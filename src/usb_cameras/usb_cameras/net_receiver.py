"""ROS 2 node on the ROBOT: receives camera frames over the network (from the laptop now,
the Raspberry Pi later) and publishes them exactly like usb_cameras does:

  /<camera>/image_raw              sensor_msgs/Image (bgr8)
  /<camera>/image_raw/compressed   sensor_msgs/CompressedImage (the JPEG as sent, no re-encoding)
  /<camera>/camera_info            sensor_msgs/CameraInfo (focal length from the sender's hfov)
  /<camera>/qr_codes               std_msgs/String JSON: QR codes the SENDER read at full resolution
                                   (qr_scanner reads these with <camera>_detections_topic)

  ros2 run usb_cameras net_receiver                  # listens on port 5600
  ros2 run usb_cameras net_receiver --ros-args -p port:=5600

The sender connects TO the robot, so nothing on the laptop/Pi needs an open port.
Any number of senders can connect; frames are told apart by camera name.
Header stamps use the ROBOT's clock at arrival (the sender's clock is not synchronised).
"""

import json
import socket
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String

from .capture import camera_info_k
from .net_protocol import DEFAULT_PORT, ProtocolError, read_frame


class NetReceiver(Node):
    def __init__(self):
        super().__init__("net_camera_receiver")
        p = self.declare_parameter
        p("port", DEFAULT_PORT)
        p("bind", "0.0.0.0")
        p("publish_raw", True)
        p("publish_compressed", True)
        p("allowed_cameras", ["left", "right"])   # ignore other names ([] = allow all)
        p("client_timeout_s", 5.0)
        g = lambda n: self.get_parameter(n).value
        self.opts = {k: g(k) for k in ("publish_raw", "publish_compressed", "client_timeout_s")}
        self.allowed = [c for c in g("allowed_cameras") if c]

        self.lock = threading.Lock()
        self.outputs = {}            # camera name -> dict of publishers + stats
        for name in self.allowed:
            self._outputs_for(name)

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((g("bind"), int(g("port"))))
        self.server.listen(4)
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        self.create_timer(10.0, self._report)
        self.get_logger().info(f"net_camera_receiver up: listening on {g('bind')}:{g('port')} "
                               f"for cameras {self.allowed or 'any'}")

    def _outputs_for(self, name):
        with self.lock:
            if name not in self.outputs:
                self.outputs[name] = {
                    "raw": self.create_publisher(Image, f"/{name}/image_raw", qos_profile_sensor_data),
                    "jpg": self.create_publisher(CompressedImage, f"/{name}/image_raw/compressed",
                                                 qos_profile_sensor_data),
                    "info": self.create_publisher(CameraInfo, f"/{name}/camera_info", qos_profile_sensor_data),
                    "qr": self.create_publisher(String, f"/{name}/qr_codes", 10),
                    "qr_count": 0, "qr_codes": 0, "scan_ms": None,
                    "count": 0, "window_start": time.time(), "window_count": 0, "last": None, "bytes": 0,
                }
            return self.outputs[name]

    # ---------- network ----------
    def _accept_loop(self):
        while self.running and rclpy.ok():
            try:
                conn, addr = self.server.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(float(self.opts["client_timeout_s"]))
            self.get_logger().info(f"sender connected from {addr[0]}:{addr[1]}")
            threading.Thread(target=self._client_loop, args=(conn, addr), daemon=True).start()

    def _client_loop(self, conn, addr):
        try:
            while self.running:
                header, jpeg = read_frame(conn)
                self._publish(header, jpeg)
        except socket.timeout:
            self.get_logger().warn(f"sender {addr[0]}: no data for {self.opts['client_timeout_s']} s, dropping")
        except ProtocolError as e:
            self.get_logger().warn(f"sender {addr[0]}: {e}")
        except (ConnectionError, OSError):
            self.get_logger().info(f"sender {addr[0]} disconnected")
        except Exception as e:                       # bad JSON etc.: drop this sender, keep the node alive
            self.get_logger().warn(f"sender {addr[0]}: bad data ({type(e).__name__}: {e}), dropping")
        finally:
            conn.close()

    # ---------- ROS output ----------
    def _publish(self, header, jpeg):
        name = str(header.get("camera", ""))
        if not name or (self.allowed and name not in self.allowed):
            return
        out = self._outputs_for(name)
        now = time.time()
        if header.get("type", "frame") == "qr":
            self._publish_qr(name, out, header, now)
            return
        if not jpeg:
            return
        sec, nsec = int(now), int((now - int(now)) * 1e9)
        frame_id = f"{name}_camera_optical"
        w, h = int(header.get("width", 0)), int(header.get("height", 0))

        if self.opts["publish_compressed"]:
            c = CompressedImage()
            c.header.stamp.sec, c.header.stamp.nanosec = sec, nsec
            c.header.frame_id = frame_id
            c.format = "jpeg"
            c.data = jpeg
            out["jpg"].publish(c)
        if self.opts["publish_raw"]:
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return
            h, w = frame.shape[:2]
            m = Image()
            m.header.stamp.sec, m.header.stamp.nanosec = sec, nsec
            m.header.frame_id = frame_id
            m.height, m.width, m.encoding = h, w, "bgr8"
            m.is_bigendian, m.step = 0, w * 3
            m.data = frame.tobytes()
            out["raw"].publish(m)
        if w and h:
            info = CameraInfo()
            info.header.stamp.sec, info.header.stamp.nanosec = sec, nsec
            info.header.frame_id = frame_id
            info.height, info.width = h, w
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            k = camera_info_k(w, h, float(header.get("hfov_deg", 70.0)))
            info.k = k
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
            out["info"].publish(info)
        with self.lock:
            out["count"] += 1
            out["window_count"] += 1
            out["bytes"] += len(jpeg)
            out["last"] = now
            t_cap = header.get("t_capture")
            out["age"] = (now - t_cap) if isinstance(t_cap, (int, float)) else None

    def _publish_qr(self, name, out, header, now):
        codes = header.get("codes") or []
        if not isinstance(codes, list):
            return
        msg = {"camera": name, "stamp": now,           # robot clock at arrival (used for TF)
               "width": int(header.get("width", 0)), "height": int(header.get("height", 0)),
               "hfov_deg": float(header.get("hfov_deg", 70.0)), "seq": header.get("seq"),
               "scan_ms": header.get("scan_ms"), "codes": codes}
        out["qr"].publish(String(data=json.dumps(msg)))
        with self.lock:
            out["qr_count"] += 1
            out["qr_codes"] += sum(1 for c in codes if isinstance(c, dict) and c.get("decoded"))
            out["scan_ms"] = header.get("scan_ms")

    def _report(self):
        now = time.time()
        with self.lock:
            for name, out in self.outputs.items():
                dt = max(1e-3, now - out["window_start"])
                fps = out["window_count"] / dt
                kbps = out["bytes"] / dt / 1024
                if out["last"] is None or now - out["last"] > 5.0:
                    self.get_logger().warn(f"{name}: no frames received in the last 5 s")
                else:
                    qr = ""
                    if out["qr_count"]:
                        qr = (f", QR results {out['qr_count'] / dt:.1f}/s from the sender "
                              f"({out['qr_codes']} code reads, {out['scan_ms']} ms per scan)")
                    self.get_logger().info(f"{name}: {fps:.1f} fps, {kbps:.0f} kB/s{qr}")
                out["window_start"], out["window_count"], out["bytes"] = now, 0, 0
                out["qr_count"], out["qr_codes"] = 0, 0

    def destroy_node(self):
        self.running = False
        try:
            self.server.close()
        except OSError:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NetReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
