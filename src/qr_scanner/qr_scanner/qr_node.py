"""ROS 2 node: reads QR codes on walls with SEVERAL cameras (e.g. front, left, right), decodes
them, localises them and reports each physical code once, whichever camera saw it.

Per camera (names listed in the `cameras` parameter, e.g. front, left, right), set either
  <name>_image_topic      a ROS image topic (raw or, with <name>_compressed, CompressedImage), or
  <name>_device           a /dev/video* device read directly by this node (no camera driver needed)
or
  <name>_detections_topic QR codes ALREADY READ by the camera's sender (cam_sender.py reads the
                          full-resolution picture on the laptop/Pi; net_receiver publishes them on
                          /<name>/qr_codes). This node then does no decoding for that camera, only
                          tracking/direction/distance/logging. An image topic, if also set, is only
                          used for the debug picture.
plus optionally
  <name>_camera_info_topic   calibration; without it, intrinsics come from <name>_hfov_deg
  <name>_depth_topic         depth ALIGNED to that colour image (else distance from the QR size)
  <name>_mount_yaw_deg       which way the camera faces: 0 front, +90 left, -90 right
  <name>_mount_pitch_deg     tilt downwards
  <name>_publish_tf + <name>_mount_xyz   publish base_frame -> <name>_qr_optical so map positions work
A camera with no topic and no device is simply off. "none" counts as empty.

Publishes:
  /qr/detections                     JSON event, once per newly confirmed code (includes "camera")
  /qr/visible                        JSON of codes in view on every scan, per camera
  /qr/sequence                       distinct codes in the order found (latched)
  /qr/markers                        RViz markers
  /qr/<name>/debug_image/compressed  annotated frames per camera (throttled)
Service ~/reset clears everything.
"""

import csv
import json
import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .localize import (backproject, bearing_deg, depth_at, depth_from_size,
                       intrinsics_from_fov, optical_mount_quaternion)
from .payload import decode_payload, parse_key_map
from .qr_decode import QrHit, QrReader, draw
from .tracker import CodeTracker

DEFAULTS = {   # name: (image_topic, camera_info_topic, mount_yaw_deg)
    "front": ("/camera/color/image_raw", "/camera/color/camera_info", 0.0),
    "left": ("", "", 90.0),
    "right": ("", "", -90.0),
}


def _on(value) -> bool:
    return bool(value) and str(value).strip().lower() not in ("none", "-", "''", '""', "false")


class CameraInput:
    """One camera: latest frame (+ depth), intrinsics, mounting."""

    def __init__(self, name):
        self.name = name
        self.frame = None          # (bgr, header-like: stamp, frame_id)
        self.depth = None
        self.K = None
        self.K_from_info = False
        self.info_frame = ""
        self.frames_seen = 0
        self.last_frame_wall = None
        self.cfg = {}
        self.capture = None        # cv2.VideoCapture when reading a device directly
        self.remote = None         # deque of QR results read by the sender (detections_topic mode)
        self.remote_msgs = 0
        self.last_remote = None    # (hits, (h, w), time) for drawing on the preview picture
        self.last_debug = 0.0


class QrScanner(Node):
    def __init__(self):
        super().__init__("qr_scanner")
        p = self.declare_parameter
        p("cameras", ["front", "left", "right"])
        p("base_frame", "base_link")
        p("depth_scale", 0.001)
        p("target_frame", "map")
        p("scan_rate_hz", 5.0)                   # per camera
        p("capture_width", 1280)                 # for <name>_device cameras
        p("capture_height", 720)
        p("full_upscale_every_n", 0)
        p("qr_size_m", 0.09)
        p("confirm_frames", 2)
        p("confirm_window_s", 3.0)
        p("same_code_dist_m", 1.5)
        p("decode_mode", "raw")
        p("caesar_shift", 0)
        p("key_map", [""])
        p("log_file", os.path.expanduser("~/qr_log.csv"))
        p("debug_image_hz", 2.0)
        g = lambda n: self.get_parameter(n).value

        self.cfg = {n: g(n) for n in ("base_frame", "target_frame", "qr_size_m", "decode_mode", "caesar_shift",
                                      "depth_scale", "log_file", "debug_image_hz")}
        self.key_map = parse_key_map([e for e in g("key_map") if e])
        self.tracker = CodeTracker(int(g("confirm_frames")), float(g("confirm_window_s")),
                                   float(g("same_code_dist_m")))
        self.lock = threading.Lock()
        self.bridge = None
        self.tf_buffer = None
        try:
            from tf2_ros import Buffer, TransformListener
            import tf2_geometry_msgs  # noqa: F401
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        except ImportError:
            self.get_logger().warn("tf2 not available: positions reported in the camera frame only")

        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_det = self.create_publisher(String, "/qr/detections", 10)
        self.pub_vis = self.create_publisher(String, "/qr/visible", 10)
        self.pub_seq = self.create_publisher(String, "/qr/sequence", latched)
        self.pub_markers = self.create_publisher(MarkerArray, "/qr/markers", latched)

        # ---- cameras ----
        self.cams = {}
        self.debug_pubs = {}
        static_tfs = []
        for name in g("cameras"):
            topic0, info0, yaw0 = DEFAULTS.get(name, ("", "", 0.0))
            for key, val in (("image_topic", topic0), ("compressed", False), ("camera_info_topic", info0),
                             ("detections_topic", ""),
                             ("depth_topic", ""), ("device", ""), ("hfov_deg", 70.0),
                             ("mount_yaw_deg", yaw0), ("mount_pitch_deg", 0.0),
                             ("publish_tf", False), ("mount_xyz", [0.0, 0.0, 0.0])):
                p(f"{name}_{key}", val)
            cam = CameraInput(name)
            cam.cfg = {k: g(f"{name}_{k}") for k in ("image_topic", "compressed", "camera_info_topic",
                                                     "detections_topic", "depth_topic",
                                                     "device", "hfov_deg", "mount_yaw_deg", "mount_pitch_deg",
                                                     "publish_tf", "mount_xyz")}
            c = cam.cfg
            if _on(c["detections_topic"]):
                cam.remote = deque(maxlen=20)
                self.create_subscription(String, c["detections_topic"],
                                         lambda m, cam=cam: self._on_remote(cam, m), 10)
            if _on(c["device"]) and cam.remote is None:
                self._open_device(cam, int(g("capture_width")), int(g("capture_height")))
            elif _on(c["image_topic"]):
                if c["compressed"]:
                    self.create_subscription(CompressedImage, c["image_topic"],
                                             lambda m, cam=cam: self._on_compressed(cam, m), qos_profile_sensor_data)
                else:
                    self.create_subscription(Image, c["image_topic"],
                                             lambda m, cam=cam: self._on_image(cam, m), qos_profile_sensor_data)
            elif cam.remote is None:
                continue                                   # camera switched off
            if _on(c["camera_info_topic"]):
                self.create_subscription(CameraInfo, c["camera_info_topic"],
                                         lambda m, cam=cam: self._on_info(cam, m), qos_profile_sensor_data)
            if _on(c["depth_topic"]):
                self.create_subscription(Image, c["depth_topic"],
                                         lambda m, cam=cam: self._on_depth(cam, m), qos_profile_sensor_data)
            if c["publish_tf"]:
                static_tfs.append(self._mount_tf(cam))
            self.cams[name] = cam
            self.debug_pubs[name] = self.create_publisher(CompressedImage, f"/qr/{name}/debug_image/compressed", 2)

        if static_tfs:
            try:
                from tf2_ros import StaticTransformBroadcaster
                self.static_broadcaster = StaticTransformBroadcaster(self)
                self.static_broadcaster.sendTransform(static_tfs)
            except ImportError:
                self.get_logger().warn("tf2_ros missing: cannot publish camera mount transforms")

        self.reader = QrReader(full_upscale_every_n=int(g("full_upscale_every_n")))
        self.create_service(Trigger, "~/reset", self._on_reset)
        self.create_timer(1.0 / max(0.5, float(g("scan_rate_hz"))), self._scan)
        self.create_timer(10.0, self._check_cameras)
        desc = ", ".join(f"{n} ({self._source(c)})" for n, c in self.cams.items())
        self.get_logger().info(f"qr_scanner up: cameras: {desc or 'NONE'}  decode_mode={self.cfg['decode_mode']} "
                               f"key_map={self.key_map or 'none'}")

    # ---------- inputs ----------
    def _cv(self, msg, encoding):
        if self.bridge is None:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)

    def _store(self, cam, bgr, stamp, frame_id):
        with self.lock:
            cam.frame = (bgr, stamp, frame_id)
            cam.frames_seen += 1
            cam.last_frame_wall = time.monotonic()

    def _on_image(self, cam, msg):
        try:
            bgr = self._cv(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"{cam.name}: image conversion failed: {e}", throttle_duration_sec=5.0)
            return
        self._store(cam, bgr, msg.header.stamp, msg.header.frame_id)

    def _on_compressed(self, cam, msg):
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            self._store(cam, bgr, msg.header.stamp, msg.header.frame_id)

    def _on_remote(self, cam, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            self.get_logger().warn(f"{cam.name}: bad QR result message", throttle_duration_sec=5.0)
            return
        with self.lock:
            cam.remote.append(data)
            cam.remote_msgs += 1
            cam.last_frame_wall = time.monotonic()

    @staticmethod
    def _source(cam):
        if cam.remote is not None:
            return f"QR results from {cam.cfg['detections_topic']}"
        return cam.cfg["device"] if _on(cam.cfg["device"]) else cam.cfg["image_topic"]

    def _on_depth(self, cam, msg):
        try:
            d = self._cv(msg, "passthrough")
        except Exception as e:
            self.get_logger().warn(f"{cam.name}: depth conversion failed: {e}", throttle_duration_sec=5.0)
            return
        with self.lock:
            cam.depth = (d, msg.header.stamp)

    def _on_info(self, cam, msg):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0][0] > 0:                       # uncalibrated drivers publish all zeros
            cam.K, cam.K_from_info = K, True
        cam.info_frame = msg.header.frame_id

    def _open_device(self, cam, w, h):
        cap = cv2.VideoCapture(cam.cfg["device"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not cap.isOpened():
            self.get_logger().error(f"{cam.name}: cannot open {cam.cfg['device']}")
            return
        cam.capture = cap

        def loop():
            while rclpy.ok() and cam.capture is not None:
                ok, frame = cap.read()
                if ok:
                    self._store(cam, frame, self.get_clock().now().to_msg(), "")
                else:
                    time.sleep(0.05)
        threading.Thread(target=loop, daemon=True).start()

    def _check_cameras(self):
        for name, cam in self.cams.items():
            if cam.last_frame_wall is None or time.monotonic() - cam.last_frame_wall > 5.0:
                what = "QR results" if cam.remote is not None else "frames"
                self.get_logger().warn(f"{name} camera: no {what} in the last 5 s from {self._source(cam)}")

    # ---------- geometry ----------
    def _mount_tf(self, cam):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.cfg["base_frame"]
        t.child_frame_id = f"{cam.name}_qr_optical"
        x, y, z = (list(cam.cfg["mount_xyz"]) + [0.0, 0.0, 0.0])[:3]
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = float(x), float(y), float(z)
        q = optical_mount_quaternion(float(cam.cfg["mount_yaw_deg"]), float(cam.cfg["mount_pitch_deg"]))
        (t.transform.rotation.x, t.transform.rotation.y,
         t.transform.rotation.z, t.transform.rotation.w) = (float(v) for v in q)
        return t

    def _intrinsics(self, cam, shape):
        if cam.K is None:                     # no (valid) camera_info: estimate from the field of view
            h, w = shape[:2]
            cam.K = intrinsics_from_fov(w, h, float(cam.cfg["hfov_deg"]))
        return cam.K

    def _locate(self, cam, hit, shape, stamp, frame_id, depth, K=None, from_info=None):
        info = {"side_px": round(hit.side_px, 1)}
        if K is None:
            K, from_info = self._intrinsics(cam, shape), cam.K_from_info
        u, v = hit.center
        z = None
        if depth is not None:
            d_img, d_stamp = depth
            age = abs(Time.from_msg(stamp).nanoseconds - Time.from_msg(d_stamp).nanoseconds) * 1e-9
            if age < 0.2:
                z = depth_at(d_img, hit.corners, shape, self.cfg["depth_scale"])
                if z:
                    info["range_source"] = "depth"
        if z is None:
            z = depth_from_size(hit.side_px, K[0][0], float(self.cfg["qr_size_m"]))
            info["range_source"] = "qr_size" + ("" if from_info else "+fov_estimate")
        cam_bearing = bearing_deg(u, K)
        robot_bearing = (cam_bearing + float(cam.cfg["mount_yaw_deg"]) + 180.0) % 360.0 - 180.0
        info["bearing_deg"] = round(robot_bearing, 1)          # relative to the robot, + = left
        if z is None:
            return info
        x, y, zz = backproject(u, v, z, K)
        info["distance_m"] = round(float(math.sqrt(x * x + y * y + zz * zz)), 2)
        info["camera_point"] = [round(x, 3), round(y, 3), round(zz, 3)]
        tf_frame = f"{cam.name}_qr_optical" if cam.cfg["publish_tf"] else (frame_id or cam.info_frame)
        info["camera_frame"] = tf_frame
        target = self.cfg["target_frame"]
        if target and self.tf_buffer is not None and tf_frame:
            ps = PointStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = tf_frame
            ps.point.x, ps.point.y, ps.point.z = float(x), float(y), float(zz)
            try:
                out = self.tf_buffer.transform(ps, target, timeout=Duration(seconds=0.05))
                info["map_point"] = [round(out.point.x, 3), round(out.point.y, 3), round(out.point.z, 3)]
            except Exception as e:
                info["tf_error"] = type(e).__name__
        return info

    # ---------- reporting ----------
    def _confirm(self, code, cam_name, info):
        decoded = decode_payload(code.text, self.cfg["decode_mode"], int(self.cfg["caesar_shift"]), self.key_map)
        code.report = {
            "text": code.text, "decoded": decoded, "camera": cam_name,
            "map_point": code.mean_position(),
            "frame": self.cfg["target_frame"] if code.positions else info.get("camera_frame"),
            "camera_point": info.get("camera_point"), "distance_m": info.get("distance_m"),
            "bearing_deg": info.get("bearing_deg"), "range_source": info.get("range_source"),
            "first_seen_unix": round(time.time(), 2), "first_seen_local": time.strftime("%H:%M:%S"),
            "index": sum(1 for c in self.tracker.codes if c.confirmed),
        }
        self.pub_det.publish(String(data=json.dumps(code.report)))
        self.pub_seq.publish(String(data="".join(self.tracker.sequence)))
        self.get_logger().info(f"QR #{code.report['index']} ({cam_name} camera): '{code.text}' -> '{decoded}' "
                               f"{info.get('distance_m')} m, bearing {info.get('bearing_deg')} deg")
        self._log(code.report)
        self._publish_markers()

    def _log(self, rep):
        path = self.cfg["log_file"]
        if not path:
            return
        new = not os.path.exists(path)
        try:
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["time", "index", "text", "decoded", "camera", "x", "y", "z", "frame",
                                "distance_m", "bearing_deg"])
                pt = rep.get("map_point") or rep.get("camera_point") or [None] * 3
                w.writerow([rep["first_seen_local"], rep["index"], rep["text"], rep["decoded"], rep["camera"],
                            *pt, rep["frame"], rep["distance_m"], rep["bearing_deg"]])
        except OSError as e:
            self.get_logger().warn(f"could not write {path}: {e}")

    def _publish_markers(self):
        arr = MarkerArray()
        frame = self.cfg["target_frame"] or self.cfg["base_frame"]
        clear = Marker()
        clear.header.frame_id = frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, c in enumerate(c for c in self.tracker.codes if c.confirmed and c.positions):
            x, y, z = c.mean_position()
            for kind in (Marker.CUBE, Marker.TEXT_VIEW_FACING):
                m = Marker()
                m.header.frame_id = frame
                m.ns, m.id = "qr", i * 2 + (kind == Marker.TEXT_VIEW_FACING)
                m.type, m.action = kind, Marker.ADD
                m.pose.position.x, m.pose.position.y = x, y
                m.pose.position.z = z + (0.25 if kind == Marker.TEXT_VIEW_FACING else 0.0)
                m.pose.orientation.w = 1.0
                if kind == Marker.CUBE:
                    m.scale.x = m.scale.y = m.scale.z = 0.12
                    m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.8, 0.2, 0.9
                else:
                    m.scale.z = 0.2
                    m.color.r = m.color.g = m.color.b = m.color.a = 1.0
                    m.text = f"{c.text} ({c.report.get('decoded', '')})"
                arr.markers.append(m)
        self.pub_markers.publish(arr)

    # ---------- main loop ----------
    def _scan(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        visible = []
        for name, cam in self.cams.items():
            if cam.remote is not None:
                self._scan_remote(cam, t, visible)
                continue
            with self.lock:
                frame, depth = cam.frame, cam.depth
                cam.frame = None                               # each frame is processed once
            if frame is None:
                continue
            bgr, stamp, frame_id = frame
            hits = self.reader.read(bgr)
            for h in hits:
                info = self._locate(cam, h, bgr.shape, stamp, frame_id, depth)
                visible.append(self._handle_hit(cam, h, info, t))
            self._debug_image(cam, bgr, hits, stamp)
        self.pub_vis.publish(String(data=json.dumps({"t": round(t, 2), "codes": visible})))

    def _handle_hit(self, cam, h, info, t):
        entry = {"camera": cam.name, "text": h.text if h.decoded else None, "decoded_ok": h.decoded,
                 "method": h.method, **info}
        if not h.decoded:
            entry["hint"] = "QR spotted but too small/blurred: slow down or move closer"
        else:
            code, new = self.tracker.observe(h.text, cam.name, t, info.get("map_point"))
            if new:
                self._confirm(code, cam.name, info)
            entry["already_reported"] = code.confirmed
        return entry

    def _scan_remote(self, cam, t, visible):
        """QR codes the sender already read at full resolution: locate + track them."""
        with self.lock:
            batch = list(cam.remote)
            cam.remote.clear()
            frame = cam.frame
            cam.frame = None
        last_hits, last_shape = None, None
        for msg in batch:
            W, H = int(msg.get("width") or 0), int(msg.get("height") or 0)
            if W <= 0 or H <= 0:
                continue
            # corners are in FULL-resolution pixels, so use full-resolution intrinsics
            K = intrinsics_from_fov(W, H, float(msg.get("hfov_deg") or cam.cfg["hfov_deg"]))
            stamp = self.get_clock().now().to_msg()
            if isinstance(msg.get("stamp"), (int, float)):
                stamp.sec, stamp.nanosec = int(msg["stamp"]), int((msg["stamp"] % 1.0) * 1e9)
            hits = []
            for c in msg.get("codes") or []:
                try:
                    corners = np.asarray(c["corners"], np.float32).reshape(4, 2)
                except (KeyError, TypeError, ValueError):
                    continue
                hits.append(QrHit(str(c.get("text") or ""), corners, str(c.get("method", "remote")),
                                  bool(c.get("decoded"))))
            for h in hits:
                if h.decoded and not h.text:
                    continue
                info = self._locate(cam, h, (H, W), stamp, f"{cam.name}_camera_optical", None, K, False)
                visible.append(self._handle_hit(cam, h, info, t))
            last_hits, last_shape = hits, (H, W)
        if last_hits is not None:
            cam.last_remote = (last_hits, last_shape, time.monotonic())
        if frame is not None:                    # preview picture: draw the latest result scaled down
            bgr, stamp, _ = frame
            hits = []
            if cam.last_remote is not None and time.monotonic() - cam.last_remote[2] < 1.0:
                r_hits, r_shape, _ = cam.last_remote
                k = bgr.shape[1] / float(r_shape[1])
                hits = [QrHit(h.text, h.corners * k, h.method, h.decoded) for h in r_hits]
            self._debug_image(cam, bgr, hits, stamp)

    def _debug_image(self, cam, bgr, hits, stamp):
        hz = float(self.cfg["debug_image_hz"])
        now = time.monotonic()
        if hz <= 0 or now - cam.last_debug < 1.0 / hz:
            return
        cam.last_debug = now
        out = draw(bgr, hits)
        cv2.putText(out, cam.name, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.format = "jpeg"
            msg.data = buf.tobytes()
            self.debug_pubs[cam.name].publish(msg)

    def _on_reset(self, request, response):
        self.tracker.reset()
        self.pub_seq.publish(String(data=""))
        self._publish_markers()
        response.success, response.message = True, "QR history cleared"
        return response

    def destroy_node(self):
        for cam in self.cams.values():
            if cam.capture is not None:
                cap, cam.capture = cam.capture, None
                cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QrScanner()
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
