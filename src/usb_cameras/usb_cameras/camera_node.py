"""ROS 2 node: publishes several USB webcams (e.g. left + right) as standard image topics.

  ros2 run usb_cameras usb_cameras --ros-args -p names:="[left, right]" \
      -p devices:="[/dev/video0, /dev/video2]"

For every camera <name>:
  /<name>/image_raw              sensor_msgs/Image (bgr8)         - for nodes on this computer
  /<name>/image_raw/compressed   sensor_msgs/CompressedImage JPEG - for viewing over the network
  /<name>/camera_info            sensor_msgs/CameraInfo (focal length from hfov_deg; uncalibrated)
frame_id: <name>_camera_optical

Each camera runs in its own thread and reconnects automatically if it is unplugged.
Use /dev/v4l/by-path/... names (stable per USB port) so left and right never swap after a reboot.
"""

import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from .capture import CameraThread, camera_info_k


class UsbCameras(Node):
    def __init__(self):
        super().__init__("usb_cameras")
        p = self.declare_parameter
        p("names", ["left", "right"])
        p("devices", ["/dev/video0", "/dev/video2"])
        p("width", 640)
        p("height", 480)
        p("fps", 15.0)
        p("mjpg", True)                    # needed for two cameras on one USB bus
        p("hfov_deg", 70.0)                # for camera_info (check your webcam's spec)
        p("publish_raw", True)
        p("publish_compressed", True)
        p("jpeg_quality", 80)
        g = lambda n: self.get_parameter(n).value
        names, devices = list(g("names")), list(g("devices"))
        if len(names) != len(devices):
            raise ValueError(f"names {names} and devices {devices} must have the same length")
        self.opts = {k: g(k) for k in ("hfov_deg", "publish_raw", "publish_compressed", "jpeg_quality")}

        self.cams = []
        for name, dev in zip(names, devices):
            if not dev or str(dev).lower() == "none":
                continue
            cam = CameraThread(name, dev, int(g("width")), int(g("height")), float(g("fps")), bool(g("mjpg")),
                               log=self.get_logger().info)
            cam.pub_raw = self.create_publisher(Image, f"/{name}/image_raw", qos_profile_sensor_data)
            cam.pub_jpg = self.create_publisher(CompressedImage, f"/{name}/image_raw/compressed",
                                                qos_profile_sensor_data)
            cam.pub_info = self.create_publisher(CameraInfo, f"/{name}/camera_info", qos_profile_sensor_data)
            cam.last_sent = None
            cam.start()
            self.cams.append(cam)

        self.create_timer(1.0 / max(1.0, float(g("fps"))), self._publish)
        self.create_timer(10.0, self._report)
        self.get_logger().info("usb_cameras up: " + ", ".join(f"{c.name}={c.device}" for c in self.cams))

    def _publish(self):
        for cam in self.cams:
            frame, stamp = cam.latest()
            if frame is None or stamp == cam.last_sent:
                continue                       # nothing new from this camera
            cam.last_sent = stamp
            h, w = frame.shape[:2]
            sec = int(stamp)
            header_stamp = (sec, int((stamp - sec) * 1e9))
            frame_id = f"{cam.name}_camera_optical"

            if self.opts["publish_raw"]:
                m = Image()
                m.header.stamp.sec, m.header.stamp.nanosec = header_stamp
                m.header.frame_id = frame_id
                m.height, m.width, m.encoding = h, w, "bgr8"
                m.is_bigendian, m.step = 0, w * 3
                m.data = frame.tobytes()
                cam.pub_raw.publish(m)
            if self.opts["publish_compressed"]:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(self.opts["jpeg_quality"])])
                if ok:
                    c = CompressedImage()
                    c.header.stamp.sec, c.header.stamp.nanosec = header_stamp
                    c.header.frame_id = frame_id
                    c.format = "jpeg"
                    c.data = buf.tobytes()
                    cam.pub_jpg.publish(c)
            info = CameraInfo()
            info.header.stamp.sec, info.header.stamp.nanosec = header_stamp
            info.header.frame_id = frame_id
            info.height, info.width = h, w
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            k = camera_info_k(w, h, float(self.opts["hfov_deg"]))
            info.k = k
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [k[0], 0.0, k[2], 0.0, 0.0, k[4], k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
            cam.pub_info.publish(info)

    def _report(self):
        for cam in self.cams:
            frame, stamp = cam.latest()
            if stamp is None or time.time() - stamp > 5.0:
                self.get_logger().warn(f"{cam.name}: no frames from {cam.device} in the last 5 s")

    def destroy_node(self):
        for cam in self.cams:
            cam.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UsbCameras()
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
