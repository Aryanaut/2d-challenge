"""Optional curtain detector: finds hanging curtains in the depth image, at a low rate, for the mission manager.

  sub  /camera/depth/image_raw      depth image (16UC1 millimetres or 32FC1 metres)
  sub  /camera/depth/camera_info    intrinsics (else depth_hfov_deg is used)
  pub  /curtain/points   std_msgs/String   JSON {"points": [[x, y, ux, uy], ...]} in the map frame (latched): confirmed
                                           curtain spots and the unit direction the camera was looking at each
  pub  /curtain/markers  MarkerArray       the remembered curtain, for RViz

The mission manager (use_curtains:=true on exploration.launch.py) reads /curtain/points; when a curtain lies along the
edge of the tile the robot is on, it aligns with it and walks through (mission/modes.py, 'curtain' mode).
The detection itself is in curtain_core.py (no ROS).
"""
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explorer import curtain_core as cc


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class CurtainDetector(Node):
    def __init__(self):
        super().__init__('curtain_detector')
        p = self.declare_parameter
        self.depth_topic = p('depth_topic', '/camera/depth/image_raw').value
        self.info_topic = p('camera_info_topic', '/camera/depth/camera_info').value
        self.hfov = math.radians(float(p('depth_hfov_deg', 81.4).value))   # used until camera_info arrives
        self.depth_scale = float(p('depth_scale', 0.001).value)           # 16UC1 -> metres
        self.global_frame = p('global_frame', 'map').value
        self.rate_hz = float(p('process_hz', 2.0).value)                  # low rate: one depth frame this often
        self.params = cc.CurtainParams(
            stride=int(p('stride', 4).value), run=int(p('run_rows', 6).value),
            max_step_m=float(p('max_step_m', 0.03).value), min_jump_m=float(p('min_jump_m', 0.25).value),
            min_range_m=float(p('min_range_m', 0.30).value), max_range_m=float(p('max_range_m', 3.5).value),
            min_columns=float(p('min_columns', 0.30).value))
        self.curtain = cc.CurtainMap(cell_m=float(p('cell_m', 0.05).value), confirm_hits=int(p('confirm_hits', 3).value))
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.intrinsics = None
        self.last = 0.0
        self.version, self.sent_version = 0, -1
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, 1)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.points_pub = self.create_publisher(String, '/curtain/points', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/curtain/markers', latched)
        self.create_timer(1.0, self._publish)
        self.get_logger().info(f'curtain detector on {self.depth_topic} at {self.rate_hz:g} Hz')

    def _on_info(self, msg):
        if msg.k[0] > 0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_depth(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last < 1.0 / self.rate_hz:
            return
        self.last = now
        if msg.encoding == '16UC1':
            depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width) * self.depth_scale
        elif msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        else:
            return
        det = cc.detect_curtain(depth, self.params)
        if det is None:
            return
        fx = fy = (msg.width / 2) / math.tan(self.hfov / 2)
        k = self.intrinsics or (fx, fy, msg.width / 2, msg.height / 2)
        try:
            t = self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time(), Duration(seconds=0.2))
        except Exception as e:  # noqa: BLE001 - TF not ready yet
            self.get_logger().warn(f'no TF {self.global_frame} <- {msg.header.frame_id}: {e}', throttle_duration_sec=5.0)
            return
        tr = t.transform.translation
        pts = cc.project_to_map(det, k, quat_to_matrix(t.transform.rotation), (tr.x, tr.y, tr.z))
        if self.curtain.add(pts, (tr.x, tr.y)):
            self.version += 1
            self.get_logger().info(f'curtain seen: {len(self.curtain)} spots, {len(det.columns)} columns now, '
                                   f'{float(np.median(det.ranges)):.2f} m ahead')

    def _publish(self):
        if self.version == self.sent_version:
            return
        self.sent_version = self.version
        pts = self.curtain.confirmed()
        self.points_pub.publish(String(data=json.dumps(self.curtain.to_json())))
        m = Marker()
        m.header.frame_id = self.global_frame
        m.ns, m.id, m.type, m.action = 'curtain', 0, Marker.POINTS, Marker.ADD
        m.scale.x = m.scale.y = 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.2, 0.9, 1.0
        m.points = [Point(x=float(x), y=float(y), z=0.05) for x, y, _, _ in pts]
        self.marker_pub.publish(MarkerArray(markers=[m]))


def main(args=None):
    rclpy.init(args=args)
    node = CurtainDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
