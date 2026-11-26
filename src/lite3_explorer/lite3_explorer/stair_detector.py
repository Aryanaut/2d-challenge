"""Optional stair detector: finds staircases in the depth image, at a low rate, for the mission manager.

  sub  /camera/depth/image_raw      depth image (16UC1 millimetres or 32FC1 metres)
  sub  /camera/depth/camera_info    intrinsics (else depth_hfov_deg is used)
  sub  /ai_mode                     std_msgs/Bool   true while the robot climbs in AI mode: detection pauses (the body
                                                    pitches on the stairs, and the treads would spoil the floor height)
  sub  /stairs/enable               std_msgs/Bool   (latched) detection on / off; start_enabled until the first message.
                                                    The mission manager turns it on for the opening scan (and for the
                                                    whole mission with use_stairs:=true)
  pub  /stairs/points   std_msgs/String   JSON {"points": [[x, y, ux, uy], ...]} in the map frame (latched): confirmed
                                          stair cells and the unit direction the steps go up
  pub  /stairs/markers  MarkerArray       the remembered stair cells and their ascent, for RViz

The 2D lidar sees a staircase's risers as a wall. The depth image is projected into the map frame (z up) and searched
for treads: flat levels at evenly spaced heights, each farther away than the one below (stairs_core.py, no ROS).
The mission manager (use_stairs:=true on exploration.launch.py) turns the tiles holding the stair cells into staircase
tiles; when one shares a side with the current tile its 'stairs' mode climbs over it in the robot's AI mode.
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
from std_msgs.msg import Bool, Empty, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explorer import stairs_core as sc
from lite3_explorer.curtain_detector import quat_to_matrix


class StairDetector(Node):
    def __init__(self):
        super().__init__('stair_detector')
        p = self.declare_parameter
        self.depth_topic = p('depth_topic', '/camera/depth/image_raw').value
        self.info_topic = p('camera_info_topic', '/camera/depth/camera_info').value
        self.hfov = math.radians(float(p('depth_hfov_deg', 81.4).value))   # used until camera_info arrives
        self.depth_scale = float(p('depth_scale', 0.001).value)           # 16UC1 -> metres
        self.global_frame = p('global_frame', 'map').value                 # z must point up in this frame
        self.rate_hz = float(p('process_hz', 2.0).value)                  # low rate: one depth frame this often
        self.params = sc.StairParams(
            stride=int(p('stride', 4).value),
            min_range_m=float(p('min_range_m', 0.25).value), max_range_m=float(p('max_range_m', 2.5).value),
            cell_m=float(p('cell_m', 0.05).value), min_cell_points=int(p('min_cell_points', 3).value),
            flat_std_m=float(p('flat_std_m', 0.015).value), level_gap_m=float(p('level_gap_m', 0.03).value),
            link_cells=int(p('link_cells', 2).value), max_level_span_m=float(p('max_level_span_m', 0.08).value),
            min_level_cells=int(p('min_level_cells', 10).value),
            min_rise_m=float(p('min_rise_m', 0.07).value), max_rise_m=float(p('max_rise_m', 0.25).value),
            min_run_m=float(p('min_run_m', 0.10).value), max_gap_m=float(p('max_gap_m', 0.35).value),
            min_levels=int(p('min_levels', 3).value))
        self.floor = sc.FloorEstimate(near_m=float(p('floor_near_m', 0.6).value),
                                      min_cells=int(p('floor_min_cells', 25).value),
                                      window=int(p('floor_window', 40).value),
                                      quantile=float(p('floor_quantile', 0.2).value))
        self.stairs = sc.StairMap(cell_m=float(p('map_cell_m', 0.10).value),
                                  confirm_hits=int(p('confirm_hits', 3).value))
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.intrinsics = None
        self.paused = False
        self.enabled = bool(p('start_enabled', True).value)
        self.last = 0.0
        self.frame = 0
        self.version, self.sent_version = 0, -1
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, 1)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Bool, p('pause_topic', '/ai_mode').value, self._on_pause, latched)
        self.create_subscription(Bool, p('enable_topic', '/stairs/enable').value, self._on_enable, latched)
        self.points_pub = self.create_publisher(String, '/stairs/points', latched)
        self.create_subscription(Empty, '/map_reset', self._on_map_reset, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/stairs/markers', latched)
        self.create_timer(1.0, self._publish)
        self.get_logger().info(f'stair detector on {self.depth_topic} at {self.rate_hz:g} Hz'
                               + ('' if self.enabled else ' (disabled until /stairs/enable)'))

    def _on_info(self, msg):
        if msg.k[0] > 0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_pause(self, msg):
        if bool(msg.data) != self.paused:
            self.paused = bool(msg.data)
            self.get_logger().info('AI mode on: detection paused.' if self.paused else 'AI mode off: detecting again.')

    def _on_map_reset(self, _msg):
        """A new map (new map frame) starts: the staircases seen so far are in the old one."""
        self.stairs = sc.StairMap(cell_m=self.stairs.cell_m, confirm_hits=self.stairs.confirm_hits)
        self.version += 1
        self.get_logger().info('new map: staircases forgotten.')

    def _on_enable(self, msg):
        if bool(msg.data) != self.enabled:
            self.enabled = bool(msg.data)
            self.get_logger().info('detection enabled.' if self.enabled else 'detection disabled.')

    def _on_depth(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.paused or not self.enabled or now - self.last < 1.0 / self.rate_hz:
            return
        self.last = now
        if msg.encoding == '16UC1':
            depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width) * self.depth_scale
        elif msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        else:
            return
        fx = fy = (msg.width / 2) / math.tan(self.hfov / 2)
        k = self.intrinsics or (fx, fy, msg.width / 2, msg.height / 2)
        try:
            t = self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time(), Duration(seconds=0.2))
        except Exception as e:  # noqa: BLE001 - TF not ready yet
            self.get_logger().warn(f'no TF {self.global_frame} <- {msg.header.frame_id}: {e}', throttle_duration_sec=5.0)
            return
        tr = t.transform.translation
        pts = sc.depth_to_points(depth, k, self.params.stride)
        pts = pts @ quat_to_matrix(t.transform.rotation).T + np.array([tr.x, tr.y, tr.z])
        levels = sc.find_levels(pts, (tr.x, tr.y), self.params)
        det = sc.chain_stairs(levels, self.params, self.floor.update(levels))
        if det is None:
            return
        self.frame += 1
        if self.stairs.add(det, self.frame):
            self.version += 1
            self.get_logger().info(f'staircase seen: {len(self.stairs)} cells, {det.steps} steps of '
                                   f'{det.rise_m * 100:.0f} cm going up towards '
                                   f'{math.degrees(math.atan2(det.ascent[1], det.ascent[0])):+.0f} deg')

    def _publish(self):
        if self.version == self.sent_version:
            return
        self.sent_version = self.version
        pts = self.stairs.confirmed()
        self.points_pub.publish(String(data=json.dumps(self.stairs.to_json())))
        m = Marker()
        m.header.frame_id = self.global_frame
        m.ns, m.id, m.type, m.action = 'stairs', 0, Marker.POINTS, Marker.ADD
        m.scale.x = m.scale.y = 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.6, 0.0, 1.0
        m.points = [Point(x=float(x), y=float(y), z=0.05) for x, y, _, _ in pts]
        arrows = Marker()
        arrows.header.frame_id = self.global_frame
        arrows.ns, arrows.id, arrows.type, arrows.action = 'stairs', 1, Marker.LINE_LIST, Marker.ADD
        arrows.scale.x = 0.015
        arrows.color.r, arrows.color.g, arrows.color.b, arrows.color.a = 1.0, 0.6, 0.0, 1.0
        for x, y, ux, uy in pts:                       # short line from each cell towards where the steps go up
            arrows.points += [Point(x=float(x), y=float(y), z=0.05),
                              Point(x=float(x + 0.08 * ux), y=float(y + 0.08 * uy), z=0.05)]
        self.marker_pub.publish(MarkerArray(markers=[m, arrows]))


def main(args=None):
    rclpy.init(args=args)
    node = StairDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
