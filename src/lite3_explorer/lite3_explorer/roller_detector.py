"""Optional roller detector: finds roller tiles with the depth camera, confirmed by the 3D lidar, at a low rate.

  sub  /camera/depth/image_raw      depth image (16UC1 millimetres or 32FC1 metres)
  sub  /camera/depth/camera_info    intrinsics (else depth_hfov_deg is used)
  sub  <cloud_topic>                3D lidar pointcloud (default /livox/points; use_lidar): the roller returns only low
                                    points and the space behind it is open (roller_core.LidarLowMap)
  sub  /roller_mode                 std_msgs/Bool   true while the mission's roller mode crosses a roller: detection
                                                    pauses (the body pitches over the roller)
  pub  /roller/points   std_msgs/String   JSON {"points": [[x, y, nx, ny, lidar], ...]} in the map frame (latched):
                                          confirmed roller spots, the unit way across the roller, and how many lidar
                                          low-only cells lie near each spot
  pub  /roller/markers  MarkerArray       the remembered roller spots (white, lidar-confirmed brighter), for RViz

The 2D lidar sees a roller lying across the middle of a tile as a wall, so the tile looks like half a tile. The depth
image is projected into the map frame (z up) and searched for a short, thin, long ridge with the floor hidden behind it
(roller_core.py, no ROS); the floor height comes from the depth image too (stairs_core.FloorEstimate). The mission
manager (use_rollers:=true on exploration.launch.py) turns the tiles holding the spots into roller tiles; when one
shares a side with the current tile its 'roller' mode squares up to it and crosses it in the h-step gait.
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
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explorer import roller_core as rc
from lite3_explorer import stairs_core as sc
from lite3_explorer.curtain_detector import quat_to_matrix


class RollerDetector(Node):
    def __init__(self):
        super().__init__('roller_detector')
        p = self.declare_parameter
        self.depth_topic = p('depth_topic', '/camera/depth/image_raw').value
        self.info_topic = p('camera_info_topic', '/camera/depth/camera_info').value
        self.hfov = math.radians(float(p('depth_hfov_deg', 81.4).value))   # used until camera_info arrives
        self.depth_scale = float(p('depth_scale', 0.001).value)           # 16UC1 -> metres
        self.global_frame = p('global_frame', 'map').value                 # z must point up in this frame
        self.rate_hz = float(p('process_hz', 2.0).value)                  # low rate: one depth frame this often
        self.params = rc.RollerParams(
            stride=int(p('stride', 4).value),
            min_range_m=float(p('min_range_m', 0.25).value), max_range_m=float(p('max_range_m', 1.6).value),
            cell_m=float(p('cell_m', 0.05).value), min_cell_points=int(p('min_cell_points', 3).value),
            min_top_m=float(p('min_top_m', 0.12).value), max_top_m=float(p('max_top_m', 0.30).value),
            tall_m=float(p('tall_m', 0.35).value), link_cells=int(p('link_cells', 2).value),
            min_cells=int(p('min_cells', 6).value), min_length_m=float(p('min_length_m', 0.40).value),
            max_width_m=float(p('max_width_m', 0.25).value),
            behind_m=(float(p('behind_near_m', 0.05).value), float(p('behind_far_m', 0.40).value)),
            behind_min_h_m=float(p('behind_min_h_m', 0.03).value),
            max_behind_frac=float(p('max_behind_frac', 0.15).value), min_points=int(p('min_points', 300).value))
        self.floor = sc.FloorEstimate(near_m=float(p('floor_near_m', 0.6).value),
                                      min_cells=int(p('floor_min_cells', 25).value),
                                      window=int(p('floor_window', 40).value),
                                      quantile=float(p('floor_quantile', 0.2).value))
        self.rollers = rc.RollerMap(cell_m=float(p('map_cell_m', 0.10).value),
                                    confirm_hits=int(p('confirm_hits', 3).value))
        self.use_lidar = bool(p('use_lidar', True).value)
        self.cloud_topic = p('cloud_topic', '/livox/points').value
        self.lidar_hz = float(p('lidar_hz', 2.0).value)
        self.lidar_match_m = float(p('lidar_match_m', 0.15).value)
        self.lidar = rc.LidarLowMap(
            cell_m=float(p('lidar_cell_m', 0.05).value),
            low_m=(float(p('lidar_low_min_m', 0.04).value), float(p('lidar_low_max_m', 0.28).value)),
            high_min_m=float(p('lidar_high_min_m', 0.32).value), min_low=int(p('lidar_min_low', 20).value),
            max_high_behind=int(p('lidar_max_high_behind', 10).value),
            max_range_m=float(p('lidar_max_range_m', 2.5).value))
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.intrinsics = None
        self.paused = False
        self.last = self.lidar_last = 0.0
        self.frame = 0
        self.version, self.sent_version = 0, -1
        self.lidar_count = 0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, 1)
        self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Bool, p('pause_topic', '/roller_mode').value, self._on_pause, latched)
        if self.use_lidar:
            self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, qos_profile_sensor_data)
        self.points_pub = self.create_publisher(String, '/roller/points', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/roller/markers', latched)
        self.create_timer(1.0, self._publish)
        self.get_logger().info(f'roller detector on {self.depth_topic} at {self.rate_hz:g} Hz'
                               + (f', confirmed by {self.cloud_topic}' if self.use_lidar else ' (no lidar)'))

    def _on_info(self, msg):
        if msg.k[0] > 0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_pause(self, msg):
        if bool(msg.data) != self.paused:
            self.paused = bool(msg.data)
            self.get_logger().info('Crossing a roller: detection paused.' if self.paused else 'Detecting again.')

    def _lookup(self, frame_id):
        try:
            return self.tf.lookup_transform(self.global_frame, frame_id, Time(), Duration(seconds=0.2))
        except Exception as e:  # noqa: BLE001 - TF not ready yet
            self.get_logger().warn(f'no TF {self.global_frame} <- {frame_id}: {e}', throttle_duration_sec=5.0)
            return None

    def _on_depth(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.paused or now - self.last < 1.0 / self.rate_hz:
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
        t = self._lookup(msg.header.frame_id)
        if t is None:
            return
        tr = t.transform.translation
        pts = sc.depth_to_points(depth, k, self.params.stride)
        pts = pts @ quat_to_matrix(t.transform.rotation).T + np.array([tr.x, tr.y, tr.z])
        floor_z = self.floor.update(sc.find_levels(pts, (tr.x, tr.y)))
        if floor_z is None:
            return
        det = rc.detect_roller(pts, (tr.x, tr.y), floor_z, self.params)
        if det is None:
            return
        self.frame += 1
        if self.rollers.add(det, self.frame):
            self.version += 1
            cx, cy = det.centre
            self.get_logger().info(f'roller seen: {len(self.rollers)} spots, {det.length_m:.2f} m long, top '
                                   f'{det.top_m:.2f} m above the floor, at ({cx:.2f}, {cy:.2f}), across it towards '
                                   f'{math.degrees(math.atan2(det.normal[1], det.normal[0])):+.0f} deg; '
                                   f'lidar confirms {self._lidar_confirmed()} spot(s)')

    def _on_cloud(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.paused or now - self.lidar_last < 1.0 / self.lidar_hz or self.floor.value is None:
            return                                     # the floor height comes from the depth camera
        self.lidar_last = now
        t = self._lookup(msg.header.frame_id)
        if t is None:
            return
        r = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts = np.stack([r['x'], r['y'], r['z']], axis=1).astype(float)
        pts = pts[np.linalg.norm(pts, axis=1) > 0.05]                 # Livox reports no-return as (0, 0, 0)
        tr = t.transform.translation
        pts = pts @ quat_to_matrix(t.transform.rotation).T + np.array([tr.x, tr.y, tr.z])
        self.lidar.add(pts, (tr.x, tr.y), self.floor.value)
        if len(self.rollers):
            n = self._lidar_confirmed()
            if n != self.lidar_count:
                self.lidar_count = n
                self.version += 1
                self.get_logger().info(f'lidar confirms {n} of {len(self.rollers)} roller spot(s) (low returns only, '
                                       f'open behind)')

    def _spots(self):
        base = self.rollers.confirmed()
        if not self.use_lidar or len(base) == 0:
            return base
        lidar_xy = self.lidar.confirmed(near_xy=base[:, :2], radius_m=self.lidar_match_m + 0.05)
        return self.rollers.confirmed(lidar_xy, self.lidar_match_m)

    def _lidar_confirmed(self):
        return int((self._spots()[:, 4] > 0).sum())

    def _publish(self):
        if self.version == self.sent_version:
            return
        self.sent_version = self.version
        pts = self._spots()
        self.points_pub.publish(String(data=json.dumps({'points': [[round(float(v), 3) for v in r] for r in pts]})))
        m = Marker()
        m.header.frame_id = self.global_frame
        m.ns, m.id, m.type, m.action = 'roller', 0, Marker.CUBE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.08, 0.08, 0.17                 # roller-high block per spot
        m.color.r, m.color.g, m.color.b, m.color.a = 0.95, 0.95, 0.95, 0.8
        m.points = [Point(x=float(x), y=float(y), z=0.085) for x, y, *_ in pts]
        lid = Marker()
        lid.header.frame_id = self.global_frame
        lid.ns, lid.id, lid.type, lid.action = 'roller', 1, Marker.SPHERE_LIST, Marker.ADD
        lid.pose.orientation.w = 1.0
        lid.scale.x = lid.scale.y = lid.scale.z = 0.05                    # green dot: the lidar confirms this spot
        lid.color.r, lid.color.g, lid.color.b, lid.color.a = 0.1, 0.9, 0.2, 1.0
        lid.points = [Point(x=float(x), y=float(y), z=0.20) for x, y, _, _, n in pts if n > 0]
        self.marker_pub.publish(MarkerArray(markers=[m, lid]))


def main(args=None):
    rclpy.init(args=args)
    node = RollerDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
