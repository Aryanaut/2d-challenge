"""Optional curtain detector: finds hanging curtains in the depth image, at a low rate, for the mission manager.

  sub  /camera/depth/image_raw      depth image (16UC1 millimetres or 32FC1 metres)
  sub  /camera/depth/camera_info    intrinsics (else depth_hfov_deg is used)
  sub  <cloud_topic>                3D lidar pointcloud (default /livox/points; use_lidar): a curtain edge returns only floor-level
                                    points while a standard wall (0.5-0.6 m) returns them up to its top (curtain_core.LidarCurtainMap)
  sub  <colour_topic>               front colour camera (use_colour, default off): a pale, smooth, upright surface above the floor,
                                    as bright as the floor = the fabric curtain (curtain_colour.py); needs the depth image too
  pub  /curtain/points   std_msgs/String   JSON {"points": [[x, y, ux, uy], ...]} in the map frame (latched): confirmed
                                           curtain spots and the unit direction the camera was looking at each
  pub  /curtain/markers  MarkerArray       the remembered curtain, for RViz
  srv  ~/detect          std_srvs/Trigger  run the depth (+ colour) checks on the latest depth frame NOW (ignores process_hz) and
                                           reply with JSON in `message`: {found, methods, spots, nearest: {x, y, range_m,
                                           bearing_deg}} (found = this frame showed a curtain; spots/nearest = everything confirmed)

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
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explorer import curtain_colour as ccol
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
        self.top_height_m = float(p('curtain_top_height_m', 1.0).value)    # a wall whose top is this high is a curtain; 0 = off
        self.top_tol_m = float(p('curtain_top_tol_m', 0.15).value)
        self.floor_z = float(p('floor_z', 0.0).value)                     # map-frame z of the floor
        self.curtain = cc.CurtainMap(cell_m=float(p('cell_m', 0.05).value), confirm_hits=int(p('confirm_hits', 3).value))
        self.use_lidar = bool(p('use_lidar', True).value)
        self.cloud_topic = p('cloud_topic', '/livox/points').value
        self.lidar_hz = float(p('lidar_hz', 2.0).value)
        self.lidar_last = 0.0
        self.lidar = cc.LidarCurtainMap(
            cell_m=float(p('lidar_cell_m', 0.05).value), low_max_m=float(p('lidar_low_max_m', 0.20).value),
            high_min_m=float(p('lidar_high_min_m', 0.30).value), min_low=int(p('lidar_min_low', 20).value),
            min_high_behind=int(p('lidar_min_high_behind', 10).value), floor_z=self.floor_z)
        # colour camera check (curtain_colour.py): off by default
        self.use_colour = bool(p('use_colour', False).value)
        self.colour_topic = p('colour_topic', '/camera/color/image_raw/compressed').value
        self.colour_info_topic = p('colour_info_topic', '/camera/color/camera_info').value
        self.colour_max_age = float(p('colour_max_age_s', 0.5).value)
        d = ccol.ColourParams()
        self.colour_params = ccol.ColourParams(
            stride=int(p('colour_stride', d.stride).value),
            max_range_m=float(p('colour_max_range_m', d.max_range_m).value),
            min_height_m=float(p('colour_min_height_m', d.min_height_m).value),
            max_texture=float(p('colour_max_texture', d.max_texture).value),
            max_saturation=float(p('colour_max_saturation', d.max_saturation).value),
            min_rel_brightness=float(p('colour_min_rel_brightness', d.min_rel_brightness).value),
            min_area_m2=float(p('colour_min_area_m2', d.min_area_m2).value),
            min_fraction=float(p('colour_min_fraction', d.min_fraction).value))
        self.colour_cell_m = self.curtain.cell_m
        self.colour_msg = None
        self.colour_k = None
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
        if self.use_colour:
            ctype = CompressedImage if self.colour_topic.endswith('/compressed') else Image
            self.create_subscription(ctype, self.colour_topic, self._on_colour, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, self.colour_info_topic, self._on_colour_info, 1)
        if self.use_lidar:
            self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, qos_profile_sensor_data)
        self.depth_msg = None                                  # latest depth frame, for ~/detect
        self.create_service(Trigger, '~/detect', self._on_detect)
        self.create_timer(1.0, self._publish)
        self.get_logger().info(f'curtain detector on {self.depth_topic} at {self.rate_hz:g} Hz'
                               + (f', colour check on {self.colour_topic}' if self.use_colour else ''))

    def _on_info(self, msg):
        if msg.k[0] > 0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_colour(self, msg):
        self.colour_msg = msg                                  # decoded only when a depth frame is processed

    def _on_colour_info(self, msg):
        if msg.k[0] > 0:
            self.colour_k = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _colour_rgb(self):
        """Latest colour image as HxWx3 RGB uint8, or None (missing / too old / unknown encoding)."""
        m = self.colour_msg
        if m is None:
            return None
        age = (self.get_clock().now() - Time.from_msg(m.header.stamp)).nanoseconds * 1e-9
        if age > self.colour_max_age:
            return None
        if isinstance(m, CompressedImage):
            import cv2                                         # only needed for the compressed topic
            bgr = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
            return None if bgr is None else bgr[..., ::-1]
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
        if m.encoding == 'rgb8':
            return img
        if m.encoding == 'bgr8':
            return img[..., ::-1]
        return None

    def _colour_check(self, depth, msg, k):
        """Colour + depth curtain test (curtain_colour.py); adds the curtain to self.curtain."""
        rgb = self._colour_rgb()
        if rgb is None or self.colour_k is None:
            self.get_logger().warn('colour check: no recent colour image / camera_info', throttle_duration_sec=10.0)
            return False
        try:
            tc = self.tf.lookup_transform(self.colour_msg.header.frame_id, msg.header.frame_id, Time(),
                                          Duration(seconds=0.2))
            tm = self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time(), Duration(seconds=0.2))
        except Exception as e:  # noqa: BLE001 - TF not ready yet
            self.get_logger().warn(f'colour check: no TF: {e}', throttle_duration_sec=5.0)
            return False
        r_dc = quat_to_matrix(tc.transform.rotation)
        t_dc = (tc.transform.translation.x, tc.transform.translation.y, tc.transform.translation.z)
        rot = quat_to_matrix(tm.transform.rotation)
        tr = tm.transform.translation
        up = rot.T @ np.array([0.0, 0.0, 1.0])                 # map 'up' in the depth optical frame
        det = ccol.detect_curtain_colour(depth, rgb, k, self.colour_k, (r_dc, t_dc), up, self.colour_params)
        if det is None:
            return False
        pts = det.points @ rot.T + np.array([tr.x, tr.y, tr.z])
        pts = ccol.one_per_cell(pts[:, :2], self.colour_cell_m)   # one hit per spot per image
        if self.curtain.add(pts, (tr.x, tr.y)):
            self.version += 1
            self.get_logger().info(f'curtain seen by the colour camera: {det.area_m2:.2f} m2 '
                                   f'({100 * det.fraction:.0f} % of the image), {float(np.median(det.ranges)):.2f} m '
                                   f'ahead; {len(self.curtain)} spots')
        return True

    def _on_depth(self, msg):
        self.depth_msg = msg
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last < 1.0 / self.rate_hz:
            return
        self.last = now
        self._process_depth(msg)

    def _process_depth(self, msg):
        """Run the depth / colour checks on one frame. Returns (methods that saw a curtain, camera pose (x, y, heading) or None)."""
        methods, pose = [], None
        if msg.encoding == '16UC1':
            depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width) * self.depth_scale
        elif msg.encoding == '32FC1':
            depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        else:
            return methods, pose
        fx = fy = (msg.width / 2) / math.tan(self.hfov / 2)
        k = self.intrinsics or (fx, fy, msg.width / 2, msg.height / 2)
        if self.use_colour and self._colour_check(depth, msg, k):
            methods.append('colour')
        det = cc.detect_curtain(depth, self.params)
        top = cc.detect_wall_top(depth, self.params) if self.top_height_m > 0 else None
        if det is None and top is None:
            return methods, pose
        try:
            t = self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time(), Duration(seconds=0.2))
        except Exception as e:  # noqa: BLE001 - TF not ready yet
            self.get_logger().warn(f'no TF {self.global_frame} <- {msg.header.frame_id}: {e}', throttle_duration_sec=5.0)
            return methods, pose
        tr = t.transform.translation
        rot = quat_to_matrix(t.transform.rotation)
        pose = (tr.x, tr.y, math.atan2(rot[1, 2], rot[0, 2]))   # camera optical z (forward) in the map
        if det is not None:
            methods.append('depth_hem')
            pts = cc.project_to_map(det, k, rot, (tr.x, tr.y, tr.z))
            if self.curtain.add(pts, (tr.x, tr.y)):
                self.version += 1
                self.get_logger().info(f'curtain seen: {len(self.curtain)} spots, {len(det.columns)} columns now, '
                                       f'{float(np.median(det.ranges)):.2f} m ahead')
        if top is not None:
            # a wall whose top edge is ~1 m above the floor is a curtain (curtains hang from a bar at that height)
            p3 = cc.project_to_map_3d(top, k, rot, (tr.x, tr.y, tr.z))
            z = p3[:, 2] - self.floor_z
            ok = np.abs(z - self.top_height_m) <= self.top_tol_m
            if ok.sum() >= self.params.min_columns * max(1, len(top.columns)):
                methods.append('wall_top')
            if ok.sum() >= self.params.min_columns * max(1, len(top.columns)) and self.curtain.add(p3[ok, :2], (tr.x, tr.y)):
                self.version += 1
                self.get_logger().info(f'curtain seen (wall top at {float(np.median(z[ok])):.2f} m): {len(self.curtain)} spots')
        return methods, pose

    def _on_cloud(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.lidar_last < 1.0 / self.lidar_hz:
            return
        self.lidar_last = now
        try:
            t = self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time(), Duration(seconds=0.2))
        except Exception as e:  # noqa: BLE001 - TF not ready yet
            self.get_logger().warn(f'no TF {self.global_frame} <- {msg.header.frame_id}: {e}', throttle_duration_sec=5.0)
            return
        r = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts = np.stack([r['x'], r['y'], r['z']], axis=1).astype(float)
        tr = t.transform.translation
        pts = pts @ quat_to_matrix(t.transform.rotation).T + np.array([tr.x, tr.y, tr.z])
        before = len(self.lidar)
        self.lidar.add(pts, (tr.x, tr.y))
        if len(self.lidar) != before:
            self.version += 1
            self.get_logger().info(f'curtain seen by the lidar (floor-level returns, no wall height): '
                                   f'{len(self.lidar)} spots')

    def _on_detect(self, request, response):
        """~/detect: process the latest depth frame now, then report this frame's result and the confirmed curtain."""
        msg = self.depth_msg
        if msg is None:
            response.success, response.message = False, json.dumps({'error': 'no depth image received yet'})
            return response
        methods, pose = self._process_depth(msg)
        self.last = self.get_clock().now().nanoseconds * 1e-9   # the periodic check need not repeat this frame
        pts = self._all_points()
        out = {'found': bool(methods), 'methods': methods, 'spots': int(len(pts)), 'nearest': None}
        if pose is None:                                         # no detection this frame: still locate the robot for 'nearest'
            try:
                t = self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time(), Duration(seconds=0.2))
                rot = quat_to_matrix(t.transform.rotation)
                pose = (t.transform.translation.x, t.transform.translation.y, math.atan2(rot[1, 2], rot[0, 2]))
            except Exception as e:  # noqa: BLE001 - TF not ready yet
                out['warning'] = f'no TF {self.global_frame} <- {msg.header.frame_id}'
                self.get_logger().warn(f'~/detect: {e}', throttle_duration_sec=5.0)
        if pose is not None and len(pts):
            d = pts[:, :2] - np.array(pose[:2])
            i = int(np.argmin(np.hypot(d[:, 0], d[:, 1])))
            bearing = math.degrees((math.atan2(d[i, 1], d[i, 0]) - pose[2] + math.pi) % (2 * math.pi) - math.pi)
            out['nearest'] = {'x': round(float(pts[i, 0]), 3), 'y': round(float(pts[i, 1]), 3),
                              'range_m': round(float(np.hypot(*d[i])), 2), 'bearing_deg': round(bearing, 1)}
        response.success, response.message = True, json.dumps(out)
        return response

    def _all_points(self):
        return np.vstack([self.curtain.confirmed(), self.lidar.confirmed()]) if self.use_lidar else self.curtain.confirmed()

    def _publish(self):
        if self.version == self.sent_version:
            return
        self.sent_version = self.version
        pts = self._all_points()
        self.points_pub.publish(String(data=json.dumps({'points': [[round(float(v), 3) for v in r] for r in pts]})))
        m = Marker()
        m.header.frame_id = self.global_frame
        m.ns, m.id, m.type, m.action = 'curtain', 0, Marker.CUBE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.06, 0.06, 0.6          # a curtain-high strip of magenta per confirmed spot
        m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.2, 0.9, 0.6
        m.points = [Point(x=float(x), y=float(y), z=0.3) for x, y, _, _ in pts]
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
