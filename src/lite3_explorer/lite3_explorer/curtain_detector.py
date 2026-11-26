"""Optional curtain detector: finds hanging curtains in the depth image, at a low rate, for the mission manager.

  sub  /camera/depth/image_raw      depth image (16UC1 millimetres or 32FC1 metres)
  sub  /camera/depth/camera_info    intrinsics (else depth_hfov_deg is used)
  sub  <cloud_topic>                3D lidar pointcloud (default /livox/points; use_lidar): a curtain edge returns only floor-level
                                    points while a standard wall (0.5-0.6 m) returns them up to its top (curtain_core.LidarCurtainMap)
  sub  <colour_topic>               front colour camera (use_colour, default off): a pale, smooth, upright surface above the floor,
                                    as bright as the floor = the fabric curtain (curtain_colour.py); needs the depth image too
  pub  /curtain/points   std_msgs/String   JSON {"points": [[x, y, ux, uy], ...], "camera": n} in the map frame (latched):
                                           confirmed curtain spots and the unit direction the camera was looking at each;
                                           the first n rows are camera spots (depth hem / colour), the rest lidar-only
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
import threading
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from std_msgs.msg import Empty, String
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
        self.curtain = cc.CurtainMap(cell_m=float(p('cell_m', 0.05).value), confirm_hits=int(p('confirm_hits', 3).value),
                                    min_view_sep_m=float(p('min_view_sep_m', 0.15).value))
        self.use_lidar = bool(p('use_lidar', True).value)
        self.cloud_topic = p('cloud_topic', '/livox/points').value
        self.lidar_hz = float(p('lidar_hz', 2.0).value)
        self.lidar_count = 0                                   # confirmed lidar spots at the last scan
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
            min_rel_warmth=(lambda v: None if v < 0 else v)(float(p('colour_min_rel_warmth', d.min_rel_warmth).value)),
            min_area_m2=float(p('colour_min_area_m2', d.min_area_m2).value),
            min_fraction=float(p('colour_min_fraction', d.min_fraction).value),
            extend_min_m=float(p('colour_extend_min_m', d.extend_min_m).value),
            extend_max_m=float(p('colour_extend_max_m', d.extend_max_m).value),
            min_extend_m2=float(p('colour_min_extend_m2', d.min_extend_m2).value),
            extend_tol_m=float(p('colour_extend_tol_m', d.extend_tol_m).value),
            open_cells=int(p('colour_open_cells', d.open_cells).value),
            require_floor=bool(p('colour_require_floor', d.require_floor).value))
        self.colour_cell_m = self.curtain.cell_m
        self.colour_k = None
        self.tf = Buffer()
        # /tf on its OWN node and thread: on this node's executor /tf (~57 Hz) starves behind the depth / colour / cloud
        # callbacks (~0.8 s of work a second). TransformListener(buf, self, spin_thread=True) does NOT avoid that: in
        # humble it adds this whole node to a second executor, which runs the heavy callbacks too. Bag fullrun_25_11_215528:
        # the buffer ran 5-14 s behind (curtain placed 35 deg off); fullrun_25_11_233315: up to 40 s behind, and as the
        # map lookups below are made at the sensor's stamp (a stale buffer skips the frame instead of placing it with an
        # old pose) every single frame of the run was skipped: the curtain was never looked for.
        self.tf_node = rclpy.create_node('curtain_detector_tf', start_parameter_services=False)
        self.tf_listener = TransformListener(self.tf, self.tf_node, spin_thread=True)
        self.intrinsics = None
        self.version, self.sent_version = 0, -1
        # The sensor callbacks only keep the NEWEST message (queue 1) and return; the work (hem ~0.25 s, colour ~0.05 s,
        # lidar map up to ~0.15 s a frame) runs in timers on their own threads. Before, all of it ran in the callbacks
        # on one thread, ~0.9 s of work a second: the queues (5 deep) backed up, the oldest frame was processed, and the
        # colour image paired with a depth frame was up to 5.8 s away from it - skipped (bag fuck_26_11_212613, 277 s:
        # the only drive past the curtain, camera spots 0 all run). The colour frame is now the one NEAREST the depth
        # frame's stamp from a short buffer, not the last one received.
        self.lock = threading.Lock()                           # self.curtain / self.lidar / self.version
        self.rx_group = ReentrantCallbackGroup()
        self.depth_group = MutuallyExclusiveCallbackGroup()    # depth timer + ~/detect: one frame at a time
        self.lidar_group = MutuallyExclusiveCallbackGroup()
        newest = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.depth_msg, self.depth_done = None, None           # latest depth frame (also for ~/detect), last one processed
        self.cloud_msg = None
        self.colour_buf = deque(maxlen=max(1, int(p('colour_buffer', 12).value)))
        self.create_subscription(CameraInfo, self.info_topic, self._on_info, 1, callback_group=self.rx_group)
        self.create_subscription(Image, self.depth_topic, self._on_depth, newest, callback_group=self.rx_group)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.points_pub = self.create_publisher(String, '/curtain/points', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/curtain/markers', latched)
        if self.use_colour:
            ctype = CompressedImage if self.colour_topic.endswith('/compressed') else Image
            # not queue 1: the colour camera runs faster than depth, keep the frames around the depth stamp
            self.create_subscription(ctype, self.colour_topic, self._on_colour, qos_profile_sensor_data,
                                     callback_group=self.rx_group)
            self.create_subscription(CameraInfo, self.colour_info_topic, self._on_colour_info, 1,
                                     callback_group=self.rx_group)
        if self.use_lidar:
            self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, newest, callback_group=self.rx_group)
            self.create_timer(1.0 / max(self.lidar_hz, 0.1), self._tick_lidar, callback_group=self.lidar_group)
        self.create_subscription(Empty, '/map_reset', self._on_map_reset, 10, callback_group=self.depth_group)
        self.create_timer(1.0 / max(self.rate_hz, 0.1), self._tick_depth, callback_group=self.depth_group)
        self.create_service(Trigger, '~/detect', self._on_detect, callback_group=self.depth_group)
        self.create_timer(1.0, self._publish)
        self.get_logger().info(f'curtain detector on {self.depth_topic} at {self.rate_hz:g} Hz'
                               + (f', colour check on {self.colour_topic}' if self.use_colour else ''))

    def _on_info(self, msg):
        if msg.k[0] > 0:
            self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _on_colour(self, msg):
        self.colour_buf.append(msg)                            # decoded only when a depth frame is processed

    def _on_colour_info(self, msg):
        if msg.k[0] > 0:
            self.colour_k = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def _map_tf(self, msg):
        """global_frame <- msg's frame at the moment the sensor took it (raises when TF has no pose for then). The
        depth frames are ~0.25 s and the lidar clouds ~0.9 s older than the newest TF, so the pose is normally there;
        a lookup at Time() instead silently returned whatever pose the buffer had last, however old."""
        return self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                                        Duration(seconds=0.2))

    def _colour_rgb(self, ref_stamp):
        """The buffered colour image nearest the depth frame's stamp (ref_stamp) as (HxWx3 RGB uint8, msg, ''), or
        (None, None, why) (none / too far from the depth frame / unknown encoding). Judged against the depth frame's
        stamp, not the clock now: the frames arrive 0.15-0.5 s old over the domain bridge."""
        buf = list(self.colour_buf)
        if not buf:
            return None, None, 'no colour image received'
        ref = Time.from_msg(ref_stamp)
        m = min(buf, key=lambda c: abs((ref - Time.from_msg(c.header.stamp)).nanoseconds))
        age = abs((ref - Time.from_msg(m.header.stamp)).nanoseconds) * 1e-9
        if age > self.colour_max_age:
            return None, None, f'colour image {age:.2f} s from the depth frame (colour_max_age_s {self.colour_max_age:g})'
        if isinstance(m, CompressedImage):
            import cv2                                         # only needed for the compressed topic
            bgr = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
            return (None, None, 'compressed colour image did not decode') if bgr is None else (bgr[..., ::-1], m, '')
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
        if m.encoding == 'rgb8':
            return img, m, ''
        if m.encoding == 'bgr8':
            return img[..., ::-1], m, ''
        return None, None, f'colour encoding {m.encoding} (rgb8 / bgr8 / compressed only)'

    def _colour_check(self, depth, msg, k):
        """Colour + depth curtain test (curtain_colour.py); adds the curtain to self.curtain."""
        rgb, cmsg, why = self._colour_rgb(msg.header.stamp)
        if rgb is None or self.colour_k is None:
            why = why or f'no camera_info on {self.colour_info_topic}'
            self.get_logger().warn(f'colour check skipped: {why}', throttle_duration_sec=10.0)
            return False
        try:
            tc = self.tf.lookup_transform(cmsg.header.frame_id, msg.header.frame_id, Time(),
                                          Duration(seconds=0.2))
            tm = self._map_tf(msg)
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
        with self.lock:
            newly = self.curtain.add(pts, (tr.x, tr.y))
            if newly:
                self.version += 1
        if newly:
            self.get_logger().info(f'curtain seen by the colour camera: {det.area_m2:.2f} m2 '
                                   f'({100 * det.fraction:.0f} % of the image, {det.extend_m2:.2f} m2 above the walls), '
                                   f'{float(np.median(det.ranges)):.2f} m '
                                   f'ahead; {len(self.curtain)} spots')
        return True

    def _on_depth(self, msg):
        self.depth_msg = msg                                   # processed by _tick_depth (process_hz)

    def _tick_depth(self):
        msg = self.depth_msg
        if msg is None or msg is self.depth_done:
            return
        self.depth_done = msg
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
            t = self._map_tf(msg)
        except Exception as e:  # noqa: BLE001 - TF not ready yet
            self.get_logger().warn(f'no TF {self.global_frame} <- {msg.header.frame_id}: {e}', throttle_duration_sec=5.0)
            return methods, pose
        tr = t.transform.translation
        rot = quat_to_matrix(t.transform.rotation)
        pose = (tr.x, tr.y, math.atan2(rot[1, 2], rot[0, 2]))   # camera optical z (forward) in the map
        if det is not None:
            methods.append('depth_hem')
            pts = cc.project_to_map(det, k, rot, (tr.x, tr.y, tr.z))
            with self.lock:
                newly = self.curtain.add(pts, (tr.x, tr.y))
                if newly:
                    self.version += 1
            if newly:
                self.get_logger().info(f'curtain seen: {len(self.curtain)} spots, {len(det.columns)} columns now, '
                                       f'{float(np.median(det.ranges)):.2f} m ahead')
        if top is not None:
            # a wall whose top edge is ~1 m above the floor is a curtain (curtains hang from a bar at that height)
            p3 = cc.project_to_map_3d(top, k, rot, (tr.x, tr.y, tr.z))
            z = p3[:, 2] - self.floor_z
            ok = np.abs(z - self.top_height_m) <= self.top_tol_m
            if ok.sum() >= self.params.min_columns * max(1, len(top.columns)):
                methods.append('wall_top')
            if ok.sum() >= self.params.min_columns * max(1, len(top.columns)):
                with self.lock:
                    newly = self.curtain.add(p3[ok, :2], (tr.x, tr.y))
                    if newly:
                        self.version += 1
                if newly:
                    self.get_logger().info(f'curtain seen (wall top at {float(np.median(z[ok])):.2f} m): '
                                           f'{len(self.curtain)} spots')
        return methods, pose

    def _on_cloud(self, msg):
        self.cloud_msg = msg                                   # processed by _tick_lidar (lidar_hz)

    def _tick_lidar(self):
        msg = self.cloud_msg
        if msg is None:
            return
        self.cloud_msg = None
        try:
            t = self._map_tf(msg)
        except Exception as e:  # noqa: BLE001 - TF not ready yet
            self.get_logger().warn(f'no TF {self.global_frame} <- {msg.header.frame_id}: {e}', throttle_duration_sec=5.0)
            return
        r = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        pts = np.stack([r['x'], r['y'], r['z']], axis=1).astype(float)
        tr = t.transform.translation
        pts = pts @ quat_to_matrix(t.transform.rotation).T + np.array([tr.x, tr.y, tr.z])
        with self.lock:
            self.lidar.add(pts, (tr.x, tr.y))
            n = len(self.lidar)                                # confirmed() once per scan (cached for _publish)
            changed = n != self.lidar_count
            self.lidar_count = n
            if changed:
                self.version += 1
        if changed:
            self.get_logger().info(f'curtain seen by the lidar (floor-level returns, no wall height): {n} spots')

    def _on_map_reset(self, _msg):
        """A new map (new map frame) starts after the stairs: the curtain spots seen so far are in the old one. (The
        mission manager carries its own memory of the curtain into the new frame.)"""
        with self.lock:
            self._reset_maps()

    def _reset_maps(self):
        c, lid = self.curtain, self.lidar
        self.lidar_count = 0
        self.curtain = cc.CurtainMap(cell_m=c.cell_m, confirm_hits=c.confirm_hits, min_view_sep_m=c.min_view_sep_m)
        self.lidar = cc.LidarCurtainMap(cell_m=lid.cell_m, low_max_m=lid.low_max, high_min_m=lid.high_min,
                                        max_high_frac=lid.max_high_frac, min_low=lid.min_low,
                                        min_high_behind=lid.min_high_behind, behind_m=lid.behind_m,
                                        min_range_m=lid.min_range, max_range_m=lid.max_range, floor_z=lid.floor_z,
                                        near_m=lid.near_cells * lid.cell_m)
        self.version += 1
        self.get_logger().info('new map: curtain spots forgotten.')

    def _on_detect(self, request, response):
        """~/detect: process the latest depth frame now, then report this frame's result and the confirmed curtain."""
        msg = self.depth_msg
        if msg is None:
            response.success, response.message = False, json.dumps({'error': 'no depth image received yet'})
            return response
        methods, pose = self._process_depth(msg)
        self.depth_done = msg                                    # the periodic check need not repeat this frame
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
        with self.lock:
            cam = self.curtain.confirmed()
            return np.vstack([cam, self.lidar.confirmed()]) if self.use_lidar else cam

    def _publish(self):
        with self.lock:
            if self.version == self.sent_version:
                return
            self.sent_version = self.version
            cam = self.curtain.confirmed()
            lid = self.lidar.confirmed() if self.use_lidar else np.zeros((0, 4))
        pts = np.vstack([cam, lid])
        self.points_pub.publish(String(data=json.dumps({'points': [[round(float(v), 3) for v in r] for r in pts],
                                                        'camera': int(len(cam))})))
        markers = []
        # a curtain-high strip per confirmed spot: magenta = camera (depth hem / colour), orange = lidar only
        for i, (ns, rows, rgb) in enumerate((('curtain_camera', cam, (0.9, 0.2, 0.9)),
                                             ('curtain_lidar', lid, (1.0, 0.55, 0.0)))):
            m = Marker()
            m.header.frame_id = self.global_frame
            m.ns, m.id, m.type = ns, i, Marker.CUBE_LIST
            m.action = Marker.ADD if len(rows) else Marker.DELETE
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = 0.06, 0.06, 0.6
            m.color.r, m.color.g, m.color.b, m.color.a = rgb[0], rgb[1], rgb[2], 0.6
            m.points = [Point(x=float(r[0]), y=float(r[1]), z=0.3) for r in rows]
            markers.append(m)
        self.marker_pub.publish(MarkerArray(markers=markers))


def main(args=None):
    rclpy.init(args=args)
    node = CurtainDetector()
    executor = MultiThreadedExecutor(num_threads=4)          # sensor callbacks / depth timer / lidar timer / publish
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.tf_node.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
