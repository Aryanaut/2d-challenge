"""Classify the terrain around the Lite3 as rugged / non_rugged.

Primary sensor: Livox MID-360 point cloud (/livox/points). The cloud is moved into
base_link with TF, a few frames are stacked (motion compensated through odom when that TF
exists), and the box around the robot is checked for steps, bumps and holes.

Verification: Orbbec depth image (/camera/depth/image_raw). Only the bottom part of the
frame is used (depth.roi_top_fraction), because the camera is pitched down and that part
shows the ground just ahead of the robot.

Slats are found with Hough line transforms on the depth and colour images (slat_lines.py).

Output: /terrain_type (std_msgs/String, latched): "rugged", "non_rugged" or "unknown".
/terrain_kind (JSON, latched): the terrain KIND of the floor around the robot (flat / uneven / slats / ...) from the
configurable strategies in terrain_kinds.py (config/terrain_kinds.yaml). It knows nothing about tiles: whoever calls
~/classify (the explorer, when it looks at a tile) matches the answer to the place it was looking at via "seq".
Optional (elevation.enabled): a ground reference plane is fitted from the lidar floor while the robot is assumed to stand
on flat floor at the start; /terrain_kind then also carries elevation_m (+ higher / - lower than that floor),
elevation_class (higher / lower / level) and slope_deg of the patch (elevation.py).
Details for tuning: ~/details (JSON string), ~/lidar_points and ~/depth_points
(PointCloud2, intensity = height above the fitted ground plane in metres).

The node only reads sensors, so it works the same whether the dog is driven with the
remote (manual) or by Nav2 / the frontier explorer (auto). Pause it with
  ros2 service call /terrain_classifier/set_enabled std_srvs/srv/SetBool "{data: false}"
"""
from collections import deque
from dataclasses import fields
import json
import os

import cv2
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from lite3_terrain_classifier import elevation as elev
from lite3_terrain_classifier import terrain_core as tc
from lite3_terrain_classifier.terrain_kinds import KindClassifier
from lite3_terrain_classifier import slat_lines
from lite3_terrain_classifier.msg_utils import cloud_to_xyz, depth_to_metres, image_to_gray8, xyzi_cloud

# Per-sensor defaults that differ from RoughnessParams (the depth camera is noisier with range).
LIDAR_DEFAULTS = {}
DEPTH_DEFAULTS = {'cell_spread_m': 0.10, 'min_points': 300, 'min_points_per_cell': 10}


class TerrainClassifier(Node):
    def __init__(self):
        super().__init__('terrain_classifier')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731

        # ---- general ----
        self.base_frame = p('base_frame', 'base_link')
        self.odom_frame = p('odom_frame', 'odom')
        self.terrain_topic = p('terrain_topic', '/terrain_type')
        rate_hz = p('rate_hz', 5.0)
        self.fusion_mode = p('fusion_mode', 'lidar_primary')
        if self.fusion_mode not in tc.FUSION_MODES:
            raise ValueError(f'fusion_mode must be one of {tc.FUSION_MODES}')
        self.depth_fallback = p('depth_fallback', True)
        self.flat_needs_lidar = p('flat_needs_lidar', True)     # a depth-only flat reading cannot leave RUGGED
        self.use_depth = p('use_depth', True)
        self.publish_debug_clouds = p('publish_debug_clouds', True)
        self.enabled = p('start_enabled', True)
        self.debounce = tc.Debouncer(p('confirm_frames', 3))
        # trigger_mode: classify only when ~/classify is called (e.g. by the mission manager on entering a
        # new tile), for trigger_duration_s, then hold the last label. One run also happens at startup.
        self.trigger_mode = p('trigger_mode', False)
        self.trigger_duration_s = p('trigger_duration_s', 1.5)
        self.active_until_s = None      # trigger_mode: classify until this time (None = idle)

        # ---- lidar ----
        self.cloud_topic = p('lidar.topic', '/livox/points')
        self.accumulate_s = p('lidar.accumulate_s', 0.3)
        self.lidar_max_age_s = p('lidar.max_age_s', 1.0)
        self.lidar_box = self._box('lidar', (-1.5, 2.5), (-1.5, 1.5), (-1.0, 0.4))
        self.body_x = (p('lidar.body_x_min', -0.45), p('lidar.body_x_max', 0.45))
        self.body_y = (p('lidar.body_y_min', -0.30), p('lidar.body_y_max', 0.30))
        self.lidar_params = self._roughness('lidar', LIDAR_DEFAULTS)

        # ---- depth camera ----
        self.depth_topic = p('depth.image_topic', '/camera/depth/image_raw')
        self.info_topic = p('depth.info_topic', '/camera/depth/camera_info')
        self.mm_scale = p('depth.mm_scale', 0.001)
        self.roi_top = p('depth.roi_top_fraction', 0.5)
        self.roi_bottom_margin = p('depth.roi_bottom_margin', 0.0)
        self.roi_side_margin = p('depth.roi_side_margin', 0.05)
        self.depth_stride = p('depth.stride', 4)
        self.depth_range = (p('depth.min_range_m', 0.25), p('depth.max_range_m', 2.0))
        self.depth_max_age_s = p('depth.max_age_s', 1.0)
        self.depth_use_tf = p('depth.use_tf', True)
        self.depth_box = self._box('depth', (0.0, 2.5), (-1.5, 1.5), (-1.0, 0.4))
        self.depth_params = self._roughness('depth', DEPTH_DEFAULTS)

        # ---- colour camera (only for the slats lines) ----
        self.use_color = p('use_color', True)
        self.color_topic = p('color.image_topic', '/camera/color/image_raw')
        # the camera often only publishes the compressed colour image: both topics are listened to
        self.color_compressed_topic = p('color.compressed_topic', self.color_topic + '/compressed')
        self.color_max_age_s = p('color.max_age_s', 1.0)
        self.color_msg, self.color_recv_s, self.color_seen = None, None, None

        # ---- elevation against a ground reference taken at the start (optional) ----
        self.elev_enabled = p('elevation.enabled', False)
        self.elev_frame = p('elevation.frame', 'odom')           # odom: absolute | gravity: rotation only (see elevation.py)
        if self.elev_frame not in ('odom', 'gravity'):
            raise ValueError("elevation.frame must be 'odom' or 'gravity'")
        elev_cfg = {k: type(v)(p(f'elevation.{k}', v)) for k, v in elev.DEFAULTS.items()}
        self.elev_ref = elev.GroundReference(elev_cfg)
        self.elev_est = elev.ElevationEstimator(self.elev_ref, elev_cfg)
        self.elev_issue = ''

        # ---- state ----
        self.rng = np.random.default_rng(0)
        self.clouds = deque()           # (recv_s, frame, points)
        self.lidar_issue = f'no cloud on {self.cloud_topic} yet'
        self.depth_msg = None
        self.depth_recv_s = None
        self.camera_info = None
        self.last_lidar = None
        self.last_depth = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.terrain_pub = self.create_publisher(String, self.terrain_topic, latched)
        self.details_pub = self.create_publisher(String, '~/details', 10)
        self.kind_pub = self.create_publisher(String, '/terrain_kind', latched)
        # debug: what the Hough line detection sees (only computed while someone subscribes)
        self.ground_pub = self.create_publisher(String, '~/ground_reference', latched)
        self.slats_depth_pub = self.create_publisher(Image, '~/slats_depth_debug', 2)
        self.slats_color_pub = self.create_publisher(Image, '~/slats_color_debug', 2)
        self.slats_json_pub = self.create_publisher(String, '~/slats_detections', 10)
        self.lidar_dbg_pub = self.create_publisher(PointCloud2, '~/lidar_points', 2)
        self.depth_dbg_pub = self.create_publisher(PointCloud2, '~/depth_points', 2)

        self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, qos_profile_sensor_data)
        if self.use_color:
            self.create_subscription(Image, self.color_topic, self._on_color, qos_profile_sensor_data)
            if self.color_compressed_topic:
                self.create_subscription(CompressedImage, self.color_compressed_topic, self._on_color_compressed,
                                         qos_profile_sensor_data)
        if self.use_depth:
            self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_service(SetBool, '~/set_enabled', self._on_set_enabled)
        self.create_service(Trigger, '~/classify', self._on_classify)
        self.create_service(Trigger, '~/recalibrate_ground', self._on_recalibrate)
        self.create_timer(1.0 / max(rate_hz, 0.1), self._tick)

        # terrain kinds (strategies): which kind of floor is this? independent of any tile grid
        self.kinds_enabled = p('kinds.enabled', True)
        self.kinds_window_s = p('kinds.window_s', 5.0)       # continuous mode: start a new patch this often
        kinds_file = p('kinds_file', '') or os.path.join(
            get_package_share_directory('lite3_terrain_classifier'), 'config', 'terrain_kinds.yaml')
        with open(kinds_file, encoding='utf-8') as fh:
            self.kinds = KindClassifier(yaml.safe_load(fh) or {})
        self.kinds_seq = 0                      # counts triggers: a result belongs to the trigger with this number
        self.kinds_started_s = self._now_s()
        self.kind_last = None
        self.get_logger().info(f'Terrain kinds from {kinds_file}: {[s.name for s in self.kinds.strategies]}')

        self._publish_label(tc.UNKNOWN)
        if self.trigger_mode:
            self.active_until_s = self._now_s() + self.trigger_duration_s
            self._new_patch()
        self.get_logger().info(
            f'{"TRIGGERED (service ~/classify)" if self.trigger_mode else "continuous"}: '
            f'Lidar {self.cloud_topic} (primary), depth {self.depth_topic if self.use_depth else "off"} '
            f'(verify), fusion={self.fusion_mode} -> {self.terrain_topic}')

    # ------------------------------------------------------------------ parameters
    def _box(self, prefix, x, y, z):
        p = lambda name, default: self.declare_parameter(f'{prefix}.{name}', default).value  # noqa: E731
        return ((p('x_min', x[0]), p('x_max', x[1])),
                (p('y_min', y[0]), p('y_max', y[1])),
                (p('z_min', z[0]), p('z_max', z[1])))

    def _roughness(self, prefix, overrides):
        values = {}
        defaults = tc.RoughnessParams()
        for f in fields(tc.RoughnessParams):
            default = overrides.get(f.name, getattr(defaults, f.name))
            values[f.name] = type(default)(self.declare_parameter(f'{prefix}.{f.name}', default).value)
        return tc.RoughnessParams(**values)

    # ------------------------------------------------------------------ helpers
    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _lookup(self, target, source):
        """Latest transform target <- source as (R, t), or None."""
        if not source or source == target:
            return np.eye(3), np.zeros(3)
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time())
        except TransformException:
            return None
        q, t = tf.transform.rotation, tf.transform.translation
        return tc.quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------------ callbacks
    def _on_set_enabled(self, request, response):
        self.enabled = bool(request.data)
        self.debounce.reset()
        self.clouds.clear()
        if self.trigger_mode:
            self.active_until_s = None
        self._publish_label(tc.UNKNOWN)
        response.success = True
        response.message = 'terrain classifier ' + ('enabled' if self.enabled else 'paused')
        self.get_logger().info(response.message)
        return response

    def _new_patch(self):
        """Start collecting a new patch of floor for the terrain kind."""
        self.kinds.reset()
        self.elev_est.reset()
        self.kinds_seq += 1
        self.kinds_started_s = self._now_s()
        self.kind_last = None

    def _on_classify(self, request, response):
        if self.trigger_mode:
            self.active_until_s = self._now_s() + self.trigger_duration_s
            self._new_patch()
            response.message = f'classifying for {self.trigger_duration_s:g} s'
        else:
            response.message = 'continuous mode: already classifying'
        response.success = True
        return response

    def _on_recalibrate(self, request, response):
        """Take a new ground reference NOW: the robot must stand on the floor that is to count as elevation 0."""
        if not self.elev_enabled:
            response.success, response.message = False, 'elevation.enabled is false'
            return response
        self.elev_ref.reset()
        self.elev_est.reset()
        self.elev_issue = ''
        response.success = True
        response.message = f'collecting a new ground reference for {self.elev_ref.cfg["calibrate_s"]:g} s'
        self.get_logger().info(response.message)
        return response

    def _on_color_compressed(self, msg):
        """JPEG / PNG colour frame -> an Image-like record the rest of the code understands (grey only)."""
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        out = Image(header=msg.header, height=img.shape[0], width=img.shape[1], encoding='mono8', is_bigendian=0,
                    step=img.shape[1], data=img.tobytes())
        self._on_color(out)

    def _on_color(self, msg):
        self.color_msg = msg
        self.color_recv_s = self._now_s()

    def _on_info(self, msg):
        self.camera_info = msg

    def _on_depth(self, msg):
        self.depth_msg = msg
        self.depth_recv_s = self._now_s()

    def _on_cloud(self, msg):
        if not self.enabled:
            return
        if self.trigger_mode and (self.active_until_s is None or self._now_s() > self.active_until_s + 0.5) \
                and not self._calibrating():
            return                       # idle: do not even convert the cloud
        recv_s = self._now_s()
        try:
            xyz = cloud_to_xyz(msg)
        except (ValueError, KeyError) as exc:
            self.lidar_issue = f'cannot read {self.cloud_topic}: {exc}'
            return
        to_base = self._lookup(self.base_frame, msg.header.frame_id)
        if to_base is None:
            self.lidar_issue = (f'no TF {self.base_frame} <- {msg.header.frame_id}; '
                                'launch with publish_lidar_tf:=true and the lidar mount pose')
            return
        pts = tc.transform_points(xyz, *to_base)
        # Coarse crop now (generous margin for motion while stacking), exact crop at evaluation.
        (x0, x1), (y0, y1), (z0, z1) = self.lidar_box
        m = 0.5
        pts = tc.crop_box(pts, (x0 - m, x1 + m), (y0 - m, y1 + m), (z0 - m, z1 + m))

        frame = self.base_frame
        if self.accumulate_s > 0 and self.odom_frame:
            to_odom = self._lookup(self.odom_frame, self.base_frame)
            if to_odom is not None:
                pts, frame = tc.transform_points(pts, *to_odom), self.odom_frame
        self.clouds.append((recv_s, frame, pts))
        self.lidar_issue = ''
        while self.clouds and recv_s - self.clouds[0][0] > self.accumulate_s + 1e-3:
            self.clouds.popleft()

    # ------------------------------------------------------------------ assessment
    def _lidar_points(self, now_s):
        if not self.clouds:
            return None, self.lidar_issue or f'no cloud on {self.cloud_topic}'
        newest_s = self.clouds[-1][0]
        if now_s - newest_s > self.lidar_max_age_s:
            return None, f'{self.cloud_topic} is stale ({now_s - newest_s:.1f} s old)'
        chunks = []
        from_odom = None
        for i, (recv_s, frame, pts) in enumerate(self.clouds):
            if frame == self.base_frame:
                # Not motion compensated: only the newest one is usable.
                if i == len(self.clouds) - 1:
                    chunks.append(pts)
                continue
            if from_odom is None:
                from_odom = self._lookup(self.base_frame, frame) or False
            if from_odom:
                chunks.append(tc.transform_points(pts, *from_odom))
        if not chunks:
            return None, f'no TF {self.base_frame} <- {self.odom_frame}'
        pts = np.concatenate(chunks)
        (x, y, z) = self.lidar_box
        return tc.crop_box(pts, x, y, z, self.body_x, self.body_y), ''

    def _assess_lidar(self, now_s):
        pts, issue = self._lidar_points(now_s)
        if pts is None:
            return tc.invalid(issue), None
        result, used, res = tc.assess_terrain(pts, self.lidar_params, up_known=True, rng=self.rng)
        return result, (self.base_frame, used, res)

    def _assess_depth(self, now_s):
        if not self.use_depth:
            return None, None
        msg, info = self.depth_msg, self.camera_info
        if msg is None:
            return tc.invalid(f'no image on {self.depth_topic}'), None
        if now_s - self.depth_recv_s > self.depth_max_age_s:
            return tc.invalid(f'{self.depth_topic} is stale ({now_s - self.depth_recv_s:.1f} s old)'), None
        if info is None:
            return tc.invalid(f'no CameraInfo on {self.info_topic}'), None
        try:
            depth = depth_to_metres(msg, self.mm_scale)
        except ValueError as exc:
            return tc.invalid(str(exc)), None
        h, w = depth.shape
        # Intrinsics may be for a different resolution than the image (binning).
        sx = w / info.width if info.width else 1.0
        sy = h / info.height if info.height else 1.0
        k = info.k
        fx, cx, fy, cy = k[0] * sx, k[2] * sx, k[4] * sy, k[5] * sy
        if fx <= 0 or fy <= 0:
            return tc.invalid('CameraInfo has no intrinsics'), None
        side = int(w * self.roi_side_margin)
        pts = tc.depth_to_points(depth, fx, fy, cx, cy,
                                 row_start=int(h * self.roi_top), row_stop=h - int(h * self.roi_bottom_margin),
                                 col_start=side, col_stop=w - side, stride=self.depth_stride,
                                 min_range=self.depth_range[0], max_range=self.depth_range[1])
        frame = msg.header.frame_id
        up_known = False
        if self.depth_use_tf:
            to_base = self._lookup(self.base_frame, frame)
            if to_base is not None:
                pts = tc.crop_box(tc.transform_points(pts, *to_base), *self.depth_box)
                frame, up_known = self.base_frame, True
        result, used, res = tc.assess_terrain(pts, self.depth_params, up_known=up_known, rng=self.rng)
        if not up_known and self.depth_use_tf:
            result.reason += f' (camera frame: no TF {self.base_frame} <- {msg.header.frame_id})'
        return result, (frame, used, res)

    # ------------------------------------------------------------------ main loop
    def _tick(self):
        if not self.enabled:
            return
        now_s = self._now_s()
        if self._calibrating():
            self._calibrate_ground(now_s)
        if self.trigger_mode and (self.active_until_s is None or now_s > self.active_until_s):
            return                       # idle: the last label stays latched on the terrain topic
        lidar, lidar_dbg = self._assess_lidar(now_s)
        depth, depth_dbg = self._assess_depth(now_s)
        fused = tc.fuse(lidar, depth, self.fusion_mode, self.depth_fallback)
        previous = self.debounce.state
        state = self.debounce.update(tc.vote(fused, previous, self.flat_needs_lidar))
        self._publish_label(state)

        if state != previous:
            log = self.get_logger().warn if state == tc.UNKNOWN else self.get_logger().info
            log(f'Terrain: {state.upper()}  [{fused.source}, verified={fused.verified}]  '
                f'lidar={lidar.label} ({lidar.reason})  '
                f'depth={depth.label + " (" + depth.reason + ")" if depth else "off"}')
        elif state == tc.UNKNOWN:
            self.get_logger().warn(f'Terrain unknown. lidar: {lidar.reason}. '
                                   f'depth: {depth.reason if depth else "off"}',
                                   throttle_duration_sec=5.0)
        if fused.agree is False:
            self.get_logger().info(f'Lidar says {lidar.label}, depth camera says {depth.label}',
                                   throttle_duration_sec=2.0)

        self._update_kind(lidar_dbg, now_s)
        details = {
            'terrain': state, 'candidate': fused.label, 'source': fused.source,
            'verified': fused.verified, 'sensors_agree': fused.agree,
            'fusion_mode': self.fusion_mode,
            'lidar': lidar.to_dict(), 'depth': depth.to_dict() if depth else None,
        }
        self.details_pub.publish(String(data=json.dumps(details)))
        if self.publish_debug_clouds:
            self._publish_debug(self.lidar_dbg_pub, lidar_dbg)
            self._publish_debug(self.depth_dbg_pub, depth_dbg)

    def _update_kind(self, lidar_dbg, now_s):
        """Feed this assessment's floor points to the kind strategies; publish the kind when it changes."""
        if not self.kinds_enabled:
            return
        if not self.trigger_mode and now_s - self.kinds_started_s > self.kinds_window_s:
            self._new_patch()
        if lidar_dbg is not None and len(lidar_dbg[1]):          # lidar floor cells (may be missing: images still count)
            _frame, used, res = lidar_dbg
            fixed = None
            to_odom = self._lookup(self.odom_frame, self.base_frame) if self.odom_frame else None
            if to_odom:
                fixed = tc.transform_points(used, *to_odom)[:, :2]
            self.kinds.add(used[:, :2], res, fixed)
        self._feed_images(now_s)
        self._add_elevation(lidar_dbg)
        result = self.kinds.classify()
        if self.elev_enabled:
            result.update(self.elev_est.estimate())
            result['elevation_ready'] = self.elev_ref.ready
        key = (result['kind'], result.get('n_cells', 0) // 25,       # republish as the evidence grows
               round(result.get('elevation_m', 0.0) / 0.02), result.get('elevation_class'))
        if key != self.kind_last:
            self.kind_last = key
            if result['kind'] != 'unknown':
                self.get_logger().info(f'Terrain kind: {result["kind"].upper()} {result}')
            self.kind_pub.publish(String(data=json.dumps(dict(result, seq=self.kinds_seq))))

    def _add_image(self, camera, gray, pub, header):
        """Run the slat lines on one frame; publish the annotated picture and the counts for debugging."""
        watch = pub.get_subscription_count() > 0 or self.slats_json_pub.get_subscription_count() > 0
        out = self.kinds.add_image(camera, gray, debug=watch)
        if not watch:
            return
        res, picture = out
        if pub.get_subscription_count() > 0:
            pub.publish(Image(header=header, height=picture.shape[0], width=picture.shape[1], encoding='bgr8',
                              is_bigendian=0, step=picture.shape[1] * 3, data=picture.tobytes()))
        self.slats_json_pub.publish(String(data=json.dumps(dict(res, camera=camera, frames=len(self.kinds.images[camera])))))

    def _feed_images(self, now_s):
        """Depth and colour frame (each only once) -> Hough lines for the slats strategy."""
        if self.depth_msg is not None and self.use_depth and now_s - self.depth_recv_s <= self.depth_max_age_s \
                and self.depth_msg is not getattr(self, 'depth_seen', None):
            self.depth_seen = self.depth_msg
            try:
                depth = depth_to_metres(self.depth_msg, self.mm_scale)
                # no depth (0 / NaN) between the slats shows as a dark streak: keep it dark
                gray = slat_lines.to_gray8(np.nan_to_num(depth, nan=0.0))
                self._add_image('depth', gray, self.slats_depth_pub, self.depth_msg.header)
            except ValueError as exc:
                self.get_logger().warn(f'depth image for slats: {exc}', throttle_duration_sec=10.0)
        if self.color_msg is not None and self.use_color and now_s - self.color_recv_s <= self.color_max_age_s \
                and self.color_msg is not self.color_seen:
            self.color_seen = self.color_msg
            try:
                self._add_image('color', slat_lines.to_gray8(image_to_gray8(self.color_msg)), self.slats_color_pub,
                                self.color_msg.header)
            except ValueError as exc:
                self.get_logger().warn(f'colour image for slats: {exc}', throttle_duration_sec=10.0)

    # ------------------------------------------------------------------ elevation
    def _calibrating(self):
        return self.elev_enabled and self.enabled and not self.elev_ref.ready

    def _to_fixed(self, base_points):
        """base_link points -> the frame the elevation is measured in (z up), or None while there is no odom TF."""
        if not self.odom_frame or len(base_points) == 0:
            return None
        to_odom = self._lookup(self.odom_frame, self.base_frame)
        if to_odom is None:
            return None
        rot, trans = to_odom
        return tc.transform_points(base_points, rot, trans if self.elev_frame == 'odom' else np.zeros(3))

    def _calibrate_ground(self, now_s):
        """At the start (robot assumed flat on level floor): collect the lidar floor for calibrate_s, fit the reference."""
        pts, issue = self._lidar_points(now_s)
        fixed = self._to_fixed(pts) if pts is not None else None
        if fixed is None:
            self.elev_issue = issue or (f'no lidar points in the area' if pts is not None and len(pts) == 0
                                        else f'no TF {self.odom_frame} <- {self.base_frame}')
            self.elev_ref.started_s = None                  # the timer runs only while points come in
            self.get_logger().warn(f'Ground reference waiting: {self.elev_issue}', throttle_duration_sec=5.0)
            return
        self.elev_ref.add(fixed, now_s)
        outcome = self.elev_ref.finish(now_s)
        if outcome is True:
            self.elev_issue = ''
            info = dict(self.elev_ref.info, frame=self.elev_frame, odom_frame=self.odom_frame)
            self.ground_pub.publish(String(data=json.dumps(info)))
            self.get_logger().info(f'Ground reference taken ({self.elev_frame} frame): {info}')
        elif outcome:
            self.elev_issue = outcome
            self.get_logger().warn(f'Ground reference refused: {outcome}; collecting again.')

    def _add_elevation(self, lidar_dbg):
        """Feed this assessment's floor points (not walls / objects) to the elevation estimator."""
        if not (self.elev_enabled and self.elev_ref.ready and lidar_dbg is not None):
            return
        _frame, used, res = lidar_dbg
        if len(used) == 0:
            return
        floor = used[(res > -0.15) & (res < 0.25)]
        fixed = self._to_fixed(floor)
        if fixed is not None:
            self.elev_est.add(fixed)

    def _publish_label(self, label):
        self.terrain_pub.publish(String(data=label))

    def _publish_debug(self, pub, dbg):
        if dbg is None or pub.get_subscription_count() == 0:
            return
        frame, pts, res = dbg
        header = Header(frame_id=frame, stamp=self.get_clock().now().to_msg())
        pub.publish(xyzi_cloud(header, pts, res, PointCloud2, PointField))


def main(args=None):
    rclpy.init(args=args)
    node = TerrainClassifier()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
