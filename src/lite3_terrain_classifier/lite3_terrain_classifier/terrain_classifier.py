"""Classify the terrain around the Lite3 as rugged / non_rugged.

Primary sensor: Livox MID-360 point cloud (/livox/points). The cloud is moved into
base_link with TF, a few frames are stacked (motion compensated through odom when that TF
exists), and the box around the robot is checked for steps, bumps and holes.

Verification: Orbbec depth image (/camera/depth/image_raw). Only the bottom part of the
frame is used (depth.roi_top_fraction), because the camera is pitched down and that part
shows the ground just ahead of the robot.

Output: /terrain_type (std_msgs/String, latched): "rugged", "non_rugged" or "unknown".
Details for tuning: ~/details (JSON string), ~/lidar_points and ~/depth_points
(PointCloud2, intensity = height above the fitted ground plane in metres).

The node only reads sensors, so it works the same whether the dog is driven with the
remote (manual) or by Nav2 / the frontier explorer (auto). Pause it with
  ros2 service call /terrain_classifier/set_enabled std_srvs/srv/SetBool "{data: false}"
"""
from collections import deque
from dataclasses import fields
import json

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from lite3_terrain_classifier import terrain_core as tc
from lite3_terrain_classifier.msg_utils import cloud_to_xyz, depth_to_metres, xyzi_cloud

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
        self.lidar_dbg_pub = self.create_publisher(PointCloud2, '~/lidar_points', 2)
        self.depth_dbg_pub = self.create_publisher(PointCloud2, '~/depth_points', 2)

        self.create_subscription(PointCloud2, self.cloud_topic, self._on_cloud, qos_profile_sensor_data)
        if self.use_depth:
            self.create_subscription(Image, self.depth_topic, self._on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, self.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_service(SetBool, '~/set_enabled', self._on_set_enabled)
        self.create_service(Trigger, '~/classify', self._on_classify)
        self.create_timer(1.0 / max(rate_hz, 0.1), self._tick)

        self._publish_label(tc.UNKNOWN)
        if self.trigger_mode:
            self.active_until_s = self._now_s() + self.trigger_duration_s
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

    def _on_classify(self, request, response):
        if self.trigger_mode:
            self.active_until_s = self._now_s() + self.trigger_duration_s
            response.message = f'classifying for {self.trigger_duration_s:g} s'
        else:
            response.message = 'continuous mode: already classifying'
        response.success = True
        return response

    def _on_info(self, msg):
        self.camera_info = msg

    def _on_depth(self, msg):
        self.depth_msg = msg
        self.depth_recv_s = self._now_s()

    def _on_cloud(self, msg):
        if not self.enabled:
            return
        if self.trigger_mode and (self.active_until_s is None or self._now_s() > self.active_until_s + 0.5):
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
        if self.trigger_mode and (self.active_until_s is None or now_s > self.active_until_s):
            return                       # idle: the last label stays latched on the terrain topic
        lidar, lidar_dbg = self._assess_lidar(now_s)
        depth, depth_dbg = self._assess_depth(now_s)
        fused = tc.fuse(lidar, depth, self.fusion_mode, self.depth_fallback)
        previous = self.debounce.state
        state = self.debounce.update(fused.label)
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
