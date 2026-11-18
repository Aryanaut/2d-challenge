"""Grade how rough the ground ahead is and slow down / pause the Lite3 to match.

Runs next to terrain_classifier and reuses its work:
  ~/lidar_points from terrain_classifier (PointCloud2 in base_link, intensity = height above
  the fitted ground plane), /terrain_type ("rugged" / "non_rugged" / "unknown"),
  /imu/data (wobble, tipping), /leg_odom2 (slip: commanded vs measured speed).

Outputs:
  /terrain_severity              std_msgs/String (latched)  smooth|rough|extreme
  /terrain_severity/speed_factor std_msgs/Float32            0.0 .. 1.0
  /terrain_severity/state        std_msgs/String             RUNNING|CREEP|PAUSED|BLOCKED
  /terrain_severity/blocked      std_msgs/Bool (latched)     true => explorer should pick another goal
  ~/details                      std_msgs/String             JSON with every metric, for tuning

Speed governor (auto mode only; the handheld remote does not go through ROS):
  cmd_in (/cmd_vel_raw, remap Nav2 / explorer output here) -> scaled -> cmd_out (/cmd_vel)

Services: ~/set_enabled (std_srvs/SetBool), ~/reset (std_srvs/Trigger) clears PAUSED/BLOCKED.
"""
from dataclasses import fields
import json
import time

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger

from lite3_terrain_severity import severity_core as sc

_DT = {PointField.INT8: np.int8, PointField.UINT8: np.uint8, PointField.INT16: np.int16,
       PointField.UINT16: np.uint16, PointField.INT32: np.int32, PointField.UINT32: np.uint32,
       PointField.FLOAT32: np.float32, PointField.FLOAT64: np.float64}


def _json_default(o):
    """Let json.dumps handle numpy scalars/arrays."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def cloud_to_xyzh(msg, h_field='intensity'):
    """PointCloud2 -> N x 4 (x, y, z, h)."""
    f = {fd.name: fd for fd in msg.fields}
    names = ['x', 'y', 'z', h_field]
    missing = [n for n in names if n not in f]
    if missing:
        raise KeyError(f'cloud has no field(s) {missing}')
    end = '>' if msg.is_bigendian else '<'
    dtype = np.dtype({'names': names,
                      'formats': [np.dtype(_DT[f[n].datatype]).newbyteorder(end) for n in names],
                      'offsets': [f[n].offset for n in names], 'itemsize': msg.point_step})
    buf = bytes(msg.data)
    if msg.row_step == msg.width * msg.point_step:
        arr = np.frombuffer(buf, dtype=dtype, count=msg.width * msg.height)
    else:
        arr = np.concatenate([np.frombuffer(buf[r * msg.row_step:r * msg.row_step + msg.width * msg.point_step],
                                            dtype=dtype) for r in range(msg.height)])
    out = np.stack([arr[n].astype(np.float64) for n in names], axis=1)
    return out[np.isfinite(out).all(axis=1)]


class TerrainSeverity(Node):
    def __init__(self):
        super().__init__('terrain_severity')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731

        self.points_topic = p('points_topic', '/terrain_classifier/lidar_points')
        self.h_field = p('height_field', 'intensity')
        self.terrain_topic = p('terrain_topic', '/terrain_type')
        self.imu_topic = p('imu_topic', '/imu/data')
        self.odom_topic = p('odom_topic', '/leg_odom2')
        self.cmd_in = p('cmd_in', '/cmd_vel_raw')
        self.cmd_out = p('cmd_out', '/cmd_vel')
        self.use_governor = p('use_governor', True)
        self.use_imu = p('use_imu', True)
        self.use_slip = p('use_slip', True)
        rate_hz = p('rate_hz', 20.0)
        self.points_max_age = p('points_max_age_s', 1.0)
        self.cmd_timeout = p('cmd_timeout_s', 0.5)
        self.enabled = p('start_enabled', True)
        imu_window = p('imu_window_s', 1.5)

        self.P = self._dataclass_params('severity', sc.SeverityParams)
        self.governor = sc.Governor(self._dataclass_params('governor', sc.GovernorParams))

        # state
        self.points, self.points_t = None, None
        self.terrain_type = 'unknown'
        self.roll = self.pitch = 0.0
        self.imu_win = sc.Window(imu_window)
        self.cmd, self.cmd_t = Twist(), -1e9
        self.cmd_stale_sent = True
        self.slip_hist = []
        self.geo_level, self.geo_metrics = None, {'reason': 'waiting for points'}
        self.last_label, self.last_state = None, None

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.level_pub = self.create_publisher(String, '/terrain_severity', latched)
        self.blocked_pub = self.create_publisher(Bool, '/terrain_severity/blocked', latched)
        self.factor_pub = self.create_publisher(Float32, '/terrain_severity/speed_factor', 10)
        self.state_pub = self.create_publisher(String, '/terrain_severity/state', 10)
        self.details_pub = self.create_publisher(String, '~/details', 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_out, 10) if self.use_governor else None

        self.create_subscription(PointCloud2, self.points_topic, self._on_points, qos_profile_sensor_data)
        self.create_subscription(String, self.terrain_topic, self._on_terrain, latched)
        if self.use_imu:
            self.create_subscription(Imu, self.imu_topic, self._on_imu, qos_profile_sensor_data)
        if self.use_slip:
            self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos_profile_sensor_data)
        if self.use_governor:
            self.create_subscription(Twist, self.cmd_in, self._on_cmd, 10)
        self.create_service(SetBool, '~/set_enabled', self._on_set_enabled)
        self.create_service(Trigger, '~/reset', self._on_reset)
        self.create_timer(1.0 / max(rate_hz, 1.0), self._tick)

        self._publish_level('unknown', False)
        self.get_logger().info(
            f'severity from {self.points_topic} + {self.terrain_topic}; governor '
            f'{self.cmd_in} -> {self.cmd_out}' if self.use_governor else 'governor off (report only)')

    # ------------------------------------------------------------------ parameters
    def _dataclass_params(self, prefix, cls):
        defaults, values = cls(), {}
        for f in fields(cls):
            d = getattr(defaults, f.name)
            v = self.declare_parameter(f'{prefix}.{f.name}', d).value
            values[f.name] = [float(x) for x in v] if isinstance(d, list) else type(d)(v)
        return cls(**values)

    # ------------------------------------------------------------------ callbacks
    def _now(self):
        return time.monotonic()

    def _on_points(self, msg):
        if not self.enabled:
            return
        try:
            self.points = cloud_to_xyzh(msg, self.h_field)
            self.points_t = self._now()
        except (KeyError, ValueError) as exc:
            self.get_logger().warn(f'cannot read {self.points_topic}: {exc}', throttle_duration_sec=5.0)
            return
        self.geo_level, self.geo_metrics = sc.assess_geometry(self.points, self._direction(), self.P)

    def _on_terrain(self, msg):
        self.terrain_type = msg.data.strip().lower()

    def _on_imu(self, msg):
        q = msg.orientation
        self.roll, self.pitch = sc.quat_to_roll_pitch(q.x, q.y, q.z, q.w)
        self.imu_win.push(self._now(), self.roll, self.pitch)

    def _on_odom(self, msg):
        if self._now() - self.cmd_t < self.cmd_timeout:
            self.slip_hist.append(abs(self.cmd.linear.x - msg.twist.twist.linear.x))
            self.slip_hist = self.slip_hist[-25:]
        else:
            self.slip_hist.clear()      # standing still or remote driving: no slip estimate

    def _on_cmd(self, msg):
        self.cmd, self.cmd_t = msg, self._now()

    def _on_set_enabled(self, req, res):
        self.enabled = bool(req.data)
        self.governor.reset()
        res.success, res.message = True, 'terrain_severity ' + ('enabled' if self.enabled else 'paused')
        self.get_logger().info(res.message)
        return res

    def _on_reset(self, req, res):
        self.governor.reset()
        res.success, res.message = True, 'governor reset'
        return res

    # ------------------------------------------------------------------ main loop
    def _direction(self):
        if self._now() - self.cmd_t > self.cmd_timeout:
            return 'forward'
        vx, wz = self.cmd.linear.x, self.cmd.angular.z
        if vx < -0.05:
            return 'backward'
        if abs(vx) < 0.05 and abs(wz) > 0.1:
            return 'turn'
        return 'forward'

    def _tick(self):
        now = self._now()
        if not self.enabled:
            self._forward_cmd(now, passthrough=True)
            return

        geo_level, geo_metrics = self.geo_level, self.geo_metrics
        if self.points_t is None or now - self.points_t > self.points_max_age:
            geo_level, geo_metrics = None, {'reason': f'{self.points_topic} stale or missing'}

        imu_std = self.imu_win.std_deg() if self.use_imu else 0.0
        slip = float(np.mean(self.slip_hist)) if (self.use_slip and len(self.slip_hist) >= 5) else 0.0
        pro_level, alarm, pro_metrics = sc.assess_proprio(
            np.degrees(self.roll), np.degrees(self.pitch), imu_std, slip, self.P)
        if not self.use_imu:
            alarm = False

        raw = sc.combine(geo_level, pro_level, self.terrain_type, self.P)
        state, factor = self.governor.update(raw, alarm, now)
        label = sc.LEVELS[self.governor.level]

        self._publish_level(label, state == sc.BLOCKED)
        self.factor_pub.publish(Float32(data=float(factor)))
        self.state_pub.publish(String(data=state))
        if (label, state) != (self.last_label, self.last_state):
            log = self.get_logger().warn if state in (sc.PAUSED, sc.BLOCKED) else self.get_logger().info
            why = geo_metrics.get('why') or geo_metrics.get('reason', '')
            walls = geo_metrics.get('wall_cells')
            log(f'severity {label.upper()} -> {state} x{factor:.2f} | terrain={self.terrain_type} | '
                f'lidar: {why}' + (f' (ignored {walls} wall cells)' if walls else '') +
                f' | robot: {pro_metrics["level"]} wobble={pro_metrics["imu_std"]} deg'
                f'{" | TILT ALARM" if alarm else ""}')
            self.last_label, self.last_state = label, state

        self.details_pub.publish(String(data=json.dumps({
            'severity': label, 'raw_level': sc.LEVELS[raw], 'state': state, 'speed_factor': round(factor, 2),
            'terrain_type': self.terrain_type, 'pauses': self.governor.pauses,
            'geometry': geo_metrics, 'proprio': pro_metrics}, default=_json_default)))
        self._forward_cmd(now)

    def _forward_cmd(self, now, passthrough=False):
        if not self.use_governor:
            return
        if now - self.cmd_t > self.cmd_timeout:
            if not self.cmd_stale_sent:             # input stopped: stop once, then stay quiet
                self.cmd_pub.publish(Twist())
                self.cmd_stale_sent = True
            return
        self.cmd_stale_sent = False
        c, out = self.cmd, Twist()
        if passthrough:
            out = c
        else:
            out.linear.x, out.linear.y, out.angular.z = self.governor.scale(c.linear.x, c.linear.y, c.angular.z)
        self.cmd_pub.publish(out)

    def _publish_level(self, label, blocked):
        if label != getattr(self, '_pub_label', None):
            self.level_pub.publish(String(data=label))
            self._pub_label = label
        if blocked != getattr(self, '_pub_blocked', None):
            self.blocked_pub.publish(Bool(data=blocked))
            self._pub_blocked = blocked

    def stop(self):
        if self.cmd_pub is not None:
            self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = TerrainSeverity()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
