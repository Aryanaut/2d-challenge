#!/usr/bin/env python3
"""ROS 2 wrapper around the stair detector + state machine.

Subscribes
    <cloud_topic>               sensor_msgs/PointCloud2   3D lidar
    /imu/data                   sensor_msgs/Imu           roll / pitch
    /leg_odom2                  nav_msgs/Odometry         x, y, z, yaw
    /stair_climber/enable       std_msgs/Bool             start / stop the behaviour
    /stair_climber/vlm_result   std_msgs/Bool             optional camera confirmation

Publishes
    /cmd_vel                    geometry_msgs/Twist       only while active
    /simple_cmd                 transfer_interfaces/MotionSimpleCMD   gait switches
    /stair_climber/state        std_msgs/String           current state
    /stair_climber/active       std_msgs/Bool             True = we own /cmd_vel
    /stair_climber/debug        std_msgs/String           JSON: pitch, detection, ...
    /stair_climber/vlm_request  std_msgs/Empty            ask the VLM node to check
    /stair_climber/marker       visualization_msgs/Marker detected step edge (RViz)

Set dry_run:=true to publish velocities on /stair_climber/cmd_vel_preview
and log gait switches instead of sending them. Use that with a rosbag.
"""

import json
import math
import time
from collections import deque
from dataclasses import fields, replace

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Bool, Empty, String
from visualization_msgs.msg import Marker

try:
    from sensor_msgs_py import point_cloud2
except ImportError:  # pragma: no cover
    point_cloud2 = None

try:
    from transfer_interfaces.msg import MotionSimpleCMD
except ImportError:  # robot's custom message not sourced
    MotionSimpleCMD = None

try:
    import tf2_ros
except ImportError:  # pragma: no cover
    tf2_ros = None

from .stair_detector import (DetectorParams, StairObservation, detect_stairs,
                             estimate_floor_z, front_wall_distance, quaternion_to_matrix,
                             quaternion_to_rpy, rpy_to_matrix, to_level_base_frame)
from .state_machine import Observation, SMParams, State, StairStateMachine

GAIT_CODE_PARAMS = ('stair_gait_code', 'flat_gait_code', 'top_gait_code', 'descend_gait_code',
                    'ai_enter_code', 'ai_exit_code')


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """PointCloud2 -> (N,3) float array, NaNs removed.

    Reads x/y/z straight from the byte buffer, so clouds with mixed field types
    (the Livox driver adds uint8 tag/line fields) work too.
    """
    fields = {f.name: f for f in msg.fields}
    np_types = {7: np.float32, 8: np.float64}           # PointField.FLOAT32 / FLOAT64
    if all(k in fields and fields[k].datatype in np_types for k in ('x', 'y', 'z')):
        endian = '>' if msg.is_bigendian else '<'
        dt = np.dtype({'names': ['x', 'y', 'z'],
                       'formats': [np.dtype(np_types[fields[k].datatype]).newbyteorder(endian)
                                   for k in ('x', 'y', 'z')],
                       'offsets': [fields[k].offset for k in ('x', 'y', 'z')],
                       'itemsize': msg.point_step})
        n = msg.width * msg.height
        arr = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
        pts = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float64)
        return pts[np.isfinite(pts).all(axis=1)]
    arr = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
    arr = np.asarray(list(arr) if not isinstance(arr, np.ndarray) else arr)
    if arr.dtype.names:
        return np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float64)
    return arr.astype(np.float64).reshape(-1, 3)


class StairClimberNode(Node):

    def __init__(self):
        super().__init__('stair_climber')

        # ---------------- node-level parameters ----------------
        self.declare_parameter('cloud_topic', '/points')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('odom_topic', '/leg_odom2')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('simple_cmd_topic', '/simple_cmd')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('use_tf', True)
        # Fallback lidar extrinsics (Lab Exercise 7 table), used if TF fails
        self.declare_parameter('lidar_xyz', [0.18678, 0.0, 0.12498])
        self.declare_parameter('lidar_rpy_deg', [0.0, 15.0, 0.0])
        self.declare_parameter('auto_calibrate_floor', True)
        self.declare_parameter('floor_z', math.nan)
        self.declare_parameter('max_cloud_points', 40000)
        self.declare_parameter('detect_rate', 5.0)
        self.declare_parameter('detection_max_age', 0.6)
        # Median of the last N detections (distance / angle / centre). 1 = off.
        # Close-range single scans jitter a few degrees (stairs_run2), so use ~3.
        self.declare_parameter('detect_smooth_n', 1)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('autostart', False)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('gait_cmd_repeats', 3)
        # Gaits only change in Move Mode (DEEP Robotics doc sec 1.2.5), so this
        # is sent right before every gait code. "0x0" disables it.
        self.declare_parameter('move_mode_code', '0x21010D06')
        # 18 Nov: /cmd_vel was ignored until Move Mode was sent, so it is also sent
        # when the node is enabled (before ALIGN drives). Optionally also after Enter AI.
        self.declare_parameter('move_mode_on_enable', True)
        self.declare_parameter('move_mode_after_ai_enter', False)
        # Enter/Exit AI: the manual test that walked sent each code ONCE.
        self.declare_parameter('ai_cmd_repeats', 1)

        # ---------------- algorithm parameters ----------------
        self.det_params = self._declare_dataclass(DetectorParams, 'detector.')
        self.sm_params = self._declare_dataclass(SMParams, '')
        self.sm = StairStateMachine(self.sm_params)

        gp = self.get_parameter
        self.dry_run = gp('dry_run').value
        self.use_tf = gp('use_tf').value and tf2_ros is not None
        self.base_frame = gp('base_frame').value
        self.floor_z = gp('floor_z').value
        self.auto_floor = gp('auto_calibrate_floor').value
        self.max_pts = gp('max_cloud_points').value
        self.detect_period = 1.0 / gp('detect_rate').value
        self.det_max_age = gp('detection_max_age').value
        self.smooth = deque(maxlen=max(1, int(gp('detect_smooth_n').value)))
        self.gait_repeats = gp('gait_cmd_repeats').value
        self.move_mode_code = int(str(gp('move_mode_code').value).strip() or '0', 0)
        self.move_mode_on_enable = gp('move_mode_on_enable').value
        self.move_mode_after_ai = gp('move_mode_after_ai_enter').value
        self.ai_repeats = gp('ai_cmd_repeats').value
        xyz = gp('lidar_xyz').value
        rpy = [math.radians(a) for a in gp('lidar_rpy_deg').value]
        self.fallback_R = rpy_to_matrix(*rpy)
        self.fallback_t = np.array(xyz, dtype=float)

        # ---------------- state ----------------
        self.enabled = gp('autostart').value
        self.imu_t = None
        self.odom_t = None
        self.roll = self.pitch = 0.0
        self.x = self.y = self.z = self.yaw = 0.0
        self.stair = None
        self.stair_t = 0.0
        self.wall_dist = math.inf      # lidar: tall obstacle straight ahead (checked in every state)
        self.wall_t = 0.0
        self.stair_seq = 0
        self.last_detect_wall = 0.0
        self.vlm = None
        self.was_active = False
        self.last_state = None
        self.pending_gait = []   # [(code, remaining_repeats)]

        # ---------------- ROS interfaces ----------------
        self.create_subscription(PointCloud2, gp('cloud_topic').value, self.on_cloud,
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, gp('imu_topic').value, self.on_imu,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, gp('odom_topic').value, self.on_odom, 10)
        self.create_subscription(Bool, '/stair_climber/enable', self.on_enable, 10)
        self.create_subscription(Bool, '/stair_climber/vlm_result', self.on_vlm, 10)
        # the mission manager puts the robot in AI mode itself (before enabling this node) and says so here (latched):
        # no Move Mode on enable (it would leave AI mode) and no second Enter AI
        self.ai_on_already = False
        self.create_subscription(Bool, '/stair_climber/ai_on', self.on_ai_on,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL))

        cmd_topic = '/stair_climber/cmd_vel_preview' if self.dry_run else gp('cmd_vel_topic').value
        # The robot's /cmd_vel subscriber asks for TRANSIENT_LOCAL durability (18 Nov:
        # "incompatible QoS ... DURABILITY"); a default VOLATILE publisher is ignored.
        # TRANSIENT_LOCAL also still works with ordinary VOLATILE subscribers.
        cmd_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, cmd_qos)
        self.simple_pub = None
        if MotionSimpleCMD is not None and not self.dry_run:
            # The robot's /simple_cmd subscriber asks for TRANSIENT_LOCAL durability
            # (same as `ros2 topic pub`); a default VOLATILE publisher is ignored by it.
            simple_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                    durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.simple_pub = self.create_publisher(MotionSimpleCMD, gp('simple_cmd_topic').value,
                                                    simple_qos)
        self.state_pub = self.create_publisher(String, '/stair_climber/state', 10)
        self.active_pub = self.create_publisher(Bool, '/stair_climber/active', 10)
        self.debug_pub = self.create_publisher(String, '/stair_climber/debug', 10)
        self.vlm_req_pub = self.create_publisher(Empty, '/stair_climber/vlm_request', 10)
        self.marker_pub = self.create_publisher(Marker, '/stair_climber/marker', 10)

        if self.use_tf:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_timer(1.0 / gp('control_rate').value, self.control_loop)

        if MotionSimpleCMD is None:
            self.get_logger().warn(
                'transfer_interfaces not found: gait switching is DISABLED. '
                'Run `source ~/lite3_ws/install/setup.bash` before launching.')
        self.get_logger().info(
            f'stair_climber ready (dry_run={self.dry_run}, enabled={self.enabled}). '
            f'Enable with: ros2 topic pub --once /stair_climber/enable std_msgs/msg/Bool "{{data: true}}"')

    # ------------------------------------------------------------ params

    def _declare_dataclass(self, cls, prefix):
        """Declare one ROS parameter per dataclass field and build the dataclass."""
        values = {}
        for f in fields(cls):
            name = prefix + f.name
            default = getattr(cls(), f.name)
            if f.name in GAIT_CODE_PARAMS:
                # Hex strings like "0x21010300" -- YAML/ROS ints don't do hex reliably
                self.declare_parameter(name, hex(default))
                raw = str(self.get_parameter(name).value).strip()
                values[f.name] = int(raw, 0) if raw else 0
            else:
                self.declare_parameter(name, default)
                values[f.name] = type(default)(self.get_parameter(name).value)
        return cls(**values)

    # ---------------------------------------------------------- callbacks

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_enable(self, msg: Bool):
        if msg.data != self.enabled:
            self.get_logger().info(f'stair_climber {"ENABLED" if msg.data else "DISABLED"}')
        self.enabled = msg.data
        if not msg.data:
            self.pending_gait.clear()       # disabled: send no more gait changes

    def on_ai_on(self, msg: Bool):
        self.ai_on_already = bool(msg.data)
        self.get_logger().info(f'AI mode already on (set by the mission): {self.ai_on_already}')

    def on_vlm(self, msg: Bool):
        self.vlm = msg.data

    def on_imu(self, msg: Imu):
        q = msg.orientation
        self.roll, self.pitch, _ = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        self.imu_t = self.now()

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x, self.y, self.z = p.x, p.y, p.z
        _, _, self.yaw = quaternion_to_rpy(q.x, q.y, q.z, q.w)
        self.odom_t = self.now()

    def _sensor_pose(self, frame_id):
        if self.use_tf and frame_id and frame_id != self.base_frame:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, frame_id, rclpy.time.Time())
                r, t = tf.transform.rotation, tf.transform.translation
                return quaternion_to_matrix(r.x, r.y, r.z, r.w), np.array([t.x, t.y, t.z])
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f'TF {frame_id}->{self.base_frame} failed, '
                                       f'using lidar_xyz/lidar_rpy_deg: {e}',
                                       throttle_duration_sec=5.0)
        if frame_id == self.base_frame:
            return np.eye(3), np.zeros(3)
        return self.fallback_R, self.fallback_t

    def on_cloud(self, msg: PointCloud2):
        # The wall check runs on every scan in every state (cheap). Stair detection only while it is
        # needed (SEARCH / ALIGN / IDLE), rate-limited (CPU).
        detect = self.sm.state in (State.SEARCH, State.ALIGN, State.IDLE)
        wall = time.monotonic()
        detect = detect and wall - self.last_detect_wall >= self.detect_period
        if not detect and not self.enabled:
            return

        pts = cloud_to_xyz(msg)
        if len(pts) > self.max_pts:
            pts = pts[:: int(math.ceil(len(pts) / self.max_pts))]
        R, t = self._sensor_pose(msg.header.frame_id)
        level = to_level_base_frame(pts, R, t, self.roll, self.pitch)
        body_x = pts @ R[0] + t[0]                      # x in base_link before levelling
        self.wall_dist, self.wall_t = front_wall_distance(level, self.det_params, body_x), self.now()
        if not detect:
            return
        self.last_detect_wall = wall

        if math.isnan(self.floor_z) and self.auto_floor:
            fz = estimate_floor_z(level)
            if not math.isnan(fz):
                self.floor_z = fz
                self.get_logger().info(f'floor calibrated: floor_z = {fz:.3f} m '
                                       '(robot must be on flat ground with clear space ahead)')
        if math.isnan(self.floor_z):
            return

        obs = detect_stairs(level, self.floor_z, self.det_params)
        now = self.now()
        if obs.detected and self.smooth.maxlen > 1:
            while self.smooth and now - self.smooth[0][0] > 1.0:
                self.smooth.popleft()
            self.smooth.append((now, obs))
            med = lambda k: float(np.median([getattr(o, k) for _, o in self.smooth]))
            obs = replace(obs, distance=med('distance'), edge_yaw=med('edge_yaw'),
                          center_y=med('center_y'))
        self.stair = obs
        self.stair_t = now
        self.stair_seq += 1
        self._publish_marker(msg.header.stamp)

    # -------------------------------------------------------------- loop

    def control_loop(self):
        t = self.now()
        fresh_stair = self.stair if (t - self.stair_t) <= self.det_max_age else None
        obs = Observation(
            t=t,
            enabled=self.enabled,
            imu_age=(t - self.imu_t) if self.imu_t is not None else 1e9,
            odom_age=(t - self.odom_t) if self.odom_t is not None else 1e9,
            roll=self.roll, pitch=self.pitch, yaw=self.yaw,
            x=self.x, y=self.y, z=self.z,
            stair=fresh_stair, stair_seq=self.stair_seq, vlm=self.vlm,
            wall_dist=self.wall_dist if (t - self.wall_t) <= self.det_max_age else math.inf,
            ai_on_already=self.ai_on_already,
        )
        cmd = self.sm.update(obs)

        if cmd.state != self.last_state:
            why = self.sm.transitions[-1][3] if self.sm.transitions else ''
            text = f'[{self.last_state.value if self.last_state else "-"} -> {cmd.state.value}] {why}'
            if cmd.state == State.ABORT:
                self.get_logger().error(text)
            else:
                self.get_logger().info(text)
            self.last_state = cmd.state

        if cmd.request_vlm:
            self.vlm = None
            self.vlm_req_pub.publish(Empty())

        entered_search = (cmd.state == State.SEARCH and self.sm.transitions
                          and self.sm.transitions[-1][0] == t
                          and self.sm.transitions[-1][1] == State.IDLE)
        if entered_search and self.move_mode_on_enable and self.move_mode_code and not self.ai_on_already:
            self.pending_gait.append([self.move_mode_code, self.gait_repeats])
        if cmd.gait_code:
            if self.move_mode_code and not cmd.raw_code:
                self.pending_gait.append([self.move_mode_code, self.gait_repeats])
            ai = cmd.gait_code in (self.sm_params.ai_enter_code, self.sm_params.ai_exit_code)
            self.pending_gait.append([cmd.gait_code, self.ai_repeats if ai else self.gait_repeats])
            if (cmd.gait_code == self.sm_params.ai_enter_code and self.move_mode_after_ai
                    and self.move_mode_code):
                self.pending_gait.append([self.move_mode_code, self.gait_repeats])
        self._send_gait()

        if cmd.active:
            tw = Twist()
            tw.linear.x, tw.linear.y, tw.angular.z = cmd.vx, cmd.vy, cmd.wz
            self.cmd_pub.publish(tw)
        elif self.was_active:
            self.cmd_pub.publish(Twist())   # one zero when handing control back
        self.was_active = cmd.active

        self.state_pub.publish(String(data=cmd.state.value))
        self.active_pub.publish(Bool(data=cmd.active))
        s = fresh_stair
        self.debug_pub.publish(String(data=json.dumps({
            'state': cmd.state.value, 'note': cmd.note,
            'nose_up_deg': round(math.degrees(self.sm_params.nose_up_sign * self.pitch), 1),
            'roll_deg': round(math.degrees(self.roll), 1),
            'z': round(self.z, 3), 'floor_z': None if math.isnan(self.floor_z) else round(self.floor_z, 3),
            'cmd': [round(cmd.vx, 3), round(cmd.vy, 3), round(cmd.wz, 3)],
            'wall_m': None if math.isinf(obs.wall_dist) else round(obs.wall_dist, 2),
            'stairs': None if s is None else {
                'detected': s.detected, 'reason': s.reason,
                'distance': None if math.isnan(s.distance) else round(s.distance, 3),
                'step_cm': None if math.isnan(s.step_height) else round(s.step_height * 100, 1),
                'yaw_deg': round(math.degrees(s.edge_yaw), 1),
                'center_y': round(s.center_y, 3), 'width': round(s.width, 2),
                'close': s.close},
        })))

    def _send_gait(self):
        """Send each gait code a few times, one per tick (the robot may drop one)."""
        if not self.pending_gait:
            return
        item = self.pending_gait[0]
        code = item[0]
        if self.dry_run:
            for c, _ in self.pending_gait:
                self.get_logger().info(f'[dry_run] would send /simple_cmd cmd_code={hex(c)}')
            self.pending_gait.clear()
            return
        if self.simple_pub is None:
            self.pending_gait.clear()
            self.sm.force_abort('cannot send gait command: transfer_interfaces not sourced')
            return
        msg = MotionSimpleCMD()
        msg.cmd_code = code
        msg.type = 0
        self.simple_pub.publish(msg)
        self.get_logger().info(f'sent /simple_cmd cmd_code={hex(code)}')
        item[1] -= 1
        if item[1] <= 0:
            self.pending_gait.pop(0)

    def _publish_marker(self, stamp):
        s = self.stair
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = stamp
        m.ns, m.id, m.type = 'stair_edge', 0, Marker.LINE_STRIP
        if not (s and s.detected):
            m.action = Marker.DELETE
            self.marker_pub.publish(m)
            return
        m.action = Marker.ADD
        m.scale.x = 0.02
        m.color.r, m.color.g, m.color.a = 1.0, 0.5, 1.0
        a = -math.tan(s.edge_yaw)   # x = a*y + b
        b = s.distance - a * s.center_y
        z = self.floor_z + s.step_height
        for yy in (s.center_y - s.width / 2, s.center_y + s.width / 2):
            m.points.append(Point(x=a * yy + b, y=yy, z=z))
        self.marker_pub.publish(m)

    def stop(self):
        if rclpy.ok():                      # after Ctrl+C the context is already gone
            self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = StairClimberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
