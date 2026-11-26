"""/cmd_vel gate: wait for the h-step gait, then balance pauses while Nav2 drives the robot up onto a higher tile.

exploration.launch.py (step_pause:=true) remaps Nav2's velocity smoother output from /cmd_vel to in_topic
(/cmd_vel_nav_smoothed); this node forwards it to /cmd_vel unchanged, except during a pause (step_pause.py): while
/step_up is true and the IMU shows the nose going up the step, it sends zero velocity for pause_s at each of
pause_offsets_s after the tilt (1-2 short stops so the robot can balance on the step).

Gait wait (wait_for_hstep, step_pause.GaitWait): from /step_up going on, zero velocity until /robot_state reports the
h-step gait (robot_gait_state hstep_state, basic 6), so the robot never drives onto the step in the flat gait. Gives up
after gait_wait_max_s (0 = never).

Gait switch hold: while /gait_switching (gait_switcher) is true the robot is held too (zero velocity), so a gait is
only switched while it stands still. This one holds even when Nav2 is not driving.

Pauses only start while Nav2 is actually driving FORWARD (a command with linear speed >= min_forward_mps arrived in the
last driving_timeout_s): turning in place at the step edge tilts the nose too, and must not use the pauses up before
the climb (fullrun_26_11_210206). The curtain and roller modes also set /step_up (h-step gait) but publish /cmd_vel
themselves, and are left alone.

  sub  in_topic (/cmd_vel_nav_smoothed)  geometry_msgs/Twist   Nav2 velocity smoother output
  sub  /step_up                          std_msgs/Bool         mission manager: driving to a higher tile
  sub  /imu/data                         sensor_msgs/Imu       robot IMU (pitch)
  sub  /gait_switching                   std_msgs/Bool         gait_switcher: a gait switch is running
  sub  /robot_state                      std_msgs/Int32MultiArray  [basic, gait, policy, motion] (robot_state_bridge)
  pub  /cmd_vel                          geometry_msgs/Twist
  pub  ~/paused                          std_msgs/Bool         true during a pause
"""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Int32MultiArray

from lite3_explorer.robot_state import BASIC_STANDING
from lite3_explorer.step_pause import GaitWait, StepPause, StepPauseParams


class StepPauseGate(Node):
    def __init__(self):
        super().__init__('step_pause_gate')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.enabled = bool(p('enabled', True))
        self.logic = StepPause(StepPauseParams(
            pause_offsets_s=[float(x) for x in p('pause_offsets_s', [0.0, 0.7])],
            pause_s=float(p('pause_s', 0.2)),
            climb_start_deg=float(p('climb_start_deg', 6.0)),
            tilt_hold_s=float(p('tilt_hold_s', 0.1)),
            level_deg=float(p('level_deg', 3.0)),
            rearm_s=float(p('rearm_s', 1.0))))
        self.nose_up_sign = float(p('nose_up_sign', -1.0))     # ROS pitch: + = nose DOWN
        self.imu_max_age_s = float(p('imu_max_age_s', 0.5))
        self.driving_timeout_s = float(p('driving_timeout_s', 0.3))
        self.min_forward_mps = float(p('min_forward_mps', 0.05))
        self.wait_for_hstep = bool(p('wait_for_hstep', True))
        self.gait_wait = GaitWait(hstep_state=int(p('hstep_state', 13)), max_wait_s=float(p('gait_wait_max_s', 4.0)))
        self.robot_state_max_age_s = float(p('robot_state_max_age_s', 1.0))
        self.gait_state = None        # (robot_gait_state, monotonic time) while basic state 6 (standing / stepping)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.step_up = False
        self.gait_switching = False
        self.nose = None              # (nose_up_deg, monotonic time)
        self.last_drive_t = -math.inf
        self.last_forward_t = -math.inf
        self.was_paused = False
        self.was_gait_held = False
        self.cmd_pub = self.create_publisher(Twist, p('out_topic', '/cmd_vel'), 10)
        self.paused_pub = self.create_publisher(Bool, '~/paused', latched)
        self.paused_pub.publish(Bool(data=False))
        self.create_subscription(Twist, p('in_topic', '/cmd_vel_nav_smoothed'), self._on_cmd, 10)
        self.create_subscription(Bool, p('step_up_topic', '/step_up'), self._on_step_up, latched)
        self.create_subscription(Bool, p('gait_switching_topic', '/gait_switching'), self._on_gait_switching, latched)
        self.create_subscription(Imu, p('imu_topic', '/imu/data'), self._on_imu, qos_profile_sensor_data)
        self.create_subscription(Int32MultiArray, p('robot_state_topic', '/robot_state'), self._on_robot_state, 10)
        self.create_timer(0.02, self._tick)
        o = self.logic.offsets
        self.get_logger().info(
            f'Step-up pauses {"on" if self.enabled else "OFF (plain relay)"}: {len(o)} x {self.logic.p.pause_s:.2f} s '
            f'at {o} s after the nose tilts up > {self.logic.p.climb_start_deg:.0f} deg while /step_up is true; '
            + (f'hold on /step_up until the robot reports the h-step gait (give up after '
               f'{self.gait_wait.max_wait_s:.1f} s).' if self.enabled and self.wait_for_hstep else 'no h-step gait wait.'))

    def _on_step_up(self, msg):
        self.step_up = bool(msg.data)

    def _on_gait_switching(self, msg):
        self.gait_switching = bool(msg.data)

    def _on_robot_state(self, msg):
        if len(msg.data) >= 2:
            gait = int(msg.data[1]) if int(msg.data[0]) == BASIC_STANDING else None
            self.gait_state = (gait, time.monotonic())

    def _on_imu(self, msg):
        q = msg.orientation
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        self.nose = (self.nose_up_sign * math.degrees(pitch), time.monotonic())

    def _on_cmd(self, msg):
        if abs(msg.linear.x) > 1e-3 or abs(msg.linear.y) > 1e-3 or abs(msg.angular.z) > 1e-3:
            self.last_drive_t = time.monotonic()
        if math.hypot(msg.linear.x, msg.linear.y) >= self.min_forward_mps:
            self.last_forward_t = time.monotonic()
        if not self._tick():
            self.cmd_pub.publish(msg)

    def _tick(self) -> bool:
        """Update the pause logic; while paused publish zero velocity. Returns whether the robot is held."""
        if not self.enabled:
            return False
        now = time.monotonic()
        nose = self.nose[0] if self.nose is not None and now - self.nose[1] <= self.imu_max_age_s else None
        driving = now - self.last_drive_t <= self.driving_timeout_s
        held = False
        if self.wait_for_hstep:
            gait = (self.gait_state[0] if self.gait_state is not None
                    and now - self.gait_state[1] <= self.robot_state_max_age_s else None)
            held = self.gait_wait.update(now, self.step_up, gait, driving=driving)
            if held and not self.was_gait_held:
                self.get_logger().info(f'Step up ahead: holding the robot until it reports the h-step gait '
                                       f'(gait state now {gait}).')
            self.was_gait_held = held
            if self.gait_wait.event:
                log = self.get_logger().warn if self.gait_wait.gave_up else self.get_logger().info
                log(self.gait_wait.event)
        forward = now - self.last_forward_t <= self.driving_timeout_s
        paused = self.logic.update(now, self.step_up, nose, driving=forward)
        if self.logic.started:
            self.get_logger().info(self.logic.started)
        paused = paused or held or self.gait_switching
        if paused:
            self.cmd_pub.publish(Twist())
        if paused != self.was_paused:
            self.was_paused = paused
            self.paused_pub.publish(Bool(data=paused))
        return paused


def main(args=None):
    rclpy.init(args=args)
    node = StepPauseGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
