"""Switch the Lite3 gait from the terrain classifier.

  /terrain_type rugged     -> rug_hstep    (0x21010407)
  /terrain_type non_rugged -> flat_medium  (0x21010307, same as lite3_flat_medium)

Subscribes /terrain_type (std_msgs/String, latched) and publishes a
transfer_interfaces/MotionSimpleCMD {cmd_code, type: 0} on /simple_cmd, the same
message the lite3_<gait> shell aliases send with `ros2 topic pub --once`.
/step_up (Bool, from the mission manager): true while the robot drives to a tile that is higher than the one it stands on
(terrain classifier elevation option): the step_up_gait (rug_hstep) is used instead of the terrain gait.
/gait_freeze (Bool, latched, from the mission manager): true while the recovery mode runs; the robot keeps the gait it
already has (no command is sent for its new targets or a new terrain) until it is released. /step_up still switches
to the step-up gait while frozen (the mission manager only sets it then when recovery's step_up_gait is on).
/robot_state (Int32MultiArray [basic, gait, policy, motion], robot_state_bridge): while /step_up holds the step-up gait,
it is sent again every step_up_retry_s until the robot reports it (terrain_gait.GaitConfirm; 0 = no re-sends).
Switching rules (flat delay, minimum interval) live in terrain_gait.py. The gait only changes when the robot starts
for a new tile (/mission_manager/next_target names a tile) or when a new terrain is classified (/terrain_type changes);
a finished spin / animation brings the terrain gait back. While the robot is in AI mode (/ai_mode true: the mission
manager's stairs mode climbs a staircase) no gait is sent at all: a gait command would take it out of AI mode.
Speed per gait: hstep_speed_limit (m/s, default 0.38; 0 = no cap) for rug_hstep, and gait_speed_limits ("gait:m/s",
e.g. flat_medium:0.4) for any other gait, cap Nav2's linear speed while that gait is in use,
as a nav2_msgs/SpeedLimit (absolute) on speed_limit_topic (the controller server's, default /speed_limit); any other
gait lifts the cap (speed_limit 0). Modes that publish /cmd_vel themselves (curtain, roller, stairs) use their own speeds.
Gait switches only while the robot stands still (switch_hold, terrain_gait.SwitchHold): every gait command is preceded by
switch_stop_s of zero velocity on /cmd_vel, and zero velocity goes on until the robot reports the new gait (at least
switch_settle_s; at most switch_stop_s + switch_max_wait_s). /gait_switching (Bool, latched) is true meanwhile: the
step_pause_gate (Nav2 output) and the mission manager's own /cmd_vel stop forwarding motion while it is.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav2_msgs.msg import SpeedLimit
from std_msgs.msg import Bool, Int32MultiArray, String
from transfer_interfaces.msg import MotionSimpleCMD

from lite3_explorer.robot_state import BASIC_STANDING
from lite3_explorer.terrain_gait import GAIT_CODES, GaitCommand, GaitConfirm, SwitchHold, TerrainGaitSwitcher


class GaitSwitcher(Node):
    def __init__(self):
        super().__init__('gait_switcher')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.switcher = TerrainGaitSwitcher(
            flat_gait=p('flat_gait', 'flat_medium'),
            rugged_gait=p('rugged_gait', 'rug_hstep'),
            flat_delay_s=p('flat_delay_s', 2.0),
            min_switch_interval_s=p('min_switch_interval_s', 1.0))
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, p('terrain_topic', '/terrain_type'), self._on_terrain, latched)
        # Match the domain bridge subscriber (RELIABLE + TRANSIENT_LOCAL); a VOLATILE publisher won't connect.
        cmd_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.create_publisher(MotionSimpleCMD, '/simple_cmd', cmd_qos)
        # The gait in use, latched (the tmux status window echoes it): 'flat_slow  (terrain)' etc.
        self.gait_pub = self.create_publisher(String, '~/gait', latched)
        self._publish_gait('none sent yet')
        # While the explorer spins to look around, hold the crawl gait (set from the reporter).
        self.crawl_hold = False
        self.hold_gait = 'flat_crawl'                     # gait held while crawl_hold is set
        self.repeats = int(p('command_repeats', 3))       # publishes per switch (a lone one can be lost)
        self.repeat_period_s = p('repeat_period_s', 0.3)
        self._burst = None                                # (MotionSimpleCMD, publishes left)
        # the robot is attempting a new tile: a gait switch is allowed
        self.create_subscription(String, p('target_topic', '/mission_manager/next_target'), self._on_target, latched)
        self.step_up_gait = p('step_up_gait', 'rug_hstep')
        self.create_subscription(Bool, p('step_up_topic', '/step_up'), self._on_step_up, latched)
        # step-up gait: re-send until /robot_state reports it
        self.confirm = GaitConfirm(retry_s=float(p('step_up_retry_s', 1.5)))
        self.robot_state_max_age_s = float(p('robot_state_max_age_s', 1.0))
        self.robot_gait = None                            # (robot_gait_state or None, time) from /robot_state
        self._last_publish_at = None                      # last /simple_cmd publish (burst repeats included)
        self.create_subscription(Int32MultiArray, p('robot_state_topic', '/robot_state'), self._on_robot_state, 10)
        self.create_subscription(Bool, '/spin_crawl', self._on_crawl_hold, 10)
        self.create_subscription(Bool, '/spin_flat', self._on_flat_request, 10)
        self.animating = False                            # no gait commands while a robot animation runs
        self.pre_animation_gait = None                    # the gait in use when the animation started: sent again after
        self.create_subscription(Bool, '/animation_active', self._on_animation, 10)
        # /gait_freeze (mission manager, recovery mode): keep the gait the robot already has, no commands at all
        self.create_subscription(Bool, p('freeze_topic', '/gait_freeze'), self._on_freeze, latched)
        self.ai_mode = False                              # no gait commands while the robot is in AI mode (stairs)
        self.create_subscription(Bool, '/ai_mode', self._on_ai_mode, latched)
        # Nav2 speed cap per gait ("gait:m/s"); gaits not listed: no cap
        self.speed_limits = {}
        for item in p('gait_speed_limits', ['']) or []:
            if item and ':' in item:
                gait, mps = item.rsplit(':', 1)
                self.speed_limits[gait.strip()] = float(mps)
        # h-step gait speed cap (m/s; 0 = no cap): overrides a rug_hstep entry in gait_speed_limits
        hstep_mps = float(p('hstep_speed_limit', 0.38))
        if hstep_mps > 0:
            self.speed_limits['rug_hstep'] = hstep_mps
        else:
            self.speed_limits.pop('rug_hstep', None)
        self.speed_pub = self.create_publisher(SpeedLimit, p('speed_limit_topic', '/speed_limit'), 10)
        self._limit_sent = None                           # (limit, subscriber count) last published
        # gait switches only while the robot stands still: zero /cmd_vel before and during each switch
        self.hold = (SwitchHold(stop_s=float(p('switch_stop_s', 0.4)), settle_s=float(p('switch_settle_s', 1.0)),
                                max_wait_s=float(p('switch_max_wait_s', 3.0)))
                     if bool(p('switch_hold', True)) else None)
        self.vel_pub = self.create_publisher(Twist, p('cmd_vel_topic', '/cmd_vel'), 10)
        self.switching_pub = self.create_publisher(Bool, p('switching_topic', '/gait_switching'), latched)
        self.switching_pub.publish(Bool(data=False))
        self._was_holding = False
        self.create_timer(0.02, self._hold_tick)
        self.create_timer(0.2, self._tick)
        self.get_logger().info(f'gait switcher: non_rugged -> {self.switcher.flat_gait}, '
                               f'rugged -> {self.switcher.rugged_gait}; Nav2 speed limits {self.speed_limits or "none"}')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_gait(self, why):
        self.gait_pub.publish(String(data=f'{self.switcher.gait or "-"}  ({why})'))

    def _on_terrain(self, msg):
        if self.switcher.update_terrain(msg.data, self._now()):
            self.get_logger().info(f'terrain: {msg.data}')

    def _on_target(self, msg):
        if msg.data.startswith('tile '):          # '(none)' = no target
            self.switcher.arm()

    def _on_freeze(self, msg):
        if self.switcher.freeze(bool(msg.data)):
            self.get_logger().info(f'gait frozen at {self.switcher.gait or "-"} (recovery)' if msg.data
                                   else 'gait freeze released: terrain / target rules again')
            # a switch still being repeated is the gait the robot has now: its repeats go on (see _tick)
            self._publish_gait('frozen: recovery' if msg.data else f'terrain {self.switcher.terrain}')

    def _on_step_up(self, msg):
        if self.switcher.set_override(self.step_up_gait if msg.data else None):
            self.get_logger().info(f'higher tile ahead: {"h-step gait" if msg.data else "terrain gait again"}')

    def _on_robot_state(self, msg):
        if len(msg.data) >= 2:
            gait = int(msg.data[1]) if int(msg.data[0]) == BASIC_STANDING else None
            self.robot_gait = (gait, self._now())

    def _confirm_step_up(self, now):
        """While /step_up holds the step-up gait and it was sent, send it again until /robot_state reports it."""
        sw = self.switcher
        want = self.step_up_gait if sw.override == self.step_up_gait and sw.gait == self.step_up_gait else None
        gait = self._robot_gait_now(now)
        resend = self.confirm.update(now, want, gait, self._last_publish_at)
        if self.confirm.event:
            (self.get_logger().warn if resend else self.get_logger().info)(f'step up: {self.confirm.event}')
        if resend:
            self._send(GaitCommand(want, GAIT_CODES[want]), now)

    def _send(self, cmd, now):
        """Switch to `cmd`: with switch_hold the robot is stopped first (_hold_tick publishes it), else at once."""
        if self.hold is None:
            self._publish_cmd(cmd, now)
            return
        if not self.hold.holding:
            self.get_logger().info(f'gait switch to {cmd.gait}: stopping the robot first')
        self.hold.request(cmd, now)
        self._hold_tick()

    def _robot_gait_now(self, now):
        return (self.robot_gait[0] if self.robot_gait is not None
                and now - self.robot_gait[1] <= self.robot_state_max_age_s else None)

    def _hold_tick(self):
        """Send the gait once the robot was stopped; zero /cmd_vel while the switch hold lasts."""
        if self.hold is None:
            return
        now = self._now()
        if self.hold.cmd is not None and (self.ai_mode or (self.animating and not self.crawl_hold)):
            # AI mode (a gait would end it) / animation (sends its own gait back after): drop the switch not sent yet
            self.hold.cancel()
            self.switcher.armed = True
        cmd = self.hold.update(now, self._robot_gait_now(now))
        if cmd is not None:
            self._publish_cmd(cmd, now)
        if self.hold.event:
            self.get_logger().info(f'gait switch: {self.hold.event}')
        holding = self.hold.holding
        if holding:
            self.vel_pub.publish(Twist())
        if holding != self._was_holding:
            self._was_holding = holding
            self.switching_pub.publish(Bool(data=holding))

    def _publish_cmd(self, cmd, now):
        out = MotionSimpleCMD()
        out.cmd_code, out.type = cmd.cmd_code, 0
        self.cmd_pub.publish(out)
        self._last_publish_at = now
        self._burst = (out, self.repeats - 1, now)
        self.switcher.mark_sent(cmd, now)
        self.get_logger().info(f'gait -> {cmd.gait} (0x{cmd.cmd_code:08X})')
        self._publish_gait('crawl hold: spinning' if self.crawl_hold else 'animation' if self.animating
                           else f'step up {self.step_up_gait}' if self.switcher.override
                           else f'terrain {self.switcher.terrain}')

    def _on_crawl_hold(self, msg):
        if self.ai_mode:
            return
        now = self._now()
        was_holding, self.crawl_hold = self.crawl_hold, bool(msg.data)
        if self.crawl_hold:
            # Sent at once (no interval wait): the reporter starts the spin after a short pause.
            self.hold_gait = 'flat_crawl'
            self._send(GaitCommand('flat_crawl', GAIT_CODES['flat_crawl']), now)
        elif was_holding:
            # Release: the next tick restores the gait for the current terrain.
            self.switcher.restore()

    def _on_flat_request(self, msg):
        # Hold flat_slow for the opening spin: the terrain loop must not switch back until the reporter releases it.
        if self.ai_mode:
            return
        if msg.data:
            self.crawl_hold = True
            self.hold_gait = 'flat_slow'
            self._send(GaitCommand('flat_slow', GAIT_CODES['flat_slow']), self._now())
        elif self.crawl_hold and self.hold_gait == 'flat_slow':
            self.crawl_hold = False
            self.switcher.restore()                       # the next tick restores the gait for the current terrain

    def _on_animation(self, msg):
        was, self.animating = self.animating, bool(msg.data)
        if self.animating and not was:
            self.pre_animation_gait = self.switcher.gait
        elif was and not self.animating:
            # Move mode is back and the robot has dropped its gait: send the one it had before the animation again
            # (the terrain loop takes over from there at the next event, as before).
            gait, self.pre_animation_gait = self.pre_animation_gait, None
            if gait in GAIT_CODES:
                self._send(GaitCommand(gait, GAIT_CODES[gait]), self._now())
            else:
                self.switcher.restore()                   # none sent before: the terrain gait

    def _on_ai_mode(self, msg):
        was, self.ai_mode = self.ai_mode, bool(msg.data)
        if self.ai_mode and not was:
            self._burst = None                            # a gait still being repeated would end AI mode
            if self.hold is not None and self.hold.holding:
                self.hold.cancel()                        # nor any gait still waiting to be sent
                self._hold_tick()
            self._publish_gait('AI mode: stairs')
        elif was and not self.ai_mode:
            self.switcher.restore()                       # AI mode is over: restore the terrain gait

    def _repeat_burst(self, now):
        out, left, last = self._burst
        if left <= 0:
            self._burst = None
        elif now - last >= self.repeat_period_s:
            self.cmd_pub.publish(out)
            self._last_publish_at = now
            self._burst = (out, left - 1, now)

    def _update_speed_limit(self):
        """Publish the speed cap of the gait in use when it changed (or the controller server just subscribed)."""
        limit = float(self.speed_limits.get(self.switcher.gait, 0.0))
        subs = self.speed_pub.get_subscription_count()
        if subs == 0 or self._limit_sent == (limit, subs):
            return
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage, msg.speed_limit = False, limit
        self.speed_pub.publish(msg)
        if self._limit_sent is None or self._limit_sent[0] != limit:
            self.get_logger().info(f'Nav2 speed limit: {f"{limit:.2f} m/s" if limit > 0 else "none"} '
                                   f'(gait {self.switcher.gait or "-"})')
        self._limit_sent = (limit, subs)

    def _tick(self):
        now = self._now()
        self._update_speed_limit()
        if self.ai_mode:
            return
        if self._burst is not None:
            self._repeat_burst(now)
        if self.crawl_hold or self.animating:
            return
        # Don't send (and record) a gait before the robot's subscriber is matched: it would be lost.
        if self.cmd_pub.get_subscription_count() == 0:
            return
        if self.hold is not None and self.hold.cmd is not None:
            return                                        # a switch is waiting for the robot to stop
        cmd = self.switcher.pending_command(now)
        if cmd is not None:
            self._send(cmd, now)
        else:
            self._confirm_step_up(now)


def main(args=None):
    rclpy.init(args=args)
    node = GaitSwitcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
