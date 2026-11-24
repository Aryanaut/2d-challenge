"""Switch the Lite3 gait from the terrain classifier.

  /terrain_type rugged     -> rug_hstep    (0x21010407)
  /terrain_type non_rugged -> flat_medium  (0x21010307, same as lite3_flat_medium)

Subscribes /terrain_type (std_msgs/String, latched) and publishes a
transfer_interfaces/MotionSimpleCMD {cmd_code, type: 0} on /simple_cmd, the same
message the lite3_<gait> shell aliases send with `ros2 topic pub --once`.
/step_up (Bool, from the mission manager): true while the robot drives to a tile that is higher than the one it stands on
(terrain classifier elevation option): the step_up_gait (rug_hstep) is used instead of the terrain gait.
Switching rules (flat delay, minimum interval) live in terrain_gait.py. The gait only changes when the robot starts
for a new tile (/mission_manager/next_target names a tile) or when a new terrain is classified (/terrain_type changes);
a finished spin / animation brings the terrain gait back. While the robot is in AI mode (/ai_mode true: the mission
manager's stairs mode climbs a staircase) no gait is sent at all: a gait command would take it out of AI mode.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from transfer_interfaces.msg import MotionSimpleCMD

from lite3_explorer.terrain_gait import GAIT_CODES, GaitCommand, TerrainGaitSwitcher


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
        self.create_subscription(Bool, '/spin_crawl', self._on_crawl_hold, 10)
        self.create_subscription(Bool, '/spin_flat', self._on_flat_request, 10)
        self.animating = False                            # no gait commands while a robot animation runs
        self.pre_animation_gait = None                    # the gait in use when the animation started: sent again after
        self.create_subscription(Bool, '/animation_active', self._on_animation, 10)
        self.ai_mode = False                              # no gait commands while the robot is in AI mode (stairs)
        self.create_subscription(Bool, '/ai_mode', self._on_ai_mode, latched)
        self.create_timer(0.2, self._tick)
        self.get_logger().info(f'gait switcher: non_rugged -> {self.switcher.flat_gait}, '
                               f'rugged -> {self.switcher.rugged_gait}')

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

    def _on_step_up(self, msg):
        if self.switcher.set_override(self.step_up_gait if msg.data else None):
            self.get_logger().info(f'higher tile ahead: {"h-step gait" if msg.data else "terrain gait again"}')

    def _send(self, cmd, now):
        out = MotionSimpleCMD()
        out.cmd_code, out.type = cmd.cmd_code, 0
        self.cmd_pub.publish(out)
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
            self._publish_gait('AI mode: stairs')
        elif was and not self.ai_mode:
            self.switcher.restore()                       # AI mode is over: restore the terrain gait

    def _repeat_burst(self, now):
        out, left, last = self._burst
        if left <= 0:
            self._burst = None
        elif now - last >= self.repeat_period_s:
            self.cmd_pub.publish(out)
            self._burst = (out, left - 1, now)

    def _tick(self):
        now = self._now()
        if self.ai_mode:
            return
        if self._burst is not None:
            self._repeat_burst(now)
        if self.crawl_hold or self.animating:
            return
        # Don't send (and record) a gait before the robot's subscriber is matched: it would be lost.
        if self.cmd_pub.get_subscription_count() == 0:
            return
        cmd = self.switcher.pending_command(now)
        if cmd is not None:
            self._send(cmd, now)


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
