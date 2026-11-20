"""Switch the Lite3 gait from the terrain classifier.

  /terrain_type rugged     -> rug_general  (0x21010401, same as lite3_rug_general)
  /terrain_type non_rugged -> flat_medium  (0x21010307, same as lite3_flat_medium)

Subscribes /terrain_type (std_msgs/String, latched) and publishes a
transfer_interfaces/MotionSimpleCMD {cmd_code, type: 0} on /simple_cmd, the same
message the lite3_<gait> shell aliases send with `ros2 topic pub --once`.
Switching rules (flat delay, minimum interval) live in terrain_gait.py.
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
            rugged_gait=p('rugged_gait', 'rug_general'),
            flat_delay_s=p('flat_delay_s', 2.0),
            min_switch_interval_s=p('min_switch_interval_s', 1.0))
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, p('terrain_topic', '/terrain_type'), self._on_terrain, latched)
        # Match the domain bridge subscriber (RELIABLE + TRANSIENT_LOCAL); a VOLATILE publisher won't connect.
        cmd_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.create_publisher(MotionSimpleCMD, '/simple_cmd', cmd_qos)
        # While the explorer spins to look around, hold the crawl gait (set from the reporter).
        self.crawl_hold = False
        self.hold_gait = 'flat_crawl'                     # gait held while crawl_hold is set
        self.repeats = int(p('command_repeats', 3))       # publishes per switch (a lone one can be lost)
        self.repeat_period_s = p('repeat_period_s', 0.3)
        self._burst = None                                # (MotionSimpleCMD, publishes left)
        self.create_subscription(Bool, '/spin_crawl', self._on_crawl_hold, 10)
        self.create_subscription(Bool, '/spin_flat', self._on_flat_request, 10)
        self.animating = False                            # no gait commands while a robot animation runs
        self.create_subscription(Bool, '/animation_active', self._on_animation, 10)
        self.create_timer(0.2, self._tick)
        self.get_logger().info(f'gait switcher: non_rugged -> {self.switcher.flat_gait}, '
                               f'rugged -> {self.switcher.rugged_gait}')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_terrain(self, msg):
        if self.switcher.update_terrain(msg.data, self._now()):
            self.get_logger().info(f'terrain: {msg.data}')

    def _send(self, cmd, now):
        out = MotionSimpleCMD()
        out.cmd_code, out.type = cmd.cmd_code, 0
        self.cmd_pub.publish(out)
        self._burst = (out, self.repeats - 1, now)
        self.switcher.mark_sent(cmd, now)
        self.get_logger().info(f'gait -> {cmd.gait} (0x{cmd.cmd_code:08X})')

    def _on_crawl_hold(self, msg):
        now = self._now()
        was_holding, self.crawl_hold = self.crawl_hold, bool(msg.data)
        if self.crawl_hold:
            # Sent at once (no interval wait): the reporter starts the spin after a short pause.
            self.hold_gait = 'flat_crawl'
            self._send(GaitCommand('flat_crawl', GAIT_CODES['flat_crawl']), now)
        elif was_holding:
            # Release: the next tick restores the gait for the current terrain.
            self.switcher.gait = None

    def _on_flat_request(self, msg):
        # Hold flat_slow for the opening spin: the terrain loop must not switch back until the reporter releases it.
        if msg.data:
            self.crawl_hold = True
            self.hold_gait = 'flat_slow'
            self._send(GaitCommand('flat_slow', GAIT_CODES['flat_slow']), self._now())
        elif self.crawl_hold and self.hold_gait == 'flat_slow':
            self.crawl_hold = False
            self.switcher.gait = None                     # the next tick restores the gait for the current terrain

    def _on_animation(self, msg):
        was, self.animating = self.animating, bool(msg.data)
        if was and not self.animating:
            self.switcher.gait = None                     # move mode is back: restore the terrain gait

    def _repeat_burst(self, now):
        out, left, last = self._burst
        if left <= 0:
            self._burst = None
        elif now - last >= self.repeat_period_s:
            self.cmd_pub.publish(out)
            self._burst = (out, left - 1, now)

    def _tick(self):
        now = self._now()
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
