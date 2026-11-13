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
from std_msgs.msg import String
from transfer_interfaces.msg import MotionSimpleCMD

from lite3_kitchen_explorer.terrain_gait import TerrainGaitSwitcher


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
        self.cmd_pub = self.create_publisher(MotionSimpleCMD, '/simple_cmd', 10)
        self.create_timer(0.2, self._tick)
        self.get_logger().info(f'gait switcher: non_rugged -> {self.switcher.flat_gait}, '
                               f'rugged -> {self.switcher.rugged_gait}')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_terrain(self, msg):
        if self.switcher.update_terrain(msg.data, self._now()):
            self.get_logger().info(f'terrain: {msg.data}')

    def _tick(self):
        now = self._now()
        cmd = self.switcher.pending_command(now)
        if cmd is None:
            return
        out = MotionSimpleCMD()
        out.cmd_code, out.type = cmd.cmd_code, 0
        self.cmd_pub.publish(out)
        self.switcher.mark_sent(cmd, now)
        self.get_logger().info(f'gait -> {cmd.gait} (0x{cmd.cmd_code:08X})')


def main(args=None):
    rclpy.init(args=args)
    node = GaitSwitcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
