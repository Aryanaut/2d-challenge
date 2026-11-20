"""Play one robot animation through the animation_player and exit when it has finished.

  ros2 run lite3_explorer animation_test --ros-args -p animation:=quickscan
(the launch file test_animation.launch.py starts animation_player and this node together)

Exit code 0 when the animation ran to the end, 1 on a timeout or when the player never started it.
"""
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

from lite3_explorer.animations import ANIMATIONS


class AnimationTest(Node):
    def __init__(self):
        super().__init__('animation_test')
        self.name = self.declare_parameter('animation', 'quickscan').value
        self.timeout_s = self.declare_parameter('timeout_s', 60.0).value
        self.start_timeout_s = self.declare_parameter('start_timeout_s', 10.0).value
        self.result = None                      # True = ran to the end, False = failed
        self.started = False
        if self.name not in ANIMATIONS:
            self.get_logger().error(f'unknown animation {self.name!r}; use one of {sorted(ANIMATIONS)}')
            self.result = False
            return
        self.pub = self.create_publisher(String, '/play_animation', 10)
        self.create_subscription(Bool, '/animation_active', self._on_active, 10)
        self.t0 = self.get_clock().now()
        self.sent_at = None
        self.create_timer(0.2, self._tick)

    def _elapsed(self, since):
        return (self.get_clock().now() - since).nanoseconds * 1e-9

    def _on_active(self, msg):
        if msg.data and not self.started:
            self.started = True
            self.get_logger().info(f'{self.name} started')
        elif not msg.data and self.started:
            self.get_logger().info(f'{self.name} finished')
            self.result = True

    def _tick(self):
        if self.sent_at is None:
            if self.pub.get_subscription_count() == 0:        # wait for animation_player
                if self._elapsed(self.t0) > self.start_timeout_s:
                    self.get_logger().error('animation_player is not running (nobody listens on /play_animation)')
                    self.result = False
                return
            self.pub.publish(String(data=self.name))
            self.sent_at = self.get_clock().now()
            self.get_logger().info(f'playing {self.name} ({ANIMATIONS[self.name].category})')
            return
        if not self.started and self._elapsed(self.sent_at) > self.start_timeout_s:
            self.get_logger().error('the animation did not start (no /simple_cmd subscriber yet? see animation_player log)')
            self.result = False
        elif self._elapsed(self.sent_at) > self.timeout_s:
            self.get_logger().error(f'{self.name} did not finish within {self.timeout_s:.0f} s')
            self.pub.publish(String(data='stop'))
            self.result = False


def main(args=None):
    rclpy.init(args=args)
    node = AnimationTest()
    try:
        while rclpy.ok() and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.pub.publish(String(data='stop'))          # leave pose mode
    code = 0 if node.result else 1
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(code)
