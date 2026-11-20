"""Play robot animations (see animations.py).

  /play_animation   (std_msgs/String)  animation name, e.g. `quickscan`; `stop` cancels the current one
  /animation_active (std_msgs/Bool)    true while an animation is running (gait_switcher pauses on it)

Publishes transfer_interfaces/MotionSimpleCMD on /simple_cmd at publish_rate_hz, so pose commands
are held for as long as the animation needs them (like `ros2 topic pub -r 20`).

  ros2 topic pub --once /play_animation std_msgs/msg/String "{data: quickscan}"
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from transfer_interfaces.msg import MotionSimpleCMD

from lite3_explorer.animations import ANIMATIONS


class AnimationPlayer(Node):
    def __init__(self):
        super().__init__('animation_player')
        p = lambda name, default: self.declare_parameter(name, default).value  # noqa: E731
        self.params = {
            'quickscan': dict(pause_s=p('quickscan.pause_s', 0.5), move_s=p('quickscan.move_s', 1.0),
                              lean=p('quickscan.lean', 20000), yaw=p('quickscan.yaw', 20000)),
        }
        self.common = dict(mode_settle_s=p('mode_settle_s', 1.0), mode_repeats=p('mode_repeats', 3))
        # Match the domain bridge subscriber (RELIABLE + TRANSIENT_LOCAL), as gait_switcher does.
        cmd_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.create_publisher(MotionSimpleCMD, '/simple_cmd', cmd_qos)
        self.active_pub = self.create_publisher(Bool, '/animation_active', 10)
        self.create_subscription(String, p('animation_topic', '/play_animation'), self._on_play, 10)
        self.anim = None
        self.create_timer(1.0 / p('publish_rate_hz', 20.0), self._tick)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish(self, cmds):
        for code, size in cmds:
            msg = MotionSimpleCMD()
            msg.cmd_code, msg.size, msg.type = code, size, 0
            self.cmd_pub.publish(msg)

    def _set_active(self, active):
        self.active_pub.publish(Bool(data=active))

    def _on_play(self, msg):
        name = msg.data.strip().lower()
        if name == 'stop':
            if self.anim is not None:
                self._publish(self.anim.cancel(self._now()))
                self.anim = None
                self._set_active(False)
                self.get_logger().info('animation cancelled')
            return
        if name not in ANIMATIONS:
            self.get_logger().warn(f'unknown animation {name!r}; use one of {sorted(ANIMATIONS)}')
            return
        if self.anim is not None:
            self.get_logger().warn(f'{self.anim.name} is still running; ignoring {name}')
            return
        if self.cmd_pub.get_subscription_count() == 0:
            self.get_logger().warn('no /simple_cmd subscriber yet; ignoring ' + name)
            return
        kwargs = {**self.common, **self.params.get(name, {})}
        self.anim = ANIMATIONS[name](**kwargs)
        self.anim.start(self._now())
        self._set_active(True)
        self.get_logger().info(f'animation {name} ({self.anim.category}) started')

    def _tick(self):
        if self.anim is None:
            return
        self._publish(self.anim.tick(self._now()))
        if self.anim.finished:
            self.get_logger().info(f'animation {self.anim.name} finished')
            self.anim = None
            self._set_active(False)


def main(args=None):
    rclpy.init(args=args)
    node = AnimationPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.anim is not None:                      # leave pose mode on Ctrl-C
            node._publish(node.anim.cancel(node._now()))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
