"""Broadcast odom -> base_link TF from the Lite3 leg odometry (/leg_odom2).

slam_toolbox and Nav2 need the odom->base_link transform, stamped on the SAME clock
as the lidar scans. /leg_odom2 is stamped by the Lite3 motion host, whose clock is
usually not synced with the perception host. Then slam_toolbox cannot match scans to
poses ("Message Filter dropping message ... queue is full") and the map frame never
appears.

stamp_mode:
  auto     (default) use the message stamp while it is within max_stamp_offset_s of
           this computer's clock, otherwise stamp with the receive time (and say so)
  now      always stamp with the receive time
  message  always keep the message stamp

If something else already publishes odom->base_link on /tf, this node does not
broadcast a second one (two parents break the TF tree). It still republishes
/odom and warns if the existing transform's stamps are off.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def stamp_to_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class OdomTfBridge(Node):
    def __init__(self):
        super().__init__('odom_tf_bridge')
        p = self.declare_parameter
        self.odom_topic = p('odom_topic', '/leg_odom2').value
        self.odom_frame = p('odom_frame', 'odom').value
        self.base_frame = p('base_frame', 'base_link').value
        self.republish_topic = p('republish_topic', '/odom').value
        self.stamp_mode = p('stamp_mode', 'auto').value
        self.max_stamp_offset_s = p('max_stamp_offset_s', 0.2).value
        self.auto_detect_s = p('auto_detect_s', 3.0).value
        if self.stamp_mode not in ('auto', 'now', 'message'):
            raise ValueError("stamp_mode must be 'auto', 'now' or 'message'")

        self.broadcast_tf = None if self.auto_detect_s > 0 else True
        self.using_receive_time = self.stamp_mode == 'now'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detect_timer = self.create_timer(self.auto_detect_s or 1.0, self._decide)
        self.broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, self.republish_topic, 10) if self.republish_topic else None
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos_profile_sensor_data)
        self.get_logger().info(f'{self.odom_topic} -> {self.odom_frame}->{self.base_frame} (stamp_mode={self.stamp_mode})')

    def _decide(self):
        self.detect_timer.cancel()
        if self.broadcast_tf is not None:
            return
        try:
            existing = self.tf_buffer.lookup_transform(self.odom_frame, self.base_frame, Time())
        except TransformException:
            self.broadcast_tf = True
            self.get_logger().info(f'Broadcasting {self.odom_frame}->{self.base_frame}.')
            return
        self.broadcast_tf = False
        offset = self.get_clock().now().nanoseconds * 1e-9 - stamp_to_s(existing.header.stamp)
        self.get_logger().warn(f'{self.odom_frame}->{self.base_frame} is already on /tf; not broadcasting it.')
        if abs(offset) > 1.0:
            self.get_logger().error(
                f'The existing {self.odom_frame}->{self.base_frame} TF is stamped {offset:+.1f} s away from this '
                'computer\'s clock. slam_toolbox will drop every scan. Sync the clocks (chrony/NTP), or stop '
                'the node publishing that TF and let this bridge publish it.')

    def _stamp_for(self, msg):
        now = self.get_clock().now()
        if self.stamp_mode == 'now' or not msg.header.stamp.sec:
            return now.to_msg()
        if self.stamp_mode == 'message':
            return msg.header.stamp
        offset = now.nanoseconds * 1e-9 - stamp_to_s(msg.header.stamp)
        bad = abs(offset) > self.max_stamp_offset_s
        if bad != self.using_receive_time:
            self.using_receive_time = bad
            if bad:
                self.get_logger().warn(
                    f'{self.odom_topic} stamps are {offset:+.3f} s off this computer\'s clock '
                    '(motion host clock not synced); stamping with receive time instead.')
            else:
                self.get_logger().info(f'{self.odom_topic} stamps are in sync again; using them.')
        return now.to_msg() if bad else msg.header.stamp

    def _on_odom(self, msg: Odometry):
        stamp = self._stamp_for(msg)
        if self.broadcast_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            pos = msg.pose.pose.position
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = pos.x, pos.y, pos.z
            t.transform.rotation = msg.pose.pose.orientation
            self.broadcaster.sendTransform(t)
        if self.odom_pub:
            msg.header.stamp = stamp
            msg.header.frame_id = self.odom_frame
            msg.child_frame_id = self.base_frame
            self.odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBridge()
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
