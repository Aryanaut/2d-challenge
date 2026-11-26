"""Broadcast odom -> base_link TF from the Lite3 leg odometry (/leg_odom2).

slam_toolbox and Nav2 need the odom->base_link transform, stamped on the SAME clock
as the lidar scans. /leg_odom2 is stamped by the Lite3 motion host, whose clock is
usually not synced with the perception host. Then slam_toolbox cannot match scans to
poses ("Message Filter dropping message ... queue is full") and the map frame never
appears.

AI gait (/ai_mode true): the robot's leg odometry is not trustworthy while it walks in AI mode (it jumps by metres and
degrees). The bridge then HOLDS the last odom->base_link it published, and when AI mode ends it carries on from the
held pose (an offset applied to the raw odometry). The mission manager then publishes the pose the robot is expected to
be at (the centre of the tile beyond the staircase, in the map frame) on /odom_rebase; the bridge converts it with the
latest map->odom and moves the odom pose there. slam_toolbox only has to scan-match the small remainder.

hold_until_rebase_s: after AI mode ends the pose stays HELD until that rebase arrives (at most this long, 0 = off). Bag
fullrun_25_11_183742: released at AI off, the robot (beyond the stairs) shifted a few cm while AI mode wound down, slam_toolbox
matched those scans at the PRE-climb pose and turned map->odom by up to 14 deg in the 2 s before the rebase, which was
then converted with that broken map->odom. Held, slam_toolbox sees no motion and integrates nothing until the rebase.
The rebase also publishes the old pose 1 ms before the new one, so no scan stamped in between gets a pose interpolated
half-way across the jump.

stamp_mode:
  auto     (default) use the message stamp while it is within max_stamp_offset_s of
           this computer's clock, otherwise stamp with the receive time (and say so)
  now      always stamp with the receive time
  message  always keep the message stamp

If something else already publishes odom->base_link on /tf, this node does not
broadcast a second one (two parents break the TF tree). It still republishes
/odom and warns if the existing transform's stamps are off.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def stamp_to_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def se2_mul(a, b):
    """a * b for SE(2) poses (x, y, yaw)."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2])


def se2_inv(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), s * a[0] - c * a[1], -a[2])


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

        self.map_frame = p('map_frame', 'map').value
        self.hold_in_ai_mode = p('hold_in_ai_mode', True).value
        # one odom message further than this from the last one is not motion (the leg odometry jumps by metres when AI
        # mode ends, a moment AFTER /ai_mode goes false): absorbed into the offset so the published pose stays continuous
        self.max_jump_m = p('max_jump_m', 0.5).value
        self.max_jump_rad = p('max_jump_rad', 1.0).value
        self.hold_until_rebase_s = p('hold_until_rebase_s', 10.0).value
        self.ai_on = False
        self.release_at = None            # AI mode over, still held: waiting for /odom_rebase until this time (s)
        self.rebased_in_ai = False        # a rebase already came while AI mode was on: no need to wait for another
        self.last_out = None
        # odom pose published = offset * raw leg odometry; frozen: publish `held` instead of tracking the raw pose
        self.offset = (0.0, 0.0, 0.0)
        self.raw = None
        self.frozen, self.held = False, None
        self.last_msg = None

        self.broadcast_tf = None if self.auto_detect_s > 0 else True
        self.using_receive_time = self.stamp_mode == 'now'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.detect_timer = self.create_timer(self.auto_detect_s or 1.0, self._decide)
        self.broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(Odometry, self.republish_topic, 10) if self.republish_topic else None
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/ai_mode', self._on_ai_mode, latched)
        self.create_subscription(PoseStamped, '/odom_rebase', self._on_rebase, 10)
        # RViz "2D Pose Estimate" (/initialpose): put the robot where the operator clicked, e.g. after a fall threw the
        # leg odometry off. Same as an /odom_rebase; slam_toolbox scan-matches the remainder.
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._on_initialpose, 10)
        self.get_logger().info(f'{self.odom_topic} -> {self.odom_frame}->{self.base_frame} (stamp_mode={self.stamp_mode})')

    def _output_pose(self):
        """The odom pose to publish: the held one while frozen, else offset * raw."""
        if self.frozen and self.held is not None:
            return self.held
        return se2_mul(self.offset, self.raw) if self.raw is not None else None

    def _now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _release(self, why):
        """Stop holding: carry on from the held pose (offset * raw_now == held)."""
        if self.raw is not None and self.held is not None:
            self.offset = se2_mul(self.held, se2_inv(self.raw))
        self.frozen, self.held, self.release_at, self.rebased_in_ai = False, None, None, False
        self.get_logger().info(f'{why}: odom->base_link continues from the held pose.')

    def _on_ai_mode(self, msg):
        on = bool(msg.data) and self.hold_in_ai_mode
        self.ai_on = on
        if on and not self.frozen:
            self.held = self._output_pose()
            self.frozen = self.held is not None
            self.release_at, self.rebased_in_ai = None, False
            if self.frozen:
                self.get_logger().info('AI mode on: holding odom->base_link at the last pose until it ends.')
        elif on and self.release_at is not None:
            self.release_at = None                     # AI mode on again before the rebase: keep holding
        elif not on and self.frozen and self.release_at is None:
            if self.hold_until_rebase_s > 0.0 and not self.rebased_in_ai:
                self.release_at = self._now_s() + self.hold_until_rebase_s
                self.get_logger().info(f'AI mode off: still holding odom->base_link until /odom_rebase (at most '
                                       f'{self.hold_until_rebase_s:.0f} s).')
            else:
                self._release('AI mode off')

    def _check_release(self):
        if self.release_at is not None and self.frozen and self._now_s() >= self.release_at:
            self.get_logger().warn('No /odom_rebase after AI mode: releasing the held pose anyway.')
            self._release('Hold timed out')

    def _on_initialpose(self, msg: PoseWithCovarianceStamped):
        self.get_logger().warn('2D Pose Estimate from RViz: moving the robot to the clicked pose.')
        self._on_rebase(PoseStamped(header=msg.header, pose=msg.pose.pose))

    def _on_rebase(self, msg: PoseStamped):
        """Move the robot to the pose in msg (map frame, or odom frame when msg.header.frame_id == odom_frame)."""
        want = (msg.pose.position.x, msg.pose.position.y, yaw_of(msg.pose.orientation))
        frame = msg.header.frame_id or self.map_frame
        if frame != self.odom_frame:
            try:
                mo = self.tf_buffer.lookup_transform(self.map_frame, self.odom_frame, Time())
            except TransformException as e:
                self.get_logger().error(f'Rebase to {frame} ignored: no {self.map_frame}->{self.odom_frame} ({e}).')
                return
            t = mo.transform
            want = se2_mul(se2_inv((t.translation.x, t.translation.y, yaw_of(t.rotation))), want)
        before = self.last_out
        if self.frozen and self.ai_on:
            self.held = want                                  # applied when AI mode ends
            self.rebased_in_ai = True
        elif self.raw is not None:
            self.offset = se2_mul(want, se2_inv(self.raw))
            if self.frozen:                                   # held after AI mode, waiting for exactly this
                self.frozen, self.held, self.release_at, self.rebased_in_ai = False, None, None, False
        else:
            self.get_logger().error('Rebase ignored: no leg odometry yet.')
            return
        self.get_logger().info(f'Rebased {self.odom_frame}->{self.base_frame} to ({want[0]:.2f}, {want[1]:.2f}, '
                               f'{math.degrees(want[2]):.0f} deg).')
        if self.last_msg is not None and not self.frozen:
            stamp = self._stamp_for(self.last_msg)
            if before is not None:
                # the old pose again 1 ms earlier: TF interpolates between samples, and the scans arrive ~1 s late, so a
                # scan stamped between the last old sample and the jump would get a pose half-way across it
                ns = stamp.sec * 1_000_000_000 + stamp.nanosec - 1_000_000
                self._publish(self.last_msg, pose=before, stamp=Time(nanoseconds=ns).to_msg())
            self._publish(self.last_msg, stamp=stamp)

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
        new = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_of(msg.pose.pose.orientation))
        if self.raw is not None and not self.frozen and self.last_out is not None:
            dyaw = math.atan2(math.sin(new[2] - self.raw[2]), math.cos(new[2] - self.raw[2]))
            step = math.hypot(new[0] - self.raw[0], new[1] - self.raw[1])
            if step > self.max_jump_m or abs(dyaw) > self.max_jump_rad:
                self.offset = se2_mul(self.last_out, se2_inv(new))          # output stays where it was
                self.get_logger().warn(f'{self.odom_topic} jumped {step:.2f} m / {math.degrees(dyaw):.0f} deg in one '
                                       'message: absorbed (not motion).')
        self.raw = new
        self.last_msg = msg
        self._check_release()
        self._publish(msg)

    def _publish(self, msg: Odometry, pose=None, stamp=None):
        """pose / stamp: publish this pose / stamp instead of the current output (the rebase's 'old pose' sample)."""
        stamp = stamp if stamp is not None else self._stamp_for(msg)
        x, y, yaw = pose if pose is not None else self._output_pose()
        if pose is None:
            self.last_out = (x, y, yaw)
        q = Quaternion(z=math.sin(0.5 * yaw), w=math.cos(0.5 * yaw))
        # roll / pitch of the raw odometry are dropped when the pose is corrected or held (SE(2) only)
        corrected = self.frozen or self.offset != (0.0, 0.0, 0.0)
        pos = msg.pose.pose.position
        if self.broadcast_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, pos.z
            t.transform.rotation = q if corrected else msg.pose.pose.orientation
            self.broadcaster.sendTransform(t)
        if self.odom_pub:
            out = Odometry()
            out.header.stamp = stamp
            out.header.frame_id = self.odom_frame
            out.child_frame_id = self.base_frame
            out.pose = msg.pose
            out.twist = msg.twist
            out.pose.pose.position.x, out.pose.pose.position.y = x, y
            if corrected:
                out.pose.pose.orientation = q
            self.odom_pub.publish(out)


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
