"""Pass /scan_raw to /scan only while the body is as flat as it was at the start of the run.

The robot IMU and the lidar's own IMU each give a gravity direction. The first reference_s seconds after the first reading of
each (the robot stands still then) are the reference. A scan is dropped while either IMU is tilted more than max_tilt_deg away
from its reference, plus hold_s afterwards (the body is still settling, and the filtered gravity lags). slam_toolbox keeps its
last pose, odometry carries the robot over the stairs, and scan matching resumes on the flat.

While scans are dropped slam_toolbox publishes no new map->odom (it stamps that transform from the last scan it processed), so
Nav2 fails with "extrapolation into the future" on odom->map for as long as the gate blocks. The gate therefore keeps map->odom
alive meanwhile: it caches slam_toolbox's last transform while the body is level and republishes that one, stamped now, at
keepalive_hz while blocking (the odometry carries the robot over the step; slam_toolbox's correction is frozen at its last value).

An IMU that never publishes is ignored (warned once), so a missing topic cannot freeze mapping.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from lite3_explorer.tilt_core import GravityTracker, too_tilted


class TiltGate(Node):
    def __init__(self):
        super().__init__('tilt_gate')
        p = self.declare_parameter
        self.scan_in = p('scan_in', '/scan_raw').value
        self.scan_out = p('scan_out', '/scan').value
        imus = {'robot': p('robot_imu_topic', '/imu/data').value, 'lidar': p('lidar_imu_topic', '/livox/imu').value}
        self.max_tilt = math.radians(p('max_tilt_deg', 6.0).value)
        self.hold_s = p('hold_s', 1.0).value
        ref_s = p('reference_s', 3.0).value
        alpha = p('filter_alpha', 0.2).value
        self.map_frame = p('map_frame', 'map').value
        self.odom_frame = p('odom_frame', 'odom').value
        self.keepalive_hz = p('keepalive_hz', 20.0).value          # 0 = do not keep map->odom alive while blocking
        self.keepalive_lead_s = p('keepalive_lead_s', 0.3).value   # stamp the held transform this far ahead of now
        self.trackers = {n: GravityTracker(ref_s, alpha) for n, t in imus.items() if t}
        for name, topic in imus.items():
            if topic:
                self.create_subscription(Imu, topic, lambda m, n=name: self._on_imu(n, m), qos_profile_sensor_data)
        self.pub = self.create_publisher(LaserScan, self.scan_out, qos_profile_sensor_data)
        self.state_pub = self.create_publisher(Bool, '/tilt_gate/blocking', 1)
        self.create_subscription(LaserScan, self.scan_in, self._on_scan, qos_profile_sensor_data)
        self.create_timer(5.0, self._warn_missing)
        self._held = None                                # last map->odom from slam_toolbox while the body was level
        if self.keepalive_hz > 0:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.broadcaster = TransformBroadcaster(self)
            self.create_timer(1.0 / self.keepalive_hz, self._keepalive)
        self._blocked_until = 0.0
        self._blocking = False
        self._dropped = 0
        self._warned = False
        self.get_logger().info(f'{self.scan_in} -> {self.scan_out}, IMUs {imus}, max tilt {math.degrees(self.max_tilt):.1f} deg')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_imu(self, name, msg: Imu):
        a = msg.linear_acceleration
        tr = self.trackers[name]
        was = tr.ready
        tr.add(self._now(), (a.x, a.y, a.z))
        if tr.ready and not was:
            self.get_logger().info(f'{name} IMU reference taken (flat start pose).')

    def _warn_missing(self):
        if self._warned:
            return
        silent = [n for n, t in self.trackers.items() if t.current is None]
        if silent:
            self._warned = True
            self.get_logger().warn(f'No readings yet from the {silent} IMU: not gating on it.')

    def _keepalive(self):
        if self.now_blocking():
            if self._held is not None:
                tf = TransformStamped()
                tf.header.frame_id, tf.child_frame_id = self.map_frame, self.odom_frame
                tf.header.stamp = (self.get_clock().now() + Duration(seconds=self.keepalive_lead_s)).to_msg()
                tf.transform = self._held.transform
                self.broadcaster.sendTransform(tf)
            return
        try:
            self._held = self.tf_buffer.lookup_transform(self.map_frame, self.odom_frame, Time())
        except TransformException:
            pass

    def now_blocking(self):
        return self._now() < self._blocked_until

    def _on_scan(self, msg: LaserScan):
        now = self._now()
        bad = too_tilted({n: t.tilt() for n, t in self.trackers.items()}, self.max_tilt)
        if bad:
            self._blocked_until = now + self.hold_s
            if not self._blocking:
                tilt = math.degrees(self.trackers[bad].tilt())
                self.get_logger().warn(f'{bad} IMU tilted {tilt:.1f} deg from its start reference: ignoring scans.')
        blocking = now < self._blocked_until
        if blocking != self._blocking:
            self._blocking = blocking
            self.state_pub.publish(Bool(data=blocking))
            if not blocking:
                self.get_logger().info(f'Body flat again; scans resume (dropped {self._dropped}).')
                self._dropped = 0
        if blocking:
            self._dropped += 1
            return
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TiltGate()
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
