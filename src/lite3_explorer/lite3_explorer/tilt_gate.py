"""Pass /scan_raw to /scan only while the body is as flat as it was at the start of the run.

The robot IMU and the lidar's own IMU each give a gravity direction. The first reference_s seconds after the first reading of
each (the robot stands still then) are the reference. A scan is dropped while either IMU is tilted more than max_tilt_deg away
from its reference, plus hold_s afterwards (the body is still settling, and the filtered gravity lags). slam_toolbox keeps its
last pose, odometry carries the robot over the stairs, and scan matching resumes on the flat.

An IMU that never publishes is ignored (warned once), so a missing topic cannot freeze mapping.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool

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
        self.trackers = {n: GravityTracker(ref_s, alpha) for n, t in imus.items() if t}
        for name, topic in imus.items():
            if topic:
                self.create_subscription(Imu, topic, lambda m, n=name: self._on_imu(n, m), qos_profile_sensor_data)
        self.pub = self.create_publisher(LaserScan, self.scan_out, qos_profile_sensor_data)
        self.state_pub = self.create_publisher(Bool, '/tilt_gate/blocking', 1)
        self.create_subscription(LaserScan, self.scan_in, self._on_scan, qos_profile_sensor_data)
        self.create_timer(5.0, self._warn_missing)
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
