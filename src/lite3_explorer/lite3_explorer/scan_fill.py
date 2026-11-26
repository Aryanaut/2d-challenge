"""/scan -> /scan_slam for slam_toolbox: rays with no return (inf / nan / out of range) inside the front sector become
`fill_m` (just below the scan's range_max).

slam_toolbox (Karto) ignores readings >= the scan's range_max (inf included), so a ray into open space clears nothing and
open floor beyond the maze stays unknown (bag endconditiontest_25_11_203502: the corridor's exit tile 1,0 stayed 56 %
mapped, below tiles.min_known, and was never a target). A reading between max_laser_range (slam_toolbox.yaml, 1.7 m) and
range_max is traced as free up to max_laser_range with no wall at its end.

Only within +-fill_half_angle_deg of straight ahead: behind the robot the body blocks the lidar (60-97 % of rays beyond
+-110 deg have no return in bags endconditiontest 201356 / 203502), and filling there would clear walls behind it.
Every other /scan consumer (Nav2, mission manager, LLM lidar summary) keeps the unfilled /scan.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def fill_ranges(ranges, angle_min: float, angle_increment: float, range_min: float, range_max: float,
                fill_m: float, half_angle_rad: float) -> np.ndarray:
    r = np.asarray(ranges, dtype=np.float32).copy()
    ang = angle_min + np.arange(len(r)) * angle_increment
    ang = np.arctan2(np.sin(ang), np.cos(ang))
    none = ~(np.isfinite(r) & (r >= range_min) & (r < range_max))
    r[none & (np.abs(ang) <= half_angle_rad)] = fill_m
    return r


class ScanFill(Node):
    def __init__(self):
        super().__init__('scan_fill')
        p = self.declare_parameter
        self.half = math.radians(p('fill_half_angle_deg', 100.0).value)
        self.margin = p('fill_below_range_max_m', 0.5).value
        self.pub = self.create_publisher(LaserScan, p('scan_out', '/scan_slam').value, qos_profile_sensor_data)
        self.create_subscription(LaserScan, p('scan_in', '/scan').value, self._on_scan, qos_profile_sensor_data)

    def _on_scan(self, msg: LaserScan):
        fill = max(msg.range_min, msg.range_max - self.margin)
        msg.ranges = fill_ranges(msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max,
                                 fill, self.half).tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ScanFill()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
