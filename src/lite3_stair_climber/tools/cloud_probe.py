#!/usr/bin/env python3
"""Print what the lidar sees straight ahead, as a height profile.

    python3 cloud_probe.py            # robot standing still, level

For each 10 cm slice in front of the robot (central strip |y| < 0.3 m) it prints
how many points there are and how high they are above the floor. Use it to find
the lidar's close-range blind zone and to see what the stair detector sees.
"""
import math

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

TOPIC = '/livox/points'
T = np.array([0.18678, 0.0, 0.12498])          # lidar position in base_link
P = math.radians(15.0)                           # lidar pitch (tilted down)
R = np.array([[math.cos(P), 0, math.sin(P)], [0, 1, 0], [-math.sin(P), 0, math.cos(P)]])


def main():
    rclpy.init()
    node = rclpy.create_node('cloud_probe')
    got = {}
    node.create_subscription(PointCloud2, TOPIC, lambda m: got.setdefault('m', m),
                             qos_profile_sensor_data)
    frames = []
    while len(frames) < 5:                       # merge 5 scans (0.5 s) for density
        got.clear()
        while 'm' not in got:
            rclpy.spin_once(node, timeout_sec=1.0)
        s = point_cloud2.read_points(got['m'], field_names=('x', 'y', 'z'), skip_nans=True)
        frames.append(np.stack([s['x'], s['y'], s['z']], axis=1).astype(float))
    pts = np.vstack(frames) @ R.T + T

    near = pts[(np.abs(pts[:, 1]) < 1.5) & (pts[:, 0] > 0.0) & (pts[:, 0] < 4.0)]
    floor_z = np.percentile(near[:, 2], 5)
    print(f'{len(pts)} points, floor_z ~ {floor_z:.3f} m (base_link)')
    c = pts[(np.abs(pts[:, 1]) < 0.3) & (pts[:, 0] > 0.0)]
    print('  x range  | points | height above floor: 20%   80%   max  (cm)')
    for x0 in np.arange(0.0, 3.0, 0.1):
        b = c[(c[:, 0] >= x0) & (c[:, 0] < x0 + 0.1)]
        if len(b) == 0:
            print(f' {x0:4.1f}-{x0 + 0.1:3.1f} |      0 |   (nothing seen)')
            continue
        h = (b[:, 2] - floor_z) * 100
        print(f' {x0:4.1f}-{x0 + 0.1:3.1f} | {len(b):6d} |  {np.percentile(h, 20):6.0f} '
              f'{np.percentile(h, 80):5.0f} {h.max():5.0f}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
