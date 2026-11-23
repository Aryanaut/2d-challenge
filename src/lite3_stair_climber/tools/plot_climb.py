#!/usr/bin/env python3
"""Plot body pitch and odometry height from a rosbag of a MANUAL stair run.

Use it to choose climb_start_pitch_deg, level_pitch_deg, min_height_gain, ...

    ros2 bag record -o stairs_run1 /imu/data /leg_odom2 <cloud_topic>
    # drive over the stairs by hand in RUG gait, Ctrl+C
    python3 tools/plot_climb.py stairs_run1 --nose-up-sign -1

Needs a sourced ROS 2 environment (rosbag2_py) and matplotlib.
"""

import argparse
import math

import matplotlib.pyplot as plt
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def quat_pitch_roll(q):
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    return pitch, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('--imu', default='/imu/data')
    ap.add_argument('--odom', default='/leg_odom2')
    ap.add_argument('--nose-up-sign', type=float, default=-1.0)
    ap.add_argument('--climb-start', type=float, default=8.0)
    ap.add_argument('--level', type=float, default=4.0)
    a = ap.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=a.bag, storage_id=''),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    t0 = None
    imu_t, nose, roll = [], [], []
    odom_t, z, x = [], [], []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in (a.imu, a.odom):
            continue
        t0 = t if t0 is None else t0
        ts = (t - t0) * 1e-9
        msg = deserialize_message(data, get_message(types[topic]))
        if topic == a.imu:
            p, r = quat_pitch_roll(msg.orientation)
            imu_t.append(ts)
            nose.append(math.degrees(a.nose_up_sign * p))
            roll.append(math.degrees(r))
        else:
            odom_t.append(ts)
            z.append(msg.pose.pose.position.z)
            x.append(msg.pose.pose.position.x)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(11, 8))
    ax[0].plot(imu_t, nose, label='nose-up pitch (deg)')
    for v, s in ((a.climb_start, '--'), (-a.climb_start, '--'), (a.level, ':'), (-a.level, ':')):
        ax[0].axhline(v, ls=s, c='gray')
    ax[0].legend(); ax[0].grid()
    ax[1].plot(imu_t, roll, c='tab:red', label='roll (deg)'); ax[1].legend(); ax[1].grid()
    ax[2].plot(odom_t, z, c='tab:green', label='odom z (m)')
    ax2 = ax[2].twinx(); ax2.plot(odom_t, x, c='tab:purple', alpha=.4, label='odom x (m)')
    ax[2].legend(loc='upper left'); ax2.legend(loc='upper right'); ax[2].grid()
    ax[2].set_xlabel('time (s)')
    if nose:
        print(f'nose-up: max {max(nose):.1f} deg, min {min(nose):.1f} deg')
        print(f'roll: max |{max(abs(r) for r in roll):.1f}| deg')
    if z:
        print(f'odom z: min {min(z):.3f} m, max {max(z):.3f} m (gain {max(z) - z[0]:.3f} m)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
