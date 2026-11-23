#!/usr/bin/env python3
"""Emergency stop for the stair climber. Start it BEFORE enabling the node:

    python3 ~/ros2_ws/src/stair_climber/tools/estop.py

Press ENTER at any time: the stair node is disabled at once (it stops sending
velocities and gait changes) and the robot is told to stand still. The robot stays
in auto mode; run `lite3_manual` afterwards to take over with the handheld controller.

Unlike `ros2 topic pub --once`, this is already connected, so there is no 1-2 s delay.
"""
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


def main():
    rclpy.init()
    node = rclpy.create_node('stair_estop')
    enable = node.create_publisher(Bool, '/stair_climber/enable', 10)
    # same QoS as the robot's /cmd_vel subscriber (TRANSIENT_LOCAL), or it is ignored
    cmd = node.create_publisher(Twist, '/cmd_vel', QoSProfile(
        depth=1, reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL))
    t_end = time.monotonic() + 1.0
    while time.monotonic() < t_end:                  # let discovery finish
        rclpy.spin_once(node, timeout_sec=0.1)
    print('E-STOP READY. Press ENTER to stop the stair climber.')
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print('quit WITHOUT stopping anything (press ENTER, not Ctrl+C, to stop the robot)')
        return
    for _ in range(20):                              # 1 s of repeats
        enable.publish(Bool(data=False))
        cmd.publish(Twist())
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(0.05)
    print('STOPPED: stair node disabled, zero velocity sent.')
    print('To drive with the handheld controller now, run:  lite3_manual')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
