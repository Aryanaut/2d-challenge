#!/usr/bin/env python3
"""Flat-floor test of AI mode, sending EXACTLY what the stair node sends (same QoS).

    python3 ~/ros2_ws/src/stair_climber/tools/ai_walk_test.py --speed 0.1 --repeats 1

Run `lite3_auto` first. Robot on FLAT floor, 2 m of free space in front.
Steps: Enter AI (x repeats) -> stand still `--wait` s (zero /cmd_vel, like the node)
-> forward at `--speed` for `--time` s -> zero -> Exit AI.
Prints how far /leg_odom2 says it moved, so you can compare with what you saw.
Ctrl+C at any point: sends zero velocity and Exit AI.
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from transfer_interfaces.msg import MotionSimpleCMD

ENTER_AI, EXIT_AI = 0x21010528, 0x2101052B
TL = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=0.1)
    ap.add_argument('--time', type=float, default=3.0)
    ap.add_argument('--wait', type=float, default=5.0)
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--no-ai', action='store_true', help='skip Enter/Exit AI (normal gait)')
    a = ap.parse_args()

    try:   # keep ROS alive on Ctrl+C so we can still send zero + Exit AI
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    n = rclpy.create_node('ai_walk_test')
    cmd = n.create_publisher(Twist, '/cmd_vel', TL)
    simple = n.create_publisher(MotionSimpleCMD, '/simple_cmd', TL)
    pos = {}
    n.create_subscription(Odometry, '/leg_odom2',
                          lambda m: pos.update(x=m.pose.pose.position.x, y=m.pose.pose.position.y), 10)

    def spin_for(sec, vx):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            tw = Twist()
            tw.linear.x = vx
            cmd.publish(tw)
            rclpy.spin_once(n, timeout_sec=0.05)     # ~20 Hz, like the node

    def send(code):
        for _ in range(a.repeats):
            m = MotionSimpleCMD()
            m.cmd_code, m.type = code, 0
            simple.publish(m)
            print(f'sent /simple_cmd {hex(code)}')
            rclpy.spin_once(n, timeout_sec=0.05)

    try:
        spin_for(1.0, 0.0)                           # discovery + first odom
        if 'x' not in pos:
            print('WARNING: no /leg_odom2 received')
        if not a.no_ai:
            send(ENTER_AI)
        print(f'standing {a.wait:.0f} s (zero velocity)...')
        spin_for(a.wait, 0.0)
        x0, y0 = pos.get('x', math.nan), pos.get('y', math.nan)
        print(f'forward at {a.speed} m/s for {a.time:.0f} s  <-- WATCH: does it walk?')
        spin_for(a.time, a.speed)
        spin_for(1.0, 0.0)
        moved = math.hypot(pos.get('x', math.nan) - x0, pos.get('y', math.nan) - y0)
        print(f'odometry says it moved {moved:.2f} m (expected ~{a.speed * a.time:.2f})')
    except KeyboardInterrupt:
        print('interrupted')
    finally:
        spin_for(0.5, 0.0)
        if not a.no_ai:
            send(EXIT_AI)
            spin_for(0.5, 0.0)
        print('done (zero velocity sent, AI exited)')
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
