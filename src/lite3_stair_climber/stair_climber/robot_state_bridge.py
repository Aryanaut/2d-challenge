#!/usr/bin/env python3
"""Robot state (UDP 0x0901 from the motion host, 50 Hz) -> ROS.

Publishes
    /robot_state        std_msgs/Int32MultiArray   [robot_basic_state, robot_gait_state, robot_policy_state,
                                                    robot_motion_state]   every packet (50 Hz)
    /robot_state/text   std_msgs/String (latched)  '(6,13,1) stepping, RUG H-step  battery 87%' on every change

stair_climber and the mission manager read /robot_state to check that a gait / AI command took effect and send it
again until it has. Run this node on the computer the motion host sends its state reports to.

Parameters
    bind_ip     ''       listen on all interfaces
    port        43897    UDP port the motion host reports to (check the robot's network setup)
    report_rate 50.0     publish at most this often
"""
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32MultiArray, String

from .robot_state import describe, parse


class RobotStateBridge(Node):

    def __init__(self):
        super().__init__('robot_state_bridge')
        p = lambda n, d: self.declare_parameter(n, d).value  # noqa: E731
        ip, port = p('bind_ip', ''), int(p('port', 43897))
        self.min_period = 1.0 / max(1e-3, float(p('report_rate', 50.0)))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, 'SO_REUSEPORT'):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)
        self.pub = self.create_publisher(Int32MultiArray, '/robot_state', 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.text_pub = self.create_publisher(String, '/robot_state/text', latched)
        self.last_mode, self.last_pub, self.last_rx, self.packets = None, 0.0, None, 0
        self.create_timer(0.005, self._poll)
        self.create_timer(2.0, self._watchdog)
        self.get_logger().info(f'robot_state_bridge: listening for 0x0901 on udp {ip or "*"}:{port}')

    def _poll(self):
        while True:
            try:
                data = self.sock.recv(2048)
            except (BlockingIOError, InterruptedError):
                return
            s = parse(data)
            if s is None:
                continue
            now = time.monotonic()
            self.packets += 1
            self.last_rx = now
            changed = s.mode != self.last_mode
            if changed or now - self.last_pub >= self.min_period:
                self.pub.publish(Int32MultiArray(data=list(s.mode)))
                self.last_pub = now
            if changed:
                self.last_mode = s.mode
                text = f'{describe(s)}  battery {s.battery:.0f}%' + (f'  error 0x{s.error:X}' if s.error else '')
                self.text_pub.publish(String(data=text))
                self.get_logger().info(f'robot state {text}')

    def _watchdog(self):
        if self.last_rx is None or time.monotonic() - self.last_rx > 2.0:
            self.get_logger().warn('no 0x0901 robot state packets: is the motion host reporting to this computer / '
                                   'port? gait checks fall back to timed waits', throttle_duration_sec=10.0)


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001  SIGTERM (tmux session stop): the context is already shut down
        if rclpy.ok():
            raise
    finally:
        node.sock.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
