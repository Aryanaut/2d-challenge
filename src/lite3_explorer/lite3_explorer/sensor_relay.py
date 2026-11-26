"""Relay the robot's sensor topics to local copies, so each one crosses the link exactly once.

All robot sensors (/imu/data, /leg_odom2, /livox/points, the cameras) come from one robot process,
lite3_domain_bridge_1, which sends every message once per subscriber in a single loop. Every extra PC process that
subscribes to the cloud or a camera image adds a full copy per frame, the loop slows down, and the IMU / odometry
(one latest sample per loop) drop from 200 Hz to 1-3 Hz (bags imu_26_11_190945, explorer_26_11_191117).

This node is the only PC subscriber to those topics. It republishes each as <prefix><topic> (default /local/imu/data
...), and every other node subscribes to the local copy (exploration.launch.py remaps them).

  sub  <topic>            serialized, BEST_EFFORT (no resends asked of the robot), any type
  pub  <prefix><topic>    same bytes, RELIABLE (matches both reliable and best-effort local subscribers)

Lazy: the robot topic is only subscribed while its local copy has at least one subscriber (after linger_s with none,
it is dropped again), so e.g. the depth image costs nothing on a run without the curtain / stair / roller detectors.

Parameters
  topics    list of robot topics to relay
  prefix    '/local'
  linger_s  3.0   keep the robot subscription this long after the last local subscriber went away
  depth     queue depth for both sides (10)
"""
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message

DEFAULT_TOPICS = ['/imu/data', '/leg_odom2', '/livox/points', '/livox/imu',
                  '/camera/depth/image_raw', '/camera/depth/camera_info',
                  '/camera/color/image_raw', '/camera/color/image_raw/compressed', '/camera/color/camera_info']


class _Relay:
    def __init__(self, src, dst):
        self.src, self.dst = src, dst
        self.type_name = None
        self.pub = None
        self.sub = None
        self.idle_since = None
        self.count = 0
        self.group = MutuallyExclusiveCallbackGroup()   # one per topic: a 600 KB image never delays the IMU


class SensorRelay(Node):
    def __init__(self):
        super().__init__('sensor_relay')
        topics = self.declare_parameter('topics', DEFAULT_TOPICS).value
        prefix = '/' + self.declare_parameter('prefix', '/local').value.strip('/')
        self.linger_s = float(self.declare_parameter('linger_s', 3.0).value)
        depth = int(self.declare_parameter('depth', 10).value)
        self.sub_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth,
                                  reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.pub_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth,
                                  reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.relays = [_Relay(t, prefix + t) for t in dict.fromkeys(t for t in topics if t)]
        self.get_logger().info(f'Relaying {len(self.relays)} robot topics to {prefix}/... (subscribed only while used)')
        self.create_timer(0.5, self._update)
        self.create_timer(10.0, self._report)
        self._last_report = time.monotonic()

    def _update(self):
        now = time.monotonic()
        for r in self.relays:
            if r.pub is None:                       # the type is only known once the robot publishes the topic
                types = {info.topic_type for info in self.get_publishers_info_by_topic(r.src)}
                if not types:
                    continue
                if len(types) > 1:
                    self.get_logger().error(f'{r.src}: publishers disagree on the type {sorted(types)}; not relayed',
                                            throttle_duration_sec=30.0)
                    continue
                r.type_name = types.pop()
                r.pub = self.create_publisher(get_message(r.type_name), r.dst, self.pub_qos)
            wanted = r.pub.get_subscription_count() > 0
            if wanted and r.sub is None:
                r.sub = self.create_subscription(get_message(r.type_name), r.src, self._forward(r), self.sub_qos,
                                                 raw=True, callback_group=r.group)
                self.get_logger().info(f'{r.src} -> {r.dst} ({r.type_name}): subscribed')
            if wanted:
                r.idle_since = None
            elif r.sub is not None:
                r.idle_since = r.idle_since or now
                if now - r.idle_since >= self.linger_s:
                    self.destroy_subscription(r.sub)
                    r.sub, r.idle_since = None, None
                    self.get_logger().info(f'{r.src}: no local subscribers, unsubscribed')

    @staticmethod
    def _forward(r):
        publish = r.pub.publish

        def cb(data):
            publish(data)
            r.count += 1
        return cb

    def _report(self):
        """Count the messages relayed per topic (a cheap rate check without one more subscriber on the robot)."""
        now = time.monotonic()
        dt = now - self._last_report
        self._last_report = now
        rates = []
        for r in self.relays:
            if r.sub is not None:
                rates.append(f'{r.src} {r.count / dt:.0f} Hz')
            r.count = 0
        if rates:
            self.get_logger().info('relayed: ' + ', '.join(rates))


def main():
    rclpy.init()
    node = SensorRelay()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
