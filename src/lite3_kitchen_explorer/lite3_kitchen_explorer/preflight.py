"""Preflight check: run once on the dog before launching exploration.

    ros2 run lite3_kitchen_explorer preflight

Listens for a few seconds and reports what the launch file needs: the message types,
the frame_ids, whether the TF links exist, and whether something consumes /cmd_vel.
It prints the launch arguments to use. It does not move the robot.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, PointCloud2
from tf2_ros import Buffer, TransformListener

CLOUD, IMAGE, ODOM = '/livox/points', '/camera/color/image_raw/compressed', '/leg_odom2'


class Preflight(Node):
    def __init__(self):
        super().__init__('lite3_preflight')
        self.listen_s = self.declare_parameter('listen_s', 5.0).value
        self.base = self.declare_parameter('base_frame', 'base_link').value
        self.odom = self.declare_parameter('odom_frame', 'odom').value
        self.seen = {}
        self.tf = Buffer()
        TransformListener(self.tf, self)
        for topic, typ in ((CLOUD, PointCloud2), (IMAGE, CompressedImage), (ODOM, Odometry)):
            self.create_subscription(typ, topic, lambda m, t=topic: self._on(t, m), qos_profile_sensor_data)

    def _on(self, topic, msg):
        info = self.seen.setdefault(topic, {'count': 0, 'msg': msg})
        info['count'] += 1
        info['msg'] = msg

    def report(self):
        ok = True
        lines = []

        def check(cond, good, bad):
            nonlocal ok
            ok &= bool(cond)
            lines.append(('  OK   ' if cond else '  FIX  ') + (good if cond else bad))

        types = dict(self.get_topic_names_and_types())
        lines.append('Topic types:')
        for topic in (CLOUD, '/livox/lidar', IMAGE, ODOM, '/cmd_vel'):
            lines.append(f'  {topic:40s} {", ".join(types.get(topic, ["(not present)"]))}')

        cloud = self.seen.get(CLOUD)
        cloud_frame = cloud['msg'].header.frame_id if cloud else None
        check(cloud, f'{CLOUD}: {cloud["count"] / self.listen_s:.1f} Hz, frame_id="{cloud_frame}"' if cloud else '',
              f'No PointCloud2 on {CLOUD}. If /livox/lidar is sensor_msgs/PointCloud2, use cloud_topic:=/livox/lidar; '
              'if it is livox_ros_driver2/CustomMsg, set xfer_format: 0 in the Livox driver config.')
        img = self.seen.get(IMAGE)
        check(img, f'{IMAGE}: {img["count"] / self.listen_s:.1f} Hz, format="{img["msg"].format}"' if img else '',
              f'No images on {IMAGE}. Use image_topic:=/camera/color/image_raw image_compressed:=false.')
        odom = self.seen.get(ODOM)
        if odom:
            m = odom['msg']
            lines.append(f'  info {ODOM}: frame_id="{m.header.frame_id}" child_frame_id="{m.child_frame_id}"')
        check(odom, f'{ODOM} is publishing', f'No messages on {ODOM}. Is the Lite3 transfer node running?')

        now_s = self.get_clock().now().nanoseconds * 1e-9
        for topic in (CLOUD, IMAGE, ODOM):
            info = self.seen.get(topic)
            if not info:
                continue
            st = info['msg'].header.stamp
            offset = now_s - (st.sec + st.nanosec * 1e-9)
            if abs(offset) < 0.5:
                lines.append(f'  OK   {topic} timestamps match this clock ({offset:+.3f} s)')
            elif topic == ODOM:  # odom_tf_bridge re-stamps it, so not a blocker
                lines.append(f'  info {topic} timestamps are {offset:+.1f} s off (motion host clock); '
                             'odom_tf_bridge re-stamps them (stamp_mode: auto).')
            else:
                check(False, '', f'{topic} timestamps are {offset:+.1f} s off this computer\'s clock. '
                      'Sync clocks (chrony/PTP) or slam_toolbox / the costmaps will drop the data.')

        has_odom_tf = self.tf.can_transform(self.odom, self.base, Time())
        lines.append(f'  info TF {self.odom}->{self.base} already published: {has_odom_tf} '
                     f'({"odom_tf_bridge will stay quiet" if has_odom_tf else "odom_tf_bridge will publish it"})')
        if cloud_frame:
            has_lidar_tf = self.tf.can_transform(self.base, cloud_frame, Time())
            check(has_lidar_tf, f'TF {self.base}->{cloud_frame} exists',
                  f'No TF {self.base}->{cloud_frame}. Measure the MID-360 mount and launch with '
                  f'publish_lidar_tf:=true lidar_frame:={cloud_frame} lidar_x:=.. lidar_y:=.. lidar_z:=.. '
                  '(and lidar_pitch:=.. if the lidar is tilted).')
        frames = self.tf.all_frames_as_string().strip()
        cmd_subs = self.count_subscribers('/cmd_vel')
        check(cmd_subs > 0, f'/cmd_vel has {cmd_subs} subscriber(s) (motion bridge running)',
              '/cmd_vel has no subscriber: nothing forwards velocity commands, so the dog will not move.')
        # Who publishes the robot state, and which velocity topics does anything listen to?
        for info in self.get_publishers_info_by_topic(ODOM):
            node = f'{info.node_namespace.rstrip("/")}/{info.node_name}'
            subs = self.get_subscriber_names_and_types_by_node(info.node_name, info.node_namespace)
            lines.append(f'  info {ODOM} comes from node {node}; it subscribes to: '
                         + (', '.join(f'{t} [{"/".join(ty)}]' for t, ty in subs) or '(nothing)'))
        twist_types = ('geometry_msgs/msg/Twist', 'geometry_msgs/msg/TwistStamped')
        listeners = [(t, self.count_subscribers(t)) for t, ty in types.items()
                     if any(x in twist_types for x in ty)]
        listeners = [f'{t} ({n} sub)' for t, n in listeners if n > 0]
        lines.append('  info velocity topics with a subscriber: ' + (', '.join(listeners) or 'none'))

        print('\n'.join(['', '=== Lite3 kitchen explorer preflight ==='] + lines))
        print('\nTF frames seen:\n' + (frames or '  (none)'))
        print('\nResult:', 'READY' if ok else 'fix the FIX items above first')
        return ok


def main(args=None):
    rclpy.init(args=args)
    node = Preflight()
    end = time.monotonic() + node.listen_s
    while rclpy.ok() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    ok = node.report()
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
