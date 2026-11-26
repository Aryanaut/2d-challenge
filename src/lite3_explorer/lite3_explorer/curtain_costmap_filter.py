"""Clear the detected curtain out of Nav2's inputs, so Nav2 plans (and drives) through it.

A hanging curtain is a wall to the 2D lidar: slam_toolbox maps it as occupied and the costmaps' obstacle layers mark
it from /scan, so Nav2 has no path to the tile behind it (bag explorer_26_11_223230: the curtain on the start tile's
W edge, 57 camera spots, and only 'no Nav2 path' for the rest of the run). The mission manager picks the ONE tile edge
the curtain lies on and publishes it on /curtain/wall; this node republishes

  /map   -> /map_nav    occupancy grid with the cells within clear_half_width_m of that edge set free
                        (the global costmap's static layer reads /map_nav)
  /scan  -> /scan_nav   laser scan with the rays that hit within scan_half_width_m of that edge dropped (range inf),
                        so the obstacle layers (global and local costmap) never mark the curtain

Without a curtain wall both pass through unchanged, so the node runs in every launch. slam_toolbox, the mission's tile
logic and the terrain nodes keep the raw /map and /scan (the curtain stays a wall in the tile map; only Nav2 ignores it).
end_margin_m at each end of the edge is not cleared: the wall posts the curtain hangs between stay obstacles.

  sub  /map            nav_msgs/OccupancyGrid  (transient local)
  sub  /scan           sensor_msgs/LaserScan
  sub  /curtain/wall   std_msgs/String JSON {"segment": [[x0, y0], [x1, y1]] | null}  (latched, mission_manager)
  pub  /map_nav        nav_msgs/OccupancyGrid  (transient local)
  pub  /scan_nav       sensor_msgs/LaserScan
"""
import json
import math

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from lite3_explorer import curtain_core as cc


def clear_map(msg: OccupancyGrid, seg, half_width_m: float, end_margin_m: float) -> int:
    """Set the cells of msg (in place) within half_width_m of seg free. Returns how many occupied cells were freed."""
    info = msg.info
    w, h, res = info.width, info.height, info.resolution
    if not w or not h or seg is None:
        return 0
    q = info.origin.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    # only the cells around the segment's bounding box (the map can be large)
    (x0, y0), (x1, y1) = seg
    pad = half_width_m + res
    c, s = math.cos(yaw), math.sin(yaw)
    ox, oy = info.origin.position.x, info.origin.position.y
    corners = [(x, y) for x in (min(x0, x1) - pad, max(x0, x1) + pad) for y in (min(y0, y1) - pad, max(y0, y1) + pad)]
    cols = [((x - ox) * c + (y - oy) * s) / res for x, y in corners]
    rows = [(-(x - ox) * s + (y - oy) * c) / res for x, y in corners]
    c0, c1 = max(0, int(math.floor(min(cols)))), min(w, int(math.ceil(max(cols))) + 1)
    r0, r1 = max(0, int(math.floor(min(rows)))), min(h, int(math.ceil(max(rows))) + 1)
    if c0 >= c1 or r0 >= r1:
        return 0
    rr, cc_ = np.mgrid[r0:r1, c0:c1]
    u, v = (cc_ + 0.5) * res, (rr + 0.5) * res
    xs, ys = ox + c * u - s * v, oy + s * u + c * v
    band = cc.segment_band_mask(xs, ys, seg, half_width_m, end_margin_m)
    data = np.asarray(msg.data, np.int8).reshape(h, w)
    win = data[r0:r1, c0:c1]
    freed = int(np.count_nonzero(band & (win > 0)))
    win[band] = 0
    msg.data = data.reshape(-1).tolist()
    return freed


def scan_hits(msg: LaserScan, tf_xy_yaw, seg, half_width_m: float, end_margin_m: float) -> np.ndarray:
    """Indices of the rays whose hit lies within half_width_m of seg. tf_xy_yaw: the scan frame's pose in the map
    (x, y, yaw); the scan is taken as planar."""
    r = np.asarray(msg.ranges, float)
    ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
    if seg is None or not ok.any():
        return np.zeros(0, int)
    a = msg.angle_min + np.arange(len(r)) * msg.angle_increment + tf_xy_yaw[2]
    rr = np.where(ok, r, 0.0)
    xs, ys = tf_xy_yaw[0] + rr * np.cos(a), tf_xy_yaw[1] + rr * np.sin(a)
    return np.nonzero(ok & cc.segment_band_mask(xs, ys, seg, half_width_m, end_margin_m))[0]


class CurtainCostmapFilter(Node):
    def __init__(self):
        super().__init__('curtain_costmap_filter')
        p = self.declare_parameter
        self.global_frame = p('global_frame', 'map').value
        self.map_half = float(p('clear_half_width_m', 0.15).value)
        self.scan_half = float(p('scan_half_width_m', 0.20).value)
        self.end_margin = float(p('end_margin_m', 0.08).value)
        self.seg = None
        self.map_msg = None
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.map_pub = self.create_publisher(OccupancyGrid, p('map_out', '/map_nav').value, latched)
        self.scan_pub = self.create_publisher(LaserScan, p('scan_out', '/scan_nav').value, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, p('map_in', '/map').value, self._on_map, latched)
        self.create_subscription(LaserScan, p('scan_in', '/scan').value, self._on_scan, qos_profile_sensor_data)
        self.create_subscription(String, '/curtain/wall', self._on_wall, latched)
        self._tf_warned = False
        self.get_logger().info('curtain costmap filter: /map -> /map_nav, /scan -> /scan_nav (curtain cleared once '
                               'the mission manager publishes /curtain/wall)')

    def _on_wall(self, msg):
        try:
            seg = json.loads(msg.data).get('segment')
            seg = None if not seg else ((float(seg[0][0]), float(seg[0][1])), (float(seg[1][0]), float(seg[1][1])))
        except (ValueError, TypeError, IndexError, KeyError, AttributeError):
            return
        if seg == self.seg:
            return
        self.seg = seg
        self.get_logger().info('no curtain: /map_nav and /scan_nav pass through' if seg is None else
                               f'curtain wall ({seg[0][0]:.2f}, {seg[0][1]:.2f}) -> ({seg[1][0]:.2f}, {seg[1][1]:.2f}): '
                               f'cleared from /map_nav and /scan_nav')
        if self.map_msg is not None:
            self._publish_map()

    def _on_map(self, msg):
        self.map_msg = msg
        self._publish_map()

    def _publish_map(self):
        src = self.map_msg
        if self.seg is None:
            self.map_pub.publish(src)
            return
        out = OccupancyGrid(header=src.header, info=src.info, data=src.data)
        clear_map(out, self.seg, self.map_half, self.end_margin)
        self.map_pub.publish(out)

    def _on_scan(self, msg):
        if self.seg is None:
            self.scan_pub.publish(msg)
            return
        try:
            t = self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time.from_msg(msg.header.stamp),
                                         Duration(seconds=0.05))
        except Exception:  # noqa: BLE001 - no pose at the scan's stamp: use the newest one
            try:
                t = self.tf.lookup_transform(self.global_frame, msg.header.frame_id, Time())
            except Exception as e:  # noqa: BLE001 - TF not ready: pass the scan through unchanged
                if not self._tf_warned:
                    self._tf_warned = True
                    self.get_logger().warn(f'no TF {self.global_frame} <- {msg.header.frame_id}: scan passed through ({e})')
                self.scan_pub.publish(msg)
                return
        q, tr = t.transform.rotation, t.transform.translation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        idx = scan_hits(msg, (tr.x, tr.y, yaw), self.seg, self.scan_half, self.end_margin)
        if len(idx):
            ranges = list(msg.ranges)
            for i in idx:
                ranges[int(i)] = float('inf')                      # no return: not marked by the obstacle layers
            msg.ranges = ranges
        self.scan_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CurtainCostmapFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
