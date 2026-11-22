"""lite3_sim node: a simulated Lite3 in a maze, standing in for the robot, the lidar, slam_toolbox and Nav2.

Run it with lite3_explorer's UNMODIFIED mission_manager (launch/sim.launch.py does both, plus RViz). The mission
manager cannot tell the difference: it gets the same topics, TF, actions and services as on the robot.

Provides (what the mission manager uses)
  TF        map -> odom -> base_link (map = the robot's start pose, like slam_toolbox), base_link -> sim_lidar
  pub /map                               nav_msgs/OccupancyGrid   SLAM map from the simulated scans (latched)
  pub /global_costmap/costmap            nav_msgs/OccupancyGrid   static map + /virtual_walls + inflation (latched)
  pub /local_costmap/costmap             nav_msgs/OccupancyGrid   4 m rolling window, odom frame (latched)
  pub /curtain/points                    std_msgs/String          confirmed curtain spots, curtain_detector format
  pub /stairs/points                     std_msgs/String          confirmed stair cells, stair_detector format
  sub /simple_cmd                        transfer_interfaces/MotionSimpleCMD   AI mode on / off (stairs mode); without
                                         transfer_interfaces installed: /ai_mode (std_msgs/Bool) instead
  act spin, compute_path_to_pose, navigate_to_pose, follow_waypoints          (nav2_msgs)
  srv /behavior_server/get_state                                   lifecycle_msgs/GetState (always active)
  srv /global_costmap/clear_entirely_global_costmap, /local_costmap/clear_entirely_local_costmap
  sub /cmd_vel (curtain / mppi modes), /virtual_walls (mission manager -> global costmap)

For RViz / debugging
  pub /scan (sensor_msgs/LaserScan), /odom, /plan (the path being followed)
  pub /sim/maze        MarkerArray (latched): ground-truth walls (grey), slit panels (blue), curtains (cyan), scene
                       tiles (orange), tile
                       labels, the maze's own tile grid (thin white), the start pose
  pub /sim/robot       MarkerArray: body footprint, heading, mission mode, depth-camera view, curtain spots
  pub /sim/status      std_msgs/String JSON (1 Hz): sim time, pose, maze tile, Nav2 task, ground-truth coverage
  srv ~/reset_curtains std_srvs/Trigger   forget the detected curtain spots

Staircases ('M'): the risers block the lidar, and the robot unless it is in AI mode; the stair camera runs
lite3_explorer's stairs_core on simulated tread / floor points. The climb itself is not simulated (no pitch, no
height). Not simulated: terrain, gaits, scene / pose animations, the AI. Parameters: config/sim.yaml.
"""
from __future__ import annotations

import array
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Twist
from lifecycle_msgs.msg import State as LifecycleState
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose, FollowWaypoints, NavigateToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from lite3_sim import world as W
from lite3_sim.maze import CURTAIN, SLIT, STAIRS, load_maze

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def unflatten(flat: dict) -> dict:
    out = {}
    for key, value in flat.items():
        node = out
        parts = key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat(yaw):
    from geometry_msgs.msg import Quaternion
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


class Lite3Sim(Node):
    def __init__(self):
        super().__init__('lite3_sim', allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        flat = {k: p.value for k, p in self.get_parameters_by_prefix('').items() if k != 'use_sim_time'}
        cfg = unflatten(flat)
        maze_path = cfg.pop('maze', '')
        if not maze_path:
            raise RuntimeError('parameter "maze" (path to a maze YAML) is required')
        self.maze = load_maze(maze_path)
        self.world = W.World(self.maze, cfg, logger=lambda m: self.get_logger().info(f'[world] {m}'))
        self.cfg = self.world.cfg
        self.rate_hz = float(cfg.get('physics_hz', 50.0))
        self.lock = threading.RLock()
        self.get_logger().info(self.maze.summary())
        self.get_logger().info(
            f'map frame = {"the robot start pose (like SLAM)" if self.cfg["map_origin"] == "start" else "the maze corner"}'
            f'; controller {self.cfg["controller"]["type"]}; lidar range {self.cfg["lidar"]["range_max"]} m; '
            f'speed x{self.cfg["speed_scale"]}')

        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.action_group = ReentrantCallbackGroup()

        # publishers
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', LATCHED)
        self.gcm_pub = self.create_publisher(OccupancyGrid, '/global_costmap/costmap', LATCHED)
        self.lcm_pub = self.create_publisher(OccupancyGrid, '/local_costmap/costmap', LATCHED)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.plan_pub = self.create_publisher(Path, '/plan', 10)
        self.curtain_pub = self.create_publisher(String, '/curtain/points', LATCHED)
        self.stairs_pub = self.create_publisher(String, '/stairs/points', LATCHED)
        self.maze_pub = self.create_publisher(MarkerArray, '/sim/maze', LATCHED)
        self.robot_pub = self.create_publisher(MarkerArray, '/sim/robot', 10)
        self.status_pub = self.create_publisher(String, '/sim/status', 10)

        # inputs
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd_vel, 10, callback_group=self.timer_group)
        self.create_subscription(PointCloud2, '/virtual_walls', self._on_virtual_walls, 10,
                                 callback_group=self.timer_group)
        self.mode_text = ''
        self.create_subscription(String, '/mission_manager/mode', self._on_mode, LATCHED,
                                 callback_group=self.timer_group)
        self.create_subscription(Bool, '/exploration_complete', self._on_complete, LATCHED,
                                 callback_group=self.timer_group)
        # robot AI mode (stairs mode): the real /simple_cmd codes; /ai_mode when transfer_interfaces is not installed
        try:
            from transfer_interfaces.msg import MotionSimpleCMD
            self.create_subscription(MotionSimpleCMD, '/simple_cmd', self._on_simple_cmd, QoSProfile(
                depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE),
                callback_group=self.timer_group)
        except ImportError:
            self.get_logger().warning('transfer_interfaces not installed: AI mode follows /ai_mode instead of '
                                      '/simple_cmd')
            self.create_subscription(Bool, '/ai_mode', self._on_ai_mode, LATCHED, callback_group=self.timer_group)

        # Nav2 stand-ins
        accept = lambda _g: GoalResponse.ACCEPT       # noqa: E731
        cancel = lambda _g: CancelResponse.ACCEPT     # noqa: E731
        kw = dict(goal_callback=accept, cancel_callback=cancel, callback_group=self.action_group)
        self._servers = [
            ActionServer(self, Spin, 'spin', self._exec_spin, **kw),
            ActionServer(self, NavigateToPose, 'navigate_to_pose', self._exec_nav, **kw),
            ActionServer(self, FollowWaypoints, 'follow_waypoints', self._exec_waypoints, **kw),
            ActionServer(self, ComputePathToPose, 'compute_path_to_pose', self._exec_plan, **kw),
        ]
        self.create_service(GetState, '/behavior_server/get_state', self._on_get_state,
                            callback_group=self.action_group)
        for name in ('/global_costmap/clear_entirely_global_costmap', '/local_costmap/clear_entirely_local_costmap'):
            self.create_service(ClearEntireCostmap, name, self._on_clear, callback_group=self.action_group)
        self.create_service(Trigger, '~/reset_curtains', self._on_reset_curtains, callback_group=self.action_group)

        # state for publishing
        self._last = {'scan': None, 'map': -1, 'global': -1, 'local': None, 'curtain': -1, 'stairs': -1}
        self._task_path_id = None
        self.truth_visited = set()
        self.reachable = self.maze.reachable_tiles()
        self.t_start = time.monotonic()
        self.complete_at = None

        self._publish_static()
        self._publish_maze()
        self._last_step = time.monotonic()
        self.create_timer(1.0 / self.rate_hz, self._step, callback_group=self.timer_group)
        self.create_timer(0.1, self._publish_robot, callback_group=self.timer_group)
        self.create_timer(1.0, self._publish_status, callback_group=self.timer_group)
        self.create_timer(2.0, self._publish_maze, callback_group=self.timer_group)   # late RViz subscribers

    # =========================================================================================== loop
    def _step(self):
        now = time.monotonic()
        dt = min(0.1, now - self._last_step)
        self._last_step = now
        with self.lock:
            self.world.step(dt)
            self._truth_track()
            pose = list(self.world.pose)
            vel = list(self.world.vel)
            scan = self.world.last_scan
            map_v, glob_v, loc = self.world.map_version, self.world.global_version, self.world.local_costmap
            curt_v = self.world.curtain_version
            stair_v = self.world.stair_version
            task = self.world.task
            path = getattr(task, 'path', None) if task is not None else None
        stamp = self.get_clock().now().to_msg()
        self._publish_tf(pose, vel, stamp)
        if scan is not None and scan is not self._last['scan']:
            self._last['scan'] = scan
            self._publish_scan(scan, stamp)
        if map_v != self._last['map']:
            self._last['map'] = map_v
            with self.lock:
                grid = self.world.map_grid
            self.map_pub.publish(self._grid_msg(grid, self.world.info, 'map', stamp))
        if glob_v != self._last['global']:
            self._last['global'] = glob_v
            with self.lock:
                grid = self.world.global_costmap
            self.gcm_pub.publish(self._grid_msg(grid, self.world.info, 'map', stamp))
        if loc is not None and loc is not self._last['local']:
            self._last['local'] = loc
            self.lcm_pub.publish(self._grid_msg(loc[0], loc[1], 'odom', stamp))
        if curt_v != self._last['curtain']:
            self._last['curtain'] = curt_v
            with self.lock:
                pts = self.world.curtain_points()
            self.curtain_pub.publish(String(data=json.dumps(
                {'points': [[round(float(v), 3) for v in r] for r in pts]})))
        if stair_v != self._last['stairs']:
            self._last['stairs'] = stair_v
            with self.lock:
                pts = self.world.stair_points()
            self.stairs_pub.publish(String(data=json.dumps(
                {'points': [[round(float(v), 3) for v in r] for r in pts]})))
        if path is not None and id(path) != self._task_path_id:
            self._task_path_id = id(path)
            self.plan_pub.publish(self._path_msg(path, stamp))

    def _truth_track(self):
        mx, my, _ = W.compose(W.invert(self.world.map_T_maze), tuple(self.world.pose))
        s = self.maze.tile_size
        i, j = int(mx // s), int(my // s)
        if math.hypot(mx - (i + 0.5) * s, my - (j + 0.5) * s) <= 0.45 and (i, j) not in self.truth_visited:
            self.truth_visited.add((i, j))

    # =========================================================================================== inputs
    def _on_cmd_vel(self, msg):
        with self.lock:
            self.world.set_cmd_vel(msg.linear.x, msg.linear.y, msg.angular.z)

    def _on_simple_cmd(self, msg):
        with self.lock:
            self.world.simple_cmd(msg.cmd_code)

    def _on_ai_mode(self, msg):
        with self.lock:
            self.world.simple_cmd(W.AI_ON if msg.data else W.AI_OFF)

    def _on_virtual_walls(self, msg):
        pts = np.array([(p[0], p[1]) for p in point_cloud2.read_points(msg, field_names=('x', 'y'), skip_nans=True)],
                       float).reshape(-1, 2)
        with self.lock:
            self.world.set_virtual_walls(pts)

    def _on_mode(self, msg):
        self.mode_text = msg.data

    def _on_complete(self, msg):
        if not msg.data or self.complete_at is not None:
            return
        self.complete_at = time.monotonic()
        missed = sorted(self.reachable - self.truth_visited)
        self.get_logger().info(
            f'EXPLORATION COMPLETE after {self.complete_at - self.t_start:.0f} s: reached '
            f'{len(self.truth_visited & self.reachable)} of the {len(self.reachable)} reachable maze tiles'
            + (f'; never reached (maze i,j): {missed}' if missed else ' (all of them)'))

    def _on_get_state(self, request, response):
        response.current_state.id = LifecycleState.PRIMARY_STATE_ACTIVE
        response.current_state.label = 'active'
        return response

    def _on_clear(self, request, response):
        with self.lock:
            self.world.clear_costmaps()
        return response

    def _on_reset_curtains(self, request, response):
        with self.lock:
            if self.world.curtains is not None:
                self.world.curtains._hits.clear()
                self.world.curtain_version += 1
        response.success, response.message = True, 'curtain spots forgotten'
        return response

    # =========================================================================================== actions
    def _run(self, goal_handle, task, result, feedback_fn=None):
        """Wait for a world task, mirroring its end state onto the goal handle."""
        last_fb = 0.0
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                with self.lock:
                    self.world.cancel(task)
                goal_handle.canceled()
                return result
            with self.lock:
                state, reason = task.state, task.reason
            if state == W.SUCCEEDED:
                goal_handle.succeed()
                return result
            if state in (W.FAILED, W.CANCELLED):
                if reason:
                    self.get_logger().info(f'{task.kind} goal aborted: {reason}')
                goal_handle.abort()
                return result
            if feedback_fn is not None and time.monotonic() - last_fb > 0.2:
                last_fb = time.monotonic()
                fb = feedback_fn(task)
                if fb is not None:
                    goal_handle.publish_feedback(fb)
            time.sleep(0.02)
        goal_handle.abort()
        return result

    def _exec_spin(self, gh):
        g = gh.request
        allowance = g.time_allowance.sec + g.time_allowance.nanosec * 1e-9 or 20.0
        with self.lock:
            task = self.world.spin(g.target_yaw, allowance)
        res = Spin.Result()
        out = self._run(gh, task, res, lambda t: self._fb(Spin.Feedback(), angular_distance_traveled=float(
            t.feedback.get('angular_distance_traveled', 0.0))))
        try:
            res.total_elapsed_time = DurationMsg(sec=int(self.world.time - task.started))
        except (AttributeError, TypeError):
            pass
        return out

    def _exec_nav(self, gh):
        p = gh.request.pose.pose
        with self.lock:
            task = self.world.navigate((p.position.x, p.position.y, yaw_of(p.orientation)))
        return self._run(gh, task, NavigateToPose.Result(), self._nav_feedback)

    def _exec_waypoints(self, gh):
        goals = [(ps.pose.position.x, ps.pose.position.y, yaw_of(ps.pose.orientation)) for ps in gh.request.poses]
        with self.lock:
            task = self.world.follow_waypoints(goals)
        res = FollowWaypoints.Result()

        def fb(t):
            return self._fb(FollowWaypoints.Feedback(), current_waypoint=int(t.feedback.get('current_waypoint', 0)))
        out = self._run(gh, task, res, fb)
        self._set_missed(res, task.missed)
        return out

    @staticmethod
    def _set_missed(res, missed):
        try:                                              # Humble: int32[]
            res.missed_waypoints = [int(i) for i in missed]
        except (AssertionError, TypeError, ValueError):   # Jazzy+: nav2_msgs/WaypointStatus[]
            from nav2_msgs.msg import WaypointStatus
            out = []
            for i in missed:
                w = WaypointStatus()
                w.waypoint_index = int(i)
                w.waypoint_status = WaypointStatus.FAILED
                out.append(w)
            res.missed_waypoints = out

    def _exec_plan(self, gh):
        g = gh.request
        t0 = time.monotonic()
        with self.lock:
            start = (g.start.pose.position.x, g.start.pose.position.y) if g.use_start else tuple(self.world.pose[:2])
            path = self.world.plan(start, (g.goal.pose.position.x, g.goal.pose.position.y))
        res = ComputePathToPose.Result()
        if path is None:
            gh.abort()
            return res
        res.path = self._path_msg(path, self.get_clock().now().to_msg(), goal_yaw=yaw_of(g.goal.pose.orientation))
        dt = time.monotonic() - t0
        res.planning_time = DurationMsg(sec=int(dt), nanosec=int((dt % 1.0) * 1e9))
        gh.succeed()
        return res

    def _nav_feedback(self, task):
        fb = NavigateToPose.Feedback()
        try:
            fb.distance_remaining = float(task.feedback.get('distance_remaining', 0.0))
        except (AttributeError, TypeError):
            pass
        return fb

    @staticmethod
    def _fb(msg, **fields):
        for k, v in fields.items():
            try:
                setattr(msg, k, v)
            except (AttributeError, TypeError, AssertionError):
                pass
        return msg

    # =========================================================================================== messages
    def _grid_msg(self, grid, info, frame, stamp):
        msg = OccupancyGrid()
        msg.header.frame_id, msg.header.stamp = frame, stamp
        msg.info.resolution = float(info.resolution)
        msg.info.width, msg.info.height = int(info.width), int(info.height)
        msg.info.origin.position.x, msg.info.origin.position.y = float(info.origin_x), float(info.origin_y)
        msg.info.origin.orientation.w = 1.0
        msg.data = array.array('b', np.clip(grid, -1, 100).astype(np.int8).tobytes())
        return msg

    def _path_msg(self, pts, stamp, goal_yaw=None):
        msg = Path()
        msg.header.frame_id, msg.header.stamp = 'map', stamp
        for k, (x, y) in enumerate(pts):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            if k + 1 < len(pts):
                yaw = math.atan2(pts[k + 1][1] - y, pts[k + 1][0] - x)
            else:
                yaw = goal_yaw if goal_yaw is not None else (
                    math.atan2(y - pts[k - 1][1], x - pts[k - 1][0]) if k else 0.0)
            ps.pose.orientation = quat(yaw)
            msg.poses.append(ps)
        return msg

    def _publish_static(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = 'base_link', 'sim_lidar'
        t.transform.translation.x = float(self.cfg['lidar']['x_offset_m'])
        t.transform.rotation.w = 1.0
        self.static_tf.sendTransform(t)

    def _publish_tf(self, pose, vel, stamp):
        a = TransformStamped()
        a.header.stamp, a.header.frame_id, a.child_frame_id = stamp, 'map', 'odom'
        a.transform.rotation.w = 1.0
        b = TransformStamped()
        b.header.stamp, b.header.frame_id, b.child_frame_id = stamp, 'odom', 'base_link'
        b.transform.translation.x, b.transform.translation.y = float(pose[0]), float(pose[1])
        b.transform.rotation = quat(pose[2])
        self.tf.sendTransform([a, b])
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, 'odom', 'base_link'
        od.pose.pose.position.x, od.pose.pose.position.y = float(pose[0]), float(pose[1])
        od.pose.pose.orientation = quat(pose[2])
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.angular.z = (float(v) for v in vel)
        self.odom_pub.publish(od)

    def _publish_scan(self, scan, stamp):
        rel, rng, _ = scan
        lc = self.cfg['lidar']
        msg = LaserScan()
        msg.header.frame_id, msg.header.stamp = 'sim_lidar', stamp
        msg.angle_min, msg.angle_increment = float(rel[0]), float(lc['angle_increment'])
        msg.angle_max = float(rel[0] + lc['angle_increment'] * (len(rel) - 1))
        msg.scan_time = 1.0 / float(lc['rate_hz'])
        msg.range_min, msg.range_max = float(lc['range_min']), float(lc['range_max'])
        msg.ranges = array.array('f', np.asarray(rng, np.float32).tobytes())
        self.scan_pub.publish(msg)

    # =========================================================================================== RViz markers
    def _marker(self, ns, mid, mtype, stamp, rgba, frame='map'):
        m = Marker()
        m.header.frame_id, m.header.stamp = frame, stamp
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in rgba)
        m.pose.orientation.w = 1.0
        return m

    def _publish_maze(self):
        stamp = self.get_clock().now().to_msg()
        g = self.world.geometry
        markers = []
        colours = {'wall': (0.55, 0.55, 0.55, 0.9), SLIT: (0.45, 0.55, 0.85, 0.9), CURTAIN: (0.0, 0.85, 0.95, 0.35),
                   STAIRS: (1.0, 0.8, 0.0, 0.9)}
        for k, ((x0, y0, x1, y1), kind) in enumerate(zip(g.segs, g.kinds)):
            m = self._marker('maze_' + kind, k, Marker.CUBE, stamp, colours.get(kind, colours['wall']))
            length = math.hypot(x1 - x0, y1 - y0)
            low, high = (0.15, 0.55) if kind == CURTAIN else (0.0, self.maze.stair_rise_m) if kind == STAIRS \
                else (0.0, 0.5)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (x0 + x1) / 2, (y0 + y1) / 2, (low + high) / 2
            m.pose.orientation = quat(math.atan2(y1 - y0, x1 - x0))
            m.scale.x, m.scale.y, m.scale.z = max(length, 0.01), max(2 * g.half_thickness, 0.02), high - low
            markers.append(m)
        # the maze's tile grid (ground truth), thin lines at floor level
        s, mz = self.maze.tile_size, self.maze
        grid = self._marker('maze_tile_grid', 0, Marker.LINE_LIST, stamp, (1.0, 1.0, 1.0, 0.25))
        grid.scale.x = 0.01
        for i in range(mz.cols + 1):
            for a, b in (((i * s, 0.0), (i * s, mz.rows * s)),):
                for x, y in (a, b):
                    px, py, _ = self.world.maze_to_map(x, y)
                    grid.points.append(Point(x=px, y=py, z=0.005))
        for j in range(mz.rows + 1):
            for x, y in ((0.0, j * s), (mz.cols * s, j * s)):
                px, py, _ = self.world.maze_to_map(x, y)
                grid.points.append(Point(x=px, y=py, z=0.005))
        markers.append(grid)
        for n, (tile, text) in enumerate(sorted(mz.labels.items())):
            px, py, _ = self.world.maze_to_map(*mz.tile_centre(tile))
            t = self._marker('maze_labels', n, Marker.TEXT_VIEW_FACING, stamp, (1.0, 1.0, 0.6, 0.9))
            t.pose.position.x, t.pose.position.y, t.pose.position.z = px, py - 0.3, 0.05
            t.scale.z = 0.18
            t.text = text
            markers.append(t)
        for n, tile in enumerate(sorted(mz.solid)):
            px, py, _ = self.world.maze_to_map(*mz.tile_centre(tile))
            b = self._marker('maze_solid', n, Marker.CUBE, stamp, (0.3, 0.3, 0.3, 0.6))
            b.pose.position.x, b.pose.position.y, b.pose.position.z = px, py, 0.25
            b.pose.orientation = quat(self.world.map_T_maze[2])
            b.scale.x = b.scale.y = s * 0.98
            b.scale.z = 0.5
            markers.append(b)
        margin = float(self.world.cfg['scene']['margin_m'])
        for n, tile in enumerate(sorted(mz.scenes)):        # scene tiles: the cluttered square, low and orange
            px, py, _ = self.world.maze_to_map(*mz.tile_centre(tile))
            b = self._marker('maze_scene', n, Marker.CUBE, stamp, (1.0, 0.55, 0.1, 0.45))
            b.pose.position.x, b.pose.position.y, b.pose.position.z = px, py, 0.15
            b.pose.orientation = quat(self.world.map_T_maze[2])
            b.scale.x = b.scale.y = s - 2 * margin
            b.scale.z = 0.3
            markers.append(b)
        n = 0                                              # staircases: every tread / the landing as a gold slab
        for tile in sorted(mz.stairs):
            k = 6
            for a in range(k):
                for b in range(k):
                    u, v = tile[0] * s + (a + 0.5) * s / k, tile[1] * s + (b + 0.5) * s / k
                    h = mz.stair_height(u, v)
                    px, py, _ = self.world.maze_to_map(u, v)
                    c = self._marker('maze_stairs', n, Marker.CUBE, stamp, (1.0, 0.8, 0.0, 0.35 + 0.5 * h))
                    c.pose.position.x, c.pose.position.y, c.pose.position.z = px, py, h / 2
                    c.pose.orientation = quat(self.world.map_T_maze[2])
                    c.scale.x = c.scale.y = s / k
                    c.scale.z = max(h, 0.01)
                    markers.append(c)
                    n += 1
        sx, sy, syaw = self.world.maze_to_map(*mz.start)
        a = self._marker('maze_start', 0, Marker.ARROW, stamp, (0.2, 1.0, 0.2, 0.6))
        a.pose.position.x, a.pose.position.y = sx, sy
        a.pose.orientation = quat(syaw)
        a.scale.x, a.scale.y, a.scale.z = 0.4, 0.06, 0.06
        markers.append(a)
        self.maze_pub.publish(MarkerArray(markers=markers))

    def _publish_robot(self):
        stamp = self.get_clock().now().to_msg()
        with self.lock:
            x, y, yaw = self.world.pose
            bumped = self.world.bumped
            seen = self.world.curtain_seen_now.copy()
            confirmed = self.world.curtain_points()
            stairs_now = self.world.stair_seen_now.copy()
            stairs_ok = self.world.stair_points()
            ai = self.world.ai_mode
            task = self.world.task
        markers = []
        body = self._marker('robot', 0, Marker.CUBE, stamp, (1.0, 0.2, 0.2, 0.8) if bumped else (1.0, 0.55, 0.1, 0.6))
        body.pose.position.x, body.pose.position.y, body.pose.position.z = x, y, 0.15
        body.pose.orientation = quat(yaw)
        body.scale.x, body.scale.y, body.scale.z = self.world.length, self.world.width, 0.25
        markers.append(body)
        head = self._marker('robot', 1, Marker.ARROW, stamp, (1.0, 1.0, 1.0, 0.9))
        head.pose.position.x, head.pose.position.y, head.pose.position.z = x, y, 0.3
        head.pose.orientation = quat(yaw)
        head.scale.x, head.scale.y, head.scale.z = 0.45, 0.05, 0.05
        markers.append(head)
        txt = self._marker('robot', 2, Marker.TEXT_VIEW_FACING, stamp, (1.0, 1.0, 1.0, 1.0))
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = x, y, 0.8
        txt.scale.z = 0.2
        txt.text = (self.mode_text or '(no mission_manager)') + (f'\nnav2: {task.kind}' if task is not None else '') \
            + ('\nAI MODE' if ai else '') + ('\nBUMPED' if bumped else '')
        markers.append(txt)
        cc = self.cfg['curtain_camera']
        if cc['enabled']:
            fov = self._marker('robot', 3, Marker.LINE_STRIP, stamp, (0.0, 0.85, 0.95, 0.5))
            fov.scale.x = 0.01
            cx, cy = x + math.cos(yaw) * cc['x_offset_m'], y + math.sin(yaw) * cc['x_offset_m']
            half = math.radians(cc['hfov_deg']) / 2
            r = cc['range_max']
            for p in ((cx, cy), (cx + r * math.cos(yaw + half), cy + r * math.sin(yaw + half)),
                      (cx + r * math.cos(yaw - half), cy + r * math.sin(yaw - half)), (cx, cy)):
                fov.points.append(Point(x=p[0], y=p[1], z=0.3))
            markers.append(fov)
        for mid, pts, rgba, size in ((4, seen, (0.0, 1.0, 1.0, 1.0), 0.04), (5, confirmed[:, :2] if len(confirmed)
                                                                             else confirmed, (1.0, 0.0, 1.0, 1.0), 0.06)):
            m = self._marker('curtain_spots', mid, Marker.POINTS, stamp, rgba)
            m.scale.x = m.scale.y = size
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.35) for p in pts]
            if not m.points:
                m.action = Marker.DELETE
            markers.append(m)
        for mid, pts, rgba, size in ((6, stairs_now, (1.0, 0.9, 0.3, 1.0), 0.04),
                                     (7, stairs_ok[:, :2] if len(stairs_ok) else stairs_ok, (1.0, 0.5, 0.0, 1.0), 0.07)):
            m = self._marker('stair_cells', mid, Marker.POINTS, stamp, rgba)
            m.scale.x = m.scale.y = size
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.4) for p in pts]
            if not m.points:
                m.action = Marker.DELETE
            markers.append(m)
        self.robot_pub.publish(MarkerArray(markers=markers))

    def _publish_status(self):
        with self.lock:
            w = self.world
            task = w.task
            st = {
                'sim_time_s': round(w.time, 1),
                'pose_map': [round(v, 3) for v in w.pose],
                'maze_tile': list(w.tile_of_maze()),
                'nav2_task': None if task is None else {'kind': task.kind, 'state': task.state,
                                                        **{k: v for k, v in task.feedback.items()
                                                           if isinstance(v, (int, float))}},
                'bumped': w.bumped,
                'in_inscribed_zone': w.robot_in_inscribed(),
                'curtain_spots': int(len(w.curtain_points())),
                'stair_cells': int(len(w.stair_points())),
                'ai_mode': w.ai_mode,
                'truth_reached': len(self.truth_visited & self.reachable),
                'truth_reachable': len(self.reachable),
                'truth_not_yet': [list(t) for t in sorted(self.reachable - self.truth_visited)],
                'mission_mode': self.mode_text,
            }
        self.status_pub.publish(String(data=json.dumps(st)))


def main(args=None):
    rclpy.init(args=args)
    node = Lite3Sim()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
