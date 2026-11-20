"""Mission manager node: tile exploration of the arena with switchable mission modes.

The SLAM map is cut into 1 m tiles; the robot visits every identified walkable tile and the mission ends when the
configured end condition is met (default: all identified tiles explored). WHERE to go next comes from exploration
profiles (tiles by default, frontiers optionally), HOW to drive from traversal profiles.

Modes (mission/modes.py, configured in config/mission.yaml):
  idle -> scanning (tile grid + turn in place) -> exploring (next tile sharing a side with an explored tile,
          Nav2-reachable; optionally a frontier) -> scanning -> ...
  recovery: when stuck (no motion while driving, or sitting in the global costmap's obstacle zone) while unexplored
            tiles remain, or when no unexplored tile has a Nav2 path: drive to the nearest tile centre, then rescan.
  complete: the end condition is met; /exploration_complete is published.

When a NEW tile is reached, the AI client (lite3_llm_tools llm_client) is asked in the background what it looks like;
the answers make up the tile report (reports_dir/tile_report_<run>.md).

Interfaces
  sub  /map, /global_costmap/costmap, /local_costmap/costmap      nav_msgs/OccupancyGrid
  sub  ~/mode_cmd               std_msgs/String   switch mode by name (e.g. "idle", "scanning", "exploring", "recovery")
  srv  ~/set_active             std_srvs/SetBool  true: start / resume (idle or complete -> start_mode); false: idle
  srv  ~/report_now             std_srvs/Trigger  describe the tile the robot is in right now
  pub  ~/mode                   std_msgs/String   "<mode>:<phase>" (latched)
  pub  ~/status                 std_msgs/String   JSON: mode, phase, target, tile counts, end-condition state (1 Hz)
  pub  ~/events                 std_msgs/String   JSON: every decision (also run_dir/events.jsonl)
  pub  ~/current_goal           geometry_msgs/PoseStamped (latched)   goal sent to Nav2
  pub  ~/next_target            std_msgs/String   (latched) the tile being driven to
  pub  /exploration_complete    std_msgs/Bool     (latched)
  pub  /tile_map                std_msgs/String   JSON: grid, trail, every tile's status and sides (latched)
  pub  /tile_report             std_msgs/String   JSON: one tile report entry (latched, depth 100)
  pub  ~/tile_markers, ~/markers  visualization_msgs/MarkerArray   tiles, trail, target / tile descriptions
  pub  /spin_crawl, /spin_flat  std_msgs/Bool     gait holds for gait_switcher while turning
  act  spin, compute_path_to_pose, navigate_to_pose, follow_waypoints   (Nav2 clients)
  srv  /global_costmap/clear_entirely_global_costmap, /local_costmap/clear_entirely_local_costmap  (recovery)
"""
import datetime as dt
import json
import math
import os
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import State as LifecycleState
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explorer import map_grid as G
from lite3_explorer import tiles as T
from lite3_explorer import turn_check as TC
from lite3_explorer import nav2_traversal  # noqa: F401  (registers nav2_pose / nav2_route)
from lite3_explorer.llm_link import LlmLink
from lite3_explorer.mission.config import DEFAULTS, merge_flat
from lite3_explorer.mission.core import ModeMachine
from lite3_explorer.mission.traversal import FAILED, SUCCEEDED, build_traversal
from lite3_explorer.tile_reports import TileReportBook
from lite3_explorer.tile_tracker import TileTracker

SIDE_NAMES = {'N': 'north', 'E': 'east', 'S': 'south', 'W': 'west'}
TILE_COLOURS = {'visited': (0.2, 0.5, 1.0, 0.45), 'to_visit': (0.1, 0.8, 0.2, 0.35),
                'blocked': (0.8, 0.1, 0.1, 0.25), 'unreachable': (0.6, 0.0, 0.6, 0.35)}


# ------------------------------------------------------------------------------- Nav2 helpers
class Nav2Scanner:
    """Turn in place with Nav2's Spin behaviour (relative yaw)."""

    def __init__(self, node):
        self.node = node
        self.client = ActionClient(node, Spin, 'spin')
        self.state_client = node.create_client(GetState, '/behavior_server/get_state')
        self.behavior_active = False
        self._state_future = None
        self.seq, self.handle, self.status = 0, None, None

    def poll_behavior_server(self):
        """The spin server exists once behavior_server is configured, but rejects goals until it is ACTIVE."""
        if self.behavior_active:
            return
        f = self._state_future
        if f is not None:
            if not f.done():
                return
            res, self._state_future = f.result(), None
            if res is not None and res.current_state.id == LifecycleState.PRIMARY_STATE_ACTIVE:
                self.behavior_active = True
                self.node.get_logger().info('Nav2 behavior_server is active.')
                return
        if self.state_client.service_is_ready():
            self._state_future = self.state_client.call_async(GetState.Request())

    @property
    def ready(self):
        return self.client.server_is_ready() and self.behavior_active

    def start(self, deg, time_allowance_s):
        self.seq += 1
        seq = self.seq
        self.handle, self.status = None, None
        goal = Spin.Goal()
        goal.target_yaw = math.radians(deg)
        goal.time_allowance = DurationMsg(sec=int(time_allowance_s))
        self.node.get_logger().info(f'Spin {deg:+.0f} deg.')
        if not self.client.wait_for_server(timeout_sec=0.5):
            self.status = FAILED
            return

        def on_response(future):
            handle = future.result()
            if seq != self.seq:
                if handle is not None and handle.accepted:
                    handle.cancel_goal_async()
                return
            if handle is None or not handle.accepted:
                self.status = FAILED
                return
            self.handle = handle
            handle.get_result_async().add_done_callback(on_result)

        def on_result(future):
            if seq == self.seq:
                st = future.result().status
                self.status = SUCCEEDED if st == GoalStatus.STATUS_SUCCEEDED else FAILED
                self.handle = None

        self.client.send_goal_async(goal).add_done_callback(on_response)

    def poll(self):
        return self.status

    def cancel(self):
        self.seq += 1
        if self.handle is not None:
            self.handle.cancel_goal_async()
        self.handle, self.status = None, None

    def has_room(self):
        return self.node.turn_has_room()


class Nav2PathChecker:
    """Ask Nav2's planner for a path from the robot to each goal; results arrive asynchronously."""

    def __init__(self, node):
        self.node = node
        self.client = ActionClient(node, ComputePathToPose, 'compute_path_to_pose')
        self.seq, self.goals, self.results = 0, [], {}

    @property
    def ready(self):
        return self.client.server_is_ready()

    def start(self, goals):
        if not self.client.server_is_ready():
            return False
        self.seq += 1
        seq = self.seq
        self.goals, self.results = list(goals), {}
        for idx, (x, y) in enumerate(self.goals):
            goal = ComputePathToPose.Goal()
            goal.goal.header.frame_id = self.node.global_frame
            goal.goal.header.stamp = self.node.get_clock().now().to_msg()
            goal.goal.pose.position.x, goal.goal.pose.position.y = float(x), float(y)
            goal.goal.pose.orientation.w = 1.0
            goal.use_start = False                 # plan from the robot's current pose
            self.client.send_goal_async(goal).add_done_callback(
                lambda fut, i=idx: self._on_response(seq, i, fut))
        return True

    def _on_response(self, seq, idx, future):
        if seq != self.seq:
            return
        handle = future.result()
        if handle is None or not handle.accepted:
            self.results[idx] = None
            return
        handle.get_result_async().add_done_callback(lambda fut: self._on_result(seq, idx, fut.result()))

    def _on_result(self, seq, idx, wrapped):
        if seq != self.seq:
            return
        path = None
        if wrapped is not None and wrapped.status == GoalStatus.STATUS_SUCCEEDED and wrapped.result.path.poses:
            path = [(p.pose.position.x, p.pose.position.y) for p in wrapped.result.path.poses]
        self.results[idx] = path

    def poll(self):
        if len(self.results) < len(self.goals):
            return None
        return [self.results.get(i) for i in range(len(self.goals))]

    def cancel(self):
        self.seq += 1
        self.goals, self.results = [], {}


# ------------------------------------------------------------------------------- the node
class MissionManager(Node):
    def __init__(self):
        super().__init__('mission_manager', allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        flat = {name: p.value for name, p in self.get_parameters_by_prefix('').items()}
        self.cfg = cfg = merge_flat(DEFAULTS, flat)
        self.global_frame, self.robot_frame = cfg['global_frame'], cfg['robot_frame']
        self.t0 = time.monotonic()

        # run files + tile report
        self.output_dir = os.path.expanduser(cfg['output_dir'])
        self.run_dir = os.path.join(self.output_dir, 'run_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.run_dir, exist_ok=True)
        self.use_ai = bool(cfg['use_ai'])
        self.llm = LlmLink(self, enabled=self.use_ai, default_timeout_s=float(cfg['llm_report_timeout_s']))
        stamp = os.path.basename(self.run_dir)[len('run_'):]
        tcfg = cfg['tiles']
        self.book = TileReportBook(os.path.join(os.path.expanduser(cfg['reports_dir']), f'tile_report_{stamp}.md'),
                                   self.run_dir, self.use_ai, float(tcfg['reach_radius_m']))
        self.report_args, self.report_tries, self.report_calls = {}, {}, {}
        self.query_count = 0
        self.text_markers = []

        self.tiles = TileTracker(float(tcfg['size_m']), float(tcfg['reach_radius_m']), int(tcfg['max_failures']),
                                 on_reached=self._on_tile_reached, on_enter=self._on_tile_enter)
        self.target = None                     # (label, goal_xy) for the RViz highlight
        self._published_tiles = -1

        # TF, Nav2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._action_clients = {}
        self._traversals = {}
        self.scanner = Nav2Scanner(self)
        self.paths = Nav2PathChecker(self)
        self.clear_clients = [self.create_client(ClearEntireCostmap, n) for n in
                              ('/global_costmap/clear_entirely_global_costmap',
                               '/local_costmap/clear_entirely_local_costmap')]

        # topics
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, cfg['map_topic'], self._on_map, latched)
        self.global_costmap = None             # (grid, GridInfo, received)
        self.local_costmap = None              # (grid, origin, res, frame, received)
        self.create_subscription(OccupancyGrid, cfg['global_costmap_topic'], self._on_global_costmap, latched)
        self.create_subscription(OccupancyGrid, cfg['local_costmap_topic'], self._on_local_costmap, latched)
        self.create_subscription(String, '~/mode_cmd', self._on_mode_cmd, 10)
        # robot animations (animation_player): /play_animation starts one, /animation_active says when it is over
        self.animation_pub = self.create_publisher(String, '/play_animation', 10)
        self.create_subscription(Bool, '/animation_active', self._on_animation_active, 10)
        self._animation = 'done'               # pending (asked, not started) | running | done
        self.terrain_client = self.create_client(Trigger, cfg['terrain_classify_service']) \
            if cfg['terrain_classify_service'] else None
        self.create_service(SetBool, '~/set_active', self._on_set_active)
        self.create_service(Trigger, '~/report_now', self._on_report_now)
        self.mode_pub = self.create_publisher(String, '~/mode', latched)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.events_pub = self.create_publisher(String, '~/events', 50)
        self.goal_pub = self.create_publisher(PoseStamped, '~/current_goal', latched)
        self.walls_pub = self.create_publisher(PointCloud2, '/virtual_walls', 1)
        self.route_pub = self.create_publisher(Path, '~/route', latched)
        self.intent_pub = self.create_publisher(String, '~/next_target', latched)
        self.complete_pub = self.create_publisher(Bool, '/exploration_complete', latched)
        self.tile_pub = self.create_publisher(String, '/tile_map', latched)
        self.tile_report_pub = self.create_publisher(String, '/tile_report', QoSProfile(
            depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.tile_marker_pub = self.create_publisher(MarkerArray, '~/tile_markers', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', latched)
        self.crawl_pub = self.create_publisher(Bool, '/spin_crawl', 10)
        self.flat_pub = self.create_publisher(Bool, '/spin_flat', 10)
        self.complete_pub.publish(Bool(data=False))

        self.machine = ModeMachine(self, cfg)
        self.get_logger().info(
            f'Mission modes: {list(self.machine.modes)}; end condition: {cfg["end_condition"]["conditions"]} '
            f'({cfg["end_condition"]["combine"]}); AI {"on" if self.use_ai else "off"}; run files in {self.run_dir}; '
            f'tile report: {self.book.path}')
        self.machine.start()
        self.create_timer(float(cfg['tick_period_s']), self._tick)
        self.create_timer(0.5, self._track_tiles)
        self.create_timer(1.0, self._refresh_tiles)
        self.create_timer(1.0, self._publish_virtual_walls)     # re-marked every second: the lidar must not clear them
        self.create_timer(1.0, self._publish_status)

    # ===================================================== context API used by the modes
    def now(self):
        return time.monotonic()

    def log(self, level, msg):
        # One call site per severity: rclpy raises "Logger severity cannot be changed between calls" otherwise.
        logger = self.get_logger()
        if level in ('warn', 'warning'):
            logger.warning(msg)
        elif level == 'error':
            logger.error(msg)
        elif level == 'debug':
            logger.debug(msg)
        else:
            logger.info(msg)

    def pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, Time(),
                                                timeout=Duration(seconds=0.1))
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return [round(t.transform.translation.x, 3), round(t.transform.translation.y, 3), round(yaw, 4)]

    def local_map_ready(self, pose, radius_m=1.5, min_known=0.5):
        if self.tiles.map is None:
            return False
        grid, info = self.tiles.map
        r, c = info.world_to_cell(pose[0], pose[1])
        rad = int(radius_m / info.resolution)
        win = grid[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1]
        return win.size > 0 and (win >= 0).mean() >= min_known

    def ready_for_start(self):
        missing = []
        if self.pose() is None:
            missing.append(f'TF {self.global_frame}->{self.robot_frame}')
        if self.tiles.map is None:
            missing.append(self.cfg['map_topic'])
        turning = any(self.cfg['modes'].get(n, {}).get('turn', True) for n, m in self.machine.modes.items()
                      if m.type_name == 'scanning')
        if turning and not self.scanner.ready:
            missing.append('Nav2 spin (behavior_server active)')
        if not self.paths.ready:
            missing.append('Nav2 compute_path_to_pose')
        if self.use_ai and not self.llm.ready:
            missing.append('AI client (llm_client)')
        return not missing, missing

    def animation_play(self, name):
        self._animation = 'pending'
        self.animation_pub.publish(String(data=name))
        return True

    def animation_state(self):
        return self._animation

    def animation_cancel(self):
        self.animation_pub.publish(String(data='stop'))
        self._animation = 'done'

    def _on_animation_active(self, msg):
        if msg.data and self._animation == 'pending':
            self._animation = 'running'
        elif not msg.data and self._animation == 'running':
            self._animation = 'done'

    def gait_hold(self, kind):
        self.crawl_pub.publish(Bool(data=kind == 'crawl'))
        self.flat_pub.publish(Bool(data=kind == 'flat'))

    def action_client(self, action_type, name):
        key = (action_type.__name__, name)
        if key not in self._action_clients:
            self._action_clients[key] = ActionClient(self, action_type, name)
        return self._action_clients[key]

    def traversal(self, profile):
        if profile not in self._traversals:
            self._traversals[profile] = build_traversal(self, profile, self.cfg['traversals'])
        return self._traversals[profile]

    def costmap_cost_at_robot(self):
        if self.global_costmap is None or time.monotonic() - self.global_costmap[2] > 10.0:
            return None
        pose = self.pose()
        if pose is None:
            return None
        return G.value_at(self.global_costmap[0], self.global_costmap[1], pose[0], pose[1])

    def clear_costmaps(self):
        for client in self.clear_clients:
            if client.service_is_ready():
                client.call_async(ClearEntireCostmap.Request())
        self.get_logger().info('Cleared the Nav2 costmaps.')

    def _publish_route(self, goal_xy, route):
        """Sparse route (waypoints + goal) as a Path, instead of Nav2's dense /plan."""
        msg = Path()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        stops = route or ([(goal_xy[0], goal_xy[1], 0.0)] if goal_xy else [])   # no target: empty path clears RViz
        for x, y, yaw in stops:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            msg.poses.append(ps)
        self.route_pub.publish(msg)

    def set_target(self, label, goal_xy, route):
        self._publish_route(goal_xy, route)
        new = (label, tuple(goal_xy)) if label else None
        if new == self.target:
            return
        self.target = new
        self.intent_pub.publish(String(data=f'tile {label} at ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) '
                                             f'[{self.machine.current.name}]' if label else '(none)'))
        if self.tiles.ready:
            self._publish_tile_markers()

    def publish_goal(self, pose_stamped, profile):
        self.goal_pub.publish(pose_stamped)

    def event(self, kind, **data):
        rec = {'t': round(time.monotonic() - self.t0, 2), 'wall_time': dt.datetime.now().isoformat(timespec='seconds'),
               'kind': kind, 'mode': self.machine.current.name if getattr(self, 'machine', None) and
               self.machine.current else None, **data}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        self.events_pub.publish(String(data=line))
        try:
            with open(os.path.join(self.run_dir, 'events.jsonl'), 'a', encoding='utf-8') as fh:
                fh.write(line + '\n')
        except OSError as exc:
            self.get_logger().error(f'Cannot write events.jsonl: {exc}', throttle_duration_sec=30.0)

    def on_mode_change(self, mode, phase):
        self.mode_pub.publish(String(data=f'{mode}:{phase}' if phase else mode))

    def on_complete(self, reason):
        self.get_logger().info(f'MISSION COMPLETE: {reason}')
        self.complete_pub.publish(Bool(data=True))
        if self.tiles.ready:
            self.tiles.update(force=True)
            self._save_json('tiles.json', self.tiles.as_dict(self.global_frame))
        self._write_book(finished=True)
        self.get_logger().info(f'{len(self.book.entries)} tile(s) reported. Tile report: {self.book.path}')

    def turn_has_room(self):
        """False if the body would touch the local costmap's inscribed / lethal cells while turning."""
        if self.local_costmap is None:
            return True
        grid, origin, res, frame, received = self.local_costmap
        if time.monotonic() - received > 10.0:
            return True
        try:
            t = self.tf_buffer.lookup_transform(frame, self.robot_frame, Time(), timeout=Duration(seconds=0.1))
        except TransformException:
            return True
        ok, blocked = TC.can_turn_in_place(grid, origin, res, t.transform.translation.x, t.transform.translation.y,
                                           float(self.cfg['robot_length_m']), float(self.cfg['robot_width_m']),
                                           float(self.cfg['turn_room_margin_m']))
        if not ok:
            self.get_logger().info(f'No room to turn: {blocked} of 72 body orientations touch the obstacle zone.')
        return ok

    # ===================================================== inputs
    def _on_map(self, msg):
        i = msg.info
        self.tiles.set_map(G.grid_from_msg_data(msg.data, i.width, i.height),
                           G.GridInfo(i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y))

    def _on_global_costmap(self, msg):
        i = msg.info
        self.global_costmap = (G.grid_from_msg_data(msg.data, i.width, i.height),
                               G.GridInfo(i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y),
                               time.monotonic())

    def _on_local_costmap(self, msg):
        i = msg.info
        self.local_costmap = (G.grid_from_msg_data(msg.data, i.width, i.height),
                              (i.origin.position.x, i.origin.position.y), i.resolution, msg.header.frame_id,
                              time.monotonic())

    def _on_mode_cmd(self, msg):
        name = msg.data.strip()
        ok, why = self.machine.request(name, 'operator command')
        (self.get_logger().info if ok else self.get_logger().warning)(f'Mode command "{name}": {why}')

    def _on_set_active(self, request, response):
        response.message = self.machine.activate(bool(request.data))
        response.success = True
        return response

    def _on_report_now(self, request, response):
        pose = self.pose()
        if not self.tiles.ready or pose is None:
            response.success, response.message = False, 'no tile grid or robot pose yet'
        else:
            tile = self.tiles.tile_of(pose[0], pose[1])
            self._on_tile_reached(tile, pose, 'manual')
            response.success, response.message = True, f'describing tile {T.tile_label(tile)}'
        return response

    # ===================================================== loops
    def _tick(self):
        self.scanner.poll_behavior_server()
        self.llm.poll()
        self._poll_tile_reports()
        self.machine.tick()

    def _track_tiles(self):
        if self.tiles.ready:
            self.tiles.track(self.pose())

    def _publish_virtual_walls(self):
        """Virtual walls as a point cloud (Nav2 global costmap, source 'virtual_walls'): the wall behind the robot at
        the start and every tile edge with more than half of it on a mapped wall."""
        if not self.tiles.ready:
            return
        step = 0.05
        pts = []
        for (x0, y0), (x1, y1) in self.tiles.virtual_walls():
            n = max(1, int(math.dist((x0, y0), (x1, y1)) / step))
            pts += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, 0.1) for k in range(n + 1)]
        if not pts:
            return
        header = PoseStamped().header
        header.frame_id = self.global_frame
        header.stamp = self.get_clock().now().to_msg()
        self.walls_pub.publish(point_cloud2.create_cloud_xyz32(header, pts))

    def _refresh_tiles(self):
        self.tiles.update()
        if self.tiles.ready and self.tiles.version != self._published_tiles:
            self._published_tiles = self.tiles.version
            self.tile_pub.publish(String(data=json.dumps(self.tiles.as_dict(self.global_frame), default=str)))
            self._publish_tile_markers()

    def _publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.machine.status(), default=str)))

    # ===================================================== tile reports (AI descriptions)
    def _on_tile_enter(self, tile):
        """Entering another tile: ask the (triggered) terrain classifier to classify once."""
        if self.terrain_client is not None and self.terrain_client.service_is_ready():
            self.terrain_client.call_async(Trigger.Request())

    def _on_tile_reached(self, tile, pose, source):
        label = T.tile_label(tile)
        self.get_logger().info(f'Reached tile {label} ({source}).')
        self.event('tile_reached', tile=label, source=source, pose=pose)
        sides = T.tile_sides(*self.tiles.map, self.tiles.grid, tile) if self.tiles.map else {}
        self.book.reached(label, pose, source, sides)
        self.report_calls.pop(label, None)
        if not (self.use_ai and self.llm.ready):
            self.book.set_no_ai(label, 'AI disabled (use_ai:=false)' if not self.use_ai else 'llm_client is not running')
            self._write_book()
            self._publish_tile_report(label)
            return
        self.query_count += 1
        name = f'tile_{tile[0]}_{tile[1]}_q{self.query_count:04d}'
        walls = ', '.join(f'{SIDE_NAMES[k]} {v}' for k, v in sides.items())
        size = self.tiles.size_m
        context = (f'TILE CONTEXT: the robot has just reached the centre of tile {label} of its {size:g} m x {size:g} m '
                   f'tile grid (position ({pose[0]:.2f}, {pose[1]:.2f}) in the map frame).'
                   + (f' The lidar map says: {walls}.' if walls else '')) if pose else \
            f'TILE CONTEXT: the robot has just reached tile {label}.'
        self.report_args[label] = dict(timeout_s=float(self.cfg['llm_report_timeout_s']), name=name, tile=label,
                                       pose=pose, context=context, save_dir=self.run_dir)
        self.report_tries[label] = 1
        self.report_calls[label] = self.llm.request('describe_tile', **self.report_args[label])
        self._write_book()

    def _poll_tile_reports(self):
        for label, call in list(self.report_calls.items()):
            if not call.done:
                continue
            del self.report_calls[label]
            if call.ok:
                self.book.set_result(label, call.result)
                self.get_logger().info(f'Tile {label}: {call.result.get("scene_description")}')
            elif self.report_tries.get(label, 1) < int(self.cfg['llm_report_retries']) and self.llm.ready:
                self.report_tries[label] = self.report_tries.get(label, 1) + 1
                self.get_logger().warning(f'Tile {label}: AI description failed ({call.failure()}); retry '
                                          f'{self.report_tries[label]}/{self.cfg["llm_report_retries"]}')
                self.report_calls[label] = self.llm.request('describe_tile', **self.report_args[label])
                continue
            else:
                self.book.set_error(label, call.failure())
            self._write_book()
            self._publish_tile_report(label)

    def _write_book(self, finished=False):
        try:
            self.book.write(grid=self.tiles.grid.as_dict() if self.tiles.ready else None,
                            tile_counts=self.tiles.counts() if self.tiles.tiles else None,
                            to_visit=self.tiles.unexplored() if self.tiles.ready else None, finished=finished)
        except OSError as exc:
            self.get_logger().error(f'Cannot write the tile report {self.book.path}: {exc}', throttle_duration_sec=30.0)

    def _publish_tile_report(self, label):
        entry = self.book.entries[label]
        self.tile_report_pub.publish(String(data=json.dumps(entry, ensure_ascii=False, default=str)))
        if not self.tiles.ready:
            return
        cx, cy = self.tiles.grid.center(tuple(int(v) for v in label.split(',')))
        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'tile_report', entry['order'], Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.35
        m.pose.orientation.w = 1.0
        m.scale.z = 0.16
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.8, 0.1, 1.0) if entry['status'] == 'done' else \
            (0.8, 0.8, 0.8, 1.0)
        text = entry.get('description') if entry['status'] == 'done' else f'({entry["status"]})'
        m.text = f'{label}: {text[:60]}'
        self.text_markers = [x for x in self.text_markers if x.id != m.id] + [m]
        self.marker_pub.publish(MarkerArray(markers=self.text_markers))

    # ===================================================== RViz
    def _publish_tile_markers(self):
        tg = self.tiles.grid
        stamp = self.get_clock().now().to_msg()
        markers = []
        for n, (status, (r, g, b, a)) in enumerate(TILE_COLOURS.items()):
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = 'tiles', n, Marker.CUBE_LIST, Marker.ADD
            m.pose.position.x, m.pose.position.y = tg.origin_x, tg.origin_y
            m.pose.orientation.z, m.pose.orientation.w = math.sin(tg.yaw / 2), math.cos(tg.yaw / 2)
            m.scale.x = m.scale.y = 0.94 * tg.size
            m.scale.z = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            for (i, j), t in self.tiles.tiles.items():
                if t['status'] == status:
                    p = type(m.pose.position)()
                    p.x, p.y = (i + 0.5) * tg.size, (j + 0.5) * tg.size
                    m.points.append(p)
            markers.append(m)
        trail = Marker()
        trail.header.frame_id = self.global_frame
        trail.header.stamp = stamp
        trail.ns, trail.id, trail.type, trail.action = 'tiles', 10, Marker.LINE_STRIP, Marker.ADD
        trail.pose = markers[0].pose
        trail.scale.x = 0.06
        trail.color.r, trail.color.g, trail.color.b, trail.color.a = 0.0, 0.15, 0.6, 1.0
        for i, j in self.tiles.trail:
            p = type(trail.pose.position)()
            p.x, p.y, p.z = (i + 0.5) * tg.size, (j + 0.5) * tg.size, 0.05
            trail.points.append(p)
        if len(trail.points) < 2:
            trail.action = Marker.DELETE
        markers.append(trail)
        markers.extend(self._target_markers(stamp))
        self.tile_marker_pub.publish(MarkerArray(markers=markers))

    def _target_markers(self, stamp):
        """Orange highlight on the tile being driven to (DELETE when there is none, so RViz clears it)."""
        tg = self.tiles.grid
        out = []
        for n, ns, mtype in ((0, 'next_tile_cell', Marker.CUBE), (1, 'next_tile_goal', Marker.SPHERE),
                             (2, 'next_tile_label', Marker.TEXT_VIEW_FACING)):
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type = ns, n, mtype
            m.action = Marker.ADD if self.target else Marker.DELETE
            m.pose.orientation.z, m.pose.orientation.w = math.sin(tg.yaw / 2), math.cos(tg.yaw / 2)
            if self.target:
                label, goal = self.target
                cx, cy = tg.center(tuple(int(v) for v in label.split(',')))
                rgb = (1.0, 0.5, 0.0) if self.machine.current.type_name != 'recovery' else (0.9, 0.1, 0.9)
                m.color.r, m.color.g, m.color.b, m.color.a = (*rgb, 0.6 if n == 0 else 1.0)
                if n == 0:
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.03
                    m.scale.x = m.scale.y = 0.94 * tg.size
                    m.scale.z = 0.06
                elif n == 1:
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(goal[0]), float(goal[1]), 0.1
                    m.scale.x = m.scale.y = m.scale.z = 0.25
                else:
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.5
                    m.text = f'{self.machine.current.name.upper()} {label}'
                    m.scale.z = 0.3
            out.append(m)
        return out

    # ===================================================== output / shutdown
    def _save_json(self, name, data):
        with open(os.path.join(self.run_dir, name), 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    def shutdown(self):
        try:
            mode = self.machine.current
            if mode is not None:
                mode.exit('shutdown', 'node shutting down')
            self.gait_hold(None)
        finally:
            self._write_book(finished=self.machine.finished_reason is not None)


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
