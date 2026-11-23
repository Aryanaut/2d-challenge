"""Mission manager node: tile exploration of the arena with switchable mission modes.

The SLAM map is cut into 1 m tiles; the robot visits every identified walkable tile and the mission ends when the
configured end condition is met (default: all identified tiles explored). WHERE to go next comes from exploration
profiles (tiles by default, frontiers optionally), HOW to drive from traversal profiles.

Modes (mission/modes.py, configured in config/mission.yaml):
  idle -> scanning (tile grid + turn in place) -> exploring (next tile sharing a side with an explored tile,
          Nav2-reachable; optionally a frontier) -> scanning -> ...
  recovery: when stuck (no motion while driving, or sitting in the global costmap's obstacle zone) while unexplored
            tiles remain, or when no unexplored tile has a Nav2 path: drive to the nearest tile centre, then rescan.
  backtracking: a tile with two or more tiles to travel to is marked the 'last split'; when no exploration target is left
            but tiles are still unexplored the robot drives
            back to it and on to the tile it has not travelled to yet, then returns to exploring.
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
  sub  /curtain/points, /stairs/points   std_msgs/String   curtain / stair detectors (use_curtains, use_stairs)
  pub  /cmd_vel                 geometry_msgs/Twist   direct velocity (curtain, stairs and mppi modes)
  pub  /simple_cmd              transfer_interfaces/MotionSimpleCMD   robot AI mode on / off (stairs mode)
  pub  /ai_mode                 std_msgs/Bool     (latched) true while AI mode is on: gait_switcher and stair_detector pause
  pub  /stair_climber/enable    std_msgs/Bool     stairs mode with climber: true: hand the climb to stair_climber
  sub  /stair_climber/state     std_msgs/String   ... and watch it (DONE = crossed, ABORT = failed)
  act  spin, compute_path_to_pose, navigate_to_pose, follow_waypoints   (Nav2 clients)
  srv  /global_costmap/clear_entirely_global_costmap, /local_costmap/clear_entirely_local_costmap  (recovery)
"""
import datetime as dt
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped, Twist
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
from lite3_explorer.mission.config import DEFAULTS, merge_flat, mode_cfg
from lite3_explorer.mission.core import ModeMachine
from lite3_explorer.mission.traversal import FAILED, SUCCEEDED, build_traversal
from lite3_explorer.tile_reports import TileReportBook
from lite3_explorer.tile_tracker import TileTracker

SIDE_NAMES = {'N': 'north', 'E': 'east', 'S': 'south', 'W': 'west'}
TILE_COLOURS = {'visited': (0.2, 0.5, 1.0, 0.45), 'to_visit': (0.1, 0.8, 0.2, 0.35),
                'blocked': (0.8, 0.1, 0.1, 0.25), 'unreachable': (0.6, 0.0, 0.6, 0.35),
                'staircase': (1.0, 0.85, 0.0, 0.6)}


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
        self.goals, self.results = list(goals), {}
        self._send(self.seq, 0)
        return True

    def _send(self, seq, idx):
        """Goals go out ONE AT A TIME: Nav2's planner server runs a single goal and a newer one replaces the pending
        one, so sending them all at once aborts every goal but the first and the last (reported as 'no path')."""
        if seq != self.seq or idx >= len(self.goals):
            return
        x, y = self.goals[idx]
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = self.node.global_frame
        goal.goal.header.stamp = self.node.get_clock().now().to_msg()
        goal.goal.pose.position.x, goal.goal.pose.position.y = float(x), float(y)
        goal.goal.pose.orientation.w = 1.0
        goal.use_start = False                 # plan from the robot's current pose
        self.client.send_goal_async(goal).add_done_callback(lambda fut: self._on_response(seq, idx, fut))

    def _on_response(self, seq, idx, future):
        if seq != self.seq:
            return
        handle = future.result()
        if handle is None or not handle.accepted:
            self.results[idx] = None
            self._send(seq, idx + 1)
            return
        handle.get_result_async().add_done_callback(lambda fut: self._on_result(seq, idx, fut.result()))

    def _on_result(self, seq, idx, wrapped):
        if seq != self.seq:
            return
        path = None
        if wrapped is not None and wrapped.status == GoalStatus.STATUS_SUCCEEDED and wrapped.result.path.poses:
            path = [(p.pose.position.x, p.pose.position.y) for p in wrapped.result.path.poses]
        self.results[idx] = path
        self._send(seq, idx + 1)

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
        self.odom_frame = cfg.get('odom_frame', 'odom')
        self.t0 = time.monotonic()

        # run files + tile report
        self.output_dir = os.path.expanduser(cfg['output_dir'])
        self.run_dir = os.path.join(self.output_dir, 'run_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.run_dir, exist_ok=True)
        self.use_ai = bool(cfg['use_ai'])
        self.llm = LlmLink(self, enabled=self.use_ai, default_timeout_s=float(cfg['llm_report_timeout_s']))
        stamp = os.path.basename(self.run_dir)[len('run_'):]
        tcfg = cfg['tiles']
        self.scene_images_dir = os.path.join(os.path.expanduser(cfg['scene_images_dir']), os.path.basename(self.run_dir)[len('run_'):])
        self.book = TileReportBook(os.path.join(os.path.expanduser(cfg['reports_dir']), f'tile_report_{stamp}.md'),
                                   self.run_dir, self.use_ai, float(tcfg['reach_radius_m']))
        self.report_args, self.report_tries, self.report_calls = {}, {}, {}
        self.query_count = 0

        self.tiles = TileTracker(float(tcfg['size_m']), float(tcfg['reach_radius_m']), int(tcfg['max_failures']),
                                 on_reached=self._on_tile_reached,
                                 min_known=float(tcfg.get('min_known', 0.5)),
                                 scene_min_known=float(tcfg.get('scene_min_known', 0.95)),
                                 scene_max_free=float(tcfg.get('scene_max_free', 0.80)))
        self.tiles.skip_scene = bool(mode_cfg(cfg, 'exploring').get('scenes', False))   # scene tiles are looked at, never entered
        self.tiles.stair_min_points = int(tcfg.get('stair_min_points', 4))
        self.target = None                     # (label, goal_xy) for the RViz highlight
        self._published_tiles = -1

        # TF, Nav2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._action_clients = {}
        self._traversals = {}
        self.scanner = Nav2Scanner(self)
        self.paths = Nav2PathChecker(self)
        self.wall_paths = Nav2PathChecker(self)      # own checker: re-evaluates the tiles after the virtual walls change
        self._walls_sig = None
        self.opening_done = False                    # virtual walls + tile maths wait for the opening spin
        self.initial_heading = None                  # yaw the robot had when the opening spin began (it returns to it)
        self._reeval_at = None
        self._reeval_tiles = []
        self._reeval_back = {}
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
        self.terrain_marker_pub = self.create_publisher(MarkerArray, '~/terrain_markers', latched)
        # terrain kind (flat / uneven / slats ...) from the terrain classifier, which knows nothing about tiles: the tile
        # that was being looked at when it was asked gets the answer
        self.create_subscription(String, '/terrain_kind', self._on_terrain_kind, latched)
        # fallback when no kind could be named (e.g. the lidar patch is too sparse): rugged / non_rugged still decides
        # between hatched (uneven) and nothing
        self.create_subscription(String, '/terrain_classifier/details', self._on_terrain_details, 10)
        self._terrain_label, self._terrain_after = None, 0.0
        self._terrain_seen, self._terrain_asked = set(), {}
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', latched)
        self.crawl_pub = self.create_publisher(Bool, '/spin_crawl', 10)
        self.flat_pub = self.create_publisher(Bool, '/spin_flat', 10)
        self.step_up_pub = self.create_publisher(Bool, '/step_up', latched)      # h-step gait while driving to a higher tile
        self._step_up = None
        self._rise_target, self._rise_from, self._rise_max = None, 0.0, None
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
        self.create_timer(0.5, self._reevaluate_tiles)
        self.create_timer(1.0, self._publish_virtual_walls)     # re-marked every second: the lidar must not clear them
        self.create_timer(1.0, self._publish_status)
        # curtain mode (use_curtains): the detector's confirmed curtain spots, and direct velocity commands for the walk through it
        self._curtain_pts = np.zeros((0, 4))
        self.create_subscription(String, '/curtain/points', self._on_curtain_points, latched)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._cmd = None                                  # (vx, wz, expires): republished at 20 Hz until it expires
        self.create_timer(0.05, self._publish_cmd)
        # stairs mode (use_stairs): the stair detector's confirmed stair cells, and the robot's AI mode on / off
        self.create_subscription(String, '/stairs/points', self._on_stair_points, latched)
        self.ai_pub = self.create_publisher(Bool, '/ai_mode', latched)
        self.ai_pub.publish(Bool(data=False))
        # stairs mode with climber: true: the stair_climber package does the alignment + AI crossing
        self.climber_enable_pub = self.create_publisher(Bool, '/stair_climber/enable', 10)
        self._climber_state = (None, 0.0)
        self.create_subscription(String, '/stair_climber/state', self._on_climber_state, 10)
        self.simple_cmd_pub, self._simple_cmd_type = None, None
        try:
            from transfer_interfaces.msg import MotionSimpleCMD
            # same QoS as gait_switcher: the robot's domain bridge subscribes RELIABLE + TRANSIENT_LOCAL
            self.simple_cmd_pub = self.create_publisher(MotionSimpleCMD, '/simple_cmd', QoSProfile(
                depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
            self._simple_cmd_type = MotionSimpleCMD
        except ImportError:
            self.get_logger().warning('transfer_interfaces is not installed: the stairs mode cannot switch AI mode.')

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

    @staticmethod
    def _yaw_of(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def pose(self):
        """[x, y, yaw] of the robot in the global frame.

        Composed from the NEWEST map->odom and the NEWEST odom->base_link, each looked up separately. A single
        lookup_transform(map, base_link, Time()) returns the pose at the latest time the WHOLE chain has data for, i.e. the
        stamp of map->odom, and SLAM publishes that late (up to several seconds under load in the bags): the pose then
        froze for seconds and the omni recovery drove straight through its goal."""
        odom_frame = getattr(self, 'odom_frame', 'odom')
        try:
            if odom_frame in (self.global_frame, self.robot_frame):
                raise TransformException('no separate odom frame')
            a = self.tf_buffer.lookup_transform(self.global_frame, odom_frame, Time(), timeout=Duration(seconds=0.1))
            b = self.tf_buffer.lookup_transform(odom_frame, self.robot_frame, Time(), timeout=Duration(seconds=0.1))
            ya, yb = self._yaw_of(a.transform.rotation), self._yaw_of(b.transform.rotation)
            ca, sa = math.cos(ya), math.sin(ya)
            bx, by = b.transform.translation.x, b.transform.translation.y
            x = a.transform.translation.x + ca * bx - sa * by
            y = a.transform.translation.y + sa * bx + ca * by
            yaw = math.atan2(math.sin(ya + yb), math.cos(ya + yb))
            return [round(x, 3), round(y, 3), round(yaw, 4)]
        except TransformException:
            pass
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, Time(),
                                                timeout=Duration(seconds=0.1))
        except TransformException:
            return None
        return [round(t.transform.translation.x, 3), round(t.transform.translation.y, 3),
                round(self._yaw_of(t.transform.rotation), 4)]

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

    def _on_curtain_points(self, msg):
        try:
            self._curtain_pts = np.array(json.loads(msg.data).get('points', []), float).reshape(-1, 4)
        except (ValueError, TypeError):
            pass

    def curtain_points(self):
        return self._curtain_pts

    def _on_stair_points(self, msg):
        try:
            pts = np.array(json.loads(msg.data).get('points', []), float).reshape(-1, 4)
        except (ValueError, TypeError):
            return
        if self.tiles.set_stair_points(pts):
            self.tiles.update()
            self.log('info', f'Staircase tile(s): {", ".join(T.tile_label(t) for t in sorted(self.tiles.stairs))}.')
            self.event('staircases', tiles=[T.tile_label(t) for t in sorted(self.tiles.stairs)])

    def ai_flag(self, on):
        """/ai_mode only (gait_switcher / stair_detector pause), no /simple_cmd: stair_climber sends the AI codes."""
        self.ai_pub.publish(Bool(data=bool(on)))

    def stair_climber_enable(self, on):
        """Enable / disable the stair_climber node. Enabling sends false first, so it restarts from IDLE after an
        earlier DONE / ABORT."""
        if on:
            self.climber_enable_pub.publish(Bool(data=False))
            self._climber_state = (None, 0.0)          # ignore the state from before
        self.climber_enable_pub.publish(Bool(data=bool(on)))

    def _on_climber_state(self, msg):
        self._climber_state = (msg.data, time.monotonic())

    def stair_climber_state(self):
        """(last /stair_climber/state or None, seconds since it arrived)."""
        state, t = self._climber_state
        return (state, time.monotonic() - t) if state is not None else (None, math.inf)

    def ai_mode(self, on, cmd_code):
        """Robot AI mode on / off: MotionSimpleCMD {cmd_code, type: 0} on /simple_cmd (what `ros2 topic pub --once
        /simple_cmd ...` sends), and /ai_mode for gait_switcher / stair_detector. False when no robot subscriber heard it."""
        self.ai_pub.publish(Bool(data=bool(on)))
        if self.simple_cmd_pub is None:
            return False
        msg = self._simple_cmd_type()
        msg.cmd_code, msg.type = int(cmd_code), 0
        self.simple_cmd_pub.publish(msg)
        # a lone command can be lost (the gait switcher repeats its commands too): repeat it, newest command only
        self._ai_repeat = (msg, 2)
        if getattr(self, '_ai_repeat_timer', None) is None:
            self._ai_repeat_timer = self.create_timer(0.3, self._repeat_ai_cmd)
        return self.simple_cmd_pub.get_subscription_count() > 0

    def _repeat_ai_cmd(self):
        rep = getattr(self, '_ai_repeat', None)
        if rep is None or rep[1] <= 0:
            self._ai_repeat = None
            return
        self.simple_cmd_pub.publish(rep[0])
        self._ai_repeat = (rep[0], rep[1] - 1)

    def drive(self, vx, wz, vy=0.0):
        """Direct velocity command (curtain / mppi modes); lapses 0.5 s after the last call, so a stalled mission stops the robot."""
        self._cmd = (float(vx), float(wz), float(vy), time.monotonic() + 0.5)

    def stop(self):
        if self._cmd is not None:
            self._cmd = None
            self.cmd_pub.publish(Twist())

    def _publish_cmd(self):
        if self._cmd is None:
            return
        vx, wz, vy, until = self._cmd
        if time.monotonic() > until:
            self.stop()
            return
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        self.cmd_pub.publish(msg)

    def costmap_snapshot(self):
        """(grid, GridInfo) of the latest global costmap, or None when it is missing or stale."""
        if self.global_costmap is None or time.monotonic() - self.global_costmap[2] > 10.0:
            return None
        return self.global_costmap[0], self.global_costmap[1]

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
        self._update_step_up()
        if self.tiles.ready:
            self._publish_tile_markers()

    def _update_step_up(self):
        """/step_up is true while the robot drives to a target tile that is higher than the tile it started from. Both elevations
        are absolute, against the ground plane fitted at the start of the run (terrain classifier elevation option; a start tile
        with no measurement counts as the start floor, 0 m). The start elevation is taken once, when the target is set, and the
        largest elevation measured for the target is kept, so neither the robot moving over the step edge nor a later low reading
        of the raised tile switches the h-step gait off. It stays on until the target changes (the tile is reached, or given up)."""
        on = False
        if self.cfg.get('step_up_gait', True) and self.target and self.tiles.ready:
            if self._rise_target != self.target[0]:
                self._rise_target, self._rise_max = self.target[0], None
                pose = self.pose()
                here = self.tiles.elevation(T.tile_label(self.tiles.grid.tile_of(pose[0], pose[1]))) if pose is not None else None
                self._rise_from = here['elevation_m'] if here else 0.0
            goal = self.tiles.elevation(self.target[0])
            if goal:
                rise = goal['elevation_m'] - self._rise_from
                self._rise_max = rise if self._rise_max is None else max(self._rise_max, rise)
            on = self._rise_max is not None and self._rise_max >= float(self.cfg.get('step_up_min_rise_m', 0.04))
        else:
            self._rise_target, self._rise_max = None, None
        if on != self._step_up:
            self._step_up = on
            self.step_up_pub.publish(Bool(data=on))
            self.log('info', f'Target tile {self.target[0] if self.target else "-"}: '
                             + (f'{self._rise_max:+.2f} m above the start tile: h-step gait.' if on else 'no step up: terrain gait.'))
            self.event('step_up', on=on, tile=self.target[0] if self.target else None)

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
        if self.tiles.ready and self.opening_done:
            self.tiles.track(self.pose())

    def _publish_virtual_walls(self):
        """Virtual walls as a point cloud (Nav2 global costmap, source 'virtual_walls'): the wall behind the robot at
        the start and every tile edge with more than half of it on a mapped wall."""
        if not self.tiles.ready or not self.opening_done:
            return
        step = 0.05
        pts = []
        walls = self.tiles.virtual_walls()
        opened = self.tiles.open_curtains and bool(self.tiles.curtain_edges)
        if opened != getattr(self, '_curtains_opened', False):
            # the obstacle layer keeps virtual-wall marks (clearing: false): wipe them so the curtain edges really open
            # (and are re-marked from the next cloud once backtracking is over)
            self._curtains_opened = opened
            self.log('info', f'Curtain virtual walls {"lifted (backtracking)" if opened else "restored"}.')
            self.clear_costmaps()
        sig = (tuple(sorted((round(a, 2), round(b, 2), round(c, 2), round(d, 2)) for (a, b), (c, d) in walls)),
               sum(1 for t in self.tiles.tiles.values() if t.get('status') != 'unknown'))   # new tiles re-trigger it
        if sig != self._walls_sig:
            self._walls_sig = sig
            self._reeval_at = self.now() + 3.0       # let the global costmap take the new walls in first
        info = self.tiles.map[1] if self.tiles.map else None
        for (x0, y0), (x1, y1) in walls:
            n = max(1, int(math.dist((x0, y0), (x1, y1)) / step))
            pts += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n, 0.1) for k in range(n + 1)]
        if info is not None:   # only inside the SLAM map: the global costmap has its size and logs 'worldToMap failed'
            pts = [p for p in pts if 0 <= info.world_to_cell(p[0], p[1])[0] < info.height - 1
                   and 0 <= info.world_to_cell(p[0], p[1])[1] < info.width - 1]
        if not pts:
            return
        header = PoseStamped().header
        header.frame_id = self.global_frame
        header.stamp = self.get_clock().now().to_msg()
        self.walls_pub.publish(point_cloud2.create_cloud_xyz32(header, pts))

    def _reevaluate_tiles(self):
        """After the virtual walls changed: ask Nav2 for a path to every unexplored tile; a tile with none is eliminated.
        Nothing is eliminated when NO tile got a path (the planner failing from where the robot stands is not a wall)."""
        if self._reeval_tiles:
            results = self.wall_paths.poll()
            if results is None:
                if self.now() - self._reeval_started > 10.0:
                    self.wall_paths.cancel()
                    self._reeval_tiles = []
                    for label, why in self._reeval_back.items():
                        self.tiles.mark_unreachable(label, why)
                    self._reeval_back = {}
                return
            tiles, self._reeval_tiles = self._reeval_tiles, []
            back, self._reeval_back = self._reeval_back, {}
            if not any(results):
                self.log('warn', 'Virtual walls: no tile has a Nav2 path from here; eliminating nothing.')
                for label, why in back.items():           # keep the earlier eliminations
                    self.tiles.mark_unreachable(label, why)
                return
            doomed = [t['tile'] for t, path in zip(tiles, results)
                      if not path and t['tile'] in self.tiles.unexplored()]
            for label in doomed:
                self.tiles.mark_unreachable(label, 'no Nav2 path with the virtual walls')
            restored = [lab for lab in back if lab not in doomed]
            if restored:
                self.log('info', f'Newly scanned area: tile(s) {", ".join(restored)} have a Nav2 path again: restored.')
                self.event('restored', tiles=restored, why='Nav2 path after new scans')
                self._publish_tile_markers()
            if doomed:
                self.log('warn', f'Virtual walls: no Nav2 path to tile(s) {", ".join(doomed)}: eliminated.')
                self.event('eliminated', tiles=doomed, why='no Nav2 path with the virtual walls')
                self._publish_tile_markers()
            return
        if (self._reeval_at is None or self.now() < self._reeval_at or not self.tiles.ready
                or not self.opening_done):
            return
        pose = self.pose()
        if pose is None or not self.wall_paths.ready:
            return
        self._reeval_at = None
        # tiles eliminated for the virtual walls are checked again: new scans can open a way to them
        back = {lab: why for lab, why in self.tiles.unreachable.items()
                if why.startswith('no Nav2 path with the virtual walls')}
        for lab in back:
            self.tiles.unreachable.pop(lab)
        if back:
            self.tiles.invalidate()
            self.tiles.update()
        targets = self.tiles.targets(pose[:2], limit=10 ** 6, min_clearance_m=0.30)
        if targets and self.wall_paths.start([t['goal_xy'] for t in targets]):
            self._reeval_tiles, self._reeval_started, self._reeval_back = targets, self.now(), back
        else:
            for lab, why in back.items():
                self.tiles.mark_unreachable(lab, why)

    def _refresh_tiles(self):
        if not self.opening_done:
            return
        self.tiles.update()
        if self.tiles.ready and self.tiles.version != self._published_tiles:
            self._published_tiles = self.tiles.version
            self.tile_pub.publish(String(data=json.dumps(self.tiles.as_dict(self.global_frame), default=str)))
            self._publish_tile_markers()

    def _publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.machine.status(), default=str)))

    # ===================================================== tile reports (AI descriptions)
    def classify_tile(self, label):
        """Ask the terrain classifier what the floor is; the answer goes to tile `label` (the one the robot is pointing at)."""
        self._terrain_label, self._terrain_after = label, self.now()
        self._terrain_seen.discard(label)
        self.tiles.begin_terrain_look(label)
        self._terrain_asked[label] = self.now()
        if self.terrain_client is not None and self.terrain_client.service_is_ready():
            self.terrain_client.call_async(Trigger.Request())

    def terrain_status(self, label):
        """'pending' (asked, no answer yet) | 'seen' (the classifier named the floor) | 'unseen' (no answer within
        terrain_wait_s: the floor could not be seen). Without a classifier service every tile counts as seen."""
        if self.terrain_client is None or label in self._terrain_seen:
            return 'seen'
        if self.now() - self._terrain_asked.get(label, self.now()) >= float(self.cfg.get('terrain_wait_s', 3.0)):
            return 'unseen'
        return 'pending'

    TERRAIN_DEBUG_COLOURS = {'flat': [0.1, 0.8, 0.2, 0.25]}
    UNEVEN_MARKER = {'style': 'hatch', 'color': [1.0, 0.75, 0.0, 1.0], 'label': 'UNEVEN'}

    def _on_terrain_details(self, msg):
        """The classifier's rugged / non_rugged answer for the tile being looked at; used when no kind was named."""
        label = self._terrain_label
        if label is None or label in self._terrain_seen or self.now() < self._terrain_after:
            return
        try:
            details = json.loads(msg.data)
        except ValueError:
            return
        state = details.get('candidate', details.get('terrain'))      # this frame's answer, not the held label
        if state == 'rugged':
            self.tiles.set_terrain(label, {'kind': 'uneven', 'source': 'terrain_type', 'marker': self.UNEVEN_MARKER})
            self.get_logger().info(f'Tile {label}: terrain UNEVEN (rugged, no finer kind yet).')
            self.event('terrain', tile=label, terrain='uneven', info={'source': 'terrain_type'})
            self._publish_terrain_markers()
            self._terrain_seen.add(label)
        elif state == 'non_rugged':
            self.tiles.set_terrain(label, {'kind': 'flat', 'source': 'terrain_type', 'marker': {'style': 'none'}})
            self._terrain_seen.add(label)

    def _on_terrain_kind(self, msg):
        label = self._terrain_label
        try:
            result = json.loads(msg.data)
        except ValueError:
            return
        if label is None or result.get('kind') in (None, 'unknown') or self.now() < self._terrain_after:
            return
        self._terrain_seen.add(label)
        if result['kind'] == 'flat' and self.tiles.terrain.get(label, {}).get('kind') not in (None, 'flat'):
            kept = self.tiles.terrain[label]['kind']
            self.get_logger().info(f'Tile {label}: the classifier now says flat (spread {result.get("spread_median_m")} m, '
                                   f'rough {result.get("rough_fraction")}), but this look already found it {kept.upper()}: keeping that.')
            self.event('terrain_kept', tile=label, kept=kept, ignored='flat')
            return
        before = self.tiles.elevation(label)
        if self.tiles.set_terrain(label, result):
            self.get_logger().info(f'Tile {label}: terrain {result["kind"].upper()} '
                                   f'(spread {result.get("spread_median_m")} m, cells {result.get("n_cells")}).')
            self.event('terrain', tile=label, terrain=result['kind'], info={k: v for k, v in result.items()
                                                                          if k not in ('marker', 'kind')})
        self._report_elevation(label, before, self.tiles.elevation(label))
        self._update_step_up()
        self._publish_terrain_markers()

    def _report_elevation(self, label, before, now):
        """The classifier's elevation of tile `label` (the tile the robot chose to go to) against the ground reference taken at
        the start of the run (classifier option elevation.enabled). Logged and sent as an event when it appears or changes."""
        if not now or (before and before.get('elevation_class') == now.get('elevation_class')
                       and abs(before['elevation_m'] - now['elevation_m']) < 0.02):
            return
        rel = 'level with' if now['elevation_class'] == 'level' else f'{now["elevation_class"]} than'
        self.get_logger().info(f'Tile {label}: elevation {now["elevation_m"]:+.2f} m ({rel} the start floor), '
                               f'slope {now.get("slope_deg")} deg.')
        self.event('elevation', tile=label, **now)

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

    def describe_scene(self, label, pose):
        """Scene mode: ask the AI what the scene tile (the robot stands on a neighbour, facing it) looks like.
        Returns an LlmCall (done / ok / result / failure()); an AI that is off gives a call that has already failed."""
        self.query_count += 1
        size = self.tiles.size_m
        here = T.tile_label(self.tiles.grid.tile_of(pose[0], pose[1])) if self.tiles.ready else '?'
        context = (f"SCENE TILE CONTEXT: tile {label} of the robot's {size:g} m x {size:g} m tile grid is a scene tile "
                   f'(a scanned, mostly non-traversable tile the robot does not enter). The robot stands on the '
                   f'neighbouring tile {here} at ({pose[0]:.2f}, {pose[1]:.2f}) in the map frame and is facing tile {label}. '
                   f'Describe what is on tile {label}.')
        return self.llm.request('describe_tile', timeout_s=float(self.cfg['llm_report_timeout_s']),
                                name=f'scene_{label.replace(",", "_")}_q{self.query_count:04d}', tile=label, pose=pose,
                                context=context, save_dir=self.scene_images_dir)

    def record_scene(self, label, pose, call):
        """Write the scene tile's description (or the failure) into the tile report."""
        self.book.reached(label, pose, 'scene tile (looked at, not entered)', {})
        if call.ok:
            result = dict(call.result)
            if result.get('image'):
                result['image'] = os.path.join(self.scene_images_dir, result['image'])   # the report joins run_dir: absolute wins
            self.book.set_result(label, result)
            self.get_logger().info(f'Scene tile {label}: {call.result.get("scene_description")}')
        elif not self.use_ai:
            self.book.set_no_ai(label, 'AI disabled (use_ai:=false)')
        else:
            self.book.set_error(label, call.failure())
        self._write_book()
        self._publish_tile_report(label)

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
        self.tile_report_pub.publish(String(data=json.dumps(entry, ensure_ascii=False, default=str)))   # no RViz text: the descriptions are in the tile report

    # ===================================================== RViz
    def _publish_terrain_markers(self):
        """Each tile drawn in the marker style its terrain kind has in the classifier's config (solid / hatch)."""
        tg = self.tiles.grid
        if tg is None:
            return
        stamp = self.get_clock().now().to_msg()
        markers = [Marker(action=Marker.DELETEALL)]
        c, sn = math.cos(tg.yaw), math.sin(tg.yaw)
        hatch_m = 0.12
        for label, res in self.tiles.terrain.items():
            mk = dict(res.get('marker') or {})
            if mk.get('style', 'none') not in ('solid', 'hatch'):
                # debugging: a tile classified as plain ground is still drawn (faint solid + kind label)
                mk = {'style': 'solid', 'color': self.TERRAIN_DEBUG_COLOURS.get(res.get('kind'), [0.6, 0.6, 0.6, 0.2]),
                      'label': str(res.get('kind', '?')).upper()}
            res = dict(res, marker=mk)
            style = mk['style']
            i, j = (int(v) for v in label.split(','))
            cx, cy = tg.center((i, j))
            r, g, b, a = (float(v) for v in (list((res['marker']).get('color') or [1.0, 1.0, 1.0, 0.5]) + [1.0] * 4)[:4])
            base = (i + 500) * 3000 + (j + 500) * 3
            if style == 'solid':
                m = Marker()
                m.type, m.id = Marker.CUBE, base
                m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.03
                m.pose.orientation.z, m.pose.orientation.w = math.sin(tg.yaw / 2), math.cos(tg.yaw / 2)
                m.scale.x = m.scale.y = 0.96 * tg.size
                m.scale.z = 0.02
            else:                                                     # hatch: outline + diagonal lines
                m = Marker()
                m.type, m.id = Marker.LINE_LIST, base + 2
                m.pose.orientation.w = 1.0
                m.scale.x = 0.012
                half = 0.48 * tg.size

                def pt(lu, lv):
                    q = type(m.pose.position)()
                    q.x, q.y, q.z = cx + c * lu - sn * lv, cy + sn * lu + c * lv, 0.04
                    return q
                corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
                for q in range(4):
                    m.points += [pt(*corners[q]), pt(*corners[(q + 1) % 4])]
                d = -2 * half + hatch_m
                while d < 2 * half - 1e-6:
                    u0, u1 = max(-half, -half - d), min(half, half - d)
                    if u1 > u0:
                        m.points += [pt(u0, u0 + d), pt(u1, u1 + d)]
                    d += hatch_m
            m.header.frame_id, m.header.stamp, m.ns, m.action = self.global_frame, stamp, 'terrain', Marker.ADD
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            markers.append(m)
            text = (res['marker']).get('label')
            elevation = self.tiles.elevation(label)
            if elevation:                                  # second line: elevation against the start floor, level tiles too
                text = f'{text or str(res.get("kind", "")).upper()}\n{elevation["elevation_m"]:+.2f} m'
            if text:
                t = Marker()
                t.header.frame_id, t.header.stamp, t.ns, t.action = self.global_frame, stamp, 'terrain', Marker.ADD
                t.type, t.id = Marker.TEXT_VIEW_FACING, base + 1
                t.pose.position.x, t.pose.position.y, t.pose.position.z = cx, cy, 0.25
                t.pose.orientation.w = 1.0
                t.scale.z = 0.14
                t.color.r = t.color.g = t.color.b = t.color.a = 1.0
                t.text = str(text)
                markers.append(t)
        self.terrain_marker_pub.publish(MarkerArray(markers=markers))

    def _publish_tile_markers(self):
        tg = self.tiles.grid
        stamp = self.get_clock().now().to_msg()
        markers = [Marker(action=Marker.DELETEALL)]      # clears every namespace: scene tiles that stopped being scene disappear
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
        trail.pose = markers[1].pose
        trail.scale.x = 0.06
        trail.color.r, trail.color.g, trail.color.b, trail.color.a = 0.0, 0.15, 0.6, 1.0
        for i, j in self.tiles.trail:
            p = type(trail.pose.position)()
            p.x, p.y, p.z = (i + 0.5) * tg.size, (j + 0.5) * tg.size, 0.05
            trail.points.append(p)
        if len(trail.points) < 2:
            trail.action = Marker.DELETE
        markers.append(trail)
        markers.extend(self._scene_markers(stamp))
        markers.extend(self._split_markers(stamp))
        markers.extend(self._free_label_markers(stamp))
        markers.extend(self._target_markers(stamp))
        self.tile_marker_pub.publish(MarkerArray(markers=markers))

    def _hatch_marker(self, tg, tile, mid, ns, stamp, z=0.04, rgba=(0.2, 0.5, 1.0, 1.0), hatch_m=0.12):
        """A tile drawn as an outline with diagonal hatch lines."""
        cx, cy = tg.center(tile)
        c, sn = math.cos(tg.yaw), math.sin(tg.yaw)
        m = Marker()
        m.header.frame_id, m.header.stamp, m.ns, m.id = self.global_frame, stamp, ns, mid
        m.type, m.action = Marker.LINE_LIST, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.012
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        half = 0.48 * tg.size

        def pt(lu, lv):
            q = type(m.pose.position)()
            q.x, q.y, q.z = cx + c * lu - sn * lv, cy + sn * lu + c * lv, z
            return q
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        for q in range(4):
            m.points += [pt(*corners[q]), pt(*corners[(q + 1) % 4])]
        d = -2 * half + hatch_m
        while d < 2 * half - 1e-6:
            u0, u1 = max(-half, -half - d), min(half, half - d)
            if u1 > u0:
                m.points += [pt(u0, u0 + d), pt(u1, u1 + d)]
            d += hatch_m
        return m

    def _scene_markers(self, stamp):
        """Scene tiles (completely scanned, less than scene_max_free traversable) as blue hatched tiles."""
        tg = self.tiles.grid
        return [self._hatch_marker(tg, tile, 100 + n, 'scene_tiles', stamp, z=0.05)
                for n, tile in enumerate(self.tiles.scene_tiles())]

    def _split_markers(self, stamp):
        """The last split tile (two or more tiles to travel to; backtracking returns there) as a solid black tile."""
        top = self.tiles.last_split()
        if top is None:
            return []
        tg = self.tiles.grid
        cx, cy = tg.center(top['tile'])
        m = Marker()
        m.header.frame_id, m.header.stamp = self.global_frame, stamp
        m.ns, m.id, m.type, m.action = 'last_split', 0, Marker.CUBE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.04
        m.pose.orientation.z, m.pose.orientation.w = math.sin(tg.yaw / 2), math.cos(tg.yaw / 2)
        m.scale.x = m.scale.y = 0.94 * tg.size
        m.scale.z = 0.03
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.0, 0.0, 0.9
        return [m]

    def _free_label_markers(self, stamp):
        """Per-tile text: traversable share (compare with scene_max_free) and scanned share (compare with scene_min_known)."""
        tg = self.tiles.grid
        out = []
        for n, (tile, d) in enumerate(sorted(self.tiles.tiles.items())):
            cx, cy = tg.center(tile)
            m = Marker()
            m.header.frame_id, m.header.stamp = self.global_frame, stamp
            m.ns, m.id, m.type, m.action = 'tile_free', n, Marker.TEXT_VIEW_FACING, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.15
            m.pose.orientation.w = 1.0
            m.scale.z = 0.14
            m.color.r = m.color.g = m.color.b = m.color.a = 1.0
            m.text = f"free {d['free']:.2f}\nknown {d['known']:.2f}"
            out.append(m)
        return out

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
                rgb = {'recovery': (0.9, 0.1, 0.9), 'backtracking': (0.0, 0.8, 0.9)}.get(
                    self.machine.current.type_name, (1.0, 0.5, 0.0))
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
