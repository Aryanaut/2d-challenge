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
  pub  ~/curtain_markers         visualization_msgs/MarkerArray   the curtain wall (one tile edge) in yellow, for the whole run
  pub  /spin_crawl, /spin_flat  std_msgs/Bool     gait holds for gait_switcher while turning
  pub  /gait_freeze             std_msgs/Bool     (latched) true in recovery mode: gait_switcher keeps the current gait
  sub  /curtain/points, /stairs/points   std_msgs/String   curtain / stair detectors (use_curtains, use_stairs)
  sub  /roller/points           std_msgs/String   roller detector (use_rollers): roller tiles
  pub  /roller_mode             std_msgs/Bool     (latched) true while the roller mode holds the h-step gait (roller_detector pauses)
  pub  /cmd_vel                 geometry_msgs/Twist   direct velocity (curtain, stairs, roller and mppi modes)
  pub  /simple_cmd              transfer_interfaces/MotionSimpleCMD   robot AI mode on / off (stairs mode)
  sub  /robot_state             std_msgs/Int32MultiArray  [basic, gait, policy, motion] (lite3_explorer robot_state_bridge):
                                                  AI on / off is sent again until the robot reports it
  pub  /odom_rebase             geometry_msgs/PoseStamped  where the robot is after the AI-mode climb (odom_tf_bridge re-anchors odom)
  pub  /ai_mode                 std_msgs/Bool     (latched) true while AI mode is on: gait_switcher and stair_detector pause
  pub  /stairs/enable           std_msgs/Bool     (latched) stair_detector on / off: on for the opening scan
                                                  (modes.scanning.opening_stairs), all the time with use_stairs
  act  spin, compute_path_to_pose, navigate_to_pose, navigate_through_poses (follow_waypoints)   (Nav2 clients)
  srv  /global_costmap/clear_entirely_global_costmap, /local_costmap/clear_entirely_local_costmap  (recovery)
"""
import datetime as dt
import json
import math
import os
import subprocess
import time
from turtle import st
from unittest import result

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PoseStamped, Twist
from lifecycle_msgs.msg import State as LifecycleState
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose, Spin
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Path
from rcl_interfaces.msg import Log, SetParametersResult
from sensor_msgs.msg import CameraInfo, Imu, LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Empty, Int32MultiArray, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explorer import curtain_core as CC
from lite3_explorer import map_grid as G
from lite3_explorer import prop_votes as PV
from lite3_explorer import roller_core as RC
from lite3_explorer import tiles as T
from lite3_explorer import turn_check as TC
from lite3_explorer import nav2_traversal  # noqa: F401  (registers nav2_pose / nav2_route)
from lite3_explorer.llm_link import LlmLink
from lite3_explorer.mission.config import DEFAULTS, merge_flat, mode_cfg
from lite3_explorer.mission.core import ModeMachine
from lite3_explorer.mission.traversal import FAILED, SUCCEEDED, build_traversal
from lite3_explorer.mission_log import MissionLog, kind_of
from lite3_explorer.tile_reports import TileReportBook
from lite3_explorer.tile_tracker import TileTracker

SIDE_NAMES = {'N': 'north', 'E': 'east', 'S': 'south', 'W': 'west'}
TILE_COLOURS = {'visited': (0.2, 0.5, 1.0, 0.45), 'to_visit': (0.1, 0.8, 0.2, 0.35),
                'blocked': (0.8, 0.1, 0.1, 0.25), 'unreachable': (0.6, 0.0, 0.6, 0.35),
                'staircase': (1.0, 0.85, 0.0, 0.6), 'roller': (0.95, 0.95, 0.95, 0.6),
                'closed': (0.25, 0.25, 0.25, 0.6)}           # below the climbed staircase (j < its j): never again


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
        # The dog keeps turning after Spin's last command (bag fullrun_26_11_210206: -60 deg ended 7 deg past, +80 deg
        # ~10 deg past, in rug_hstep): ask for this much less, never less than half the turn (modes.scanning.spin_lead_deg).
        self.lead_deg = 6.0

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
        ask = math.copysign(max(abs(deg) - self.lead_deg, 0.5 * abs(deg)), deg)
        goal.target_yaw = math.radians(ask)
        goal.time_allowance = DurationMsg(sec=int(time_allowance_s))
        self.node.get_logger().info(f'Spin {deg:+.0f} deg (asking {ask:+.0f}: coasts the rest).')
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
        mlc = cfg.get('mission_log', {})
        self.mission_log = MissionLog(self.run_dir, time.time(), int(mlc.get('log_tail', 300))) \
            if mlc.get('enabled', True) else None
        self.book.mission_log = self.mission_log

        self.tiles = TileTracker(float(tcfg['size_m']), float(tcfg['reach_radius_m']), int(tcfg['max_failures']),
                                 on_reached=self._on_tile_reached,
                                 min_known=float(tcfg.get('min_known', 0.5)),
                                 scene_min_known=float(tcfg.get('scene_min_known', 0.85)),
                                 scene_max_free=float(tcfg.get('scene_max_free', 0.80)),
                                 scene_margin_m=float(tcfg.get('scene_margin_m', 0.10)))
        self.tiles.align_on_entry = bool(tcfg.get('align_on_entry', True))
        self.tiles.initial_clear_m = float(tcfg.get('initial_clear_m', 0.40))
        self.tiles.align_max_shift_m = float(tcfg.get('align_max_shift_m', 0.15))
        self.tiles.align_max_rot_deg = float(tcfg.get('align_max_rot_deg', 6.0))
        self.tiles.align_radius_m = float(tcfg.get('align_radius_m', 2.0))
        self.tiles.align_min_quality = float(tcfg.get('align_min_quality', 1.5))
        self.tiles.on_aligned = lambda r: (
            self.log('info', f'Tile {r["tile"][0]},{r["tile"][1]}: grid aligned to the walls (turn {r["rot_deg"]:+.1f} deg, '
                             f'shift {r["shift_u"]:+.2f} / {r["shift_v"]:+.2f} m).'),
            self.event('tile_aligned', **r))
        self.tiles.skip_scene = bool(mode_cfg(cfg, 'exploring').get('scenes', False))   # scene tiles are looked at, never entered
        self.tiles.stair_min_points = int(tcfg.get('stair_min_points', 4))
        self.tiles.stair_floor_min_known = float(tcfg.get('stair_floor_min_known', 0.90))
        self.tiles.stair_floor_min_free = float(tcfg.get('stair_floor_min_free', 0.95))
        self.tiles.stair_shift_max_m = float(tcfg.get('stair_shift_max_m', 0.35))
        self.tiles.roller_min_points = int(tcfg.get('roller_min_points', 4))
        self.tiles.roller_min_lidar = int(tcfg.get('roller_min_lidar', 2))
        self.tiles.roller_depth_only_points = int(tcfg.get('roller_depth_only_points', 0))
        self.tiles.roller_edge_margin_m = float(tcfg.get('roller_edge_margin_m', 0.15))
        self.tiles.prop_votes = bool(tcfg.get('prop_votes', False))
        self.tiles.scene_lidar_min_known = float(tcfg.get('scene_lidar_min_known', 0.0))
        self.tiles.prop_min_frames = int(tcfg.get('prop_min_frames', 8))
        self.tiles.prop_min_score = float(tcfg.get('prop_min_score', 4.0))
        self.target = None                     # (label, goal_xy) for the RViz highlight
        self._published_tiles = -1

        # TF, Nav2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._action_clients = {}
        self._traversals = {}
        self.scanner = Nav2Scanner(self)
        self.scanner.lead_deg = float(cfg['modes'].get('scanning', {}).get('spin_lead_deg', 6.0))
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
        self._markers = []                     # /animation_marker names not read yet
        self.create_subscription(String, '/animation_marker', lambda m: self._markers.append(m.data), 10)
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
        self.scene_type_pub = self.create_publisher(String, '/scene/result', QoSProfile(depth=20, durability=DurabilityPolicy.TRANSIENT_LOCAL)) ##added by debbie, publisher of scene type
        # {"tile": "i,j", "pos": [x, y]} while scene mode faces that scene tile, {} once it stops (mission reporter)
        self.scene_looking_pub = self.create_publisher(String, '/scene/looking', latched)
        self.robot_pos_pub = self.create_publisher(String, '/robot/position', 10) ##added by debbie, publisher of robot position
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
        # recovery mode (modes.recovery.keep_gait): the robot keeps the gait it already has (gait_switcher /gait_freeze)
        self.gait_freeze_pub = self.create_publisher(Bool, '/gait_freeze', latched)
        self._gait_frozen = False
        self._frozen_step_up = True                    # recovery step_up_gait: the elevation rule still switches while frozen
        self._step_tile = None                         # tile the robot is driving into now when it is not the target
                                                       # (backtracking: the next tile of the route, hop by hop)
        self.gait_freeze_pub.publish(Bool(data=False))
        self._hstep_hold = False                                                 # roller mode: h-step gait while crossing
        self.roller_mode_pub = self.create_publisher(Bool, '/roller_mode', latched)
        self.roller_mode_pub.publish(Bool(data=False))
        self._rise_target, self._rise_from, self._rise_max = None, 0.0, None
        # depth fallback (step_up_depth): a step up seen ahead by step_ahead_detector switches /step_up for the tile being
        # driven to as well, where the tile elevations miss it
        self._step_ahead = None                        # (detection dict, time) of the last {"step": true} frame
        self._depth_tile, self._depth_step = None, None  # the step tile a depth step was seen for, and that detection
        self.create_subscription(String, '/step_ahead', self._on_step_ahead, 10)
        self.complete_pub.publish(Bool(data=False))

        if self.mission_log is not None:
            self._setup_mission_log(cfg.get('mission_log', {}))

        self.machine = ModeMachine(self, cfg)
        self.get_logger().info(
            f'Mission modes: {list(self.machine.modes)}; end condition: {cfg["end_condition"]["conditions"]} '
            f'({cfg["end_condition"]["combine"]}); AI {"on" if self.use_ai else "off"}; run files in {self.run_dir}; '
            f'tile report: {self.book.path}')
        self.machine.start()
        self.create_timer(float(cfg['tick_period_s']), self._tick)
        self.create_timer(0.5, self._track_tiles)
        self._slam_mo, self._slam_hold_until = None, 0.0   # last map->odom seen, tile reaching held until (monotonic)
        if float(cfg['slam_jump_m']) > 0.0:
            self.create_timer(0.2, self._watch_slam_jump)
        self.create_timer(1.0, self._refresh_tiles)
        self.create_timer(0.5, self._reevaluate_tiles)
        self.create_timer(0.5, self._publish_robot_position) #publish robot position every 0.5 seconds
        self.create_timer(1.0, self._publish_virtual_walls)     # re-marked every second: the lidar must not clear them
        self.create_timer(1.0, self._publish_status)
        # curtain mode (use_curtains): the detector's confirmed curtain spots, and direct velocity commands for the walk through it
        self._curtain_pts = np.zeros((0, 5))           # x, y, ux, uy, 1 = camera spot / 0 = lidar only
        # the maze has ONE curtain: every spot ever confirmed is remembered for the rest of the run (the detector's own
        # set can shrink: lidar spots un-confirm as more scans come in). 5 cm cell -> [x, y, camera]; carried into the
        # new map's frame after the stairs (_carry_grid). From them the ONE tile edge the curtain lies on is picked
        # (curtain_core.pick_curtain_wall) as (tile, side), and only that wall is drawn in yellow on ~/curtain_markers.
        # It moves only to an edge with curtain_wall_switch_ratio x its camera spots (a grey wall seen close up can
        # give the first few spots). An edge the curtain mode actually walked through (tiles.curtain_edges) wins.
        self._curtain_seen = {}
        self._curtain_wall = None                      # (tile, side) of the curtain wall, once found
        self._curtain_switch_ratio = float(cfg['modes'].get('exploring', {}).get('curtain_wall_switch_ratio', 2.0))
        self._curtain_drawn = None                     # grid the wall was last drawn with (redrawn when the grid moves)
        self._curtain_ignore_until = 0.0               # after a new map: the detector's last old-frame points are ignored
        self.curtain_marker_pub = self.create_publisher(MarkerArray, '~/curtain_markers', latched)
        self.create_timer(1.0, self._update_curtain_wall)
        self.create_subscription(String, '/curtain/points', self._on_curtain_points, latched)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._cmd = None                                  # (vx, wz, expires): republished at 20 Hz until it expires
        self.create_timer(0.05, self._publish_cmd)
        # gait_switcher: a gait switch is running, the robot must stand still (direct /cmd_vel is zeroed meanwhile)
        self._gait_switching = False
        self.create_subscription(Bool, '/gait_switching', self._on_gait_switching, latched)
        # stairs mode (use_stairs): the stair detector's confirmed stair cells, and the robot's AI mode on / off
        self.create_subscription(String, '/stairs/points', self._on_stair_points, latched)
        # roller mode (use_rollers): the roller detector's confirmed roller spots
        self.create_subscription(String, '/roller/points', self._on_roller_points, latched)
        # scene-prop votes (tiles.prop_votes): the detector's scene props placed on the map keep a tile a scene tile.
        # Always subscribed (the points are collected even while off), so switching it on mid-run uses what was seen.
        self._prop_cam = (float(tcfg.get('prop_fx', 462.6)), float(tcfg.get('prop_cx', 320.1)))
        self._prop_last_pose = None                       # (monotonic t, yaw) of the previous detection message
        self._prop_logged = set()
        self.create_subscription(String, tcfg.get('prop_detections_topic', '/perception/detections'), self._on_detections, 10)
        if tcfg.get('prop_camera_info_topic'):
            self.create_subscription(CameraInfo, tcfg['prop_camera_info_topic'], self._on_prop_camera_info,
                                     qos_profile_sensor_data)
        self.add_on_set_parameters_callback(self._on_set_params)
        self.ai_pub = self.create_publisher(Bool, '/ai_mode', latched)
        self.ai_pub.publish(Bool(data=False))
        # stair detector on / off: always on when an exploring mode climbs staircases (use_stairs), else only while a
        # mode asks for it (the opening scan, modes.scanning.opening_stairs)
        self._stairs_always = any(m.type_name == 'exploring' and m.cfg.get('stairs', False)
                                  for m in self.machine.modes.values())
        self._stairs_enabled = None
        self.stairs_enable_pub = self.create_publisher(Bool, '/stairs/enable', latched)
        self.stair_detection(False)
        self.odom_rebase_pub = self.create_publisher(PoseStamped, '/odom_rebase', 10)
        # new map after the stairs (modes.stairs.new_map): save the map, restart slam_toolbox, start a new tile grid
        self._last_map, self._map_rx_at = None, 0.0
        self._nm = None                                   # the new-map job in progress (see start_new_map)
        self._map_parts = 1                               # tile report / map file number (part 1 = the first map)
        self.save_map_client = None
        try:
            from slam_toolbox.srv import SaveMap
            self._save_map_type = SaveMap
            self.save_map_client = self.create_client(SaveMap, '/slam_toolbox/save_map')
        except ImportError:
            self._save_map_type = None
        self.map_reset_pub = self.create_publisher(Empty, '/map_reset', 10)    # detectors forget what they saw
        self.create_timer(0.2, self._new_map_tick)
        # robot state [basic, gait, policy, motion] from robot_state_bridge (0x0901 UDP report):
        # the stairs mode sends AI on / off again until the robot reports it
        self._robot_state = (None, 0.0)
        self.create_subscription(Int32MultiArray, '/robot_state', self._on_robot_state, 10)
        # stairs FIX 3: the IMU stays live in AI mode (odom_tf_bridge freezes the map pose): pitch + yaw for the crossing
        self._imu = None
        self.create_subscription(Imu, '/imu/data', self._on_imu, qos_profile_sensor_data)
        # stairs: the wall beyond the staircase. The map pose is frozen in AI mode (and so is the local costmap), so the
        # crossing looks at the raw scan (base_link frame) for something in the body's corridor ahead (front_gap)
        self._scan = None
        self.create_subscription(LaserScan, cfg.get('front_scan_topic', '/scan'), self._on_scan, qos_profile_sensor_data)
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

    def _publish_robot_position(self):
        """Publish the tile the robot stands on, "i,j", for the mission reporter."""
        p = self.pose()
        if p is None or not self.tiles.ready:
            return
        i, j = self.tiles.grid.tile_of(p[0], p[1])
        k = 1.02                                           # tile position multiplier
        self.robot_pos_pub.publish(String(data=f'{i * k:.2f},{j * k:.2f}'))

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
        self._markers = []
        self.animation_pub.publish(String(data=name))
        return True

    def animation_markers(self):
        """The animation's marker names reached since the last call."""
        out, self._markers = self._markers, []
        return out

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
            data = json.loads(msg.data)
            pts = np.array(data.get('points', []), float).reshape(-1, 4)
            n_cam = int(data.get('camera', len(pts)))          # no count (older publishers, the sim): all camera spots
            self._curtain_pts = np.hstack([pts, (np.arange(len(pts)) < n_cam).astype(float)[:, None]])
        except (ValueError, TypeError):
            return
        if time.monotonic() >= self._curtain_ignore_until and len(pts):
            self._remember_curtain(self._curtain_pts[:, [0, 1, 4]])

    def _remember_curtain(self, rows):
        """Add confirmed curtain spots (x, y, camera) to the run's curtain memory, then look for the curtain wall."""
        for x, y, cam in np.asarray(rows, float).reshape(-1, 3):
            key = (int(math.floor(x / 0.05)), int(math.floor(y / 0.05)))
            e = self._curtain_seen.setdefault(key, [float(x), float(y), 0.0])
            e[2] = max(e[2], float(cam))
        self._update_curtain_wall()

    def _update_curtain_wall(self):
        """Pick (once) the tile edge the curtain lies on and draw it. A curtain edge the robot walked through overrides
        the pick (that is certain); otherwise the pick stays for the rest of the run, unless the curtain mode rejects
        it (timed out / blocked walking through it): then it is unmarked and the curtain is looked for elsewhere. Spots on
        a rejected edge or on the initial tile's edges never count (tiles.not_curtain_mask)."""
        tg = self.tiles.grid
        if tg is None:
            return
        if self._curtain_wall is not None and self.tiles.is_not_curtain(*self._curtain_wall):
            tile, side = self._curtain_wall
            self._curtain_wall = None
            self.log('warn', f'Curtain on the {SIDE_NAMES[side]} edge of tile {T.tile_label(tile)} was no curtain: '
                             f'unmarked, looking for it elsewhere.')
            self.event('curtain_wall_cleared', tile=T.tile_label(tile), side=side)
            self._publish_curtain_markers()
        if self._curtain_wall is not None and self._curtain_drawn != tg.as_dict():
            self._publish_curtain_markers()            # new / nudged grid (or a new map): the edge moved with it
        walked = next(iter(self.tiles.curtain_edges.values()), None)
        wall, found = None, None
        if walked is not None:
            tile, nb = walked
            wall = (tuple(tile), T.side_towards(tuple(tile), tuple(nb)))
            if self._curtain_wall is not None and T.edge_key(tg, *self._curtain_wall) == T.edge_key(tg, *wall):
                wall = None                                # already the drawn one
        elif self._curtain_seen:
            # redone on every update: a detected wall moves to an edge with clearly more camera spots (pick_curtain_wall)
            seen = np.array(list(self._curtain_seen.values()), float)
            found = CC.pick_curtain_wall(tg, seen[self.tiles.not_curtain_mask(seen)], current=self._curtain_wall,
                                         switch_ratio=self._curtain_switch_ratio)
            if found is not None:
                wall = (tuple(found['tile']), found['side'])
        if wall is None:
            return
        was = self._curtain_wall
        self._curtain_wall = wall
        (x0, y0), (x1, y1) = tg.edge(*wall)
        why = 'walked through' if walked is not None else 'moved' if was is not None else 'detected'
        extra = {} if found is None else {'camera_spots': found['camera_spots'], 'spots': found['spots']}
        if was is not None:
            extra['was'] = [T.tile_label(was[0]), was[1]]
        self.log('info', f'Curtain {why} on the {SIDE_NAMES[wall[1]]} edge of tile {T.tile_label(wall[0])}'
                         + (f' (was the {SIDE_NAMES[was[1]]} edge of tile {T.tile_label(was[0])})' if was else '')
                         + ': marked in yellow.')
        self.event('curtain_wall', tile=T.tile_label(wall[0]), side=wall[1], how=why,
                   segment=[[round(x0, 2), round(y0, 2)], [round(x1, 2), round(y1, 2)]], **extra)
        self._publish_curtain_markers()

    def _publish_curtain_markers(self):
        """The curtain wall: a yellow, curtain-high panel along its tile edge (kept for the rest of the run)."""
        m = Marker()
        m.header.frame_id, m.header.stamp = self.global_frame, self.get_clock().now().to_msg()
        m.ns, m.id, m.type = 'curtain', 0, Marker.CUBE
        tg = self.tiles.grid
        if self._curtain_wall is None or tg is None:
            m.action = Marker.DELETE
        else:
            (x0, y0), (x1, y1) = tg.edge(*self._curtain_wall)
            yaw = math.atan2(y1 - y0, x1 - x0)
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (x0 + x1) / 2, (y0 + y1) / 2, 0.35
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.scale.x, m.scale.y, m.scale.z = math.dist((x0, y0), (x1, y1)), 0.06, 0.7
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.9, 0.0, 0.9
        self._curtain_drawn = tg.as_dict() if m.action == Marker.ADD else None
        self.curtain_marker_pub.publish(MarkerArray(markers=[m]))

    def curtain_points(self):
        return self._curtain_pts

    def _on_stair_points(self, msg):
        try:
            pts = np.array(json.loads(msg.data).get('points', []), float).reshape(-1, 4)
        except (ValueError, TypeError):
            return
        if self.tiles.set_stair_points(pts):
            self.tiles.update()
            self.log('info', f'Staircase tile {", ".join(T.tile_label(t) for t in sorted(self.tiles.stairs))}: '
                             f'marked as the staircase for the rest of the run.')
            self.event('staircases', tiles=[T.tile_label(t) for t in sorted(self.tiles.stairs)])

    def _on_roller_points(self, msg):
        try:
            pts = RC.points_from_json(json.loads(msg.data))
        except (ValueError, TypeError):
            return
        if self.tiles.set_roller_points(pts):
            self.tiles.update()
            self.log('info', f'Roller tile(s): {", ".join(self.tiles.rollers_left()) or "(all crossed)"}.')
            self.event('rollers', tiles=self.tiles.rollers_left())

    def hstep(self, on):
        """Hold the h-step gait (/step_up true: gait_switcher sends rug_hstep) while the roller or curtain mode crosses;
        /roller_mode tells the roller detector to pause."""
        on = bool(on)
        if on != self._hstep_hold:
            self._hstep_hold = on
            self.roller_mode_pub.publish(Bool(data=on))
            self._update_step_up()

    def rebase_odom(self, x, y, yaw):
        """Tell odom_tf_bridge the robot is at (x, y, yaw) in the global frame (after AI mode threw the leg odometry
        off): it moves odom->base_link there; slam_toolbox scan-matches the remainder."""
        ps = PoseStamped()
        ps.header.frame_id = self.global_frame
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
        self.odom_rebase_pub.publish(ps)

    def _on_prop_camera_info(self, msg):
        if msg.k[0] > 0:
            self._prop_cam = (float(msg.k[0]), float(msg.k[2]))

    def _on_set_params(self, params):
        """Runtime switches: tiles.prop_votes / prop_min_frames / prop_min_score (ros2 param set /mission_manager ...)."""
        for p in params:
            if p.name == 'tiles.prop_votes':
                self.tiles.prop_votes = bool(p.value)
                self.log('info', f'Scene-prop votes {"ON" if p.value else "OFF"}.')
            elif p.name == 'tiles.prop_min_frames':
                self.tiles.prop_min_frames = int(p.value)
            elif p.name == 'tiles.prop_min_score':
                self.tiles.prop_min_score = float(p.value)
            elif p.name == 'tiles.scene_lidar_min_known':
                self.tiles.scene_lidar_min_known = float(p.value)
            else:
                continue
            self.tiles.invalidate()
            self.tiles.update()
        return SetParametersResult(successful=True)

    def _prop_this_side(self, p, x0, y0, push):
        """A prop point that prop_votes.place pushed across a WALLED tile edge goes back to the camera's side of it: the
        ray stopped at the prop (or the wall it stands against), and the camera cannot see through a wall. Bag
        fuck_26_11_212613: the cleaning-room cart against tile 2,0's E wall voted 3,0 a scene tile 9 times, 2,0 never,
        and the robot drove into the cleaning room."""
        r = math.hypot(p[0] - x0, p[1] - y0)
        if r < 1e-6:
            return p
        back = push + 0.10                                   # behind the hit, a little inside the near tile
        bx, by = p[0] - back * (p[0] - x0) / r, p[1] - back * (p[1] - y0) / r
        far, near = tuple(self.tiles.grid.tile_of(p[0], p[1])), tuple(self.tiles.grid.tile_of(bx, by))
        if far != near and T.side_towards(near, far) is not None and not self.tiles._open_between(near, far):
            return (round(bx, 3), round(by, 3)) + tuple(p[2:])
        return p

    def _on_detections(self, msg):
        """/perception/detections: place the scene props on the map (prop_votes.place) and count them per tile."""
        if not self.tiles.ready or self.tiles.map is None:
            return
        try:
            dets = json.loads(msg.data).get('detections') or []
        except (ValueError, AttributeError):
            return
        tcfg = self.cfg['tiles']
        labels = [l for l in tcfg.get('prop_labels', PV.DEFAULT_PROPS) if l]
        if not any(d.get('label') in labels for d in dets):
            return
        pose = self.pose()
        if pose is None:
            return
        now = time.monotonic()
        last, self._prop_last_pose = self._prop_last_pose, (now, pose[2])
        if last is not None and now - last[0] < 2.0:      # turning fast: the frame was taken at another heading
            rate = abs(math.degrees(math.atan2(math.sin(pose[2] - last[1]), math.cos(pose[2] - last[1])))) / max(now - last[0], 1e-3)
            if rate > float(tcfg.get('prop_max_turn_dps', 30.0)):
                return
        grid, info = self.tiles.map
        cam_x, push = float(tcfg.get('prop_camera_x_m', 0.252)), float(tcfg.get('prop_push_m', 0.15))
        pts = PV.place(dets, pose, grid, info, fx=self._prop_cam[0], cx=self._prop_cam[1],
                       cam_x=cam_x, props=labels, push_m=push,
                       min_conf=float(tcfg.get('prop_min_conf', 0.35)), max_range_m=float(tcfg.get('prop_max_range_m', 3.0)))
        x0, y0 = pose[0] + cam_x * math.cos(pose[2]), pose[1] + cam_x * math.sin(pose[2])
        pts = [self._prop_this_side(p, x0, y0, push) for p in pts]
        here = tuple(self.tiles.grid.tile_of(pose[0], pose[1]))
        pts = [p for p in pts if tuple(self.tiles.grid.tile_of(p[0], p[1])) != here]   # the tile the robot stands on is no scene
        for tile in self.tiles.add_prop_points(pts):
            label = T.tile_label(tile)
            if label in self._prop_logged:
                continue
            self._prop_logged.add(label)
            e = self.tiles.prop_counts().get(tile, {})
            state = 'kept a scene tile' if self.tiles.prop_votes else 'not used (tiles.prop_votes off)'
            self.log('info', f'Tile {label}: scene props seen ({e.get("frames")} sightings, {e.get("labels")}): {state}.')
            self.event('prop_scene', tile=label, frames=e.get('frames'), score=e.get('score'), labels=e.get('labels'),
                       used=bool(self.tiles.prop_votes))

    def _on_imu(self, msg):
        q, w = msg.orientation, msg.angular_velocity
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        self._imu = (pitch, self._yaw_of(q), w.z, time.monotonic())

    def _on_scan(self, msg):
        r = np.asarray(msg.ranges, dtype=float)
        a = msg.angle_min + msg.angle_increment * np.arange(r.size)
        ok = np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)
        # how old the scan already is when it arrives (bag fullrun_25_11_183742: ~1.1 s from the Livox to /scan)
        lag = (self.get_clock().now().nanoseconds - Time.from_msg(msg.header.stamp).nanoseconds) * 1e-9
        self._scan = (r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), msg.header.frame_id, max(0.0, lag), time.monotonic())

    def front_gap(self, half_width_m=0.20, nose_m=0.305):
        """(gap_m, age_s): how far ahead of the nose (nose_m in front of the scan frame's origin) the nearest scan return
        lies in the corridor |y| < half_width_m; gap_m None = no return in the corridor. age_s = how old the scan is NOW
        (transport lag + time since it arrived): the robot has kept walking since. None = no scan yet."""
        if self._scan is None:
            return None
        x, y, _frame, lag, received = self._scan
        sel = (x > 0.0) & (np.abs(y) < half_width_m)
        gap = float(np.min(x[sel])) - nose_m if np.any(sel) else None
        return gap, lag + (time.monotonic() - received)

    def imu(self):
        """(pitch_rad (ROS: + = nose DOWN), yaw_rad, gyro_z_rad_s, age_s) of the latest /imu/data, or None."""
        if self._imu is None:
            return None
        pitch, yaw, gz, t = self._imu
        return pitch, yaw, gz, time.monotonic() - t

    def _on_robot_state(self, msg):
        if len(msg.data) >= 4:
            self._robot_state = (tuple(int(v) for v in msg.data[:4]), time.monotonic())

    def robot_state(self, max_age_s=0.5):
        """(basic, gait, policy, motion) from /robot_state, or None when there is none this fresh."""
        st, t = self._robot_state
        return st if st is not None and time.monotonic() - t <= max_age_s else None

    def stair_detection(self, on):
        """Stair detector on / off (/stairs/enable). It stays on regardless with use_stairs (exploring climbs staircases)."""
        on = bool(on) or self._stairs_always
        if on != self._stairs_enabled:
            self._stairs_enabled = on
            self.stairs_enable_pub.publish(Bool(data=on))
            self.get_logger().info(f'Stair detector {"on" if on else "off"}.')

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

    def _on_gait_switching(self, msg):
        self._gait_switching = bool(msg.data)

    def _publish_cmd(self):
        if self._cmd is None:
            return
        vx, wz, vy, until = self._cmd
        if time.monotonic() > until:
            self.stop()
            return
        msg = Twist()
        if not self._gait_switching:                     # a gait switch runs: hold still (gait_switcher sends zero too)
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
        self._step_tile = None
        self.intent_pub.publish(String(data=f'tile {label} at ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) '
                                             f'[{self.machine.current.name}]' if label else '(none)'))
        self._update_step_up()
        if self.tiles.ready:
            self._publish_tile_markers()

    def set_step_tile(self, label):
        """The tile the robot is driving into now, for the step-up check (backtracking: the next tile of the route, hop
        by hop; None = the target tile). Its rise is measured from the tile the robot is on when it changes."""
        if label == self._step_tile:
            return
        self._step_tile = label
        self._update_step_up()

    def _update_step_up(self):
        """/step_up is true while the robot drives to a target tile that is higher than the tile it started from. Both elevations
        are absolute, against the ground plane fitted at the start of the run (terrain classifier elevation option; a start tile
        with no measurement counts as the start floor, 0 m). The start elevation is taken once, when the target is set, and the
        largest elevation measured for the target is kept, so neither the robot moving over the step edge nor a later low reading
        of the raised tile switches the h-step gait off. It stays on until the target changes (the tile is reached, or given up)."""
        if self._gait_frozen and not self._frozen_step_up:
            return                                # recovery keeps the gait: /step_up stays as it was
        on = False
        step_tile = (self._step_tile or self.target[0]) if self.target else None
        depth = self._update_depth_step(step_tile)
        if self.cfg.get('step_up_gait', True) and step_tile and self.tiles.ready:
            if self._rise_target != step_tile:
                self._rise_target, self._rise_max = step_tile, None
                pose = self.pose()
                here = self.tiles.elevation(T.tile_label(self.tiles.grid.tile_of(pose[0], pose[1]))) if pose is not None else None
                max_slope = float(self.cfg.get('step_up_from_max_slope_deg', 15.0))
                if here and max_slope > 0 and float(here.get('slope_deg') or 0.0) > max_slope:
                    self.log('info', f'Current tile elevation {here["elevation_m"]:+.2f} m at slope {here["slope_deg"]} deg '
                                     f'is not a floor: step up measured from the start floor.')
                    here = None
                self._rise_from = here['elevation_m'] if here else 0.0
            goal = self.tiles.elevation(step_tile)
            if goal:
                rise = goal['elevation_m'] - self._rise_from
                self._rise_max = rise if self._rise_max is None else max(self._rise_max, rise)
            on = self._rise_max is not None and self._rise_max >= float(self.cfg.get('step_up_min_rise_m', 0.04))
        else:
            self._rise_target, self._rise_max = None, None
        rise = on
        on = on or depth is not None or getattr(self, '_hstep_hold', False)   # depth fallback; the roller mode crossing
        if on != self._step_up:
            self._step_up = on
            self.step_up_pub.publish(Bool(data=on))
            self.log('info', f'Target tile {step_tile or "-"}: '
                     + (f'{self._rise_max:+.2f} m above the start tile: h-step gait.' if rise else
                        f'depth camera sees a step up {depth["rise_m"] * 100:+.0f} cm, {depth["dist_m"]:.2f} m ahead: '
                        f'h-step gait.' if depth is not None else
                        'crossing a roller / curtain: h-step gait.' if on else 'no step up: terrain gait.'))
            self.event('step_up', on=on, tile=step_tile, by='elevation' if rise else 'depth' if depth is not None
                       else 'hstep_hold' if on else None)

    def _on_step_ahead(self, msg):
        """step_ahead_detector: one JSON per depth frame; {"step": true, ...} = a step up ahead of the body."""
        try:
            det = json.loads(msg.data)
        except ValueError:
            return
        if not det.get('step'):
            return
        self._step_ahead = (det, self.now())
        if self._depth_step is None and self.cfg.get('step_up_depth', True):
            self._update_step_up()

    def _update_depth_step(self, step_tile):
        """The depth fallback for /step_up: a step seen ahead (in the last step_up_depth_max_age_s) while driving to
        step_tile holds the h-step gait until step_tile changes, like the elevation rule (the step leaves the camera view
        before the rear legs are up). Returns that detection, or None."""
        if not (self.cfg.get('step_up_gait', True) and self.cfg.get('step_up_depth', True) and step_tile):
            self._depth_tile, self._depth_step = None, None
            return None
        if self._depth_tile != step_tile:
            self._depth_tile, self._depth_step = step_tile, None
        if self._depth_step is None and self._step_ahead is not None:
            det, t = self._step_ahead
            if self.now() - t <= float(self.cfg.get('step_up_depth_max_age_s', 1.0)):
                self._depth_step = det
                self.event('step_ahead', tile=step_tile, **{k: det.get(k) for k in ('rise_m', 'dist_m', 'at_edge')})
        return self._depth_step

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
        m = self.machine.modes.get(mode) if getattr(self, 'machine', None) else None
        freeze = m is not None and m.type_name == 'recovery' and bool(m.cfg.get('keep_gait', True))
        if freeze != self._gait_frozen:
            self._gait_frozen = freeze
            if freeze:
                self._frozen_step_up = bool(m.cfg.get('step_up_gait', True))
            self.gait_freeze_pub.publish(Bool(data=freeze))
            self.log('info', ('Recovery: keeping the current gait' + (' (step ups still switch to h-step).'
                              if self._frozen_step_up else '.')) if freeze else 'Gait follows terrain / targets again.')
            if not freeze:
                self._update_step_up()            # the target may have changed meanwhile

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
        grid = G.grid_from_msg_data(msg.data, i.width, i.height)
        info = G.GridInfo(i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y)
        self._last_map = (grid, info)
        self._map_rx_at = time.monotonic()
        self.tiles.set_map(grid, info)

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

    def _watch_slam_jump(self):
        """map->odom only moves by scan-matching corrections of a few cm; a jump of slam_jump_m / slam_jump_deg in one
        check is SLAM re-placing the robot (a loop closure), possibly falsely. Hold tile reaching for slam_jump_hold_s so
        the jumped pose marks no tile (fullrun_25_11_215528: 'tile_reached 1,-2' 1 s after the robot was on 0,-4)."""
        now = time.monotonic()
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.odom_frame, Time())
        except TransformException:
            t = None
        if t is not None:
            cur = (t.transform.translation.x, t.transform.translation.y, self._yaw_of(t.transform.rotation))
            nm = getattr(self, '_nm', None)
            if nm is not None and nm['state'] == 'working':
                cur = None                       # new map: slam_toolbox restarts map->odom; the next one is the baseline
            prev, self._slam_mo = self._slam_mo, cur
            if prev is not None and cur is not None:
                d = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
                dyaw = math.degrees(abs(math.atan2(math.sin(cur[2] - prev[2]), math.cos(cur[2] - prev[2]))))
                if d > float(self.cfg['slam_jump_m']) or dyaw > float(self.cfg['slam_jump_deg']):
                    self._slam_hold_until = now + float(self.cfg['slam_jump_hold_s'])
                    self.log('warn', f'SLAM jump: {self.global_frame}->{self.odom_frame} moved {d:.2f} m / {dyaw:.0f} deg '
                                     f'at once (loop closure?); no tile counts as reached for '
                                     f'{float(self.cfg["slam_jump_hold_s"]):.0f} s.')
                    self.event('slam_jump', moved_m=round(d, 3), turned_deg=round(dyaw, 1),
                               before=[round(v, 3) for v in prev], after=[round(v, 3) for v in cur])
        self.tiles.reach_hold = now < self._slam_hold_until

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
    ELEV_FIELDS = ('elevation_m', 'elevation_class', 'slope_deg', 'elevation_spread_m', 'elevation_points')

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
        before = self.tiles.elevation(label)
        if result['kind'] == 'flat' and self.tiles.terrain.get(label, {}).get('kind') not in (None, 'flat'):
            kept = self.tiles.terrain[label]['kind']
            self.get_logger().info(f'Tile {label}: the classifier now says flat (spread {result.get("spread_median_m")} m, '
                                   f'rough {result.get("rough_fraction")}), but this look already found it {kept.upper()}: keeping that.')
            self.event('terrain_kept', tile=label, kept=kept, ignored='flat')
            # The kind is kept, but the elevation is still news: steptest_26_11_133619 tile 1,0 was UNEVEN (/terrain_type)
            # before any elevation came in, every later answer (flat, 0.51 m higher) was dropped here, /step_up stayed
            # off and the robot hit the step in the flat gait.
            elev = {k: result[k] for k in self.ELEV_FIELDS if k in result}
            if elev:
                self.tiles.set_terrain(label, dict(self.tiles.terrain[label], **elev))
        elif self.tiles.set_terrain(label, result):
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

    def describe_scene(self, label, pose, view=None):
        """Scene mode: ask the AI what the scene tile (the robot stands on a neighbour, facing it) looks like.
        Returns an LlmCall (done / ok / result / failure()); an AI that is off gives a call that has already failed."""
        self.query_count += 1
        size = self.tiles.size_m
        here = T.tile_label(self.tiles.grid.tile_of(pose[0], pose[1])) if self.tiles.ready else '?'
        context = (f"SCENE TILE CONTEXT: tile {label} of the robot's {size:g} m x {size:g} m tile grid is a scene tile "
                   f'(a scanned, mostly non-traversable tile the robot does not enter). The robot stands on the '
                   f'neighbouring tile {here} at ({pose[0]:.2f}, {pose[1]:.2f}) in the map frame and is facing tile {label}. '
                   f'Classify what is on tile {label}: one of the scene types listed above, or traversable.')
        if view:
            context += (f' This picture is the {view} view of three (left, centre, right: the robot yaws its body to '
                        f'each), so part of the tile may be out of frame.')
        return self.llm.request('describe_tile', timeout_s=float(self.cfg['llm_report_timeout_s']),
                                name=f'scene_{label.replace(",", "_")}_q{self.query_count:04d}' + (f'_{view}' if view else ''), tile=label, pose=pose,
                                context=context, save_dir=self.scene_images_dir, prompt='describe_scene')

    def scene_looking(self, label):
        """Scene mode faces scene tile `label` now (None: no longer). The mission reporter gives what perception sees
        meanwhile that tile's room and position (tile x 1.02) once the AI has named it."""
        msg = {}
        if label:
            i, j = (int(v) for v in label.split(','))
            msg = {'tile': label, 'pos': [round(i * 1.02, 2), round(j * 1.02, 2)]}
        self.scene_looking_pub.publish(String(data=json.dumps(msg)))

    def record_scene(self, label, pose, call):
        """Write the scene tile's description (or the failure) into the tile report."""
        self.book.reached(label, pose, 'scene tile (looked at, not entered)', {})
        # every scene tile's answer goes to the mission reporter, traversable / failed ones too (scene_type ""):
        # it releases what perception saw while facing the tile. pos = the SCENE tile x 1.02 (as /robot/position)
        i, j = (int(v) for v in label.split(','))
        scene_msg = {'tile': label, 'scene_type': '', 'pos': [round(i * 1.02, 2), round(j * 1.02, 2)], 'confidence': None}
        if call.ok:
            result = dict(call.result)
            if result.get('scene_type') == 'traversable':
                tile = tuple(int(v) for v in label.split(','))
                if self.tiles.is_prop_scene(tile):                 # scene props were seen on it: a bad picture, not a free tile
                    e = self.tiles.prop_counts().get(tile, {})
                    self.log('warn', f'Scene tile {label}: the AI said traversable, but scene props were seen on it '
                                     f'({e.get("frames")} sightings, {e.get("labels")}): it stays a scene tile.')
                    self.event('scene_traversable_vetoed', tile=label, frames=e.get('frames'), labels=e.get('labels'))
                else:
                    self.tiles.mark_traversable(tile)              # no scene: an ordinary tile again
                    self.event('scene_traversable', tile=label)
            scene_msg['scene_type'] = result.get('scene_type') or ''
            scene_msg['confidence'] = result.get('scene_confidence', result.get('confidence'))
            if result.get('image'):
                result['image'] = os.path.join(self.scene_images_dir, result['image'])   # the report joins run_dir: absolute wins
            if result.get('images'):
                result['images'] = {k: os.path.join(self.scene_images_dir, v) for k, v in result['images'].items()}
            self.book.set_result(label, result)
            self.get_logger().info(f'Scene tile {label}: {call.result.get("scene_description")}')
        elif not self.use_ai:
            self.book.set_no_ai(label, 'AI disabled (use_ai:=false)')
        else:
            self.book.set_error(label, call.failure())
        self.scene_type_pub.publish(String(data=json.dumps(scene_msg)))
        self._write_book(finished=self.machine.finished_reason is not None)   # a background answer may come after the end
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

    # ===================================================== perception + mission reporter, into the tile report
    def _setup_mission_log(self, mlc):
        self._ml_dirty = False
        self._ml_groups = {**{n.lstrip('/'): 'perception' for n in mlc.get('perception_nodes', [])},
                           **{n.lstrip('/'): 'reporter' for n in mlc.get('reporter_nodes', [])}}
        self._ml_min_level = int(mlc.get('min_log_level', 20))
        def safe(cb):          # the report is a side job: a bad message must never take the mission manager down
            def run(msg):
                try:
                    cb(msg)
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().error(f'tile report ({cb.__name__}): {exc!r}', throttle_duration_sec=10.0)
            return run
        if mlc.get('event_topic'):
            self.create_subscription(String, mlc['event_topic'], safe(self._on_perception_event), 50)
        if mlc.get('qr_topic'):
            self.create_subscription(String, mlc['qr_topic'], safe(self._on_qr_detection), 20)
        if mlc.get('qr_text_topic'):
            self.create_subscription(String, mlc['qr_text_topic'], safe(self._on_qr_text), 20)
        if mlc.get('result_topic'):    # latched + reliable like the reporter's publisher
            self.create_subscription(String, mlc['result_topic'], safe(self._on_mission_result), QoSProfile(
                depth=50, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        if self._ml_groups:
            self.create_subscription(Log, '/rosout', safe(self._on_rosout), QoSProfile(
                depth=1000, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE))
        self.create_timer(5.0, self._flush_mission_log)    # log lines only mark the report dirty

    def _tile_at(self, xy):
        """Tile label "i,j" of a map point (None before the tile grid exists)."""
        if xy is None or not self.tiles.ready:
            return None
        return T.tile_label(self.tiles.grid.tile_of(float(xy[0]), float(xy[1])))

    def _record_sighting(self, kind, label, thing_xy=None, robot_xy=None, t=None, **kw):
        if robot_xy is None:
            p = self.pose()
            robot_xy = p[:2] if p is not None else None
        tile = self._tile_at(thing_xy if thing_xy is not None else robot_xy)
        if self.tiles.ready:
            self.mission_log.fill_tiles(self._tile_at)   # those seen before the grid existed (so they merge with this one)
        s = self.mission_log.sighting(t or time.time(), kind, label, tile=tile,
                                      pos=thing_xy if thing_xy is not None else robot_xy,
                                      robot_tile=self._tile_at(robot_xy), robot_pos=robot_xy, **kw)
        if s is None:
            return
        self.event('detection', what=kind, label=label, tile=tile, pos=s['pos'], robot_tile=s['robot_tile'])
        self.log('info', f'Detected {kind} {label!r} on tile {tile or "?"} (robot on {s["robot_tile"] or "?"}).')
        self._write_book(finished=self.machine.finished_reason is not None)

    @staticmethod
    def _json(msg):
        try:
            d = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            return None
        return d if isinstance(d, dict) else None

    def _on_perception_event(self, msg):
        ev = self._json(msg)
        if ev is None:
            return
        loc, rp = ev.get('location') or {}, ev.get('robot_pose') or {}
        xy = loc.get('xy')
        robot_xy = [rp['x'], rp['y']] if 'x' in rp and 'y' in rp else None
        st = ev.get('stamp') or {}
        t = st.get('sec', 0) + st.get('nanosec', 0) * 1e-9 if st.get('sec') else None
        kind = kind_of(ev)
        notes = [f'{loc["range_m"]:.1f} m' if isinstance(loc.get('range_m'), (int, float)) else '']
        if kind == 'survivor' and 'qr_in_view' in ev:
            notes.append('QR in view' if ev['qr_in_view'] else 'no QR in view')
        if ev.get('simulated'):
            notes.append('sim')
        self._record_sighting(kind, str(ev.get('label', '')), xy if xy and len(xy) >= 2 else None, robot_xy, t,
                              confidence=ev.get('confidence'), source='perception',
                              note=', '.join(n for n in notes if n))

    def _on_qr_detection(self, msg):
        d = self._json(msg)
        if d is None:
            return
        mp = d.get('map_point')
        xy = mp[:2] if isinstance(mp, (list, tuple)) and len(mp) >= 2 and None not in mp[:2] else None
        text = str(d.get('decoded') or d.get('text') or '')
        note = f'raw {d["text"]!r}' if d.get('text') and d.get('decoded') and d['text'] != d['decoded'] else ''
        if isinstance(d.get('distance_m'), (int, float)):
            note = ', '.join(x for x in (note, f'{d["distance_m"]:.1f} m') if x)
        self._record_sighting('qr', text, xy, t=d.get('first_seen_unix'),
                              source=f'qr_scanner ({d.get("camera", "?")} camera)', note=note)

    def _on_qr_text(self, msg):
        txt = msg.data.strip()
        if txt.startswith('{'):
            d = self._json(msg) or {}
            txt = str(d.get('message') or d.get('decoded') or d.get('text') or '').strip()
        if txt:
            self._record_sighting('qr', txt, source='perception qr_text')

    def _on_mission_result(self, msg):
        d = self._json(msg)
        if d is None:
            return
        self.mission_log.result(time.time(), d)
        self.event('mission_result', result=d)
        self._write_book(finished=self.machine.finished_reason is not None)

    def _on_rosout(self, msg):
        group = self._ml_groups.get(msg.name.lstrip('/').split('.')[-1]) or self._ml_groups.get(msg.name.lstrip('/'))
        if group is None or msg.level < self._ml_min_level:
            return
        self.mission_log.log_line(group, msg.stamp.sec + msg.stamp.nanosec * 1e-9, msg.level, msg.name, msg.msg)
        self._ml_dirty = True

    def _flush_mission_log(self):
        if self.tiles.ready and self.mission_log.fill_tiles(self._tile_at):
            self._ml_dirty = True
        if self._ml_dirty:
            self._ml_dirty = False
            self._write_book(finished=self.machine.finished_reason is not None)

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
        markers.extend(self._stair_ref_markers(stamp))
        self.tile_marker_pub.publish(MarkerArray(markers=markers))

    def _stair_ref_markers(self, stamp):
        """Reference tiles beyond staircases (odom is rebased to them after a climb): an off-grid outline turned to the
        stairs with an arrow up the steps. Orange while planned, green once odom was rebased to it."""
        out = []
        for n, ref in enumerate(getattr(self.tiles, 'stair_refs', [])):
            rgba = (0.1, 0.8, 0.2, 1.0) if ref.get('used') else (1.0, 0.55, 0.0, 1.0)
            m = Marker()
            m.header.frame_id, m.header.stamp = self.global_frame, stamp
            m.ns, m.id, m.type, m.action = 'stair_ref', 2 * n, Marker.LINE_STRIP, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.04
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            for x, y in list(ref['corners']) + [ref['corners'][0]]:
                q = type(m.pose.position)()
                q.x, q.y, q.z = x, y, 0.07
                m.points.append(q)
            a = Marker()
            a.header.frame_id, a.header.stamp = self.global_frame, stamp
            a.ns, a.id, a.type, a.action = 'stair_ref', 2 * n + 1, Marker.ARROW, Marker.ADD
            a.pose.position.x, a.pose.position.y, a.pose.position.z = ref['centre_xy'][0], ref['centre_xy'][1], 0.07
            a.pose.orientation.z, a.pose.orientation.w = math.sin(ref['yaw'] / 2), math.cos(ref['yaw'] / 2)
            a.scale.x, a.scale.y, a.scale.z = 0.4 * ref['size'], 0.06, 0.06
            a.color.r, a.color.g, a.color.b, a.color.a = rgba
            out += [m, a]
        return out

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
        """Scene tiles (almost completely scanned, free/known < scene_max_free) as blue hatched tiles."""
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
        """Per-tile text: traversable share (free/known is compared with scene_max_free) and scanned share (compare with scene_min_known)."""
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
            # the scene test is free/KNOWN < scene_max_free (and known >= scene_min_known), over the tile's interior only
            # (scene_margin_m in from the edges: 'inner'); 'free' alone is free/TOTAL over the whole tile, which reads lower
            # than the tested ratio whenever the tile is not fully scanned (0.79 free of 0.97 known = 0.81)
            ratio = d['free'] / d['known'] if d['known'] > 0 else 0.0
            m.text = (f"inner free/known {d.get('scene_free', ratio):.2f}\nfree/known {ratio:.2f}\n"
                      f"known {d['known']:.2f}\nfree {d['free']:.2f}")
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
                rgb = {'recovery': (0.9, 0.1, 0.9), 'backtracking': (0.0, 0.8, 0.9), 'ending': (0.1, 0.9, 0.2)}.get(
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
    # ===================================================== new map after the stairs
    def start_new_map(self, out_dir=None, timeout_s=60.0):
        """Stairs mode (new_map): save the map so far under out_dir (default reports_dir; <run>_part<n>.pgm / .yaml, tile
        report, tiles json), then restart slam_toolbox (Humble has no reset service; the launch file respawns it) and
        start a fresh tile grid. Poll new_map_state(): 'working' | 'done' | 'failed'. A failure leaves the old map, grid
        and report exactly as they were."""
        if self._nm is not None and self._nm['state'] == 'working':
            return True
        out = os.path.expanduser(out_dir or self.cfg['reports_dir'])
        stamp = os.path.basename(self.run_dir)[len('run_'):]
        self._nm = {'state': 'working', 'step': 'save', 'dir': out, 't0': self.now(), 'timeout': float(timeout_s),
                    'base': os.path.join(out, f'map_{stamp}_part{self._map_parts}'), 'future': None}
        return True

    def new_map_state(self):
        return self._nm['state'] if self._nm is not None else 'failed'

    def _new_map_fail(self, why):
        self.log('error', f'New map after the stairs: {why}; carrying on with the old map.')
        self.event('new_map_failed', why=why, step=self._nm['step'])
        self._nm['state'] = 'failed'

    def _new_map_tick(self):
        nm = self._nm
        if nm is None or nm['state'] != 'working':
            return
        if self.now() - nm['t0'] > nm['timeout']:
            return self._new_map_fail(f'timed out in step "{nm["step"]}"')
        if nm['step'] == 'save':
            if self.save_map_client is None or not self.save_map_client.wait_for_service(timeout_sec=0.0):
                if self.now() - nm['t0'] > 5.0:
                    self._new_map_fail('slam_toolbox save_map service not available')
                return
            try:
                os.makedirs(nm['dir'], exist_ok=True)
                if self.tiles.ready:
                    self.tiles.update(force=True)
                    with open(nm['base'] + '_tiles.json', 'w', encoding='utf-8') as fh:
                        json.dump(self.tiles.as_dict(self.global_frame), fh, indent=2, ensure_ascii=False, default=str)
            except OSError as exc:
                return self._new_map_fail(f'cannot write to {nm["dir"]}: {exc}')
            self._write_book(finished=True)
            req = self._save_map_type.Request()
            req.name.data = nm['base']
            nm['future'] = self.save_map_client.call_async(req)
            nm['step'] = 'saving'
        elif nm['step'] == 'saving':
            if not nm['future'].done():
                return
            res = nm['future'].result()
            if res is None or res.result != 0 or not os.path.exists(nm['base'] + '.yaml'):
                return self._new_map_fail('slam_toolbox could not save the map')
            nm['old_pose'] = self.pose()                 # same robot, both maps: gives the old frame -> new frame transform
            nm['old_grid'], nm['crossed'] = self.tiles.grid, set(self.tiles.stairs_crossed)
            nm['initial'] = (self.tiles.initial_tile, self.tiles.initial_left)
            nm['not_curtains'] = list(self.tiles.not_curtains)
            nm['forward'] = self.tiles.forward          # 'behind the stairs' stays relative to the original heading
            self.log('info', f'Map up to the staircase saved: {nm["base"]}.pgm/.yaml (+ _tiles.json, tile report {self.book.path}).')
            self.event('map_saved', path=nm['base'], tile_report=self.book.path)
            # clear: stop slam_toolbox; the launch file respawns it with an empty map (map frame = odom at that moment)
            if subprocess.run(['pkill', '-f', 'async_slam_toolbox_node'], check=False).returncode != 0:
                return self._new_map_fail('could not stop slam_toolbox (pkill found no async_slam_toolbox_node)')
            nm['killed_at'] = time.monotonic()
            nm['step'] = 'restart'
        elif nm['step'] == 'restart':
            # a map published well after the kill = the respawned slam_toolbox is mapping from scratch
            if self._map_rx_at < nm['killed_at'] + 1.0 or self._last_map is None:
                return
            if self.pose() is None:
                return
            self._new_map_apply()
            nm['state'] = 'done'

    def _new_map_apply(self):
        """slam_toolbox runs an empty map: forget the old grid and everything on it. The next scan (opening scan:
        spin, tile grid on the new map with the robot's tile as start tile and the wall behind it, i.e. against the
        staircase) establishes the new grid."""
        self._map_parts += 1
        stamp = os.path.basename(self.run_dir)[len('run_'):]
        self.book = TileReportBook(os.path.join(os.path.expanduser(self.cfg['reports_dir']),
                                                f'tile_report_{stamp}_part{self._map_parts}.md'),
                                   self.run_dir, self.use_ai, float(self.cfg['tiles']['reach_radius_m']))
        self.book.mission_log = self.mission_log      # detections / reporter results go on in the new part
        self.report_args, self.report_tries, self.report_calls = {}, {}, {}
        self.tiles.reset()
        self.tiles.set_map(*self._last_map)
        n_curtain = len(self._curtain_seen)
        self._carry_grid()
        if n_curtain and self.tiles.carry is None:
            # no transform to the new frame (nor the same tile numbering): the old curtain would be drawn in the wrong place
            self._curtain_seen, self._curtain_wall = {}, None
            self.log('warn', 'New map: the curtain seen before cannot be carried over (no transform); forgotten.')
        self._curtain_ignore_until = time.monotonic() + 2.0   # the detector's old-frame spots still in flight
        self._publish_curtain_markers()
        self.machine.reset_for_new_map()
        self.opening_done = False
        self._walls_sig, self._reeval_at, self._reeval_tiles, self._reeval_back = None, None, [], {}
        self._published_tiles = -1
        self.target = None
        self.map_reset_pub.publish(Empty())
        self.clear_costmaps()              # the old virtual walls are in the old map's frame
        self.tile_marker_pub.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))   # old tiles off RViz
        self.log('info', f'New map started (part {self._map_parts}); tile report {self.book.path}.')
        self.event('new_map', part=self._map_parts, tile_report=self.book.path)

    def _carry_grid(self):
        """The new map has its own frame (slam_toolbox starts at the robot). The robot's pose before and after tells the
        old-frame -> new-frame transform; the old tile grid moved through it keeps its tile numbering and coordinates
        (start tile = the tile the robot is on, e.g. 0,3 after crossing from 0,1), and so do the staircase walls."""
        nm = self._nm
        old, a, b = nm.get('old_grid'), nm.get('old_pose'), self.pose()
        if old is None or a is None or b is None:
            self.log('warn', 'New map: previous tile grid or pose unknown; the new map gets its own tile numbering.')
            return
        d = T.wrap(b[2] - a[2])
        c, s = math.cos(d), math.sin(d)
        ox, oy = old.origin_x - a[0], old.origin_y - a[1]
        grid = T.TileGrid(b[0] + c * ox - s * oy, b[1] + s * ox + c * oy, T.wrap(old.yaw + d), old.size)
        # the remembered curtain moves through the same old-frame -> new-frame transform
        seen, self._curtain_seen = list(self._curtain_seen.values()), {}
        for x, y, cam in seen:
            ux, uy = x - a[0], y - a[1]
            nx, ny = b[0] + c * ux - s * uy, b[1] + s * ux + c * uy
            self._curtain_seen[(int(math.floor(nx / 0.05)), int(math.floor(ny / 0.05)))] = [nx, ny, cam]
        # the curtain wall is kept as (tile, side): the carried grid has the same numbering, so it stays where it was
        self.tiles.carry = {'grid': grid, 'stairs_crossed': nm.get('crossed', set()), 'forward': nm.get('forward'),
                           'initial_tile': nm.get('initial', (None, False))[0],
                           'initial_left': nm.get('initial', (None, False))[1],
                           'not_curtains': nm.get('not_curtains', [])}
        self.log('info', f'New map: tile grid carried over (robot on tile {T.tile_label(grid.tile_of(b[0], b[1]))}, '
                         f'frame turned {math.degrees(d):+.1f} deg).')

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
