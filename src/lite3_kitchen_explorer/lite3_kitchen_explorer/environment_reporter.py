"""Environment reporter: the tile-by-tile maze exploration manager, and the author of the tile report.

Start-up: before any exploring, the node builds a 1 m x 1 m tile grid. It proposes initial-tile
placements around the robot (snapped to the nearby walls, centred on the robot, ...) and takes the
first, wall-snapped one (tile_ai_choice: true lets the AI client pick instead). The grid is anchored
to that tile, and the robot's starting tile counts as reached. From then on every tile is classified
(visited / to_visit / blocked / unknown) and the robot's path is kept as a trail of visited tiles.

A tile is REACHED when the robot gets within tile_reach_radius_m of its centre (or Nav2 reports the
tile's goal reached there). Each time a tile is reached:
  - the AI client is asked, in the background, what the tile looks like (camera frame + lidar ranges
    + what the map says about the tile's sides). The frame is taken at once, and nothing waits for the answer;
  - the answer fills in that tile's entry in the tile report, one markdown file for the whole run
    (reports_dir, default ~/2d/reports/tile_report_<run>.md), rewritten after every change.

Each time the frontier explorer pauses at a reached goal (and at the start), this node keeps it paused, makes
one quick turn in place (turn_deg, default 360) so the lidar maps all around the tile, and then decides
where to go next: the unvisited tiles Nav2 can reach are ranked by path length and the nearest wins
(2-3 waypoints along the Nav2 path, filtered by the heading rule); frontiers are the fallback when no
tile is connected. The choice goes to the explorer on /frontier_explorer/preferred_goal (+ preferred_route).
With tile_ai_choice / tile_target_ai_choice / ai_frontier_choice set to true, the AI client makes those
choices from a map image instead (off by default: the AI's job is the tile report).

The AI is NOT part of this package. It lives in the lite3_llm_tools package (node llm_client), which this
node reaches through JSON messages on /llm/request, /llm/response and /llm/status (llm_link.py). With
use_ai:=false, or when llm_client is not running, the tile report still lists every tile reached, without
descriptions.

Interfaces
  sub  /frontier_reached            std_msgs/Bool
  sub  /exploration_complete        std_msgs/Bool (latched)
  sub  /map                         nav_msgs/OccupancyGrid
  sub  /frontier_candidates         std_msgs/String (from the explorer)
  pub  /frontier_explorer/preferred_goal  geometry_msgs/PointStamped (the choice)
  pub  /frontier_explorer/preferred_route geometry_msgs/PoseArray (2-3 waypoints + final pose, each with yaw;
                                          sent just before preferred_goal, driven without stopping)
  pub  /tile_map                    std_msgs/String  JSON: grid, trail, every tile's status and sides
  pub  /tile_report                 std_msgs/String  JSON: one finished tile report (latched)
  pub  ~/tile_markers               visualization_msgs/MarkerArray  tiles + trail + next target (orange) for RViz
  pub  ~/markers                    visualization_msgs/MarkerArray  what each tile looks like, labelled at its centre
  pub  ~/next_target                std_msgs/String  the tile being driven to
  pub  /spin_crawl                  std_msgs/Bool    crawl gait while turning (gait_switcher)
  act  spin                         nav2_msgs/Spin (client) the scan turn
  pub  ~/state                      std_msgs/String  WAITING_START / INIT_TILE / EXPLORING / DECIDING:<phase> / COMPLETE
  srv  ~/report_now                 std_srvs/Trigger  describe the tile the robot is in right now (manual)
  AI   /llm/request, /llm/response, /llm/status   see llm_link.py
Files: reports_dir/tile_report_<run>.md; output_dir/run_<time>/: stops.jsonl (one decision per stop), tiles.json,
       and the AI client's camera frames, answers and map images.
"""
import datetime as dt
import json
import math
import os
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Pose, PoseArray, PointStamped
from nav_msgs.msg import OccupancyGrid
from lifecycle_msgs.msg import State as LifecycleState
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import ComputePathToPose, Spin
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_kitchen_explorer import frontier_core as fc
from lite3_kitchen_explorer.llm_link import LlmLink
from lite3_kitchen_explorer.tile_reports import TileReportBook, within_reach
from lite3_kitchen_explorer import turn_check as TC
from lite3_kitchen_explorer import approach as A
from lite3_kitchen_explorer import tiles as T


class EnvironmentReporter(Node):
    WAITING_START, INIT_TILE, EXPLORING, DECIDING, COMPLETE = (
        'WAITING_START', 'INIT_TILE', 'EXPLORING', 'DECIDING', 'COMPLETE')

    def __init__(self):
        super().__init__('environment_reporter')
        p = self.declare_parameter
        self.report_at_start = p('report_at_start', True).value
        # One quick turn in place at every stop (incl. the start tile) so the lidar sees all around and the
        # neighbouring tiles form on the map before the next tile is chosen. No AI query waits for it.
        self.turn_at_stop = p('turn_at_stop', True).value
        self.turn_deg = float(p('turn_deg', 360.0).value)
        self.turn_time_allowance_s = p('turn_time_allowance_s', 20.0).value
        self.turn_use_crawl = p('turn_use_crawl', True).value     # crawl gait for the turn (gait_switcher, /spin_crawl)
        self.crawl_settle_s = p('crawl_settle_s', 1.0).value
        # Turn only where Nav2's Spin can finish: the body is 0.61 x 0.37 m and a 1 m corridor is usually too narrow, and
        # an aborted spin leaves the robot rotated across the corridor (route_debug_080014). Checked on the local costmap.
        self.turn_check_room = p('turn_check_room', True).value
        self.robot_length_m = p('robot_length_m', 0.61).value
        self.robot_width_m = p('robot_width_m', 0.37).value
        self.turn_room_margin_m = p('turn_room_margin_m', 0.0).value
        self.local_costmap_topic = p('local_costmap_topic', '/local_costmap/costmap').value
        self.start_wait_s = p('start_wait_s', 60.0).value
        self.global_frame = p('global_frame', 'map').value
        self.robot_frame = p('robot_frame', 'base_link').value
        self.explorer_service = p('explorer_service', '/frontier_explorer/set_active').value
        self.output_dir = os.path.expanduser(p('output_dir', '~/lite3_survey_runs').value)
        # Master switch: false = never use the AI (no tile descriptions, no AI choices). Tiles are placed by
        # geometry and the next target is the nearest Nav2-connected tile either way.
        self.use_ai = bool(p('use_ai', True).value)
        self.llm_timeout_s = p('llm_timeout_s', 120.0).value             # one map / tile choice request
        self.llm_report_timeout_s = p('llm_report_timeout_s', 180.0).value  # one tile description
        self.reports_dir = os.path.expanduser(p('reports_dir', '~/2d/reports').value)
        self.tile_reach_radius_m = p('tile_reach_radius_m', 0.30).value  # a tile is reached this close to its centre
        self.ai_frontier_choice = p('ai_frontier_choice', False).value   # AI picks frontiers from a map image
        self.min_wall_clearance_m = p('min_wall_clearance_m', 0.35).value
        self.choice_min_candidates = p('choice_min_candidates', 2).value
        self.candidates_max_age_s = p('candidates_max_age_s', 3.0).value
        self.tiles_enabled = p('tiles_enabled', True).value
        self.tile_size_m = p('tile_size_m', 1.0).value
        self.tile_wait_s = p('tile_wait_s', 30.0).value
        self.tile_ai_choice = p('tile_ai_choice', False).value            # AI places the initial tile
        self.tile_targets_first = p('tile_targets_first', True).value     # go tile by tile
        self.tile_prompt_choice = p('tile_target_ai_choice', False).value # AI picks among nearest tiles
        self.tile_target_limit = p('tile_target_limit', 6).value
        if not self.use_ai:
            self.tile_ai_choice = False
            self.tile_prompt_choice = False
        self.path_check_timeout_s = p('path_check_timeout_s', 10.0).value
        self.target_min_clearance_m = p('target_min_clearance_m', 0.30).value

        self.run_dir = os.path.join(self.output_dir, 'run_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.run_dir, exist_ok=True)
        self.llm = LlmLink(self, enabled=self.use_ai, default_timeout_s=self.llm_timeout_s)
        run_stamp = os.path.basename(self.run_dir)[len('run_'):]
        self.book = TileReportBook(os.path.join(self.reports_dir, f'tile_report_{run_stamp}.md'), self.run_dir,
                                   self.use_ai, self.tile_reach_radius_m)
        self.report_args = {}         # tile label -> describe_tile request fields (for retries)
        self.report_tries = {}
        self.report_max_tries = 5
        self.report_calls = {}        # tile label -> (LlmCall, image name): descriptions being written in the background
        self.get_logger().info(f'AI {"on (via llm_client)" if self.use_ai else "off"}; run files in {self.run_dir}; '
                               f'tile report: {self.book.path}')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.spin_client = ActionClient(self, Spin, 'spin')
        # The spin action server exists once behavior_server is configured, but it rejects goals until it is ACTIVE.
        self.behavior_state_client = self.create_client(GetState, '/behavior_server/get_state')
        self.behavior_active = False
        self.behavior_state_future = None
        self.path_client = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.crawl_pub = self.create_publisher(Bool, '/spin_crawl', 10)   # gait_switcher: crawl while turning
        self.flat_pub = self.create_publisher(Bool, '/spin_flat', 10)     # gait_switcher: flat_medium for the opening spin
        self.opening, self.spin_retries = False, 0
        self.opening_retries = int(p('opening_spin_retries', 2).value)   # retries of the aborted opening sweep
        self.start_sweep_deg = p('start_sweep_deg', 90.0).value          # opening scan covers -this .. +this
        self.spin_queue = []         # spins still to run after the current one
        self.pending_spin = None     # the turn waiting for the crawl gait to settle
        self.action_handle = None
        self.action_done = None
        self.action_seq = 0
        self.explorer_client = self.create_client(SetBool, self.explorer_service)

        self.create_subscription(Bool, '/frontier_reached', self._on_frontier_reached, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/exploration_complete', self._on_exploration_complete, latched)
        self.tile_report_pub = self.create_publisher(String, '/tile_report', QoSProfile(
            depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.state_pub = self.create_publisher(String, '~/state', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', latched)
        self.create_service(Trigger, '~/report_now', self._on_report_now)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)
        self.create_subscription(String, '/frontier_candidates', self._on_candidates, latched)
        self.create_subscription(String, '/frontier_explorer/goal_failed', self._on_goal_failed, 10)
        self.local_costmap = None     # (grid, (origin_x, origin_y), resolution, frame_id, received monotonic time)
        self.create_subscription(OccupancyGrid, self.local_costmap_topic, self._on_local_costmap, latched)
        self.failed_tiles = {}        # tile label -> why the explorer gave up on it; not offered again
        self.preferred_pub = self.create_publisher(PointStamped, '/frontier_explorer/preferred_goal', 10)
        self.route_pub = self.create_publisher(PoseArray, '/frontier_explorer/preferred_route', 10)
        self.intent_pub = self.create_publisher(String, '~/next_target', QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.grid = None
        self.grid_info = None
        self.candidates = None
        self.candidates_at = 0.0
        self.tile_grid = None
        self.tile_init = None        # how the initial tile was chosen
        self.next_target = None      # (tile, goal_xy) the robot is driving to; highlighted in RViz
        self.path_checks = None      # Nav2 path checks for candidate tiles, while in 'checking_paths'
        self.trail = []              # visited tiles in visit order (consecutive repeats removed)
        self.trail_log = []          # [{tile, time, pose}]
        self.tiles = {}
        self.map_version = 0
        self.tiles_version = -1
        self.tile_pub = self.create_publisher(String, '/tile_map', latched)
        self.tile_marker_pub = self.create_publisher(MarkerArray, '~/tile_markers', latched)
        self.create_timer(0.5, self._track_tiles)
        self.create_timer(1.0, self._update_tiles)

        self.pending = None           # (LlmCall, capture): the single map query in flight (tile / target choice)
        self.query_count = 0
        self.reports = []
        self.markers = []
        self.complete_pending = False
        self.state = None
        self.phase = ''
        self.phase_since = 0.0
        self.started_at = time.monotonic()
        self.stop = None              # the stop being decided
        if self.report_at_start or self.tiles_enabled:
            self._set_state(self.WAITING_START)
        else:
            self._set_state(self.EXPLORING)
            self._explorer_active(True)
        self.create_timer(0.2, self._tick)

    # ------------------------------------------------------------------ inputs
    def _on_map(self, msg):
        i = msg.info
        self.grid_info = fc.GridInfo(i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y)
        self.grid = fc.grid_from_msg_data(msg.data, i.width, i.height)
        self.map_version += 1

    def _on_local_costmap(self, msg):
        i = msg.info
        self.local_costmap = (fc.grid_from_msg_data(msg.data, i.width, i.height), (i.origin.position.x, i.origin.position.y),
                              i.resolution, msg.header.frame_id, time.monotonic())

    def _on_goal_failed(self, msg):
        """The explorer gave up on a tile goal (stuck, rejected, timed out): choose another tile, not a frontier."""
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        goal, reason = data.get('goal'), data.get('reason', 'failed')
        if self.next_target is not None and goal and math.hypot(goal[0] - self.next_target[1][0],
                                                                 goal[1] - self.next_target[1][1]) <= 0.2:
            label = T.tile_label(self.next_target[0])
            self.failed_tiles[label] = reason
            self.get_logger().warn(f'Tile {label} could not be reached ({reason}); choosing another tile.')
            self.next_target = None
            self._publish_tile_markers()
        if self.state == self.EXPLORING:
            self._begin_stop('goal failed', turn=False)

    def _on_candidates(self, msg):
        try:
            self.candidates = json.loads(msg.data)
            self.candidates_at = time.monotonic()
        except ValueError:
            pass

    def _on_frontier_reached(self, msg):
        if not msg.data:
            return
        self._mark_goal_reached()          # Nav2 says the goal was reached: that tile counts if the robot is at it
        if self.state == self.EXPLORING:
            self._begin_stop('goal reached')

    def _on_exploration_complete(self, msg):
        if not msg.data:
            return
        if self.state == self.DECIDING:
            self.complete_pending = True      # finish this stop first
        elif self.state in (self.EXPLORING, self.WAITING_START):
            self._finish()

    def _on_report_now(self, request, response):
        pose = self._robot_pose()
        if self.tile_grid is None or pose is None:
            response.success, response.message = False, 'no tile grid or robot pose yet'
        else:
            tile = self.tile_grid.tile_of(pose[0], pose[1])
            self._start_tile_report(tile, pose, 'manual')
            response.success, response.message = True, f'describing tile {T.tile_label(tile)}'
        return response

    def _robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.global_frame, self.robot_frame, Time(),
                                                timeout=Duration(seconds=0.1))
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return [round(t.transform.translation.x, 3), round(t.transform.translation.y, 3), round(yaw, 4)]

    # -------------------------------------------------------------- main loop
    def _poll_behavior_server(self):
        if self.behavior_active:
            return
        f = self.behavior_state_future
        if f is not None:
            if not f.done():
                return
            res = f.result()
            self.behavior_state_future = None
            if res is not None and res.current_state.id == LifecycleState.PRIMARY_STATE_ACTIVE:
                self.behavior_active = True
                self.get_logger().info('Nav2 behavior_server is active.')
                return
        if self.behavior_state_client.service_is_ready():
            self.behavior_state_future = self.behavior_state_client.call_async(GetState.Request())

    def _tick(self):
        self._poll_behavior_server()
        self.llm.poll()
        self._poll_pending()
        self._poll_tile_reports()
        now = time.monotonic()
        if self.state == self.WAITING_START:
            ready = self._robot_pose() is not None and self.explorer_client.service_is_ready() \
                and (not self.turn_at_stop or (self.spin_client.server_is_ready() and self.behavior_active)) and (not self.use_ai or self.llm.ready)
            if ready or now - self.started_at > self.start_wait_s:
                if not ready:
                    self.get_logger().warn('TF / Nav2 spin / explorer / AI client not all ready; starting anyway '
                                           '(steps that need a missing one are skipped).')
                if self.tiles_enabled:
                    self._explorer_active(False)
                    self._set_state(self.INIT_TILE, 'map')
                else:
                    self._after_start_setup()
        elif self.state == self.INIT_TILE and self.phase == 'map':
            pose = self._robot_pose()
            if pose is not None and self.grid is not None and self._local_map_ready(pose):
                self._begin_tile_init(pose)
            elif now - self.phase_since > self.tile_wait_s:
                self.get_logger().warn('Local map still sparse; creating the initial tile with what is mapped.')
                self._begin_tile_init(pose or [0.0, 0.0, 0.0])
        elif self.state == self.DECIDING:
            self._tick_deciding(now)

    def _after_start_setup(self):
        if self.report_at_start:
            self._begin_stop('start')
        else:
            self._set_state(self.EXPLORING)
            self._explorer_active(True)

    # ------------------------------------------------------------ tile system
    def _local_map_ready(self, pose, radius_m=1.5, min_known=0.5):
        r, c = self.grid_info.world_to_cell(pose[0], pose[1])
        rad = int(radius_m / self.grid_info.resolution)
        win = self.grid[max(0, r - rad):r + rad + 1, max(0, c - rad):c + rad + 1]
        return win.size > 0 and (win >= 0).mean() >= min_known

    def _ai_available(self):
        return self.use_ai and self.llm.ready

    def _tiles_state(self):
        """The tile grid, tile statuses and trail as the AI client needs them to draw the map."""
        return {'tile_grid': self.tile_grid.as_dict() if self.tile_grid is not None else None,
                'tiles': [{'tile': list(t), 'status': d['status']} for t, d in self.tiles.items()],
                'trail': [list(t) for t in self.trail]}

    def _begin_tile_init(self, pose):
        if self.grid is None:
            self._establish_tiles(None, 'no map: grid centred on the robot along the map axes', pose)
            return
        cands = T.initial_candidates(self.grid, self.grid_info, pose, size=self.tile_size_m)
        if len(cands) == 1 or not self.tile_ai_choice or not self._ai_available():
            how = 'only one placement' if len(cands) == 1 else (
                'AI choice disabled' if not self.tile_ai_choice else 'AI not available')
            self._establish_tiles(cands[0], how, pose)
            return
        call = self.llm.request(
            'place_initial_tile', name='initial_tile', save_dir=self.run_dir, pose=pose, size_m=self.tile_size_m,
            wall_band_m=self.min_wall_clearance_m,
            candidates=[{'id': c['id'], 'kind': c['kind'], 'sides': c['sides'], 'grid': c['grid'].as_dict()}
                        for c in cands])
        self.pending = (call, {'kind': 'tile', 'candidates': cands, 'pose': pose})
        self.get_logger().info(f'Asking the AI to place the initial {self.tile_size_m:g} m tile '
                               f'({len(cands)} options, see initial_tile.png).')
        self._set_phase('choosing')

    def _on_tile_choice(self, call, capture):
        cands = {c['id']: c for c in capture['candidates']}
        first = capture['candidates'][0]
        if not call.ok:
            self.get_logger().error(f'Initial-tile query failed: {call.failure()}; using tile {first["id"]}.')
            self._establish_tiles(first, f'AI query failed ({call.failure()}); default placement', capture['pose'])
            return
        res = call.result
        answer, ai_sides, choice = res.get('answer'), res.get('ai_sides') or {}, res.get('choice')
        if choice not in cands:
            self.get_logger().warn(f'AI gave no valid tile ("{(answer or {}).get("scene_description")}"); '
                                   f'using tile {first["id"]}.')
            self._establish_tiles(first, 'AI answer had no valid tile; default placement', capture['pose'],
                                  ai_sides, answer)
        else:
            self._establish_tiles(cands[choice], 'chosen by the AI', capture['pose'], ai_sides, answer)

    def _establish_tiles(self, cand, how, pose, ai_sides=None, answer=None):
        if cand is None:
            half = self.tile_size_m / 2.0
            grid = T.TileGrid(pose[0] - half, pose[1] - half, 0.0, self.tile_size_m)
            cand = {'id': '-', 'kind': 'centred on the robot, along the map axes', 'grid': grid, 'sides': {}}
        self.tile_grid = cand['grid']
        self.tile_init = {'placement': cand['id'], 'kind': cand['kind'], 'how': how,
                          'grid': self.tile_grid.as_dict(), 'map_sides': cand.get('sides', {}),
                          'ai_sides': ai_sides or {}, 'ai_answer': answer}
        start = self.tile_grid.tile_of(pose[0], pose[1])
        self.trail = [start]
        self.trail_log = [{'tile': list(start), 'time': dt.datetime.now().isoformat(timespec='seconds'),
                           'pose': pose}]
        self.tiles_version = -1
        self._update_tiles()
        self.get_logger().info(
            f'Initial tile {cand["id"]} ({cand["kind"]}, {how}); grid yaw {math.degrees(self.tile_grid.yaw):.1f} deg. '
            f'Sides from map: {cand.get("sides")}, from AI: {ai_sides or "-"}. Starting exploration.')
        self._start_tile_report(start, pose, 'start')     # the tile the robot starts in counts as reached
        self._after_start_setup()

    def _track_tiles(self):
        """A tile is reached when the robot is within tile_reach_radius_m of its centre."""
        if self.tile_grid is None:
            return
        pose = self._robot_pose()
        if pose is None:
            return
        tile = self.tile_grid.tile_of(pose[0], pose[1])
        if within_reach(pose, self.tile_grid.center(tile), self.tile_reach_radius_m):
            self._mark_reached(tile, pose, 'centre reached')

    def _mark_goal_reached(self):
        """Nav2 reported the goal reached: count the target tile if the robot really is at its goal point.

        The goal point is the free cell nearest the tile centre that keeps clear of walls, so beside a wall it can
        be a little off-centre; this keeps such a tile from being driven to forever.
        """
        if self.tile_grid is None or self.next_target is None:
            return
        tile, goal = self.next_target
        pose = self._robot_pose()
        if within_reach(pose, goal, 1.5 * self.tile_reach_radius_m):
            self._mark_reached(tile, pose, 'goal reached')

    def _mark_reached(self, tile, pose, source):
        if self.trail and self.trail[-1] == tile:
            return
        new = tile not in set(self.trail)
        self.trail.append(tile)
        self.trail_log.append({'tile': list(tile), 'time': dt.datetime.now().isoformat(timespec='seconds'),
                               'pose': pose})
        self.tiles_version = -1
        self.get_logger().info(f'Reached tile {T.tile_label(tile)} ({source})' + (' (new)' if new else ' (revisit)'))
        if new:
            self._start_tile_report(tile, pose, source)

    # ------------------------------------------------------------ tile report
    def _tile_sides(self, tile):
        if self.grid is None or self.tile_grid is None:
            return {}
        return T.tile_sides(self.grid, self.grid_info, self.tile_grid, tile)

    def _start_tile_report(self, tile, pose, source):
        """Open the tile's entry and, if the AI is available, ask it in the background what the tile looks like."""
        label = T.tile_label(tile)
        sides = self._tile_sides(tile)
        self.book.reached(label, pose, source, sides)
        self.report_calls.pop(label, None)
        if not self._ai_available():
            self.book.set_no_ai(label, 'AI disabled (use_ai:=false)' if not self.use_ai else 'llm_client is not running')
            self._write_book()
            self._publish_tile_report(label)
            return
        self.query_count += 1
        name = f'tile_{tile[0]}_{tile[1]}_q{self.query_count:04d}'
        walls = ', '.join(f'{ {"N": "north", "E": "east", "S": "south", "W": "west"}[k]} {v}' for k, v in sides.items())
        context = (f'TILE CONTEXT: the robot has just reached the centre of tile {label} of its '
                   f'{self.tile_size_m:g} m x {self.tile_size_m:g} m tile grid (position ({pose[0]:.2f}, {pose[1]:.2f}) '
                   f'in the map frame).' + (f' The lidar map says: {walls}.' if walls else '')) if pose else \
            f'TILE CONTEXT: the robot has just reached tile {label}.'
        self.report_args[label] = dict(timeout_s=self.llm_report_timeout_s, name=name, tile=label,
                                       pose=pose, context=context, save_dir=self.run_dir)
        self.report_tries[label] = 1
        call = self.llm.request('describe_tile', **self.report_args[label])
        self.report_calls[label] = (call, name)
        self.get_logger().info(f'Tile {label} reached ({source}): asking the AI what it looks like (in the background).')
        self._write_book()

    def _poll_tile_reports(self):
        for label, (call, name) in list(self.report_calls.items()):
            if not call.done:
                continue
            del self.report_calls[label]
            if call.ok:
                self.book.set_result(label, call.result)
                self.get_logger().info(f'Tile {label}: {call.result.get("scene_description")} '
                                       f'(conf {float(call.result.get("confidence", 0.0)):.2f})')
            elif self.report_tries.get(label, 1) < self.report_max_tries and self._ai_available():
                # No description yet: ask again; the report only lists the tile once one has arrived.
                self.report_tries[label] = self.report_tries.get(label, 1) + 1
                self.get_logger().warn(f'Tile {label}: the AI description failed ({call.failure()}); '
                                       f'retrying ({self.report_tries[label]}/{self.report_max_tries})')
                retry = self.llm.request('describe_tile', **self.report_args[label])
                self.report_calls[label] = (retry, name)
                continue
            else:
                self.book.set_error(label, call.failure())
                self.get_logger().warn(f'Tile {label}: the AI description failed ({call.failure()})')
            self._write_book()
            self._publish_tile_report(label)

    def _write_book(self, finished=False):
        counts = T.tile_counts(self.tiles) if self.tiles else None
        to_visit = [T.tile_label(t) for t, d in sorted(self.tiles.items()) if d['status'] == 'to_visit']
        try:
            self.book.write(grid=self.tile_grid.as_dict() if self.tile_grid is not None else None,
                            tile_counts=counts, to_visit=to_visit, finished=finished)
        except OSError as exc:
            self.get_logger().error(f'Cannot write the tile report {self.book.path}: {exc}', throttle_duration_sec=30.0)

    def _publish_tile_report(self, label):
        entry = self.book.entries[label]
        self.tile_report_pub.publish(String(data=json.dumps(entry, ensure_ascii=False, default=str)))
        if self.tile_grid is None:
            return
        tile = tuple(int(v) for v in label.split(','))
        cx, cy = self.tile_grid.center(tile)
        m = Marker()
        m.header.frame_id = self.global_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'tile_report', entry['order'], Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.35
        m.pose.orientation.w = 1.0
        m.scale.z = 0.16
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.8, 0.1, 1.0) if entry['status'] == 'done' else (0.8, 0.8, 0.8, 1.0)
        text = entry.get('description') if entry['status'] == 'done' else f'({entry["status"]})'
        m.text = f'{label}: {text[:60]}'
        self.markers = [x for x in self.markers if x.id != m.id] + [m]
        self.marker_pub.publish(MarkerArray(markers=self.markers))

    def _environment_hint(self):
        """The newest tile description, as context for the AI's map choices."""
        done = [e for e in self.book.entries.values() if e['status'] == 'done']
        return done[-1]['description'] if done else 'unknown'

    def _update_tiles(self):
        if self.tile_grid is None or self.grid is None:
            return
        if self.tiles_version == self.map_version:
            return
        self.tiles_version = self.map_version
        self.tiles = T.classify_tiles(self.grid, self.grid_info, self.tile_grid, self.trail)
        for tile, t in self.tiles.items():
            if t['status'] != 'unknown':
                t['sides'] = T.tile_sides(self.grid, self.grid_info, self.tile_grid, tile)
        data = self.tile_map_dict()
        self.tile_pub.publish(String(data=json.dumps(data)))
        self._publish_tile_markers()

    def tile_map_dict(self):
        return {'frame': self.global_frame, 'grid': self.tile_grid.as_dict(), 'initial_tile': self.tile_init,
                'current_tile': T.tile_label(self.trail[-1]) if self.trail else None,
                'counts': T.tile_counts(self.tiles),
                'trail': [T.tile_label(t) for t in self.trail],
                'tiles': {T.tile_label(k): v for k, v in sorted(self.tiles.items())}}

    def _publish_tile_markers(self):
        tg = self.tile_grid
        stamp = self.get_clock().now().to_msg()
        colours = {'visited': (0.2, 0.5, 1.0, 0.45), 'to_visit': (0.1, 0.8, 0.2, 0.35),
                   'blocked': (0.8, 0.1, 0.1, 0.25)}
        markers = []
        for n, (status, (r, g, b, a)) in enumerate(colours.items()):
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = 'tiles', n, Marker.CUBE_LIST, Marker.ADD
            m.pose.position.x, m.pose.position.y = tg.origin_x, tg.origin_y
            m.pose.orientation.z, m.pose.orientation.w = math.sin(tg.yaw / 2), math.cos(tg.yaw / 2)
            m.scale.x = m.scale.y = 0.94 * tg.size
            m.scale.z = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            for (i, j), t in self.tiles.items():
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
        for i, j in self.trail:
            p = type(trail.pose.position)()
            p.x, p.y, p.z = (i + 0.5) * tg.size, (j + 0.5) * tg.size, 0.05
            trail.points.append(p)
        if len(trail.points) < 2:
            trail.action = Marker.DELETE
        markers.append(trail)
        markers.extend(self._next_target_markers(stamp))
        self.tile_marker_pub.publish(MarkerArray(markers=markers))

    def _next_target_markers(self, stamp):
        """Orange highlight on the tile being driven to, its goal point and a label.

        Shown until the tile is visited. The same ids are always published, with DELETE
        when there is no target, so RViz removes the old highlight.
        """
        tg = self.tile_grid
        shown = None
        if self.next_target is not None:
            tile, goal = self.next_target
            if self.tiles.get(tile, {}).get('status') != 'visited':
                shown = (tile, goal)
        out = []
        for n, ns, mtype in ((0, 'next_tile_cell', Marker.CUBE),
                             (1, 'next_tile_goal', Marker.SPHERE),
                             (2, 'next_tile_label', Marker.TEXT_VIEW_FACING)):
            m = Marker()
            m.header.frame_id = self.global_frame
            m.header.stamp = stamp
            m.ns, m.id, m.type = ns, n, mtype
            m.action = Marker.ADD if shown else Marker.DELETE
            m.pose.orientation.z, m.pose.orientation.w = math.sin(tg.yaw / 2), math.cos(tg.yaw / 2)
            if shown:
                tile, goal = shown
                cx, cy = tg.center(tile)
                if n == 0:
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.03
                    m.scale.x = m.scale.y = 0.94 * tg.size
                    m.scale.z = 0.06
                    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 0.6
                elif n == 1:
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = float(goal[0]), float(goal[1]), 0.1
                    m.scale.x = m.scale.y = m.scale.z = 0.25
                    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
                else:
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = cx, cy, 0.5
                    m.text = f'NEXT {T.tile_label(tile)}'
                    m.scale.z = 0.3
                    m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
            out.append(m)
        return out

    def _begin_stop(self, reason, turn=True):
        """The explorer reached its goal: keep it paused, turn once in place so the lidar maps around the tile, then decide.

        Nothing here waits for the AI: the tile description was requested when the tile was reached and arrives in
        the background.
        """
        self._explorer_active(False)
        self.stop = {'stop': len(self.reports) + 1, 'reason': reason,
                     'wall_time': dt.datetime.now().isoformat(timespec='seconds'), 'pose': self._robot_pose()}
        self._set_state(self.DECIDING)
        if turn and self._start_turn(initial=(reason == 'start')):
            self.get_logger().info(f'Stop {self.stop["stop"]} ({reason}): turning {self.turn_deg:.0f} deg to scan, '
                                   'then choosing the next tile.')
        else:
            self.get_logger().info(f'Stop {self.stop["stop"]} ({reason}): choosing the next tile.')
            self._finish_stop()

    def _start_turn(self, initial=False):
        """Begin the quick scan turn. False (nothing started) if turning is off, the scan is done, or Nav2's spin is not up."""
        if not self.turn_at_stop or self.complete_pending or abs(self.turn_deg) < 1.0:
            self.get_logger().info(f'Scan turn skipped: turn_at_stop={self.turn_at_stop}, '
                                   f'complete_pending={self.complete_pending}, turn_deg={self.turn_deg}.')
            return False
        if not self.spin_client.server_is_ready():
            self.get_logger().warn('Nav2 spin action not available: skipping the scan turn.')
            return False
        # The opening spin ignores the room check (the costmap is still sparse / inflated at the start) and runs in flat_medium.
        if not initial and not self._turn_has_room():
            return False
        make_goal = self._spin_goal
        self.opening, self.spin_retries = bool(initial), 0
        goal = make_goal(self.turn_deg)
        if initial:
            # Opening sweep from -start_sweep_deg to +start_sweep_deg (Nav2's Spin is relative): turn right, then left.
            half = abs(self.start_sweep_deg)
            goal = make_goal(-half)
            self.spin_queue = [make_goal(2 * half)]
            self.flat_pub.publish(Bool(data=True))        # flat_slow gait (held by gait_switcher) for the opening spin, let it settle
            self.get_logger().info(f'Opening scan: spin {-half:.0f} then {half:.0f} deg after {self.crawl_settle_s:g} s.')
            self.pending_spin = goal
            self._set_phase('crawl_wait')
        elif self.turn_use_crawl:
            self.crawl_pub.publish(Bool(data=True))       # hold the crawl gait, let it settle, then spin
            self.pending_spin = goal
            self._set_phase('crawl_wait')
        else:
            self._send_action(self.spin_client, goal)
            self._set_phase('turning')
        return True

    def _spin_goal(self, deg):
        g = Spin.Goal()
        g.target_yaw = math.radians(deg)
        g.time_allowance = DurationMsg(sec=int(self.turn_time_allowance_s))
        return g

    def _turn_has_room(self):
        """False if the body would touch the local costmap's inscribed / lethal cells while turning (Nav2's Spin would abort)."""
        if not self.turn_check_room or self.local_costmap is None:
            return True
        grid, origin, res, frame, received = self.local_costmap
        if time.monotonic() - received > 10.0:
            return True                          # no fresh costmap to judge by: let Nav2 decide
        try:
            t = self.tf_buffer.lookup_transform(frame, self.robot_frame, Time(), timeout=Duration(seconds=0.1))
        except TransformException:
            return True
        ok, blocked = TC.can_turn_in_place(grid, origin, res, t.transform.translation.x, t.transform.translation.y,
                                           self.robot_length_m, self.robot_width_m, self.turn_room_margin_m)
        if not ok:
            self.get_logger().info(f'No room to turn here ({blocked} of 72 body orientations touch the costmap\'s obstacle '
                                   'zone; Nav2 would abort the spin): skipping the scan turn.')
        return ok

    def _release_crawl(self):
        self.crawl_pub.publish(Bool(data=False))
        self.flat_pub.publish(Bool(data=False))

    def _tick_deciding(self, now):
        if self.phase == 'crawl_wait':
            if now - self.phase_since >= self.crawl_settle_s and self.pending_spin is not None:
                goal, self.pending_spin = self.pending_spin, None
                self.get_logger().info(f'Sending spin goal {math.degrees(goal.target_yaw):.0f} deg.')
                self._send_action(self.spin_client, goal)
                self._set_phase('turning')
        elif self.phase == 'turning':
            timed_out = now - self.phase_since > self.turn_time_allowance_s + 5.0
            if self.action_done == GoalStatus.STATUS_SUCCEEDED and self.spin_queue:
                self._send_action(self.spin_client, self.spin_queue.pop(0))
                self._set_phase('turning')
                return
            if self.opening and self.spin_retries < self.opening_retries and \
                    (timed_out or (self.action_done is not None and self.action_done != GoalStatus.STATUS_SUCCEEDED)):
                # Aborted (e.g. "Collision Ahead"): the opening scan must finish before navigating. Retry the
                # sweep the other way round, keeping the held gait.
                if self.action_done is None:
                    self.action_seq += 1
                    if self.action_handle is not None:
                        self.action_handle.cancel_goal_async()
                self.spin_retries += 1
                half = abs(self.start_sweep_deg) * (1 if self.spin_retries % 2 else -1)
                self.get_logger().warn(f'Opening spin aborted (status {self.action_done}); retry '
                                       f'{self.spin_retries}/{self.opening_retries} turning {half:+.0f} deg first.')
                self.pending_spin = self._spin_goal(half)
                self.spin_queue = [self._spin_goal(-2 * half)]
                self.action_done = None
                self._set_phase('crawl_wait')
                return
            if self.action_done is not None or timed_out:
                self.spin_queue = []
                if self.action_done is None:
                    self._cancel_action()
                self._release_crawl()
                if self.action_done != GoalStatus.STATUS_SUCCEEDED:
                    self.get_logger().warn('The scan turn did not finish; choosing the next tile with the map so far.')
                self._finish_stop()
        elif self.phase == 'checking_paths':
            pc = self.path_checks
            if pc is None:
                self._set_phase('resume')
            elif len(pc['results']) == len(pc['targets']) or now - self.phase_since > self.path_check_timeout_s:
                if len(pc['results']) < len(pc['targets']):
                    self.get_logger().warn('Some Nav2 path checks timed out; those tiles are not offered.')
                    pc['results'].update({i: None for i in range(len(pc['targets'])) if i not in pc['results']})
                self._finish_path_checks()
        elif self.phase == 'choose_wait':
            fresh = self.candidates is not None and now - self.candidates_at <= self.candidates_max_age_s
            if fresh or now - self.phase_since > 6.0:
                self._ask_frontier_choice()
        elif self.phase == 'resume' and now - self.phase_since >= 0.5:
            self._resume_after_stop()

    # ------------------------------------------------------------------ queries
    def _poll_pending(self):
        """The single map query in flight (initial tile, tile / frontier choice), when the AI makes those choices."""
        if self.pending is None or not self.pending[0].done:
            return
        call, capture = self.pending
        self.pending = None
        if capture.get('kind') == 'frontier':
            self._on_frontier_choice(call, capture)
        elif capture.get('kind') == 'tile':
            self._on_tile_choice(call, capture)

    def _finish_stop(self):
        # The tile pipeline (nearest connected tile) runs whenever tiles are on; the AI only joins in if asked to.
        choose = (self.tiles_enabled and self.tile_targets_first) or (self.use_ai and self.ai_frontier_choice)
        if choose and not self.complete_pending:
            self._set_phase('choose_wait')       # wait for fresh /frontier_candidates
        else:
            self._publish_stop_report()
            self._set_phase('resume')

    # ------------------------------------------------ AI frontier choice
    def _ask_frontier_choice(self):
        stop = self.stop
        data = self.candidates or {}
        cands = data.get('candidates', [])
        if self.grid is None or (self.tile_grid is None and len(cands) < max(1, self.choice_min_candidates)):
            why = 'no map yet' if self.grid is None else f'{len(cands)} candidate(s)'
            stop['next_frontier'] = {'chosen_by': 'explorer', 'note': f'AI not asked ({why})'}
            self._publish_stop_report()
            self._set_phase('resume')
            return
        pose = self._robot_pose() or stop['pose']
        visited = [r['pose'] for r in self.reports if r.get('pose')]
        tile_summary = ''
        if self.tile_grid is not None:
            self.tiles_version = -1
            self._update_tiles()
            visited_set = set(self.trail)
            for c in cands:
                tile = self.tile_grid.tile_of(*c['goal_xy'])
                c['tile'] = T.tile_label(tile)
                c['tile_status'] = 'visited' if tile in visited_set else \
                    self.tiles.get(tile, {}).get('status', 'unknown')
            counts = T.tile_counts(self.tiles)
            tile_summary = (f"current tile {T.tile_label(self.trail[-1])}; {counts['visited']} tiles visited, "
                            f"{counts['to_visit']} tiles still to visit; trail: "
                            + ' -> '.join(T.tile_label(t) for t in self.trail[-15:]))
        ctx = {'pose': pose, 'visited': visited, 'cands': cands, 'tile_summary': tile_summary}
        if self.tile_grid is not None and self.tile_targets_first:
            # A tile counts as connected only if Nav2 finds a path to it. Unvisited tiles are the
            # raw candidates; _finish_path_checks keeps the reachable ones and calls _offer_choice.
            raw = T.nearest_connected_targets(
                self.grid, self.grid_info, self.tile_grid, self.tiles, self.trail[-1], pose[:2],
                limit=4 * self.tile_target_limit, min_clearance_m=self.target_min_clearance_m,
                grid_connected=False)
            skipped = [t['tile'] for t in raw if t['tile'] in self.failed_tiles]
            raw = [t for t in raw if t['tile'] not in self.failed_tiles]
            if skipped:
                self.get_logger().info(f'Not offering tile(s) the robot could not reach: {", ".join(skipped)}.')
            if raw:
                self._start_path_checks(raw, ctx)
                return
            self.get_logger().info('No unvisited tile to check; choosing among frontiers instead.')
        self._offer_choice(ctx, [])

    def _offer_choice(self, ctx, targets):
        """Pick the next target from the Nav2-verified tiles, or fall back to frontiers; the AI client chooses if on."""
        stop = self.stop
        pose, visited, cands, tile_summary = ctx['pose'], ctx['visited'], ctx['cands'], ctx['tile_summary']
        mode = 'frontier'
        self.next_target = None      # replaced below if a tile is chosen
        if self.tile_grid is not None:
            self._publish_tile_markers()
        ai = self._ai_available()
        frontier_ai = ai and self.ai_frontier_choice
        if self.tile_grid is not None and self.tile_targets_first:
            if targets and (not self.tile_prompt_choice or not ai):
                why = 'AI disabled' if not self.use_ai else ('AI not available' if not ai else 'AI choice disabled')
                self._choose_target(targets[0], f'nearest tile ({why})')
                self._publish_stop_report()
                self._set_phase('resume')
                return
            if targets:
                mode, cands = 'tile', targets
            else:
                self.get_logger().info('No unvisited tile has a Nav2 path; choosing among frontiers instead.')
        if mode == 'frontier' and not frontier_ai:
            note = ('AI disabled (use_ai:=false)' if not self.use_ai else
                    'AI not available (llm_client is not running)' if not ai else 'explorer ranking (ai_frontier_choice is off)')
            stop['next_frontier'] = {'chosen_by': 'explorer', 'note': note}
            self._publish_stop_report()
            self._set_phase('resume')
            return
        if mode == 'frontier' and len(cands) < max(1, self.choice_min_candidates):
            stop['next_frontier'] = {'chosen_by': 'explorer', 'note': f'AI not asked ({len(cands)} frontier(s))'}
            self._publish_stop_report()
            self._set_phase('resume')
            return
        name = f'stop{stop["stop"]:03d}_map'
        self._map_args = (pose, visited)         # for the route overlay drawn once a target is chosen
        call = self.llm.request(
            'choose_target', mode=mode, name=name, save_dir=self.run_dir, pose=pose, visited=visited,
            candidates=cands, environment=self._environment_hint(), stops_so_far=len(self.reports) + 1,
            tile_summary=tile_summary, min_clearance_m=self.target_min_clearance_m,
            wall_band_m=self.min_wall_clearance_m, **self._tiles_state())
        self.pending = (call, {'kind': 'frontier', 'mode': mode, 'name': name, 'candidates': cands})
        what = 'connected unvisited tiles' if mode == 'tile' else 'frontiers'
        self.get_logger().info(f'Asking the AI to choose among {len(cands)} {what} (map: {name}.png).')
        self._set_phase('choosing')

    # ------------------------------------------------- Nav2 path check for tiles
    def _start_path_checks(self, raw, ctx):
        """Ask Nav2 for a path from the robot to each candidate tile's goal point.

        Answers arrive asynchronously; _tick_deciding finishes the check once all have
        answered or the timeout passes.
        """
        self.path_checks = {'targets': raw, 'results': {}, 'ctx': ctx}
        if not self.path_client.server_is_ready():
            self.get_logger().warn('Nav2 planner (compute_path_to_pose) not available; no tile is connected.')
            self.path_checks['results'] = {i: None for i in range(len(raw))}
            self._finish_path_checks()
            return
        for idx, t in enumerate(raw):
            goal = ComputePathToPose.Goal()
            goal.goal.header.frame_id = self.global_frame
            goal.goal.header.stamp = self.get_clock().now().to_msg()
            goal.goal.pose.position.x, goal.goal.pose.position.y = float(t['goal_xy'][0]), float(t['goal_xy'][1])
            goal.goal.pose.orientation.w = 1.0
            goal.use_start = False          # plan from the robot's current pose
            self.path_client.send_goal_async(goal).add_done_callback(
                lambda fut, i=idx: self._on_path_goal_response(i, fut))
        self._set_phase('checking_paths')

    def _on_path_goal_response(self, idx, future):
        if self.path_checks is None:
            return
        handle = future.result()
        if handle is None or not handle.accepted:
            self.path_checks['results'][idx] = None
            return
        handle.get_result_async().add_done_callback(
            lambda fut: self._record_path(idx, fut.result()))

    def _record_path(self, idx, wrapped):
        if self.path_checks is None:
            return
        path = None
        if wrapped is not None and wrapped.status == GoalStatus.STATUS_SUCCEEDED and wrapped.result.path.poses:
            path = wrapped.result.path
        self.path_checks['results'][idx] = path

    def _finish_path_checks(self):
        pc = self.path_checks
        self.path_checks = None
        reachable = []
        for idx, t in enumerate(pc['targets']):
            path = pc['results'].get(idx)
            if path is None:
                self.get_logger().info(f'Tile {t["tile"]}: no Nav2 path; not offered.')
                continue
            t = dict(t)
            pts = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            t['steps'] = len(path.poses)
            t['path_m'] = round(sum(math.dist(a, b) for a, b in zip(pts, pts[1:])), 2)
            t['path_xy'] = [[round(x, 3), round(y, 3)] for x, y in pts]
            t['start_xy'] = t['path_xy'][0]
            t['hint_xy'] = [[round(x, 2), round(y, 2)] for x, y, _ in A.sample_path(pts)]
            reachable.append(t)
        reachable.sort(key=lambda t: t['path_m'])
        targets = reachable[:self.tile_target_limit]
        for i, t in enumerate(targets):
            t['id'] = i + 1
        self.get_logger().info(f'Nav2 path to {len(targets)} of {len(pc["targets"])} unvisited tile(s).')
        self._offer_choice(pc['ctx'], targets)

    def _send_preferred(self, goal_xy):
        msg = PointStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y = float(goal_xy[0]), float(goal_xy[1])
        self.preferred_pub.publish(msg)

    def _send_route(self, route):
        msg = PoseArray()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, yaw in route:
            pose = Pose()
            pose.position.x, pose.position.y = float(x), float(y)
            pose.orientation.z, pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            msg.poses.append(pose)
        self.route_pub.publish(msg)

    def _plan_route(self, target, ai):
        """Waypoints + yaws for the target: the AI's if usable, else evenly spaced along the Nav2 path.

        ai: the AI answer {scene_description, evidence, confidence, extras} or None.
        """
        path = target.get('path_xy') or []
        grid_yaw = self.tile_grid.yaw if self.tile_grid is not None else 0.0
        if ai is not None:
            route, note = A.parse_route(ai['extras'], path, target['goal_xy'], grid_yaw)
        else:
            route, note = A.default_route(path, target['goal_xy'], grid_yaw), 'default (no AI answer)'
        if path:
            route, dropped = A.filter_route(route, path[0])
            if dropped:
                note += f'; dropped {len(dropped)} waypoint(s): ' + '; '.join(dropped)
                self.get_logger().warn('Route filter: ' + '; '.join(dropped))
        return route, note

    def _save_route_image(self, route):
        """stopNNN_route.png: the tile map with the route's waypoints (numbered, arrow = yaw), drawn by the AI client."""
        args = getattr(self, '_map_args', None)
        if args is None or self.grid is None or not self._ai_available():
            return
        self.llm.request('render_map', timeout_s=30.0, track=False, name=f'stop{self.stop["stop"]:03d}_route',
                         save_dir=self.run_dir, pose=args[0], visited=args[1], candidates=[],
                         wall_band_m=self.min_wall_clearance_m, route=[list(p) for p in route], **self._tiles_state())

    def _choose_target(self, target, chosen_by, ai=None):
        """Send a tile target to the explorer and record the decision. ai: the AI answer dict, if the AI chose."""
        route, route_note = self._plan_route(target, ai)
        self._send_route(route)          # before the goal, so the explorer has it when it plans
        self._send_preferred(target['goal_xy'])
        self.intent_pub.publish(String(data=f'tile {target["tile"]} at ({target["goal_xy"][0]:.2f}, '
                                            f'{target["goal_xy"][1]:.2f}) by {chosen_by}'))
        self.next_target = (self.tile_grid.tile_of(*target['goal_xy']), target['goal_xy'])
        self._publish_tile_markers()
        self._save_route_image(route)
        self.stop['next_frontier'] = {
            'chosen_by': chosen_by, 'kind': 'tile', 'id': target['id'], 'tile': target['tile'],
            'steps': target['steps'], 'goal_xy': target['goal_xy'], 'clearance_m': target['clearance_m'],
            'reason': ai['scene_description'] if ai else 'nearest tile with a Nav2 path',
            'evidence': ai['evidence'] if ai else [], 'confidence': ai['confidence'] if ai else None,
            'route': A.route_json(route), 'route_source': route_note}
        self.get_logger().info(f'Next: tile {target["tile"]} ({target["steps"]} step(s), '
                               f'{target["clearance_m"]:.2f} m from walls) chosen by {chosen_by}; route '
                               f'[{route_note}]: ' + ' > '.join(f'({x:.2f},{y:.2f},{math.degrees(w):+.0f}deg)'
                                                               for x, y, w in route))

    def _on_frontier_choice(self, call, capture):
        stop = self.stop
        cands = {c['id']: c for c in capture['candidates']}
        tile_mode = capture.get('mode') == 'tile'
        if not call.ok:
            if tile_mode:
                self.get_logger().error(f'Tile choice failed: {call.failure()}; going to the nearest connected tile.')
                self._choose_target(capture['candidates'][0], f'nearest tile (AI query failed: {call.failure()})')
            else:
                stop['next_frontier'] = {'chosen_by': 'explorer', 'note': f'AI query failed: {call.failure()}'}
                self.get_logger().error(f'Frontier choice failed: {call.failure()}; explorer will choose.')
        else:
            res = call.result
            answer, choice = res['answer'], res.get('choice')
            ai = {**answer, 'extras': res.get('extras') or {}}
            if choice not in cands:
                choice = None
            if tile_mode:
                if choice is None:
                    self.get_logger().warn(f'AI gave no valid target ("{answer["scene_description"]}"); '
                                           'going to the nearest connected tile.')
                    self._choose_target(capture['candidates'][0], 'nearest tile (AI answer invalid)', ai)
                else:
                    self._choose_target(cands[choice], 'AI', ai)
            elif choice is None:
                stop['next_frontier'] = {'chosen_by': 'explorer', 'ai_answer': answer,
                                         'note': 'AI chose none or an invalid id'}
                self.get_logger().warn(f'AI chose no valid frontier ("{answer["scene_description"]}"); '
                                       'explorer will choose.')
            else:
                c = cands[choice]
                self._send_preferred(c['goal_xy'])
                stop['next_frontier'] = {'chosen_by': 'AI', 'kind': 'frontier', 'id': choice,
                                         'goal_xy': c['goal_xy'], 'clearance_m': c['clearance_m'],
                                         'reason': answer['scene_description'],
                                         'evidence': answer['evidence'], 'confidence': answer['confidence']}
                self.get_logger().info(f'AI chose frontier {choice} at {c["goal_xy"]} '
                                       f'(wall clearance {c["clearance_m"]:.2f} m): {answer["scene_description"]}')
        self._publish_stop_report()
        self._set_phase('resume')     # short delay so the preference arrives before the explorer replans

    def _publish_stop_report(self):
        """Record this stop's decision (stops.jsonl). What the tiles look like is in the tile report."""
        stop = self.stop
        report = {k: stop[k] for k in ('stop', 'reason', 'wall_time', 'pose')}
        if 'next_frontier' in stop:
            report['next_frontier'] = stop['next_frontier']
        tile = None
        if self.tile_grid is not None:
            self.tiles_version = -1
            self._update_tiles()
            tile = self.tile_grid.tile_of(stop['pose'][0], stop['pose'][1]) if stop['pose'] else None
            report['tile'] = T.tile_label(tile) if tile else None
            report['tile_sides'] = self.tiles.get(tile, {}).get('sides') if tile else None
            report['tile_counts'] = T.tile_counts(self.tiles)
            self._save_json('tiles.json', self.tile_map_dict())
        self.reports.append(report)
        nxt = report.get('next_frontier') or {}
        where = f'tile {report["tile"]}' if report.get('tile') else f'{report["pose"]}'
        what = (f'tile {nxt["tile"]} by {nxt["chosen_by"]}' if nxt.get('kind') == 'tile' else
                f'frontier {nxt["id"]} by {nxt["chosen_by"]}' if nxt.get('kind') == 'frontier' else
                f'explorer decides ({nxt.get("note", "")})')
        self.get_logger().info(f'STOP {report["stop"]} at {where}: next {what}')
        with open(os.path.join(self.run_dir, 'stops.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(report, ensure_ascii=False, default=str) + '\n')
        self._write_book()

    def _resume_after_stop(self):
        self.stop = None
        if self.complete_pending:
            self._finish()
        else:
            self._set_state(self.EXPLORING)
            self._explorer_active(True)

    def _finish(self):
        self._set_state(self.COMPLETE)
        if self.tile_grid is not None:
            self.tiles_version = -1
            self._update_tiles()
            self._save_json('tiles.json', self.tile_map_dict())
        self._write_book(finished=True)
        self.get_logger().info(f'Exploration complete after {len(self.reports)} stop(s), '
                               f'{len(self.book.entries)} tile(s) reported. Tile report: {self.book.path}')

    # ----------------------------------------------------------------- output
    def _save_json(self, name, data):
        with open(os.path.join(self.run_dir, name), 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    # ------------------------------------------------------- nav2 / explorer
    def _explorer_active(self, active):
        if not self.explorer_client.service_is_ready():
            self.get_logger().error(f'{self.explorer_service} not available', throttle_duration_sec=5.0)
            return
        self.explorer_client.call_async(SetBool.Request(data=active))

    def _send_action(self, client, goal):
        self.action_seq += 1
        seq = self.action_seq
        self.action_handle = None
        if not client.wait_for_server(timeout_sec=0.5):
            self.get_logger().error('Nav2 spin action not available.')
            self.action_done = GoalStatus.STATUS_ABORTED
            return False
        self.action_done = None

        def on_response(future):
            handle = future.result()
            if seq != self.action_seq:
                if handle is not None and handle.accepted:
                    handle.cancel_goal_async()
                return
            if handle is None or not handle.accepted:
                self.get_logger().warn('Nav2 rejected the spin goal.')
                self.action_done = GoalStatus.STATUS_ABORTED
                return
            self.action_handle = handle
            handle.get_result_async().add_done_callback(on_result)

        def on_result(future):
            if seq == self.action_seq:
                self.action_done = future.result().status
                self.get_logger().info(f'Spin finished with status {self.action_done} (4 = succeeded).')
                self.action_handle = None

        client.send_goal_async(goal).add_done_callback(on_response)
        return True

    def _cancel_action(self):
        self.action_seq += 1
        if self.action_handle is not None:
            self.action_handle.cancel_goal_async()
        self.action_handle = None
        self.pending_spin = None
        self.spin_queue = []
        self._release_crawl()

    # ----------------------------------------------------------------- state
    def _set_state(self, state, phase=''):
        if state != self.state:
            self.get_logger().info(f'State: {self.state} -> {state}')
        self.state = state
        self._set_phase(phase)

    def _set_phase(self, phase):
        self.phase = phase
        self.phase_since = time.monotonic()
        self.state_pub.publish(String(data=f'{self.state}:{phase}' if phase else self.state))

    def destroy_node(self):
        self._write_book(finished=self.state == self.COMPLETE)     # whatever has arrived so far
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EnvironmentReporter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._cancel_action()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()