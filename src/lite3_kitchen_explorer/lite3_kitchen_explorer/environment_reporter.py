"""Environment reporter: at every stop of the frontier explorer, ask the AI what kind of place this is.

Start-up: before any exploring, the node builds a 1 m x 1 m tile grid. It proposes
initial-tile placements around the robot (snapped to the nearby walls, centred on
the robot, ...), draws them over the local map, and asks the AI to pick the one that
best fits the immediate walls and frontiers and to describe its four sides. The
grid is anchored to that tile, and only then does exploration begin. From then on
every tile is classified (visited / to_visit / blocked / unknown) and the robot's
path is kept as a trail of visited tiles.

The frontier explorer visits every unknown frontier. Each time it pauses (at the
start, and at every reached frontier), this node:
  1. keeps the explorer paused,
  2. looks around in scan_steps headings (Nav2 Spin), sending one camera frame per
     heading with the survey prompt to the AI server (same /analyze request as client.py),
  3. combines the answers into the stop's environment,
  4. (ai_frontier_choice) draws the SLAM map with walls, the robot and the numbered
     candidate frontiers, and asks the AI which frontier to explore next while
     avoiding walls; the choice goes to the explorer on /frontier_explorer/preferred_goal
     (the explorer still rejects a choice too close to a wall),
  5. publishes the stop report (environment + navigation decision) and resumes exploring.

Interfaces
  sub  camera image                 sensor_msgs/CompressedImage or Image
  sub  /frontier_reached            std_msgs/Bool
  sub  /exploration_complete        std_msgs/Bool (latched)
  sub  /map                         nav_msgs/OccupancyGrid
  sub  /frontier_candidates         std_msgs/String (from the explorer)
  pub  /frontier_explorer/preferred_goal  geometry_msgs/PointStamped (AI choice)
  pub  /tile_map                    std_msgs/String  JSON: grid, trail, every tile's status and sides
  pub  ~/tile_markers               visualization_msgs/MarkerArray  tiles + trail + next target (orange) for RViz
  pub  /environment_report          std_msgs/String  JSON stop report (one per stop)
  pub  ~/state                      std_msgs/String  WAITING_START / EXPLORING / SCANNING:<phase> / COMPLETE
  pub  ~/last_result                std_msgs/String  JSON answer for each image
  pub  ~/markers                    visualization_msgs/MarkerArray  environment label at each stop
  srv  ~/report_now                 std_srvs/Trigger  pause and report here (manual)
Files in output_dir/run_<time>/: every image + answer, reports.jsonl, environment_report.md
"""
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import math
import os
import time

import cv2
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import Spin
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_kitchen_explorer import frontier_core as fc
from lite3_kitchen_explorer.frontier_choice import build_prompt, build_tile_prompt, parse_choice
from lite3_kitchen_explorer.image_utils import encode_jpeg, image_msg_to_bgr
from lite3_kitchen_explorer.map_render import render_map
from lite3_kitchen_explorer.perception_client import PerceptionClient, PerceptionError, read_prompt
from lite3_kitchen_explorer.report_utils import render_markdown, summarize_views
from lite3_kitchen_explorer import tiles as T
from lite3_kitchen_explorer.map_render import render_initial_tile_panels


class EnvironmentReporter(Node):
    WAITING_START, INIT_TILE, EXPLORING, SCANNING, COMPLETE = (
        'WAITING_START', 'INIT_TILE', 'EXPLORING', 'SCANNING', 'COMPLETE')

    def __init__(self):
        super().__init__('environment_reporter')
        p = self.declare_parameter
        server_url = p('server_url', 'http://127.0.0.1:8000').value
        group_id = p('group_id', '').value
        prompt_file = p('prompt_file', '').value
        read_timeout = p('read_timeout_s', 75.0).value
        connect_timeout = p('connect_timeout_s', 5.0).value
        self.image_topic = p('image_topic', '/camera/color/image_raw/compressed').value
        self.image_compressed = p('image_compressed', True).value
        self.max_image_age_s = p('max_image_age_s', 1.5).value
        self.max_image_width = p('max_image_width', 1024).value
        self.jpeg_quality = p('jpeg_quality', 85).value
        self.error_backoff_s = p('error_backoff_s', 5.0).value
        self.query_retries = p('query_retries', 1).value
        self.scan_steps = p('scan_steps', 4).value
        self.spin_time_allowance_s = p('spin_time_allowance_s', 20.0).value
        self.settle_s = p('settle_s', 1.0).value
        self.report_at_start = p('report_at_start', True).value
        self.start_wait_s = p('start_wait_s', 60.0).value
        self.global_frame = p('global_frame', 'map').value
        self.robot_frame = p('robot_frame', 'base_link').value
        self.explorer_service = p('explorer_service', '/frontier_explorer/set_active').value
        self.output_dir = os.path.expanduser(p('output_dir', '~/lite3_survey_runs').value)
        self.ai_frontier_choice = p('ai_frontier_choice', True).value
        frontier_prompt_file = p('frontier_prompt_file', '').value
        self.min_wall_clearance_m = p('min_wall_clearance_m', 0.35).value
        self.choice_min_candidates = p('choice_min_candidates', 2).value
        self.candidates_max_age_s = p('candidates_max_age_s', 3.0).value
        self.tiles_enabled = p('tiles_enabled', True).value
        self.tile_size_m = p('tile_size_m', 1.0).value
        self.tile_wait_s = p('tile_wait_s', 30.0).value
        self.tile_ai_choice = p('tile_ai_choice', True).value
        self.tile_targets_first = p('tile_targets_first', True).value     # go tile by tile
        self.tile_prompt_choice = p('tile_target_ai_choice', True).value  # AI picks among nearest tiles
        self.tile_target_limit = p('tile_target_limit', 6).value
        self.target_min_clearance_m = p('target_min_clearance_m', 0.30).value
        tile_target_prompt_file = p('tile_target_prompt_file', '').value
        tile_prompt_file = p('tile_prompt_file', '').value

        from ament_index_python.packages import get_package_share_directory
        prompts_dir = os.path.join(get_package_share_directory('lite3_kitchen_explorer'), 'prompts')
        prompt_file = prompt_file or os.path.join(prompts_dir, 'environment_report.txt')
        frontier_prompt_file = frontier_prompt_file or os.path.join(prompts_dir, 'frontier_choice.txt')
        self.frontier_prompt = read_prompt(frontier_prompt_file) if self.ai_frontier_choice else ''
        tile_prompt_file = tile_prompt_file or os.path.join(prompts_dir, 'tile_init.txt')
        self.tile_prompt = read_prompt(tile_prompt_file) if self.tiles_enabled and self.tile_ai_choice else ''
        tile_target_prompt_file = tile_target_prompt_file or os.path.join(prompts_dir, 'tile_choice.txt')
        self.tile_target_prompt = read_prompt(tile_target_prompt_file) if self.tiles_enabled else ''
        self.client = PerceptionClient(server_url, group_id, read_prompt(prompt_file),
                                       read_timeout_s=read_timeout, connect_timeout_s=connect_timeout)
        self.run_dir = os.path.join(self.output_dir, 'run_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.run_dir, exist_ok=True)
        self.get_logger().info(f'AI server {self.client.url} group={group_id} prompt={prompt_file}; '
                               f'reports in {self.run_dir}')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.explorer_client = self.create_client(SetBool, self.explorer_service)

        msg_type = CompressedImage if self.image_compressed else Image
        self.create_subscription(msg_type, self.image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(Bool, '/frontier_reached', self._on_frontier_reached, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/exploration_complete', self._on_exploration_complete, latched)
        self.report_pub = self.create_publisher(String, '/environment_report', QoSProfile(
            depth=50, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.state_pub = self.create_publisher(String, '~/state', latched)
        self.result_pub = self.create_publisher(String, '~/last_result', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '~/markers', latched)
        self.create_service(Trigger, '~/report_now', self._on_report_now)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._on_map, map_qos)
        self.create_subscription(String, '/frontier_candidates', self._on_candidates, latched)
        self.preferred_pub = self.create_publisher(PointStamped, '/frontier_explorer/preferred_goal', 10)
        self.grid = None
        self.grid_info = None
        self.candidates = None
        self.candidates_at = 0.0
        self.tile_grid = None
        self.tile_init = None        # how the initial tile was chosen
        self.next_target = None      # (tile, goal_xy) the robot is driving to; highlighted in RViz
        self.trail = []              # visited tiles in visit order (consecutive repeats removed)
        self.trail_log = []          # [{tile, time, pose}]
        self.tiles = {}
        self.map_version = 0
        self.tiles_version = -1
        self.tile_pub = self.create_publisher(String, '/tile_map', latched)
        self.tile_marker_pub = self.create_publisher(MarkerArray, '~/tile_markers', latched)
        self.create_timer(0.5, self._track_tiles)
        self.create_timer(1.0, self._update_tiles)

        self.worker = ThreadPoolExecutor(max_workers=1)
        self.latest_image = None
        self.latest_image_at = 0.0
        self.pending = None           # (future, capture)
        self.query_count = 0
        self.retry_until = 0.0
        self.reports = []
        self.markers = []
        self.complete_pending = False
        self.state = None
        self.phase = ''
        self.phase_since = 0.0
        self.started_at = time.monotonic()
        self.stop = None              # stop being scanned
        self.action_handle = None
        self.action_done = None
        self.action_seq = 0
        if self.report_at_start or self.tiles_enabled:
            self._set_state(self.WAITING_START)
        else:
            self._set_state(self.EXPLORING)
            self._explorer_active(True)
        self.create_timer(0.2, self._tick)

    # ------------------------------------------------------------------ inputs
    def _on_image(self, msg):
        self.latest_image = msg
        self.latest_image_at = time.monotonic()

    def _on_map(self, msg):
        i = msg.info
        self.grid_info = fc.GridInfo(i.width, i.height, i.resolution, i.origin.position.x, i.origin.position.y)
        self.grid = fc.grid_from_msg_data(msg.data, i.width, i.height)
        self.map_version += 1

    def _on_candidates(self, msg):
        try:
            self.candidates = json.loads(msg.data)
            self.candidates_at = time.monotonic()
        except ValueError:
            pass

    def _on_frontier_reached(self, msg):
        if msg.data and self.state == self.EXPLORING:
            self._begin_stop('frontier reached')

    def _on_exploration_complete(self, msg):
        if not msg.data:
            return
        if self.state == self.SCANNING:
            self.complete_pending = True      # finish this stop first
        elif self.state in (self.EXPLORING, self.WAITING_START):
            self._finish()

    def _on_report_now(self, request, response):
        if self.state == self.EXPLORING:
            self._begin_stop('manual request')
            response.success, response.message = True, 'stopping to report'
        else:
            response.success, response.message = False, f'busy ({self.state})'
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
    def _tick(self):
        self._poll_pending()
        now = time.monotonic()
        if self.state == self.WAITING_START:
            ready = self._robot_pose() is not None and self.spin_client.server_is_ready() \
                and self.explorer_client.service_is_ready()
            if ready or now - self.started_at > self.start_wait_s:
                if not ready:
                    self.get_logger().warn('TF / Nav2 / explorer not all ready; starting anyway.')
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
        elif self.state == self.SCANNING:
            self._tick_scanning(now)

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

    def _begin_tile_init(self, pose):
        if self.grid is None:
            self._establish_tiles(None, 'no map: grid centred on the robot along the map axes', pose)
            return
        cands = T.initial_candidates(self.grid, self.grid_info, pose, size=self.tile_size_m)
        if len(cands) == 1 or not self.tile_ai_choice:
            self._establish_tiles(cands[0], 'only one placement' if len(cands) == 1 else 'AI choice disabled', pose)
            return
        img = render_initial_tile_panels(self.grid, self.grid_info, pose, cands,
                                         wall_band_m=self.min_wall_clearance_m)
        ok, buf = cv2.imencode('.png', img)
        name = 'initial_tile'
        with open(os.path.join(self.run_dir, name + '.png'), 'wb') as fh:
            fh.write(buf.tobytes())
        table = '\n'.join(
            [f"{c['id']} | {c['kind']} | " + ', '.join(f'{k}={v}' for k, v in c['sides'].items()) for c in cands])
        from string import Template
        prompt = Template(self.tile_prompt).safe_substitute(
            candidates='id | placement | sides measured from the map\n' + table,
            ids=', '.join(c['id'] for c in cands), size=f'{self.tile_size_m:g}')
        self._save_json(name + '_prompt.json', {'prompt': prompt,
                                                'candidates': [{**c, 'grid': c['grid'].as_dict()} for c in cands]})
        capture = {'kind': 'tile', 'name': name, 'candidates': cands, 'pose': pose}
        self.pending = (self.worker.submit(self.client.analyze, buf.tobytes(), name + '.png', 'image/png', prompt),
                        capture)
        self.get_logger().info(f'Asking the AI to place the initial {self.tile_size_m:g} m tile '
                               f'({len(cands)} options, see {name}.png).')
        self._set_phase('choosing')

    def _on_tile_choice(self, future, capture):
        cands = {c['id']: c for c in capture['candidates']}
        first = capture['candidates'][0]
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Initial-tile query failed: {exc}; using tile {first["id"]}.')
            self._establish_tiles(first, f'AI query failed ({exc}); default placement', capture['pose'])
            return
        answer = result.as_dict()
        self._save_json(capture['name'] + '_answer.json', {**answer, 'raw': result.raw})
        self.result_pub.publish(String(data=json.dumps({'initial_tile': answer})))
        choice, ai_sides = T.parse_tile_choice(result.scene_description, result.evidence, cands)
        if choice is None:
            self.get_logger().warn(f'AI gave no valid tile ("{result.scene_description}"); using tile {first["id"]}.')
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
        self._after_start_setup()

    def _track_tiles(self):
        if self.tile_grid is None:
            return
        pose = self._robot_pose()
        if pose is None:
            return
        tile = self.tile_grid.tile_of(pose[0], pose[1])
        if not self.trail or self.trail[-1] != tile:
            new = tile not in set(self.trail)
            self.trail.append(tile)
            self.trail_log.append({'tile': list(tile), 'time': dt.datetime.now().isoformat(timespec='seconds'),
                                   'pose': pose})
            self.tiles_version = -1
            self.get_logger().info(f'Entered tile {T.tile_label(tile)}' + (' (new)' if new else ' (revisit)'))

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

    def _begin_stop(self, reason):
        self._explorer_active(False)
        self.stop = {'stop': len(self.reports) + 1, 'reason': reason,
                     'wall_time': dt.datetime.now().isoformat(timespec='seconds'),
                     'pose': self._robot_pose(), 'views': [], 'heading': 0, 'attempts': 0}
        self.get_logger().info(f'Stop {self.stop["stop"]} ({reason}): looking around in {self.scan_steps} directions.')
        self._set_state(self.SCANNING, 'settle')

    def _tick_scanning(self, now):
        if self.phase == 'settle' and now - self.phase_since >= self.settle_s:
            self._set_phase('query')
        elif self.phase == 'query':
            if self.pending is not None or now < self.retry_until:
                return
            if self._capture():
                self._set_phase('wait_result')
            elif now - self.phase_since > 5.0:
                self._record_view({'error': f'no fresh image on {self.image_topic}'})
        elif self.phase == 'spinning' and self.action_done is not None:
            if self.action_done != GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().warn('Spin did not succeed; taking the next view from here.')
            self._set_phase('settle')
        elif self.phase == 'choose_wait':
            fresh = self.candidates is not None and now - self.candidates_at <= self.candidates_max_age_s
            if fresh or now - self.phase_since > 6.0:
                self._ask_frontier_choice()
        elif self.phase == 'resume' and now - self.phase_since >= 0.5:
            self._resume_after_stop()

    # ------------------------------------------------------------------ queries
    def _capture(self):
        now = time.monotonic()
        if self.latest_image is None or now - self.latest_image_at > self.max_image_age_s:
            self.get_logger().warn(f'No fresh image on {self.image_topic}', throttle_duration_sec=5.0)
            return False
        try:
            jpeg = encode_jpeg(image_msg_to_bgr(self.latest_image), self.max_image_width, self.jpeg_quality)
        except ValueError as exc:
            self.get_logger().error(f'Image conversion failed: {exc}', throttle_duration_sec=5.0)
            return False
        self.query_count += 1
        name = f'stop{self.stop["stop"]:03d}_view{self.stop["heading"] + 1}_q{self.query_count:04d}'
        with open(os.path.join(self.run_dir, name + '.jpg'), 'wb') as fh:
            fh.write(jpeg)
        capture = {'name': name, 'pose': self._robot_pose()}
        self.pending = (self.worker.submit(self.client.analyze, jpeg, name + '.jpg', 'image/jpeg'), capture)
        return True

    def _poll_pending(self):
        if self.pending is None or not self.pending[0].done():
            return
        future, capture = self.pending
        self.pending = None
        if capture.get('kind') == 'frontier':
            self._on_frontier_choice(future, capture)
            return
        if capture.get('kind') == 'tile':
            self._on_tile_choice(future, capture)
            return
        try:
            result = future.result()
        except PerceptionError as exc:
            self.retry_until = time.monotonic() + max(self.error_backoff_s, exc.retry_after_s or 0.0)
            self._on_query_failed(capture, str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - the worker must never kill the node
            self.retry_until = time.monotonic() + self.error_backoff_s
            self._on_query_failed(capture, repr(exc))
            return
        view = {**result.as_dict(), 'image': capture['name'] + '.jpg', 'elapsed_s': round(result.elapsed_s, 2)}
        self._save_json(capture['name'] + '.json', {**view, 'pose': capture['pose'], 'raw': result.raw})
        self.result_pub.publish(String(data=json.dumps(view)))
        self.get_logger().info(f'  view {self.stop["heading"] + 1}: {result.scene_description} '
                               f'(identified={result.relevant_to_mission}, conf {result.confidence:.2f})')
        self._record_view(view, capture['pose'])

    def _on_query_failed(self, capture, error):
        self.get_logger().error(f'Query {capture["name"]} failed: {error}')
        self._save_json(capture['name'] + '.json', {'error': error, 'pose': capture['pose']})
        if self.stop is None:
            return
        self.stop['attempts'] += 1
        if self.stop['attempts'] <= self.query_retries:
            self._set_phase('query')          # retry this heading after the backoff
        else:
            self._record_view({'error': error, 'image': capture['name'] + '.jpg'}, capture['pose'])

    def _record_view(self, view, pose=None):
        stop = self.stop
        pose = pose or self._robot_pose()
        view['heading_deg'] = round(math.degrees(pose[2]), 1) if pose else 360.0 * stop['heading'] / self.scan_steps
        stop['views'].append(view)
        stop['heading'] += 1
        stop['attempts'] = 0
        if stop['heading'] >= self.scan_steps:
            self._finish_stop()
            return
        goal = Spin.Goal()
        goal.target_yaw = 2.0 * math.pi / self.scan_steps
        goal.time_allowance = DurationMsg(sec=int(self.spin_time_allowance_s))
        self._send_action(self.spin_client, goal)
        self._set_phase('spinning')

    def _finish_stop(self):
        self.stop['summary'] = summarize_views(self.stop['views'])
        if self.ai_frontier_choice and not self.complete_pending:
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
        mode = 'frontier'
        self.next_target = None      # replaced below if a tile is chosen
        if self.tile_grid is not None:
            self._publish_tile_markers()
        if self.tile_grid is not None and self.tile_targets_first:
            targets = T.nearest_connected_targets(
                self.grid, self.grid_info, self.tile_grid, self.tiles, self.trail[-1], pose[:2],
                limit=self.tile_target_limit, min_clearance_m=self.target_min_clearance_m)
            if len(targets) == 1 or (targets and not self.tile_prompt_choice):
                self._choose_target(targets[0], 'nearest tile (only option)' if len(targets) == 1
                                    else 'nearest tile (AI choice disabled)')
                self._publish_stop_report()
                self._set_phase('resume')
                return
            if targets:
                mode, cands = 'tile', targets
            else:
                self.get_logger().info('No unvisited tile is connected; choosing among frontiers instead.')
        if mode == 'frontier' and len(cands) < max(1, self.choice_min_candidates):
            stop['next_frontier'] = {'chosen_by': 'explorer', 'note': f'AI not asked ({len(cands)} frontier(s))'}
            self._publish_stop_report()
            self._set_phase('resume')
            return
        try:
            img = render_map(self.grid, self.grid_info, pose, cands, visited,
                             wall_band_m=self.min_wall_clearance_m, tile_grid=self.tile_grid,
                             tiles=self.tiles, trail=self.trail)
        except ValueError as exc:
            stop['next_frontier'] = {'chosen_by': 'explorer', 'note': f'map render failed: {exc}'}
            self._publish_stop_report()
            self._set_phase('resume')
            return
        ok, buf = cv2.imencode('.png', img)
        name = f'stop{stop["stop"]:03d}_map'
        with open(os.path.join(self.run_dir, name + '.png'), 'wb') as fh:
            fh.write(buf.tobytes())
        if mode == 'tile':
            prompt = build_tile_prompt(self.tile_target_prompt, cands, stop['summary']['environment'],
                                       self.target_min_clearance_m, len(self.reports) + 1, tile_summary)
        else:
            prompt = build_prompt(self.frontier_prompt, cands, stop['summary']['environment'],
                                  self.min_wall_clearance_m, len(self.reports) + 1, tile_summary)
        self._save_json(name + '_prompt.json', {'prompt': prompt, 'mode': mode, 'candidates': cands})
        capture = {'kind': 'frontier', 'mode': mode, 'name': name, 'candidates': cands}
        self.pending = (self.worker.submit(self.client.analyze, buf.tobytes(), name + '.png', 'image/png', prompt),
                        capture)
        what = 'connected unvisited tiles' if mode == 'tile' else 'frontiers'
        self.get_logger().info(f'Asking the AI to choose among {len(cands)} {what} (map: {name}.png).')
        self._set_phase('choosing')

    def _send_preferred(self, goal_xy):
        msg = PointStamped()
        msg.header.frame_id = self.global_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y = float(goal_xy[0]), float(goal_xy[1])
        self.preferred_pub.publish(msg)

    def _choose_target(self, target, chosen_by, result=None):
        """Send a tile target to the explorer and record the decision."""
        self._send_preferred(target['goal_xy'])
        self.next_target = (self.tile_grid.tile_of(*target['goal_xy']), target['goal_xy'])
        self._publish_tile_markers()
        self.stop['next_frontier'] = {
            'chosen_by': chosen_by, 'kind': 'tile', 'id': target['id'], 'tile': target['tile'],
            'steps': target['steps'], 'goal_xy': target['goal_xy'], 'clearance_m': target['clearance_m'],
            'reason': result.scene_description if result else 'nearest connected unvisited tile',
            'evidence': result.evidence if result else [], 'confidence': result.confidence if result else None}
        self.get_logger().info(f'Next: tile {target["tile"]} ({target["steps"]} step(s), '
                               f'{target["clearance_m"]:.2f} m from walls) chosen by {chosen_by}.')

    def _on_frontier_choice(self, future, capture):
        stop = self.stop
        cands = {c['id']: c for c in capture['candidates']}
        tile_mode = capture.get('mode') == 'tile'
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            if tile_mode:
                self.get_logger().error(f'Tile choice failed: {exc}; going to the nearest connected tile.')
                self._choose_target(capture['candidates'][0], f'nearest tile (AI query failed: {exc})')
            else:
                stop['next_frontier'] = {'chosen_by': 'explorer', 'note': f'AI query failed: {exc}'}
                self.get_logger().error(f'Frontier choice failed: {exc}; explorer will choose.')
        else:
            choice = parse_choice(result.scene_description, result.evidence, cands)
            answer = result.as_dict()
            self._save_json(capture['name'] + '_answer.json', {**answer, 'raw': result.raw})
            self.result_pub.publish(String(data=json.dumps({'navigation_choice': answer})))
            if tile_mode:
                if choice is None:
                    self.get_logger().warn(f'AI gave no valid target ("{result.scene_description}"); '
                                           'going to the nearest connected tile.')
                    self._choose_target(capture['candidates'][0], 'nearest tile (AI answer invalid)', result)
                else:
                    self._choose_target(cands[choice], 'AI', result)
            elif choice is None:
                stop['next_frontier'] = {'chosen_by': 'explorer', 'ai_answer': answer,
                                         'note': 'AI chose none or an invalid id'}
                self.get_logger().warn(f'AI chose no valid frontier ("{result.scene_description}"); '
                                       'explorer will choose.')
            else:
                c = cands[choice]
                self._send_preferred(c['goal_xy'])
                stop['next_frontier'] = {'chosen_by': 'AI', 'kind': 'frontier', 'id': choice,
                                         'goal_xy': c['goal_xy'], 'clearance_m': c['clearance_m'],
                                         'reason': result.scene_description,
                                         'evidence': result.evidence, 'confidence': result.confidence}
                self.get_logger().info(f'AI chose frontier {choice} at {c["goal_xy"]} '
                                       f'(wall clearance {c["clearance_m"]:.2f} m): {result.scene_description}')
        self._publish_stop_report()
        self._set_phase('resume')     # short delay so the preference arrives before the explorer replans

    def _publish_stop_report(self):
        stop = self.stop
        report = {k: stop[k] for k in ('stop', 'reason', 'wall_time', 'pose', 'views', 'summary')}
        if 'next_frontier' in stop:
            report['next_frontier'] = stop['next_frontier']
        if self.tile_grid is not None:
            self.tiles_version = -1
            self._update_tiles()
            tile = self.tile_grid.tile_of(stop['pose'][0], stop['pose'][1]) if stop['pose'] else None
            report['tile'] = T.tile_label(tile) if tile else None
            report['tile_sides'] = self.tiles.get(tile, {}).get('sides') if tile else None
            report['tile_counts'] = T.tile_counts(self.tiles)
            self._save_json('tiles.json', self.tile_map_dict())
        self.reports.append(report)
        s = report['summary']
        self.get_logger().info(
            f'STOP {report["stop"]} REPORT at {report["pose"]}: {s["environment"]} '
            f'(conf {s["confidence"]:.2f}, {s["identified_views"]}/{s["total_views"]} views identified) '
            f'evidence: {s["evidence"]}')
        self.report_pub.publish(String(data=json.dumps(report)))
        with open(os.path.join(self.run_dir, 'reports.jsonl'), 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(report, ensure_ascii=False) + '\n')
        self._write_markdown()
        self._publish_marker(report)

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
        self._write_markdown(finished=True)
        self.get_logger().info(f'Exploration complete after {len(self.reports)} stop(s). '
                               f'Report: {os.path.join(self.run_dir, "environment_report.md")}')

    # ----------------------------------------------------------------- output
    def _save_json(self, name, data):
        with open(os.path.join(self.run_dir, name), 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

    def _write_markdown(self, finished=False):
        with open(os.path.join(self.run_dir, 'environment_report.md'), 'w', encoding='utf-8') as fh:
            fh.write(render_markdown(self.reports, self.global_frame, finished))

    def _publish_marker(self, report):
        if not report['pose']:
            return
        x, y, _ = report['pose']
        dot, text = Marker(), Marker()
        for i, m in enumerate((dot, text)):
            m.header.frame_id = self.global_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.action = 'environment', 2 * report['stop'] + i, Marker.ADD
            m.pose.position.x, m.pose.position.y = x, y
            m.pose.orientation.w = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.8, 0.1, 1.0
        dot.type = Marker.CYLINDER
        dot.scale.x, dot.scale.y, dot.scale.z = 0.25, 0.25, 0.05
        text.type = Marker.TEXT_VIEW_FACING
        text.text = f'{report["stop"]}: {report["summary"]["environment"]}'
        text.pose.position.z = 0.6
        text.scale.z = 0.22
        self.markers += [dot, text]
        self.marker_pub.publish(MarkerArray(markers=self.markers))

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
                self.action_done = GoalStatus.STATUS_ABORTED
                return
            self.action_handle = handle
            handle.get_result_async().add_done_callback(on_result)

        def on_result(future):
            if seq == self.action_seq:
                self.action_done = future.result().status
                self.action_handle = None

        client.send_goal_async(goal).add_done_callback(on_response)
        return True

    def _cancel_action(self):
        self.action_seq += 1
        if self.action_handle is not None:
            self.action_handle.cancel_goal_async()
        self.action_handle = None

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
        self.worker.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EnvironmentReporter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._cancel_action()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
