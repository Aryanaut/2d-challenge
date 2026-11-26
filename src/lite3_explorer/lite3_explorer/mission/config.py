"""Mission configuration: defaults and merging of the flat ROS parameters (no ROS).

ROS 2 YAML nests parameters with dots (``modes.exploring.traversal``). The mission manager is created with
``automatically_declare_parameters_from_overrides=True``, so ANY key in the YAML reaches it, and this module
merges those flat keys over DEFAULTS. That is what makes modes, traversals and end conditions pluggable from
config alone: a new mode ``modes.exploring_careful: {type: exploring, traversal: careful}`` or a new
traversal profile ``traversals.careful: {type: nav2_pose, ...}`` needs no new declare_parameter calls.

ROS parameters cannot be empty lists of unknown type; use [''] for "none" (empty strings are dropped).
"""
from __future__ import annotations

import copy
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    'global_frame': 'map',
    'robot_frame': 'base_link',
    'map_topic': '/map',
    'global_costmap_topic': '/global_costmap/costmap',
    'local_costmap_topic': '/local_costmap/costmap',
    'tick_period_s': 0.2,
    # SLAM jump guard: map->odom moving more than this between two checks (0.2 s) is a loop closure / relocalisation,
    # not motion (odom->base_link carries the motion). Logged as a 'slam_jump' event; no tile counts as reached from the
    # pose for slam_jump_hold_s afterwards. 0 = off. fullrun_25_11_215528: a false loop closure moved it 2.1 m.
    'slam_jump_m': 0.5,
    'slam_jump_deg': 15.0,
    'slam_jump_hold_s': 5.0,
    'output_dir': '~/lite3_survey_runs',
    'reports_dir': '~/2d/reports',
    'scene_images_dir': '~/2d/images/runs',   # the images sent to the AI for scene tiles: <dir>/<run>/scene_<tile>_q<n>.jpg (+ .json answer)
    'use_ai': True,                       # AI tile descriptions via lite3_llm_tools/llm_client
    'terrain_classify_service': '/terrain_classifier/classify',   # asked for the tile being driven to ('' = never)
    'terrain_wait_s': 3.0,            # no terrain answer this long after asking = the floor of that tile cannot be seen
    'step_up_gait': True,             # target tile higher than the current one (classifier elevation option): gait_switcher uses h-step
    'prefer_flat_tiles': True,        # classifier elevation option on: go to the reachable tile with the least elevation change (up or down) from the current one
    'flat_tile_tolerance_m': 0.03,    # ... elevation changes within this of the best count as equal (the strategy's ranking decides)
    'step_up_min_rise_m': 0.04,       # ... when it is at least this much higher
    'step_up_depth': True,            # ... and when step_ahead_detector sees a step up ahead in the depth image while driving to a
                                      # tile (fallback where the tile elevations miss it); held until the tile changes
    'step_up_depth_max_age_s': 1.0,   # ... a detection older than this does not count for a new tile
    'step_up_from_max_slope_deg': 15.0,  # ... the current tile's elevation is ignored (start floor, 0 m, used instead) when
                                      # its plane is steeper than this: a wall panel read as floor (fullrun_26_11_120359: the
                                      # current tile read 0.55 m at 22.6 deg, so the higher target looked lower)
    'llm_report_timeout_s': 180.0,
    'llm_report_retries': 5,
    'robot_length_m': 0.61,               # turn-room check (match the local costmap footprint)
    'robot_width_m': 0.37,
    'turn_room_margin_m': 0.0,

    'tiles': {
        'size_m': 1.0,
        'reach_radius_m': 0.30,           # a tile is reached (explored) this close to its centre
        'max_failures': 2,                # failed attempts before a tile is given up on (unreachable); 0 = never
        'no_curtain_on_initial_tile': False,   # true: curtain spots on the initial tile's edges are ignored
        'min_known': 0.5,                 # share of a tile that must be scanned before it becomes a tile
        'scene_min_known': 0.85,          # scene tile: scanned at least this much ...
        'scene_max_free': 0.80,           # ... and less than this share of the SCANNED part traversable (drawn blue hatched in RViz)
        'scene_lidar_min_known': 0.0,     # a scene from the lidar alone needs this much scanned; less: only with a scene-prop sighting
        'initial_clear_m': 0.40,          # the initial tile is closed once the robot is this far outside it (body cleared)
        'align_on_entry': True,           # on every tile entry: nudge the grid so that tile's edges lie on the walls around it
        'align_max_shift_m': 0.15,        # ... an edge further than this from its wall is not that wall's: left alone
        'align_max_rot_deg': 6.0,         # ... turn at most this much towards the wall angle
        'align_radius_m': 2.0,            # ... walls looked at within this of the robot
        'align_min_quality': 1.5,         # ... wall-angle quality (estimate_wall_angle) needed
        'scene_margin_m': 0.10,           # ... measured this far inside the tile edges (walls on a slightly-off grid are no scene)
        'stair_min_points': 4,            # confirmed stair cells (10 cm, /stairs/points) that make a tile a staircase tile
        'stair_floor_min_known': 0.90,    # a tile scanned this much ...
        'stair_floor_min_free': 0.95,     # ... with this share of its interior free is open floor: never a staircase
        'stair_shift_max_m': 0.35,        # stair cells this close inside a traversable tile's edge go to the tile beyond it
        'roller_min_points': 4,           # confirmed roller spots (10 cm, /roller/points) that make a tile a roller tile ...
        'roller_min_lidar': 2,            # ... of which this many are confirmed by the 3D lidar (0 = depth camera alone)
        'roller_depth_only_points': 12,   # ... or a tile with this many spots needs no lidar (0 = lidar always required)
        'roller_edge_margin_m': 0.15,     # spots this close to a tile edge do not count (the roller lies mid-tile)
        # scene-prop votes (prop_votes.py): the detector's scene props placed on the lidar map keep a tile a scene tile
        'prop_votes': False,              # on / off (also at runtime: ros2 param set /mission_manager tiles.prop_votes true)
        'prop_min_frames': 8,             # prop sightings on a tile needed ...
        'prop_min_score': 4.0,            # ... and their summed confidence
        'prop_detections_topic': '/perception/detections',
        'prop_camera_info_topic': '/camera/color/camera_info',
        'prop_fx': 462.6,                 # colour camera intrinsics until camera_info arrives (Orbbec 640x480)
        'prop_cx': 320.1,
        'prop_camera_x_m': 0.252,         # camera ahead of base_link
        'prop_min_conf': 0.35,
        # chosen-tile check: the tile the exploration picks as its target is run against the scene-prop sightings; any
        # on it (this many sightings, summed confidence) -> a scene tile, not driven into. Independent of prop_votes;
        # an AI 'traversable' answer reopens it.
        'chosen_prop_check': True,
        'chosen_prop_min_frames': 1,
        'chosen_prop_min_score': 0.0,
        'prop_max_range_m': 3.0,
        'prop_max_turn_dps': 30.0,        # ignore detections while the robot turns faster than this (pose / frame mismatch)
        'prop_labels': ['tool_bench', 'hand_tools', 'kitchen', 'utensils', 'market_stall', 'cart', 'food',
                        'cleaning_cart', 'medical_kit'],   # any of them = "a prop is here"; the label is not trusted
    },

    # Tile report: detections (when + tile of every QR / hazard / survivor), the mission reporter's results and the
    # perception / reporter /rosout lines. Filled only while those nodes run (explorer.sh --reporter, the sim).
    'mission_log': {
        'enabled': True,
        'event_topic': '/perception/event',        # object / hazard / survivor sightings
        'qr_topic': '/qr/detections',              # qr_scanner: once per newly confirmed code
        'qr_text_topic': '/perception/qr_text',    # perception's own decoder (same text only once)
        'result_topic': '/mission/result',         # what the reporter sent
        'perception_nodes': ['object_detector', 'qr_scanner', 'llm_event_bridge', 'sim_perception'],
        'reporter_nodes': ['mission_reporter'],
        'min_log_level': 20,                       # /rosout lines at or above this (20 INFO, 30 WARN)
        'log_tail': 300,                           # log lines shown per group in the report (all of them in run_dir/*.log)
    },

    'mission': {
        # Mode instances to build. Each name is configured under modes.<name>; modes.<name>.type picks the class
        # (defaults to the name). Add your own, e.g. 'exploring_stairs' with type: exploring.
        'modes': ['idle', 'scanning', 'exploring', 'quickscan', 'curtain', 'curtain_scan', 'stairs', 'roller', 'scene',
                  'mppi', 'backtracking', 'recovery', 'ending', 'complete'],
        'initial_mode': 'idle',
        'start_mode': 'scanning',         # where set_active(true) / auto start goes
        'recovery_mode': 'recovery',      # where the stuck supervisor sends the robot
        'complete_mode': 'complete',
    },

    'stuck': {
        'enabled': True,
        'watch_modes': ['exploring', 'backtracking'],     # modes the stuck supervisor watches
        'triggers': ['no_motion', 'costmap'],
        'window_s': 60.0,                 # no_motion: driving this long ...
        'min_motion_m': 0.15,             # ... without moving this far = stuck
        'costmap_cost': 99,               # costmap: robot centre on a cell this costly (99 inscribed, 100 lethal) ...
        'costmap_hold_s': 20.0,           # ... for this long = stuck in the global costmap
        'cooldown_s': 5.0,                # ignore triggers this long after (re-)entering a watched mode
        'costmap_mode': 'mppi',           # stuck in the costmap obstacle zone -> this mode moves the robot out ('' = recovery)
        'per_mode': {},                   # <mode>: {costmap_hold_s, cooldown_s} overriding the above while that mode runs
    },

    'end_condition': {
        'combine': 'any',                 # any | all
        'conditions': ['all_tiles_explored'],
        'all_tiles_explored': {
            'min_visited': 1,             # never finish before this many tiles were reached
            'ignore_unreachable': False,  # False: tiles given up on still count as unexplored (retried), no early finish
        },
        'time_limit': {'seconds': 0.0},   # 0 = off
        'max_tiles_visited': {'count': 0},
        'no_frontiers_left': {'min_cluster_cells': 8, 'goal_clearance_m': 0.35, 'min_clearance_m': 0.30},
        # the robot entered a tile with no walls on its edges (triggers; 'hallway' = >= min_hallway_tiles in a row): the
        # tile before it on the robot's route with exactly `walls` walls is the ending tile. Met at its centre; `mode` drives there.
        # "no walls": no edge read as a wall (mapped wall along >= solid_wall_share of it); never the initial tile
        'ending_tile': {'triggers': ['next_tile', 'open_tile'], 'min_seen_share': 0.5, 'min_hallway_tiles': 4, 'walls': 2, 'entered_only': True,
                        'solid_wall_share': 0.4,
                        'preceding': True, 'reach_radius_m': 0.30, 'lock': True, 'mode': 'ending'},
    },

    'explorations': {
        # Named exploration profiles: WHERE to go next. type = a registered strategy (see mission/exploration.py).
        'tiles': {
            'type': 'tiles',
            'max_adjacency': 2,           # 0 open side w/ current tile, 1 open side w/ a visited tile,
                                          # 2 any side (also across a wall) w/ a visited tile, 3 anywhere
            'neighbours_only': True,      # choose only among the tiles sharing a side with the current tile (the rest: irrelevant)
            'neighbours_fallback': True,  # ... unless none of them is left (dead end): then offer the others (False: wait / recover)
            'fallback_any': False,         # no tile within max_adjacency -> offer the rest (else: none)
            'reject_via_other_tiles': True,   # drop a tile whose path runs through other tiles (not the current one / itself)
            'via_tile_tolerance_m': 0.15,
            'direct_path_only': True,     # next tile: a neighbour of the current tile, path through their shared side
            'direct_fallback': True,      # none qualifies -> best reachable tile (False: none)
            'prefer_current': True,       # rank neighbours of the current tile before other visited tiles' neighbours
            'eliminate_after': 2,         # consecutive failed Nav2 path checks before a tile is eliminated; 0 = never
            'candidate_limit': 12,        # tiles sent to the Nav2 path check per decision
            'target_limit': 6,
            'min_clearance_m': 0.30,
        },
        'frontier': {                     # optional: add to modes.exploring.exploration to use it
            'type': 'frontier',
            'min_cluster_cells': 8,
            'goal_clearance_m': 0.35,
            'min_clearance_m': 0.30,
            'distance_weight': 1.0,
            'size_weight': 0.5,
            'min_goal_distance_m': 0.6,
            'blacklist_radius_m': 0.6,
            'candidate_limit': 8,
            'target_limit': 6,
            'replan_when_explored': True,
            'explored_radius_m': 0.5,
        },
    },

    'traversals': {
        # Named traversal profiles. type = a registered traversal class (see mission/traversal.py).
        'route': {'type': 'nav2_route',   # NavigateThroughPoses: N waypoints, then the tile centre, without stopping in between
                  'backend': 'through_poses',   # through_poses | follow_waypoints (FollowWaypoints: stops at every waypoint)
                  'route_mode': 'straight',   # straight: waypoints on the line current tile centre -> next tile centre; path: along the Nav2 path
                  'waypoints': 3,         # intermediate waypoints, evenly spaced along the path
                  'skip_start_m': 0.0,    # 0 = equal gaps start->wp->goal; >0 skips this much of the path first (first gap longer)
                  'max_waypoint_deviation_m': 0.25,   # drop waypoints further than this off the straight line to the tile centre (0 = keep all)
                  'drop_backward_waypoints': False},   # True: drop waypoints facing away from the goal (fewer than N)
        'direct': {'type': 'nav2_pose', 'behavior_tree': ''},   # NavigateToPose straight to the goal
        # recovery: NavigateToPose followed by the holonomic RecoveryMPPI controller (nav2_params*.yaml), vx / vy / wz
        'recovery_mppi': {'type': 'nav2_pose', 'behavior_tree': 'recovery_mppi.xml'},
        # recovery fallback without Nav2: straight linear x / y velocity to the goal, heading held (mission/traversal.py)
        'omni': {'type': 'omni', 'speed_mps': 0.20, 'min_speed_mps': 0.06, 'gain': 0.8, 'tolerance_m': 0.10,
                 'heading_kp': 1.5, 'wz_max': 0.3, 'block_cost': 100, 'lookahead_m': 0.30, 'body_length_m': 0.61, 'body_width_m': 0.37,
                 'body_pad_m': 0.03, 'timeout_s': 30.0,
                 'stall_s': 6.0, 'min_progress_m': 0.05,
                 'lead_s': 0.0, 'heading_latch_s': 0.0},   # lag compensation, off here (mission.yaml sets the robot's)
    },

    'modes': {
        'idle': {
            'type': 'idle',
            'auto_start': True,           # leave idle by itself once TF / Nav2 / AI are ready
            'start_wait_s': 60.0,         # ... or after this long anyway
        },
        'scanning': {
            'type': 'scanning',
            'turn': True,                 # false = no turning, just refresh the tiles
            'turn_deg': 360.0,
            'opening_setpoints_deg': [0.0, 80.0, -80.0],   # opening spin: yaw setpoints relative to the start heading (front sweep, never through the back), then back to 0
            'opening_retries': 5,           # the opening scan must finish before exploring
            'opening_min_turn_deg': 160.0,    # the opening spin may stop early (map not sparse) only after turning this far
            'return_tolerance_deg': 3.0,  # after the opening sweep turn back to the starting heading
            'time_allowance_s': 20.0,
            'spin_lead_deg': 6.0,         # Nav2 Spin asks for this much less (the dog coasts past; Nav2Scanner only)
            'use_crawl': False,           # true: crawl gait while turning (gait_switcher /spin_crawl); false: keep the current gait
            'settle_s': 1.0,
            'check_room': True,           # skip the turn where the body would hit the local costmap
            'map_wait_s': 30.0,           # max wait for the map around the start before placing the tile grid
            'next_mode': 'exploring',
            'opening_stairs': False,      # stair detector on for the opening scan; a staircase beside the start tile -> stairs_mode
            'opening_stairs_wait_s': 2.0, # stand still this long after the opening scan so the detector confirms what it sees
            'stairs_mode': 'stairs',
            'stair_min_facing': 0.5,      # as modes.exploring.stair_min_facing
            'relocate_wait_s': 3.0,       # after the stairs: stand still this long before the tile is checked against SLAM (the
                                          # scan reaches slam_toolbox ~1.1 s late; the check must see a scan matched after the rebase)
        },
        'exploring': {
            'type': 'exploring',
            'exploration': ['tiles'],     # exploration profiles tried in order (WHERE); e.g. ['tiles', 'frontier']
            'traversal': 'route',         # traversal profile used to drive (HOW)
            'kind_traversals': [''],      # per target kind "kind:profile", e.g. "frontier:direct"
            'tile_traversals': [''],      # per-tile overrides "i,j:profile", e.g. "3,1:stairs"
            'path_check_timeout_s': 10.0,
            'goal_timeout_s': 120.0,
            'scan_after_reach': True,
            'scenes': True,               # a scene tile (scanned, mostly not traversable) shares a side with the current tile -> scene mode
            'scene_mode': 'scene',        # (scene tiles are never driven to; the scene mode looks at them from the neighbouring tile)
            'curtains': False,            # use_curtains:=true: a curtain on the current tile's edge -> curtain mode (walk through it)
            'curtain_mode': 'curtain',
            'curtain_scan_when_stuck': True,   # no reachable tile: face this tile's edges for a curtain first (curtain_scan_mode)
            'curtain_scan_mode': 'curtain_scan',
            'curtain_edge_tol_m': 0.30,   # curtain spots within this of the tile edge line count as on the edge
            'curtain_min_span_m': 0.40,   # ... spread over at least this much of it
            'curtain_camera_tol_m': 0.15,  # the camera spots confirming an edge: within this of its line ...
            'curtain_camera_min_span_m': 0.30,   # ... and spread over this much of it by themselves
            'curtain_wall_switch_ratio': 2.0,   # the yellow curtain wall moves to another edge with this many times its camera spots
            'curtain_last': True,         # another reachable tile beside the curtain's tile: go there first, the tile is the
                                          # last split, backtracking walks through the curtain later (needs backtracking)
            'stairs': False,              # use_stairs:=true: a staircase tile shares a side with the current tile -> stairs
            'stairs_mode': 'stairs',      # mode (AI mode on, climb over it into the tile beyond, AI mode off, wall it off)
            'stair_min_facing': 0.5,      # climb only from a side the steps go up from (cos of ascent vs the way across)
            'stairs_when_stuck': True,    # nothing reachable left on this side (after retries / recovery): give up on the
                                          # rest and climb the staircase instead of retrying forever
            'stairs_after_retries': 1,    # ... after this many retry rounds without a new tile
            'stairs_backtrack_attempts': 3,   # forced stairs: backtrack to the tile in front of the staircase (instead of
                                          # recovery) at most this many times, then the usual handling (0 = no limit)
            'rollers': False,             # use_rollers:=true: a roller tile shares a side with the current tile (or the robot
            'roller_mode': 'roller',      # stands in front of its roller) -> roller mode (square up, h-step gait, cross it)
            'roller_min_facing': 0.8,     # cross only when the roller lies across the way (|cos| of its normal vs the side)
            'backtracking': True,         # two or more tiles to travel to -> the tile is the last split; no exploration
                                          # target left but tiles unexplored -> backtracking mode (else retry_wait)
            'backtracking_mode': 'backtracking',
            'split_on_discovery': True,   # a new tile found beside a tile visited earlier (sharing a side that is no wall)
                                          # makes that tile a split again, so backtracking can come back for it
            'classify_target': True,      # ask the terrain classifier about the target tile once the robot points at it;
                                          # floor not visible -> choose another tile
            'classify_max_angle_deg': 20.0,   # "pointing at it": heading within this angle of the tile
            'retry_period_s': 15.0,       # nothing reachable but tiles left: wait, then retry every tile (never finish early)
            'retry_rescan': True,         # rescan first: new map data may open a Nav2 path
            'retry_clear_costmaps': True,
            'scan_mode': 'scanning',
            'face_away_from_walls': True,   # tile goals: final heading N/W/E/S facing away from the most walls
            'on_goal_failed': 'decide',   # decide (choose another tile) | recovery
            'no_path_retries': 2,         # rescan + retry this many times (retry_period_s apart) before going to recovery
            'on_no_reachable_tile': 'recovery',   # recovery | scanning | idle
            'quickscan_on_edge': False,   # true: while driving, stop on a tile's edge and run the quickscan animation
            'quickscan_edge_margin_m': 0.15,   # "on the edge" = within this distance of the tile boundary
            'quickscan_mode': 'quickscan',
        },
        'quickscan': {
            'type': 'quickscan',
            'animation': 'quickscan',     # robot animation (animations.py) played by the animation_player node
            'settle_s': 1.0,              # wait for the robot to stand still before the animation starts
            'start_timeout_s': 3.0,       # animation_player did not start it: skip
            'timeout_s': 30.0,            # give up on (and cancel) an animation that runs this long
            'next_mode': 'exploring',     # a new decision is made afterwards (the cancelled goal is re-chosen)
        },
        'scene': {
            'type': 'scene',
            'next_mode': 'exploring',     # the scene tile is marked explored (looked at, not entered), then exploration goes on
            'rest_s': 1.5,                # stand still first (a scan spin just before is still coasting)
            'center_tol_m': 0.06,         # then drive (vx / vy, heading held) to within this of the current tile's centre
            'center_speed_mps': 0.15,
            'center_min_speed_mps': 0.05,
            'center_gain': 1.0,
            'heading_gain': 1.0,          # wz that holds the heading while centring
            'center_timeout_s': 15.0,     # not there by then: turn to face the scene tile from where it is
            'face': 'back_wall',          # back_wall: square up to the scene tile's back wall (lidar map); none found / centre: face its centre
            'back_wall_max_deg': 20.0,    # the back wall may be turned this far from the tile grid
            'back_wall_beyond_m': 0.4,    # ... and lie in the scene tile's far half or up to this past its far edge
            'back_wall_min_coverage': 0.8,   # ... a continuous line over at least this share of the tile's middle 0.8
            'align_tol_deg': 6.0,         # only then turn in place until facing it within this
            'commit_deg': 150.0,          # target further round than this: keep the direction the turn started in (no flip at 180 deg)
            'align_timeout_s': 15.0,
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.4,
            'turn_gain': 1.0,
            'settle_s': 1.0,              # stand still facing it before the AI looks (sharp image)
            'quickscan': True,            # take 3 pictures (yaw left, centre, right: the scenescan animation) instead of one
            'animation': 'scenescan',
            'start_timeout_s': 3.0,       # animation_player did not start it: one picture from where it stands
            'timeout_s': 30.0,            # give up on (and cancel) the animation after this long
            'llm_attempts': 2,            # AI description tries; after the last the tile is marked explored anyway
            'background': False,          # true: once the pictures are taken, go on exploring; the answer is recorded when it comes
        },
        'mppi': {
            'type': 'mppi',
            'next_mode': 'exploring',     # clear of the costmap obstacle zone: decide again
            'fail_mode': 'recovery',      # could not get out (timeout, no way out, moved too far)
            'speed_mps': 0.15,            # x / y speed while adjusting
            'hold_heading': True,         # keep the heading the robot had on entering (wz P-controller); False = wz 0
            'heading_kp': 1.5,            # rad/s of wz per rad of heading error
            'wz_max': 0.3,                # rad/s limit of that correction
            'clear_cost': 70,             # clear = the robot's cell costs less than this (and the way out aims for such a cell)
            'clear_hold_s': 0.6,          # ... for this long
            'search_m': 1.0,              # how far around the robot to look for clear floor
            'max_travel_m': 1.0,          # give up after moving this far from where it started
            'timeout_s': 20.0,
        },
        'curtain_scan': {
            'type': 'curtain_scan',
            'next_mode': 'exploring',
            'dwell_s': 3.0,               # stand still facing each edge this long (the detector confirms a spot over frames)
            'align_tol_deg': 10.0,
            'turn_timeout_s': 12.0,       # the turn to an edge did not finish: look from where it stopped
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.4,
            'turn_gain': 1.0,
        },
        'curtain': {
            'type': 'curtain',
            'next_mode': 'exploring',
            'rebase_odom': True,          # through: odom re-anchored at the centre of the tile behind (as after the stairs) ...
            'scan_after': True,           # ... then a scan, and the tile is corrected from where SLAM puts the robot
            'scan_mode': 'scanning',
            'speed_mps': 0.25,            # forward speed (vx) through the curtain
            'hstep': True,                # cross in the h-step gait only (rug_hstep, held via /step_up) ...
            'gait_settle_s': 2.0,         # ... standing this long after switching it on and after releasing it
            'align_tol_deg': 5.0,         # square to the edge within this before walking
            'align_timeout_s': 15.0,
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.4,
            'turn_gain': 1.0,
            'heading_gain': 1.0,          # wz that holds the heading while walking
            'extra_m': 0.0,               # stop this much beyond the next tile's centre
            'cross_timeout_s': 20.0,
            'low_move_s': 2.0,            # barely moving (position AND heading) this long while driving / turning: low_move_mode (0 = off)
            'low_move_m': 0.05,           # ... "barely" = less than this far ...
            'low_move_deg': 3.0,          # ... and less than this much turned
            'low_move_grace_s': 2.0,      # ... judged only this long after each phase starts (standing robot takes ~1.5 s to get going)
            'low_move_mode': 'recovery',
        },
        'stairs': {
            'type': 'stairs',
            'next_mode': 'exploring',     # over the staircase: exploring again
            'fail_mode': 'recovery',      # gave up after AI mode was on (somewhere on / near the stairs): recovery
            'ai_on_code': 0x21010528,     # /simple_cmd MotionSimpleCMD {cmd_code, type: 0}: enter AI mode
            'ai_off_code': 0x2101052B,    # ... and leave it
            'ai_settle_s': 2.0,           # stand still this long after each AI mode command (no /robot_state)
            'ai_verify': True,            # /robot_state: wait until the robot reports AI on (16,0,1) / off (basic 6)
            'ai_retry_s': 3.0,            # ... and send the AI command again if it is not there this long after the last
            'ai_max_attempts': 0,         # 0 = keep trying; N = give up after N sends
            'ai_confirm_s': 0.3,          # the reported state must hold this long
            'ai_policy_state': 0,         # robot_policy_state for AI on: 0 = AI basic gait (-1 = any)
            'center_tol_m': 0.06,         # first drive (vx / vy) to within this of the centre of the tile in front of it
            'center_speed_mps': 0.15,
            'center_min_speed_mps': 0.05,
            'center_gain': 1.0,
            'center_timeout_s': 15.0,     # not there by then: square up from where it is
            'align_tol_deg': 5.0,         # square to the edge within this before climbing
            'align_timeout_s': 15.0,      # could not square up: give up (AI mode never turned on)
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.4,
            'turn_gain': 1.0,
            'speed_mps': 0.20,            # forward speed (vx) over the stairs in AI mode
            'heading_gain': 1.0,          # wz that holds the heading while climbing
            'lateral_gain': 0.5,          # vy that keeps the robot on the line through the tile centres (0 = off)
            'lateral_max_mps': 0.08,
            'extra_m': 0.0,               # stop this much beyond the centre of the tile beyond the staircase
            'cross_timeout_s': 60.0,      # not over it by then: stop, AI mode off, fail_mode
            'stall_m': 0.05,              # climbing: less progress than this ...
            'stall_s': 10.0,              # ... in this long = stuck on the stairs
            'pose_lost_s': 3.0,           # no pose (SLAM lost on the stairs): keep walking straight this long, then stop
            'imu_lost_s': 3.0,            # IMU crossing: no fresh /imu/data this long -> stop (beyond the stairs already: counts as over them)
            # IMU crossing, level or going down: the wall beyond the staircase from the raw /scan (base_link; the map pose and
            # costmap are frozen in AI mode). Gap = nose to the nearest return in the body corridor, minus what was walked
            # since the scan was taken (~1.1 s lag). Slow from wall_slow_m, stop (= over the stairs) at wall_stop_m when level.
            'wall_stop_m': 0.15,
            'wall_slow_m': 0.60,
            'wall_creep_mps': 0.10,       # slowest walk while the nose is still down (off the last step)
            'wall_half_width_m': 0.20,    # corridor |y| (footprint half width 0.185)
            'wall_nose_m': 0.305,         # nose ahead of base_link (footprint)
            'wall_capture_level_deg': 8.0,  # stop only on a scan taken this level (nose down, the floor below is in the band)
            'wall_touch_m': 0.08,         # a gap this small, then an empty corridor (closer than range_min: inf) = touching
            'wall_touch_hold_s': 3.0,
            'wall_scan_max_age_s': 2.5,   # older scan: no wall check
            'scan_after': True,           # after the climb: scan_mode (a scan), then the tile the robot is on is corrected from SLAM
            'scan_mode': 'scanning',
            'new_map': False,             # after the climb: save the map under reports/, clear it, start a new map from the tile beyond
            'new_map_dir': '',            # ... (virtual wall against the staircase); '' = reports_dir. Needs scan_mode.
            'new_map_timeout_s': 60.0,    # save + slam_toolbox restart longer than this: give up, keep the old map
            'rebase_odom': True,          # after the climb: tell odom_tf_bridge the robot is at the centre of the reference tile beyond the
                                          # staircase (one tile past it along the steps, turned to them; AI gait odometry is off)
            'rebase_yaw': 'heading',      # rebase pose: 'heading' (line walked + squared-up heading + IMU drift) or 'stairs' (snap to the reference tile)
        },
        'roller': {
            'type': 'roller',
            'next_mode': 'exploring',     # over the roller: exploring again (the roller tile is marked explored)
            'fail_mode': 'recovery',      # gave up with the robot past the edge (on / at the roller): recovery
            'gait_settle_s': 2.0,         # stand still this long after asking for the h-step gait (and after releasing it)
            'center_tol_m': 0.10,         # first drive (turn, then vx / wz) to within this of where it squares up
            'center_speed_mps': 0.15,
            'center_min_speed_mps': 0.05,
            'center_gain': 1.0,
            'center_turn_deg': 30.0,
            'center_heading_gain': 1.5,
            'center_timeout_s': 15.0,     # not there by then: square up from where it is
            'align_tol_deg': 4.0,         # square to the roller within this before stepping over it
            'approach_angle_deg': 20.0,   # approach with the body this far off the roller plane (+ = left), vx / vy walk the straight line
            'approach_until_m': 0.0,      # ... until the roller tile's edge line; then square up and cross
            'align_timeout_s': 15.0,      # could not square up: give up (never crossed), back to the mode before
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.4,
            'turn_gain': 1.0,
            'speed_mps': 0.20,            # forward speed (vx) over the roller in the h-step gait
            'heading_gain': 1.0,          # wz that holds the heading while crossing
            'lateral_gain': 0.5,          # vy that keeps the robot on the line through the tile centres (0 = off)
            'lateral_max_mps': 0.08,
            'extra_m': 0.0,               # stop this much beyond the centre of the tile beyond the roller
            'recenter': True,             # at the edge after the roller tile: stop, recentre on its midpoint with omni vx / vy / wz
            'recenter_tol_m': 0.06,
            'recenter_speed_mps': 0.08,
            'recenter_gain': 0.8,
            'recenter_timeout_s': 8.0,    # not there by then: cross on from where it is
            'restore_heading': True,      # at the midpoint also set the yaw setpoint to the mission's original heading (held to the end)
            'restore_max_deg': 100.0,     # ... unless it is further than this from the way across (crossing back over a roller)
            'recenter_yaw_tol_deg': 5.0,  # recentring ends when the heading is within this of the setpoint too
            'cross_timeout_s': 40.0,
            'stall_m': 0.05,              # crossing: less progress than this ...
            'stall_s': 6.0,               # ... in this long = stuck at the roller: back off and try again
            'max_attempts': 2,            # crossing tries (back off to where it squared up, square up again, cross)
            'back_speed_mps': 0.12,       # backing off after a stuck try
            'back_timeout_s': 10.0,
            'pose_lost_s': 3.0,           # no pose while crossing: keep walking straight this long, then stop
        },
        'backtracking': {
            'type': 'backtracking',
            'traversal': 'route',         # NavigateThroughPoses back to the split tile, then to its untravelled tile
            'waypoints_per_tile': 3,      # waypoints on the straight line between each pair of tile centres on the way
            'align_at_tiles': True,       # at each tile centre turn in place to face the next tile, then drive that hop
            'align_tol_deg': 5.0,
            'align_timeout_s': 15.0,      # not facing it by then: drive on anyway
            'align_settle_s': 0.6,        # squared up: stand still this long before Nav2 drives the hop
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.4,
            'turn_gain': 1.0,
            'min_clearance_m': 0.25,      # goal point in a tile: this far from walls
            'path_check_timeout_s': 10.0,
            'goal_timeout_s': 120.0,
            'face_away_from_walls': True,
            'scan_after_reach': True,     # reached the untravelled tile: scan_mode (which hands over to exploring) ...
            'scan_mode': 'scanning',
            'next_mode': 'exploring',     # ... else straight back to exploring (also after a failure)
            'stairs_mode': 'stairs',      # split tile in front of a forced staircase (exploring.stairs_backtrack): climb it
            'stair_min_facing': 0.5,      # as modes.exploring.stair_min_facing
        },
        'recovery': {
            'type': 'recovery',
            'traversal': 'recovery_mppi',     # Nav2 + holonomic RecoveryMPPI controller (strafes with vx / vy)
            'fallback_traversal': 'omni',     # that failed: straight x / y velocity to the same centre ('' = none)
            'keep_gait': True,            # keep the gait the robot already has while recovering (/gait_freeze)
            'step_up_gait': True,         # ... except a step up: a higher recovery target still switches to h-step (/step_up)
            'candidates': 3,
            'min_distance_m': 0.25,       # skip centres the robot is already on
            'adjacent_only': True,        # only the robot's own tile / tiles sharing a side with it (never through another tile)
            'min_clearance_m': 0.25,
            'attempt_timeout_s': 45.0,
            'clear_costmaps': True,
            'clear_settle_s': 1.0,
            'give_up_after': 3,           # consecutive recoveries without reaching a new tile; 0 = never give up
            'next_mode': 'scanning',
            'fail_mode': 'scanning',
        },
        'ending': {
            'type': 'ending',
            'traversal': 'direct',            # NavigateToPose to the ending tile's centre
            'fallback_traversal': 'omni',     # failed / stopped off-centre: straight x / y velocity to the centre ('' = none)
            'centre_tolerance_m': 0.10,       # Nav2 arrived further than this from the centre: finish with the fallback
            'attempt_timeout_s': 60.0,
            'clear_costmaps': False,
            'give_up_after': 0,               # failed visits (each followed by a rescan) before completing anyway; 0 = never
            'fail_mode': 'scanning',
        },
        'complete': {'type': 'complete'},
    },
}


def merge_flat(defaults: Dict[str, Any], flat: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy of defaults with dotted keys (a.b.c) set to the given values."""
    cfg = copy.deepcopy(defaults)
    for key, value in flat.items():
        if key in ('use_sim_time',) or key.startswith('qos_overrides'):
            continue
        parts = key.split('.')
        node = cfg
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = clean(value)
    return cfg


def clean(value):
    """ROS string arrays use [''] for an empty list."""
    if isinstance(value, (list, tuple)):
        return [v for v in value if v != '']
    return value


def mode_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The configuration of mode `name`, with the defaults of its type underneath."""
    own = dict(cfg.get('modes', {}).get(name, {}))
    mtype = own.get('type', name)
    base = dict(DEFAULTS['modes'].get(mtype, {}))
    base.update(own)
    base['type'] = mtype
    return base
