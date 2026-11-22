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
    'output_dir': '~/lite3_survey_runs',
    'reports_dir': '~/2d/reports',
    'scene_images_dir': '~/2d/images/runs',   # the images sent to the AI for scene tiles: <dir>/<run>/scene_<tile>_q<n>.jpg (+ .json answer)
    'use_ai': True,                       # AI tile descriptions via lite3_llm_tools/llm_client
    'terrain_classify_service': '/terrain_classifier/classify',   # asked for the tile being driven to ('' = never)
    'terrain_wait_s': 3.0,            # no terrain answer this long after asking = the floor of that tile cannot be seen
    'llm_report_timeout_s': 180.0,
    'llm_report_retries': 5,
    'robot_length_m': 0.61,               # turn-room check (match the local costmap footprint)
    'robot_width_m': 0.37,
    'turn_room_margin_m': 0.0,

    'tiles': {
        'size_m': 1.0,
        'reach_radius_m': 0.30,           # a tile is reached (explored) this close to its centre
        'max_failures': 2,                # failed attempts before a tile is given up on (unreachable); 0 = never
        'min_known': 0.5,                 # share of a tile that must be scanned before it becomes a tile
        'scene_min_known': 0.95,          # scene tile: scanned at least this much ...
        'scene_max_free': 0.80,           # ... but less than this share of it traversable (drawn blue hatched in RViz)
        'stair_min_points': 4,            # confirmed stair cells (10 cm, /stairs/points) that make a tile a staircase tile
    },

    'mission': {
        # Mode instances to build. Each name is configured under modes.<name>; modes.<name>.type picks the class
        # (defaults to the name). Add your own, e.g. 'exploring_stairs' with type: exploring.
        'modes': ['idle', 'scanning', 'exploring', 'quickscan', 'curtain', 'stairs', 'scene', 'mppi', 'backtracking',
                  'recovery', 'complete'],
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
        'route': {'type': 'nav2_route',   # FollowWaypoints: N waypoints along the Nav2 path, then the tile centre
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
                 'heading_kp': 1.5, 'wz_max': 0.3, 'block_cost': 100, 'lookahead_m': 0.30, 'timeout_s': 30.0,
                 'stall_s': 6.0, 'min_progress_m': 0.05},
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
            'opening_setpoints_deg': [0.0, 80.0, -160.0, -80.0],   # opening spin: yaw setpoints relative to the start heading, then back to 0
            'opening_retries': 5,           # the opening scan must finish before exploring
            'opening_min_turn_deg': 160.0,    # the opening spin may stop early (map not sparse) only after turning this far
            'return_tolerance_deg': 3.0,  # after the opening sweep turn back to the starting heading
            'time_allowance_s': 20.0,
            'use_crawl': True,            # crawl gait while turning (gait_switcher /spin_crawl)
            'settle_s': 1.0,
            'check_room': True,           # skip the turn where the body would hit the local costmap
            'map_wait_s': 30.0,           # max wait for the map around the start before placing the tile grid
            'next_mode': 'exploring',
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
            'curtain_edge_tol_m': 0.30,   # curtain spots within this of the tile edge line count as on the edge
            'curtain_min_span_m': 0.40,   # ... spread over at least this much of it
            'curtain_last': True,         # another reachable tile beside the curtain's tile: go there first, the tile is the
                                          # last split, backtracking walks through the curtain later (needs backtracking)
            'stairs': False,              # use_stairs:=true: a staircase tile shares a side with the current tile -> stairs
            'stairs_mode': 'stairs',      # mode (AI mode on, climb over it into the tile beyond, AI mode off, wall it off)
            'stair_min_facing': 0.5,      # climb only from a side the steps go up from (cos of ascent vs the way across)
            'backtracking': True,         # two or more tiles to travel to -> the tile is the last split; no exploration
                                          # target left but tiles unexplored -> backtracking mode (else retry_wait)
            'backtracking_mode': 'backtracking',
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
            'align_tol_deg': 6.0,         # turn in place until facing the scene tile's centre within this
            'commit_deg': 150.0,          # target further round than this: keep the direction the turn started in (no flip at 180 deg)
            'align_timeout_s': 15.0,
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.6,
            'turn_gain': 1.0,
            'settle_s': 1.0,              # stand still facing it before the AI looks (sharp image)
            'llm_attempts': 2,            # AI description tries; after the last the tile is marked explored anyway
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
        'curtain': {
            'type': 'curtain',
            'next_mode': 'exploring',
            'speed_mps': 0.25,            # forward speed (vx) through the curtain
            'align_tol_deg': 5.0,         # square to the edge within this before walking
            'align_timeout_s': 15.0,
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.6,
            'turn_gain': 1.0,
            'heading_gain': 1.0,          # wz that holds the heading while walking
            'extra_m': 0.0,               # stop this much beyond the next tile's centre
            'cross_timeout_s': 20.0,
        },
        'stairs': {
            'type': 'stairs',
            'next_mode': 'exploring',     # over the staircase: exploring again
            'fail_mode': 'recovery',      # gave up after AI mode was on (somewhere on / near the stairs): recovery
            'ai_on_code': 0x21010528,     # /simple_cmd MotionSimpleCMD {cmd_code, type: 0}: enter AI mode
            'ai_off_code': 0x2101052B,    # ... and leave it
            'ai_settle_s': 2.0,           # stand still this long after each AI mode command
            'center_tol_m': 0.06,         # first drive (vx / vy) to within this of the centre of the tile in front of it
            'center_speed_mps': 0.15,
            'center_min_speed_mps': 0.05,
            'center_gain': 1.0,
            'center_timeout_s': 15.0,     # not there by then: square up from where it is
            'align_tol_deg': 5.0,         # square to the edge within this before climbing
            'align_timeout_s': 15.0,      # could not square up: give up (AI mode never turned on)
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.6,
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
        },
        'backtracking': {
            'type': 'backtracking',
            'traversal': 'route',         # FollowWaypoints back to the split tile, then to its untravelled tile
            'waypoints_per_tile': 3,      # waypoints on the straight line between each pair of tile centres on the way
            'align_at_tiles': True,       # at each tile centre turn in place to face the next tile, then drive that hop
            'align_tol_deg': 5.0,
            'align_timeout_s': 15.0,      # not facing it by then: drive on anyway
            'min_turn_rad_s': 0.2,
            'max_turn_rad_s': 0.6,
            'turn_gain': 1.0,
            'min_clearance_m': 0.25,      # goal point in a tile: this far from walls
            'path_check_timeout_s': 10.0,
            'goal_timeout_s': 120.0,
            'face_away_from_walls': True,
            'scan_after_reach': True,     # reached the untravelled tile: scan_mode (which hands over to exploring) ...
            'scan_mode': 'scanning',
            'next_mode': 'exploring',     # ... else straight back to exploring (also after a failure)
        },
        'recovery': {
            'type': 'recovery',
            'traversal': 'recovery_mppi',     # Nav2 + holonomic RecoveryMPPI controller (strafes with vx / vy)
            'fallback_traversal': 'omni',     # that failed: straight x / y velocity to the same centre ('' = none)
            'tile_statuses': ['visited', 'to_visit'],   # which tiles' centres are recovery targets
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
