# lite3_explorer — tile exploration with a mission manager

ROS 2 (Humble) package for the DEEP Robotics **Lite3**, derived from `lite3_kitchen_explorer`. The SLAM map is cut
into 1 m × 1 m tiles and the dog visits **every identified tile**. One node, `mission_manager`, runs the whole
mission as a set of switchable **modes**. Inside a mode, **where** to go next comes from *exploration profiles*
(tiles by default; frontiers optional) and **how** to drive there from *traversal profiles*.

What changed from `lite3_kitchen_explorer`:

| `lite3_kitchen_explorer` | `lite3_explorer` |
|---|---|
| `frontier_explorer` + `environment_reporter`, two nodes coordinating over topics and `set_active` | one `mission_manager` node with a mode state machine (`mission/`) |
| frontiers chosen when no tile was reachable; done when no frontier left | tiles by default, frontiers an optional fallback profile; done when the **configurable end condition** is met (default: all identified tiles explored) |
| next tile = nearest tile Nav2 can reach | next tile must **share a side with an explored tile**; tiles Nav2 repeatedly can't plan to are **eliminated** |
| stuck → blacklist the goal, try another | stuck → **recovery mode**: drive to the nearest tile centre, then rescan |
| traversal hard-coded (FollowWaypoints) | named **traversal profiles** per mode, and per tile |
| AI could place the first tile / pick targets (off by default) | not ported: tiles snap to the walls, target = the profile's top-ranked candidate (`choose_target()` hook left for an AI picker) |

The AI tile report is unchanged: when a new tile is reached, `lite3_llm_tools`'s `llm_client` is asked in the
background what it looks like, and the answers are compiled into `~/2d/reports/tile_report_<run>.md`.

## Modes

```
 idle ──auto start / set_active(true)──► scanning ──scan finished──► exploring ──end condition met──► complete
  ▲                                       ▲     ▲                      │    │
  └── set_active(false) (from any mode)   │     └────tile reached──────┘    │ stuck (no motion / in the costmap)
                                          │                                 │ or no Nav2 path to any unexplored tile
                                          └──recovered / gave up── recovery ◄┘   (drive to the nearest tile centre)
```

| Mode | What it does | Phases |
|---|---|---|
| `idle` | Nothing moves. With `auto_start` it leaves once TF, `/map`, Nav2 (spin, planner) and the AI client are up, or after `start_wait_s`. After a pause it only leaves on a command. | `waiting_start`, `paused` |
| `scanning` | First time: waits for the map around the robot, **places the tile grid** (snapped to nearby walls) and does the opening sweep −90…+90° in the flat gait. Later: one in-place turn (360°) in the crawl gait, skipped where the body has no room. Then re-classifies the tiles. | `wait_map`, `settle`, `turning`, `done` |
| `exploring` | **Checkpoint**: checks the end condition. Asks its exploration profiles in order (default `[tiles]`) for candidates, has Nav2 check a path to each, and drives to the best reachable one with a traversal profile. If a profile has nothing reachable, the next one is asked. Reached → `scanning`. Failed → the profile records it (tile failure / frontier blacklist) and it re-decides. | `decide`, `checking_paths`, `driving`, `wait_planner`, `wait_traversal` |
| `quickscan` | Optional (`quickscan_on_edge`). Entered from `exploring` when the robot is within `quickscan_edge_margin_m` (0.15 m) of a tile boundary, once per tile. Cancels the Nav2 goal, waits `settle_s`, plays the `quickscan` robot animation and waits for it to end, then returns to `exploring`, which re-decides (the cancelled goal is normally re-chosen; it does not count as a failure). Skipped if `animation_player` doesn't start it within `start_timeout_s`; cancelled after `timeout_s`. | `settle`, `playing` |
| `stairs` | Optional (`use_stairs:=true`, `./explorer.sh --stairs`). Entered from `exploring` at a decision when a **staircase tile** (found by `stair_detector` in the depth image) shares a side with the current tile and its steps go up away from it. Drives to the current tile's centre, squares up to that side, turns the robot's **AI mode on** (`/simple_cmd` `0x21010528`), walks forward over the staircase tile(s) to the centre of the tile beyond, turns **AI mode off** (`0x2101052B`), walls the staircase off (virtual walls on all four edges: never entered again) and returns to `exploring`. Stuck / timed out on the stairs: AI mode off, walled off, `fail_mode` (`recovery`). Any other exit (operator command, shutdown) also turns AI mode off. While AI mode is on, `/ai_mode` is true: `gait_switcher` sends no gait and `stair_detector` pauses. | `center`, `align`, `ai_on`, `cross`, `ai_off` |
| `roller` | Optional (`use_rollers:=true`, `./explorer.sh --roller`). A **roller tile** has a cylinder (~0.17 m high) lying across its middle: the 2D lidar maps it as a wall, so the tile looks like half a tile. `roller_detector` finds it in the depth image (a short, thin, long ridge with the floor hidden behind it) and the 3D lidar confirms it (only low returns there, open space behind). Entered from `exploring` at a decision (before the end condition is checked) when a roller tile shares a side with the current tile and its roller lies across the way, or the robot stands in front of the roller on its tile. Drives to the centre of the tile in front, squares up to the roller, holds the **h-step gait** (`/step_up` true: `gait_switcher` sends `rug_hstep`; `/roller_mode` true pauses the detector), walks forward with vx over the roller to the centre of the tile beyond, releases the gait, marks the **roller tile explored** (on the trail, tile report written) and returns to `exploring`. Stuck at the roller: backs off, squares up and tries again (`max_attempts`); given up past the edge: `fail_mode` (`recovery`), before it: the mode before. Crossing a roller from a tile with other untravelled tiles beside it makes that tile the last split; `backtracking` comes back over the roller the same way it walks back through a curtain (drives to the tile before it, phase `to_roller`, hands over to `roller`, which hands back with `roller_ok`; not crossed: another way is looked for). | `center`, `align`, `gait_on`, `cross`, `back`, `gait_off` |
| `recovery` | Clears the costmaps, then drives to the **nearest tile centre** (visited or to-visit tiles, excluding the tile it got stuck on and the centre it already stands on), trying up to `candidates` of them. Then `scanning`. After `give_up_after` recoveries in a row without reaching a new tile, the failing tiles are marked unreachable. | `clearing`, `driving`, `wait_traversal`, `give_up` |
| `complete` | The end condition is met. Publishes `/exploration_complete`, writes `tiles.json` and the final tile report. `set_active(true)` starts again. | `done` |

**The stuck supervisor** watches the modes in `stuck.watch_modes` (default `exploring`). If unexplored tiles remain
and either trigger fires, it switches to `recovery` (if none remain, it checks the end condition instead):

- `no_motion`: the robot has been driving for `window_s` (20 s) and moved less than `min_motion_m` (0.15 m).
- `costmap`: the robot's centre has been on a global-costmap cell of cost ≥ `costmap_cost` (99 = inscribed) for
  `costmap_hold_s` (3 s). From there Nav2's planner can't plan out.

`exploring` also goes to `recovery` when **no candidate of any profile has a Nav2 path** (`on_no_reachable_tile`).

**Tile states:** `visited` (explored), `to_visit`, `blocked`, `unknown` (not identified yet: doesn't count), and
`unreachable`: a `to_visit` tile that was **eliminated**: no Nav2 path `eliminate_after` decisions in a row, no point
clear of the walls, `tiles.max_failures` failed attempts (Nav2 abort, stuck, timeout), or repeated recoveries.
Reaching an eliminated tile by any route clears that status. `staircase` (drawn yellow): a tile holding at least
`tiles.stair_min_points` confirmed stair cells: never a Nav2 target, never counted as unexplored; only the `stairs` mode
enters it. `roller` (drawn white): a tile holding at least `tiles.roller_min_points` confirmed roller spots away from its
edges, `tiles.roller_min_lidar` of them confirmed by the lidar: never a Nav2 target, never counted as unexplored, never on
a Nav2 route back; once the `roller` mode has crossed it, it is `visited`.

## Robot animations

Timed sequences of `/simple_cmd` (`MotionSimpleCMD`) commands, defined in `animations.py` and played by the `animation_player` node.
Two categories:

- **movement** (`MovementAnimation`): moves the robot around; none defined yet.
- **pose** (`PoseAnimation`): enters **pose mode** (`0x21010D05`) when it starts and **move mode** (`0x21010D06`) when it ends (also if cancelled).
  The robot only holds a pose while the pose commands keep being published, so the player republishes them at `publish_rate_hz` (20) for the whole animation.
  Keyframes ramp linearly between poses (`move_s` per move) and a pause is a keyframe that repeats the previous pose.

| Animation | Category | Sequence |
|---|---|---|
| `quickscan` | pose | pose mode → (optional lean forward, `quickscan.lean`, default 0) → pause → yaw left → yaw right → pause → centred → move mode |
| `scenescan` | pose | pose mode → yaw left, hold → centre, hold → yaw right, hold → centred → move mode. Publishes `/animation_marker` (`String`: `left`, `centre`, `right`) at the end of each hold; the scene mode takes a picture (one AI call) at each. Parameters `scenescan.pause_s` (1.0), `scenescan.move_s` (1.0), `scenescan.yaw` (20000). |

Pose commands: lean `0x21010130` (±20000, positive = forward), yaw `0x21010135` (values within ±9553 are a dead zone and ignored, so use more, default 20000; positive = right, so left is negative).

`animation_player` parameters: `quickscan.pause_s` (0.5, the pause at the forward lean and at the right extreme), `quickscan.move_s` (1.0),
`quickscan.lean` (0 = no lean; up to 20000 to lean forward), `quickscan.yaw` (20000), `mode_settle_s` (1.0, wait after pose mode), `mode_repeats` (3, mode commands are sent several times
in case one is lost), `publish_rate_hz`. Topics: `/play_animation` (`std_msgs/String`: an animation name, or `stop`), `/animation_active` (`Bool`).
`gait_switcher` sends no gait commands while `/animation_active` is true and sends the gait that was in use before the animation again afterwards.

**Scene mode:** with `modes.scene.quickscan: true` (default) the robot, once aligned and settled, plays `scenescan` and takes three pictures (left, centre, right) instead of one; the descriptions are joined in the tile report and the scene type is the commonest one named. If the animation cannot start (no `animation_player`) it falls back to one picture.

```bash
ros2 run lite3_explorer animation_player
ros2 topic pub --once /play_animation std_msgs/msg/String "{data: quickscan}"
```

**Test an animation** (starts the player, plays it once, exits when it ends; exit code 1 if it never started or timed out):
```bash
ros2 launch lite3_explorer test_animation.launch.py                                   # quickscan
ros2 launch lite3_explorer test_animation.launch.py animation:=quickscan pause_s:=1.0 lean:=15000 yaw:=4000
```
Arguments: `animation` and the optional overrides `pause_s`, `move_s`, `lean`, `yaw`.

**Quickscan on tile edges:** `./scripts/explorer.sh --quickscan`, or `quickscan_on_edge:=true` on `exploration.launch.py`
(also starts `animation_player`), or `modes.exploring.quickscan_on_edge: true` in `mission.yaml`.

**Add an animation:** subclass `PoseAnimation`, return the keyframes `[(seconds, {cmd_code: size}), ...]`, and add the class to `ANIMATIONS`.

## Exploration profiles (where to go)

`modes.exploring.exploration` lists the profiles to ask, in order; later ones are fallbacks. Profiles live under
`explorations.<name>`, exactly like traversal profiles.

**`tiles`** (default). At each decision:
1. Take the unexplored tiles with a goal point clear of walls (tiles with none are eliminated).
2. **Adjacency**: keep the tiles that share a side with an explored (visited) tile. Each gets a level:
   `0` open side with the current tile, `1` open side with any visited tile, `2` touches a visited tile only across a
   wall, `3` touches none. Kept: level ≤ `max_adjacency` (default 2). If none qualifies, the rest are offered
   (`fallback_any`, default off: neighbours that are only briefly "no path" must not make the robot jump to a non-adjacent tile).
3. **Reachability**: Nav2 plans a path to each. Tiles without a path are dropped and get a strike; after
   `eliminate_after` (2) strikes in a row a tile is eliminated. Strikes are only given when some other tile did get a
   path, so a planner failing from where the robot stands (stuck in an inflated wall) eliminates nothing — that goes
   to recovery instead.
4. **Ranking**: adjacency level first (`prefer_current: false` treats levels 0 and 1 alike), then Nav2 path length.

**`frontier`** (optional, off by default). Frontier clusters (free cells next to unknown space), goal point clear of
walls, ranked by `distance_weight × path length − size_weight × size`. Failed goals are blacklisted; a goal is dropped
once its frontier gets mapped on the way. Use it as a fallback to open up parts of the arena the lidar hasn't
mapped well enough for tiles to be identified:

```yaml
modes:
  exploring: {exploration: [tiles, frontier], kind_traversals: ['frontier:direct']}
end_condition:                        # don't stop while frontiers remain
  combine: all
  conditions: [all_tiles_explored, no_frontiers_left]
```

If Nav2's action servers are not up, the modes **wait** (`wait_traversal`) rather than failing tiles.

## End conditions

Configured under `end_condition`. `conditions` lists them, `combine` is `any` (stop when one is met) or `all`.

| Name | Checked | Met when | Settings |
|---|---|---|---|
| `all_tiles_explored` | at checkpoints (each `exploring` decision, after a scan) | no `to_visit` tile left and at least `min_visited` visited | `min_visited` (1), `ignore_unreachable` (true) |
| `no_frontiers_left` | at checkpoints | no frontier cluster with a clear goal point left | `min_cluster_cells` (8), `min_clearance_m` (0.30) |
| `time_limit` | every tick | mission running ≥ `seconds` | `seconds` (0 = off) |
| `max_tiles_visited` | every tick | ≥ `count` tiles visited | `count` (0 = off) |

E.g. finish when all tiles are done **or** after 10 minutes:
```yaml
end_condition:
  combine: any
  conditions: [all_tiles_explored, time_limit]
  time_limit: {seconds: 600.0}
```

## Running

```bash
cd ~/2d-challenge && colcon build --packages-select lite3_explorer && source install/setup.bash
ros2 run lite3_explorer preflight                      # topics, frames, TF, clocks, /cmd_vel consumer
ros2 launch lite3_explorer exploration.launch.py server_url:=http://192.168.10.141:8000 group_id:=team7
# or everything in tmux with status panes:
SERVER_URL=http://192.168.10.141:8000 GROUP_ID=team7 src/lite3_explorer/scripts/run_explorer_tmux.sh
```

Launch arguments are the same as before (`use_slam`, `use_nav2`, `use_terrain`, `use_gait`, lidar TF, `use_ai`,
`server_url`, ...), plus `mission_params:=<yaml>`, `auto_start:=true|false`, `use_mission:=false` and
`quickscan_on_edge:=true` (see [Robot animations](#robot-animations)), `use_curtains:=true` and `use_stairs:=true`
(starts `stair_detector`, `config/stairs.yaml`, and turns on `modes.exploring.stairs`), `use_rollers:=true` (starts
`roller_detector`, `config/roller.yaml`, and turns on `modes.exploring.rollers`).
`mission_only.launch.py` skips odom TF, cloud→scan, SLAM and Nav2 for robots that already run them.

### Local controller: pure pursuit or holonomic MPPI

`controller:=rpp` (default) uses Nav2's regulated pure pursuit (`config/nav2_params.yaml`): it only drives forward
(vx) and turns (wz). `controller:=mppi` uses Nav2's **MPPI** controller with the **Omni** (holonomic) motion model
(`config/nav2_params_mppi.yaml`): it also commands sideways speed vy, so the robot can strafe along a path and turn while
moving (the Lite3 takes vx, vy and wz on `/cmd_vel`).

```bash
ros2 launch lite3_explorer exploration.launch.py controller:=mppi
```

The MPPI file differs from `nav2_params.yaml` only in `controller_server`: the `FollowPath` block, the
`min_y_velocity_threshold` (0.001, so vy is not thrown away) and the velocity smoother's vy limits (0.3 m/s). Main knobs
in `FollowPath`: `vx_max` / `vy_max` / `wz_max` (speed limits), `batch_size` (sampled trajectories per cycle; lower it
if the controller misses its 10 Hz rate), `time_steps` x `model_dt` (the horizon; `model_dt` must equal the controller
period, 0.1 s), the critic weights (`PathAlignCritic` / `PathFollowCritic` keep it on the path, `CostCritic` keeps it
off walls using the body footprint, `GoalCritic` / `GoalAngleCritic` shape the arrival). `visualize: true` publishes the
sampled and the chosen trajectories on `/trajectories` for RViz. Use another file with `nav2_params:=/path/file.yaml`.

Commands while it runs:
```bash
ros2 service call /mission_manager/set_active std_srvs/srv/SetBool "{data: false}"    # pause (idle)
ros2 service call /mission_manager/set_active std_srvs/srv/SetBool "{data: true}"     # start / resume
ros2 topic pub --once /mission_manager/mode_cmd std_msgs/msg/String "{data: recovery}" # force a mode
ros2 service call /mission_manager/report_now std_srvs/srv/Trigger                     # describe this tile now
ros2 topic echo /mission_manager/status --field data                                   # mode, target, tiles, end condition
```

Topics: `~/mode` (`mode:phase`, latched), `~/status` (JSON, 1 Hz), `~/events` (every decision as JSON, also
`run_dir/events.jsonl`), `~/current_goal`, `~/next_target`, `/tile_map`, `/tile_report`, `~/tile_markers`
(visited blue, to-visit green, blocked red, unreachable purple; target orange, recovery target magenta), `~/markers`
(tile descriptions), `/exploration_complete`.

## Configuration: `config/mission.yaml`

Every setting is in `config/mission.yaml` with a comment; defaults live in `lite3_explorer/mission/config.py`. The
node accepts **any** parameter key (`automatically_declare_parameters_from_overrides`), so new modes, profiles and
end-condition settings need no code change. ROS 2 can't type an empty list: write `['']` for "none".

### Traversal profiles

A mode drives through a named profile, never Nav2 directly:

```yaml
traversals:
  route:                                # FollowWaypoints: 3 waypoints along the Nav2 path, then the tile centre
    type: nav2_route
    waypoints: 3                        # evenly spaced along the path (after skip_start_m), driven through
    skip_start_m: 0.20
    drop_backward_waypoints: false      # true: drop waypoints facing away from the goal (then fewer than 3)
  direct: {type: nav2_pose}             # NavigateToPose straight to the goal (behavior_tree: optional BT XML)
modes:
  exploring: {traversal: route, tile_traversals: ['3,1:direct'], kind_traversals: ['frontier:direct']}
  recovery:  {traversal: direct}
```

For a tile, the goal is the **exact tile centre** when it is at least `min_clearance_m` from the walls, otherwise
the clear point nearest the centre. With the `route` profile Nav2 gets 3 intermediate waypoints spaced evenly
along its own planned path (so they follow the corridor round walls), each facing the next, then the centre facing
along the tile grid. E.g. a 1 m hop: waypoints at 0.4, 0.6 and 0.8 m, goal at 1.0 m.

### Several instances of one mode type

```yaml
mission:
  modes: [idle, scanning, exploring, explore_careful, recovery, complete]
modes:
  explore_careful: {type: exploring, traversal: direct, scan_after_reach: false}
```
Switch to it with `mode_cmd`, or point another mode's `next_mode` / `scan_mode` at it. Every mode a mode can
switch to must exist in `mission.modes`; the node checks this at start-up.

## Extending

All mission logic in `lite3_explorer/mission/` is ROS-free and unit tested (`test/test_mission.py` runs whole
missions in simulated corridors and rooms, including a fake lidar that reveals the map as the robot scans).

- **A new traversal** (e.g. stair climbing): subclass `mission.traversal.Traversal` (`available`, `start`, `poll`,
  `cancel`), decorate it with `@register_traversal('stairs')`, import its module in `mission_manager.py`, add
  `traversals.stairs: {type: stairs}` and select it per mode or per tile. `ExploringMode.traversal_for()` is the
  hook for choosing it automatically (e.g. from terrain severity).
- **A new mode**: subclass `mission.core.Mode` (`enter`, `tick` → `Transition` or None, `exit`, `targets`), decorate it
  with `@register_mode('name')` in `mission/modes.py`, and list an instance in `mission.modes`.
- **A new end condition**: subclass `mission.end_conditions.EndCondition` (`evaluate(snapshot) -> (met, why)`,
  `continuous = True` to check it every tick), decorate it with `@register('name')`, and list it in
  `end_condition.conditions`.
- **A new exploration strategy**: subclass `mission.exploration.Exploration` (`candidates`, `select`, optionally
  `still_valid`, `on_reached`, `on_failed`), decorate it with `@register_exploration('name')`, add
  `explorations.<x>: {type: name}` and list it in `modes.exploring.exploration`.
- **A different final pick** (e.g. the AI): override `ExploringMode.choose_target(reachable)`. Tile targets keep the
  `lite3_llm_tools` `choose_target` wire format.

## Files

| File | Role |
|---|---|
| `mission_manager.py` | The ROS node: TF, maps, costmaps, Nav2 spin / planner clients, tile reports, RViz, services. It is the `ctx` the modes use. |
| `config/nav2_params_mppi.yaml` | Nav2 parameters with the holonomic MPPI local controller (`controller:=mppi`). |
| `nav2_traversal.py` | `nav2_pose` and `nav2_route` traversal types. |
| `mission/core.py` | `Mode`, `Transition`, `ModeMachine` (switching, stuck supervisor, end-condition checks). |
| `mission/modes.py` | `idle`, `scanning`, `exploring`, `quickscan`, `curtain`, `stairs`, `roller`, `scene`, `mppi`, `backtracking`, `recovery`, `complete`. |
| `stairs_core.py`, `stair_detector.py`, `config/stairs.yaml` | Staircase detection in the depth image (treads at evenly spaced heights, each farther than the one below; tuned on `bags/data/stairs`), staircase tiles, the side to climb from; the optional node publishing `/stairs/points`. |
| `roller_core.py`, `roller_detector.py`, `config/roller.yaml` | Roller detection: depth camera (low, thin, long ridge with the floor hidden behind it) confirmed by the 3D lidar (low returns only, open behind); tuned on `bags/data/roller`. Roller tiles, the side to cross from; the optional node publishing `/roller/points`. |
| `animations.py`, `animation_player.py`, `animation_test.py` | Robot animations (movement / pose), the node that plays them, and the one-shot test node (`launch/test_animation.launch.py`). |
| `mission/exploration.py` | Exploration strategies (`tiles`, `frontier`) and their registry. |
| `mission/stuck.py`, `mission/end_conditions.py`, `mission/traversal.py`, `mission/config.py` | Stuck detector, end conditions, traversal interface, config defaults. |
| `frontiers.py` | Frontier clustering (only used by the optional `frontier` profile and `no_frontiers_left`). |
| `tile_tracker.py` | Tile grid, classification, trail, failures, unreachable tiles. |
| `tiles.py` (incl. adjacency levels), `map_grid.py`, `approach.py`, `turn_check.py`, `tile_reports.py`, `llm_link.py` | Tile geometry, map helpers, waypoint routes, turn-room check, tile report, AI link (from the kitchen explorer). |
| `gait_switcher.py` | Terrain gait switching; pauses while an animation runs and while the robot is in AI mode (`/ai_mode`). |
| `terrain_gait.py`, `odom_tf_bridge.py`, `preflight.py`, `survey_monitor.py` | Unchanged helpers. |

Run files: `~/lite3_survey_runs/run_<time>/events.jsonl` (every mode switch, target, failure, stuck event,
recovery), `tiles.json`, and the AI client's images and answers.
