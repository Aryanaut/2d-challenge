# lite3_sim: maze simulator for lite3_explorer

Tests **exploration, the tile grid, curtain mode and stairs mode** of `lite3_explorer` in mazes you draw yourself, and
shows everything in RViz. It does **not** simulate gaits / gait switching, terrain, the climb itself (no pitch or
height), pose / scene animations (quickscan) or the AI.

`lite3_sim` stands in for everything *around* the mission manager: the robot, the lidar, slam_toolbox and Nav2.
Your **unmodified** `mission_manager` runs against it. It gets the same topics, TF, actions and services it uses on
the dog, so what you debug here is the real node.

| On the robot | In lite3_sim |
|---|---|
| Lite3 + leg odometry | kinematic body 0.61 x 0.37 m that collides with wall and slit panels (not curtains); `/cmd_vel` (vx, vy, wz) |
| Livox → `cloud_to_scan` `/scan` | ray-cast 2D scan, range 0.35–4 m, curtains look like walls |
| slam_toolbox `/map` | log-odds map from the scans with a perfect pose; **map frame = start pose**, like SLAM |
| Nav2 global / local costmap | SLAM map + `/virtual_walls` + inflation, your footprints and radii |
| NavFn `compute_path_to_pose` | grid Dijkstra: inscribed = wall, unknown allowed, 0.4 m goal tolerance |
| RPP / MPPI controller, `navigate_to_pose`, `follow_waypoints` | pure pursuit (rotate-to-heading, collision detection, 3 s failure tolerance), or omni (vx, vy, wz); goal checker 0.4 m / 0.6 rad; progress checker 0.3 m in 30 s; missed waypoints |
| bt_navigator default tree | on a planner / controller failure: clear costmaps → spin 90° → wait 5 s → back up 0.3 m (round robin), 6 retries, then abort |
| behavior_server `spin` | relative turn at ≤ 1 rad/s, aborts on collision / time allowance |
| `curtain_detector` (depth camera) | curtain points in the camera's 81° view, line of sight, ≥ 30 % of the image width → `/curtain/points` |
| `stair_detector` (depth camera) | tread and floor points in the camera's 81° view (2.5 m, past walls) run through lite3_explorer's real `stairs_core` → `/stairs/points`; paused in AI mode |
| Lite3 AI mode (`/simple_cmd` `0x21010528` / `0x2101052B`) | the robot gets over stair risers only in AI mode (`/ai_mode` when `transfer_interfaces` is not installed) |

## Build and run

```bash
cd ~/2d-challenge
colcon build --packages-select lite3_explorer lite3_sim && source install/setup.bash

ros2 launch lite3_sim sim.launch.py                               # mazes/arena_example.yaml + mission_manager + RViz
ros2 launch lite3_sim sim.launch.py maze:=dead_ends               # a bundled maze by name
ros2 launch lite3_sim sim.launch.py maze:=~/mazes/my_maze.yaml    # your own maze
```

Launch arguments:

| Argument | Default | |
|---|---|---|
| `maze` | `arena_example` | path to a maze YAML, or a bundled name: `corridor`, `curtain_corridor`, `curtain_loop`, `dead_ends`, `stairs_loop`, `arena_example` |
| `use_curtains` | `true` | the mission's curtain mode (`modes.exploring.curtains`) |
| `use_stairs` | `true` | the mission's stairs mode (`modes.exploring.stairs`) |
| `controller` | `rpp` | `rpp` (vx, wz) or `omni` (vx, vy, wz, like `controller:=mppi` on the robot) |
| `map_origin` | `start` | `start`: map frame at the start pose, like SLAM. `maze`: at the maze's south-west corner |
| `speed_scale` | `1.0` | robot speed multiplier. The mission's timeouts are wall-clock time, so stay ≤ ~3 |
| `lidar_range` | from `config/sim.yaml` (4.0) | shorter range = tiles identified later |
| `auto_start` | `true` | `false`: start with `ros2 service call /mission_manager/set_active std_srvs/srv/SetBool "{data: true}"` |
| `use_mission` | `true` | `false`: sim + RViz only, so you can run `mission_manager` yourself (e.g. under a debugger) |
| `mission_params` | lite3_explorer's `config/mission.yaml` | your mission config |
| `sim_params`, `sim_mission_params`, `rviz` / `rviz_config` | | |

The mission manager gets `mission_params`, then `config/sim_mission.yaml` on top (`use_ai: false`, no terrain
classifier, `classify_target: false`, `quickscan_on_edge: false`, `curtains: true`, run files in `~/lite3_sim_runs`),
then `tiles.size_m` from the maze file. Everything else is your real configuration.

Running the mission manager yourself:
```bash
ros2 launch lite3_sim sim.launch.py use_mission:=false maze:=dead_ends
ros2 run lite3_explorer mission_manager --ros-args \
  --params-file install/lite3_explorer/share/lite3_explorer/config/mission.yaml \
  --params-file install/lite3_sim/share/lite3_sim/config/sim_mission.yaml -p tiles.size_m:=1.02
```

## Drawing a maze

A maze is a YAML file with an ASCII layout. One tile = 4 characters wide, 2 lines tall; walls sit on the tile edges.
The top line is north. Tile `(i, j)` counts east from the left column and north from the bottom row, starting at 0.

```yaml
name: my_maze
tile_size: 1.02          # m
wall_thickness: 0.02
slit_m: 0.04             # gap at both ends of a slit panel
layout: |
  +---+---+---+---+---+
  | > |       ~   | ST|
  +   +   +---+   +   +
  |   | R1    !   |   |
  +   +===+   +---+   +
  |       | R2  R3    |
  +---+   +---+---+   +
  | # |               |
  +---+---+---+---+---+
start:                   # all optional
  heading: E             # N E S W or degrees (used with 'S'; arrows set their own)
  offset_m: [0.0, 0.0]   # from the tile centre
  yaw_offset_deg: 4.0    # the dog is never put down square to the walls
extra:                   # optional free geometry, metres, origin = SW corner of the maze
  walls:    [[x0, y0, x1, y1]]
  curtains: [[x0, y0, x1, y1]]
  boxes:    [[x, y, w, h]]
```

| Where | Characters |
|---|---|
| horizontal edge (between `+`) | `---` wall panel, `~~~` curtain, `===` slit panel, spaces = open |
| vertical edge | `\|` wall panel, `~` curtain, `!` slit panel, space = open |
| inside a tile | `>` `<` `^` `v` start facing E W N S (or `S` + `start.heading`), `#` solid (not floor), `0` scene tile, `M` staircase, anything else = label shown in RViz (`R1`...) |

- **wall panel**: blocks the lidar and the robot.
- **curtain**: the lidar sees a wall (it hangs in the scan band); the robot walks through. The simulated depth camera
  reports it on `/curtain/points` for curtain mode.
- **slit panel**: the aluminium-frame panels. They block the robot, but the lidar sees through the slit at each end.
- **staircase** (`M`): steps go up from every open side of the tile to a landing (`stairs: {steps: 2, rise_m: 0.13,
  run_m: 0.20}` in the maze file). The risers on those sides look like walls to the lidar and block the robot unless it
  is in AI mode; the simulated depth camera reports the treads on `/stairs/points`. The mission's stairs mode squares
  up to it, turns AI mode on, climbs over it into the tile beyond, turns AI mode off and walls it off. Since it is
  walled off afterwards, give the tiles beyond it another way back (see `mazes/stairs_loop.yaml`), or they are cut off.

Check a layout before running it:
```bash
ros2 run lite3_sim maze_preview ~/mazes/my_maze.yaml            # summary, redraw from the parsed edges, closed-off tiles
ros2 run lite3_sim maze_preview ~/mazes/my_maze.yaml --png m.png
```
If the redraw doesn't match your drawing, a character is off by a column.

## What RViz shows (`rviz/lite3_sim.rviz`, top-down, follows the robot)

| Display | Topic | |
|---|---|---|
| SLAM map | `/map` | what the mission manager sees |
| Global costmap | `/global_costmap/costmap` | inflated walls + **virtual walls** |
| Maze ground truth | `/sim/maze` | walls grey, slit panels blue, curtains cyan (translucent), the maze's own tile grid (thin white), labels, start arrow |
| Robot + curtain camera | `/sim/robot` | body (turns **red when it bumps a wall**), heading, `mission mode:phase` + Nav2 task, camera field of view, curtain spots seen now (cyan) and confirmed (magenta) |
| Tiles | `/mission_manager/tile_markers` | visited blue, to-visit green, blocked red, unreachable purple, target orange, trail |
| Nav2 path / mission route | `/plan`, `/mission_manager/route` | the path being followed / the waypoints |
| Virtual walls | `/virtual_walls` | red |
| Scan | `/scan` | yellow |

Compare the mission's tiles with the white ground-truth grid to see how well the grid snapped to the walls.

`ros2 topic echo /sim/status --field data` (1 Hz JSON) gives the pose, the maze tile, the running Nav2 task, whether
the robot is bumping or in the inscribed zone, and **ground truth**: how many reachable maze tiles were reached and
which are left. When `/exploration_complete` goes true the sim logs which reachable tiles were never reached.
`ros2 service call /lite3_sim/reset_curtains std_srvs/srv/Trigger` forgets the detected curtain spots.
Mission run files (`events.jsonl`, `tiles.json`) go to `~/lite3_sim_runs/run_<time>/`.

## Headless: test a maze in seconds

The same mission modes, exploration profiles, tile tracker and end conditions, called directly without ROS and
faster than real time (about 20 min of mission in under a minute):

```bash
ros2 run lite3_sim headless --maze ~/mazes/my_maze.yaml \
  --mission-params install/lite3_explorer/share/lite3_explorer/config/mission.yaml
ros2 run lite3_sim headless --maze dead_ends.yaml --set explorations.tiles.reject_via_other_tiles=false -v
```
It prints whether the mission completed, the tile counts, unreachable tiles and the ground-truth coverage (exit code
0 = completed). `-v` prints the mission log. Not included: the node's re-evaluation of tiles after the virtual walls
change, RViz, AI / terrain. Tests: `colcon test --packages-select lite3_sim` (or `python3 -m pytest src/lite3_sim/test`).

## Configuration: `config/sim.yaml`

Robot size, lidar (range, resolution, noise), SLAM resolution, both costmaps (footprints, inflation), planner
(tolerance, unknown cost), controller (type, speeds, lookahead, goal / progress checker, collision detection),
spin, and curtain camera (field of view, range, `min_columns`, confirm hits). Defaults are copied from
lite3_explorer's `nav2_params.yaml` and `curtain.yaml`. Change those files on the robot? Mirror the values here.

## Files

| File | |
|---|---|
| `lite3_sim/maze.py` | maze format and parser |
| `lite3_sim/world.py` | geometry, lidar, robot motion, Nav2 stand-ins (spin, navigate, waypoints), curtain camera |
| `lite3_sim/costmap.py` | occupancy mapper, Nav2 inflation, NavFn-like planner (scipy used when present) |
| `lite3_sim/sim_node.py` | the ROS 2 node: action servers, services, TF, topics, RViz markers |
| `lite3_sim/headless.py` | mission logic without ROS |
| `lite3_sim/maze_preview.py` | maze check / redraw |
| `launch/sim.launch.py`, `config/`, `mazes/`, `rviz/` | |
