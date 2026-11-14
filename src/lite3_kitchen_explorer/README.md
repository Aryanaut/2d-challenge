# lite3_kitchen_explorer — tile-by-tile maze exploration with an AI tile report

> **The AI client now lives in its own package, [`lite3_llm_tools`](../lite3_llm_tools/README.md).** This package
> has no HTTP, prompt or image code any more. It reaches the AI through JSON messages on `/llm/request`,
> `/llm/response` and `/llm/status`, and works without it (`use_ai:=false`, or the `llm_client` node not running):
> everything below happens except the AI's descriptions. The server address, group id, camera topic and prompts
> are configured in `lite3_llm_tools/config/llm_client.yaml`.

ROS 2 (Humble) package for the DEEP Robotics **Lite3**. The dog explores a maze **tile by tile**. At every stop (and at the start) it makes **one quick turn in place**
(360° by default) so the lidar maps all around the tile and the neighbouring tiles form on the map; nothing waits for the AI
during it. Navigation is geometric (the lidar map and Nav2). The **AI's job is the report**: when a tile is **reached**, your LAN **AI server** (GPT 6.1 Terra, the same `/analyze` API as `client.py`)
is asked **in the background** what that tile looks like, from the camera frame taken on arrival (before the turn), the lidar ranges and what
the map says about the tile's sides. All the tile descriptions are compiled into **one file**,
`~/2d/reports/tile_report_<run>.md`, which is rewritten after every change.

**When is a tile reached?** When the robot is within `tile_reach_radius_m` (0.30 m) of the tile's **centre**. Tile goals are the
tile centre (the free point nearest the centre when a wall is in the way), and Nav2's `xy_goal_tolerance` is 0.25 m so it
stops close enough. If Nav2 reports a goal reached and the robot is at the tile's goal point (up to 0.45 m), that also
counts, so a tile beside a wall is not driven to forever. The tile the robot starts in counts as reached at the start.
Merely entering a tile does not count.

**Optional AI choices (off by default).** `tile_ai_choice` (initial tile placement), `tile_target_ai_choice` (confirm the next tile) and
`ai_frontier_choice` (pick a frontier) let the AI choose from a map image instead. They are slow and block the robot while
it waits, so they are off: tiles are fitted to the walls geometrically and the next tile is the nearest connected one.

**1 m tile system.** Before exploring, the dog sets up a grid of 1 m × 1 m tiles:
1. The code proposes initial-tile placements around the start position: snapped to the nearby walls (the wall angle and the grid offset are measured from the lidar map), centred on the robot, or along the map axes.
2. It takes the first, wall-snapped one (with `tile_ai_choice: true` the AI picks among them from `initial_tile.png`).
3. The grid is anchored to that tile, the start tile is reported, and **then exploration begins**.

**Tile by tile: the nearest connected tile.** Each time the explorer reaches a goal, the code finds the tiles the dog has not visited yet that it can walk to from its current tile, going from tile to neighbouring tile without crossing a wall. It ranks them by the number of tile steps. The up-to-6 nearest are drawn on the map, and listed for the AI with their step count, wall clearance, and whether they contain unexplored space (`prompts/tile_choice.txt`). The AI is told to **go to the nearest connected tile**, and only skip it if the map shows it is unsafe. The explorer then drives to a clear point in the chosen tile. If the AI's answer is invalid or the server fails, the dog goes to the nearest one anyway. Frontiers are only used when no unvisited tile is connected; this is also how the dog reaches new parts of the maze.

From then on, every tile is classified as **visited**, **to visit**, **blocked** or **unknown**, with wall/open sides read from the map. The robot's path is kept as a **trail** of visited tiles in order. The tiles and trail are drawn on the map the AI uses to choose frontiers, and the AI is told to prefer frontiers in tiles not yet visited. They're also published on `/tile_map`, shown in RViz, and saved in `tiles.json`.

(The package and launch-file names still say "kitchen" so your workspace and commands keep working.)

```
 Lite3 /leg_odom2 ──► odom_tf_bridge ──► TF odom→base_link, /odom
 /livox/points ──► pointcloud_to_laserscan ──► /scan ──► slam_toolbox ──► /map
                                                     │
 /map ──► frontier_explorer ──NavigateToPose──► Nav2 ──► /cmd_vel ──► Lite3 motion host
              │  ▲  /frontier_reached                  ▲ Spin (look-around)
              ▼  │  ~/set_active (pause/resume)        │
          environment_reporter ── /llm/request (describe_tile, in the background) ──► llm_client (lite3_llm_tools) ◄── camera, /map, /scan
              ▲  /llm/response, /llm/status               │  POST /analyze (group_id, prompt, image)
              │                                           ▼
              │                      AI server → {"scene_description", "relevant_to_mission", "evidence", "confidence"}
              ▼  /tile_report (one JSON per tile) → survey_monitor, RViz labels, ~/2d/reports/tile_report_<run>.md
```

## Nodes

| Node | What it does |
|---|---|
| `frontier_explorer` | Finds every frontier (free space next to unknown), clusters them with BFS, scores them with `cost = distance − 0.5 × size`, and publishes the top 8 as numbered candidates with their wall clearance (`/frontier_candidates`). It sends the AI's pick (`~/preferred_goal`) to Nav2 if that pick is safe, and otherwise its own best. It drops noise clusters, picks a goal point clear of obstacles, blacklists failed goals, detects being stuck, and replans if a goal gets mapped on the way. It keeps going until no reachable frontier is left, then publishes `/exploration_complete`. |
| `environment_reporter` | The tile manager. It builds the tile grid, marks a tile **reached** when the robot is near its centre, and asks the `llm_client` in the background what each reached tile looks like. Each time the explorer reaches a goal it briefly keeps it paused, finds the unvisited tiles Nav2 can reach, picks the nearest (2–3 waypoints along the Nav2 path), and resumes. It writes the tile report. You can describe the tile the dog is in right now with `ros2 service call /environment_reporter/report_now std_srvs/srv/Trigger`. |
| `llm_client` *(package `lite3_llm_tools`)* | Everything that talks to the AI server: it keeps camera frames, draws the lidar map for the AI, adds the lidar ranges to the prompts, retries failed requests and returns the answers. See its README. |
| `survey_monitor` | Prints each tile report (`/tile_report`) as readable lines. The tmux script uses it. |
| `odom_tf_bridge` | Turns `/leg_odom2` into `/odom` and TF `odom→base_link`, re-stamping it if the motion host's clock is off. It skips the TF if something else already publishes it. |
| `preflight` | Run it once before launching. It checks the topics, frames, TF links, clock offsets and the `/cmd_vel` consumer. |

## The tile report

One file for the whole run: `~/2d/reports/tile_report_<run>.md` (`reports_dir` in `reporter.yaml`). It is created when the
exploration starts and rewritten atomically after every change, so it is always current. It contains:
- **A summary:** the run, whether the AI descriptions are on, how many are done / pending / failed, the tile grid (origin, yaw), and the tile counts (visited, to visit, blocked, unknown).
- **A table, one row per tile reached, in the order reached:** tile `i,j`, time, position, the **map sides** (N/E/S/W: wall, open, frontier, unknown), **what it looks like**, and the confidence.
- **One section per tile:** the same, plus the evidence the AI gave and the path of its camera image.
- **The tiles still to visit.**

An entry starts as `(waiting for the AI ...)` and is filled in when the answer comes back. A failed query shows
`(AI query failed: ...)`; with the AI off the rows say so and still list every tile reached with its map sides.

Also published and written:
- `/tile_report`: one JSON per finished tile entry (latched). Yellow labels at the tile centres in RViz (`/environment_reporter/markers`).
- `~/lite3_survey_runs/run_<time>/`: `tile_<i>_<j>_qNNNN.jpg/.json` (the camera frame and the server's raw answer for each tile), `stops.jsonl` (one line per decision: which tile was chosen next and how), `tiles.json` (the grid, the trail, every tile's status and sides, updated at every stop). With the optional AI choices on, also `stopNNN_map.png`, `_prompt.json`, `_answer.json` and `initial_tile.*`.

## Install (on the Lite3 perception host / your NUC)

```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
                 ros-humble-pointcloud-to-laserscan python3-opencv
pip install httpx            # used by lite3_llm_tools (same HTTP library as client.py)
cd ~/ros2_ws/src && cp -r /path/to/lite3_kitchen_explorer /path/to/lite3_llm_tools .
cd ~/ros2_ws && colcon build --symlink-install --packages-select lite3_llm_tools lite3_kitchen_explorer
source install/setup.bash
```

This runs alongside the Lite3 ROS 2 `transfer` package. That package is what forwards `/cmd_vel` to the motion host and publishes `/leg_odom2`.

## Configure (once)

1. In `../lite3_llm_tools/config/llm_client.yaml`, set `server_url` and `group_id` (the camera topic is already set). Test them with `./scripts/test_llm.py` from the workspace root.
2. Run the preflight with the dog standing and its drivers running:
   ```bash
   ros2 run lite3_kitchen_explorer preflight
   ```
   Fix anything it marks **FIX**. The most likely one is a missing TF from `base_link` to the Livox frame
   (usually `livox_frame`). If so, measure where the MID-360 sits on the body and add
   `publish_lidar_tf:=true lidar_x:=.. lidar_y:=.. lidar_z:=.. lidar_pitch:=..` to the launch command.
   If `/livox/points` turns out not to be a PointCloud2, the preflight tells you what to change.
3. **Uneven tiles:** `scan_min_height` / `scan_max_height` (in metres, in `base_link`) set which lidar
   points count as obstacles. Set the minimum high enough that terrain tiles and ramps are *not* walls, and low enough that
   real obstacles are. Then check `/scan` and `/map` in RViz before you let it drive.
4. Check the footprint and speed limits in `config/nav2_params.yaml`. The defaults are conservative: 0.35–0.4 m/s and 0.6 rad/s.

## Run

Stand the dog up and put it in the mode that accepts `/cmd_vel` from the perception host (see the
Lite3 manual / remote). Keep the remote in hand. Then:

```bash
ros2 launch lite3_kitchen_explorer kitchen_exploration.launch.py \
    server_url:=http://192.168.10.141:8000 group_id:=YOUR_GROUP
# plus publish_lidar_tf:=true lidar_x:=... etc. if the preflight asked for it
```

If you already run SLAM and Nav2, use `explorer_only.launch.py` instead, with the same `server_url:=` / `group_id:=` arguments.

## Run

Stand the dog up and put it in the mode that accepts `/cmd_vel` (navigation/auto mode). Keep the remote in hand.

```bash
chmod +x scripts/run_survey_tmux.sh
SERVER_URL=http://192.168.10.141:8000 GROUP_ID=YOUR_GROUP ./scripts/run_survey_tmux.sh
# extra launch args: LAUNCH_ARGS="publish_lidar_tf:=true lidar_x:=0.15 lidar_z:=0.12"
./scripts/run_survey_tmux.sh stop     # Ctrl-C the launch and close everything
```

**Running without the AI.** `use_ai:=false` (launch argument, or `use_ai` in `config/reporter.yaml`) means the
`llm_client` is not started and the server is never queried. Navigation is identical (the AI only writes descriptions), and the tile report still lists every tile reached with its map sides. With the
top-level tmux script: `./scripts/run_kitchen_tmux.sh --no-ai` or `USE_AI=false ./scripts/run_kitchen_tmux.sh`
(`--ai` / `USE_AI=true` turns it back on).

| Window | Panes |
|---|---|
| `explorer` | the launch output |
| `status` | reporter state · explorer status · **tile reports** · last AI answer (`/llm/last_result`) · `/cmd_vel` |
| `health` | `/scan` rate · `/map` rate · `odom→base_link` TF · shell with pause/resume/report-now commands |

Switch windows with `Ctrl-b n` / `Ctrl-b p` or the mouse. Detach with `Ctrl-b d`; rerun the script to re-attach.

Without tmux:
```bash
ros2 launch lite3_kitchen_explorer kitchen_exploration.launch.py \
    server_url:=http://192.168.10.141:8000 group_id:=YOUR_GROUP
ros2 run lite3_kitchen_explorer survey_monitor        # in a second terminal
```
To explore without the tile manager and its reports, add `use_reporter:=false`. If you already run SLAM and Nav2, use `explorer_only.launch.py`.

In RViz, add `/map`, `/frontier_explorer/markers`, `/environment_reporter/markers` and
`/environment_reporter/tile_markers` (blue = visited tiles with the trail line, green = to visit, red = blocked).
For the explorer's current goal, add `/frontier_explorer/current_goal` as a **Pose** display (an arrow, latched), and `/plan` as a **Path** for Nav2's route to it.
For frontiers, green is the chosen one, blue are candidates and red are blacklisted. Yellow labels at the tile centres are the tile descriptions.

## Test the prompt against the server first (prompts are in `lite3_llm_tools/prompts/`)

```bash
python client.py --server http://192.168.10.141:8000 --group YOUR_GROUP \
    --image arena_photo.jpg --prompt-file ../lite3_llm_tools/prompts/environment_report.txt
```

Try your arena photos from the dog's height: the block field, the ramps, the beam/ladder section, the stairs, a dead end. The
prompt keeps the four-key JSON format from your `prompt.txt`, so your server's replies don't change. The
keys now mean the following:
- `scene_description` starts with the **location type**, for example "kitchen area with a toy stove".
- `relevant_to_mission` is **true when the image clearly identifies the environment**, and false for a blank wall or a blurred image.
- `confidence` is still confidence in the description.

## Key tuning knobs

- `tile_targets_first` (true) in `reporter.yaml`: go to the nearest connected unvisited tile at each stop. Set it to false to choose among frontiers instead. With `tile_target_ai_choice: false` the nearest tile is always taken without asking the AI.
- `tiles_enabled` / `tile_size_m` / `tile_ai_choice` in `reporter.yaml`: switch the tile system on or off, change the tile size, or let the AI place the first tile (`tile_ai_choice: true`; by default the code snaps it to the walls).
- `ai_frontier_choice` (false) in `reporter.yaml`: set it to true to let the AI pick frontiers from a map image; false = the explorer's own ranking.
- `min_wall_clearance_m` (0.35) in `reporter.yaml`: the width of the red no-go band and the distance the AI is told to keep from walls. `preferred_min_clearance_m` (0.30) in `explorer.yaml` is the hard limit: AI picks closer to a wall than this are refused. Nav2's own `inflation_radius` (0.45) still keeps the path itself away from walls.

- **The scan turn needs room.** The dog is 0.61 × 0.37 m, so turning in place sweeps a 0.36 m radius, and Nav2's Spin aborts as soon as the body touches an inscribed or lethal cell of the local costmap. In a 1 m maze corridor that is usually the case, and an aborted spin leaves the robot rotated across the corridor, where it can then neither turn back nor drive (`route_debug_080014`). So with `turn_check_room: true` the reporter checks the local costmap first (every 5° of body orientation, same rule as Nav2) and **skips the turn** where it cannot finish; the log says `No room to turn here`. The lidar covers 360° anyway, so the map still forms. Set `turn_check_room: false` to always try, or `turn_room_margin_m` for a stricter check. Tiles open on several sides (junctions) still get the turn.
- **A tile goal that fails is not replaced by a frontier.** If the explorer gives up on a tile goal (stuck, timed out, rejected), it holds and publishes `/frontier_explorer/goal_failed`; the reporter marks that tile as unreachable, does not turn, and picks the next-nearest reachable tile. Frontiers are the fallback only when no other tile is connected. If the reporter is not running, the explorer resumes by itself after `hold_timeout_s`.
- `turn_at_stop` (true), `turn_deg` (360), `turn_use_crawl` (true), `turn_time_allowance_s` (20) in `reporter.yaml`: the quick scan turn at each stop. The turn speed is Nav2's `behavior_server` `max_rotational_vel` (1.0 rad/s, so about 6 s for 360°), `rotational_acc_lim` and `simulate_ahead_time` in `nav2_params.yaml`. If the turn fails or times out, the next tile is chosen with the map so far.
- `tile_reach_radius_m` (0.30) in `reporter.yaml`: how close to a tile's centre the robot must get for the tile to count as reached. Keep Nav2's `xy_goal_tolerance` (`nav2_params.yaml`, 0.25) below it. `reports_dir` (`~/2d/reports`): where the tile report goes. `llm_report_timeout_s` (180): a description that takes longer is marked failed.
- `report_at_start` (true): decide the first target as soon as the start tile is set up.
- `size_weight` (0.5) and `min_cluster_cells` (8) in `explorer.yaml`: these decide which frontiers get visited first, and how small a frontier can be before it's ignored. Lowering `min_cluster_cells` means more, shorter hops, so more stops and reports.
- `goal_timeout_s` / `progress_timeout_s`: how quickly an unreachable frontier is given up and blacklisted.

## Limits

- **Stairs:** Nav2 plans in 2D, so the staircase shows up as an obstacle and the explorer goes around it.
  A tile description may say "staircase" if the camera sees one, but the robot won't climb it.
- The tile photo is taken while the robot is walking, as it arrives at the centre (before the scan turn), so it can be a little blurred and shows what is ahead of the robot.
- A slow server does not slow the robot: the descriptions arrive in the background (the report shows `waiting` until then).
- The AI judges a picture of the map. It can misread it, which is why the explorer re-checks each pick's wall clearance before driving there.
- Run `python3 -m pytest test` to check the frontier, parsing and report logic.
  The ROS nodes have not been run on a Lite3 or in simulation yet.
