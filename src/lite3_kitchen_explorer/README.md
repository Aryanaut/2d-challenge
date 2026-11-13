# lite3_kitchen_explorer — frontier survey with AI environment reports

ROS 2 (Humble) package for the DEEP Robotics **Lite3**. The dog explores **every unknown
frontier** (the method from
[shimmer0909/Frontier-Exploration](https://github.com/shimmer0909/Frontier-Exploration)).
At the start and at **every stop**, it looks around and asks your LAN **AI server**
(GPT 6.1 Terra, the same `/analyze` API as `client.py`) what kind of environment it is in.
It then publishes and logs a report for that stop, and moves on until no frontiers are left.

**The AI also steers the exploration.** At every stop it gets a top-down image of the SLAM map
(walls in black, a red no-go band next to walls, the robot, and the numbered candidate frontiers), and is told
to **pick the next frontier while avoiding walls**. Nav2 then drives there. The explorer only follows the
AI's pick if that frontier still exists and is at least `preferred_min_clearance_m` from a wall. Otherwise,
or if the server fails, it falls back to its own distance/size ranking.

**1 m tile system.** Before exploring, the dog sets up a grid of 1 m × 1 m tiles:
1. The code proposes initial-tile placements around the start position: snapped to the nearby walls (the wall angle and the grid offset are measured from the lidar map), centred on the robot, or along the map axes.
2. It draws them side by side (`initial_tile.png`), and the AI picks the one that best fits the immediate walls and frontiers. The AI also classifies the tile's N/E/S/W sides as wall, open, frontier or unknown.
3. The grid is anchored to that tile, and **then exploration begins**.

**Tile by tile: the nearest connected tile.** At every stop, the code finds the tiles the dog has not visited yet that it can walk to from its current tile, going from tile to neighbouring tile without crossing a wall. It ranks them by the number of tile steps. The up-to-6 nearest are drawn on the map, and listed for the AI with their step count, wall clearance, and whether they contain unexplored space (`prompts/tile_choice.txt`). The AI is told to **go to the nearest connected tile**, and only skip it if the map shows it is unsafe. The explorer then drives to a clear point in the chosen tile. If the AI's answer is invalid or the server fails, the dog goes to the nearest one anyway. Frontiers are only used when no unvisited tile is connected; this is also how the dog reaches new parts of the maze.

From then on, every tile is classified as **visited**, **to visit**, **blocked** or **unknown**, with wall/open sides read from the map. The robot's path is kept as a **trail** of visited tiles in order. The tiles and trail are drawn on the map the AI uses to choose frontiers, and the AI is told to prefer frontiers in tiles not yet visited. They're also published on `/tile_map`, shown in RViz, and saved in `tiles.json`.

(The package and launch-file names still say "kitchen" so your workspace and commands keep working.)

```
 Lite3 /leg_odom2 ──► odom_tf_bridge ──► TF odom→base_link, /odom
 /livox/points ──► pointcloud_to_laserscan ──► /scan ──► slam_toolbox ──► /map
                                                     │
 /map ──► frontier_explorer ──NavigateToPose──► Nav2 ──► /cmd_vel ──► Lite3 motion host
              │  ▲  /frontier_reached                  ▲ Spin (look-around)
              ▼  │  ~/set_active (pause/resume)        │
          environment_reporter ◄── /camera/color/image_raw/compressed
              │  POST /analyze (group_id, prompt, image) per view
              ▼
        AI server → {"scene_description", "relevant_to_mission", "evidence", "confidence"}
              │
              ▼  /environment_report (one JSON report per stop) → survey_monitor, RViz labels, files
```

## Nodes

| Node | What it does |
|---|---|
| `frontier_explorer` | Finds every frontier (free space next to unknown), clusters them with BFS, scores them with `cost = distance − 0.5 × size`, and publishes the top 8 as numbered candidates with their wall clearance (`/frontier_candidates`). It sends the AI's pick (`~/preferred_goal`) to Nav2 if that pick is safe, and otherwise its own best. It drops noise clusters, picks a goal point clear of obstacles, blacklists failed goals, detects being stuck, and replans if a goal gets mapped on the way. It keeps going until no reachable frontier is left, then publishes `/exploration_complete`. |
| `environment_reporter` | At the start and at every reached frontier, it pauses the explorer and turns in place 4 × 90°. At each heading it sends one camera frame with `prompts/environment_report.txt` to the AI server, then combines the 4 answers into one **stop report**. It then asks the AI to choose the next frontier from the rendered map (`prompts/frontier_choice.txt`), adds that decision to the report, and resumes exploring. You can also trigger a report by hand: `ros2 service call /environment_reporter/report_now std_srvs/srv/Trigger`. |
| `survey_monitor` | Prints each stop report as readable lines. The tmux script uses it. |
| `odom_tf_bridge` | Turns `/leg_odom2` into `/odom` and TF `odom→base_link`, re-stamping it if the motion host's clock is off. It skips the TF if something else already publishes it. |
| `preflight` | Run it once before launching. It checks the topics, frames, TF links, clock offsets and the `/cmd_vel` consumer. |

## What a stop report contains

For each stop:
- **Where:** the stop number, why it stopped (`start`, `frontier reached` or `manual request`), the time, and the robot's pose `[x, y, yaw]` in `map`.
- **What it saw:** every view's heading, `scene_description`, `relevant_to_mission`, `evidence`, `confidence` and image file.
- **Next frontier:** which frontier the AI chose and why, with that frontier's distance from the nearest wall. If the explorer decided instead, it says why.
- **Summary:** the stop's **environment**. This is the most confident view that clearly identifies the place (`relevant_to_mission: true`), with its confidence, its evidence, how many views identified the place, and a count of all the descriptions given.

The reports are published latched on `/environment_report`, shown as yellow labels in RViz
(`/environment_reporter/markers`), and written to `~/lite3_survey_runs/run_<time>/`:
- `reports.jsonl`: one JSON line per stop.
- `environment_report.md`: a readable table of all the stops plus every view. It is rewritten after each stop, and marked complete at the end.
- `stopNNN_viewK_qNNNN.jpg/.json`: every image and the server's raw answer.
- `stopNNN_map.png`, `_prompt.json`, `_answer.json`: the map the AI saw, the exact prompt with the candidate table, and the AI's choice.
- `initial_tile.png`, `initial_tile_prompt.json`, `initial_tile_answer.json`: the start-up tile placement.
- `tiles.json`: the grid, the initial tile, the trail, and every tile's status and sides. It is updated at every stop.

## Install (on the Lite3 perception host / your NUC)

```bash
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
                 ros-humble-pointcloud-to-laserscan python3-opencv
pip install httpx            # same HTTP library as client.py
cd ~/ros2_ws/src && cp -r /path/to/lite3_kitchen_explorer .
cd ~/ros2_ws && colcon build --symlink-install --packages-select lite3_kitchen_explorer
source install/setup.bash
```

This runs alongside the Lite3 ROS 2 `transfer` package. That package is what forwards `/cmd_vel` to the motion host and publishes `/leg_odom2`.

## Configure (once)

1. In `config/reporter.yaml`, set `server_url` and `group_id`. The camera topic is already set.
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
    server_url:=http://192.168.1.100:8000 group_id:=YOUR_GROUP
# plus publish_lidar_tf:=true lidar_x:=... etc. if the preflight asked for it
```

If you already run SLAM and Nav2, use `explorer_only.launch.py` instead, with the same `server_url:=` / `group_id:=` arguments.

## Run

Stand the dog up and put it in the mode that accepts `/cmd_vel` (navigation/auto mode). Keep the remote in hand.

```bash
chmod +x scripts/run_survey_tmux.sh
SERVER_URL=http://192.168.1.100:8000 GROUP_ID=YOUR_GROUP ./scripts/run_survey_tmux.sh
# extra launch args: LAUNCH_ARGS="publish_lidar_tf:=true lidar_x:=0.15 lidar_z:=0.12"
./scripts/run_survey_tmux.sh stop     # Ctrl-C the launch and close everything
```

| Window | Panes |
|---|---|
| `explorer` | the launch output |
| `status` | reporter state · explorer status · **environment report per stop** · last AI answer per view · `/cmd_vel` |
| `health` | `/scan` rate · `/map` rate · `odom→base_link` TF · shell with pause/resume/report-now commands |

Switch windows with `Ctrl-b n` / `Ctrl-b p` or the mouse. Detach with `Ctrl-b d`; rerun the script to re-attach.

Without tmux:
```bash
ros2 launch lite3_kitchen_explorer kitchen_exploration.launch.py \
    server_url:=http://192.168.1.100:8000 group_id:=YOUR_GROUP
ros2 run lite3_kitchen_explorer survey_monitor        # in a second terminal
```
To explore without the AI reports, add `use_reporter:=false`. If you already run SLAM and Nav2, use `explorer_only.launch.py`.

In RViz, add `/map`, `/frontier_explorer/markers`, `/environment_reporter/markers` and
`/environment_reporter/tile_markers` (blue = visited tiles with the trail line, green = to visit, red = blocked).
For the explorer's current goal, add `/frontier_explorer/current_goal` as a **Pose** display (an arrow, latched), and `/plan` as a **Path** for Nav2's route to it.
For frontiers, green is the chosen one, blue are candidates and red are blacklisted. Yellow labels are the stop reports.

## Test the prompt against the server first

```bash
python client.py --server http://192.168.1.100:8000 --group YOUR_GROUP \
    --image arena_photo.jpg --prompt-file prompts/environment_report.txt
```

Try your arena photos from the dog's height: the block field, the ramps, the beam/ladder section, the stairs, a dead end. The
prompt keeps the four-key JSON format from your `prompt.txt`, so your server's replies don't change. The
keys now mean the following:
- `scene_description` starts with the **location type**, for example "kitchen area with a toy stove".
- `relevant_to_mission` is **true when the image clearly identifies the environment**, and false for a blank wall or a blurred image.
- `confidence` is still confidence in the description.

## Key tuning knobs

- `tile_targets_first` (true) in `reporter.yaml`: go to the nearest connected unvisited tile at each stop. Set it to false to choose among frontiers instead. With `tile_target_ai_choice: false` the nearest tile is always taken without asking the AI.
- `tiles_enabled` / `tile_size_m` / `tile_ai_choice` in `reporter.yaml`: switch the tile system on or off, change the tile size, or let the code snap the first tile to the walls without asking the AI.
- `ai_frontier_choice` (true) in `reporter.yaml`: set it to false to go back to the explorer's own ranking.
- `min_wall_clearance_m` (0.35) in `reporter.yaml`: the width of the red no-go band and the distance the AI is told to keep from walls. `preferred_min_clearance_m` (0.30) in `explorer.yaml` is the hard limit: AI picks closer to a wall than this are refused. Nav2's own `inflation_radius` (0.45) still keeps the path itself away from walls.

- `scan_steps` (4) in `reporter.yaml`: the number of views per stop. Use 6–8 for a narrow camera; use 1 to skip turning and just look ahead.
- `settle_s` (1 s): the pause after each turn so the image isn't blurred.
- `report_at_start` (true): report the starting position before exploring.
- `size_weight` (0.5) and `min_cluster_cells` (8) in `explorer.yaml`: these decide which frontiers get visited first, and how small a frontier can be before it's ignored. Lowering `min_cluster_cells` means more, shorter hops, so more stops and reports.
- `goal_timeout_s` / `progress_timeout_s`: how quickly an unreachable frontier is given up and blacklisted.

## Limits

- **Stairs:** Nav2 plans in 2D, so the staircase shows up as an obstacle and the explorer goes around it.
  The robot will report a "staircase" environment if it sees one, but it won't climb it.
- The stop pose is where the robot stood. Each view's heading tells you which way the camera was facing.
- Each stop takes `scan_steps` × (turn + settle + server reply), plus one more reply for the frontier choice, so with a slow server the stops get long.
- The AI judges a picture of the map. It can misread it, which is why the explorer re-checks each pick's wall clearance before driving there.
- Run `python3 -m pytest test` to check the frontier, parsing and report logic.
  The ROS nodes have not been run on a Lite3 or in simulation yet.
