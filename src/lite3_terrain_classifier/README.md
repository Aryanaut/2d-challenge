# lite3_terrain_classifier

ROS 2 (Humble) package for the DEEP Robotics **Lite3**. It checks the ground around the dog and
publishes whether it is **rugged** (not flat: steps, blocks, bumps, holes) or **non_rugged** (flat).

```
/livox/points ──► base_link (TF) ──► stack 0.3 s ──► area around the robot ──┐   primary
                                                                             ├─► fuse ─► debounce ─► /terrain_type
/camera/depth/image_raw ──► bottom half of the frame ──► 3D points ──────────┘   verify
```

| Topic | Type | Content |
|---|---|---|
| `/terrain_type` | `std_msgs/String` (latched) | `rugged`, `non_rugged` or `unknown` (no usable sensor data / paused) |
| `/terrain_classifier/details` | `std_msgs/String` (JSON) | per-sensor metrics, `source`, `verified`, `sensors_agree` |
| `/terrain_classifier/lidar_points` | `PointCloud2` | lidar points used; `intensity` = height above the fitted ground (m) |
| `/terrain_classifier/depth_points` | `PointCloud2` | the same for the depth camera |
| `/terrain_classifier/set_enabled` | `std_srvs/SetBool` | pause / resume |

## How it decides

The same test runs on each sensor (`terrain_core.py`):

1. Fit the main ground plane (RANSAC, then a least-squares refinement).
2. Split the area into 0.2 m cells and look at how high each cell sits relative to that plane.
3. A cell is rugged if its ground is more than `step_height_m` (5 cm) above or below the plane, or if
   heights inside it spread more than `cell_spread_m`. Cells taller than `obstacle_height_m` (35 cm) are
   walls, boxes or people, not terrain, so they are ignored.
4. The terrain is rugged when at least `min_rugged_cells` (4) cells **and** `min_rugged_fraction` (2 %) of the
   cells are rugged.

**Lidar (primary):** a 4 m × 3 m area around the robot (`lidar.x_min` … `lidar.y_max`, base_link), with the
body cut out. MID-360 frames are sparse, so 0.3 s of frames is stacked. While walking, this uses the
`odom → base_link` TF. Without that TF, only the latest frame is used.

**Depth camera (verify):** only the bottom part of the image is used (`depth.roi_top_fraction: 0.5`),
because the camera points slightly down and that part shows the ground just ahead. Range is capped at
2 m. If TF `base_link ← camera frame` exists, the check runs in base_link; if not, it still works in the
camera frame, because heights relative to a fitted plane don't depend on the frame.

**Fusion** (`fusion_mode`):

| Mode | Behaviour |
|---|---|
| `lidar_primary` (default) | the lidar decides; `verified: true` in details when the camera agrees |
| `require_agreement` | the output only changes when both sensors agree |
| `any_rugged` | rugged if either sensor says rugged (most cautious) |

If the lidar has no usable data, the camera result is used alone (`depth_fallback: true`). The output only
switches after the same new answer has come in `confirm_frames` (3) times in a row, which takes about 0.6 s at 5 Hz.

## Build

```bash
cd ~/ros2_ws/src && cp -r /path/to/lite3_terrain_classifier .
cd ~/ros2_ws && colcon build --symlink-install --packages-select lite3_terrain_classifier lite3_kitchen_explorer
source install/setup.bash
```

## Manual mode (driving with the remote)

```bash
ros2 launch lite3_terrain_classifier terrain_classifier.launch.py
ros2 topic echo /terrain_type
```

The node only reads sensors and never sends motion commands, so it is safe to run next to the remote.
If nothing publishes `base_link → livox_frame` on `/tf` (the `lite3_kitchen_explorer` preflight tells you),
add `publish_lidar_tf:=true lidar_x:=.. lidar_z:=.. lidar_pitch:=..`.

## Auto mode (frontier exploration)

`lite3_kitchen_explorer`'s `kitchen_exploration.launch.py` (and `explorer_only.launch.py`) start the
classifier automatically. It uses the same `cloud_topic` and lidar TF as the rest of the stack:

```bash
ros2 launch lite3_kitchen_explorer kitchen_exploration.launch.py server_url:=... group_id:=...
# use_terrain:=false to leave it out, terrain_params:=/path/to/my_terrain.yaml to tune
```

In both modes, any node can subscribe to `/terrain_type`. The topic is latched, so late subscribers still get the
current value.

## Tuning on the real robot

1. Stand the dog on flat floor, then in front of the uneven tiles, and run
   `ros2 topic echo /terrain_classifier/details`. Compare `max_step_m`, `rugged_cells` and `n_cells` for the two cases.
2. In RViz, show `/terrain_classifier/lidar_points` and `/terrain_classifier/depth_points` coloured by
   `intensity`. They show which area is checked and which spots stick out.
3. Flat floor reported as rugged: raise `step_height_m` / `cell_spread_m` or `min_rugged_cells`
   (check the lidar TF first: a wrong pitch makes far points look higher than near ones).
   Rugged tiles missed: lower them, or make the lidar area smaller so the tiles make up more of it.
4. Furniture or walls inside the lidar area that are lower than `obstacle_height_m` can count as rugged.
   Shrink the area (`lidar.x_min` … `lidar.y_max`) if that happens.
5. `max_tilt_deg > 0` also calls ramps rugged. It is off by default because body pitch while walking
   and lidar mounting errors look like tilt.

Run `python3 -m pytest test` to check the maths. If a locally installed pytest plugin fails to load, prefix the command with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
