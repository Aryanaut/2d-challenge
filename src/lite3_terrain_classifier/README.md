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
| `/terrain_kind` | `std_msgs/String` (JSON, latched) | terrain **kind** of the floor around the robot: `flat`, `uneven`, `slats` (see below) |
| `/terrain_classifier/slats_depth_debug` | `sensor_msgs/Image` (bgr8) | debug: the depth frame the slats test saw, with the Hough lines drawn on it |
| `/terrain_classifier/slats_color_debug` | `sensor_msgs/Image` (bgr8) | debug: the same for the colour frame |
| `/terrain_classifier/slats_detections` | `std_msgs/String` (JSON) | debug: per frame `{camera, rungs, converging, segments, orientation, angle_deg, frames}` |
| `/terrain_classifier/set_enabled` | `std_srvs/SetBool` | pause / resume |
| `/terrain_classifier/classify` | `std_srvs/Trigger` | `trigger_mode` only: start a new patch and classify for `trigger_duration_s` |

## How it decides

The same test runs on each sensor (`terrain_core.py`):

1. Fit the main ground plane (RANSAC, then a least-squares refinement).
2. Split the area into 0.2 m cells and look at how high each cell sits relative to that plane.
3. A cell is rugged if its ground is more than `step_height_m` (5 cm) above or below the plane, or if
   heights inside it spread more than `cell_spread_m`. Cells taller than `obstacle_height_m` (35 cm) are
   walls, boxes or people, not terrain, so they are ignored.
4. The terrain is rugged when at least `min_rugged_cells` (4) cells **and** `min_rugged_fraction` (2 %) of the
   cells are rugged.

**Lidar (primary):** only the floor directly in front of the robot: 1.0 m × 0.9 m, from 0.4 m to 1.4 m ahead and
±0.45 m to the sides (`lidar.x_min` … `lidar.y_max`, base_link). It used to be 4 m × 3 m around the robot, which
also picked up tiles beside and behind it; widen the four values to go back. The body is cut out. MID-360 frames are sparse, so 0.3 s of frames is stacked. While walking, this uses the
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

## Terrain kinds (flat / uneven / slats)

On top of rugged / non_rugged the classifier names the **kind** of floor. It knows nothing about tiles or any map:
it looks at the floor around the robot, and whoever asks (`~/classify`) matches the answer to where it was looking.
The explorer does that: it triggers the classifier when it looks at a tile and stores the answer on that tile.

Every kind is a strategy (`terrain_kinds.py`). `config/terrain_kinds.yaml` lists them in priority order and the
**first** one whose rule accepts the floor wins:

```yaml
common:                      # measurement settings shared by all kinds
  cell_m: 0.30               # lidar spread is measured per cell of this size
  min_cells: 5               # lidar floor cells needed before uneven / flat can be decided
  image_window: 15           # per camera: how many of the latest frames vote on slats
images:                      # Hough line detection settings (slat_lines.py), shared and per camera
  common: {roi_top_fraction: 0.5, hough_threshold: 30, min_length_fraction: 0.10}
terrains:
  - name: slats              # parallel straight lines in the depth and / or colour image
    type: hough_slats
    sources: [depth, color]
    mode: any                # any: one camera is enough | all: every camera that has frames must agree
    require: [depth]         # cameras that MUST confirm whatever the mode (colour alone is fooled by block edges)
    min_lines:               # distinct slat-like lines for one frame to vote "slats", per camera and family
      depth: {rungs: 4, converging: 3}
      color: {rungs: 15, converging: 6}
    min_vote_fraction: 0.6   # share of a camera's latest frames that must vote (at least min_frames, 5, frames)
    marker: {style: solid, color: [1.0, 0.0, 0.0, 0.55], label: SLATS}
  - {name: uneven, type: uneven, rugged_median_m: 0.12, rugged_fraction: 0.60,
     marker: {style: hatch, color: [1.0, 0.75, 0.0, 1.0], label: UNEVEN}}
  - {name: flat, type: flat, marker: {style: none}}          # everything left
```

**Slats (Hough lines).** Slats show up as several parallel straight edges in the lower half of the images (the floor in
front of the robot): *rungs* (near-horizontal lines at different image heights) or *converging* lines (running away from
the camera, crossing the same image row at different places). In the depth image the gaps between slats are dark
streaks. `slat_lines.py` runs Canny + probabilistic Hough (`cv2.HoughLinesP`) on each frame and counts the *distinct*
parallel lines in each family. **Not slats: uneven ground.** Ramps, wedges, wood grain and curtain folds also give many straight lines, but at mixed angles.
A family only counts when its lines are slat-like: *rungs* must be parallel (angle spread at most
`max_rung_angle_std_deg`, 5°) **and long** (each rung runs across at least `min_rung_span`, 35 %, of the image width: the
edges of a field of blocks or steps are parallel too, but short pieces), *converging* lines must meet at one vanishing point (mean distance at most
`max_vanishing_residual` of the image width). A family that fails counts 0 lines; the raw counts stay in `rungs_raw` /
`converging_raw`, and the debug picture header shows `accepted/raw`.
**Depth must confirm.** `require: [depth]` means the colour image alone can never name slats: block edges, wood grain and
curtain folds give long straight lines in colour, but the depth camera sees them as a plain floor or as short cracks. A
camera with no frames yet does not confirm either, so slats stay undetected until depth has run; set `require: []` to
go back to the either-camera behaviour.
A frame votes "slats" when either family reaches `min_lines`; a camera says slats when
`min_vote_fraction` of its latest `image_window` frames voted, so a single noisy frame does not decide. The depth
image comes from `depth.image_topic`, the colour image from `color.image_topic` (`/camera/color/image_raw`,
`use_color: false` turns it off). The colour image is read from `color.image_topic` and from `<image_topic>/compressed` (JPEG, as the camera often only
publishes that). The slats strategy does not need lidar cells, so it works even when the lidar patch
is too sparse. **Orientation.** Every frame also says which way the slats run in the image, from the family with more distinct lines:
`horizontal` = rungs, the slats lie across the view (and across the robot's path), `angle_deg` is their mean angle from
the image's horizontal (-20..20); `vertical` = converging lines, the slats run away from the camera, along the robot's
path, `angle_deg` is the mean line angle (90 = straight ahead); `none` when no line was found. When the kind is slats,
`/terrain_kind` carries `slats_orientation` and `slats_angle_deg`, taken from the frames that voted slats (majority
orientation, mean angle). The explorer stores them with the tile's terrain. The debug pictures print the orientation in
their header.

**Debugging the lines.** The debug topics above are only computed while something subscribes to them. In the pictures
the cyan line is the top of the searched region, green segments are rungs, orange are converging lines and grey ones were
ignored; the header shows the counts and the verdict (the colour camera needs 4 lines, depth 3). View them with
`ros2 run rqt_image_view rqt_image_view /terrain_classifier/slats_depth_debug` or as Image displays in RViz, and read the
numbers with `ros2 topic echo /terrain_classifier/slats_detections`.

A lidar-height variant of the slats test (`type: slats`) is still there, commented out in the yaml.

**Uneven** and **flat** come from the lidar: uneven = median cell spread at least `rugged_median_m` and at least
`rugged_fraction` of the cells rough; flat = everything left.

`/terrain_kind` carries `kind`, the measurements (`n_cells`, `spread_median_m`, `rough_fraction`, `coherence`), the
`marker` style from the config, and `seq`, which goes up by one on every `~/classify` call. In `trigger_mode` each call
starts a fresh patch of floor; in continuous mode a new patch starts every `kinds.window_s` (5 s). Use another file with
`--ros-args -p kinds_file:=/path/terrain_kinds.yaml`, and turn the kinds off with `-p kinds.enabled:=false`.

To add a kind, write a `TerrainStrategy` subclass in `terrain_kinds.py` (`match(stats)` returns a dict of
measurements to report, or `None`), decorate it with `@register_terrain('mytype')` and list it in the yaml.

**Validation.** Flat versus rugged (lidar) is measured: on a recorded run flat floor had a median cell spread of
0.04–0.08 m and the slats and uneven tiles 0.18–0.19 m. The Hough slats test was checked on the pictures in `images/`
(`test/test_slat_lines.py`) and on a recorded run, `bags/data/terrains` (depth + compressed colour, 27 s): first a grid of
cubes, then a slats tile (ribs running away from the robot), then cubes again, a metal staircase and cubes. Per frame:

| | depth converging lines | colour converging lines | colour rungs |
|---|---|---|---|
| cubes | 0–2 | up to 4 | 3–7 |
| slats | 3–5 | 9–13 | 0–5 |
| metal stairs | 0 | 1 | 12 |

so colour counts only with many lines (`min_lines.color`: 6 converging, 15 rungs), depth with 3 converging / 4 rungs, and
depth must confirm (`require: [depth]`). The converging lines are kept by a RANSAC vanishing-point test, so noise lines
do not hide a real set. Replayed as 1.5 s triggers of 5 Hz frames, 24 of 26 triggers were right: every cubes and stairs
trigger was *not* slats, the slats tile was slats except the first and last second of the transition. The staircase is not
named (kind `unknown`), only not called slats. Colour rungs are the weak spot (stairs give 12): with ribs seen *across*
the view (rungs) only depth can confirm them, and that is not validated on a recording. Tune on runs that cross slats,
uneven and flat tiles.

## Elevation against a start-of-run ground reference (optional)

`elevation.enabled: true` (or `elevation:=true` on the launch file) makes the classifier measure how high or low the floor
is compared with the floor the robot started on. It is off by default.

1. **Reference.** After start, the robot is assumed to stand on flat floor. For `elevation.calibrate_s` (2 s) the lidar floor
   in front of it is collected in a fixed, gravity-aligned frame and one plane is fitted (`elevation.py`). A start floor that is
   not flat (`max_rms_m`) or not level (`max_tilt_deg`) is refused and collected again. The result is on the latched
   `~/ground_reference` topic. `~/recalibrate_ground` (Trigger) takes a new reference where the robot stands now.
2. **Patch elevation.** The floor points of the patch the kind is computed from (the same patch `~/classify` starts) are
   compared with the reference plane. The dominant floor level of the patch counts, so a wall or a step edge inside the area
   does not move it.
3. **Output.** `/terrain_kind` gets `elevation_m` (+ higher, - lower than the start floor), `elevation_class`
   (`higher` / `lower` / `level` within `level_tolerance_m`, 4 cm), `slope_deg` (patch against the reference, for ramps),
   `elevation_spread_m`, `elevation_points` and `elevation_ready` (false while there is no reference yet).

`elevation.frame: odom` (default) is absolute and needs an odometry whose z is real; on a raised tile the robot's own height
is part of odom, so the tile ahead is still measured against the start floor. With planar odometry (z always 0) use `gravity`:
only the robot's pitch and roll are removed, so on a raised tile everything reads lower by that height. It needs the
`odom -> base_link` TF and `kinds.enabled`. Not tested on the real robot: check `elevation_m` on flat floor (should be about 0) and
in front of a block of known height first.

The explorer's mission manager asks for the target tile, so the answer (elevation included) is stored on that tile: it is in the
tile's terrain in `/tile_map` and `tiles.json`, logged, sent as an `elevation` event, and shown on the tile in RViz (`~/terrain_markers`, second line of its label, level tiles too).

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

## Running on its own

The classifier needs only the lidar cloud and TF (`odom → base_link`, `base_link → livox_frame`), not the explorer,
SLAM or Nav2.

```bash
# continuous (about 5 Hz)
ros2 launch lite3_terrain_classifier terrain_classifier.launch.py
# on demand: each call starts a fresh patch and classifies for trigger_duration_s (1.5 s)
ros2 launch lite3_terrain_classifier terrain_classifier.launch.py trigger_mode:=true
ros2 service call /terrain_classifier/classify std_srvs/srv/Trigger
ros2 topic echo /terrain_kind
```

Useful options: `use_depth:=false` (lidar only), `publish_lidar_tf:=true` (with `lidar_x`, `lidar_z`, ...) when
nothing else publishes `base_link → livox_frame`. On a recorded bag (it needs `/livox/points`, `/tf` and `/tf_static`):
`ros2 bag play <bag> --topics /livox/points /tf /tf_static`.

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
