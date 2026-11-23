# stair_climber

This ROS 2 package lets the DEEP Robotics Lite3 **detect the metal stairs and cross them (up, then down) on its own**, without manual input.

```
stair_climber/
├── stair_climber/
│   ├── stair_detector.py      lidar point cloud -> "stairs at 0.8 m, 12 cm steps, turn 3°"   (pure numpy)
│   ├── state_machine.py       SEARCH -> ALIGN -> CLIMB -> ... -> DONE                     (pure python)
│   ├── stair_climber_node.py  ROS 2 node: subscribes to sensors, runs the two files above
│   └── vlm_stair_checker.py   optional camera confirmation via the Lab 6 server (needs 1 TODO)
├── config/stair_climber.yaml  every threshold, all marked # MEASURE where you must check them
├── launch/stair_climber.launch.py
├── test/                      21 tests, run on a laptop without ROS
└── tools/plot_climb.py        plot pitch / height from a rosbag to choose thresholds
```

## How it works

1. **SEARCH**: the lidar looks at a box in front of the robot and builds a height profile. A first rise of 6–20 cm, with nothing steeper than 45° behind it, counts as **stairs**. Five detections in a row are required.
2. **CONFIRM** *(optional, `use_vlm: true`)*: one camera frame goes to the scene-understanding server with the question "staircase ahead? YES/NO".
3. **ALIGN**: the robot turns square to the first step, centres itself (strafing with `linear.y`) and stops 0.45 m before the step.
4. **SWITCH_TO_STAIR_GAIT**: the node sends the RUG gait code on `/simple_cmd` and waits 2 s.
5. **APPROACH → CLIMB_UP → TOP_PLATFORM → DESCEND → CLEAR_BOTTOM**: the robot walks slowly while holding its heading. **IMU pitch** decides each transition (nose-up = climbing, level = on top, nose-down = descending, level = at the bottom). `/leg_odom2` height is a second check.
6. **SWITCH_TO_FLAT_GAIT → DONE**: the robot switches back to the flat gait and stops publishing `/cmd_vel` (control returns to navigation).
7. **ABORT** (from any state): too much roll or pitch, stale sensors, a timeout, or being stuck. The robot stands still and keeps control so nothing else drives it. Your operator takes over.

## Install (on the robot's group computer)

```bash
cp -r stair_climber ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --symlink-install --packages-select stair_climber
source ~/lite3_ws/install/setup.bash   # robot's custom msgs (MotionSimpleCMD)
source ~/ros2_ws/install/setup.bash
```

## Run

```bash
ros2 launch stair_climber stair_climber.launch.py
lite3_auto                                            # robot into Auto mode
ros2 topic pub --once /stair_climber/enable std_msgs/msg/Bool "{data: true}"
ros2 topic echo /stair_climber/debug                  # live view of what it thinks
```
Stop any time with `{data: false}`, or with the controller's STOP button.

## Checklist for the practice arena (23–26 Nov). Do these IN ORDER.

1. **Find the point cloud topic**: `ros2 topic list`, then set `cloud_topic`.
2. **Measure the stairs**: step height (≤12 cm → RUG *General*, ≤15 cm → *H-Step*), stair width, platform length.
3. **Test the gait codes by hand** (on flat ground first). They are already in the config: RUG General `0x21010401`, H-Step `0x21010407`, Flat Slow `0x21010300`, Move Mode `0x21010D06`:
   `ros2 topic pub --once /simple_cmd transfer_interfaces/msg/MotionSimpleCMD "{cmd_code: 0x2101XXXX, type: 0}"`
4. **Check the pitch sign**: launch with `dry_run:=true`, tilt the robot nose-up by hand, and confirm `nose_up_deg` in `/stair_climber/debug` is **positive**. If not, set `nose_up_sign: 1.0`.
5. **Record a manual crossing** in RUG gait:
   `ros2 bag record -o stairs_run1 /imu/data /leg_odom2 <cloud_topic>`
   Then run `python3 tools/plot_climb.py stairs_run1` and set `climb_start_pitch_deg`, `level_pitch_deg` and `min_height_gain` from the plot. If odom z doesn't rise on the stairs, set `use_odom_height: false`.
6. **Test detection on the bag** (the robot can't move in dry run):
   `ros2 launch stair_climber stair_climber.launch.py dry_run:=true autostart:=true use_sim_time:=true`
   `ros2 bag play stairs_run1 --clock`
   Watch `/stair_climber/debug` and the `/stair_climber/marker` line in RViz.
7. **Check the close-range blind zone**: stand the robot 0.45 m from the first step. Is `stairs.detected` still true? If not, increase `align_standoff`.
8. **Live runs**: operator holds STOP. Start with `search_speed: 0.0`, standing about 1 m in front of the stairs.

## Integrating with navigation

The node publishes `/stair_climber/active` (`true` while it owns `/cmd_vel`). Your navigation code should pause or cancel its Nav2 goal while this is `true`, or feed both into a `twist_mux` with stair_climber at the higher priority. When the state reaches `DONE`, `active` goes `false` and navigation resumes.

## Known limitations / TODOs

- Gait codes come from the DEEP Robotics "Motion Host Communication Interface" doc (V1.1.0-0). Test each one by hand once before trusting it.
- `vlm_stair_checker.query_scene_server()` must be filled in from the Lab 6 `client.py`.
- Centring needs both ends of the first step inside the lidar box (`roi_half_width` > half the stair width).
- A single 6–20 cm box wider than 35 cm could look like a step. Turn on the VLM check if the arena has such props.
- The thresholds are guesses until you've done steps 4–5 above. The tests only prove the logic, not the numbers.

## Tests

```bash
python3 -m pytest test/ -q     # needs only numpy + pytest
```
