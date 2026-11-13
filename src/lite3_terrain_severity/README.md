# lite3_terrain_severity: plug-and-play setup

A separate ROS 2 package that runs next to your `lite3_terrain_classifier`. **You don't need to change anything in your existing package.**

It grades the ground ahead and sets the speed:

| Level | Speed |
|---|---|
| smooth | 100 % |
| mild | 70 % |
| moderate | 45 % |
| severe | 25 % |
| extreme | pause (then creep, then `BLOCKED` after 3 pauses) |

---

## Step 1: Open the robot's workspace in VS Code

The code has to run on the Lite3's perception host, so the easiest way is to let VS Code work **directly on the robot**:

1. In VS Code, open **Extensions** (`Ctrl+Shift+X`), search for **Remote - SSH** (by Microsoft) and install it.
2. Connect your laptop to the robot's network (Wi-Fi or Ethernet).
3. Press `F1` and choose **Remote-SSH: Connect to Host…**, then type `user@<perception-host-ip>` (the same login you use with `ssh`). Enter the password when asked.
4. In the new window, choose **File → Open Folder…** and open your workspace, for example `/home/<user>/ros2_ws`. You should see the `src` folder, with `lite3_terrain_classifier` inside it.

> If you develop on your laptop (with ROS 2 installed) and copy to the robot later, just open your laptop's `ros2_ws` folder instead. The steps below are the same.

## Step 2: Drop the package in

1. Unzip `lite3_terrain_severity.zip` on your laptop.
2. Drag the **`lite3_terrain_severity`** folder from your file manager into the `src` folder in VS Code's Explorer panel. With Remote-SSH, this uploads it to the robot.

It should look like this:
```
ros2_ws/
└── src/
    ├── lite3_terrain_classifier/      (yours, unchanged)
    └── lite3_terrain_severity/        (new)
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── resource/lite3_terrain_severity
        ├── lite3_terrain_severity/
        │   ├── __init__.py
        │   ├── severity_core.py       (the maths: levels, pause/creep logic)
        │   └── terrain_severity.py    (the ROS 2 node)
        ├── config/terrain_severity.yaml   (all thresholds and speeds: tune here)
        ├── launch/terrain_severity.launch.py
        └── test/test_severity_core.py
```
**Check:** `package.xml` must be directly inside `src/lite3_terrain_severity/`, not one folder deeper. If unzipping created `lite3_terrain_severity/lite3_terrain_severity/package.xml` one level too deep, move the inner folder up.

## Step 3: Build

Open a terminal in VS Code (**Terminal → New Terminal**, or `` Ctrl+` ``). Because of Remote-SSH, it runs on the robot.

```bash
cd ~/ros2_ws
source /opt/ros/$ROS_DISTRO/setup.bash     # or: source /opt/ros/humble/setup.bash
colcon build --packages-select lite3_terrain_severity --symlink-install
source install/setup.bash
```
With `--symlink-install`, edits to the `.py` and `.yaml` files apply the next time you **restart the node**, without rebuilding.

Optional self-test (11 tests should pass):
```bash
python3 -m pytest -q src/lite3_terrain_severity/test
```

## Step 4: Make sure the classifier feeds it

This node reads the classifier's debug cloud, so the classifier must run with debug clouds on. That's the default:
```bash
ros2 param get /terrain_classifier publish_debug_clouds    # should be: True
ros2 topic list | grep terrain_classifier                  # should list /terrain_classifier/lidar_points
```

## Step 5: Run in report-only mode (safe, doesn't touch velocity)

```bash
ros2 launch lite3_terrain_severity terrain_severity.launch.py
```
In a second terminal (click **+** in the terminal panel; run `source ~/ros2_ws/install/setup.bash` first):
```bash
ros2 topic echo /terrain_severity                # smooth / mild / moderate / severe / extreme
ros2 topic echo /terrain_severity/details        # all metrics + which one set the level
```
Drive the robot over each tile with the remote, and over the stairs, and watch the level. If a tile shows the wrong level, look at `geometry.levels` in the details output to see which metric caused it. Then edit that row in `config/terrain_severity.yaml` and restart the launch.

## Step 6: Let it control the speed (auto mode)

1. Rename your explorer's or Nav2's output topic from `/cmd_vel` to `/cmd_vel_raw`. In their launch file, add to the node that publishes velocity:
   ```python
   remappings=[('/cmd_vel', '/cmd_vel_raw')]
   ```
2. Start the severity node with the governor on:
   ```bash
   ros2 launch lite3_terrain_severity terrain_severity.launch.py use_governor:=true
   ```
3. Check that only this node publishes to the robot:
   ```bash
   ros2 topic info /cmd_vel -v        # Publisher count: 1 (terrain_severity)
   ```

Keep the handheld remote in your hand for the first runs. The remote doesn't go through ROS, so it always overrides.

## Everyday commands

```bash
ros2 topic echo /terrain_severity/state      # RUNNING / CREEP / PAUSED / BLOCKED
ros2 topic echo /terrain_severity/blocked    # true => explorer should pick another frontier
ros2 service call /terrain_severity/reset std_srvs/srv/Trigger                 # clear PAUSED/BLOCKED
ros2 service call /terrain_severity/set_enabled std_srvs/srv/SetBool "{data: false}"   # pass speed through untouched
```

## If something's wrong

| Symptom | Fix |
|---|---|
| `Package 'lite3_terrain_severity' not found` | run `source ~/ros2_ws/install/setup.bash` in that terminal |
| colcon builds nothing | `package.xml` is one folder too deep (see Step 2) |
| Level stuck at **moderate** with reason "stale or missing" | the classifier isn't running, or `publish_debug_clouds` is false |
| Robot doesn't move at all with the governor on | your explorer still publishes `/cmd_vel` (do the remap), or the robot isn't in navigation/auto mode |
| Pauses on ramps | raise `pitch_alarm_deg` above your steepest ramp angle |
| Too slow on the block tiles | raise the `severe` and `moderate` columns of `edge_m` / `bump_m`, or raise `speed_factor` |
| Nav2 aborts during pauses | raise `movement_time_allowance` in Nav2's progress checker |
| Your topic names differ | edit the topic names at the top of `config/terrain_severity.yaml` (`imu_topic`, `odom_topic`, …) |
