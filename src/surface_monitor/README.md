# surface_monitor

A ROS 2 node that reports which surface the robot is standing on, using the downward-facing Logitech webcam:

| Surface seen | Label | Meaning |
|---|---|---|
| OSB wood (flat boards, panels, rubble blocks, wooden steps) | `wood` | on the course |
| Silver aluminium checker plate | `metal` | climbing the stairs |
| Anything else (grey lab floor, etc.) | `other` | off the course; after time on the course this means **exited / complete** |

## How it decides

1. **Local classifier, every frame (10 Hz, a few ms).** It measures colour and texture on the centre of the image:
   - **Lab floor:** almost no texture.
   - **Metal:** no colour and very strong edges.
   - **Wood:** warm colour and medium texture.

   A 1-second majority vote smooths out single bad frames.
2. **VLM (the Lab 6 gateway).** It is asked whenever the surface changes, plus every 8 s. It goes through your existing `client.py`, so the request format is exactly what the server expects. The VLM can override a weak local vote.
3. **Course complete.** This is latched and only fires when all three of these hold:
   - the robot has spent at least 3 s on wood or metal (so a start area on lab floor doesn't count),
   - "other" has held for 2 s,
   - the VLM confirms "other".

   If the gateway gives no answer within 15 s, it completes on local evidence alone. A VLM answer of wood or metal vetoes the exit.

Modes (`mode:=`): `hybrid` (default), `vlm_only`, `local_only`.

## Topics and services

| Name | Type | Notes |
|---|---|---|
| `/surface/state` | `std_msgs/String` | `wood` / `metal` / `other` / `unknown` (no camera frames), 5 Hz |
| `/surface/on_stairs` | `std_msgs/Bool` | true while on metal; use it to switch the stair gait |
| `/surface/course_complete` | `std_msgs/Bool` | latched (transient local); late subscribers still get it |
| `/surface/vlm_result` | `std_msgs/String` | JSON of every VLM answer, including errors and latency |
| `/surface/debug` | `std_msgs/String` | JSON of features, votes and timers, 1 Hz |
| `/surface_monitor/reset` | `std_srvs/Trigger` | clears `course_complete` between practice runs |

## Setup

1. **Copy the package and build it.**
   ```bash
   cp -r surface_monitor ~/ros2_ws/src/
   cd ~/ros2_ws && colcon build --symlink-install --packages-select surface_monitor
   source install/setup.bash
   ```
2. **Copy the Lab 6 `client.py`** to the machine that runs the node, for example `~/scene_understanding/client.py`. Install whatever it imports (`pip install requests` if it uses requests). If it lives in a venv, pass that venv's python as `client_python`.
3. **Check the webcam is visible** on that machine: `ls /dev/v4l/by-id/`. Use the stable `/dev/v4l/by-id/usb-046d_...-video-index0` path rather than `/dev/video0`, because the depth camera also creates `/dev/video*` devices.
   - On the Lite3, check that your **group computer** container can see the USB device. If it can't, ask the instructors.
   - Install the camera driver if needed: `sudo apt install ros-$ROS_DISTRO-v4l2-camera`.
4. **Lock exposure and white balance** so colour and brightness stay stable. Control names vary, so list them first:
   ```bash
   v4l2-ctl -d /dev/v4l/by-id/<webcam> --list-ctrls
   # e.g. (names differ between kernels):
   v4l2-ctl -d /dev/v4l/by-id/<webcam> -c white_balance_automatic=0 -c white_balance_temperature=4500
   ```
   The camera sits in the robot's shadow. A small white LED next to it makes both the local check and the VLM far more reliable.
5. **Test the VLM path** on one saved frame. This also checks your group ID format and the `client.py` path:
   ```bash
   ros2 run surface_monitor surface_tune frame.jpg --vlm --group group07 \
       --client-script ~/scene_understanding/client.py
   ```

## Tune the local thresholds (important)

The defaults come from phone photos of the arena. Texture numbers depend on camera height, focus and lighting, so re-tune on the robot:

```bash
ros2 run surface_monitor surface_tune --camera /dev/v4l/by-id/<webcam>
```

Hold or walk the robot over each surface: flat wood, wood panels, rubble blocks, wooden steps, metal stairs and lab floor. Note the printed `Lstd`, `grad`, `chroma` and `b` values, then adjust the `th_*` values in `config/surface_monitor.yaml`:

- **Lab floor:** `Lstd` below `th_other_lstd_max` and `grad` below `th_other_grad_max`.
- **Metal:** `chroma` below `th_metal_chroma_max`, and `grad` above `th_metal_grad_strong`, or above `th_metal_grad_min` together with `Lstd` above `th_metal_lstd_min`.
- **Wood:** everything else that is yellowish (`b` at least `th_wood_b_min`).

Values measured on your photos for reference:

| Surface | Lstd | grad | chroma | b |
|---|---|---|---|---|
| Wood (flat, panels, rubble, steps) | 11–41 | 26–54 | 2–18 (pale photo: 2–10) | −1 to 17 |
| Metal stairs | 19–39 | 50–139 | 3–7 | 2–7 |
| Lab floor | 5–8 | 15–16 | 4–18 | −17 to 4 |

## Train a classifier (instead of hand-set thresholds)

Everything runs locally on the CPU: OpenCV and scikit-image for features, scikit-learn for the model.

```bash
pip install scikit-learn scikit-image joblib     # on the machine that will run the node
```

1. **Record frames** from the real downward webcam, mounted at its real height:
   ```bash
   # live, with labels on keys w / m / o (needs a display):
   ros2 run surface_monitor surface_collect --camera /dev/v4l/by-id/<webcam> --session run1
   # or headless over SSH:
   ros2 run surface_monitor surface_collect --camera <dev> --label metal --session stairs1 --seconds 60
   # or from a bag of the robot walking (one bag per surface is easiest):
   ros2 bag record -o wood_flat1 /down_camera/image_raw
   ros2 run surface_monitor surface_extract wood_flat1 --label wood
   ```
   **Aim for 3 or more separate runs per surface.** Use different spots, lighting and times. Cover every kind of wood (flat boards, rubble blocks, wooden steps), plus the metal stairs and all off-course floor. Walk normally so some frames are blurred.
2. **Train:**
   ```bash
   ros2 run surface_monitor surface_train data/ --out ~/surface_model.joblib
   ```
   The scores are measured on whole runs the model never saw. The per-session table flags any run it gets wrong; that usually means that kind of surface or lighting needs more recordings.
3. **Check it live:**
   ```bash
   ros2 run surface_monitor surface_tune --camera <dev> --model ~/surface_model.joblib
   ```
4. **Use it.** In `config/surface_monitor.yaml` set `classifier: model` and `model_path`, then rebuild and launch, e.g. with `mode:=local_only` for no network at all.

Train on the same machine, or at least the same scikit-learn version, as the one that runs the node. Saved models don't load reliably across scikit-learn versions.

## Run

```bash
ros2 launch surface_monitor surface_monitor.launch.py group:=group07 \
    client_script:=/home/user/scene_understanding/client.py \
    start_camera:=true video_device:=/dev/v4l/by-id/<webcam>

ros2 topic echo /surface/state
ros2 topic echo /surface/debug
ros2 topic echo /surface/course_complete
ros2 service call /surface_monitor/reset std_srvs/srv/Trigger   # between runs
```

If the camera is already published elsewhere, leave `start_camera` off and set `image_topic:=/your/topic`. For a compressed topic, also set `compressed: true` in the YAML.

## Tests (no ROS needed)

```bash
cd surface_monitor && python3 -m pytest test/ -q
```
