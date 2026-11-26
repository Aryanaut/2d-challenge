# lite3_teleop_vlm

Drive the Lite3 from the keyboard while you test the VLM, with surface and QR detection running alongside.

## Architecture

```
        KEYBOARD (SSH terminal on the robot)
                    │
            ┌───────▼────────┐  /cmd_vel (Twist, hold-to-move, dead-man 0.6 s)
            │ teleop_console ├──────────────────────────────────────────► Lite3 driver
            │  (terminal UI) ├──/simple_cmd (auto / manual / stand-sit)─►
            └──┬─────────▲───┘
   /vlm/request│         │ /vlm/result, /vlm/status, /surface/*, /qr/*
               │         │
        ┌──────▼─────────┴──┐   one request at a time   ┌──────────────────────┐
        │    vlm_broker     ├──── client.py (Lab 6) ───►│ VLM gateway          │
        │ queue + priorities│◄──────────────────────────│ 192.168.10.141:8000  │
        │ latest frame/cam  │                           └──────────────────────┘
        │ logs every query  │◄── front camera (colour)
        └──────▲────────────┘◄── down camera (Logitech)
   /vlm/request│ /vlm/result
        ┌──────┴────────────┐          ┌──────────────┐
        │  surface_monitor  │          │  qr_scanner  │  (local OpenCV/zbar, no VLM)
        │ vlm_backend:broker│          │              │
        └───────────────────┘          └──────────────┘
```

**Why a broker:** the gateway allows **one active request per group**. Without the broker, your keyboard tests and `surface_monitor` would collide and get errors. The broker is the only thing that calls the gateway:

- It queues requests, with your keypresses (priority 10) ahead of background nodes (priority 5).
- It keeps one waiting request per sender.
- It attaches the latest frame from the camera you asked for.
- It answers every request, including dropped ones.

Later, the mission manager sends its requests the same way.

## Messages

`/vlm/request` (`std_msgs/String`, JSON) — any node can send one:
```json
{"id": "a1b2", "source": "mission_manager", "camera": "front", "prompt": "room",
 "priority": 5, "max_age_s": 30}
```
- `prompt` is a file in `prompts/`: `scene`, `room`, `hazard`, `survivor` or `surface`. You can send `"prompt_text": "..."` instead.
- `"snapshot_only": true` just saves both camera frames.

`/vlm/result` returns `id`, `source`, `ok`, `result` (the model's text), `parsed` (if it answered JSON), `error`, `latency_s`, `queue_wait_s` and `saved`.

`/vlm/status` (1 Hz) shows the queue, the request in flight, ok/failed counts and camera frame ages.

The broker saves every query to `~/vlm_tests/<date>/` as image + prompt + answer, ready for reviewing prompts or for classifier training.

## Keys

| Group | Key | Action |
|---|---|---|
| Drive | `w` / `s` or ↑ / ↓ | forward / back |
| | `a` / `d` or ← / → | turn left / right |
| | `q` / `e` | strafe left / right |
| | **space / x / Esc** | **stop** |
| | `+` / `-` | faster / slower (capped at `max_linear`) |
| Robot | `n` | auto/navigation mode (`0x21010C03`): robot follows `/cmd_vel` |
| | `m` | manual mode (`0x21010C02`): handheld controller |
| | `t` `t` | stand / sit (`0x21010202`), press twice within 1.5 s |
| VLM | `v` | ask about the **front** camera with the selected prompt |
| | `b` | ask about the **down** camera (surface prompt) |
| | `p` / `o` | next / previous front prompt (scene, room, hazard, survivor) |
| | `k` | snapshot both cameras |
| Other | `r` `r` | reset `surface_monitor` + `qr_scanner` (press twice) |
| | Ctrl+C | quit: stops the robot, back to manual if you switched to auto |

**How driving works:**
- **Hold to move.** The robot moves only while a key repeats, and stops 0.6 s after you let go. Over SSH, key repeat starts after about 0.3–0.5 s, so there's a short pause at the very start of each press; that's normal.
- **Latch mode.** If you prefer teleop_twist_keyboard's "keeps going until stopped", set `latch: true` in `config/test_session.yaml`.
- **Speed.** It starts at 0.3 m/s, as in the lab handout.

The screen shows your current command, the robot mode you last sent, the surface state and course-complete flag, the QR codes found, the VLM queue and last answer, and a short event log.

## Setup (once)

```bash
cd ~/ros2_ws/src && unzip lite3_ws_src.zip   # lite3_teleop_vlm, surface_monitor, qr_scanner
cd ~/ros2_ws && colcon build --symlink-install
source install/setup.bash
```

`lite3_teleop_vlm` runs without the other two; their parts of the screen just stay blank.

## A test session

**On the robot's group computer: SSH terminal 1** (broker plus detectors):
```bash
source ~/transfer_ws/install/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch lite3_teleop_vlm test_session.launch.py group:=group07 \
    client_script:=/home/user/scene_understanding/client.py \
    front_image_topic:=/camera/color/image_raw \
    down_image_topic:=/down_camera/image_raw start_down_camera:=true \
    down_video_device:=/dev/v4l/by-id/<logitech-webcam>
```
- Check the real topic names first with `ros2 topic list`.
- Leave out `start_down_camera` if the webcam is already published elsewhere.
- Use `start_qr:=false` or `start_surface:=false` to test the VLM alone.

**SSH terminal 2** (the console; it needs its own terminal):
```bash
source ~/transfer_ws/install/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 run lite3_teleop_vlm teleop_console --ros-args \
    --params-file ~/ros2_ws/install/lite3_teleop_vlm/share/lite3_teleop_vlm/config/test_session.yaml
```

Then:
1. Stand and calibrate the robot with the handheld controller, as in the lab.
2. **Check no other group is driving it.**
3. Press `n` for auto mode. Drive with `w` `a` `s` `d`.
4. Face something and press `v`. The answer appears in about 3–8 s.
5. Use `p` to switch to `room` / `hazard` / `survivor` and press `v` again.
6. Walk over each surface and watch the Surface line.
7. Press `b` to ask the VLM directly about the floor.
8. Ctrl+C when done: the robot stops and returns to manual mode.

**Why run the console on the robot, not the workstation:**
- The `n` / `m` / `t` keys need the Lite3's `transfer_interfaces` messages, which are only there after `source ~/transfer_ws/install/setup.bash`. Without them the console still drives and tests the VLM, but tells you to use `lite3_auto` / `lite3_manual` in another terminal.
- Keyboard-to-robot latency doesn't depend on Wi-Fi.

## Safety

- The dead-man stop is on by default. Space, x and Esc all stop immediately.
- Stand/sit and reset need a double press.
- Quitting always publishes zero velocity, and returns to manual mode if this console switched to auto.
- Two groups share each robot: check before pressing `n`.
- Keep the handheld controller within reach. In manual mode it overrides everything.

## Tests (no ROS needed)

```bash
cd lite3_teleop_vlm && python3 -m pytest test/ -q
```
