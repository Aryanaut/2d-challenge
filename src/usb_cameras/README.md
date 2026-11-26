# usb_cameras

Publishes the left and right USB webcams as standard ROS 2 image topics, so `qr_scanner`, `rqt_image_view` and the VLM broker can use them.

| Topic (per camera: `left`, `right`) | Type | Use |
|---|---|---|
| `/<name>/image_raw` | `sensor_msgs/Image` (bgr8) | nodes on the same computer (`qr_scanner`) |
| `/<name>/image_raw/compressed` | `sensor_msgs/CompressedImage` (JPEG) | viewing over Remote Desktop / other computers |
| `/<name>/camera_info` | `sensor_msgs/CameraInfo` | focal length from `hfov_deg` (uncalibrated) |

Frame IDs are `<name>_camera_optical`.

**How it handles the cameras:**
- Each camera is read in its own thread.
- If a camera is unplugged, it reconnects every 2 s and warns once.
- **MJPG** is used by default, so two webcams fit on one USB 2.0 bus. With raw video, the second camera often delivers no frames.

## Needs: the webcams visible on this computer

The group computer is a container. The webcams must be **passed through** to it, so that `ls /dev/video*` shows them. That's a one-time job for the instructor on the robot's host. For a Proxmox LXC it's usually:

```
lxc.cgroup2.devices.allow: c 81:* rwm
lxc.mount.entry: /dev/video0 dev/video0 none bind,optional,create=file
lxc.mount.entry: /dev/video2 dev/video2 none bind,optional,create=file
```

Then, inside the container, `sudo usermod -aG video user` and log in again.

## 1. Find the cameras

```bash
ros2 run usb_cameras probe_cameras
```

It lists every `/dev/video*`, says which ones give pictures, and saves a photo from each to `~/cam_probe/`. Each webcam creates two devices; only one gives pictures. Open the photos, or hold your hand in front of one camera while probing, to see which is left and which is right.

**Use the `/dev/v4l/by-path/...` names it prints.** They're tied to the USB port, so left and right never swap after a reboot or re-plug.

To check both cameras can stream at the same time:
```bash
ros2 run usb_cameras probe_cameras --both /dev/video0 /dev/video2
```
Each should show at least 5 fps. If not, use a lower resolution (`--width 320 --height 240`) or plug the cameras into different USB ports.

## 2. Run

```bash
ros2 launch usb_cameras usb_cameras.launch.py \
    left_device:=/dev/v4l/by-path/<left>-video-index0 \
    right_device:=/dev/v4l/by-path/<right>-video-index0
```

Use `right_device:=none` to run one camera only. Resolution, fps and field of view are in `config/usb_cameras.yaml`.

Check:
```bash
ros2 topic hz /left/image_raw          # about 15 Hz
ros2 run rqt_image_view rqt_image_view /left/image_raw/compressed
```

## 3. Feed the QR scanner

```bash
ros2 launch lite3_teleop_vlm test_session.launch.py ... \
    left_image_topic:=/left/image_raw  left_camera_info:=/left/camera_info \
    right_image_topic:=/right/image_raw right_camera_info:=/right/camera_info
```

## Tests (no ROS, no cameras)

```bash
cd usb_cameras && python3 -m pytest test/ -q
```

## Cameras on another computer (laptop now, Raspberry Pi later)

When the webcams are plugged into a different computer from the robot's ROS session, that computer streams them over the network. The robot republishes them on the **same topics** (`/left/image_raw`, `/right/image_raw`, `/compressed`, `camera_info`), so nothing else changes.

```
laptop / Pi: scripts/cam_sender.py ──TCP 5600 (JPEG frames)──► robot: ros2 run usb_cameras net_receiver
```

- **Sender:** `scripts/cam_sender.py` is one standalone file; it needs only `pip install opencv-python numpy`, no ROS. It works on Windows and Linux (including a Pi).
- **Connections:** the sender connects *to* the robot, so the laptop/Pi needs no open ports. Both ends reconnect automatically.
- **What's sent:** only the newest frame per camera, so a slow link drops frames rather than building up delay. At 640×480 with JPEG quality 80, that's about 200–300 kB/s per camera.
- **Timestamps:** stamped with the robot's clock on arrival.

**Robot:**
```bash
ros2 run usb_cameras net_receiver            # listens on TCP 5600; logs fps per camera every 10 s
```

**Laptop / Pi:**
```bash
python cam_sender.py --list                                   # find camera numbers, saves cam_<n>.jpg
python cam_sender.py --robot 192.168.1.71 --left 1 --right 2 --preview
```
On a Pi, use device paths: `--left /dev/video0 --right /dev/video2`.

### QR codes read on the sender (default)

`cam_sender.py` captures at **1920×1080** (usually the webcam's widest view) and reads QR codes on the laptop/Pi from that full picture: codes about 3× smaller/further away are read than from a 640-wide image. The robot receives:

- a small preview per camera (`--send-width 640`), published as before on `/left/image_raw` etc.
- the decoded results, published as `std_msgs/String` JSON on **`/left/qr_codes`** and **`/right/qr_codes`**: text, corners in full-res pixels, full-res width/height and `hfov_deg`.

`qr_scanner` reads these via `left_detections_topic` / `right_detections_topic` (or simply `net_cameras:=true` in `test_session.launch.py`). It then does the direction, distance, report-once tracking and logging without decoding the side cameras itself.

| Sender option | Default | Notes |
|---|---|---|
| `--width --height` | 1920 1080 | capture size used for QR reading |
| `--send-width` | 640 | preview width sent to the robot (0 = full) |
| `--scan-hz` | 4 | QR scans per second per camera (Pi: 2) |
| `--hfov` | 70 | horizontal FOV **in that capture mode**; it drives the distances, so measure it |
| `--qr-size` | 0.09 | printed black square (m), for the distance printed on the sender |
| `--no-decode` | off | old behaviour: just a camera link, the robot decodes the preview |

Measuring hfov: stand the camera D metres from a wall, mark the left and right picture edges on the wall (width W); hfov = 2·atan(W / 2D).

