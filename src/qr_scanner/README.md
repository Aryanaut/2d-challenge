# qr_scanner

Now supports **several cameras at once** (front + left + right); see "Multiple cameras" below.

A ROS 2 node that reads QR codes on the arena walls from the front camera's **colour** stream, at any distance it can resolve. It decodes ("decrypts") the content, works out where each code is on the map, and reports each code once.

The arena codes seen so far contain single letters: the far code reads `I` and the near one `J`. What a letter means comes from the mission brief, so the meaning is configured (`key_map`) rather than hard-coded.

## How it reads codes at different distances

| Step | What it does | Limit / cost |
|---|---|---|
| 1. zbar on the full frame | reads most codes | down to ~50 px wide, ~25 ms at 1280×720 |
| 2. QR finder (OpenCV, ArUco-based) | spots codes it can't read yet | ~15 ms |
| 3. Zoom on each spotted-but-unread code | crops it, enlarges it to ~240 px, boosts contrast, retries zbar and OpenCV | down to ~35–40 px when sharp |
| 4. Still unread | published as a "move closer / hold still" candidate with its bearing | — |

Measured on your arena photos, where code size means the width of the black square in pixels:

| Code size | Result |
|---|---|
| ≥ 70 px | reads even with strong motion blur |
| ~50 px | reads while walking slowly |
| ~35–40 px | reads only when the robot is still |
| < 30 px | not readable; get closer |

**Distance this gives you.** Distance ≈ fx × QR size / pixels. Assuming a 9 cm code and a typical depth camera with a ~69° horizontal field of view:

| Colour stream | fx | Read while walking (50 px) | Read when still (35 px) |
|---|---|---|---|
| 1280×720 | ~910 | ~1.6 m | ~2.3 m |
| 640×480 | ~465 | ~0.8 m | ~1.2 m |

**Use the highest colour resolution the camera offers.** During exploration, pause for about half a second facing each wall. The viewpoint scans already do this.

## Topics

| Topic | Type | Contents |
|---|---|---|
| `/qr/detections` | `std_msgs/String` (JSON) | Published once per new code, after it is read in 2 frames: `text`, `decoded`, `map_point`, `distance_m`, `bearing_deg`, `index`, time |
| `/qr/visible` | `std_msgs/String` (JSON) | Every scan: codes in view, including unread candidates with a "move closer" hint and bearing. Use it to steer towards a code. |
| `/qr/sequence` | `std_msgs/String` (latched) | Distinct codes in the order found, e.g. `IJ` |
| `/qr/markers` | `visualization_msgs/MarkerArray` | Codes on the map in RViz |
| `/qr/<camera>/debug_image/compressed` | `sensor_msgs/CompressedImage` | Annotated frames per camera (`front`, `left`, `right`): green = read, orange = spotted but not yet read |
| `/qr_scanner/reset` | `std_srvs/Trigger` | Forget found codes between practice runs |

Every confirmed code is also appended to `~/qr_log.csv` with its time, letter, meaning and coordinates, ready for the competition report.

## Setup

1. **Install the decoder library and copy the package in, then build.**
   ```bash
   sudo apt install libzbar0 && pip install pyzbar     # zbar decoder (much better at small codes)
   cp -r qr_scanner ~/ros2_ws/src/
   cd ~/ros2_ws && colcon build --symlink-install --packages-select qr_scanner
   source install/setup.bash
   ```
2. **Find your camera's topics:** `ros2 topic list | grep -i -E "color|depth|info"`. You need:
   - a colour image (raw or `/compressed`),
   - its `camera_info`,
   - optionally a depth image **aligned to colour** (RealSense: `align_depth.enable:=true`).

   With no depth, distance is estimated from the printed QR size, so set `qr_size_m`.
3. **Measure the printed QR** (the black square, not the paper) and set `qr_size_m` in `config/qr_scanner.yaml`.
4. **Try it without ROS first:**
   ```bash
   ros2 run qr_scanner qr_test --camera /dev/video4 --fx 910 --qr-size 0.09
   ros2 run qr_scanner qr_test photos/*.jpg --save annotated/
   ```
5. **Run it:**
   ```bash
   ros2 launch qr_scanner qr_scanner.launch.py \
       image_topic:=/camera/color/image_raw \
       camera_info_topic:=/camera/color/camera_info \
       depth_topic:=/camera/aligned_depth_to_color/image_raw
   ros2 topic echo /qr/detections
   ```
   Map positions need TF from the camera frame to `map`. That means SLAM running, plus the static transform for the camera mount, as in Lab 7. Without it, positions are still reported in the camera frame.

## Decoding the content ("decrypt")

Set this in `config/qr_scanner.yaml` once you know the scheme from the mission brief:

```yaml
decode_mode: raw                       # raw | base64 | caesar | json | auto
caesar_shift: 0                        # for caesar
key_map: ["I:storeroom", "J:kitchen"]  # letter -> meaning, applied after decoding
```

With a `key_map`, a multi-letter code such as `IJ` becomes `storeroom kitchen`. `/qr/sequence` keeps the order the codes were found in, in case the letters together spell a word.

## Tests (no ROS needed)

```bash
cd qr_scanner && python3 -m pytest test/ -q
```

## Multiple cameras

One node reads every camera and keeps one shared list of codes. A code seen by the front camera and later by a side camera is reported **once**. Every report says which `camera` saw it. `bearing_deg` is relative to the **robot** (+ = left), using each camera's mounting yaw.

Cameras are named in `cameras: [front, left, right]`. Give each one either a topic or a device; one with neither is switched off:

| Launch argument | Meaning |
|---|---|
| `<name>_image_topic` | ROS image topic (add `<name>_compressed:=true` for a `/compressed` topic) |
| `<name>_device` | read a `/dev/video*` device directly, no camera driver needed |
| `<name>_camera_info` | calibration topic; without it, focal length comes from `<name>_hfov_deg` in the YAML |
| `<name>_depth_topic` | depth **aligned** to that colour image (otherwise distance comes from the QR size) |

Mounting is set in `config/qr_scanner.yaml`:

- `<name>_mount_yaw_deg`: 0 front, +90 left, −90 right
- `<name>_mount_pitch_deg`: tilt down
- `<name>_hfov_deg`: horizontal field of view
- `<name>_publish_tf` and `<name>_mount_xyz`: with `publish_tf: true`, the node publishes `base_link → <name>_qr_optical`, so side-camera codes get map positions when SLAM runs. Measure `mount_xyz` (metres, x forward, y left, z up).

Example:
```bash
ros2 launch qr_scanner qr_scanner.launch.py \
    front_image_topic:=/camera/color/image_raw front_camera_info:=/camera/color/camera_info \
    left_image_topic:=/left_cam/image_raw right_image_topic:=/right_cam/image_raw
```

Cost: about 40 ms per camera per scan, so 3 cameras at 5 Hz ≈ 0.6 CPU core. If the robot struggles, lower `scan_rate_hz` (e.g. 3) or use `/compressed` topics.

### Cameras whose codes are read elsewhere (`<name>_detections_topic`)

For side webcams on the laptop/Pi, `cam_sender.py` reads the codes at 1920×1080 and the robot's `net_receiver` publishes them on `/left/qr_codes` and `/right/qr_codes`. Set `left_detections_topic:=/left/qr_codes` (same for right), or use `net_cameras:=true` in `test_session.launch.py`. This node then:

- **does not decode** that camera; the preview image topic is used only for the debug picture, with the boxes scaled down;
- builds the intrinsics from the **full-res** size and the `hfov_deg` in each message, because the corners are full-res pixels; `<name>_hfov_deg` and the preview's `camera_info` are ignored for that camera;
- uses the robot's arrival time as the timestamp for TF.

Bearings, distances, `/qr/detections`, `/qr/visible`, `/qr/sequence`, markers and the CSV log work exactly as for decoded cameras.

