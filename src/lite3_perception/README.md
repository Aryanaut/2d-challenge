# lite3_perception

Watches the Lite3's front camera and publishes what it finds for the mission reporter. It needs no PyTorch on the robot: it runs a trained `detector.onnx` with onnxruntime, plus OpenCV.

| What | How | Published as an event? |
|---|---|---|
| **Hand tools** (M01) | `models/detector.onnx` | yes, `type: object`, `label: hand_tools` |
| **Hazard signs** (M02): `high_voltage`, `radioactive`, `chemical`, `biological`, `flammable` | matched against the real arena signs in `hazard_templates/` | yes, `type: hazard` |
| **Survivors** (M03 / M05): printed pictures of people | `models/detector.onnx` | yes, `type: object`, `label: survivor`, with `qr_in_view` true/false |
| **QR codes** | OpenCV | **no**. Only spotted, to set `qr_in_view` on survivors. The decoded text is still published on `/perception/qr_text` |
| Other objects (food, utensils, medical kit, carts, benches…) | `models/detector.onnx` | no, only in `/perception/detections` |

The room name (`scene_type`) is **not** worked out here (`scene_enable: false`). A separate scene package provides it.

## Topics

| Topic | Type | Content |
|---|---|---|
| in: `/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | front camera (`image_topic`) |
| out: `/perception/event` | `std_msgs/String` (JSON) | one message per confirmed, new hand tool / hazard / survivor. **The mission reporter reads this** |
| out: `/perception/keyframe` | `sensor_msgs/CompressedImage` | the photo for each event, with boxes drawn on it |
| out: `/perception/detections` | `std_msgs/String` (JSON) | every detection, every frame (for debugging) |
| out: `/perception/qr_text` | `std_msgs/String` | each newly decoded QR text, exactly as read |
| out: `/mission_reporter/observation` | `std_msgs/String` (JSON) | the same findings in the newer reporter's format (unused by the current reporter, harmless) |

Example event:
```json
{"event_id": "5748e440", "type": "object", "label": "survivor", "confidence": 0.71,
 "bbox_xyxy": [60, 40, 300, 400], "location": {"bearing_deg": 4.2, "xy": [2.04, 3.06]},
 "robot_pose": {"x": 1.5, "y": 2.0, "yaw_deg": 90.0, "frame": "map"},
 "qr_in_view": true, "bbox_h_frac": 0.75, "confirm_hits": 3, "scene_type": ""}
```

## How an item becomes an event

1. It is detected in at least **N frames within 2 s**: 3 by default, 4 for `hand_tools` (`confirm_hits`, `class_confirm_overrides`).
2. **Where it is:**
   - with depth on: the map position of the object
   - with depth off (the default): the robot's map position
3. It is **de-duplicated**: the same label near the same spot is not reported again.
4. **Hazard signs** must be a full yellow triangle with a dark border, not cut off at the picture edge. Uncertain matches are flagged `hazard_certain: false`, and the reporter ignores those.

## Install

```bash
cd ~/2d/src && unzip lite3_perception.zip
# put your trained model here (exact name):
cp ~/detector.onnx ~/2d/src/lite3_perception/models/detector.onnx
pip install onnxruntime            # once; add --break-system-packages if pip refuses
cd ~/2d && colcon build --packages-select lite3_perception && source install/setup.bash
```

## Run

```bash
ros2 launch lite3_perception perception.launch.py
```
In the startup log, check for:
- `model .../detector.onnx (onnx), classes: [...]`
- `hazard classes: ['biological', 'chemical', 'flammable', 'high_voltage', 'radioactive']`
- `listening on /camera/color/image_raw/compressed`

A red `pthread_setaffinity_np failed` line is harmless.

Watch it work:
```bash
ros2 topic echo /perception/event --full-length
ros2 topic echo /perception/detections --full-length
ros2 topic hz /perception/detections          # aim for >= 2 per second
```

## Main settings (`config/detector.yaml`, rebuild after editing)

| Setting | Default | Meaning |
|---|---|---|
| `image_topic` / `camera_info_topic` | `/camera/color/image_raw/compressed` / `/camera_info` | camera input |
| `use_depth` | `false` | `true` = real object positions (needs depth aligned to colour) |
| `map_frame` / `base_frame` | `map` / `base_link` | for positions (needs SLAM). `odom` works for bench tests |
| `report_objects` | `["survivor", "hand_tools"]` | the only objects ever turned into events |
| `class_conf_overrides` | `{"survivor": 0.35, "hand_tools": 0.45}` | confidence needed per label (higher = stricter) |
| `class_confirm_overrides` | `{"hand_tools": 4}` | frames needed per label |
| `qr_report` | `false` | `true` = also send QR codes as events (M04) |
| `qr_assoc_s` | `3.0` | a QR seen within this many seconds of a survivor = "has QR" (M03) |
| `scene_enable` | `false` | `true` = guess the room from objects in view (`scene_rules`, `scene_decisive`) and publish it on `scene_topic` |
| `hazard_min_score` / `hazard_min_margin` | `0.6` / `0.02` | how sure a sign match must be |
| `detect_hz` | `3.0` | frames processed per second |

## Folders

| Folder | Content |
|---|---|
| `models/` | put `detector.onnx` here |
| `hazard_templates/` | one picture per hazard sign. **The file name is the label.** To change a sign, replace its PNG and rebuild |
| `training/` | `train_lite3_detector.ipynb` (Colab: auto-label, train, export `detector.onnx`); `bag_to_frames.py` (rosbag to training frames) |
| `lite3_perception/llm_event_bridge.py` | optional: asks the LAN LLM to double-check events (`with_bridge:=true`). `query_llm()` must be filled in first |

## Retraining the model

1. Record a bag with the robot's camera:

   ```bash
   ros2 bag record -o ~/bags/myscene /camera/color/image_raw/compressed
   ```
2. Turn it into frames:

   ```bash
   python3 training/bag_to_frames.py ~/bags/myscene ~/train_frames/frames
   ```
3. Zip the frames and run `training/train_lite3_detector.ipynb` in Colab (T4 GPU). It saves its progress to Google Drive.
4. Copy the new `detector.onnx` into `models/` and rebuild.
