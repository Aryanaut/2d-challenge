# lite3_mission_reporter

Turns what the robot finds into the organisers' result messages, **one per mission (M01–M05)**. It publishes them on ROS, saves them to a file, and (optionally) POSTs them to the task server.

```json
{"group_id": "group01_s3xy", "task_id": "M01", "task_type": "object", "scene_type": "workshop",
 "label": "hand_tools", "pos": [3.06, 2.04], "message": ""}
```

## Where each part comes from

| Input topic (all `std_msgs/String`) | From | Used for |
|---|---|---|
| `/perception/event` (JSON) | `lite3_perception` | hand tools, hazard signs, survivors (with `qr_in_view`) |
| `scene_topic` (default `/mission/scene_type`) | **your scene package** | `scene_type` of every report (the last message received). Plain text `kitchen` or JSON `{"scene_type": "kitchen"}` |
| `position_topic` (default `""` = off) | **your position package** | `pos` of every report except M04 (the last position received). Type set by `position_msg_type` |
| `qr_message_topic` (default `/perception/qr_text`) | **your QR package** | answers M04; its text becomes `message`. Plain text, or JSON with `message` / `text` / `decoded` |
| `/mission/manual_report` (JSON) | you, by hand | send or override any field of a result |
| `/perception/llm_result` (JSON) | `llm_event_bridge` | only with `use_llm_result: true` |

| Output | Content |
|---|---|
| `/mission/result` (`std_msgs/String`, latched) | each result JSON, once per mission |
| `~/lite3_mission_results/results.json` | all results this run (rewritten each time) |
| `~/lite3_mission_results/results.jsonl` | every result from every run, plus the server's replies |
| `POST server_url` | each result, if `server_url` is set |

## The five missions (fixed)

| Mission | `task_type` | Answered by |
|---|---|---|
| M01 | `object`, `label: hand_tools` | the first `hand_tools` event |
| M02 | `hazard`, `label` = the sign found | the first **certain** hazard sign (any of the 5) |
| M03 | `survivor` | a survivor **with** a QR code beside it (`qr_in_view`, or a QR seen within 1.5 m) |
| M04 | `qr`, `pos: null`, `message` = exact QR text | the first text on `qr_message_topic` |
| M05 | `survivor` | a survivor **without** a QR, seen close enough to be sure (it fills >= 25% of the image height) |

**Rules:**
- Each mission is answered **once**. Later sightings log `no open matching task`.
- The **same survivor** (within 1 m) can't answer both M03 and M05.
- With `require_position: true`, a sighting without a map position doesn't use up a mission. It logs `no position ... task kept open` and waits for a sighting that has one.

## Install

```bash
cd ~/2d/src && unzip lite3_mission_reporter.zip
cd ~/2d && colcon build --packages-select lite3_mission_reporter && source install/setup.bash
```

## Configure (`config/mission_reporter.yaml`, rebuild after editing)

| Setting | Set to |
|---|---|
| `group_id` | your exact group ID + pin |
| `scene_topic` | the topic your scene package publishes |
| `qr_message_topic` | the topic your QR package publishes |
| `position_topic` | the topic your position package publishes (`""` = use perception's position) |
| `position_msg_type` | `string` (`"[x, y]"`, `"x,y"`, `{"x":..,"y":..}`), `point_stamped`, `pose_stamped`, `pose_cov`, `odometry`, `point`, `pose` |
| `require_position` | `true` for real runs; `false` to allow `"pos": null` when testing without SLAM |
| `server_url` | `""` (no sending), `"http://127.0.0.1:8001/task_report"` (practice server), or the organisers' address |

Already set for the fixed missions (don't change):
```yaml
task_ids:        ["M01", "M02", "M03", "M04", "M05"]
task_types:      ["object", "hazard", "survivor", "qr", "survivor"]
task_labels:     ["hand_tools", "", "", "", ""]
task_conditions: ["", "", "qr", "", "no_qr"]
```

Other settings:

| Setting | Default | Meaning |
|---|---|---|
| `scene_from` | `topic` | `topic` = only the scene package decides the room |
| `same_survivor_radius_m` | `1.0` | survivors closer than this = the same one |
| `qr_near_survivor_m` | `1.0` | a QR position (map frame, from `/qr/visible`) this close to the survivor's map position = that survivor has a QR |
| `no_qr_wait_s` | `10.0` | a survivor with no QR yet waits this long before it can be sent as M05 |
| `no_qr_min_bbox_h_frac` | `0.25` | how close (by size in the image) a survivor must be before calling it "no QR" |
| `use_llm_result` | `false` | `true` = use LLM-checked results from `llm_event_bridge` |
| `output_dir` | `~/lite3_mission_results` | where the result files go |

## Run

```bash
rm -f ~/lite3_mission_results/results.json      # optional: clear the last run's file
ros2 launch lite3_mission_reporter mission_reporter.launch.py
ros2 topic echo /mission/result --full-length --qos-durability transient_local    # watch answers
```
The startup log shows:
- `pos from <position topic> (<type>)`, then `first position from ...` once a position arrives
- `scene from <scene topic>`
- `M04 message from <QR topic>`
- `mission_reporter up: group=... tasks=[...] server=...`

**Between runs:** restart the node. It forgets answered missions on restart.

## Test by hand

```bash
ros2 topic pub --once /mission/scene_type std_msgs/String "{data: kitchen}"
ros2 topic pub --once /perception/qr_text std_msgs/String "{data: 'Please help me.'}"     # -> M04
ros2 topic pub --once /mission/manual_report std_msgs/String \
  "{data: '{\"task_id\": \"M01\", \"task_type\": \"object\", \"label\": \"hand_tools\", \"pos\": [3.06, 2.04]}'}"
```

## Practice server

`mock_task_server.py`, kept in `~/2d/scripts`, stands in for the organisers' server. It checks each result's format, and you can view the results in a browser.
```bash
python3 ~/2d/scripts/mock_task_server.py           # then server_url: "http://127.0.0.1:8001/task_report"
```
To view the results, forward port 8001 in VS Code's **Ports** tab, then open `localhost:8001`. **[Clear]** empties the table.

## Log messages (normal, not errors)

| Message | Meaning |
|---|---|
| `no open matching task, not reported` | that mission is already answered |
| `no position ... task kept open` | no map position yet (SLAM / TF) |
| `same one already reported as M03, skipped` | the same survivor was seen again |
| `too far to be sure ... not reported as M05 yet` | it needs to get closer to check for a QR |
| `not sure which sign, not reported` | unclear hazard sign |
| `qr task already answered, not reported` | another QR text arrived; the first one counts |
| `server M01: {...}` / `send FAILED` | the server's reply / the server is unreachable (the result is still saved to file) |
