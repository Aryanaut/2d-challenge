# lite3_mission_reporter

Turns perception events into the result messages the organisers require (one JSON per answered task, `M01`…`Mxx`). It publishes them on a topic, writes them to disk, and optionally POSTs them to the organisers' server.

Everything lives in one node: [mission_reporter_node.py](lite3_mission_reporter/mission_reporter_node.py).

## Data flow

```
/perception/event ──────────┐   (default)
/perception/llm_result ─────┤   (use_llm_result:=true, replaces /perception/event)
/mission/scene_type ────────┼──► mission_reporter ──► /mission/result  (std_msgs/String, JSON)
/mission/manual_report ─────┘                    ├──► <output_dir>/results.jsonl + results.json
                                                 └──► POST server_url  (optional)
```

### Topics

| Topic | Dir | Type | Notes |
|---|---|---|---|
| `/perception/event` | in | String (JSON) | From the object detector. `type` is `object`, `hazard` or `qr`. |
| `/perception/llm_result` | in | String (JSON) | Only subscribed when `use_llm_result` is true. Uses the LLM's verified label/room. |
| `/mission/scene_type` | in | String | Current room name, e.g. `kitchen`. Last message wins. |
| `/mission/manual_report` | in | String (JSON) | Manual override. Any of `task_id`, `task_type`, `scene_type`, `label`, `pos`, `message`. Marks that task done and emits immediately. |
| `/mission/result` | out | String (JSON) | One message per answered task. QoS is reliable and transient-local, so late subscribers get them. |

### Result format

```json
{"group_id": "group01_s3xy", "task_id": "M01", "task_type": "object",
 "scene_type": "clinic", "label": "hand_tools", "pos": [3.06, 2.04], "message": ""}
```

- `pos` is the map-frame `[x, y]`, rounded to 2 decimals. The reporter tries `location.xy`, then `location.position_map`, then falls back to `robot_pose`.
- `qr` results have `pos: null`, an empty `label`, and the decoded text in `message`.
- `survivor` results have an empty `label`.

## How a task gets answered

Tasks are defined in the YAML as parallel lists (same index = same task). They are fixed for both practice and competition.

| Parameter | Meaning |
|---|---|
| `task_ids` | `M01`… |
| `task_types` | `object`, `hazard`, `survivor` or `qr` |
| `task_labels` | Required label. `""` accepts any label of that type. |
| `task_scenes` | `""` reports the room perception says the robot is in. A name forces or requires that room. |
| `task_conditions` | Survivor tasks only. `qr` = survivor with a QR beside it. `no_qr` = survivor without one. |

For each incoming event (`process()`):

1. **Filter.**
   - Drop it if the source is not confirmed or `confidence < min_confidence`.
   - On the `/perception/event` path, drop hazards with `hazard_certain == false`. Only the LLM path may decide those.
   - If `require_position` is true and a non-QR event has no position, drop it and keep the task open.
2. **Classify.** The event becomes `qr`, `hazard` or `survivor` (label `survivor`). Anything else is `object`.
3. **Match a task.** The first open task of that type whose `label` is empty or equal to the event label. With `report_each_task_once` (default), a task closes after its first answer.
4. **Pick the scene.** `task_scenes` if set. Otherwise the LLM's `room_type`, then the event's `scene_type`, then the last `/mission/scene_type`.
5. **Emit.** Publish, append to the files, and POST if `server_url` is set.

### Survivor logic

Survivors have extra rules (`pick_survivor_task()`):

- **De-duplication.** A survivor within `same_survivor_radius_m` of one already reported is skipped.
- **QR association.** QR positions are remembered. A survivor "has a QR" if the event has `qr_in_view`, or a QR was seen within `qr_near_survivor_m`.
- **Conditions.** `qr` tasks only take survivors with a QR. `no_qr` tasks only take survivors without one.
- **Distance check for `no_qr`.** The survivor must fill at least `no_qr_min_bbox_h_frac` of the image height. A distant survivor might have an unseen QR, so it is not reported yet.
- **Unknown room.** If the task requires a scene and the room is unknown, the survivor is held when `hold_unknown_scene_survivors` is true.

## Outputs

- `/mission/result` topic, one message per task.
- `<output_dir>/results.jsonl` (default `~/lite3_mission_results`) is append-only. It holds each result, plus a `{"server_reply_for": ...}` line per server POST.
- `<output_dir>/results.json` is the full list of results, rewritten on each emit.
- Server POST, if `server_url` is set. It runs in a background thread, so a slow or failed server never blocks the node. A failure is logged and the result stays in the files.
- `republish_hz > 0` re-sends all results periodically.

## Running

```bash
ros2 launch lite3_mission_reporter mission_reporter.launch.py
ros2 launch lite3_mission_reporter mission_reporter.launch.py url:=A use_llm_result:=true
```

Launch arguments ([mission_reporter.launch.py](launch/mission_reporter.launch.py)):

| Arg | Default | Effect |
|---|---|---|
| `use_llm_result` | `false` | Listen to `/perception/llm_result` instead of `/perception/event`. |
| `url` | `""` | `A` / `URLA` posts to `<server>/task_report_A`. `B` / `URLB` posts to `<server>/task_report_B`. A full `http(s)://` URL is used as given. Empty uses `server_url` from the YAML. Anything else raises an error. |

The organiser server address is hard-coded as `SERVER` at the top of the launch file. Update it if the network changes.

Set the room as the robot moves:

```bash
ros2 topic pub --once /mission/scene_type std_msgs/String "{data: kitchen}"
```

Send a manual result:

```bash
ros2 topic pub --once /mission/manual_report std_msgs/String \
  '{data: "{\"task_id\": \"M01\", \"label\": \"hand_tools\", \"pos\": [1.0, 2.0]}"}'
```

Watch the results:

```bash
ros2 topic echo /mission/result
```

## Configuration

All parameters are in [config/mission_reporter.yaml](config/mission_reporter.yaml).

| Parameter | Default | Meaning |
|---|---|---|
| `group_id` | `""` | Your group id, copied exactly from the organisers. Results are rejected if it is empty (the node warns). |
| `default_scene_type` | `""` | Initial room, until `/mission/scene_type` updates it. |
| `use_llm_result` | `false` | Use the LLM-verified result stream. |
| `require_survivor_confirmed` | `true` | (LLM mode) skip survivors the LLM says are not survivors. |
| `min_confidence` | `0.0` | Minimum event confidence. |
| `same_survivor_radius_m` | `1.0` | Radius for treating two survivor sightings as the same one. |
| `hold_unknown_scene_survivors` | `true` | Hold scene-bound survivor tasks while the room is unknown. |
| `qr_near_survivor_m` | `1.5` | A QR within this distance counts as beside the survivor. |
| `no_qr_min_bbox_h_frac` | `0.25` | Minimum survivor size before reporting "no QR". |
| `require_position` | `false` | Don't answer non-QR tasks without a position. The YAML comment suggests turning this on once position sourcing is confirmed. |
| `report_each_task_once` | `true` | Close a task after its first answer. |
| `output_dir` | `~/lite3_mission_results` | Where the result files go. |
| `republish_hz` | `0.0` | Periodically re-send all results (0 = off). |
| `server_url` | `""` | POST target. Empty means topic and files only. |
| `server_timeout_s` | `5.0` | HTTP timeout. |

## Gotchas

- Without a position source (no depth and no `map->base` TF), `pos` is `null`. Turn on `require_position` to hold the task open until a located sighting arrives.
- Uncertain hazards are only reported through the LLM path, so with `use_llm_result:=false` they are silently ignored.
- There is no unit test for this package yet. The logic is in `process()` and `pick_survivor_task()`.
