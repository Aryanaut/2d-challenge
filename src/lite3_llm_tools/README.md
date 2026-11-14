# lite3_llm_tools — AI server client for maze explorers

ROS 2 (Humble) package that holds **everything that talks to the AI server** (the LAN GPT server with the
`POST /analyze` API). It was split out of `lite3_kitchen_explorer` so that any maze explorer can use it:
the explorer sends a small JSON request, the `llm_client` node builds the context from live **camera images
and lidar data**, queries the server, and sends back a JSON answer.

```
 camera ──┐                                                       ┌─► AI server  POST /analyze
 /map  ───┼──► llm_client ── prompts + map image + lidar line ────┤      (group_id, prompt, image)
 /scan ───┘       ▲   │                                           └── {"scene_description", "relevant_to_mission",
                  │   ▼                                                  "evidence", "confidence"}
        /llm/request  /llm/response  /llm/status  /llm/last_result      (all std_msgs/String, JSON)
                  ▲   │
              your explorer (e.g. lite3_kitchen_explorer's environment_reporter)
```

## What the AI sees

| Context | Where it comes from | How the AI gets it |
|---|---|---|
| Camera view | `image_topic` (Orbbec colour camera) | a JPEG, shrunk to fit 1024×768 (the server rejects larger images) |
| Lidar map | `/map` (SLAM toolbox, built from the lidar) | a top-down PNG: walls black, a red band next to walls, explored floor white, unknown grey, the robot, numbered candidates, the tile grid, the trail and the route |
| Lidar ranges | `/scan` | one line of text added to the prompt: `LIDAR (robot frame): nearest obstacle ahead 0.62 m, left 0.35 m, behind clear …` (`lidar_context: false` turns it off) |

## What it does (the tasks)

| Task | What the explorer sends | What it gets back (`result`) |
|---|---|---|
| `describe_tile` | `name`, `tile`, `pose`, `context`, `save_dir` | **a tile report**: keeps the next camera frame, then asks what that tile looks like (prompt `tile_report.txt`, plus the lidar line and your `context`, e.g. which tile and what the map says about its sides): `{tile, scene_description, relevant_to_mission, evidence, confidence, image}`. This is what `lite3_kitchen_explorer` uses: one per tile reached, sent in the background |
| `capture_view` | `name`, `pose`, `save_dir` | keeps the **next** camera frame (and the lidar ranges at that moment) under `name`; `{name, image, lidar}` |
| `describe_views` | `views: [{name, heading_deg, …}]`, `save_dir` | what kind of place each kept frame shows: `views: [{scene_description, relevant_to_mission, evidence, confidence, …} or {error}]`. Failed views are retried (`query_retries`) |
| `place_initial_tile` | `pose`, `candidates: [{id, kind, sides, grid}]`, `size_m` | which proposed 1 m tile placement fits the walls best: `{choice: "B", ai_sides: {N: "wall", …}, answer}` |
| `choose_target` | `mode: "tile"` or `"frontier"`, `candidates`, `pose`, `tile_grid`, `tiles`, `trail`, … | the next target: `{choice: <candidate id or null>, answer, extras: {target_yaw_deg, waypoints}}` |
| `render_map` | same state, plus `route` | draws the map PNG (no AI call) |
| `ping` | — | `{pong: true}` |

The prompts (`prompts/*.txt`) are plain text templates and can be replaced with the `*_prompt_file` parameters.
They are specific to the competition maze (chipboard walls, ramps, beams); change them for another maze.

## Messages

Everything is a JSON object in a `std_msgs/String`, so a run can be recorded and replayed with `ros2 bag`.

| Topic | Direction | Content |
|---|---|---|
| `/llm/request` | explorer → client | `{"id": "<unique>", "task": "choose_target", …}` |
| `/llm/response` | client → explorer | `{"id": "<same>", "task": …, "ok": true, "result": {…}, "elapsed_s": 3.1}` or `{"ok": false, "error": "…"}` |
| `/llm/status` | client → all (latched, 1 Hz) | `{"ready": true, "server": …, "group_set": true, "have_map": true, "have_image": true, "have_scan": true, "queries": 12, "last_error": ""}` |
| `/llm/last_result` | client → all | the result of each AI answer, for a status pane |

An explorer should treat a missing `/llm/status`, `ok: false` or a timeout as "the AI is not available" and carry on
with its own logic. `lite3_kitchen_explorer` does exactly that (`llm_link.py`, about 100 lines, is a ready-made
client for the protocol).

Images and answers are written to `output_dir/run_<time>/` (or the request's `save_dir`): the camera frames,
`<name>.png` (exactly the image the AI saw), `<name>_prompt.json` and `<name>_answer.json`.

## Install and run

```bash
pip install httpx
cd ~/ros2_ws && colcon build --symlink-install --packages-select lite3_llm_tools
source install/setup.bash
ros2 launch lite3_llm_tools llm_client.launch.py server_url:=http://192.168.10.141:8000 group_id:=YOUR_GROUP
```

`lite3_kitchen_explorer`'s launch file starts it for you (and skips it with `use_ai:=false`). Settings are in
`config/llm_client.yaml`; any launch argument left empty keeps the file's value. Quote the group id in the YAML:
a bare number would be read as an integer.

Check the server and the credentials without ROS (from the workspace root):

```bash
./scripts/test_llm.py                   # newest camera frame in ~/lite3_survey_runs, server + group from the config
GROUP_ID=<value> ./scripts/test_llm.py  # try another group id
```

Watch it work: `ros2 topic echo /llm/status`, `ros2 topic echo /llm/last_result`.

## Failure handling

- **Connect errors / HTTP 429, 502, 503, 504**: retried inside the client with back-off (`connect_retries` 3, `connect_backoff_s` 2 s, then 4 s, 8 s). A 4xx such as `invalid_group` is not retried; the error comes back to the explorer.
- **A failed camera view** is retried once more after `error_backoff_s`; if it still fails, that view carries an `error` and the others are unaffected.
- **Image too large**: everything sent is shrunk to `max_image_width` × `max_image_height` (1024×768, the server's limit).

## Reusing it for another explorer

1. Run `llm_client`.
2. Publish a request to `/llm/request` and wait for the `/llm/response` with the same `id`.
3. For a new kind of decision, add a task in `lite3_llm_tools/tools.py` (a method that builds the context, calls
   `client.analyze(image, name, mime, prompt)` and returns a dict), register it in `LlmTools.handle`, and add a prompt in
   `prompts/`. `tools.py` has no ROS code, so it is tested with a fake HTTP server (`test/test_tools.py`).

## Tests

```bash
cd src/lite3_llm_tools && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test -q
```

## Files

| File | Purpose |
|---|---|
| `llm_client.py` | the ROS node: sensors in, `/llm/*` messages |
| `tools.py` | the tasks (no ROS) |
| `perception_client.py` | the `POST /analyze` client, retries, reply parsing |
| `prompts.py` | prompt building and answer parsing (tile letter, target id) |
| `map_render.py` | the top-down map image for the AI |
| `lidar_context.py` | `/scan` → one line of text |
| `image_utils.py`, `gridmap.py`, `tile_view.py` | image conversion / resizing, occupancy-grid helpers, tile-grid geometry from JSON |
