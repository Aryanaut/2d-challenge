#!/usr/bin/env python3
"""
Mission reporter: perception events -> organiser result messages.

Subscribes
  /perception/event        std_msgs/String  JSON from object_detector (type object | hazard | qr)
  /perception/llm_result   std_msgs/String  JSON from the LLM bridge (only if use_llm_result:=true)
  /mission/scene_type      std_msgs/String  current room name, e.g. "kitchen" (latched-style, last wins)
  /mission/manual_report   std_msgs/String  JSON override: any of task_id/task_type/scene_type/label/pos/message

Publishes
  /mission/result          std_msgs/String  ONE JSON message per answered task, exactly:
      {"group_id": "...", "task_id": "M01", "task_type": "object", "scene_type": "clinic",
       "label": "medical_kit", "pos": [3.06, 2.04], "message": ""}
  qr results have "pos": null and the decoded text in "message".

Optional: server_url (e.g. http://<server>:8001/task_report) -> each result is also POSTed there as JSON;
the reply is logged and appended to results.jsonl.
"""
import json
import math
import os
import threading
import time
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

TASK_TYPES = ("object", "hazard", "survivor", "qr")


def build_result(group_id, task_id, task_type, scene_type, label, pos, message, swap_xy=False):
    """The exact organiser format (key order preserved). swap_xy: report [y, x] instead of [x, y]."""
    if pos and swap_xy:
        pos = [pos[1], pos[0]]
    return {
        "group_id": group_id,
        "task_id": task_id,
        "task_type": task_type,
        "scene_type": scene_type or "",
        "label": label or "",
        "pos": [round(float(pos[0]), 2), round(float(pos[1]), 2)] if pos else None,
        "message": message or "",
    }


def event_task_type(ev, label):
    """Map a perception event to a mission task_type."""
    if ev.get("type") == "qr":
        return "qr"
    if ev.get("type") == "hazard":
        return "hazard"
    if label == "survivor" or ev.get("label") == "survivor":
        return "survivor"
    return "object"


def event_position(ev):
    """Best available map-frame [x, y] of the thing seen (falls back to the robot's own pose)."""
    loc = ev.get("location") or {}
    if loc.get("xy"):
        return loc["xy"][:2]
    pm = loc.get("position_map")
    if pm:
        return [pm["x"], pm["y"]]
    rp = ev.get("robot_pose")
    if rp:
        return [rp["x"], rp["y"]]
    return None


class MissionReporter(Node):
    def __init__(self):
        super().__init__("mission_reporter")
        p = self.declare_parameter
        self.group_id = p("group_id", "").value
        ids = list(p("task_ids", ["M01"]).value)
        types = list(p("task_types", ["object"]).value)
        labels = list(p("task_labels", [""] * len(ids)).value)
        scenes = list(p("task_scenes", [""] * len(ids)).value)
        conds = list(p("task_conditions", [""] * len(ids)).value)
        n = len(ids)
        labels += [""] * (n - len(labels))
        scenes += [""] * (n - len(scenes))
        conds += [""] * (n - len(conds))
        self.tasks = []
        for i in range(n):
            if types[i] not in TASK_TYPES:
                self.get_logger().error(f"task {ids[i]}: unknown task_type '{types[i]}' (use {TASK_TYPES})")
            self.tasks.append({"id": ids[i], "type": types[i], "label": labels[i], "scene": scenes[i],
                               "cond": conds[i], "done": False})
        self.scene = p("default_scene_type", "").value
        self.use_llm = p("use_llm_result", False).value
        self.need_confirmed = p("require_survivor_confirmed", True).value
        self.min_conf = float(p("min_confidence", 0.0).value)
        self.once = p("report_each_task_once", True).value
        self.out_dir = os.path.expanduser(p("output_dir", "~/lite3_mission_results").value)
        hz = float(p("republish_hz", 0.0).value)
        self.server_url = p("server_url", "").value
        self.require_pos = p("require_position", True).value
        self.swap_xy = p("swap_xy", False).value   # reported "pos" is [y, x] (map x/y flipped); internal maths stays [x, y]
        self.same_survivor_r = float(p("same_survivor_radius_m", 1.0).value)
        self.hold_unknown = p("hold_unknown_scene_survivors", True).value
        self.require_scene = p("require_scene", True).value
        self.pending_max = int(p("pending_max", 50).value)
        # "scene:task_id:label": answer that task as soon as the room is identified as `scene`
        self.scene_tasks = [r.split(":") for r in p("scene_tasks", [""]).value if r.count(":") == 2]
        # a scene_tasks answer waits this long: another tile named the same scene meanwhile -> the higher confidence wins
        # (the AI sometimes names a scene it sees BEYOND a wall, from the tile in front of it)
        self.scene_task_hold_s = float(p("scene_task_hold_s", 45.0).value)
        # sightings while the robot faces a scene tile (and this long after) are on THAT tile: its room and position
        self.scene_look_grace_s = float(p("scene_look_grace_s", 3.0).value)
        self.scene_held_timeout_s = float(p("scene_held_timeout_s", 90.0).value)   # no answer for the tile by then: release
        self.looking = None                      # {"tile": "i,j", "pos": [x, y]} while facing a scene tile
        self.looking_ended = (None, 0.0)         # (that look, monotonic time it ended)
        self.held = {}                           # scene tile -> [(monotonic time, (ev, label, confirmed, room))]
        self.scene_task_cands = {}               # task_id -> best scene_tasks answer so far, sent at its deadline
        self.pending = []                        # sightings / QR texts waiting for the room to become known
        self.survivors = []                      # (task_id, [x, y]) already reported
        self.survivor_xy = []                    # their map positions (same-survivor test; pos may be a tile)
        # the room of an object = the room of the scene tile it stands on (its map position on the mission's tile grid),
        # not the room named last: a hazard in the kitchen seen right after the workshop was named went out as
        # 'workshop' (sim mission_check, 26 Nov). /tile_map gives the grid and which tiles are scene tiles.
        self.room_from_tile = bool(p("object_room_from_tile", True).value)
        self.room_margin_m = float(p("object_room_margin_m", 0.35).value)   # also a scene tile this close (range error)
        self.tile_pos_scale = float(p("tile_pos_scale", 1.02).value)        # report pos of tile i,j = [i, j] x this
        self.grid = None                         # (origin_x, origin_y, yaw rad, size) of the mission's tile grid
        self.scene_flags = set()                 # tile labels the mission manager flags as scene tiles
        self.tile_sides = {}                     # tile label -> {"N": "wall" | "open" | ...} from /tile_map
        self.room_of = {}                        # tile label -> scene type from /scene/result ('traversable' too)
        self.room_pos = {}                       # tile label -> its pos from /scene/result
        # an object on a scene tile the robot has not looked at yet waits this long for that tile's room (the look can
        # come minutes later: sim mission_check, hand tools held 90 s then sent with the last room, 'kitchen')
        self.room_wait_s = float(p("object_room_wait_s", 600.0).value)
        self.room_wait_tiles = set()
        self.qr_positions = []                   # where QR codes were seen ([x, y], map frame)
        self.qr_near_r = float(p("qr_near_survivor_m", 1.5).value)
        self.no_qr_min_h = float(p("no_qr_min_bbox_h_frac", 0.25).value)
        # "survivor WITH a QR" (M03) also when the front camera had a QR in view within qr_window_s (either side) of the
        # survivor sighting, read straight from the qr_scanner (/qr/visible) - not only perception's qr_in_view, which
        # missed a QR read 0.7 s before (plusscenevote 26 Nov: sent as M05). A survivor without one waits until
        # qr_window_s after its sighting before it can go out as "no QR" (M05), while a "qr" task is still open.
        self.qr_window_s = float(p("qr_window_s", 3.0).value)
        # how long a survivor with no QR yet waits before it can go out as "no QR" (M05). Longer than qr_window_s: the QR
        # beside a survivor is often only read once the robot has turned to it (fullrun_26_11_141104: survivor at
        # 14:12:47, its QR read 7.4 s later 0.18 m from it - sent as M05). A QR read meanwhile near it makes it M03.
        self.no_qr_wait_s = max(self.qr_window_s, float(p("no_qr_wait_s", 10.0).value))
        self.qr_camera = p("qr_camera", "front").value
        self.qr_seen_times = []                  # unix times a QR was in view (qr_camera), last 15 min
        self.survivor_wait = []                  # (due unix time, (ev, label, confirmed, room)) waiting for a late QR
        self.server_timeout = float(p("server_timeout_s", 5.0).value)

        if not self.group_id:
            self.get_logger().warn("group_id is empty - results will be rejected. Set it in mission_reporter.yaml")

        self.lock = threading.Lock()
        self.results = []
        os.makedirs(self.out_dir, exist_ok=True)

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, "/mission/result", qos)
        self.scene_from = p("scene_from", "topic").value           # "topic" = only the scene topic decides
        scene_topic = p("scene_topic", "/mission/scene_type").value
        # latched like the publisher (mission_manager /scene/result), so a room published before we started is still received
        scene_qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, scene_topic, self.on_scene, scene_qos)
        tile_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, p("tile_map_topic", "/tile_map").value, self.on_tile_map, tile_qos)
        look_topic = p("scene_looking_topic", "/scene/looking").value
        if look_topic:
            look_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(String, look_topic, self.on_looking, look_qos)
        self.create_timer(1.0, self._tick_scene)
        self.get_logger().info(f"scene from {scene_topic} (scene_from={self.scene_from})")
        self.create_subscription(String, "/mission/manual_report", self.on_manual, 10)
        # pos: taken from your position package's topic (same idea as the scene topic)
        self.topic_pos = None
        self.pos_topic = p("position_topic", "").value
        self.pos_type = p("position_msg_type", "string").value
        if self.pos_topic:
            self._subscribe_position(self.pos_topic, self.pos_type)
        # M04: the QR message text comes from another package's topic (std_msgs/String, plain text or JSON)
        self.qr_topic = p("qr_message_topic", "/perception/qr_text").value
        if self.qr_topic:
            self.create_subscription(String, self.qr_topic, self.on_qr_message, 10)
            self.get_logger().info(f"M04 message from {self.qr_topic}")
        qr_vis = p("qr_visible_topic", "/qr/visible").value
        if qr_vis:
            self.create_subscription(String, qr_vis, self.on_qr_visible, 10)
        if self.use_llm:
            self.create_subscription(String, "/perception/llm_result", self.on_llm, 10)
        else:
            self.create_subscription(String, "/perception/event", self.on_event, 10)
        if hz > 0:
            self.create_timer(1.0 / hz, self.republish)
        self.get_logger().info(
            f"mission_reporter up: group={self.group_id!r} tasks={[t['id'] + ':' + t['type'] for t in self.tasks]} "
            f"source={'llm_result' if self.use_llm else 'event'} out={self.out_dir} "
            f"server={self.server_url or 'off (topic + file only)'}")

    # ---- inputs -------------------------------------------------------------------------------
    def on_scene(self, msg):
        """/scene/result: {"tile", "scene_type", "pos", "confidence"} for every scene tile looked at (plain text and
        {"scene_type": ...} work too). "" / traversable = no room: the current room stays."""
        txt = msg.data.strip()
        scene_pos, tile, conf = None, None, None
        if txt.startswith("{"):
            try:
                d = json.loads(txt)
                txt = str(d.get("scene_type") or d.get("scene") or d.get("room") or "").strip()
                scene_pos, tile, conf = d.get("pos"), d.get("tile"), d.get("confidence")
            except json.JSONDecodeError:
                pass
        named = txt if txt and self._norm(txt) != "traversable" else ""
        if tile is not None:
            with self.lock:
                self.room_of[str(tile)] = named or "traversable"
                if scene_pos is not None:
                    self.room_pos[str(tile)] = list(scene_pos)
        if named and named != self.scene:
            self.scene = named
            self.get_logger().info(f"scene_type -> '{self.scene}'" + (f" (tile {tile})" if tile else ""))
        if tile is not None:
            self._release_held(str(tile), named)   # seen while facing this tile: its room (or, if none, the usual way)
        self._flush_pending()          # a real detection waiting in the queue wins over a scene rule
        if named:
            self._apply_scene_tasks(named, scene_pos, conf, tile)

    def on_tile_map(self, msg):
        """/tile_map (mission_manager): the tile grid and which tiles are scene tiles."""
        try:
            d = json.loads(msg.data)
            g = d["grid"]
            grid = (float(g["origin_xy"][0]), float(g["origin_xy"][1]), math.radians(float(g.get("yaw_deg", 0.0))),
                    float(g.get("size_m", 1.0)))
        except (ValueError, KeyError, TypeError, IndexError):
            return
        tiles = {k: v for k, v in (d.get("tiles") or {}).items() if isinstance(v, dict)}
        flags = {k for k, v in tiles.items() if v.get("scene")}
        sides = {k: v["sides"] for k, v in tiles.items() if isinstance(v.get("sides"), dict)}
        with self.lock:
            self.grid = grid
            self.scene_flags = flags
            self.tile_sides = sides

    def _tile_candidates(self, xy):
        """Called with self.lock held: tile labels for a map point, its own tile first, then the neighbours whose
        edge is within room_margin_m - unless /tile_map has a wall on that edge: the camera cannot see through it, so
        the object is on this side (fuck_26_11_212613: the M03 poster on tile 2,0's N wall, 0.10 m from the edge, was
        held for scene tile 2,1 behind the wall, never looked at, and never reported)."""
        ox, oy, yaw, size = self.grid
        dx, dy = xy[0] - ox, xy[1] - oy
        u = math.cos(yaw) * dx + math.sin(yaw) * dy
        v = -math.sin(yaw) * dx + math.cos(yaw) * dy
        i, j = int(math.floor(u / size)), int(math.floor(v / size))
        fu, fv = u / size - i, v / size - j
        m = self.room_margin_m / size
        out = [(i, j)]
        own = self.tile_sides.get(f"{i},{j}") or {}
        for di, dj, near, side, back in ((-1, 0, fu < m, "W", "E"), (1, 0, fu > 1 - m, "E", "W"),
                                         (0, -1, fv < m, "S", "N"), (0, 1, fv > 1 - m, "N", "S")):
            other = self.tile_sides.get(f"{i + di},{j + dj}") or {}
            if near and own.get(side) != "wall" and other.get(back) != "wall":
                out.append((i + di, j + dj))
        return [f"{a},{b}" for a, b in out]

    def _object_room(self, ev):
        """Called with self.lock held: ("room", tile, room) when the object stands on a scene tile already named,
        ("wait", tile, None) when on a scene tile not answered yet, None when on no scene tile (or not located)."""
        loc = ev.get("location") or {}
        xy = loc.get("xy") or ([loc["position_map"]["x"], loc["position_map"]["y"]] if loc.get("position_map") else None)
        if not self.room_from_tile or self.grid is None or not xy:
            return None
        cands = self._tile_candidates(xy)
        for t in cands:
            r = self.room_of.get(t)
            if r and r != "traversable":
                return "room", t, r
        for t in cands:
            if t not in self.room_of and (t in self.scene_flags or (self.looking or {}).get("tile") == t):
                return "wait", t, None
        return None

    def _tile_pos(self, tile):
        if tile in self.room_pos:
            return list(self.room_pos[tile])
        i, j = (int(v) for v in tile.split(","))
        return [round(i * self.tile_pos_scale, 3), round(j * self.tile_pos_scale, 3)]

    def on_looking(self, msg):
        """/scene/looking: {"tile", "pos"} while scene mode faces a scene tile, {} when it stops."""
        try:
            d = json.loads(msg.data or "{}")
        except json.JSONDecodeError:
            d = {}
        with self.lock:
            if d.get("tile"):
                self.looking = {"tile": str(d["tile"]), "pos": d.get("pos")}
                self.get_logger().info(f"facing scene tile {d['tile']}: sightings wait for its room")
            elif self.looking is not None:
                self.looking_ended = (self.looking, time.monotonic())
                self.looking = None

    def _current_look(self):
        """Called with self.lock held: the scene tile the camera faces now (or did, scene_look_grace_s ago)."""
        if self.looking is not None:
            return self.looking
        last, t = self.looking_ended
        if last is not None and time.monotonic() - t <= self.scene_look_grace_s:
            return last
        return None

    def _release_held(self, tile, room):
        """The AI answered for scene tile `tile`: what was seen while facing it is reported now, with `room` when it
        named one (else the usual way: the current room, or the queue until one is known)."""
        with self.lock:
            items = self.held.pop(tile, [])
            self.room_wait_tiles.discard(tile)
        if items:
            self.get_logger().info(f"scene tile {tile} -> '{room or 'no scene'}': {len(items)} held sighting(s) released")
        for _, (ev, label, confirmed, rm) in items:
            if room:
                ev["_scene_room"] = room
            ev["_released_unix"] = time.time()      # the no-QR wait starts again here, not at the (held) sighting
            self.process(ev, label, confirmed, rm)

    def _apply_scene_tasks(self, scene, scene_pos, conf=None, tile=None):
        """scene_tasks rules: the identified room alone answers the task (pos = the scene tile x 1.02). Held for
        scene_task_hold_s: if another tile is named the same scene meanwhile, the higher confidence wins."""
        for room, task_id, label in self.scene_tasks:
            if self._norm(room) != self._norm(scene):
                continue
            with self.lock:
                task = next((t for t in self.tasks if t["id"] == task_id and not t["done"]), None)
                if task is None:
                    continue
                if self.require_pos and scene_pos is None:
                    self.get_logger().warn(f"scene '{scene}': no position - task {task_id} kept open")
                    continue
                c = float(conf) if conf is not None else 0.0
                cand = self.scene_task_cands.get(task_id)
                if cand is None:
                    self.scene_task_cands[task_id] = {"deadline": time.monotonic() + self.scene_task_hold_s,
                                                      "scene": scene, "pos": scene_pos, "conf": c, "label": label,
                                                      "tile": tile}
                    self.get_logger().info(f"scene '{scene}' on tile {tile} (confidence {c:.2f}): {task_id} sent in "
                                           f"{self.scene_task_hold_s:.0f} s unless another tile is named '{scene}'")
                elif c > cand["conf"]:
                    self.get_logger().info(f"scene '{scene}' also on tile {tile} ({c:.2f} > {cand['conf']:.2f} on "
                                           f"tile {cand['tile']}): {task_id} will use tile {tile}")
                    cand.update(scene=scene, pos=scene_pos, conf=c, tile=tile)
                else:
                    self.get_logger().info(f"scene '{scene}' also on tile {tile} ({c:.2f} <= {cand['conf']:.2f}): "
                                           f"{task_id} keeps tile {cand['tile']}")
        if self.scene_task_hold_s <= 0:
            self._tick_scene()

    def _tick_scene(self):
        """1 Hz: send scene_tasks answers whose hold is over; release sightings held for a tile that never got an answer."""
        now = time.monotonic()
        with self.lock:
            for tid, c in [(k, v) for k, v in self.scene_task_cands.items() if now >= v["deadline"]]:
                del self.scene_task_cands[tid]
                task = next((t for t in self.tasks if t["id"] == tid and not t["done"]), None)
                if task is None:
                    continue                    # a real detection answered it meanwhile
                task["done"] = True
                self._emit(build_result(self.group_id, tid, task["type"], c["scene"], c["label"], c["pos"], "", swap_xy=self.swap_xy))
            stale = [tile for tile, items in self.held.items()
                     if items and now - items[0][0] > (self.room_wait_s if tile in self.room_wait_tiles
                                                       else self.scene_held_timeout_s)]
            self.room_wait_tiles -= set(stale)
            due = [w for w in self.survivor_wait if time.time() >= w[0]]
            self.survivor_wait = [w for w in self.survivor_wait if w not in due]
        for _, args in due:
            args[0]["_qr_wait_done"] = True
            self.process(*args)                      # QR window over: decided now (a late QR makes it M03)
        for tile in stale:
            self.get_logger().warn(f"scene tile {tile}: no answer after {self.scene_held_timeout_s:.0f} s - "
                                   f"releasing what was seen while facing it")
            self._release_held(tile, "")

    def _queue(self, item):
        """Called with self.lock held: keep a sighting until the room is known (oldest dropped past pending_max)."""
        self.pending.append(item)
        if len(self.pending) > self.pending_max:
            self.pending.pop(0)
        self.get_logger().warn(f"room unknown - {len(self.pending)} sighting(s) queued until it is known",
                               throttle_duration_sec=5.0)

    def _flush_pending(self):
        """The room is known now: answer the queued sightings, oldest first (each still needs an open matching task)."""
        if not self.scene:
            return
        with self.lock:
            items, self.pending = self.pending, []
        if items:
            self.get_logger().info(f"room known ('{self.scene}'): replaying {len(items)} queued sighting(s)")
        items.sort(key=lambda it: it[0] != "qr")       # QR texts first, then the sightings (stable: oldest first)
        for kind, a in items:
            if kind == "qr":
                self._answer_qr(a)
            else:
                self.process(*a)

    def _subscribe_position(self, topic, kind):
        """kind: string | point_stamped | pose_stamped | pose_cov | odometry | point | pose"""
        def xy_of(m):
            if kind == "string":
                t = m.data.strip()
                try:
                    d = json.loads(t)
                except json.JSONDecodeError:
                    d = [float(v) for v in t.replace(",", " ").split()[:2]]       # "x y" or "x,y"
                if isinstance(d, dict):
                    d = d.get("pos") or [d.get("x"), d.get("y")]
                return [float(d[0]), float(d[1])]
            if kind in ("point_stamped",):
                return [m.point.x, m.point.y]
            if kind == "point":
                return [m.x, m.y]
            if kind == "pose_stamped":
                return [m.pose.position.x, m.pose.position.y]
            if kind == "pose":
                return [m.position.x, m.position.y]
            if kind in ("pose_cov", "odometry"):
                return [m.pose.pose.position.x, m.pose.pose.position.y]
            raise ValueError(kind)
        if kind == "string":
            mtype = String
        elif kind in ("point_stamped", "pose_stamped", "pose_cov", "point", "pose"):
            from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, Point, Pose
            mtype = {"point_stamped": PointStamped, "pose_stamped": PoseStamped,
                     "pose_cov": PoseWithCovarianceStamped, "point": Point, "pose": Pose}[kind]
        elif kind == "odometry":
            from nav_msgs.msg import Odometry
            mtype = Odometry
        else:
            self.get_logger().error(f"position_msg_type '{kind}' not supported")
            return

        def cb(m):
            try:
                xy = [round(v, 2) for v in xy_of(m)]
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"position message not understood: {e}", throttle_duration_sec=5.0)
                return
            if self.topic_pos is None:
                self.get_logger().info(f"first position from {topic}: {xy}")
            self.topic_pos = xy
        self.create_subscription(mtype, topic, cb, 10)
        self.get_logger().info(f"pos from {topic} ({kind})")

    def _pos(self, ev):
        """Position for a report: your position topic if set, otherwise perception's."""
        if ev.get("_pos_at_sighting") is not None:
            return list(ev["_pos_at_sighting"])           # queued sighting: where the robot WAS, not where it is now
        if self.pos_topic:
            return list(self.topic_pos) if self.topic_pos is not None else None
        return event_position(ev)

    def on_qr_message(self, msg):
        """A QR text from the QR package -> answer the qr task (pos null, current scene)."""
        txt = msg.data
        if txt.strip().startswith("{"):
            try:
                d = json.loads(txt)
                txt = str(d.get("message") or d.get("text") or d.get("decoded") or d.get("data") or "")
            except json.JSONDecodeError:
                pass
        if not txt.strip():
            return
        self._note_qr_seen(time.time())
        self._answer_qr(txt)

    def on_qr_visible(self, msg):
        """/qr/visible (qr_scanner, every scan): note when qr_camera had a code in view (read or not)."""
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        codes = (d.get("codes") or []) if isinstance(d, dict) else []
        mine = [c for c in codes if isinstance(c, dict) and c.get("camera", self.qr_camera) == self.qr_camera]
        if mine:
            self._note_qr_seen(float(d.get("t") or time.time()))
        with self.lock:
            for c in mine:                       # where it is (map frame): a survivor seen near it has a QR (M03)
                mp = c.get("map_point")
                if isinstance(mp, (list, tuple)) and len(mp) >= 2 and None not in mp[:2]:
                    self.qr_positions.append([float(mp[0]), float(mp[1])])
            del self.qr_positions[:-500]

    def _note_qr_seen(self, t):
        with self.lock:
            self.qr_seen_times.append(t)
            self.qr_seen_times = [x for x in self.qr_seen_times if x > t - 900.0]

    @staticmethod
    def _seen_unix(ev):
        """When the sighting happened (unix): perception's stamp, else when it reached us."""
        if ev.get("_seen_unix") is None:
            st = ev.get("stamp") or {}
            ev["_seen_unix"] = (st["sec"] + st.get("nanosec", 0) * 1e-9) if isinstance(st, dict) and st.get("sec") \
                else time.time()
        return ev["_seen_unix"]

    def _answer_qr(self, txt):
        with self.lock:
            task = next((t for t in self.tasks if t["type"] == "qr" and not (self.once and t["done"])), None)
            if task is None:
                self.get_logger().info(f"QR '{txt}': qr task already answered, not reported",
                                       throttle_duration_sec=5.0)
                return
            scene_out = task["scene"] or self.scene
            if self.require_scene and not self._norm(scene_out):
                self._queue(("qr", txt))
                return
            task["done"] = True
            res = build_result(self.group_id, task["id"], "qr", scene_out, "", None, txt)
            self._emit(res)

    def on_event(self, msg):
        try:
            ev = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if ev.get("type") == "qr" and self.qr_topic and self.qr_topic != "/perception/qr_text":
            return                                     # M04 is answered from the QR package's topic instead
        if ev.get("type") == "hazard" and ev.get("hazard_certain") is False:
            # unsure which sign it is: only the LLM (use_llm_result) may decide those
            self.get_logger().info(f"hazard {ev.get('hazard_top3')}: not sure which sign, not reported",
                                   throttle_duration_sec=5.0)
            return
        self.process(ev, ev.get("label", ""), True, None)

    def on_llm(self, msg):
        try:
            res = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        ev = res.get("event") or {}
        label = res.get("hazard") or res.get("label") or ev.get("label", "")
        confirmed = True
        if ev.get("label") == "survivor" and self.need_confirmed and res.get("llm") is not None:
            confirmed = bool(res.get("survivor_confirmed"))
        if res.get("llm") is not None and not res.get("accepted", True):
            confirmed = False                     # LLM disagreed with the detector
        room = (res.get("llm") or {}).get("room_type")
        self.process(ev, label, confirmed, room)

    def on_manual(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error("manual_report: invalid JSON")
            return
        with self.lock:
            task = next((t for t in self.tasks if t["id"] == d.get("task_id")), None)
            ttype = d.get("task_type") or (task or {}).get("type", "object")
            res = build_result(self.group_id, d.get("task_id", ""), ttype, d.get("scene_type", self.scene),
                               d.get("label", ""), d.get("pos"), d.get("message", ""), swap_xy=self.swap_xy)
            if task:
                task["done"] = True
            self._emit(res)

    # ---- core ---------------------------------------------------------------------------------
    def process(self, ev, label, confirmed, room=None):
        if not confirmed or float(ev.get("confidence", 1.0)) < self.min_conf:
            return
        self._seen_unix(ev)
        ttype = event_task_type(ev, label)
        if ttype != "qr" and "_scene_tile" not in ev:
            with self.lock:
                where = self._object_room(ev)
                if where is not None:
                    kind, tile, room_t = where
                    ev = dict(ev)
                    ev["_scene_tile"] = tile
                    ev["_pos_at_sighting"] = self._tile_pos(tile)
                    if kind == "wait":
                        self.held.setdefault(tile, []).append((time.monotonic(), (ev, label, confirmed, room)))
                        self.room_wait_tiles.add(tile)
                        self.get_logger().info(f"event {ttype}/{label}: on scene tile {tile} - held for its room")
                        return
                    ev["_scene_room"] = room_t
                    self.get_logger().info(f"event {ttype}/{label}: on scene tile {tile} = '{room_t}'")
        if ttype != "qr" and "_scene_tile" not in ev:
            with self.lock:
                look = self._current_look()
                if look is not None:
                    # facing a scene tile: it is on THAT tile. Wait for the AI's answer for it (its room), pos = the tile
                    ev_h = dict(ev)
                    ev_h["_scene_tile"] = look["tile"]
                    ev_h["_pos_at_sighting"] = list(look["pos"]) if look.get("pos") is not None else self._pos(ev)
                    self.held.setdefault(look["tile"], []).append((time.monotonic(), (ev_h, label, confirmed, room)))
                    self.get_logger().info(f"event {ttype}/{label}: seen while facing scene tile {look['tile']} - "
                                           f"held for that tile's room")
                    return
        if ttype != "qr" and self.require_pos and self._pos(ev) is None:
            # don't use up a task on a sighting with no position; wait for one that has it
            self.get_logger().warn(
                f"event {ttype}/{label}: no position (no depth and no map->base TF?) - task kept open",
                throttle_duration_sec=5.0)
            return
        with self.lock:
            if ev.get("_scene_room"):
                scene_now = ev["_scene_room"]                  # seen while facing that scene tile: its room
            elif self.scene_from == "topic":
                scene_now = self.scene                         # only the scene package decides the room
            else:
                scene_now = room or ev.get("scene_type") or self.scene
            if ttype == "qr" and event_position(ev) is not None:
                self.qr_positions.append(event_position(ev))
            if ttype == "survivor":
                task = self.pick_survivor_task(self._pos(ev), scene_now, ev)
                if task == "wait_qr":
                    ev_w = dict(ev)
                    ev_w["_pos_at_sighting"] = self._pos(ev)       # where the robot was, not where it is in 3 s
                    self.survivor_wait.append((self._no_qr_from(ev_w) + self.no_qr_wait_s,
                                               (ev_w, label, confirmed, room)))
                    self.get_logger().info(f"survivor at {ev_w['_pos_at_sighting']}: no QR yet - waiting "
                                           f"{self.no_qr_wait_s:.0f} s for one before calling it 'no QR'")
                    return
                if task is False:
                    return                               # same survivor again, or waiting for the scene
            else:
                task = next((t for t in self.tasks if t["type"] == ttype and not (self.once and t["done"])
                             and (not t["label"] or t["label"] == label)), None)
            if task is None:
                self.get_logger().info(f"event {ttype}/{label}: no open matching task, not reported")
                return
            if self.require_scene and not self._norm(task["scene"] or scene_now):
                ev_q = dict(ev)
                ev_q["_pos_at_sighting"] = self._pos(ev)       # freeze the position now; the replay may be much later
                self._queue(("event", (ev_q, label, confirmed, room)))
                return
            task["done"] = True
            if ttype == "qr":
                pos, out_label, message = None, "", ev.get("qr_text", "")
            elif ttype == "survivor":
                pos, out_label, message = self._pos(ev), "", ""
            else:
                pos, out_label, message = self._pos(ev), label, ""
            if ttype == "survivor":
                self.survivors.append((task["id"], pos))
                self.survivor_xy.append((task["id"], event_position(ev)))
                scene_out = task["scene"] or scene_now           # M03: the mission's exact room name; M05: the room it was in
            else:
                scene_out = task["scene"] or scene_now
            res = build_result(self.group_id, task["id"], ttype, scene_out, out_label, pos, message, swap_xy=self.swap_xy)
            self._emit(res)

    @staticmethod
    def _norm(s):
        return (s or "").strip().lower().replace(" ", "_").replace("-", "_")

    def _has_qr(self, ev, pos):
        """True if a QR code was seen beside this survivor (same moment, or a QR seen near this spot)."""
        if ev.get("qr_in_view"):
            return True
        t = self._seen_unix(ev)
        if any(abs(q - t) <= self.qr_window_s for q in self.qr_seen_times):
            return True                              # the scanner had a QR in view around the sighting
        here = event_position(ev)                    # map frame, like qr_positions (pos may be the course frame)
        if here is not None:
            for q in self.qr_positions:
                if ((here[0] - q[0]) ** 2 + (here[1] - q[1]) ** 2) ** 0.5 < self.qr_near_r:
                    return True                      # a QR seen beside where the survivor is (before or since)
        return False

    def _no_qr_from(self, ev):
        """When the no-QR wait starts: the sighting, or when it was released if it was held for its room (a held
        sighting must not use up its wait while held - survey run_20261126_210217: held 12 s, went out as M05)."""
        return max(self._seen_unix(ev), float(ev.get("_released_unix") or 0.0))

    def _is_qr_room(self, scene):
        """True if scene_tasks says the survivor WITH a QR is in this room: a survivor there is never M05."""
        if not scene:
            return False
        qr_ids = {t["id"] for t in self.tasks if t["type"] == "survivor" and t["cond"] == "qr"}
        return any(self._norm(room) == scene and tid in qr_ids for room, tid, _ in self.scene_tasks)

    def pick_survivor_task(self, pos, scene_now, ev=None):
        """Survivor -> task.  task_conditions: "qr" = survivor WITH a QR beside it (M03),
        "no_qr" = survivor WITHOUT one (M05), "" = any.  task_scenes (optional) = room it must be in.
        Returns task, None (no open task), False (skip for now) or "wait_qr" (no QR yet: decide after qr_window_s)."""
        ev = ev or {}
        here = event_position(ev) if (ev.get("location") or {}) else None
        if here is not None:                         # map positions: two survivors can share a reported tile pos
            for tid, p in self.survivor_xy:
                if p is not None and ((here[0] - p[0]) ** 2 + (here[1] - p[1]) ** 2) ** 0.5 < self.same_survivor_r:
                    self.get_logger().info(f"survivor at {here}: same one already reported as {tid}, skipped",
                                           throttle_duration_sec=5.0)
                    return False
        elif pos is not None:
            for tid, p in self.survivors:
                if p is not None and ((pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2) ** 0.5 < self.same_survivor_r:
                    self.get_logger().info(f"survivor at {pos}: same one already reported as {tid}, skipped",
                                           throttle_duration_sec=5.0)
                    return False
        scene = self._norm(scene_now)
        has_qr = self._has_qr(ev, pos) or self._is_qr_room(scene)
        open_ = [t for t in self.tasks if t["type"] == "survivor" and not (self.once and t["done"])]

        def fits(t):
            if t["cond"] == "qr" and not has_qr:
                return False
            if t["cond"] == "no_qr" and has_qr:
                return False
            if t["scene"] and not (scene and self._norm(t["scene"]) == scene):
                return False
            return True

        cands = [t for t in open_ if fits(t)]
        if not cands:
            return None
        task = cands[0]
        if task["cond"] == "no_qr" and not ev.get("_qr_wait_done") \
                and time.time() < self._no_qr_from(ev) + self.no_qr_wait_s \
                and any(t["cond"] == "qr" for t in open_):
            return "wait_qr"                         # a QR may still come into view: decide after the window
        if task["cond"] == "no_qr" and any(t["cond"] == "qr" for t in open_):
            # only call it "no QR" if we were close enough that a QR beside it would have been seen. Once the "qr"
            # task is answered there is no doubt left: any other survivor is the one without a QR (sim mission_check:
            # the market survivor was only ever seen from 0.9 m, never reported)
            h = float(ev.get("bbox_h_frac", 1.0))
            if h < self.no_qr_min_h:
                self.get_logger().info(
                    f"survivor at {pos}: no QR seen but too far to be sure (size {h:.2f} < {self.no_qr_min_h}) "
                    f"- not reported as {task['id']} yet", throttle_duration_sec=5.0)
                return False
        if task["scene"] and not scene and self.hold_unknown:
            return False
        return task

    def _emit(self, res):
        self.results.append(res)
        self.pub.publish(String(data=json.dumps(res)))
        self.get_logger().info(f"RESULT {json.dumps(res)}")
        try:
            with open(os.path.join(self.out_dir, "results.jsonl"), "a") as f:
                f.write(json.dumps(res) + "\n")
            with open(os.path.join(self.out_dir, "results.json"), "w") as f:
                json.dump(self.results, f, indent=2)
        except OSError as e:
            self.get_logger().error(f"could not write results: {e}")
        if self.server_url:
            threading.Thread(target=self._post, args=(dict(res),), daemon=True).start()

    def _post(self, res):
        """Send one result to the organisers' server; never blocks the node."""
        req = urllib.request.Request(self.server_url, data=json.dumps(res).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.server_timeout) as r:
                reply = {"status": r.status, "body": r.read().decode(errors="replace")[:500]}
            self.get_logger().info(f"server {res['task_id']}: {reply}")
        except Exception as e:  # noqa: BLE001
            reply = {"error": repr(e)}
            self.get_logger().error(f"server {res['task_id']}: send FAILED {e} - result is still in results.json")
        try:
            with open(os.path.join(self.out_dir, "results.jsonl"), "a") as f:
                f.write(json.dumps({"server_reply_for": res["task_id"], **reply}) + "\n")
        except OSError:
            pass

    def republish(self):
        with self.lock:
            for r in self.results:
                self.pub.publish(String(data=json.dumps(r)))


def main():
    rclpy.init()
    node = MissionReporter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
