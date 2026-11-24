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
import os
import threading
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String

TASK_TYPES = ("object", "hazard", "survivor", "qr")


def build_result(group_id, task_id, task_type, scene_type, label, pos, message):
    """The exact organiser format (key order preserved)."""
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
        self.same_survivor_r = float(p("same_survivor_radius_m", 1.0).value)
        self.hold_unknown = p("hold_unknown_scene_survivors", True).value
        self.survivors = []                      # (task_id, [x, y]) already reported
        self.qr_positions = []                   # where QR codes were seen ([x, y])
        self.qr_near_r = float(p("qr_near_survivor_m", 1.5).value)
        self.no_qr_min_h = float(p("no_qr_min_bbox_h_frac", 0.25).value)
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
        self.create_subscription(String, scene_topic, self.on_scene, 10)
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
        txt = msg.data.strip()
        if txt.startswith("{"):                                # also accept JSON, e.g. {"scene_type": "kitchen"}
            try:
                d = json.loads(txt)
                txt = str(d.get("scene_type") or d.get("scene") or d.get("room") or "").strip()
            except json.JSONDecodeError:
                pass
        if txt and txt != self.scene:
            self.scene = txt
            self.get_logger().info(f"scene_type -> '{self.scene}'")

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
        with self.lock:
            task = next((t for t in self.tasks if t["type"] == "qr" and not (self.once and t["done"])), None)
            if task is None:
                self.get_logger().info(f"QR '{txt}': qr task already answered, not reported",
                                       throttle_duration_sec=5.0)
                return
            task["done"] = True
            res = build_result(self.group_id, task["id"], "qr", task["scene"] or self.scene, "", None, txt)
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
                               d.get("label", ""), d.get("pos"), d.get("message", ""))
            if task:
                task["done"] = True
            self._emit(res)

    # ---- core ---------------------------------------------------------------------------------
    def process(self, ev, label, confirmed, room=None):
        if not confirmed or float(ev.get("confidence", 1.0)) < self.min_conf:
            return
        ttype = event_task_type(ev, label)
        if ttype != "qr" and self.require_pos and self._pos(ev) is None:
            # don't use up a task on a sighting with no position; wait for one that has it
            self.get_logger().warn(
                f"event {ttype}/{label}: no position (no depth and no map->base TF?) - task kept open",
                throttle_duration_sec=5.0)
            return
        with self.lock:
            if self.scene_from == "topic":
                scene_now = self.scene                         # only the scene package decides the room
            else:
                scene_now = room or ev.get("scene_type") or self.scene
            if ttype == "qr" and event_position(ev) is not None:
                self.qr_positions.append(event_position(ev))
            if ttype == "survivor":
                task = self.pick_survivor_task(self._pos(ev), scene_now, ev)
                if task is False:
                    return                               # same survivor again, or waiting for the scene
            else:
                task = next((t for t in self.tasks if t["type"] == ttype and not (self.once and t["done"])
                             and (not t["label"] or t["label"] == label)), None)
            if task is None:
                self.get_logger().info(f"event {ttype}/{label}: no open matching task, not reported")
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
                scene_out = task["scene"] or scene_now           # M03: the mission's exact room name; M05: the room it was in
            else:
                scene_out = task["scene"] or scene_now
            res = build_result(self.group_id, task["id"], ttype, scene_out, out_label, pos, message)
            self._emit(res)

    @staticmethod
    def _norm(s):
        return (s or "").strip().lower().replace(" ", "_").replace("-", "_")

    def _has_qr(self, ev, pos):
        """True if a QR code was seen beside this survivor (same moment, or a QR seen near this spot)."""
        if ev.get("qr_in_view"):
            return True
        if pos is not None:
            for q in self.qr_positions:
                if ((pos[0] - q[0]) ** 2 + (pos[1] - q[1]) ** 2) ** 0.5 < self.qr_near_r:
                    return True
        return False

    def pick_survivor_task(self, pos, scene_now, ev=None):
        """Survivor -> task.  task_conditions: "qr" = survivor WITH a QR beside it (M03),
        "no_qr" = survivor WITHOUT one (M05), "" = any.  task_scenes (optional) = room it must be in.
        Returns task, None (no open task) or False (skip for now)."""
        ev = ev or {}
        if pos is not None:
            for tid, p in self.survivors:
                if p is not None and ((pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2) ** 0.5 < self.same_survivor_r:
                    self.get_logger().info(f"survivor at {pos}: same one already reported as {tid}, skipped",
                                           throttle_duration_sec=5.0)
                    return False
        has_qr = self._has_qr(ev, pos)
        scene = self._norm(scene_now)
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
        if task["cond"] == "no_qr":
            # only call it "no QR" if we were close enough that a QR beside it would have been seen
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
