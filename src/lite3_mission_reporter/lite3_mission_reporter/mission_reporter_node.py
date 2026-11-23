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
"""
import json
import os
import threading

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
        n = len(ids)
        labels += [""] * (n - len(labels))
        scenes += [""] * (n - len(scenes))
        self.tasks = []
        for i in range(n):
            if types[i] not in TASK_TYPES:
                self.get_logger().error(f"task {ids[i]}: unknown task_type '{types[i]}' (use {TASK_TYPES})")
            self.tasks.append({"id": ids[i], "type": types[i], "label": labels[i], "scene": scenes[i], "done": False})
        self.scene = p("default_scene_type", "").value
        self.use_llm = p("use_llm_result", False).value
        self.need_confirmed = p("require_survivor_confirmed", True).value
        self.min_conf = float(p("min_confidence", 0.0).value)
        self.once = p("report_each_task_once", True).value
        self.out_dir = os.path.expanduser(p("output_dir", "~/lite3_mission_results").value)
        hz = float(p("republish_hz", 0.0).value)

        if not self.group_id:
            self.get_logger().warn("group_id is empty - results will be rejected. Set it in mission_reporter.yaml")

        self.lock = threading.Lock()
        self.results = []
        os.makedirs(self.out_dir, exist_ok=True)

        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, "/mission/result", qos)
        self.create_subscription(String, "/mission/scene_type", self.on_scene, 10)
        self.create_subscription(String, "/mission/manual_report", self.on_manual, 10)
        if self.use_llm:
            self.create_subscription(String, "/perception/llm_result", self.on_llm, 10)
        else:
            self.create_subscription(String, "/perception/event", self.on_event, 10)
        if hz > 0:
            self.create_timer(1.0 / hz, self.republish)
        self.get_logger().info(
            f"mission_reporter up: group={self.group_id!r} tasks={[t['id'] + ':' + t['type'] for t in self.tasks]} "
            f"source={'llm_result' if self.use_llm else 'event'} out={self.out_dir}")

    # ---- inputs -------------------------------------------------------------------------------
    def on_scene(self, msg):
        self.scene = msg.data.strip()
        self.get_logger().info(f"scene_type -> '{self.scene}'")

    def on_event(self, msg):
        try:
            ev = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.process(ev, ev.get("label", ""), True)

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
        self.process(ev, label, confirmed)

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
    def process(self, ev, label, confirmed):
        if not confirmed or float(ev.get("confidence", 1.0)) < self.min_conf:
            return
        ttype = event_task_type(ev, label)
        with self.lock:
            task = next((t for t in self.tasks if t["type"] == ttype and not (self.once and t["done"])
                         and (not t["label"] or t["label"] == label)), None)
            if task is None:
                self.get_logger().info(f"event {ttype}/{label}: no open matching task, not reported")
                return
            task["done"] = True
            if ttype == "qr":
                pos, out_label, message = None, "", ev.get("qr_text", "")
            elif ttype == "survivor":
                pos, out_label, message = event_position(ev), "", ""
            else:
                pos, out_label, message = event_position(ev), label, ""
            res = build_result(self.group_id, task["id"], ttype, task["scene"] or self.scene,
                               out_label, pos, message)
            self._emit(res)

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
