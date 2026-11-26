"""VLM broker: the ONE node that talks to the scene-understanding gateway.

The gateway allows one active request per group, so every node (keyboard console,
surface_monitor, a future mission manager) sends its request here instead and the broker
queues them, attaches the latest frame from the requested camera, calls the gateway and
publishes the answer.

Subscribes:
  /vlm/request            std_msgs/String JSON {"id", "source", "camera": "front"|"down",
                          "prompt": "<name in prompts/>" or "prompt_text": "...",
                          "priority": int, "max_age_s": float, "snapshot_only": bool}
  <front_image_topic>, <down_image_topic>   camera frames (raw or compressed)
Publishes:
  /vlm/result             std_msgs/String JSON - one answer for EVERY request (also dropped ones)
  /vlm/status             std_msgs/String JSON - queue, in-flight request, counters (1 Hz)
Saves every query (image + prompt + answer) under log_dir/<date>/ for review and for training data.
"""

import glob
import json
import os
import tempfile
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from . import gateway
from .broker_queue import Request, RequestQueue


def _share_prompts():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("lite3_teleop_vlm"), "prompts")
    except Exception:
        return os.path.join(os.path.dirname(__file__), "..", "prompts")


class VlmBroker(Node):
    def __init__(self):
        super().__init__("vlm_broker")
        p = self.declare_parameter
        p("server", "http://192.168.10.141:8000")
        p("group", "groupXX")
        p("client_python", "python3")
        p("client_script", os.path.expanduser("~/scene_understanding/client.py"))
        p("prompts_dir", _share_prompts())
        p("front_image_topic", "/camera/color/image_raw")
        p("front_compressed", False)
        p("down_image_topic", "/down_camera/image_raw")
        p("down_compressed", False)
        p("min_interval_s", 2.0)         # spacing between gateway calls
        p("timeout_s", 20.0)
        p("frame_max_age_s", 2.0)        # refuse to send a frame older than this
        p("max_width", 1024)
        p("max_height", 768)
        p("log_dir", os.path.expanduser("~/vlm_tests"))
        g = lambda n: self.get_parameter(n).value
        self.cfg = {n: g(n) for n in ("server", "group", "client_python", "client_script", "prompts_dir",
                                      "min_interval_s", "timeout_s", "frame_max_age_s", "max_width",
                                      "max_height", "log_dir")}
        for problem in gateway.config_problems(self.cfg["group"], self.cfg["client_script"]):
            self.get_logger().error(f"VLM config: {problem}")
        self.prompts = self._load_prompts()

        self.lock = threading.Condition()
        self.queue = RequestQueue()
        self.frames = {}                 # camera -> (bgr, stamp_s, received_monotonic)
        self.current = None
        self.stats = {"ok": 0, "failed": 0, "dropped": 0, "last_error": "", "last_latency_s": None}
        self.last_call = 0.0
        self.bridge = None

        self.pub_result = self.create_publisher(String, "/vlm/result", 20)
        self.pub_status = self.create_publisher(String, "/vlm/status", 5)
        self.create_subscription(String, "/vlm/request", self._on_request, 20)
        self._camera_sub("front", g("front_image_topic"), g("front_compressed"))
        self._camera_sub("down", g("down_image_topic"), g("down_compressed"))
        self.create_timer(1.0, self._publish_status)

        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        self.get_logger().info(f"vlm_broker up: group={self.cfg['group']} prompts={sorted(self.prompts)} "
                               f"front={g('front_image_topic')} down={g('down_image_topic')}")

    # ---------- setup ----------
    def _load_prompts(self):
        prompts = {}
        for path in glob.glob(os.path.join(self.cfg["prompts_dir"], "*.txt")):
            with open(path) as f:
                prompts[os.path.splitext(os.path.basename(path))[0]] = f.read()
        if not prompts:
            self.get_logger().warn(f"no prompts found in {self.cfg['prompts_dir']}")
        return prompts

    def _camera_sub(self, name, topic, compressed):
        if not topic:
            return
        if compressed:
            cb = lambda m: self._store(name, cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR), m)
            self.create_subscription(CompressedImage, topic, cb, qos_profile_sensor_data)
        else:
            def cb(m):
                if self.bridge is None:
                    from cv_bridge import CvBridge
                    self.bridge = CvBridge()
                try:
                    self._store(name, self.bridge.imgmsg_to_cv2(m, desired_encoding="bgr8"), m)
                except Exception as e:
                    self.get_logger().warn(f"{name} image conversion failed: {e}", throttle_duration_sec=5.0)
            self.create_subscription(Image, topic, cb, qos_profile_sensor_data)

    def _store(self, name, bgr, msg):
        if bgr is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            self.frames[name] = (bgr, stamp, time.monotonic())

    # ---------- requests ----------
    def _on_request(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"bad /vlm/request (not JSON): {msg.data[:100]}")
            return
        req = Request.from_dict(d, time.monotonic())
        with self.lock:
            replaced = self.queue.push(req)
            self.lock.notify()
        if replaced:
            self._answer(replaced, {"ok": False, "error": "replaced by a newer request from the same source"})

    def _answer(self, req, out, image_stamp=None, saved=None):
        if not out.get("ok"):
            key = "dropped" if out.get("error", "").startswith(("replaced", "expired")) else "failed"
            self.stats[key] += 1
            if key == "failed":
                self.stats["last_error"] = out.get("error", "")
        else:
            self.stats["ok"] += 1
        res = {"id": req.id, "source": req.source, "camera": req.camera, "prompt": req.prompt or "custom",
               "snapshot_only": req.snapshot_only, "image_stamp": image_stamp,
               "queue_wait_s": round(time.monotonic() - req.t_req, 2), "saved": saved, **out}
        self.pub_result.publish(String(data=json.dumps(res)))

    # ---------- worker ----------
    def _work(self):
        while rclpy.ok():
            with self.lock:
                wait = self.cfg["min_interval_s"] - (time.monotonic() - self.last_call)
                if not len(self.queue) or wait > 0:
                    self.lock.wait(timeout=max(0.05, wait if len(self.queue) else 0.5))
                    continue
                req, expired = self.queue.pop(time.monotonic())
                frame = self.frames.get(req.camera) if req else None
                self.current = req
            for r in expired:
                self._answer(r, {"ok": False, "error": "expired while queued"})
            if req is None:
                continue
            try:
                self._handle(req, frame)
            except Exception as e:  # never let the worker die
                self._answer(req, {"ok": False, "error": f"broker error: {e}"})
            finally:
                with self.lock:
                    self.current = None

    def _handle(self, req, frame):
        day_dir = os.path.join(self.cfg["log_dir"], time.strftime("%Y-%m-%d"))
        os.makedirs(day_dir, exist_ok=True)
        stem = os.path.join(day_dir, f"{time.strftime('%H%M%S')}_{req.source}_{req.prompt or 'custom'}_{req.camera}")

        if req.snapshot_only:
            saved = []
            with self.lock:
                frames = dict(self.frames)
            for cam, (bgr, stamp, _) in frames.items():
                path = f"{stem.rsplit('_', 1)[0]}_{cam}.jpg"
                cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved.append(path)
            self._answer(req, {"ok": bool(saved), "error": "" if saved else "no camera frames yet"}, saved=saved)
            return

        if frame is None:
            self._answer(req, {"ok": False, "error": f"no frames received from the {req.camera} camera"})
            return
        bgr, stamp, rx = frame
        age = time.monotonic() - rx
        if age > self.cfg["frame_max_age_s"]:
            self._answer(req, {"ok": False, "error": f"{req.camera} camera frame is {age:.1f} s old (camera stalled?)"})
            return
        prompt = req.prompt_text or self.prompts.get(req.prompt)
        if not prompt:
            self._answer(req, {"ok": False, "error": f"unknown prompt '{req.prompt}'. Have: {sorted(self.prompts)}"})
            return

        h, w = bgr.shape[:2]
        s = min(self.cfg["max_width"] / w, self.cfg["max_height"] / h, 1.0)
        img = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        cv2.imwrite(stem + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            prompt_path = f.name
        try:
            self.last_call = time.monotonic()
            out = gateway.query(self.cfg["client_python"], self.cfg["client_script"], self.cfg["server"],
                                self.cfg["group"], stem + ".jpg", prompt_path, float(self.cfg["timeout_s"]))
        finally:
            os.remove(prompt_path)
        self.stats["last_latency_s"] = out.get("latency_s")
        with open(stem + ".json", "w") as f:
            json.dump({"request": req.__dict__, "prompt": prompt, "response": out}, f, indent=2, default=str)
        self._answer(req, out, image_stamp=stamp, saved=[stem + ".jpg", stem + ".json"])

    def _publish_status(self):
        with self.lock:
            cur = self.current
            status = {"queue": self.queue.summary(), "in_flight": None if cur is None else
                      {"source": cur.source, "prompt": cur.prompt or "custom", "camera": cur.camera},
                      "cameras": {k: round(time.monotonic() - v[2], 1) for k, v in self.frames.items()},
                      **self.stats}
        self.pub_status.publish(String(data=json.dumps(status)))


def main(args=None):
    rclpy.init(args=args)
    node = VlmBroker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
