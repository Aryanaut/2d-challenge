"""ROS 2 node: reports which surface the robot is standing on, from the downward camera.

Subscribes:
  <image_topic>                sensor_msgs/Image (or CompressedImage if compressed:=true)
Publishes:
  /surface/state               std_msgs/String  "wood" | "metal" | "other" | "unknown"   (5 Hz)
  /surface/on_stairs           std_msgs/Bool    true while state == "metal"
  /surface/course_complete     std_msgs/Bool    latched (transient local): true once the robot has left the course
  /surface/vlm_result          std_msgs/String  JSON of every VLM answer (for logging / debugging)
  /surface/debug               std_msgs/String  JSON of local features + fusion state (1 Hz)
Services:
  ~/reset                      std_srvs/Trigger clear course_complete and history (between practice runs)
"""

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .fusion import FusionConfig, SurfaceFusion
from .local_classifier import Thresholds, classify_image
from .vlm_client import GatewayClient, VlmResult, parse_surface_result


def _default_prompt_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("surface_monitor"), "prompts", "surface_prompt.txt")
    except Exception:
        return ""


class SurfaceMonitor(Node):
    def __init__(self):
        super().__init__("surface_monitor")

        # ---- parameters ----
        p = self.declare_parameter
        p("image_topic", "/down_camera/image_raw")
        p("compressed", False)
        p("local_rate_hz", 10.0)
        p("publish_rate_hz", 5.0)
        p("mode", "hybrid")                       # hybrid | vlm_only | local_only
        p("server", "http://192.168.10.141:8000")
        p("group", "groupXX")                     # MUST be your group01..group16
        p("client_python", "python3")             # python that has client.py's deps (e.g. requests)
        p("client_script", os.path.expanduser("~/scene_understanding/client.py"))
        p("prompt_file", _default_prompt_path())
        p("vlm_timeout_s", 20.0)
        p("vlm_backend", "direct")                # direct = call client.py here | broker = via vlm_broker
        p("classifier", "threshold")              # threshold | model (trained with surface_train)
        p("model_path", os.path.expanduser("~/surface_model.joblib"))
        p("model_min_conf", 0.6)                  # below this the frame votes "unclear"
        for name, val in FusionConfig().__dict__.items():
            if name != "mode":
                p(name, val)
        for name, val in Thresholds().as_dict().items():
            p("th_" + name, val)

        g = lambda n: self.get_parameter(n).value
        fcfg = FusionConfig(mode=g("mode"), **{k: float(g(k)) for k in FusionConfig().__dict__ if k != "mode"})
        self.th = Thresholds(**{k: float(g("th_" + k)) for k in Thresholds().as_dict()})
        self.fusion = SurfaceFusion(fcfg)
        self.local_period = 1.0 / max(0.5, float(g("local_rate_hz")))

        self.model = None
        if g("classifier") == "model":
            from .model_classifier import ModelClassifier
            try:
                self.model = ModelClassifier(g("model_path"), float(g("model_min_conf")))
                info = self.model.info
                self.get_logger().info(
                    f"trained model loaded: {info.get('model_type')} cv_macro_f1={info.get('cv_macro_f1')} "
                    f"trained {info.get('trained_at')} on {info.get('images_per_label')}")
            except Exception as e:
                self.get_logger().error(f"could not load model {g('model_path')}: {e}; "
                                        "falling back to threshold classifier")

        self.backend = g("vlm_backend")
        self.vlm_timeout_s = float(g("vlm_timeout_s"))
        self.client = GatewayClient(g("client_python"), g("client_script"), g("server"), g("group"),
                                    g("prompt_file"), timeout_s=self.vlm_timeout_s)
        self.prompt_text = ""
        if self.backend == "broker":
            try:
                with open(g("prompt_file")) as f:
                    self.prompt_text = f.read()
            except OSError as e:
                self.get_logger().error(f"cannot read prompt_file {g('prompt_file')}: {e}")
        elif fcfg.mode != "local_only":
            for problem in self.client.config_problems():
                self.get_logger().error(f"VLM config: {problem}")

        # ---- state shared with the VLM worker thread ----
        self.lock = threading.Lock()
        self.latest_frame = None          # (bgr, t)
        self.vlm_done = []                # [(t_rx, t_img, VlmResult)]
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.last_local_t = None
        self.last_features = {}
        self.last_local_label = "unclear"
        self.published_complete = None
        self.last_debug_t = None
        self.bridge = None
        self.broker_pending = {}          # request id -> (t_sent_node_clock, t_sent_monotonic)

        # ---- ROS interfaces ----
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_state = self.create_publisher(String, "/surface/state", 10)
        self.pub_stairs = self.create_publisher(Bool, "/surface/on_stairs", 10)
        self.pub_complete = self.create_publisher(Bool, "/surface/course_complete", latched)
        self.pub_vlm = self.create_publisher(String, "/surface/vlm_result", 10)
        self.pub_debug = self.create_publisher(String, "/surface/debug", 10)

        if g("compressed"):
            self.create_subscription(CompressedImage, g("image_topic"), self._on_compressed,
                                     qos_profile_sensor_data)
        else:
            self.create_subscription(Image, g("image_topic"), self._on_image, qos_profile_sensor_data)
        self.create_service(Trigger, "~/reset", self._on_reset)
        if self.backend == "broker":
            self.pub_req = self.create_publisher(String, "/vlm/request", 10)
            self.create_subscription(String, "/vlm/result", self._on_broker_result, 20)
        self.create_timer(1.0 / max(0.5, float(g("publish_rate_hz"))), self._tick)

        self.get_logger().info(
            f"surface_monitor up: mode={fcfg.mode} topic={g('image_topic')} group={g('group')}")

    # ---------- time ----------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------- image input ----------
    def _on_compressed(self, msg: CompressedImage):
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            self._handle_frame(bgr)

    def _on_image(self, msg: Image):
        if self.bridge is None:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # unsupported encoding etc.
            self.get_logger().warn(f"image conversion failed: {e}", throttle_duration_sec=5.0)
            return
        self._handle_frame(bgr)

    def _handle_frame(self, bgr):
        t = self._now()
        with self.lock:
            self.latest_frame = (bgr, t)
        if self.last_local_t is not None and t - self.last_local_t < self.local_period:
            return
        self.last_local_t = t
        if self.model is not None:
            label, conf, probs = self.model.predict(bgr)
            feats = {"model_conf": conf, **{"p_" + k: v for k, v in probs.items()}}
        else:
            label, feats = classify_image(bgr, self.th)
        with self.lock:
            self.fusion.add_local(t, label)
            self.last_features = feats
            self.last_local_label = label

    # ---------- VLM worker ----------
    def _vlm_job(self, bgr, t_img):
        res = self.client.query(bgr)
        with self.lock:
            self.vlm_done.append((self._now(), t_img, res))

    def _ask_broker(self, t):
        rid = uuid.uuid4().hex[:8]
        self.broker_pending[rid] = (t, time.monotonic())
        self.pub_req.publish(String(data=json.dumps({
            "id": rid, "source": "surface_monitor", "camera": "down", "prompt": "surface",
            "prompt_text": self.prompt_text, "priority": 5, "max_age_s": 10.0})))

    def _on_broker_result(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if d.get("source") != "surface_monitor":
            return
        with self.lock:
            sent = self.broker_pending.pop(d.get("id"), None)
            if sent is None:
                return
            if d.get("ok"):
                surface, conf, evidence = parse_surface_result(d.get("result", ""))
                res = VlmResult(ok=True, surface=surface, confidence=conf, evidence=evidence,
                                latency_s=float(d.get("latency_s") or 0.0))
            else:
                res = VlmResult(ok=False, error=str(d.get("error")), latency_s=float(d.get("latency_s") or 0.0))
            self.vlm_done.append((self._now(), sent[0], res))   # image time ~ request time

    # ---------- main loop ----------
    def _tick(self):
        t = self._now()
        with self.lock:
            done, self.vlm_done = self.vlm_done, []
            for t_rx, t_img, res in done:
                self.fusion.add_vlm(t_rx, t_img, res.ok, res.surface, res.confidence)
                self.pub_vlm.publish(String(data=json.dumps({
                    "ok": res.ok, "surface": res.surface, "confidence": res.confidence,
                    "evidence": res.evidence, "error": res.error,
                    "latency_s": round(res.latency_s, 2), "image_age_s": round(t_rx - t_img, 2)})))
                if not res.ok:
                    self.get_logger().warn(f"VLM request failed: {res.error}", throttle_duration_sec=10.0)

            state = self.fusion.step(t)

            # a broker answer that never comes (broker not running) must not block future requests
            for rid, (t_sent, mono) in list(self.broker_pending.items()):
                if time.monotonic() - mono > self.vlm_timeout_s + 30.0:
                    self.broker_pending.pop(rid)
                    self.fusion.add_vlm(t, t_sent, False)
                    self.get_logger().warn("no answer from vlm_broker (is it running?)", throttle_duration_sec=30.0)

            if self.fusion.should_query_vlm(t) and self.latest_frame is not None:
                bgr, t_img = self.latest_frame
                self.fusion.mark_vlm_requested(t)
                if self.backend == "broker":
                    self._ask_broker(t)
                else:
                    self.pool.submit(self._vlm_job, bgr.copy(), t_img)

            complete = self.fusion.course_complete
            reason = self.fusion.complete_reason
            feats = dict(self.last_features)
            local_label = self.last_local_label

        self.pub_state.publish(String(data=state))
        self.pub_stairs.publish(Bool(data=(state == "metal")))

        if complete != self.published_complete:
            self.pub_complete.publish(Bool(data=complete))
            self.published_complete = complete
            if complete:
                self.get_logger().info(f"COURSE COMPLETE: {reason}")

        if self.last_debug_t is None or t - self.last_debug_t >= 1.0:
            self.last_debug_t = t
            f = self.fusion
            self.pub_debug.publish(String(data=json.dumps({
                "state": state, "local_label": local_label,
                "local_majority": [f.last_local_label, round(f.last_local_frac, 2)],
                "features": {k: round(v, 3) for k, v in feats.items()},
                "vlm_last": None if f.vlm_last is None else
                    [f.vlm_last.surface, f.vlm_last.confidence, round(t - f.vlm_last.t_img, 1)],
                "vlm_in_flight": f.vlm_in_flight, "vlm_fail_streak": f.vlm_fail_streak,
                "on_course_time_s": round(f.on_course_time, 1),
                "exit_candidate_s": None if f.exit_candidate_since is None
                    else round(t - f.exit_candidate_since, 1),
                "course_complete": complete, "reason": reason})))

    def _on_reset(self, request, response):
        with self.lock:
            self.fusion.reset()
        self.published_complete = None
        response.success = True
        response.message = "surface history and course_complete cleared"
        self.get_logger().info(response.message)
        return response

    def destroy_node(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SurfaceMonitor()
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
