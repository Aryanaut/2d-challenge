#!/usr/bin/env python3
"""
Object detector + trigger node for the Lite3 challenge.

Pipeline (runs at `detect_hz`, always on the *latest* frame, never queues frames):
  camera frame -> YOLO / YOLO-World detection -> multi-frame confirmation
  -> depth + TF localisation -> de-duplication -> publish ONE event per new object

QR codes: with qr_source=scanner they come from the qr_scanner node (/qr/visible, /qr/detections);
with qr_source=builtin they are decoded here with cv2 (deterministic, no LLM needed).

Publishes
  /perception/event       std_msgs/String           JSON, one per confirmed NEW object / QR  <- LLM package subscribes here
  /perception/keyframe    sensor_msgs/CompressedImage annotated JPEG for that event (same header.stamp as event "stamp")
  /perception/detections  std_msgs/String           JSON of raw detections every tick (debug / monitoring)

The detector only answers "something worth looking at is HERE".
The LLM package answers "what does it mean" (room type, hazard symbol, survivor clue).
"""
import json
import ast
import math
import os
import time
import uuid

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transforms)

from lite3_perception.hazard_classifier import HazardClassifier


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image -> BGR ndarray without cv_bridge."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("bgr8", "rgb8"):
        img = buf.reshape(msg.height, msg.step // 3, 3)[:, : msg.width]
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
    if msg.encoding in ("bgra8", "rgba8"):
        img = buf.reshape(msg.height, msg.step // 4, 4)[:, : msg.width]
        code = cv2.COLOR_RGBA2BGR if msg.encoding == "rgba8" else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(img, code)
    if msg.encoding == "mono8":
        return cv2.cvtColor(buf.reshape(msg.height, msg.step)[:, : msg.width], cv2.COLOR_GRAY2BGR)
    if msg.encoding in ("yuv422", "yuyv", "yuv422_yuy2"):
        return cv2.cvtColor(buf.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUY2)
    raise ValueError(f"unsupported encoding {msg.encoding}")


def depth_msg_to_m(msg: Image) -> np.ndarray:
    """Depth image -> float32 metres (handles 16UC1 mm and 32FC1 m)."""
    if msg.encoding in ("16UC1", "mono16"):
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
        return d.astype(np.float32) / 1000.0
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)[:, : msg.width].copy()
    raise ValueError(f"unsupported depth encoding {msg.encoding}")


def observation_from_event(event, label=None, scene_type=None):
    """/perception/event -> lite3_mission_reporter observation dict (None for QR: the reporter's qr_detector does M04)."""
    if event["type"] == "qr":
        return None
    lab = label or event["label"]
    if lab == "survivor":
        kind, lab = "survivor", ""
    elif event["type"] == "hazard":
        kind = "hazard"
    else:
        kind = "object"
    loc = event.get("location") or {}
    pose = event.get("robot_pose") or {}
    if loc.get("xy"):
        pos = loc["xy"]                                    # where the thing is (depth on)
    elif "x" in pose:
        pos = [pose["x"], pose["y"]]                       # fallback: where the robot was (depth off)
    else:
        pos = None
    obs = {"kind": kind, "label": lab, "pos": pos, "confidence": event.get("confidence")}
    scene_type = scene_type or event.get("scene_type")
    if scene_type:
        obs["scene_type"] = scene_type
    return obs


class ObjectDetectorNode(Node):
    def __init__(self):
        super().__init__("object_detector")

        p = self.declare_parameter
        # --- inputs
        p("image_topic", "/camera/color/image_raw/compressed")
        p("depth_topic", "/camera/depth/image_raw")          # must be ALIGNED to colour for localisation
        p("camera_info_topic", "/camera/color/camera_info")
        p("use_depth", True)
        # --- model
        p("backend", "onnx")                                   # "onnx" (no PyTorch needed) | "ultralytics"
        p("model_path", "detector.onnx")                       # bare name = looked up in <pkg share>/models/
        p("class_names", [""])                                 # only needed if the .onnx has no embedded names
        p("nms_iou", 0.45)
        p("open_vocab_classes", [
            "person", "toy figure", "doll", "first aid kit", "red cross sign",
            "fire extinguisher", "electrical panel", "warning sign", "qr code",
            "stove", "refrigerator", "sink", "bed", "sofa", "toilet", "curtain",
        ])
        p("device", "")                                        # "", "cpu", "cuda:0"
        p("imgsz", 640)
        p("conf_threshold", 0.35)
        p("class_conf_overrides", "{}")                        # JSON e.g. {"person":0.5,"qr code":0.25}
        p("ignore_classes", ["curtain"])                       # detected (for /perception/detections) but never triggers an event
        p("report_objects", ["survivor", "hand_tools"])        # ONLY these object labels become events / reports
                                                               # (hazards and QR codes are always reported)
        # --- hazard warning signs (yellow triangles, template matching, no training)
        p("hazard_enable", True)
        p("hazard_template_dir", "")                           # "" = <pkg share>/hazard_templates (filename = label)
        p("hazard_min_area", 400)                              # px^2 of the yellow triangle in the image
        p("hazard_min_score", 0.6)                             # below this / margin -> "uncertain", LLM decides
        p("hazard_min_margin", 0.02)
        # --- survivors: detector labels that mean "survivor" + bystander filter
        p("label_map", '{"person": "survivor", "toy figure": "survivor", "doll": "survivor"}')
        p("survivor_max_height_m", 1.0)                          # taller than this (needs depth) = real human bystander
        p("survivor_new_if_moved_m", 1.5)                        # no-depth de-dup: same survivor if robot moved less than this
        p("survivor_new_if_bearing_deg", 25.0)                   # ... and the bearing differs by less than this
        # --- scene (room type) from what is in view: scene -> labels that suggest it
        p("scene_enable", False)                               # off: a separate scene package provides the room
        p("scene_rules", '{"workshop": ["tool_bench", "hand_tools"], "kitchen": ["kitchen", "utensils"], '
                         '"market": ["market_stall", "cart", "food"], "clinic": ["medical_kit"]}')
        p("scene_decisive", '{"utensils": "kitchen", "kitchen": "kitchen", "tool_bench": "workshop", '
                            '"hand_tools": "workshop", "medical_kit": "clinic"}')   # seeing one of these DECIDES the scene
        p("scene_decisive_hits", 2)                            # ... if seen in at least N frames of the window
        p("scene_decisive_conf", 0.35)                         # ... with at least this confidence
        p("scene_window_s", 6.0)                               # look at detections from the last N seconds
        p("scene_min_score", 1.5)                              # sum of confidences needed to name a scene
        p("scene_topic", "/mission/scene_type")
        # --- QR beside a survivor (M03 = survivor WITH a QR, M05 = survivor without)
        p("qr_assoc_s", 3.0)                                   # QR seen within N s of the survivor = "has QR"
        p("qr_report", False)                                  # False: QR codes are only identified (for M03/M05),
                                                               #        never sent as events / M04 reports
        p("qr_text_topic", "/perception/qr_text")
        p("qr_source", "builtin")                              # scanner = use the qr_scanner node's topics
                                                               # builtin = decode here with cv2 (check_qr)
                                                               # both    = both feed the same QR state
        p("qr_visible_topic", "/qr/visible")                   # qr_scanner: codes in view, every scan
        p("qr_detections_topic", "/qr/detections")             # qr_scanner: once per newly confirmed code
        p("qr_camera", "front")                                # only this qr_scanner camera counts as "in view"
        # --- mission reporter hand-off (lite3_mission_reporter)
        p("publish_observations", True)                        # set false if llm_event_bridge publishes them instead
        p("observation_topic", "/mission_reporter/observation")
        # --- trigger logic
        p("detect_hz", 3.0)
        p("confirm_hits", 3)                                   # need >= N hits ...
        p("class_confirm_overrides", '{"medical_kit": 2}')     # per-label N (lower = more sensitive)
        p("confirm_window_s", 2.0)                             # ... within this window
        p("dedup_radius_m", 1.0)                               # same label within this distance = same object
        p("dedup_cooldown_s", 20.0)                            # used when no map position is available
        p("max_range_m", 4.0)
        # --- frames
        p("map_frame", "map")                                  # or "odom" if you have no map frame
        p("base_frame", "base_link")
        p("jpeg_quality", 80)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.conf_thr = float(g("conf_threshold"))
        self.class_conf = json.loads(g("class_conf_overrides"))
        self.ignore = set(g("ignore_classes"))
        self.report_objects = set(x for x in g("report_objects") if x)
        self.confirm_hits = int(g("confirm_hits"))
        self.class_confirm = {k: int(v) for k, v in json.loads(g("class_confirm_overrides")).items()}
        self.confirm_window = float(g("confirm_window_s"))
        self.dedup_r = float(g("dedup_radius_m"))
        self.dedup_cooldown = float(g("dedup_cooldown_s"))
        self.max_range = float(g("max_range_m"))
        self.map_frame = g("map_frame")
        self.base_frame = g("base_frame")
        self.imgsz = int(g("imgsz"))
        self.device = g("device") or None
        self.jpeg_q = int(g("jpeg_quality"))

        # --- model
        self.backend = g("backend")
        self.nms_iou = float(g("nms_iou"))
        model_path = g("model_path")
        if not os.path.isabs(model_path):
            try:
                from ament_index_python.packages import get_package_share_directory
                cand = os.path.join(get_package_share_directory("lite3_perception"), "models", model_path)
                if os.path.exists(cand):
                    model_path = cand
            except Exception:  # noqa: BLE001
                pass
        if self.backend == "onnx":
            self.load_onnx(model_path, [c for c in g("class_names") if c])
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            classes = list(g("open_vocab_classes"))
            if "world" in model_path.lower() and classes:
                self.model.set_classes(classes)  # first call needs CLIP weights (internet)
            self.names = dict(self.model.names)
        self.get_logger().info(f"model {model_path} ({self.backend}), classes: {list(self.names.values())}")

        self.qr = cv2.QRCodeDetector()

        self.hazard = None
        if g("hazard_enable"):
            tdir = g("hazard_template_dir")
            if not tdir:
                try:
                    from ament_index_python.packages import get_package_share_directory
                    tdir = os.path.join(get_package_share_directory("lite3_perception"), "hazard_templates")
                except Exception:  # noqa: BLE001
                    tdir = ""
            try:
                self.hazard = HazardClassifier(tdir)
                self.get_logger().info(f"hazard classes: {self.hazard.labels}")
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"hazard classifier disabled: {e}")
        self.label_map = json.loads(g("label_map"))
        self.survivor_max_h = float(g("survivor_max_height_m"))
        self.survivor_moved = float(g("survivor_new_if_moved_m"))
        self.survivor_bearing = float(g("survivor_new_if_bearing_deg"))
        self.scene_enable = bool(g("scene_enable"))
        self.scene_rules = json.loads(g("scene_rules"))
        self.scene_window = float(g("scene_window_s"))
        self.scene_decisive = json.loads(g("scene_decisive"))
        self.scene_dec_hits = int(g("scene_decisive_hits"))
        self.scene_dec_conf = float(g("scene_decisive_conf"))
        self.scene_min = float(g("scene_min_score"))
        self.scene_hist = []                                   # (t, label, conf)
        self.scene = ""
        self.qr_assoc = float(g("qr_assoc_s"))
        self.qr_report = bool(g("qr_report"))
        self.last_qr_seen = -1e9                               # any QR in view (decoded or not)
        self.last_qr_text = ""
        self.qr_source = str(g("qr_source")).strip().lower()
        if self.qr_source not in ("scanner", "builtin", "both"):
            self.get_logger().warn(f"qr_source '{self.qr_source}' unknown, using 'builtin'")
            self.qr_source = "builtin"
        self.qr_camera = str(g("qr_camera"))
        self.last_scanner_msg = None                           # time of the last /qr/visible (watchdog)
        self.started = time.time()
        self.last_frame = None                                 # (bgr, header) of the last processed frame
        self.hz_min_area = int(g("hazard_min_area"))
        self.hz_min_score = float(g("hazard_min_score"))
        self.hz_min_margin = float(g("hazard_min_margin"))

        # --- state
        self.latest_img = None          # (bgr, header)
        self.latest_depth = None        # (metres, header)
        self.K = None                   # fx, fy, cx, cy
        self.hits = {}                  # label -> list[(t, det)]
        self.reported = []              # list of dict(label, xy, t)
        self.reported_qr = set()

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- io
        topic = g("image_topic")
        if topic.endswith("compressed"):
            self.create_subscription(CompressedImage, topic, self.on_compressed, qos_profile_sensor_data)
        else:
            self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)
        if g("use_depth"):
            self.create_subscription(Image, g("depth_topic"), self.on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, g("camera_info_topic"), self.on_info, qos_profile_sensor_data)

        self.pub_event = self.create_publisher(String, "/perception/event", 10)
        self.pub_scene = self.create_publisher(String, g("scene_topic"), 10)
        self.pub_qr_text = self.create_publisher(String, g("qr_text_topic"), 10)
        self.publish_obs = bool(self.get_parameter("publish_observations").value)
        self.pub_obs = self.create_publisher(String, self.get_parameter("observation_topic").value, 10)
        self.pub_key = self.create_publisher(CompressedImage, "/perception/keyframe", 10)
        self.pub_dets = self.create_publisher(String, "/perception/detections", 10)
        if self.qr_source in ("scanner", "both"):
            self.create_subscription(String, g("qr_visible_topic"), self.on_qr_visible, 10)
            self.create_subscription(String, g("qr_detections_topic"), self.on_qr_detection, 10)
            self.create_timer(5.0, lambda: self.check_scanner_alive(time.time()))

        self.create_timer(1.0 / float(g("detect_hz")), self.tick)
        self.get_logger().info(f"listening on {topic}; QR from {self.qr_source}"
                               + (f" ({g('qr_visible_topic')}, {g('qr_detections_topic')}, "
                                  f"camera '{self.qr_camera}')" if self.qr_source != "builtin" else ""))

    # ------------------------------------------------------------------ callbacks
    def on_compressed(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            self.latest_img = (img, msg.header)

    def on_image(self, msg):
        try:
            self.latest_img = (image_msg_to_bgr(msg), msg.header)
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)

    def on_depth(self, msg):
        try:
            self.latest_depth = (depth_msg_to_m(msg), msg.header)
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)

    def on_info(self, msg):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    # ------------------------------------------------------------------ main loop
    def tick(self):
        if self.latest_img is None:
            return
        img, header = self.latest_img
        self.latest_img = None   # process each frame once
        now = time.time()

        dets = self.detect(img)
        hazards = self.detect_hazards(img)
        self.pub_dets.publish(String(data=json.dumps({"stamp": now, "detections": dets, "hazards": hazards})))

        self.last_frame = (img, header)
        if self.qr_source in ("builtin", "both"):
            self.check_qr(img, header, dets)
        if self.scene_enable:
            self.update_scene(dets, now)

        candidates = []
        for d in dets:
            if d["label"] in self.ignore:
                continue
            mapped = self.label_map.get(d["label"], d["label"])
            if self.report_objects and mapped not in self.report_objects:
                continue                                       # still used for the scene, never reported
            if mapped != d["label"]:
                d = dict(d, label=mapped, detector_label=d["label"])
            candidates.append(("object", mapped, d))
        for hz in hazards:
            key = hz["label"] if hz["certain"] else "unknown_hazard"
            candidates.append(("hazard", key, hz))

        for kind, key, d in candidates:
            hkey = f"{kind}:{key}"
            lst = [h for h in self.hits.get(hkey, []) if now - h[0] < self.confirm_window]
            lst.append((now, d))
            self.hits[hkey] = lst
            if len(lst) < self.class_confirm.get(key, self.confirm_hits):
                continue

            loc = self.localise(d, img.shape, header)
            if loc and loc.get("range_m", 0) > self.max_range:
                continue  # too far: wait until we are closer for a better look
            extra = {}
            if key == "survivor" and loc.get("range_m") and self.K:
                h_m = (d["bbox_xyxy"][3] - d["bbox_xyxy"][1]) * loc["range_m"] / self.K[1]
                extra["height_m"] = round(h_m, 2)
                if h_m > self.survivor_max_h:
                    self.get_logger().info(f"ignoring person {h_m:.2f} m tall (real bystander?)",
                                           throttle_duration_sec=5.0)
                    self.hits[hkey] = []
                    continue
            pose = self.robot_pose()
            if self.already_reported(hkey, loc, now, pose):
                continue

            mean_conf = float(np.mean([h[1]["confidence"] for h in lst]))
            self.hits[hkey] = []
            self.reported.append({"label": hkey, "xy": (loc or {}).get("xy"), "t": now,
                                  "pose": pose, "bearing": (loc or {}).get("bearing_deg")})
            extra.update({"confirm_hits": len(lst), "mean_confidence": round(mean_conf, 3)})
            if key == "survivor":
                extra["qr_in_view"] = (time.time() - self.last_qr_seen) < self.qr_assoc
                extra["qr_text_recent"] = self.last_qr_text if extra["qr_in_view"] else ""
                extra["bbox_h_frac"] = round((d["bbox_xyxy"][3] - d["bbox_xyxy"][1]) / img.shape[0], 3)
            if kind == "hazard":
                # most common top-1 over the confirming frames, plus the latest top-3 for the LLM
                votes = {}
                for _, h in lst:
                    votes[h["label"]] = votes.get(h["label"], 0) + 1
                extra.update({"hazard_votes": votes, "hazard_top3": d["top3"],
                              "hazard_certain": d["certain"]})
            self.emit(kind, img, header, dets, d, loc, extra=extra)

    # ------------------------------------------------------------------ ONNX backend (no PyTorch)
    def load_onnx(self, path, names_param):
        self.ort = None
        names = None
        try:
            import onnxruntime as ort
            self.ort = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            inp = self.ort.get_inputs()[0]
            self.ort_in = inp.name
            self.onnx_size = inp.shape[2] if isinstance(inp.shape[2], int) else self.imgsz
            meta = self.ort.get_modelmeta().custom_metadata_map
            if "names" in meta:
                names = ast.literal_eval(meta["names"])   # ultralytics embeds {0: 'cls', ...}
        except ImportError:
            self.get_logger().info("onnxruntime not installed - using OpenCV DNN")
            self.net = cv2.dnn.readNetFromONNX(path)
            self.onnx_size = self.imgsz
        if names_param:
            names = {i: n for i, n in enumerate(names_param)}
        self.names = names or {}

    def detect_onnx(self, img):
        S = self.onnx_size
        h, w = img.shape[:2]
        r = min(S / h, S / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top, left = (S - nh) // 2, (S - nw) // 2
        canvas = np.full((S, S, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)

        if self.ort is not None:
            out = self.ort.run(None, {self.ort_in: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()
        pred = out[0]
        if pred.shape[0] > pred.shape[1]:      # (N, 4+nc) -> (4+nc, N)
            pred = pred.T
        scores = pred[4:].T                    # N x nc
        cls = scores.argmax(1)
        conf = scores.max(1)
        min_thr = min([self.conf_thr] + list(self.class_conf.values()))
        keep = conf >= min_thr
        if not keep.any():
            return []
        cx, cy, bw, bh = pred[0][keep], pred[1][keep], pred[2][keep], pred[3][keep]
        cls, conf = cls[keep], conf[keep]
        x1 = (cx - bw / 2 - left) / r
        y1 = (cy - bh / 2 - top) / r
        bw, bh = bw / r, bh / r
        rects = [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, bw, bh)]
        idx = cv2.dnn.NMSBoxesBatched(rects, conf.tolist(), cls.tolist(), min_thr, self.nms_iou) \
            if hasattr(cv2.dnn, "NMSBoxesBatched") else cv2.dnn.NMSBoxes(rects, conf.tolist(), min_thr, self.nms_iou)
        out_dets = []
        for i in np.array(idx).flatten():
            label = self.names.get(int(cls[i]), f"class_{int(cls[i])}")
            c = float(conf[i])
            if c < float(self.class_conf.get(label, self.conf_thr)):
                continue
            x, y, ww, hh = rects[i]
            out_dets.append({"label": label, "confidence": round(c, 3),
                             "bbox_xyxy": [max(0, round(x)), max(0, round(y)),
                                           min(w - 1, round(x + ww)), min(h - 1, round(y + hh))]})
        return out_dets

    # ------------------------------------------------------------------ pieces
    def detect(self, img):
        if self.backend == "onnx":
            return self.detect_onnx(img)
        r = self.model.predict(img, imgsz=self.imgsz, conf=min([self.conf_thr] + list(self.class_conf.values())),
                               device=self.device, verbose=False)[0]
        out = []
        for box in r.boxes:
            label = r.names[int(box.cls)]
            conf = float(box.conf)
            if conf < float(self.class_conf.get(label, self.conf_thr)):
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            out.append({"label": label, "confidence": round(conf, 3), "bbox_xyxy": [round(x1), round(y1), round(x2), round(y2)]})
        return out

    def update_scene(self, dets, now):
        """Name the room from what has been in view recently (rules in scene_rules)."""
        self.scene_hist = [h for h in self.scene_hist if now - h[0] < self.scene_window]
        self.scene_hist += [(now, d.get("detector_label", d["label"]), d["confidence"]) for d in dets]
        # 1) deciding objects: e.g. a cooking pot (utensils) means kitchen, whatever else is around
        dec_hits = {}
        for _, lab, c in self.scene_hist:
            if lab in self.scene_decisive and c >= self.scene_dec_conf:
                dec_hits[lab] = dec_hits.get(lab, 0) + 1
        decided = [(n, self.scene_decisive[lab]) for lab, n in dec_hits.items() if n >= self.scene_dec_hits]
        if decided:
            best = max(decided)[1]                             # the deciding object seen most often
            if best != self.scene:
                self.scene = best
                self.pub_scene.publish(String(data=best))
                self.get_logger().info(f"scene -> {best}  (decided by {dict(dec_hits)})")
            return
        # 2) otherwise: vote with everything in view (scene_rules)
        scores = {}
        for _, lab, c in self.scene_hist:
            scenes = [sc for sc, labs in self.scene_rules.items() if lab in labs]
            for sc in scenes:
                scores[sc] = scores.get(sc, 0.0) + c / len(scenes)
        if not scores:
            return
        best = max(scores, key=scores.get)
        if scores[best] >= self.scene_min and best != self.scene:
            self.scene = best
            self.pub_scene.publish(String(data=best))
            self.get_logger().info(f"scene -> {best}  (scores {({k: round(v, 1) for k, v in scores.items()})})")

    def detect_hazards(self, img):
        if self.hazard is None:
            return []
        out = []
        for h in self.hazard.detect(img, self.hz_min_area):
            h["certain"] = h["score"] >= self.hz_min_score and h["margin"] >= self.hz_min_margin
            h["confidence"] = h["score"]
            out.append(h)
        return out

    def check_qr(self, img, header, dets):
        try:
            ok, decoded, points, _ = self.qr.detectAndDecodeMulti(img)
        except cv2.error:
            return
        if not ok:
            return
        self.last_qr_seen = time.time()                        # a QR is in view (even if not decodable yet)
        for text, pts in zip(decoded, points):
            if text:
                self.last_qr_text = text
            if not text or text in self.reported_qr:
                continue
            self.reported_qr.add(text)
            self.pub_qr_text.publish(String(data=text))        # exact decoded string, for other packages
            if not self.qr_report:
                continue                                       # identified only, not reported
            xs, ys = pts[:, 0], pts[:, 1]
            d = {"label": "qr code", "confidence": 1.0,
                 "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]}
            loc = self.localise(d, img.shape, header)
            self.emit("qr", img, header, dets, d, loc, extra={"qr_text": text})

    # ------------------------------------------------------------------ QR from the qr_scanner node
    def on_qr_visible(self, msg):
        """/qr/visible (every scan): a code from qr_camera in view, read or not, = 'a QR is in view'."""
        self.last_scanner_msg = time.time()
        try:
            codes = json.loads(msg.data).get("codes") or []
        except (json.JSONDecodeError, AttributeError):
            return
        mine = [c for c in codes if isinstance(c, dict) and c.get("camera") == self.qr_camera]
        if not mine:
            return
        self.last_qr_seen = time.time()
        texts = [c["text"] for c in mine if c.get("text")]
        if texts:
            self.last_qr_text = texts[-1]

    def on_qr_detection(self, msg):
        """/qr/detections (once per new code): forward the exact raw text, like check_qr does."""
        try:
            det = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(det, dict):
            return
        text = str(det.get("text") or "")                      # raw text; NOT 'decoded' (key_map/caesar)
        if not text or text in self.reported_qr:
            return
        if det.get("camera", self.qr_camera) == self.qr_camera:
            self.last_qr_seen = time.time()
            self.last_qr_text = text
        self.reported_qr.add(text)
        self.pub_qr_text.publish(String(data=text))
        if not self.qr_report or self.last_frame is None:
            return
        img, header = self.last_frame
        loc = {}
        if det.get("bearing_deg") is not None:
            loc["bearing_deg"] = det["bearing_deg"]
        if det.get("distance_m") is not None:
            loc["range_m"] = det["distance_m"]
        mp = det.get("map_point")
        if mp and det.get("frame") == self.map_frame:
            loc["xy"] = [round(mp[0], 2), round(mp[1], 2)]
            loc["position_map"] = {"x": loc["xy"][0], "y": loc["xy"][1], "z": round(mp[2], 2),
                                   "frame": self.map_frame}
        d = {"label": "qr code", "confidence": 1.0, "bbox_xyxy": [0, 0, 0, 0]}
        self.emit("qr", img, header, [], d, loc, extra={"qr_text": text, "qr_camera": det.get("camera")})

    def check_scanner_alive(self, now):
        """Warn if qr_scanner is silent: M03/M04 would quietly never trigger."""
        last = self.last_scanner_msg if self.last_scanner_msg is not None else self.started
        if now - last > 10.0:
            self.get_logger().warn("no /qr/visible for >10 s: is qr_scanner running? (M03/M04 need it)",
                                   throttle_duration_sec=30.0)

    def localise(self, d, shape, header):
        """Bearing always; 3-D map position if depth + intrinsics + TF are available."""
        h, w = shape[:2]
        x1, y1, x2, y2 = d["bbox_xyxy"]
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        fx, fy, cx, cy = self.K if self.K else (w / 2.0 / math.tan(math.radians(35)), None, w / 2.0, h / 2.0)
        loc = {"bearing_deg": round(-math.degrees(math.atan2(u - cx, fx)), 1)}  # +left, camera frame

        if self.K is None or self.latest_depth is None:
            return loc
        depth, dh = self.latest_depth
        sx, sy = depth.shape[1] / w, depth.shape[0] / h   # tolerate different resolutions
        bw, bh = (x2 - x1), (y2 - y1)
        cx0, cx1 = int((x1 + 0.35 * bw) * sx), int((x2 - 0.35 * bw) * sx)
        cy0, cy1 = int((y1 + 0.35 * bh) * sy), int((y2 - 0.35 * bh) * sy)
        patch = depth[max(cy0, 0):max(cy1, cy0 + 1), max(cx0, 0):max(cx1, cx0 + 1)]
        valid = patch[(patch > 0.1) & np.isfinite(patch)]
        if valid.size < 5:
            return loc
        z = float(np.median(valid))
        p = PointStamped()
        p.header.frame_id = dh.frame_id or header.frame_id
        p.header.stamp = dh.stamp
        p.point.x = (u - cx) * z / fx
        p.point.y = (v - cy) * z / fy
        p.point.z = z
        loc["range_m"] = round(z, 2)
        try:
            pm = self.tf_buffer.transform(p, self.map_frame, timeout=Duration(seconds=0.1))
            loc["xy"] = [round(pm.point.x, 2), round(pm.point.y, 2)]
            loc["position_map"] = {"x": loc["xy"][0], "y": loc["xy"][1], "z": round(pm.point.z, 2), "frame": self.map_frame}
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"TF to {self.map_frame} failed: {e}", throttle_duration_sec=5.0)
        return loc

    def already_reported(self, label, loc, now, pose=None):
        loc = loc or {}
        for r in self.reported:
            if r["label"] != label:
                continue
            if loc.get("xy") and r["xy"]:                      # best: compare map positions
                if math.dist(loc["xy"], r["xy"]) < self.dedup_r:
                    return True
            elif pose and r.get("pose") and loc.get("bearing_deg") is not None and r.get("bearing") is not None:
                # no depth: same thing if robot is in about the same spot looking the same way
                moved = math.dist((pose["x"], pose["y"]), (r["pose"]["x"], r["pose"]["y"]))
                dyaw = abs(((pose["yaw_deg"] + loc["bearing_deg"]) - (r["pose"]["yaw_deg"] + r["bearing"]) + 180) % 360 - 180)
                if moved < self.survivor_moved and dyaw < self.survivor_bearing:
                    return True
            elif now - r["t"] < self.dedup_cooldown:
                return True
        return False

    def robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            return {"x": round(t.transform.translation.x, 2), "y": round(t.transform.translation.y, 2),
                    "yaw_deg": round(math.degrees(yaw), 1), "frame": self.map_frame}
        except Exception:  # noqa: BLE001
            return None

    def emit(self, kind, img, header, all_dets, d, loc, extra):
        stamp = self.get_clock().now().to_msg()
        event = {
            "event_id": uuid.uuid4().hex[:8],
            "type": kind,                                   # "object" | "qr" | "hazard"
            "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
            "label": d["label"],
            "confidence": d["confidence"],
            "bbox_xyxy": d["bbox_xyxy"],
            "image_size": [img.shape[1], img.shape[0]],
            "location": loc or {},
            "robot_pose": self.robot_pose(),
            "detector_label": d.get("detector_label", d["label"]),
            "other_labels_in_frame": sorted({x["label"] for x in all_dets if x is not d}),
            "scene_type": self.scene,
            **extra,
        }

        vis = img.copy()
        for x in all_dets + ([] if any(x is d for x in all_dets) else [d]):
            a, b, c, e = x["bbox_xyxy"]
            col = (0, 0, 255) if x is d else (0, 200, 0)
            cv2.rectangle(vis, (a, b), (c, e), col, 2)
            cv2.putText(vis, f'{x["label"]} {x["confidence"]:.2f}', (a, max(b - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        if ok:
            km = CompressedImage()
            km.header.stamp = stamp
            km.header.frame_id = header.frame_id
            km.format = "jpeg"
            km.data = jpg.tobytes()
            self.pub_key.publish(km)   # publish image FIRST so the LLM side already has it when the event lands
        self.pub_event.publish(String(data=json.dumps(event)))
        if self.publish_obs:
            obs = observation_from_event(event)
            if obs:
                self.pub_obs.publish(String(data=json.dumps(obs)))
        self.get_logger().info(f"EVENT {kind}: {d['label']} {loc}")


def main():
    rclpy.init()
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
