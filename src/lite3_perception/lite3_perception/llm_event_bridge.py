#!/usr/bin/env python3
"""
Reference consumer for /perception/event -> copy this pattern into your LLM package.

- Pairs each event with its keyframe JPEG (matched on stamp).
- Calls the LAN AI server on a worker thread with timeout + retries, so a slow or
  dead server never blocks navigation (booklet rule 14).
- Cross-checks the LLM answer against the detector label (booklet design note 2)
  and publishes the result on /perception/llm_result (JSON).

Replace `query_llm()` with the client your package already uses for the AI server.
"""
import base64
import json
import queue
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

PROMPT = """You are the scene-understanding module of a search-and-rescue robot dog in a
damaged shopping mall (props may be toys or printed pictures; interpret them as the real thing).
The onboard detector flagged: {label} (conf {confidence}), other objects in view: {others}.
Mission tasks: {mission}
Look at the image and answer ONLY with JSON:
{{"object": str, "agrees_with_detector": bool, "room_type": str|null,
  "hazard_symbol": str|null, "is_survivor": bool, "confidence": 0-1, "notes": str,
  "medical_kit_visible": bool, "medical_kit_hint": str|null}}
Medical kits are a priority. Look carefully for ANY first-aid / medical item even if small,
partly hidden or not the flagged object: a box or bag with a red/green/white cross, a white or
red case, bandages, pill bottles, medicine boxes. Set "medical_kit_visible" accordingly.
If none is visible, use "medical_kit_hint" to say where in THIS scene one is most likely
(e.g. "wall-mounted box left of the door", "shelf behind the counter", "next to the survivor"),
otherwise null."""


def query_llm(prompt: str, jpeg_b64: str, timeout_s: float) -> dict:
    """PLACEHOLDER - call your LAN AI server here and return the parsed JSON dict."""
    raise NotImplementedError("wire this to your existing AI-server client")


class LlmEventBridge(Node):
    def __init__(self):
        super().__init__("llm_event_bridge")
        self.declare_parameter("mission_text", "")
        self.declare_parameter("timeout_s", 8.0)
        self.declare_parameter("retries", 2)
        self.declare_parameter("min_accept_confidence", 0.6)
        self.declare_parameter("object_labels", ["survivor", "medical_kit", "hand_tools", "food", "water", "utensils"])
        self.declare_parameter("hazard_list", ["high_voltage", "radioactive", "chemical", "biological", "flammable"])   # standard hazard names

        self.keyframes = {}                 # (sec, nanosec) -> jpeg bytes
        self.jobs = queue.Queue(maxsize=20)
        self.create_subscription(CompressedImage, "/perception/keyframe", self.on_key, 10)
        self.create_subscription(String, "/perception/event", self.on_event, 10)
        self.pub = self.create_publisher(String, "/perception/llm_result", 10)
        # turn on (and set object_detector publish_observations:=false) once query_llm() works:
        # then only LLM-checked results go to the mission reporter, with the room type attached
        self.declare_parameter("publish_observations", False)
        self.pub_obs = self.create_publisher(String, "/mission_reporter/observation", 10)
        threading.Thread(target=self.worker, daemon=True).start()

    def on_key(self, msg):
        self.keyframes[(msg.header.stamp.sec, msg.header.stamp.nanosec)] = bytes(msg.data)
        if len(self.keyframes) > 30:
            self.keyframes.pop(next(iter(self.keyframes)))

    def on_event(self, msg):
        ev = json.loads(msg.data)
        jpg = self.keyframes.pop((ev["stamp"]["sec"], ev["stamp"]["nanosec"]), None)
        if ev["type"] == "qr":
            # QR text is already decoded locally - you may still want the LLM to *interpret* the clue
            pass
        try:
            self.jobs.put_nowait((ev, jpg))
        except queue.Full:
            self.get_logger().warn("LLM queue full, dropping event")

    def worker(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        while rclpy.ok():
            ev, jpg = self.jobs.get()
            labels = [l for l in g("object_labels") if l]
            prompt = PROMPT.format(label=ev["label"], confidence=ev["confidence"],
                                   others=", ".join(ev["other_labels_in_frame"]) or "none",
                                   mission=g("mission_text") or "not given")
            if ev["type"] == "object":
                prompt += (f'\nSet "object" to EXACTLY one of {labels} (or null if none fits). '
                           f'The detector says "{ev["label"]}".')
            if ev["label"] == "survivor":
                prompt += (f'\nThe red box was detected as "{ev.get("detector_label")}". In this arena, survivors '
                           f'are shown as PICTURES of people, toy figures or dolls. Real humans standing outside the arena '
                           f'(marshals, spectators) are NOT survivors. Set "is_survivor" true only for a picture/figure '
                           f'inside the arena, and say in "notes" where in the room it is (left/right/centre).')
            if ev["type"] == "hazard":
                options = [h for h in g("hazard_list") if h] or [t[0] for t in ev.get("hazard_top3", [])]
                prompt += (f'\nThe red box is a yellow triangular hazard warning sign. The onboard template matcher '
                           f'suggests {ev.get("hazard_top3")} (certain={ev.get("hazard_certain")}). '
                           f'Identify the symbol, choosing EXACTLY one of: {options}. Put that exact string in "hazard_symbol".')
            if ev.get("qr_text"):
                prompt += f'\nA QR code in view decodes to: "{ev["qr_text"]}". Explain what it tells us.'
            b64 = base64.b64encode(jpg).decode() if jpg else ""

            answer, err = None, None
            for attempt in range(int(g("retries")) + 1):
                try:
                    answer = query_llm(prompt, b64, float(g("timeout_s")))
                    break
                except Exception as e:  # noqa: BLE001
                    err = repr(e)
                    time.sleep(0.5 * (attempt + 1))

            result = {"event": ev, "llm": answer, "error": err, "accepted": False}
            if answer and ev["type"] == "object":
                obj = answer.get("object")
                answer["object"] = obj if obj in [l for l in g("object_labels") if l] else None
                result["label"] = answer["object"] or ev["label"]
            if answer:
                result["medical_kit_visible"] = bool(answer.get("medical_kit_visible"))
                result["medical_kit_hint"] = answer.get("medical_kit_hint")
            if answer:
                result["accepted"] = bool(answer.get("agrees_with_detector")) or \
                    float(answer.get("confidence", 0)) >= float(g("min_accept_confidence"))
                if ev["label"] == "survivor":
                    result["survivor_confirmed"] = bool(answer.get("is_survivor"))
                if ev["type"] == "hazard":
                    # final hazard = detector's label if it was certain and the LLM agrees, else the LLM's pick
                    llm_h = answer.get("hazard_symbol")
                    if llm_h not in [h for h in g("hazard_list") if h]:
                        llm_h = None                                   # LLM answered outside the standard list
                    result["hazard"] = ev["label"] if (ev.get("hazard_certain") and llm_h in (None, ev["label"])) \
                        else (llm_h or ev["label"])
            else:
                # graceful fallback: keep the detector's own label, flagged as unverified
                result["fallback_label"] = ev["label"]
                result["label"] = ev["label"]
                if ev["type"] == "hazard":
                    result["hazard"] = ev["label"]
            self.pub.publish(String(data=json.dumps(result)))
            if g("publish_observations"):
                from lite3_perception.object_detector_node import observation_from_event
                final = result.get("hazard") or result.get("label") or ev["label"]
                if ev["label"] == "survivor" and answer and not result.get("survivor_confirmed"):
                    continue                                   # LLM says: real bystander, not a survivor
                if answer and not result["accepted"]:
                    continue
                obs = observation_from_event(ev, label=final,
                                             scene_type=(answer or {}).get("room_type"))
                if obs:
                    self.pub_obs.publish(String(data=json.dumps(obs)))
            self.get_logger().info(f"LLM result for {ev['event_id']}: accepted={result['accepted']} err={err}")


def main():
    rclpy.init()
    node = LlmEventBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
