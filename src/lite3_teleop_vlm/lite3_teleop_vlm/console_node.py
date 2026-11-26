"""Keyboard console: drive the Lite3 and test the VLM from one terminal.

Run it in its own terminal (it needs the keyboard), ideally over SSH on the robot's group
computer so driving does not depend on Wi-Fi latency:

  ros2 run lite3_teleop_vlm teleop_console

Keys
  drive   w/s or Up/Down  forward/back     a/d or Left/Right  turn     q/e  strafe
          (hold to move: stops 0.6 s after you let go)
          space / x / Esc  STOP            + / -  faster / slower
  robot   n  auto (navigation) mode        m  manual mode            t t  stand/sit (press twice)
  VLM     v  ask about the FRONT camera with the selected prompt
          b  ask about the DOWN camera (surface prompt)
          p / o  next / previous prompt    k  save a snapshot of both cameras
  other   r r  reset surface_monitor and qr_scanner (press twice)    h  help    Ctrl+C  quit
"""

import json
import shutil
import sys
import textwrap
import threading
import time
import uuid
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from .keyboard import RawKeyboard
from .teleop_logic import Teleop, TeleopConfig

# Lite3 /simple_cmd codes (from the lab handouts)
CMD_AUTO = 0x21010C03       # "Navigation" mode - robot follows /cmd_vel (lite3_auto)
CMD_MANUAL = 0x21010C02     # "Manual" mode - handheld controller (lite3_manual)
CMD_STAND_SIT = 0x21010202  # toggle stand / sit

HELP = __doc__.split("Keys", 1)[1].rstrip()


class ConsoleNode(Node):
    def __init__(self):
        super().__init__("teleop_console")
        p = self.declare_parameter
        p("cmd_vel_topic", "/cmd_vel")
        p("simple_cmd_topic", "/simple_cmd")
        p("linear_speed", 0.3)
        p("angular_speed", 0.6)
        p("max_linear", 0.5)
        p("max_angular", 1.2)
        p("hold_s", 0.6)
        p("latch", False)
        p("publish_hz", 10.0)
        p("front_prompts", ["scene", "room", "hazard", "survivor"])
        p("down_prompt", "surface")
        p("manual_on_exit", True)        # switch back to manual mode on quit if we switched to auto
        g = lambda n: self.get_parameter(n).value

        self.teleop = Teleop(TeleopConfig(
            linear_speed=float(g("linear_speed")), angular_speed=float(g("angular_speed")),
            max_linear=float(g("max_linear")), max_angular=float(g("max_angular")),
            hold_s=float(g("hold_s")), latch=bool(g("latch"))))
        self.prompts = list(g("front_prompts"))
        self.prompt_i = 0
        self.down_prompt = g("down_prompt")
        self.manual_on_exit = bool(g("manual_on_exit"))

        self.lock = threading.Lock()
        self.events = deque(maxlen=7)
        self.state = {"surface": "-", "complete": None, "qr_seq": "", "qr_last": None,
                      "vlm_status": None, "vlm_last": None, "mode": "unknown (not set from here)"}
        self.pending_ids = {}
        self.was_moving = False
        self.zero_sends = 0
        self.sent_auto = False

        self.pub_cmd = self.create_publisher(Twist, g("cmd_vel_topic"), 10)
        self.pub_req = self.create_publisher(String, "/vlm/request", 10)
        self.pub_mode = None
        try:
            from transfer_interfaces.msg import MotionSimpleCMD
            self.MotionSimpleCMD = MotionSimpleCMD
            self.pub_mode = self.create_publisher(MotionSimpleCMD, g("simple_cmd_topic"), 10)
        except ImportError:
            self.MotionSimpleCMD = None
            self.event("transfer_interfaces not sourced: n/m/t disabled. "
                       "Run on the robot after `source ~/transfer_ws/install/setup.bash`, "
                       "or use lite3_auto / lite3_manual in another terminal.")

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/surface/state", lambda m: self._set("surface", m.data), 10)
        self.create_subscription(Bool, "/surface/course_complete", self._on_complete, latched)
        self.create_subscription(String, "/qr/sequence", lambda m: self._set("qr_seq", m.data), latched)
        self.create_subscription(String, "/qr/detections", self._on_qr, 10)
        self.create_subscription(String, "/vlm/status", self._on_vlm_status, 10)
        self.create_subscription(String, "/vlm/result", self._on_vlm_result, 20)
        self.reset_clients = [self.create_client(Trigger, "/surface_monitor/reset"),
                              self.create_client(Trigger, "/qr_scanner/reset")]
        self.create_timer(1.0 / max(1.0, float(g("publish_hz"))), self._publish_cmd)

    # ---------- helpers ----------
    def event(self, text):
        with self.lock:
            self.events.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")

    def _set(self, key, value):
        with self.lock:
            self.state[key] = value

    def _now(self):
        return time.monotonic()

    # ---------- subscriptions ----------
    def _on_complete(self, msg):
        with self.lock:
            changed = self.state["complete"] != msg.data
            self.state["complete"] = msg.data
        if changed and msg.data:
            self.event("SURFACE MONITOR: COURSE COMPLETE")

    def _on_qr(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._set("qr_last", d)
        self.event(f"QR #{d.get('index')}: '{d.get('text')}' -> '{d.get('decoded')}' at {d.get('map_point')}")

    def _on_vlm_status(self, msg):
        try:
            self._set("vlm_status", json.loads(msg.data))
        except json.JSONDecodeError:
            pass

    def _on_vlm_result(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._set("vlm_last", d)
        mine = self.pending_ids.pop(d.get("id"), None)
        who = "you" if mine else d.get("source")
        if d.get("snapshot_only"):
            self.event(f"snapshot saved: {', '.join(d.get('saved') or []) or d.get('error')}")
        elif d.get("ok"):
            self.event(f"VLM ({who}, {d.get('prompt')}/{d.get('camera')}) answered in {d.get('latency_s')} s")
        else:
            self.event(f"VLM ({who}, {d.get('prompt')}) failed: {d.get('error')}")

    # ---------- outputs ----------
    def _publish_cmd(self):
        vx, vy, wz = self.teleop.velocity(self._now())
        moving = (vx, vy, wz) != (0.0, 0.0, 0.0)
        if moving:
            self.zero_sends = 3
        if moving or self.zero_sends > 0:      # after stopping, send zero a few times then go quiet
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
            self.pub_cmd.publish(t)
            if not moving:
                self.zero_sends -= 1

    def send_mode(self, code, label):
        if self.pub_mode is None:
            self.event(f"cannot send {label}: transfer_interfaces not available here")
            return
        msg = self.MotionSimpleCMD()
        msg.cmd_code = code
        msg.type = 0
        self.pub_mode.publish(msg)
        if code == CMD_AUTO:
            self.sent_auto = True
            self._set("mode", "AUTO (navigation) - robot follows /cmd_vel")
        elif code == CMD_MANUAL:
            self._set("mode", "MANUAL - handheld controller")
        self.event(f"sent {label}")

    def ask_vlm(self, camera, prompt, snapshot_only=False):
        rid = uuid.uuid4().hex[:8]
        self.pending_ids[rid] = time.time()
        self.pub_req.publish(String(data=json.dumps({
            "id": rid, "source": "console", "camera": camera, "prompt": prompt,
            "priority": 10, "max_age_s": 30.0, "snapshot_only": snapshot_only})))
        self.event("snapshot requested" if snapshot_only else f"asked VLM: {prompt} on {camera} camera")

    def reset_monitors(self):
        for c in self.reset_clients:
            if c.service_is_ready():
                c.call_async(Trigger.Request())
                self.event(f"reset sent to {c.srv_name}")
            else:
                self.event(f"{c.srv_name} not running")

    # ---------- key handling ----------
    def on_key(self, key):
        action = self.teleop.key(key, self._now())
        if action is None:
            return
        if action == "stop":
            self.event("STOP")
        elif action == "speed":
            self.event(f"speed {self.teleop.linear:.2f} m/s, turn {self.teleop.angular:.2f} rad/s")
        elif action == "mode_auto":
            self.send_mode(CMD_AUTO, "AUTO / navigation mode (check no other group is driving!)")
        elif action == "mode_manual":
            self.teleop.stop()
            self.send_mode(CMD_MANUAL, "MANUAL mode")
        elif action == "confirm_stand_sit":
            self.event("press t again within 1.5 s to stand/sit")
        elif action == "stand_sit":
            self.teleop.stop()
            self.send_mode(CMD_STAND_SIT, "stand/sit toggle")
        elif action == "vlm_front":
            self.ask_vlm("front", self.prompts[self.prompt_i])
        elif action == "vlm_down":
            self.ask_vlm("down", self.down_prompt)
        elif action in ("prompt_next", "prompt_prev"):
            self.prompt_i = (self.prompt_i + (1 if action == "prompt_next" else -1)) % len(self.prompts)
            self.event(f"front prompt: {self.prompts[self.prompt_i]}")
        elif action == "snapshot":
            self.ask_vlm("front", "", snapshot_only=True)
        elif action == "confirm_reset":
            self.event("press r again within 1.5 s to reset surface_monitor and qr_scanner")
        elif action == "reset":
            self.reset_monitors()
        elif action == "help":
            self.event("keys: see the bottom of the screen")

    def shutdown_robot(self):
        self.teleop.stop()
        for _ in range(3):
            self.pub_cmd.publish(Twist())
            time.sleep(0.05)
        if self.manual_on_exit and self.sent_auto:
            self.send_mode(CMD_MANUAL, "MANUAL mode (on exit)")
            time.sleep(0.2)

    # ---------- screen ----------
    def render(self):
        cols = shutil.get_terminal_size((100, 40)).columns
        width = max(60, min(cols, 110)) - 2
        vx, vy, wz = self.teleop.velocity(self._now())
        with self.lock:
            st = dict(self.state)
            events = list(self.events)
        vs = st["vlm_status"] or {}
        last = st["vlm_last"] or {}
        lines = ["LITE3 TELEOP + VLM TEST CONSOLE".ljust(width), "-" * width]
        lines.append(f"Robot   mode: {st['mode']}")
        moving = "MOVING" if (vx, vy, wz) != (0.0, 0.0, 0.0) else "stopped"
        lines.append(f"        cmd: vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}  [{moving}]   "
                     f"speed {self.teleop.linear:.2f} m/s / {self.teleop.angular:.2f} rad/s")
        comp = {None: "-", True: "YES", False: "no"}[st["complete"]]
        lines.append(f"Surface {st['surface']:8s} course complete: {comp}")
        ql = st["qr_last"] or {}
        lines.append(f"QR      found: {st['qr_seq'] or '-'}   last: "
                     f"{ql.get('text', '-')} -> {ql.get('decoded', '-')}")
        inflight = vs.get("in_flight")
        lines.append(f"VLM     prompt [p/o]: {self.prompts[self.prompt_i]:9s} | in flight: "
                     f"{(inflight or {}).get('source', '-')}/{(inflight or {}).get('prompt', '')} | "
                     f"queue {len(vs.get('queue', []))} | ok {vs.get('ok', '-')} fail {vs.get('failed', '-')}"
                     + ("" if vs else "  (no broker status - is vlm_broker running?)"))
        cams = vs.get("cameras") or {}
        if cams:
            lines.append("        camera frame age: " + ", ".join(f"{k} {v:.1f}s" for k, v in cams.items()))
        lines.append("-" * width)
        if last and not last.get("snapshot_only"):
            head = (f"Last VLM answer ({last.get('source')}, {last.get('prompt')}/{last.get('camera')}, "
                    f"{last.get('latency_s')} s):")
            lines.append(head)
            body = last.get("result") if last.get("ok") else "ERROR: " + str(last.get("error"))
            parsed = last.get("parsed")
            if parsed:
                body = json.dumps(parsed, ensure_ascii=False)
            for ln in textwrap.wrap(str(body), width - 2)[:8]:
                lines.append("  " + ln)
        else:
            lines.append("Last VLM answer: -")
        lines.append("-" * width)
        lines.append("Events:")
        lines += ["  " + e[:width - 2] for e in events] or ["  -"]
        lines.append("-" * width)
        lines += [ln[:width] for ln in HELP.strip("\n").splitlines()]
        # redraw in place (cursor home, clear each line, clear below) - no flicker over SSH
        sys.stdout.write("\x1b[H" + "".join(ln + "\x1b[K\r\n" for ln in lines) + "\x1b[J")
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = ConsoleNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        with RawKeyboard() as kb:
            last_draw = 0.0
            while rclpy.ok():
                for k in kb.read(0.05):
                    node.on_key(k)
                if time.monotonic() - last_draw > 0.2:
                    node.render()
                    last_draw = time.monotonic()
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(e)
    finally:
        node.shutdown_robot()
        sys.stdout.write("\r\nconsole stopped: robot commanded to stop\r\n")
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
