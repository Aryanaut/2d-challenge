"""The explorer's side of the AI link: JSON requests to the llm_client node (lite3_llm_tools) and their replies.

The explorer has no AI code of its own: no HTTP, no prompts, no image handling. It sends
  /llm/request   {"id": ..., "task": ..., ...}   (std_msgs/String, JSON)
and reads
  /llm/response  {"id": ..., "ok": true, "result": {...}} or {"ok": false, "error": "..."}
  /llm/status    {"ready": true, ...}   (latched, ~1 Hz)
If the node is not running (or use_ai is false) `ready` is False and every call fails at once, so the
caller falls back to its non-AI behaviour. The protocol is described in lite3_llm_tools/tools.py.
"""
import json
import time
from typing import Optional

from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

REQUEST_TOPIC, RESPONSE_TOPIC, STATUS_TOPIC = '/llm/request', '/llm/response', '/llm/status'


def _json_default(obj):
    if hasattr(obj, 'item'):          # numpy scalars
        return obj.item()
    if hasattr(obj, 'tolist'):        # numpy arrays
        return obj.tolist()
    return str(obj)


class LlmCall:
    """One request in flight. Poll `done`; then `ok` / `result` / `error`."""

    def __init__(self, call_id: str, task: str, timeout_s: float):
        self.id, self.task, self.timeout_s = call_id, task, timeout_s
        self.sent_at = time.monotonic()
        self.response: Optional[dict] = None
        self.error = ''

    @property
    def done(self) -> bool:
        return self.response is not None or bool(self.error)

    @property
    def ok(self) -> bool:
        return self.response is not None and bool(self.response.get('ok'))

    @property
    def result(self) -> dict:
        return (self.response or {}).get('result') or {}

    def failure(self) -> str:
        return self.error or str((self.response or {}).get('error', 'unknown error'))


class LlmLink:
    STATUS_MAX_AGE_S = 5.0            # the node publishes status every second

    def __init__(self, node, enabled: bool = True, default_timeout_s: float = 120.0):
        self.node = node
        self.enabled = enabled
        self.default_timeout_s = default_timeout_s
        self.status: Optional[dict] = None
        self.status_at = 0.0
        self.calls = {}
        self._n = 0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = node.create_publisher(String, REQUEST_TOPIC, QoSProfile(depth=50))
        node.create_subscription(String, RESPONSE_TOPIC, self._on_response, QoSProfile(depth=50))
        node.create_subscription(String, STATUS_TOPIC, self._on_status, latched)

    @property
    def ready(self) -> bool:
        """The AI is switched on and the llm_client node is up."""
        return (self.enabled and self.status is not None and bool(self.status.get('ready'))
                and time.monotonic() - self.status_at <= self.STATUS_MAX_AGE_S)

    def request(self, task: str, timeout_s: Optional[float] = None, track: bool = True, **fields) -> LlmCall:
        """Send a request. With track=False the reply is ignored (fire and forget, e.g. debug images)."""
        self._n += 1
        call = LlmCall(f'{self.node.get_name()}-{self._n}', task, timeout_s or self.default_timeout_s)
        if not self.ready:
            call.error = 'AI disabled' if not self.enabled else 'llm_client is not running (no /llm/status)'
            return call
        self._pub.publish(String(data=json.dumps({'id': call.id, 'task': task, **fields}, default=_json_default)))
        if track:
            self.calls[call.id] = call
        return call

    def poll(self):
        """Time out calls that never got a reply. Call this from the node's main tick."""
        now = time.monotonic()
        for call in list(self.calls.values()):
            if call.done:
                self.calls.pop(call.id, None)
            elif now - call.sent_at > call.timeout_s:
                call.error = f'no reply from llm_client within {call.timeout_s:g} s'
                self.calls.pop(call.id, None)

    def _on_response(self, msg: String):
        try:
            reply = json.loads(msg.data)
        except ValueError:
            return
        call = self.calls.pop(reply.get('id'), None)
        if call is not None:
            call.response = reply

    def _on_status(self, msg: String):
        try:
            self.status = json.loads(msg.data)
            self.status_at = time.monotonic()
        except ValueError:
            pass
