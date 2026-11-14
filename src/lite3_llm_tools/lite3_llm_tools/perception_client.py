"""Client for the LAN perception server (no ROS imports).

Sends exactly the same request as the reference client.py:
    POST {server}/analyze   multipart form
        group_id = <group>
        prompt   = <mission prompt text>
        image    = (<name>, <bytes>, <mime>)
and then pulls the four-key mission JSON out of whatever the server wraps it in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
import time
from typing import Any, List, Optional

REQUIRED_KEYS = ('scene_description', 'relevant_to_mission', 'evidence', 'confidence')
_WRAPPER_KEYS = ('result', 'response', 'output', 'answer', 'content', 'text', 'message',
                 'data', 'analysis', 'json', 'choices', 'output_text')


class PerceptionError(Exception):
    """Request failed or the reply did not contain a usable mission JSON."""

    def __init__(self, message: str, retry_after_s: Optional[float] = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


@dataclass
class MissionResult:
    scene_description: str
    relevant_to_mission: bool
    evidence: List[str]
    confidence: float
    raw: Any = None
    http_status: int = 0
    elapsed_s: float = 0.0
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in REQUIRED_KEYS}


def read_prompt(path: str) -> str:
    # utf-8-sig also accepts the optional BOM written by some Windows editors.
    with open(path, encoding='utf-8-sig') as fh:
        prompt = fh.read()
    if not prompt.strip():
        raise ValueError(f'Prompt file {path} is blank.')
    return prompt


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        return value.strip().lower() == 'true'
    raise ValueError(f'relevant_to_mission is not a boolean: {value!r}')


def _to_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError('confidence is a boolean')
    conf = float(value)
    if not math.isfinite(conf):
        raise ValueError('confidence is not finite')
    return min(1.0, max(0.0, conf))


def _from_dict(obj: dict) -> MissionResult:
    evidence = obj['evidence']
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        raise ValueError('evidence is not a list')
    return MissionResult(
        scene_description=str(obj['scene_description']),
        relevant_to_mission=_to_bool(obj['relevant_to_mission']),
        evidence=[str(e) for e in evidence][:3],
        confidence=_to_confidence(obj['confidence']),
        extras={k: v for k, v in obj.items() if k not in REQUIRED_KEYS},
    )


def _json_candidates(text: str):
    """Yield JSON objects found in free text (handles code fences and leading chatter)."""
    stripped = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.IGNORECASE)
    try:
        yield json.loads(stripped)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except ValueError:
            continue
        yield obj


def extract_mission_result(payload: Any, _depth: int = 0) -> MissionResult:
    """Find the {scene_description, relevant_to_mission, evidence, confidence} object."""
    if _depth > 6:
        raise ValueError('response nested too deeply')
    if isinstance(payload, dict):
        if all(k in payload for k in REQUIRED_KEYS):
            return _from_dict(payload)
        ordered = [payload[k] for k in _WRAPPER_KEYS if k in payload]
        ordered += [v for k, v in payload.items() if k not in _WRAPPER_KEYS]
        for value in ordered:
            if isinstance(value, (dict, list, str)):
                try:
                    return extract_mission_result(value, _depth + 1)
                except (ValueError, KeyError, TypeError):
                    continue
    elif isinstance(payload, list):
        for item in payload:
            try:
                return extract_mission_result(item, _depth + 1)
            except (ValueError, KeyError, TypeError):
                continue
    elif isinstance(payload, str):
        for obj in _json_candidates(payload):
            if isinstance(obj, (dict, list)):
                try:
                    return extract_mission_result(obj, _depth + 1)
                except (ValueError, KeyError, TypeError):
                    continue
    raise ValueError('no mission JSON with the four required keys found in server reply')


class PerceptionClient:
    def __init__(self, server: str, group_id: str, prompt: str,
                 read_timeout_s: float = 75.0, connect_timeout_s: float = 5.0):
        if not group_id:
            raise ValueError('group_id must be set')
        if not prompt.strip():
            raise ValueError('Prompt must not be blank.')
        if not math.isfinite(read_timeout_s) or read_timeout_s <= 0:
            raise ValueError('read timeout must be finite and positive')
        self.url = server.rstrip('/') + '/analyze'
        self.group_id = group_id
        self.prompt = prompt
        self.read_timeout_s = read_timeout_s
        self.connect_timeout_s = connect_timeout_s

    def analyze(self, image_bytes: bytes, filename: str = 'frame.jpg',
                mime: str = 'image/jpeg', prompt: Optional[str] = None) -> MissionResult:
        import httpx  # imported lazily so the parser can be used/tested without it

        started = time.monotonic()
        try:
            response = httpx.post(
                self.url,
                data={'group_id': self.group_id, 'prompt': prompt or self.prompt},
                files={'image': (filename, image_bytes, mime)},
                timeout=httpx.Timeout(self.read_timeout_s, connect=self.connect_timeout_s),
            )
        except httpx.HTTPError as error:
            raise PerceptionError(f'{type(error).__name__}: {error}') from error
        elapsed = time.monotonic() - started

        if response.is_error:
            retry_after = None
            if 'retry-after' in response.headers:
                try:
                    retry_after = float(response.headers['retry-after'])
                except ValueError:
                    retry_after = None
            raise PerceptionError(
                f'HTTP {response.status_code} {response.reason_phrase}: {response.text[:300]}',
                retry_after_s=retry_after)
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        try:
            result = extract_mission_result(payload)
        except ValueError as error:
            raise PerceptionError(f'{error}; reply was: {str(payload)[:300]}') from error
        result.raw = payload
        result.http_status = response.status_code
        result.elapsed_s = elapsed
        return result
