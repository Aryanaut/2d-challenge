"""Scene-understanding gateway client for surface classification.

Calls the course's own Lab 6 client.py (the one that already works for you) as a
subprocess, so the request format is exactly what the gateway expects:

    python client.py --server <url> --group <groupNN> --image <jpg> --prompt-file <txt>

and parses the JSON it prints:
    {"success": true, "group_id": "...", "result": "<model text>", "processing_time_s": ...}
The model text is expected to be JSON: {"surface": ..., "confidence": ..., "evidence": ...}.
"""

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

VALID = ("wood", "metal", "other", "unclear")
GROUP_RE = re.compile(r"^group(0[1-9]|1[0-6])$")

# Map loose model wording onto the four labels.
SYNONYMS = {
    "osb": "wood", "chipboard": "wood", "plywood": "wood", "timber": "wood", "wooden": "wood",
    "aluminium": "metal", "aluminum": "metal", "steel": "metal", "silver": "metal",
    "checker": "metal", "tread": "metal", "floor": "other", "vinyl": "other",
    "none": "other", "neither": "other", "unknown": "unclear", "uncertain": "unclear",
}


@dataclass
class VlmResult:
    ok: bool
    surface: str = "unclear"
    confidence: float = 0.0
    evidence: str = ""
    latency_s: float = 0.0
    error: str = ""
    raw: dict = field(default_factory=dict)


def _first_json_object(text: str) -> Optional[dict]:
    """Return the outermost {...} in text as a dict, or None."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def normalise_label(value) -> str:
    s = str(value or "").strip().lower()
    if s in VALID:
        return s
    for key, label in SYNONYMS.items():
        if key in s:
            return label
    return "unclear"


def parse_surface_result(result) -> tuple:
    """Parse the model's answer (dict or string, possibly in ```json fences) into
    (surface, confidence, evidence). Falls back to keyword matching if not JSON."""
    if isinstance(result, dict):
        data = result
    else:
        text = str(result)
        text = re.sub(r"```(?:json)?", "", text)
        data = _first_json_object(text)
        if data is None:  # free text: look for a label word
            return normalise_label(text), 0.3, text[:200]
    surface = normalise_label(data.get("surface") or data.get("label"))
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    evidence = data.get("evidence", "")
    if isinstance(evidence, list):
        evidence = "; ".join(map(str, evidence))
    return surface, conf, str(evidence)[:300]


def parse_client_output(text: str) -> VlmResult:
    """Parse everything client.py printed (status line, elapsed line, JSON, footer)."""
    payload = _first_json_object(text)
    if payload is None:
        return VlmResult(ok=False, error=f"no JSON in client output: {text.strip()[:200]}")
    if not payload.get("success", False):
        err = payload.get("error") or {}
        code = err.get("code", "unknown") if isinstance(err, dict) else str(err)
        msg = err.get("message", "") if isinstance(err, dict) else ""
        return VlmResult(ok=False, error=f"{code}: {msg}", raw=payload)
    surface, conf, evidence = parse_surface_result(payload.get("result", ""))
    return VlmResult(ok=True, surface=surface, confidence=conf, evidence=evidence, raw=payload)


class GatewayClient:
    def __init__(self, client_python: str, client_script: str, server: str, group: str,
                 prompt_file: str, timeout_s: float = 20.0, max_width: int = 640,
                 max_height: int = 480, jpeg_quality: int = 85):
        self.client_python = client_python
        self.client_script = client_script
        self.server = server
        self.group = group
        self.prompt_file = prompt_file
        self.timeout_s = timeout_s
        self.max_w = min(max_width, 1024)   # gateway rejects anything over 1024x768
        self.max_h = min(max_height, 768)
        self.jpeg_quality = jpeg_quality

    def config_problems(self) -> list:
        problems = []
        if not GROUP_RE.match(self.group):
            problems.append(f"group '{self.group}' is not group01..group16 (server answers invalid_group)")
        if not os.path.isfile(self.client_script):
            problems.append(f"client_script not found: {self.client_script}")
        if not os.path.isfile(self.prompt_file):
            problems.append(f"prompt_file not found: {self.prompt_file}")
        return problems

    def _fit(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        scale = min(self.max_w / w, self.max_h / h, 1.0)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return bgr

    def query(self, bgr: np.ndarray) -> VlmResult:
        t0 = time.monotonic()
        img = self._fit(bgr)
        fd, path = tempfile.mkstemp(suffix=".jpg", prefix="surface_")
        os.close(fd)
        try:
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            cmd = [self.client_python, self.client_script, "--server", self.server,
                   "--group", self.group, "--image", path, "--prompt-file", self.prompt_file]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
            res = parse_client_output(proc.stdout + "\n" + proc.stderr)
        except subprocess.TimeoutExpired:
            res = VlmResult(ok=False, error=f"timeout after {self.timeout_s:.0f} s")
        except OSError as e:
            res = VlmResult(ok=False, error=f"could not run client: {e}")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        res.latency_s = time.monotonic() - t0
        return res
