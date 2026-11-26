"""Request queue policy for the VLM broker (pure Python, unit-tested).

Rules:
  - one waiting request per source: a newer request from the same source replaces the older one
    (the replaced one is answered with error "replaced", so the sender is never left waiting)
  - highest priority first, then oldest
  - requests older than their max_age_s are dropped (answered with error "expired")
"""

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

_counter = itertools.count()


@dataclass
class Request:
    id: str
    source: str
    camera: str = "front"           # front | down
    prompt: str = ""                # name of a file in prompts/ (without .txt)
    prompt_text: str = ""           # or the prompt itself
    priority: int = 5               # higher first; console = 10, background nodes = 5
    max_age_s: float = 30.0
    snapshot_only: bool = False     # just save the camera frames, no VLM call
    t_req: float = 0.0
    seq: int = field(default_factory=lambda: next(_counter))

    @classmethod
    def from_dict(cls, d: dict, now: float) -> "Request":
        return cls(id=str(d.get("id", "")), source=str(d.get("source", "unknown")),
                   camera=str(d.get("camera", "front")), prompt=str(d.get("prompt", "")),
                   prompt_text=str(d.get("prompt_text", "")), priority=int(d.get("priority", 5)),
                   max_age_s=float(d.get("max_age_s", 30.0)),
                   snapshot_only=bool(d.get("snapshot_only", False)), t_req=now)


class RequestQueue:
    def __init__(self):
        self.items: List[Request] = []

    def __len__(self):
        return len(self.items)

    def push(self, req: Request) -> Optional[Request]:
        """Add a request; returns the request it replaced (same source), if any."""
        replaced = None
        for i, r in enumerate(self.items):
            if r.source == req.source:
                replaced = self.items.pop(i)
                break
        self.items.append(req)
        return replaced

    def pop(self, now: float) -> Tuple[Optional[Request], List[Request]]:
        """Return (next request or None, expired requests that were dropped)."""
        expired = [r for r in self.items if now - r.t_req > r.max_age_s]
        self.items = [r for r in self.items if r not in expired]
        if not self.items:
            return None, expired
        best = max(self.items, key=lambda r: (r.priority, -r.seq))
        self.items.remove(best)
        return best, expired

    def summary(self):
        return [{"source": r.source, "prompt": r.prompt or "custom", "camera": r.camera,
                 "priority": r.priority} for r in sorted(self.items, key=lambda r: (-r.priority, r.seq))]
