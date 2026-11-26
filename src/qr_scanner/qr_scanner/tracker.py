"""Shared QR code tracking across all cameras (pure Python, unit-tested).

A code is identified by its text. The same text seen by several cameras, or again later,
is the same code - unless map positions are known for both and they are further apart than
same_code_dist_m (then it is a second physical code with the same text).
A code is reported ("confirmed") once it has been read in confirm_frames frames within
confirm_window_s, from any camera(s).
"""

from typing import List, Optional

import numpy as np


class TrackedCode:
    def __init__(self, text: str, t: float):
        self.text = text
        self.first_seen = t
        self.last_seen = t
        self.frames = 0
        self.cameras = set()
        self.positions: List[list] = []     # map-frame points
        self.confirmed = False
        self.report = {}

    def mean_position(self) -> Optional[list]:
        if not self.positions:
            return None
        return np.median(np.array(self.positions), axis=0).tolist()


class CodeTracker:
    def __init__(self, confirm_frames: int = 2, confirm_window_s: float = 3.0,
                 same_code_dist_m: float = 1.5, max_positions: int = 30):
        self.confirm_frames = confirm_frames
        self.confirm_window_s = confirm_window_s
        self.same_code_dist_m = same_code_dist_m
        self.max_positions = max_positions
        self.codes: List[TrackedCode] = []
        self.sequence: List[str] = []       # distinct texts in the order they were confirmed

    def reset(self):
        self.codes, self.sequence = [], []

    def _match(self, text, pos, t) -> TrackedCode:
        for c in self.codes:
            if c.text != text:
                continue
            mp = c.mean_position()
            if pos is None or mp is None or np.linalg.norm(np.array(mp) - np.array(pos)) < self.same_code_dist_m:
                return c
        c = TrackedCode(text, t)
        self.codes.append(c)
        return c

    def observe(self, text: str, camera: str, t: float, map_point: Optional[list] = None):
        """Record one read. Returns (code, newly_confirmed: bool)."""
        c = self._match(text, map_point, t)
        if t - c.last_seen > self.confirm_window_s:
            c.frames = 0
        c.frames += 1
        c.last_seen = t
        c.cameras.add(camera)
        if map_point is not None:
            c.positions.append(list(map_point))
            c.positions = c.positions[-self.max_positions:]
        if not c.confirmed and c.frames >= self.confirm_frames:
            c.confirmed = True
            if c.text not in self.sequence:
                self.sequence.append(c.text)
            return c, True
        return c, False
