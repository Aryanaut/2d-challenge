"""Stuck detection (no ROS).

Two independent triggers, each switchable in stuck.triggers:
  no_motion  the robot has been driving for window_s and has not moved min_motion_m in that window
  costmap    the robot's centre sits on a global-costmap cell with cost >= costmap_cost for costmap_hold_s
             (inside an inflated wall Nav2's planner cannot plan out, so every goal fails)
"""
from __future__ import annotations

from collections import deque
import math
from typing import Optional, Sequence


class StuckDetector:
    def __init__(self, triggers: Sequence[str] = ('no_motion', 'costmap'), window_s: float = 20.0,
                 min_motion_m: float = 0.15, costmap_cost: int = 99, costmap_hold_s: float = 3.0):
        self.triggers = set(triggers)
        unknown = self.triggers - {'no_motion', 'costmap'}
        if unknown:
            raise ValueError(f'unknown stuck trigger(s): {sorted(unknown)}')
        self.window_s = window_s
        self.min_motion_m = min_motion_m
        self.costmap_cost = costmap_cost
        self.costmap_hold_s = costmap_hold_s
        self.reset()

    def reset(self):
        self.history = deque()          # (t, x, y) while driving
        self.driving_since = None
        self.in_cost_since = None
        self.trigger = None             # which trigger fired last: 'costmap' | 'no_motion'

    def update(self, now: float, pose: Optional[Sequence[float]], driving: bool,
               cost: Optional[int] = None, costmap_hold_s: Optional[float] = None) -> Optional[str]:
        """Feed one sample; returns a reason string when stuck, else None. costmap_hold_s overrides the configured hold
        for this sample (per-mode patience, e.g. shorter while backtracking)."""
        reason = None
        hold = self.costmap_hold_s if costmap_hold_s is None else costmap_hold_s
        if 'costmap' in self.triggers:
            if cost is not None and cost >= self.costmap_cost:
                if self.in_cost_since is None:
                    self.in_cost_since = now
                elif now - self.in_cost_since >= hold:
                    reason = f'in the global costmap obstacle zone (cost {cost}) for {now - self.in_cost_since:.0f} s'
                    self.trigger = 'costmap'
            else:
                self.in_cost_since = None
        if 'no_motion' in self.triggers:
            if not driving or pose is None:
                self.history.clear()
                self.driving_since = None
            else:
                if self.driving_since is None:
                    self.driving_since = now
                self.history.append((now, pose[0], pose[1]))
                while self.history and now - self.history[0][0] > self.window_s:
                    self.history.popleft()
                if reason is None and now - self.driving_since >= self.window_s and self.history:
                    x0, y0 = self.history[0][1], self.history[0][2]
                    moved = max(math.hypot(x - x0, y - y0) for _, x, y in self.history)
                    if moved < self.min_motion_m:
                        reason = f'moved only {moved:.2f} m in {self.window_s:.0f} s while driving'
                        self.trigger = 'no_motion'
        return reason
