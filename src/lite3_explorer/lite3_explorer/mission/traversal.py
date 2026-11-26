"""Traversal strategies: HOW the robot drives to a goal (no ROS in this module).

A mode never talks to Nav2 directly; it asks the context for a traversal profile by name
(``ctx.traversal('route')``) and drives it through this small interface:

    available() -> bool                       the backend is up (False: the mode waits instead of failing the tile)
    start(goal_xy, yaw, route=None) -> bool   begin driving; route = [(x, y, yaw), ...] ending at the goal
    poll() -> TraversalResult                 RUNNING / SUCCEEDED / FAILED (+ reason)
    cancel()

Profiles are configured under ``traversals.<name>`` with a ``type`` naming a registered class. The Nav2
implementations (nav2_pose, nav2_route) live in nav2_traversal.py and register themselves on import.
To add e.g. a stair-climbing traversal: subclass Traversal, decorate it with @register_traversal('stairs'),
import it in the mission manager, add ``traversals.stairs: {type: stairs, ...}`` and point a mode (or a tile,
via modes.exploring.tile_traversals: ["3,1:stairs"]) at it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple, Type

RUNNING, SUCCEEDED, FAILED = 'running', 'succeeded', 'failed'
TRAVERSAL_TYPES: Dict[str, Type['Traversal']] = {}


def register_traversal(type_name: str):
    def deco(cls):
        cls.type_name = type_name
        TRAVERSAL_TYPES[type_name] = cls
        return cls
    return deco


@dataclass
class TraversalResult:
    state: str
    reason: str = ''

    @property
    def done(self) -> bool:
        return self.state != RUNNING


class Traversal:
    type_name = 'base'
    uses_route = False

    def __init__(self, ctx, name: str, cfg: dict):
        self.ctx, self.name, self.cfg = ctx, name, dict(cfg or {})

    def available(self) -> bool:
        return True

    def start(self, goal_xy: Sequence[float], yaw: float,
              route: Optional[Sequence[Tuple[float, float, float]]] = None) -> bool:
        raise NotImplementedError

    def poll(self) -> TraversalResult:
        raise NotImplementedError

    def cancel(self):
        raise NotImplementedError


def build_traversal(ctx, name: str, profiles: dict) -> Traversal:
    cfg = profiles.get(name)
    if cfg is None:
        raise KeyError(f'no traversal profile {name!r} (traversals: {sorted(profiles)})')
    ttype = cfg.get('type', name)
    if ttype not in TRAVERSAL_TYPES:
        raise KeyError(f'traversal profile {name!r} has unknown type {ttype!r} (known: {sorted(TRAVERSAL_TYPES)})')
    return TRAVERSAL_TYPES[ttype](ctx, name, cfg)


@register_traversal('omni')
class OmniTraversal(Traversal):
    """Holonomic straight line to the goal with linear x / y velocity (no Nav2): ctx.drive(vx, wz, vy).

    The heading the robot has at start() is held (wz P-controller); the robot slides towards the goal in its own frame.
    Speed ramps down near the goal (gain * distance, between min_speed_mps and speed_mps). Fails when a global-costmap
    cell of at least block_cost lies under the body footprint within lookahead_m ahead on the way (a wall, not mere inflation), on timeout_s,
    or when the robot has not got closer by min_progress_m within stall_s.

    Options (traversals.<name>.*): speed_mps, min_speed_mps, gain, tolerance_m, heading_kp, wz_max, block_cost,
    lookahead_m, body_length_m, body_width_m, body_pad_m, require_costmap, timeout_s, stall_s, min_progress_m."""

    def __init__(self, ctx, name, cfg):
        super().__init__(ctx, name, cfg)
        self.goal, self.heading, self.result = None, None, TraversalResult(FAILED, 'not started')

    def available(self):
        return hasattr(self.ctx, 'drive') and hasattr(self.ctx, 'stop')

    def _c(self, key, default):
        return float(self.cfg.get(key, default))

    def start(self, goal_xy, yaw, route=None):
        pose = self.ctx.pose()
        if pose is None:
            self.result = TraversalResult(FAILED, 'no pose')
            return False
        self.goal = (float(goal_xy[0]), float(goal_xy[1]))
        self.heading = pose[2] if len(pose) > 2 else 0.0
        self.started = self.ctx.now()
        self.best, self.best_at = math.dist(pose[:2], self.goal), self.started
        self.result = TraversalResult(RUNNING)
        return True

    def _footprint_hits(self, grid, info, x, y, yaw, block):
        """Set of costmap cells >= block under the body rectangle (body_length_m x body_width_m, padded) at (x, y, yaw)."""
        from lite3_explorer.costmap_escape import cost_at
        pad = self._c('body_pad_m', 0.03)
        hl = self._c('body_length_m', 0.61) / 2 + pad
        hw = self._c('body_width_m', 0.37) / 2 + pad
        step = max(info.resolution, 0.02)
        c, s = math.cos(yaw), math.sin(yaw)
        hits = set()
        nu, nv = int(2 * hl / step) + 1, int(2 * hw / step) + 1
        for i in range(nu + 1):
            u = -hl + 2 * hl * i / nu
            for j in range(nv + 1):
                v = -hw + 2 * hw * j / nv
                px, py = x + u * c - v * s, y + u * s + v * c
                cost = cost_at(grid, info, px, py)
                if cost is not None and cost >= block:
                    hits.add(info.world_to_cell(px, py))
        return hits

    def _blocked(self, pose, ux, uy):
        """True when the BODY (not just its centre) would touch a wall cell within lookahead_m along (ux, uy).

        The heading is held, so the rectangle keeps the starting yaw. Cells already under the body now are ignored
        (a recovery starts inside the obstacle zone; it must be able to slide out). A missing / stale global costmap
        blocks (fail closed) unless require_costmap is false."""
        snap = self.ctx.costmap_snapshot() if hasattr(self.ctx, 'costmap_snapshot') else None
        if snap is None:
            return bool(self.cfg.get('require_costmap', False)) and hasattr(self.ctx, 'costmap_snapshot')
        grid, info = snap
        block = self._c('block_cost', 100)
        ahead = min(self._c('lookahead_m', 0.30), math.dist(pose[:2], self.goal))
        steps = max(1, int(ahead / max(info.resolution, 0.02)))
        yaw = self.heading if self.heading is not None else (pose[2] if len(pose) > 2 else 0.0)
        already = self._footprint_hits(grid, info, pose[0], pose[1], yaw, block)
        for k in range(1, steps + 1):
            d = ahead * k / steps
            hits = self._footprint_hits(grid, info, pose[0] + ux * d, pose[1] + uy * d, yaw, block)
            if hits - already:
                return True
        return False

    def poll(self):
        if self.result.state != RUNNING:
            return self.result
        pose = self.ctx.pose()
        now = self.ctx.now()
        if pose is None:
            return self.result
        dx, dy = self.goal[0] - pose[0], self.goal[1] - pose[1]
        dist = math.hypot(dx, dy)
        if dist <= self._c('tolerance_m', 0.10):
            return self._end(SUCCEEDED)
        if now - self.started > self._c('timeout_s', 30.0):
            return self._end(FAILED, 'timed out')
        if dist < self.best - self._c('min_progress_m', 0.05):
            self.best, self.best_at = dist, now
        elif now - self.best_at > self._c('stall_s', 6.0):
            return self._end(FAILED, f'no progress for {self._c("stall_s", 6.0):.0f} s')
        ux, uy = dx / dist, dy / dist
        if self._blocked(pose, ux, uy):
            return self._end(FAILED, 'wall in the global costmap on the way')
        v = max(self._c('min_speed_mps', 0.06), min(self._c('speed_mps', 0.2), self._c('gain', 0.8) * dist))
        yaw = pose[2] if len(pose) > 2 else 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        err = math.atan2(math.sin(self.heading - yaw), math.cos(self.heading - yaw))
        wlim = self._c('wz_max', 0.3)
        wz = max(-wlim, min(wlim, self._c('heading_kp', 1.5) * err))
        self.ctx.drive(v * (ux * c + uy * s), wz, v * (-ux * s + uy * c))      # map -> robot frame
        return self.result

    def _end(self, state, reason=''):
        self.ctx.stop()
        self.result = TraversalResult(state, reason)
        return self.result

    def cancel(self):
        if self.result.state == RUNNING:
            self.ctx.stop()
            self.result = TraversalResult(FAILED, 'cancelled')
