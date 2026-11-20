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
