"""Which way out of a costmap obstacle zone (no ROS).

Inside an inflated wall Nav2's planner cannot plan, so every goal fails. The mppi mode (mission/modes.py) nudges the
robot out with x / y velocity instead: escape_direction() looks along a ring of directions in the costmap and returns the
one that reaches clear floor soonest without crossing a lethal cell.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

LETHAL = 100


def cost_at(grid: np.ndarray, info, x: float, y: float) -> Optional[int]:
    """Costmap value at a world point (unknown -1 counts as free), or None outside the grid."""
    r, c = info.world_to_cell(x, y)
    if not info.contains(r, c):
        return None
    return max(0, int(grid[r, c]))


def escape_direction(grid: np.ndarray, info, x: float, y: float, *, clear_cost: int = 70, max_m: float = 1.0,
                     step_m: float = 0.025, n_dirs: int = 24) -> Optional[Tuple[float, float, float]]:
    """(ux, uy, distance_m): the unit direction (map frame) that reaches a cell cheaper than clear_cost at the shortest
    distance, never passing a lethal cell on the way; None when no direction does within max_m."""
    best = None
    for k in range(n_dirs):
        a = 2 * math.pi * k / n_dirs
        ux, uy = math.cos(a), math.sin(a)
        d = step_m
        while d <= max_m + 1e-9:
            v = cost_at(grid, info, x + ux * d, y + uy * d)
            if v is None or v >= LETHAL:
                break                                   # off the map or a wall: this way is shut
            if v < clear_cost:
                if best is None or d < best[2]:
                    best = (ux, uy, d)
                break
            d += step_m
    return best
