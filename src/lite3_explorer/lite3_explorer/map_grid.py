"""Occupancy-grid helpers (pure numpy, no ROS).

Only map geometry lives here. Frontier clustering (for the optional frontier exploration profile) is in
frontiers.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np

UNKNOWN = -1
_NEIGHBOURS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class GridInfo:
    """Geometry of a nav_msgs/OccupancyGrid (origin yaw assumed 0, as slam_toolbox and Nav2 publish)."""
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float

    def cell_to_world(self, row: float, col: float) -> Tuple[float, float]:
        return (self.origin_x + (col + 0.5) * self.resolution,
                self.origin_y + (row + 0.5) * self.resolution)

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor((y - self.origin_y) / self.resolution)),
                int(math.floor((x - self.origin_x) / self.resolution)))

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width


def grid_from_msg_data(data: Sequence[int], width: int, height: int) -> np.ndarray:
    """OccupancyGrid.data is row-major, row 0 at the map origin."""
    return np.asarray(data, dtype=np.int16).reshape(height, width)


def _shift(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r, c] = mask[r + dr, c + dc] (False outside the grid)."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    r0, r1 = max(0, -dr), min(h, h - dr)
    c0, c1 = max(0, -dc), min(w, w - dc)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = mask[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def free_mask(grid: np.ndarray, free_threshold: int) -> np.ndarray:
    return (grid >= 0) & (grid <= free_threshold)


def unmapped_edge_cells(grid: np.ndarray, free_threshold: int = 25) -> np.ndarray:
    """Free cells next to unknown space. Used only to tell whether a tile still borders unmapped space
    (reported as a tile property), never to choose where to drive."""
    unknown = grid == UNKNOWN
    touches = np.zeros_like(unknown)
    for dr, dc in _NEIGHBOURS_4:
        touches |= _shift(unknown, dr, dc)
    return free_mask(grid, free_threshold) & touches


def wall_clearance_m(grid: np.ndarray, info: GridInfo, xy: Tuple[float, float],
                     occupied_threshold: int = 65, max_m: float = 2.0) -> float:
    """Distance from xy to the nearest occupied (wall) cell, capped at max_m."""
    r, c = info.world_to_cell(*xy)
    rad = max(1, int(math.ceil(max_m / info.resolution)))
    r0, r1 = max(0, r - rad), min(info.height, r + rad + 1)
    c0, c1 = max(0, c - rad), min(info.width, c + rad + 1)
    if r0 >= r1 or c0 >= c1:
        return max_m
    rows, cols = np.nonzero(grid[r0:r1, c0:c1] >= occupied_threshold)
    if rows.size == 0:
        return max_m
    d = np.sqrt((rows + r0 - r) ** 2 + (cols + c0 - c) ** 2).min() * info.resolution
    return float(min(max_m, d))


def value_at(grid: np.ndarray, info: GridInfo, x: float, y: float):
    """Cell value at a world point, or None outside the grid."""
    r, c = info.world_to_cell(x, y)
    return int(grid[r, c]) if info.contains(r, c) else None
