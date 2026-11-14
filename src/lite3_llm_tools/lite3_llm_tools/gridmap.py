"""Occupancy-grid helpers for rendering the SLAM (lidar) map for the AI (no ROS).

Deliberately small and independent of the explorer packages, so the AI tools can be reused with any maze
explorer that publishes a nav_msgs/OccupancyGrid.
"""
from dataclasses import dataclass
import math
from typing import Sequence, Tuple

import numpy as np


@dataclass
class GridInfo:
    """Geometry of a nav_msgs/OccupancyGrid (origin yaw assumed 0, as SLAM toolbox publishes)."""
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


def dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Binary dilation with a disc of the given radius (pure numpy)."""
    if radius_cells <= 0:
        return mask.copy()
    out = mask.copy()
    r2 = radius_cells * radius_cells
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            if (dr or dc) and dr * dr + dc * dc <= r2:
                out |= _shift(mask, dr, dc)
    return out
