"""Can the robot turn in place here? Checked against the local costmap, the way Nav2's Spin checks it (no ROS).

The dog's body is 0.61 x 0.37 m, so turning in place sweeps a circle of radius 0.36 m, and Nav2's Spin aborts
("Collision Ahead - Exiting Spin") as soon as the footprint touches an inscribed or lethal cell of the local costmap
(OccupancyGrid values 99 and 100). In a 1 m wide maze corridor that is nearly always the case, and an aborted turn
leaves the robot rotated across the corridor. Checking first avoids starting a turn that cannot finish.
"""
import math
from typing import List, Sequence, Tuple

import numpy as np

INSCRIBED = 99      # nav_msgs/OccupancyGrid value of Nav2's INSCRIBED_INFLATED_OBSTACLE (253); lethal is 100


def footprint_corners(x: float, y: float, yaw: float, length: float, width: float) -> List[Tuple[float, float]]:
    hl, hw = length / 2.0, width / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + fx * c - fy * s, y + fx * s + fy * c) for fx, fy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def footprint_cost(grid: np.ndarray, origin_xy: Sequence[float], resolution: float, x: float, y: float, yaw: float,
                   length: float, width: float, margin: float = 0.0) -> int:
    """Highest costmap value under the footprint outline (a cell outside the grid counts as free)."""
    pts = footprint_corners(x, y, yaw, length + 2 * margin, width + 2 * margin)
    worst = 0
    h, w = grid.shape
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        n = max(2, int(math.dist((ax, ay), (bx, by)) / resolution * 3))
        for k in range(n + 1):
            px, py = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
            r, c = int((py - origin_xy[1]) / resolution), int((px - origin_xy[0]) / resolution)
            if 0 <= r < h and 0 <= c < w:
                worst = max(worst, int(grid[r, c]))
    return worst


def blocked_orientations(grid: np.ndarray, origin_xy: Sequence[float], resolution: float, x: float, y: float,
                         length: float = 0.61, width: float = 0.37, margin: float = 0.0, block_value: int = INSCRIBED,
                         step_deg: int = 5) -> List[int]:
    """Body orientations (degrees, 0..355) at (x, y) in which the footprint touches a cell >= block_value."""
    return [a for a in range(0, 360, step_deg)
            if footprint_cost(grid, origin_xy, resolution, x, y, math.radians(a), length, width, margin) >= block_value]


def can_turn_in_place(grid: np.ndarray, origin_xy: Sequence[float], resolution: float, x: float, y: float,
                      length: float = 0.61, width: float = 0.37, margin: float = 0.0,
                      block_value: int = INSCRIBED) -> Tuple[bool, int]:
    """(True if a full turn is free of obstacles, number of blocked orientations out of 72)."""
    blocked = blocked_orientations(grid, origin_xy, resolution, x, y, length, width, margin, block_value)
    return not blocked, len(blocked)
