"""Optional frontier exploration helpers (pure numpy, no ROS).

Used only by the `frontier` exploration strategy (mission/exploration.py) and the `no_frontiers_left` end
condition. A frontier cell is a known-free cell next to unknown space; cells are grouped with 8-connected BFS
and each cluster gets a goal point clear of walls. Clusters are ranked by cost = distance_weight * distance
- size_weight * size. (Ported from lite3_kitchen_explorer.frontier_core.)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Iterable, List, Optional, Tuple

import numpy as np

from lite3_explorer.map_grid import (GridInfo, _shift, free_mask, grid_from_msg_data, unmapped_edge_cells,  # noqa: F401
                                     wall_clearance_m)

_NEIGHBOURS_8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def detect_frontier_cells(grid: np.ndarray, free_threshold: int = 25) -> np.ndarray:
    return unmapped_edge_cells(grid, free_threshold)


def occupied_mask(grid: np.ndarray, occupied_threshold: int) -> np.ndarray:
    return grid >= occupied_threshold


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


@dataclass
class FrontierCluster:
    cells: List[Tuple[int, int]]
    centroid_xy: Tuple[float, float]
    goal_xy: Tuple[float, float]
    size_m: float
    distance_m: float = 0.0
    cost: float = 0.0
    blacklisted: bool = False
    extra: dict = field(default_factory=dict)










def cluster_frontiers(frontier: np.ndarray) -> List[List[Tuple[int, int]]]:
    """8-connected BFS connected components over frontier cells."""
    visited = np.zeros_like(frontier, dtype=bool)
    h, w = frontier.shape
    clusters: List[List[Tuple[int, int]]] = []
    for r, c in zip(*np.nonzero(frontier)):
        if visited[r, c]:
            continue
        queue = deque([(int(r), int(c))])
        visited[r, c] = True
        cells = []
        while queue:
            cr, cc = queue.popleft()
            cells.append((cr, cc))
            for dr, dc in _NEIGHBOURS_8:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and frontier[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        clusters.append(cells)
    return clusters


def build_clusters(grid: np.ndarray, info: GridInfo, *, free_threshold: int = 25,
                   occupied_threshold: int = 65, min_cluster_cells: int = 8,
                   clearance_m: float = 0.30) -> List[FrontierCluster]:
    frontier = detect_frontier_cells(grid, free_threshold)
    if not frontier.any():
        return []
    clearance_cells = int(math.ceil(clearance_m / info.resolution)) if clearance_m > 0 else 0
    near_obstacle = dilate(occupied_mask(grid, occupied_threshold), clearance_cells)

    result = []
    for cells in cluster_frontiers(frontier):
        if len(cells) < min_cluster_cells:
            continue
        arr = np.asarray(cells, dtype=float)
        mean_r, mean_c = arr.mean(axis=0)
        # Goal = frontier cell with obstacle clearance closest to the centroid.
        safe = [cell for cell in cells if not near_obstacle[cell]]
        candidates = np.asarray(safe if safe else cells, dtype=float)
        idx = int(np.argmin((candidates[:, 0] - mean_r) ** 2 + (candidates[:, 1] - mean_c) ** 2))
        goal_r, goal_c = candidates[idx]
        goal_xy = info.cell_to_world(goal_r, goal_c)
        result.append(FrontierCluster(
            cells=cells,
            centroid_xy=info.cell_to_world(mean_r, mean_c),
            goal_xy=goal_xy,
            size_m=len(cells) * info.resolution,
            extra={'has_clearance': bool(safe),
                   'clearance_m': wall_clearance_m(grid, info, goal_xy, occupied_threshold)},
        ))
    return result




def is_blacklisted(xy: Tuple[float, float], blacklist: Iterable[Tuple[float, float]], radius_m: float) -> bool:
    return any(math.hypot(xy[0] - bx, xy[1] - by) <= radius_m for bx, by in blacklist)


def select_frontier(clusters: List[FrontierCluster], robot_xy: Tuple[float, float], *,
                    distance_weight: float = 1.0, size_weight: float = 0.5,
                    min_goal_distance_m: float = 0.5,
                    blacklist: Iterable[Tuple[float, float]] = (),
                    blacklist_radius_m: float = 0.5) -> Optional[FrontierCluster]:
    """Score every cluster (cost = w_d * distance - w_s * size) and return the cheapest usable one."""
    blacklist = list(blacklist)
    best = None
    for cluster in clusters:
        cluster.distance_m = math.hypot(cluster.goal_xy[0] - robot_xy[0], cluster.goal_xy[1] - robot_xy[1])
        cluster.cost = distance_weight * cluster.distance_m - size_weight * cluster.size_m
        cluster.blacklisted = is_blacklisted(cluster.goal_xy, blacklist, blacklist_radius_m)
        cluster.extra['usable'] = not cluster.blacklisted and cluster.distance_m >= min_goal_distance_m
        if not cluster.extra['usable']:
            continue
        if best is None or cluster.cost < best.cost:
            best = cluster
    return best


def rank_candidates(clusters: List[FrontierCluster], limit: int = 8) -> List[dict]:
    """Usable clusters (after select_frontier scored them) as numbered candidates, best cost first."""
    usable = sorted((c for c in clusters if not c.blacklisted and c.extra.get('usable', True)),
                    key=lambda c: c.cost)[:limit]
    r = lambda v, n: round(float(v), n)  # noqa: E731  (plain floats for JSON)
    return [{'id': i + 1, 'goal_xy': [r(c.goal_xy[0], 3), r(c.goal_xy[1], 3)],
             'centroid_xy': [r(c.centroid_xy[0], 3), r(c.centroid_xy[1], 3)],
             'distance_m': r(c.distance_m, 2), 'size_m': r(c.size_m, 2),
             'clearance_m': r(c.extra.get('clearance_m', 0.0), 2), 'cost': r(c.cost, 3)}
            for i, c in enumerate(usable)]


def frontier_near(grid: np.ndarray, info: GridInfo, xy: Tuple[float, float], radius_m: float,
                  free_threshold: int = 25) -> bool:
    """True if any frontier cell remains within radius of xy (goal still worth visiting)."""
    r, c = info.world_to_cell(*xy)
    rad = max(1, int(math.ceil(radius_m / info.resolution)))
    r0, r1 = max(0, r - rad - 1), min(info.height, r + rad + 2)
    c0, c1 = max(0, c - rad - 1), min(info.width, c + rad + 2)
    if r0 >= r1 or c0 >= c1:
        return False
    # Pad by one so neighbours of the window edge are evaluated correctly.
    pr0, pr1 = max(0, r0 - 1), min(info.height, r1 + 1)
    pc0, pc1 = max(0, c0 - 1), min(info.width, c1 + 1)
    window = detect_frontier_cells(grid[pr0:pr1, pc0:pc1], free_threshold)
    window = window[r0 - pr0:r0 - pr0 + (r1 - r0), c0 - pc0:c0 - pc0 + (c1 - c0)]
    rows, cols = np.nonzero(window)
    if rows.size == 0:
        return False
    d2 = (rows + r0 - r) ** 2 + (cols + c0 - c) ** 2
    return bool((d2 <= rad * rad).any())
