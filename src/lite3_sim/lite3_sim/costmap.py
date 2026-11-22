"""Occupancy maps, Nav2-style costmaps and a NavFn-like planner (no ROS).

All grids are numpy arrays indexed [row, col] with row = y, like nav_msgs/OccupancyGrid. Values follow the
OccupancyGrid conventions the mission manager reads: -1 unknown, 0 free, 100 occupied / lethal, 99 inscribed,
1..98 inflated cost (Nav2's cost translation table).
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:                                           # optional: fast Dijkstra in C
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as _sp_dijkstra
except ImportError:                            # pragma: no cover - depends on the machine
    csr_matrix = None
    _sp_dijkstra = None

LETHAL, INSCRIBED = 100, 99


@dataclass
class GridInfo:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor((y - self.origin_y) / self.resolution)),
                int(math.floor((x - self.origin_x) / self.resolution)))

    def cell_to_world(self, row: float, col: float) -> Tuple[float, float]:
        return (self.origin_x + (col + 0.5) * self.resolution, self.origin_y + (row + 0.5) * self.resolution)

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width


# --------------------------------------------------------------------------------------------- SLAM-like map
class OccupancyMapper:
    """Log-odds occupancy grid filled from 2D scans (stands in for slam_toolbox, with a perfect pose)."""

    def __init__(self, info: GridInfo, hit: float = 0.9, miss: float = 0.4, clamp=(-2.0, 3.5),
                 occupied_at: float = 0.5, free_at: float = -0.35):
        self.info = info
        self.logodds = np.zeros((info.height, info.width), np.float32)
        self.touched = np.zeros((info.height, info.width), bool)
        self.hit, self.miss, self.clamp = hit, miss, clamp
        self.occupied_at, self.free_at = occupied_at, free_at
        self.version = 0

    def _cells(self, xs, ys):
        info = self.info
        c = np.floor((xs - info.origin_x) / info.resolution).astype(np.int64)
        r = np.floor((ys - info.origin_y) / info.resolution).astype(np.int64)
        ok = (r >= 0) & (r < info.height) & (c >= 0) & (c < info.width)
        return np.unique(r[ok] * info.width + c[ok])

    def integrate(self, ox: float, oy: float, angles: np.ndarray, ranges: np.ndarray, hit: np.ndarray,
                  range_max: float):
        """angles in the map frame; ranges (inf / > range_max = nothing hit); hit = the beam ended on an obstacle."""
        res = self.info.resolution
        r = np.where(np.isfinite(ranges), np.minimum(ranges, range_max), range_max)
        steps = np.arange(0.0, range_max, res * 0.5)
        free = steps[None, :] < (r[:, None] - 0.6 * res)
        xs = ox + np.cos(angles)[:, None] * steps[None, :]
        ys = oy + np.sin(angles)[:, None] * steps[None, :]
        free_idx = self._cells(xs[free], ys[free])
        hx = ox + np.cos(angles[hit]) * (r[hit] + 0.25 * res)
        hy = oy + np.sin(angles[hit]) * (r[hit] + 0.25 * res)
        hit_idx = self._cells(hx, hy)
        flat_l, flat_t = self.logodds.reshape(-1), self.touched.reshape(-1)
        flat_l[free_idx] -= self.miss
        flat_l[hit_idx] += self.hit
        np.clip(flat_l, *self.clamp, out=flat_l)
        flat_t[free_idx] = True
        flat_t[hit_idx] = True
        self.version += 1

    def grid(self) -> np.ndarray:
        g = np.full(self.logodds.shape, -1, np.int16)
        g[self.touched & (self.logodds <= self.free_at)] = 0
        g[self.touched & (self.logodds >= self.occupied_at)] = 100
        return g


# --------------------------------------------------------------------------------------------- costmaps
def nav2_to_occ(cost: np.ndarray) -> np.ndarray:
    """Nav2 costs (0..252) -> OccupancyGrid values 0..98 (Nav2's cost translation table)."""
    out = np.zeros(cost.shape, np.int16)
    mid = (cost >= 1) & (cost <= 252)
    out[mid] = 1 + (97 * (cost[mid].astype(np.int32) - 1)) // 251
    return out


def inflate(lethal: np.ndarray, unknown: np.ndarray, res: float, inscribed_m: float, inflation_m: float,
            cost_scaling: float) -> np.ndarray:
    """Nav2 InflationLayer on a lethal mask -> OccupancyGrid-style costmap (-1, 0..98, 99, 100)."""
    h, w = lethal.shape
    reach = inflation_m                      # Nav2 inflates only out to inflation_radius, even when it is
    rad = int(math.ceil(reach / res))        # smaller than the inscribed radius (the local costmap here)
    dist = np.full((h, w), np.inf, np.float32)
    dist[lethal] = 0.0
    if rad > 0 and lethal.any():
        pad = np.pad(lethal, rad)
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                d = math.hypot(dr, dc) * res
                if d == 0.0 or d > reach + 1e-9:
                    continue
                shifted = pad[rad + dr:rad + dr + h, rad + dc:rad + dc + w]
                np.minimum(dist, np.where(shifted, d, np.inf), out=dist)
    out = np.zeros((h, w), np.int16)
    infl = np.isfinite(dist) & (dist > inscribed_m) & (dist <= inflation_m)
    if infl.any():
        c = np.floor(252.0 * np.exp(-cost_scaling * (dist[infl] - inscribed_m))).astype(np.int32)
        out[infl] = nav2_to_occ(np.clip(c, 0, 252))
    out[np.isfinite(dist) & (dist <= inscribed_m)] = INSCRIBED
    out[lethal] = LETHAL
    # Nav2 (inflate_unknown false): unknown cells only take inscribed / lethal costs
    out[unknown & (out < INSCRIBED)] = -1
    return out


def footprint_inscribed_radius(length: float, width: float, padding: float = 0.0) -> float:
    return min(length, width) / 2.0 + padding


# --------------------------------------------------------------------------------------------- planner
class GridPlanner:
    """Shortest path on a costmap, NavFn style: inscribed / lethal cells are walls, unknown cells are allowed (at a
    high cost), the robot's own cell is always free, and a goal in an obstacle is moved to the reachable cell
    nearest to it within `tolerance_m`."""

    def __init__(self, unknown_cost: float = 4.0, cost_weight: float = 0.8, tolerance_m: float = 0.4,
                 allow_unknown: bool = True):
        self.unknown_cost, self.cost_weight = unknown_cost, cost_weight
        self.tolerance_m, self.allow_unknown = tolerance_m, allow_unknown
        self.costmap, self.info, self.version = None, None, None
        self._weights = None
        self._graph = None
        self._cache = {}

    def set_costmap(self, costmap: np.ndarray, info: GridInfo, version):
        if version == self.version and self.costmap is not None:
            return
        self.costmap, self.info, self.version = costmap, info, version
        cm = costmap.astype(np.float32)
        nav = np.where(cm > 0, (cm - 1) * 251.0 / 97.0 + 1.0, 0.0)
        w = 1.0 + self.cost_weight * nav / 50.0
        w[costmap == -1] = self.unknown_cost if self.allow_unknown else np.inf
        w[costmap >= INSCRIBED] = np.inf
        self._weights = w
        self._graph = None
        self._cache = {}

    # -- graph
    def _base_graph(self):
        """Edges between passable 8-neighbours: (a, b, cost); built once per costmap."""
        if self._graph is None:
            h, w = self._weights.shape
            wt = self._weights.reshape(-1)
            ok = np.isfinite(wt)
            idx = np.arange(h * w).reshape(h, w)
            aa, bb, dd = [], [], []
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                a = idx[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)].reshape(-1)
                b = a + dr * w + dc
                m = ok[a] & ok[b]
                a, b = a[m], b[m]
                aa.append(a)
                bb.append(b)
                dd.append(math.hypot(dr, dc) * 0.5 * (wt[a] + wt[b]))
            self._graph = (np.concatenate(aa), np.concatenate(bb), np.concatenate(dd))
        return self._graph

    def _distances(self, start_idx):
        if start_idx in self._cache:
            return self._cache[start_idx]
        h, w = self._weights.shape
        n = h * w
        a, b, d = self._base_graph()
        wt = self._weights.reshape(-1)
        if not np.isfinite(wt[start_idx]):         # NavFn clears the robot's own cell: edges out of it
            r, c = divmod(start_idx, w)
            ea, eb, ed = [], [], []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr or dc) and 0 <= rr < h and 0 <= cc < w and np.isfinite(wt[rr * w + cc]):
                        ea.append(start_idx)
                        eb.append(rr * w + cc)
                        ed.append(math.hypot(dr, dc) * 0.5 * (1.0 + wt[rr * w + cc]))
            if ea:
                a, b, d = np.concatenate([a, ea]), np.concatenate([b, eb]), np.concatenate([d, ed])
        if _sp_dijkstra is not None:
            g = csr_matrix((d, (a, b)), shape=(n, n))
            dist, pred = _sp_dijkstra(g, directed=False, indices=start_idx, return_predecessors=True)
        else:
            dist, pred = _py_dijkstra(n, a, b, d, start_idx)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[start_idx] = (dist, pred)
        return dist, pred

    def plan(self, start_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        if self.costmap is None:
            return None
        info = self.info
        sr, sc = info.world_to_cell(*start_xy)
        gr, gc = info.world_to_cell(*goal_xy)
        if not info.contains(sr, sc):
            return None
        start = sr * info.width + sc
        dist, pred = self._distances(start)
        goal = None
        if info.contains(gr, gc) and np.isfinite(dist[gr * info.width + gc]):
            goal = gr * info.width + gc
        else:
            rad = int(self.tolerance_m / info.resolution)
            best = None
            for r in range(max(0, gr - rad), min(info.height, gr + rad + 1)):
                for c in range(max(0, gc - rad), min(info.width, gc + rad + 1)):
                    dd = math.hypot(r - gr, c - gc)
                    if dd <= rad and np.isfinite(dist[r * info.width + c]) and (best is None or dd < best[0]):
                        best = (dd, r * info.width + c)
            if best is None:
                return None
            goal = best[1]
        cells = [goal]
        while cells[-1] != start:
            p = int(pred[cells[-1]])
            if p < 0:
                return None
            cells.append(p)
        cells.reverse()
        pts = [info.cell_to_world(i // info.width, i % info.width) for i in cells]
        pts[0] = (float(start_xy[0]), float(start_xy[1]))
        if goal == gr * info.width + gc:
            pts[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        return smooth_path(pts)


def _py_dijkstra(n, a, b, d, start):
    adj = [[] for _ in range(n)]
    for u, v, w in zip(a.tolist(), b.tolist(), d.tolist()):
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = np.full(n, np.inf)
    pred = np.full(n, -9999, np.int64)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        du, u = heapq.heappop(heap)
        if du > dist[u]:
            continue
        for v, w in adj[u]:
            nd = du + w
            if nd < dist[v]:
                dist[v], pred[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return dist, pred


def smooth_path(pts: Sequence[Tuple[float, float]], window: int = 2) -> List[Tuple[float, float]]:
    """Moving-average smoothing of a cell path (end points kept): removes the 8-connected staircase."""
    if len(pts) < 3:
        return list(pts)
    arr = np.asarray(pts, float)
    out = arr.copy()
    for i in range(1, len(arr) - 1):
        lo, hi = max(0, i - window), min(len(arr), i + window + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return [tuple(p) for p in out]


def path_length(pts) -> float:
    return float(sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))) if pts else 0.0
