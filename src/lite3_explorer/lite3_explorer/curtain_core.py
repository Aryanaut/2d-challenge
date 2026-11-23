"""Curtain detection from the depth camera (no ROS), and where a curtain sits on a tile's edge.

A curtain hangs from a bar and stops above the floor, so the 2D lidar sees a wall while the robot can walk under it.
The depth image shows what a wall never does: reading a column from the top down it is a near, almost vertical
surface, then the depth JUMPS FARTHER at the hem (the floor behind the curtain), then shrinks again as the floor comes
towards the camera:

    row   0..20   far background above the curtain bar
    row  40..220  curtain   0.63 0.65 ... 0.76 0.78 m   (slightly slanted, never jumps)
    row 240       hem       0.96 m                      <- jump of +0.2 m or more
    row 260..460  floor     0.85 0.75 ... 0.36 m        (getting closer)

No camera height, tilt or lidar calibration is needed. Curtain points are projected into the map frame and remembered
in a CurtainMap; edge_curtain() says whether they lie along one of a tile's four edges (mission/modes.py then aligns
with that edge and walks through). Detection idea taken from the explorer_alt package of the backup workspace.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

@dataclass
class CurtainParams:
    stride: int = 4                 # use every n-th pixel (rows and columns)
    run: int = 6                    # rows (after the stride) of near, smooth surface above the hem
    max_step_m: float = 0.03        # largest depth change between neighbouring rows on the curtain
    min_jump_m: float = 0.12        # the depth behind the hem is at least this much farther
    min_range_m: float = 0.30
    max_range_m: float = 3.50
    min_columns: float = 0.12       # fraction of the columns that must show the curtain
    range_band_m: float = 0.30      # columns count together when their ranges differ by less than this


@dataclass
class Detection:
    columns: np.ndarray             # image columns (full resolution) that show the curtain
    rows: np.ndarray                # hem row (full resolution) of each of those columns
    ranges: np.ndarray              # depth of the curtain at the hem, metres


def detect_curtain(depth_m: np.ndarray, p: CurtainParams = CurtainParams()) -> Optional[Detection]:
    """Curtain columns in a depth image (metres, 0 / nan = no reading), or None."""
    d = np.nan_to_num(np.asarray(depth_m, np.float32)[::p.stride, ::p.stride], nan=0.0)
    h, w = d.shape
    if h < p.run + 8:
        return None
    cols, rows, rngs = [], [], []
    for c in range(w):
        col = d[:, c]
        for r in range(p.run, h - 6):
            above = col[r - p.run:r]
            if (above <= 0).any():
                continue
            dc = float(above[-1])
            if not p.min_range_m < dc < p.max_range_m or np.abs(np.diff(above)).max() > p.max_step_m:
                continue
            below = col[r + 1:r + 6]
            below = below[below > 0]
            if len(below) < 3:
                continue
            med = float(np.median(below))
            # farther behind the hem, then (floor) coming back towards the camera
            if below.min() > dc + 0.5 * p.min_jump_m and med > dc + p.min_jump_m and col[r + 6] < med:
                cols.append(c * p.stride)
                rows.append(r * p.stride)
                rngs.append(dc)
                break
    if not cols:
        return None
    cols, rows, rngs = np.array(cols), np.array(rows), np.array(rngs)
    keep = np.abs(rngs - np.median(rngs)) <= p.range_band_m
    if keep.sum() < p.min_columns * w:
        return None
    return Detection(cols[keep], rows[keep], rngs[keep])


def project_to_map_3d(det: Detection, intrinsics: Tuple[float, float, float, float],
                      rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Detection points as Nx3 map-frame (x, y, z). intrinsics = fx, fy, cx, cy of the depth image;
    rotation (3x3) / translation take optical-frame points (x right, y down, z forward) to the map."""
    fx, fy, cx, cy = intrinsics
    z = det.ranges
    pts = np.stack([(det.columns - cx) * z / fx, (det.rows - cy) * z / fy, z], axis=1)
    return pts @ np.asarray(rotation, float).T + np.asarray(translation, float)


def project_to_map(det: Detection, intrinsics: Tuple[float, float, float, float],
                   rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Curtain points as Nx2 map-frame (x, y) (see project_to_map_3d)."""
    return project_to_map_3d(det, intrinsics, rotation, translation)[:, :2]


def detect_wall_top(depth_m: np.ndarray, p: CurtainParams = CurtainParams()) -> Optional[Detection]:
    """A near, smooth, vertical surface whose TOP edge is in view (the depth jumps farther, or is empty, above it).

    `rows` is the top-edge row of each column. The caller projects the edge into the map frame and keeps it when its
    height is ~curtain_top_height_m: curtains hang from a bar about 1 m above the floor, so a 'wall' that ends there
    is a curtain (with the 2D lidar it is indistinguishable from a wall)."""
    d = np.nan_to_num(np.asarray(depth_m, np.float32)[::p.stride, ::p.stride], nan=0.0)
    h, w = d.shape
    if h < p.run + 8:
        return None
    cols, rows, rngs = [], [], []
    for c in range(w):
        col = d[:, c]
        for r in range(2, h - p.run - 1):
            below = col[r:r + p.run]
            if (below <= 0).any():
                continue
            dc = float(below[0])
            if not p.min_range_m < dc < p.max_range_m or np.abs(np.diff(below)).max() > p.max_step_m:
                continue
            above = col[max(0, r - 3):r]
            above = above[above > 0]
            if len(above) and float(np.median(above)) < dc + p.min_jump_m:
                continue                      # no jump to something farther above the surface: not its top edge
            cols.append(c * p.stride)
            rows.append(r * p.stride)
            rngs.append(dc)
            break
    if not cols:
        return None
    cols, rows, rngs = np.array(cols), np.array(rows), np.array(rngs)
    keep = np.abs(rngs - np.median(rngs)) <= p.range_band_m
    if keep.sum() < p.min_columns * w:
        return None
    return Detection(cols[keep], rows[keep], rngs[keep])


class CurtainMap:
    """Curtain points seen so far, kept on a coarse grid; a spot counts once it was seen `confirm_hits` times."""

    def __init__(self, cell_m: float = 0.05, confirm_hits: int = 3):
        self.cell_m = cell_m
        self.confirm_hits = confirm_hits
        self._hits: Dict[Tuple[int, int], List[float]] = {}   # cell -> [count, x, y, dx, dy] (sums)

    def add(self, points_xy: np.ndarray, viewpoint_xy: Sequence[float]) -> bool:
        """Record curtain points seen from viewpoint_xy. True if a new spot became confirmed."""
        newly = False
        for x, y in np.asarray(points_xy, float).reshape(-1, 2):
            dx, dy = x - viewpoint_xy[0], y - viewpoint_xy[1]
            n = float(np.hypot(dx, dy)) or 1.0
            key = (int(np.floor(x / self.cell_m)), int(np.floor(y / self.cell_m)))
            e = self._hits.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            e[0] += 1
            e[1] += x
            e[2] += y
            e[3] += dx / n
            e[4] += dy / n
            newly |= e[0] == self.confirm_hits
        return newly

    def confirmed(self) -> np.ndarray:
        """Nx4 rows of x, y and the unit direction the robot was looking when it saw the spot (away from the robot)."""
        rows = []
        for n, sx, sy, sdx, sdy in self._hits.values():
            if n >= self.confirm_hits:
                u = float(np.hypot(sdx, sdy)) or 1.0
                rows.append((sx / n, sy / n, sdx / u, sdy / u))
        return np.array(rows, float).reshape(-1, 4)

    def to_json(self) -> dict:
        return {'points': [[round(float(v), 3) for v in r] for r in self.confirmed()]}

    @staticmethod
    def from_json(data: dict) -> np.ndarray:
        return np.array(data.get('points', []), float).reshape(-1, 4)

    def __len__(self):
        return len(self.confirmed())


# ------------------------------------------------------------------------------------------ from the lidar pointcloud
class LidarCurtainMap:
    """Curtain spots from a 3D lidar pointcloud (map frame), for when the depth camera cannot see the curtain bar.

    The standard walls are solid panels 0.5-0.6 m tall: the lidar sees returns up to that height on them. A hanging
    curtain gives almost only floor-level returns (the hem, below `low_max_m`) and nothing at wall height, while the
    real wall behind it is seen further on. A cell counts as curtain when, over all scans so far, it has at least
    `min_low` low returns, (nearly) none at or above `high_min_m`, and tall returns lie beyond it along the way the
    robot looked (the space behind the curtain). The 2D scan / occupancy map shows such a curtain as a wall."""

    def __init__(self, cell_m: float = 0.05, low_max_m: float = 0.20, high_min_m: float = 0.30,
                 max_high_frac: float = 0.03, min_low: int = 20, min_high_behind: int = 10,
                 behind_m: Tuple[float, float] = (0.30, 0.80), min_range_m: float = 0.50, max_range_m: float = 3.0,
                 floor_z: float = 0.0, near_m: float = 0.20):
        self.cell_m, self.low_max, self.high_min = cell_m, low_max_m, high_min_m
        self.max_high_frac, self.min_low, self.min_high_behind = max_high_frac, min_low, min_high_behind
        self.near_cells = max(1, int(round(near_m / cell_m)))
        self.behind_m, self.min_range, self.max_range, self.floor_z = behind_m, min_range_m, max_range_m, floor_z
        # cell -> [low, high, sum x of low, sum y of low, sum dir x, sum dir y]
        self._cells: Dict[Tuple[int, int], List[float]] = {}
        self._cache: Optional[np.ndarray] = None

    def add(self, points_xyz: np.ndarray, viewpoint_xy: Sequence[float]) -> None:
        """Record one scan (Nx3 map-frame points) seen from viewpoint_xy."""
        p = np.asarray(points_xyz, float).reshape(-1, 3)
        d = p[:, :2] - np.asarray(viewpoint_xy, float)
        r = np.hypot(d[:, 0], d[:, 1])
        keep = (r >= self.min_range) & (r <= self.max_range)
        p, d, r = p[keep], d[keep], r[keep]
        if len(p) == 0:
            return
        z = p[:, 2] - self.floor_z
        low, high = z <= self.low_max, z >= self.high_min
        use = low | high
        p, d, r, low, high = p[use], d[use], r[use], low[use], high[use]
        ij = np.floor(p[:, :2] / self.cell_m).astype(np.int64)
        keys, inv = np.unique(ij, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        agg = np.zeros((len(keys), 6))
        np.add.at(agg[:, 0], inv, low)
        np.add.at(agg[:, 1], inv, high)
        np.add.at(agg[:, 2], inv, p[:, 0] * low)
        np.add.at(agg[:, 3], inv, p[:, 1] * low)
        np.add.at(agg[:, 4], inv, d[:, 0] / r * low)
        np.add.at(agg[:, 5], inv, d[:, 1] / r * low)
        for k, row in zip(map(tuple, keys.tolist()), agg):
            e = self._cells.setdefault(k, [0.0] * 6)
            for i in range(6):
                e[i] += row[i]
        self._cache = None

    def confirmed(self) -> np.ndarray:
        """Nx4 rows of x, y and the unit direction the robot looked (away from it), like CurtainMap.confirmed()."""
        if self._cache is not None:
            return self._cache
        step = self.cell_m
        high = {k: e[1] for k, e in self._cells.items() if e[1] > 0}
        lows = {k: e[0] for k, e in self._cells.items() if e[0] > 0}
        r = self.near_cells
        rows = []
        for k, (lo, hi, sx, sy, dx, dy) in self._cells.items():
            if lo < self.min_low:
                continue
            # a wall's low returns come with wall-height ones in the same few cells (its lower part is simply seen more
            # often, and a tilted / drifting scan smears it by a cell or two): judge the neighbourhood, not the cell
            near = [(k[0] + i, k[1] + j) for i in range(-r, r + 1) for j in range(-r, r + 1)]
            if sum(high.get(c, 0.0) for c in near) > self.max_high_frac * sum(lows.get(c, 0.0) for c in near):
                continue
            n = float(np.hypot(dx, dy))
            if n < 1e-9:
                continue
            ux, uy = dx / n, dy / n
            cx, cy = (k[0] + 0.5) * step, (k[1] + 0.5) * step
            behind = 0.0
            for t in np.arange(self.behind_m[0], self.behind_m[1] + 1e-9, step):
                behind += high.get((int(np.floor((cx + ux * t) / step)), int(np.floor((cy + uy * t) / step))), 0.0)
            if behind >= self.min_high_behind:
                rows.append((sx / lo, sy / lo, ux, uy))
        self._cache = np.array(rows, float).reshape(-1, 4)
        return self._cache

    def to_json(self) -> dict:
        return {'points': [[round(float(v), 3) for v in r] for r in self.confirmed()]}

    def __len__(self):
        return len(self.confirmed())


# ------------------------------------------------------------------------------------------ on a tile's edge
SIDE_STEPS = {'E': (1, 0), 'W': (-1, 0), 'N': (0, 1), 'S': (0, -1)}


def edge_curtain(grid, tile, points, visited=(), skip=(), *, tol_m: float = 0.30, min_span_m: float = 0.40,
                 min_points: int = 6, min_facing: float = 0.3) -> Optional[dict]:
    """The curtain lying along one of `tile`'s four edges, or None.

    grid: a tiles.TileGrid; points: Nx4 (x, y, ux, uy) from CurtainMap.confirmed(). A side counts when at least
    min_points curtain spots lie within tol_m of that edge line (and within its length), spread over min_span_m of it,
    seen from inside this tile (their viewing direction points out through the edge). Sides whose neighbour tile was
    visited, or is in `skip`, are ignored: there is nothing new behind them. The side with the most spots wins.
    Returns {side, neighbour, centre_xy, normal, yaw, through_m, spots} (normal / yaw point out through the curtain,
    through_m is how far to drive from the edge line to the neighbour's centre) or None."""
    pts = np.asarray(points, float).reshape(-1, 4)
    if len(pts) == 0:
        return None
    size = grid.size
    u, v = grid.to_local(pts[:, 0], pts[:, 1])
    u, v = np.asarray(u, float) / size, np.asarray(v, float) / size
    c, s = math.cos(grid.yaw), math.sin(grid.yaw)
    visited, skip = set(map(tuple, visited)), set(map(tuple, skip))
    tol = tol_m / size
    best = None
    for side, (di, dj) in SIDE_STEPS.items():
        neighbour = (tile[0] + di, tile[1] + dj)
        if neighbour in visited or neighbour in skip:
            continue
        if di:                                         # E / W edge: u constant, v along it
            across, along = u - (tile[0] + (1 if di > 0 else 0)), v - tile[1]
        else:                                          # N / S edge: v constant, u along it
            across, along = v - (tile[1] + (1 if dj > 0 else 0)), u - tile[0]
        normal = (c * di - s * dj, s * di + c * dj)    # world direction out through the edge
        facing = pts[:, 2] * normal[0] + pts[:, 3] * normal[1]
        on = (np.abs(across) <= tol) & (along >= -0.1) & (along <= 1.1) & (facing >= min_facing)
        if on.sum() < min_points:
            continue
        span = float(along[on].max() - along[on].min()) * size
        if span < min_span_m:
            continue
        if best is None or on.sum() > best['spots']:
            best = dict(edge_crossing(grid, tile, side), spots=int(on.sum()), span_m=round(span, 2))
    return best


def edge_crossing(grid, tile, side: str) -> dict:
    """How to walk from `tile` through its `side` edge into the neighbour (the curtain mode's `edge`): {side,
    neighbour, centre_xy, normal, yaw, through_m, spots, span_m}; normal / yaw point out through the edge."""
    di, dj = SIDE_STEPS[side]
    size = grid.size
    c, s = math.cos(grid.yaw), math.sin(grid.yaw)
    normal = (c * di - s * dj, s * di + c * dj)
    mid_u = tile[0] + (0.5 if dj else (1.0 if di > 0 else 0.0))        # the middle of that edge, in tile units
    mid_v = tile[1] + (0.5 if di else (1.0 if dj > 0 else 0.0))
    cx, cy = grid.to_world(mid_u * size, mid_v * size)
    return {'side': side, 'neighbour': (tile[0] + di, tile[1] + dj), 'centre_xy': (cx, cy), 'normal': normal,
            'yaw': math.atan2(normal[1], normal[0]), 'through_m': 0.5 * size, 'spots': 0, 'span_m': size}
