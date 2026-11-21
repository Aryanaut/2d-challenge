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


def project_to_map(det: Detection, intrinsics: Tuple[float, float, float, float],
                   rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Curtain points as Nx2 map-frame (x, y). intrinsics = fx, fy, cx, cy of the depth image;
    rotation (3x3) / translation take optical-frame points (x right, y down, z forward) to the map."""
    fx, fy, cx, cy = intrinsics
    z = det.ranges
    pts = np.stack([(det.columns - cx) * z / fx, (det.rows - cy) * z / fy, z], axis=1)
    out = pts @ np.asarray(rotation, float).T + np.asarray(translation, float)
    return out[:, :2]


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
            mid_u = tile[0] + (0.5 if dj else (1.0 if di > 0 else 0.0))        # the middle of that edge, in tile units
            mid_v = tile[1] + (0.5 if di else (1.0 if dj > 0 else 0.0))
            cx, cy = grid.to_world(mid_u * size, mid_v * size)
            best = {'side': side, 'neighbour': neighbour, 'centre_xy': (cx, cy), 'normal': normal,
                    'yaw': math.atan2(normal[1], normal[0]), 'through_m': 0.5 * size,
                    'spots': int(on.sum()), 'span_m': round(span, 2)}
    return best
