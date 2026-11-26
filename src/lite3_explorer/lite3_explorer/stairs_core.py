"""Staircase detection from the depth camera (no ROS), and which tiles a staircase stands on.

The 2D lidar sees the risers of a staircase as a wall. The depth camera sees what a wall never shows: HORIZONTAL
surfaces (treads) stacked at evenly spaced heights, each one a little farther away than the one below it. Side view
(forward distance vs height above the floor) of the metal staircase in bags/data/stairs:

    h 0.45 |                 ___________  landing
    h 0.30 |           _____|
    h 0.15 |     _____|                    tread depth ~0.25 m, rise ~0.15 m
    h 0.00 |____|                          floor
           0.4   0.7   1.0   1.3   m ahead

detect_stairs() works on points in a frame whose z axis points up (the map frame via TF): a height map of 5 cm cells,
the cells whose heights hardly spread are flat, flat cells of about the same height that touch are one LEVEL, and a
staircase is a chain of at least min_levels levels where each rises min_rise_m..max_rise_m above the previous one and
lies farther away from the camera. A wall or curtain has no raised flat level, a box or table only one. No camera
height or floor height is needed: the floor, when it is in view, is simply the lowest level of the chain.

Detected stair cells are remembered in a StairMap; stair_tiles() says which tiles hold enough of them, and
stair_crossing() which side of the current tile the staircase can be climbed from (mission/modes.py, 'stairs' mode).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lite3_explorer.curtain_core import SIDE_STEPS, edge_crossing


@dataclass
class StairParams:
    stride: int = 4                 # use every n-th depth pixel (rows and columns)
    min_range_m: float = 0.25       # horizontal distance from the camera
    max_range_m: float = 2.5
    cell_m: float = 0.05            # height-map cell
    min_cell_points: int = 3
    flat_std_m: float = 0.015       # a cell is flat (tread / floor) when its heights spread less than this
    level_gap_m: float = 0.03       # neighbouring flat cells closer in height than this are one level ...
    link_cells: int = 2             # ... 'neighbouring' = at most this many cells apart (bridges reflection holes)
    max_level_span_m: float = 0.08  # a level thicker than this is a slope, not a tread
    min_level_cells: int = 10       # flat cells a level needs (10 x 25 cm^2)
    min_rise_m: float = 0.07        # height of one step
    max_rise_m: float = 0.25
    min_run_m: float = 0.10         # each level lies at least this much farther away than the one below ...
    max_gap_m: float = 0.35         # ... and starts no farther than this past where the one below ends
    min_levels: int = 3             # floor + two treads, or three treads


@dataclass
class Level:
    height: float
    xy: np.ndarray                  # Mx2 cell centres
    near: float                     # 10th / 50th / 90th percentile of the cells' distance from the camera
    mid: float
    far: float

    @property
    def centroid(self):
        return self.xy.mean(axis=0)


@dataclass
class StairDetection:
    levels: List[Level]             # the chain, lowest first
    xy: np.ndarray                  # Nx2 stair cells (every level of the chain above the lowest)
    ascent: Tuple[float, float]     # unit vector, horizontal, in the direction the steps go up
    rise_m: float                   # median step height

    @property
    def steps(self) -> int:
        return len(self.levels) - 1


def depth_to_points(depth_m: np.ndarray, intrinsics: Tuple[float, float, float, float], stride: int = 4) -> np.ndarray:
    """Nx3 optical-frame points (x right, y down, z forward) from a depth image in metres (0 / nan = no reading)."""
    fx, fy, cx, cy = intrinsics
    d = np.nan_to_num(np.asarray(depth_m, np.float32)[::stride, ::stride], nan=0.0)
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    u, v = u * stride, v * stride
    ok = d > 0
    z = d[ok]
    return np.stack([(u[ok] - cx) * z / fx, (v[ok] - cy) * z / fy, z], axis=1)


def _levels(cells_xy, cells_z, cell_ij, view_xy, p: StairParams) -> List[Level]:
    """Flat cells -> levels by region growing: a cell joins a neighbour (at most link_cells away) whose height differs
    by less than level_gap_m. A slightly tilted tread (the body pitches) still grows into one level; the riser's jump
    keeps two treads apart. Levels thicker than max_level_span_m (ramps, slopes) are dropped."""
    index = {tuple(ij): k for k, ij in enumerate(cell_ij)}
    seen = set()
    out = []
    for start in range(len(cells_z)):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            k = stack.pop()
            comp.append(k)
            ci, cj = cell_ij[k]
            for di in range(-p.link_cells, p.link_cells + 1):
                for dj in range(-p.link_cells, p.link_cells + 1):
                    n = index.get((ci + di, cj + dj))
                    if n is not None and n not in seen and abs(cells_z[n] - cells_z[k]) < p.level_gap_m:
                        seen.add(n)
                        stack.append(n)
        if len(comp) < p.min_level_cells:
            continue
        z = cells_z[comp]
        if np.percentile(z, 95) - np.percentile(z, 5) > p.max_level_span_m:
            continue
        xy = cells_xy[comp]
        dist = np.hypot(xy[:, 0] - view_xy[0], xy[:, 1] - view_xy[1])
        near, mid, far = np.percentile(dist, [10, 50, 90])
        out.append(Level(float(np.median(z)), xy, float(near), float(mid), float(far)))
    return sorted(out, key=lambda lv: lv.height)


def _longest_chain(levels: List[Level], p: StairParams) -> List[int]:
    """Longest chain of levels (lowest first) where each one is a step up and farther away than the previous."""
    n = len(levels)
    best = [[i] for i in range(n)]
    for j in range(n):
        for i in range(j):
            a, b = levels[i], levels[j]
            rise = b.height - a.height
            if not p.min_rise_m <= rise <= p.max_rise_m:
                continue
            # near edges, not medians: the floor wraps around the sides of a staircase, so its median is far away
            if b.near - a.near < p.min_run_m or b.near - a.far > p.max_gap_m:
                continue
            if len(best[i]) + 1 > len(best[j]):
                best[j] = best[i] + [j]
    return max(best, key=len) if best else []


def find_levels(points: np.ndarray, view_xy: Sequence[float], p: StairParams = StairParams()) -> List[Level]:
    """The flat levels (floor, treads, table tops ...) in Nx3 points of a z-up frame seen from view_xy, lowest first."""
    pts = np.asarray(points, float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return []
    dist = np.hypot(pts[:, 0] - view_xy[0], pts[:, 1] - view_xy[1])
    pts = pts[(dist >= p.min_range_m) & (dist <= p.max_range_m)]
    if len(pts) < p.min_cell_points * p.min_level_cells:
        return []
    ij = np.floor(pts[:, :2] / p.cell_m).astype(np.int64)
    keys, inv, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    mean = np.bincount(inv, pts[:, 2]) / counts
    std = np.sqrt(np.maximum(np.bincount(inv, pts[:, 2] ** 2) / counts - mean ** 2, 0.0))
    flat = (counts >= p.min_cell_points) & (std <= p.flat_std_m)
    if flat.sum() < p.min_level_cells:
        return []
    return _levels((keys[flat] + 0.5) * p.cell_m, mean[flat], keys[flat], view_xy, p)


def chain_stairs(levels: List[Level], p: StairParams = StairParams(),
                 floor_z: Optional[float] = None) -> Optional[StairDetection]:
    """A staircase among the levels, or None. floor_z (FloorEstimate): the height of the floor the robot stands on. Close
    to a staircase the floor is often out of view (or only seen beside the steps), so it is added as a level of its own
    that is nearer than everything: then two treads above the floor are enough (a box on the floor is still only two
    levels)."""
    levels = list(levels)
    if floor_z is not None:
        levels.append(Level(float(floor_z), np.zeros((0, 2)), 0.0, 0.0, math.inf))   # nearer than, and touching, all
        levels.sort(key=lambda lv: lv.height)
    chain = _longest_chain(levels, p)
    if len(chain) < p.min_levels:
        return None
    steps = [levels[i] for i in chain]
    rise = float(np.median(np.diff([lv.height for lv in steps])))
    return StairDetection(steps, np.vstack([lv.xy for lv in steps[1:]]), _ascent(steps), rise)


def _ascent(steps: List[Level]) -> Tuple[float, float]:
    """The way the steps go up: perpendicular to the treads' long axis (their cells' principal axis, averaged over the
    treads by doubled angle so the axis sign does not matter), pointing from the lower treads to the higher ones. The
    centroids alone are skewed by what the camera's field of view cuts off each tread."""
    real = [lv for lv in steps if len(lv.xy) >= 3]
    s = 0j
    for lv in real:
        w, v = np.linalg.eigh(np.cov((lv.xy - lv.xy.mean(axis=0)).T))
        elong = (w[1] - w[0]) / (w[1] + 1e-12)                     # 0 = round blob (no axis), 1 = a line
        s += len(lv.xy) * elong * np.exp(2j * math.atan2(v[1, 1], v[0, 1]))
    rough = real[-1].centroid - real[0].centroid
    if abs(s) < 1e-9:
        n = rough
    else:
        ang = np.angle(s) / 2 + math.pi / 2
        n = np.array([math.cos(ang), math.sin(ang)])
        if float(n @ rough) < 0:
            n = -n
    norm = float(np.hypot(*n)) or 1.0
    return float(n[0] / norm), float(n[1] / norm)


def detect_stairs(points: np.ndarray, view_xy: Sequence[float], p: StairParams = StairParams(),
                  floor_z: Optional[float] = None) -> Optional[StairDetection]:
    """A staircase in Nx3 points of a z-up frame seen from view_xy (the camera's x, y in that frame), or None."""
    return chain_stairs(find_levels(points, view_xy, p), p, floor_z)


class FloorEstimate:
    """Height of the floor the robot stands on, in the detection frame: the lowest big level that starts right in front
    of the camera, over the last `window` frames, taken at a low quantile. Standing at the foot of a staircase the first
    tread can be the nearest level, but a tread is always ABOVE the floor, so the low quantile ignores those frames.
    (The detector node also stops feeding it while the robot climbs in AI mode.)"""

    def __init__(self, near_m: float = 0.6, min_cells: int = 25, window: int = 40, quantile: float = 0.2):
        self.near_m, self.min_cells, self.window, self.quantile = near_m, min_cells, window, quantile
        self._seen: List[float] = []

    def update(self, levels: List[Level]) -> Optional[float]:
        near = [lv for lv in levels if lv.near <= self.near_m and len(lv.xy) >= self.min_cells]
        if near:
            self._seen = (self._seen + [min(lv.height for lv in near)])[-self.window:]
        return self.value

    @property
    def value(self) -> Optional[float]:
        return float(np.quantile(self._seen, self.quantile)) if len(self._seen) >= 3 else None


class StairMap:
    """Stair cells seen so far, on a coarse grid; a cell counts once it was seen `confirm_hits` times. Each cell keeps
    the mean ascent direction it was seen with (the stairs go up that way)."""

    def __init__(self, cell_m: float = 0.10, confirm_hits: int = 3):
        self.cell_m = cell_m
        self.confirm_hits = confirm_hits
        self._hits: Dict[Tuple[int, int], List[float]] = {}   # cell -> [count, x, y, ux, uy] (sums)
        self._frames: Dict[Tuple[int, int], int] = {}        # cell -> last frame it was counted in

    def add(self, det: StairDetection, frame: int = 0) -> bool:
        """Record one detection (`frame` counts each cell once per frame). True if a new cell became confirmed."""
        newly = False
        ux, uy = det.ascent
        for x, y in np.asarray(det.xy, float).reshape(-1, 2):
            key = (int(np.floor(x / self.cell_m)), int(np.floor(y / self.cell_m)))
            if self._frames.get(key) == frame:
                continue
            self._frames[key] = frame
            e = self._hits.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            e[0] += 1
            e[1] += x
            e[2] += y
            e[3] += ux
            e[4] += uy
            newly |= e[0] == self.confirm_hits
        return newly

    def confirmed(self) -> np.ndarray:
        """Nx4 rows of x, y and the unit ascent direction (the way the steps go up)."""
        rows = []
        for n, sx, sy, sux, suy in self._hits.values():
            if n >= self.confirm_hits:
                u = float(np.hypot(sux, suy)) or 1.0
                rows.append((sx / n, sy / n, sux / u, suy / u))
        return np.array(rows, float).reshape(-1, 4)

    def to_json(self) -> dict:
        return {'points': [[round(float(v), 3) for v in r] for r in self.confirmed()]}

    @staticmethod
    def from_json(data: dict) -> np.ndarray:
        return np.array(data.get('points', []), float).reshape(-1, 4)

    def __len__(self):
        return len(self.confirmed())


# ------------------------------------------------------------------------------------------ on the tile grid
def stair_tiles(grid, points, min_points: int = 4, edge_margin_m: float = 0.10, traversable=(),
                shift_max_m: float = 0.35, min_facing: float = 0.5) -> Dict[tuple, dict]:
    """Tiles holding at least min_points confirmed stair cells: {tile: {'points': n, 'ascent': (ux, uy)}}. Cells within
    edge_margin_m of a tile's edge do not count: the tile grid is snapped to the walls only to a few cm, and a
    staircase filling its tile would otherwise spill a row of cells into the tile beside it.

    `traversable` tiles (walked on, or scanned open floor) are never staircases: a staircase only stands at the edge of
    a traversable tile. A stair cell inside one is the foot of the steps seen across that edge (a grid that is off the
    walls puts it there): it is given to the neighbour beyond the edge the steps climb through (ascent within
    acos(min_facing) of the edge's outward normal), if it lies within shift_max_m of that edge; else it is dropped.
    Bag curtaintest_25_11_135748: 13 of 23 cells 0-0.26 m inside open tile 0,2, ascent out of its W edge -> the
    staircase is tile -1,2."""
    pts = np.asarray(points, float).reshape(-1, 4)
    if grid is None or len(pts) == 0:
        return {}
    traversable = set(map(tuple, traversable))
    c, s = math.cos(grid.yaw), math.sin(grid.yaw)
    acc: Dict[tuple, List[float]] = {}
    for x, y, ux, uy in pts:
        tile = tuple(grid.tile_of(x, y))
        if tile in traversable:
            u, v = grid.to_local(x, y)
            fu, fv = float(u) / grid.size - tile[0], float(v) / grid.size - tile[1]   # 0..1 inside the tile
            best = None
            for side, (di, dj) in SIDE_STEPS.items():
                facing = ux * (c * di - s * dj) + uy * (s * di + c * dj)          # ascent . outward normal
                dist = grid.size * ((1.0 - fu) if di > 0 else fu if di < 0 else (1.0 - fv) if dj > 0 else fv)
                if facing >= min_facing and dist <= shift_max_m and (best is None or facing > best[0]):
                    best = (facing, (tile[0] + di, tile[1] + dj))
            if best is None or best[1] in traversable:
                continue
            tile = best[1]
        elif grid.edge_distance(x, y) < edge_margin_m:
            continue
        e = acc.setdefault(tile, [0, 0.0, 0.0])
        e[0] += 1
        e[1] += ux
        e[2] += uy
    out = {}
    for tile, (n, sux, suy) in acc.items():
        if n >= min_points:
            u = math.hypot(sux, suy) or 1.0
            out[tile] = {'points': int(n), 'ascent': (sux / u, suy / u)}
    return out


def stair_crossing(grid, tile, stairs: Dict[tuple, dict], skip=(), min_facing: float = 0.5) -> Optional[dict]:
    """A staircase tile sharing a side with `tile` that can be climbed from it, or None.

    stairs: stair_tiles(). A side counts when the steps go up away from `tile` (the ascent direction points out through
    that side, cos >= min_facing: from beside the staircase its stringer is in the way). Tiles in `skip` (crossed or
    given up on) are ignored. Returns curtain_core.edge_crossing() for that side plus 'stairs' (the staircase tiles
    in a row along the way), 'beyond' (the first tile after them, where the robot stops) and 'through_m' (from the
    edge line to the centre of 'beyond')."""
    skip = set(map(tuple, skip))
    c, s = math.cos(grid.yaw), math.sin(grid.yaw)
    best = None
    for side, (di, dj) in SIDE_STEPS.items():
        first = (tile[0] + di, tile[1] + dj)
        if first not in stairs or first in skip:
            continue
        normal = (c * di - s * dj, s * di + c * dj)
        ux, uy = stairs[first]['ascent']
        facing = ux * normal[0] + uy * normal[1]
        if facing < min_facing:
            continue
        row, nxt = [first], (first[0] + di, first[1] + dj)
        while nxt in stairs and nxt not in skip:          # a staircase deeper than one tile: cross all of it
            row.append(nxt)
            nxt = (nxt[0] + di, nxt[1] + dj)
        if best is None or stairs[first]['points'] > best['spots']:
            e = edge_crossing(grid, tile, side)
            e.update(spots=stairs[first]['points'], facing=round(facing, 2), stairs=row, beyond=nxt,
                     through_m=(len(row) + 0.5) * grid.size)
            best = e
    return best


def beyond_reference(grid, row: Sequence[tuple], stairs: Dict[tuple, dict],
                     fallback_dir: Optional[Sequence[float]] = None) -> dict:
    """The tile beyond the staircase, in the staircase's own direction: the reference odometry is rebased to after a climb.

    The tile grid may sit a few degrees off the staircase, so the grid tile beyond it (stair_crossing()'s 'beyond') is
    not where a climb along the steps ends. This tile is not on the grid: it is a size x size square turned to the
    ascent direction (the stair detector's, averaged over the crossed staircase tiles `row`, weighted by their points;
    fallback_dir, e.g. the crossing's edge normal, when there is none), its centre one tile past the staircase along
    it: from the middle of the staircase tiles, (len(row) / 2 + 0.5) tiles up the ascent.
    Returns {'centre_xy', 'yaw', 'ascent', 'corners' (sw, se, ne, nw: back-right, front-right, front-left, back-left),
    'size', 'stairs' (row), 'grid_tile' (the grid tile holding the centre)}."""
    row = [tuple(t) for t in row]
    size = grid.size
    sx = sy = 0.0
    for t in row:
        d = stairs.get(t)
        if d:
            sx += d['points'] * d['ascent'][0]
            sy += d['points'] * d['ascent'][1]
    n = math.hypot(sx, sy)
    if n < 1e-9:
        if fallback_dir is None:
            raise ValueError('beyond_reference: no ascent direction for the staircase and no fallback')
        sx, sy = fallback_dir
        n = math.hypot(sx, sy)
    ux, uy = sx / n, sy / n
    centres = [grid.center(t) for t in row]
    mx = sum(c[0] for c in centres) / len(centres)
    my = sum(c[1] for c in centres) / len(centres)
    ahead = (len(row) / 2.0 + 0.5) * size
    cx, cy = mx + ux * ahead, my + uy * ahead
    h = size / 2.0
    corners = [(cx + ux * a - uy * b, cy + uy * a + ux * b) for a, b in ((-h, -h), (h, -h), (h, h), (-h, h))]
    return {'centre_xy': (cx, cy), 'yaw': math.atan2(uy, ux), 'ascent': (ux, uy), 'corners': corners, 'size': size,
            'stairs': row, 'grid_tile': grid.tile_of(cx, cy)}
