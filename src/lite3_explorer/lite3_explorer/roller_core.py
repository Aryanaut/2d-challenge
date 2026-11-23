"""Roller detection from the depth camera and the 3D lidar (no ROS), and which tiles hold a roller.

A roller tile has a horizontal cylinder (~0.15 m across, its top ~0.17 m above the floor) lying across the middle of
the tile in a wooden frame. The robot can only get over it in the h-step gait (rug_hstep). The 2D scan sees the
roller as a wall across the middle of the tile, so the tile looks like HALF a tile, with nothing mapped behind it.

Depth camera (detect_roller): side view of bags/data/roller (forward distance vs height above the floor, 5 cm columns,
camera ~0.30 m high):

    h 0.17 |             _.     <- top of the roller: a short, thin ridge
    h 0.08 |            |       front face of the roller / frame board
    h 0.00 |____________|       floor ... then NOTHING for >1 m: the floor behind is hidden by the roller
           0.30  0.50  0.70 m ahead

In a z-up frame (the map via TF), every point is binned into 5 cm cells. A roller is a cluster of cells whose
highest point is min_top_m..max_top_m above the floor (no wall-height point near them), that is long (min_length_m
along its axis) and thin (max_width_m across), with (almost) nothing visible above the floor just behind it. A wall
is too tall, a step / box shows its top surface behind the edge, a stair tread has the next tread behind it, a cable
is too low.

3D lidar (LidarLowMap): the roller's face returns only low points (low_m), never wall-height ones (high_min_m) in its
neighbourhood, and the space just behind it is open (few wall-height returns there). That is what makes the 2D scan's
"wall" across the half tile a roller and not a real wall. The lidar alone is not specific (any low clutter passes), so
it CONFIRMS the depth camera's roller spots: RollerMap.confirmed() counts lidar low-only cells near each spot.

Confirmed spots are remembered in a RollerMap; roller_tiles() says which tiles hold enough of them, roller_crossing()
from where the robot can cross one (mission/modes.py, 'roller' mode).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lite3_explorer.curtain_core import SIDE_STEPS, edge_crossing


@dataclass
class RollerParams:
    stride: int = 4                 # use every n-th depth pixel (rows and columns)
    min_range_m: float = 0.25       # horizontal distance from the camera
    max_range_m: float = 1.6        # farther away the depth is too sparse / noisy (seen again on the approach anyway)
    cell_m: float = 0.05
    min_cell_points: int = 3
    min_top_m: float = 0.12         # top of the roller above the floor (bag: 0.17 m) ...
    max_top_m: float = 0.30         # ... the standard walls are 0.5-0.6 m
    tall_m: float = 0.35            # a point this high in a cell or its 8 neighbours: a wall / box, not a roller
    link_cells: int = 2             # cells at most this many apart belong to the same roller
    min_cells: int = 6
    min_length_m: float = 0.40      # along the roller axis
    max_width_m: float = 0.25       # across it (the end brackets widen it a little)
    behind_m: Tuple[float, float] = (0.05, 0.40)   # band behind the far edge of the roller ...
    behind_min_h_m: float = 0.03    # ... where points higher than this above the floor ...
    max_behind_frac: float = 0.15   # ... number at most this fraction of the roller's points (occluded floor)
    min_points: int = 300           # points (after the stride) on the roller cells


@dataclass
class RollerDetection:
    xy: np.ndarray                  # Nx2 cell centres of the roller
    axis: Tuple[float, float]       # unit vector along the roller
    normal: Tuple[float, float]     # unit vector across it, pointing away from the camera (the way across)
    top_m: float                    # height of its top above the floor
    length_m: float
    width_m: float
    points: int

    @property
    def centre(self):
        return self.xy.mean(axis=0)


def detect_roller(points: np.ndarray, view_xy: Sequence[float], floor_z: float,
                  p: RollerParams = RollerParams()) -> Optional[RollerDetection]:
    """The roller in Nx3 points of a z-up frame seen from view_xy (the camera's x, y), floor at floor_z, or None.
    The longest roller-like cluster wins."""
    pts = np.asarray(points, float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    dist = np.hypot(pts[:, 0] - view_xy[0], pts[:, 1] - view_xy[1])
    pts = pts[(dist >= p.min_range_m) & (dist <= p.max_range_m)]
    if len(pts) < p.min_points:
        return None
    h = pts[:, 2] - floor_z
    ij = np.floor(pts[:, :2] / p.cell_m).astype(np.int64)
    keys, inv, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    hmax = np.full(len(keys), -np.inf)
    np.maximum.at(hmax, inv, h)
    tall = {tuple(k) for k, hm in zip(keys.tolist(), hmax) if hm > p.tall_m}
    cand = [i for i in np.flatnonzero((counts >= p.min_cell_points) & (hmax >= p.min_top_m) & (hmax <= p.max_top_m))
            if not any((keys[i][0] + a, keys[i][1] + b) in tall for a in (-1, 0, 1) for b in (-1, 0, 1))]
    index = {(int(keys[i][0]), int(keys[i][1])): i for i in cand}
    seen, best = set(), None
    view = np.asarray(view_xy, float)[:2]
    for start in cand:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:                                   # cells linked within link_cells
            k = stack.pop()
            comp.append(k)
            ci, cj = keys[k]
            for di in range(-p.link_cells, p.link_cells + 1):
                for dj in range(-p.link_cells, p.link_cells + 1):
                    n = index.get((int(ci + di), int(cj + dj)))
                    if n is not None and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(comp) < p.min_cells:
            continue
        det = _roller_of(comp, keys, counts, hmax, pts, h, view, p)
        if det is not None and (best is None or det.length_m > best.length_m):
            best = det
    return best


def _roller_of(comp, keys, counts, hmax, pts, h, view, p: RollerParams) -> Optional[RollerDetection]:
    xy = (keys[comp] + 0.5) * p.cell_m
    c = xy.mean(axis=0)
    _, vec = np.linalg.eigh(np.cov((xy - c).T))
    axis, normal = vec[:, 1], vec[:, 0]
    if float((c - view) @ normal) < 0:
        normal = -normal                               # across the roller, away from the camera
    along, across = (xy - c) @ axis, (xy - c) @ normal
    length = float(along.max() - along.min()) + p.cell_m
    width = float(np.percentile(across, 95) - np.percentile(across, 5)) + p.cell_m
    n_pts = int(counts[comp].sum())
    if length < p.min_length_m or width > p.max_width_m or n_pts < p.min_points:
        return None
    # just behind the roller the floor is hidden: (almost) nothing above the floor may be seen there, and nothing of wall
    # height right behind it (a ledge in front of a wall)
    q = pts[:, :2] - c
    pa, pn = q @ axis, q @ normal
    far = float(np.percentile(across, 95)) + p.cell_m / 2
    beside = (pa >= along.min()) & (pa <= along.max()) & (pn < far + p.behind_m[1])
    band = beside & (((pn > far + p.behind_m[0]) & (h > p.behind_min_h_m)) | ((pn > far) & (h > p.tall_m)))
    if band.sum() > p.max_behind_frac * n_pts:
        return None
    return RollerDetection(xy, (float(axis[0]), float(axis[1])), (float(normal[0]), float(normal[1])),
                           float(np.median(hmax[comp])), round(length, 3), round(width, 3), n_pts)


# ------------------------------------------------------------------------------------------ lidar
class LidarLowMap:
    """Cells where the 3D lidar (map frame) sees only LOW returns: at least `min_low` returns between low_m above the
    floor (summed over scans), (nearly) none at wall height (high_min_m) in the neighbourhood (near_m), and few
    wall-height returns behind the cell along the way the robot looked (max_high_behind over behind_m): the space
    behind a roller is open. A curtain hem (wall behind it) or a real wall does not pass."""

    def __init__(self, cell_m: float = 0.05, low_m: Tuple[float, float] = (0.04, 0.28), high_min_m: float = 0.32,
                 max_high_frac: float = 0.03, min_low: int = 20, behind_m: Tuple[float, float] = (0.30, 0.80),
                 max_high_behind: int = 10, min_range_m: float = 0.40, max_range_m: float = 2.5, near_m: float = 0.10):
        self.cell_m, self.low_m, self.high_min = cell_m, low_m, high_min_m
        self.max_high_frac, self.min_low = max_high_frac, min_low
        self.behind_m, self.max_high_behind = behind_m, max_high_behind
        self.min_range, self.max_range = min_range_m, max_range_m
        self.near_cells = max(1, int(round(near_m / cell_m)))
        self._cells: Dict[Tuple[int, int], List[float]] = {}   # cell -> [low, high, sum x, sum y, sum dir x, sum dir y]
        self._cache: Optional[np.ndarray] = None

    def add(self, points_xyz: np.ndarray, viewpoint_xy: Sequence[float], floor_z: float) -> None:
        """Record one scan (Nx3 map-frame points) seen from viewpoint_xy, with the floor at floor_z."""
        p = np.asarray(points_xyz, float).reshape(-1, 3)
        d = p[:, :2] - np.asarray(viewpoint_xy, float)[:2]
        r = np.hypot(d[:, 0], d[:, 1])
        keep = (r >= self.min_range) & (r <= self.max_range)
        p, d, r = p[keep], d[keep], r[keep]
        z = p[:, 2] - floor_z
        low, high = (z >= self.low_m[0]) & (z <= self.low_m[1]), z >= self.high_min
        use = low | high
        p, d, r, low, high = p[use], d[use], r[use], low[use], high[use]
        if len(p) == 0:
            return
        ij = np.floor(p[:, :2] / self.cell_m).astype(np.int64)
        keys, inv = np.unique(ij, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        agg = np.zeros((len(keys), 6))
        for col, val in enumerate((low, high, p[:, 0] * low, p[:, 1] * low, d[:, 0] / r * low, d[:, 1] / r * low)):
            np.add.at(agg[:, col], inv, val)
        for k, row in zip(map(tuple, keys.tolist()), agg):
            e = self._cells.setdefault(k, [0.0] * 6)
            for i in range(6):
                e[i] += row[i]
        self._cache = None

    def confirmed(self, near_xy: Optional[np.ndarray] = None, radius_m: float = 0.25) -> np.ndarray:
        """Nx2 x, y of the low-only cells; with near_xy only those within radius_m of one of those points (cheap: the
        whole map is not searched)."""
        if near_xy is None and self._cache is not None:
            return self._cache
        step, r = self.cell_m, self.near_cells
        high = {k: e[1] for k, e in self._cells.items() if e[1] > 0}
        lows = {k: e[0] for k, e in self._cells.items() if e[0] > 0}
        if near_xy is not None:
            near_xy = np.asarray(near_xy, float).reshape(-1, 2)
            reach = int(math.ceil(radius_m / step))
            keys = {(int(np.floor(x / step)) + i, int(np.floor(y / step)) + j)
                    for x, y in near_xy for i in range(-reach, reach + 1) for j in range(-reach, reach + 1)}
            cells = [(k, self._cells[k]) for k in keys if k in self._cells]
        else:
            cells = list(self._cells.items())
        rows = []
        for k, (lo, _, sx, sy, dx, dy) in cells:
            if lo < self.min_low:
                continue
            near = [(k[0] + i, k[1] + j) for i in range(-r, r + 1) for j in range(-r, r + 1)]
            if sum(high.get(c, 0.0) for c in near) > self.max_high_frac * sum(lows.get(c, 0.0) for c in near):
                continue
            n = math.hypot(dx, dy)
            if n < 1e-9:
                continue
            ux, uy = dx / n, dy / n
            cx, cy = (k[0] + 0.5) * step, (k[1] + 0.5) * step
            behind = sum(high.get((int(np.floor((cx + ux * t) / step)), int(np.floor((cy + uy * t) / step))), 0.0)
                         for t in np.arange(self.behind_m[0], self.behind_m[1] + 1e-9, step))
            if behind <= self.max_high_behind:
                rows.append((sx / lo, sy / lo))
        out = np.array(rows, float).reshape(-1, 2)
        if near_xy is None:
            self._cache = out
        return out

    def __len__(self):
        return len(self.confirmed())


# ------------------------------------------------------------------------------------------ remembered spots
class RollerMap:
    """Roller cells seen by the depth camera so far, on a coarse grid; a cell counts once it was seen in `confirm_hits`
    frames. Each cell keeps the mean way across the roller (sign-aligned: a roller seen from both sides keeps one
    direction; the mission picks the side to cross from)."""

    def __init__(self, cell_m: float = 0.10, confirm_hits: int = 3):
        self.cell_m = cell_m
        self.confirm_hits = confirm_hits
        self._hits: Dict[Tuple[int, int], List[float]] = {}   # cell -> [count, x, y, nx, ny] (sums)
        self._frames: Dict[Tuple[int, int], int] = {}

    def add(self, det: RollerDetection, frame: int = 0) -> bool:
        """Record one detection (`frame` counts each cell once per frame). True if a new cell became confirmed."""
        newly = False
        nx, ny = det.normal
        for x, y in np.asarray(det.xy, float).reshape(-1, 2):
            key = (int(np.floor(x / self.cell_m)), int(np.floor(y / self.cell_m)))
            if self._frames.get(key) == frame:
                continue
            self._frames[key] = frame
            e = self._hits.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            sign = -1.0 if e[3] * nx + e[4] * ny < 0 else 1.0
            e[0] += 1
            e[1] += x
            e[2] += y
            e[3] += sign * nx
            e[4] += sign * ny
            newly |= e[0] == self.confirm_hits
        return newly

    def confirmed(self, lidar_xy: Optional[np.ndarray] = None, match_m: float = 0.15) -> np.ndarray:
        """Nx5 rows of x, y, the unit way across (nx, ny) and the number of lidar low-only cells (LidarLowMap) within
        match_m of the spot (0 without lidar)."""
        lid = np.asarray(lidar_xy if lidar_xy is not None else np.zeros((0, 2)), float).reshape(-1, 2)
        rows = []
        for n, sx, sy, snx, sny in self._hits.values():
            if n < self.confirm_hits:
                continue
            x, y = sx / n, sy / n
            u = math.hypot(snx, sny) or 1.0
            near = int((np.hypot(lid[:, 0] - x, lid[:, 1] - y) <= match_m).sum()) if len(lid) else 0
            rows.append((x, y, snx / u, sny / u, near))
        return np.array(rows, float).reshape(-1, 5)

    def __len__(self):
        return len(self.confirmed())


def points_from_json(data: dict) -> np.ndarray:
    """/roller/points JSON -> Nx5 (x, y, nx, ny, lidar)."""
    rows = [list(r) + [0.0] * (5 - len(r)) for r in data.get('points', [])]
    return np.array(rows, float).reshape(-1, 5)


# ------------------------------------------------------------------------------------------ on the tile grid
def roller_tiles(grid, points, min_points: int = 4, min_lidar: int = 2, edge_margin_m: float = 0.15) -> Dict[tuple, dict]:
    """Tiles holding a roller: at least min_points confirmed roller spots (Nx5 from RollerMap.confirmed()) away from the
    tile's edges (edge_margin_m: the roller lies across the middle of its tile, spots near an edge are spill-over into
    the neighbour), of which at least min_lidar are confirmed by the lidar (0 = the depth camera alone decides).
    {tile: {'points', 'lidar', 'normal' (unit way across, either sign), 'line_xy' (mean spot)}}."""
    pts = np.asarray(points, float).reshape(-1, 5)
    if grid is None or len(pts) == 0:
        return {}
    acc: Dict[tuple, list] = {}
    for x, y, nx, ny, lid in pts:
        if grid.edge_distance(x, y) < edge_margin_m:
            continue
        e = acc.setdefault(tuple(grid.tile_of(x, y)), [0, 0, 0.0, 0.0, 0.0, 0.0])
        sign = -1.0 if e[4] * nx + e[5] * ny < 0 else 1.0
        e[0] += 1
        e[1] += lid > 0
        e[2] += x
        e[3] += y
        e[4] += sign * nx
        e[5] += sign * ny
    out = {}
    for tile, (n, nl, sx, sy, snx, sny) in acc.items():
        if n >= min_points and nl >= min_lidar:
            u = math.hypot(snx, sny) or 1.0
            out[tile] = {'points': int(n), 'lidar': int(nl), 'normal': (snx / u, sny / u), 'line_xy': (sx / n, sy / n)}
    return out


def _crossing(grid, frm, side, roller, info, facing):
    di, dj = SIDE_STEPS[side]
    e = edge_crossing(grid, frm, side)                # edge between `frm` and the roller tile
    e.update(neighbour=tuple(roller), roller=tuple(roller), beyond=(roller[0] + di, roller[1] + dj), from_tile=tuple(frm),
             through_m=1.5 * grid.size, spots=info['points'], lidar=info.get('lidar', 0), facing=round(facing, 2),
             line_xy=tuple(info['line_xy']), start_xy=tuple(grid.center(tuple(frm))))
    return e


def crossing_between(grid, frm, roller, info: Optional[dict] = None) -> Optional[dict]:
    """How to cross `roller` from the tile `frm` beside it (the way back over a roller crossed before), or None when the
    two do not share a side. info: the roller tile's roller_tiles() entry (its line / normal), if still known."""
    frm, roller = tuple(frm), tuple(roller)
    step = (roller[0] - frm[0], roller[1] - frm[1])
    side = next((sd for sd, st in SIDE_STEPS.items() if st == step), None)
    if side is None:
        return None
    info = info or {'points': 0, 'lidar': 0, 'line_xy': tuple(grid.center(roller)), 'normal': (1.0, 0.0)}
    return _crossing(grid, frm, side, roller, info, 1.0)


def roller_crossing(grid, tile, rollers: Dict[tuple, dict], skip=(), min_facing: float = 0.8,
                    robot_xy: Optional[Sequence[float]] = None, clear_m: float = 0.35) -> Optional[dict]:
    """How to cross a roller tile from `tile` (the robot's tile), or None.

    rollers: roller_tiles(). A roller tile sharing a side with `tile` counts when its roller lies across the way through
    that side (|cos| of the roller's normal vs the side's normal >= min_facing); the robot crosses it into the tile
    beyond. When `tile` is itself a roller tile (Nav2 drove into its front half before the roller was confirmed), it is
    crossed from where the robot stands (robot_xy) away from the roller: the crossing is then planned from the tile
    behind the robot, and start_xy is the point on the line through the tile centres level with the robot (at least
    clear_m before the roller). Tiles in `skip` (crossed or given up on) are ignored.
    Returns curtain_core.edge_crossing() for the edge into the roller tile plus 'roller', 'beyond', 'from_tile',
    'start_xy' (where the robot squares up), 'line_xy', and through_m (edge line to the centre of 'beyond')."""
    skip = set(map(tuple, skip))
    c, s = math.cos(grid.yaw), math.sin(grid.yaw)
    tile = tuple(tile)
    normals = {side: (c * di - s * dj, s * di + c * dj) for side, (di, dj) in SIDE_STEPS.items()}
    if tile in rollers and tile not in skip:
        info = rollers[tile]
        side = max(normals, key=lambda sd: abs(normals[sd][0] * info['normal'][0] + normals[sd][1] * info['normal'][1]))
        nx, ny = normals[side]
        facing = abs(nx * info['normal'][0] + ny * info['normal'][1])
        if facing < min_facing:
            return None
        here = robot_xy if robot_xy is not None else grid.center(tile)
        lx, ly = info['line_xy']
        if (lx - here[0]) * nx + (ly - here[1]) * ny < 0:          # the roller is behind: cross the other way
            side = {'E': 'W', 'W': 'E', 'N': 'S', 'S': 'N'}[side]
            nx, ny = -nx, -ny
        di, dj = SIDE_STEPS[side]
        e = _crossing(grid, (tile[0] - di, tile[1] - dj), side, tile, info, facing)
        ex, ey = e['centre_xy']
        prog = (here[0] - ex) * nx + (here[1] - ey) * ny               # robot past the edge line into the roller tile
        line = (lx - ex) * nx + (ly - ey) * ny
        prog = max(-0.5 * grid.size, min(prog, line - clear_m))
        e['start_xy'] = (ex + nx * prog, ey + ny * prog)
        return e
    best = None
    for side, (di, dj) in SIDE_STEPS.items():
        roller = (tile[0] + di, tile[1] + dj)
        if roller not in rollers or roller in skip:
            continue
        info = rollers[roller]
        nx, ny = normals[side]
        facing = abs(nx * info['normal'][0] + ny * info['normal'][1])
        if facing < min_facing:
            continue                                   # the roller lies along the way: not crossed from this side
        if best is None or info['points'] > best['spots']:
            best = _crossing(grid, tile, side, roller, info, facing)
    return best
