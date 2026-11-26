"""1 m x 1 m tile system over the SLAM map (no ROS).

The grid is anchored by the initial tile (chosen with the AI at start-up) and may be
rotated to line up with the maze walls. Tile (0, 0) is the initial tile. The grid yaw is the wall angle
in [0, 90) deg of the map frame (along the map axes when no walls are found): i grows along the grid's +x,
j along its +y. The robot's heading does not turn the grid; it only picks the start tile's back wall.

Tile status
  visited   the robot's centre has been inside the tile (part of the trail)
  to_visit  the map shows walkable floor in the tile, but the robot hasn't been there
  blocked   the tile is mapped but has (almost) no walkable floor (walls / outside)
  unknown   not mapped yet (not "identified": it does not count towards the end condition)
  unreachable  (set by tile_tracker.py, not here) a to_visit tile the robot gave up on after repeated failures
Tile sides (N, E, S, W): wall / open / unknown, from the map cells along each edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lite3_explorer.map_grid import GridInfo, unmapped_edge_cells

SIDES = ('N', 'E', 'S', 'W')
Tile = Tuple[int, int]


@dataclass
class TileGrid:
    origin_x: float          # world position of the initial tile's south-west corner
    origin_y: float
    yaw: float               # grid rotation in the map frame (rad)
    size: float = 1.0

    def to_local(self, x, y):
        dx, dy = np.asarray(x, float) - self.origin_x, np.asarray(y, float) - self.origin_y
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def to_world(self, u, v):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.origin_x + c * u - s * v, self.origin_y + s * u + c * v

    def tile_of(self, x: float, y: float) -> Tile:
        u, v = self.to_local(x, y)
        return int(math.floor(float(u) / self.size)), int(math.floor(float(v) / self.size))

    def center(self, tile: Tile) -> Tuple[float, float]:
        return self.to_world((tile[0] + 0.5) * self.size, (tile[1] + 0.5) * self.size)

    def corners(self, tile: Tile) -> List[Tuple[float, float]]:
        i, j = tile
        return [self.to_world(u * self.size, v * self.size)
                for u, v in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]

    def edge(self, tile: Tile, side: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        sw, se, ne, nw = self.corners(tile)
        return {'S': (sw, se), 'E': (se, ne), 'N': (nw, ne), 'W': (sw, nw)}[side]

    def edge_distance(self, x: float, y: float) -> float:
        """Distance (m) from a point to the nearest boundary of the tile it is in."""
        u, v = self.to_local(x, y)
        fu, fv = float(u) / self.size % 1.0, float(v) / self.size % 1.0
        return self.size * min(fu, 1.0 - fu, fv, 1.0 - fv)

    def as_dict(self) -> dict:
        return {'origin_xy': [round(self.origin_x, 3), round(self.origin_y, 3)],
                'yaw_deg': round(math.degrees(self.yaw), 2), 'size_m': self.size}


# --------------------------------------------------------------- alignment
def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))



def _occupied_points(grid: np.ndarray, info: GridInfo, center: Sequence[float], radius_m: float,
                     occupied_threshold: int) -> np.ndarray:
    rows, cols = np.nonzero(grid >= occupied_threshold)
    xs = info.origin_x + (cols + 0.5) * info.resolution
    ys = info.origin_y + (rows + 0.5) * info.resolution
    keep = (xs - center[0]) ** 2 + (ys - center[1]) ** 2 <= radius_m ** 2
    return np.stack([xs[keep], ys[keep]], axis=1)


def estimate_wall_angle(points: np.ndarray, bin_m: float = 0.05) -> Optional[Tuple[float, float]]:
    """Grid rotation in [0, 90) deg that makes walls axis-aligned, and a quality score (>1 is good)."""
    if len(points) < 15:
        return None
    scores = []
    for deg in range(90):
        a = math.radians(deg)
        u = math.cos(a) * points[:, 0] + math.sin(a) * points[:, 1]
        v = -math.sin(a) * points[:, 0] + math.cos(a) * points[:, 1]
        score = 0.0
        for w in (u, v):
            hist = np.bincount(np.floor((w - w.min()) / bin_m).astype(int))
            score += float((hist.astype(float) ** 2).sum())
        scores.append(score)
    scores = np.asarray(scores)
    best = int(scores.argmax())
    return math.radians(best), float(scores[best] / max(1e-9, np.median(scores)))


def find_wall(along: np.ndarray, across: np.ndarray, *, d_min: float, d_max: float, half_span: float = 0.30,
              band_m: float = 0.03, slot_m: float = 0.05, min_coverage: float = 0.8,
              step_m: float = 0.01) -> Optional[float]:
    """Nearest straight wall in one direction: points given as (along = distance in that direction, across =
    sideways offset), both relative to the robot. A wall is a line `along = d` (d_min <= d <= d_max) whose points
    cover at least min_coverage of the slot_m slots over |across| <= half_span, within band_m of the line: continuous,
    so scattered clutter (objects, a scene tile) is not taken for a wall. Returns d (the best-covered line of the
    nearest such wall) or None."""
    keep = (np.abs(across) <= half_span) & (along >= d_min - band_m) & (along <= d_max + band_m)
    along, across = along[keep], across[keep]
    if len(along) == 0:
        return None
    n_slots = max(1, int(round(2 * half_span / slot_m)))
    slot = np.clip(((across + half_span) / slot_m).astype(int), 0, n_slots - 1)
    found = None
    for d in np.arange(d_min, d_max + 1e-9, step_m):
        on = np.abs(along - d) <= band_m
        cover = len(np.unique(slot[on])) / n_slots
        if cover >= min_coverage:
            if found is None:
                found = (d, cover, int(on.sum()))
            elif d - found[0] > 2 * band_m:
                break                                  # past the nearest wall
            elif (cover, int(on.sum())) > (found[1], found[2]):
                found = (d, cover, int(on.sum()))
        elif found is not None and d - found[0] > 2 * band_m:
            break
    return None if found is None else float(np.median(along[np.abs(along - found[0]) <= band_m]))


def back_wall_heading(grid: np.ndarray, info: GridInfo, tg: TileGrid, here: Tile, scene: Tile, *,
                      max_deg: float = 20.0, step_deg: float = 0.5, beyond_m: float = 0.4, span: float = 0.4,
                      band_m: float = 0.03, slot_m: float = 0.05, min_coverage: float = 0.8,
                      occupied_threshold: int = 65) -> Optional[dict]:
    """The back wall of a scene tile seen from `here` (a tile sharing a side with it): the straight wall in the scene
    tile's far half or up to beyond_m past its far edge, turned at most max_deg from the grid. The lidar does not see
    past the back wall, so per slot_m slot across |across| <= span * size only the FARTHEST occupied cell counts
    (along = distance from the scene tile's centre away from `here`): clutter in front of the wall (a table against
    it, a solid block) never does. At each angle the line `along = d` through the most of those cells (within band_m)
    is found; a wall needs them in at least min_coverage of the slots (continuous). Best = highest coverage, then the
    tightest; a straight-line fit to its cells then gives the angle finer than step_deg (map cells are coarse).
    Returns {'yaw': heading square to that wall, facing it from `here`, 'offset_deg': its turn from the grid,
    'distance_m': d, 'coverage'} or None (not a neighbour / no such wall)."""
    di, dj = int(scene[0]) - int(here[0]), int(scene[1]) - int(here[1])
    if abs(di) + abs(dj) != 1:
        return None
    base = tg.yaw + math.atan2(dj, di)
    half, half_span = tg.size / 2.0, span * tg.size
    sc = tg.center(tuple(scene))
    pts = _occupied_points(grid, info, sc, math.hypot(half + beyond_m, half_span) + band_m, occupied_threshold)
    if len(pts) < 5:
        return None
    dx, dy = pts[:, 0] - sc[0], pts[:, 1] - sc[1]
    n_slots = max(1, int(round(2 * half_span / slot_m)))
    best = None
    for deg in np.arange(-max_deg, max_deg + 1e-9, step_deg):
        a = base + math.radians(float(deg))
        c, s = math.cos(a), math.sin(a)
        along, across = c * dx + s * dy, -s * dx + c * dy
        keep = (np.abs(across) <= half_span) & (along >= 0.0) & (along <= half + beyond_m)
        if not keep.any():
            continue
        al, ac = along[keep], across[keep]
        slot = np.clip(((ac + half_span) / slot_m).astype(int), 0, n_slots - 1)
        order = np.lexsort((al, slot))                         # per slot, the farthest cell last
        last = np.r_[slot[order][1:] != slot[order][:-1], True]
        far_al, far_ac = al[order][last], ac[order][last]
        for d in far_al:                                       # the candidate lines: through each farthest cell
            on = np.abs(far_al - d) <= band_m
            cover = int(on.sum()) / n_slots
            key = (round(cover, 3), -float(np.std(far_al[on])), -abs(float(deg)))
            if best is None or key > best[0]:
                best = (key, float(deg), far_ac[on], far_al[on])
    if best is None or best[0][0] < min_coverage:
        return None
    _, deg, ac, al = best
    dist = float(np.mean(al))
    if len(ac) >= 3:                                           # fit along = k * across + d: the wall is turned by -atan(k)
        k = float(np.polyfit(ac, al, 1)[0])
        ref = deg - math.degrees(math.atan(k))
        if abs(ref) <= max_deg:
            deg = ref
    return {'yaw': wrap(base + math.radians(deg)), 'offset_deg': round(deg, 2),
            'distance_m': round(dist, 3), 'coverage': best[0][0]}


def align_to_walls(grid: np.ndarray, info: GridInfo, tg: TileGrid, tile: Tile, robot_xy: Sequence[float], *,
                   radius_m: float = 2.0, max_shift_m: float = 0.15, max_rot_deg: float = 6.0,
                   min_quality: float = 1.5, occupied_threshold: int = 65) -> Optional[dict]:
    """A small correction of the tile grid so the edges of `tile` (the one the robot stands in) lie on the walls seen
    around the robot. Turn: the wall angle (estimate_wall_angle) nearest the grid's own, within max_rot_deg. Edges: per
    axis the walls of the robot's own tile (start_tile_walls; two opposite walls are averaged) pull that tile edge onto
    them, within max_shift_m (further off: not a wall of this tile, the axis is left alone). The grid turns about the tile's
    south-west corner. Returns {'grid': TileGrid, 'rot_deg', 'shift_u', 'shift_v', 'walls', 'quality'} or None (no
    walls / nothing worth correcting)."""
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    pts = _occupied_points(grid, info, (rx, ry), radius_m, occupied_threshold)
    est = estimate_wall_angle(pts)
    if est is None or est[1] < min_quality:
        return None
    half_turn = math.pi / 2
    d_yaw = wrap(est[0] - tg.yaw + math.pi / 4) % half_turn - math.pi / 4      # nearest wall angle, in [-45, 45) deg
    if abs(d_yaw) > math.radians(max_rot_deg):
        d_yaw = 0.0                                                            # walls are not this grid's: keep the turn
    yaw = tg.yaw + d_yaw
    c, s = math.cos(yaw), math.sin(yaw)
    walls = start_tile_walls(pts, (rx, ry), yaw, tg.size)
    sw = tg.corners(tile)[0]
    low = [c * sw[0] + s * sw[1], -s * sw[0] + c * sw[1]]                      # the tile's SW corner in the new frame
    old = list(low)
    r = (c * rx + s * ry, -s * rx + c * ry)
    shifts = []
    for ax, axis in enumerate(('u', 'v')):
        implied = [r[ax] + walls[k] - tg.size if k[0] == '+' else r[ax] - walls[k]
                   for k in ('+' + axis, '-' + axis) if k in walls]
        shift = 0.0
        if implied:
            want = sum(implied) / len(implied)
            if abs(want - low[ax]) <= max_shift_m:
                shift = want - low[ax]
                low[ax] = want
        shifts.append(shift)
    if abs(d_yaw) < math.radians(0.2) and max(abs(shifts[0]), abs(shifts[1])) < 0.01:
        return None
    ou, ov = low[0] - tile[0] * tg.size, low[1] - tile[1] * tg.size            # grid origin in the new frame
    new = TileGrid(c * ou - s * ov, s * ou + c * ov, yaw, tg.size)
    if tuple(new.tile_of(rx, ry)) != tuple(tile):
        return None                                                            # would put the robot on another tile
    return {'grid': new, 'rot_deg': round(math.degrees(d_yaw), 2), 'shift_u': round(shifts[0], 3),
            'shift_v': round(shifts[1], 3), 'walls': {k: round(d, 3) for k, d in walls.items()},
            'quality': round(est[1], 2)}


def start_tile_walls(pts: np.ndarray, robot_xy: Sequence[float], yaw: float, size: float) -> dict:
    """Walls of the robot's own tile in the grid frame turned by yaw: {'+u', '-u', '+v', '-v'} -> distance from the
    robot (m) or missing. Searched from a quarter to 0.7 of a tile away (a robot standing near the tile centre)."""
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = pts[:, 0] - robot_xy[0], pts[:, 1] - robot_xy[1]
    u, v = c * dx + s * dy, -s * dx + c * dy
    out = {}
    for key, along, across in (('+u', u, v), ('-u', -u, v), ('+v', v, u), ('-v', -v, u)):
        d = find_wall(along, across, d_min=0.25 * size, d_max=0.7 * size)
        if d is not None:
            out[key] = d
    return out


def initial_candidates(grid: np.ndarray, info: GridInfo, robot_pose: Sequence[float], *,
                       size: float = 1.0, radius_m: float = 2.5, occupied_threshold: int = 65) -> List[dict]:
    """Placements for the initial tile (always containing the robot), best guess first.

    A: turned to match the walls (estimate_wall_angle), with the edges on the walls of the robot's OWN tile: per axis
    the wall behind the robot (opposite its heading, robot_pose[2]) if there is one, else the walls found on that
    axis (two opposite walls: averaged), else centred on the robot. Only continuous walls next to the robot count
    (start_tile_walls), so clutter further away (objects, a scene tile) does not move the grid.
    Then the alternatives: centred on the robot (turned to the walls) and along the map axes."""
    rx, ry = robot_pose[0], robot_pose[1]
    heading = robot_pose[2] if len(robot_pose) > 2 else None

    def centred(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        half = size / 2.0
        return TileGrid(rx - c * half + s * half, ry - s * half - c * half, yaw, size)

    # The grid yaw is the wall angle itself, never mixed with the heading. (An earlier "+j along the heading" step
    # returned heading - 90 deg snapped to quarter turns, which threw the wall angle away: in bag
    # stairs69_25_11_105440 the walls were at 22 deg and the grid at 31.3 deg.)
    cands = []
    pts = _occupied_points(grid, info, (rx, ry), radius_m, occupied_threshold)
    est = estimate_wall_angle(pts)
    if est is not None:
        yaw, quality = est
        c, s = math.cos(yaw), math.sin(yaw)
        walls = start_tile_walls(pts, (rx, ry), yaw, size)
        back = None
        if heading is not None:                         # the grid direction pointing away from the heading
            k = round(wrap(heading + math.pi - yaw) / (math.pi / 2)) % 4
            back = ['+u', '+v', '-u', '-v'][k]
        lows, used = [], []
        for axis in ('u', 'v'):
            keys = [k for k in ('+' + axis, '-' + axis) if k in walls]
            if back in keys:
                keys = [back]                            # the back wall wins on its axis
            # edge of the tile on a wall: + side at d -> the tile starts at d - size; - side at d -> at -d
            implied = [walls[k] - size if k[0] == '+' else -walls[k] for k in keys]
            lows.append(sum(implied) / len(implied) if implied else -size / 2.0)
            used += [f'{k} {walls[k]:.2f} m' + (' (back)' if k == back else '') for k in keys]
        ru, rv = c * rx + s * ry, -s * rx + c * ry
        cu, cv = ru + lows[0], rv + lows[1]
        aligned = TileGrid(c * cu - s * cv, s * cu + c * cv, yaw, size)
        cands.append({'grid': aligned, 'kind': 'turned to the walls, edges on the start tile\'s walls'
                      + (f' ({", ".join(used)})' if used else ' (none found: centred on the robot)'),
                      'walls': {k: round(d, 3) for k, d in walls.items()}, 'back': back,
                      'alignment_quality': round(quality, 2)})
        centre = centred(yaw)
        du = abs(centre.to_local(rx, ry)[0] - aligned.to_local(rx, ry)[0])
        dv = abs(centre.to_local(rx, ry)[1] - aligned.to_local(rx, ry)[1])
        if du > 0.05 or dv > 0.05:
            cands.append({'grid': centre, 'kind': 'centred on the robot, turned to match the walls'})
    if est is None or abs(math.degrees(est[0]) % 90) > 3:
        cands.append({'grid': centred(0.0), 'kind': 'centred on the robot, along the map axes'})
    for letter, cand in zip('ABCD', cands):
        cand['id'] = letter
        cand['sides'] = tile_sides(grid, info, cand['grid'], (0, 0), occupied_threshold=occupied_threshold)
    return cands


# ----------------------------------------------------------- classification
def side_shares(grid: np.ndarray, info: GridInfo, tg: TileGrid, tile: Tile, *, free_threshold: int = 25,
                occupied_threshold: int = 65, samples: int = 20) -> Dict[str, Tuple[float, float]]:
    """(wall share, free share) of each side: the share of points along the edge with mapped wall within ±1 cell, and
    the share whose ±1 cell window is all free. A maze wall reads ~1.0 wall; clutter on an edge (furniture legs, a box)
    reads partly (bag endconditiontest_25_11_201356, tile 2,0 outside the maze: 0.55-0.65)."""
    result = {}
    for side in SIDES:
        (x0, y0), (x1, y1) = tg.edge(tile, side)
        occ = free = 0
        for t in (np.arange(samples) + 0.5) / samples:
            r, c = info.world_to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            win = grid[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            if win.size == 0:
                continue
            if (win >= occupied_threshold).any():
                occ += 1
            elif ((win >= 0) & (win <= free_threshold)).all():
                free += 1
        result[side] = (round(occ / samples, 2), round(free / samples, 2))
    return result


def sides_from_shares(shares: Dict[str, Tuple[float, float]], wall_fraction: float = 0.4) -> Dict[str, str]:
    result = {}
    for side, (occ, free) in shares.items():
        if occ >= wall_fraction:
            result[side] = 'wall'
        elif free >= 0.5:
            result[side] = 'open'
        else:
            result[side] = 'unknown'
    return result


def tile_sides(grid: np.ndarray, info: GridInfo, tg: TileGrid, tile: Tile, *, free_threshold: int = 25,
               occupied_threshold: int = 65, samples: int = 20, wall_fraction: float = 0.4) -> Dict[str, str]:
    """wall / open / unknown for each side, sampling the map along the edge (±1 cell)."""
    return sides_from_shares(side_shares(grid, info, tg, tile, free_threshold=free_threshold,
                                         occupied_threshold=occupied_threshold, samples=samples), wall_fraction)


def edge_key(tg: TileGrid, tile: Tile, side: str) -> tuple:
    """The same key for a shared edge from either tile (rounded, sorted end points)."""
    (x0, y0), (x1, y1) = tg.edge(tile, side)
    return tuple(sorted(((round(x0, 2), round(y0, 2)), (round(x1, 2), round(y1, 2)))))


def side_towards(a: Tile, b: Tile) -> Optional[str]:
    """The side of tile a that it shares with tile b (None: they share no side)."""
    for side, (di, dj, _) in _STEPS.items():
        if (a[0] + di, a[1] + dj) == tuple(b):
            return side
    return None


def edge_coverage(grid: np.ndarray, info: GridInfo, tg: TileGrid, tiles, *, occupied_threshold: int = 65,
                  samples: int = 20, band_m: float = 0.0
                  ) -> List[Tuple[tuple, Tuple[Tuple[float, float], Tuple[float, float]], float]]:
    """Every tile edge (each shared edge once) as (key, world segment, share of it that lies on mapped wall).

    A sample point along the edge is on wall when a mapped wall cell lies within ±1 cell of it or, with band_m > 0,
    within ±1 cell of any point up to band_m across the edge (either side). The grid is often a few cm off the maze
    walls (only the tiles the robot enters are aligned): bag fullrun_25_11_225456, tile 0,-1 W, wall 8-15 cm inside
    the tile, 25% on the ±1 cell line -> never a virtual wall; 70% with a 0.15 m band."""
    out, seen = [], set()
    h, w = grid.shape
    occ_map = grid >= occupied_threshold
    dil = occ_map.copy()                          # wall within ±1 cell (the 3x3 window around a sample)
    dil[1:, :] |= occ_map[:-1, :]
    dil[:-1, :] |= occ_map[1:, :]
    row = dil.copy()
    dil[:, 1:] |= row[:, :-1]
    dil[:, :-1] |= row[:, 1:]
    f = (np.arange(samples) + 0.5) / samples
    offsets = (np.arange(-band_m, band_m + 1e-9, info.resolution) if band_m > 0 else np.zeros(1))
    for tile, t in tiles.items():
        if t.get('status') == 'unknown':
            continue
        for side in SIDES:
            (x0, y0), (x1, y1) = tg.edge(tile, side)
            key = edge_key(tg, tile, side)
            if key in seen:
                continue
            seen.add(key)
            length = math.hypot(x1 - x0, y1 - y0) or 1.0
            nx, ny = -(y1 - y0) / length, (x1 - x0) / length          # across the edge
            px = x0 + f[:, None] * (x1 - x0) + offsets[None, :] * nx  # samples x offsets
            py = y0 + f[:, None] * (y1 - y0) + offsets[None, :] * ny
            r = np.floor((py - info.origin_y) / info.resolution).astype(np.int64)
            c = np.floor((px - info.origin_x) / info.resolution).astype(np.int64)
            inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            hit = np.zeros(r.shape, bool)
            hit[inside] = dil[r[inside], c[inside]]
            out.append((key, ((x0, y0), (x1, y1)), float(hit.any(axis=1).sum()) / samples))
    return out


def edge_line_share(grid: np.ndarray, info: GridInfo, tg: TileGrid, tile: Tile, side: str, *, band_m: float = 0.15,
                    thick_cells: int = 4, occupied_threshold: int = 65, samples: int = 20) -> float:
    """Share of a tile edge covered by ONE straight line of mapped wall parallel to it: the best stripe thick_cells
    wide anywhere within band_m across the edge (either side), counting the samples along the edge with an occupied
    cell in that stripe. A maze wall is such a line; scene clutter near the edge is not: it is spread over the whole
    band, so edge_coverage (any wall cell within the band per sample) reads it as a wall and this does not
    (sim bag backtracking_20261126_131935, clutter 0.15 m in from the edge: 0.80-1.00 there, 0.00-0.10 here).
    Coarser walls of the real lidar map lose some share here (blockedscenetest_26_11_084131: 1.00 -> 0.60), so this
    only decides when edge_coverage calls every side of a scene tile a wall (TileTracker._view_open)."""
    occ = grid >= occupied_threshold
    h, w = occ.shape
    (x0, y0), (x1, y1) = tg.edge(tile, side)
    length = math.hypot(x1 - x0, y1 - y0) or 1.0
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length
    f = (np.arange(samples) + 0.5) / samples
    offsets = np.arange(-band_m, band_m + 1e-9, info.resolution)
    px = x0 + f[:, None] * (x1 - x0) + offsets[None, :] * nx
    py = y0 + f[:, None] * (y1 - y0) + offsets[None, :] * ny
    r = np.floor((py - info.origin_y) / info.resolution).astype(np.int64)
    c = np.floor((px - info.origin_x) / info.resolution).astype(np.int64)
    inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    hit = np.zeros(r.shape, bool)
    hit[inside] = occ[r[inside], c[inside]]
    k = max(1, min(int(thick_cells), hit.shape[1]))
    return max(float(hit[:, s:s + k].any(axis=1).mean()) for s in range(hit.shape[1] - k + 1))


def wall_edges(grid: np.ndarray, info: GridInfo, tg: TileGrid, tiles, *, min_fraction: float = 0.5,
               occupied_threshold: int = 65, samples: int = 20) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Tile edges (world segments, each shared edge once) where MORE than min_fraction of the edge lies on mapped wall."""
    return [seg for _, seg, share in edge_coverage(grid, info, tg, tiles, occupied_threshold=occupied_threshold,
                                                   samples=samples) if share > min_fraction]


def classify_tiles(grid: np.ndarray, info: GridInfo, tg: TileGrid, visited, *, free_threshold: int = 25,
                   occupied_threshold: int = 65, min_free: float = 0.15, min_known: float = 0.70,
                   scene_min_known: float = 0.85, scene_max_free: float = 0.80,
                   scene_margin_m: float = 0.10) -> Dict[Tile, dict]:
    """Status and coverage for every tile that has any mapped cell (plus every visited tile).

    A tile is defined (to_visit / blocked) only once at least min_known of its area has been
    scanned by the lidar; before that it is 'unknown' and is not offered as a target. Visited
    tiles (including the initial tile the robot stands in) are always defined.

    'scene' marks a SCENE tile: (almost) completely scanned (known >= scene_min_known) and less than scene_max_free of
    its SCANNED area traversable (free / known). Relative to the scanned area, not the whole tile: a cluttered tile always
    has corners hidden behind its own clutter until the robot stands on it, so an absolute free share with a ~0.95 scan
    gate flagged it too late (after it was entered). It is a flag only: the status (and so the tile selection) is unchanged.
    The free/known ratio of the scene test ('scene_free') is taken over the tile's INTERIOR only, scene_margin_m in from
    every edge: maze walls lie on the tile edges, and with the grid a few cm off them (bag curtaintest_25_11_132200: grid
    centred on the robot, ~8 cm from the walls) the walls of a dead-end fall inside it and read as clutter (empty tile
    2,0: free/known 0.85 on the whole tile -> flagged, 0.98 inside the margin). A scene's props stand inside the tile."""
    h, w = grid.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs = info.origin_x + (cols + 0.5) * info.resolution
    ys = info.origin_y + (rows + 0.5) * info.resolution
    u, v = tg.to_local(xs, ys)
    ti = np.floor(u / tg.size).astype(np.int64)
    tj = np.floor(v / tg.size).astype(np.int64)
    known = grid >= 0
    if not known.any() and not visited:
        return {}
    i0, j0 = ti[known].min() if known.any() else 0, tj[known].min() if known.any() else 0
    ni = (ti[known].max() - i0 + 1) if known.any() else 1
    nj = (tj[known].max() - j0 + 1) if known.any() else 1
    inside = (ti >= i0) & (ti < i0 + ni) & (tj >= j0) & (tj < j0 + nj)
    key = np.where(inside, (ti - i0) * nj + (tj - j0), 0)
    n = ni * nj

    def count(mask):
        return np.bincount(key[mask & inside], minlength=n)

    total = count(np.ones_like(known))
    free_c = count((grid >= 0) & (grid <= free_threshold))
    occ_c = count(grid >= occupied_threshold)
    edge_c = count(unmapped_edge_cells(grid, free_threshold))
    known_c = count(known)
    m = min(max(0.0, scene_margin_m / tg.size), 0.45)            # the scene test's inset, as a share of a tile side
    fu, fv = u / tg.size - ti, v / tg.size - tj
    interior = (fu >= m) & (fu < 1.0 - m) & (fv >= m) & (fv < 1.0 - m)
    in_known_c = count(known & interior)
    in_free_c = count((grid >= 0) & (grid <= free_threshold) & interior)

    visited = set(map(tuple, visited))
    tiles = {}
    for k in np.nonzero(known_c)[0]:
        tile = (int(k // nj + i0), int(k % nj + j0))
        tot = max(1, int(total[k]))
        scene_free = in_free_c[k] / in_known_c[k] if in_known_c[k] else 1.0   # free/known inside the margin
        info_t = {'free': round(free_c[k] / tot, 2), 'occupied': round(occ_c[k] / tot, 2),
                  'known': round(known_c[k] / tot, 2), 'unmapped_edge': bool(edge_c[k] > 0),
                  'scene_free': round(scene_free, 2),
                  'scene': bool(known_c[k] / tot >= scene_min_known and scene_free < scene_max_free)}
        if tile in visited:
            status = 'visited'
        elif info_t['known'] < min_known:
            status = 'unknown'
        elif info_t['free'] >= min_free:
            status = 'to_visit'
        else:
            status = 'blocked'
        info_t['status'] = status
        tiles[tile] = info_t
    for tile in visited:
        tiles.setdefault(tile, {'free': 0.0, 'occupied': 0.0, 'known': 0.0, 'unmapped_edge': False, 'scene': False})
        tiles[tile]['status'] = 'visited'
    return tiles


def tile_counts(tiles: Dict[Tile, dict]) -> Dict[str, int]:
    counts = {'visited': 0, 'to_visit': 0, 'blocked': 0, 'unknown': 0, 'unreachable': 0}
    for t in tiles.values():
        counts[t['status']] = counts.get(t['status'], 0) + 1
    return counts


def tile_label(tile: Tile) -> str:
    return f'{tile[0]},{tile[1]}'


# ------------------------------------------------------- connected tiles
_STEPS = {'E': (1, 0, 'W'), 'W': (-1, 0, 'E'), 'N': (0, 1, 'S'), 'S': (0, -1, 'N')}


def _connected(tiles: Dict[Tile, dict], a: Tile, side: str) -> Optional[Tile]:
    di, dj, back = _STEPS[side]
    b = (a[0] + di, a[1] + dj)
    tb = tiles.get(b)
    if tb is None or tb['status'] not in ('visited', 'to_visit'):
        return None
    if tiles[a].get('sides', {}).get(side) == 'wall' or tb.get('sides', {}).get(back) == 'wall':
        return None
    return b


def connected_steps(tiles: Dict[Tile, dict], start: Tile) -> Dict[Tile, Tuple[int, List[Tile]]]:
    """BFS over walkable tiles (visited / to_visit) whose shared side is not a wall.
    Returns {tile: (steps, path from start)}."""
    from collections import deque
    if start not in tiles:
        return {}
    seen = {start: (0, [start])}
    queue = deque([start])
    while queue:
        a = queue.popleft()
        steps, path = seen[a]
        for side in SIDES:
            b = _connected(tiles, a, side)
            if b is not None and b not in seen:
                seen[b] = (steps + 1, path + [b])
                queue.append(b)
    return seen


# ------------------------------------------------------- ending tile (end condition ending_tile)
_WALKABLE = ('visited', 'to_visit', 'unreachable')
_AXES = (('E', 'W', ('N', 'S')), ('N', 'S', ('E', 'W')))   # (forward, backward, the two long sides)


def wall_count(tiles: Dict[Tile, dict], tile: Tile) -> int:
    sides = tiles.get(tile, {}).get('sides', {})
    return sum(1 for s in SIDES if sides.get(s) == 'wall')


def _step(tile: Tile, side: str) -> Tile:
    di, dj, _ = _STEPS[side]
    return (tile[0] + di, tile[1] + dj)


def _joined(tiles: Dict[Tile, dict], a: Tile, side: str) -> bool:
    """No wall on the side a shares with its neighbour (from either tile)."""
    b = _step(a, side)
    back = _STEPS[side][2]
    return tiles.get(a, {}).get('sides', {}).get(side) != 'wall' and \
        tiles.get(b, {}).get('sides', {}).get(back) != 'wall'


def hallways(tiles: Dict[Tile, dict], min_tiles: int = 4) -> List[List[Tile]]:
    """Straight one-tile-wide hallways: runs of at least min_tiles walkable tiles in a row (along i or j), each with a
    mapped wall on both long sides, consecutive tiles joined by a side that is not a wall. Each run is maximal."""
    out = []
    for fwd, back, lateral in _AXES:
        def ok(t):
            d = tiles.get(t)
            return d is not None and d['status'] in _WALKABLE and \
                all(d.get('sides', {}).get(s) == 'wall' for s in lateral)
        for t in sorted(tiles):
            if not ok(t):
                continue
            prev = _step(t, back)
            if ok(prev) and _joined(tiles, prev, fwd):
                continue                                   # not the start of a run
            run = [t]
            while ok(_step(run[-1], fwd)) and _joined(tiles, run[-1], fwd):
                run.append(_step(run[-1], fwd))
            if len(run) >= min_tiles:
                out.append(run)
    return out


def open_tiles(tiles: Dict[Tile, dict], solid_wall_share: Optional[float] = None) -> List[Tile]:
    """Walkable tiles with no wall on any edge, every side mapped (an unknown side does not count).
    solid_wall_share None: every side reads 'open'. Else no side is a SOLID wall (mapped wall along at least this share
    of the edge, tile 'wall_share'): clutter partly on an edge (outside the maze) is no maze wall."""
    out = []
    for t, d in sorted(tiles.items()):
        sides = d.get('sides', {})
        if d['status'] not in _WALKABLE or len(sides) < 4 or 'unknown' in sides.values():
            continue
        if solid_wall_share is None:
            ok = all(v == 'open' for v in sides.values())
        else:
            share = d.get('wall_share') or {s: (1.0 if v == 'wall' else 0.0) for s, v in sides.items()}
            ok = all(share.get(s, 0.0) < solid_wall_share for s in SIDES)
        if ok:
            out.append(t)
    return out


def looks_open(d: dict, solid_wall_share: Optional[float] = 0.4, min_seen_share: float = 0.5) -> bool:
    """A tile the lidar shows with no wall: no edge with mapped wall along solid_wall_share of it or more (0.4 = the wall
    / open classification of tile_sides: any edge read as a wall, maze wall or clutter, rules the tile out; an open edge
    reads <= ~0.25 from the wall ends at its corners), and every
    edge mapped (wall + free) along at least min_seen_share of it, so an edge the scan has not reached does not pass as
    open. Needs the tracker's wall_share / free_share (a tile still 'unknown' has none).
    solid_wall_share None: every side must read 'open' instead."""
    occ, free = d.get('wall_share'), d.get('free_share')
    if d.get('status') not in _WALKABLE or not occ or not free:
        return False
    if solid_wall_share is None:
        return all(d.get('sides', {}).get(s) == 'open' for s in SIDES)
    return all(occ.get(s, 0.0) < solid_wall_share and occ.get(s, 0.0) + free.get(s, 0.0) >= min_seen_share
               for s in SIDES)


def next_open_tile(tiles: Dict[Tile, dict], current: Tile, *, solid_wall_share: Optional[float] = 0.4,
                   min_seen_share: float = 0.5, exclude=()) -> Optional[Tile]:
    """The tile next to `current`, through an open side of it, that the scan shows with no walls (looks_open): the
    outside of the maze seen from its last tile. None when the current tile looks open itself (the robot is already
    outside) or no neighbour does. Tiles in `exclude` (walked already) are skipped."""
    d = tiles.get(tuple(current))
    if d is None or looks_open(d, solid_wall_share, min_seen_share):
        return None
    for side in SIDES:
        if d.get('sides', {}).get(side) != 'open':
            continue
        n = _step(tuple(current), side)
        if n not in exclude and n in tiles and looks_open(tiles[n], solid_wall_share, min_seen_share):
            return n
    return None


def find_ending_tile(tiles: Dict[Tile, dict], trail: Sequence[Tile], *, min_hallway_tiles: int = 4,
                     triggers: Sequence[str] = ('hallway', 'open_tile'), walls: int = 2,
                     entered_only: bool = False, preceding: bool = True,
                     solid_wall_share: Optional[float] = None, min_seen_share: float = 0.5) -> Optional[dict]:
    """The ENDING tile: a hallway of at least min_hallway_tiles tiles and/or a tile with no walls (the TRIGGER) was
    found; the ending tile is the tile preceding it on the robot's route that has exactly `walls` walls on its edges
    (and is not part of a trigger itself).

    The route is the trail (the tiles the robot walked, in order). A trigger the robot has not walked into yet is
    reached by the shortest walkable route from the current tile, appended to the trail. Triggers on the trail are tried
    in the order the robot met them; one with no such tile before it (e.g. the start tile) is skipped.
    entered_only: only triggers on the trail count (the robot stood in them, so the lidar saw all their edges), never one
    seen from a distance. preceding False: the trigger tile itself is the ending tile.
    Trigger next_tile (checked first): the scan from the CURRENT tile (the trail's last) shows the tile next to it with
    no walls (next_open_tile): the current tile is the ending tile and the robot never enters the outside.
    The INITIAL tile (the trail's first) is never the ending tile.
    Returns {'tile', 'trigger_tile', 'trigger'} or None."""
    start = tuple(trail[0]) if trail else None
    if 'next_tile' in triggers and trail and tuple(trail[-1]) != start:
        cur = tuple(trail[-1])
        n = next_open_tile(tiles, cur, solid_wall_share=solid_wall_share, min_seen_share=min_seen_share,
                           exclude={tuple(t) for t in trail})
        if n is not None:
            return {'tile': cur, 'trigger_tile': n, 'trigger': f'the scan shows next tile {tile_label(n)} with no walls'}
    feature: Dict[Tile, str] = {}
    if 'hallway' in triggers:
        for run in hallways(tiles, min_hallway_tiles):
            for t in run:
                feature.setdefault(t, f'hallway of {len(run)} tiles {tile_label(run[0])}..{tile_label(run[-1])}')
    if 'open_tile' in triggers:
        for t in open_tiles(tiles, solid_wall_share):
            feature.setdefault(t, f'tile {tile_label(t)} has no walls')
    if not feature:
        return None
    route = []
    for t in trail:
        t = tuple(t)
        if not route or route[-1] != t:
            route.append(t)

    def before(route, hit):
        if not preceding:
            if route[hit] == start:
                return None
            return {'tile': route[hit], 'trigger_tile': route[hit], 'trigger': feature[route[hit]]}
        for k in range(hit - 1, -1, -1):
            t = route[k]
            if t != start and t not in feature and wall_count(tiles, t) == walls:
                return {'tile': t, 'trigger_tile': route[hit], 'trigger': feature[route[hit]]}
        return None

    for hit, t in enumerate(route):
        if t in feature:
            found = before(route, hit)
            if found is not None:
                return found
    if not route or entered_only:
        return None
    on_trail = set(route)
    reach = [(steps, path) for t, (steps, path) in connected_steps(tiles, route[-1]).items()
             if t in feature and t not in on_trail]
    if not reach:
        return None
    _, path = min(reach, key=lambda r: (r[0], r[1][-1]))
    ext = route + path[1:]
    hit = next(k for k in range(len(route), len(ext)) if ext[k] in feature)
    return before(ext, hit)


def nearest_connected_targets(grid: np.ndarray, info: GridInfo, tg: TileGrid, tiles: Dict[Tile, dict],
                              current: Tile, robot_xy: Sequence[float], *, limit: int = 6,
                              min_clearance_m: float = 0.30, occupied_threshold: int = 65,
                              free_threshold: int = 25, grid_connected: bool = True) -> List[dict]:
    """Not-yet-visited tiles reachable through connected tiles, nearest (fewest steps) first.

    With grid_connected=False every unvisited tile is a candidate (the caller checks that
    the robot can actually reach it, e.g. with a Nav2 path), ordered by distance from the robot.

    Each target gets a goal point inside the tile: the free cell nearest the tile centre
    that keeps min_clearance_m from walls (else the free cell with the most clearance).
    Tiles whose best point is closer than min_clearance_m to a wall are left out.
    """
    if grid_connected:
        reach = connected_steps(tiles, current)
    else:
        reach = {t: (0, []) for t, v in tiles.items() if v['status'] == 'to_visit'}
    if not reach:
        return []
    dist = clearance_map(grid, info, occupied_threshold)
    free = (grid >= 0) & (grid <= free_threshold)
    out = []
    order = sorted((t for t in reach if tiles[t]['status'] == 'to_visit'),
                   key=lambda t: (reach[t][0], math.dist(tg.center(t), robot_xy)))
    for tile in order:
        point = tile_goal_point(info, tg, tile, free, dist, min_clearance_m)
        if point is None:
            continue
        gx, gy, clearance = point
        steps, path = reach[tile]
        cx, cy = tg.center(tile)
        out.append({'id': len(out) + 1, 'tile': tile_label(tile), 'steps': steps,
                    'path': [tile_label(t) for t in path],
                    'goal_xy': [round(gx, 3), round(gy, 3)],
                    'centre_xy': [round(cx, 3), round(cy, 3)],
                    'centre_offset_m': round(math.dist((gx, gy), (cx, cy)), 2),
                    'distance_m': round(math.dist((gx, gy), robot_xy), 2),
                    'clearance_m': round(clearance, 2),
                    # wire name kept for lite3_llm_tools' choose_target prompt: the tile borders unmapped space
                    'has_frontier': bool(tiles[tile].get('unmapped_edge'))})
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------- goal points in a tile
def clearance_map(grid: np.ndarray, info: GridInfo, occupied_threshold: int = 65) -> np.ndarray:
    """Distance (m) from every cell to the nearest wall cell."""
    import cv2
    not_wall = (grid < occupied_threshold).astype(np.uint8)
    return cv2.distanceTransform(not_wall, cv2.DIST_L2, 5) * info.resolution


def tile_goal_point(info: GridInfo, tg: TileGrid, tile: Tile, free: np.ndarray, dist: np.ndarray,
                    min_clearance_m: float) -> Optional[Tuple[float, float, float]]:
    """(x, y, clearance) of the goal point in a tile: the exact tile centre when its cell is free and keeps
    min_clearance_m from walls, else the free cell nearest the centre that does; None if the tile has no such cell."""
    xs, ys = zip(*tg.corners(tile))
    r0, c0 = info.world_to_cell(min(xs), min(ys))
    r1, c1 = info.world_to_cell(max(xs), max(ys))
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(info.height - 1, r1), min(info.width - 1, c1)
    if r0 > r1 or c0 > c1:
        return None
    rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    wx = info.origin_x + (cols + 0.5) * info.resolution
    wy = info.origin_y + (rows + 0.5) * info.resolution
    u, v = tg.to_local(wx, wy)
    inside = (np.floor(u / tg.size) == tile[0]) & (np.floor(v / tg.size) == tile[1]) & free[rows, cols]
    if not inside.any():
        return None
    clear = dist[rows, cols]
    cx, cy = tg.center(tile)
    rc, cc = info.world_to_cell(cx, cy)
    if info.contains(rc, cc) and free[rc, cc] and dist[rc, cc] >= min_clearance_m:
        return float(cx), float(cy), float(dist[rc, cc])
    d_centre = np.hypot(wx - cx, wy - cy)
    ok = inside & (clear >= min_clearance_m)
    if ok.any():
        k = np.argmin(np.where(ok, d_centre, np.inf))
    else:
        k = np.argmax(np.where(inside, clear, -1.0))
    k = np.unravel_index(k, inside.shape)
    if clear[k] < min_clearance_m:
        return None
    return float(wx[k]), float(wy[k]), float(clear[k])


def nearest_tile_centres(grid: np.ndarray, info: GridInfo, tg: TileGrid, tiles: Dict[Tile, dict],
                         robot_xy: Sequence[float], *, statuses=('visited', 'to_visit'), limit: int = 5,
                         min_distance_m: float = 0.0, min_clearance_m: float = 0.25,
                         occupied_threshold: int = 65, free_threshold: int = 25,
                         exclude: Sequence[Tile] = (), adjacent_to: Optional[Tile] = None) -> List[dict]:
    """Recovery targets: walkable tiles ordered by the straight-line distance from the robot to their centre.

    The goal is the tile centre itself when it is free and clear of walls, else the clear free point nearest
    the centre (tile_goal_point). Goals closer than min_distance_m to the robot are skipped (the robot is
    already there). Nav2 path checks are deliberately not made: the robot is stuck, so the planner may fail
    from where it stands even when the tile is reachable.
    """
    dist = clearance_map(grid, info, occupied_threshold)
    free = (grid >= 0) & (grid <= free_threshold)
    excluded = set(map(tuple, exclude))
    if adjacent_to is not None:
        # only the robot's own tile and the tiles sharing a side with it: anything else (diagonal, further away)
        # can only be reached THROUGH another tile
        near = {tuple(adjacent_to)} | {(adjacent_to[0] + di, adjacent_to[1] + dj)
                                      for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))}
        excluded |= {t for t in tiles if t not in near}
    order = sorted((t for t, d in tiles.items() if d.get('status') in statuses and t not in excluded),
                   key=lambda t: math.dist(tg.center(t), robot_xy))
    out = []
    for tile in order:
        point = tile_goal_point(info, tg, tile, free, dist, min_clearance_m)
        if point is None:
            continue
        gx, gy, clearance = point
        d = math.dist((gx, gy), robot_xy)
        if d < min_distance_m and not (adjacent_to is not None and tile == tuple(adjacent_to)):
            continue        # already there (the robot's own tile is kept: recovery stays in the tile it stands in)
        cx, cy = tg.center(tile)
        out.append({'tile': tile_label(tile), 'goal_xy': [round(gx, 3), round(gy, 3)],
                    'centre_xy': [round(cx, 3), round(cy, 3)], 'distance_m': round(d, 2),
                    'clearance_m': round(clearance, 2), 'status': tiles[tile].get('status')})
    # nearest by the distance to the point the robot will really drive to (the goal point moves off the centre when
    # the centre is blocked), not to the tile centre
    own = tile_label(tuple(adjacent_to)) if adjacent_to is not None else None
    out.sort(key=lambda c: (c['tile'] != own, c['distance_m']))     # the tile the robot is in comes first
    return out[:limit]


# ------------------------------------------------------- adjacency to explored tiles
ADJ_CURRENT, ADJ_VISITED, ADJ_WALL, ADJ_NONE = 0, 1, 2, 3
ADJACENCY_NAMES = {ADJ_CURRENT: 'open side with the current tile', ADJ_VISITED: 'open side with a visited tile',
                   ADJ_WALL: 'shares only a wall side with a visited tile', ADJ_NONE: 'shares no side with a visited tile'}


def direct_path_through_shared_side(tg: TileGrid, current: Tile, tile: Tile, path, tol_m: float = 0.15) -> bool:
    """True when `tile` shares a side with `current` and the path (map-frame points) runs from one into the other
    through that side only: every point inside the two tiles (tol_m slack), and the ends on opposite sides of it."""
    di, dj = tile[0] - current[0], tile[1] - current[1]
    if abs(di) + abs(dj) != 1 or len(path) < 2:
        return False
    tol = tol_m / tg.size
    lo = (min(current[0], tile[0]) - tol, min(current[1], tile[1]) - tol)
    hi = (max(current[0], tile[0]) + 1 + tol, max(current[1], tile[1]) + 1 + tol)
    pts = []
    for x, y in (p[:2] for p in path):
        u, v = tg.to_local(x, y)
        u, v = float(u) / tg.size, float(v) / tg.size
        if not (lo[0] <= u <= hi[0] and lo[1] <= v <= hi[1]):
            return False
        pts.append((u, v))
    axis = 0 if di else 1                                       # the coordinate that changes across the shared side
    edge = max(current[axis], tile[axis])
    sign = di + dj                                              # +1: the next tile is on the high side of the edge
    return (pts[0][axis] - edge) * sign < tol and (pts[-1][axis] - edge) * sign > 0


def tiles_crossed(tg: TileGrid, path, tol_m: float = 0.15) -> set:
    """Tiles a path (map-frame points) runs through. A point only counts for a tile when it is more than tol_m
    inside it, so cutting a corner or hugging a shared side does not count."""
    out = set()
    for x, y in (p[:2] for p in path):
        u, v = tg.to_local(x, y)
        fu, fv = float(u) / tg.size, float(v) / tg.size
        i, j = math.floor(fu), math.floor(fv)
        inside = tg.size * min(fu - i, i + 1 - fu, fv - j, j + 1 - fv)
        if inside > tol_m:
            out.add((int(i), int(j)))
    return out


def adjacency_level(tiles: Dict[Tile, dict], tile: Tile, visited, current: Optional[Tile]) -> int:
    """How a candidate tile touches the explored area (lower = better):

      0 ADJ_CURRENT  shares a side with the current tile, and that side is not a wall
      1 ADJ_VISITED  shares a non-wall side with some visited tile
      2 ADJ_WALL     shares a side with a visited tile, but only across a wall
      3 ADJ_NONE     shares no side with any visited tile
    """
    visited = set(map(tuple, visited))
    best = ADJ_NONE
    for side in SIDES:
        di, dj, back = _STEPS[side]
        nb = (tile[0] + di, tile[1] + dj)
        if nb not in visited:
            continue
        wall = tiles.get(tile, {}).get('sides', {}).get(side) == 'wall' or \
            tiles.get(nb, {}).get('sides', {}).get(back) == 'wall'
        level = ADJ_WALL if wall else (ADJ_CURRENT if nb == current else ADJ_VISITED)
        best = min(best, level)
    return best
