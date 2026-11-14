"""1 m x 1 m tile system over the SLAM map (no ROS).

The grid is anchored by the initial tile (chosen with the AI at start-up) and may be
rotated to line up with the maze walls. Tile (0, 0) is the initial tile; i grows
along the grid's x axis ("east"), j along its y axis ("north").

Tile status
  visited   the robot's centre has been inside the tile (part of the trail)
  to_visit  the map shows walkable floor in the tile, but the robot hasn't been there
  blocked   the tile is mapped but has (almost) no walkable floor (walls / outside)
  unknown   not mapped yet
Tile sides (N, E, S, W): wall / open / unknown, from the map cells along each edge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lite3_kitchen_explorer.frontier_core import GridInfo, detect_frontier_cells

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

    def as_dict(self) -> dict:
        return {'origin_xy': [round(self.origin_x, 3), round(self.origin_y, 3)],
                'yaw_deg': round(math.degrees(self.yaw), 2), 'size_m': self.size}


# --------------------------------------------------------------- alignment
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


def best_offset(w: np.ndarray, size: float = 1.0, band_m: float = 0.08, step_m: float = 0.05) -> Tuple[float, int]:
    """Offset o in [0, size) putting the most wall points on the lines w = o + k * size."""
    best = (0.0, -1)
    for o in np.arange(0.0, size, step_m):
        d = np.mod(w - o, size)
        hits = int(((d < band_m) | (d > size - band_m)).sum())
        if hits > best[1]:
            best = (float(o), hits)
    return best


def initial_candidates(grid: np.ndarray, info: GridInfo, robot_pose: Sequence[float], *,
                       size: float = 1.0, radius_m: float = 2.5, occupied_threshold: int = 65) -> List[dict]:
    """Placements for the initial tile (always containing the robot), best guess first."""
    rx, ry = robot_pose[0], robot_pose[1]

    def centred(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        half = size / 2.0
        return TileGrid(rx - c * half + s * half, ry - s * half - c * half, yaw, size)

    cands = []
    pts = _occupied_points(grid, info, (rx, ry), radius_m, occupied_threshold)
    est = estimate_wall_angle(pts)
    if est is not None:
        yaw, quality = est
        c, s = math.cos(yaw), math.sin(yaw)
        u = c * pts[:, 0] + s * pts[:, 1]
        v = -s * pts[:, 0] + c * pts[:, 1]
        ou, hu = best_offset(u, size)
        ov, hv = best_offset(v, size)
        ru, rv = c * rx + s * ry, -s * rx + c * ry
        cu = ou + math.floor((ru - ou) / size) * size       # tile containing the robot
        cv = ov + math.floor((rv - ov) / size) * size
        aligned = TileGrid(c * cu - s * cv, s * cu + c * cv, yaw, size)
        cands.append({'grid': aligned, 'kind': 'snapped to the nearby walls',
                      'wall_points_on_edges': hu + hv, 'alignment_quality': round(quality, 2)})
        centre = centred(yaw)
        du = abs(((centre.to_local(rx, ry)[0]) - (aligned.to_local(rx, ry)[0])))
        dv = abs(((centre.to_local(rx, ry)[1]) - (aligned.to_local(rx, ry)[1])))
        if du > 0.15 or dv > 0.15:
            cands.append({'grid': centre, 'kind': 'centred on the robot, turned to match the walls'})
    if est is None or abs(math.degrees(est[0]) % 90) > 3:
        cands.append({'grid': centred(0.0), 'kind': 'centred on the robot, along the map axes'})
    for letter, cand in zip('ABCD', cands):
        cand['id'] = letter
        cand['sides'] = tile_sides(grid, info, cand['grid'], (0, 0), occupied_threshold=occupied_threshold)
    return cands


# ----------------------------------------------------------- classification
def tile_sides(grid: np.ndarray, info: GridInfo, tg: TileGrid, tile: Tile, *, free_threshold: int = 25,
               occupied_threshold: int = 65, samples: int = 20, wall_fraction: float = 0.4) -> Dict[str, str]:
    """wall / open / unknown for each side, sampling the map along the edge (±1 cell)."""
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
        if occ >= wall_fraction * samples:
            result[side] = 'wall'
        elif free >= 0.5 * samples:
            result[side] = 'open'
        else:
            result[side] = 'unknown'
    return result


def classify_tiles(grid: np.ndarray, info: GridInfo, tg: TileGrid, visited, *, free_threshold: int = 25,
                   occupied_threshold: int = 65, min_free: float = 0.15, min_known: float = 0.70) -> Dict[Tile, dict]:
    """Status and coverage for every tile that has any mapped cell (plus every visited tile).

    A tile is defined (to_visit / blocked) only once at least min_known of its area has been
    scanned by the lidar; before that it is 'unknown' and is not offered as a target. Visited
    tiles (including the initial tile the robot stands in) are always defined."""
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
    frontier_c = count(detect_frontier_cells(grid, free_threshold))
    known_c = count(known)

    visited = set(map(tuple, visited))
    tiles = {}
    for k in np.nonzero(known_c)[0]:
        tile = (int(k // nj + i0), int(k % nj + j0))
        tot = max(1, int(total[k]))
        info_t = {'free': round(free_c[k] / tot, 2), 'occupied': round(occ_c[k] / tot, 2),
                  'known': round(known_c[k] / tot, 2), 'frontier': bool(frontier_c[k] > 0)}
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
        tiles.setdefault(tile, {'free': 0.0, 'occupied': 0.0, 'known': 0.0, 'frontier': False})
        tiles[tile]['status'] = 'visited'
    return tiles


def tile_counts(tiles: Dict[Tile, dict]) -> Dict[str, int]:
    counts = {'visited': 0, 'to_visit': 0, 'blocked': 0, 'unknown': 0}
    for t in tiles.values():
        counts[t['status']] += 1
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
    import cv2
    if grid_connected:
        reach = connected_steps(tiles, current)
    else:
        reach = {t: (0, []) for t, v in tiles.items() if v['status'] == 'to_visit'}
    if not reach:
        return []
    not_wall = (grid < occupied_threshold).astype(np.uint8)
    dist = cv2.distanceTransform(not_wall, cv2.DIST_L2, 5) * info.resolution
    free = (grid >= 0) & (grid <= free_threshold)
    out = []
    order = sorted((t for t in reach if tiles[t]['status'] == 'to_visit'),
                   key=lambda t: (reach[t][0], math.dist(tg.center(t), robot_xy)))
    for tile in order:
        xs, ys = zip(*tg.corners(tile))
        r0, c0 = info.world_to_cell(min(xs), min(ys))
        r1, c1 = info.world_to_cell(max(xs), max(ys))
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(info.height - 1, r1), min(info.width - 1, c1)
        if r0 > r1 or c0 > c1:
            continue
        rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
        wx = info.origin_x + (cols + 0.5) * info.resolution
        wy = info.origin_y + (rows + 0.5) * info.resolution
        u, v = tg.to_local(wx, wy)
        inside = (np.floor(u / tg.size) == tile[0]) & (np.floor(v / tg.size) == tile[1]) & free[rows, cols]
        if not inside.any():
            continue
        clear = dist[rows, cols]
        cx, cy = tg.center(tile)
        d_centre = np.hypot(wx - cx, wy - cy)
        ok = inside & (clear >= min_clearance_m)
        if ok.any():
            k = np.argmin(np.where(ok, d_centre, np.inf))
        else:
            k = np.argmax(np.where(inside, clear, -1.0))
        k = np.unravel_index(k, inside.shape)
        if clear[k] < min_clearance_m:
            continue
        gx, gy = float(wx[k]), float(wy[k])
        steps, path = reach[tile]
        out.append({'id': len(out) + 1, 'tile': tile_label(tile), 'steps': steps,
                    'path': [tile_label(t) for t in path],
                    'goal_xy': [round(gx, 3), round(gy, 3)],
                    'distance_m': round(math.dist((gx, gy), robot_xy), 2),
                    'clearance_m': round(float(clear[k]), 2),
                    'has_frontier': bool(tiles[tile].get('frontier'))})
        if len(out) >= limit:
            break
    return out
