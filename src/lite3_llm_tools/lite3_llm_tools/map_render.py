"""Render the SLAM occupancy grid as top-down images for the AI (no ROS).

Colours (also explained in the prompts):
  black      maze walls (occupied)
  light red  band within wall_band_m of a wall (too close to walk)
  white      explored free floor
  grey       unexplored
  blue arrow robot, pointing the way it faces
  magenta    numbered candidate frontiers
  tiles      1 m tile grid lines; visited tiles tinted blue with their visit order and
             joined by the dark-blue trail; tiles yet to visit outlined in green
North (+y in the map) is up; a 1 m scale bar is drawn in the corner.
"""
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from lite3_llm_tools.gridmap import GridInfo, dilate

FONT = cv2.FONT_HERSHEY_SIMPLEX
ROBOT = (255, 80, 0)
CANDIDATE = (200, 0, 200)
TRAIL = (140, 40, 0)
VISITED_TINT = (255, 200, 120)
TO_VISIT = (0, 170, 0)


def _base(grid, info, window, wall_band_m, free_threshold, occupied_threshold, target_px, max_px_per_cell):
    """Coloured map crop for window = (r0, r1, c0, c1); returns image and world->pixel function."""
    r0, r1, c0, c1 = window
    sub = np.full((r1 - r0, c1 - c0), -1, np.int16)
    gr0, gr1, gc0, gc1 = max(0, r0), min(info.height, r1), max(0, c0), min(info.width, c1)
    if gr0 < gr1 and gc0 < gc1:
        sub[gr0 - r0:gr1 - r0, gc0 - c0:gc1 - c0] = grid[gr0:gr1, gc0:gc1]
    img = np.full(sub.shape + (3,), 150, np.uint8)
    free = (sub >= 0) & (sub <= free_threshold)
    occupied = sub >= occupied_threshold
    img[free] = (255, 255, 255)
    img[dilate(occupied, int(math.ceil(wall_band_m / info.resolution))) & free] = (190, 190, 255)
    img[occupied] = (0, 0, 0)
    scale = max(1, min(max_px_per_cell, target_px // max(sub.shape)))
    img = cv2.resize(img, (sub.shape[1] * scale, sub.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    img = np.ascontiguousarray(img[::-1])
    h = img.shape[0]

    def to_px(x, y):
        c = (x - info.origin_x) / info.resolution - c0
        r = (y - info.origin_y) / info.resolution - r0
        return int(round(c * scale)), int(round(h - r * scale))
    return img, to_px, scale


def _draw_robot(img, to_px, pose):
    x, y, yaw = pose
    start = to_px(x, y)
    tip = to_px(x + 0.5 * math.cos(yaw), y + 0.5 * math.sin(yaw))
    cv2.circle(img, start, 7, ROBOT, -1)
    cv2.arrowedLine(img, start, tip, ROBOT, 3, tipLength=0.4)


def _scale_bar(img, info, scale):
    h = img.shape[0]
    bar = int(round(1.0 / info.resolution * scale))
    cv2.line(img, (10, h - 12), (10 + bar, h - 12), (0, 0, 255), 3)
    cv2.putText(img, '1 m', (12, h - 18), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def _legend(img, text):
    """A white strip above the image with the legend, wrapped onto as many lines as the image width needs."""
    items, lines, line = [t for t in text.split('  ') if t], [], ''
    for item in items:
        trial = f'{line}  {item}' if line else item
        if line and cv2.getTextSize(trial, FONT, 0.45, 1)[0][0] > img.shape[1] - 12:
            lines.append(line)
            line = item
        else:
            line = trial
    lines.append(line)
    strip = np.full((8 + 20 * len(lines), img.shape[1], 3), 255, np.uint8)
    for n, ln in enumerate(lines):
        cv2.putText(strip, ln, (6, 19 + 20 * n), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def _poly(to_px, corners):
    return np.array([to_px(x, y) for x, y in corners], np.int32)


def draw_tiles(img, to_px, tile_grid, tiles: Dict, trail: Sequence, label_tiles: bool = True):
    """Grid lines for known tiles, blue tint + visit order on visited tiles, trail, green to-visit outlines."""
    walls = (img == 0).all(axis=2)
    overlay = img.copy()
    order = {}
    for n, t in enumerate(trail):
        order.setdefault(tuple(t), n + 1)
    for tile, info_t in tiles.items():
        poly = _poly(to_px, tile_grid.corners(tile))
        if info_t['status'] == 'visited':
            cv2.fillPoly(overlay, [poly], VISITED_TINT)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)
    img[walls] = 0                                   # keep walls black under the tint
    for tile, info_t in tiles.items():
        poly = _poly(to_px, tile_grid.corners(tile))
        cv2.polylines(img, [poly], True, (120, 120, 120), 1, cv2.LINE_AA)
        if info_t['status'] == 'to_visit':
            centre = poly.mean(axis=0)
            inset = (centre + 0.82 * (poly - centre)).astype(np.int32)
            cv2.polylines(img, [inset], True, TO_VISIT, 2, cv2.LINE_AA)
    pts = [to_px(*tile_grid.center(tuple(t))) for t in trail]
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, TRAIL, 3, cv2.LINE_AA)
    if label_tiles:
        for tile, n in order.items():
            cx, cy = to_px(*tile_grid.center(tile))
            cv2.putText(img, str(n), (cx - 14, cy - 8), FONT, 0.45, TRAIL, 1, cv2.LINE_AA)


WAYPOINT = (0, 170, 0)


def draw_route(img, to_px, route: Sequence[Sequence[float]], arrow_m: float = 0.3):
    """Numbered waypoint dots, each with an arrow along its yaw; the last pose is the tile target."""
    for i, (x, y, yaw) in enumerate(route, 1):
        start = to_px(x, y)
        tip = to_px(x + arrow_m * math.cos(yaw), y + arrow_m * math.sin(yaw))
        cv2.arrowedLine(img, start, tip, WAYPOINT, 3, cv2.LINE_AA, tipLength=0.45)
        cv2.circle(img, start, 6, WAYPOINT, -1, cv2.LINE_AA)
        cv2.circle(img, start, 6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(i), (start[0] + 7, start[1] - 7), FONT, 0.5, (0, 100, 0), 2, cv2.LINE_AA)


def render_map(grid: np.ndarray, info: GridInfo, robot_pose: Optional[Sequence[float]],
               candidates: List[dict], visited: Sequence[Sequence[float]] = (), *,
               tile_grid=None, tiles: Optional[Dict] = None, trail: Sequence = (),
               free_threshold: int = 25, occupied_threshold: int = 65, wall_band_m: float = 0.35,
               route: Sequence[Sequence[float]] = (), margin_m: float = 1.0, target_px: int = 900,
               max_px_per_cell: int = 12) -> np.ndarray:
    """Whole explored map: walls, tiles + trail, past stops, robot, numbered candidate frontiers."""
    known = grid >= 0
    if not known.any():
        raise ValueError('map has no known cells yet')
    rows, cols = np.nonzero(known)
    pad = int(math.ceil(margin_m / info.resolution))
    window = (max(0, rows.min() - pad), min(info.height, rows.max() + pad + 1),
              max(0, cols.min() - pad), min(info.width, cols.max() + pad + 1))
    img, to_px, scale = _base(grid, info, window, wall_band_m, free_threshold, occupied_threshold,
                              target_px, max_px_per_cell)
    if tile_grid is not None and tiles:
        draw_tiles(img, to_px, tile_grid, tiles, trail)
    for vx, vy, *_ in visited:
        cv2.circle(img, to_px(vx, vy), max(4, scale), (0, 140, 255), -1)
    for cand in candidates:
        px = to_px(*cand['goal_xy'])
        radius = max(9, int(0.18 / info.resolution * scale))
        cv2.circle(img, px, radius, CANDIDATE, -1)
        cv2.circle(img, px, radius, (80, 0, 80), 2)
        label = str(cand['id'])
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
        cv2.putText(img, label, (px[0] - tw // 2, px[1] + th // 2), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if len(route):
        draw_route(img, to_px, route)
    if robot_pose is not None:
        _draw_robot(img, to_px, robot_pose)
    _scale_bar(img, info, scale)
    legend = ('black=wall  red=near wall  white=free  grey=unknown  blue=robot  magenta#=candidate  '
              'orange=past stop')
    if tile_grid is not None:
        legend += '  blue tiles=visited (order)  green box=tile to visit'
    if len(route):
        legend += '  green dot+arrow=waypoint (yaw)'
    return _legend(img, legend)


def render_initial_tile_panels(grid: np.ndarray, info: GridInfo, robot_pose: Sequence[float],
                               candidates: List[dict], *, half_width_m: float = 2.5,
                               free_threshold: int = 25, occupied_threshold: int = 65,
                               wall_band_m: float = 0.35, panel_px: int = 420) -> np.ndarray:
    """One panel per initial-tile candidate (A, B, C...), side by side, around the robot."""
    rx, ry = robot_pose[0], robot_pose[1]
    half = int(math.ceil(half_width_m / info.resolution))
    rc, cc = info.world_to_cell(rx, ry)
    window = (rc - half, rc + half + 1, cc - half, cc + half + 1)
    panels = []
    for cand in candidates:
        img, to_px, scale = _base(grid, info, window, wall_band_m, free_threshold, occupied_threshold,
                                  panel_px, 40)
        tg = cand['grid']
        for di in (-2, -1, 0, 1, 2):            # faint neighbouring tiles show the grid alignment
            for dj in (-2, -1, 0, 1, 2):
                if di or dj:
                    cv2.polylines(img, [_poly(to_px, tg.corners((di, dj)))], True, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.polylines(img, [_poly(to_px, tg.corners((0, 0)))], True, (0, 120, 255), 4, cv2.LINE_AA)
        for side in ('N', 'E', 'S', 'W'):
            (x0, y0), (x1, y1) = tg.edge((0, 0), side)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            cx, cy = tg.center((0, 0))
            lx, ly = mx + 0.28 * (mx - cx) / 0.5, my + 0.28 * (my - cy) / 0.5   # just outside the edge
            px = to_px(lx, ly)
            cv2.putText(img, side, (px[0] - 7, px[1] + 7), FONT, 0.7, (0, 90, 200), 2, cv2.LINE_AA)
        _draw_robot(img, to_px, robot_pose)
        _scale_bar(img, info, scale)
        title = np.full((34, img.shape[1], 3), 255, np.uint8)
        cv2.putText(title, f"tile {cand['id']}: {cand['kind']}", (6, 23), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(np.vstack([title, img]))
        panels.append(np.full((panels[-1].shape[0], 8, 3), 60, np.uint8))
    sheet = np.hstack(panels[:-1])
    return _legend(sheet, 'orange square=proposed 1 m tile  N/E/S/W=its sides  black=wall  red=near wall  '
                          'white=free  grey=unknown  blue=robot')
