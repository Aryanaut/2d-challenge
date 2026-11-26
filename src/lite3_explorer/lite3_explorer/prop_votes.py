"""Scene-prop votes (no ROS): place the detector's scene props on the lidar map, so a tile with props stays a scene tile.

The object detector (lite3_perception, /perception/detections) reports props such as food, utensils or a cart with a
bounding box. place() turns each one into a map point: the box centre gives a bearing (camera intrinsics), the robot pose
gives the ray, and the ray is walked over the occupancy grid to the first occupied cell (the prop, or what it stands on),
then pushed a little further so the point lands inside the tile behind that edge. Depth is not used (it is not aligned
to the colour image on the robot).

The tile tracker counts the points per tile (tile_tracker.prop_tiles): a tile with enough of them is kept a scene tile
whatever the lidar scene test says, and an AI 'traversable' answer does not reopen it. The props only say "there is a
scene here": the room NAME they suggest is unreliable (fullrun 26 Nov: the toy kitchen was detected as food).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

# detector labels that count as scene props (survivor / water do not: they stand anywhere). The label is not trusted (the
# model mixes them up: market stall -> medical_kit, toy kitchen -> food); any of them only says "a prop is on this tile"
DEFAULT_PROPS = ('tool_bench', 'hand_tools', 'kitchen', 'utensils', 'market_stall', 'cart', 'food', 'cleaning_cart',
                 'medical_kit')


def cast(grid: np.ndarray, info, x0: float, y0: float, ang: float, *, max_range_m: float = 4.0,
         occupied: int = 65, push_m: float = 0.15, min_unknown_m: float = 0.5):
    """Walk from (x0, y0) along ang to the first occupied cell. Returns (x, y, range) push_m beyond it, or None if the ray
    leaves the map, runs into unscanned cells (past min_unknown_m) or finds nothing within max_range_m."""
    res = info.resolution
    c, s = math.cos(ang), math.sin(ang)
    r = 0.1
    while r < max_range_m:
        x, y = x0 + r * c, y0 + r * s
        col, row = int((x - info.origin_x) / res), int((y - info.origin_y) / res)
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            return None
        v = grid[row, col]
        if v >= occupied:
            return x + push_m * c, y + push_m * s, r
        if v < 0 and r > min_unknown_m:
            return None
        r += res / 2.0
    return None


def place(dets: Iterable[dict], pose: Sequence[float], grid: np.ndarray, info, *, fx: float, cx: float,
          cam_x: float = 0.252, props: Sequence[str] = DEFAULT_PROPS, min_conf: float = 0.35,
          max_range_m: float = 4.0, push_m: float = 0.15) -> List[Tuple[float, float, float, str, float]]:
    """Map points (x, y, confidence, label, range_m) for the scene props among dets (detector dicts with label,
    confidence, bbox_xyxy). pose = robot [x, y, yaw] in the map frame; the camera sits cam_x ahead of it, facing forward."""
    x, y, yaw = pose[0], pose[1], pose[2]
    x0, y0 = x + cam_x * math.cos(yaw), y + cam_x * math.sin(yaw)
    out = []
    for d in dets:
        lab, conf = d.get('label'), float(d.get('confidence', 0.0))
        if lab not in props or conf < min_conf or not d.get('bbox_xyxy'):
            continue
        x1, _, x2, _ = d['bbox_xyxy']
        bearing = -math.atan2((x1 + x2) / 2.0 - cx, fx)          # + = left
        hit = cast(grid, info, x0, y0, yaw + bearing, max_range_m=max_range_m, push_m=push_m)
        if hit is not None:
            out.append((round(hit[0], 3), round(hit[1], 3), round(conf, 3), lab, round(hit[2], 2)))
    return out
