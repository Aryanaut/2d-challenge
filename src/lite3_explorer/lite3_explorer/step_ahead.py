"""Step ahead in the depth image: the fallback trigger for the h-step gait (no ROS).

The mission manager's elevation rule (/step_up when the target tile is higher than the current one) needs the terrain
classifier to have measured the target tile, and only knows whole tiles: a step inside a tile, a tile whose elevation never
came in or came in late, or a backtracking hop between two tiles of the same height across a raised patch never switch
the gait. This looks straight ahead of the robot instead.

Each depth frame is projected into a level frame at base_link (x forward, y left, z up): the static base_link <- camera
transform, then the IMU's roll and pitch (the odom -> base_link TF carries yaw only, so the body tilt on the step comes
from the IMU). The points in a corridor ahead of the body (|y| <= half_width_m) are binned along x into a height profile
(median per bin; a bin whose heights spread more than max_bin_spread_m is a riser face or clutter and is skipped).

A step up is a flat top (at least min_top_m deep, its bins within top_flat_m of each other) min_rise_m..max_rise_m above
the floor, reached by a jump (the floor-level bin before it at most max_edge_gap_m back): a ramp rises bin by bin and
has no such jump. The floor is the nearest visible ground; when that is itself raised (the robot is at the edge already,
and the camera, 20 deg down from 0.25 m ahead of base_link, cannot see the floor under the front feet) the nominal foot
level, stand_height_m under base_link, is used instead.

fullrun_26_11_210206, tile 1,1 (+0.14 m by the classifier): the depth profile shows the floor 0.33 m under base_link and a
flat top ~0.2 m above it from 0.75 m ahead at 50.8 s; the elevation rule switched at 52.8 s, the front legs went up at 54.5 s.
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StepAheadParams:
    stride: int = 4                  # use every 4th pixel
    min_depth_m: float = 0.15        # depth readings outside this range are dropped
    max_depth_m: float = 3.0
    half_width_m: float = 0.20       # corridor half width (left / right of base_link) ...
    strip_m: float = 0.10            # ... cut into strips this wide along x, each profiled on its own (an edge at an angle
                                     # to the heading smears over the x bins of one wide profile)
    min_strips: int = 2              # a step must be found in at least this many strips
    near_m: float = 0.30             # profile from this far ahead of base_link ...
    far_m: float = 1.2               # ... a step must start within this far ahead (its top may reach min_top_m further)
    bin_m: float = 0.05              # profile bin along x
    min_bin_points: int = 6
    max_bin_spread_m: float = 0.03   # 20..80 % height spread of a bin: more = riser face / clutter, not a surface
    stand_height_m: float = 0.33     # floor under base_link standing (fullrun_26_11_210206: 0.31..0.34)
    ref_band_m: float = 0.15         # the floor = the median of the visible ground this deep from the nearest bin
    min_rise_m: float = 0.06         # step height ...
    max_rise_m: float = 0.35         # ... (higher = a wall / obstacle, not a step)
    max_edge_gap_m: float = 0.20     # the last floor-level bin is at most this far before the top
    min_top_m: float = 0.15          # top at least this deep ...
    top_flat_m: float = 0.03         # ... its bins within this of the first top bin
    max_tilt_deg: float = 6.0        # body roll / pitch beyond this: no detection (on the step already, or turning on it)


@dataclass
class StepAhead:
    dist_m: float                    # where the top starts, ahead of base_link
    rise_m: float                    # top above the floor
    floor_z: float                   # floor height in the level base_link frame
    top_z: float
    at_edge: bool                    # the nearest visible ground was raised already: floor = nominal foot level

    def to_dict(self):
        return {'dist_m': round(self.dist_m, 2), 'rise_m': round(self.rise_m, 3), 'floor_z': round(self.floor_z, 3),
                'top_z': round(self.top_z, 3), 'at_edge': self.at_edge}


def rpy_from_quat(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """(roll, pitch, yaw) in rad, ROS convention (pitch + = nose DOWN)."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def level_rotation(roll: float, pitch: float) -> np.ndarray:
    """Body -> level frame (same heading, z up) for the IMU's roll and pitch."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return ry @ rx


def depth_to_points(depth: np.ndarray, k: Sequence[float], p: StepAheadParams) -> np.ndarray:
    """Depth image (metres) -> Nx3 points in the optical frame (x right, y down, z forward), every stride-th pixel."""
    fx, fy, cx, cy = k
    s = max(1, int(p.stride))
    d = depth[::s, ::s].astype(np.float32)
    v, u = np.mgrid[0:depth.shape[0]:s, 0:depth.shape[1]:s]
    ok = np.isfinite(d) & (d > p.min_depth_m) & (d < p.max_depth_m)
    z = d[ok]
    return np.stack([(u[ok] - cx) * z / fx, (v[ok] - cy) * z / fy, z], axis=1)


def to_level_base(pts_optical: np.ndarray, r_base_cam: np.ndarray, t_base_cam: Sequence[float],
                  roll: float, pitch: float) -> np.ndarray:
    """Optical-frame points -> level frame at base_link (x forward, y left, z up)."""
    pts = pts_optical @ np.asarray(r_base_cam).T + np.asarray(t_base_cam, dtype=float)
    return pts @ level_rotation(roll, pitch).T


def height_profile(pts: np.ndarray, p: StepAheadParams, y_range: Optional[Tuple[float, float]] = None):
    """Corridor ahead of the body (or the strip y_range of it) binned along x: (bin centres, median z, point count,
    20..80 % spread). Bins with too few points have NaN height."""
    y0, y1 = y_range if y_range is not None else (-p.half_width_m, p.half_width_m)
    edges = np.arange(p.near_m, p.far_m + p.min_top_m + 1e-9, p.bin_m)
    centres = edges[:-1] + p.bin_m / 2
    n = len(centres)
    z_med, spread, count = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, int)
    if len(pts) == 0 or n == 0:
        return centres, z_med, count, spread
    sel = (pts[:, 1] >= y0) & (pts[:, 1] <= y1) & (pts[:, 0] >= edges[0]) & (pts[:, 0] < edges[-1])
    x, z = pts[sel, 0], pts[sel, 2]
    idx = np.minimum(((x - edges[0]) / p.bin_m).astype(int), n - 1)
    order = np.argsort(idx, kind='stable')
    idx, z = idx[order], z[order]
    bounds = np.searchsorted(idx, np.arange(n + 1))
    for i in range(n):
        zs = z[bounds[i]:bounds[i + 1]]
        count[i] = len(zs)
        if len(zs) >= p.min_bin_points:
            q20, q50, q80 = np.percentile(zs, (20, 50, 80))
            z_med[i], spread[i] = q50, q80 - q20
    return centres, z_med, count, spread


def find_step(centres: np.ndarray, z: np.ndarray, spread: np.ndarray, p: StepAheadParams) -> Optional[StepAhead]:
    """The nearest step up in a height profile, or None."""
    valid = [i for i in range(len(centres)) if np.isfinite(z[i]) and spread[i] <= p.max_bin_spread_m]
    if not valid:
        return None
    first = valid[0]
    ref = float(np.median([z[i] for i in valid if centres[i] <= centres[first] + p.ref_band_m]))
    nominal = -p.stand_height_m
    at_edge = ref - nominal >= p.min_rise_m
    if at_edge:
        ref = nominal
    pos = 0
    while pos < len(valid):
        j = valid[pos]
        if centres[j] - p.bin_m / 2 > p.far_m:
            break
        if z[j] - ref < p.min_rise_m:
            pos += 1
            continue
        if pos == 0:
            edge_ok = at_edge
        else:
            k = valid[pos - 1]
            edge_ok = z[k] - ref <= p.min_rise_m / 2 and centres[j] - centres[k] <= p.max_edge_gap_m
        top: List[int] = [j]
        nxt = pos + 1
        while nxt < len(valid):
            m = valid[nxt]
            if centres[m] - centres[top[-1]] > p.max_edge_gap_m or abs(z[m] - z[j]) > p.top_flat_m:
                break
            top.append(m)
            nxt += 1
        if edge_ok and centres[top[-1]] - centres[j] + p.bin_m >= p.min_top_m - 1e-9:
            top_z = float(np.median([z[i] for i in top]))
            rise = top_z - ref
            if p.min_rise_m <= rise <= p.max_rise_m:
                return StepAhead(dist_m=float(centres[j] - p.bin_m / 2), rise_m=rise, floor_z=ref, top_z=top_z,
                                 at_edge=at_edge and pos == 0)
        pos = max(nxt, pos + 1)
    return None


def strips(p: StepAheadParams) -> List[Tuple[float, float]]:
    n = max(1, int(round(2 * p.half_width_m / p.strip_m)))
    w = 2 * p.half_width_m / n
    return [(-p.half_width_m + i * w, -p.half_width_m + (i + 1) * w) for i in range(n)]


def detect(pts_level: np.ndarray, p: StepAheadParams) -> Optional[StepAhead]:
    """A step found in at least min_strips strips of the corridor: the nearest start, the median rise."""
    found = []
    for yr in strips(p):
        centres, z, _, spread = height_profile(pts_level, p, yr)
        det = find_step(centres, z, spread, p)
        if det is not None:
            found.append(det)
    if len(found) < max(1, p.min_strips):
        return None
    rise = float(np.median([d.rise_m for d in found]))
    floor = float(np.median([d.floor_z for d in found]))
    return StepAhead(dist_m=min(d.dist_m for d in found), rise_m=rise, floor_z=floor, top_z=floor + rise,
                     at_edge=all(d.at_edge for d in found))


def tilted(roll: float, pitch: float, p: StepAheadParams) -> bool:
    lim = math.radians(p.max_tilt_deg)
    return abs(roll) > lim or abs(pitch) > lim


class StepConfirm:
    """A step counts once seen in confirm_frames frames in a row (a frame with none starts over); a tilted frame (no
    detection made) neither counts nor breaks the run."""

    def __init__(self, confirm_frames: int = 2):
        self.confirm_frames = max(1, int(confirm_frames))
        self.hits = 0

    def update(self, det: Optional[StepAhead]) -> bool:
        self.hits = self.hits + 1 if det is not None else 0
        return self.hits >= self.confirm_frames
