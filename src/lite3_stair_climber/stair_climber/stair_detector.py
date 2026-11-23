"""Stair detection from a 3D lidar point cloud (pure numpy, no ROS needed).

Coordinate frame used everywhere in this file ("level base frame"):
    origin  = robot base_link
    +x      = forward, +y = left, +z = up
    rotated so the floor is horizontal (the robot's own roll/pitch removed
    using the IMU), but NOT rotated by yaw, so +x is still "where the robot
    is facing".

Idea: crop a box in front of the robot and look at how high the points are
above the floor.
    open floor -> everything near height 0
    wall       -> one huge height jump at a single distance
    stairs     -> a first jump of 6-20 cm, then more plateaus further away
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorParams:
    # Region of interest in front of the robot (level base frame, metres)
    roi_x_min: float = 0.25
    roi_x_max: float = 1.6
    roi_half_width: float = 0.6
    # Profile + wall checks only look at this central corridor, so handrail
    # posts at the sides of the staircase don't look like a wall
    corridor_half_width: float = 0.25
    # A point within this height of the floor counts as "floor"
    floor_tolerance: float = 0.04
    # Valid step (riser) heights. Lite3 RUG General <= 12 cm, H-Step <= 15 cm.
    min_step_height: float = 0.06
    max_step_height: float = 0.20
    # Profile resolution along x
    bin_size: float = 0.05
    min_points_per_bin: int = 5
    # Slice width along y used to fit the first step edge
    edge_slice_width: float = 0.05
    # The first step must span at least this width (stairs, not a small box)
    min_edge_width: float = 0.35
    # Anything rising faster than this behind the first edge is a wall/box
    max_stair_slope_deg: float = 45.0
    # Need this many points in the ROI at all
    min_roi_points: int = 50

    # ---- close-range mode (23 Nov) ----
    # Under ~1.3 m the lidar cannot see the floor or the first step, but it still
    # sees the higher step faces (stairs_run2: steps clear from 0.6 m). When the
    # normal detector fails, find the NEAREST VISIBLE step face instead.
    close_range: bool = False
    close_x_min: float = 0.55          # skip lidar self-hits (robot body) closer than this
    close_x_max: float = 2.0
    close_min_height: float = 0.08     # ignore floor-level points
    close_max_height: float = 0.65     # anything taller (top step is ~0.5 m) = wall
    close_bin: float = 0.025
    close_min_points: int = 40         # points on the stairs in the centre strip, one scan
    close_min_rise: float = 0.06       # need >= 2 levels: a second step this much higher
    close_depth: float = 0.5           # look this far behind the nearest face for the rise
    close_slice_width: float = 0.10    # y slices for the edge (angle / centre) fit
    close_slice_min_points: int = 6
    # The nearest VISIBLE face is usually the 2nd step (the 1st is in the blind
    # zone). distance = face - this, so it is roughly the distance to the 1st step.
    close_distance_offset: float = 0.13

    # ---- wall ahead (24 Nov) ----
    # Points straight ahead, higher than wall_min_z above base_link (levelled with the IMU). The state machine
    # only acts on it once the robot is going down / past the stairs: on the way up, handrail posts and the
    # staircase itself also reach this high (stairs_run2).
    wall_half_width: float = 0.25      # corridor the robot walks through
    wall_x_min: float = 0.35           # closer than this = lidar self-hits / blind zone
    wall_x_max: float = 2.0
    wall_min_z: float = 0.20           # above base_link (~0.6 m above the floor the robot stands on)
    wall_max_z: float = 1.0            # higher = ceiling (pitched 37 deg nose-down, the ceiling behind looks 'ahead')
    wall_min_points: int = 15

@dataclass
class StairObservation:
    detected: bool
    reason: str = ''
    distance: float = math.nan     # x distance to the first step edge at center_y (m)
    step_height: float = math.nan  # height of the first riser (m)
    edge_yaw: float = 0.0          # rad; turn the robot by this to face the steps squarely
    center_y: float = 0.0          # lateral offset of the staircase centre (m, +left)
    width: float = 0.0             # visible width of the first step (m)
    num_steps: int = 0             # plateaus seen (rough)
    close: bool = False            # found by the close-range (step-face) detector


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Angles in radians."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_to_rpy(x: float, y: float, z: float, w: float):
    """Returns (roll, pitch, yaw) in radians, ROS convention.

    Note: in ROS a POSITIVE pitch means the nose points DOWN.
    """
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def to_level_base_frame(points_sensor: np.ndarray,
                        sensor_rotation: np.ndarray,
                        sensor_translation: np.ndarray,
                        robot_roll: float,
                        robot_pitch: float) -> np.ndarray:
    """Sensor-frame points (N,3) -> level base frame (N,3).

    sensor_rotation / sensor_translation: pose of the sensor in base_link
    (from TF, or from the extrinsics table in Lab Exercise 7).
    robot_roll / robot_pitch: from the IMU, used to make the floor horizontal.
    """
    pts_base = points_sensor @ sensor_rotation.T + sensor_translation
    level = rpy_to_matrix(robot_roll, robot_pitch, 0.0)
    return pts_base @ level.T


def estimate_floor_z(points_level: np.ndarray,
                     x_min: float = 0.25, x_max: float = 0.7,
                     half_width: float = 0.3) -> float:
    """Height of the floor in the level base frame.

    Call this while the robot stands on FLAT ground with nothing just ahead.
    Returns nan if there are not enough points.
    """
    x, y, z = points_level[:, 0], points_level[:, 1], points_level[:, 2]
    m = (x >= x_min) & (x <= x_max) & (np.abs(y) <= half_width)
    if m.sum() < 20:
        return math.nan
    return float(np.median(z[m]))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _height_profile(x, h, p: DetectorParams):
    """Top-surface height (80th percentile) of each x bin; nan where too few points."""
    edges = np.arange(p.roi_x_min, p.roi_x_max + 1e-9, p.bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1
    prof = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        sel = idx == i
        if sel.sum() >= p.min_points_per_bin:
            prof[i] = np.percentile(h[sel], 80)
    return centers, prof


def _count_plateaus(heights: np.ndarray, min_rise: float) -> int:
    heights = heights[~np.isnan(heights)]
    if len(heights) == 0:
        return 0
    count, level = 1, heights[0]
    for hh in heights[1:]:
        if hh - level >= min_rise:
            count += 1
            level = hh
    return count


def front_wall_distance(points_level: np.ndarray, p: DetectorParams = DetectorParams(),
                        body_x: np.ndarray = None) -> float:
    """Distance (m, along x) to the nearest tall obstacle straight ahead, or inf if none is seen.

    body_x: the points' x in the (not levelled) base_link frame. Points not in front of the body are dropped
    first: with the body pitched, points behind / above the robot rotate into 'ahead' when levelled."""
    if points_level is None or len(points_level) == 0:
        return math.inf
    x, y, z = points_level[:, 0], points_level[:, 1], points_level[:, 2]
    m = ((x >= p.wall_x_min) & (x <= p.wall_x_max) & (np.abs(y) <= p.wall_half_width)
         & (z >= p.wall_min_z) & (z <= p.wall_max_z))
    if body_x is not None:
        m &= body_x >= p.wall_x_min
    if m.sum() < p.wall_min_points:
        return math.inf
    return float(np.percentile(x[m], 5))


def detect_stairs(points_level: np.ndarray, floor_z: float,
                  p: DetectorParams = DetectorParams()) -> StairObservation:
    """Look for an ascending staircase in front of the robot.

    Uses the normal (floor + first step) detector; if that fails and
    p.close_range is on, tries the close-range step-face detector.
    """
    obs = detect_stairs_far(points_level, floor_z, p)
    if obs.detected or not p.close_range:
        return obs
    close = detect_stairs_close(points_level, floor_z, p)
    if close.detected:
        return close
    close.reason = f'far: {obs.reason}; close: {close.reason}'
    return close


def detect_stairs_far(points_level: np.ndarray, floor_z: float,
                      p: DetectorParams = DetectorParams()) -> StairObservation:
    """Normal detector: needs the floor in front of the first step."""
    if points_level is None or len(points_level) == 0 or math.isnan(floor_z):
        return StairObservation(False, 'no points or floor unknown')

    x, y = points_level[:, 0], points_level[:, 1]
    h = points_level[:, 2] - floor_z
    roi = (x >= p.roi_x_min) & (x <= p.roi_x_max) & (np.abs(y) <= p.roi_half_width)
    x, y, h = x[roi], y[roi], h[roi]
    if len(x) < p.min_roi_points:
        return StairObservation(False, f'only {len(x)} points in ROI')

    # 1) Height profile along x in the central corridor ahead
    corr = np.abs(y) <= p.corridor_half_width
    centers, prof = _height_profile(x[corr], h[corr], p)
    valid = ~np.isnan(prof)
    if valid.sum() < 3:
        return StairObservation(False, 'profile too sparse')

    rise_thresh = 0.6 * p.min_step_height
    edge_i = None
    seen_floor = False
    prev_h = None
    for i in np.nonzero(valid)[0]:
        if prof[i] <= p.floor_tolerance:
            seen_floor = True
            prev_h = prof[i]
            continue
        if prof[i] >= rise_thresh:
            edge_i = i
            break
        prev_h = prof[i]
    if edge_i is None:
        return StairObservation(False, 'no rise ahead (flat)')
    if not seen_floor:
        return StairObservation(False, 'no floor before the rise (too close / on it?)')

    # 2) First tread height = the first step's riser height. The edge bin
    #    itself mixes riser + floor points, so use the next few bins (the
    #    first ~15 cm of tread) instead.
    after = prof[edge_i:]
    after_valid = after[~np.isnan(after)]
    tread = after_valid[1:4] if len(after_valid) >= 2 else after_valid[:1]
    step_h = float(np.median(tread)) - (prev_h or 0.0)
    if step_h > p.max_step_height + p.floor_tolerance:
        return StairObservation(False, f'first rise {step_h:.2f} m too tall (wall?)')
    if step_h < p.min_step_height:
        return StairObservation(False, f'first rise {step_h:.2f} m too small')

    edge_x = centers[edge_i] - 0.5 * p.bin_size

    # 4) Fit the first step edge across y slices: x_edge = a*y + b
    near = (x >= edge_x - 2 * p.bin_size) & (x <= edge_x + 0.35)
    ys, xs = [], []
    y_edges = np.arange(-p.roi_half_width, p.roi_half_width + 1e-9, p.edge_slice_width)
    for j in range(len(y_edges) - 1):
        sl = near & (y >= y_edges[j]) & (y < y_edges[j + 1]) & (h >= 0.5 * step_h)
        if sl.sum() >= 3:
            ys.append(0.5 * (y_edges[j] + y_edges[j + 1]))
            xs.append(float(np.percentile(x[sl], 5)))
    if len(ys) < 3:
        return StairObservation(False, 'cannot fit step edge')
    ys, xs = np.array(ys), np.array(xs)
    a, b = np.polyfit(ys, xs, 1)
    # drop outliers once and refit
    resid = np.abs(xs - (a * ys + b))
    keep = resid < 0.06
    if keep.sum() >= 3:
        ys, xs = ys[keep], xs[keep]
        a, b = np.polyfit(ys, xs, 1)
    width = float(ys.max() - ys.min() + p.edge_slice_width)
    if width < p.min_edge_width:
        return StairObservation(False, f'step only {width:.2f} m wide')

    center_y = float(0.5 * (ys.max() + ys.min()))

    # 5) Nothing in the stair corridor may rise steeper than a staircase could
    behind = (x > edge_x) & (np.abs(y - center_y) <= p.corridor_half_width)
    if behind.any():
        slope = math.tan(math.radians(p.max_stair_slope_deg))
        limit = (x[behind] - edge_x) * slope + p.max_step_height + p.floor_tolerance
        if np.mean(h[behind] > limit) > 0.05:
            return StairObservation(False, 'too steep behind first edge (wall/box)')

    return StairObservation(
        detected=True,
        reason='ok',
        distance=float(a * center_y + b),
        step_height=step_h,
        edge_yaw=float(math.atan(-a)),
        center_y=center_y,
        width=width,
        num_steps=_count_plateaus(after, rise_thresh),
    )


# ---------------------------------------------------------------------------
# Close-range detection (nearest visible step face)
# ---------------------------------------------------------------------------

def detect_stairs_close(points_level: np.ndarray, floor_z: float,
                        p: DetectorParams = DetectorParams()) -> StairObservation:
    """Find the stairs when the floor and first step are in the blind zone.

    Looks only at points above the floor: the nearest vertical face, then at
    least one more, higher level behind it (stairs, not a box), and nothing
    taller than a staircase (not a wall). distance = nearest visible face
    minus close_distance_offset (roughly the 1st step).
    """
    if points_level is None or len(points_level) == 0 or math.isnan(floor_z):
        return StairObservation(False, 'no points or floor unknown')
    x, y = points_level[:, 0], points_level[:, 1]
    h = points_level[:, 2] - floor_z
    roi = ((x >= p.close_x_min) & (x <= p.close_x_max) & (np.abs(y) <= p.roi_half_width)
           & (h >= p.close_min_height))
    x, y, h = x[roi], y[roi], h[roi]

    # 1) centre strip: nearest face + height profile behind it
    c = np.abs(y) <= p.corridor_half_width
    xc, hc = x[c], h[c]
    if len(xc) < p.close_min_points:
        return StairObservation(False, f'close: only {len(xc)} points ahead')
    edges = np.arange(p.close_x_min, p.close_x_max + 1e-9, p.close_bin)
    counts, _ = np.histogram(xc, bins=edges)
    first = np.nonzero(counts >= 3)[0]
    if len(first) == 0:
        return StairObservation(False, 'close: no surface ahead')
    face_x = edges[first[0]]
    seg = (xc >= face_x - p.close_bin) & (xc <= face_x + p.close_depth)
    if seg.sum() < p.close_min_points:
        return StairObservation(False, f'close: only {int(seg.sum())} points on the stairs')
    if np.mean(hc[seg] > p.close_max_height) > 0.05:
        return StairObservation(False, 'close: too tall (wall?)')
    # top-surface height per bin behind the face
    prof = []
    for lo in np.arange(face_x, face_x + p.close_depth, 0.05):
        sb = seg & (xc >= lo) & (xc < lo + 0.05)
        if sb.sum() >= 3:
            prof.append(float(np.percentile(hc[sb], 90)))
    if len(prof) < 2:
        return StairObservation(False, 'close: profile too sparse')
    rise = max(prof[1:]) - prof[0]
    if rise < p.close_min_rise:
        return StairObservation(False, f'close: no second step (rise {rise:.2f} m; box?)')

    # 2) per y slice: x of the nearest face -> edge line (angle) and width (centre)
    near = (x >= face_x - 0.2) & (x <= face_x + 0.25)
    ys, xs = [], []
    sl_edges = np.arange(-p.roi_half_width, p.roi_half_width + 1e-9, p.close_slice_width)
    for a, b in zip(sl_edges[:-1], sl_edges[1:]):
        sl = near & (y >= a) & (y < b)
        if sl.sum() < p.close_slice_min_points:
            continue
        if np.any(h[(y >= a) & (y < b)] > p.close_max_height):
            continue                                  # side wall / post in this slice
        ys.append(0.5 * (a + b))
        xs.append(float(np.percentile(x[sl], 10)))
    if len(ys) < 3:
        return StairObservation(False, 'close: cannot fit step edge')
    ys, xs = np.array(ys), np.array(xs)
    a, b = np.polyfit(ys, xs, 1)
    keep = np.abs(xs - (a * ys + b)) < 0.05
    if keep.sum() >= 3:
        ys, xs = ys[keep], xs[keep]
        a, b = np.polyfit(ys, xs, 1)
    width = float(ys.max() - ys.min() + p.close_slice_width)
    if width < p.min_edge_width:
        return StairObservation(False, f'close: step only {width:.2f} m wide')
    center_y = float(0.5 * (ys.max() + ys.min()))
    return StairObservation(
        detected=True, reason='ok (close)',
        distance=float(a * center_y + b) - p.close_distance_offset,
        step_height=float(rise),
        edge_yaw=float(math.atan(-a)),
        center_y=center_y, width=width, num_steps=len(prof), close=True)
