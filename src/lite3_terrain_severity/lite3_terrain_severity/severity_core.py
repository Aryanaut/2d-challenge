"""How rough is the ground ahead, and how fast may the Lite3 go over it? (no ROS here)

Input points are N x 4: x, y, z in base_link and h = height above the ground plane fitted by
terrain_classifier (its ~/lidar_points debug cloud, intensity channel).

Three levels:
  0 smooth   -> normal speed
  1 rough    -> slow down
  2 extreme  -> pause, then creep (a step / drop the Lite3 cannot handle, or the robot tipping)

Maze-safe: anything taller than wall_min_h_m (the 50 cm OSB walls) is treated as a wall, not as
terrain, and is removed together with a margin around its base before anything is measured.
The ground level is then re-fitted locally, so a ground plane tilted by walls doesn't look like a bump.
"""
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

LEVELS = ('smooth', 'rough', 'extreme')
SMOOTH, ROUGH, EXTREME = range(3)

RUNNING, CREEP, PAUSED, BLOCKED = 'RUNNING', 'CREEP', 'PAUSED', 'BLOCKED'


@dataclass
class SeverityParams:
    # ---- where to look (base_link, metres) ----
    cell_m: float = 0.05
    x_start_m: float = 0.30          # front of the body footprint
    lookahead_m: float = 1.0         # how far ahead to grade
    half_width_m: float = 0.25       # Lite3 is ~0.4 m wide; corridor is 1 m
    turn_radius_m: float = 0.55      # region checked when turning on the spot
    min_cells: int = 12              # fewer usable cells => "not enough data"
    # ---- walls (not terrain) ----
    wall_min_h_m: float = 0.30       # taller than this = wall / obstacle, ignored here
    wall_margin_cells: int = 2       # also ignore this many cells around a wall's base
    # ---- ROUGH if any of these is reached ----
    rough_edge_m: float = 0.06       # step between neighbouring 5 cm cells
    rough_bump_m: float = 0.08       # highest point above the local ground
    rough_hole_m: float = 0.08       # deepest dip below the local ground
    rough_std_m: float = 0.03        # spread of heights (bumpy surface)
    rough_area: float = 0.20         # share of cells more than rough_area_h_m off the ground
    rough_area_h_m: float = 0.05
    # ---- EXTREME (pause) ----
    extreme_edge_m: float = 0.20     # a single step/drop bigger than the Lite3 can take (~15-18 cm)
    extreme_within_m: float = 0.60   # ... only when it is this close; farther away => ROUGH
    # ---- proprioception (the robot itself is struggling) ----
    rough_imu_std_deg: float = 3.0   # roll/pitch wobble over the window
    extreme_imu_std_deg: float = 8.0
    rough_slip_mps: float = 0.25     # |commanded vx - measured vx|
    roll_alarm_deg: float = 18.0     # body roll beyond this => pause immediately
    pitch_alarm_deg: float = 30.0    # high, because ramps are legal terrain
    # ---- how the other classifier's label is used ----
    non_rugged_cap: int = SMOOTH     # classifier says non_rugged and robot is steady => at most this
    unknown_level: int = ROUGH       # no usable data => at least this


@dataclass
class GovernorParams:
    speed_factor: list = field(default_factory=lambda: [1.0, 0.4, 0.0])  # smooth, rough, extreme
    creep_factor: float = 0.15       # speed while edging forward after a pause
    turn_exponent: float = 0.5       # angular factor = linear factor ** this (turns slow down less)
    min_turn_factor: float = 0.3     # turning never drops below this; also the BLOCKED turn speed
    down_hold_s: float = 1.0         # a lower level must last this long before speeding up
    factor_rise_per_s: float = 0.5   # speed back up gently, slow down instantly
    pause_s: float = 1.5             # stand still, let the gait settle and the sensors re-scan
    creep_s: float = 2.0             # then edge forward this long before pausing again
    max_pauses: int = 3              # still extreme after this many pauses => BLOCKED


def _region(pts, direction, P):
    x, y = pts[:, 0], pts[:, 1]
    if direction == 'backward':
        m = (x <= -P.x_start_m) & (x >= -P.x_start_m - P.lookahead_m) & (np.abs(y) <= P.half_width_m)
        dist = -x
    elif direction == 'turn':
        dist = np.hypot(x, y)
        m = dist <= P.turn_radius_m
    else:
        m = (x >= P.x_start_m) & (x <= P.x_start_m + P.lookahead_m) & (np.abs(y) <= P.half_width_m)
        dist = x
    return pts[m], dist[m]


def _dilate(mask, n):
    out = mask.copy()
    for _ in range(n):
        m = out.copy()
        m[1:, :] |= out[:-1, :]
        m[:-1, :] |= out[1:, :]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


def assess_geometry(pts, direction='forward', P=None):
    """pts: N x 4 (x, y, z, h). Returns (level or None, metrics dict)."""
    P = P or SeverityParams()
    if pts is None or len(pts) == 0:
        return None, {'reason': 'no points'}
    sel, dist = _region(np.asarray(pts, float), direction, P)
    if len(sel) == 0:
        return None, {'reason': f'no points in {direction} region'}

    # ---- height grid: highest point per 5 cm cell (= where a foot lands) ----
    x0, y0 = sel[:, 0].min(), sel[:, 1].min()
    gx = np.floor((sel[:, 0] - x0) / P.cell_m).astype(int)
    gy = np.floor((sel[:, 1] - y0) / P.cell_m).astype(int)
    nx, ny = gx.max() + 1, gy.max() + 1
    flat = gx * ny + gy
    hmax = np.full(nx * ny, -np.inf)
    near = np.full(nx * ny, np.inf)
    np.maximum.at(hmax, flat, sel[:, 3])
    np.minimum.at(near, flat, dist)
    seen = np.isfinite(hmax).reshape(nx, ny)
    hmax = hmax.reshape(nx, ny)
    near = near.reshape(nx, ny)

    # ---- remove walls (and the band at their base) ----
    wall = seen & (hmax >= P.wall_min_h_m)
    wall_zone = _dilate(wall, P.wall_margin_cells)
    use = seen & ~wall_zone
    n_cells = int(use.sum())
    base = dict(direction=direction, cells=n_cells, wall_cells=int(wall.sum()))
    if n_cells < P.min_cells:
        return None, {**base, 'reason': f'only {n_cells} ground cells visible (rest is wall or unseen)'}

    # ---- re-fit the ground locally (robust: fit, drop outliers, fit again) ----
    cx = (np.arange(nx) + 0.5) * P.cell_m + x0
    cy = (np.arange(ny) + 0.5) * P.cell_m + y0
    X, Y = np.meshgrid(cx, cy, indexing='ij')
    A = np.column_stack([X[use], Y[use], np.ones(n_cells)])
    z = hmax[use]
    keep = np.ones(n_cells, bool)
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        r = z - A @ coef
        keep = np.abs(r - np.median(r)) < max(0.03, 2.5 * np.std(r[keep]))
        if keep.sum() < 6:
            keep[:] = True
            break
    resid = z - A @ coef
    tilt_deg = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
    if tilt_deg > 20.0:          # fit went wrong (e.g. mostly a staircase): fall back to a flat offset
        resid = z - np.median(z)
        tilt_deg = 0.0
    hloc = np.full((nx, ny), np.nan)
    hloc[use] = resid

    s_top, s_bot = np.sort(resid), np.sort(resid)
    bump = float(s_top[-2]) if n_cells > 1 else float(s_top[-1])     # 2nd highest: ignore 1 stray point
    hole = max(0.0, float(-s_bot[1]) if n_cells > 1 else float(-s_bot[0]))
    spread = float(np.std(resid))
    area = float(np.mean(np.abs(resid) > P.rough_area_h_m))

    # ---- step edges between neighbouring ground cells ----
    raw = np.where(use, hmax, np.nan)
    ex = np.abs(np.diff(raw, axis=0))
    ey = np.abs(np.diff(raw, axis=1))
    all_e = np.concatenate([ex[np.isfinite(ex)], ey[np.isfinite(ey)]])
    all_e.sort()
    edge = float(all_e[-3]) if all_e.size >= 3 else (float(all_e[-1]) if all_e.size else 0.0)

    # distance to the closest "too big" edge
    big = np.zeros((nx, ny), bool)
    with np.errstate(invalid='ignore'):
        bx = ex >= P.extreme_edge_m
        by = ey >= P.extreme_edge_m
    big[:-1, :] |= bx
    big[1:, :] |= bx
    big[:, :-1] |= by
    big[:, 1:] |= by
    big &= use
    # require at least 2 such cells so one noisy cell can't stop the robot
    big_dist = float(np.sort(near[big])[1]) if big.sum() >= 2 else None

    reasons = []
    if edge >= P.rough_edge_m:
        reasons.append(f'edge {edge:.2f}')
    if bump >= P.rough_bump_m:
        reasons.append(f'bump {bump:.2f}')
    if hole >= P.rough_hole_m:
        reasons.append(f'hole {hole:.2f}')
    if spread >= P.rough_std_m:
        reasons.append(f'spread {spread:.3f}')
    if area >= P.rough_area:
        reasons.append(f'area {area:.2f}')
    level = ROUGH if reasons else SMOOTH
    if big_dist is not None and big_dist <= P.extreme_within_m:
        level = EXTREME
        reasons.insert(0, f'step >= {P.extreme_edge_m:.2f} m at {big_dist:.2f} m')

    return level, {**base, 'edge': round(edge, 3), 'bump': round(bump, 3), 'hole': round(hole, 3),
                   'spread': round(spread, 3), 'area': round(area, 3), 'ground_tilt_deg': round(tilt_deg, 1),
                   'big_step_dist': None if big_dist is None else round(big_dist, 2),
                   'why': ', '.join(reasons) or 'flat'}


def assess_proprio(roll_deg, pitch_deg, imu_std_deg, slip, P=None):
    """Returns (level, alarm, metrics). alarm=True => pause right now (tipping)."""
    P = P or SeverityParams()
    roll_deg, pitch_deg, imu_std_deg, slip = (float(v) for v in (roll_deg, pitch_deg, imu_std_deg, slip))
    level = SMOOTH
    if imu_std_deg >= P.rough_imu_std_deg or slip >= P.rough_slip_mps:
        level = ROUGH
    if imu_std_deg >= P.extreme_imu_std_deg:
        level = EXTREME
    alarm = bool(abs(roll_deg) >= P.roll_alarm_deg or abs(pitch_deg) >= P.pitch_alarm_deg)
    return level, alarm, dict(roll=round(roll_deg, 1), pitch=round(pitch_deg, 1),
                              imu_std=round(imu_std_deg, 2), slip=round(slip, 3),
                              level=LEVELS[level], tilt_alarm=alarm)


def combine(geo_level, proprio_level, terrain_type, P=None):
    """Merge geometry, proprioception and the rugged/non_rugged label into one level."""
    P = P or SeverityParams()
    if geo_level is None:                       # can't see the ground: be careful
        geo_level = P.unknown_level
    if terrain_type == 'non_rugged' and proprio_level == SMOOTH and geo_level < EXTREME:
        geo_level = min(geo_level, P.non_rugged_cap)
    if terrain_type == 'unknown':
        geo_level = max(geo_level, P.unknown_level)
    return max(geo_level, proprio_level)


class Governor:
    """Turns the level stream into a speed factor plus a RUNNING/CREEP/PAUSED/BLOCKED state."""

    def __init__(self, G=None):
        self.G = G or GovernorParams()
        self.state = RUNNING
        self.level = SMOOTH            # debounced level
        self.factor = 1.0
        self._higher_seen_t = -1e9
        self._state_until = 0.0
        self._creep_until = 0.0
        self._last_t = None
        self.pauses = 0

    def _debounce(self, raw, now):
        if raw >= self.level:
            self.level, self._higher_seen_t = raw, now
        elif now - self._higher_seen_t >= self.G.down_hold_s:
            self.level, self._higher_seen_t = raw, now
        return self.level

    def update(self, raw_level, tilt_alarm, now):
        G = self.G
        lvl = self._debounce(raw_level, now)

        if tilt_alarm and self.state != PAUSED:
            self.state, self._state_until = PAUSED, now + G.pause_s
        elif self.state == PAUSED:
            if now >= self._state_until:
                if lvl < EXTREME and not tilt_alarm:
                    self.state, self.pauses = RUNNING, 0
                elif self.pauses >= G.max_pauses:
                    self.state = BLOCKED
                else:
                    self.state, self._creep_until = CREEP, now + G.creep_s
        elif self.state == BLOCKED:
            if lvl < EXTREME:
                self.state, self.pauses = RUNNING, 0
        elif lvl == EXTREME:
            if self.state == RUNNING or (self.state == CREEP and now >= self._creep_until):
                self.pauses += 1
                self.state, self._state_until = PAUSED, now + G.pause_s
        else:
            self.state, self.pauses = RUNNING, 0

        target = {RUNNING: G.speed_factor[lvl], CREEP: G.creep_factor, PAUSED: 0.0, BLOCKED: 0.0}[self.state]
        dt = 0.0 if self._last_t is None else max(0.0, min(0.5, now - self._last_t))
        self._last_t = now
        if target < self.factor:
            self.factor = target                                 # slow down at once
        else:
            self.factor = min(target, self.factor + G.factor_rise_per_s * dt)
        return self.state, self.factor

    def scale(self, vx, vy, wz):
        """Apply the current factor. BLOCKED still lets the robot back up and turn away."""
        G, f = self.G, self.factor
        if self.state == BLOCKED:
            return min(vx, 0.0) * G.creep_factor, vy * G.creep_factor, wz * G.min_turn_factor
        if self.state == PAUSED or f <= 0.0:
            return 0.0, 0.0, 0.0
        return vx * f, vy * f, wz * max(f ** G.turn_exponent, G.min_turn_factor)

    def reset(self):
        self.__init__(self.G)


class Window:
    """Rolling roll/pitch window for the IMU wobble metric."""

    def __init__(self, seconds=1.5):
        self.s, self.buf = seconds, deque()

    def push(self, t, roll, pitch):
        self.buf.append((t, roll, pitch))
        while self.buf and t - self.buf[0][0] > self.s:
            self.buf.popleft()

    def std_deg(self):
        if len(self.buf) < 5:
            return 0.0
        a = np.array([(r, p) for _, r, p in self.buf])
        return float(np.degrees(a.std(axis=0).max()))


def quat_to_roll_pitch(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch