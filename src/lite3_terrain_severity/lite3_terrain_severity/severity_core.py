"""How rough is the ground ahead, and how fast may the Lite3 go over it? (no ROS here)

Input points are N x 4: x, y, z in base_link and h = height above the fitted ground plane
(exactly what terrain_classifier publishes on ~/lidar_points as intensity).

Levels: 0 smooth, 1 mild, 2 moderate, 3 severe, 4 extreme (pause).
"""
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

LEVELS = ('smooth', 'mild', 'moderate', 'severe', 'extreme')
SMOOTH, MILD, MODERATE, SEVERE, EXTREME = range(5)

RUNNING, CREEP, PAUSED, BLOCKED = 'RUNNING', 'CREEP', 'PAUSED', 'BLOCKED'


@dataclass
class SeverityParams:
    # ---- where to look (base_link, metres) ----
    cell_m: float = 0.05
    x_start_m: float = 0.30          # front of the body footprint
    lookahead_m: float = 1.2
    half_width_m: float = 0.28       # Lite3 is ~0.4 m wide, plus margin
    turn_radius_m: float = 0.70      # region checked when turning on the spot
    min_cells: int = 12              # fewer observed cells => "not enough data"
    # ---- thresholds for levels mild, moderate, severe, extreme ----
    edge_m: list = field(default_factory=lambda: [0.03, 0.06, 0.10, 0.15])   # step between cells
    bump_m: list = field(default_factory=lambda: [0.04, 0.08, 0.12, 0.18])   # highest point
    hole_m: list = field(default_factory=lambda: [0.04, 0.07, 0.11, 0.16])   # deepest dip
    roughness_m: list = field(default_factory=lambda: [0.012, 0.025, 0.045, 0.07])  # height std
    rough_area: list = field(default_factory=lambda: [0.05, 0.15, 0.30, 0.50])  # share of cells off-ground
    rough_area_h_m: float = 0.04     # what counts as "off-ground" for rough_area
    extreme_within_m: float = 0.70   # geometry only says EXTREME once the hazard is this close
    # ---- proprioception (the robot is actually struggling) ----
    imu_std_deg: list = field(default_factory=lambda: [1.5, 3.0, 5.0, 8.0])  # roll/pitch wobble over window
    slip_mps: list = field(default_factory=lambda: [0.10, 0.20, 0.30, 0.45])  # |cmd vx - odom vx|
    roll_alarm_deg: float = 18.0     # body roll beyond this => pause immediately
    pitch_alarm_deg: float = 30.0    # pitch alarm is high because ramps are legal terrain
    # ---- what the other classifier's label does ----
    non_rugged_cap: int = MILD       # classifier says non_rugged and robot is steady => at most this
    unknown_level: int = MODERATE    # no usable data => at least this


@dataclass
class GovernorParams:
    speed_factor: list = field(default_factory=lambda: [1.0, 0.7, 0.45, 0.25, 0.0])  # per level
    creep_factor: float = 0.15       # speed while edging forward after a pause
    turn_exponent: float = 0.5       # angular factor = linear factor ** this (turns slow down less)
    min_turn_factor: float = 0.3     # turning never drops below this factor; also the BLOCKED turn speed
    down_hold_s: float = 1.0         # a lower level must last this long before speeding up
    factor_rise_per_s: float = 0.5   # speed back up gently, slow down instantly
    pause_s: float = 1.5             # stand still, let the gait settle and the sensors re-scan
    creep_s: float = 2.0             # then edge forward this long before pausing again
    max_pauses: int = 3              # still extreme after this many pauses => BLOCKED


def level_of(value, thresholds):
    if value is None or not np.isfinite(value):
        return SMOOTH
    return int(sum(value >= t for t in thresholds))


def _region(pts, direction, P):
    x, y = pts[:, 0], pts[:, 1]
    if direction == 'backward':
        m = (x <= -P.x_start_m) & (x >= -P.x_start_m - P.lookahead_m) & (np.abs(y) <= P.half_width_m)
        xs = -x
    elif direction == 'turn':
        m = np.hypot(x, y) <= P.turn_radius_m
        xs = np.hypot(x, y)
    else:
        m = (x >= P.x_start_m) & (x <= P.x_start_m + P.lookahead_m) & (np.abs(y) <= P.half_width_m)
        xs = x
    return pts[m], xs[m]


def assess_geometry(pts, direction='forward', P=None):
    """pts: N x 4 (x, y, z, h). Returns (level, metrics dict)."""
    P = P or SeverityParams()
    if pts is None or len(pts) == 0:
        return None, {'reason': 'no points'}
    sel, dist = _region(np.asarray(pts, float), direction, P)
    if len(sel) == 0:
        return None, {'reason': f'no points in {direction} region'}

    # grid in the region's own (x, y); for 'turn' this is just a square around the robot
    gx = np.floor(sel[:, 0] / P.cell_m).astype(int)
    gy = np.floor(sel[:, 1] / P.cell_m).astype(int)
    gx -= gx.min()
    gy -= gy.min()
    nx, ny = gx.max() + 1, gy.max() + 1
    flat = gx * ny + gy
    h = sel[:, 3]
    hmax = np.full(nx * ny, -np.inf)
    hmin = np.full(nx * ny, np.inf)
    np.maximum.at(hmax, flat, h)
    np.minimum.at(hmin, flat, h)
    near = np.full(nx * ny, np.inf)
    np.minimum.at(near, flat, dist)
    seen = np.isfinite(hmax)
    n_cells = int(seen.sum())
    if n_cells < P.min_cells:
        return None, {'reason': f'only {n_cells} cells observed', 'cells': n_cells}

    top = hmax[seen]
    bot = hmin[seen]
    srt_top, srt_bot = np.sort(top), np.sort(bot)
    bump = float(srt_top[-2]) if top.size > 1 else float(top[0])     # 2nd highest: ignore one stray point
    hole = float(-srt_bot[1]) if bot.size > 1 else float(-bot[0])
    hole = max(hole, 0.0)
    roughness = float(np.std(top))
    off = (top > P.rough_area_h_m) | (bot < -P.rough_area_h_m)
    rough_area = float(off.mean())

    grid = np.where(seen, hmax, np.nan).reshape(nx, ny)
    diffs = np.concatenate([np.abs(np.diff(grid, axis=0)).ravel(), np.abs(np.diff(grid, axis=1)).ravel()])
    diffs = np.sort(diffs[np.isfinite(diffs)])
    edge = float(diffs[-3]) if diffs.size >= 3 else 0.0             # 3rd largest: ignore noise

    levels = {
        'edge': level_of(edge, P.edge_m), 'bump': level_of(bump, P.bump_m),
        'hole': level_of(hole, P.hole_m), 'roughness': level_of(roughness, P.roughness_m),
        'rough_area': level_of(rough_area, P.rough_area),
    }
    level = max(levels.values())

    # how far away is the worst stuff? far-away extremes only mean "severe, slow down"
    # EXTREME (=> pause) needs a real tall bump or deep hole close ahead. Lots of medium
    # blocks or a far-away hazard only mean SEVERE: slow right down and keep going.
    tall = min(P.bump_m[3], P.edge_m[3])
    hazard = (hmax >= tall) | (hmin <= -P.hole_m[3])
    hazard_cells = seen & hazard
    hazard_dist = float(near[hazard_cells].min()) if hazard_cells.any() else None
    if level == EXTREME and (hazard_dist is None or hazard_dist > P.extreme_within_m):
        level = SEVERE

    metrics = dict(direction=direction, cells=n_cells, edge=round(edge, 3), bump=round(bump, 3),
                   hole=round(hole, 3), roughness=round(roughness, 3), rough_area=round(rough_area, 3),
                   hazard_dist=None if hazard_dist is None else round(hazard_dist, 2), levels=levels)
    return level, metrics


def assess_proprio(roll_deg, pitch_deg, imu_std_deg, slip, P=None):
    """Returns (level, alarm, metrics). alarm=True => pause right now (tipping)."""
    P = P or SeverityParams()
    levels = {'imu_std': level_of(imu_std_deg, P.imu_std_deg), 'slip': level_of(slip, P.slip_mps)}
    alarm = bool(abs(roll_deg) >= P.roll_alarm_deg or abs(pitch_deg) >= P.pitch_alarm_deg)
    return max(levels.values()), alarm, dict(roll=round(float(roll_deg), 1), pitch=round(float(pitch_deg), 1),
                                              imu_std=round(float(imu_std_deg), 2), slip=round(float(slip), 3),
                                              levels=levels, tilt_alarm=alarm)


def combine(geo_level, proprio_level, terrain_type, P=None):
    """Merge geometry, proprioception and the rugged/non_rugged label into one level."""
    P = P or SeverityParams()
    if geo_level is None:                       # can't see the ground: be careful
        geo_level = P.unknown_level
    if terrain_type == 'non_rugged' and proprio_level <= SMOOTH:
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
        self._higher_seen_t = -1e9     # last time raw level was >= current debounced level
        self._state_until = 0.0
        self._creep_until = 0.0
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
        if target < self.factor:
            self.factor = target                                 # slow down at once
        else:
            self.factor = min(target, self.factor + G.factor_rise_per_s * self._dt(now))
        self._last_t = now
        return self.state, self.factor

    def _dt(self, now):
        last = getattr(self, '_last_t', now)
        return max(0.0, min(0.5, now - last))

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