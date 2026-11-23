"""Optional elevation of the floor relative to a ground reference taken at the start of the run. No ROS, no tiles.

At the start the robot is assumed to stand on flat floor. GroundReference collects the floor points the lidar sees in
that time (in a fixed frame, see below) and fits one plane to them: the ground reference. Later, ElevationEstimator
collects the floor points of a patch (the same patch the terrain kind is computed from) and reports how far that
floor sits above (+) or below (-) the reference plane, plus the slope of the patch against the reference.

Frames. Points must be in a frame whose z axis points up (gravity), so the robot's own pitch and roll while walking do
not change the answer:
  * `odom`    : the real odom frame. The elevation is ABSOLUTE (relative to the start floor), even when the robot stands
                on a raised tile. Needs an odometry whose z is real (leg or lidar-inertial odometry).
  * `gravity` : base_link rotated into the odom orientation, no translation. Works with planar odometry (z always 0),
                but then the elevation is relative to the floor the robot STANDS ON now plus its start height: on a raised
                tile everything reads that much lower. Use it when odom z is not available.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from lite3_terrain_classifier import terrain_core as tc

DEFAULTS = {
    'calibrate_s': 2.0,            # collect this long at the start for the reference plane
    'min_points': 300,             # points needed for the reference
    'max_rms_m': 0.03,             # the reference floor must be this flat (rms of the points around the fitted plane)
    'max_tilt_deg': 10.0,          # ... and this close to level (the robot is assumed to stand flat)
    'level_tolerance_m': 0.04,     # a patch within this of the reference is 'level', otherwise 'higher' / 'lower'
    'inlier_band_m': 0.08,         # patch points further than this from the patch's own dominant level are ignored
    'min_patch_points': 150,
    'upper_level_fraction': 0.2,   # a level this much higher than the dominant one counts as the patch's floor when it holds
                                   # this fraction of the dominant level's points (a step in the patch: the tile is 'higher')
    'max_points': 30000,
}


class GroundReference:
    """The reference ground plane, fitted from the points collected while the robot stands on flat floor."""

    def __init__(self, cfg: Optional[dict] = None, rng: Optional[np.random.Generator] = None):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.rng = rng or np.random.default_rng(0)
        self.normal: Optional[np.ndarray] = None
        self.d = 0.0
        self.info: dict = {}
        self.reset()

    @property
    def ready(self) -> bool:
        return self.normal is not None

    def reset(self):
        """Forget the reference and start collecting again."""
        self.normal, self.d, self.info = None, 0.0, {}
        self._chunks: list = []
        self._count = 0
        self.started_s: Optional[float] = None

    def add(self, points, now_s: float):
        """Collect floor points (N, 3) in the fixed up-is-z frame while calibrating."""
        if self.ready:
            return
        if self.started_s is None:
            self.started_s = now_s
        points = np.asarray(points, float)
        points = points[np.isfinite(points).all(axis=1)] if len(points) else points.reshape(0, 3)
        if len(points):
            self._chunks.append(points)
            self._count += len(points)

    def finish(self, now_s: float):
        """When calibrate_s has passed: fit the plane. Returns None (still collecting), True (reference ready) or
        a string (why it was refused; collection restarts)."""
        if self.ready:
            return True
        if self.started_s is None or now_s - self.started_s < float(self.cfg['calibrate_s']):
            return None
        pts = np.concatenate(self._chunks) if self._chunks else np.zeros((0, 3))
        n = len(pts)
        self._chunks, self._count, self.started_s = [], 0, None
        if n < int(self.cfg['min_points']):
            return f'only {n} floor points (need {self.cfg["min_points"]})'
        if n > int(self.cfg['max_points']):
            pts = pts[self.rng.choice(n, int(self.cfg['max_points']), replace=False)]
        up = np.array([0.0, 0.0, 1.0])
        plane = tc.fit_plane(pts, 60, 0.03, up, 35.0, self.rng)
        if plane is None:
            return 'plane fit failed'
        normal, d = plane
        if normal @ up < 0:
            normal, d = -normal, -d
        res = pts @ normal + d
        inl = np.abs(res) < 0.10
        if inl.sum() < int(self.cfg['min_points']):
            return 'the floor points do not form one plane'
        rms = float(np.sqrt(np.mean(res[inl] ** 2)))
        tilt = math.degrees(math.acos(min(1.0, float(normal @ up))))
        if rms > float(self.cfg['max_rms_m']):
            return f'the start floor is not flat (rms {rms:.3f} m > {self.cfg["max_rms_m"]} m)'
        if tilt > float(self.cfg['max_tilt_deg']):
            return f'the start floor is tilted {tilt:.1f} deg (max {self.cfg["max_tilt_deg"]})'
        self.normal, self.d = normal, float(d)
        self.info = {'normal': [round(float(v), 4) for v in normal], 'd': round(self.d, 4), 'n_points': int(inl.sum()),
                     'rms_m': round(rms, 4), 'tilt_deg': round(tilt, 2)}
        return True

    def height(self, points) -> np.ndarray:
        """Signed height (m) of each point above the reference plane."""
        return np.asarray(points, float) @ self.normal + self.d


class ElevationEstimator:
    """Elevation of one patch of floor against the ground reference. reset() starts a new patch."""

    def __init__(self, reference: GroundReference, cfg: Optional[dict] = None):
        self.ref = reference
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.reset()

    def reset(self):
        self._pts = np.zeros((0, 3))

    def add(self, points):
        """points (N, 3): floor points of the patch in the same fixed frame as the reference (no walls / objects)."""
        if not self.ref.ready:
            return
        points = np.asarray(points, float)
        if len(points) == 0:
            return
        self._pts = np.vstack([self._pts, points])
        if len(self._pts) > int(self.cfg['max_points']):
            self._pts = self._pts[np.random.default_rng(0).choice(len(self._pts), int(self.cfg['max_points']),
                                                                   replace=False)]

    @staticmethod
    def _dominant_level(h, band: float) -> float:
        """The densest 2 * band window of heights (stray points on a wall or a step edge drop out): its median."""
        order = np.sort(h)
        hi = np.searchsorted(order, order + 2 * band, side='right')
        start = int(np.argmax(hi - np.arange(len(order))))
        return float(np.median(order[start:hi[start]]))

    def estimate(self) -> dict:
        """{} while there is no reference or too few points, else the elevation of the patch:
        elevation_m (+ higher than the start floor), elevation_class, slope_deg, elevation_spread_m, elevation_points."""
        if not self.ref.ready:
            return {}
        pts = self._pts
        if len(pts) < int(self.cfg['min_patch_points']):
            return {}
        h = self.ref.height(pts)
        band = float(self.cfg['inlier_band_m'])
        centre = self._dominant_level(h, band)
        sel = np.abs(h - centre) <= band
        # A step inside the patch: the robot's own (nearer, denser) floor wins the dominant level, so the raised part would be
        # missed and the tile read 'level'. Take the highest level that holds enough points as the patch's floor instead.
        frac = float(self.cfg['upper_level_fraction'])
        n_dom, upper = int(sel.sum()), h > centre + 1.5 * band
        while frac > 0 and upper.sum() >= max(int(self.cfg['min_patch_points']) // 2, frac * n_dom):
            c2 = self._dominant_level(h[upper], band)
            sel2 = np.abs(h - c2) <= band
            if sel2.sum() < max(int(self.cfg['min_patch_points']) // 2, frac * n_dom):
                break
            centre, sel = c2, sel2
            upper = h > centre + 1.5 * band
        hs, ps = h[sel], pts[sel]
        if len(hs) < int(self.cfg['min_patch_points']) // 2:
            return {}
        elevation = float(np.median(hs))
        # slope of the patch against the reference plane: least squares h = a u + b v + c in the plane's own axes
        u, v = tc._plane_basis(self.ref.normal)
        design = np.column_stack((ps @ u, ps @ v, np.ones(len(ps))))
        slope = 0.0
        try:
            (a, b, _c), *_ = np.linalg.lstsq(design, hs, rcond=None)
            slope = math.degrees(math.atan(math.hypot(float(a), float(b))))
        except np.linalg.LinAlgError:
            pass
        tol = float(self.cfg['level_tolerance_m'])
        cls = 'level' if abs(elevation) <= tol else ('higher' if elevation > 0 else 'lower')
        return {'elevation_m': round(elevation, 3), 'elevation_class': cls, 'slope_deg': round(slope, 1),
                'elevation_spread_m': round(float(np.percentile(hs, 90) - np.percentile(hs, 10)), 3),
                'elevation_points': int(len(hs))}
