"""Terrain KINDS (flat / uneven / slats / ...) as configurable strategies. No ROS, no tiles: it classifies a patch of
floor around the robot, and does not know about the explorer's tile grid.

The classifier node feeds every assessment (lidar points in base_link and their height above the fitted floor plane)
to a KindClassifier; classify() gives the kind. A kind is a strategy class registered with @register_terrain('type');
the first strategy in the configured list (config/terrain_kinds.yaml) whose match() accepts the stats wins:

    terrains:                     # priority order
      - {name: slats,  type: slats,  ...}
      - {name: uneven, type: uneven, ...}
      - {name: flat,   type: flat}

To add a kind write a TerrainStrategy subclass (match(stats) -> info dict or None), register it and list it.

Numbers (bag terrains_20_11_091239, 0.3 m cells, spread = p90 - p10 of height): flat floor 0.04-0.08 m median spread,
slats and uneven floor 0.18-0.19 m. The slats-versus-uneven split (`coherence`) is NOT validated: both rugged tiles
there scored about 0.2, so the slats threshold is a starting value.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Type

import numpy as np

from lite3_terrain_classifier import slat_lines

# measurement settings shared by all strategies (`common:` in the config)
DEFAULTS = {
    'cell_m': 0.30,               # spread is measured per cell of this size
    'min_points_per_cell': 6,
    'wall_height_m': 0.30,        # a cell with a point higher than this holds a wall / object: not floor
    'min_cells': 5,               # floor cell measurements needed before there is a kind
    'spread_threshold_m': 0.08,   # a cell is rough when its p90 - p10 height exceeds this
    'coherence_bin_m': 0.02,
    'max_floor_points': 20000,
    'image_window': 15,           # per camera: how many of the latest frames take part in the slats vote
}

DEFAULT_TERRAINS = [
    {'name': 'slats', 'type': 'hough_slats', 'sources': ['depth', 'color'], 'mode': 'any',
     'marker': {'style': 'solid', 'color': [1.0, 0.0, 0.0, 0.55], 'label': 'SLATS'}},
    {'name': 'uneven', 'type': 'uneven', 'rugged_median_m': 0.12, 'rugged_fraction': 0.60,
     'marker': {'style': 'hatch', 'color': [1.0, 0.75, 0.0, 1.0], 'label': 'UNEVEN'}},
    {'name': 'flat', 'type': 'flat', 'marker': {'style': 'none'}},
]


def line_coherence(x, y, h, bin_m: float = 0.02):
    """Share of the height variance explained by averaging along parallel lines, over all directions.

    Parallel ridges (slats) average coherently along their own direction, so one angle scores high; random
    bumps average out at every angle. Returns (share, angle_deg of the best line direction normal)."""
    total = float(((h - h.mean()) ** 2).sum())
    if total < 1e-9 or len(h) < 100:
        return 0.0, 0.0
    best = (0.0, 0.0)
    for th in np.radians(np.arange(0, 180, 5)):
        d = x * math.cos(th) + y * math.sin(th)
        b = np.floor(d / bin_m).astype(int)
        b -= b.min()
        cnt = np.bincount(b)
        ok = cnt >= 3
        if ok.sum() < 8:
            continue
        prof = np.bincount(b, weights=h)[ok] / cnt[ok]
        w = cnt[ok]
        mu = float((prof * w).sum() / w.sum())
        r = float((w * (prof - mu) ** 2).sum()) / total
        if r > best[0]:
            best = (r, math.degrees(th))
    return best


# ------------------------------------------------------------------------------------------ strategies
class TerrainStats:
    """What the strategies get to look at for one tile."""

    def __init__(self, spreads, floor_points, cfg, rng, images=None):
        self.images = images or {}                # camera name -> detections of its latest frames (slat_lines.detect)
        self.spreads = np.asarray(spreads, float)
        self.floor = floor_points                 # (N, 3) u, v, height in metres, or None
        self.cfg, self.rng = cfg, rng
        self.n_cells = len(self.spreads)
        self.median = float(np.median(self.spreads)) if self.n_cells else 0.0
        self.rough_fraction = float((self.spreads > cfg['spread_threshold_m']).mean()) if self.n_cells else 0.0
        self._coh = None

    def coherence(self):
        """(share of the height variance explained by parallel lines, best angle) - computed once, on demand."""
        if self._coh is None:
            fl = self.floor
            if fl is None or len(fl) < 300:
                self._coh = (0.0, 0.0)
            else:
                if len(fl) > 3000:
                    fl = fl[self.rng.choice(len(fl), 3000, replace=False)]
                self._coh = line_coherence(fl[:, 0], fl[:, 1], fl[:, 2], self.cfg['coherence_bin_m'])
        return self._coh


TERRAIN_TYPES: Dict[str, Type['TerrainStrategy']] = {}


def register_terrain(type_name: str):
    def deco(cls):
        cls.type_name = type_name
        TERRAIN_TYPES[type_name] = cls
        return cls
    return deco


class TerrainStrategy:
    """One terrain kind. cfg is its entry in the config (name, type, its own parameters, marker)."""
    type_name = 'base'
    needs_cells = True          # False: this kind does not need lidar floor cells (e.g. image based)

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg)
        self.name = self.cfg.get('name', self.type_name)
        self.marker = dict(self.cfg.get('marker') or {})

    def match(self, stats: TerrainStats) -> Optional[dict]:
        """None = not this terrain; a dict (measurements to report) = it is."""
        raise NotImplementedError

    def _rugged(self, stats) -> bool:
        return (stats.median >= float(self.cfg.get('rugged_median_m', 0.12))
                and stats.rough_fraction >= float(self.cfg.get('rugged_fraction', 0.60)))


@register_terrain('flat')
class FlatTerrain(TerrainStrategy):
    """Matches everything the earlier strategies did not take (or, with rugged_* set, only small spreads)."""

    def match(self, stats):
        if 'rugged_median_m' in self.cfg and self._rugged(stats):
            return None
        return {}


@register_terrain('uneven')
class UnevenTerrain(TerrainStrategy):
    """Rugged floor: a large height spread in most cells."""

    def match(self, stats):
        return {} if self._rugged(stats) else None


@register_terrain('slats')
class SlatsTerrain(TerrainStrategy):
    """Rugged floor whose height is made of straight parallel ridges (one direction explains min_coherence)."""

    def match(self, stats):
        if not self._rugged(stats):
            return None
        coh, ang = stats.coherence()
        if coh < float(self.cfg.get('min_coherence', 0.35)):
            return None
        return {'coherence': round(coh, 2), 'ridge_normal_deg': round(ang)}


@register_terrain('hough_slats')
class HoughSlats(TerrainStrategy):
    """Slats seen as several parallel straight lines (Hough transform) in the depth and / or colour camera image.

    Per camera, every recent frame votes "slats" when it has at least min_lines[camera] distinct parallel lines
    (slat_lines.is_slats); the camera says slats when at least min_vote_fraction of its last frames voted. mode `any`:
    one camera is enough, `all`: every listed camera that has frames must say it. `require` (default [depth]) lists
    cameras that must ALWAYS confirm, whatever the mode: a camera that has no frames yet does not confirm."""
    needs_cells = False

    def match(self, stats):
        min_lines = dict({'depth': {'rungs': 4, 'converging': 3}, 'color': {'rungs': 15, 'converging': 6}}, **(self.cfg.get('min_lines') or {}))
        need_frames = int(self.cfg.get('min_frames', 5))
        need_share = float(self.cfg.get('min_vote_fraction', 0.6))
        says, votes = {}, {}
        for cam in self.cfg.get('sources', ['depth', 'color']):
            frames = stats.images.get(cam, [])
            if len(frames) < need_frames:
                continue
            share = sum(slat_lines.is_slats(r, min_lines.get(cam, 3)) for r in frames) / len(frames)
            votes[cam] = round(share, 2)
            says[cam] = share >= need_share
        if not says:
            return None
        ok = all(says.values()) if self.cfg.get('mode', 'any') == 'all' else any(says.values())
        # confirmation: these cameras MUST have frames and say slats (default: depth). The colour image alone is easily
        # fooled by block edges, wood grain and curtain folds, so it cannot name slats without the depth camera.
        for cam in self.cfg.get('require', ['depth']):
            if not says.get(cam):
                return None
        if not ok:
            return None
        # which way do they run? the frames that voted slats, from the cameras that say slats
        voters = [r for cam, frames in stats.images.items() if says.get(cam)
                  for r in frames if slat_lines.is_slats(r, min_lines.get(cam, 3)) and r.get('orientation') != 'none']
        out = {'slats_votes': votes}
        if voters:
            horizontal = [r for r in voters if r['orientation'] == 'horizontal']
            vertical = [r for r in voters if r['orientation'] == 'vertical']
            chosen, name = (horizontal, 'horizontal') if len(horizontal) >= len(vertical) else (vertical, 'vertical')
            angles = [r['angle_deg'] for r in chosen if r.get('angle_deg') is not None]
            out.update(slats_orientation=name, slats_angle_deg=round(float(np.mean(angles)), 1) if angles else None)
        return out


class KindClassifier:
    """Collects floor measurements for a while (reset() starts a new patch), then classify() names the terrain kind.

    Spread is measured per assessment and per 0.3 m cell in the robot frame, so odometry drift between assessments
    does not blur the heights. Cells that hold a wall or object are skipped."""

    def __init__(self, cfg: Optional[dict] = None):
        """cfg: {'common': {...measurement settings...}, 'terrains': [ {name, type, ...}, ... ]} (both optional)."""
        cfg = cfg or {}
        self.cfg = dict(DEFAULTS, **(cfg.get('common') or {}))
        self.strategies: List[TerrainStrategy] = []
        for entry in cfg.get('terrains') or DEFAULT_TERRAINS:
            if entry.get('type') not in TERRAIN_TYPES:
                raise ValueError(f"unknown terrain type {entry.get('type')!r} (known: {sorted(TERRAIN_TYPES)})")
            self.strategies.append(TERRAIN_TYPES[entry['type']](entry))
        self.image_params = cfg.get('images') or {}       # slat_lines.detect settings, per camera or shared
        self.rng = np.random.RandomState(0)
        self.reset()

    def reset(self):
        self.spreads: List[float] = []
        self.floor = np.zeros((0, 3))
        self.images: Dict[str, list] = {}

    def add_image(self, camera: str, gray8: np.ndarray, debug: bool = False):
        """One camera frame (single channel, uint8): look for parallel lines, keep the latest image_window results.
        Returns the detection; with debug=True also the annotated BGR picture: (result, picture)."""
        params = dict(self.image_params.get('common') or {}, **(self.image_params.get(camera) or {}))
        res, segments = slat_lines.analyse(gray8, params)
        lst = self.images.setdefault(camera, [])
        lst.append(res)
        del lst[:-int(self.cfg['image_window'])]
        if not debug:
            return res
        min_lines = dict({'depth': {'rungs': 4, 'converging': 3}, 'color': {'rungs': 15, 'converging': 6}}, **{})
        for strat in self.strategies:
            if strat.type_name == 'hough_slats':
                min_lines.update(strat.cfg.get('min_lines') or {})
        return res, slat_lines.annotate(gray8, res, segments, params, min_lines.get(camera, 3))

    def add(self, xy, h, xy_fixed=None):
        """xy: (N, 2) positions in the robot frame; h: (N,) height above the floor plane (walls included);
        xy_fixed: the same points in a fixed frame (odom), for the ridge test across assessments."""
        c = self.cfg
        xy, h = np.asarray(xy, float), np.asarray(h, float)
        if len(h) == 0:
            return
        ci, cj = np.floor(xy[:, 0] / c['cell_m']).astype(int), np.floor(xy[:, 1] / c['cell_m']).astype(int)
        key = (ci + 500) * 1000 + (cj + 500)
        order = np.argsort(key)
        key_s, h_s = key[order], h[order]
        for grp in np.split(h_s, np.flatnonzero(np.diff(key_s)) + 1):
            if len(grp) < c['min_points_per_cell'] or grp.max() > c['wall_height_m']:
                continue
            self.spreads.append(float(np.percentile(grp, 90) - np.percentile(grp, 10)))
        sel = (h > -0.12) & (h < c['wall_height_m'] - 0.05)
        if sel.any():
            pts = np.c_[(xy if xy_fixed is None else np.asarray(xy_fixed, float))[sel], h[sel]]
            self.floor = np.vstack([self.floor, pts])
            if len(self.floor) > c['max_floor_points']:
                self.floor = self.floor[self.rng.choice(len(self.floor), int(c['max_floor_points']), replace=False)]

    def classify(self) -> dict:
        have_cells = len(self.spreads) >= self.cfg['min_cells']
        stats = TerrainStats(self.spreads, self.floor, self.cfg, self.rng, self.images)
        info = {'n_cells': stats.n_cells}
        if have_cells:
            info.update(spread_median_m=round(stats.median, 3), rough_fraction=round(stats.rough_fraction, 2))
        for strat in self.strategies:
            if strat.needs_cells and not have_cells:
                continue
            found = strat.match(stats)
            if found is not None:
                return dict(info, **found, kind=strat.name, marker=strat.marker)
        return dict(info, kind='unknown')
