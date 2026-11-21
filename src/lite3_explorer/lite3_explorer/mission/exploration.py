"""Exploration strategies: WHERE the robot goes next (no ROS). The counterpart of traversal.py (HOW it drives).

The exploring mode holds an ordered list of exploration profiles (modes.<name>.exploration, e.g. [tiles] or
[tiles, frontier]). At each decision it asks them in order; the first one that yields a reachable target wins,
so later profiles are fallbacks. Profiles are configured under ``explorations.<name>`` with a ``type`` naming a
registered class:

  tiles     (default) unvisited tiles that share a side with an explored tile, filtered by Nav2 reachability
  frontier  (optional) frontier clusters (free space next to unknown space), ranked by distance and size

The interface a strategy implements (a target is a dict with at least label, tile, goal_xy, clearance_m, kind):

  candidates(pose) -> [target]          ranked candidates for this decision ([] = nothing to offer)
  check_paths                           True: the mode asks Nav2 for a path to each candidate first
  select(checked, pose) -> [target]     checked = [(target, path or None)]; returns the reachable targets ranked,
                                        each with path_xy and path_m. Elimination by reachability happens here.
  still_valid(target) -> bool           polled while driving; False cancels the goal and re-decides
  on_reached(target) / on_failed(target, reason) -> bool (True = given up on for good)

Add one with @register_exploration('name') on an Exploration subclass and an ``explorations.<x>: {type: name}``
profile.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Type

from lite3_explorer import frontiers as F
from lite3_explorer import tiles as T

EXPLORATION_TYPES: Dict[str, Type['Exploration']] = {}


def register_exploration(type_name: str):
    def deco(cls):
        cls.type_name = type_name
        EXPLORATION_TYPES[type_name] = cls
        return cls
    return deco


def path_length(points) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


class Exploration:
    type_name = 'base'
    check_paths = True

    def __init__(self, ctx, name: str, cfg: dict):
        self.ctx, self.name, self.cfg = ctx, name, dict(cfg or {})

    avoid: frozenset = frozenset()      # tile labels whose goal just failed (set by the exploring mode)

    def candidates(self, pose) -> List[dict]:
        raise NotImplementedError

    def select(self, checked, pose) -> List[dict]:
        out = []
        for target, path in checked:
            if path:
                pts = [tuple(p[:2]) for p in path]
                out.append(dict(target, path_xy=pts, path_m=round(path_length(pts), 2)))
        return sorted(out, key=lambda t: t['path_m'])

    def still_valid(self, target) -> bool:
        return True

    def on_reached(self, target):
        pass

    def on_failed(self, target, reason) -> bool:
        return False

    def describe(self) -> dict:
        return {'type': self.type_name}


# ------------------------------------------------------------------------------------------------ tiles
@register_exploration('tiles')
class TileExploration(Exploration):
    """Unvisited, identified tiles.

    Candidates are tiles with a goal point clear of walls whose adjacency level (tiles.adjacency_level) is at most
    max_adjacency: 0 = open side with the current tile, 1 = open side with any visited tile, 2 = touches a visited
    tile across a wall, 3 = anywhere. Default 2: the next tile must share a side with an explored tile. Only if NO
    tile qualifies are the others offered (fallback_any: true), so the mission can't stall behind an eliminated tile.

    Neighbours only (neighbours_only): the next tile is chosen ONLY among the tiles that share a side with the current
    tile; every other unexplored tile is marked irrelevant (tile dict 'relevant': False) for this decision and is
    not offered. With none left (a dead end) the other tiles are offered after all (neighbours_fallback: true), so
    the mission can't stall; with it false the mission waits / recovers instead.

    Reachability: every candidate gets a Nav2 path check. A tile with no path is dropped for this decision and
    gets a strike; after eliminate_after strikes in a row it is ELIMINATED (marked unreachable, so the end condition
    no longer waits for it). Strikes are only given when some other candidate did get a path, so a planner that
    fails from where the robot stands (stuck in an inflated wall) eliminates nothing: that case goes to recovery.

    Direct path (direct_path_only): the next tile must be a neighbour of the current tile with a Nav2 path that stays
    inside the two tiles and crosses their shared side. When no tile qualifies (a dead end, everything near is
    explored) the best reachable tile is used instead (direct_fallback: true), else none is chosen.

    Ranking: by adjacency level (with prefer_current: false, levels 0 and 1 count the same), then Nav2 path length.
    """

    def __init__(self, ctx, name, cfg):
        super().__init__(ctx, name, cfg)
        self.strikes: Dict[str, int] = {}

    def _level_key(self, level):
        if not self.cfg.get('prefer_current', True) and level == T.ADJ_CURRENT:
            return T.ADJ_VISITED
        return level

    def candidates(self, pose):
        tiles = self.ctx.tiles
        if not tiles.ready:
            return []
        clearance = float(self.cfg.get('min_clearance_m', 0.30))
        raw = tiles.targets(pose[:2], limit=10 ** 6, min_clearance_m=clearance)
        # Identified tiles with no goal point clear of the walls: nothing could drive there. Eliminate them.
        with_point = {t['tile'] for t in raw}
        unclear = [lab for lab in tiles.unexplored() if lab not in with_point]
        for label in unclear:
            tiles.mark_unreachable(label, 'no goal point clear of the walls')
        if unclear:
            self.ctx.log('warn', f'Tile(s) {", ".join(unclear)} have no point clear of walls: eliminated.')
            self.ctx.event('eliminated', tiles=unclear, why='no clear goal point')
        if self.avoid:
            # a goal that just failed: choose another tile first (it is only retried when nothing else is left)
            others = [t for t in raw if t['tile'] not in self.avoid]
            if others:
                raw = others
        raw = self._neighbours_only(raw)
        visited, current = set(tiles.trail), tiles.current
        max_level = int(self.cfg.get('max_adjacency', T.ADJ_WALL))
        within, beyond = [], []
        for t in raw:
            tile = tuple(int(v) for v in t['tile'].split(','))
            level = T.adjacency_level(tiles.tiles, tile, visited, current)
            (within if level <= max_level else beyond).append(
                dict(t, label=t['tile'], kind='tile', adjacency=level, adjacency_note=T.ADJACENCY_NAMES[level]))
        out = within
        if not within and beyond and self.cfg.get('fallback_any', True):
            # e.g. the only tile linking the explored area to the rest was eliminated: without this the tiles behind
            # it could never be chosen and the mission would stall. They still need a Nav2 path.
            self.ctx.log('info', f'No unexplored tile within adjacency level {max_level}; '
                                 f'falling back to {len(beyond)} tile(s) further away.')
            out = beyond
        elif not within and beyond:
            self.ctx.log('info', f'{len(beyond)} unexplored tile(s), none within adjacency level {max_level}.')
        out.sort(key=lambda t: (self._level_key(t['adjacency']), t['distance_m']))
        return out[:int(self.cfg.get('candidate_limit', 12))]

    def _neighbours_only(self, raw):
        """Keep the tiles that share a side with the current tile; mark the unexplored rest irrelevant."""
        tiles = self.ctx.tiles
        if not self.cfg.get('neighbours_only', True) or tiles.current is None:
            return raw
        ci, cj = tiles.current
        near = {(ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)}
        for tile, d in tiles.tiles.items():
            d['relevant'] = tile in near
        keep = [t for t in raw if tuple(int(v) for v in t['tile'].split(',')) in near]
        skipped = [t['tile'] for t in raw if t not in keep]
        if skipped:
            self.ctx.log('info', f'Tiles not sharing a side with {T.tile_label(tiles.current)} are irrelevant now: '
                                 f'{", ".join(skipped)}.')
        if keep or not self.cfg.get('neighbours_fallback', True):
            return keep
        if raw:
            self.ctx.log('info', f'No unexplored neighbour of tile {T.tile_label(tiles.current)}: '
                                 f'offering the other tiles.')
            for d in tiles.tiles.values():
                d['relevant'] = True
        return raw

    def select(self, checked, pose):
        tiles = self.ctx.tiles
        reachable = super().select(checked, pose)
        if reachable:
            limit = int(self.cfg.get('eliminate_after', 2))
            for target, path in checked:
                label = target['tile']
                if path:
                    self.strikes.pop(label, None)
                    continue
                self.strikes[label] = self.strikes.get(label, 0) + 1
                if limit and self.strikes[label] >= limit:
                    tiles.mark_unreachable(label, f'no Nav2 path {self.strikes[label]} times in a row')
                    self.ctx.log('warn', f'Tile {label}: no Nav2 path {self.strikes[label]} times: eliminated.')
                    self.ctx.event('eliminated', tiles=[label], why='no Nav2 path')
                    self.strikes.pop(label, None)
                else:
                    self.ctx.log('info', f'Tile {label}: no Nav2 path (strike {self.strikes[label]}/{limit}).')
        if reachable and self.cfg.get('reject_via_other_tiles', True) and tiles.grid is not None:
            # a tile that can only be reached THROUGH other tiles is not a choice: it comes up once the robot is next to it
            here = {tuple(tiles.current)} if tiles.current else set()
            here.add(tuple(tiles.grid.tile_of(pose[0], pose[1])))
            tol = float(self.cfg.get('via_tile_tolerance_m', 0.15))
            direct, via = [], []
            for t in reachable:
                goal = tuple(int(v) for v in t['tile'].split(','))
                crossed = T.tiles_crossed(tiles.grid, t['path_xy'], tol) - here - {goal}
                (via if crossed else direct).append((t, crossed))
            if via:
                self.ctx.log('info', 'Not chosen (path runs through other tiles): ' + ', '.join(
                    f'{t["tile"]} via {"/".join(T.tile_label(c) for c in sorted(cr))}' for t, cr in via))
                self.ctx.event('via_other_tiles', tiles=[t['tile'] for t, _ in via])
            reachable = [t for t, _ in direct]
        if reachable and self.cfg.get('direct_path_only', True) and tiles.grid is not None and tiles.current:
            direct = [t for t in reachable if T.direct_path_through_shared_side(
                tiles.grid, tiles.current, tuple(int(v) for v in t['tile'].split(',')), t['path_xy'])]
            if direct:
                reachable = direct
            elif self.cfg.get('direct_fallback', True):
                self.ctx.log('info', f'No neighbour of tile {T.tile_label(tiles.current)} has a direct path through '
                                     f'the shared side: using the best reachable tile instead.')
            else:
                reachable = []
        reachable.sort(key=lambda t: (self._level_key(t['adjacency']), t['path_m']))
        return reachable[:int(self.cfg.get('target_limit', 6))]

    def still_valid(self, target):
        """Keep driving unless the tile turned out blocked / unreachable on the way (visited is fine: Nav2 centres it)."""
        return self.ctx.tiles.tiles.get(tuple(int(v) for v in target['tile'].split(',')), {}).get('status') \
            in ('to_visit', 'visited', None)

    def on_reached(self, target):
        self.strikes.pop(target['tile'], None)

    def on_failed(self, target, reason):
        return self.ctx.tiles.record_failure(target['tile'], reason)

    def describe(self):
        return {'type': self.type_name, 'strikes': dict(self.strikes)}


# --------------------------------------------------------------------------------------------- frontier
@register_exploration('frontier')
class FrontierExploration(Exploration):
    """Optional: frontier clusters, cost = distance_weight * distance - size_weight * size (lowest first).

    Mostly useful as a fallback after `tiles` (exploration: [tiles, frontier]) to open up unmapped parts of the
    arena when no identified tile is left or reachable. Failed goals are blacklisted. While driving, the goal is
    dropped once its frontier has been mapped (replan_when_explored). Pair it with the no_frontiers_left end
    condition (combine: all) if the mission should not end while frontiers remain.
    """

    def __init__(self, ctx, name, cfg):
        super().__init__(ctx, name, cfg)
        self.blacklist: List = []

    def candidates(self, pose):
        if self.ctx.tiles.map is None:
            return []
        grid, info = self.ctx.tiles.map
        c = self.cfg
        clusters = F.build_clusters(grid, info, min_cluster_cells=int(c.get('min_cluster_cells', 8)),
                                    clearance_m=float(c.get('goal_clearance_m', 0.35)))
        F.select_frontier(clusters, pose[:2], distance_weight=float(c.get('distance_weight', 1.0)),
                          size_weight=float(c.get('size_weight', 0.5)),
                          min_goal_distance_m=float(c.get('min_goal_distance_m', 0.6)),
                          blacklist=self.blacklist, blacklist_radius_m=float(c.get('blacklist_radius_m', 0.6)))
        min_clear = float(c.get('min_clearance_m', 0.30))
        usable = sorted((cl for cl in clusters if cl.extra.get('usable')
                         and cl.extra.get('clearance_m', 0.0) >= min_clear), key=lambda cl: cl.cost)
        out = []
        for i, cl in enumerate(usable[:int(c.get('candidate_limit', 8))]):
            gx, gy = cl.goal_xy
            tile = self.ctx.tiles.tile_of(gx, gy)
            out.append({'kind': 'frontier', 'label': f'frontier#{i + 1}', 'tile': T.tile_label(tile) if tile else None,
                        'goal_xy': [round(gx, 3), round(gy, 3)], 'centroid_xy': list(cl.centroid_xy),
                        'clearance_m': round(float(cl.extra.get('clearance_m', 0.0)), 2),
                        'size_m': round(cl.size_m, 2), 'distance_m': round(cl.distance_m, 2), 'cost': cl.cost})
        return out

    def select(self, checked, pose):
        reachable = super().select(checked, pose)
        dw, sw = float(self.cfg.get('distance_weight', 1.0)), float(self.cfg.get('size_weight', 0.5))
        for t in reachable:
            t['cost'] = round(dw * t['path_m'] - sw * t['size_m'], 3)
        reachable.sort(key=lambda t: t['cost'])
        return reachable[:int(self.cfg.get('target_limit', 6))]

    def still_valid(self, target):
        if not self.cfg.get('replan_when_explored', True) or self.ctx.tiles.map is None:
            return True
        grid, info = self.ctx.tiles.map
        return F.frontier_near(grid, info, target['goal_xy'], float(self.cfg.get('explored_radius_m', 0.5)))

    def on_failed(self, target, reason):
        self.blacklist.append(tuple(target['goal_xy']))
        return True

    def describe(self):
        return {'type': self.type_name, 'blacklisted': len(self.blacklist)}


def build_exploration(ctx, name: str, profiles: dict) -> Exploration:
    cfg = profiles.get(name)
    if cfg is None:
        raise KeyError(f'no exploration profile {name!r} (explorations: {sorted(profiles)})')
    etype = cfg.get('type', name)
    if etype not in EXPLORATION_TYPES:
        raise KeyError(f'exploration profile {name!r} has unknown type {etype!r} (known: {sorted(EXPLORATION_TYPES)})')
    return EXPLORATION_TYPES[etype](ctx, name, cfg)
