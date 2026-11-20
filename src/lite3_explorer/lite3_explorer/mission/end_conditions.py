"""Pluggable mission end conditions (no ROS).

end_condition.conditions lists condition names; each is configured under end_condition.<name>.
end_condition.combine is 'any' (finish when one is met) or 'all' (finish when every one is met).

Conditions are either checked at CHECKPOINTS only (after a scan, when the next tile is chosen: the tile map is
current then) or CONTINUOUSLY (every tick, e.g. a time limit). To add one, subclass EndCondition and decorate it
with @register('my_name'); then list 'my_name' in end_condition.conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

REGISTRY: Dict[str, Type['EndCondition']] = {}


def register(name: str):
    def deco(cls):
        cls.name = name
        REGISTRY[name] = cls
        return cls
    return deco


@dataclass
class MissionSnapshot:
    counts: Dict[str, int]                       # tile status -> count
    visited: int                                 # distinct tiles reached
    unexplored: List[str] = field(default_factory=list)     # to_visit tile labels
    unreachable: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0                       # since the mission started (left idle the first time)
    tiles_ready: bool = True
    map: Optional[tuple] = None                  # (grid, GridInfo) of the SLAM map, for map-based conditions


class EndCondition:
    name = 'base'
    continuous = False                           # True: checked every tick, not only at checkpoints

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    def evaluate(self, snap: MissionSnapshot) -> Tuple[bool, str]:
        raise NotImplementedError


@register('all_tiles_explored')
class AllTilesExplored(EndCondition):
    """Every identified walkable tile has been reached (identified = defined by the map: to_visit / visited)."""

    def evaluate(self, snap):
        if not snap.tiles_ready:
            return False, 'no tile grid yet'
        min_visited = int(self.cfg.get('min_visited', 1))
        if snap.visited < min_visited:
            return False, f'{snap.visited}/{min_visited} tiles visited'
        if snap.unexplored:
            return False, f'{len(snap.unexplored)} tile(s) left: {", ".join(snap.unexplored[:8])}'
        if snap.unreachable and not self.cfg.get('ignore_unreachable', True):
            return False, f'{len(snap.unreachable)} unreachable tile(s) left'
        extra = f', {len(snap.unreachable)} unreachable skipped' if snap.unreachable else ''
        return True, f'all identified tiles explored ({snap.visited} visited{extra})'


@register('time_limit')
class TimeLimit(EndCondition):
    continuous = True

    def evaluate(self, snap):
        limit = float(self.cfg.get('seconds', 0.0))
        if limit <= 0:
            return False, 'no time limit'
        return snap.elapsed_s >= limit, f'{snap.elapsed_s:.0f}/{limit:.0f} s'


@register('max_tiles_visited')
class MaxTilesVisited(EndCondition):
    continuous = True

    def evaluate(self, snap):
        n = int(self.cfg.get('count', 0))
        if n <= 0:
            return False, 'no tile limit'
        return snap.visited >= n, f'{snap.visited}/{n} tiles visited'


@register('no_frontiers_left')
class NoFrontiersLeft(EndCondition):
    """No frontier cluster (free space next to unknown) of at least min_cluster_cells is left on the map.
    Use with combine: all next to all_tiles_explored when frontier exploration is on."""

    def evaluate(self, snap):
        if snap.map is None:
            return False, 'no map yet'
        from lite3_explorer import frontiers as F
        grid, info = snap.map
        clusters = F.build_clusters(grid, info, min_cluster_cells=int(self.cfg.get('min_cluster_cells', 8)),
                                    clearance_m=float(self.cfg.get('goal_clearance_m', 0.35)))
        min_clear = float(self.cfg.get('min_clearance_m', 0.30))
        usable = [c for c in clusters if c.extra.get('clearance_m', 0.0) >= min_clear]
        return not usable, f'{len(usable)} frontier cluster(s) left'


class CompositeEndCondition:
    def __init__(self, conditions: List[EndCondition], combine: str = 'any'):
        if combine not in ('any', 'all'):
            raise ValueError(f"end_condition.combine must be 'any' or 'all', not {combine!r}")
        if not conditions:
            raise ValueError('end_condition.conditions is empty: the mission could never finish')
        self.conditions = conditions
        self.combine = combine
        self.last: Dict[str, Tuple[bool, str]] = {}

    def check(self, snap: MissionSnapshot, checkpoint: bool) -> Optional[str]:
        """The reason to finish, or None. Checkpoint-only conditions keep their last checkpoint result in between."""
        for cond in self.conditions:
            if checkpoint or cond.continuous:
                self.last[cond.name] = cond.evaluate(snap)
        results = [self.last.get(c.name, (False, 'not checked yet')) for c in self.conditions]
        met = [(c.name, why) for c, (ok, why) in zip(self.conditions, results) if ok]
        if self.combine == 'any' and met:
            return f'{met[0][0]}: {met[0][1]}'
        if self.combine == 'all' and len(met) == len(self.conditions):
            return '; '.join(f'{n}: {w}' for n, w in met)
        return None

    def describe(self) -> Dict[str, str]:
        return {name: ('MET ' if ok else '') + why for name, (ok, why) in self.last.items()}


def build_end_condition(cfg: dict) -> CompositeEndCondition:
    names = [n for n in cfg.get('conditions', []) if n]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f'unknown end condition(s) {unknown}; known: {sorted(REGISTRY)}')
    return CompositeEndCondition([REGISTRY[n](cfg.get(n, {})) for n in names], cfg.get('combine', 'any'))
