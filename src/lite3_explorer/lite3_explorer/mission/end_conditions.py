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
    tiles: Optional[dict] = None                 # tile -> classification (status, sides), for tile-layout conditions
    trail: List[tuple] = field(default_factory=list)        # tiles the robot walked, in order
    pose: Optional[list] = None                  # [x, y, yaw] of the robot
    grid: Optional[object] = None                # tiles.TileGrid


class EndCondition:
    name = 'base'
    continuous = False                           # True: checked every tick, not only at checkpoints

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}

    def evaluate(self, snap: MissionSnapshot) -> Tuple[bool, str]:
        raise NotImplementedError

    def goal(self) -> Optional[dict]:
        """A place the robot must go to before this condition can be met: {'tile', 'mode', 'reason'} or None."""
        return None

    def reset(self):
        """A new map (and tile grid) starts."""


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


@register('ending_tile')
class EndingTile(EndCondition):
    """next_tile trigger (default): the scan from the current tile shows the tile next to it with no walls (the outside
    of the maze): the CURRENT tile is the ending tile, the robot centres itself there and never leaves the maze.
    open_tile trigger (safety net): the robot entered a tile with no walls on its edges (triggers: open_tile; optionally hallway: a hallway longer
    than 3 tiles, min_hallway_tiles): the tile preceding it on the robot's route with exactly 2 walls on its edges is the
    ENDING tile (tiles.find_ending_tile; preceding: false -> the entered tile itself). "No walls" = no edge with mapped
    wall along solid_wall_share or more of it (0.4: any edge read as a wall rules the tile out; 0 = every side reads open).
    The initial tile is never the ending tile. entered_only (default): only tiles
    the robot stood in count, never ones seen from a distance (the 1.7 m lidar cannot see a far tile's edges).
    Met once the robot stands at the ending tile's centre (within reach_radius_m). Until then goal() names the tile and
    the mode (`mode`, default 'ending') that drives there. Once found the ending tile is kept (lock) until a new map."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.found: Optional[dict] = None

    @property
    def mode(self) -> str:
        return self.cfg.get('mode', 'ending')

    def reset(self):
        self.found = None

    def _solid(self):
        v = float(self.cfg.get('solid_wall_share', 0.4))
        return v if v > 0 else None

    def evaluate(self, snap):
        if not snap.tiles_ready or snap.tiles is None or snap.grid is None:
            return False, 'no tile grid yet'
        if self.found is None or not self.cfg.get('lock', True):
            from lite3_explorer import tiles as T
            self.found = T.find_ending_tile(snap.tiles, snap.trail, min_hallway_tiles=int(self.cfg.get('min_hallway_tiles', 4)),
                                            triggers=[t for t in self.cfg.get('triggers', ['hallway', 'open_tile']) if t],
                                            walls=int(self.cfg.get('walls', 2)),
                                            entered_only=bool(self.cfg.get('entered_only', True)),
                                            preceding=bool(self.cfg.get('preceding', True)),
                                            solid_wall_share=self._solid(),
                                            min_seen_share=float(self.cfg.get('min_seen_share', 0.5)))
        if self.found is None:
            return False, 'no hallway / open tile found yet'
        label = '%d,%d' % self.found['tile']
        if snap.pose is None:
            return False, f'ending tile {label} ({self.found["trigger"]}): no pose'
        cx, cy = snap.grid.center(self.found['tile'])
        d = ((snap.pose[0] - cx) ** 2 + (snap.pose[1] - cy) ** 2) ** 0.5
        if d <= float(self.cfg.get('reach_radius_m', 0.30)):
            return True, f'centre of ending tile {label} reached ({self.found["trigger"]}; {d:.2f} m off centre)'
        return False, f'ending tile {label} ({self.found["trigger"]}): {d:.2f} m from its centre'

    def goal(self):
        if self.found is None:
            return None
        return {'tile': self.found['tile'], 'mode': self.mode, 'condition': self.name,
                'reason': f'ending tile {"%d,%d" % self.found["tile"]} found ({self.found["trigger"]}): centring on it'}


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

    def goal(self) -> Optional[dict]:
        """The first condition's place to go to before it can be met (see EndCondition.goal), or None."""
        for cond in self.conditions:
            g = cond.goal()
            if g is not None:
                return g
        return None

    def reset(self):
        for cond in self.conditions:
            cond.reset()
        self.last = {}

    def describe(self) -> Dict[str, str]:
        return {name: ('MET ' if ok else '') + why for name, (ok, why) in self.last.items()}


def build_end_condition(cfg: dict) -> CompositeEndCondition:
    names = [n for n in cfg.get('conditions', []) if n]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f'unknown end condition(s) {unknown}; known: {sorted(REGISTRY)}')
    return CompositeEndCondition([REGISTRY[n](cfg.get(n, {})) for n in names], cfg.get('combine', 'any'))
