"""Tile bookkeeping for the mission (no ROS): grid, classification, trail, failures and unreachable tiles.

The mission manager feeds it the SLAM map and the robot pose; the mission modes ask it what is left
to explore. "Identified" tiles are the ones the map has defined (to_visit / visited / blocked); a tile
is "explored" once the robot has reached it (visited). Tiles the robot failed to reach too many times
become "unreachable" so the end condition can ignore them.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Callable, Dict, List, Optional, Sequence

from lite3_explorer import tiles as T
from lite3_explorer.tile_reports import within_reach


class TileTracker:
    def __init__(self, size_m: float = 1.0, reach_radius_m: float = 0.30, max_failures: int = 2,
                 on_reached: Optional[Callable[[tuple, list, str], None]] = None,
                 on_enter: Optional[Callable[[tuple], None]] = None, min_known: float = 0.5,
                 scene_min_known: float = 0.95, scene_max_free: float = 0.80):
        self.size_m = size_m
        self.min_known = min_known           # share of a tile the lidar must have scanned before it becomes a tile
        self.scene_min_known, self.scene_max_free = scene_min_known, scene_max_free   # scene tile: scanned >= this, traversable < this
        self.reach_radius_m = reach_radius_m
        self.wall_release_fraction = 0.25               # a virtual wall stays until less than this share of its edge is mapped wall
        self._wall_keys = set()                         # edges that are virtual walls now (hysteresis)
        self.skip_scene = False                        # True: scene tiles are never driven to (the scene mode looks at them)
        self.described: set = set()                    # scene tiles already looked at: explored without being entered
        self.max_failures = max_failures     # 0 = never give up on a tile
        self.on_reached = on_reached                   # called (tile, pose, source) for every NEW tile reached
        self.on_enter = on_enter                       # called (tile) whenever the robot enters another tile
        self.grid = None                               # T.TileGrid
        self.init_info: Optional[dict] = None
        self.terrain: Dict[str, dict] = {}              # tile label -> what the terrain classifier said about it
        self._look_rugged = set()                      # tiles that were called non-flat during their current terrain look
        self.start_wall = None                         # ((x, y), (x, y)): virtual wall behind the robot at the start
        self.tiles: Dict[tuple, dict] = {}
        self.trail: List[tuple] = []
        self.trail_log: List[dict] = []
        self.failures: Dict[str, List[str]] = {}       # tile label -> reasons
        self.unreachable: Dict[str, str] = {}          # tile label -> why
        self._map = None                               # (grid array, GridInfo)
        self._map_version = 0
        self._classified_version = -1
        self.version = 0                               # bumped on every re-classification (for publishers)

    # ------------------------------------------------------------------ setup
    @property
    def ready(self) -> bool:
        return self.grid is not None

    def set_map(self, grid, info):
        self._map = (grid, info)
        self._map_version += 1

    @property
    def map(self):
        return self._map

    def establish(self, pose, how: str = 'wall-snapped'):
        """Anchor the tile grid at the robot (snapped to nearby walls when the map allows) and visit the start tile."""
        cand = None
        if self._map is not None:
            cands = T.initial_candidates(self._map[0], self._map[1], pose, size=self.size_m)
            cand = cands[0] if cands else None
        if cand is None:
            half = self.size_m / 2.0
            cand = {'id': '-', 'kind': 'centred on the robot, along the map axes', 'sides': {},
                    'grid': T.TileGrid(pose[0] - half, pose[1] - half, 0.0, self.size_m)}
            how = 'no map'
        self.grid = cand['grid']
        self.terrain = {}
        self.init_info = {'placement': cand['id'], 'kind': cand['kind'], 'how': how,
                          'grid': self.grid.as_dict(), 'map_sides': cand.get('sides', {})}
        start = self.grid.tile_of(pose[0], pose[1])
        # virtual wall along the BACK edge of the start tile: the side of the tile that faces away from the robot's
        # (original) heading
        yaw = pose[2] if len(pose) > 2 else 0.0
        normals = {'E': 0.0, 'N': math.pi / 2, 'W': math.pi, 'S': -math.pi / 2}     # outward normals vs. the grid yaw
        back = max(normals, key=lambda sd: math.cos(self.grid.yaw + normals[sd] - (yaw + math.pi)))
        self.start_wall = self.grid.edge(start, back)
        self.start_wall_side = back
        self.trail, self.trail_log = [], []
        self._wall_keys = set()
        self.described = set()
        self._visit(start, pose, 'start')
        self.update(force=True)
        return start

    # ---------------------------------------------------------- classification
    def update(self, force: bool = False) -> bool:
        """Re-classify the tiles if the map changed. True if anything was recomputed."""
        if self.grid is None or self._map is None:
            return False
        if not force and self._classified_version == self._map_version:
            return False
        self._classified_version = self._map_version
        self.version += 1
        grid, info = self._map
        self.tiles = T.classify_tiles(grid, info, self.grid, list(self.trail) + sorted(self.described), min_known=self.min_known,
                                      scene_min_known=self.scene_min_known, scene_max_free=self.scene_max_free)
        for tile, t in self.tiles.items():
            if t['status'] != 'unknown':
                t['sides'] = T.tile_sides(grid, info, self.grid, tile)
            if t['status'] == 'to_visit' and T.tile_label(tile) in self.unreachable:
                t['status'] = 'unreachable'
        return True

    def virtual_walls(self, min_fraction: float = 0.5, start_wall: bool = True):
        """World segments to block in the planner: the wall behind the start pose and every tile edge more than
        min_fraction covered by mapped wall."""
        segs = []
        if start_wall and getattr(self, 'start_wall', None):
            segs.append(self.start_wall)
        if self.grid is not None and self._map is not None:
            # Hysteresis: an edge becomes a wall above min_fraction but is only released below wall_release_fraction. Next
            # to a wall the SLAM map thins it (the robot is close, grazing rays), so the share hovers around min_fraction
            # and the wall would flicker or vanish while it is still there.
            keys = set()
            for key, seg, share in T.edge_coverage(self._map[0], self._map[1], self.grid, self.tiles):
                if share > min_fraction or (key in self._wall_keys and share >= self.wall_release_fraction):
                    keys.add(key)
                    segs.append(seg)
            self._wall_keys = keys
        return segs

    def begin_terrain_look(self, label: str):
        """The classifier is asked about this tile again: answers of this look are compared with each other only."""
        self._look_rugged.discard(label)

    def set_terrain(self, label: str, result: dict) -> bool:
        """Store the terrain classifier's answer for a tile. True if the kind changed.

        Within one look a "flat" never replaces an answer that said the floor is not flat (uneven / slats): the classifier's
        answers flip around its thresholds (and the lidar sees little at first), and calling rough ground flat is the
        costly mistake. A new look (begin_terrain_look) starts afresh."""
        kind = result.get('kind')
        if kind == 'flat' and label in self._look_rugged:
            return False
        if kind and kind != 'flat':
            self._look_rugged.add(label)
        changed = self.terrain.get(label, {}).get('kind') != kind
        self.terrain[label] = result
        self.version += 1                      # republish the tile map
        return changed

    def invalidate(self):
        self._classified_version = -1

    # ------------------------------------------------------------- reaching
    def track(self, pose) -> Optional[tuple]:
        """A tile is reached when the robot is within reach_radius_m of its centre. Returns the tile if newly entered."""
        if self.grid is None or pose is None:
            return None
        tile = self.grid.tile_of(pose[0], pose[1])
        if within_reach(pose, self.grid.center(tile), self.reach_radius_m):
            return self._visit(tile, pose, 'centre reached')
        return None

    def mark_goal_reached(self, tile, goal_xy, pose) -> Optional[tuple]:
        """Nav2 reported a tile goal reached: count it if the robot is at the goal point (beside a wall the goal is off-centre)."""
        if self.grid is None or pose is None:
            return None
        if within_reach(pose, goal_xy, 1.5 * self.reach_radius_m):
            return self._visit(tuple(tile), pose, 'goal reached')
        return None

    def _visit(self, tile, pose, source):
        if self.trail and self.trail[-1] == tile:
            return None
        new = tile not in set(self.trail)
        self.trail.append(tile)
        self.trail_log.append({'tile': list(tile), 'time': dt.datetime.now().isoformat(timespec='seconds'),
                               'pose': list(pose) if pose else None, 'source': source})
        self.invalidate()
        if self.on_enter is not None:
            self.on_enter(tile)
        label = T.tile_label(tile)
        self.failures.pop(label, None)
        self.unreachable.pop(label, None)
        if new and self.on_reached is not None:
            self.on_reached(tile, pose, source)
        return tile if new else None

    @property
    def current(self) -> Optional[tuple]:
        return self.trail[-1] if self.trail else None

    def visited_count(self) -> int:
        return len(set(self.trail))

    # ------------------------------------------------------------- failures
    def record_failure(self, label: str, reason: str) -> bool:
        """Count a failed attempt; True if the tile is now given up on (unreachable)."""
        reasons = self.failures.setdefault(label, [])
        reasons.append(reason)
        if self.max_failures and len(reasons) >= self.max_failures:
            self.mark_unreachable(label, f'{len(reasons)} failed attempts: {"; ".join(reasons[-3:])}')
            return True
        return False

    def mark_unreachable(self, label: str, why: str):
        self.unreachable[label] = why
        self.invalidate()
        self.update()

    def failed_recently(self, label: str) -> bool:
        return label in self.failures

    # ------------------------------------------------------------- queries
    def counts(self) -> Dict[str, int]:
        return T.tile_counts(self.tiles)

    def _beside_current(self, tile) -> bool:
        return self.current is not None and abs(tile[0] - self.current[0]) + abs(tile[1] - self.current[1]) == 1

    def scene_tiles(self) -> List[tuple]:
        """Scene tiles that count: completely scanned but less than scene_max_free traversable, AND sharing a side with the
        current tile (the scene detection is not used anywhere else), plus the ones already looked at."""
        return [t for t, d in sorted(self.tiles.items())
                if t in self.described or (d.get('scene') and self._beside_current(t))]

    def mark_described(self, tile):
        """A scene tile was looked at from a neighbouring tile: it counts as explored (status visited) although the robot
        never entered it. The trail (and so the current tile) is untouched."""
        self.described.add(tuple(tile))
        self.invalidate()
        self.update()

    def is_scene(self, tile) -> bool:
        tile = tuple(tile)
        return tile in self.described or (bool(self.tiles.get(tile, {}).get('scene')) and self._beside_current(tile))

    def scene_neighbours(self, tile) -> List[tuple]:
        """Scene tiles that share a side with `tile` and have not been described yet."""
        i, j = tile
        return [n for n in ((i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1))
                if n not in self.described and self.tiles.get(n, {}).get('scene')]    # (call with the current tile)

    def _drivable(self, label: str) -> bool:
        return not (self.skip_scene and self.is_scene(tuple(int(v) for v in label.split(','))))

    def unexplored(self) -> List[str]:
        """Identified, walkable tiles not visited yet and not given up on."""
        return [T.tile_label(t) for t, d in sorted(self.tiles.items()) if d['status'] == 'to_visit']

    def targets(self, robot_xy, *, limit: int = 24, min_clearance_m: float = 0.30,
                skip_failed: bool = True) -> List[dict]:
        """Unvisited tiles with a clear goal point, nearest first (the caller checks Nav2 reachability)."""
        if self.grid is None or self._map is None or self.current is None:
            return []
        grid, info = self._map
        raw = T.nearest_connected_targets(grid, info, self.grid, self.tiles, self.current, robot_xy,
                                          limit=limit, min_clearance_m=min_clearance_m, grid_connected=False)
        if skip_failed:
            raw = [t for t in raw if t['tile'] not in self.unreachable]
        return [t for t in raw if self._drivable(t['tile'])]

    def recovery_targets(self, robot_xy, *, statuses=('visited', 'to_visit'), limit: int = 5,
                         min_distance_m: float = 0.25, min_clearance_m: float = 0.25,
                         exclude: Sequence[str] = (), adjacent_only: bool = False) -> List[dict]:
        if self.grid is None or self._map is None:
            return []
        grid, info = self._map
        excl = [tuple(int(v) for v in lab.split(',')) for lab in exclude]
        out = T.nearest_tile_centres(grid, info, self.grid, self.tiles, robot_xy, statuses=statuses, limit=limit,
                                     min_distance_m=min_distance_m, min_clearance_m=min_clearance_m,
                                     exclude=excl,
                                     adjacent_to=self.grid.tile_of(robot_xy[0], robot_xy[1]) if adjacent_only else None)
        return [t for t in out if self._drivable(t['tile'])]

    def tile_of(self, x, y):
        return self.grid.tile_of(x, y) if self.grid is not None else None

    def as_dict(self, frame: str = 'map') -> dict:
        return {'frame': frame, 'grid': self.grid.as_dict() if self.grid else None, 'initial_tile': self.init_info,
                'current_tile': T.tile_label(self.current) if self.current else None,
                'counts': self.counts(), 'trail': [T.tile_label(t) for t in self.trail],
                'unreachable': dict(self.unreachable),
                'tiles': {T.tile_label(k): (dict(v, terrain=self.terrain[T.tile_label(k)])
                                       if T.tile_label(k) in self.terrain else v) for k, v in sorted(self.tiles.items())}}


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
