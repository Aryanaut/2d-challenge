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

import numpy as np

from lite3_explorer import roller_core as RC
from lite3_explorer import stairs_core as SC
from lite3_explorer import tiles as T
from lite3_explorer.tile_reports import within_reach


class TileTracker:
    def __init__(self, size_m: float = 1.0, reach_radius_m: float = 0.30, max_failures: int = 2,
                 on_reached: Optional[Callable[[tuple, list, str], None]] = None,
                 on_enter: Optional[Callable[[tuple], None]] = None, min_known: float = 0.5,
                 scene_min_known: float = 0.85, scene_max_free: float = 0.80):
        self.size_m = size_m
        self.min_known = min_known           # share of a tile the lidar must have scanned before it becomes a tile
        self.scene_min_known, self.scene_max_free = scene_min_known, scene_max_free   # scene tile: scanned >= this, free/scanned < this
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
        self._look_elev: Dict[str, dict] = {}          # tile -> the highest elevation answer of its current terrain look
        self.start_wall = None                         # ((x, y), (x, y)): virtual wall behind the robot at the start
        self.tiles: Dict[tuple, dict] = {}
        self.trail: List[tuple] = []
        self.trail_log: List[dict] = []
        self.failures: Dict[str, List[str]] = {}       # tile label -> reasons
        self.unreachable: Dict[str, str] = {}          # tile label -> why
        # split tiles: a tile from which two or more tiles could be travelled to. A stack (most recent last) of
        # {'tile': (i, j), 'options': [labels]}; the "last split" is the most recent one that still has an option
        # the robot has not travelled to (see last_split()). Backtracking returns to it.
        self.splits: List[dict] = []
        # tile edges the robot has walked through a curtain on: {edge key: (tile, neighbour)}. The lidar sees a curtain as
        # a wall, so these edges become virtual walls; with open_curtains (backtracking) they are left out.
        self.curtain_edges: Dict[tuple, tuple] = {}
        self.open_curtains = False
        # curtains left for later: a curtain on an edge of a split tile whose other way was taken first.
        # {tile behind the curtain: {'tile': split tile, 'side': edge side}}. Backtracking walks through them.
        self.deferred_curtains: Dict[tuple, dict] = {}
        # staircases (stairs mode): the stair detector's confirmed stair cells make STAIRCASE tiles (status 'staircase':
        # never a Nav2 target, never unexplored; the lidar sees the risers as a wall anyway). The stairs mode climbs over
        # one in AI mode; afterwards its four edges are virtual walls so it is never entered again (stairs_crossed).
        self.stair_points = np.zeros((0, 4))
        self.stair_min_points = 4                      # confirmed stair cells a tile needs to be a staircase tile
        self.stairs: Dict[tuple, dict] = {}            # staircase tile -> {'points': n, 'ascent': (ux, uy)}
        self.stairs_crossed: set = set()
        # roller tiles (roller mode): a roller lying across the middle of a tile, which the lidar maps as a wall (the tile
        # looks like half a tile). Status 'roller' (never a Nav2 target, never unexplored) until the roller mode has
        # crossed it in the h-step gait: then it is marked explored (on the trail) and stays out of Nav2 routes.
        self.roller_points = np.zeros((0, 5))
        self.roller_min_points, self.roller_min_lidar, self.roller_edge_margin_m = 4, 2, 0.15
        self.rollers: Dict[tuple, dict] = {}           # roller tile -> {'points', 'lidar', 'normal', 'line_xy'}
        self.rollers_crossed: set = set()
        # roller crossings the robot made: {sorted (tile before, tile beyond): roller tile}. Backtracking may cross back
        # over them with the roller mode (known_route links the two tiles, roller_between names the roller tile).
        self.roller_crossings: Dict[tuple, tuple] = {}
        self._map = None                              # (grid array, GridInfo)
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
        self.splits = []
        self.curtain_edges = {}
        self.deferred_curtains = {}
        self.stairs, self.stairs_crossed = {}, set()
        self.rollers, self.rollers_crossed, self.roller_crossings = {}, set(), {}
        self._visit(start, pose, 'start')
        self.set_stair_points(self.stair_points)       # stairs seen before the grid existed
        self.set_roller_points(self.roller_points)     # rollers too
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
        for tile in set(self.stairs) | self.stairs_crossed:
            t = self.tiles.setdefault(tile, {'free': 0.0, 'occupied': 0.0, 'known': 0.0, 'unmapped_edge': False,
                                             'scene': False})
            t['status'], t['scene'] = 'staircase', False
        for tile in set(self.rollers) - self.rollers_crossed:
            t = self.tiles.setdefault(tile, {'free': 0.0, 'occupied': 0.0, 'known': 0.0, 'unmapped_edge': False,
                                             'scene': False})
            t['status'], t['scene'] = 'roller', False
        self._flag_last_split()
        return True

    def virtual_walls(self, min_fraction: float = 0.5, start_wall: bool = True):
        """World segments to block in the planner: the wall behind the start pose, every tile edge more than
        min_fraction covered by mapped wall, and all four edges of every staircase the robot has climbed over. With
        open_curtains (backtracking) edges with a curtain are left out."""
        segs = []
        skip = set(self.curtain_edges) if self.open_curtains else set()
        if start_wall and getattr(self, 'start_wall', None):
            segs.append(self.start_wall)
        if self.grid is not None:
            for tile in sorted(self.stairs_crossed):
                segs += [self.grid.edge(tile, side) for side in 'NESW']
        if self.grid is not None and self._map is not None:
            # Hysteresis: an edge becomes a wall above min_fraction but is only released below wall_release_fraction. Next
            # to a wall the SLAM map thins it (the robot is close, grazing rays), so the share hovers around min_fraction
            # and the wall would flicker or vanish while it is still there.
            keys = set()
            for key, seg, share in T.edge_coverage(self._map[0], self._map[1], self.grid, self.tiles):
                if share > min_fraction or (key in self._wall_keys and share >= self.wall_release_fraction):
                    keys.add(key)
                    if key not in skip:
                        segs.append(seg)
            self._wall_keys = keys
        return segs

    def begin_terrain_look(self, label: str):
        """The classifier is asked about this tile again: answers of this look are compared with each other only."""
        self._look_rugged.discard(label)
        self._look_elev.pop(label, None)

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
        result = self._keep_highest_elevation(label, result)
        changed = self.terrain.get(label, {}).get('kind') != kind
        self.terrain[label] = result
        self.version += 1                      # republish the tile map
        return changed

    _ELEV_KEYS = ('elevation_m', 'elevation_class', 'slope_deg', 'elevation_spread_m', 'elevation_points')

    def _keep_highest_elevation(self, label: str, result: dict) -> dict:
        """Within one look a lower elevation never replaces a higher one (same idea as flat vs not flat): on the approach the
        robot's own floor and the body tilt move into the measured patch and the raised tile reads 'level' a moment after it
        read 'higher', which switched the h-step gait off just before the step. A new look (begin_terrain_look) starts afresh."""
        if 'elevation_m' not in result:
            return result
        best = self._look_elev.get(label)
        if best is not None and best['elevation_m'] - result['elevation_m'] > 0.04:
            return dict(result, **{k: best[k] for k in self._ELEV_KEYS if k in best})
        self._look_elev[label] = {k: result[k] for k in self._ELEV_KEYS if k in result}
        return result

    def elevation(self, label: str) -> Optional[dict]:
        """Elevation of a tile against the start-of-run ground reference ({elevation_m, elevation_class, slope_deg, ...}),
        or None when the classifier's elevation option is off or has not measured that tile yet."""
        res = self.terrain.get(label) or {}
        if 'elevation_m' not in res:
            return None
        return {k: res[k] for k in ('elevation_m', 'elevation_class', 'slope_deg', 'elevation_spread_m') if k in res}

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
        """Scene tiles, wherever they are: (almost) completely scanned, less than scene_max_free of the scanned area
        traversable (never entered; the scene mode looks at them, only from a tile that shares a side with them:
        scene_neighbours), plus the ones already looked at."""
        return [t for t, d in sorted(self.tiles.items()) if t in self.described or self._scene_flag(t)]

    def _scene_flag(self, tile) -> bool:
        """Looks like a scene tile and the robot has never stood on it (a tile it walked through stays a tile). A
        staircase tile is never a scene tile (the stairs mode climbs it)."""
        tile = tuple(tile)
        return (bool(self.tiles.get(tile, {}).get('scene')) and tile not in set(self.trail)
                and tile not in self.stairs and tile not in self.stairs_crossed and tile not in self.rollers)

    def mark_described(self, tile):
        """A scene tile was looked at from a neighbouring tile: it counts as explored (status visited) although the robot
        never entered it. The trail (and so the current tile) is untouched."""
        self.described.add(tuple(tile))
        self.invalidate()
        self.update()

    def is_scene(self, tile) -> bool:
        tile = tuple(tile)
        return tile in self.described or self._scene_flag(tile)

    def _open_between(self, a, b) -> bool:
        """Tiles a and b share a side the robot can look across: not a wall panel. Stricter than the tile sides
        (wall_fraction 0.4): clutter standing against the edge of a scene tile does not make the side a wall, only an
        edge that is (almost) entirely occupied does."""
        side = T.side_towards(tuple(a), tuple(b))
        if side is None or self.grid is None or self._map is None:
            return False
        grid, info = self._map
        return T.tile_sides(grid, info, self.grid, tuple(a), wall_fraction=0.9).get(side) != 'wall'

    def scene_neighbours(self, tile) -> List[tuple]:
        """Scene tiles not looked at yet that share an open side with `tile` (call with the current tile): the only
        ones the scene mode is started for."""
        i, j = tile
        return [n for n in ((i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1))
                if n not in self.described and self._scene_flag(n) and self._open_between(tile, n)]

    def pending_scenes(self) -> List[tuple]:
        """Scene tiles not looked at yet (and not given up on)."""
        return [t for t in self.scene_tiles() if t not in self.described and T.tile_label(t) not in self.unreachable]

    def scene_viewpoints(self, scene) -> List[tuple]:
        """Tiles to look at a scene tile from: neighbours across an open side that the robot can stand on (visited or
        to_visit, not a scene tile themselves)."""
        i, j = scene
        return [n for n in ((i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1))
                if self.tiles.get(n, {}).get('status') in ('visited', 'to_visit') and not self.is_scene(n)
                and self._open_between(scene, n)]



    def _drivable(self, label: str) -> bool:
        tile = tuple(int(v) for v in label.split(','))
        if tile in self.stairs or tile in self.stairs_crossed:
            return False                                # Nav2 never drives onto stairs: the stairs mode climbs them
        if tile in self.rollers:
            return False                                # nor to a roller tile: the roller mode crosses it (h-step gait)
        return not (self.skip_scene and self.is_scene(tile))

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

    def goal_point(self, tile, min_clearance_m: float = 0.25) -> Optional[List[float]]:
        """The point inside a tile the robot drives to: its centre, or the clear free point nearest it (None: none)."""
        if self.grid is None or self._map is None:
            return None
        grid, info = self._map
        p = T.tile_goal_point(info, self.grid, tuple(tile), (grid >= 0) & (grid <= 25),
                              T.clearance_map(grid, info), min_clearance_m)
        return [round(p[0], 3), round(p[1], 3)] if p else None

    # ------------------------------------------------------------- curtains
    def add_curtain_edge(self, tile, neighbour):
        """The robot walked from `tile` through a curtain into `neighbour` (tiles sharing a side)."""
        side = T.side_towards(tuple(tile), tuple(neighbour))
        if self.grid is None or side is None:
            return
        self.curtain_edges[T.edge_key(self.grid, tuple(tile), side)] = (tuple(tile), tuple(neighbour))

    def is_curtain_edge(self, a, b) -> bool:
        side = T.side_towards(tuple(a), tuple(b))
        return self.grid is not None and side is not None and T.edge_key(self.grid, tuple(a), side) in self.curtain_edges

    def defer_curtain(self, tile, neighbour, side: str):
        """A curtain on `side` of `tile` (with `neighbour` behind it) is left for later: another way is taken first."""
        self.deferred_curtains[tuple(neighbour)] = {'tile': tuple(tile), 'side': side}
        self.version += 1

    def curtains_left(self) -> List[str]:
        """Tiles behind a deferred curtain that have not been visited yet (still to explore)."""
        visited = set(self.trail)
        return [T.tile_label(t) for t in sorted(self.deferred_curtains) if t not in visited]

    def deferred_curtain(self, label: str) -> Optional[dict]:
        """The deferred curtain in front of tile `label` (not visited yet), or None."""
        tile = tuple(int(v) for v in label.split(','))
        return self.deferred_curtains.get(tile) if tile not in set(self.trail) else None

    # ------------------------------------------------------------- staircases
    def set_stair_points(self, points) -> bool:
        """Confirmed stair cells (Nx4 x, y, ux, uy, from the stair detector) -> staircase tiles. A tile the robot has
        already stood on is floor, never a staircase (stair cells can spill over the edge of the tile in front of the
        steps); a staircase tile stays one. True if the set of staircase tiles changed."""
        self.stair_points = np.asarray(points, float).reshape(-1, 4)
        if self.grid is None:
            return False
        found = SC.stair_tiles(self.grid, self.stair_points, self.stair_min_points)
        floor = set(self.trail) - set(self.stairs) - self.stairs_crossed
        stairs = {t: d for t, d in found.items() if t not in floor}
        for t, d in self.stairs.items():
            stairs.setdefault(t, d)                     # once a staircase, always a staircase
        changed = set(stairs) != set(self.stairs)
        self.stairs = stairs
        if changed:
            self.invalidate()
            self.version += 1
        return changed

    def mark_stairs_crossed(self, tiles):
        """The robot climbed over these staircase tiles: wall them off (virtual walls on all four edges)."""
        self.stairs_crossed |= set(map(tuple, tiles))
        self.invalidate()
        self.update()
        self.version += 1

    # ------------------------------------------------------------- rollers
    def set_roller_points(self, points) -> bool:
        """Confirmed roller spots (Nx5 x, y, nx, ny, lidar, from the roller detector) -> roller tiles. Unlike a staircase
        a roller tile may already be on the trail (Nav2 can drive into its front half before the roller is confirmed);
        one already crossed stays crossed. A roller tile stays one. True if the set of roller tiles changed."""
        self.roller_points = np.asarray(points, float).reshape(-1, 5)
        if self.grid is None:
            return False
        found = RC.roller_tiles(self.grid, self.roller_points, self.roller_min_points, self.roller_min_lidar,
                                self.roller_edge_margin_m)
        rollers = dict(self.rollers)
        rollers.update({t: d for t, d in found.items() if t not in self.rollers_crossed})
        changed = set(rollers) != set(self.rollers)
        self.rollers = rollers
        if changed:
            self.invalidate()
            self.version += 1
        return changed

    def mark_roller_crossed(self, roller, beyond, pose, from_tile=None) -> None:
        """The roller mode crossed `roller` (from `from_tile`) into `beyond`: the roller tile is explored (on the trail,
        its tile report is written) and the robot stands on `beyond`. The crossing is remembered: backtracking may cross
        back over it."""
        roller, beyond = tuple(roller), tuple(beyond)
        self.rollers_crossed.add(roller)
        if from_tile is not None:
            self.roller_crossings[tuple(sorted((tuple(from_tile), beyond)))] = roller
        self._visit(roller, pose, 'roller crossed')
        self._visit(beyond, pose, 'over the roller')
        self.invalidate()
        self.update()
        self.version += 1

    def roller_between(self, a, b) -> Optional[tuple]:
        """The roller tile the robot crossed between tiles a and b (either way), or None."""
        return self.roller_crossings.get(tuple(sorted((tuple(a), tuple(b)))))

    def rollers_left(self) -> List[str]:
        """Roller tiles not crossed yet."""
        return [T.tile_label(t) for t in sorted(set(self.rollers) - self.rollers_crossed)]

    def known_route(self, start, goal) -> Optional[List[tuple]]:
        """Tiles from start to goal along moves the robot has already made (consecutive trail entries, either way),
        fewest steps first; None when the trail does not link them. Never over a staircase it climbed (walled off) nor
        INTO a roller tile (Nav2 cannot drive over the roller: the lidar maps it as a wall); a roller the robot crossed is
        one hop from the tile before it to the tile beyond it (roller_between: the roller mode crosses it back)."""
        from collections import deque
        start, goal = tuple(start), tuple(goal)
        links: Dict[tuple, set] = {}
        for a, b in zip(self.trail, self.trail[1:]):
            if a in self.stairs_crossed or b in self.stairs_crossed or a in self.rollers or b in self.rollers:
                continue
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
        for a, b in self.roller_crossings:
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
        prev = {start: None}
        queue = deque([start])
        while queue:
            a = queue.popleft()
            if a == goal:
                route = []
                while a is not None:
                    route.append(a)
                    a = prev[a]
                return route[::-1]
            for b in sorted(links.get(a, ())):
                if b not in prev:
                    prev[b] = a
                    queue.append(b)
        return None

    # ------------------------------------------------------------- split tiles (backtracking)
    def mark_split(self, tile, options: Sequence[str]) -> bool:
        """The robot stands on `tile` with two or more tiles it could travel to: remember it as the last split.
        True if this is a new split (or its options changed)."""
        tile, options = tuple(tile), sorted(set(options))
        old = next((s for s in self.splits if s['tile'] == tile), None)
        if old is not None:
            self.splits.remove(old)
        self.splits.append({'tile': tile, 'options': options})      # most recent last
        self._flag_last_split()
        self.version += 1
        return old is None or old['options'] != options

    def split_remaining(self, split: dict, include_unreachable: bool = False) -> List[str]:
        """Options of a split the robot has not travelled to yet (still to_visit, drivable; with include_unreachable
        also the ones given up on)."""
        ok = ('to_visit', 'unreachable') if include_unreachable else ('to_visit',)
        out = []
        for label in split['options']:
            tile = tuple(int(v) for v in label.split(','))
            behind = self.deferred_curtain(label)
            if behind is not None and behind['tile'] == split['tile']:
                out.append(label)                   # behind a curtain left for later (often unmapped: lidar sees a wall)
            elif self.tiles.get(tile, {}).get('status') in ok and self._drivable(label):
                out.append(label)
        return out

    def last_split(self, exclude=None, include_unreachable: bool = True) -> Optional[dict]:
        """The most recent split tile that still has an untravelled option (exhausted splits are forgotten). Options
        given up on (unreachable) count unless include_unreachable is False."""
        self.splits = [s for s in self.splits if self.split_remaining(s, include_unreachable=True)]
        for s in reversed(self.splits):
            if not self.split_remaining(s, include_unreachable):
                continue
            if exclude is None or s['tile'] != tuple(exclude):
                return s
        return None

    def drop_split(self, tile):
        tile = tuple(tile)
        self.splits = [s for s in self.splits if s['tile'] != tile]
        self._flag_last_split()
        self.version += 1

    def _flag_last_split(self):
        """tiles[t]['last_split'] = True on the last split tile only (published with the tile map)."""
        for d in self.tiles.values():
            d.pop('last_split', None)
        top = self.last_split() if self.tiles else None
        if top is not None and top['tile'] in self.tiles:
            self.tiles[top['tile']]['last_split'] = True

    def as_dict(self, frame: str = 'map') -> dict:
        top = self.last_split() if self.tiles else None
        return {'frame': frame, 'grid': self.grid.as_dict() if self.grid else None, 'initial_tile': self.init_info,
                'current_tile': T.tile_label(self.current) if self.current else None,
                'counts': self.counts(), 'trail': [T.tile_label(t) for t in self.trail],
                'unreachable': dict(self.unreachable),
                'staircases': {T.tile_label(t): {'points': d['points'], 'ascent': [round(v, 2) for v in d['ascent']],
                                                 'crossed': t in self.stairs_crossed}
                               for t, d in sorted(self.stairs.items())},
                'rollers': {T.tile_label(t): {'points': d['points'], 'lidar': d.get('lidar', 0),
                                              'normal': [round(v, 2) for v in d['normal']],
                                              'crossed': t in self.rollers_crossed}
                            for t, d in sorted(self.rollers.items())},
                'last_split': ({'tile': T.tile_label(top['tile']), 'options': top['options'],
                                'remaining': self.split_remaining(top)} if top else None),
                'tiles': {T.tile_label(k): (dict(v, terrain=self.terrain[T.tile_label(k)])
                                       if T.tile_label(k) in self.terrain else v) for k, v in sorted(self.tiles.items())}}


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
