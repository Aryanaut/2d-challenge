"""Tile bookkeeping for the mission (no ROS): grid, classification, trail, failures and unreachable tiles.

The mission manager feeds it the SLAM map and the robot pose; the mission modes ask it what is left
to explore. "Identified" tiles are the ones the map has defined (to_visit / visited / blocked); a tile
is "explored" once the robot has reached it (visited). Tiles the robot failed to reach too many times
become "unreachable" so the end condition can ignore them.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lite3_explorer import roller_core as RC
from lite3_explorer import stairs_core as SC
from lite3_explorer import tiles as T
from lite3_explorer.tile_reports import within_reach


class TileTracker:
    def __init__(self, size_m: float = 1.0, reach_radius_m: float = 0.30, max_failures: int = 2,
                 on_reached: Optional[Callable[[tuple, list, str], None]] = None,
                 on_enter: Optional[Callable[[tuple], None]] = None, min_known: float = 0.5,
                 scene_min_known: float = 0.85, scene_max_free: float = 0.80, scene_margin_m: float = 0.10):
        self._init_args = (size_m, reach_radius_m, max_failures, on_reached, on_enter, min_known, scene_min_known,
                           scene_max_free, scene_margin_m)
        self.size_m = size_m
        # new map after the stairs: {'grid': TileGrid, 'stairs_crossed': set} of the previous map, expressed in the new map's
        # frame. establish() then keeps that grid (same tile numbering and coordinates) instead of placing a new one.
        self.carry: Optional[dict] = None
        # align_on_entry: every time the robot enters a tile, the grid is nudged so that tile's edges lie on the walls around
        # it (tiles.align_to_walls), within these limits
        self.align_on_entry, self.align_radius_m, self.align_max_shift_m = True, 2.0, 0.15
        self.align_max_rot_deg, self.align_min_quality = 6.0, 1.5
        self.on_aligned: Optional[Callable[[dict], None]] = None
        self.min_known = min_known           # share of a tile the lidar must have scanned before it becomes a tile
        self.scene_min_known, self.scene_max_free = scene_min_known, scene_max_free   # scene tile: scanned >= this, free/scanned < this
        self.scene_margin_m = scene_margin_m           # ... free/scanned measured this far inside the tile edges (walls)
        # a tile scanned at least scene_min_known but less than this is a scene only with a scene-prop sighting on it
        # (prop_points): partly scanned tiles beside the start (outside the maze, a wall edge) read cluttered. 0 = off
        self.scene_lidar_min_known = 0.0
        self.reach_radius_m = reach_radius_m
        # True while SLAM's map->odom has just jumped (mission manager, slam_jump_*): the robot pose may be metres off,
        # so no tile counts as reached from it ('centre reached' / 'goal reached')
        self.reach_hold = False
        self.wall_release_fraction = 0.25               # a virtual wall stays until less than this share of its edge is mapped wall
        self._wall_keys = set()                         # edges that are virtual walls now (hysteresis)
        # a mapped wall this far across a tile edge (either side) still counts as on the edge (T.edge_coverage band_m):
        # the grid is often a few cm off the maze walls in tiles the robot has not entered (no align_on_entry there)
        self.wall_band_m = 0.15
        # a shared edge at least this much mapped wall (within wall_band_m) is a wall between two tiles: no scene mode
        # across it (_open_between); clutter against the edge of a scene tile reads less
        self.scene_wall_fraction = 0.8
        # tiles with a virtual wall on all four edges (map walls, not curtains): never a target ('blocked'), see update()
        self.enclosed: set = set()
        self.skip_scene = False                        # True: scene tiles are never driven to (the scene mode looks at them)
        self.described: set = set()                    # scene tiles already looked at: explored without being entered
        self.traversable: set = set()                  # tiles the AI said are no scene (a wall seen from the side): ordinary tiles
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
        self._split_seen: set = set()                  # tiles already seen as to_visit (note_discovered_splits)
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
        # a TRAVERSABLE tile is never a staircase (the steps stand at its edge): walked on, called traversable by the AI,
        # or scanned open floor (known >= stair_floor_min_known and interior free/known >= stair_floor_min_free)
        self.stair_floor_min_known = 0.90
        self.stair_floor_min_free = 0.95
        self.stair_shift_max_m = 0.35                  # stair cells this close inside a traversable tile's edge move across it
        self.stairs: Dict[tuple, dict] = {}            # staircase tile -> {'points': n, 'ascent': (ux, uy)}
        self.stairs_crossed: set = set()
        # the exploring mode gave up on what is left on this side (nothing reachable after retries / recovery): the
        # staircase may be taken although tiles are still unexplored (stairs_ready)
        self.stairs_forced = False
        # (axis, sign): the grid axis (0 = i, 1 = j) closest to the robot's original heading in the start tile and the
        # direction along it (+1 / -1), set by establish, carried over to a new map. 'Behind' / 'beyond' the staircase
        # are measured along it (is_closed).
        self.forward: Optional[Tuple[int, int]] = None
        # reference tiles beyond staircases (stairs_core.beyond_reference): off-grid squares turned to the staircase, one
        # tile past it along the steps. The stairs mode makes one per climb and rebases odom to it afterwards ('used').
        self.stair_refs: List[dict] = []
        # the mission's initial tile: closed (never a target, never routed through, walled off for Nav2) once the robot
        # has left it. Carried over to a new map (same tile numbering).
        self.initial_tile: Optional[tuple] = None
        self.initial_left = False
        # the robot has left the initial tile only once its pose is this far outside it (the whole body has cleared
        # it, not just its centre crossed the edge): walling it off earlier would put the robot inside the walls
        self.initial_clear_m = 0.40
        # tile edges that are no curtain: (tile, side) the curtain mode gave up on (timed out / blocked walking through).
        # Curtain spots on them, or on any edge of the initial tile, are ignored (not_curtain_mask).
        self.not_curtains: List[tuple] = []
        # roller tiles (roller mode): a roller lying across the middle of a tile, which the lidar maps as a wall (the tile
        # looks like half a tile). Status 'roller' (never a Nav2 target, never unexplored) until the roller mode has
        # crossed it in the h-step gait: then it is marked explored (on the trail) and stays out of Nav2 routes.
        self.roller_points = np.zeros((0, 5))
        self.roller_min_points, self.roller_min_lidar, self.roller_edge_margin_m = 4, 2, 0.15
        self.roller_depth_only_points = 0
        self.rollers: Dict[tuple, dict] = {}           # roller tile -> {'points', 'lidar', 'normal', 'line_xy'}
        self.rollers_crossed: set = set()
        # roller crossings the robot made: {sorted (tile before, tile beyond): roller tile}. Backtracking may cross back
        # over them with the roller mode (known_route links the two tiles, roller_between names the roller tile).
        self.roller_crossings: Dict[tuple, tuple] = {}
        # scene-prop votes (prop_votes.py): map points of scene props the detector saw. A tile (not on the trail) with at
        # least prop_min_frames of them summing to prop_min_score confidence stays a scene tile (update) and an AI
        # 'traversable' answer does not reopen it (mission manager). prop_votes False = points are kept but not used.
        self.prop_votes = False
        self.prop_min_frames = 8
        self.prop_min_score = 4.0
        self.prop_points: List[tuple] = []             # (x, y, confidence, label, range_m) in the map frame
        # chosen-tile check (check_chosen_props): tiles made scene tiles when they were picked as the target with ANY
        # scene-prop sighting on them (below the prop_min_frames vote). Unlike a prop-vote tile, an AI 'traversable'
        # answer reopens one (mark_traversable): a single false sighting must not wall off a corridor.
        self.prop_chosen: set = set()
        self._map = None                              # (grid array, GridInfo)
        self._map_version = 0
        self._classified_version = -1
        self.version = 0                               # bumped on every re-classification (for publishers)

    # ------------------------------------------------------------------ setup
    # settings the mission manager sets from mission.yaml after construction: they survive reset()
    _CONFIG = ('align_on_entry', 'align_radius_m', 'align_max_shift_m', 'align_max_rot_deg', 'align_min_quality',
               'on_aligned', 'skip_scene', 'wall_release_fraction', 'wall_band_m', 'scene_wall_fraction',
               'stair_min_points', 'stair_floor_min_known',
               'stair_floor_min_free', 'stair_shift_max_m', 'roller_min_points', 'roller_min_lidar',
               'roller_depth_only_points', 'roller_edge_margin_m', 'prop_votes', 'prop_min_frames', 'prop_min_score', 'initial_clear_m')

    def reset(self):
        """Forget the grid, the map and everything learned on it (tiles, trail, failures, staircases, rollers, curtains,
        splits ...): a NEW map is starting (stairs mode, new_map). The configured settings stay. The next establish()
        places a new grid; version moves on so publishers redraw."""
        keep = {k: getattr(self, k) for k in self._CONFIG}
        version = self.version
        self.__init__(*self._init_args)
        for k, v in keep.items():
            setattr(self, k, v)
        self.version = version + 1

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
        carry, self.carry = self.carry, None
        if carry is not None:
            cand = {'id': 'inherited', 'kind': 'grid of the previous map (same tile numbering)', 'sides': {},
                    'grid': carry['grid']}
            how = 'inherited from the previous map'
        elif self._map is not None:
            cands = T.initial_candidates(self._map[0], self._map[1], pose, size=self.size_m)
            cand = cands[0] if cands else None
        if cand is None:
            half = self.size_m / 2.0
            cand = {'id': '-', 'kind': 'centred on the robot, along the map axes', 'sides': {},
                    'grid': T.TileGrid(pose[0] - half, pose[1] - half, 0.0, self.size_m)}
            how = 'no map'
        self.grid = cand['grid']
        self.terrain = {}
        yaw0 = pose[2] if len(pose) > 2 else 0.0
        if carry is not None and carry.get('forward'):
            self.forward = tuple(carry['forward'])
        else:
            self.forward = forward_axis(yaw0 - self.grid.yaw)
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
        self.start_tile = tuple(start)
        if carry is not None:
            self.initial_tile, self.initial_left = carry.get('initial_tile'), bool(carry.get('initial_left'))
        else:
            self.initial_tile, self.initial_left = tuple(start), False
        self.trail, self.trail_log = [], []
        self._wall_keys = set()
        self.described = set()
        self.traversable = set()
        self.splits = []
        self._split_seen = set()
        self.curtain_edges = {}
        self.deferred_curtains = {}
        self.not_curtains = [tuple(e) for e in carry.get('not_curtains', ())] if carry else []
        self.stairs, self.stairs_crossed = {}, set(carry['stairs_crossed'] if carry else ())
        self.stair_refs = []
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
                                      scene_min_known=self.scene_min_known, scene_max_free=self.scene_max_free,
                                      scene_margin_m=self.scene_margin_m)
        for tile, t in self.tiles.items():
            if t['status'] != 'unknown':
                shares = T.side_shares(grid, info, self.grid, tile)
                t['sides'] = T.sides_from_shares(shares)
                t['wall_share'] = {s: occ for s, (occ, _) in shares.items()}
                t['free_share'] = {s: free for s, (_, free) in shares.items()}
            if t['status'] == 'to_visit' and T.tile_label(tile) in self.unreachable:
                t['status'] = 'unreachable'
        for tile in self.traversable:
            t = self.tiles.get(tile)
            if t is not None:
                t['scene'] = False
                if t['status'] == 'blocked':                   # the AI looked: it can be walked on
                    t['status'] = 'to_visit'
        if self.scene_lidar_min_known > 0:                     # partly scanned: the lidar alone is not enough
            props = self.prop_counts()
            for tile, t in self.tiles.items():
                if t.get('scene') and t['known'] < self.scene_lidar_min_known and tile not in self.described \
                        and not props.get(tile, {}).get('frames'):
                    t['scene'] = False
                    t['scene_weak'] = True                     # (in /tile_map) would be a scene at scene_min_known
        if self.prop_votes:                                    # scene props seen on it: a scene tile, whatever the lidar / AI say
            trail = set(self.trail)
            for tile in self.prop_tiles():
                t = self.tiles.get(tile)
                if t is not None and tile not in trail:
                    t['scene'], t['prop_scene'] = True, True
        trail = set(self.trail)
        for tile in self.prop_chosen - trail - self.traversable:      # picked as the target with props seen on it
            t = self.tiles.get(tile)
            if t is not None and t.get('status') not in ('staircase', 'roller', 'closed'):
                t['scene'], t['prop_scene'], t['prop_chosen'] = True, True, True
        if len(self.stair_points) and self._assign_stairs():   # the map grew: the staircase may only now be found
            self.version += 1
        for tile in set(self.stairs) | self.stairs_crossed:
            t = self.tiles.setdefault(tile, {'free': 0.0, 'occupied': 0.0, 'known': 0.0, 'unmapped_edge': False,
                                             'scene': False})
            t['status'], t['scene'] = 'staircase', False
        for tile in set(self.rollers) - self.rollers_crossed:
            t = self.tiles.setdefault(tile, {'free': 0.0, 'occupied': 0.0, 'known': 0.0, 'unmapped_edge': False,
                                             'scene': False})
            t['status'], t['scene'] = 'roller', False
        for tile, t in self.tiles.items():
            if self.is_closed(tile):
                t['status'], t['scene'] = 'closed', False      # beyond the staircase (before the climb) / behind it (after), the initial tile once left
        self._mark_enclosed()
        self._flag_last_split()
        return True

    def _enclosed_tiles(self) -> set:
        """Tiles whose four edges are all virtual walls from the map (_wall_keys, as last computed by virtual_walls).
        Curtain edges do not count (the robot walks through them). Tiles the robot stands on or has walked on are never
        enclosed (it got in, so a side is open), nor staircase / roller tiles (their own modes cross them)."""
        if self.grid is None or not self._wall_keys:
            return set()
        walls = self._wall_keys - set(self.curtain_edges)
        on_trail = set(self.trail)
        out = set()
        for tile in self.tiles:
            if (tile in on_trail or tile == self.current or tile in self.stairs or tile in self.stairs_crossed
                    or tile in self.rollers):
                continue
            if all(T.edge_key(self.grid, tile, side) in walls for side in T.SIDES):
                out.add(tile)
        return out

    def _mark_enclosed(self):
        """A tile walled in on all four sides is not traversable: status 'blocked' (never a target, not unexplored)
        and flagged 'enclosed'. A scene tile already looked at keeps 'visited' (it is explored) but is flagged too."""
        self.enclosed = self._enclosed_tiles()
        for tile in self.enclosed:
            t = self.tiles[tile]
            t['enclosed'] = True
            if t['status'] in ('to_visit', 'unreachable', 'unknown'):
                t['status'] = 'blocked'

    def virtual_walls(self, min_fraction: float = 0.5, start_wall: bool = True):
        """World segments to block in the planner: the wall behind the start pose, every tile edge more than
        min_fraction covered by mapped wall, all four edges of every staircase the robot has climbed over and the
        staircase column's edge towards the closed side (beyond it before the climb, behind it after: is_closed). With
        open_curtains (backtracking) edges with a curtain are left out."""
        segs = []
        skip = set(self.curtain_edges) if self.open_curtains else set()
        if start_wall and getattr(self, 'start_wall', None):
            segs.append(self.start_wall)
        if self.grid is not None:
            for tile in sorted(self.stairs_crossed):
                segs += [self.grid.edge(tile, side) for side in 'NESW']
            segs += self._closed_wall()                 # the closed side of the staircase (is_closed)
            if self._initial_closed() and self.current != self.initial_tile:
                segs += [self.grid.edge(self.initial_tile, side) for side in 'NESW']
        if self.grid is not None and self._map is not None:
            # Hysteresis: an edge becomes a wall above min_fraction but is only released below wall_release_fraction. Next
            # to a wall the SLAM map thins it (the robot is close, grazing rays), so the share hovers around min_fraction
            # and the wall would flicker or vanish while it is still there.
            keys = set()
            for key, seg, share in T.edge_coverage(self._map[0], self._map[1], self.grid, self.tiles,
                                                   band_m=self.wall_band_m):
                if share > min_fraction or (key in self._wall_keys and share >= self.wall_release_fraction):
                    keys.add(key)
                    if key not in skip:
                        segs.append(seg)
            if keys != self._wall_keys:
                self._wall_keys = keys
                if self._enclosed_tiles() != self.enclosed:
                    self.invalidate()                   # re-classify: a tile became / stopped being enclosed
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
        self._check_initial_cleared(pose)
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

    def relocate(self, assumed, pose, source: str = 'relocated') -> Optional[tuple]:
        """The robot was put on tile `assumed` (e.g. after an AI-mode climb, from dead reckoning) but a scan has since
        placed it by SLAM: the tile its pose is really in. A different tile: `assumed` is taken off the trail (when the
        robot never stood on it before) and the real one visited instead. Returns the real tile."""
        if self.grid is None or pose is None:
            return None
        assumed = tuple(assumed)
        actual = tuple(self.grid.tile_of(pose[0], pose[1]))
        if actual == assumed:
            return actual
        if self.trail.count(assumed) == 1:                   # the robot never stood on it otherwise
            i = self.trail.index(assumed)
            del self.trail[i], self.trail_log[i]
        self._visit(actual, pose, source)
        self.invalidate()
        return actual

    def align_grid(self, tile, pose) -> Optional[dict]:
        """Nudge the grid so the edges of `tile` (the robot's) lie on the detected walls (tiles.align_to_walls). Everything
        keyed on world edges (virtual walls, curtain edges, the start wall) is re-derived. Returns the correction or None."""
        if not self.align_on_entry or self.grid is None or self._map is None or pose is None:
            return None
        res = T.align_to_walls(self._map[0], self._map[1], self.grid, tuple(tile), pose[:2], radius_m=self.align_radius_m,
                               max_shift_m=self.align_max_shift_m, max_rot_deg=self.align_max_rot_deg,
                               min_quality=self.align_min_quality)
        if res is None:
            return None
        old, new = self.grid, res['grid']
        curtains = list(self.curtain_edges.values())
        self.grid.origin_x, self.grid.origin_y, self.grid.yaw = new.origin_x, new.origin_y, new.yaw   # in place: others hold it
        self.curtain_edges = {}
        for a, b in curtains:
            self.add_curtain_edge(a, b)
        st = getattr(self, 'start_tile', None)
        if st is not None and getattr(self, 'start_wall_side', None):
            self.start_wall = self.grid.edge(st, self.start_wall_side)
        self._wall_keys = set()
        if self.init_info is not None:
            self.init_info['grid'] = self.grid.as_dict()
        self.invalidate()
        self.version += 1
        res = dict(res, tile=list(tile))
        res.pop('grid')
        if self.on_aligned is not None:
            self.on_aligned(res)
        return res

    def _visit(self, tile, pose, source):
        if self.trail and self.trail[-1] == tile:
            return None
        if self.reach_hold and source in ('centre reached', 'goal reached'):
            return None
        if source in ('centre reached', 'goal reached'):
            self.align_grid(tile, pose)
        new = tile not in set(self.trail)
        self.trail.append(tile)
        self._check_initial_cleared(pose)
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

    def prop_counts(self) -> Dict[tuple, dict]:
        """Per tile: {'frames': n, 'score': summed confidence, 'labels': {label: n}} of the scene-prop points on it
        (counted on the CURRENT grid, so a grid nudge moves them with the walls)."""
        out: Dict[tuple, dict] = {}
        if self.grid is None:
            return out
        for x, y, conf, lab, _ in self.prop_points:
            e = out.setdefault(tuple(self.grid.tile_of(x, y)), {'frames': 0, 'score': 0.0, 'labels': {}})
            e['frames'] += 1
            e['score'] = round(e['score'] + conf, 2)
            e['labels'][lab] = e['labels'].get(lab, 0) + 1
        return out

    def prop_tiles(self) -> set:
        """Tiles with enough scene-prop votes (prop_min_frames points, prop_min_score summed confidence)."""
        return {t for t, e in self.prop_counts().items()
                if e['frames'] >= self.prop_min_frames and e['score'] >= self.prop_min_score}

    def is_prop_scene(self, tile) -> bool:
        """prop_votes on and the tile has enough scene-prop votes."""
        return bool(self.prop_votes) and tuple(tile) in self.prop_tiles()

    def check_chosen_props(self, tile, min_frames: int = 1, min_score: float = 0.0) -> Optional[dict]:
        """The exploration picked this tile as its target: scene props seen on it (at least min_frames sightings summing to
        min_score) -> it becomes a scene tile (looked at, never entered). Returns its prop counts then, else None. Not for
        a tile on the trail, one the AI said is traversable, a staircase / roller tile or one already a scene tile."""
        tile = tuple(tile)
        e = self.prop_counts().get(tile)
        if (not e or e['frames'] < max(int(min_frames), 1) or e['score'] < float(min_score)
                or tile in set(self.trail) or tile in self.traversable or tile in self.stairs
                or tile in self.stairs_crossed or tile in self.rollers or self.is_scene(tile)):
            return None
        self.prop_chosen.add(tile)
        self.invalidate()
        self.update()
        return e

    def add_prop_points(self, points, max_points: int = 20000) -> List[tuple]:
        """Add scene-prop map points; returns the tiles that just got enough votes (re-classifies when there are any)."""
        if not points:
            return []
        before = self.prop_tiles()
        self.prop_points.extend(points)
        if len(self.prop_points) > max_points:
            del self.prop_points[:len(self.prop_points) - max_points]
        new = sorted(self.prop_tiles() - before)
        weak = self.scene_lidar_min_known > 0 and self.grid is not None and any(
            self.tiles.get(tuple(self.grid.tile_of(p[0], p[1])), {}).get('scene_weak') for p in points)
        if (new and self.prop_votes) or weak:               # weak: a partly scanned cluttered tile, now with a prop on it
            self.invalidate()
            self.update()
        return new

    def mark_described(self, tile):
        """A scene tile was looked at from a neighbouring tile: it counts as explored (status visited) although the robot
        never entered it. The trail (and so the current tile) is untouched."""
        self.described.add(tuple(tile))
        self.invalidate()
        self.update()

    def mark_traversable(self, tile):
        """The AI said the scene tile is none of the scene types (e.g. a dark panel that is only the wall of a traversable
        tile): it becomes an ordinary tile to explore and is never a scene tile again."""
        self.traversable.add(tuple(tile))
        self.described.discard(tuple(tile))
        self.prop_chosen.discard(tuple(tile))
        self.invalidate()
        self.update()

    def is_scene(self, tile) -> bool:
        tile = tuple(tile)
        return tile in self.described or self._scene_flag(tile)

    def _open_between(self, a, b) -> bool:
        """Tiles a and b share a side the robot can look across: no wall between them. The edge counts as a wall when at
        least scene_wall_fraction of it has mapped wall within wall_band_m across it (either side, as for the virtual
        walls): the grid is often a few cm off the maze walls in tiles the robot has not entered, and a wall 12 cm off the
        edge reads ~0.1 on the edge line itself, so a scene behind it was looked at through the wall. Stricter than the
        virtual walls (> 0.5): clutter standing against the edge of a scene tile (half the edge: ~0.6) is no wall."""
        side = T.side_towards(tuple(a), tuple(b))
        if side is None or self.grid is None or self._map is None:
            return False
        grid, info = self._map
        key = T.edge_key(self.grid, tuple(a), side)
        for k, _, share in T.edge_coverage(grid, info, self.grid, {tuple(a): {'status': 'edge'}}, band_m=self.wall_band_m):
            if k == key:
                return share < self.scene_wall_fraction
        return True

    @staticmethod
    def _seg_key(seg) -> tuple:
        """edge_key of a world segment (rounded, sorted end points)."""
        (x0, y0), (x1, y1) = seg
        return tuple(sorted(((round(x0, 2), round(y0, 2)), (round(x1, 2), round(y1, 2)))))

    def _virtual_wall_keys(self) -> set:
        """Edge keys of the virtual walls: map walls (_wall_keys, curtain edges left out), the start wall, the edges of
        crossed staircases and the closed side (as in virtual_walls)."""
        keys = set(self._wall_keys) - set(self.curtain_edges)
        if getattr(self, 'start_wall', None):
            keys.add(self._seg_key(self.start_wall))
        if self.grid is not None:
            segs = [self.grid.edge(tile, side) for tile in self.stairs_crossed for side in T.SIDES]
            segs += self._closed_wall()
            if self._initial_closed() and self.current != self.initial_tile:
                segs += [self.grid.edge(self.initial_tile, side) for side in T.SIDES]
            keys |= {self._seg_key(s) for s in segs}
        return keys

    def _virtual_wall_scanned(self, a, b) -> bool:
        """The side tiles a and b share is a virtual wall AND the lidar has mapped wall on it (T.tile_sides 'wall':
        >= 0.4 of the edge line within ±1 cell): a real wall, the camera cannot look across it, however little of it the
        wider wall band (_open_between, scene_wall_fraction) has seen yet. Bag fullrun_26_11_141104: scene 0,-1 looked
        at from the start tile across the start wall (band share 0.50 < 0.8, edge line 0.45 = 'wall'): the pictures
        were of the wall."""
        side = T.side_towards(tuple(a), tuple(b))
        if side is None or self.grid is None or self._map is None:
            return False
        if T.edge_key(self.grid, tuple(a), side) not in self._virtual_wall_keys():
            return False
        grid, info = self._map
        return T.tile_sides(grid, info, self.grid, tuple(a)).get(side) == 'wall'

    def _view_open(self, viewer, scene) -> bool:
        """The scene tile can be looked at from its neighbour `viewer` across their shared side: _open_between, or,
        when _open_between calls EVERY side of the scene tile a wall, a side with no straight wall line on it
        (T.edge_line_share < scene_wall_fraction). The lidar scanned the scene tile, so some side of it is open: all
        four reading as wall means clutter close to the edges fills the wall band (sim scene clutter sits 0.15 m in,
        exactly wall_band_m: bag backtracking_20261126_131935, scenes 0,-1 / 2,-1 / 0,-2 'without a viewpoint' while
        the robot stood beside them, then retrying them forever). Not a scene tile: _open_between."""
        viewer, scene = tuple(viewer), tuple(scene)
        if self._virtual_wall_scanned(viewer, scene):
            return False
        if self._open_between(viewer, scene):
            return True
        if T.side_towards(viewer, scene) is None or self.grid is None or self._map is None or not self.is_scene(scene):
            return False
        i, j = scene
        if any(self._open_between(scene, n) for n in ((i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1))):
            return False                                # a real open side elsewhere: look from there
        grid, info = self._map
        return T.edge_line_share(grid, info, self.grid, viewer, T.side_towards(viewer, scene),
                                 band_m=self.wall_band_m) < self.scene_wall_fraction

    def wall_between(self, a, b) -> bool:
        """Neighbouring tiles a and b are separated by a wall (see _open_between; b a scene tile: _view_open).
        Not neighbours: False."""
        return T.side_towards(tuple(a), tuple(b)) is not None and not self._view_open(a, b)

    def scene_neighbours(self, tile) -> List[tuple]:
        """Scene tiles not looked at yet that share an open side with `tile` (call with the current tile): the only
        ones the scene mode is started for."""
        i, j = tile
        return [n for n in ((i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1))
                if n not in self.described and self._scene_flag(n) and self._view_open(tile, n)]

    def pending_scenes(self) -> List[tuple]:
        """Scene tiles not looked at yet (and not given up on)."""
        return [t for t in self.scene_tiles() if t not in self.described and T.tile_label(t) not in self.unreachable]

    def scene_viewpoints(self, scene) -> List[tuple]:
        """Tiles to look at a scene tile from: neighbours across an open side that the robot can stand on (visited or
        to_visit, not a scene tile themselves)."""
        i, j = scene
        return [n for n in ((i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1))
                if self.tiles.get(n, {}).get('status') in ('visited', 'to_visit') and not self.is_scene(n)
                and self._view_open(n, scene)]



    def _drivable(self, label: str) -> bool:
        tile = tuple(int(v) for v in label.split(','))
        if tile in self.stairs or tile in self.stairs_crossed:
            return False                                # Nav2 never drives onto stairs: the stairs mode climbs them
        if self.is_closed(tile):
            return False                                # the closed side of the staircase
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

    def recovery_targets(self, robot_xy, *, limit: int = 5, min_distance_m: float = 0.25,
                         min_clearance_m: float = 0.25, exclude: Sequence[str] = (),
                         adjacent_only: bool = False) -> List[dict]:
        """Recovery setpoints: centres of previously explored tiles (ones the robot has stood on), nearest first.

        Never a tile that was only seen (to_visit) or only described from a neighbour: the robot is stuck, so it only
        goes back to ground it has already walked. adjacent_only prefers the robot's own tile and the tiles sharing a
        side with it; when none of those is an explored, clear tile, the nearest explored tiles anywhere are offered
        so recovery always has a setpoint."""
        if self.grid is None or self._map is None:
            return []
        grid, info = self._map
        explored = set(self.trail)
        roller_tiles = set(self.rollers) | set(self.rollers_crossed)
        excl = {tuple(int(v) for v in lab.split(',')) for lab in exclude}
        # never recover onto a roller tile (crossed or not): the lidar maps it as a wall, only the roller mode crosses it
        excl |= roller_tiles | {t for t in self.tiles if t not in explored}
        here = self.grid.tile_of(robot_xy[0], robot_xy[1])

        def find(adjacent_to):
            out = T.nearest_tile_centres(grid, info, self.grid, self.tiles, robot_xy, statuses=('visited',),
                                         limit=limit + len(excl), min_distance_m=min_distance_m,
                                         min_clearance_m=min_clearance_m, exclude=sorted(excl),
                                         adjacent_to=adjacent_to)
            return [t for t in out if self._drivable(t['tile'])][:limit]

        found = find(here) if adjacent_only else []
        return found or find(None)

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

    def reject_curtain(self, tile, side: str) -> None:
        """The curtain on `side` of `tile` was no curtain (the curtain mode timed out / was blocked walking through it):
        never a curtain again, and no longer a deferred one."""
        tile = tuple(tile)
        if (tile, side) not in self.not_curtains:
            self.not_curtains.append((tile, side))
        di, dj = {'E': (1, 0), 'W': (-1, 0), 'N': (0, 1), 'S': (0, -1)}[side]
        self.deferred_curtains.pop((tile[0] + di, tile[1] + dj), None)
        self.version += 1

    def not_curtain_edges(self) -> List[tuple]:
        """(tile, side) of every edge that cannot hold a curtain: the rejected ones and the four edges of the initial
        tile."""
        out = list(self.not_curtains)
        if self.initial_tile is not None:
            out += [(tuple(self.initial_tile), side) for side in 'NESW']
        return out

    def is_not_curtain(self, tile, side: str) -> bool:
        """True when this edge (from either tile) cannot hold a curtain (not_curtain_edges)."""
        if self.grid is None:
            return False
        key = T.edge_key(self.grid, tuple(tile), side)
        return any(T.edge_key(self.grid, t, sd) == key for t, sd in self.not_curtain_edges())

    def not_curtain_mask(self, points, tol_m: float = 0.15) -> np.ndarray:
        """Rows of `points` (x, y, ...) that may be curtain spots: False within tol_m of an edge that cannot hold a
        curtain (not_curtain_edges)."""
        pts = np.asarray(points, float)
        keep = np.ones(len(pts), bool)
        if self.grid is None or not len(pts):
            return keep
        xy = pts[:, :2]
        for tile, side in self.not_curtain_edges():
            (x0, y0), (x1, y1) = self.grid.edge(tile, side)
            a, d = np.array([x0, y0]), np.array([x1 - x0, y1 - y0])
            t = np.clip(((xy - a) @ d) / max(float(d @ d), 1e-9), 0.0, 1.0)
            keep &= np.hypot(*(xy - (a + t[:, None] * d)).T) > tol_m
        return keep

    def curtains_left(self) -> List[str]:
        """Tiles behind a deferred curtain that have not been visited yet (still to explore)."""
        visited = set(self.trail)
        return [T.tile_label(t) for t in sorted(self.deferred_curtains)
                if t not in visited and not self.is_closed(t) and not self.is_closed(self.deferred_curtains[t]['tile'])]

    def deferred_curtain(self, label: str) -> Optional[dict]:
        """The deferred curtain in front of tile `label` (not visited yet), or None."""
        tile = tuple(int(v) for v in label.split(','))
        if tile in set(self.trail) or self.is_closed(tile):
            return None
        info = self.deferred_curtains.get(tile)
        return None if info is None or self.is_closed(info['tile']) else info

    # ------------------------------------------------------------- staircases
    def set_stair_points(self, points) -> bool:
        """Confirmed stair cells (Nx4 x, y, ux, uy, from the stair detector) -> the staircase tile. The maze has ONE
        staircase: the first tile found is locked as it for the rest of the run (see _assign_stairs). A tile the robot has
        already stood on is floor, never a staircase (stair cells can spill over the edge of the tile in front of the
        steps). True if the staircase tile changed (i.e. it was just found)."""
        self.stair_points = np.asarray(points, float).reshape(-1, 4)
        if self.grid is None:
            return False
        changed = self._assign_stairs()
        if changed:
            self.invalidate()
            self.version += 1
        return changed

    def _assign_stairs(self) -> bool:
        """self.stairs from the stair cells. There is only one staircase in the maze: once a tile is a staircase it stays
        THE staircase for the rest of the run (more stair cells, a scan that later looks like open floor or cells spilling
        into other tiles change nothing), and once it has been climbed (stairs_crossed, carried over to a new map) no
        other tile becomes one. The first time, the tile with the most stair cells wins. True if it was just found."""
        if self.stairs_crossed:
            return False                                # climbed already: the one staircase is known and walled off
        if self.stairs:
            tile = next(iter(self.stairs))              # locked: only its stair data is refreshed
            found = SC.stair_tiles(self.grid, self.stair_points, self.stair_min_points)
            if tile in found:
                self.stairs[tile] = found[tile]
            return False
        scanned_open = self.scanned_open_tiles()
        floor = set(self.trail) | self.traversable | scanned_open
        found = {t: d for t, d in SC.stair_tiles(self.grid, self.stair_points, self.stair_min_points, traversable=floor,
                                                 shift_max_m=self.stair_shift_max_m).items() if t not in floor}
        if not found:
            return False
        tile = max(sorted(found), key=lambda t: found[t]['points'])
        self.stairs = {tile: found[tile]}
        return True

    def stairs_ready(self) -> bool:
        """The staircase may be taken only once everything on this side is done: no tile left to visit, no scene tile
        left to look at, no tile behind a curtain left for later (given-up tiles do not count). stairs_forced: the
        exploring mode found nothing reachable left on this side, so the rest no longer holds the stairs back."""
        if self.staircase_tile is None or self.stairs_crossed:
            return False
        if self.stairs_forced:
            return True
        return (not self.unexplored() and not (self.skip_scene and self.pending_scenes())
                and not self.curtains_left())

    def scanned_open_tiles(self) -> set:
        """Tiles the lidar has scanned as open floor (stair_floor_min_known / stair_floor_min_free): no staircase there."""
        return {tile for tile, t in self.tiles.items()
                if t.get('known', 0.0) >= self.stair_floor_min_known
                and t.get('scene_free', 0.0) >= self.stair_floor_min_free}

    def mark_stairs_crossed(self, tiles):
        """The robot climbed over these staircase tiles: wall them off (virtual walls on all four edges), and close every
        tile behind it along the robot's original heading (is_closed)."""
        self.stairs_crossed |= set(map(tuple, tiles))
        self.invalidate()
        self.update()
        self.version += 1

    @property
    def staircase_tile(self) -> Optional[tuple]:
        """THE staircase (one per maze): the climbed one, else the locked detected one, else None."""
        if self.stairs_crossed:
            return min(self.stairs_crossed)
        return next(iter(self.stairs)) if self.stairs else None

    def is_closed(self, tile) -> bool:
        """Tiles the robot may not traverse because of the staircase, along the grid axis (i or j) of the robot's
        original heading in the start tile (forward):
          before the climb (staircase found, not crossed): every tile BEYOND it (further along forward than it);
          after the climb: every tile BEHIND it (on the start side) -- no way back over or around the stairs.
        Tiles level with the staircase (same i, or same j when heading along j) stay open. Nothing is closed before a
        staircase is found.
        The initial tile is closed too once the robot has left it (_initial_closed)."""
        if self._initial_closed() and tuple(tile) == self.initial_tile:
            return True
        stair, fwd = self.staircase_tile, self.forward
        if stair is None or not fwd:
            return False
        tile = tuple(tile)
        if tile == stair:
            return False                                # the staircase keeps its own status
        axis, s = fwd
        d = s * (tile[axis] - stair[axis])
        return d < 0 if self.stairs_crossed else d > 0

    def _check_initial_cleared(self, pose) -> bool:
        """Close the initial tile (initial_left) once the robot's pose is at least initial_clear_m outside it: its body
        has cleared the tile. Entering the next tile is not enough (the robot still straddles the edge)."""
        if self.initial_left or self.initial_tile is None or self.grid is None or pose is None:
            return self.initial_left
        u, v = self.grid.to_local(pose[0], pose[1])
        i, j = self.initial_tile
        s = self.grid.size
        du = max(i * s - float(u), 0.0, float(u) - (i + 1) * s)
        dv = max(j * s - float(v), 0.0, float(v) - (j + 1) * s)
        if math.hypot(du, dv) >= self.initial_clear_m:
            self.initial_left = True                    # the initial tile is closed from now on (is_closed)
            self.invalidate()
            self.version += 1
        return self.initial_left

    def _initial_closed(self) -> bool:
        """The robot has left the initial tile: it is not traversable any more."""
        return self.initial_tile is not None and self.initial_left

    def _closed_wall(self):
        """The virtual wall (list of tile-edge segments) between the staircase's row/column (across forward) and the
        closed side: its edge on the beyond side before the climb, on the behind side after it, across every known tile
        (+2 either side)."""
        stair, fwd = self.staircase_tile, self.forward
        if stair is None or not fwd or self.grid is None:
            return []
        axis, s = fwd
        ahead = ('E' if s > 0 else 'W') if axis == 0 else ('N' if s > 0 else 'S')
        side = {'E': 'W', 'W': 'E', 'N': 'S', 'S': 'N'}[ahead] if self.stairs_crossed else ahead
        other = 1 - axis
        ks = [t[other] for t in self.tiles] + [stair[other]]
        out = []
        for k in range(min(ks) - 2, max(ks) + 3):
            tile = [0, 0]
            tile[axis], tile[other] = stair[axis], k
            out.append(self.grid.edge(tuple(tile), side))
        return out

    # ------------------------------------------------------------- rollers
    def set_roller_points(self, points) -> bool:
        """Confirmed roller spots (Nx5 x, y, nx, ny, lidar, from the roller detector) -> roller tiles. Unlike a staircase
        a roller tile may already be on the trail (Nav2 can drive into its front half before the roller is confirmed);
        one already crossed stays crossed. A roller tile stays one. True if the set of roller tiles changed."""
        self.roller_points = np.asarray(points, float).reshape(-1, 5)
        if self.grid is None:
            return False
        found = RC.roller_tiles(self.grid, self.roller_points, self.roller_min_points, self.roller_min_lidar,
                                self.roller_edge_margin_m, self.roller_depth_only_points)
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
            if self.is_closed(a) or self.is_closed(b):
                continue                                # never through the closed side of the staircase
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
        for a, b in self.roller_crossings:
            if self.is_closed(a) or self.is_closed(b):
                continue
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

    def note_discovered_splits(self) -> List[dict]:
        """Tiles that became to_visit since the last call and share a side (no wall on it) with a tile the robot visited
        before (not the current one): that earlier tile becomes a split again, its options all of its untravelled
        neighbours across an open side, so backtracking can come back for the new tile. Most recently visited last
        (so the latest of them is the last split). Returns [{'tile', 'options', 'new'}] for the splits that changed."""
        fresh = {t for t, d in self.tiles.items() if d.get('status') == 'to_visit'} - self._split_seen
        self._split_seen |= fresh
        visited = set(self.trail)
        found: Dict[tuple, List[str]] = {}
        for tile in sorted(fresh):
            if not self._drivable(T.tile_label(tile)):
                continue
            for side in T.SIDES:
                prev = T._step(tile, side)
                if prev in visited and prev != self.current and T._joined(self.tiles, tile, side) \
                        and not self.is_closed(prev):
                    found.setdefault(prev, []).append(T.tile_label(tile))
        last_seen = {t: i for i, t in enumerate(self.trail)}
        out = []
        for prev in sorted(found, key=lambda t: last_seen[t]):
            options = set(found[prev])
            for side in T.SIDES:
                b = T._step(prev, side)
                if self.tiles.get(b, {}).get('status') in ('to_visit', 'unreachable') and T._joined(self.tiles, prev, side) \
                        and self._drivable(T.tile_label(b)):
                    options.add(T.tile_label(b))
            old = next((s for s in self.splits if s['tile'] == prev), None)
            if old is not None:
                options |= set(self.split_remaining(old, include_unreachable=True))
            if self.mark_split(prev, sorted(options)):
                out.append({'tile': T.tile_label(prev), 'options': sorted(options), 'new': sorted(found[prev])})
        return out

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
            elif self.stairs_forced and tile in self.stairs and tile not in self.stairs_crossed:
                out.append(label)                   # the staircase to climb (the stairs mode starts at the split tile)
            elif self.tiles.get(tile, {}).get('status') in ok and self._drivable(label):
                out.append(label)
        return out

    def last_split(self, exclude=None, include_unreachable: bool = True) -> Optional[dict]:
        """The most recent split tile that still has an untravelled option (exhausted splits are forgotten). Options
        given up on (unreachable) count unless include_unreachable is False."""
        self.splits = [s for s in self.splits
                       if not self.is_closed(s['tile']) and self.split_remaining(s, include_unreachable=True)]
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
                'stair_refs': [{'stairs': [T.tile_label(t) for t in r['stairs']],
                                'centre_xy': [round(v, 3) for v in r['centre_xy']],
                                'yaw_deg': round(math.degrees(r['yaw']), 1), 'size_m': r['size'],
                                'grid_tile': T.tile_label(r['grid_tile']), 'used': bool(r.get('used'))}
                               for r in self.stair_refs],
                'rollers': {T.tile_label(t): {'points': d['points'], 'lidar': d.get('lidar', 0),
                                              'normal': [round(v, 2) for v in d['normal']],
                                              'crossed': t in self.rollers_crossed}
                            for t, d in sorted(self.rollers.items())},
                'last_split': ({'tile': T.tile_label(top['tile']), 'options': top['options'],
                                'remaining': self.split_remaining(top)} if top else None),
                'tiles': {T.tile_label(k): (dict(v, terrain=self.terrain[T.tile_label(k)])
                                       if T.tile_label(k) in self.terrain else v) for k, v in sorted(self.tiles.items())}}


def forward_axis(heading: float) -> Tuple[int, int]:
    """(axis, sign) of a heading in the grid frame (rad, 0 = +i, pi/2 = +j): the grid axis it is closest to (0 = i,
    1 = j) and the direction along it (+1 / -1). Exactly diagonal goes to i."""
    c, s = math.cos(heading), math.sin(heading)
    if abs(c) >= abs(s):
        return 0, (1 if c >= 0.0 else -1)
    return 1, (1 if s >= 0.0 else -1)


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
