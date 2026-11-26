import math

import numpy as np

from lite3_explorer import map_grid as fc
from lite3_explorer import tiles as T

RES = 0.05
INFO = fc.GridInfo(200, 200, RES, -5.0, -5.0)
ANGLE = math.radians(12)
OFFSET = (0.3, 0.6)           # maze grid lines at u = 0.3 + k, v = 0.6 + k (rotated frame)


def maze():
    """Rotated maze, walls on a 1 m grid; robot starts in a cell with walls N and E."""
    g = np.full((200, 200), -1, np.int16)
    cols, rows = np.meshgrid(np.arange(200), np.arange(200))
    x = INFO.origin_x + (cols + 0.5) * RES
    y = INFO.origin_y + (rows + 0.5) * RES
    c, s = math.cos(ANGLE), math.sin(ANGLE)
    u, v = c * x + s * y - OFFSET[0], -s * x + c * y - OFFSET[1]
    g[(u > -2) & (u < 3) & (v > -1) & (v < 3) & (u ** 2 + v ** 2 < 9)] = 0

    def wall(m):
        g[m & (g == 0)] = 100
    t = 0.04
    wall((abs(u + 2) < t) & (v > -1) & (v < 3)); wall((abs(u - 3) < t) & (v > -1) & (v < 3))
    wall((abs(v + 1) < t) & (u > -2) & (u < 3)); wall((abs(v - 3) < t) & (u > -2) & (u < 1))
    wall((abs(u - 1) < t) & (v > -1) & (v < 2)); wall((abs(v - 1) < t) & (u > -1) & (u < 1))
    wall((abs(u + 1) < t) & (v > 0) & (v < 3))
    rx = c * (0.45 + OFFSET[0]) - s * (0.4 + OFFSET[1])
    ry = s * (0.45 + OFFSET[0]) + c * (0.4 + OFFSET[1])
    return g, (rx, ry, ANGLE + 0.3)


def test_grid_round_trip_and_geometry():
    tg = T.TileGrid(1.0, 2.0, math.radians(30))
    x, y = tg.center((2, -1))
    assert tg.tile_of(x, y) == (2, -1)
    u, v = tg.to_local(*tg.to_world(0.25, 0.75))
    assert abs(u - 0.25) < 1e-9 and abs(v - 0.75) < 1e-9
    sw, se, ne, nw = tg.corners((0, 0))
    assert abs(math.dist(sw, se) - 1.0) < 1e-9 and tg.edge((0, 0), 'N') == (nw, ne)


def test_initial_candidate_snaps_to_walls_and_reads_sides():
    g, robot = maze()
    cands = T.initial_candidates(g, INFO, robot)
    a = cands[0]
    assert a['id'] == 'A' and 'walls' in a['kind']
    assert abs(math.degrees(a['grid'].yaw) - 12) < 1.01
    c, s = math.cos(ANGLE), math.sin(ANGLE)
    expect = (c * OFFSET[0] - s * OFFSET[1], s * OFFSET[0] + c * OFFSET[1])
    assert math.dist((a['grid'].origin_x, a['grid'].origin_y), expect) < 0.08
    assert a['grid'].tile_of(*robot[:2]) == (0, 0)
    assert a['sides'] == {'N': 'wall', 'E': 'wall', 'S': 'open', 'W': 'open'}
    assert all(cd['grid'].tile_of(*robot[:2]) == (0, 0) for cd in cands)     # robot inside every option
    assert len({cd['id'] for cd in cands}) == len(cands) >= 2


def test_classification_and_trail_status():
    g, robot = maze()
    tg = T.initial_candidates(g, INFO, robot)[0]['grid']
    tiles = T.classify_tiles(g, INFO, tg, [(0, 0), (0, -1)])
    assert tiles[(0, 0)]['status'] == 'visited' and tiles[(0, -1)]['status'] == 'visited'
    assert tiles[(1, 0)]['status'] == 'to_visit'
    counts = T.tile_counts(tiles)
    assert counts['visited'] == 2 and counts['to_visit'] > 5
    far = T.classify_tiles(g, INFO, tg, [])
    assert all(t['status'] != 'visited' for t in far.values())


def _classified():
    g, robot = maze()
    tg = T.initial_candidates(g, INFO, robot)[0]['grid']
    tiles = T.classify_tiles(g, INFO, tg, [(0, 0)])
    for t, d in tiles.items():
        if d['status'] != 'unknown':
            d['sides'] = T.tile_sides(g, INFO, tg, t)
    return g, robot, tg, tiles


def test_connected_steps_respect_walls():
    g, robot, tg, tiles = _classified()
    reach = T.connected_steps(tiles, (0, 0))
    assert reach[(0, -1)][0] == 1 and reach[(-1, 0)][0] == 1
    assert (1, 0) not in reach or reach[(1, 0)][0] > 1          # wall on the east side of the start tile
    assert reach[(-2, 0)][1] == [(0, 0), (0, -1), (-1, -1), (-2, -1), (-2, 0)]   # around the wall, not through


def test_nearest_connected_targets():
    g, robot, tg, tiles = _classified()
    targets = T.nearest_connected_targets(g, INFO, tg, tiles, (0, 0), robot[:2], limit=6)
    assert [t['id'] for t in targets] == list(range(1, len(targets) + 1))
    assert [t['steps'] for t in targets] == sorted(t['steps'] for t in targets)
    assert targets[0]['steps'] == 1 and all(t['clearance_m'] >= 0.30 for t in targets)
    for t in targets:                                   # goal point lies inside its tile
        assert T.tile_label(tg.tile_of(*t['goal_xy'])) == t['tile']
    assert '0,0' not in [t['tile'] for t in targets]    # visited tiles are not targets


def test_adjacency_levels():
    tiles = {(0, 0): {'sides': {'E': 'open'}}, (1, 0): {'sides': {'W': 'open', 'N': 'wall'}},
             (1, 1): {}, (2, 0): {'sides': {}}, (5, 5): {}}
    visited = [(0, 0), (1, 0)]
    assert [T.adjacency_level(tiles, t, visited, (1, 0)) for t in [(2, 0), (0, 1), (1, 1), (5, 5)]] == \
        [T.ADJ_CURRENT, T.ADJ_VISITED, T.ADJ_WALL, T.ADJ_NONE]


def test_scene_tiles_are_scanned_but_less_than_80_percent_traversable():
    info = fc.GridInfo(40, 20, 0.05, 0.0, 0.0)                 # two 1 m tiles side by side
    grid = np.zeros((20, 40), dtype=np.int16)
    grid[0:8, 20:40] = 100                                     # tile 1,0: 40% occupied (an object / scene)
    tg = T.TileGrid(0.0, 0.0, 0.0)
    tiles = T.classify_tiles(grid, info, tg, [(0, 0)])
    assert tiles[(0, 0)]['scene'] is False and tiles[(1, 0)]['scene'] is True
    assert tiles[(1, 0)]['status'] == 'to_visit'               # a flag only: tile selection is unchanged
    part = grid.copy()
    part[:, 30:] = -1                                          # half of tile 1,0 never scanned: not completely scanned
    assert T.classify_tiles(part, info, tg, [(0, 0)], min_known=0.4)[(1, 0)]['scene'] is False



def _tile_row(fractions):
    """One row of 1 m tiles (20 x 20 cells of 5 cm); each (occupied, unscanned[, where]) share fills that tile's cells in
    order: occupied first, then unscanned, the rest free. where = 'edge' (default) fills the 10 cm ring along the tile
    edges first (walls), 'middle' the interior first (a scene's props)."""
    info = fc.GridInfo(20 * len(fractions), 20, 0.05, 0.0, 0.0)
    grid = np.zeros((20, 20 * len(fractions)), dtype=np.int16)
    r, c = np.divmod(np.arange(400), 20)
    ring = (r < 2) | (r >= 18) | (c < 2) | (c >= 18)
    for i, frac in enumerate(fractions):
        occ, unknown = frac[0], frac[1]
        where = frac[2] if len(frac) > 2 else 'edge'
        order = np.argsort(~ring if where == 'edge' else ring, kind='stable')
        cells = np.zeros(400, dtype=np.int16)
        n_occ, n_unk = round(occ * 400), round(unknown * 400)
        cells[order[:n_occ]] = 100
        cells[order[n_occ:n_occ + n_unk]] = -1
        grid[:, 20 * i:20 * (i + 1)] = cells.reshape(20, 20)
    return grid, info


def test_scene_flag_uses_the_scanned_part_so_cluttered_tiles_are_caught_before_entry():
    """Bag 'generational run' (23 Nov): the scene tiles 2,0 and 2,-3 were never fully scanned before the robot drove
    at them (part of a cluttered tile hides behind its own clutter), so the old rule (known >= 0.95 and free/TILE < 0.85)
    flagged 2,0 only after it was entered and 2,-3 only 24 s into the drive. Numbers are /tile_map values at the time
    each tile was chosen as a target; 1,-3 is a real maze tile with the lowest free/known ratio seen before entry. The
    scene props stand in the middle of their tiles; the maze tiles' occupied cells are walls along their edges."""
    grid, info = _tile_row([
        (0.00, 0.00),     # 0,0: the robot's tile
        (0.24, 0.10, 'middle'),   # 1,0 ~ scene 2,0:  known 0.90, free 0.66 (free/known 0.73)
        (0.21, 0.07, 'middle'),   # 2,0 ~ scene 2,-3: known 0.93, free 0.72 (free/known 0.77)
        (0.17, 0.07),     # 3,0 ~ real 1,-3:  known 0.93, free 0.76 (free/known 0.82)
        (0.06, 0.00),     # 4,0 ~ real 0,-1:  known 1.00, free 0.94 (walls on its edges)
        (0.40, 0.50),     # 5,0: half of it never scanned -> not enough to call it a scene yet
    ])
    tg = T.TileGrid(0.0, 0.0, 0.0)
    tiles = T.classify_tiles(grid, info, tg, [(0, 0)])
    assert [tiles[(i, 0)]['scene'] for i in range(6)] == [False, True, True, False, False, False]
    old = T.classify_tiles(grid, info, tg, [(0, 0)], scene_min_known=0.95, scene_max_free=0.85)
    assert not old[(1, 0)]['scene'] and not old[(2, 0)]['scene']   # the old thresholds miss both (absolute share)

def test_edge_distance():
    from lite3_explorer.tiles import TileGrid
    g = TileGrid(0.0, 0.0, 0.0, 1.0)
    assert abs(g.edge_distance(0.5, 0.5) - 0.5) < 1e-9
    assert abs(g.edge_distance(0.1, 0.5) - 0.1) < 1e-9
    assert abs(g.edge_distance(1.95, 0.5) - 0.05) < 1e-9


def test_virtual_wall_survives_a_thinned_map_but_not_a_cleared_one():
    from lite3_explorer.tile_tracker import TileTracker
    info = fc.GridInfo(40, 20, 0.05, 0.0, 0.0)                 # two 1 m tiles side by side, wall on their shared edge x=1
    grid = np.zeros((20, 40), dtype=np.int16)
    grid[:, 19:21] = 100
    t = TileTracker()
    t.grid = T.TileGrid(0.0, 0.0, 0.0)
    t.start_wall = None

    def walls(g):
        t.set_map(g, info)
        t.update(force=True)
        return [s for s in t.virtual_walls(start_wall=False) if abs(s[0][0] - 1.0) < 1e-6 and abs(s[1][0] - 1.0) < 1e-6]
    assert walls(grid)                                         # fully mapped wall: virtual wall
    thin = grid.copy()
    thin[:11, 19:21] = 0                                       # the map thins it to 45% of the edge (robot close by)
    assert walls(thin)                                         # kept (was: dropped below 50%)
    gone = grid.copy()
    gone[:, 19:21] = 0                                         # really gone
    assert not walls(gone)
    assert not walls(thin)                                     # and 45% alone does not bring it back


def _start_room(robot_xy=(0.0, 0.0), back_wall=True, clutter=True, size=1.02):
    """Start tile centred on the origin (walls N and S, a back wall W, open E) and, two tiles east, a tile full of
    scattered clutter (a scene tile) that outnumbers the wall cells."""
    g = np.full((200, 200), 0, np.int16)
    cols, rows = np.meshgrid(np.arange(200), np.arange(200))
    x = INFO.origin_x + (cols + 0.5) * RES
    y = INFO.origin_y + (rows + 0.5) * RES
    h = size / 2
    g[(np.abs(y - h) < RES / 2 + 1e-9) & (x > -h) & (x < 3 * size)] = 100            # north wall
    g[(np.abs(y + h) < RES / 2 + 1e-9) & (x > -h) & (x < 3 * size)] = 100            # south wall
    if back_wall:
        g[(np.abs(x + h) < RES / 2 + 1e-9) & (np.abs(y) < h)] = 100                   # back (west) wall
    if clutter:
        rng = np.random.default_rng(1)
        box = (x > 1.5 * size + 0.15) & (x < 2.5 * size - 0.15) & (np.abs(y) < h - 0.15)
        g[box & (rng.random(g.shape) < 0.6)] = 100
    return g, (robot_xy[0], robot_xy[1], 0.0)


def test_start_tile_sits_on_the_back_wall_and_ignores_clutter():
    g, robot = _start_room(robot_xy=(0.08, 0.05))          # robot a bit off the tile centre
    a = T.initial_candidates(g, INFO, robot, size=1.02)[0]
    assert '(back)' in a['kind'] and a['back'] == '-u'
    x0, y0 = a['grid'].corners((0, 0))[0]
    assert abs(x0 - (-0.51)) < 0.04                         # back edge on the back wall, not the robot centred
    assert abs(y0 - (-0.51)) < 0.04                         # between the side walls
    assert a['grid'].tile_of(*robot[:2]) == (0, 0)


def test_grid_keeps_the_wall_angle_when_the_heading_is_off_axis():
    # bag stairs69_25_11_105440: walls at ~22 deg, robot facing -58.7 deg; the grid was placed at 31.3 deg
    # (heading - 90 deg + 180), 9 deg off the walls
    angle, heading = math.radians(22), math.radians(-58.7)
    g = np.full((200, 200), 0, np.int16)
    cols, rows = np.meshgrid(np.arange(200), np.arange(200))
    x = INFO.origin_x + (cols + 0.5) * RES
    y = INFO.origin_y + (rows + 0.5) * RES
    c, s = math.cos(angle), math.sin(angle)
    u, v = c * x + s * y, -s * x + c * y
    for k in (-2, -1, 1, 2):
        g[(np.abs(u - (k - 0.5 * np.sign(k))) < RES / 2 + 1e-9) & (np.abs(v) < 2.5)] = 100
        g[(np.abs(v - (k - 0.5 * np.sign(k))) < RES / 2 + 1e-9) & (np.abs(u) < 2.5)] = 100
    tg = T.initial_candidates(g, INFO, (0.0, 0.0, heading))[0]['grid']
    assert abs(math.degrees(tg.yaw) - 22) < 1.01             # the wall angle, not the heading


def test_start_tile_without_walls_nearby_is_centred_on_the_robot():
    g, robot = _start_room(robot_xy=(0.0, 0.0), back_wall=False)
    a = T.initial_candidates(g, INFO, robot, size=1.02)[0]
    cx, cy = a['grid'].center((0, 0))
    assert abs(cx) < 0.04 and abs(cy) < 0.04               # no back wall: centred along the heading


from lite3_explorer.tile_tracker import TileTracker  # noqa: E402


def test_tile_elevation_comes_from_the_terrain_answer():
    t = TileTracker()
    assert t.elevation('1,0') is None
    t.set_terrain('1,0', {'kind': 'flat'})
    assert t.elevation('1,0') is None                      # classifier elevation option off
    t.set_terrain('1,0', {'kind': 'flat', 'elevation_m': 0.12, 'elevation_class': 'higher', 'slope_deg': 1.0, 'n_cells': 9})
    assert t.elevation('1,0') == {'elevation_m': 0.12, 'elevation_class': 'higher', 'slope_deg': 1.0}


def test_lower_elevation_does_not_replace_a_higher_one_within_a_look():
    from lite3_explorer.tile_tracker import TileTracker
    t = TileTracker(1.0, 0.3, 2)
    t.set_terrain('0,-1', {'kind': 'flat', 'elevation_m': 0.30, 'elevation_class': 'higher'})
    t.set_terrain('0,-1', {'kind': 'flat', 'elevation_m': 0.002, 'elevation_class': 'level'})   # the approach: own floor in the patch
    assert t.elevation('0,-1')['elevation_class'] == 'higher' and t.elevation('0,-1')['elevation_m'] == 0.30
    t.begin_terrain_look('0,-1')                                                                  # a new look measures again
    t.set_terrain('0,-1', {'kind': 'flat', 'elevation_m': 0.002, 'elevation_class': 'level'})
    assert t.elevation('0,-1')['elevation_class'] == 'level'


def test_scene_flag_ignores_walls_on_the_edges_of_a_slightly_off_grid():
    """Bag curtaintest_25_11_132200: no start-tile wall was found, so the grid was centred on the robot, ~8 cm off the
    walls. The empty dead-end 2,0 (walls N, E, S) then had its walls inside the tile: free/known 0.85 < 0.86 over the
    whole tile -> flagged a scene, and the AI took 35 s to call it traversable. Inside a 0.10 m margin it is ~all free.
    A tile with a prop standing in its middle is still a scene."""
    info = fc.GridInfo(40, 20, 0.05, 0.0, 0.0)
    grid = np.zeros((20, 40), dtype=np.int16)
    # tile 0,0: dead-end, walls 2 cells thick lying up to 10 cm inside its N, E and S edges (bag: 2.90-2.95 m for an
    # edge at 3 m)
    grid[0:2, 0:20] = 100
    grid[18:20, 0:20] = 100
    grid[:, 18:20] = 100
    # tile 1,0: a 0.45 m x 0.45 m prop in the middle
    grid[6:15, 26:35] = 100
    tg = T.TileGrid(0.0, 0.0, 0.0)
    whole = T.classify_tiles(grid, info, tg, [], scene_max_free=0.86, scene_margin_m=0.0)
    assert whole[(0, 0)]['scene']                                  # the old whole-tile test flags the empty dead-end
    tiles = T.classify_tiles(grid, info, tg, [], scene_max_free=0.86, scene_margin_m=0.10)
    assert not tiles[(0, 0)]['scene'] and tiles[(0, 0)]['scene_free'] > 0.95
    assert tiles[(1, 0)]['scene']


def lattice_room():
    """3 x 3 cells of 1 m, walls on every lattice line, turned by ANGLE; the middle cell is closed all round."""
    g = np.full((200, 200), -1, np.int16)
    cols, rows = np.meshgrid(np.arange(200), np.arange(200))
    x, y = INFO.origin_x + (cols + 0.5) * RES, INFO.origin_y + (rows + 0.5) * RES
    c, s = math.cos(ANGLE), math.sin(ANGLE)
    u, v = c * x + s * y, -s * x + c * y
    inside = (u > 0) & (u < 3) & (v > 0) & (v < 3)
    g[inside] = 0
    near = lambda w: abs(w - np.round(w)) < 0.04
    g[inside & (near(u) | near(v))] = 100
    return g, (c * 1.5 - s * 1.5, s * 1.5 + c * 1.5)


def test_align_to_walls_pulls_the_tile_edges_onto_the_walls():
    g, (rx, ry) = lattice_room()
    true = T.TileGrid(0.0, 0.0, ANGLE)                    # the tile (1, 1) is the middle cell
    off = T.TileGrid(0.07, -0.06, ANGLE + math.radians(3))
    assert off.tile_of(rx, ry) == (1, 1)
    res = T.align_to_walls(g, INFO, off, (1, 1), (rx, ry))
    assert res is not None
    new = res['grid']
    assert new.tile_of(rx, ry) == (1, 1)
    assert abs(T.wrap(new.yaw - ANGLE)) < math.radians(0.8)
    for a, b in zip(new.corners((1, 1)), true.corners((1, 1))):
        assert math.dist(a, b) < 0.05, (a, b)
    # already aligned: nothing to do; far off (> max shift / rot): left alone
    assert T.align_to_walls(g, INFO, true, (1, 1), (rx, ry)) is None
    far = T.TileGrid(0.4, 0.0, ANGLE + math.radians(20))
    res = T.align_to_walls(g, INFO, far, far.tile_of(rx, ry), (rx, ry))
    assert res is None or abs(res['rot_deg']) < 1e-6


def test_tracker_aligns_the_grid_once_per_tile_entry():
    from lite3_explorer.tile_tracker import TileTracker
    g, (rx, ry) = lattice_room()
    tr = TileTracker()
    tr.set_map(g, INFO)
    tr.grid = T.TileGrid(0.07, -0.06, ANGLE + math.radians(3))
    tr.start_wall, tr.start_wall_side, tr.start_tile = None, 'S', (0, 0)
    seen = []
    tr.on_aligned = seen.append
    pose = (rx, ry, 0.0)
    tr.track(pose)                                        # enters (1, 1) at its centre-ish
    assert len(seen) == 1 and tr.current == (1, 1)
    tr.track(pose)                                        # same tile: no second run
    assert len(seen) == 1
    assert abs(T.wrap(tr.grid.yaw - ANGLE)) < math.radians(0.8)


def test_no_tile_is_reached_while_a_slam_jump_holds_reaching():
    """fullrun_25_11_215528: a false loop closure moved map->odom 2.1 m and the jumped pose 'reached' tile 1,-2 on the
    far side of the stairs. The mission manager sets reach_hold after such a jump."""
    tr = TileTracker()
    tr.grid = T.TileGrid(0.0, 0.0, 0.0)
    centre = tr.grid.center((1, 0))
    tr.reach_hold = True
    assert tr.track((centre[0], centre[1], 0.0)) is None
    assert tr.mark_goal_reached((1, 0), centre, (centre[0], centre[1], 0.0)) is None
    assert (1, 0) not in tr.trail
    tr.reach_hold = False
    assert tr.track((centre[0], centre[1], 0.0)) == (1, 0)


def _tracker_with(tiles, trail):
    tr = TileTracker()
    tr.tiles = {t: dict(d) for t, d in tiles.items()}
    tr.trail = list(trail)
    return tr


def _tile(status, **walls):
    return {'status': status, 'sides': {s: walls.get(s, 'open') for s in 'NESW'}}


def test_new_tile_beside_an_earlier_tile_makes_it_a_split():
    # robot went 0,0 -> 1,0 -> 2,0; then the map shows a new tile 1,1 north of 1,0 (visited earlier, not the current)
    tiles = {(0, 0): _tile('visited'), (1, 0): _tile('visited'), (2, 0): _tile('visited')}
    tr = _tracker_with(tiles, [(0, 0), (1, 0), (2, 0)])
    assert tr.note_discovered_splits() == []                       # nothing new yet
    tr.tiles[(1, 1)] = _tile('to_visit')
    got = tr.note_discovered_splits()
    assert got == [{'tile': '1,0', 'options': ['1,1'], 'new': ['1,1']}]
    assert tr.last_split()['tile'] == (1, 0)
    assert tr.note_discovered_splits() == []                       # only once per new tile


def test_new_tile_behind_a_wall_or_beside_the_current_tile_is_no_split():
    tiles = {(0, 0): _tile('visited'), (1, 0): _tile('visited', N='wall'), (2, 0): _tile('visited')}
    tr = _tracker_with(tiles, [(0, 0), (1, 0), (2, 0)])
    tr.note_discovered_splits()
    tr.tiles[(1, 1)] = _tile('to_visit')                            # walled off from 1,0
    tr.tiles[(2, 1)] = _tile('to_visit')                            # beside the current tile: exploring handles it
    assert tr.note_discovered_splits() == []
    assert tr.last_split() is None


def test_most_recently_visited_tile_is_the_last_split():
    tiles = {(0, 0): _tile('visited'), (1, 0): _tile('visited'), (2, 0): _tile('visited'), (3, 0): _tile('visited')}
    tr = _tracker_with(tiles, [(0, 0), (1, 0), (2, 0), (3, 0)])
    tr.initial_tile = None
    tr.note_discovered_splits()
    tr.tiles[(0, -1)] = _tile('to_visit')
    tr.tiles[(2, 1)] = _tile('to_visit')
    assert [s['tile'] for s in tr.note_discovered_splits()] == ['0,0', '2,0']
    assert tr.last_split()['tile'] == (2, 0)


def test_virtual_wall_a_few_cm_off_the_grid_line():
    """bag fullrun_25_11_225456, tile 0,-1: the wall stood ~10 cm inside the tile (grid not aligned there) and only
    25% of the edge read as wall on the ±1 cell line: no virtual wall. The band across the edge finds it."""
    from lite3_explorer.tile_tracker import TileTracker
    info = fc.GridInfo(40, 20, 0.05, 0.0, 0.0)
    grid = np.zeros((20, 40), dtype=np.int16)
    grid[:, 22] = 100                                          # wall 10-15 cm east of the shared edge x=1
    t = TileTracker()
    t.grid = T.TileGrid(0.0, 0.0, 0.0)
    t.start_wall = None
    t.set_map(grid, info)
    t.update(force=True)

    def on_edge():
        return [s for s in t.virtual_walls(start_wall=False) if abs(s[0][0] - 1.0) < 1e-6 and abs(s[1][0] - 1.0) < 1e-6]
    t.wall_band_m = 0.0
    assert not on_edge()
    t._wall_keys = set()
    t.wall_band_m = 0.15
    assert on_edge()


def test_tile_with_four_virtual_walls_is_blocked():
    from lite3_explorer.tile_tracker import TileTracker
    info = fc.GridInfo(60, 20, 0.05, 0.0, 0.0)                 # tiles 0,0 1,0 2,0 in a row
    grid = np.zeros((20, 60), dtype=np.int16)
    grid[0, :] = grid[-1, :] = 100                             # long walls S and N
    grid[:, 20] = grid[:, 40] = 100                            # walls on both edges of the middle tile
    t = TileTracker()
    t.grid = T.TileGrid(0.0, 0.0, 0.0)
    t.start_wall = None
    t.set_map(grid, info)
    t._visit((0, 0), (0.5, 0.5, 0.0), 'start')
    t.update(force=True)
    assert t.tiles[(1, 0)]['status'] == 'to_visit'
    t.virtual_walls(start_wall=False)
    t.update()
    assert (1, 0) in t.enclosed
    assert t.tiles[(1, 0)]['status'] == 'blocked' and t.tiles[(1, 0)]['enclosed']
    assert '1,0' not in t.unexplored()
    assert t.tiles[(0, 0)]['status'] == 'visited'              # the robot's own tile is never enclosed
    open_ = grid.copy()
    open_[:, 40] = 0                                           # the east wall was a mapping artefact: open again
    t.set_map(open_, info)
    t.virtual_walls(start_wall=False)
    t.update()
    assert (1, 0) not in t.enclosed and t.tiles[(1, 0)]['status'] == 'to_visit'
