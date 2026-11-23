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
