import math

import numpy as np

from lite3_kitchen_explorer import frontier_core as fc
from lite3_kitchen_explorer import tiles as T

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
