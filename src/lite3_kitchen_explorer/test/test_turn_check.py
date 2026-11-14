import glob
import os
import sqlite3

import numpy as np
import pytest

from lite3_kitchen_explorer import turn_check as TC

RES = 0.05


def room(half_width_m, length_m=4.0):
    """A corridor along y, walls (lethal = 100, with a 2-cell inscribed halo = 99) at x = +-half_width_m around x = 0."""
    n = int(length_m / RES)
    g = np.zeros((n, n), np.int16)
    origin = (-length_m / 2, -length_m / 2)
    for col in range(n):
        x = origin[0] + (col + 0.5) * RES
        if abs(x) >= half_width_m:
            g[:, col] = 100 if abs(x) < half_width_m + 3 * RES else 0
            if abs(x) >= half_width_m + 3 * RES:
                g[:, col] = 0
    for col in range(n):                      # inscribed halo on the free side of each wall
        x = origin[0] + (col + 0.5) * RES
        if half_width_m - 2 * RES <= abs(x) < half_width_m:
            g[:, col] = 99
    return g, origin


def test_open_space_allows_a_full_turn():
    g = np.zeros((80, 80), np.int16)
    ok, blocked = TC.can_turn_in_place(g, (-2.0, -2.0), RES, 0.0, 0.0)
    assert ok and blocked == 0


def test_a_wide_corridor_allows_a_turn_and_a_narrow_one_does_not():
    wide, origin = room(0.55)                 # 1.1 m between the walls, 1.0 m with the halo
    assert TC.can_turn_in_place(wide, origin, RES, 0.0, 0.0)[0]
    narrow, origin = room(0.40)               # the dog's rotation radius is 0.357 m, plus the halo
    ok, blocked = TC.can_turn_in_place(narrow, origin, RES, 0.0, 0.0)
    assert not ok and blocked > 40


def test_only_the_corridor_axis_orientations_are_free_in_a_narrow_corridor():
    narrow, origin = room(0.42)
    blocked = set(TC.blocked_orientations(narrow, origin, RES, 0.0, 0.0))
    assert 90 not in blocked and 270 not in blocked        # aligned with the corridor (along y)
    assert 0 in blocked and 180 in blocked                 # across it


def test_off_centre_is_worse_than_centred_and_margin_makes_the_check_stricter():
    g, origin = room(0.50)
    assert TC.can_turn_in_place(g, origin, RES, 0.0, 0.0)[0]
    assert not TC.can_turn_in_place(g, origin, RES, 0.12, 0.0)[0]
    assert not TC.can_turn_in_place(g, origin, RES, 0.0, 0.0, margin=0.10)[0]


def test_lethal_only_threshold_is_more_permissive_than_inscribed():
    g, origin = room(0.42)
    strict = len(TC.blocked_orientations(g, origin, RES, 0.0, 0.0, block_value=99))
    lethal_only = len(TC.blocked_orientations(g, origin, RES, 0.0, 0.0, block_value=100))
    assert lethal_only < strict


def test_the_cells_outside_the_grid_count_as_free():
    g = np.zeros((10, 10), np.int16)
    assert TC.can_turn_in_place(g, (-0.25, -0.25), RES, 0.0, 0.0)[0]


BAG = os.path.expanduser('~/2d/bags/route_debug_080014')


@pytest.mark.skipif(not glob.glob(BAG + '/*.db3'), reason='the recorded bag is not on this machine')
def test_the_recorded_corridor_at_the_start_tile_cannot_be_turned_in():
    """route_debug_080014: Nav2's Spin aborted here twice ("Collision Ahead - Exiting Spin")."""
    pytest.importorskip('rclpy')
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    c = sqlite3.connect(glob.glob(BAG + '/*.db3')[0])
    t0 = c.execute('select min(timestamp) from messages').fetchone()[0]
    topics = {n: (i, t) for i, n, t in c.execute('select id,name,type from topics')}

    def at(name, t):
        i, ty = topics[name]
        row = c.execute('select data from messages where topic_id=? and timestamp<=? order by timestamp desc limit 1',
                        (i, t0 + int(t * 1e9))).fetchone()
        return deserialize_message(row[0], get_message(ty))
    lc, odom = at('/local_costmap/costmap', 41.5), at('/odom', 41.5).pose.pose.position
    grid = np.array(lc.data, np.int16).reshape(lc.info.height, lc.info.width)
    ok, blocked = TC.can_turn_in_place(grid, (lc.info.origin.position.x, lc.info.origin.position.y), lc.info.resolution,
                                       odom.x, odom.y)
    assert not ok and blocked > 40
