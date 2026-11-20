import math

from lite3_explorer import approach as A

PATH = [[0.0, 0.0], [0.6, 0.0], [1.2, 0.0], [1.2, 0.6], [1.2, 1.2], [1.8, 1.2]]   # 3.0 m, one right-angle turn


def test_sample_path_is_even_and_ordered():
    s = A.sample_path(PATH, 5)                             # no skip: at 0.5, 1.0, ... 2.5 m
    assert len(s) == 5 and [k for _, _, k in s] == sorted(k for _, _, k in s)
    assert s[0][:2] == (0.5, 0.0) and math.isclose(s[2][0], 1.2) and math.isclose(s[2][1], 0.3)


def test_default_route_yaws_and_final_axis():
    r = A.default_route(PATH, (1.8, 1.2), grid_yaw=math.radians(4))
    assert len(r) == 4 and r[-1][:2] == (1.8, 1.2)          # 3 waypoints + goal
    assert math.isclose(r[0][2], math.atan2(r[1][1] - r[0][1], r[1][0] - r[0][0]))   # faces the next waypoint
    assert 0.0 <= r[0][2] < math.pi / 2                                   # heading east, curving north
    assert math.isclose(r[-1][2], math.radians(4), abs_tol=1e-6)         # last leg is +x -> grid axis at 4 deg
    assert any(math.isclose(y, math.pi / 2) for _, _, y in r)             # turns north on the vertical leg


def test_parse_route_snaps_and_falls_back():
    ai = {'target_yaw_deg': 90, 'waypoints': [{'x': 0.52, 'y': 0.1, 'yaw_deg': 10}, {'x': 9, 'y': 9, 'yaw_deg': 'bad'}]}
    r, note = A.parse_route(ai, PATH, (1.8, 1.2), 0.0)
    d = A.default_route(PATH, (1.8, 1.2), 0.0)
    assert r[0][:2] == (0.6, 0.0) and math.isclose(r[0][2], math.radians(10))   # snapped onto the path, AI yaw
    assert r[1] == d[1]                                                       # too far / bad yaw -> default
    assert math.isclose(r[3][2], math.pi / 2) and 'AI' in note
    r2, note2 = A.parse_route({}, PATH, (1.8, 1.2), 0.0)
    assert r2 == d and note2.startswith('default')
    assert len(A.parse_route({'waypoints': 'x'}, [], (1, 1), 0.0)[0]) == 1    # no path: just the goal


def test_filter_route_drops_bunched_and_backward_waypoints():
    import math
    from lite3_explorer import approach as A
    # the 1,1 -> 0,1 route recorded in route_debug_031713 (yaws in degrees)
    raw = [(1.05, 0.76, -7), (1.06, 0.76, -169), (1.04, 0.76, 170), (0.89, 0.79, 153), (0.71, 0.87, 149), (0.55, 0.97, -175)]
    route = [(x, y, math.radians(d)) for x, y, d in raw]
    kept, dropped = A.filter_route(route, (1.07, 0.79))
    assert [round(x, 2) for x, _, _ in kept] == [0.89, 0.71, 0.55]
    assert len(dropped) == 3


def test_filter_route_drops_backward_yaw_far_from_start():
    import math
    from lite3_explorer import approach as A
    route = [(1.0, 0.0, math.pi), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]   # first waypoint faces away from the goal
    kept, dropped = A.filter_route(route, (0.0, 0.0))
    assert kept == route[1:] and len(dropped) == 1


def test_always_three_waypoints_then_the_tile_centre():
    line = lambda length: [[0.0, 0.0], [length, 0.0]]       # noqa: E731
    for length in (0.3, 1.0, 1.4, 6.0):
        r = A.default_route(line(length), (length, 0.0), 0.0)
        assert len(r) == 4 and r[-1][:2] == (length, 0.0)            # 3 waypoints + the tile centre
        xs = [x for x, _, _ in r]
        assert xs == sorted(xs) and len(set(xs)) == 4                 # in order along the path, all distinct


def test_waypoints_skip_the_start_and_are_evenly_spaced():
    r = A.default_route([[0.0, 0.0], [1.0, 0.0]], (1.0, 0.0), 0.0)    # a 1 m tile hop, 0.2 m skipped
    assert [round(x, 2) for x, _, _ in r] == [0.4, 0.6, 0.8, 1.0]
    r = A.default_route([[0.0, 0.0], [0.4, 0.0]], (0.4, 0.0), 0.0)    # short: skip capped at a quarter
    assert [round(x, 3) for x, _, _ in r] == [0.175, 0.25, 0.325, 0.4]
    assert len(A.default_route([[0.0, 0.0], [1.0, 0.0]], (1.0, 0.0), 0.0, n=5)) == 6   # configurable


def test_waypoints_follow_the_path_round_a_wall():
    # U-shaped path: tile 1,0 seen through a wall from 0,0, reached via the row above
    path = [[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]]
    r = A.default_route(path, (1.5, 0.5), 0.0, skip_start_m=0.0)
    assert [(round(x, 2), round(y, 2)) for x, y, _ in r[:3]] == [(0.5, 1.25), (1.0, 1.5), (1.5, 1.25)]
