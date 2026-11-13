import math

import numpy as np

from lite3_kitchen_explorer import frontier_core as fc

INFO = fc.GridInfo(width=40, height=30, resolution=0.1, origin_x=-1.0, origin_y=-2.0)


def room_grid():
    """Known free room on the left, unknown on the right, wall along the top/bottom."""
    g = np.full((INFO.height, INFO.width), -1, dtype=np.int16)
    g[:, :20] = 0
    g[0, :20] = 100
    g[-1, :20] = 100
    return g


def test_cell_world_roundtrip():
    x, y = INFO.cell_to_world(5, 7)
    assert INFO.world_to_cell(x, y) == (5, 7)


def test_detects_single_frontier_column():
    f = fc.detect_frontier_cells(room_grid())
    rows, cols = np.nonzero(f)
    assert set(cols) == {19}
    assert set(rows) == set(range(1, INFO.height - 1))  # walls are not frontier


def test_cluster_and_goal_has_clearance():
    clusters = fc.build_clusters(room_grid(), INFO, min_cluster_cells=3, clearance_m=0.3)
    assert len(clusters) == 1
    c = clusters[0]
    assert len(c.cells) == INFO.height - 2
    gr, gc = INFO.world_to_cell(*c.goal_xy)
    assert gc == 19 and 3 <= gr <= INFO.height - 4
    assert c.extra['has_clearance']


def test_small_clusters_dropped():
    g = np.zeros((10, 10), dtype=np.int16)
    g[5, 5] = -1  # a single unknown pixel -> 4 frontier cells, noise
    info = fc.GridInfo(10, 10, 0.05, 0.0, 0.0)
    assert fc.build_clusters(g, info, min_cluster_cells=8) == []


def test_selection_prefers_cost_and_respects_blacklist():
    g = np.full((30, 60), 0, dtype=np.int16)
    g[:, 55:] = -1          # big frontier far away (x ~ 5.5 m)
    g[0:4, 0:4] = -1        # small frontier near origin
    info = fc.GridInfo(60, 30, 0.1, 0.0, 0.0)
    clusters = fc.build_clusters(g, info, min_cluster_cells=3, clearance_m=0.0)
    assert len(clusters) == 2
    robot = (1.0, 1.0)
    near_only = fc.select_frontier(clusters, robot, size_weight=0.0, min_goal_distance_m=0.0)
    assert near_only.distance_m < 2.0
    big_wins = fc.select_frontier(clusters, robot, size_weight=2.0, min_goal_distance_m=0.0)
    assert big_wins.goal_xy[0] > 5.0
    blocked = fc.select_frontier(clusters, robot, size_weight=2.0, min_goal_distance_m=0.0,
                                 blacklist=[big_wins.goal_xy], blacklist_radius_m=0.5)
    assert blocked is not None and blocked.goal_xy[0] < 2.0


def test_min_goal_distance():
    clusters = fc.build_clusters(room_grid(), INFO, min_cluster_cells=3)
    goal = clusters[0].goal_xy
    assert fc.select_frontier(clusters, goal, min_goal_distance_m=0.5) is None


def test_frontier_near():
    g = room_grid()
    x, y = INFO.cell_to_world(15, 19)
    assert fc.frontier_near(g, INFO, (x, y), 0.3)
    g[:, 20:] = 0  # area mapped
    assert not fc.frontier_near(g, INFO, (x, y), 0.3)
    assert not fc.frontier_near(g, INFO, (100.0, 100.0), 0.3)


def test_dilate_disc():
    m = np.zeros((9, 9), dtype=bool)
    m[4, 4] = True
    d = fc.dilate(m, 2)
    assert d[4, 6] and d[2, 4] and not d[2, 2] and d.sum() == 13


def test_msg_data_layout():
    data = list(range(6))
    g = fc.grid_from_msg_data(data, width=3, height=2)
    assert g[1, 0] == 3 and math.isclose(g.mean(), 2.5)


def test_wall_clearance_and_candidates():
    g = np.zeros((40, 40), dtype=np.int16)
    g[:, 0] = 100                       # wall along x = 0
    info = fc.GridInfo(40, 40, 0.1, 0.0, 0.0)
    assert abs(fc.wall_clearance_m(g, info, (1.05, 2.0)) - 1.0) < 1e-6
    assert fc.wall_clearance_m(g, info, (3.95, 2.0), max_m=1.0) == 1.0
    g[30:, 10:] = -1                    # unknown corner -> one frontier
    g[0:5, 35:] = -1                    # second, smaller frontier
    clusters = fc.build_clusters(g, info, min_cluster_cells=3, clearance_m=0.0)
    fc.select_frontier(clusters, (1.0, 1.0), min_goal_distance_m=0.0)
    cands = fc.rank_candidates(clusters, limit=8)
    assert [c['id'] for c in cands] == [1, 2]
    assert cands[0]['cost'] <= cands[1]['cost']
    assert all(c['clearance_m'] > 0.5 for c in cands)
