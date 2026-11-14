"""The explorer must follow a tile target, even when a frontier is nearby (route_debug_074926)."""
import time
from unittest import mock

import numpy as np
import pytest

pytest.importorskip('rclpy')
pytest.importorskip('nav2_msgs')
from lite3_kitchen_explorer import frontier_core as fc   # noqa: E402
from lite3_kitchen_explorer import frontier_explorer as FE   # noqa: E402


def explorer(route=None, preferred=(2.0, 2.0)):
    n = FE.FrontierExplorer.__new__(FE.FrontierExplorer)
    g = np.full((100, 100), -1, np.int16)
    g[10:90, 10:90] = 0                                         # a big free area (the cell size below makes it 4 m wide)
    n.grid, n.info = g, fc.GridInfo(100, 100, 0.05, 0.0, 0.0)
    n.free_threshold, n.occupied_threshold = 25, 65
    n.preferred_timeout_s, n.preferred_match_radius_m, n.preferred_min_clearance_m = 60.0, 0.6, 0.30
    n.route_match_m, n.blacklist_radius_m, n.blacklist = 0.10, 0.6, []
    n.route, n.preferred = route, (*preferred, time.monotonic())
    n._robot_pose = lambda: (1.0, 1.0, 0.0)
    log = mock.Mock()
    n.get_logger = lambda: log
    n.log = log
    return n


def cluster(x, y, clearance, name):
    return fc.FrontierCluster(cells=[], centroid_xy=(x, y), goal_xy=(x, y), size_m=1.0, distance_m=1.0,
                              extra={'usable': True, 'clearance_m': clearance, 'name': name})


ROUTE = [(1.5, 1.5, 0.8), (2.0, 2.0, 1.0)]
FRONTIER_BESIDE_THE_TILE = cluster(2.3, 2.2, 0.10, 'frontier hugging a wall, 0.36 m from the tile goal')
FRONTIER_ELSEWHERE = cluster(4.0, 4.0, 0.9, 'best cost-ranked frontier elsewhere')


def test_tile_target_beside_a_wall_hugging_frontier_is_followed_with_its_route():
    n = explorer(route=list(ROUTE))
    chosen = n._apply_preference([FRONTIER_BESIDE_THE_TILE, FRONTIER_ELSEWHERE], FRONTIER_ELSEWHERE)
    assert chosen.goal_xy == (2.0, 2.0) and chosen.extra['direct'] and chosen.extra['route'] == ROUTE
    assert 'tile' in chosen.extra['source'].lower()
    assert not n.log.warn.called                                 # no "using cost ranking"


def test_without_a_route_the_old_frontier_matching_still_applies():
    n = explorer(route=None)                                     # the AI chose a frontier: no route is sent for those
    chosen = n._apply_preference([cluster(2.1, 2.0, 0.9, 'x')], FRONTIER_ELSEWHERE)
    assert chosen.extra['source'] == 'FRONTIER (AI-chosen)' and chosen.goal_xy == (2.1, 2.0)
    n = explorer(route=None)
    chosen = n._apply_preference([FRONTIER_BESIDE_THE_TILE], FRONTIER_ELSEWHERE)
    assert chosen is FRONTIER_ELSEWHERE                          # a frontier too close to a wall is still refused


def test_a_route_for_a_different_point_is_not_used():
    n = explorer(route=[(0.5, 0.5, 0.0), (3.0, 3.0, 0.0)])       # stale route: ends elsewhere
    chosen = n._apply_preference([FRONTIER_BESIDE_THE_TILE], FRONTIER_ELSEWHERE)
    assert chosen is FRONTIER_ELSEWHERE                          # falls back exactly as before


def test_unusable_tile_goals_still_fall_back_to_the_explorer():
    n = explorer(route=list(ROUTE), preferred=(0.2, 0.2))        # outside the free area (unknown cells)
    n.route = [(0.5, 0.5, 0.0), (0.2, 0.2, 0.0)]
    chosen = n._apply_preference([], FRONTIER_ELSEWHERE)
    assert chosen is FRONTIER_ELSEWHERE and n.log.warn.called
    n = explorer(route=list(ROUTE))
    n.blacklist = [(2.0, 2.0)]                                   # this tile goal failed before
    assert n._apply_preference([], FRONTIER_ELSEWHERE) is FRONTIER_ELSEWHERE


def test_the_route_is_consumed_so_it_is_never_reused():
    n = explorer(route=list(ROUTE))
    n._apply_preference([], FRONTIER_ELSEWHERE)
    assert n.route is None


# ------------------------------------------------------------------ a failed tile goal is handed back, not replaced by a frontier
def navigating(direct, hold_after_reach=True):
    n = explorer()
    n.state, n.goal_xy, n.goal_handle, n.goal_direct = FE.FrontierExplorer.NAVIGATING, (2.0, 2.0), None, direct
    n.hold_after_reach, n.hold_since = hold_after_reach, 0.0
    n.goal_failed_pub, n.goal_info_pub, n.status_pub = mock.Mock(), mock.Mock(), mock.Mock()
    return n


def test_a_failed_tile_goal_is_reported_and_the_explorer_holds_for_the_reporter():
    n = navigating(direct=True)
    n._fail_goal('made no progress (stuck)')
    import json
    sent = json.loads(n.goal_failed_pub.publish.call_args[0][0].data)
    assert sent == {'goal': [2.0, 2.0], 'reason': 'made no progress (stuck)'}
    assert n.state == FE.FrontierExplorer.HOLD and n.blacklist == [(2.0, 2.0)] and n.goal_xy is None


def test_a_failed_frontier_goal_still_replans_by_itself():
    n = navigating(direct=False)
    n._fail_goal('timed out')
    assert n.state == FE.FrontierExplorer.IDLE and not n.goal_failed_pub.publish.called


def test_without_hold_after_reach_a_failed_tile_goal_replans_as_before():
    n = navigating(direct=True, hold_after_reach=False)
    n._fail_goal()
    assert n.state == FE.FrontierExplorer.IDLE and not n.goal_failed_pub.publish.called
