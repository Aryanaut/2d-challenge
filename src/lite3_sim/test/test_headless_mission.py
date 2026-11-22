"""lite3_explorer's real mission logic against the simulated world (no ROS)."""
import os

import pytest

pytest.importorskip('lite3_explorer')
from lite3_sim.headless import HeadlessRun  # noqa: E402

MAZES = os.path.join(os.path.dirname(__file__), '..', 'mazes')


def run(name, max_time=900, **over):
    r = HeadlessRun(os.path.join(MAZES, name + '.yaml'), overrides=over)
    return r, r.run(max_time)


def test_corridor_is_explored_completely():
    r, res = run('corridor')
    assert res['complete'] and res['complete'].startswith('all_tiles_explored')
    assert res['truth_tiles_visited'] == res['truth_tiles_reachable'] == 5
    assert r.ctx.tiles.grid.size == pytest.approx(1.02)                    # tiles.size_m taken from the maze


def test_curtain_mode_walks_through_the_curtain():
    r, res = run('curtain_corridor')
    assert [e[2]['side'] for e in res['curtain_events']] == ['E']
    assert {(3, 0), (4, 0)} <= set(r.truth_visited)                        # the tiles behind it


def test_without_curtain_mode_the_far_tiles_stay_unexplored():
    r, res = run('curtain_corridor', **{'modes.exploring.curtains': False})
    assert res['complete'] and not res['curtain_events']
    assert {(3, 0), (4, 0)}.isdisjoint(r.truth_visited)
