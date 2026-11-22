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


def test_scene_tile_is_looked_at_from_beside_it_and_never_entered():
    r, res = run('scene_corridor')
    assert res['complete'] and res['complete'].startswith('all_tiles_explored')
    assert res['truth_scenes'] == [(2, 1)]
    kinds = [(k, d['tile']) for _, k, d in res['scene_events']]
    assert kinds[0][0] == 'scene' and ('scene_done', kinds[0][1]) in kinds    # scene mode ran on it once
    assert 'scene' in res['modes']
    assert (2, 1) not in r.truth_visited                                       # looked at, never entered
    assert r.ctx.tiles.tiles[r.ctx.tiles.grid.tile_of(*r.world.maze_to_map(*r.maze.tile_centre((2, 1)))[:2])][
        'status'] == 'visited'                                                 # ... and counted as explored
