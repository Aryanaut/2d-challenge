"""Staircases ('M') in the simulator: maze parsing, risers (lidar / AI mode), the stair camera, and the mission's stairs
mode climbing over one (headless, no ROS)."""
import math
import os

import numpy as np
import pytest

from lite3_sim.maze import STAIRS, MazeError, load_maze, load_maze_dict
from lite3_sim.maze_preview import render
from lite3_sim.world import AI_OFF, AI_ON, World

MAZES = os.path.join(os.path.dirname(__file__), '..', 'mazes')
CORRIDOR = {'layout': '+---+---+---+---+\n| >       M     |\n+---+---+---+---+\n'}


# ------------------------------------------------------------------------------------------ maze
def test_m_is_a_staircase_climbed_from_its_open_sides():
    m = load_maze_dict(CORRIDOR)
    assert m.stairs == {(2, 0): ['E', 'W']}
    risers = [s for s in m.segments if s.kind == STAIRS]
    assert sorted((s.x0, s.x1) for s in risers) == pytest.approx([(2.04, 2.04), (3.06, 3.06)])
    assert m.edge((2, 0), 'E') is None                           # open for the robot's path; the riser is a segment
    assert (2, 0) not in m.reachable_tiles() and (3, 0) in m.reachable_tiles()
    assert (3, 0) not in m.reachable_tiles(through_stairs=False)
    assert 'M' in render(m)


def test_steps_go_up_from_each_open_side_to_a_landing():
    m = load_maze_dict(CORRIDOR)
    s = m.tile_size
    h = [m.stair_height(2 * s + u, 0.5) for u in (0.05, 0.25, 0.51, 0.8, 0.97)]
    assert h == pytest.approx([0.13, 0.26, 0.26, 0.26, 0.13])
    assert m.stair_height(1.5, 0.5) == 0.0


def test_staircase_errors():
    with pytest.raises(MazeError):
        load_maze_dict({'layout': '+---+---+\n| > | M |\n+---+---+\n'})      # walled in: no side to climb from
    with pytest.raises(MazeError):
        load_maze_dict({'layout': '+---+---+\n| M     |\n+---+---+\n', 'start': {'tile': [0, 0]}})


# ------------------------------------------------------------------------------------------ world
def test_lidar_sees_the_riser_as_a_wall():
    w = World(load_maze_dict(CORRIDOR), {'lidar': {'noise_m': 0.0}})
    w.step(0.05)
    rel, rng, _ = w.last_scan
    assert rng[np.argmin(np.abs(rel))] == pytest.approx(2.04 - 0.51, abs=0.02)


def drive_east(w, seconds):
    for _ in range(int(seconds / 0.05)):
        w.set_cmd_vel(0.3, 0.0, 0.0)
        w.step(0.05)


def test_risers_stop_the_robot_unless_it_is_in_ai_mode():
    w = World(load_maze_dict(CORRIDOR), {'map_origin': 'maze', 'stair_camera': {'enabled': False}})
    drive_east(w, 12.0)
    assert w.bumped and w.pose[0] < 2.04 - 0.3                       # stopped at the first riser
    w.simple_cmd(0x21010307)                                         # a gait code: not AI mode
    assert not w.ai_mode
    w.simple_cmd(AI_ON)
    drive_east(w, 6.0)
    assert w.ai_mode and w.pose[0] > 3.06 + 0.3                       # over the staircase
    w.simple_cmd(AI_OFF)
    assert not w.ai_mode


def test_stair_camera_finds_the_staircase_tile_ahead():
    w = World(load_maze_dict(CORRIDOR), {'map_origin': 'maze'})
    for _ in range(int(4.0 / 0.05)):
        w.step(0.05)
    pts = w.stair_points()
    assert len(pts) > 10                                              # confirmed after a few frames
    assert np.all((pts[:, 0] > 2.0) & (pts[:, 0] < 3.1))              # on the staircase tile
    assert np.median(pts[:, 2]) > 0.9                                 # going up towards +x


def test_stair_camera_pauses_in_ai_mode_and_sees_nothing_without_stairs():
    w = World(load_maze_dict(CORRIDOR), {'map_origin': 'maze'})
    w.simple_cmd(AI_ON)
    for _ in range(int(4.0 / 0.05)):
        w.step(0.05)
    assert len(w.stair_points()) == 0
    plain = World(load_maze_dict({'layout': '+---+---+---+\n| >         |\n+---+---+---+\n'}), {'map_origin': 'maze'})
    assert plain.stairs is None and len(plain.stair_points()) == 0


# ------------------------------------------------------------------------------------------ the mission
pytest.importorskip('lite3_explorer')
from lite3_sim.headless import HeadlessRun  # noqa: E402


def run(name='stairs_loop', max_time=900, ctx_patch=None, **over):
    r = HeadlessRun(os.path.join(MAZES, name + '.yaml'), overrides=over)
    if ctx_patch:
        ctx_patch(r)
    return r, r.run(max_time)


def test_stairs_mode_climbs_over_the_staircase_in_ai_mode_then_explores_the_rest():
    r, res = run()
    assert res['complete'] and res['complete'].startswith('all_tiles_explored')
    assert res['truth_tiles_visited'] == res['truth_tiles_reachable'] == 9
    assert res['modes'].count('stairs') == 1
    assert [(on, code) for _, on, code in res['ai_commands']] == [(True, '0x21010528'), (False, '0x2101052B')]
    kinds = [k for _, k, _ in res['stairs_events']]
    assert kinds == ['stairs', 'stairs_crossed']
    assert res['staircase_tiles'] == ['2,0'] or len(res['staircase_tiles']) == 1
    assert not r.world.ai_mode
    stair = r.ctx.tiles.grid.tile_of(*r.world.maze_to_map(*r.maze.tile_centre((2, 0)))[:2])
    assert tuple(stair) in r.ctx.tiles.stairs_crossed                 # walled off afterwards


def test_without_the_stairs_mode_the_robot_goes_round_and_never_climbs():
    r, res = run(**{'modes.exploring.stairs': False})
    assert res['complete'] and res['truth_tiles_visited'] == 9        # the loop reaches every tile anyway
    assert not res['ai_commands'] and 'stairs' not in res['modes']
    assert (2, 0) not in r.truth_visited


def test_a_mission_that_never_turns_ai_mode_on_bumps_into_the_riser():
    def deaf(r):
        r.ctx.ai_mode = lambda on, code: False                         # the robot never hears /simple_cmd
    r, res = run(ctx_patch=deaf, max_time=300)
    failed = [d for _, k, d in res['stairs_events'] if k == 'stairs_failed']
    assert failed and 'stuck' in failed[0]['why']
    assert not r.world.ai_mode
