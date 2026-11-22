"""World simulation (no ROS): lidar, SLAM map, costmaps, planner, robot motion, curtain camera."""
import math

import numpy as np
import pytest

from lite3_sim.maze import load_maze_dict
from lite3_sim.world import FAILED, SUCCEEDED, World, invert, compose

CORRIDOR = {'layout': '+---+---+---+---+\n| >     ~       |\n+---+---+---+---+\n'}


def run(w, task, seconds):
    for _ in range(int(seconds / 0.05)):
        w.step(0.05)
        if task.done:
            break
    return task


def test_map_frame_is_the_start_pose():
    m = load_maze_dict(dict(CORRIDOR, start={'yaw_offset_deg': 10}))
    w = World(m)
    assert w.pose == pytest.approx([0.0, 0.0, 0.0])
    x, y, yaw = w.maze_to_map(*m.start)
    assert (x, y, yaw) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    w2 = World(m, {'map_origin': 'maze'})
    assert w2.pose == pytest.approx(list(m.start))


def test_lidar_sees_curtain_as_wall_and_never_leaks():
    w = World(load_maze_dict(CORRIDOR), {'lidar': {'noise_m': 0.0}})
    w.step(0.05)
    rel, rng, _ = w.last_scan
    ahead = rng[np.argmin(np.abs(rel))]
    assert ahead == pytest.approx(1.53, abs=0.02)          # the curtain on the east edge of tile 1,0
    # drive right up to the curtain (inside the lidar's blind zone) and through it: nothing outside gets mapped
    w.set_cmd_vel(0.3, 0.0, 0.0)
    for _ in range(200):
        w.set_cmd_vel(0.3, 0.0, 0.0)
        w.step(0.05)
    assert w.pose[0] > 2.0                                   # walked through the curtain
    g, info = w.mapper.grid(), w.info
    rows, cols = np.nonzero(g >= 0)
    ys = info.origin_y + (rows + 0.5) * info.resolution
    assert ys.min() > -0.51 - 0.1 and ys.max() < 0.51 + 0.1  # nothing known beyond the corridor's side walls


def test_robot_cannot_pass_walls():
    w = World(load_maze_dict(CORRIDOR))
    for _ in range(100):
        w.set_cmd_vel(0.0, 0.3, 0.0)                         # sideways into the north wall
        w.step(0.05)
    assert w.bumped and w.pose[1] < 0.51 - 0.185 + 0.01


def test_navigate_plans_around_and_reaches_goal():
    w = World(load_maze_dict({'layout': '+---+---+\n| > |   |\n+   +   +\n|       |\n+---+---+\n'}),
              {'controller': {'xy_goal_tolerance': 0.15}})
    for _ in range(40):
        w.step(0.05)
    task = run(w, w.navigate((0.0, -1.02, 0.0)), 30)        # map frame: start tile 0,1 -> the tile south of it
    assert task.state == SUCCEEDED
    goal = w.maze_to_map(1.5 * 1.02, 1.5 * 1.02)            # tile 1,1: only reachable round through the bottom row
    for _ in range(60):
        w.step(0.05)
    path = w.plan(w.pose[:2], goal[:2])
    assert path is not None and max(math.dist(p, w.pose[:2]) for p in path) > 0.5
    task = run(w, w.navigate((goal[0], goal[1], 0.0)), 90)
    assert task.state == SUCCEEDED and w.tile_of_maze() == (1, 1)


def test_spin_and_waypoints():
    w = World(load_maze_dict(CORRIDOR))
    s = run(w, w.spin(math.pi, 20.0), 20)
    assert s.state == SUCCEEDED and abs(abs(w.pose[2]) - math.pi) < 0.1
    wp = run(w, w.follow_waypoints([(0.5, 0.0, 0.0), (5.0, 3.0, 0.0), (1.02, 0.0, 0.0)]), 90)
    assert wp.state == SUCCEEDED and wp.missed == [1]       # the middle pose is outside the maze


def test_collision_detection_aborts_instead_of_pushing_into_walls():
    w = World(load_maze_dict(CORRIDOR))
    w.pose = [0.0, 0.25, math.pi / 2]                        # facing the north wall, 0.26 m away
    w.step(0.05)
    task = w.navigate((0.0, 0.0, 0.0))
    from lite3_sim.world import NavTask
    t = NavTask((0.0, 0.9, 0.0))                             # a goal inside the wall
    w.start_task(t)
    run(w, t, 20)
    assert t.state == FAILED


def test_curtain_camera_confirms_spots_in_front():
    w = World(load_maze_dict(CORRIDOR))
    for _ in range(60):
        w.step(0.05)
    pts = w.curtain_points()
    assert len(pts) >= 6
    assert np.all(np.abs(pts[:, 0] - 1.53) < 0.08) and np.all(pts[:, 2] > 0.5)    # on the curtain, seen looking +x
    w2 = World(load_maze_dict(dict(CORRIDOR, start={'heading': 'W'})))
    for _ in range(60):
        w2.step(0.05)
    assert len(w2.curtain_points()) == 0                     # behind the camera


def test_pose_helpers():
    a = (1.0, 2.0, 0.7)
    assert compose(a, invert(a)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)
