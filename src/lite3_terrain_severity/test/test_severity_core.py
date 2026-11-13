"""Offline tests for severity_core (no ROS): python3 -m pytest -q test/test_severity_core.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lite3_terrain_severity'))
import severity_core as sc  # noqa: E402

rng = np.random.default_rng(1)


def ground(h_fn, noise=0.004, step=0.02):
    xs = np.arange(-1.5, 2.5, step)
    ys = np.arange(-1.0, 1.0, step)
    X, Y = [a.ravel() for a in np.meshgrid(xs, ys)]
    keep = ~((np.abs(X) < 0.45) & (np.abs(Y) < 0.30))     # classifier removes the body footprint
    X, Y = X[keep], Y[keep]
    H = np.array([h_fn(x, y) for x, y in zip(X, Y)]) + rng.normal(0, noise, X.size)
    return np.column_stack([X, Y, H - 0.3, H])


def lvl(h_fn, direction='forward'):
    return sc.assess_geometry(ground(h_fn), direction)[0]


def test_flat_is_smooth():
    assert lvl(lambda x, y: 0.0) == sc.SMOOTH


def test_small_bumps_mild_or_moderate():
    assert lvl(lambda x, y: 0.03 * np.sin(18 * x) * np.cos(15 * y)) in (sc.MILD, sc.MODERATE)


def test_block_field_is_severe():
    blocks = lambda x, y: 0.11 if (int(np.floor(x / 0.2)) + int(np.floor(y / 0.2))) % 2 else 0.0
    assert lvl(blocks) == sc.SEVERE


def test_tall_obstacle_close_is_extreme_far_is_severe():
    close = lambda x, y: 0.22 if 0.45 < x < 0.6 else 0.0
    far = lambda x, y: 0.22 if 1.3 < x < 1.45 else 0.0
    assert lvl(close) == sc.EXTREME
    assert lvl(far) == sc.SEVERE


def test_hole_detected():
    hole = lambda x, y: -0.18 if 0.5 < x < 0.8 else 0.0
    assert lvl(hole) == sc.EXTREME


def test_backward_looks_behind():
    behind = lambda x, y: 0.22 if -0.65 < x < -0.5 else 0.0
    assert lvl(behind, 'forward') == sc.SMOOTH
    assert lvl(behind, 'backward') == sc.EXTREME


def test_combine_rules():
    assert sc.combine(sc.SEVERE, 0, 'non_rugged') == sc.MILD          # trust the classifier when steady
    assert sc.combine(sc.SEVERE, sc.MODERATE, 'non_rugged') == sc.SEVERE  # robot wobbling: trust geometry
    assert sc.combine(None, 0, 'rugged') == sc.MODERATE              # blind => careful
    assert sc.combine(sc.SMOOTH, sc.SEVERE, 'rugged') == sc.SEVERE    # robot wobbling wins


def test_proprio_alarm():
    _, alarm, _ = sc.assess_proprio(20.0, 0.0, 0.5, 0.0)
    assert alarm
    _, alarm, _ = sc.assess_proprio(2.0, 15.0, 0.5, 0.0)            # ramp pitch is fine
    assert not alarm


def test_governor_pause_creep_block_cycle():
    g = sc.Governor()
    t = 0.0
    state, f = g.update(sc.SMOOTH, False, t)
    assert state == sc.RUNNING
    state, f = g.update(sc.EXTREME, False, t := t + 0.05)
    assert state == sc.PAUSED and f == 0.0
    assert g.scale(0.5, 0.0, 0.5) == (0.0, 0.0, 0.0)
    state, _ = g.update(sc.EXTREME, False, t := t + 1.6)             # pause over, still extreme
    assert state == sc.CREEP
    for _ in range(int(2.1 / 0.05)):
        state, f = g.update(sc.EXTREME, False, t := t + 0.05)
    assert state == sc.PAUSED and g.pauses == 2
    state, _ = g.update(sc.EXTREME, False, t := t + 1.6)
    for _ in range(int(2.1 / 0.05)):
        state, _ = g.update(sc.EXTREME, False, t := t + 0.05)
    state, _ = g.update(sc.EXTREME, False, t := t + 1.6)
    assert state == sc.BLOCKED
    vx, vy, wz = g.scale(0.5, 0.0, 0.5)
    assert vx == 0.0 and wz > 0                                       # may turn away, not drive in
    assert g.scale(-0.3, 0, 0)[0] < 0                                  # may back off
    for _ in range(30):
        state, _ = g.update(sc.MILD, False, t := t + 0.05)
    assert state == sc.RUNNING


def test_governor_slows_instantly_speeds_up_gently():
    g = sc.Governor()
    t = 0.0
    g.update(sc.SMOOTH, False, t)
    _, f = g.update(sc.SEVERE, False, t := t + 0.05)
    assert f == 0.25
    _, f = g.update(sc.SMOOTH, False, t := t + 0.5)                  # within down_hold_s
    assert f == 0.25
    _, f = g.update(sc.SMOOTH, False, t := t + 0.6)                  # hold over, ramping
    assert 0.25 < f < 1.0


def test_tilt_alarm_pauses_immediately():
    g = sc.Governor()
    g.update(sc.SMOOTH, False, 0.0)
    state, f = g.update(sc.SMOOTH, True, 0.05)
    assert state == sc.PAUSED and f == 0.0
