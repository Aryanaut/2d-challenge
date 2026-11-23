"""Detector tests on synthetic point clouds. Run: python3 -m pytest test/"""

import math

import numpy as np

from stair_climber.stair_detector import (DetectorParams, detect_stairs, estimate_floor_z,
                                          rpy_to_matrix, to_level_base_frame)

FLOOR_Z = -0.30   # floor height in the level base frame
RNG = np.random.default_rng(0)


def _grid(u0, u1, v0, v1, step=0.02):
    u, v = np.meshgrid(np.arange(u0, u1, step), np.arange(v0, v1, step))
    return u.ravel(), v.ravel()


def floor(x0=0.0, x1=2.0, half_w=0.8):
    u, v = _grid(x0, x1, -half_w, half_w)
    return np.column_stack([u, v, np.full_like(u, FLOOR_Z)])


def stairs(edge_x=0.8, step_h=0.12, tread=0.28, n=3, width=0.9, yaw_deg=0.0,
           center_y=0.0, with_floor=True, noise=0.005):
    """Ascending stairs. Local frame: u = along the stair normal, v = along the edge."""
    pts = []
    for k in range(n):
        u0 = k * tread
        top = (k + 1) * step_h
        # riser (vertical face) at u0
        z, v = _grid(k * step_h, top, -width / 2, width / 2)
        pts.append(np.column_stack([np.full_like(z, u0), v, z]))
        # tread (horizontal) from u0 to u0+tread
        u, v = _grid(u0, u0 + tread, -width / 2, width / 2)
        pts.append(np.column_stack([u, v, np.full_like(u, top)]))
    local = np.vstack(pts)
    th = math.radians(yaw_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    xy = local[:, :2] @ R.T + np.array([edge_x, center_y])
    out = np.column_stack([xy, local[:, 2] + FLOOR_Z])
    if with_floor:
        f = floor()
        # remove floor that would be under the staircase
        rel = (f[:, :2] - np.array([edge_x, center_y])) @ R
        f = f[rel[:, 0] < 0]
        out = np.vstack([f, out])
    return out + RNG.normal(0, noise, out.shape)


def test_flat_floor_not_detected():
    obs = detect_stairs(floor(), FLOOR_Z)
    assert not obs.detected
    assert 'flat' in obs.reason


def test_straight_stairs_detected():
    obs = detect_stairs(stairs(edge_x=0.8, step_h=0.12), FLOOR_Z)
    assert obs.detected, obs.reason
    assert abs(obs.distance - 0.8) < 0.04
    assert abs(obs.step_height - 0.12) < 0.025
    assert abs(math.degrees(obs.edge_yaw)) < 2.0
    assert abs(obs.center_y) < 0.05
    assert obs.num_steps >= 2


def test_tall_steps_15cm_detected():
    obs = detect_stairs(stairs(edge_x=0.7, step_h=0.15), FLOOR_Z)
    assert obs.detected, obs.reason
    assert abs(obs.step_height - 0.15) < 0.025


def test_rotated_stairs_give_yaw_correction():
    for yaw in (-12.0, 10.0):
        obs = detect_stairs(stairs(edge_x=0.9, yaw_deg=yaw), FLOOR_Z)
        assert obs.detected, obs.reason
        assert abs(math.degrees(obs.edge_yaw) - yaw) < 2.5, (yaw, math.degrees(obs.edge_yaw))


def test_offset_stairs_give_lateral_correction():
    # Centring only works if the whole step width fits inside the ROI
    # (roi_half_width must be a bit more than half the staircase width).
    obs = detect_stairs(stairs(edge_x=0.8, center_y=0.15, width=0.6), FLOOR_Z)
    assert obs.detected, obs.reason
    assert abs(obs.center_y - 0.15) < 0.05


def test_wall_rejected():
    z, y = _grid(0.0, 1.0, -0.8, 0.8)
    wall = np.column_stack([np.full_like(z, 0.8), y, z + FLOOR_Z])
    f = floor(0.0, 0.8)
    obs = detect_stairs(np.vstack([f, wall]), FLOOR_Z)
    assert not obs.detected
    assert 'wall' in obs.reason or 'too tall' in obs.reason or 'steep' in obs.reason


def test_handrail_posts_do_not_block_detection():
    pts = [stairs(edge_x=0.8, width=0.8)]
    for post_y in (-0.45, 0.45):                 # vertical posts beside the steps
        for post_x in (0.8, 1.1, 1.4):
            z, _ = _grid(0.0, 1.0, 0.0, 0.04)
            pts.append(np.column_stack([np.full_like(z, post_x), np.full_like(z, post_y),
                                        z + FLOOR_Z]))
    obs = detect_stairs(np.vstack(pts), FLOOR_Z)
    assert obs.detected, obs.reason
    assert abs(obs.distance - 0.8) < 0.05
    assert abs(obs.center_y) < 0.06


def test_narrow_box_rejected():
    obs = detect_stairs(stairs(edge_x=0.8, width=0.2, n=1), FLOOR_Z)
    assert not obs.detected


def test_too_close_rejected():
    # Edge closer than the ROI start: no floor visible before the rise
    obs = detect_stairs(stairs(edge_x=0.2), FLOOR_Z)
    assert not obs.detected


def test_transform_roundtrip_and_floor_estimate():
    """Points generated in the level frame -> sensor frame -> back again."""
    level_pts = stairs()
    lidar_R = rpy_to_matrix(0.0, math.radians(15), 0.0)
    lidar_t = np.array([0.18678, 0.0, 0.12498])
    roll, pitch = math.radians(2.0), math.radians(-3.0)
    body = rpy_to_matrix(roll, pitch, 0.0)
    # level = body @ (R p + t)  =>  p = R^T (body^T level - t)
    sensor_pts = ((level_pts @ body) - lidar_t) @ lidar_R
    back = to_level_base_frame(sensor_pts, lidar_R, lidar_t, roll, pitch)
    assert np.allclose(back, level_pts, atol=1e-9)
    assert abs(estimate_floor_z(back) - FLOOR_Z) < 0.01
    assert detect_stairs(back, estimate_floor_z(back)).detected


def test_params_are_respected():
    p = DetectorParams(max_step_height=0.10)
    obs = detect_stairs(stairs(step_h=0.15), FLOOR_Z, p)
    assert not obs.detected


def arena_stairs(edge_x=1.6, noise=0.005):
    """The measured arena stairs: risers 19.5 / 20 / 12 cm, treads 13.5 / 12.5 / 18 cm,
    100 cm wide, then the same back down."""
    risers, treads = [0.195, 0.20, 0.12], [0.135, 0.125, 0.18]
    pts, x, z = [], edge_x, 0.0
    for h, t in zip(risers, treads):
        zz, v = _grid(z, z + h, -0.5, 0.5)
        pts.append(np.column_stack([np.full_like(zz, x), v, zz]))
        z += h
        u, v = _grid(x, x + t, -0.5, 0.5)
        pts.append(np.column_stack([u, v, np.full_like(u, z)]))
        x += t
    f = floor(0.0, edge_x)
    out = np.vstack([f] + pts)
    out[:, 2] += np.where(out[:, 2] == FLOOR_Z, 0.0, FLOOR_Z)
    return out + RNG.normal(0, noise, out.shape)


def test_arena_stairs_20cm_risers_detected():
    p = DetectorParams(max_step_height=0.24)
    obs = detect_stairs(arena_stairs(edge_x=1.2), FLOOR_Z, p)
    assert obs.detected, obs.reason
    assert abs(obs.distance - 1.2) < 0.05
    assert abs(obs.step_height - 0.195) < 0.03, obs.step_height
