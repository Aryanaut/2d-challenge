import math
from types import SimpleNamespace

import numpy as np
import pytest

from lite3_terrain_classifier import terrain_core as tc
from lite3_terrain_classifier.msg_utils import cloud_to_xyz, depth_to_metres

GROUND_Z = -0.30   # ground height in base_link


def floor(rng, n=4000, noise=0.01, x=(-1.5, 2.5), y=(-1.5, 1.5)):
    pts = np.column_stack((rng.uniform(*x, n), rng.uniform(*y, n), np.full(n, GROUND_Z)))
    pts[:, 2] += rng.normal(0, noise, n)
    return pts


def add_blocks(pts, rng, heights=(0.08, 0.12, 0.10)):
    """Raise several 0.4 m square tiles, like uneven terrain blocks."""
    out = pts.copy()
    for i, h in enumerate(heights):
        x0, y0 = 0.6 + 0.5 * i, -0.8 + 0.6 * i
        m = (out[:, 0] > x0) & (out[:, 0] < x0 + 0.4) & (out[:, 1] > y0) & (out[:, 1] < y0 + 0.4)
        out[m, 2] += h
    return out


def assess(pts, up_known=True, **kw):
    params = tc.RoughnessParams(**kw)
    result, _, _ = tc.assess_terrain(pts, params, up_known=up_known, rng=np.random.default_rng(1))
    return result


def test_flat_floor_is_non_rugged():
    r = assess(floor(np.random.default_rng(0)))
    assert r.valid and r.label == tc.NON_RUGGED, r


def test_blocks_are_rugged():
    rng = np.random.default_rng(0)
    r = assess(add_blocks(floor(rng), rng))
    assert r.valid and r.label == tc.RUGGED, r
    assert r.max_step_m > 0.07


def test_single_small_bump_is_not_enough():
    rng = np.random.default_rng(0)
    r = assess(add_blocks(floor(rng), rng, heights=(0.08,)), cell_size_m=0.4)
    assert r.label == tc.NON_RUGGED, r


def test_wall_is_obstacle_not_rugged():
    rng = np.random.default_rng(0)
    pts = floor(rng)
    wall = np.column_stack((np.full(1500, 2.0) + rng.normal(0, 0.01, 1500),
                            rng.uniform(-1.5, 1.5, 1500), rng.uniform(GROUND_Z, GROUND_Z + 0.7, 1500)))
    r = assess(np.vstack((pts, wall)))
    assert r.label == tc.NON_RUGGED, r
    assert r.obstacle_cells > 0


def test_tilted_flat_plane_in_camera_frame_is_non_rugged():
    rng = np.random.default_rng(0)
    pts = floor(rng, noise=0.008, x=(0.3, 2.0), y=(-0.8, 0.8))
    rot = tc.quat_to_matrix(0.0, math.sin(0.4), 0.0, math.cos(0.4))   # arbitrary sensor rotation
    r = assess(tc.transform_points(pts, rot, [0.1, 0.2, 0.3]), up_known=False)
    assert r.label == tc.NON_RUGGED, r
    rugged = tc.transform_points(add_blocks(pts, rng), rot, [0.1, 0.2, 0.3])
    assert assess(rugged, up_known=False).label == tc.RUGGED


def test_tilt_test_optional():
    rng = np.random.default_rng(0)
    pts = floor(rng)
    pts[:, 2] += 0.3 * pts[:, 0]   # ~17 deg ramp
    assert assess(pts).label == tc.NON_RUGGED
    assert assess(pts, max_tilt_deg=10.0).label == tc.RUGGED


def test_too_few_points_is_unknown():
    r = assess(floor(np.random.default_rng(0), n=50))
    assert not r.valid and r.label == tc.UNKNOWN


def camera_depth_image(blocks=(), w=320, h=240, f=250.0, cam_h=0.5, pitch_deg=30.0, noise=0.0, seed=0):
    """Ray-cast a z-depth image of a floor seen by a camera pitched down.

    blocks: (x0, x1, y0, y1, height) boxes on the floor, x forward / y left of the camera.
    """
    pitch = math.radians(pitch_deg)
    vv, uu = np.mgrid[0:h, 0:w]
    a, b = (uu - w / 2) / f, (vv - h / 2) / f            # optical ray (a, b, 1)
    down = b * math.cos(pitch) + math.sin(pitch)          # world-down component of the ray
    fwd = math.cos(pitch) - b * math.sin(pitch)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(down > 0, cam_h / down, np.nan)      # ray length to z = 0
        gx, gy = t * fwd, -t * a
        for x0, x1, y0, y1, height in blocks:
            hit = (gx > x0) & (gx < x1) & (gy > y0) & (gy < y1)
            t = np.where(hit, (cam_h - height) / down, t)
    depth = t.astype(np.float32)                           # z-depth == ray parameter for (a, b, 1)
    if noise:
        depth += np.random.default_rng(seed).normal(0, noise, depth.shape).astype(np.float32) * depth
    return depth, f


def depth_assess(depth, f, top_fraction=0.5):
    h, w = depth.shape
    pts = tc.depth_to_points(depth, f, f, w / 2, h / 2, int(h * top_fraction), h, 16, w - 16, 4, 0.25, 2.0)
    return assess(pts, up_known=False, cell_spread_m=0.10, min_points=300, min_points_per_cell=10)


def test_depth_flat_floor_is_non_rugged():
    r = depth_assess(*camera_depth_image(noise=0.005))
    assert r.valid and r.label == tc.NON_RUGGED, r


def test_depth_blocks_are_rugged():
    blocks = [(0.55, 0.80, -0.30, -0.05, 0.08), (0.55, 0.80, 0.05, 0.30, 0.12), (0.70, 0.95, -0.10, 0.15, 0.06)]
    r = depth_assess(*camera_depth_image(blocks, noise=0.005))
    assert r.valid and r.label == tc.RUGGED, r


def test_depth_top_half_is_ignored():
    # Something odd far away (top half of the frame) must not change the answer.
    far = [(1.6, 1.9, -1.0, 1.0, 0.10)]
    depth, f = camera_depth_image(far)
    assert depth_assess(depth, f).label == tc.NON_RUGGED


def test_fusion_modes():
    flat = tc.TerrainAssessment(valid=True, rugged=False)
    rough = tc.TerrainAssessment(valid=True, rugged=True)
    none = tc.invalid('x')
    assert tc.fuse(rough, rough).verified
    r = tc.fuse(rough, flat, 'lidar_primary')
    assert r.label == tc.RUGGED and not r.verified and r.agree is False
    assert tc.fuse(rough, flat, 'require_agreement').label is None
    assert tc.fuse(flat, rough, 'any_rugged').label == tc.RUGGED
    assert tc.fuse(none, rough).label == tc.RUGGED and tc.fuse(none, rough).source == 'depth'
    assert tc.fuse(none, rough, depth_fallback=False).label == tc.UNKNOWN
    assert tc.fuse(flat, None).label == tc.NON_RUGGED
    with pytest.raises(ValueError):
        tc.fuse(flat, flat, 'bogus')


def test_debouncer():
    d = tc.Debouncer(3)
    assert [d.update(tc.RUGGED) for _ in range(3)] == [tc.UNKNOWN, tc.UNKNOWN, tc.RUGGED]
    assert d.update(tc.NON_RUGGED) == tc.RUGGED
    assert d.update(None) == tc.RUGGED        # hold resets the streak
    assert d.update(tc.NON_RUGGED) == tc.RUGGED
    d.update(tc.NON_RUGGED)
    assert d.update(tc.NON_RUGGED) == tc.NON_RUGGED


def test_cloud_decode_with_padding():
    pts = np.array([[1, 2, 3, 9], [4, 5, 6, 9], [np.nan, 0, 0, 9]], dtype='<f4')
    data = np.zeros((3, 20), dtype=np.uint8)
    data[:, :16] = pts.view(np.uint8).reshape(3, 16)
    field = lambda n, o: SimpleNamespace(name=n, offset=o, datatype=7)  # noqa: E731
    msg = SimpleNamespace(fields=[field('x', 0), field('y', 4), field('z', 8), field('intensity', 12)],
                          is_bigendian=False, point_step=20, row_step=60, width=3, height=1,
                          data=data.tobytes())
    np.testing.assert_allclose(cloud_to_xyz(msg), [[1, 2, 3], [4, 5, 6]])


def test_depth_decode_16uc1():
    img = np.array([[0, 1000], [2500, 65535]], dtype='<u2')
    msg = SimpleNamespace(encoding='16UC1', is_bigendian=False, step=4, height=2, width=2,
                          data=img.tobytes())
    out = depth_to_metres(msg)
    assert np.isnan(out[0, 0]) and out[0, 1] == pytest.approx(1.0) and out[1, 0] == pytest.approx(2.5)
