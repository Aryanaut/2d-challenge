"""Colour + depth curtain check (curtain_colour.py) on synthetic camera views: camera 0.46 m above the floor, level,
looking at an upright surface 1 m ahead. Optical frame: x right, y down, z forward."""
import numpy as np

from lite3_explorer import curtain_colour as ccol

K = (372.0, 372.0, 320.0, 240.0)
UP = (0.0, -1.0, 0.0)                     # 'up' in the optical frame (y points down)
IDENT = (np.eye(3), (0.0, 0.0, 0.0))      # colour camera = depth camera
CAM_H = 0.46


def scene(surface_rgb, floor_rgb=(130, 132, 120), wall_x=1.0, wall_from=0.0, noise=0.0, no_wall=False, seed=0):
    """Depth (m) and RGB image of a floor + an upright surface wall_x ahead that starts wall_from above the floor."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:480, 0:640].astype(float)
    ry = (v - K[3]) / K[1]                  # ray direction (x, y, 1)
    with np.errstate(divide='ignore'):
        t_floor = np.where(ry > 1e-6, CAM_H / ry, np.inf)
    t_wall = np.full_like(ry, np.inf) if no_wall else np.full_like(ry, wall_x)
    y_at_wall = ry * wall_x                 # + = below the camera
    t_wall[y_at_wall > CAM_H - wall_from] = np.inf    # below the surface's lower edge: see past it
    t = np.minimum(t_floor, t_wall)
    depth = np.where(np.isfinite(t) & (t < 4.0), t, 0.0).astype(np.float32)
    rgb = np.zeros((480, 640, 3), float)
    on_wall = t_wall <= t_floor
    rgb[on_wall] = surface_rgb
    rgb[~on_wall] = floor_rgb
    rgb += rng.normal(0, 1.0, rgb.shape)    # a little sensor noise everywhere
    if noise:
        rgb[on_wall] += rng.normal(0, noise, (on_wall.sum(), 1))   # texture (wood grain)
    return depth, np.clip(rgb, 0, 255).astype(np.uint8)


def detect(depth, rgb):
    return ccol.detect_curtain_colour(depth, rgb, K, K, IDENT, UP)


def test_pale_smooth_curtain_is_found():
    d = detect(*scene((146, 137, 114), wall_from=0.15))           # cream, hangs from 0.15 m
    assert d is not None
    assert abs(float(np.median(d.ranges)) - 1.0) < 0.05
    heights = CAM_H - d.points[:, 1]
    assert heights.min() > 0.12


def test_neutral_white_curtain_close_up_is_found():
    assert detect(*scene((123, 123, 127), wall_x=0.6, wall_from=0.17)) is not None


def test_grey_wall_is_not_a_curtain():
    assert detect(*scene((75, 79, 80))) is None                   # arena wall panel, darker than the floor


def test_textured_wood_is_not_a_curtain():
    assert detect(*scene((129, 116, 81), noise=12.0)) is None     # OSB: yellow and grainy


def test_floor_only_is_not_a_curtain():
    assert detect(*scene((0, 0, 0), no_wall=True)) is None


def test_low_white_object_is_not_a_curtain():
    # pale surface only up to 0.10 m (e.g. a ledge / skirting): wall_from at floor, but cut it to 0.10 m tall
    depth, rgb = scene((143, 145, 133))
    v = np.arange(480)[:, None] * np.ones((1, 640))
    y_at_wall = (v - K[3]) / K[1] * 1.0
    above = y_at_wall < CAM_H - 0.10                               # rows looking above 0.10 m on the surface
    depth[above] = 0.0                                             # nothing there (open space far away)
    assert detect(depth, rgb) is None


def test_one_per_cell():
    pts = np.array([[0.01, 0.01], [0.02, 0.03], [0.30, 0.0]])
    out = ccol.one_per_cell(pts, 0.05)
    assert len(out) == 2
