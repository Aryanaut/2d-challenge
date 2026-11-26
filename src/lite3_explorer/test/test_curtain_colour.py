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


def detect(depth, rgb, **params):
    return ccol.detect_curtain_colour(depth, rgb, K, K, IDENT, UP, ccol.ColourParams(**params))


def test_pale_smooth_curtain_is_found():
    d = detect(*scene((146, 137, 114), wall_from=0.15))           # cream, hangs from 0.15 m
    assert d is not None
    assert abs(float(np.median(d.ranges)) - 1.0) < 0.05
    heights = CAM_H - d.points[:, 1]
    assert heights.min() > 0.12


# colours measured in bag fullrun_25_11_233315 (median RGB of the masked pixels / of the floor in the same frame)
FLOOR_25NOV = (128, 130, 130)
CURTAIN_25NOV = (129, 124, 114)


def test_cream_curtain_close_up_is_found():
    # 0.5 m away the camera sees only up to ~0.8 m above the floor: no extend band needed (the 25 Nov bag's curtain was
    # 15-44 % of the image from 0.4-0.9 m away, extend 0)
    d = detect(*scene(CURTAIN_25NOV, floor_rgb=FLOOR_25NOV, wall_x=0.5, wall_from=0.15))
    assert d is not None and d.fraction > 0.3


def test_grey_wall_as_bright_as_the_floor_is_not_a_curtain():
    # bags fullrun_25_11_231517 / 232027 / 233315: under the 25 Nov lights the grey wall panels were 0.85-1.1x as
    # bright as the floor, low saturation and smooth = curtain-like on every other test, at 20-55 % of the image. They
    # are not warmer than the floor; the cream curtain is.
    for wall, floor in (((98, 100, 101), (112, 112, 111)), ((89, 94, 96), (92, 92, 89))):
        assert detect(*scene(wall, floor_rgb=floor)) is None
        assert detect(*scene(wall, floor_rgb=floor), min_rel_warmth=None) is not None   # warmth off: found


def test_neutral_white_curtain_close_up_is_found_without_the_warmth_test():
    assert detect(*scene((123, 123, 127), wall_x=0.6, wall_from=0.17), min_rel_warmth=None) is not None


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


def _cut_top(depth, top_m, wall_x=1.0):
    """Nothing above top_m on the surface (open space far away): a panel that is only top_m tall."""
    v = np.arange(480)[:, None] * np.ones((1, 640))
    above = (v - K[3]) / K[1] * wall_x < CAM_H - top_m
    depth = depth.copy()
    depth[above] = 0.0
    return depth


def test_pale_wall_only_as_tall_as_the_arena_walls_is_not_a_curtain_with_the_extend_check():
    # bag fullrun_25_11_215528: grey wall panels (0.5-0.6 m tall) looked as pale as the curtain under the arena
    # lights. The curtain hangs higher: the same surface must go on up into 0.6-1.2 m above the floor.
    # (min_extend_m2 is off by default now: the warmth test rejects the grey walls)
    ext = {'min_extend_m2': 0.06}
    depth, rgb = scene((146, 137, 114), wall_from=0.15)
    assert detect(_cut_top(depth, 0.55), rgb, **ext) is None
    d = detect(_cut_top(depth, 1.20), rgb, **ext)
    assert d is not None and d.extend_m2 >= 0.06
    # a taller wall FURTHER BACK does not count for the low panel in front of it
    low = _cut_top(depth, 0.55)
    assert detect(np.where(low == 0.0, 2.5, low).astype(np.float32), rgb, **ext) is None
