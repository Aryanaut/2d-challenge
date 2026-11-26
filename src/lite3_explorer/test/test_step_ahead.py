"""step_ahead.py: step-up detection in ray-cast depth images from the Lite3 camera mount (no ROS)."""
import math

import numpy as np

from lite3_explorer import step_ahead as sa

K = (372.25, 372.25, 317.9, 243.5)          # fullrun_26_11_210206 camera_info, 640 x 480
W, H = 640, 480
FLOOR = -0.33                                # floor under base_link
CAM_T = np.array([0.25228, -0.0005, 0.0741])  # base_link -> camera_link (tf_static)
LINK_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])   # optical -> camera_link axes


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


R_BASE_OPT = rot_y(math.radians(20.0)) @ LINK_OPT   # camera 20 deg nose down


def render(height_at, body_pitch=0.0, max_t=4.0, step=0.01):
    """Depth image of the height field z = height_at(x, y) (level base_link frame), by ray marching from the camera.
    body_pitch: ROS pitch of the body (+ = nose down)."""
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    d_opt = np.stack([(u - K[2]) / K[0], (v - K[3]) / K[1], np.ones_like(u)], axis=-1)   # optical z = 1: t is depth
    r = sa.level_rotation(0.0, body_pitch) @ R_BASE_OPT
    d = d_opt @ r.T
    c = sa.level_rotation(0.0, body_pitch) @ CAM_T
    depth = np.zeros((H, W))
    hit = np.zeros((H, W), bool)
    for t in np.arange(0.05, max_t, step):
        p = c + t * d
        under = (~hit) & (p[..., 2] <= height_at(p[..., 0], p[..., 1]))
        depth[under], hit = t, hit | under
    return depth


def level_points(depth, body_pitch=0.0, p=sa.StepAheadParams()):
    return sa.to_level_base(sa.depth_to_points(depth, K, p), R_BASE_OPT, CAM_T, 0.0, body_pitch)


def step_field(x0, rise, yaw_deg=0.0, depth_m=2.0):
    """Floor, and a raised block rise high from the line x = x0 (turned yaw_deg about z through (x0, 0)) for depth_m."""
    a = math.radians(yaw_deg)

    def f(x, y):
        along = (x - x0) * math.cos(a) + y * math.sin(a)
        return np.where((along >= 0) & (along < depth_m), FLOOR + rise, FLOOR)
    return f


def detect(field, body_pitch=0.0, **kw):
    p = sa.StepAheadParams(**kw)
    return sa.detect(level_points(render(field, body_pitch), body_pitch, p), p)


def test_flat_floor_is_no_step():
    assert detect(lambda x, y: np.full_like(x, FLOOR)) is None


def test_floor_heights_project_back():
    pts = level_points(render(lambda x, y: np.full_like(x, FLOOR)))
    ahead = pts[(pts[:, 0] > 0.5) & (pts[:, 0] < 1.2)]
    assert len(ahead) > 1000
    assert abs(np.median(ahead[:, 2]) - FLOOR) < 0.01


def test_step_ahead_found_with_its_height_and_distance():
    det = detect(step_field(0.80, 0.14))
    assert det is not None
    assert abs(det.rise_m - 0.14) < 0.02
    assert 0.70 <= det.dist_m <= 0.85
    assert not det.at_edge


def test_low_step_found():
    det = detect(step_field(0.70, 0.08))
    assert det is not None and abs(det.rise_m - 0.08) < 0.02


def test_step_below_min_rise_ignored():
    assert detect(step_field(0.70, 0.04)) is None


def test_step_at_an_angle_found():
    det = detect(step_field(0.80, 0.13, yaw_deg=35.0))
    assert det is not None and abs(det.rise_m - 0.13) < 0.03


def test_step_too_far_ignored():
    assert detect(step_field(1.6, 0.14)) is None


def test_at_the_edge_uses_the_nominal_floor():
    det = detect(step_field(0.30, 0.13))         # the nearest visible ground (~0.45 m) is the top already
    assert det is not None and det.at_edge
    assert abs(det.rise_m - 0.13) < 0.02


def test_pitched_body_is_levelled_by_the_imu():
    det = detect(step_field(0.80, 0.14), body_pitch=math.radians(-4.0))   # nose up 4 deg
    assert det is not None and abs(det.rise_m - 0.14) < 0.02


def test_ramp_is_no_step():
    ramp = lambda x, y: FLOOR + np.clip((x - 0.6) * math.tan(math.radians(15.0)), 0.0, 0.25)  # noqa: E731
    assert detect(ramp) is None


def test_wall_and_tall_box_are_no_step():
    assert detect(step_field(0.8, 1.0)) is None
    assert detect(step_field(0.8, 0.45)) is None


def test_thin_ledge_is_no_step():
    assert detect(step_field(0.8, 0.14, depth_m=0.08)) is None     # top shallower than min_top_m


def test_narrow_object_needs_two_strips():
    def post(x, y):
        return np.where((x > 0.8) & (x < 1.2) & (np.abs(y - 0.15) < 0.03), FLOOR + 0.12, FLOOR)
    assert detect(post) is None


def test_tilted():
    p = sa.StepAheadParams(max_tilt_deg=6.0)
    assert not sa.tilted(0.0, math.radians(-5.0), p)
    assert sa.tilted(0.0, math.radians(-8.0), p)
    assert sa.tilted(math.radians(7.0), 0.0, p)


def test_rpy_from_quat():
    a = math.radians(10.0)
    roll, pitch, yaw = sa.rpy_from_quat(0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
    assert abs(pitch - a) < 1e-9 and abs(roll) < 1e-9 and abs(yaw) < 1e-9


def test_confirm_needs_frames_in_a_row():
    c = sa.StepConfirm(2)
    hit = sa.StepAhead(0.7, 0.12, FLOOR, FLOOR + 0.12, False)
    assert not c.update(hit)
    assert not c.update(None)
    assert not c.update(hit)
    assert c.update(hit)
    assert c.update(hit)
    assert not c.update(None)
