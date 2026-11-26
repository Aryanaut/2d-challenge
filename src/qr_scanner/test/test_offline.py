"""Offline tests (no ROS): python3 -m pytest test/ -q"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qr_scanner.localize import backproject, bearing_deg, depth_at, depth_from_size  # noqa: E402
from qr_scanner.payload import caesar, decode_payload, parse_key_map  # noqa: E402
from qr_scanner.qr_decode import QrReader  # noqa: E402


def _scene(text, qr_px, at=(200, 500), size=(720, 1280), blur=0, angle=0):
    """A 1280x720 'wall' (OSB-like noise texture) with a QR code of qr_px width pasted on a
    white sheet of paper."""
    rng = np.random.default_rng(1)
    wall = rng.integers(120, 200, size=size, dtype=np.uint8)
    wall = cv2.GaussianBlur(wall, (7, 7), 0)
    qr = cv2.QRCodeEncoder.create().encode(text)               # includes quiet zone
    modules = qr.shape[0]
    side = int(qr_px * modules / (modules - 8))                # qr_px = black square width
    qr = cv2.resize(qr, (side, side), interpolation=cv2.INTER_NEAREST)
    paper = np.full((int(side * 1.4), int(side * 1.8)), 245, np.uint8)
    py, px = (paper.shape[0] - side) // 2, (paper.shape[1] - side) // 2
    paper[py:py + side, px:px + side] = qr
    if angle:
        M = cv2.getRotationMatrix2D((paper.shape[1] / 2, paper.shape[0] / 2), angle, 1.0)
        paper = cv2.warpAffine(paper, M, (paper.shape[1], paper.shape[0]), borderValue=170)
    y, x = at
    wall[y:y + paper.shape[0], x:x + paper.shape[1]] = paper
    if blur:
        wall = cv2.GaussianBlur(wall, (blur, blur), 0)
    return cv2.cvtColor(wall, cv2.COLOR_GRAY2BGR)


def test_reads_large_and_medium_codes():
    r = QrReader()
    for px in (200, 120, 70, 50):
        hits = [h for h in r.read(_scene("J", px)) if h.decoded]
        assert [h.text for h in hits] == ["J"], px
        assert abs(hits[0].side_px - px) < 0.35 * px, (px, hits[0].side_px)


def test_reads_rotated_and_blurred_medium_code():
    hits = [h for h in QrReader().read(_scene("I", 80, blur=5, angle=15)) if h.decoded]
    assert [h.text for h in hits] == ["I"]


def test_two_codes_in_one_frame():
    img = _scene("I", 90, at=(100, 100))
    img2 = _scene("J", 60, at=(350, 800))
    img[300:, 700:] = img2[300:, 700:]
    texts = sorted(h.text for h in QrReader().read(img) if h.decoded)
    assert texts == ["I", "J"]


def test_no_false_positive_on_plain_wall():
    rng = np.random.default_rng(3)
    wall = cv2.GaussianBlur(rng.integers(100, 220, (720, 1280), dtype=np.uint8), (5, 5), 0)
    assert QrReader().read(cv2.cvtColor(wall, cv2.COLOR_GRAY2BGR)) == []


def test_payload_modes():
    km = parse_key_map(["I:storeroom", "J:kitchen", "bad-entry"])
    assert km == {"I": "storeroom", "J": "kitchen"}
    assert decode_payload("J", key_map=km) == "kitchen"
    assert decode_payload("IJ", key_map=km) == "storeroom kitchen"
    assert decode_payload("J") == "J"
    assert caesar("KHOOR", 3) == "HELLO"
    assert decode_payload("KHOOR", "caesar", 3) == "HELLO"
    assert decode_payload("aGVsbG8=", "base64") == "hello"
    assert decode_payload("I", "auto") == "I"            # short text is not mistaken for base64
    assert decode_payload('{"room": "kitchen"}', "auto") == '{"room":"kitchen"}'


def test_geometry():
    K = [[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]]
    assert abs(depth_from_size(45, 900, 0.09) - 1.8) < 1e-6
    x, y, z = backproject(640 + 90, 360, 2.0, K)
    assert abs(x - 0.2) < 1e-6 and y == 0 and z == 2.0
    assert bearing_deg(400, K) > 0 and bearing_deg(900, K) < 0     # left of centre = positive
    depth = np.full((480, 640), 1500, np.uint16)                  # half-res aligned depth, mm
    corners = np.float32([[600, 300], [700, 300], [700, 400], [600, 400]])
    assert abs(depth_at(depth, corners, (720, 1280)) - 1.5) < 1e-6
    depth[:] = 0
    assert depth_at(depth, corners, (720, 1280)) is None


# ---------- multi-camera tracking ----------
def test_tracker_dedups_across_cameras():
    from qr_scanner.tracker import CodeTracker
    tr = CodeTracker(confirm_frames=2, confirm_window_s=3.0, same_code_dist_m=1.5)
    assert tr.observe("I", "front", 0.0)[1] is False
    code, new = tr.observe("I", "left", 0.5)            # second read from another camera confirms
    assert new and code.cameras == {"front", "left"}
    assert tr.observe("I", "right", 1.0)[1] is False     # never reported twice
    assert tr.sequence == ["I"]


def test_tracker_window_and_separate_physical_codes():
    from qr_scanner.tracker import CodeTracker
    tr = CodeTracker(confirm_frames=2, confirm_window_s=3.0, same_code_dist_m=1.5)
    tr.observe("J", "front", 0.0)
    assert tr.observe("J", "front", 10.0)[1] is False     # too far apart in time: count restarts
    assert tr.observe("J", "front", 11.0)[1] is True
    # same letter on another wall 4 m away (map positions known) = a second code
    tr2 = CodeTracker()
    tr2.observe("I", "front", 0, [0, 0, 0]); tr2.observe("I", "front", 0.2, [0, 0, 0])
    tr2.observe("I", "left", 5, [4, 0, 0]); _, new = tr2.observe("I", "left", 5.2, [4, 0, 0])
    assert new and len([c for c in tr2.codes if c.confirmed]) == 2 and tr2.sequence == ["I"]


def test_mount_geometry():
    from qr_scanner.localize import intrinsics_from_fov, optical_mount_quaternion, rotate
    fwd = (0, 0, 1)                                   # optical axis
    assert np.allclose(rotate(optical_mount_quaternion(0), fwd), (1, 0, 0), atol=1e-9)    # front
    assert np.allclose(rotate(optical_mount_quaternion(90), fwd), (0, 1, 0), atol=1e-9)   # left
    assert np.allclose(rotate(optical_mount_quaternion(-90), fwd), (0, -1, 0), atol=1e-9) # right
    assert np.allclose(rotate(optical_mount_quaternion(0, 30), fwd), (0.866, 0, -0.5), atol=1e-3)
    K = intrinsics_from_fov(1280, 720, 90.0)
    assert abs(K[0][0] - 640) < 1e-6 and K[0][2] == 640 and K[1][2] == 360


def test_full_res_corners_with_full_res_intrinsics_match_preview():
    """Codes read by the sender come in full-res pixels; with full-res K they give the same
    bearing/distance as the same code measured in the scaled-down preview with preview K."""
    from qr_scanner.localize import intrinsics_from_fov
    K_full = intrinsics_from_fov(1920, 1080, 78.0)
    K_prev = intrinsics_from_fov(640, 360, 78.0)
    u_full, side_full = 1500.0, 90.0
    assert abs(bearing_deg(u_full, K_full) - bearing_deg(u_full / 3, K_prev)) < 0.2
    assert abs(depth_from_size(side_full, K_full[0][0], 0.09) - depth_from_size(side_full / 3, K_prev[0][0], 0.09)) < 0.01
