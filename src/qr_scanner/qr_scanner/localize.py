"""Pixel + depth (or known QR size) -> 3D point in the camera optical frame. Pure numpy."""

from typing import Optional, Tuple

import cv2
import numpy as np


def depth_at(depth: Optional[np.ndarray], corners: np.ndarray, color_shape,
             depth_scale: float = 0.001, min_m: float = 0.1, max_m: float = 8.0) -> Optional[float]:
    """Median valid depth (m) inside the QR polygon. Handles a depth image with a different
    resolution from the colour image (assumes it is aligned to colour)."""
    if depth is None:
        return None
    dh, dw = depth.shape[:2]
    ch, cw = color_shape[:2]
    sx, sy = dw / cw, dh / ch
    quad = (corners * np.float32([sx, sy])).astype(np.int32)
    mask = np.zeros((dh, dw), np.uint8)
    cv2.fillConvexPoly(mask, quad, 1)
    vals = depth[mask.astype(bool)].astype(np.float32)
    if depth.dtype != np.float32 and depth.dtype != np.float64:
        vals = vals * depth_scale            # 16UC1 millimetres -> metres
    vals = vals[np.isfinite(vals) & (vals > min_m) & (vals < max_m)]
    if vals.size < 5:
        return None
    return float(np.median(vals))


def depth_from_size(side_px: float, fx: float, qr_size_m: float) -> Optional[float]:
    """Pinhole estimate: distance = fx * real size / pixel size."""
    if side_px <= 1 or fx <= 0 or qr_size_m <= 0:
        return None
    return fx * qr_size_m / side_px


def backproject(u: float, v: float, z: float, K) -> Tuple[float, float, float]:
    """K = 3x3 (fx 0 cx; 0 fy cy; 0 0 1). Returns (x, y, z) in the optical frame
    (x right, y down, z forward)."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


def bearing_deg(u: float, K) -> float:
    """Horizontal angle of pixel column u from the optical axis, positive = left of centre
    (ROS convention: positive yaw turns left)."""
    fx, cx = K[0][0], K[0][2]
    return float(np.degrees(np.arctan2(-(u - cx), fx)))


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """Approximate K for an uncalibrated camera from its horizontal field of view."""
    fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1]], dtype=np.float64)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def optical_mount_quaternion(yaw_deg: float, pitch_deg: float = 0.0):
    """Rotation (x, y, z, w) from the robot base frame (x forward, y left, z up) to a camera
    OPTICAL frame (x right, y down, z forward) for a camera turned yaw_deg to the left and
    tilted pitch_deg downwards."""
    y, p = np.radians(yaw_deg) / 2.0, np.radians(pitch_deg) / 2.0
    q_yaw = (0.0, 0.0, np.sin(y), np.cos(y))
    q_pitch = (0.0, np.sin(p), 0.0, np.cos(p))
    q_opt = (-0.5, 0.5, -0.5, 0.5)          # camera link -> optical frame (REP 103/105 convention)
    return _qmul(_qmul(q_yaw, q_pitch), q_opt)


def rotate(q, v):
    """Rotate vector v by quaternion q (x, y, z, w)."""
    x, y, z, w = _qmul(_qmul(q, (v[0], v[1], v[2], 0.0)), (-q[0], -q[1], -q[2], q[3]))
    return np.array([x, y, z])
