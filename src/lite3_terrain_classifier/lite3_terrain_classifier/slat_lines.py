"""Slats from straight parallel lines in a camera image (Hough transform). No ROS.

Slats show up in the images as several parallel straight edges:
  rungs        near-horizontal lines at different heights of the image (slats lying across the view),
  converging   lines running away from the camera, crossing the same image row at different places (slats along it).
In the depth image the gaps between slats are dark streaks (no depth) and the slat tops are steps, so the same edge
detector finds both. Only the lower part of the image (the floor directly in front) is used.

detect(gray) -> {'rungs': n, 'converging': n, 'segments': n, 'orientation', 'angle_deg', ...}: the number of DISTINCT
lines in each family. A family only counts when its lines really are slat-like:
  rungs        must be PARALLEL (angle spread of the lines <= max_rung_angle_std_deg),
  converging   must meet at ONE vanishing point (mean distance <= max_vanishing_residual of the image width).
  rungs        must also be LONG: the lines of a rung must run across at least min_rung_span of the image width.
Uneven ground gives lines that fail these tests: ramps and wedges (mixed angles), fields of blocks and steps (parallel
but short edges), so they are not mistaken for slats; the raw counts before the test stay in rungs_raw / converging_raw for debugging.
A frame says "slats" when either family has at least min_lines (see is_slats).

Checked on the pictures in images/ (test/test_slat_lines.py): the depth picture with dark streaks and the three colour
photos of slats are slats; the two uneven pictures (depth and colour) and the curtain pictures are not. The colour
pictures are noisy (OSB grain, curtain folds), so the colour detector needs more lines and the result is a vote over many
frames, not one frame.
"""
from __future__ import annotations

import math
from typing import Dict

import cv2
import numpy as np

DEFAULTS = {
    'roi_top_fraction': 0.5,        # only the lower part of the image: the floor in front of the robot
    'side_margin_fraction': 0.04,   # image border artefacts
    'canny_low': 0,                 # 0 = automatic (from the image median)
    'canny_high': 0,
    'hough_threshold': 30,          # votes for a line
    'min_length_fraction': 0.10,    # shortest segment, share of the image width
    'max_gap_fraction': 0.02,       # gap bridged inside a segment, share of the image width
    'rung_angle_deg': 20,           # a segment within this of horizontal is a rung
    'converging_angle_deg': 35,     # a segment at least this far from horizontal (and from vertical-ish noise) converges
    'rung_spacing': 0.03,           # two rungs closer than this (share of the image height) are the same line
    'converging_spacing': 0.04,     # the same for the crossing position (share of the image width)
    # Real slats are PARALLEL (rungs) or run to ONE vanishing point (converging). Lines at mixed angles (ramps, wedges,
    # wood grain, walls) are not slats: a family that fails its test counts as 0 lines.
    'min_rung_span': 0.35,                # a rung must run across this share of the image width (block edges are short)
    'max_rung_angle_std_deg': 5.0,        # rungs: spread of the line angles
    'max_vanishing_residual': 0.012,      # converging: a line meets the common point within this (share of the width)
}


def to_gray8(img: np.ndarray) -> np.ndarray:
    """Any single-channel image (uint8, uint16, float) -> uint8, stretched between its 2nd and 98th percentile."""
    g = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(g, [2, 98])
    return (np.clip((g - lo) / max(hi - lo, 1e-3), 0.0, 1.0) * 255).astype(np.uint8)


def _distinct(values, tol: float) -> int:
    groups = []
    for v in sorted(values):
        if groups and v - groups[-1] <= tol:
            groups[-1] = v
        else:
            groups.append(v)
    return len(groups)


def _groups(items, key, tol):
    """Group items whose key lies within tol of the group's FIRST item (no chaining, so a run of nearly touching lines
    is not merged into one)."""
    groups = []
    for it in sorted(items, key=key):
        if groups and key(it) - key(groups[-1][0]) <= tol:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def _vanishing_inliers(groups, tol_px: float):
    """The largest set of groups whose (longest) lines meet at ONE point, found by RANSAC over all pairs: noise lines
    (grain, cracks, shadows) that do not point at the common vanishing point are left out instead of ruining the count.
    Returns that list of groups (empty when fewer than 3 meet)."""
    lines = []
    for g in groups:
        x1, y1, x2, y2 = max(g, key=lambda t: t[2])[1]
        n = np.array([y2 - y1, -(x2 - x1)], float)
        n /= max(np.linalg.norm(n), 1e-9)
        lines.append((n, float(n @ np.array([x1, y1], float))))
    best = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            a = np.array([lines[i][0], lines[j][0]])
            if abs(np.linalg.det(a)) < 1e-3:                 # (nearly) parallel in the image: no vanishing point
                continue
            vp = np.linalg.solve(a, [lines[i][1], lines[j][1]])
            inl = [k for k, (n, c) in enumerate(lines) if abs(float(n @ vp) - c) <= tol_px]
            if len(inl) > len(best):
                best = inl
    return [groups[k] for k in best] if len(best) >= 3 else []


def orientation(result: Dict[str, int], angles) -> dict:
    """Which way the slats run in the image, from the family with more distinct lines.

    horizontal: rungs, the slats lie ACROSS the view (and across the robot's path); angle_deg = their mean angle
                from the image's horizontal (-20..20).
    vertical:   converging lines, the slats run AWAY from the camera (along the robot's path); angle_deg = the mean
                angle of the lines from the image's horizontal (35..145; 90 = straight ahead).
    Equal counts: horizontal. No lines: no orientation."""
    if result['rungs'] == 0 and result['converging'] == 0:
        return {'orientation': 'none', 'angle_deg': None}
    horizontal = result['rungs'] >= result['converging']
    values = angles['rung' if horizontal else 'converging']
    if horizontal:
        values = [a - 180 if a > 90 else a for a in values]               # -20..20 around horizontal
    return {'orientation': 'horizontal' if horizontal else 'vertical',
            'angle_deg': round(float(np.mean(values)), 1) if values else None}


def analyse(gray8: np.ndarray, params: dict = None):
    """Hough lines of one frame. Returns (result, segments): result as in detect(); segments is a list of
    (x1, y1, x2, y2, family) in full-image pixels, family 'rung' / 'converging' / 'other'."""
    p = dict(DEFAULTS, **(params or {}))
    h, w = gray8.shape
    r0 = int(h * p['roi_top_fraction'])
    g = cv2.GaussianBlur(gray8, (5, 5), 0)
    if p['canny_high'] > 0:
        lo, hi = int(p['canny_low']), int(p['canny_high'])
    else:
        med = float(np.median(g[r0:]))
        lo, hi = int(max(10, 0.66 * med)) // 2 + 10, int(min(250, 1.33 * med + 40)) // 2 + 30
    edges = cv2.Canny(g, lo, hi)[r0:, :]
    m = int(w * p['side_margin_fraction'])
    edges[:, :m] = 0
    edges[:, w - m:] = 0
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, int(p['hough_threshold']),
                            minLineLength=int(w * p['min_length_fraction']), maxLineGap=int(w * p['max_gap_fraction']))
    rungs, converging, segments = [], [], []        # rungs: (row, angle, length); converging: (x, line, length)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            ang = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            flat = min(ang, 180 - ang)
            length = math.hypot(x2 - x1, y2 - y1)
            family = 'other'
            if flat < p['rung_angle_deg']:
                rungs.append((((y1 + y2) / 2 + r0) / h, ang - 180 if ang > 90 else ang, length,
                              min(x1, x2) / w, max(x1, x2) / w))
                family = 'rung'
            elif 35 <= ang <= 145 and y2 != y1 and flat >= p['converging_angle_deg']:
                row = (h * 0.75 - r0)                                # where the line crosses a row near the bottom
                converging.append(((x1 + (row - y1) * (x2 - x1) / (y2 - y1)) / w,
                                   (x1, y1 + r0, x2, y2 + r0), length, ang))
                family = 'converging'
            segments.append((int(x1), int(y1) + r0, int(x2), int(y2) + r0, family))
    rung_groups = _groups(rungs, lambda t: t[0], p['rung_spacing'])
    # slat rungs are long bars across the view; the edges of blocks, steps and ramps are short pieces
    rung_groups = [g for g in rung_groups
                   if max(t[4] for t in g) - min(t[3] for t in g) >= p['min_rung_span']] if rung_groups else []
    conv_groups = _groups(converging, lambda t: t[0], p['converging_spacing'])
    raw = {'rungs_raw': len(rung_groups), 'converging_raw': len(conv_groups)}
    rung_angles = [float(np.average([t[1] for t in g], weights=[t[2] for t in g])) for g in rung_groups]
    n_rungs = len(rung_groups)
    if n_rungs >= 2 and float(np.std(rung_angles)) > p['max_rung_angle_std_deg']:
        n_rungs, rung_angles = 0, []                                  # not parallel: no rungs
    conv_ok = _vanishing_inliers(conv_groups, p['max_vanishing_residual'] * w) if len(conv_groups) >= 3 else []
    n_conv = len(conv_ok)                                            # only the lines that meet at one point count
    conv_angles = [float(np.average([t[3] for t in g], weights=[t[2] for t in g])) for g in conv_ok]
    result = {'rungs': n_rungs, 'converging': n_conv, 'segments': len(segments), **raw}
    result.update(orientation(result, {'rung': rung_angles if n_rungs else [], 'converging': conv_angles}))
    return result, segments


def detect(gray8: np.ndarray, params: dict = None) -> Dict[str, int]:
    return analyse(gray8, params)[0]


FAMILY_COLORS = {'rung': (0, 200, 0), 'converging': (0, 140, 255), 'other': (160, 160, 160)}      # BGR


def annotate(gray8: np.ndarray, result: Dict[str, int], segments, params: dict = None, min_lines: int = 3):
    """BGR debug image: the frame, the searched region, every Hough segment (green = rung, orange = converging,
    grey = ignored) and the counts (accepted/raw: lines that failed the parallel / vanishing-point test count 0) and the
    verdict."""
    p = dict(DEFAULTS, **(params or {}))
    h, w = gray8.shape
    img = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    r0 = int(h * p['roi_top_fraction'])
    cv2.line(img, (0, r0), (w - 1, r0), (255, 200, 0), 1)
    for x1, y1, x2, y2, family in segments:
        cv2.line(img, (x1, y1), (x2, y2), FAMILY_COLORS[family], 2)
    verdict = is_slats(result, min_lines)
    cv2.rectangle(img, (0, 0), (w - 1, 22), (0, 0, 0), -1)
    cv2.putText(img, f"rungs {result['rungs']}/{result['rungs_raw']}  converging {result['converging']}/{result['converging_raw']}"
                     f"  {result['orientation']}  -> {'SLATS' if verdict else 'no slats'}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255) if verdict else (255, 255, 255), 1, cv2.LINE_AA)
    return img


def is_slats(result: Dict[str, int], min_lines=3) -> bool:
    """min_lines: one number for both families, or {'rungs': n, 'converging': n}."""
    if isinstance(min_lines, dict):
        return result['rungs'] >= min_lines.get('rungs', 3) or result['converging'] >= min_lines.get('converging', 3)
    return result['rungs'] >= min_lines or result['converging'] >= min_lines
