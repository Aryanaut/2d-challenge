"""Curtain check from the front colour camera + depth camera (no ROS, numpy only).

The arena curtain is a pale (white / cream) fabric hanging above a low wooden ledge. The lidar and the depth hem test
cannot tell it from a wall when there is no gap under it. The camera can, with all of these together:

    upright    the depth surface stands within ~20 deg of vertical (not the floor, not the top of a platform)
    high       0.12-1.2 m above the floor (not the ledge / the floor / skirting rails)
    smooth     almost no texture in the colour image (OSB wood grain is 3-6x rougher)
    pale       low colour saturation (OSB is yellow, S 0.25-0.37; the curtain S 0.04-0.22)
    bright     at least 0.8x as bright as the floor IN THE SAME IMAGE (the grey arena walls are ~0.6x; the camera's
               auto exposure changes absolute brightness from frame to frame, the ratio stays)

and enough of such pixels: min_area_m2 square metres (far away) or min_fraction of the image (close up).
Colour alone is NOT enough: the floor and the white roller have the same colour as the curtain.

Measured on 5 snapshots (24 Nov, tools/curtain_snapshot.py), area / share of the image:
    curtain 0.6 m  0.08 m^2 / 32 %      curtain 1.5 m  0.25 m^2 / 7 %
    floor          0       /  0 %      grey wall      0       / 0 %      OSB board  0.01 m^2 / 0.7 %

The floor is found in the depth image itself (lowest flat surface), so the camera height / small body tilts do not need
to be known; `fallback_camera_height_m` is used only when no floor is in view.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class ColourParams:
    stride: int = 4                     # use every n-th depth pixel
    min_range_m: float = 0.25
    max_range_m: float = 2.5
    upright_max: float = 0.35           # |normal . up| below this = upright surface (within ~20 deg of vertical)
    flat_min: float = 0.80              # |normal . up| above this = flat, floor-like surface (within ~37 deg)
    min_height_m: float = 0.12          # above the floor (the wooden ledge under the curtain is ~0.09 m)
    max_height_m: float = 1.20
    max_texture: float = 0.012          # local brightness std-dev (0..1); curtain ~0.005, OSB 0.016-0.033
    max_saturation: float = 0.30        # curtain 0.04-0.22, OSB 0.25-0.37
    min_rel_brightness: float = 0.80    # vs the floor's median brightness in the same image; grey wall ~0.6
    floor_tol_m: float = 0.04           # floor pixels used for the brightness reference
    default_floor_v: float = 0.50       # brightness reference when no floor is in view
    fallback_camera_height_m: float = 0.46   # standing Lite3 (base 0.39 m + camera 0.074 m); only if no floor in view
    texture_win: int = 9                # colour pixels, box size for the texture measure
    min_area_m2: float = 0.15           # curtain-like area needed (stray patches on the wall / wood were < 0.02)...
    min_fraction: float = 0.15          # ...OR this share of the image (close up only a strip of curtain is in view:
                                        #    0.6 m -> 0.08 m^2 but 32 %; stray patches were < 1 %)


@dataclass
class ColourDetection:
    points: np.ndarray                  # Nx3 curtain points in the DEPTH OPTICAL frame (x right, y down, z forward)
    area_m2: float                      # curtain-like area in view
    fraction: float                     # share of the valid depth pixels that look like curtain
    floor_v: float                      # brightness reference used (median floor brightness)
    ranges: np.ndarray                  # depth of each point, metres


def _box(img: np.ndarray, k: int) -> np.ndarray:
    """k x k box mean (same size, edges padded), numpy only."""
    r = k // 2
    p = np.pad(img, ((r + 1, r), (r + 1, r)), mode='edge').astype(np.float64)
    c = p.cumsum(0).cumsum(1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s / (k * k))[:img.shape[0], :img.shape[1]]


def colour_features(rgb: np.ndarray, win: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Brightness V (0..1), saturation S (0..1) and texture (local std-dev of grey, 0..1) of an HxWx3 RGB uint8 image."""
    f = np.asarray(rgb, np.float32) / 255.0
    mx, mn = f.max(-1), f.min(-1)
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    g = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    m = _box(g, win)
    tex = np.sqrt(np.maximum(_box(g * g, win) - m * m, 0.0))
    return v, s, tex


def colour_mask(depth_m: np.ndarray, rgb: np.ndarray, k_depth: Sequence[float], k_colour: Sequence[float],
                depth_to_colour: Tuple[np.ndarray, Sequence[float]], up: Sequence[float],
                p: ColourParams = ColourParams(), features=None) -> dict:
    """Per-pixel curtain test on the strided depth grid. Returns a dict with 'mask', 'points' (HxWx3, depth optical
    frame), 'valid', 'area_m2', 'fraction', 'floor_v', 'uc', 'vc' (colour pixel of each cell).

    k_depth / k_colour: fx, fy, cx, cy. depth_to_colour: (R, t) taking depth-optical points into the colour optical
    frame. up: unit 'up' direction expressed in the depth optical frame (from TF; body tilt of a few degrees is fine)."""
    s = max(1, int(p.stride))
    d = np.nan_to_num(np.asarray(depth_m, np.float32), nan=0.0)[::s, ::s]
    h, w = d.shape
    fx, fy, cx, cy = k_depth
    vv, uu = np.mgrid[0:h, 0:w]
    uu, vv = uu * s, vv * s
    P = np.stack([(uu - cx) * d / fx, (vv - cy) * d / fy, d], -1)
    valid = (d > p.min_range_m) & (d < p.max_range_m)

    # surface normals from neighbours two cells apart; invalid across depth jumps
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    nb = np.zeros_like(valid)
    nb[1:-1, 1:-1] = (valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1])
    jump = (np.abs(du[..., 2]) > 0.15 * d + 0.02) | (np.abs(dv[..., 2]) > 0.15 * d + 0.02)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    good_n = valid & nb & ~jump

    up = np.asarray(up, float)
    up = up / (np.linalg.norm(up) or 1.0)
    cosu = np.abs(n @ up)
    flat = good_n & (cosu > p.flat_min)
    # refine 'up' with the flat surfaces (handles a tilted body), then find the floor = lowest flat surface
    if flat.sum() > 50:
        nf = n[flat]
        nf = nf * np.sign(nf @ up)[:, None]
        up2 = nf.mean(0)
        up = up2 / (np.linalg.norm(up2) or 1.0)
        cosu = np.abs(n @ up)
        flat = good_n & (cosu > p.flat_min)
    height = P @ up                                      # relative to the camera, + = up
    if flat.sum() > 50:
        floor = float(np.percentile(height[flat], 10))
    else:
        floor = -p.fallback_camera_height_m
    hgt = height - floor

    # colour of each depth cell
    R, t = depth_to_colour
    C = P @ np.asarray(R, float).T + np.asarray(t, float)
    fxc, fyc, cxc, cyc = k_colour
    zc = np.maximum(C[..., 2], 1e-3)
    uc = np.round(fxc * C[..., 0] / zc + cxc).astype(int)
    vc = np.round(fyc * C[..., 1] / zc + cyc).astype(int)
    H, W = rgb.shape[:2]
    inimg = (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H) & (C[..., 2] > 0.05)
    ucc, vcc = np.clip(uc, 0, W - 1), np.clip(vc, 0, H - 1)
    V, S, T = features if features is not None else colour_features(rgb, p.texture_win)
    V, S, T = V[vcc, ucc], S[vcc, ucc], T[vcc, ucc]

    floor_px = flat & inimg & (np.abs(hgt) < p.floor_tol_m)
    floor_v = float(np.median(V[floor_px])) if floor_px.sum() > 50 else p.default_floor_v

    mask = (good_n & inimg & (cosu < p.upright_max) & (hgt > p.min_height_m) & (hgt < p.max_height_m)
            & (T < p.max_texture) & (S < p.max_saturation) & (V > p.min_rel_brightness * floor_v))
    cell_area = (s * d / fx) * (s * d / fy)              # m^2 one cell covers, facing the camera
    area = float(cell_area[mask].sum())
    nvalid = int(valid.sum())
    return {'mask': mask, 'points': P, 'valid': valid, 'area_m2': area,
            'fraction': float(mask.sum()) / max(1, nvalid), 'floor_v': floor_v, 'uc': uc, 'vc': vc, 'depth': d}


def detect_curtain_colour(depth_m: np.ndarray, rgb: np.ndarray, k_depth: Sequence[float], k_colour: Sequence[float],
                          depth_to_colour: Tuple[np.ndarray, Sequence[float]], up: Sequence[float],
                          p: ColourParams = ColourParams()) -> Optional[ColourDetection]:
    """Curtain seen by the colour + depth cameras, or None (see the module doc)."""
    r = colour_mask(depth_m, rgb, k_depth, k_colour, depth_to_colour, up, p)
    if r['area_m2'] < p.min_area_m2 and r['fraction'] < p.min_fraction:
        return None
    m = r['mask']
    return ColourDetection(points=r['points'][m], area_m2=r['area_m2'], fraction=r['fraction'],
                           floor_v=r['floor_v'], ranges=r['depth'][m])


def one_per_cell(points_xy: np.ndarray, cell_m: float) -> np.ndarray:
    """One point (the mean) per cell_m grid cell, so one image counts once per spot in a CurtainMap."""
    p = np.asarray(points_xy, float).reshape(-1, 2)
    if len(p) == 0:
        return p
    keys, inv = np.unique(np.floor(p / cell_m).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    out = np.zeros((len(keys), 2))
    np.add.at(out, inv, p)
    return out / np.bincount(inv)[:, None]
