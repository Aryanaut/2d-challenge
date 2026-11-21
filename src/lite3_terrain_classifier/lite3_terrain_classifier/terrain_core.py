"""Terrain roughness maths, free of ROS so it can be unit tested.

The same pipeline is used for the Livox lidar and for the depth camera:

1. Fit the dominant ground plane with RANSAC, then refine it with SVD on the inliers.
2. Measure every point's height above that plane (the residual).
3. Bin the points into square cells on the plane (cell_size_m).
4. A cell is *rugged* when its median height is more than step_height_m above or below
   the plane (a step, block or hole), or when the heights inside it spread more than
   cell_spread_m (a bumpy surface). A cell that reaches higher than obstacle_height_m is a
   wall, box or person, not terrain, and is ignored.
5. The terrain is rugged when at least min_rugged_cells cells AND at least
   min_rugged_fraction of all cells are rugged (or, optionally, when the plane is tilted
   more than max_tilt_deg).

Residuals measured against a best-fit plane do not depend on the frame the points are in,
so the camera check also works when there is no TF from base_link to the camera.
"""
from dataclasses import asdict, dataclass
import math
from typing import Optional, Tuple

import numpy as np

RUGGED = 'rugged'
NON_RUGGED = 'non_rugged'
UNKNOWN = 'unknown'


@dataclass
class RoughnessParams:
    cell_size_m: float = 0.20
    min_points_per_cell: int = 6
    step_height_m: float = 0.05
    cell_spread_m: float = 0.08          # 0 disables the spread test
    obstacle_height_m: float = 0.35
    min_rugged_cells: int = 4
    min_rugged_fraction: float = 0.02
    max_tilt_deg: float = 0.0            # 0 disables; needs points in base_link
    min_points: int = 200
    min_cells: int = 8
    max_points: int = 6000               # random subsample above this
    ransac_iterations: int = 60
    ransac_tolerance_m: float = 0.03
    max_plane_tilt_deg: float = 35.0     # RANSAC only accepts planes this close to level (base_link only)


@dataclass
class TerrainAssessment:
    valid: bool
    rugged: bool = False
    reason: str = ''
    n_points: int = 0
    n_cells: int = 0
    rugged_cells: int = 0
    obstacle_cells: int = 0
    rugged_fraction: float = 0.0
    max_step_m: float = 0.0
    rms_m: float = 0.0
    tilt_deg: Optional[float] = None
    score: float = 0.0                   # >= 1.0 means rugged

    @property
    def label(self) -> str:
        if not self.valid:
            return UNKNOWN
        return RUGGED if self.rugged else NON_RUGGED

    def to_dict(self) -> dict:
        d = asdict(self)
        d['label'] = self.label
        for key, value in d.items():
            if isinstance(value, float):
                d[key] = round(value, 4)
        return d


def invalid(reason: str, n_points: int = 0) -> TerrainAssessment:
    return TerrainAssessment(valid=False, reason=reason, n_points=n_points)


# --------------------------------------------------------------------------- geometry

def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_points(points: np.ndarray, rotation: np.ndarray, translation) -> np.ndarray:
    return points @ rotation.T + np.asarray(translation, dtype=np.float64)


def crop_box(points: np.ndarray, x_range, y_range, z_range,
             exclude_x=None, exclude_y=None) -> np.ndarray:
    """Keep points inside the box, minus an optional body footprint box."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    keep = ((x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
            & (z >= z_range[0]) & (z <= z_range[1]))
    if exclude_x is not None and exclude_y is not None:
        body = ((x >= exclude_x[0]) & (x <= exclude_x[1])
                & (y >= exclude_y[0]) & (y <= exclude_y[1]))
        keep &= ~body
    return points[keep]


def depth_to_points(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                    row_start: int, row_stop: int, col_start: int, col_stop: int,
                    stride: int, min_range: float, max_range: float) -> np.ndarray:
    """Back-project a window of a metric depth image into the camera optical frame."""
    stride = max(1, int(stride))
    window = depth_m[row_start:row_stop:stride, col_start:col_stop:stride]
    vv, uu = np.mgrid[row_start:row_stop:stride, col_start:col_stop:stride]
    vv, uu = vv[:window.shape[0], :window.shape[1]], uu[:window.shape[0], :window.shape[1]]
    ok = np.isfinite(window) & (window > min_range) & (window < max_range)
    z = window[ok].astype(np.float64)
    x = (uu[ok] - cx) * z / fx
    y = (vv[ok] - cy) * z / fy
    return np.column_stack((x, y, z))


# --------------------------------------------------------------------------- plane fit

def _orient(normal: np.ndarray, d: float) -> Tuple[np.ndarray, float]:
    """Make the sensor origin (0,0,0) lie on the positive side: positive residual = up."""
    return (normal, d) if d >= 0 else (-normal, -d)


def _svd_plane(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    return _orient(normal, -float(normal @ centroid))


def fit_plane(points: np.ndarray, iterations: int, tolerance: float,
              up: Optional[np.ndarray], max_tilt_deg: float,
              rng: np.random.Generator) -> Optional[Tuple[np.ndarray, float]]:
    """RANSAC + SVD refinement. Returns (unit normal, d) with n.p + d = height above plane."""
    n = len(points)
    if n < 3:
        return None
    idx = rng.integers(0, n, size=(iterations, 3))
    p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-9
    if up is not None:
        cos_tilt = np.abs(normals @ up) / np.where(ok, lengths, 1.0)
        ok &= cos_tilt >= math.cos(math.radians(max_tilt_deg))
    if not ok.any():
        return _svd_plane(points)
    normals = normals[ok] / lengths[ok, None]
    ds = -np.einsum('ij,ij->i', normals, p0[ok])
    dist = np.abs(points @ normals.T + ds)
    best = int(np.argmax((dist < tolerance).sum(axis=0)))
    inliers = points[dist[:, best] < tolerance]
    if len(inliers) < 3:
        return _orient(normals[best], float(ds[best]))
    normal, d = _svd_plane(inliers)
    # One more pass with a band that adapts to the sensor noise.
    res = np.abs(points @ normal + d)
    near = res[res < 3 * tolerance]
    mad = float(np.median(near)) if len(near) >= 3 else tolerance
    band = max(tolerance, 3.0 * 1.4826 * mad)
    refine = points[res < band]
    if len(refine) >= 3:
        normal, d = _svd_plane(refine)
    return normal, d


def _plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0])
    if abs(normal @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - (ref @ normal) * normal
    u /= np.linalg.norm(u)
    return u, np.cross(normal, u)


# --------------------------------------------------------------------------- assessment

def assess_terrain(points: np.ndarray, params: RoughnessParams, up_known: bool,
                   rng: Optional[np.random.Generator] = None):
    """Classify a patch of terrain.

    points: (N, 3). When up_known is True the points are in base_link (z up), which
    enables the RANSAC level constraint and the tilt test.
    Returns (TerrainAssessment, points_used, residuals) - the last two for debugging.
    """
    empty = np.zeros((0, 3)), np.zeros(0)
    if rng is None:
        rng = np.random.default_rng(0)
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)] if len(points) else points.reshape(0, 3)
    if len(points) > params.max_points:
        points = points[rng.choice(len(points), params.max_points, replace=False)]
    if len(points) < params.min_points:
        return invalid(f'only {len(points)} points (need {params.min_points})', len(points)), *empty

    up = np.array([0.0, 0.0, 1.0]) if up_known else None
    plane = fit_plane(points, params.ransac_iterations, params.ransac_tolerance_m, up,
                      params.max_plane_tilt_deg, rng)
    if plane is None:
        return invalid('plane fit failed', len(points)), *empty
    normal, d = plane
    if up is not None and normal @ up < 0:
        normal, d = -normal, -d
    res = points @ normal + d

    # Deep negative residuals are reflections / noise, not terrain.
    keep = res > -params.obstacle_height_m
    points, res = points[keep], res[keep]

    u, v = _plane_basis(normal)
    cs = params.cell_size_m
    cells = np.column_stack((np.floor(points @ u / cs), np.floor(points @ v / cs))).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind='stable')
    splits = np.cumsum(counts)[:-1]

    n_cells = rugged_cells = obstacle_cells = 0
    max_step = 0.0
    terrain_mask = np.zeros(len(points), dtype=bool)
    for cell_idx, members in enumerate(np.split(order, splits)):
        if counts[cell_idx] < params.min_points_per_cell:
            continue
        r = res[members]
        p10, p50, p90, p95 = np.percentile(r, (10, 50, 90, 95))
        if p95 > params.obstacle_height_m:
            obstacle_cells += 1
            continue
        n_cells += 1
        terrain_mask[members] = True
        max_step = max(max_step, abs(float(p50)))
        if abs(p50) > params.step_height_m or (
                params.cell_spread_m > 0 and (p90 - p10) > params.cell_spread_m):
            rugged_cells += 1

    tilt = None
    if up is not None:
        tilt = math.degrees(math.acos(min(1.0, abs(float(normal @ up)))))
    rms = float(np.sqrt(np.mean(res[terrain_mask] ** 2))) if terrain_mask.any() else 0.0
    out = TerrainAssessment(valid=True, n_points=int(len(points)), n_cells=n_cells,
                            rugged_cells=rugged_cells, obstacle_cells=obstacle_cells,
                            max_step_m=max_step, rms_m=rms, tilt_deg=tilt)
    if n_cells < params.min_cells:
        out.valid = False
        out.reason = f'only {n_cells} terrain cells (need {params.min_cells})'
        return out, points, res

    out.rugged_fraction = rugged_cells / n_cells
    cell_score = min(rugged_cells / max(params.min_rugged_cells, 1),
                     out.rugged_fraction / params.min_rugged_fraction
                     if params.min_rugged_fraction > 0 else float('inf'))
    tilt_score = tilt / params.max_tilt_deg if (tilt is not None and params.max_tilt_deg > 0) else 0.0
    out.score = float(max(cell_score, tilt_score))
    out.rugged = out.score >= 1.0
    if out.rugged:
        out.reason = 'tilted plane' if tilt_score >= 1.0 and cell_score < 1.0 else 'uneven cells'
    else:
        out.reason = 'flat'
    return out, points, res


# --------------------------------------------------------------------------- fusion

FUSION_MODES = ('lidar_primary', 'require_agreement', 'any_rugged')


@dataclass
class FusionResult:
    label: Optional[str]      # None = hold the current state
    source: str
    verified: bool
    agree: Optional[bool]


def fuse(lidar: Optional[TerrainAssessment], depth: Optional[TerrainAssessment],
         mode: str = 'lidar_primary', depth_fallback: bool = True) -> FusionResult:
    """Lidar is the primary sensor; the depth camera verifies it.

    lidar_primary      lidar decides; `verified` says whether the camera agreed
    require_agreement  the state only changes when both sensors agree
    any_rugged         rugged if either sensor says rugged (most cautious)
    """
    if mode not in FUSION_MODES:
        raise ValueError(f'fusion_mode must be one of {FUSION_MODES}')
    lv = lidar is not None and lidar.valid
    dv = depth is not None and depth.valid
    if lv and dv:
        agree = lidar.rugged == depth.rugged
        if agree:
            return FusionResult(lidar.label, 'lidar+depth', True, True)
        if mode == 'require_agreement':
            return FusionResult(None, 'lidar+depth', False, False)
        if mode == 'any_rugged':
            return FusionResult(RUGGED, 'lidar+depth', False, False)
        return FusionResult(lidar.label, 'lidar', False, False)
    if lv:
        return FusionResult(lidar.label, 'lidar', False, None)
    if dv and depth_fallback:
        return FusionResult(depth.label, 'depth', False, None)
    return FusionResult(UNKNOWN, 'none', False, None)


def vote(fused: FusionResult, state: str, flat_needs_lidar: bool = True) -> Optional[str]:
    """The label to feed the debouncer for this tick (None = no vote).

    Going from rugged to flat is the risky direction (the flat gait on rough ground). When the lidar had too few cells and
    the depth camera alone (unverified, `source == 'depth'`) says flat, that is not enough to leave RUGGED: no vote, so
    the state stays rugged until the lidar sees the floor again."""
    if flat_needs_lidar and state == RUGGED and fused.label == NON_RUGGED and fused.source == 'depth':
        return None
    return fused.label


class Debouncer:
    """Only switch state after the same new label has been seen `confirm` times in a row."""

    def __init__(self, confirm: int = 3, initial: str = UNKNOWN):
        self.confirm = max(1, int(confirm))
        self.state = initial
        self._pending = None
        self._count = 0

    def update(self, label: Optional[str]) -> str:
        if label is None or label == self.state:
            self._pending, self._count = None, 0
            return self.state
        if label == self._pending:
            self._count += 1
        else:
            self._pending, self._count = label, 1
        if self._count >= self.confirm:
            self.state, self._pending, self._count = label, None, 0
        return self.state

    def reset(self, state: str = UNKNOWN):
        self.state, self._pending, self._count = state, None, 0
