"""Feature vector for the trained surface classifier (OpenCV + scikit-image, CPU only).

One image -> ~110 numbers describing colour and texture of the centre crop:
  colour   : mean / std / 10-50-90th percentiles of L*, a*, b*, S, V  + a*/b* histograms
  texture  : Local Binary Pattern histograms at two scales (fine flakes vs. diamond pattern)
             grey-level co-occurrence (GLCM) contrast / homogeneity / energy / correlation
             gradient magnitude statistics, Laplacian energy
  lighting : share of near-white (specular) and near-black pixels

Bump FEATURE_VERSION whenever this changes, so old models are refused instead of silently wrong.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

FEATURE_VERSION = 1
SIZE = (160, 120)


def centre_crop(bgr: np.ndarray, roi_frac: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    rw, rh = int(w * roi_frac), int(h * roi_frac)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    return bgr[y0:y0 + rh, x0:x0 + rw]


def _hist(values, bins, lo, hi):
    h, _ = np.histogram(values, bins=bins, range=(lo, hi))
    return h.astype(np.float32) / max(1, values.size)


def feature_vector(bgr: np.ndarray, roi_frac: float = 0.6) -> np.ndarray:
    roi = cv2.resize(centre_crop(bgr, roi_frac), SIZE, interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    L, a, b = lab[..., 0].ravel(), lab[..., 1].ravel() - 128, lab[..., 2].ravel() - 128
    S, V = hsv[..., 1].ravel(), hsv[..., 2].ravel()

    feats = []
    for ch in (L, a, b, S, V):
        feats += [ch.mean(), ch.std(), *np.percentile(ch, (10, 50, 90))]
    feats = [np.float32(feats)]
    feats.append(_hist(a, 16, -40, 40))
    feats.append(_hist(b, 16, -40, 40))

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        feats.append(_hist(lbp.ravel(), P + 2, 0, P + 2))

    q = (gray // 8).astype(np.uint8)                       # 32 grey levels
    glcm = graycomatrix(q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        feats.append(graycoprops(glcm, prop).mean(axis=1).astype(np.float32))   # per distance

    g = gray.astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    lap = cv2.Laplacian(g, cv2.CV_32F)
    feats.append(np.float32([mag.mean(), mag.std(), np.percentile(mag, 90),
                             np.log1p(lap.var()), (L > 200).mean(), (L < 40).mean()]))

    out = np.concatenate(feats).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------- augmentation (training only) ----------
def augment(bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random but realistic variation: crop/zoom (camera height), rotation/flip (robot
    heading), brightness/contrast/gamma and white balance (lighting, shadow), blur (walking)."""
    img = bgr
    h, w = img.shape[:2]
    s = rng.uniform(0.6, 1.0)
    ch, cw = int(h * s), int(w * s)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    img = img[y0:y0 + ch, x0:x0 + cw]
    img = np.rot90(img, int(rng.integers(0, 4)))
    if rng.random() < 0.5:
        img = img[:, ::-1]
    img = np.ascontiguousarray(img).astype(np.float32)
    img *= rng.uniform(0.92, 1.08, size=3)                      # white balance
    img = img * rng.uniform(0.55, 1.35) + rng.uniform(-20, 20)  # exposure / offset
    img = np.clip(img, 0, 255)
    img = 255.0 * (img / 255.0) ** rng.uniform(0.7, 1.4)        # gamma
    img = img.astype(np.uint8)
    k = int(rng.choice([0, 0, 3, 5, 7]))
    if k:
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:                                      # motion blur along a random axis
        n = int(rng.integers(5, 13))
        kern = np.zeros((n, n), np.float32)
        kern[n // 2, :] = 1.0 / n
        M = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        img = cv2.filter2D(img, -1, cv2.warpAffine(kern, M, (n, n)))
    return img
