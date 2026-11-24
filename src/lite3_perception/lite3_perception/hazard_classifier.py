"""
Hazard warning-sign finder + classifier (pure OpenCV/NumPy, no training, no PyTorch).

1. find yellow triangular blobs (HSV threshold + polygon approximation)
2. warp each triangle to a canonical upright triangle (handles rotation / mild perspective)
3. binarise the dark symbol inside, compare against reference templates
   (one PNG per hazard in a folder, filename = label) with normalised correlation,
   trying a few scales so signs with or without an outer yellow rim both match.

Templates: hazard_templates/<label>.png, one sign per image (any picture with the yellow triangle in it).
Standard labels: high_voltage, radioactive, chemical, biological, flammable.
Unused signs are kept in hazard_templates/extra/ (not loaded).
"""
import os

import cv2
import numpy as np

CANON = 96                                    # canonical triangle canvas (px)
DESC = 32                                     # descriptor size
_TRI = np.float32([[CANON / 2, 0], [0, CANON - 1], [CANON - 1, CANON - 1]])
_CENTROID = _TRI.mean(0)
INNER_SHRINK = 0.80                           # inner region (inside the black border) relative to outer triangle
SCALES = tuple(np.round(np.arange(1.0, 0.70, -0.03), 2))  # used only for the "is this really a sign" border check
MATCH_SCALES = (1.04, 1.0, 0.96)              # templates are the real arena signs -> compare at ~the same scale
MIN_BORDER_FIT = 30.0                         # real signs have a dark border (~70-115); yellow floor/wood ~5
BORDER_BAND = (0.86, 0.78)                    # where the black border sits in a template (fraction of outer size)
INSIDE_BAND = (0.73, 0.67)                    # yellow just inside the border


def _shrink(tri, k, c=None):
    c = tri.mean(0) if c is None else c
    return (c + (tri - c) * k).astype(np.float32)


def _tri_mask(k):
    m = np.zeros((CANON, CANON), np.uint8)
    cv2.fillConvexPoly(m, _shrink(_TRI, k, _CENTROID).astype(np.int32), 255)
    return m


_INNER_MASK = _tri_mask(INNER_SHRINK)
_BORDER_RING = (_tri_mask(BORDER_BAND[0]) > 0) & (_tri_mask(BORDER_BAND[1]) == 0)
_DESC_SEL = cv2.resize(_INNER_MASK, (DESC, DESC), interpolation=cv2.INTER_NEAREST).ravel() > 0
BLUR_SIGMA = 1.5
_INSIDE_RING = (_tri_mask(INSIDE_BAND[0]) > 0) & (_tri_mask(INSIDE_BAND[1]) == 0)


def order_triangle(pts):
    """-> float32 [apex, base_left, base_right] (apex = vertex farthest from the opposite side's midpoint, upward)."""
    pts = np.float32(pts).reshape(3, 2)
    apex_i = int(np.argmin(pts[:, 1]))        # topmost vertex = apex (signs are mounted upright)
    rest = [pts[i] for i in range(3) if i != apex_i]
    rest.sort(key=lambda p: p[0])
    return np.float32([pts[apex_i], rest[0], rest[1]])


def find_triangles(bgr, min_area=400, hsv_lo=(15, 80, 100), hsv_hi=(40, 255, 255)):
    """Return list of (tri[3x2] ordered, bbox xyxy) for yellow triangles."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, hsv_lo, hsv_hi)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        hull = cv2.convexHull(c)
        approx = cv2.approxPolyDP(hull, 0.06 * cv2.arcLength(hull, True), True)
        if len(approx) != 3:
            continue
        tri_area = cv2.contourArea(approx)
        if tri_area <= 0 or area / tri_area < 0.6:          # must actually fill its triangle
            continue
        tri = order_triangle(approx)
        sides = np.linalg.norm(tri - np.roll(tri, 1, 0), axis=1)
        if sides.min() / sides.max() < 0.55:                  # roughly equilateral
            continue
        x, y, w, h = cv2.boundingRect(approx)
        out.append((tri, [x, y, x + w, y + h]))
    return out


def _canonical(bgr, tri, scale):
    outer = _shrink(tri, 1.0 / scale)                         # undo assumed scale about centroid
    M = cv2.getAffineTransform(outer, _TRI)
    return cv2.warpAffine(bgr, M, (CANON, CANON), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


def _descriptor(canon_bgr):
    g = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY)
    vals = g[_INNER_MASK > 0]
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if vals.max() - vals.min() < 40:                          # no contrast -> empty symbol
        thr = -1
    dark = (g < thr).astype(np.float32)
    dark = cv2.GaussianBlur(dark, (0, 0), BLUR_SIGMA)
    d = cv2.resize(dark, (DESC, DESC), interpolation=cv2.INTER_AREA).ravel()[_DESC_SEL]   # inside-sign pixels only
    d = d - d.mean()
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else d


def _augment(img, rng):
    """Render a template the way the camera might see it: smaller, tilted, blurred, noisy, JPEG'd."""
    h = rng.uniform(40, 150)
    s = h / img.shape[0]
    t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pad = 12
    t = cv2.copyMakeBorder(t, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    hh, ww = t.shape[:2]
    src = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
    j = rng.uniform(-0.08, 0.08, (4, 2)).astype(np.float32) * np.float32([ww, hh])
    t = cv2.warpPerspective(t, cv2.getPerspectiveTransform(src, src + j), (ww, hh),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    t = cv2.GaussianBlur(t, (0, 0), rng.uniform(0.3, 1.5))
    t = np.clip(t.astype(np.float32) * rng.uniform(0.7, 1.2) + rng.normal(0, 4, t.shape), 0, 255).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", t, [cv2.IMWRITE_JPEG_QUALITY, 75])[1], cv2.IMREAD_COLOR)


class HazardClassifier:
    def __init__(self, template_dir, n_aug=40, seed=0):
        """Each template is expanded into n_aug synthetic views; a sign's score for a class
        is its best correlation with any view of that class."""
        rng = np.random.default_rng(seed)
        self.labels, descs, owner = [], [], []
        for f in sorted(os.listdir(template_dir)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(template_dir, f))
            k = len(self.labels)
            views = [img] + [_augment(img, rng) for _ in range(n_aug)]
            got = 0
            for v in views:
                tris = find_triangles(v, min_area=100)
                if not tris:
                    continue
                tri = max(tris, key=lambda t: cv2.contourArea(t[0]))[0]
                descs.append(_descriptor(_canonical(v, tri, 1.0)))
                owner.append(k)
                got += 1
            if got:
                self.labels.append(os.path.splitext(f)[0])
        if not self.labels:
            raise RuntimeError(f"no hazard templates found in {template_dir}")
        self.T = np.stack(descs)                              # (n_views, D)
        self.owner = np.array(owner)

    def _class_scores(self, desc):
        sc = self.T @ desc
        out = np.full(len(self.labels), -1.0)
        np.maximum.at(out, self.owner, sc)
        return out

    @staticmethod
    def _border_fit(canon_bgr):
        """How well the black border lines up with where templates have it (higher = better)."""
        g = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(g[_INSIDE_RING].mean() - g[_BORDER_RING].mean())

    def is_sign(self, bgr, tri):
        """Reject yellow triangles that are not warning signs (no dark border anywhere near the edge)."""
        return max(self._border_fit(_canonical(bgr, tri, sc)) for sc in SCALES) >= MIN_BORDER_FIT

    def classify(self, bgr, tri):
        """-> (label, score, margin, top3[(label,score)])  score = correlation in [-1,1]."""
        best = np.full(len(self.labels), -1.0)
        for sc in MATCH_SCALES:
            best = np.maximum(best, self._class_scores(_descriptor(_canonical(bgr, tri, sc))))
        order = np.argsort(-best)
        top3 = [(self.labels[i], round(float(best[i]), 3)) for i in order[:3]]
        margin = float(best[order[0]] - best[order[1]]) if len(order) > 1 else 1.0
        return self.labels[order[0]], float(best[order[0]]), margin, top3

    def detect(self, bgr, min_area=400):
        """All hazard signs in a frame -> list of dicts."""
        res = []
        H, W = bgr.shape[:2]
        for tri, bbox in find_triangles(bgr, min_area):
            x1, y1, x2, y2 = bbox
            if x1 <= 2 or y1 <= 2 or x2 >= W - 3 or y2 >= H - 3:
                continue                      # sign cut off by the picture edge: wait for a full view
            if not self.is_sign(bgr, tri):
                continue
            label, score, margin, top3 = self.classify(bgr, tri)
            res.append({"label": label, "score": round(score, 3), "margin": round(margin, 3),
                        "top3": top3, "bbox_xyxy": bbox})
        return res


def build_templates(sheet_path, out_dir, labels):
    """Cut every sign out of a reference sheet (row-major order) into out_dir/<label>.png."""
    img = cv2.imread(sheet_path)
    tris = find_triangles(img, min_area=500)
    tris.sort(key=lambda t: (round(t[1][1] / 50), t[1][0]))
    if len(tris) != len(labels):
        raise ValueError(f"found {len(tris)} signs but {len(labels)} labels")
    os.makedirs(out_dir, exist_ok=True)
    for (tri, (x1, y1, x2, y2)), lab in zip(tris, labels):
        p = 6
        crop = img[max(0, y1 - p):y2 + p, max(0, x1 - p):x2 + p]
        cv2.imwrite(os.path.join(out_dir, f"{lab}.png"), crop)
