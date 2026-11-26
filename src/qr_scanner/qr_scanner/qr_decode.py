"""QR detection + decoding that copes with codes at different distances (pure OpenCV/zbar, no ROS).

Pipeline per frame:
  1. zbar on the full grayscale frame            - reads codes down to ~50 px wide (~25 ms at 1280x720)
  2. OpenCV's ArUco-based QR detector            - *finds* codes down to ~28 px, even when too small
                                                    or blurred to read (~15 ms)
  3. for every found-but-unread code: crop around it, upscale it to ~240 px, CLAHE, then
     zbar and OpenCV's decoder on the crop       - reads codes down to ~33 px when the image is sharp
  4. anything still unread is returned as a *candidate* so the robot can walk closer
Optional: every N frames, zbar on the whole frame upscaled 2x (slow, ~100 ms) as a safety net.

Measured on the arena photos (code size = width of the black square in pixels):
  >= 70 px  reads even with strong blur
  ~50 px    reads when sharp or lightly blurred (slow walking)
  ~35-40 px reads only when sharp (robot standing still)
  < 30 px   neither read nor reliably spotted -> get closer
Walking blur is the main limit, so pause briefly when facing a wall.
"""

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

try:
    from pyzbar import pyzbar
    _ZBAR_QR = [pyzbar.ZBarSymbol.QRCODE]
except ImportError:  # still works with OpenCV only, just less sensitive
    pyzbar = None
    _ZBAR_QR = None


@dataclass
class QrHit:
    text: str                    # decoded content ('' for a candidate that could not be read)
    corners: np.ndarray          # 4x2 float32, pixel coords in the original image
    method: str = ""
    decoded: bool = True

    @property
    def center(self):
        return self.corners.mean(axis=0)

    @property
    def side_px(self) -> float:
        c = self.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


def _order_quad(pts) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(pts) != 4:  # zbar can return more polygon points: use the bounding quad
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect)
    return pts.astype(np.float32)


def _zbar(gray, method, scale=1.0, offset=(0, 0)) -> List[QrHit]:
    if pyzbar is None:
        return []
    hits = []
    for d in pyzbar.decode(gray, symbols=_ZBAR_QR):
        try:
            text = d.data.decode("utf-8")
        except UnicodeDecodeError:
            text = d.data.decode("latin-1")
        pts = [(p.x, p.y) for p in d.polygon] or [
            (d.rect.left, d.rect.top), (d.rect.left + d.rect.width, d.rect.top),
            (d.rect.left + d.rect.width, d.rect.top + d.rect.height),
            (d.rect.left, d.rect.top + d.rect.height)]
        quad = _order_quad(pts) / scale + np.float32(offset)
        hits.append(QrHit(text, quad, method))
    return hits


def _cv_decode(gray, detector, method, scale=1.0, offset=(0, 0)) -> List[QrHit]:
    try:
        ok, texts, pts, _ = detector.detectAndDecodeMulti(gray)
    except cv2.error:
        return []
    if not ok or pts is None:
        return []
    hits = []
    for text, quad in zip(texts, pts):
        if text:
            hits.append(QrHit(text, _order_quad(quad) / scale + np.float32(offset), method))
    return hits


def _overlaps(quad, hits, margin=0.5) -> bool:
    c = quad.mean(axis=0)
    for h in hits:
        if np.linalg.norm(h.center - c) < max(10.0, margin * h.side_px + 0.5 * QrHit("", quad).side_px):
            return True
    return False


class QrReader:
    def __init__(self, roi_target_px: int = 240, full_upscale_every_n: int = 0,
                 use_clahe: bool = True):
        self.roi_target_px = roi_target_px
        self.full_upscale_every_n = full_upscale_every_n
        try:  # OpenCV >= 4.8: better at spotting small codes
            params = cv2.QRCodeDetectorAruco_Params()
            params.minModuleSizeInPyramid = 1.0      # default 4 px misses small / distant codes
            self.finder = cv2.QRCodeDetectorAruco(params)
            aruco = cv2.aruco.DetectorParameters()
            aruco.minMarkerPerimeterRate = 0.01      # allow small finder patterns
            self.finder.setArucoParameters(aruco)
        except AttributeError:  # older OpenCV (e.g. Ubuntu 22.04's 4.5): standard detector
            self.finder = cv2.QRCodeDetector()
        self.decoder = cv2.QRCodeDetector()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        self.frame_count = 0

    def read(self, bgr_or_gray) -> List[QrHit]:
        """Return decoded hits plus undecoded candidates (decoded=False)."""
        self.frame_count += 1
        gray = bgr_or_gray if bgr_or_gray.ndim == 2 else cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # 1. fast full-frame read
        hits = _zbar(gray, "zbar")
        if not hits and pyzbar is None:
            hits = _cv_decode(gray, self.decoder, "opencv")

        # 2. find every QR-looking thing, including ones too small to read
        try:
            ok, quads = self.finder.detectMulti(gray)
        except cv2.error:
            ok, quads = False, None
        found = [] if not ok or quads is None else [_order_quad(q) for q in quads]

        # 3. zoom into the unread ones
        candidates = []
        for quad in found:
            if _overlaps(quad, hits):
                continue
            hit = self._read_roi(gray, quad, W, H)
            if hit:
                hits.extend(h for h in hit if not _overlaps(h.corners, hits))
            else:
                candidates.append(QrHit("", quad, "detected", decoded=False))

        # optional slow safety net
        if (not hits and self.full_upscale_every_n > 0
                and self.frame_count % self.full_upscale_every_n == 0):
            up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            for h in _zbar(up, "zbar_full_x2", scale=2.0):
                if not _overlaps(h.corners, hits):
                    hits.append(h)
            candidates = [c for c in candidates if not _overlaps(c.corners, hits)]

        return hits + candidates

    def _read_roi(self, gray, quad, W, H):
        side = QrHit("", quad).side_px
        cx, cy = quad.mean(axis=0)
        half = 0.5 * side * 1.8 + 8          # QR + quiet zone + margin
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(W, cx + half)), int(min(H, cy + half))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        roi = gray[y0:y1, x0:x1]
        scale = float(np.clip(self.roi_target_px / max(side, 1.0), 1.0, 8.0))
        up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants = [("roi_zoom", up)]
        if self.clahe is not None:
            variants.append(("roi_zoom_clahe", self.clahe.apply(up)))
        for name, img in variants:
            hit = _zbar(img, "zbar_" + name, scale=scale, offset=(x0, y0))
            if not hit:
                hit = _cv_decode(img, self.decoder, "opencv_" + name, scale=scale, offset=(x0, y0))
            if hit:
                return hit
        return []


def draw(bgr, hits: List[QrHit]):
    out = bgr.copy()
    for h in hits:
        color = (0, 200, 0) if h.decoded else (0, 165, 255)
        cv2.polylines(out, [h.corners.astype(np.int32)], True, color, 2)
        label = h.text if h.decoded else "QR? (move closer)"
        x, y = h.corners.min(axis=0).astype(int)
        cv2.putText(out, f"{label} [{h.side_px:.0f}px]", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out
