"""Fast, local (no network) surface classifier for the downward-facing camera.

Runs on every frame in a few milliseconds. It separates the three course surfaces using
colour and texture measured on a centre crop of the image:

  - other (lab floor):  almost no texture -> low brightness spread (L std) and low gradient
  - metal (stair plate): no colour (low chroma), strong texture and shiny highlights
  - wood (OSB):          warm colour (positive a*/b* in Lab), medium texture

The default thresholds were set from phone photos of the arena. Re-tune them with
`ros2 run surface_monitor surface_tune --camera /dev/video0` using the real Logitech
webcam, its mounting height and the robot's shadow; texture numbers depend on distance.
"""

from dataclasses import dataclass, asdict

import cv2
import numpy as np

LABELS = ("wood", "metal", "other")


@dataclass
class Thresholds:
    roi_frac: float = 0.6          # use the centre 60% of the frame (avoid legs / vignetting)
    dark_l_max: float = 35.0       # mean L* below this -> too dark to judge ("unclear")
    other_lstd_max: float = 10.0   # lab floor: brightness spread below this ...
    other_grad_max: float = 25.0   # ... and mean gradient below this
    metal_chroma_max: float = 9.0  # metal: almost colourless ...
    metal_grad_strong: float = 60.0  # ... and very strong edges (diamond pattern), or
    metal_grad_min: float = 45.0   # strong edges plus either
    metal_lstd_min: float = 18.0   #   a large brightness spread
    metal_bright_min: float = 0.01 #   or shiny near-white highlights
    wood_b_min: float = 2.0        # wood is yellowish: b* above this
    wood_a_min: float = -1.0       # and not greenish/bluish

    def as_dict(self):
        return asdict(self)


def extract_features(bgr: np.ndarray, roi_frac: float = 0.6) -> dict:
    """Colour/texture features of the centre crop, resized to 160x120 so numbers are
    comparable between cameras and resolutions."""
    h, w = bgr.shape[:2]
    rw, rh = int(w * roi_frac), int(h * roi_frac)
    x0, y0 = (w - rw) // 2, (h - rh) // 2
    roi = bgr[y0:y0 + rh, x0:x0 + rw]
    roi = cv2.resize(roi, (160, 120), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)

    return {
        "L": float(L.mean()),
        "Lstd": float(L.std()),
        "a": float(a.mean()),
        "b": float(b.mean()),
        "chroma": float(np.hypot(a, b).mean()),
        "grad": float(np.hypot(gx, gy).mean()),
        "bright": float((L > 200).mean()),
    }


def classify(f: dict, th: Thresholds) -> str:
    """Return 'wood', 'metal', 'other' or 'unclear' for one frame's features."""
    if f["L"] < th.dark_l_max:
        return "unclear"
    if f["chroma"] < th.metal_chroma_max and (
            f["grad"] > th.metal_grad_strong
            or (f["grad"] > th.metal_grad_min
                and (f["Lstd"] > th.metal_lstd_min or f["bright"] > th.metal_bright_min))):
        return "metal"
    if f["Lstd"] < th.other_lstd_max and f["grad"] < th.other_grad_max:
        return "other"
    if f["b"] >= th.wood_b_min and f["a"] >= th.wood_a_min:
        return "wood"
    return "other"


def classify_image(bgr: np.ndarray, th: Thresholds):
    f = extract_features(bgr, th.roi_frac)
    return classify(f, th), f
