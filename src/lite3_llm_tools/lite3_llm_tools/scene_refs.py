"""Scene reference photos for the describe_scene task (no ROS).

The AI server takes one image per request, so the live camera frame and the reference photos go up together as one
collage: the live frame at the top left (green border, "LIVE"), the references as labelled thumbnails around it.

Reference files are named <scene_type>_<n>.jpg (kitchen_1.jpg, cleaning_room_2.jpg, ...): the scene type is the file
name without the trailing _<n>. Only these scene types can be answered (see prompts.parse_scene_type).
"""
import os
import re
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
_NUMBERED = re.compile(r'^(.*?)(?:[_-]\d+)?$')

# layout of the 1024 x 768 collage
LIVE_W, LIVE_H = 640, 480
THUMB_W, THUMB_H = 192, 144
LIVE_COLOUR = (0, 200, 0)       # BGR


def label_of(filename: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    return _NUMBERED.match(stem).group(1)


def load_references(folder: str) -> List[Tuple[str, np.ndarray]]:
    """[(scene_type, bgr image)] for every readable image in `folder`, sorted by file name."""
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'scene reference folder {folder} does not exist')
    refs = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        img = cv2.imread(os.path.join(folder, name), cv2.IMREAD_COLOR)
        label = label_of(name)
        if img is not None and re.fullmatch(r'[a-z][a-z0-9_]*', label):
            refs.append((label, img))
    if not refs:
        raise FileNotFoundError(f'no scene reference images (<scene_type>_<n>.jpg) in {folder}')
    return refs


def scene_types(refs: Sequence[Tuple[str, np.ndarray]]) -> List[str]:
    return sorted({label for label, _ in refs})


def _slots(width: int, height: int) -> List[Tuple[int, int]]:
    """Top-left corners of the thumbnail slots: right of the live frame, then the strip below it."""
    slots = [(x, y) for y in range(0, LIVE_H - THUMB_H + 1, THUMB_H)
             for x in range(LIVE_W, width - THUMB_W + 1, THUMB_W)]
    slots += [(x, y) for y in range(LIVE_H, height - THUMB_H + 1, THUMB_H)
              for x in range(0, width - THUMB_W + 1, THUMB_W)]
    return slots


def capacity(width: int = 1024, height: int = 768) -> int:
    return len(_slots(width, height))


def select(refs: Sequence[Tuple[str, np.ndarray]], n: int) -> List[Tuple[str, np.ndarray]]:
    """At most n references, shared out evenly between the scene types (round robin), grouped by type."""
    by_type: Dict[str, List[np.ndarray]] = {}
    for label, img in refs:
        by_type.setdefault(label, []).append(img)
    picked: Dict[str, List[np.ndarray]] = {k: [] for k in by_type}
    left = n
    for i in range(max((len(v) for v in by_type.values()), default=0)):
        for label in sorted(by_type):
            if left and i < len(by_type[label]):
                picked[label].append(by_type[label][i])
                left -= 1
    return [(label, img) for label in sorted(picked) for img in picked[label]]


def _caption(img: np.ndarray, text: str, scale: float, colour=(255, 255, 255), bg=(0, 0, 0)):
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + base + 2 * pad), bg, -1)
    cv2.putText(img, text, (pad, th + pad), font, scale, colour, thick, cv2.LINE_AA)


def build_collage(live: np.ndarray, refs: Sequence[Tuple[str, np.ndarray]],
                  width: int = 1024, height: int = 768) -> np.ndarray:
    """The live frame (fitted into LIVE_W x LIVE_H, top left) and up to capacity() labelled reference thumbnails."""
    out = np.full((height, width, 3), 40, np.uint8)
    h, w = live.shape[:2]
    s = min(LIVE_W / w, LIVE_H / h)
    lw, lh = max(1, int(w * s)), max(1, int(h * s))
    out[:lh, :lw] = cv2.resize(live, (lw, lh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    cv2.rectangle(out, (0, 0), (lw - 1, lh - 1), LIVE_COLOUR, 4)
    _caption(out[:lh, :lw], 'LIVE', 1.0, colour=(255, 255, 255), bg=LIVE_COLOUR)
    slots = _slots(width, height)
    for (x, y), (label, img) in zip(slots, select(refs, len(slots))):
        thumb = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (THUMB_W - 1, THUMB_H - 1), (255, 255, 255), 1)
        _caption(thumb, f'REF {label}', 0.45)
        out[y:y + THUMB_H, x:x + THUMB_W] = thumb
    return out
