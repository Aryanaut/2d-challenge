"""Tilt gating maths (no ROS): compare the current gravity direction of an IMU with a reference taken at the start.

The 2D scan comes from a 3D cloud sliced at fixed heights in base_link. When the body pitches or rolls (climbing stairs) the
slice cuts through floor, risers and ceiling, the 2D scan no longer matches the map and slam_toolbox's scan matcher walks
off. We therefore drop scans while the body is tilted more than a threshold away from how it stood at the start.

The tilt is the angle between the current and the reference gravity vector of each IMU. It needs no mounting calibration
(the mounting rotation cancels in the angle) and it ignores yaw.
"""
import math
from typing import Dict, Optional, Sequence, Tuple

Vec = Tuple[float, float, float]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(c * c for c in v))


def unit(v: Sequence[float]) -> Optional[Vec]:
    n = _norm(v)
    if n < 1e-6 or not math.isfinite(n):
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def angle_between(a: Vec, b: Vec) -> float:
    """Angle in radians between two unit vectors."""
    d = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.acos(d)


class GravityTracker:
    """Low-passed gravity direction of one IMU with a reference captured over the first reference_s seconds."""

    def __init__(self, reference_s: float = 3.0, alpha: float = 0.2):
        self.reference_s = reference_s
        self.alpha = alpha
        self.current: Optional[Vec] = None
        self.reference: Optional[Vec] = None
        self._t_first: Optional[float] = None
        self._sum = [0.0, 0.0, 0.0]
        self._n = 0

    def add(self, t: float, accel: Sequence[float]) -> None:
        """accel: raw accelerometer vector (any unit) at time t seconds."""
        u = unit(accel)
        if u is None:
            return
        if self.current is None:
            self.current = u
        else:
            a = self.alpha
            self.current = unit([(1 - a) * c + a * x for c, x in zip(self.current, u)]) or self.current
        if self.reference is None:
            if self._t_first is None:
                self._t_first = t
            for i in range(3):
                self._sum[i] += u[i]
            self._n += 1
            if t - self._t_first >= self.reference_s:
                self.reference = unit(self._sum)

    @property
    def ready(self) -> bool:
        return self.reference is not None

    def tilt(self) -> Optional[float]:
        """Radians between the current and the reference gravity, None until the reference exists."""
        if self.reference is None or self.current is None:
            return None
        return angle_between(self.current, self.reference)


def too_tilted(tilts: Dict[str, Optional[float]], max_tilt: float) -> Optional[str]:
    """Name of the first IMU tilted past max_tilt radians, else None. IMUs without a reference yet never block."""
    for name, t in tilts.items():
        if t is not None and t > max_tilt:
            return name
    return None
