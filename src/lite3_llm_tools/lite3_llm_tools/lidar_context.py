"""Turn a 2D lidar scan into a one-line text context for the AI (no ROS).

The camera shows what things look like; the lidar gives the AI metric clearances it cannot read from a
picture. Angles are in the robot frame: 0 = straight ahead, positive = left (counter-clockwise).
"""
import math
from typing import Dict, Optional, Sequence

SECTORS = {            # name: (centre deg, half width deg)
    'ahead': (0.0, 30.0),
    'left': (90.0, 30.0),
    'behind': (180.0, 30.0),
    'right': (-90.0, 30.0),
}


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def sector_ranges(angle_min: float, angle_increment: float, ranges: Sequence[float],
                  range_min: float, range_max: float, forward_offset_deg: float = 0.0) -> Dict[str, Optional[float]]:
    """Nearest valid return in each sector (metres), or None if the sector has no valid return.

    forward_offset_deg: the scan's angle that points forward on the robot (0 if the lidar's x axis is forward).
    """
    best: Dict[str, Optional[float]] = {name: None for name in SECTORS}
    for i, r in enumerate(ranges):
        if not math.isfinite(r) or r < range_min or r > range_max:
            continue
        deg = _wrap_deg(math.degrees(angle_min + i * angle_increment) - forward_offset_deg)
        for name, (centre, half) in SECTORS.items():
            if abs(_wrap_deg(deg - centre)) <= half and (best[name] is None or r < best[name]):
                best[name] = r
    return best


def describe(sectors: Dict[str, Optional[float]], max_range_m: float = 12.0) -> str:
    """'LIDAR: nearest obstacle ahead 0.62 m, left 0.35 m, behind clear, right 0.41 m.'"""
    parts = []
    for name in SECTORS:
        r = sectors.get(name)
        parts.append(f'{name} {r:.2f} m' if r is not None else f'{name} clear (nothing within {max_range_m:g} m)')
    return 'LIDAR (robot frame): nearest obstacle ' + ', '.join(parts) + '.'


def summarize_scan(scan, forward_offset_deg: float = 0.0) -> Dict[str, Optional[float]]:
    """From a sensor_msgs/LaserScan (duck-typed, so tests need no ROS)."""
    return sector_ranges(scan.angle_min, scan.angle_increment, list(scan.ranges), scan.range_min, scan.range_max,
                         forward_offset_deg)
