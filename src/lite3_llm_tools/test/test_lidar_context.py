import math

from lite3_llm_tools import lidar_context as L


def _scan(values_by_deg, n=360, rmax=12.0):
    ranges = [float('inf')] * n
    for deg, r in values_by_deg.items():
        ranges[int(deg) % n] = r
    return dict(angle_min=0.0, angle_increment=math.radians(1.0), ranges=ranges, range_min=0.3, range_max=rmax)


def test_sectors_pick_nearest_in_each_direction():
    s = L.sector_ranges(**_scan({0: 1.0, 10: 0.7, 90: 0.4, 180: 2.0, 270: 0.9, 200: 5.0}))
    assert s['ahead'] == 0.7 and s['left'] == 0.4 and s['behind'] == 2.0 and s['right'] == 0.9


def test_invalid_returns_are_ignored_and_empty_sector_is_clear():
    s = L.sector_ranges(**_scan({0: 0.1, 90: 99.0, 180: float('nan')}))      # < range_min, > range_max, nan
    assert all(v is None for v in s.values())
    assert 'clear' in L.describe(s)


def test_forward_offset_rotates_the_sectors():
    s = L.sector_ranges(**_scan({90: 0.5}), forward_offset_deg=90.0)       # lidar x axis points to the robot's left
    assert s['ahead'] == 0.5 and s['left'] is None


def test_describe_text():
    text = L.describe({'ahead': 0.62, 'left': 0.35, 'behind': None, 'right': 0.41})
    assert 'ahead 0.62 m' in text and 'left 0.35 m' in text and 'behind clear' in text and text.startswith('LIDAR')
