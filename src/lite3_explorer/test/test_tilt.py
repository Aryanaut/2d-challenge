import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from lite3_explorer.tilt_core import GravityTracker, too_tilted  # noqa: E402


def accel(pitch_deg, g=9.81):
    p = math.radians(pitch_deg)
    return (-g * math.sin(p), 0.0, g * math.cos(p))


def flat_tracker(mount_pitch=0.0):
    tr = GravityTracker(reference_s=1.0, alpha=1.0)
    for i in range(12):
        tr.add(i * 0.1, accel(mount_pitch))
    assert tr.ready
    return tr


def test_no_tilt_gate_before_reference():
    tr = GravityTracker(reference_s=3.0)
    tr.add(0.0, accel(20))
    assert tr.tilt() is None and too_tilted({'a': tr.tilt()}, 0.1) is None


def test_flat_is_ok_and_pitch_is_caught():
    tr = flat_tracker()
    tr.add(2.0, accel(2))
    assert too_tilted({'a': tr.tilt()}, math.radians(6)) is None
    tr.add(2.1, accel(15))      # stairs
    assert too_tilted({'a': tr.tilt()}, math.radians(6)) == 'a'


def test_mounting_tilt_cancels():
    tr = flat_tracker(mount_pitch=25)   # sensor mounted 25 deg off
    tr.add(2.0, accel(26))
    assert math.degrees(tr.tilt()) < 2
    tr.add(2.1, accel(40))
    assert math.degrees(tr.tilt()) > 10


def test_either_imu_blocks():
    assert too_tilted({'robot': 0.0, 'lidar': 0.5}, 0.1) == 'lidar'
