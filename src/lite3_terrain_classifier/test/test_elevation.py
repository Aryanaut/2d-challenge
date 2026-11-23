import numpy as np

from lite3_terrain_classifier.elevation import ElevationEstimator, GroundReference

RNG = np.random.default_rng(0)


def floor(z, n=3000, noise=0.008, x=(0.4, 1.4), y=(-0.45, 0.45), tilt=0.0):
    pts = np.column_stack((RNG.uniform(*x, n), RNG.uniform(*y, n), np.zeros(n)))
    pts[:, 2] = z + tilt * pts[:, 0] + RNG.normal(0, noise, n)
    return pts


def calibrated(z=-0.3):
    ref = GroundReference({'calibrate_s': 1.0})
    ref.add(floor(z), 0.0)
    assert ref.finish(0.5) is None                     # still collecting
    assert ref.finish(1.1) is True
    return ref


def test_reference_is_taken_after_calibrate_s():
    ref = calibrated()
    assert ref.ready and abs(ref.info['tilt_deg']) < 1 and ref.info['rms_m'] < 0.02


def test_rough_or_tilted_start_is_refused_then_retried():
    ref = GroundReference({'calibrate_s': 1.0})
    ref.add(floor(-0.3, noise=0.06), 0.0)
    assert 'not flat' in ref.finish(1.1) and not ref.ready
    ref.add(floor(-0.3, tilt=0.4), 2.0)                # restarted: collects again
    assert 'tilted' in ref.finish(3.1)
    ref.add(floor(-0.3), 4.0)
    assert ref.finish(5.1) is True


def test_too_few_points_refused():
    ref = GroundReference({'calibrate_s': 0.5})
    ref.add(floor(-0.3, n=50), 0.0)
    assert 'floor points' in ref.finish(1.0)


def test_higher_lower_level_and_unknown():
    ref = calibrated()
    est = ElevationEstimator(ref)
    assert est.estimate() == {}                        # nothing collected yet
    for dz, cls in ((0.12, 'higher'), (-0.10, 'lower'), (0.01, 'level')):
        est.reset()
        est.add(floor(-0.3 + dz, x=(0.4, 1.4)))
        out = est.estimate()
        assert out['elevation_class'] == cls and abs(out['elevation_m'] - dz) < 0.01, out
        assert out['slope_deg'] < 1.0


def test_step_in_the_patch_takes_the_dominant_level():
    ref = calibrated()
    est = ElevationEstimator(ref)
    est.add(floor(-0.3 + 0.15, n=3000, x=(0.4, 1.2)))   # most of the patch on a raised tile
    est.add(floor(-0.3, n=500, x=(1.2, 1.4)))           # a strip of the start floor
    assert abs(est.estimate()['elevation_m'] - 0.15) < 0.015


def test_ramp_reports_slope():
    ref = calibrated()
    est = ElevationEstimator(ref, {'inlier_band_m': 0.2})
    est.add(floor(-0.3, tilt=0.2))                      # rises 0.2 m per metre = 11.3 deg
    assert abs(est.estimate()['slope_deg'] - 11.3) < 1.0


def test_no_reference_no_elevation():
    est = ElevationEstimator(GroundReference())
    est.add(floor(0.0))
    assert est.estimate() == {}


def test_raised_part_behind_a_denser_own_floor_reads_higher():
    # the robot's own floor is nearer and denser; the step top (24 cm) holds ~40% of that: the tile ahead is 'higher'
    ref = calibrated()
    est = ElevationEstimator(ref)
    est.add(floor(-0.3, n=3000, x=(0.4, 0.8)))
    est.add(floor(-0.3 + 0.24, n=1200, x=(0.8, 1.4)))
    out = est.estimate()
    assert out['elevation_class'] == 'higher' and abs(out['elevation_m'] - 0.24) < 0.02


def test_a_few_stray_high_points_do_not_make_it_higher():
    ref = calibrated()
    est = ElevationEstimator(ref)
    est.add(floor(-0.3, n=3000))
    est.add(floor(-0.3 + 0.3, n=150, x=(1.3, 1.4)))     # a stray edge strip, under the fraction
    assert est.estimate()['elevation_class'] == 'level'
