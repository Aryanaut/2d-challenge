import math

import numpy as np

from lite3_explorer.scan_fill import fill_ranges


def test_no_return_in_front_becomes_the_fill_value_behind_stays_inf():
    n = 360
    amin, inc = -math.pi, 2 * math.pi / n
    r = np.full(n, np.inf)
    r[180] = 0.8                                   # straight ahead: a wall
    r[100] = np.nan
    r[200] = 0.1                                   # below range_min (the body)
    r[210] = 13.0                                  # beyond range_max
    out = fill_ranges(r, amin, inc, 0.35, 12.0, 11.5, math.radians(100))
    assert out[180] == np.float32(0.8)
    for i in (100, 170, 200, 210, 270):            # -80 .. +90 deg
        assert out[i] == np.float32(11.5)
    for i in (0, 60, 300, 359):                    # -180, -120, +120, +179 deg: the body shadow
        assert np.isinf(out[i])
    assert np.isinf(r[170])                        # input untouched
