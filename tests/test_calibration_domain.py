"""Isotonic calibrators must not invent an adjustment outside their fitted support.

`IsotonicRegression(out_of_bounds="clip")` returns the endpoint value for any
input beyond the training range. When the top training bin happens to be all
wins, that endpoint is exactly 1.0 — so every probability above the largest
`p` the calibrator ever saw gets pushed to certainty (in practice, to the
`MAX_CALIB_P_DELTA` cap). That manufactured lift is indistinguishable from
edge downstream, which is precisely where it does damage.
"""

import numpy as np

from calibration import fit_isotonic, fit_segmented, segment_key


def _saturating_calibrator():
    """Fit on p<=0.6 with an all-win top bin, so the iso endpoint is 1.0."""
    ps = np.concatenate([np.linspace(0.05, 0.45, 60), np.full(40, 0.6)])
    ys = np.concatenate([np.zeros(60), np.ones(40)])
    return fit_isotonic(ps, ys)


def test_isotonic_endpoint_saturates_at_one():
    """Guard the premise: the fitted map really does hit 1.0 at its top knot."""
    cal = _saturating_calibrator()
    assert float(cal.iso.predict([0.6])[0]) == 1.0
    assert float(cal.iso.X_thresholds_.max()) <= 0.6


def test_transform_leaves_probability_above_fitted_support_unchanged():
    cal = _saturating_calibrator()
    assert cal.transform(0.86) == 0.86


def test_transform_leaves_probability_below_fitted_support_unchanged():
    cal = _saturating_calibrator()
    assert cal.transform(0.01) == 0.01


def test_transform_still_calibrates_inside_fitted_support():
    cal = _saturating_calibrator()
    out = cal.transform(0.6)
    assert out > 0.6


def test_segmented_bundle_does_not_extrapolate_out_of_support():
    """The live path goes through the segmented bundle; it must inherit the bound."""
    rows = [
        {"p": 0.30, "y": 0.0, "line": 2.5, "side": "yes", "games_played": 10, "weight": 1}
        for _ in range(40)
    ] + [
        {"p": 0.50, "y": 1.0, "line": 2.5, "side": "yes", "games_played": 10, "weight": 1}
        for _ in range(40)
    ]
    bundle = fit_segmented(rows, min_global=50, min_segment=30)
    assert bundle is not None
    assert segment_key(line=2.5, side="yes", games_played=10) in bundle.segments
    # 0.90 is far above the largest p this segment ever saw (0.50).
    assert bundle.transform(0.90, line=2.5, side="yes", games_played=10) == 0.90
