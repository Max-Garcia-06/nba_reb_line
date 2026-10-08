from __future__ import annotations

import numpy as np
import pytest

from calibration import PlattCalibrator, fit_platt, load_platt, save_platt


def test_identity_params_leave_p_unchanged():
    cal = PlattCalibrator(a=1.0, b=0.0)
    for p in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert cal.transform(p) == pytest.approx(p, abs=1e-6)


def test_slope_below_one_shrinks_confidence():
    """The overconfidence correction: pull extremes toward 0.5, leave the middle alone."""
    cal = PlattCalibrator(a=0.7, b=0.0)
    assert cal.transform(0.90) < 0.90
    assert cal.transform(0.10) > 0.10
    assert cal.transform(0.50) == pytest.approx(0.50, abs=1e-6)


def test_transform_respects_max_delta_cap():
    from config import MAX_CALIB_P_DELTA

    # Parameters extreme enough that the uncapped move would be huge.
    cal = PlattCalibrator(a=0.1, b=-3.0)
    p = 0.95
    assert abs(cal.transform(p) - p) <= MAX_CALIB_P_DELTA + 1e-9


def test_transform_stays_in_open_unit_interval():
    cal = PlattCalibrator(a=2.5, b=1.5)
    for p in (0.0, 1e-9, 0.5, 1.0 - 1e-9, 1.0):
        out = cal.transform(p)
        assert 0.0 < out < 1.0


def test_fit_recovers_overconfidence():
    """Probabilities that are systematically too extreme should fit a slope below 1."""
    rng = np.random.default_rng(0)
    n = 4000
    true_p = rng.uniform(0.05, 0.95, n)
    ys = (rng.uniform(size=n) < true_p).astype(float)
    # Inflate confidence: push each logit away from 0.
    logit = np.log(true_p / (1 - true_p))
    stated = 1 / (1 + np.exp(-logit * 1.6))

    cal = fit_platt(stated, ys)

    assert cal.a < 1.0
    assert cal.n_rows == n


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "platt.pkl"
    cal = PlattCalibrator(a=0.77, b=-0.55, n_rows=803)
    save_platt(cal, path)
    loaded = load_platt(path)

    assert loaded is not None
    assert loaded.a == pytest.approx(0.77)
    assert loaded.b == pytest.approx(-0.55)
    assert loaded.n_rows == 803


def test_load_missing_returns_none(tmp_path):
    assert load_platt(tmp_path / "absent.pkl") is None
