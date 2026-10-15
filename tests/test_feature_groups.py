import numpy as np
import pandas as pd
import pytest

import feature_store as fs
from tests.test_feature_store_pit import _synthetic_gamelogs, _tracking


def test_base_features_unchanged_and_groups_append_sorted():
    assert fs.model_features([]) == fs.BASE_FEATURES
    got = fs.model_features(["travel", "opp_vacuum"])
    assert got == fs.BASE_FEATURES + fs.FEATURE_GROUPS["opp_vacuum"] + fs.FEATURE_GROUPS["travel"]


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="Unknown FEATURE_GROUPS"):
        fs.model_features(["nope"])


def test_feature_version_names_groups_and_lead():
    assert fs.feature_version([], 30) == "v3-lead30-base"
    assert fs.feature_version(["travel", "lineup"], 30) == "v3-lead30-lineup+travel"


def test_build_feature_table_returns_requested_group_columns():
    gl = _synthetic_gamelogs(n_days=20)
    df = fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, groups=["opp_vacuum"])
    for c in fs.FEATURE_GROUPS["opp_vacuum"]:
        assert c in df.columns
    assert "travel_km" not in df.columns


def _train_frame():
    gl = _synthetic_gamelogs(n_days=60)
    df = fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False)
    return df[df["REB"].notna() & df["min_roll"].notna()].reset_index(drop=True)


def test_model_remembers_feature_list():
    from model_zoo import XGBNBHetero

    df = _train_frame()
    m = XGBNBHetero().fit(df, features=fs.BASE_FEATURES)
    assert m.feats == fs.BASE_FEATURES
    p = m.pmf(df.assign(extra_col=1.0))  # extra columns are ignored
    assert p.shape == (len(df), 21)


def test_legacy_model_uses_base_features():
    from model_zoo import XGBNBHetero

    df = _train_frame()
    m = XGBNBHetero().fit(df, features=fs.BASE_FEATURES)
    del m.__dict__["features"]  # pickles from before this change have no attribute
    assert m.feats == fs.BASE_FEATURES
    assert m.pmf(df).shape[0] == len(df)


def test_preflight_flags_feature_mismatch(monkeypatch):
    import calibrate_preflight as cp

    monkeypatch.setattr(cp, "_model_features", lambda: fs.BASE_FEATURES[:-1])
    res = cp.check_calibrate_preflight(live=True)
    assert any("feature set" in e for e in res.errors)


def test_date_bootstrap_constant_diff_has_degenerate_ci():
    from bakeoff import date_bootstrap

    dates = np.repeat(pd.date_range("2026-01-01", periods=10).values, 5)
    point, lo, hi = date_bootstrap(np.full(50, -0.01), dates, n_boot=200)
    assert point == pytest.approx(-0.01) and lo == pytest.approx(-0.01) and hi == pytest.approx(-0.01)


def test_line_ll_rows_matches_score():
    from bakeoff import line_frame, line_ll_rows, score

    rng = np.random.default_rng(0)
    pmf = rng.dirichlet(np.ones(21), size=30)
    y = rng.integers(0, 15, size=30).astype(float)
    feat = pd.DataFrame({"reb_roll": rng.uniform(2, 12, 30), "reb_ewm": np.nan})
    lf = line_frame(feat)
    assert line_ll_rows(pmf, y, lf).mean() == pytest.approx(score(pmf, y, lf)["ll_line"])
