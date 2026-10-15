import numpy as np
import pandas as pd
import pytest

import feature_store as fs
from tests.test_feature_store_pit import _key, _synthetic_gamelogs, _tracking

G = ["tracking"]


def _td_from(gl, seed=0):
    rng = np.random.default_rng(seed)
    t = gl[["PLAYER_ID", "TEAM_ID", "GAME_DATE", "MIN", "REB"]].copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    t["REB_CHANCES"] = t["REB"] + rng.integers(0, 6, len(t))
    t["OREB_CHANCES"] = (t["REB_CHANCES"] * 0.3).round()
    t["DREB_CHANCES"] = t["REB_CHANCES"] - t["OREB_CHANCES"]
    t["REB_CONTEST"] = (t["REB"] * 0.4).round()
    t["REB_CHANCE_DEFER"] = rng.integers(0, 3, len(t))
    t["AVG_REB_DIST"] = rng.uniform(4, 10, len(t))
    return t


def _build(gl, td, **kw):
    return fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, groups=G,
                                  tracking_daily=td, **kw)


def test_tracking_state_ratios():
    td = pd.DataFrame({"PLAYER_ID": [1] * 3, "GAME_DATE": ["2025-11-01", "2025-11-03", "2025-11-05"],
                       "MIN": [30, 30, 30], "REB": [5, 7, 9], "REB_CHANCES": [10, 10, 10],
                       "OREB_CHANCES": [3, 3, 3], "DREB_CHANCES": [7, 7, 7], "REB_CONTEST": [2, 2, 2],
                       "REB_CHANCE_DEFER": [1, 1, 1], "AVG_REB_DIST": [6, 6, 6], "TEAM_ID": 9})
    st = fs.tracking_state(td).iloc[-1]
    assert st["reb_chances_roll"] == pytest.approx(10)
    assert st["reb_conv_roll"] == pytest.approx(21 / 30)
    assert st["reb_contest_pct_roll"] == pytest.approx(6 / 21)
    assert st["reb_defer_rate_roll"] == pytest.approx(0.1)
    assert st["oreb_chance_share_roll"] == pytest.approx(0.3)
    assert st["reb_chances36_roll"] == pytest.approx(12.0)


def test_tracking_features_no_lookahead():
    gl = _synthetic_gamelogs()
    td = _td_from(gl)
    cut = pd.Timestamp("2024-11-25")
    scr = td.copy()
    late = pd.to_datetime(scr["GAME_DATE"]) >= cut
    for c in ["REB_CHANCES", "REB_CONTEST", "OREB_CHANCES", "REB_CHANCE_DEFER"]:
        scr.loc[late, c] = np.random.default_rng(3).permutation(scr.loc[late, c].values)
    a, b = _key(_build(gl, td)), _key(_build(gl, scr))
    rows = a.index[a["GAME_DATE"] <= cut]
    cols = fs.FEATURE_GROUPS["tracking"]
    assert a.loc[rows, cols].notna().any().all()
    pd.testing.assert_frame_equal(a.loc[rows, cols], b.loc[rows, cols], check_dtype=False)


def test_missing_tracking_table_gives_nan_columns():
    df = _build(_synthetic_gamelogs(n_days=10), pd.DataFrame())
    assert df[fs.FEATURE_GROUPS["tracking"]].isna().all().all()


def test_fetch_tracking_day_shapes(monkeypatch):
    import nba_daily

    raw = pd.DataFrame({"PLAYER_ID": [1], "TEAM_ID": [2], "MIN": [30.0], "REB": [8], "OREB_CHANCES": [3],
                        "DREB_CHANCES": [9], "REB_CHANCES": [12], "REB_CONTEST": [3],
                        "REB_CHANCE_DEFER": [1], "AVG_REB_DIST": [6.1], "PLAYER_NAME": ["x"]})
    seen = {}

    def fake_call(fn, **kw):
        seen.update(kw)
        return [raw]

    monkeypatch.setattr("data_engine._call", fake_call)
    out = nba_daily.fetch_tracking_day("2026-03-10", "2025-26", "Regular Season")
    assert seen["date_from_nullable"] == "03/10/2026" and seen["per_mode_simple"] == "Totals"
    assert out["GAME_DATE"].tolist() == ["2026-03-10"] and out["SEASON"].tolist() == ["2025-26"]
    assert "PLAYER_NAME" not in out.columns
