import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

import game_markets as gm


def test_two_sided_mid():
    assert gm.two_sided_mid(0.40, 0.44) == pytest.approx(0.42)
    assert np.isnan(gm.two_sided_mid(0.0, 0.44)) and np.isnan(gm.two_sided_mid(0.4, 1.0))
    assert np.isnan(gm.two_sided_mid(0.5, 0.5))


def test_ladder_median_interpolates():
    assert gm.ladder_median([220.5, 224.5, 228.5], [0.7, 0.5, 0.3]) == pytest.approx(224.5)
    assert gm.ladder_median([220.5, 224.5, 228.5], [0.8, 0.6, 0.4]) == pytest.approx(226.5)


def test_ladder_median_edge_cases():
    assert np.isnan(gm.ladder_median([220.5, 224.5], [0.7, 0.3]))                 # < 3 lines
    assert np.isnan(gm.ladder_median([220.5, 224.5, 228.5], [0.9, 0.8, 0.6]))     # median above ladder
    assert np.isnan(gm.ladder_median([220.5, 224.5, 228.5], [0.4, 0.3, 0.2]))     # median below ladder
    # non-monotone noise is smoothed, not trusted
    assert gm.ladder_median([220.5, 222.5, 224.5, 226.5], [0.7, 0.45, 0.55, 0.3]) == pytest.approx(223.5, abs=1.0)


def test_event_features_combines_spread_sides():
    rows = pd.DataFrame([
        ("KXNBATOTAL", 220.5, "", 0.7), ("KXNBATOTAL", 224.5, "", 0.5), ("KXNBATOTAL", 228.5, "", 0.3),
        # home SAS: P(SAS by > L); away NYK: P(NYK by > L) = P(margin < -L)
        ("KXNBASPREAD", 2.5, "SAS", 0.65), ("KXNBASPREAD", 6.5, "SAS", 0.45), ("KXNBASPREAD", 10.5, "SAS", 0.25),
        ("KXNBASPREAD", 1.5, "NYK", 0.20),
    ], columns=["series", "line", "team", "mid"])
    f = gm.event_features(rows, home="SAS")
    assert f["mkt_total"] == pytest.approx(224.5)
    assert 4.5 < f["mkt_home_margin"] < 6.5


def test_history_features_uses_last_fresh_candle_before_cutoff(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    tip = pd.Timestamp("2026-03-11T00:00:00Z")
    pd.DataFrame([{"GAME_ID": "0022500940", "GAME_DATE_EST": "2026-03-10", "TIP_UTC": "2026-03-11T00:00:00Z",
                   "AWAY_TRICODE": "NYK", "HOME_TRICODE": "SAS"}]).to_sql("schedule", eng, index=False)
    mk = pd.DataFrame({"ticker": [f"KXNBATOTAL-26MAR10NYKSAS-{x}" for x in (220, 224, 228)],
                       "series": "KXNBATOTAL", "game_date": "2026-03-10", "teams_code": "NYKSAS",
                       "line": [220.5, 224.5, 228.5], "team": ""})
    mk.to_sql(gm.MARKETS_TABLE, eng, index=False)
    cut = int((tip - pd.Timedelta(minutes=30)).timestamp())
    hour = cut - cut % 3600
    rows = []
    for t, (p_ok, p_late) in zip(mk["ticker"], [(0.7, 0.9), (0.5, 0.9), (0.3, 0.9)]):
        rows += [{"ticker": t, "end_ts": hour, "yes_bid": p_ok - 0.01, "yes_ask": p_ok + 0.01},
                 {"ticker": t, "end_ts": hour + 3600, "yes_bid": p_late - 0.01, "yes_ask": p_late + 0.01}]
    pd.DataFrame(rows).to_sql(gm.CANDLES_TABLE, eng, index=False)
    out = gm.history_features(eng, lead_min=30).set_index("GAME_ID")
    assert out.loc["0022500940", "mkt_total"] == pytest.approx(224.5)
    assert np.isnan(out.loc["0022500940", "mkt_home_margin"])


def test_game_market_group_columns():
    import feature_store as fs
    from tests.test_feature_store_pit import _synthetic_gamelogs, _tracking

    gl = _synthetic_gamelogs(n_days=20)
    gid = gl["GAME_ID"].iloc[-1]
    gmf = pd.DataFrame({"GAME_ID": [gid], "mkt_total": [230.0], "mkt_home_margin": [-7.0]})
    df = fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False,
                                groups=["game_market"], game_market=gmf)
    g = df[df["GAME_ID"] == gid]
    assert (g["mkt_total"] == 230.0).all() and (g["mkt_spread_abs"] == 7.0).all()
    assert g["mkt_total_resid"].notna().all()
    assert df.loc[df["GAME_ID"] != gid, "mkt_total"].isna().all()


def test_fetch_candles_reads_dollar_and_fp_fields(monkeypatch):
    import kalshi_history as kh

    live = {"end_period_ts": 1791990000, "open_interest_fp": "45.20", "price": {"previous_dollars": "0.1200"},
            "volume_fp": "3.00", "yes_ask": {"close_dollars": "0.1600"}, "yes_bid": {"close_dollars": "0.1300"}}
    legacy = {"end_period_ts": 1781406000, "open_interest": 6650.43, "price": {"close": 0.96}, "volume": 12680.65,
              "yes_ask": {"close": 1.0}, "yes_bid": {"close": 0.98}}
    monkeypatch.setattr(kh, "_get", lambda path, params: {"candlesticks": [live, legacy]})
    a, b = kh.fetch_candles("T", 0, 10, path="/series/S/markets/T/candlesticks")
    assert (a["yes_bid"], a["yes_ask"], a["last_price"], a["volume"], a["open_interest"]) == (0.13, 0.16, 0.12, 3.0, 45.2)
    assert (b["yes_bid"], b["yes_ask"], b["last_price"], b["volume"], b["open_interest"]) == (0.98, 1.0, 0.96, 12680.65, 6650.43)
