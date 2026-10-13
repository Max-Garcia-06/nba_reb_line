from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import create_engine

import data_engine as de
import slate
from identity_bridge import PlayerIndex


# ---------- ETL must not shrink a season ----------

def _eng(tmp_path):
    return create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")


def test_replace_refuses_to_shrink_a_season(tmp_path):
    eng = _eng(tmp_path)
    full = pd.DataFrame({"SEASON": ["2025-26"] * 3, "K": [1, 2, 3]})
    de._replace_seasons(full, "t", ["SEASON", "K"], ["2025-26"], eng)
    with pytest.raises(de.ETLShrinkError):
        de._replace_seasons(full.iloc[:1], "t", ["SEASON", "K"], ["2025-26"], eng)
    assert pd.read_sql("SELECT COUNT(*) n FROM t", eng)["n"][0] == 3


def test_replace_allows_growth_and_explicit_shrink(tmp_path):
    eng = _eng(tmp_path)
    df = pd.DataFrame({"SEASON": ["2025-26"] * 2, "K": [1, 2]})
    de._replace_seasons(df, "t", ["SEASON", "K"], ["2025-26"], eng)
    de._replace_seasons(pd.concat([df, df.assign(K=3).iloc[:1]]), "t", ["SEASON", "K"], ["2025-26"], eng)
    de._replace_seasons(df.iloc[:1], "t", ["SEASON", "K"], ["2025-26"], eng, allow_shrink=True)
    assert pd.read_sql("SELECT COUNT(*) n FROM t", eng)["n"][0] == 1


def test_etl_pulls_play_in_games(monkeypatch, tmp_path):
    pulled = []

    def fake_logs(season, season_type="Regular Season"):
        pulled.append(season_type)
        return pd.DataFrame({"PLAYER_ID": [len(pulled)], "GAME_ID": [f"g{season_type}"], "SEASON": [season],
                             "SEASON_TYPE": [season_type]})

    monkeypatch.setattr(de, "fetch_player_gamelogs", fake_logs)
    monkeypatch.setattr(de, "fetch_tracking_rebounds", lambda s: pd.DataFrame())
    monkeypatch.setattr(de, "fetch_schedule", lambda s: pd.DataFrame({"GAME_ID": ["x"], "SEASON": [s]}))
    monkeypatch.setattr(de, "_get_engine", lambda: _eng(tmp_path))
    monkeypatch.setattr(de, "GAMELOG_KEY", ["PLAYER_ID", "GAME_ID"])
    monkeypatch.setattr(de, "SCHEDULE_KEY", ["GAME_ID"])
    de.build_historical_store(["2025-26"])
    assert "PlayIn" in pulled


# ---------- data freshness ----------

def test_live_preflight_blocks_when_last_nights_games_are_missing(monkeypatch):
    from calibrate_preflight import check_calibrate_preflight

    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight._latest_game_date", lambda: "2026-11-01")
    monkeypatch.setattr("calibrate_preflight._model_data_end", lambda: "2026-11-01")
    monkeypatch.setattr("calibrate_preflight._latest_completed_scheduled_date", lambda: "2026-11-03")
    r = check_calibrate_preflight(live=True)
    assert not r.ok and any("player_gamelogs" in e for e in r.errors)
    monkeypatch.setattr("calibrate_preflight._latest_completed_scheduled_date", lambda: "2026-11-01")
    assert check_calibrate_preflight(live=True).ok


# ---------- identity ----------

def test_initial_fallback_requires_compatible_first_names():
    assert PlayerIndex([(2, "Jaylin Williams")]).resolve("Jalen Williams") == 0
    assert PlayerIndex([(5, "Nicolas Claxton")]).resolve("Nic Claxton") == 5
    assert PlayerIndex([(6, "Herbert Jones")]).resolve("Herb Jones") == 6
    assert PlayerIndex([(7, "Moritz Wagner")]).resolve("Moe Wagner") == 7
    assert PlayerIndex([(1, "Jalen Williams")]).resolve("J. Williams") == 1


def _ml(ticker, name, event="KXNBAREB-26MAR10DENLAL"):
    from kalshi_bridge import MarketLine

    return MarketLine(ticker=ticker, player_name=name, player_id=0, game_date="2026-03-10", line=7.5,
                      yes_ask=0.5, yes_bid=0.48, no_ask=0.52, no_bid=0.5, event_ticker=event)


def test_resolve_slate_restricts_to_the_tickers_team(monkeypatch):
    monkeypatch.setattr(slate.de, "slate_schedule_index", lambda d: {
        "DENLAL": {"game_id": "0022500900", "home_team_id": 20, "away_team_id": 10,
                   "status": "Scheduled", "start_utc": None}})
    # Same normalised name on both teams: the ticker's team prefix decides.
    monkeypatch.setattr(slate, "recent_rosters", lambda d, teams: pd.DataFrame({
        "PLAYER_ID": [1, 2], "PLAYER_NAME": ["Jalen Smith", "Jalen Smith"], "TEAM_ID": [10, 20]}))
    lines = [_ml("KXNBAREB-26MAR10DENLAL-LALJSMITH8-8", "Jalen Smith"),
             _ml("KXNBAREB-26MAR10DENLAL-DENJSMITH6-6", "Jalen Smith")]
    out, _, unresolved = slate.resolve_slate(lines, "2026-03-10")
    assert out[lines[0].ticker].player_id == 2
    assert out[lines[1].ticker].player_id == 1
    assert unresolved == []


# ---------- live population gate ----------

def test_live_rows_below_training_minutes_are_skipped(monkeypatch):
    import numpy as np

    monkeypatch.setattr(slate, "resolve_slate", lambda lines, d: (
        {lines[0].ticker: slate.SlateRow(1, "A", 10, 20, "g1", 1)}, pd.DataFrame(), []))
    monkeypatch.setattr(slate, "absences_by_team", lambda d, r, i: ({}, True))
    monkeypatch.setattr(slate.de, "slate_schedule_index", lambda d: {})
    monkeypatch.setattr(slate.de, "current_team_by_player", lambda d: {})
    monkeypatch.setattr(slate, "build_feature_table", lambda **kw: pd.DataFrame({
        "PLAYER_ID": [1], "GAME_ID": ["g1"], "_live": [True], "games_played": [40], "min_roll": [12.0]}))

    class _M:
        def pmf(self, df):
            return np.full((len(df), 30), 1 / 30)

    preds, kept, skipped = slate.build_slate_predictions([_ml("KXNBAREB-26MAR10DENLAL-DENA8-8", "A")], "2026-03-10", _M())
    assert preds == [] and any("min_roll" in s for s in skipped)


def test_freshness_ignores_preseason_schedule(tmp_path):
    import calibrate_preflight as cp

    eng = _eng(tmp_path)
    pd.DataFrame({"GAME_ID": ["0022501200", "0042500407", "0012600010", "0022600001"],
                  "GAME_DATE_EST": ["2026-04-12", "2026-06-13", "2026-10-12", "2026-10-21"]}
                 ).to_sql("schedule", eng, index=False)
    # Preseason (001) has no box scores in player_gamelogs; future games aren't due yet.
    assert cp.latest_completed_game_date(eng, today="2026-10-13") == "2026-06-13"
