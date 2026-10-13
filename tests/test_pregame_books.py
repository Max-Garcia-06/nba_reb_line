from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine

import market_snapshots
import model_vs_market as mvm
from market_snapshots import load_snapshots_before_tip

DATE = "2026-10-20"
TIP = datetime(2026, 10, 20, 23, 30, tzinfo=timezone.utc)
T_A = "KXNBAREB-26OCT20BOSNYK-BOSPLAYERA8-8"
T_NOSCHED = "KXNBAREB-26OCT20LALGSW-LALPLAYERB6-6"


def _snap(ticker, at, bid):
    return {"captured_at": at.isoformat(), "game_date": DATE, "ticker": ticker,
            "event_ticker": "-".join(ticker.split("-")[:2]), "player_name": "A", "line": 7.5,
            "yes_ask": bid + 0.02, "yes_bid": bid, "no_ask": 1 - bid, "no_bid": 0.98 - bid}


@pytest.fixture
def snaps(tmp_path, monkeypatch):
    monkeypatch.setattr(market_snapshots, "SNAPSHOT_DIR", tmp_path)
    rows = [
        _snap(T_A, TIP - timedelta(minutes=120), 0.40),
        _snap(T_A, TIP - timedelta(minutes=61), 0.42),
        _snap(T_A, TIP + timedelta(minutes=30), 0.90),  # in-game
        _snap(T_NOSCHED, TIP - timedelta(minutes=90), 0.30),
    ]
    market_snapshots.snapshot_path(DATE).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return {"BOSNYK": {"start_utc": TIP}}


def test_latest_snapshot_respects_tip_lead(snaps):
    (s,) = load_snapshots_before_tip(DATE, lead_min=60, schedule_index=snaps)
    assert s.yes_bid == 0.42


def test_earliest_snapshot_is_also_pregame(snaps):
    (s,) = load_snapshots_before_tip(DATE, lead_min=60, earliest=True, schedule_index=snaps)
    assert s.yes_bid == 0.40


def test_in_game_snapshot_never_used_even_with_zero_lead(snaps):
    (s,) = load_snapshots_before_tip(DATE, lead_min=0, schedule_index=snaps)
    assert s.yes_bid == 0.42


def test_market_rows_from_snapshots_are_pregame(snaps, monkeypatch):
    monkeypatch.setattr("data_engine.slate_schedule_index", lambda d: snaps)
    rows = mvm._market_rows_snapshots(DATE, earliest=False)
    assert list(rows["yes_bid"]) == [0.42]


def test_history_rows_drop_games_without_tip_time(tmp_path, monkeypatch):
    db = tmp_path / "h.db"
    eng = create_engine(f"sqlite:///{db}")
    pd.DataFrame([
        {"ticker": "A", "event_ticker": "E1", "player_name": "a", "line": 7.5, "teams_code": "BOSNYK",
         "game_date": DATE, "result": "yes"},
        {"ticker": "B", "event_ticker": "E2", "player_name": "b", "line": 7.5, "teams_code": "GSWMIN",
         "game_date": DATE, "result": "yes"},
    ]).to_sql("kalshi_markets", eng, index=False)
    pd.DataFrame([
        {"teams_code": "BOSNYK", "AWAY_TRICODE": "BOS", "HOME_TRICODE": "NYK", "TIP_UTC": TIP.isoformat(),
         "GAME_DATE_EST": DATE},
        {"teams_code": "GSWMIN", "AWAY_TRICODE": "GSW", "HOME_TRICODE": "MIN", "TIP_UTC": None,
         "GAME_DATE_EST": DATE},
    ]).drop(columns=["teams_code"]).to_sql("schedule", eng, index=False)
    tip = int(TIP.timestamp())
    pd.DataFrame([
        {"ticker": "A", "end_ts": tip - 7200, "yes_bid": 0.40, "yes_ask": 0.42},
        {"ticker": "A", "end_ts": tip + 3600, "yes_bid": 0.90, "yes_ask": 0.92},
        {"ticker": "B", "end_ts": tip + 3600, "yes_bid": 0.90, "yes_ask": 0.92},  # postponed game, in-game candle
    ]).to_sql("kalshi_candles", eng, index=False)
    monkeypatch.setattr(mvm, "DB_PATH", db)
    rows = mvm._market_rows_history(DATE)
    assert list(rows["ticker"]) == ["A"]
    assert list(rows["yes_bid"]) == [0.40]
