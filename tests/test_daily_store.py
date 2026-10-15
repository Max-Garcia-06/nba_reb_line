import pandas as pd
import pytest
from sqlalchemy import create_engine

import daily_store as ds


@pytest.fixture
def eng(tmp_path, monkeypatch):
    e = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    pd.DataFrame({
        "GAME_ID": ["0012500001", "0022500001", "0022500002", "0042500001", "0022500003"],
        "SEASON": ["2025-26"] * 5,
        "GAME_DATE_EST": ["2025-10-10", "2025-10-21", "2025-10-21", "2026-04-20", "2026-10-14"],
    }).to_sql("schedule", e, index=False)
    monkeypatch.setattr("data_engine._today_et", lambda: "2026-10-14")
    return e


def _fetch_ok(date, season, stype):
    return pd.DataFrame({"PLAYER_ID": [1, 2], "GAME_DATE": [date, date], "STYPE": [stype, stype]})


def test_completed_game_dates_skip_preseason_and_today(eng):
    assert ds.completed_game_dates(eng, "2025-10-01", "2026-12-31") == [
        ("2025-10-21", "2025-26", "Regular Season"), ("2026-04-20", "2025-26", "Playoffs")]


def test_run_backfill_marks_done_and_is_idempotent(eng):
    calls = []

    def fetch(d, s, t):
        calls.append(d)
        return _fetch_ok(d, s, t)

    n = ds.run_backfill(eng, source="x", table="xt", key=["PLAYER_ID", "GAME_DATE"],
                        start="2025-10-01", end="2026-12-31", fetch=fetch)
    assert n == 4 and calls == ["2025-10-21", "2026-04-20"]
    assert ds.run_backfill(eng, source="x", table="xt", key=["PLAYER_ID", "GAME_DATE"],
                           start="2025-10-01", end="2026-12-31", fetch=fetch) == 0
    assert len(calls) == 2
    assert ds.done_dates(eng, "x") == {"2025-10-21", "2026-04-20"}


def test_run_backfill_skips_today_and_empty(eng):
    def fetch(d, s, t):
        if d == "2025-10-21":
            raise RuntimeError("stats.nba.com down")
        return pd.DataFrame(columns=["PLAYER_ID", "GAME_DATE"])

    n = ds.run_backfill(eng, source="y", table="yt", key=["PLAYER_ID", "GAME_DATE"],
                        start="2025-10-01", end="2026-12-31", fetch=fetch)
    assert n == 0 and ds.done_dates(eng, "y") == set()  # failure and empty both stay undone
    ds.run_backfill(eng, source="y", table="yt", key=["PLAYER_ID", "GAME_DATE"],
                    start="2026-04-01", end="2026-12-31", fetch=fetch, allow_empty=True)
    assert ds.done_dates(eng, "y") == {"2026-04-20"}  # allowed empty is done; today never fetched


def test_store_day_replaces_rows_for_date(eng):
    key = ["PLAYER_ID", "GAME_DATE"]
    ds.store_day(eng, table="zt", source="z", date="2025-10-21", df=_fetch_ok("2025-10-21", "", ""), key=key)
    ds.store_day(eng, table="zt", source="z", date="2025-10-21",
                 df=pd.DataFrame({"PLAYER_ID": [9], "GAME_DATE": ["2025-10-21"], "STYPE": ["r"]}), key=key)
    assert pd.read_sql("SELECT PLAYER_ID FROM zt", eng)["PLAYER_ID"].tolist() == [9]


def test_store_day_moves_rows_whose_key_reappears_on_another_date(eng):
    """A postponed game is listed in two date files: the later write wins, no duplicate keys."""
    key = ["GAME_ID", "PLAYER_ID"]
    row = lambda d: pd.DataFrame({"GAME_ID": ["g1"], "PLAYER_ID": [7], "GAME_DATE": [d]})
    ds.store_day(eng, table="lt", source="l", date="2026-01-24", df=row("2026-01-24"), key=key)
    ds.store_day(eng, table="lt", source="l", date="2026-01-25", df=row("2026-01-25"), key=key)
    assert pd.read_sql("SELECT GAME_DATE FROM lt", eng)["GAME_DATE"].tolist() == ["2026-01-25"]


def test_allow_empty_can_depend_on_date(eng):
    def fetch(d, s, t):
        return pd.DataFrame(columns=["PLAYER_ID", "GAME_DATE"])

    ds.run_backfill(eng, source="e", table="et", key=["PLAYER_ID", "GAME_DATE"], start="2025-10-01",
                    end="2026-12-31", fetch=fetch, allow_empty=lambda d: d < "2026-01-01")
    assert ds.done_dates(eng, "e") == {"2025-10-21"}
