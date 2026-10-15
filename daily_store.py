"""
daily_store.py
--------------
Per-date tables filled by resumable backfills (tracking_daily, starters, lineups).

A date is marked done in DONE_TABLE only after its rows are written, in the same
transaction. Failed fetches, and empty results unless ``allow_empty``, stay undone so
the next run (nightly: a 3-day lookback) retries them. Today and future dates are
never fetched: stats for unfinished games would be stored as complete.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

import pandas as pd
from sqlalchemy import inspect, text

import data_engine as de

log = logging.getLogger(__name__)

DONE_TABLE = "daily_backfill_done"
_SEASON_TYPES = {"002": "Regular Season", "004": "Playoffs", "005": "PlayIn"}


def done_dates(engine, source: str) -> set[str]:
    with engine.connect() as conn:
        if not inspect(conn).has_table(DONE_TABLE):
            return set()
        rows = conn.execute(text(f"SELECT date FROM {DONE_TABLE} WHERE source = :s"), {"s": source}).fetchall()
    return {r[0] for r in rows}


def completed_game_dates(engine, start: str, end: str) -> list[tuple[str, str, str]]:
    """[(date, SEASON, season_type)] with regular-season/play-in/playoff games in [start, end], before today ET."""
    sch = pd.read_sql(text("SELECT GAME_ID, SEASON, GAME_DATE_EST FROM schedule "
                           "WHERE GAME_DATE_EST BETWEEN :a AND :b AND GAME_DATE_EST < :t"),
                      engine, params={"a": start, "b": end, "t": de._today_et()})
    sch["season_type"] = sch["GAME_ID"].astype(str).str[:3].map(_SEASON_TYPES)
    sch = sch.dropna(subset=["season_type"])
    return sorted(set(zip(sch["GAME_DATE_EST"], sch["SEASON"], sch["season_type"])))


def store_day(engine, *, table: str, source: str, date: str, df: pd.DataFrame, key: list[str],
              date_col: str = "GAME_DATE") -> int:
    """Replace ``table`` rows for ``date`` with ``df`` and mark (source, date) done, atomically."""
    with engine.begin() as conn:
        if inspect(conn).has_table(table):
            conn.execute(text(f"DELETE FROM {table} WHERE {date_col} = :d"), {"d": date})
        if len(df):
            df.to_sql(table, conn, if_exists="append", index=False)
        if inspect(conn).has_table(DONE_TABLE):
            conn.execute(text(f"DELETE FROM {DONE_TABLE} WHERE source = :s AND date = :d"), {"s": source, "d": date})
        pd.DataFrame({"source": [source], "date": [date]}).to_sql(DONE_TABLE, conn, if_exists="append", index=False)
    if len(df):
        de._assert_unique(table, key, engine)
    return len(df)


def run_backfill(engine, *, source: str, table: str, key: list[str], start: str, end: str,
                 fetch: Callable[[str, str, str], pd.DataFrame], refetch: bool = False,
                 allow_empty: bool = False, date_col: str = "GAME_DATE") -> int:
    done = set() if refetch else done_dates(engine, source)
    parts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for date, season, stype in completed_game_dates(engine, start, end):
        parts[date].append((season, stype))
    todo = [d for d in sorted(parts) if d not in done]
    log.info(f"[{source}] {len(todo)} date(s) to fetch ({len(parts) - len(todo)} already done)")
    n = 0
    for i, date in enumerate(todo, 1):
        try:
            df = pd.concat([fetch(date, s, t) for s, t in parts[date]], ignore_index=True)
        except Exception as e:  # network / parse: retry next run
            log.warning(f"[{source}] {date} failed: {e}")
            continue
        df = df.drop_duplicates(subset=key, keep="last") if len(df) else df
        if df.empty and not allow_empty:
            log.warning(f"[{source}] {date} returned no rows; leaving it undone")
            continue
        n += store_day(engine, table=table, source=source, date=date, df=df, key=key, date_col=date_col)
        if i % 25 == 0:
            log.info(f"[{source}] {i}/{len(todo)} dates")
    return n
