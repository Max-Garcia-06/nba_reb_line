"""
nba_daily.py
------------
Per-date nba_api pulls stored by daily_store's resumable backfill:

tracking_daily  LeagueDashPtStats Rebounding, one game date per call (Totals):
                rebound chances, contested rebounds, deferred chances, distance.
starters        LeagueDashPlayerStats StarterBench=Starters, one game date per call.

A player plays at most one game per date, so (PLAYER_ID, GAME_DATE) is the key.
"""

from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import leaguedashptstats
from sqlalchemy import inspect

import daily_store
import data_engine as de

TRACKING_TABLE = "tracking_daily"
TRACKING_KEY = ["PLAYER_ID", "GAME_DATE"]
TRACKING_COLS = ["PLAYER_ID", "TEAM_ID", "MIN", "REB", "OREB_CHANCES", "DREB_CHANCES", "REB_CHANCES",
                 "REB_CONTEST", "REB_CHANCE_DEFER", "AVG_REB_DIST"]


def _mdy(date: str) -> str:
    return f"{date[5:7]}/{date[8:10]}/{date[:4]}"


def fetch_tracking_day(date: str, season: str, season_type: str) -> pd.DataFrame:
    df = de._call(leaguedashptstats.LeagueDashPtStats, season=season, season_type_all_star=season_type,
                  pt_measure_type="Rebounding", per_mode_simple="Totals", player_or_team="Player",
                  date_from_nullable=_mdy(date), date_to_nullable=_mdy(date))[0]
    out = df[TRACKING_COLS].copy()
    out["GAME_DATE"], out["SEASON"] = date, season
    return out


def backfill_tracking(start: str, end: str, engine=None, refetch: bool = False) -> int:
    return daily_store.run_backfill(engine or de._get_engine(), source="tracking", table=TRACKING_TABLE,
                                    key=TRACKING_KEY, start=start, end=end, fetch=fetch_tracking_day,
                                    refetch=refetch)


def load_table(table: str, engine=None) -> pd.DataFrame:
    engine = engine or de._get_engine()
    if not inspect(engine).has_table(table):
        return pd.DataFrame()
    return pd.read_sql(f"SELECT * FROM {table}", engine)
