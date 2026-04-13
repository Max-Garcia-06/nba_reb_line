"""
data_engine.py
--------------
ETL pipeline that pulls NBA player box scores and tracking data
from nba_api and stores them in a local SQLite database.

Pull cadence:
  - Historical (3 seasons): run once, then incrementally.
  - Today's game context: run ~2 hours before tip-off.
"""

import time
import logging
from datetime import datetime, date
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashptstats,
    playergamelogs,
    leaguegamefinder,
    commonteamroster,
    leaguedashteamstats,
)
from nba_api.stats.static import players as nba_players_static

from config import DB_PATH, SEASONS, MIN_GAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# nba_api rate-limit guard (1 req / 0.6s is safe)
_REQUEST_DELAY = 0.65


def _get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def _sleep():
    time.sleep(_REQUEST_DELAY)


# ---------------------------------------------------------------------------
# 1. Player game logs (box score level)
# ---------------------------------------------------------------------------

def fetch_player_gamelogs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Pull every player's game log for a single season and season type."""
    log.info(f"Fetching player game logs: {season} ({season_type})")
    _sleep()
    gl = playergamelogs.PlayerGameLogs(
        season_nullable=season,
        season_type_nullable=season_type,
    )
    df = gl.get_data_frames()[0]
    df["SEASON"] = season
    df["SEASON_TYPE"] = season_type
    return df


def fetch_all_gamelogs(seasons: list[str] = SEASONS) -> pd.DataFrame:
    frames = []
    for s in seasons:
        frames.append(fetch_player_gamelogs(s, "Regular Season"))
        frames.append(fetch_player_gamelogs(s, "Playoffs"))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Player tracking — rebound chances & box-out data
# ---------------------------------------------------------------------------

def fetch_tracking_rebounds(season: str) -> pd.DataFrame:
    """
    leaguedashptstats with PtMeasureType=Rebounding returns:
      - OREB_CHANCE, DREB_CHANCE, REB_CHANCE_PCT_ADJ
      - CONTESTED_OREB, CONTESTED_DREB
      - OREB_CHANCE_DEFER (deferred rebounds)
    """
    log.info(f"Fetching tracking rebounds: {season}")
    _sleep()
    pt = leaguedashptstats.LeagueDashPtStats(
        season=season,
        season_type_all_star="Regular Season",
        pt_measure_type="Rebounding",
        per_mode_simple="PerGame",
        player_or_team="Player",
    )
    df = pt.get_data_frames()[0]
    df["SEASON"] = season
    return df


def fetch_tracking_passing(season: str) -> pd.DataFrame:
    """Box-out data lives under Efficiency measure type."""
    log.info(f"Fetching tracking efficiency/box-outs: {season}")
    _sleep()
    pt = leaguedashptstats.LeagueDashPtStats(
        season=season,
        season_type_all_star="Regular Season",
        pt_measure_type="Efficiency",
        per_mode_simple="PerGame",
        player_or_team="Player",
    )
    df = pt.get_data_frames()[0]
    df["SEASON"] = season
    return df


# ---------------------------------------------------------------------------
# 3. Opponent shooting tendencies (for rebound opportunity estimation)
# ---------------------------------------------------------------------------

def fetch_opponent_shooting(season: str) -> pd.DataFrame:
    """
    Team-level opponent shooting stats.
    We derive opp_missed_fg_per_100 and opp_shot_distance_avg from
    league dash team stats with opponent context.
    """
    log.info(f"Fetching opponent shooting: {season}")
    _sleep()
    ts = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Opponent",
        per_mode_detailed="Per100Possessions",
    )
    df = ts.get_data_frames()[0]
    df["SEASON"] = season
    return df


def fetch_team_base_stats(season: str) -> pd.DataFrame:
    """Team advanced stats including PACE."""
    log.info(f"Fetching team advanced stats: {season}")
    _sleep()
    ts = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    df = ts.get_data_frames()[0]
    df["SEASON"] = season
    return df


# ---------------------------------------------------------------------------
# 4. Injury / roster context (best-effort via roster endpoint)
# ---------------------------------------------------------------------------

def fetch_active_roster(team_id: int) -> pd.DataFrame:
    """Pull current roster for a team (used for lineup adjustment)."""
    _sleep()
    r = commonteamroster.CommonTeamRoster(team_id=str(team_id))
    return r.get_data_frames()[0]


# ---------------------------------------------------------------------------
# 5. Persist to SQLite
# ---------------------------------------------------------------------------

def _upsert(df: pd.DataFrame, table: str, engine, if_exists: str = "replace"):
    """Simple upsert via replace (idempotent full refresh per table)."""
    df.to_sql(table, engine, if_exists=if_exists, index=False, chunksize=500)
    log.info(f"  -> wrote {len(df):,} rows to [{table}]")


def build_historical_store(seasons: list[str] = SEASONS):
    """Full historical ETL. Safe to re-run (replaces tables)."""
    engine = _get_engine()

    gl_frames, trk_frames, trk_eff_frames, opp_frames, team_frames = [], [], [], [], []

    for season in seasons:
        # Regular season + playoffs for game logs
        gl_frames.append(fetch_player_gamelogs(season, "Regular Season"))
        gl_frames.append(fetch_player_gamelogs(season, "Playoffs"))
        trk_frames.append(fetch_tracking_rebounds(season))
        trk_eff_frames.append(fetch_tracking_passing(season))
        opp_frames.append(fetch_opponent_shooting(season))
        team_frames.append(fetch_team_base_stats(season))

    _upsert(pd.concat(gl_frames, ignore_index=True), "player_gamelogs", engine)
    _upsert(pd.concat(trk_frames, ignore_index=True), "tracking_rebounds", engine)
    _upsert(pd.concat(trk_eff_frames, ignore_index=True), "tracking_efficiency", engine)
    _upsert(pd.concat(opp_frames, ignore_index=True), "opponent_shooting", engine)
    _upsert(pd.concat(team_frames, ignore_index=True), "team_base_stats", engine)

    log.info("Historical store build complete.")


def load_gamelogs() -> pd.DataFrame:
    engine = _get_engine()
    return pd.read_sql("SELECT * FROM player_gamelogs", engine)


def load_tracking_rebounds() -> pd.DataFrame:
    engine = _get_engine()
    return pd.read_sql("SELECT * FROM tracking_rebounds", engine)


def load_tracking_efficiency() -> pd.DataFrame:
    engine = _get_engine()
    return pd.read_sql("SELECT * FROM tracking_efficiency", engine)


def load_opponent_shooting() -> pd.DataFrame:
    engine = _get_engine()
    return pd.read_sql("SELECT * FROM opponent_shooting", engine)


def load_team_base_stats() -> pd.DataFrame:
    engine = _get_engine()
    return pd.read_sql("SELECT * FROM team_base_stats", engine)


if __name__ == "__main__":
    build_historical_store()
