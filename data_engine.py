"""
data_engine.py
--------------
ETL pipeline that pulls NBA player box scores, season-level rebound tracking
and the league schedule from nba_api into a local SQLite database.

Tables
------
player_gamelogs   one row per (PLAYER_ID, GAME_ID); regular season + playoffs
tracking_rebounds one row per (PLAYER_ID, SEASON); season aggregates — only
                  ever joined as a PRIOR-season feature (see feature_store)
schedule          one row per GAME_ID with tip time (UTC) and team tricodes

Every write dedupes on the table's natural key. Lesson from mlb_tb_line
(SYSTEM.md §9): plain-append tables silently quadrupled training rows when a
full re-ingest ran instead of an incremental one, and the model trained on the
duplicates came out badly overconfident.

Pull cadence:
  - Historical: `etl` once (all SEASONS).
  - Nightly: `etl --incremental` re-pulls only the current season and
    replaces that season's rows.
"""

import logging
import time
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, inspect, text

from nba_api.stats.endpoints import (
    leaguedashptstats,
    playergamelogs,
    scheduleleaguev2,
)

from config import DB_PATH, SEASONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# nba_api rate-limit guard (1 req / 0.6s is safe)
_REQUEST_DELAY = 0.65
_TIMEOUT = 60
_RETRIES = 3

GAMELOG_KEY = ["PLAYER_ID", "GAME_ID"]
TRACKING_KEY = ["PLAYER_ID", "SEASON"]
SCHEDULE_KEY = ["GAME_ID"]


def _get_engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def _call(fn, *args, **kwargs) -> list[pd.DataFrame]:
    """nba_api call with rate-limit sleep and simple retry (stats.nba.com is flaky)."""
    last_exc: Optional[Exception] = None
    for attempt in range(_RETRIES):
        time.sleep(_REQUEST_DELAY * (1 + 2 * attempt))
        try:
            return fn(*args, timeout=_TIMEOUT, **kwargs).get_data_frames()
        except Exception as e:  # network / JSON errors
            last_exc = e
            log.warning(f"  nba_api {fn.__name__} attempt {attempt + 1}/{_RETRIES} failed: {e}")
    raise RuntimeError(f"nba_api {fn.__name__} failed after {_RETRIES} attempts") from last_exc


# ---------------------------------------------------------------------------
# 1. Player game logs (box score level)
# ---------------------------------------------------------------------------

def fetch_player_gamelogs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Pull every player's game log for a single season and season type."""
    log.info(f"Fetching player game logs: {season} ({season_type})")
    df = _call(
        playergamelogs.PlayerGameLogs,
        season_nullable=season,
        season_type_nullable=season_type,
    )[0]
    df["SEASON"] = season
    df["SEASON_TYPE"] = season_type
    return df


# ---------------------------------------------------------------------------
# 2. Player tracking — rebound chances (season-level)
# ---------------------------------------------------------------------------

def fetch_tracking_rebounds(season: str) -> pd.DataFrame:
    """
    leaguedashptstats PtMeasureType=Rebounding, per game:
      REB_CHANCES, REB_CHANCE_PCT_ADJ, REB_CONTEST, REB_CHANCE_DEFER, ...
    Season aggregates — feature_store joins these from the PRIOR season only.
    """
    log.info(f"Fetching tracking rebounds: {season}")
    df = _call(
        leaguedashptstats.LeagueDashPtStats,
        season=season,
        season_type_all_star="Regular Season",
        pt_measure_type="Rebounding",
        per_mode_simple="PerGame",
        player_or_team="Player",
    )[0]
    df["SEASON"] = season
    return df


# ---------------------------------------------------------------------------
# 3. Schedule (tip times + tricodes, for Kalshi event matching)
# ---------------------------------------------------------------------------

def fetch_schedule(season: str) -> pd.DataFrame:
    log.info(f"Fetching schedule: {season}")
    df = _call(scheduleleaguev2.ScheduleLeagueV2, season=season)[0]
    out = pd.DataFrame({
        "GAME_ID": df["gameId"].astype(str),
        "SEASON": season,
        "GAME_DATE_EST": pd.to_datetime(df["gameDateEst"]).dt.strftime("%Y-%m-%d"),
        "TIP_UTC": df["gameDateTimeUTC"].astype(str),
        "HOME_TEAM_ID": pd.to_numeric(df["homeTeam_teamId"], errors="coerce"),
        "AWAY_TEAM_ID": pd.to_numeric(df["awayTeam_teamId"], errors="coerce"),
        "HOME_TRICODE": df["homeTeam_teamTricode"].astype(str),
        "AWAY_TRICODE": df["awayTeam_teamTricode"].astype(str),
        "GAME_STATUS": pd.to_numeric(df["gameStatus"], errors="coerce"),
    })
    # Preseason / All-Star / placeholder rows have no real team ids.
    return out[(out["HOME_TEAM_ID"] > 0) & (out["AWAY_TEAM_ID"] > 0)]


# ---------------------------------------------------------------------------
# 4. Persist to SQLite (dedupe on natural key)
# ---------------------------------------------------------------------------

def _replace_seasons(df: pd.DataFrame, table: str, key: list[str], seasons: list[str], engine) -> None:
    """
    Replace all rows for `seasons` in `table` with `df`, deduped on `key`.
    Rows for other seasons are left untouched, so incremental runs are cheap
    and re-running the same season is idempotent.
    """
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="last")
    if len(df) != before:
        log.warning(f"  [{table}] dropped {before - len(df):,} duplicate rows on {key}")

    with engine.begin() as conn:
        if inspect(conn).has_table(table):
            placeholders = ",".join(f":s{i}" for i in range(len(seasons)))
            conn.execute(
                text(f"DELETE FROM {table} WHERE SEASON IN ({placeholders})"),
                {f"s{i}": s for i, s in enumerate(seasons)},
            )
        df.to_sql(table, conn, if_exists="append", index=False, chunksize=500)
    log.info(f"  -> wrote {len(df):,} rows to [{table}] for seasons {seasons}")
    _assert_unique(table, key, engine)


def _assert_unique(table: str, key: list[str], engine) -> None:
    cols = ", ".join(key)
    with engine.connect() as conn:
        dupes = conn.execute(
            text(f"SELECT COUNT(*) FROM (SELECT {cols} FROM {table} GROUP BY {cols} HAVING COUNT(*) > 1)")
        ).scalar()
    if dupes:
        raise RuntimeError(f"[{table}] has {dupes} duplicate {key} groups after write")


def build_historical_store(seasons: list[str] = SEASONS, incremental: bool = False) -> None:
    """
    ETL for `seasons`. With incremental=True only the most recent season is
    re-pulled (nightly use). Safe to re-run: each season's rows are replaced.
    """
    if incremental:
        seasons = [sorted(seasons)[-1]]
    engine = _get_engine()

    for season in seasons:
        frames = [fetch_player_gamelogs(season, "Regular Season")]
        po = fetch_player_gamelogs(season, "Playoffs")
        if not po.empty:
            frames.append(po)
        gl = pd.concat(frames, ignore_index=True)
        gl["GAME_ID"] = gl["GAME_ID"].astype(str)
        _replace_seasons(gl, "player_gamelogs", GAMELOG_KEY, [season], engine)

        trk = fetch_tracking_rebounds(season)
        if not trk.empty:
            _replace_seasons(trk, "tracking_rebounds", TRACKING_KEY, [season], engine)

        _replace_seasons(fetch_schedule(season), "schedule", SCHEDULE_KEY, [season], engine)

    log.info("ETL complete.")


def _read(table: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {table}", _get_engine())


def load_gamelogs() -> pd.DataFrame:
    return _read("player_gamelogs")


def load_tracking_rebounds() -> pd.DataFrame:
    return _read("tracking_rebounds")


def load_schedule() -> pd.DataFrame:
    return _read("schedule")


if __name__ == "__main__":
    import sys
    build_historical_store(incremental="--incremental" in sys.argv)
