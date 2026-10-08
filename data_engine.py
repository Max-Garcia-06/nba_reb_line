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
import re
import time
from datetime import datetime, timezone
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



# ---------------------------------------------------------------------------
# 5. Slate helpers for live scan (ported from mlb_tb_line; same contract)
# ---------------------------------------------------------------------------

# Schedule statuses where the game has started or finished (excluded from live scan).
_STARTED_GAME_STATUSES = frozenset({"In Progress", "Final"})
_STATUS_BY_CODE = {1: "Scheduled", 2: "In Progress", 3: "Final"}

_EVENT_MATCHUP_RE = re.compile(r"^KX[A-Z]+-\d{2}[A-Z]{3}\d{2}([A-Z]{3})([A-Z]{3})$")


def parse_kalshi_event_matchup(event_ticker: str) -> tuple[str, str] | None:
    """
    Parse away/home tricodes from a Kalshi NBA event ticker, e.g.
    ``KXNBAREB-26JUN13NYKSAS`` -> (``NYK``, ``SAS``). NBA tricodes are always 3 letters.
    """
    m = _EVENT_MATCHUP_RE.match((event_ticker or "").strip())
    return (m.group(1), m.group(2)) if m else None


def _parse_game_datetime_utc(raw: str) -> datetime | None:
    s = (raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _live_status_by_game_id() -> dict[str, str]:
    """Today's game statuses from the NBA live scoreboard (empty on failure)."""
    try:
        from nba_api.live.nba.endpoints import scoreboard
        games = scoreboard.ScoreBoard().get_dict()["scoreboard"]["games"]
        return {str(g["gameId"]): _STATUS_BY_CODE.get(int(g["gameStatus"]), "") for g in games}
    except Exception as e:
        log.warning(f"live scoreboard unavailable ({e}); falling back to stored schedule status")
        return {}


def slate_schedule_index(game_date: str) -> dict[str, dict]:
    """
    Map matchup slug (e.g. ``NYKSAS``) -> ``{status, start_utc, game_id, home_team_id,
    away_team_id}`` for ``game_date`` (US/Eastern date), from the stored schedule
    table with live status overlaid when available.
    """
    try:
        sch = pd.read_sql(
            text("SELECT * FROM schedule WHERE GAME_DATE_EST = :d"), _get_engine(), params={"d": game_date})
    except Exception:
        sch = pd.DataFrame()
    if sch.empty:
        season = season_for_date(game_date)
        sch = fetch_schedule(season)
        sch = sch[sch["GAME_DATE_EST"] == game_date]
    live = _live_status_by_game_id()
    out: dict[str, dict] = {}
    for r in sch.itertuples():
        status = live.get(str(r.GAME_ID)) or _STATUS_BY_CODE.get(int(r.GAME_STATUS or 1), "")
        out[matchup_slug(r.AWAY_TRICODE, r.HOME_TRICODE)] = {
            "status": status,
            "start_utc": _parse_game_datetime_utc(str(r.TIP_UTC)),
            "game_id": str(r.GAME_ID),
            "home_team_id": int(r.HOME_TEAM_ID),
            "away_team_id": int(r.AWAY_TEAM_ID),
        }
    return out


def season_for_date(game_date: str) -> str:
    """NBA season string for a date: Oct-Dec belong to the season starting that year."""
    d = datetime.strptime(game_date, "%Y-%m-%d")
    start = d.year if d.month >= 8 else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def matchup_slug(away_abbr: str, home_abbr: str) -> str:
    return f"{away_abbr}{home_abbr}"


def game_status_allows_scan(status: str) -> bool:
    return (status or "").strip() not in _STARTED_GAME_STATUSES


def _parse_game_datetime_utc(raw: str) -> datetime | None:
    """Parse MLB ``game_datetime`` (e.g. ``2026-05-24T16:15:00Z``) to aware UTC."""
    s = (raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _event_ticker_from_market_line(ml) -> str:
    et = str(getattr(ml, "event_ticker", "") or "").strip()
    if not et and getattr(ml, "ticker", ""):
        parts = str(ml.ticker).split("-")
        if len(parts) >= 2:
            et = f"{parts[0]}-{parts[1]}"
    return et

def matchup_status_map(game_date: str) -> dict[str, str]:
    """Map ``TEXCOL``-style keys to NBA schedule status for ``game_date`` (YYYY-MM-DD)."""
    return {slug: str(row.get("status", "") or "") for slug, row in slate_schedule_index(game_date).items()}


def matchup_start_time_map(game_date: str) -> dict[str, datetime]:
    """Map matchup slug -> tip-off time (UTC) for ``game_date``."""
    out: dict[str, datetime] = {}
    for slug, row in slate_schedule_index(game_date).items():
        start = row.get("start_utc")
        if isinstance(start, datetime):
            out[slug] = start
    return out


def filter_market_lines_by_start_window(
    market_lines: list,
    game_date: str,
    *,
    within_hours: float,
    now: datetime | None = None,
    schedule_index: dict[str, dict] | None = None,
) -> tuple[list, list[tuple[str, str, str, str]]]:
    """
    Keep only markets whose NBA game tips within ``within_hours`` of ``now`` (UTC).

    Returns ``(kept_lines, excluded)`` where each excluded entry is
    ``(event_ticker, matchup_slug, game_datetime_iso, reason)``.
    Unparseable event tickers or missing schedule rows are excluded (fail closed).
    """
    if within_hours <= 0:
        return list(market_lines), []

    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    idx = schedule_index if schedule_index is not None else slate_schedule_index(game_date)
    max_sec = float(within_hours) * 3600.0
    kept: list = []
    excluded: list[tuple[str, str, str, str]] = []
    seen_events: set[str] = set()

    for ml in market_lines:
        et = _event_ticker_from_market_line(ml)
        matchup = parse_kalshi_event_matchup(et) if et else None
        if not matchup:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, "", "", "unparseable_event"))
            continue
        key = matchup_slug(*matchup)
        row = idx.get(key)
        if not row:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, "", "no_schedule"))
            continue
        start = row.get("start_utc")
        if not isinstance(start, datetime):
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, "", "no_start_time"))
            continue
        start_utc = start.astimezone(timezone.utc)
        delta_sec = (start_utc - now_utc).total_seconds()
        start_iso = start_utc.isoformat().replace("+00:00", "Z")
        if delta_sec < 0:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, start_iso, "already_started"))
            continue
        if delta_sec > max_sec:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, start_iso, "too_far"))
            continue
        kept.append(ml)

    return kept, excluded


def filter_market_lines_pregame(
    market_lines: list,
    game_date: str,
    schedule_index: dict[str, dict] | None = None,
) -> tuple[list, list[tuple[str, str, str]]]:
    """
    Drop markets tied to NBA games that have already started or finished.

    Returns ``(kept_lines, excluded)`` where each excluded entry is
    ``(event_ticker, matchup_slug, status)``.
    """
    if schedule_index is not None:
        status_by_matchup = {slug: str(row.get("status", "") or "") for slug, row in schedule_index.items()}
    else:
        status_by_matchup = matchup_status_map(game_date)
    kept: list = []
    excluded: list[tuple[str, str, str]] = []
    seen_events: set[str] = set()
    for ml in market_lines:
        et = _event_ticker_from_market_line(ml)
        matchup = parse_kalshi_event_matchup(et) if et else None
        if not matchup:
            kept.append(ml)
            continue
        key = matchup_slug(*matchup)
        status = status_by_matchup.get(key, "")
        if not status or game_status_allows_scan(status):
            kept.append(ml)
            continue
        if et not in seen_events:
            seen_events.add(et)
            excluded.append((et, key, status))
    return kept, excluded




if __name__ == "__main__":
    import sys
    build_historical_store(incremental="--incremental" in sys.argv)
