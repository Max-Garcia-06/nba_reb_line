"""
lineups.py
----------
NBA daily lineups JSON (stats.nba.com/js/data/leaders/00_daily_lineups_YYYYMMDD.json):
starters with lineupStatus Expected/Confirmed and every rostered player's rosterStatus
(Active/Inactive), each with a UTC timestamp of its last update. Confirmations land
~60-70 min before tip (checked on 2026-03-10).

History: stored per date (2023-24 and 2025-26 have files; 2024-25 files are empty).
Live: re-read on every scan. The Inactive list catches players ruled out after the
last injury report (Questionable/Doubtful players who then sit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import requests
from sqlalchemy import inspect, text

import daily_store
import data_engine as de
from config import DECISION_LEAD_MIN

log = logging.getLogger(__name__)

URL = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{ymd}.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"}
TABLE = "lineups"
KEY = ["GAME_ID", "PLAYER_ID"]
# An empty file this old is final (2024-25 has no lineup data); a newer one may be a CDN
# glitch or not yet populated, so it stays undone and the nightly lookback retries it.
EMPTY_FINAL_DAYS = 7
COLS = ["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "lineup_status", "roster_status",
        "position", "ts_utc"]


@dataclass
class GameLineup:
    confirmed: bool
    starters: set[int] = field(default_factory=set)
    inactive: dict[int, set[int]] = field(default_factory=dict)


def parse_lineups(payload: dict, game_date: str) -> pd.DataFrame:
    rows = []
    for g in payload.get("games", []) or []:
        for side in ("homeTeam", "awayTeam"):
            t = g.get(side) or {}
            for p in t.get("players", []) or []:
                rows.append({
                    "GAME_DATE": game_date, "GAME_ID": str(g["gameId"]),
                    "TEAM_ID": int(p.get("teamId") or t.get("teamId")), "PLAYER_ID": int(p["personId"]),
                    "PLAYER_NAME": p.get("playerName", ""), "lineup_status": p.get("lineupStatus") or "",
                    "roster_status": p.get("rosterStatus") or "", "position": p.get("position") or "",
                    "ts_utc": p.get("timestamp") or "",
                })
    return pd.DataFrame(rows, columns=COLS)


def fetch_day(game_date: str, session=None) -> pd.DataFrame:
    r = (session or requests).get(URL.format(ymd=game_date.replace("-", "")), headers=HEADERS, timeout=30)
    r.raise_for_status()
    return parse_lineups(r.json(), game_date)


def backfill(start: str, end: str, engine=None, refetch: bool = False) -> int:
    final_before = (pd.Timestamp(de._today_et()) - pd.Timedelta(days=EMPTY_FINAL_DAYS)).strftime("%Y-%m-%d")
    return daily_store.run_backfill(engine or de._get_engine(), source="lineups", table=TABLE, key=KEY,
                                    start=start, end=end, fetch=lambda d, s, t: fetch_day(d),
                                    refetch=refetch, allow_empty=lambda d: d < final_before)


def lineup_states(df: pd.DataFrame) -> dict[str, GameLineup]:
    """Per GAME_ID: confirmed = two teams, five Confirmed starters each."""
    out: dict[str, GameLineup] = {}
    for gid, g in df.groupby("GAME_ID"):
        st = g[g["position"].astype(str) != ""]
        teams = set(g["TEAM_ID"].astype(int))
        confirmed = (len(teams) == 2 and all((st["TEAM_ID"] == t).sum() == 5 for t in teams)
                     and (st["lineup_status"] == "Confirmed").all())
        inactive = {int(t): set(g.loc[(g["TEAM_ID"] == t) & (g["roster_status"] == "Inactive"), "PLAYER_ID"]
                                .astype(int)) for t in teams}
        out[str(gid)] = GameLineup(bool(confirmed), set(st["PLAYER_ID"].astype(int)), inactive)
    return out


def todays_lineups(game_date: str) -> dict[str, GameLineup] | None:
    """Live states for ``game_date``; None when the JSON can't be read (callers skip lineup games)."""
    try:
        return lineup_states(fetch_day(game_date))
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning("Lineups unavailable for %s (%s)", game_date, e)
        return None


def pregame_inactive(engine=None, lead_min: int | None = None) -> dict[tuple[str, int], set[int]]:
    """
    {(GAME_ID, TEAM_ID): Inactive player ids} as known at tip − lead_min. A team-game is
    included only if it has a lineup row stamped at or before the cutoff (else the list
    wasn't out yet and callers fall back as for an uncovered game).
    """
    engine = engine or de._get_engine()
    lead = DECISION_LEAD_MIN if lead_min is None else lead_min
    if not inspect(engine).has_table(TABLE):
        return {}
    lu = pd.read_sql(text(f"SELECT GAME_ID, TEAM_ID, PLAYER_ID, roster_status, ts_utc FROM {TABLE}"), engine)
    sch = pd.read_sql(text("SELECT GAME_ID, TIP_UTC FROM schedule"), engine)
    lu = lu.merge(sch, on="GAME_ID")
    # ISO8601: per-element parsing; an inferred format would turn other ISO variants into NaT.
    tip = pd.to_datetime(lu["TIP_UTC"], utc=True, errors="coerce", format="ISO8601")
    ts = pd.to_datetime(lu["ts_utc"], utc=True, errors="coerce", format="ISO8601")
    lu = lu[ts.notna() & tip.notna() & (ts <= tip - pd.Timedelta(minutes=lead))]
    out: dict[tuple[str, int], set[int]] = {}
    for (gid, tid), g in lu.groupby(["GAME_ID", "TEAM_ID"]):
        out[(str(gid), int(tid))] = set(g.loc[g["roster_status"] == "Inactive", "PLAYER_ID"].astype(int))
    return out
