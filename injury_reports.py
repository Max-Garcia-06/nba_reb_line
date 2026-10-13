"""
injury_reports.py
-----------------
The NBA's official injury report (PDF, republished every 15 minutes on game days)
as the single source of "who is known to be out before tip", for both history
and live scans.

Why: the teammate-vacuum features used to count a historical absence as "no
box-score row", which includes late scratches nobody knew about pre-game, while
live scans only see an injury list. History with hindsight and live without it
is train/serve skew, and it inflated every backtest that leaned on absences
(SYSTEM.md §5). With this module both sides use the latest report published
before the decision time.

Layout (2025-26): columns Game Date | Game Time | Matchup | Team | Player Name |
Current Status | Reason. Date/time/matchup/team are printed once and carry
forward; long reasons wrap onto lines with no player; a team that has not filed
yet shows "NOT YET SUBMITTED" in the reason column.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from sqlalchemy import text

from config import DATA_DIR

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
BASE_URL = "https://ak-static.cms.nba.com/referee/injury"
CACHE_DIR = Path(DATA_DIR) / "injury_reports"
TABLE = "injury_reports"
NOT_SUBMITTED = "NOT YET SUBMITTED"
OUT_STATUSES = {"out"}
STEP_MIN = 15

# Column left edges (x0, points) in the 2025-26 layout; recomputed from the header when present.
_DEFAULT_COLS = {"date": 0, "time": 110, "matchup": 195, "team": 258, "player": 420, "status": 580, "reason": 660}
_HEADER_KEYS = {"Matchup": "matchup", "Team": "team", "Player": "player", "Current": "status", "Reason": "reason"}


# ---------------------------------------------------------------------------
# Names and teams
# ---------------------------------------------------------------------------

def to_first_last(raw: str) -> str:
    """'Smith Jr., Jabari' -> 'Jabari Smith Jr.' (names without a comma pass through)."""
    s = " ".join(str(raw).split())
    if "," not in s:
        return s
    last, first = s.split(",", 1)
    return f"{first.strip()} {last.strip()}".strip()


_TEAM_CODES: dict[str, str] | None = None


def _team_codes() -> dict[str, str]:
    global _TEAM_CODES
    if _TEAM_CODES is None:
        from nba_api.stats.static import teams

        _TEAM_CODES = {}
        for t in teams.get_teams():
            for name in (t["full_name"], f"{t['city']} {t['nickname']}"):
                _TEAM_CODES[re.sub(r"\s+", "", name).lower()] = t["abbreviation"]
        _TEAM_CODES["laclippers"] = "LAC"
    return _TEAM_CODES


def team_tricode(team_name: str) -> str:
    key = re.sub(r"\s+", "", str(team_name)).lower()
    code = _team_codes().get(key)
    if code:
        return code
    for full, abbr in _team_codes().items():  # tolerate a dropped/odd city prefix
        if full.endswith(key) or key.endswith(full):
            return abbr
    return ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _columns_from_header(words: list[dict]) -> dict[str, float] | None:
    header = next((ln for ln in _lines(words) if any(w["text"] == "Matchup" for w in ln)), None)
    if header is None:
        return None
    # Only the header line: data cells can contain the same words ("Not With Team").
    hdr = {_HEADER_KEYS[w["text"]]: float(w["x0"]) for w in header if w["text"] in _HEADER_KEYS}
    if len(hdr) < 4:
        return None
    cols = dict(_DEFAULT_COLS)
    for k, x in hdr.items():
        cols[k] = x - 6  # cells start a few points right of the header word
    return cols


def _lines(words: list[dict], tol: float = 3.0) -> list[list[dict]]:
    out: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (float(w["top"]), float(w["x0"]))):
        if out and abs(float(out[-1][0]["top"]) - float(w["top"])) <= tol:
            out[-1].append(w)
        else:
            out.append([w])
    return out


def _cells(line: list[dict], cols: dict[str, float]) -> dict[str, str]:
    order = sorted(cols.items(), key=lambda kv: kv[1])
    cells: dict[str, list[str]] = {}
    for w in sorted(line, key=lambda w: float(w["x0"])):
        col = order[0][0]
        for name, x in order:
            if float(w["x0"]) >= x:
                col = name
        cells.setdefault(col, []).append(str(w["text"]))
    return {k: " ".join(v) for k, v in cells.items()}


def parse_words(pages: list[list[dict]]) -> list[dict]:
    """
    Report rows from per-page word boxes (``x0``, ``top``, ``text``): one dict per
    listed player — game_date, game_time, matchup, team (tricode), player_name
    (First Last), status — plus a row with status NOT_SUBMITTED for each team that
    has not filed yet.
    """
    rows: list[dict] = []
    cols = dict(_DEFAULT_COLS)
    cur = {"date": "", "time": "", "matchup": "", "team": ""}
    for words in pages:
        cols = _columns_from_header(words) or cols
        for line in _lines(words):
            texts = {w["text"] for w in line}
            if "Report:" in texts or "Matchup" in texts or any(t.startswith("Page") for t in texts):
                continue
            c = _cells(line, cols)
            if c.get("date") and re.match(r"\d{2}/\d{2}/\d{4}", c["date"]):
                cur["date"] = datetime.strptime(c["date"].split()[0], "%m/%d/%Y").strftime("%Y-%m-%d")
            if c.get("time"):
                cur["time"] = c["time"].replace("(ET)", "").strip()
            if c.get("matchup") and "@" in c["matchup"]:
                cur["matchup"] = c["matchup"].replace(" ", "")
            if c.get("team"):
                code = team_tricode(c["team"])
                if code:
                    cur["team"] = code
            reason = c.get("reason", "").upper().replace(" ", "")
            base = {"game_date": cur["date"], "game_time": cur["time"], "matchup": cur["matchup"], "team": cur["team"]}
            if "NOTYETSUBMITTED" in reason:
                rows.append({**base, "player_name": "", "status": NOT_SUBMITTED})
            elif c.get("player") and c.get("status"):
                rows.append({**base, "player_name": to_first_last(c["player"]), "status": c["status"].strip()})
    return rows


def parse_pdf(path: Path) -> list[dict]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        pages = [p.extract_words(x_tolerance=1.5) for p in pdf.pages]
    return parse_words(pages)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def report_url(ts_et: datetime) -> str:
    return f"{BASE_URL}/Injury-Report_{ts_et:%Y-%m-%d}_{ts_et:%I_%M%p}.pdf"


def _report_urls(ts_et: datetime) -> list[str]:
    """Current 15-minute filename, plus the hourly one used until ~Dec 2025 for on-the-hour times."""
    urls = [report_url(ts_et)]
    if ts_et.minute == 0:
        urls.append(f"{BASE_URL}/Injury-Report_{ts_et:%Y-%m-%d}_{ts_et:%I%p}.pdf")
    return urls


def _floor_step(ts: datetime) -> datetime:
    return ts.replace(minute=ts.minute - ts.minute % STEP_MIN, second=0, microsecond=0)


def fetch_report(ts_et: datetime, cache_dir: Path = CACHE_DIR, session: requests.Session | None = None) -> Path | None:
    """Download (or reuse) the report published at ``ts_et``; None when it does not exist."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(report_url(ts_et)).name
    missing = path.with_suffix(".missing")
    if path.exists():
        return path
    if missing.exists():
        return None
    for url in _report_urls(ts_et):
        r = (session or requests).get(url, timeout=30)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            path.write_bytes(r.content)
            return path
        if r.status_code not in (403, 404):
            r.raise_for_status()
    missing.write_text("403/404")
    return None


def latest_report_before(
    cutoff_et: datetime, *, cache_dir: Path = CACHE_DIR, max_steps: int = 16, session: requests.Session | None = None
) -> tuple[datetime, Path] | None:
    """The newest report published at or before ``cutoff_et`` (searching back ``max_steps`` × 15 min)."""
    ts = _floor_step(cutoff_et)
    for _ in range(max_steps):
        p = fetch_report(ts, cache_dir, session)
        if p is not None:
            return ts, p
        ts -= timedelta(minutes=STEP_MIN)
    return None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_report(engine, report_ts: datetime, rows: Iterable[dict]) -> int:
    ts = report_ts.strftime("%Y-%m-%dT%H:%M")
    df = pd.DataFrame(list(rows))
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE IF NOT EXISTS {TABLE} (report_ts TEXT, game_date TEXT, game_time TEXT, "
                          "matchup TEXT, team TEXT, player_name TEXT, status TEXT)"))
        conn.execute(text(f"DELETE FROM {TABLE} WHERE report_ts = :ts"), {"ts": ts})
    if df.empty:
        return 0
    df.insert(0, "report_ts", ts)
    df[["report_ts", "game_date", "game_time", "matchup", "team", "player_name", "status"]].to_sql(
        TABLE, engine, if_exists="append", index=False)
    return len(df)


def load_reports(engine, start: str, end: str) -> pd.DataFrame:
    try:
        return pd.read_sql(text(f"SELECT * FROM {TABLE} WHERE game_date BETWEEN :s AND :e"), engine,
                           params={"s": start, "e": end})
    except Exception:
        return pd.DataFrame(columns=["report_ts", "game_date", "game_time", "matchup", "team", "player_name", "status"])


def backfill(start: str, end: str, *, lead_min: int, engine, schedule: pd.DataFrame,
             cache_dir: Path = CACHE_DIR) -> int:
    """
    Store, for every game in ``schedule`` (GAME_DATE_EST, TIP_UTC) between start and
    end, the latest report published at least ``lead_min`` before its tip.
    """
    sch = schedule[(schedule["GAME_DATE_EST"] >= start) & (schedule["GAME_DATE_EST"] <= end)].copy()
    tips = pd.to_datetime(sch["TIP_UTC"], utc=True, errors="coerce").dropna()
    cutoffs = sorted({_floor_step(t.tz_convert(ET).to_pydatetime().replace(tzinfo=None) - timedelta(minutes=lead_min))
                      for t in tips})
    session = requests.Session()
    stored: set[datetime] = set()
    n = 0
    for cut in cutoffs:
        found = latest_report_before(cut, cache_dir=cache_dir, session=session)
        if found is None:
            log.warning("No injury report found at or before %s", cut)
            continue
        ts, path = found
        if ts in stored:
            continue
        try:
            n += store_report(engine, ts, parse_pdf(path))
        except Exception as e:  # one unreadable PDF must not stop the backfill
            log.warning("Could not parse %s: %s", path.name, e)
            continue
        stored.add(ts)
    return n


# ---------------------------------------------------------------------------
# Knowable absences
# ---------------------------------------------------------------------------

def pregame_out_by_game(
    reports: pd.DataFrame,
    games: pd.DataFrame,
    roster: dict[int, dict[int, str]],
) -> dict[tuple[str, int], set[int]]:
    """
    {(GAME_ID, TEAM_ID): player ids listed Out} from the latest report at or before
    each game's ``cutoff`` in which that team had filed. ``games`` has GAME_ID,
    TEAM_ID, tricode, matchup, game_date, cutoff (ET, 'YYYY-MM-DDTHH:MM').
    Games no report covers before the cutoff are left out (callers fall back).
    ``roster`` maps TEAM_ID -> {PLAYER_ID: name} for name resolution.
    """
    from identity_bridge import PlayerIndex

    if reports.empty or games.empty:
        return {}
    rep = reports.set_index(["game_date", "matchup"]).sort_index()
    out: dict[tuple[str, int], set[int]] = {}
    for g in games.itertuples():
        key = (g.game_date, g.matchup)
        if key not in rep.index:
            continue
        rg = rep.loc[[key]]
        rg = rg[rg["report_ts"] <= g.cutoff]
        if rg.empty:
            continue
        names: set[str] = set()
        for ts in sorted(rg["report_ts"].unique(), reverse=True):
            tr = rg[(rg["report_ts"] == ts) & (rg["team"] == g.tricode)]
            if (tr["status"] == NOT_SUBMITTED).any():
                continue
            names = {n for n, st in zip(tr["player_name"], tr["status"]) if str(st).lower() in OUT_STATUSES}
            break
        idx = PlayerIndex((pid, nm) for pid, nm in (roster.get(int(g.TEAM_ID)) or {}).items())
        out[(str(g.GAME_ID), int(g.TEAM_ID))] = {pid for n in names if (pid := idx.resolve(n))}
    return out


def pregame_out_for_gamelogs(gl: pd.DataFrame, *, lead_min: int = 60, engine=None) -> dict[tuple[str, int], set[int]]:
    """
    Knowable-before-tip absences for every scheduled game the stored reports cover:
    {(GAME_ID, TEAM_ID): {PLAYER_ID}} using the latest report at least ``lead_min``
    before tip. Names resolve against everyone who has appeared for the team in ``gl``.
    """
    import data_engine as de

    engine = engine or de._get_engine()
    try:
        sch = pd.read_sql(text("SELECT GAME_ID, GAME_DATE_EST, TIP_UTC, AWAY_TRICODE, HOME_TRICODE, "
                               "AWAY_TEAM_ID, HOME_TEAM_ID FROM schedule"), engine)
    except Exception:
        return {}
    if sch.empty:
        return {}
    reports = load_reports(engine, str(sch["GAME_DATE_EST"].min()), str(sch["GAME_DATE_EST"].max()))
    if reports.empty:
        return {}
    sch = sch[sch["GAME_DATE_EST"].isin(set(reports["game_date"]))].copy()
    tips = pd.to_datetime(sch["TIP_UTC"], utc=True, errors="coerce")
    sch = sch[tips.notna()]
    cut = (tips[tips.notna()].dt.tz_convert(ET) - pd.Timedelta(minutes=lead_min)).dt.strftime("%Y-%m-%dT%H:%M")
    sch["cutoff"] = cut.values
    sch["matchup"] = sch["AWAY_TRICODE"] + "@" + sch["HOME_TRICODE"]
    games = pd.concat([
        sch.assign(TEAM_ID=sch["AWAY_TEAM_ID"], tricode=sch["AWAY_TRICODE"]),
        sch.assign(TEAM_ID=sch["HOME_TEAM_ID"], tricode=sch["HOME_TRICODE"]),
    ]).rename(columns={"GAME_DATE_EST": "game_date"})
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    roster: dict[int, dict[int, str]] = {}
    for pid, name, tid in gl[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]].drop_duplicates().itertuples(index=False):
        roster.setdefault(int(tid), {})[int(pid)] = str(name)
    return pregame_out_by_game(reports, games[["GAME_ID", "TEAM_ID", "tricode", "matchup", "game_date", "cutoff"]],
                               roster)


def current_out(game_date: str, *, now_et: datetime | None = None) -> dict[str, set[str]] | None:
    """
    Live: {tricode: {normalised names listed Out}} for ``game_date`` from the newest
    report published in the last 4 hours, or None when none can be fetched.
    """
    from identity_bridge import norm_player_name

    now = now_et or datetime.now(ET).replace(tzinfo=None)
    try:
        found = latest_report_before(now, max_steps=16)
    except requests.RequestException as e:
        log.warning("Injury report unavailable (%s)", e)
        return None
    if found is None:
        return None
    rows = parse_pdf(found[1])
    out: dict[str, set[str]] = {}
    for r in rows:
        if r["game_date"] == game_date and str(r["status"]).lower() in OUT_STATUSES:
            out.setdefault(r["team"], set()).add(norm_player_name(r["player_name"]))
    log.info("Injury report %s: %d team(s) with players out on %s", found[0], len(out), game_date)
    return out
