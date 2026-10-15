"""
game_markets.py
---------------
Kalshi NBA game totals (KXNBATOTAL: "Over L points scored") and spreads (KXNBASPREAD:
"<team> wins by over L points") → per-game market-implied median total and home margin.

History: market lists from /historical/markets (settled before Kalshi's historical
cutoff) plus /markets?status=settled (after it); hourly candles per ticker; the price
at decision time is the last candle closing at or before tip − DECISION_LEAD_MIN and
no more than STALE_SEC older. Live: current books from the public /markets endpoint.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import inspect, text

import kalshi_history as kh
from config import DECISION_LEAD_MIN

log = logging.getLogger(__name__)

SERIES = ("KXNBATOTAL", "KXNBASPREAD")
MARKETS_TABLE = "kalshi_game_markets"
CANDLES_TABLE = "kalshi_game_candles"
CANDLES_DONE = "kalshi_game_candles_done"
MIN_LADDER = 3
STALE_SEC = 2 * 3600
_TEAM_RE = re.compile(r"^([A-Z]+)\d+$")


def two_sided_mid(bid, ask) -> float:
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return np.nan
    return (bid + ask) / 2 if bid > 0 and ask < 1 and ask > bid else np.nan


def ladder_median(lines, p_over, min_lines: int = MIN_LADDER) -> float:
    """Line where a monotone (isotonic, decreasing) fit of P(X > line) crosses 0.5; NaN outside the ladder."""
    d = pd.DataFrame({"l": np.asarray(lines, float), "p": np.asarray(p_over, float)}).dropna()
    d = d.groupby("l", as_index=False)["p"].mean().sort_values("l")
    if len(d) < min_lines:
        return np.nan
    l = d["l"].to_numpy()
    p = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0).fit_transform(l, d["p"].to_numpy())
    if p[0] < 0.5 or p[-1] > 0.5:
        return np.nan
    i = int(np.nonzero(p <= 0.5)[0][0])
    if i == 0:
        return float(l[0])
    l0, l1, p0, p1 = l[i - 1], l[i], p[i - 1], p[i]
    return float(l0 + (p0 - 0.5) * (l1 - l0) / (p0 - p1)) if p0 != p1 else float((l0 + l1) / 2)


def _team(ticker: str) -> str:
    m = _TEAM_RE.match(str(ticker).split("-")[-1])
    return m.group(1) if m else ""


def event_features(rows: pd.DataFrame, home: str) -> dict:
    tot = rows[rows["series"] == "KXNBATOTAL"]
    sp = rows[rows["series"] == "KXNBASPREAD"]
    is_home = (sp["team"] == home).to_numpy()
    x = np.where(is_home, sp["line"], -sp["line"])      # home margin M
    p = np.where(is_home, sp["mid"], 1 - sp["mid"])     # P(M > x)
    return {"mkt_total": ladder_median(tot["line"], tot["mid"]), "mkt_home_margin": ladder_median(x, p)}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _rows(ms: list[dict], series: str) -> list[dict]:
    out = []
    for m in ms:
        game_date, teams = kh.parse_event_ticker(m.get("event_ticker", ""))
        out.append({"ticker": m["ticker"], "series": series, "event_ticker": m.get("event_ticker"),
                    "game_date": game_date, "teams_code": teams, "line": kh._f(m.get("floor_strike")),
                    "team": _team(m["ticker"]) if series == "KXNBASPREAD" else "",
                    "result": m.get("result"), "open_time": m.get("open_time"), "close_time": m.get("close_time"),
                    "occurrence_time": m.get("occurrence_datetime"), "status": m.get("status")})
    return out


def _paged(path: str, params: dict) -> list[dict]:
    out, cursor = [], None
    while True:
        p = dict(params, limit=1000, **({"cursor": cursor} if cursor else {}))
        page = kh._get(path, p)
        ms = page.get("markets", []) or []
        out.extend(ms)
        cursor = page.get("cursor")
        if not cursor or not ms:
            return out


def fetch_markets() -> pd.DataFrame:
    rows = []
    for s in SERIES:
        rows += _rows(_paged("/historical/markets", {"series_ticker": s}), s)
        rows += _rows(_paged("/markets", {"series_ticker": s, "status": "settled"}), s)
    df = pd.DataFrame(rows).drop_duplicates("ticker", keep="last")
    return df[df["game_date"].notna()]


def backfill(workers: int = 4) -> None:
    """Refresh the market list, then fetch candles for every settled ticker not yet done (resumable)."""
    mk = fetch_markets()
    eng = kh._engine()
    with eng.begin() as conn:
        if inspect(conn).has_table(MARKETS_TABLE):
            conn.execute(text(f"DELETE FROM {MARKETS_TABLE}"))
        mk.to_sql(MARKETS_TABLE, conn, if_exists="append", index=False, chunksize=1000)
    log.info(f"{len(mk):,} game markets stored")
    cutoff = kh._get("/historical/cutoff", {}).get("market_settled_ts", "")

    def path_for(m):
        hist = cutoff and str(m.get("close_time") or "") < cutoff
        return (f"/historical/markets/{m['ticker']}/candlesticks" if hist
                else f"/series/{m['series']}/markets/{m['ticker']}/candlesticks")

    kh.backfill_candles(workers=workers, markets=mk, table=CANDLES_TABLE, done_table=CANDLES_DONE,
                        lookback_hours=4, path_for=path_for)


def history_features(engine=None, lead_min: int | None = None) -> pd.DataFrame:
    engine = engine or kh._engine()
    lead = DECISION_LEAD_MIN if lead_min is None else lead_min
    cols = ["GAME_ID", "mkt_total", "mkt_home_margin"]
    if not (inspect(engine).has_table(MARKETS_TABLE) and inspect(engine).has_table(CANDLES_TABLE)):
        return pd.DataFrame(columns=cols)
    mk = pd.read_sql(text(f"SELECT ticker, series, game_date, teams_code, line, team FROM {MARKETS_TABLE}"), engine)
    sch = pd.read_sql(text("SELECT GAME_ID, GAME_DATE_EST, TIP_UTC, AWAY_TRICODE, HOME_TRICODE FROM schedule"), engine)
    sch["teams_code"] = sch["AWAY_TRICODE"] + sch["HOME_TRICODE"]
    mk = mk.merge(sch.rename(columns={"GAME_DATE_EST": "game_date"}), on=["game_date", "teams_code"])
    # Epoch seconds independent of the datetime unit (pandas 3 parses to microseconds).
    tip = pd.to_datetime(mk["TIP_UTC"], utc=True)
    mk["cut_ts"] = (tip - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1) - lead * 60
    cd = pd.read_sql(text(f"SELECT ticker, end_ts, yes_bid, yes_ask FROM {CANDLES_TABLE}"), engine)
    cd = cd.merge(mk[["ticker", "cut_ts"]], on="ticker")
    cd = cd[(cd["end_ts"] <= cd["cut_ts"]) & (cd["end_ts"] > cd["cut_ts"] - STALE_SEC)]
    last = cd.sort_values("end_ts").groupby("ticker").tail(1)
    last = last.assign(mid=[two_sided_mid(b, a) for b, a in zip(last["yes_bid"], last["yes_ask"])])
    mk = mk.merge(last[["ticker", "mid"]], on="ticker", how="left")
    out = [{"GAME_ID": str(gid), **event_features(g, home=g["HOME_TRICODE"].iloc[0])}
           for gid, g in mk.groupby("GAME_ID")]
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def live_features(game_date: str, schedule_index: dict[str, dict]) -> pd.DataFrame:
    """Current books for ``game_date`` events; schedule_index = data_engine.slate_schedule_index(game_date)."""
    import data_engine as de

    cols = ["GAME_ID", "mkt_total", "mkt_home_margin"]
    rows = []
    for s in SERIES:
        try:
            ms = _paged("/markets", {"series_ticker": s, "status": "open"})
        except Exception as e:  # treated like a missing ladder: NaN features, as in history
            log.warning("Game markets %s unavailable (%s)", s, e)
            continue
        by_ticker = {m["ticker"]: m for m in ms}
        for r in _rows(ms, s):
            m = by_ticker[r["ticker"]]
            r["mid"] = two_sided_mid(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"))
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[df["game_date"] == game_date]
    out = []
    for et, g in df.groupby("event_ticker"):
        mu = de.parse_kalshi_event_matchup(str(et))
        row = schedule_index.get(de.matchup_slug(*mu)) if mu else None
        if row:
            out.append({"GAME_ID": row["game_id"], **event_features(g, home=mu[1])})
    if not out:
        return pd.DataFrame(columns=cols)
    # a game has one totals event and one spread event: combine their features
    res = pd.DataFrame(out).groupby("GAME_ID", as_index=False).agg(
        {"mkt_total": "max", "mkt_home_margin": "max"})
    return res[cols]
