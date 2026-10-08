"""
kalshi_history.py
-----------------
Pull settled KXNBAREB markets and their hourly bid/ask candlesticks from
Kalshi's public /historical endpoints into SQLite.

Why: mlb_tb_line's most important finding (SYSTEM.md §2, §6, §8) was that the
market beat the model, and that the only unbiased way to see that is to score
the model against the book on the FULL slate of markets, not just on the
trades it chose. MLB had to collect its own snapshots for weeks before it
could do that. Kalshi keeps the full 2025-26 NBA rebound history, so here we
can backfill it and score candidate models against the market before ever
staking capital.

Tables
------
kalshi_markets  one row per market ticker (strike, result, player, teams)
kalshi_candles  hourly candles per ticker: yes bid/ask close, last price, volume

No auth needed — these endpoints are public. Resumable: tickers with candles
already stored are skipped.

Usage
-----
  python kalshi_history.py markets
  python kalshi_history.py candles [--workers 6]
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, text

from config import DB_PATH

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXNBAREB"

# Candle window: markets open days early but trade mostly on game day.
# We keep [tip - 36h, tip] so pregame snapshots at any reasonable lead exist.
LOOKBACK_HOURS = 36

_EVENT_RE = re.compile(r"^[A-Z]+-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)$")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


def _engine():
    return create_engine(f"sqlite:///{DB_PATH}")


def _ts(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def _f(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


class _RateLimiter:
    """Token-ish limiter shared across worker threads (Kalshi basic tier ~20 req/s)."""

    def __init__(self, per_sec: float):
        self.min_gap = 1.0 / per_sec
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.min_gap
        if t > now:
            time.sleep(t - now)


_limiter = _RateLimiter(per_sec=5)  # unauthenticated historical endpoints 429 above ~5/s
_session = requests.Session()


def _get(path: str, params: dict) -> dict:
    for attempt in range(6):
        _limiter.wait()
        try:
            r = _session.get(f"{BASE}{path}", params=params, timeout=30)
        except requests.RequestException as e:
            log.warning(f"{path} network error ({e}); retry {attempt + 1}")
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(30, 2 ** attempt))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {path} failed after retries")


# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------

def parse_event_ticker(event_ticker: str) -> tuple[str | None, str | None]:
    """KXNBAREB-26JUN13NYKSAS -> ('2026-06-13', 'NYKSAS'). Date is the US game date."""
    m = _EVENT_RE.match(event_ticker or "")
    if not m:
        return None, None
    yy, mon, dd, teams = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        return None, None
    return f"20{yy}-{month:02d}-{int(dd):02d}", teams


def player_from_title(title: str) -> str:
    # "Victor Wembanyama: 8+ rebounds" (newer) or "Dyson Daniels records 8+ rebounds" (older)
    return re.split(r":| records ", title, maxsplit=1)[0].strip()


def fetch_markets() -> pd.DataFrame:
    rows, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = _get("/historical/markets", params)
        ms = page.get("markets", [])
        for m in ms:
            game_date, teams = parse_event_ticker(m.get("event_ticker", ""))
            rows.append({
                "ticker": m["ticker"],
                "event_ticker": m.get("event_ticker"),
                "game_date": game_date,
                "teams_code": teams,
                "title": m.get("title", ""),
                "player_name": player_from_title(m.get("title", "")),
                "kalshi_player_key": (m.get("custom_strike") or {}).get("basketball_player"),
                "line": _f(m.get("floor_strike")),
                "result": m.get("result"),
                "volume": _f(m.get("volume_fp")),
                "open_time": m.get("open_time"),
                "close_time": m.get("close_time"),
                "occurrence_time": m.get("occurrence_datetime"),
                "status": m.get("status"),
            })
        cursor = page.get("cursor")
        if not cursor or not ms:
            break
    df = pd.DataFrame(rows).drop_duplicates("ticker", keep="last")
    log.info(f"Fetched {len(df):,} {SERIES} historical markets")
    return df


def store_markets(df: pd.DataFrame) -> None:
    eng = _engine()
    with eng.begin() as conn:
        if inspect(conn).has_table("kalshi_markets"):
            conn.execute(text("DELETE FROM kalshi_markets"))
        df.to_sql("kalshi_markets", conn, if_exists="append", index=False, chunksize=1000)
    log.info(f"  -> wrote {len(df):,} rows to [kalshi_markets]")


def load_markets() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM kalshi_markets", _engine())


# ---------------------------------------------------------------------------
# Candles
# ---------------------------------------------------------------------------

def _candle_window(m: dict, tip_utc: str | None) -> tuple[int, int]:
    """[max(open, tip-LOOKBACK), tip] — falls back to occurrence-3h when tip is unknown."""
    if tip_utc:
        end = _ts(tip_utc)
    else:
        end = _ts(m["occurrence_time"]) - 3 * 3600
    start = max(_ts(m["open_time"]), end - LOOKBACK_HOURS * 3600)
    # Round to hour boundaries so the hourly candle that closes at tip is included.
    return start - start % 3600, end - end % 3600 + 3600


def fetch_candles(ticker: str, start_ts: int, end_ts: int) -> list[dict]:
    if end_ts <= start_ts:
        return []
    data = _get(f"/historical/markets/{ticker}/candlesticks",
                {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60})
    out = []
    for c in data.get("candlesticks", []):
        bid, ask, px = c.get("yes_bid") or {}, c.get("yes_ask") or {}, c.get("price") or {}
        out.append({
            "ticker": ticker,
            "end_ts": int(c["end_period_ts"]),
            "yes_bid": _f(bid.get("close")),
            "yes_ask": _f(ask.get("close")),
            "last_price": _f(px.get("close")) if px.get("close") is not None else _f(px.get("previous")),
            "volume": _f(c.get("volume")),
            "open_interest": _f(c.get("open_interest")),
        })
    return out


def _tip_lookup() -> dict[tuple[str, str], str]:
    """(game_date_est, AWAYHOME tricodes) -> TIP_UTC from the nba_api schedule table."""
    try:
        sch = pd.read_sql("SELECT GAME_DATE_EST, TIP_UTC, AWAY_TRICODE, HOME_TRICODE FROM schedule", _engine())
    except Exception:
        log.warning("No schedule table yet — falling back to occurrence_time - 3h for tip.")
        return {}
    return {(r.GAME_DATE_EST, f"{r.AWAY_TRICODE}{r.HOME_TRICODE}"): r.TIP_UTC for r in sch.itertuples()}


def backfill_candles(workers: int = 6, flush_every: int = 500) -> None:
    eng = _engine()
    markets = load_markets()
    done: set[str] = set()
    if inspect(eng).has_table("kalshi_candles_done"):
        done = set(pd.read_sql("SELECT ticker FROM kalshi_candles_done", eng)["ticker"])
    todo = markets[~markets["ticker"].isin(done)].to_dict("records")
    tips = _tip_lookup()
    log.info(f"Candles: {len(done):,} done, {len(todo):,} to fetch ({workers} workers)")

    buf, done_buf, n = [], [], 0

    def flush():
        if not done_buf:
            return
        with eng.begin() as conn:
            if buf:
                pd.DataFrame(buf).to_sql("kalshi_candles", conn, if_exists="append", index=False, chunksize=2000)
            pd.DataFrame({"ticker": done_buf}).to_sql("kalshi_candles_done", conn, if_exists="append", index=False)
        buf.clear()
        done_buf.clear()

    def job(m):
        tip = tips.get((m["game_date"], m["teams_code"]))
        s, e = _candle_window(m, tip)
        return m["ticker"], fetch_candles(m["ticker"], s, e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, m) for m in todo]
        for fut in as_completed(futs):
            try:
                ticker, rows = fut.result()
            except Exception as e:
                log.warning(f"candle fetch failed: {e}")
                continue
            buf.extend(rows)
            done_buf.append(ticker)
            n += 1
            if n % flush_every == 0:
                flush()
                log.info(f"  {n:,}/{len(todo):,} tickers")
    flush()
    log.info("Candle backfill complete.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "markets"
    if cmd == "markets":
        store_markets(fetch_markets())
    elif cmd == "candles":
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
        backfill_candles(workers=w)
    else:
        print("usage: python kalshi_history.py markets|candles [--workers N]")
