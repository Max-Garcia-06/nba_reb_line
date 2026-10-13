"""
backtest.py
-----------
Replay the live scan stack — calibration, market blend, fee-aware edge vs
maker/taker limit, Kelly sizing, portfolio caps — over stored books and settle
against actual rebounds.

Books come from live snapshots when present, else the Kalshi historical
candles (kalshi_history.py), so the whole 2025-26 season is replayable. Models
are point-in-time by default (weekly as-of retrains, shared cache with
model_vs_market).

Fixes vs mlb_tb_line/backtest.py
--------------------------------
* `limit_price` on a NO signal is a NO-contract price (edge_detector quotes
  the NO book). MLB's `_pnl`/CLV treated it as a YES price, inverting P&L and
  CLV on every NO trade. Here every side is settled at its own price.
* P&L is net of the Kalshi fee carried on each signal.

Realism (2026-10)
-----------------
* Books are pregame only: snapshots and candles at least HISTORY_LEAD_MIN before tip.
* Blend weights are walk-forward by default: each week trades on weights fit
  only on full-slate scores from days before that week (``blend="saved"``
  uses models/blend_meta*.json, which is in-sample for any day it was fit on).
* Fills come from what traded after the decision and before tip
  (:func:`simulate_fill`), not from the book spread. ``fill_model="none"`` is
  the optimistic everything-fills bound.
* ``zero_vacuum`` zeroes the teammate-absence features: history knows every
  late scratch, live does not, so this bounds that hindsight.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

from config import BACKTEST_FILL_MODEL, EDGE_THRESHOLD, MIN_P, TAIL_EDGE_MULT, TAIL_P_CUTOFF
from edge_detector import apply_flow_guard, scan_for_edges
from kalshi_bridge import MarketLine
from market_blend import fit_blend_weights, pinned_blend_weights
from probability_engine import calculate_probabilities
from trading_stack import finalize_signals

VACUUM_COLS = ("vacuum_reb", "vacuum_min", "vacuum_n")

log = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    game_date: str
    player_name: str
    ticker: str
    side: str
    line: float
    limit_price: float          # price of the contract bought (YES price or NO price)
    contracts: int
    fee_per_contract: float
    p_model: float
    edge: float
    ev: float
    actual_reb: int
    won: bool
    pnl_usd: float
    filled: int = 0
    clv: float | None = None    # close side-mid minus entry, per contract (filled only)


@dataclass
class BacktestReport:
    game_date: str
    n_markets: int
    n_signals: int
    n_trades: int
    total_pnl: float
    total_cost: float
    roi_pct: float
    win_rate: float
    mean_clv: float | None
    trades: list[BacktestTrade]
    by_line: dict[float, dict[str, float]] = field(default_factory=dict)
    pit_as_of: str = ""
    n_filled: int = 0


def settle(side: str, line: float, actual_reb: int) -> bool:
    return actual_reb > float(line) if side == "yes" else actual_reb <= float(line)


def pnl_per_contract(won: bool, price: float, fee: float = 0.0) -> float:
    """Long one contract of the chosen side at its own price."""
    return (1.0 - price if won else -price) - fee


@dataclass(frozen=True)
class Candle:
    """A book observation after the decision: hourly candle, or a later snapshot (no trades)."""
    end_ts: int
    yes_bid: float
    yes_ask: float
    last_price: float = 0.0
    volume: float = 0.0


def simulate_fill(
    side: str,
    limit: float,
    contracts: int,
    *,
    is_taker: bool,
    later: list[Candle],
    decision_ts: int,
    tip_ts: int,
) -> int:
    """
    Contracts a limit buy would have filled between decision and tip.

    A taker order fills at once. A resting order fills only on evidence that
    someone sold at our price: the opposing quote moved through the limit (fill
    in full), or a trade printed at or through it (fill up to the traded
    volume). For NO at ``limit`` that means YES at or above ``1 - limit``.
    """
    if contracts <= 0:
        return 0
    if is_taker:
        return int(contracts)
    yes_side = str(side).lower() == "yes"
    traded = 0.0
    for c in later:
        if not (decision_ts < c.end_ts <= tip_ts):
            continue
        if yes_side:
            quote_through = 0 < c.yes_ask <= limit + 1e-9
            trade_through = c.volume > 0 and 0 < c.last_price <= limit + 1e-9
        else:
            quote_through = c.yes_bid > 0 and c.yes_bid >= 1.0 - limit - 1e-9
            trade_through = c.volume > 0 and c.last_price >= 1.0 - limit - 1e-9
        if quote_through:
            return int(contracts)
        if trade_through:
            traded += c.volume
    return int(min(contracts, int(traded)))


def _close_side_mid(later: list[Candle], tip_ts: int, side: str) -> float | None:
    """Last two-sided mid at or before tip, for the given side."""
    pre = [c for c in later if c.end_ts <= tip_ts and c.yes_bid > 0 and c.yes_ask > 0]
    if not pre:
        return None
    c = max(pre, key=lambda x: x.end_ts)
    mid = (c.yes_bid + c.yes_ask) / 2.0
    return mid if str(side).lower() == "yes" else 1.0 - mid


def _later_books(rows: pd.DataFrame, game_date: str) -> dict[str, list[Candle]]:
    """Post-decision observations per ticker: Kalshi candles, else later snapshots."""
    import model_vs_market as mvm

    tickers = list(rows["ticker"])
    out: dict[str, list[Candle]] = defaultdict(list)
    if (rows.get("source") == "snapshot").any():
        from market_snapshots import _parse_captured_at_utc, load_snapshots

        for sn in load_snapshots(game_date, latest_only=False):
            cap = _parse_captured_at_utc(sn.captured_at)
            if cap is not None:
                out[sn.ticker].append(Candle(int(cap.timestamp()), float(sn.yes_bid), float(sn.yes_ask)))
        return out
    try:
        eng = create_engine(f"sqlite:///{mvm.DB_PATH}")
        c = pd.read_sql(
            text("SELECT ticker, end_ts, yes_bid, yes_ask, last_price, volume FROM kalshi_candles WHERE ticker IN "
                 "(SELECT ticker FROM kalshi_markets WHERE game_date = :d)"),
            eng, params={"d": game_date})
    except Exception as e:
        log.debug("later candles unavailable for %s: %s", game_date, e)
        return out
    c = c[c["ticker"].isin(tickers)]
    for r in c.itertuples():
        out[r.ticker].append(Candle(int(r.end_ts), float(r.yes_bid or 0), float(r.yes_ask or 0),
                                    float(r.last_price or 0), float(r.volume or 0)))
    return out


def day_bootstrap_roi_ci(
    days: list[tuple[float, float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """ROI % point estimate and (1-alpha) CI, resampling whole days (pnl, cost)."""
    days = [d for d in days if d[1] > 0]
    if not days:
        return 0.0, 0.0, 0.0
    point = sum(p for p, _ in days) / sum(c for _, c in days) * 100
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        smp = [days[rng.randrange(len(days))] for _ in days]
        cost = sum(c for _, c in smp)
        stats.append(sum(p for p, _ in smp) / cost * 100 if cost > 0 else 0.0)
    stats.sort()
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def _market_lines(rows: pd.DataFrame) -> list[MarketLine]:
    out = []
    for r in rows.itertuples():
        out.append(MarketLine(
            ticker=r.ticker, player_name=r.player_name, player_id=int(r.PLAYER_ID),
            game_date=r.game_date, line=float(r.line),
            yes_ask=float(r.yes_ask), yes_bid=float(r.yes_bid),
            no_ask=round(1.0 - float(r.yes_bid), 4), no_bid=round(1.0 - float(r.yes_ask), 4),
            event_ticker=r.event_ticker,
        ))
    return out


def run_backtest_day(
    game_date: str,
    *,
    bankroll: float = 1000.0,
    edge_threshold: float = EDGE_THRESHOLD,
    min_p: float = MIN_P,
    tail_p_cutoff: float = TAIL_P_CUTOFF,
    tail_edge_mult: float = TAIL_EDGE_MULT,
    one_per_player: bool = True,
    max_signals: int | None = None,
    max_contracts: int | None = 250,
    pit_train: bool = True,
    fill_model: str | None = None,
    zero_vacuum: bool = False,
) -> BacktestReport | None:
    """Replay one day under whatever blend weights are in effect (see run_backtest_range)."""
    import model_vs_market as mvm

    rows = mvm._market_rows_snapshots(game_date, earliest=False)
    if rows.empty:
        rows = mvm._market_rows_history(game_date)
    if rows.empty:
        return None
    rows = rows[(rows["yes_bid"] > 0) & (rows["yes_ask"] < 1) & (rows["yes_ask"] > rows["yes_bid"])]
    rows = mvm._resolve(rows, game_date)
    if rows.empty:
        return None
    rows["game_date"] = game_date

    feats = mvm._features()
    feats = feats[pd.to_datetime(feats["GAME_DATE"]).dt.strftime("%Y-%m-%d") == game_date].set_index(
        ["PLAYER_ID", "GAME_ID"])
    if zero_vacuum:
        feats = feats.copy()
        feats[list(VACUUM_COLS)] = 0
    rows = rows[[k in feats.index for k in zip(rows["PLAYER_ID"], rows["GAME_ID"])]].reset_index(drop=True)
    if rows.empty:
        return None

    model = mvm._pit_model(game_date, pit_train)
    uniq = feats.loc[sorted(set(zip(rows["PLAYER_ID"], rows["GAME_ID"])))]
    pmf = model.pmf(uniq.reset_index())
    pmf_by = {k: pmf[i] for i, k in enumerate(uniq.index)}
    preds = [{"player_id": r.PLAYER_ID, "player_name": r.player_name, "game_date": game_date,
              "kalshi_line": float(r.line), "pmf": pmf_by[(r.PLAYER_ID, r.GAME_ID)],
              "games_played": int(feats.loc[(r.PLAYER_ID, r.GAME_ID), "games_played"])}
             for r in rows.itertuples()]
    prs = calculate_probabilities(preds)
    mvm.calibrate_results(prs, game_date, pit_train)
    lines = _market_lines(rows)

    signals = scan_for_edges(prs, lines, bankroll, edge_threshold=edge_threshold, min_p=min_p,
                             tail_p_cutoff=tail_p_cutoff, tail_edge_mult=tail_edge_mult)
    signals = apply_flow_guard(signals, lines)
    signals = finalize_signals(signals, bankroll=bankroll, one_per_player=one_per_player,
                               max_signals=max_signals, max_contracts=max_contracts)

    reb_by_ticker = dict(zip(rows["ticker"], rows["REB"]))
    dec_by_ticker = dict(zip(rows["ticker"], rows["decision_ts"]))
    tip_by_ticker = dict(zip(rows["ticker"], rows["tip_ts"]))
    mode = (fill_model or BACKTEST_FILL_MODEL).lower()
    later = _later_books(rows, game_date) if signals else {}
    trades = []
    for s in signals:
        reb = int(reb_by_ticker[s.ticker])
        won = settle(s.recommended_side, s.kalshi_line, reb)
        tip_ts = int(tip_by_ticker[s.ticker])
        if mode == "none":
            filled = int(s.recommended_contracts)
        else:
            filled = simulate_fill(
                s.recommended_side, float(s.limit_price), int(s.recommended_contracts),
                is_taker=float(s.limit_price) >= float(s.book_ask) - 1e-9,
                later=later.get(s.ticker, []), decision_ts=int(dec_by_ticker[s.ticker]), tip_ts=tip_ts,
            )
        pnl = filled * pnl_per_contract(won, s.limit_price, s.fee_per_contract)
        close = _close_side_mid(later.get(s.ticker, []), tip_ts, s.recommended_side)
        clv = (close - float(s.limit_price)) if (close is not None and filled > 0) else None
        trades.append(BacktestTrade(
            game_date=game_date, player_name=s.player_name, ticker=s.ticker, side=s.recommended_side,
            line=float(s.kalshi_line), limit_price=float(s.limit_price), contracts=int(s.recommended_contracts),
            fee_per_contract=float(s.fee_per_contract), p_model=float(s.p_model), edge=float(s.edge),
            ev=float(s.ev), actual_reb=reb, won=won, pnl_usd=float(pnl), filled=int(filled), clv=clv))

    filled_trades = [t for t in trades if t.filled > 0]
    total_pnl = sum(t.pnl_usd for t in filled_trades)
    total_cost = sum(t.limit_price * t.filled for t in filled_trades)
    clv_ctr = sum(t.filled for t in filled_trades if t.clv is not None)
    mean_clv = (sum(t.clv * t.filled for t in filled_trades if t.clv is not None) / clv_ctr) if clv_ctr else None
    by_line: dict[float, dict[str, float]] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in filled_trades:
        b = by_line[t.line]
        b["n"] += 1; b["pnl"] += t.pnl_usd; b["wins"] += int(t.won)
    return BacktestReport(
        game_date=game_date, n_markets=len(lines), n_signals=len(signals), n_trades=len(trades),
        total_pnl=float(total_pnl), total_cost=float(total_cost),
        roi_pct=float(total_pnl / total_cost * 100) if total_cost > 0 else 0.0,
        win_rate=sum(t.won for t in filled_trades) / len(filled_trades) if filled_trades else 0.0,
        mean_clv=mean_clv, trades=trades, by_line=dict(by_line), pit_as_of="weekly" if pit_train else "saved",
        n_filled=len(filled_trades))


def run_backtest_range(
    start: str,
    end: str,
    *,
    blend: str = "walk_forward",
    blend_since: str | None = None,
    **kwargs: Any,
) -> list[BacktestReport]:
    """
    Backtest each day in ``[start, end]``.

    ``blend="walk_forward"``: every week trades on blend weights fit (see
    :func:`market_blend.fit_blend_weights`) on full-slate PIT scores from
    ``blend_since`` (default: a year before ``start``) up to the day before that
    week's Monday. ``blend="saved"`` uses models/blend_meta*.json as live does.
    """
    import model_vs_market as mvm

    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    if blend != "walk_forward":
        return [r for r in (run_backtest_day(d, **kwargs) for d in dates) if r is not None]

    since = blend_since or (pd.Timestamp(start) - timedelta(days=365)).strftime("%Y-%m-%d")
    eval_kw = {k: kwargs[k] for k in ("pit_train", "zero_vacuum") if k in kwargs}
    scored: list[dict] = []
    scored_until = pd.Timestamp(since)
    reports: list[BacktestReport] = []
    weights: dict[str, tuple[float, dict[str, float]]] = {}
    for d in dates:
        week = mvm._week_start(d)
        if week not in weights:
            while scored_until < pd.Timestamp(week):
                ds = scored_until.strftime("%Y-%m-%d")
                scored.extend({"p": r.p_model_cal, "m": r.p_market_mid, "y": r.y, "date": r.game_date}
                              for r in mvm.evaluate_day(ds, **eval_kw))
                scored_until += timedelta(days=1)
            prior = [r for r in scored if r["date"] < week]
            weights[week] = fit_blend_weights(prior)
            log.info("Walk-forward blend for week %s: %d rows, w=%.2f, segments=%s",
                     week, len(prior), weights[week][0], weights[week][1])
        with pinned_blend_weights(*weights[week]):
            rep = run_backtest_day(d, **kwargs)
        if rep is not None:
            reports.append(rep)
    return reports
