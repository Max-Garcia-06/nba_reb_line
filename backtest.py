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
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config import BACKTEST_FILL_MODEL, EDGE_THRESHOLD, MIN_P, TAIL_EDGE_MULT, TAIL_P_CUTOFF
from edge_detector import apply_flow_guard, scan_for_edges
from kalshi_bridge import MarketLine
from probability_engine import calculate_probabilities
from trading_stack import fill_probability, finalize_signals

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
    fill_prob: float = 1.0


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


def settle(side: str, line: float, actual_reb: int) -> bool:
    return actual_reb > float(line) if side == "yes" else actual_reb <= float(line)


def pnl_per_contract(won: bool, price: float, fee: float = 0.0) -> float:
    """Long one contract of the chosen side at its own price."""
    return (1.0 - price if won else -price) - fee


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
    use_fill_model: bool | None = None,
) -> BacktestReport | None:
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
    fill_m = BACKTEST_FILL_MODEL if use_fill_model is None else use_fill_model
    trades = []
    for s in signals:
        reb = int(reb_by_ticker[s.ticker])
        won = settle(s.recommended_side, s.kalshi_line, reb)
        fp = fill_probability(s.book_spread, side=s.recommended_side) if fill_m else 1.0
        pnl = s.recommended_contracts * pnl_per_contract(won, s.limit_price, s.fee_per_contract) * fp
        trades.append(BacktestTrade(
            game_date=game_date, player_name=s.player_name, ticker=s.ticker, side=s.recommended_side,
            line=float(s.kalshi_line), limit_price=float(s.limit_price), contracts=int(s.recommended_contracts),
            fee_per_contract=float(s.fee_per_contract), p_model=float(s.p_model), edge=float(s.edge),
            ev=float(s.ev), actual_reb=reb, won=won, pnl_usd=float(pnl), fill_prob=float(fp)))

    total_pnl = sum(t.pnl_usd for t in trades)
    total_cost = sum(t.limit_price * t.contracts * t.fill_prob for t in trades)
    by_line: dict[float, dict[str, float]] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
    for t in trades:
        b = by_line[t.line]
        b["n"] += 1; b["pnl"] += t.pnl_usd; b["wins"] += int(t.won)
    return BacktestReport(
        game_date=game_date, n_markets=len(lines), n_signals=len(signals), n_trades=len(trades),
        total_pnl=float(total_pnl), total_cost=float(total_cost),
        roi_pct=float(total_pnl / total_cost * 100) if total_cost > 0 else 0.0,
        win_rate=sum(t.won for t in trades) / len(trades) if trades else 0.0,
        mean_clv=None, trades=trades, by_line=dict(by_line), pit_as_of="weekly" if pit_train else "saved")


def run_backtest_range(start: str, end: str, **kwargs: Any) -> list[BacktestReport]:
    reports = []
    for d in pd.date_range(start, end, freq="D"):
        rep = run_backtest_day(d.strftime("%Y-%m-%d"), **kwargs)
        if rep is not None:
            reports.append(rep)
    return reports
