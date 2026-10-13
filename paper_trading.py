"""
paper_trading.py
----------------
Dry-run paper journal: what the scan *would* have ordered, kept in
``data/paper/trades_YYYY-MM-DD.jsonl`` in the live journal's format so `mark`
and `report-range --paper` can judge a dry-run trial on fills, P&L and CLV.

Paper fills are simulated from the marks: a resting order counts as filled
only when a later pre-tip mark shows the opposing quote at or through its
limit (no trade data, so trade-through fills are missed — conservative).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from journal_reader import load_jsonl_rows, placed_with_order_id
from trade_journal import TradeRow, append_row, journal_path


def record_paper_orders(game_date: str, signals: list) -> int:
    """Journal signals as paper orders; one per (ticker, side) per day, like the live ledger."""
    seen = {(r.get("ticker"), r.get("side")) for r in placed_with_order_id(load_jsonl_rows(journal_path(game_date, paper=True)))}
    n = 0
    for s in signals:
        if (s.ticker, s.recommended_side) in seen:
            continue
        append_row(
            game_date,
            TradeRow(
                game_date=game_date,
                ticker=s.ticker,
                side=s.recommended_side,
                action="buy",
                contracts=int(s.recommended_contracts),
                limit_price=float(s.limit_price),
                order_id=f"paper-{uuid.uuid4().hex[:12]}",
                player_name=s.player_name,
                kalshi_line=float(s.kalshi_line),
                games_played=int(s.games_played),
                predicted_lambda=float(s.predicted_lambda),
                p_model=float(s.p_model),
                p_model_raw=float(s.p_model_raw),
                p_model_cal=float(s.p_model_cal),
                p_market=float(s.p_market),
                fee_per_contract=float(s.fee_per_contract),
                edge=float(s.edge),
                ev=float(s.ev),
                expected_pnl=float(s.edge) * int(s.recommended_contracts),
                book_bid=float(s.book_bid),
                book_ask=float(s.book_ask),
                book_spread=float(s.book_spread),
                note="post-submit",
                success=True,
            ).to_dict(),
            paper=True,
        )
        seen.add((s.ticker, s.recommended_side))
        n += 1
    return n


def _epoch(ts: str) -> int:
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return int(dt.timestamp())


def paper_fill_rows(placed: list[dict[str, Any]], mark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Synthetic ``note=fill`` rows for paper orders, filled per :func:`backtest.simulate_fill` on marks."""
    from backtest import Candle, simulate_fill

    marks_by_oid: dict[str, list[Candle]] = {}
    for m in mark_rows:
        try:
            ts = _epoch(m.get("ts", ""))
        except ValueError:
            continue
        marks_by_oid.setdefault(str(m.get("order_id")), []).append(
            Candle(ts, float(m.get("mark_yes_bid") or 0.0), float(m.get("mark_yes_ask") or 0.0)))
    out = []
    for r in placed:
        oid = str(r.get("order_id"))
        limit = float(r.get("limit_price", 0.0) or 0.0)
        try:
            placed_ts = _epoch(r.get("ts", ""))
        except ValueError:
            continue
        filled = simulate_fill(
            str(r.get("side", "")), limit, int(r.get("contracts", 0) or 0),
            is_taker=limit >= float(r.get("book_ask", 1.0) or 1.0) - 1e-9,
            later=marks_by_oid.get(oid, []), decision_ts=placed_ts, tip_ts=2**62,  # marks are pre-tip by construction
        )
        out.append({**r, "note": "fill", "filled_contracts": int(filled), "avg_fill_price": limit,
                    "order_status": "paper"})
    return out
