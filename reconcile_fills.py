"""
Sync Kalshi order status into trade journals as ``note=fill`` rows.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from journal_reader import (
    existing_fill_order_ids,
    journal_paths_in_date_range,
    load_jsonl_rows,
    placed_with_order_id,
)
from trade_journal import TradeRow, append_row, journal_path

log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# A slate's late games settle after ET midnight, so the fill window has to run past
# the calendar date it is journaled under.
FILL_WINDOW_HOURS = 36


@dataclass
class ReconcileResult:
    updated: int
    skipped: int
    errors: int
    days_processed: int = 0

    def merge(self, other: ReconcileResult) -> ReconcileResult:
        return ReconcileResult(
            updated=self.updated + other.updated,
            skipped=self.skipped + other.skipped,
            errors=self.errors + other.errors,
            days_processed=self.days_processed + other.days_processed,
        )


def _avg_fill_price_from_order(order: dict[str, Any], placed_row: dict[str, Any]) -> float:
    for k in ("avg_fill_price", "average_fill_price", "avg_price", "fill_price"):
        if order.get(k) is not None:
            return float(order[k])
    return float(placed_row.get("limit_price", 0.0))


def _fill_price(fill: dict[str, Any], side: str) -> float:
    """Price paid on ``side`` for one fill record, normalised to dollars."""
    keys = ("yes_price_dollars", "yes_price") if side == "yes" else ("no_price_dollars", "no_price")
    for k in keys:
        v = fill.get(k)
        if v is None:
            continue
        return float(v) if k.endswith("_dollars") else float(v) / 100.0
    return 0.0


def _fill_count(fill: dict[str, Any]) -> float:
    for k in ("count", "count_fp"):
        v = fill.get(k)
        if v is not None:
            return float(v)
    return 0.0


def _aggregate_fills(fills: list[dict[str, Any]], side: str) -> tuple[int, float]:
    """Total contracts and size-weighted average price across an order's fills."""
    total = 0.0
    notional = 0.0
    for f in fills:
        c = _fill_count(f)
        total += c
        notional += c * _fill_price(f, side)
    if total <= 0:
        return 0, 0.0
    return int(round(total)), notional / total


def _fills_by_order(client: Any, game_date: str) -> dict[str, list[dict[str, Any]]]:
    """Index the day's fills by ``order_id``."""
    start = datetime.strptime(game_date, "%Y-%m-%d").replace(tzinfo=ET)
    min_ts = int(start.timestamp())
    max_ts = int((start + timedelta(hours=FILL_WINDOW_HOURS)).timestamp())

    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in client.get_fills(min_ts=min_ts, max_ts=max_ts):
        order_id = str(f.get("order_id", "") or "")
        if order_id:
            index[order_id].append(f)
    return index


def reconcile_fills_for_date(
    game_date: str,
    *,
    client: Any,
    include_resting: bool = False,
) -> ReconcileResult:
    """
    Write ``note=fill`` rows for journaled orders not yet reconciled on ``game_date``.

    Idempotent per ``order_id``. Returns zeros if journal missing or no placed orders.
    """
    path = journal_path(game_date)
    if not path.exists():
        return ReconcileResult(updated=0, skipped=0, errors=0, days_processed=0)

    rows = load_jsonl_rows(path)
    placed = placed_with_order_id(rows)
    if not placed:
        return ReconcileResult(updated=0, skipped=0, errors=0, days_processed=0)

    existing_fill_ids = existing_fill_order_ids(rows)
    updated = 0
    skipped = 0
    errors = 0

    # /portfolio/orders/{id} carries no fill-price field at all, so the order path
    # alone can only ever report limit_price as the entry price. Fills carry the
    # true executed price and survive longer than the order records, so they are
    # the primary source here and the only source once orders are purged.
    fills_by_order: dict[str, list[dict[str, Any]]] = {}
    fills_unavailable = False
    try:
        fills_by_order = _fills_by_order(client, game_date)
    except Exception as e:
        log.error("Fills unavailable for %s (%s) — falling back to order records", game_date, e)
        fills_unavailable = True

    for r in placed:
        order_id = str(r.get("order_id"))
        if order_id in existing_fill_ids:
            skipped += 1
            continue

        side = str(r.get("side", ""))
        order_fills = fills_by_order.get(order_id, [])
        try:
            o = client.get_order(order_id)
        except Exception as e:
            # Kalshi purges /portfolio/orders/{id} after a few days, so a journal that
            # missed its nightly reconcile (host asleep) can no longer be repaired
            # order-by-order.
            if fills_unavailable:
                log.warning("Could not fetch order %s (%s)", order_id, e)
                errors += 1
                continue
            # Absent from the day's fills means the order simply never filled.
            filled, avg_fill_price = _aggregate_fills(order_fills, side)
            status = "executed" if filled > 0 else ""
        else:
            status = str(o.get("status", "") or "").lower()
            if fills_unavailable:
                # Degraded: the order record carries neither a fill price nor, in
                # practice, count/remaining_count — both keys are absent, so this
                # books the full order size at its limit price. It is wrong for
                # anything that did not fill completely, but it is all we have.
                count = int(o.get("count", r.get("contracts", 0)) or 0)
                remaining = int(o.get("remaining_count", o.get("remaining", 0)) or 0)
                filled = max(0, count - remaining)
                avg_fill_price = _avg_fill_price_from_order(o, r)
            else:
                # Fills are authoritative: no fill record means the order never
                # filled, however the order record describes itself.
                filled, avg_fill_price = _aggregate_fills(order_fills, side)

        if filled <= 0 and not include_resting:
            continue

        append_row(
            game_date,
            TradeRow(
                game_date=game_date,
                ticker=str(r.get("ticker", "")),
                side=side,
                action=str(r.get("action", "buy")),
                contracts=int(r.get("contracts", 0)),
                limit_price=float(r.get("limit_price", 0.0)),
                order_id=order_id,
                player_name=str(r.get("player_name", "")),
                kalshi_line=float(r.get("kalshi_line", 0.0)),
                predicted_lambda=float(r.get("predicted_lambda", 0.0)),
                p_model=float(r.get("p_model", 0.0)),
                p_model_raw=float(r.get("p_model_raw", 0.0) or 0.0),
                p_model_cal=float(r.get("p_model_cal", 0.0) or 0.0),
                p_market=float(r.get("p_market", 0.0)),
                fee_per_contract=float(r.get("fee_per_contract", 0.0) or 0.0),
                edge=float(r.get("edge", 0.0)),
                ev=float(r.get("ev", 0.0)),
                expected_pnl=float(r.get("expected_pnl", 0.0)),
                book_bid=float(r.get("book_bid", 0.0)),
                book_ask=float(r.get("book_ask", 0.0)),
                book_spread=float(r.get("book_spread", 0.0)),
                filled_contracts=int(filled),
                avg_fill_price=avg_fill_price,
                note="fill",
                success=True if status in {"executed", "filled"} else None,
            ).to_dict(),
        )
        updated += 1

    return ReconcileResult(
        updated=updated,
        skipped=skipped,
        errors=errors,
        days_processed=1,
    )


def reconcile_fills_for_range(
    start_d: datetime,
    end_d: datetime,
    *,
    client: Any,
    include_resting: bool = False,
) -> ReconcileResult:
    """Reconcile each journal day in ``[start_d, end_d]`` (inclusive)."""
    total = ReconcileResult(updated=0, skipped=0, errors=0, days_processed=0)
    for _path, date_str in journal_paths_in_date_range(start_d, end_d):
        day_result = reconcile_fills_for_date(
            date_str,
            client=client,
            include_resting=include_resting,
        )
        if day_result.days_processed > 0:
            total = total.merge(day_result)
    return total
