"""Shared helpers for trade report / report-range (buckets, PnL, outcomes)."""

from __future__ import annotations

from fees import order_fee_usd


def pnl_per_contract(side: str, price: float, result: str | float) -> float | None:
    """
    P&L of one long contract of ``side`` bought at ``price``. ``result`` is
    ``"yes"``/``"no"``, or the YES settlement value (dollars) of a market that
    settled scalar; anything else is unresolved (None).
    """
    if isinstance(result, (int, float)) and not isinstance(result, bool):
        yes_value = float(result)
    elif result in {"yes", "no"}:
        yes_value = 1.0 if result == "yes" else 0.0
    else:
        return None
    payout = yes_value if str(side).lower() == "yes" else 1.0 - yes_value
    return payout - float(price)


def estimated_order_fee_usd(row: dict, price: float, contracts: int) -> float:
    """
    Kalshi fee for a filled trade. Rows journaled after 2026-07-06 carry
    ``fee_per_contract`` (0.0 for resting/maker fills); older rows always
    crossed at the ask, so estimate the taker fee at the fill price.
    """
    if "fee_per_contract" in row:
        return float(row.get("fee_per_contract", 0.0) or 0.0) * int(contracts)
    return order_fee_usd(price, contracts, is_taker=True)


def market_yes_no_result(market: dict) -> str:
    return str((market or {}).get("result") or "").lower()


def market_settlement(market: dict) -> str | float:
    """
    ``"yes"``/``"no"`` for a binary settlement, the YES settlement value (float,
    dollars) for a ``scalar`` one — Kalshi settles a player who did not play at a
    fixed price — else ``""`` (unsettled). Feed it to :func:`pnl_per_contract`.
    """
    res = market_yes_no_result(market)
    if res in {"yes", "no"}:
        return res
    if res == "scalar":
        m = market or {}
        for k, scale in (("settlement_value_dollars", 1.0), ("settlement_value", 100.0)):
            v = m.get(k)
            if v in (None, ""):
                continue
            try:
                val = float(v) / scale
            except (TypeError, ValueError):
                continue
            if 0.0 <= val <= 1.0:
                return val
    return ""


def line_bucket(line: float) -> str:
    """NBA rebound strike bucket for segment reporting (Kalshi lines run 1.5–19.5)."""
    ln = float(line)
    if ln <= 4.5:
        return "<=4.5"
    if ln <= 7.5:
        return "5.5-7.5"
    if ln <= 10.5:
        return "8.5-10.5"
    return ">=11.5"


def price_bucket(price: float) -> str:
    if price < 0.20:
        return "<0.20"
    if price < 0.50:
        return "0.20-0.49"
    if price < 0.80:
        return "0.50-0.79"
    return ">=0.80"


def edge_bucket(edge: float) -> str:
    if edge < 0.05:
        return "<0.05"
    if edge < 0.10:
        return "0.05-0.09"
    if edge < 0.15:
        return "0.10-0.14"
    if edge < 0.20:
        return "0.15-0.19"
    return ">=0.20"


def spread_bucket(spread: float) -> str:
    if spread <= 0:
        return "n/a"
    if spread < 0.05:
        return "<0.05"
    if spread < 0.10:
        return "0.05-0.09"
    if spread < 0.20:
        return "0.10-0.19"
    return ">=0.20"
