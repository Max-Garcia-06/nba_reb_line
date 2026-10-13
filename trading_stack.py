"""
Shared post-edge pipeline for live scan and backtest (sizing parity).
"""

from __future__ import annotations

from typing import Any

from capital_optimizer import resize_signals_portfolio
from edge_detector import EdgeSignal, dollars_to_contracts
from identity_bridge import norm_player_name


def finalize_signals(
    signals: list[EdgeSignal],
    *,
    bankroll: float,
    one_per_player: bool = True,
    max_signals: int | None = None,
    max_contracts: int | None = None,
    existing_player_usd: dict[str, float] | None = None,
    max_player_usd: float | None = None,
) -> list[EdgeSignal]:
    """
    Mirror live scan sizing: best per player, cap count, per-player exposure cap,
    portfolio resize, contract caps.

    ``existing_player_usd`` (keyed by :func:`norm_player_name`) is what earlier scans
    already committed today; each player's new leg is capped so the day's total stays
    within ``max_player_usd``.
    """
    if not signals:
        return []

    if one_per_player:
        seen: dict[str, EdgeSignal] = {}
        for s in signals:
            if s.player_name not in seen or s.ev > seen[s.player_name].ev:
                seen[s.player_name] = s
        signals = sorted(seen.values(), key=lambda x: x.ev, reverse=True)

    if max_signals and len(signals) > max_signals:
        signals = signals[: int(max_signals)]

    if max_player_usd is not None and max_player_usd > 0:
        existing = existing_player_usd or {}
        for s in signals:
            room = float(max_player_usd) - float(existing.get(norm_player_name(s.player_name), 0.0))
            if s.bet_dollars > room:
                s.recommended_contracts = dollars_to_contracts(max(0.0, room), s.limit_price)
                s.bet_dollars = round(s.recommended_contracts * s.limit_price, 2)
        signals = [s for s in signals if s.recommended_contracts > 0]

    resize_signals_portfolio(signals, float(bankroll))

    for s in signals:
        if max_contracts:
            s.recommended_contracts = min(int(s.recommended_contracts), int(max_contracts))
        s.bet_dollars = round(s.recommended_contracts * s.limit_price, 2)

    total_raw = sum(s.bet_dollars for s in signals)
    if total_raw > bankroll and total_raw > 0:
        scale = bankroll / total_raw
        for s in signals:
            s.bet_dollars = round(s.bet_dollars * scale, 2)
            s.recommended_contracts = dollars_to_contracts(s.bet_dollars, s.limit_price)
            if max_contracts:
                s.recommended_contracts = min(int(s.recommended_contracts), int(max_contracts))
            s.bet_dollars = round(s.recommended_contracts * s.limit_price, 2)

    return [s for s in signals if s.recommended_contracts > 0]
