"""
Daily risk snapshot from trade journal + exchange state (P&L, deployment, concentration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from data_engine import matchup_slug, parse_kalshi_event_matchup
from journal_reader import (
    index_fills_by_order_id,
    index_marks_by_order_and_label,
    load_jsonl_rows,
    placed_with_order_id,
)
from kalshi_bridge import read_price_from_market_dict
from reporting_common import market_yes_no_result, pnl_per_contract
from trade_journal import journal_path

if TYPE_CHECKING:
    from edge_detector import EdgeSignal

log = logging.getLogger(__name__)


class RiskDataUnavailable(RuntimeError):
    """Exchange state needed for a live risk decision could not be read."""


def event_ticker_from_ticker(ticker: str) -> str:
    parts = str(ticker or "").split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return ""


def game_slug_from_ticker(ticker: str) -> str:
    et = event_ticker_from_ticker(ticker)
    matchup = parse_kalshi_event_matchup(et) if et else None
    if not matchup:
        return ""
    return matchup_slug(*matchup)


def player_leg_key(player_name: str, kalshi_line: float) -> tuple[str, float]:
    return (str(player_name or "").strip().lower(), float(kalshi_line))


def _latest_mark_by_order_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Most recent mark row per order_id (by ts string, lexicographic ISO works)."""
    marks = index_marks_by_order_and_label(rows)
    by_oid: dict[str, dict[str, Any]] = {}
    for (_oid, _label), row in marks.items():
        oid = str(row.get("order_id", "") or "")
        if not oid:
            continue
        prev = by_oid.get(oid)
        if prev is None or str(row.get("ts", "") or "") >= str(prev.get("ts", "") or ""):
            by_oid[oid] = row
    return by_oid


def mtm_pnl_per_contract(side: str, fill_price: float, mark_row: dict[str, Any]) -> float:
    """Unrealized P&L per contract vs latest mark mid (long binary at fill_price)."""
    side_l = str(side or "").lower()
    if side_l == "yes":
        mid = float(mark_row.get("mark_yes_mid") or 0.0)
    else:
        mid = float(mark_row.get("mark_no_mid") or 0.0)
    if mid <= 0:
        return 0.0
    return float(mid) - float(fill_price)


def _fetch_markets(client: object | None, tickers: set[str], *, strict: bool = False) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if client is None or not hasattr(client, "get_market"):
        return out
    for t in tickers:
        if not t:
            continue
        try:
            out[t] = client.get_market(t) or {}
        except Exception as e:
            # An unreadable market looks unsettled with no mark: its loss would vanish.
            if strict:
                raise RiskDataUnavailable(f"get_market({t}) failed: {e}") from e
            log.debug("get_market(%s) failed: %s", t, e)
            out[t] = {}
    return out


def _live_mid(market: dict, side: str) -> float:
    """Current side mid from a get_market payload (0 when either quote is missing)."""
    s = "yes" if str(side).lower() == "yes" else "no"
    bid = read_price_from_market_dict(market or {}, f"{s}_bid_dollars", f"{s}_bid")
    ask = read_price_from_market_dict(market or {}, f"{s}_ask_dollars", f"{s}_ask")
    return (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0


def _exchange_fills(client: object | None, game_date: str, *, strict: bool = False) -> dict[str, list[dict]]:
    """order_id -> raw Kalshi fill records for ``game_date``'s fill window.

    The journal only gets ``note=fill`` rows from the nightly reconcile, so without
    this every intraday fill (and its loss) is invisible to the risk desk.
    """
    if client is None or not hasattr(client, "get_fills"):
        return {}
    from reconcile_fills import _fills_by_order

    try:
        return dict(_fills_by_order(client, game_date))
    except Exception as e:
        if strict:
            raise RiskDataUnavailable(f"get_fills failed: {e}") from e
        log.warning("get_fills failed: %s", e)
        return {}


def open_resting_usd(client: object | None, *, strict: bool = False) -> float:
    if client is None or not hasattr(client, "get_orders"):
        return 0.0
    try:
        orders = client.get_orders(status="resting", limit=200)
    except Exception as e:
        if strict:
            raise RiskDataUnavailable(f"get_orders(resting) failed: {e}") from e
        log.warning("get_orders(resting) failed: %s", e)
        return 0.0
    total = 0.0
    for o in orders:
        rem = int(getattr(o, "remaining_count", 0) or 0)
        px = float(getattr(o, "price", 0.0) or 0.0)
        if rem > 0 and px > 0:
            total += rem * px
    return float(total)


def deployed_usd_from_journal(rows: list[dict[str, Any]]) -> float:
    total = 0.0
    for r in rows:
        if str(r.get("note", "") or "") != "post-submit" or r.get("success") is not True:
            continue
        cnt = int(r.get("contracts", 0) or 0)
        px = float(r.get("limit_price", 0.0) or 0.0)
        if cnt > 0 and px > 0:
            total += cnt * px
    return float(total)


def player_exposure_usd(game_date: str) -> dict[str, float]:
    """
    USD committed per player (``norm_player_name`` key) by today's submitted orders,
    including ambiguous submits (``success`` None) that may be live. Unfilled
    remainders count too — conservative, so later scans size around them.
    """
    from identity_bridge import norm_player_name

    out: dict[str, float] = {}
    for r in load_jsonl_rows(journal_path(game_date)):
        if str(r.get("note", "") or "") != "post-submit" or r.get("success") is False:
            continue
        key = norm_player_name(str(r.get("player_name", "") or ""))
        usd = int(r.get("contracts", 0) or 0) * float(r.get("limit_price", 0.0) or 0.0)
        if key and usd > 0:
            out[key] = out.get(key, 0.0) + usd
    return out


def proposed_deployed_usd(signals: list["EdgeSignal"]) -> float:
    return float(sum(float(s.bet_dollars) for s in signals))


def proposed_resting_usd(signals: list["EdgeSignal"]) -> float:
    return float(sum(float(s.limit_price) * int(s.recommended_contracts) for s in signals))


@dataclass
class DailyRiskSnapshot:
    realized_pnl: float = 0.0
    mtm_pnl: float = 0.0
    total_pnl_for_limit: float = 0.0
    deployed_usd: float = 0.0
    open_resting_usd: float = 0.0
    open_filled_usd: float = 0.0
    orders_placed: int = 0
    contracts_by_game: dict[str, int] = field(default_factory=dict)
    player_legs: set[tuple[str, float]] = field(default_factory=set)


def _pnl_for_date(
    game_date: str,
    client: object | None,
    *,
    strict: bool,
) -> tuple[float, float, float, dict[str, int], list[dict[str, Any]]]:
    """Realized, MTM and open-filled USD for one journal date, plus filled contracts by game."""
    from reconcile_fills import _aggregate_fills

    rows = load_jsonl_rows(journal_path(game_date))
    placed = placed_with_order_id(rows)
    if not placed:
        return 0.0, 0.0, 0.0, {}, rows
    journal_fills = index_fills_by_order_id(rows)
    exchange_fills = _exchange_fills(client, game_date, strict=strict)
    marks_by_oid = _latest_mark_by_order_id(rows)
    markets = _fetch_markets(client, {str(r.get("ticker", "") or "") for r in placed}, strict=strict)

    realized = mtm = open_filled = 0.0
    contracts_by_game: dict[str, int] = {}
    for r in placed:
        oid = str(r.get("order_id", "") or "")
        side = str(r.get("side", "") or "")
        ticker = str(r.get("ticker", "") or "")
        if oid in exchange_fills:
            filled, price = _aggregate_fills(exchange_fills[oid], side)
        else:
            fill = journal_fills.get(oid)
            filled = int(fill.get("filled_contracts", 0) if fill else 0)
            price = float(fill.get("avg_fill_price", 0.0) if fill else 0.0)
        if filled <= 0:
            continue
        if price <= 0:
            price = float(r.get("limit_price", 0.0) or 0.0)

        slug = game_slug_from_ticker(ticker)
        if slug:
            contracts_by_game[slug] = contracts_by_game.get(slug, 0) + filled

        market = markets.get(ticker, {})
        pnlpc = pnl_per_contract(side, price, market_yes_no_result(market))
        if pnlpc is not None:
            realized += pnlpc * filled
            continue
        open_filled += price * filled
        mid = _live_mid(market, side)
        if mid > 0:
            mtm += (mid - price) * filled
        elif oid in marks_by_oid:
            mtm += mtm_pnl_per_contract(side, price, marks_by_oid[oid]) * filled
    return realized, mtm, open_filled, contracts_by_game, rows


def daily_risk_snapshot(
    game_date: str,
    client: object | None = None,
    *,
    strict: bool = False,
    pnl_lookback_days: int = 1,
) -> DailyRiskSnapshot:
    """
    Risk state for ``game_date``. P&L (realized + MTM vs live mid) covers
    ``pnl_lookback_days`` journal dates ending at ``game_date`` so last night's
    settled losses still count today; orders, deployment and concentration are
    ``game_date`` only. ``strict`` raises :class:`RiskDataUnavailable` instead of
    treating unreadable exchange state as zero.
    """
    realized, mtm, open_filled, contracts_by_game, rows = _pnl_for_date(game_date, client, strict=strict)
    d0 = datetime.strptime(game_date, "%Y-%m-%d")
    for k in range(1, max(1, int(pnl_lookback_days))):
        prev = (d0 - timedelta(days=k)).strftime("%Y-%m-%d")
        r_prev, m_prev, _, _, _ = _pnl_for_date(prev, client, strict=strict)
        realized += r_prev
        mtm += m_prev

    player_legs: set[tuple[str, float]] = set()
    for r in placed_with_order_id(rows):
        pname = str(r.get("player_name", "") or "")
        if pname:
            player_legs.add(player_leg_key(pname, float(r.get("kalshi_line", 0.0) or 0.0)))

    deployed = deployed_usd_from_journal(rows)
    resting = open_resting_usd(client, strict=strict)
    orders_placed = sum(
        1 for r in rows if str(r.get("note", "") or "") == "post-submit" and r.get("success") is True
    )

    return DailyRiskSnapshot(
        realized_pnl=float(realized),
        mtm_pnl=float(mtm),
        total_pnl_for_limit=float(realized + mtm),
        deployed_usd=float(deployed),
        open_resting_usd=float(resting),
        open_filled_usd=float(open_filled),
        orders_placed=int(orders_placed),
        contracts_by_game=contracts_by_game,
        player_legs=player_legs,
    )


def merge_proposed_into_snapshot(
    snapshot: DailyRiskSnapshot,
    signals: list["EdgeSignal"],
) -> DailyRiskSnapshot:
    """Return a copy with proposed signal deployment/concentration included."""
    deployed = snapshot.deployed_usd + proposed_deployed_usd(signals)
    resting = snapshot.open_resting_usd + proposed_resting_usd(signals)
    contracts_by_game = dict(snapshot.contracts_by_game)
    player_legs = set(snapshot.player_legs)

    for s in signals:
        slug = game_slug_from_ticker(s.ticker)
        if slug:
            contracts_by_game[slug] = contracts_by_game.get(slug, 0) + int(s.recommended_contracts)
        player_legs.add(player_leg_key(s.player_name, s.kalshi_line))

    return DailyRiskSnapshot(
        realized_pnl=snapshot.realized_pnl,
        mtm_pnl=snapshot.mtm_pnl,
        total_pnl_for_limit=snapshot.total_pnl_for_limit,
        deployed_usd=deployed,
        open_resting_usd=resting,
        open_filled_usd=snapshot.open_filled_usd,
        orders_placed=snapshot.orders_placed,
        contracts_by_game=contracts_by_game,
        player_legs=player_legs,
    )


def count_new_player_legs(snapshot: DailyRiskSnapshot, signals: list["EdgeSignal"]) -> int:
    existing = snapshot.player_legs
    new_legs: set[tuple[str, float]] = set()
    for s in signals:
        key = player_leg_key(s.player_name, s.kalshi_line)
        if key not in existing:
            new_legs.add(key)
    return len(new_legs)


def total_player_legs_after_proposed(snapshot: DailyRiskSnapshot, signals: list["EdgeSignal"]) -> int:
    merged = merge_proposed_into_snapshot(snapshot, signals)
    return len(merged.player_legs)
