"""
edge_detector.py
----------------
Bridges the probability engine and Kalshi to detect +EV opportunities
and size positions using Fractional Kelly Criterion.

Key concepts
------------
Edge      : P_model - P_market  (positive → model says line is mispriced)
EV        : (b * p) - q  (expected cents gained per dollar risked)
Kelly f*  : (b*p - q) / b  * kelly_fraction  (fractional for safety)
Max Bet   : Capped at MAX_BET_PCT of current bankroll

Trigger rules
-------------
A trade is flagged when ALL of the following hold:
  1. edge > EDGE_THRESHOLD
  2. EV > 0
  3. Minimum volume on Kalshi market (avoids illiquid lines)
  4. Safety switch: no flagged injury within 15 min of tip
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import EDGE_THRESHOLD, KELLY_FRACTION, MAX_BET_PCT, MIN_P, TAIL_P_CUTOFF, TAIL_EDGE_MULT
from probability_engine import ProbabilityResult
from kalshi_bridge import MarketLine, OrderResult, get_client
from execution_engine import ExecutionLedger, LedgerKey, suggest_limit_price
from trade_journal import TradeRow, append_row

log = logging.getLogger(__name__)

MIN_MARKET_VOLUME = 0     # volume filter disabled — use spread check instead
MAX_BID_ASK_SPREAD = 0.20  # skip markets where ask - bid > 20 cents (illiquid)
SAFETY_SWITCH = False      # set True externally when a lineup alert fires


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EdgeSignal:
    """One detected edge opportunity."""
    player_name: str
    player_id: int
    game_date: str
    ticker: str
    kalshi_line: float
    predicted_lambda: float
    p_model: float          # model's P(over)
    p_market: float         # market price used for entry (YES ask or NO ask)
    edge: float             # p_model - p_market
    ev: float               # expected value per $1 risked
    kelly_f: float          # fractional Kelly stake (as fraction of bankroll)
    recommended_contracts: int
    recommended_side: str   # "yes" or "no"
    bet_dollars: float
    limit_price: float      # limit price to send (<= model fair)
    flagged: bool = True    # True = strong edge; False = informational only

    def __repr__(self) -> str:
        sign = "+" if self.edge > 0 else ""
        return (
            f"EdgeSignal({self.player_name} | line={self.kalshi_line} | "
            f"λ={self.predicted_lambda:.2f} | edge={sign}{self.edge:.3f} | "
            f"EV={self.ev:.3f} | ${self.bet_dollars:.2f} {self.recommended_side.upper()})"
        )


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

def fractional_kelly(
    p: float,
    b: float,
    fraction: float = KELLY_FRACTION,
) -> float:
    """
    Fractional Kelly stake as a fraction of bankroll.

    Parameters
    ----------
    p        : model probability of winning
    b        : net odds (payout per $1 risked).
               For Kalshi "Yes" at price c: b = (1-c)/c
    fraction : Kelly multiplier (0.25 = quarter-Kelly)

    Returns
    -------
    f* in [0, 1] — fraction of bankroll to bet
    """
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f_full = (b * p - q) / b
    f_frac = max(0.0, f_full * fraction)
    return round(f_frac, 4)


def dollars_to_contracts(dollars: float, contract_price: float) -> int:
    """
    Convert dollar amount to number of Kalshi contracts.
    Each contract pays $1 if it wins; cost = contract_price (e.g. $0.54).
    """
    if contract_price <= 0:
        return 0
    return max(1, int(dollars / contract_price))


# ---------------------------------------------------------------------------
# Edge detection
# ---------------------------------------------------------------------------

def detect_edge(
    prob_result: ProbabilityResult,
    market_line: MarketLine,
    bankroll: float,
    edge_threshold: float = EDGE_THRESHOLD,
    max_spread: float = MAX_BID_ASK_SPREAD,
    min_p: float = MIN_P,
    tail_p_cutoff: float = TAIL_P_CUTOFF,
    tail_edge_mult: float = TAIL_EDGE_MULT,
) -> Optional[EdgeSignal]:
    """
    Compare model probability vs. market price; return EdgeSignal if edge found.

    We look at both sides:
      - Over edge: p_model > p_market  → buy "Yes"
      - Under edge: (1-p_model) > (1-p_market)  → buy "No"
    """
    if SAFETY_SWITCH:
        log.warning("Safety switch is ON — skipping all trades.")
        return None

    p_model_over = prob_result.p_over
    p_model_under = prob_result.p_under

    # Use side-specific entry prices (asks). This is where "NO is API-only" can matter:
    # the NO ask can be stale/mispriced relative to 1-YES mid/ask.
    yes_spread = market_line.yes_spread
    no_spread = market_line.no_spread

    yes_edge = p_model_over - market_line.yes_ask
    no_edge = p_model_under - market_line.no_ask

    # Tail guardrails:
    # - Hard floor: don't trade tiny probabilities at all.
    # - Margin-of-safety: require a larger edge in the tails (probabilities near the floor).
    def _effective_edge_threshold(p: float) -> float:
        thr = edge_threshold
        if p < tail_p_cutoff:
            thr = thr * tail_edge_mult
        return thr

    # Determine best side, respecting per-side liquidity via spread
    best = None
    yes_thr = _effective_edge_threshold(p_model_over)
    no_thr = _effective_edge_threshold(p_model_under)

    if p_model_over >= min_p and yes_edge > yes_thr and yes_spread <= max_spread:
        best = ("yes", p_model_over, market_line.yes_ask, yes_edge, yes_spread)
    if p_model_under >= min_p and no_edge > no_thr and no_spread <= max_spread:
        cand = ("no", p_model_under, market_line.no_ask, no_edge, no_spread)
        if best is None or cand[3] > best[3]:
            best = cand

    if best is None:
        return None

    side, p, c, edge, spread = best

    # EV per $1 risked
    b = (1.0 - c) / c if c > 0 else 0.0
    ev = b * p - (1.0 - p)

    if ev <= 0:
        return None

    # EV > 2.0 almost always means a broken/stale market price — skip
    if ev > 2.0:
        log.warning(f"{market_line.ticker}: EV={ev:.3f} suspiciously high, likely bad price — skipping")
        return None

    # Kelly sizing
    kf = fractional_kelly(p, b)
    bet_dollars = min(kf * bankroll, MAX_BET_PCT * bankroll)
    contracts = dollars_to_contracts(bet_dollars, c)

    # Execution limit price: cap at fair, adjust for spread
    if side == "yes":
        bid, ask = market_line.yes_bid, market_line.yes_ask
        model_fair = p_model_over
    else:
        bid, ask = market_line.no_bid, market_line.no_ask
        model_fair = p_model_under
    limit_price = suggest_limit_price(side=side, bid=bid, ask=ask, model_fair=model_fair)

    return EdgeSignal(
        player_name=prob_result.player_name,
        player_id=prob_result.player_id,
        game_date=prob_result.game_date,
        ticker=market_line.ticker,
        kalshi_line=market_line.line,
        predicted_lambda=prob_result.predicted_lambda,
        p_model=p,
        p_market=c,
        edge=edge,
        ev=ev,
        kelly_f=kf,
        recommended_contracts=contracts,
        recommended_side=side,
        bet_dollars=round(bet_dollars, 2),
        limit_price=limit_price,
        flagged=True,
    )


def scan_for_edges(
    prob_results: list[ProbabilityResult],
    market_lines: list[MarketLine],
    bankroll: float,
    edge_threshold: float = EDGE_THRESHOLD,
    min_p: float = MIN_P,
    tail_p_cutoff: float = TAIL_P_CUTOFF,
    tail_edge_mult: float = TAIL_EDGE_MULT,
) -> list[EdgeSignal]:
    """
    Match probability results to market lines by player name and line,
    then run edge detection on each matched pair.
    """
    # Build lookup: (player_name_lower, line) -> MarketLine
    market_map: dict[tuple, MarketLine] = {}
    for ml in market_lines:
        key = (ml.player_name.lower(), ml.line)
        market_map[key] = ml

    signals = []
    for pr in prob_results:
        key = (pr.player_name.lower(), pr.kalshi_line)
        ml = market_map.get(key)
        if ml is None:
            log.debug(f"No market match for {pr.player_name} @ {pr.kalshi_line}")
            continue
        signal = detect_edge(
            pr,
            ml,
            bankroll,
            edge_threshold=edge_threshold,
            min_p=min_p,
            tail_p_cutoff=tail_p_cutoff,
            tail_edge_mult=tail_edge_mult,
        )
        if signal:
            signals.append(signal)

    signals.sort(key=lambda s: s.ev, reverse=True)
    return signals


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_signals(
    signals: list[EdgeSignal],
    dry_run: bool = True,
    todays_tickers: Optional[set[str]] = None,
    cancel_stale: bool = True,
    replace_if_price_diff: float = 0.02,
) -> list[OrderResult]:
    """
    Place orders for all flagged signals.
    dry_run=True logs without placing real orders.
    """
    client = get_client(force_mock=dry_run)
    ledger = ExecutionLedger()
    results = []

    desired = {(s.ticker, s.recommended_side): s for s in signals}

    # In live mode, cancel stale open orders for today's markets that are no longer desired.
    if not dry_run and cancel_stale and hasattr(client, "get_orders") and hasattr(client, "cancel_order"):
        try:
            open_orders = client.get_orders(status="resting", limit=200)
        except Exception as e:
            log.warning(f"Could not fetch open orders ({e}); skipping cancel/replace.")
            open_orders = []

        todays_tickers = todays_tickers or set()
        for o in open_orders:
            if todays_tickers and o.ticker not in todays_tickers:
                continue
            if (o.ticker, o.side) not in desired:
                log.info(f"Cancelling stale open order: {o.ticker} {o.side.upper()} id={o.order_id} @ {o.price:.2f}")
                client.cancel_order(o.order_id)

    for sig in signals:
        key = LedgerKey(game_date=sig.game_date, ticker=sig.ticker, side=sig.recommended_side)
        if ledger.has(key) and dry_run:
            # In dry-run we keep the ledger behavior to reduce repeated spam.
            log.info(f"Skipping duplicate (already attempted): {sig.ticker} {sig.recommended_side.upper()} {sig.game_date}")
            continue

        if dry_run:
            log.info(f"[DRY RUN] {sig}")
        ledger.add_attempt(
            key,
            price=sig.limit_price,
            contracts=sig.recommended_contracts,
            dollars=sig.bet_dollars,
            note="pre-submit",
            success=None,
        )
        append_row(
            sig.game_date,
            TradeRow(
                game_date=sig.game_date,
                ticker=sig.ticker,
                side=sig.recommended_side,
                action="buy",
                contracts=sig.recommended_contracts,
                limit_price=sig.limit_price,
                order_id="",
                player_name=sig.player_name,
                kalshi_line=sig.kalshi_line,
                predicted_lambda=sig.predicted_lambda,
                p_model=sig.p_model,
                edge=sig.edge,
                ev=sig.ev,
                note="pre-submit",
                success=None,
            ).to_dict(),
        )

        # Replace logic (live): if there's an existing resting order on the same ticker+side
        # at a meaningfully different price, cancel it and submit the updated limit.
        if not dry_run and hasattr(client, "get_orders") and hasattr(client, "cancel_order"):
            try:
                existing = [
                    o for o in client.get_orders(status="resting", ticker=sig.ticker, limit=50)
                    if o.side == sig.recommended_side
                ]
            except Exception:
                existing = []
            for o in existing:
                if abs(o.price - sig.limit_price) >= replace_if_price_diff:
                    log.info(
                        f"Replacing order: {sig.ticker} {sig.recommended_side.upper()} "
                        f"old={o.price:.2f} new={sig.limit_price:.2f} id={o.order_id}"
                    )
                    client.cancel_order(o.order_id)

        result = client.place_order(
            ticker=sig.ticker,
            side=sig.recommended_side,
            contracts=sig.recommended_contracts,
            price=sig.limit_price,
        )
        if result.success:
            log.info(f"Order placed: {sig.ticker} | {sig.recommended_side.upper()} "
                     f"x{sig.recommended_contracts} @ {sig.limit_price:.2f}")
        else:
            log.warning(f"Order failed: {result.message}")

        ledger.add_attempt(
            key,
            price=sig.limit_price,
            contracts=sig.recommended_contracts,
            dollars=sig.bet_dollars,
            note="post-submit",
            order_id=result.order_id,
            success=result.success,
        )
        append_row(
            sig.game_date,
            TradeRow(
                game_date=sig.game_date,
                ticker=sig.ticker,
                side=sig.recommended_side,
                action="buy",
                contracts=sig.recommended_contracts,
                limit_price=sig.limit_price,
                order_id=result.order_id,
                player_name=sig.player_name,
                kalshi_line=sig.kalshi_line,
                predicted_lambda=sig.predicted_lambda,
                p_model=sig.p_model,
                edge=sig.edge,
                ev=sig.ev,
                note="post-submit",
                success=result.success,
            ).to_dict(),
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# EV summary report
# ---------------------------------------------------------------------------

def summarize_signals(signals: list[EdgeSignal]) -> str:
    if not signals:
        return "No edges found."

    lines = [
        f"{'Player':<28} {'Line':>5} {'λ':>6} {'P_mdl':>6} {'P_mkt':>6} "
        f"{'Edge':>6} {'EV':>6} {'Side':>4} {'$Bet':>7}"
    ]
    lines.append("-" * 84)
    for s in signals:
        lines.append(
            f"{s.player_name:<28} {s.kalshi_line:>5.1f} {s.predicted_lambda:>6.2f} "
            f"{s.p_model:>6.3f} {s.p_market:>6.3f} {s.edge:>+6.3f} {s.ev:>6.3f} "
            f"{s.recommended_side.upper():>4} {s.bet_dollars:>7.2f}"
        )
    total_ev = sum(s.ev * s.bet_dollars for s in signals)
    lines.append("-" * 84)
    lines.append(f"Total signals: {len(signals)} | Aggregate EV: ${total_ev:.2f}")
    return "\n".join(lines)
