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

from config import EDGE_THRESHOLD, KELLY_FRACTION, MAX_BET_PCT
from probability_engine import ProbabilityResult
from kalshi_bridge import MarketLine, OrderResult, get_client

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
    p_market: float         # Kalshi's implied P(over)
    edge: float             # p_model - p_market
    ev: float               # expected value per $1 risked
    kelly_f: float          # fractional Kelly stake (as fraction of bankroll)
    recommended_contracts: int
    recommended_side: str   # "yes" or "no"
    bet_dollars: float
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

    spread = market_line.yes_ask - market_line.yes_bid
    if spread > max_spread:
        log.debug(f"{market_line.ticker}: spread {spread:.2f} too wide, skipping")
        return None

    p_model_over = prob_result.p_over
    p_model_under = prob_result.p_under
    p_market_over = market_line.implied_prob

    # Determine best side
    over_edge = p_model_over - p_market_over
    under_edge = p_model_under - (1.0 - p_market_over)

    if over_edge >= under_edge and over_edge > edge_threshold:
        side = "yes"
        p = p_model_over
        c = market_line.yes_ask  # cost of "Yes" contract
        edge = over_edge
    elif under_edge > over_edge and under_edge > edge_threshold:
        side = "no"
        p = p_model_under
        c = market_line.no_ask
        edge = under_edge
    else:
        return None  # no edge

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

    return EdgeSignal(
        player_name=prob_result.player_name,
        player_id=prob_result.player_id,
        game_date=prob_result.game_date,
        ticker=market_line.ticker,
        kalshi_line=market_line.line,
        predicted_lambda=prob_result.predicted_lambda,
        p_model=p,
        p_market=p_market_over if side == "yes" else (1.0 - p_market_over),
        edge=edge,
        ev=ev,
        kelly_f=kf,
        recommended_contracts=contracts,
        recommended_side=side,
        bet_dollars=round(bet_dollars, 2),
        flagged=True,
    )


def scan_for_edges(
    prob_results: list[ProbabilityResult],
    market_lines: list[MarketLine],
    bankroll: float,
    edge_threshold: float = EDGE_THRESHOLD,
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
        signal = detect_edge(pr, ml, bankroll, edge_threshold)
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
) -> list[OrderResult]:
    """
    Place orders for all flagged signals.
    dry_run=True logs without placing real orders.
    """
    client = get_client(force_mock=dry_run)
    results = []

    for sig in signals:
        if dry_run:
            log.info(f"[DRY RUN] {sig}")
        result = client.place_order(
            ticker=sig.ticker,
            side=sig.recommended_side,
            contracts=sig.recommended_contracts,
            price=sig.p_market,
        )
        if result.success:
            log.info(f"Order placed: {sig.ticker} | {sig.recommended_side.upper()} "
                     f"x{sig.recommended_contracts} @ {sig.p_market:.2f}")
        else:
            log.warning(f"Order failed: {result.message}")
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
