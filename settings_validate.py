"""
Validate environment-driven settings at startup (fail fast on bad config).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

log = logging.getLogger(__name__)


@dataclass
class SettingsValidation:
    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_settings() -> SettingsValidation:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        from config import (
            EDGE_THRESHOLD,
            KELLY_FRACTION,
            MAX_BET_PCT,
            MAX_PORTFOLIO_PCT,
            MIN_P,
            VPIN_MAX_TOXIC,
            DAILY_LOSS_LIMIT_USD,
            MAX_ORDERS_PER_DAY,
            MAX_DAILY_DEPLOYED_USD,
            MAX_OPEN_RESTING_USD,
            MAX_CONTRACTS_PER_GAME,
            MAX_LEGS_PER_PLAYER_DAY,
            SEGMENT_MIN_FILLS,
            SEGMENT_MIN_FILL_RATE,
            SEGMENT_MIN_AVG_CLV,
            SEGMENT_MIN_ROI_PCT,
            CALIBRATE_MAX_AGE_DAYS,
            BLEND_CAP,
            DAILY_LOSS_LIMIT_PCT,
            DEFAULT_BLEND_WEIGHT,
            KALSHI_BASE_URL,
            KALSHI_ORDER_URL,
            LOSS_LIMIT_LOOKBACK_DAYS,
            MAX_BID_ASK_SPREAD,
            MAX_CALIB_P_DELTA,
            MAX_DAILY_DEPLOYED_PCT,
            MAX_OPEN_RESTING_PCT,
            MIN_BLEND_WEIGHT,
            MIN_LIMIT_PRICE,
            SCAN_WITHIN_HOURS,
        )
    except Exception as e:
        return SettingsValidation(False, [f"config import failed: {e}"], [])

    if not (0 < EDGE_THRESHOLD < 1):
        errors.append(f"EDGE_THRESHOLD must be in (0,1), got {EDGE_THRESHOLD}")
    if not (0 < KELLY_FRACTION <= 1):
        errors.append(f"KELLY_FRACTION must be in (0,1], got {KELLY_FRACTION}")
    if not (0 < MAX_BET_PCT <= 1):
        errors.append(f"MAX_BET_PCT must be in (0,1], got {MAX_BET_PCT}")
    if not (0 < MAX_PORTFOLIO_PCT <= 1):
        errors.append(f"MAX_PORTFOLIO_PCT must be in (0,1], got {MAX_PORTFOLIO_PCT}")
    if not (0 <= MIN_P < 1):
        errors.append(f"MIN_P must be in [0,1), got {MIN_P}")
    if not (0 <= VPIN_MAX_TOXIC <= 1):
        errors.append(f"VPIN_MAX_TOXIC must be in [0,1], got {VPIN_MAX_TOXIC}")
    if DAILY_LOSS_LIMIT_USD < 0:
        errors.append(f"DAILY_LOSS_LIMIT_USD must be >= 0, got {DAILY_LOSS_LIMIT_USD}")
    if MAX_ORDERS_PER_DAY < 0:
        errors.append(f"MAX_ORDERS_PER_DAY must be >= 0, got {MAX_ORDERS_PER_DAY}")
    if MAX_DAILY_DEPLOYED_USD < 0:
        errors.append(f"MAX_DAILY_DEPLOYED_USD must be >= 0, got {MAX_DAILY_DEPLOYED_USD}")
    if MAX_OPEN_RESTING_USD < 0:
        errors.append(f"MAX_OPEN_RESTING_USD must be >= 0, got {MAX_OPEN_RESTING_USD}")
    if MAX_CONTRACTS_PER_GAME < 0:
        errors.append(f"MAX_CONTRACTS_PER_GAME must be >= 0, got {MAX_CONTRACTS_PER_GAME}")
    if MAX_LEGS_PER_PLAYER_DAY < 0:
        errors.append(f"MAX_LEGS_PER_PLAYER_DAY must be >= 0, got {MAX_LEGS_PER_PLAYER_DAY}")
    if SEGMENT_MIN_FILLS < 0:
        errors.append(f"SEGMENT_MIN_FILLS must be >= 0, got {SEGMENT_MIN_FILLS}")
    if not (0.0 <= SEGMENT_MIN_FILL_RATE <= 1.0):
        errors.append(f"SEGMENT_MIN_FILL_RATE must be in [0,1], got {SEGMENT_MIN_FILL_RATE}")
    if CALIBRATE_MAX_AGE_DAYS < 0:
        errors.append(f"CALIBRATE_MAX_AGE_DAYS must be >= 0, got {CALIBRATE_MAX_AGE_DAYS}")
    for name, v in (
        ("DAILY_LOSS_LIMIT_PCT", DAILY_LOSS_LIMIT_PCT),
        ("MAX_DAILY_DEPLOYED_PCT", MAX_DAILY_DEPLOYED_PCT),
        ("MAX_OPEN_RESTING_PCT", MAX_OPEN_RESTING_PCT),
        ("MAX_CALIB_P_DELTA", MAX_CALIB_P_DELTA),
        ("DEFAULT_BLEND_WEIGHT", DEFAULT_BLEND_WEIGHT),
        ("MIN_BLEND_WEIGHT", MIN_BLEND_WEIGHT),
    ):
        if not (0.0 <= v <= 1.0):
            errors.append(f"{name} must be in [0,1], got {v}")
    if not (BLEND_CAP > 0):
        errors.append(f"BLEND_CAP must be > 0 (logits), got {BLEND_CAP}")
    if not (0 < MAX_BID_ASK_SPREAD < 1):
        errors.append(f"MAX_BID_ASK_SPREAD must be in (0,1), got {MAX_BID_ASK_SPREAD}")
    if not (0 < MIN_LIMIT_PRICE < 1):
        errors.append(f"MIN_LIMIT_PRICE must be in (0,1), got {MIN_LIMIT_PRICE}")
    if SCAN_WITHIN_HOURS < 0:
        errors.append(f"SCAN_WITHIN_HOURS must be >= 0, got {SCAN_WITHIN_HOURS}")
    if LOSS_LIMIT_LOOKBACK_DAYS < 1:
        errors.append(f"LOSS_LIMIT_LOOKBACK_DAYS must be >= 1, got {LOSS_LIMIT_LOOKBACK_DAYS}")
    if urlparse(KALSHI_BASE_URL).netloc != urlparse(KALSHI_ORDER_URL).netloc:
        errors.append(
            f"KALSHI_ORDER_URL host ({urlparse(KALSHI_ORDER_URL).netloc}) differs from KALSHI_BASE_URL "
            f"host ({urlparse(KALSHI_BASE_URL).netloc}) — orders and market data would hit different exchanges"
        )
    if DAILY_LOSS_LIMIT_USD > 0 and MAX_DAILY_DEPLOYED_USD <= 0:
        warnings.append(
            "DAILY_LOSS_LIMIT_USD is set but MAX_DAILY_DEPLOYED_USD is off — "
            "intraday protection relies on marks for MTM or post-settlement P&L only"
        )

    return SettingsValidation(len(errors) == 0, errors, warnings)


def live_trading_errors() -> list[str]:
    """Settings that must be on before any live order: a daily loss cap and a deployed cap."""
    from config import (
        DAILY_LOSS_LIMIT_PCT,
        DAILY_LOSS_LIMIT_USD,
        MAX_DAILY_DEPLOYED_PCT,
        MAX_DAILY_DEPLOYED_USD,
    )

    errors: list[str] = []
    if DAILY_LOSS_LIMIT_USD <= 0 and DAILY_LOSS_LIMIT_PCT <= 0:
        errors.append("Live trading needs a daily loss cap: set DAILY_LOSS_LIMIT_USD or DAILY_LOSS_LIMIT_PCT")
    if MAX_DAILY_DEPLOYED_USD <= 0 and MAX_DAILY_DEPLOYED_PCT <= 0:
        errors.append("Live trading needs a deployed cap: set MAX_DAILY_DEPLOYED_USD or MAX_DAILY_DEPLOYED_PCT")
    return errors


def require_valid_settings() -> None:
    v = validate_settings()
    if not v.ok:
        raise ValueError("Invalid configuration:\n" + "\n".join(f"  - {e}" for e in v.errors))
    for w in v.warnings:
        log.warning("Config: %s", w)
