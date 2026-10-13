from settings_validate import validate_settings


def test_validate_settings_ok():
    v = validate_settings()
    assert v.ok, v.errors


import pytest
import typer

import settings_validate
from settings_validate import live_trading_errors


@pytest.mark.parametrize(
    "name,value",
    [
        ("MAX_BID_ASK_SPREAD", 2.0),
        ("MIN_LIMIT_PRICE", 1.5),
        ("SCAN_WITHIN_HOURS", -1.0),
        ("DAILY_LOSS_LIMIT_PCT", 5.0),
        ("MAX_DAILY_DEPLOYED_PCT", 1.5),
        ("MAX_OPEN_RESTING_PCT", -0.1),
        ("MAX_CALIB_P_DELTA", 2.0),
        ("DEFAULT_BLEND_WEIGHT", 1.5),
        ("MIN_BLEND_WEIGHT", -0.2),
        ("LOSS_LIMIT_LOOKBACK_DAYS", 0),
    ],
)
def test_validate_rejects_out_of_range(monkeypatch, name, value):
    monkeypatch.setattr(f"config.{name}", value)
    v = validate_settings()
    assert not v.ok and any(name in e for e in v.errors)


def test_validate_rejects_mismatched_kalshi_hosts(monkeypatch):
    monkeypatch.setattr("config.KALSHI_BASE_URL", "https://demo-api.kalshi.co/trade-api/v2")
    monkeypatch.setattr("config.KALSHI_ORDER_URL", "https://api.elections.kalshi.com/trade-api/v2")
    v = validate_settings()
    assert not v.ok and any("KALSHI_ORDER_URL" in e for e in v.errors)


def _caps(monkeypatch, loss_usd=0.0, loss_pct=0.0, dep_usd=0.0, dep_pct=0.0):
    monkeypatch.setattr("config.DAILY_LOSS_LIMIT_USD", loss_usd)
    monkeypatch.setattr("config.DAILY_LOSS_LIMIT_PCT", loss_pct)
    monkeypatch.setattr("config.MAX_DAILY_DEPLOYED_USD", dep_usd)
    monkeypatch.setattr("config.MAX_DAILY_DEPLOYED_PCT", dep_pct)


def test_live_requires_loss_and_deployed_caps(monkeypatch):
    _caps(monkeypatch)
    errs = live_trading_errors()
    assert any("loss" in e.lower() for e in errs) and any("deployed" in e.lower() for e in errs)


def test_live_ok_with_pct_caps(monkeypatch):
    _caps(monkeypatch, loss_pct=0.05, dep_pct=0.10)
    assert live_trading_errors() == []


def test_cli_callback_exits_on_invalid_settings(monkeypatch):
    from cli.main import validate_settings_callback

    monkeypatch.setattr("config.MAX_BET_PCT", 2.0)
    with pytest.raises(typer.Exit):
        validate_settings_callback()


def test_validate_rejects_non_positive_blend_cap(monkeypatch):
    monkeypatch.setattr("config.BLEND_CAP", 0.0)
    v = validate_settings()
    assert not v.ok and any("BLEND_CAP" in e for e in v.errors)
