from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import typer

from data_engine import tip_epoch_by_ticker
from journal_risk import player_exposure_usd
from kalshi_bridge import MarketLine, MockKalshiClient, OpenOrder

DATE = "2026-10-12"
T1 = "KXNBAREB-26OCT12BOSNYK-BOSPLAYERA8-8"


def _ml(ticker=T1, event="KXNBAREB-26OCT12BOSNYK"):
    return MarketLine(ticker=ticker, player_name="A", player_id=0, game_date=DATE, line=7.5,
                      yes_ask=0.5, yes_bid=0.48, no_ask=0.52, no_bid=0.5, event_ticker=event)


def test_tip_epoch_by_ticker_is_tip_minus_lead():
    tip = datetime(2026, 10, 12, 23, 30, tzinfo=timezone.utc)
    idx = {"BOSNYK": {"start_utc": tip}}
    out = tip_epoch_by_ticker([_ml()], DATE, schedule_index=idx, lead_seconds=60)
    assert out == {T1: int(tip.timestamp()) - 60}


def test_tip_epoch_by_ticker_omits_unknown_games():
    assert tip_epoch_by_ticker([_ml()], DATE, schedule_index={}) == {}


def test_player_exposure_from_successful_submits(tmp_path, monkeypatch):
    rows = [
        {"note": "post-submit", "success": True, "player_name": "Nikola Jokić", "contracts": 10, "limit_price": 0.5},
        {"note": "post-submit", "success": False, "player_name": "Nikola Jokić", "contracts": 99, "limit_price": 0.5},
        {"note": "post-submit", "success": None, "player_name": "Nikola Jokić", "contracts": 4, "limit_price": 0.5},
        {"note": "pre-submit", "player_name": "Nikola Jokić", "contracts": 99, "limit_price": 0.5},
    ]
    j = tmp_path / f"trades_{DATE}.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr("journal_risk.journal_path", lambda d: j)
    # success=None post-submit = ambiguous submit: may be live, so it counts.
    assert player_exposure_usd(DATE) == {"nikola jokic": pytest.approx(7.0)}


@pytest.fixture(autouse=True)
def _gates_before_model(monkeypatch):
    """Live/mock gates must fire before any model or preflight work."""
    def _boom(*a, **k):
        raise AssertionError("scan reached model loading")

    monkeypatch.setattr("model.load_model", _boom)


def _scan_kwargs(**over):
    kw = dict(game_date=DATE, bankroll=1000.0, threshold=None, min_p=None, tail_p_cutoff=None,
              tail_edge_mult=None, max_signals=None, one_per_player=True, max_contracts=250,
              series_ticker="KXNBAREB", dry_run=True, auto_mark=False, mark_delays="")
    kw.update(over)
    return kw


@pytest.mark.parametrize("dry_run", [True, False])
def test_scan_refuses_mock_client(monkeypatch, dry_run):
    from pipeline import commands

    monkeypatch.setattr("kalshi_bridge.get_client", lambda *a, **k: MockKalshiClient())
    with pytest.raises(typer.Exit) as e:
        commands.scan(**_scan_kwargs(dry_run=dry_run))
    assert e.value.exit_code == 1


class _Exchange:
    def __init__(self):
        self.resting = [OpenOrder("r1", T1, "yes", "", "limit", "resting", 0.49, 10, 10)]
        self.cancelled: list[str] = []

    def get_orders(self, status="resting", ticker=None, limit=200):
        return [o for o in self.resting if o.order_id not in self.cancelled]

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True

    def get_balance(self):
        return 500.0


def test_live_scan_with_kill_switch_cancels_resting_and_exits(monkeypatch, tmp_path):
    from pipeline import commands

    ex = _Exchange()
    monkeypatch.setattr("kalshi_bridge.get_client", lambda *a, **k: ex)
    monkeypatch.setattr("settings_validate.live_trading_errors", lambda: [])
    kill = tmp_path / "KILL"
    kill.write_text("halt")
    monkeypatch.setattr("risk_manager.KILL_SWITCH_PATH", kill)
    with pytest.raises(typer.Exit) as e:
        commands.scan(**_scan_kwargs(dry_run=False))
    assert e.value.exit_code == 1
    assert ex.cancelled == ["r1"]


def test_live_scan_refuses_without_risk_caps(monkeypatch):
    from pipeline import commands

    ex = _Exchange()
    monkeypatch.setattr("kalshi_bridge.get_client", lambda *a, **k: ex)
    monkeypatch.setattr("settings_validate.live_trading_errors", lambda: ["Live trading needs a daily loss cap"])
    with pytest.raises(typer.Exit) as e:
        commands.scan(**_scan_kwargs(dry_run=False))
    assert e.value.exit_code == 1
    assert ex.cancelled == []  # nothing placed, nothing touched


def test_live_scan_fails_closed_on_unreadable_balance(monkeypatch, tmp_path):
    from pipeline import commands

    class _NoBalance(_Exchange):
        def get_balance(self):
            raise ConnectionError("down")

    monkeypatch.setattr("kalshi_bridge.get_client", lambda *a, **k: _NoBalance())
    monkeypatch.setattr("settings_validate.live_trading_errors", lambda: [])
    monkeypatch.setattr("risk_manager.KILL_SWITCH_PATH", tmp_path / "KILL")
    monkeypatch.setattr("risk_manager.USE_LIVE_BALANCE", True)
    with pytest.raises(typer.Exit) as e:
        commands.scan(**_scan_kwargs(dry_run=False))
    assert e.value.exit_code == 1
