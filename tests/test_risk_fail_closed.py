from __future__ import annotations

import json

import pytest

import execution_engine
from execution_engine import ExecutionLedger, LedgerCorruptError, LedgerKey
from journal_risk import RiskDataUnavailable, daily_risk_snapshot, open_resting_usd
from kalshi_bridge import OpenOrder
from risk_manager import check_pre_trade_risk, resolve_bankroll

TICKER = "KXNBAREB-26OCT12BOSNYK-BOSPLAYER8-8"


def _placed(order_id="o1", side="yes", contracts=10, price=0.50, **kw):
    row = {
        "note": "post-submit", "success": True, "order_id": order_id, "ticker": TICKER,
        "side": side, "contracts": contracts, "limit_price": price,
        "player_name": "Player A", "kalshi_line": 7.5,
    }
    row.update(kw)
    return row


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


# ---------- execution ledger ----------

def test_corrupt_ledger_raises_instead_of_starting_empty(tmp_path):
    p = tmp_path / "ledger.json"
    p.write_text("{not json")
    with pytest.raises(LedgerCorruptError):
        ExecutionLedger(p)
    assert p.read_text() == "{not json"  # history is not overwritten


def test_ledger_save_is_atomic_replace(tmp_path, monkeypatch):
    calls = []
    real_replace = execution_engine.os.replace
    monkeypatch.setattr(execution_engine.os, "replace", lambda a, b: calls.append((a, b)) or real_replace(a, b))
    p = tmp_path / "ledger.json"
    led = ExecutionLedger(p)
    led.add_attempt(LedgerKey("2026-10-12", "T", "yes"), price=0.5, contracts=1, dollars=0.5, note="pre-submit")
    assert calls and str(calls[0][1]) == str(p)
    assert json.loads(p.read_text())["entries"]


def test_ambiguous_submit_blocks_resubmit(tmp_path):
    led = ExecutionLedger(tmp_path / "ledger.json")
    key = LedgerKey("2026-10-12", "T", "yes")
    led.add_attempt(key, price=0.5, contracts=1, dollars=0.5, note="post-submit", success=False, ambiguous=True)
    assert led.has_successful_submit(key)


def test_rejected_submit_does_not_block_resubmit(tmp_path):
    led = ExecutionLedger(tmp_path / "ledger.json")
    key = LedgerKey("2026-10-12", "T", "yes")
    led.add_attempt(key, price=0.5, contracts=1, dollars=0.5, note="post-submit", success=False)
    assert not led.has_successful_submit(key)


# ---------- bankroll / resting exposure ----------

class _BrokenBalance:
    def get_balance(self):
        raise ConnectionError("down")


class _ZeroBalance:
    def get_balance(self):
        return 0.0


@pytest.mark.parametrize("client", [_BrokenBalance(), _ZeroBalance()])
def test_strict_bankroll_fails_closed(client):
    with pytest.raises(RiskDataUnavailable):
        resolve_bankroll(1000.0, client, use_live_balance=True, strict=True)


def test_non_strict_bankroll_keeps_cli_fallback():
    assert resolve_bankroll(1000.0, _BrokenBalance(), use_live_balance=True) == 1000.0


class _BrokenOrders:
    def get_orders(self, status="resting", limit=200):
        raise ConnectionError("down")


def test_strict_open_resting_fails_closed():
    with pytest.raises(RiskDataUnavailable):
        open_resting_usd(_BrokenOrders(), strict=True)
    assert open_resting_usd(_BrokenOrders()) == 0.0


def test_open_resting_sums_remaining_notional():
    class _C:
        def get_orders(self, status="resting", limit=200):
            return [OpenOrder("o", "T", "no", "", "limit", "resting", 0.40, 10, 6)]

    assert open_resting_usd(_C(), strict=True) == pytest.approx(2.40)


def test_check_pre_trade_risk_blocks_when_risk_data_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr("risk_manager.KILL_SWITCH_PATH", tmp_path / "KILL")
    monkeypatch.setattr("risk_manager.AUTO_KILL_ON_RISK_BREACH", False)
    monkeypatch.setattr("journal_risk.journal_path", lambda d: tmp_path / f"trades_{d}.jsonl")

    class _C(_BrokenOrders):
        def get_balance(self):
            return 500.0

    res = check_pre_trade_risk(game_date="2026-10-12", cli_bankroll=1000.0, client=_C(), strict=True)
    assert not res.ok
    assert "unavailable" in res.reason.lower()


# ---------- intraday P&L from the exchange ----------

class _Exchange:
    """Unsettled market at 0.30/0.32 YES; order o1 fully filled on the exchange."""

    def __init__(self, fills=None, result=""):
        self._fills = fills if fills is not None else [
            {"order_id": "o1", "count_fp": "10.00", "yes_price_dollars": "0.5000", "no_price_dollars": "0.5000"}
        ]
        self._result = result

    def get_fills(self, min_ts, max_ts, limit=1000, max_pages=20):
        return self._fills

    def get_market(self, ticker):
        return {"result": self._result, "yes_bid_dollars": "0.3000", "yes_ask_dollars": "0.3200",
                "no_bid_dollars": "0.6800", "no_ask_dollars": "0.7000"}

    def get_orders(self, status="resting", limit=200):
        return []


def test_snapshot_counts_exchange_fills_before_nightly_reconcile(tmp_path, monkeypatch):
    j = tmp_path / "trades_2026-10-12.jsonl"
    _write(j, [_placed()])  # no note=fill rows yet
    monkeypatch.setattr("journal_risk.journal_path", lambda d: tmp_path / f"trades_{d}.jsonl")
    snap = daily_risk_snapshot("2026-10-12", client=_Exchange(), strict=True)
    # long 10 YES @ 0.50, live mid 0.31 => MTM -1.90
    assert snap.mtm_pnl == pytest.approx(-1.90)
    assert snap.contracts_by_game.get("BOSNYK") == 10


def test_snapshot_includes_prior_day_realized_loss(tmp_path, monkeypatch):
    _write(tmp_path / "trades_2026-10-11.jsonl", [_placed(order_id="old", side="yes")])
    _write(tmp_path / "trades_2026-10-12.jsonl", [])
    monkeypatch.setattr("journal_risk.journal_path", lambda d: tmp_path / f"trades_{d}.jsonl")
    ex = _Exchange(fills=[{"order_id": "old", "count_fp": "10.00", "yes_price_dollars": "0.5000"}], result="no")
    snap = daily_risk_snapshot("2026-10-12", client=ex, strict=True, pnl_lookback_days=2)
    assert snap.realized_pnl == pytest.approx(-5.0)
    assert snap.orders_placed == 0  # counts/deployment stay per-day


def test_strict_snapshot_fails_closed_when_market_fetch_fails(tmp_path, monkeypatch):
    _write(tmp_path / "trades_2026-10-12.jsonl", [_placed()])
    monkeypatch.setattr("journal_risk.journal_path", lambda d: tmp_path / f"trades_{d}.jsonl")

    class _C(_Exchange):
        def get_market(self, ticker):
            raise ConnectionError("down")

    with pytest.raises(RiskDataUnavailable):
        daily_risk_snapshot("2026-10-12", client=_C(), strict=True)


def test_loss_limit_trips_on_intraday_mtm(tmp_path, monkeypatch):
    _write(tmp_path / "trades_2026-10-12.jsonl", [_placed()])
    monkeypatch.setattr("journal_risk.journal_path", lambda d: tmp_path / f"trades_{d}.jsonl")
    monkeypatch.setattr("risk_manager.KILL_SWITCH_PATH", tmp_path / "KILL")
    monkeypatch.setattr("risk_manager.AUTO_KILL_ON_RISK_BREACH", False)
    monkeypatch.setattr("risk_manager.DAILY_LOSS_LIMIT_USD", 1.0)
    monkeypatch.setattr("risk_manager.DAILY_LOSS_LIMIT_PCT", 0.0)

    class _C(_Exchange):
        def get_balance(self):
            return 500.0

    res = check_pre_trade_risk(game_date="2026-10-12", cli_bankroll=500.0, client=_C(), strict=True)
    assert not res.ok and "loss limit" in res.reason.lower()
