from __future__ import annotations

import pytest

import edge_detector
from edge_detector import EdgeSignal, cancel_unconfirmed_orders, execute_signals
from execution_engine import ExecutionLedger, LedgerKey
from journal_risk import RiskDataUnavailable
from kalshi_bridge import MockKalshiClient, OpenOrder, OrderResult
from trading_stack import finalize_signals

DATE = "2026-10-12"
T1 = "KXNBAREB-26OCT12BOSNYK-BOSPLAYERA8-8"
T2 = "KXNBAREB-26OCT12BOSNYK-NYKPLAYERB6-6"


def _sig(ticker=T1, side="yes", player="A", contracts=10, limit=0.49, ask=0.50, ev=0.1, bet=None) -> EdgeSignal:
    return EdgeSignal(
        player_name=player, player_id=1, game_date=DATE, ticker=ticker, kalshi_line=7.5,
        predicted_lambda=8.0, p_model=0.6, p_model_raw=0.6, p_market=ask, edge=0.1, ev=ev,
        kelly_f=0.05, recommended_contracts=contracts, recommended_side=side,
        bet_dollars=bet if bet is not None else round(contracts * limit, 2), limit_price=limit,
        book_bid=0.48, book_ask=ask, book_spread=round(ask - 0.48, 2),
    )


class _Exchange:
    def __init__(self, resting=None, orders_error=False, place_result=None):
        self.resting = list(resting or [])
        self.orders_error = orders_error
        self.place_result = place_result
        self.placed: list[dict] = []
        self.cancelled: list[str] = []

    def get_orders(self, status="resting", ticker=None, limit=200):
        if self.orders_error:
            raise ConnectionError("down")
        return [o for o in self.resting if ticker is None or o.ticker == ticker]

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return True

    def place_order(self, ticker, side, contracts, price, order_type="limit", **kw):
        self.placed.append({"ticker": ticker, "side": side, "contracts": contracts, "price": price, **kw})
        if self.place_result is not None:
            return self.place_result
        return OrderResult(True, f"oid-{len(self.placed)}", ticker, side, contracts, price, "ok")


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    led_path = tmp_path / "ledger.json"
    monkeypatch.setattr(edge_detector, "ExecutionLedger", lambda: ExecutionLedger(led_path))
    monkeypatch.setattr(edge_detector, "append_row", lambda *a, **k: None)
    monkeypatch.setattr("risk_manager.KILL_SWITCH_PATH", tmp_path / "KILL")
    return led_path


EXP = {T1: 1_800_000_000, T2: 1_800_000_000}


# ---------- finalize_signals: exposure-aware sizing ----------

def test_finalize_caps_player_by_existing_exposure():
    out = finalize_signals([_sig(contracts=20, limit=0.50)], bankroll=1000, existing_player_usd={"a": 15.0},
                           max_player_usd=20.0)
    (s,) = out
    assert s.bet_dollars <= 5.0 + 1e-9
    assert s.recommended_contracts == 10


def test_finalize_drops_player_already_at_cap():
    out = finalize_signals([_sig()], bankroll=1000, existing_player_usd={"a": 25.0}, max_player_usd=20.0)
    assert out == []


# ---------- execute_signals ----------

def test_execute_refuses_mock_client(ledger):
    with pytest.raises(RuntimeError, match="mock"):
        execute_signals([_sig()], client=MockKalshiClient(), expiration_by_ticker=EXP)


def test_execute_sends_client_order_id_post_only_and_expiry(ledger):
    ex = _Exchange()
    (r,) = execute_signals([_sig(limit=0.49, ask=0.50)], client=ex, expiration_by_ticker=EXP)
    assert r.success
    (p,) = ex.placed
    assert p["client_order_id"] and set(p["client_order_id"]) <= set("0123456789abcdef-")
    assert p["post_only"] is True
    assert p["expiration_time"] == EXP[T1]


def test_taker_order_is_not_post_only(ledger):
    ex = _Exchange()
    execute_signals([_sig(limit=0.50, ask=0.50)], client=ex, expiration_by_ticker=EXP)
    assert ex.placed[0]["post_only"] is False


def test_execute_skips_order_without_known_tip_time(ledger):
    ex = _Exchange()
    assert execute_signals([_sig()], client=ex, expiration_by_ticker={}) == []
    assert ex.placed == []


def test_kill_switch_stops_orders(ledger, tmp_path):
    (tmp_path / "KILL").write_text("halt")
    ex = _Exchange()
    execute_signals([_sig(), _sig(ticker=T2, player="B")], client=ex, expiration_by_ticker=EXP)
    assert ex.placed == []


def test_existing_resting_order_at_same_price_is_not_duplicated(ledger):
    ex = _Exchange(resting=[OpenOrder("r1", T1, "yes", "", "limit", "resting", 0.49, 10, 10)])
    execute_signals([_sig(limit=0.49)], client=ex, expiration_by_ticker=EXP)
    assert ex.placed == [] and ex.cancelled == []


def test_existing_resting_order_at_stale_price_is_replaced(ledger):
    ex = _Exchange(resting=[OpenOrder("r1", T1, "yes", "", "limit", "resting", 0.44, 10, 10)])
    execute_signals([_sig(limit=0.49)], client=ex, expiration_by_ticker=EXP)
    assert ex.cancelled == ["r1"] and len(ex.placed) == 1


def test_unreadable_resting_orders_blocks_placement(ledger):
    ex = _Exchange(orders_error=True)
    execute_signals([_sig()], client=ex, expiration_by_ticker=EXP)
    assert ex.placed == []


def test_ambiguous_submit_is_never_resent(ledger):
    ex = _Exchange(place_result=OrderResult(False, "", T1, "yes", 10, 0.49, "Network error", ambiguous=True))
    execute_signals([_sig()], client=ex, expiration_by_ticker=EXP)
    assert ExecutionLedger(ledger).has_successful_submit(LedgerKey(DATE, T1, "yes"))
    ex2 = _Exchange()
    execute_signals([_sig()], client=ex2, expiration_by_ticker=EXP)
    assert ex2.placed == []


# ---------- stale-order sweep ----------

def test_sweep_cancels_unconfirmed_orders_for_the_date_only():
    ex = _Exchange(resting=[
        OpenOrder("keep", T1, "yes", "", "limit", "resting", 0.49, 10, 10),
        OpenOrder("stale", T2, "no", "", "limit", "resting", 0.40, 5, 5),
        OpenOrder("other-day", "KXNBAREB-26OCT13BOSNYK-X8-8", "yes", "", "limit", "resting", 0.5, 1, 1),
        OpenOrder("other-series", "KXMLBTB-26OCT12A-B-2", "yes", "", "limit", "resting", 0.5, 1, 1),
    ])
    cancelled = cancel_unconfirmed_orders(ex, game_date=DATE, series_ticker="KXNBAREB", keep={(T1, "yes")})
    assert cancelled == ["stale"]
    assert ex.cancelled == ["stale"]


def test_sweep_with_nothing_kept_cancels_all_of_todays_orders():
    ex = _Exchange(resting=[OpenOrder("a", T1, "yes", "", "limit", "resting", 0.49, 10, 10)])
    assert cancel_unconfirmed_orders(ex, game_date=DATE, series_ticker="KXNBAREB", keep=set()) == ["a"]


def test_sweep_raises_when_orders_unreadable():
    with pytest.raises(RiskDataUnavailable):
        cancel_unconfirmed_orders(_Exchange(orders_error=True), game_date=DATE, series_ticker="KXNBAREB", keep=set())
