from __future__ import annotations

import requests

from kalshi_bridge import KalshiClient


def _client() -> KalshiClient:
    return KalshiClient.__new__(KalshiClient)


def test_get_orders_parses_v2_schema(monkeypatch):
    # Current /portfolio/orders payload: outcome_side + *_dollars prices + *_fp counts,
    # no legacy side/count/remaining_count fields.
    payload = {
        "orders": [
            {
                "order_id": "o1",
                "ticker": "KXNBAREB-26OCT12OKCDEN-OKCJWILLIAMS8-8",
                "outcome_side": "no",
                "book_side": "ask",
                "type": "limit",
                "status": "resting",
                "yes_price_dollars": "0.6200",
                "no_price_dollars": "0.3800",
                "initial_count_fp": "10.00",
                "fill_count_fp": "4.00",
                "remaining_count_fp": "6.00",
                "client_order_id": "abc",
            }
        ]
    }
    c = _client()
    monkeypatch.setattr(c, "_get", lambda path, params=None: payload)
    (o,) = c.get_orders()
    assert o.side == "no"
    assert o.price == 0.38
    assert o.count == 10
    assert o.remaining_count == 6
    assert o.client_order_id == "abc"


def test_get_orders_prefers_outcome_side_over_book_side(monkeypatch):
    payload = {"orders": [{"order_id": "o2", "ticker": "T", "book_side": "bid", "outcome_side": "yes",
                           "yes_price_dollars": "0.4100", "remaining_count_fp": "2.00", "initial_count_fp": "2.00"}]}
    c = _client()
    monkeypatch.setattr(c, "_get", lambda path, params=None: payload)
    (o,) = c.get_orders()
    assert (o.side, o.price, o.remaining_count) == ("yes", 0.41, 2)


def test_place_order_sends_idempotency_post_only_and_expiry(monkeypatch):
    sent = {}

    def fake_post(path, body):
        sent.update(body)
        return {"order_id": "x1"}

    c = _client()
    monkeypatch.setattr(c, "_post_order", fake_post)
    r = c.place_order("T", "no", 5, 0.40, client_order_id="cid-1", post_only=True, expiration_time=1_800_000_000)
    assert r.success and r.order_id == "x1"
    assert sent["client_order_id"] == "cid-1"
    assert sent["post_only"] is True
    assert sent["expiration_time"] == 1_800_000_000
    assert sent["side"] == "ask" and sent["price"] == "0.6000"


def test_place_order_omits_optional_fields_when_unset(monkeypatch):
    sent = {}
    c = _client()
    monkeypatch.setattr(c, "_post_order", lambda path, body: sent.update(body) or {"order_id": "x"})
    c.place_order("T", "yes", 1, 0.5)
    assert "client_order_id" not in sent and "expiration_time" not in sent
    assert "post_only" not in sent


def test_place_order_network_error_is_ambiguous(monkeypatch):
    def boom(path, body):
        raise requests.ConnectionError("reset")

    c = _client()
    monkeypatch.setattr(c, "_post_order", boom)
    r = c.place_order("T", "yes", 1, 0.5, client_order_id="cid")
    assert not r.success
    assert r.ambiguous


def test_place_order_http_error_is_not_ambiguous(monkeypatch):
    def rejected(path, body):
        raise requests.HTTPError("400")

    c = _client()
    monkeypatch.setattr(c, "_post_order", rejected)
    r = c.place_order("T", "yes", 1, 0.5)
    assert not r.success and not r.ambiguous


def test_cancel_order_network_error_returns_false(monkeypatch):
    def boom(path, params=None):
        raise requests.Timeout("slow")

    c = _client()
    monkeypatch.setattr(c, "_delete", boom)
    assert c.cancel_order("o1") is False
