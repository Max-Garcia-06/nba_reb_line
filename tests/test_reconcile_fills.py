from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pytest

from journal_reader import load_jsonl_rows, parse_iso_date
from reconcile_fills import reconcile_fills_for_date, reconcile_fills_for_range


def _post_submit_row(*, order_id: str = "ord-1", contracts: int = 10) -> dict:
    return {
        "note": "post-submit",
        "success": True,
        "order_id": order_id,
        "ticker": "KXNBAREB-26MAY20BOSNYK-P-2",
        "side": "yes",
        "action": "buy",
        "contracts": contracts,
        "limit_price": 0.40,
        "player_name": "Player A",
        "kalshi_line": 1.5,
        "predicted_lambda": 1.2,
        "p_model": 0.45,
        "p_market": 0.40,
        "edge": 0.05,
        "ev": 0.1,
        "expected_pnl": 0.5,
        "book_bid": 0.38,
        "book_ask": 0.42,
        "book_spread": 0.04,
    }


def _write_journal(tmp_path, game_date: str, rows: list[dict]) -> None:
    path = tmp_path / f"trades_{game_date}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class _MockClient:
    def __init__(self, orders: dict[str, dict] | None = None):
        self.orders = orders or {}
        self.fetched: list[str] = []

    def get_order(self, order_id: str) -> dict:
        self.fetched.append(order_id)
        return self.orders[order_id]


def test_reconcile_writes_fill_row(tmp_path, monkeypatch):
    game_date = "2026-05-20"
    _write_journal(tmp_path, game_date, [_post_submit_row()])
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    client = _MockClient(
        {
            "ord-1": {
                "count": 10,
                "remaining_count": 2,
                "status": "executed",
                "avg_fill_price": 0.41,
            }
        }
    )
    result = reconcile_fills_for_date(game_date, client=client)

    assert result.updated == 1
    assert result.skipped == 0
    assert result.errors == 0
    assert result.days_processed == 1
    rows = load_jsonl_rows(tmp_path / f"trades_{game_date}.jsonl")
    fills = [r for r in rows if r.get("note") == "fill"]
    assert len(fills) == 1
    assert fills[0]["order_id"] == "ord-1"
    assert fills[0]["filled_contracts"] == 8
    assert fills[0]["avg_fill_price"] == pytest.approx(0.41)


def test_reconcile_skips_existing_fill(tmp_path, monkeypatch):
    game_date = "2026-05-20"
    rows = [
        _post_submit_row(),
        {
            "note": "fill",
            "order_id": "ord-1",
            "filled_contracts": 8,
            "avg_fill_price": 0.41,
        },
    ]
    _write_journal(tmp_path, game_date, rows)
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    client = _MockClient({"ord-1": {"count": 10, "remaining_count": 0, "status": "executed"}})
    result = reconcile_fills_for_date(game_date, client=client)

    assert result.updated == 0
    assert result.skipped == 1
    assert client.fetched == []


def test_reconcile_counts_fetch_errors(tmp_path, monkeypatch):
    game_date = "2026-05-20"
    _write_journal(tmp_path, game_date, [_post_submit_row(order_id="ord-err")])
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    class _FailClient:
        def get_order(self, order_id: str):
            raise RuntimeError("api down")

    result = reconcile_fills_for_date(game_date, client=_FailClient())
    assert result.updated == 0
    assert result.errors == 1


class _PurgedOrderClient:
    """Kalshi after the order-retention window: /portfolio/orders/{id} 404s, fills remain."""

    def __init__(self, fills: list[dict]):
        self._fills = fills

    def get_order(self, order_id: str) -> dict:
        raise RuntimeError("404 Client Error: Not Found")

    def get_fills(self, min_ts: int, max_ts: int, limit: int = 1000, max_pages: int = 20) -> list[dict]:
        return self._fills


def test_reconcile_falls_back_to_fills_when_order_purged(tmp_path, monkeypatch):
    game_date = "2026-05-20"
    _write_journal(tmp_path, game_date, [_post_submit_row()])
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    client = _PurgedOrderClient(
        [
            {"order_id": "ord-1", "count_fp": "6.00", "yes_price_dollars": "0.4000"},
            {"order_id": "ord-1", "count_fp": "2.00", "yes_price_dollars": "0.4400"},
        ]
    )
    result = reconcile_fills_for_date(game_date, client=client)

    assert result.updated == 1
    assert result.errors == 0
    fills = [r for r in load_jsonl_rows(tmp_path / f"trades_{game_date}.jsonl") if r.get("note") == "fill"]
    assert fills[0]["filled_contracts"] == 8
    # Size-weighted: (6*0.40 + 2*0.44) / 8
    assert fills[0]["avg_fill_price"] == pytest.approx(0.41)


def test_reconcile_fallback_treats_missing_fill_as_unfilled(tmp_path, monkeypatch):
    game_date = "2026-05-20"
    _write_journal(tmp_path, game_date, [_post_submit_row()])
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    result = reconcile_fills_for_date(game_date, client=_PurgedOrderClient([]))

    assert result.updated == 0
    assert result.errors == 0


def test_reconcile_prefers_executed_price_over_order_record(tmp_path, monkeypatch):
    """The order endpoint reports no fill price, so limit_price must not win over fills."""
    game_date = "2026-05-20"
    _write_journal(tmp_path, game_date, [_post_submit_row(contracts=10)])
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    class _BothSourcesClient(_MockClient):
        def get_fills(self, min_ts: int, max_ts: int, limit: int = 1000, max_pages: int = 20) -> list[dict]:
            return [{"order_id": "ord-1", "count_fp": "8.00", "yes_price_dollars": "0.3700"}]

    # Order record has no price field, so _avg_fill_price_from_order would yield the
    # journal's 0.40 limit. The executed price was 0.37.
    client = _BothSourcesClient({"ord-1": {"count": 10, "remaining_count": 2, "status": "executed"}})
    result = reconcile_fills_for_date(game_date, client=client)

    assert result.updated == 1
    fills = [r for r in load_jsonl_rows(tmp_path / f"trades_{game_date}.jsonl") if r.get("note") == "fill"]
    assert fills[0]["filled_contracts"] == 8
    assert fills[0]["avg_fill_price"] == pytest.approx(0.37)


def test_reconcile_writes_no_row_for_canceled_order(tmp_path, monkeypatch):
    """A canceled order has no fills; the order record alone would book it as fully filled."""
    game_date = "2026-05-20"
    _write_journal(tmp_path, game_date, [_post_submit_row(contracts=10)])
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    class _CanceledClient(_MockClient):
        def get_fills(self, min_ts: int, max_ts: int, limit: int = 1000, max_pages: int = 20) -> list[dict]:
            return []

    # No count/remaining_count keys, mirroring what Kalshi actually returns.
    client = _CanceledClient({"ord-1": {"status": "canceled"}})
    result = reconcile_fills_for_date(game_date, client=client)

    assert result.updated == 0
    rows = load_jsonl_rows(tmp_path / f"trades_{game_date}.jsonl")
    assert [r for r in rows if r.get("note") == "fill"] == []


def test_reconcile_range_multiple_days(tmp_path, monkeypatch):
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)
    _write_journal(tmp_path, "2026-05-20", [_post_submit_row(order_id="o1")])
    _write_journal(tmp_path, "2026-05-21", [_post_submit_row(order_id="o2")])

    client = _MockClient(
        {
            "o1": {"count": 10, "remaining_count": 0, "status": "executed", "avg_fill_price": 0.40},
            "o2": {"count": 5, "remaining_count": 0, "status": "executed", "avg_fill_price": 0.35},
        }
    )
    result = reconcile_fills_for_range(
        parse_iso_date("2026-05-20"),
        parse_iso_date("2026-05-21"),
        client=client,
    )
    assert result.days_processed == 2
    assert result.updated == 2
    assert set(client.fetched) == {"o1", "o2"}


def test_reconcile_range_skips_missing_journal_days(tmp_path, monkeypatch):
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)
    _write_journal(tmp_path, "2026-05-21", [_post_submit_row(order_id="o2")])

    client = _MockClient(
        {"o2": {"count": 5, "remaining_count": 0, "status": "executed", "avg_fill_price": 0.35}}
    )
    result = reconcile_fills_for_range(
        parse_iso_date("2026-05-20"),
        parse_iso_date("2026-05-21"),
        client=client,
    )
    assert result.days_processed == 1
    assert result.updated == 1


def test_report_calls_reconcile_by_default(tmp_path, monkeypatch):
    """report with default reconcile_first invokes reconcile_fills_for_date before loading."""
    game_date = "2026-05-20"
    _write_journal(
        tmp_path,
        game_date,
        [
            _post_submit_row(),
            {
                "note": "fill",
                "order_id": "ord-1",
                "filled_contracts": 10,
                "avg_fill_price": 0.40,
            },
        ],
    )
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    calls: list[str] = []

    def _fake_reconcile(date, *, client, include_resting=False):
        calls.append(date)
        from reconcile_fills import ReconcileResult

        return ReconcileResult(updated=0, skipped=1, errors=0, days_processed=1)

    class _MarketClient:
        def get_market(self, ticker: str):
            return {"result": "yes"}

    monkeypatch.setattr("kalshi_bridge.get_client", lambda: _MarketClient())

    with patch("reconcile_fills.reconcile_fills_for_date", side_effect=_fake_reconcile):
        from pipeline.commands import report

        report(game_date=game_date, reconcile_first=True, include_resting=False)

    assert calls == [game_date]


def test_report_skips_reconcile_when_disabled(tmp_path, monkeypatch):
    game_date = "2026-05-20"
    _write_journal(
        tmp_path,
        game_date,
        [
            _post_submit_row(),
            {
                "note": "fill",
                "order_id": "ord-1",
                "filled_contracts": 10,
                "avg_fill_price": 0.40,
            },
        ],
    )
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)

    class _MarketClient:
        def get_market(self, ticker: str):
            return {"result": "yes"}

    monkeypatch.setattr("kalshi_bridge.get_client", lambda: _MarketClient())

    with patch("reconcile_fills.reconcile_fills_for_date") as mock_rec:
        from pipeline.commands import report

        report(game_date=game_date, reconcile_first=False, include_resting=False)
        mock_rec.assert_not_called()
