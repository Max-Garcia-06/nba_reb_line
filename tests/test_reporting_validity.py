from __future__ import annotations

import json

import pytest

from journal_reader import index_fills_by_order_id, load_jsonl_rows
from reconcile_fills import reconcile_fills_for_date
from reporting_common import line_bucket, market_settlement, pnl_per_contract

DATE = "2026-05-20"
TICKER = "KXNBAREB-26MAY20BOSNYK-BOSPLAYER8-8"


# ---------- settlement ----------

def test_scalar_market_settles_at_its_settlement_value():
    m = {"result": "scalar", "settlement_value_dollars": "0.0500"}
    v = market_settlement(m)
    assert v == pytest.approx(0.05)
    assert pnl_per_contract("yes", 0.40, v) == pytest.approx(0.05 - 0.40)
    assert pnl_per_contract("no", 0.55, v) == pytest.approx(0.95 - 0.55)


def test_binary_settlement_unchanged():
    assert market_settlement({"result": "yes"}) == "yes"
    assert pnl_per_contract("yes", 0.40, "yes") == pytest.approx(0.60)
    assert pnl_per_contract("no", 0.40, "yes") == pytest.approx(-0.40)


def test_unsettled_market_has_no_pnl():
    assert market_settlement({"result": ""}) == ""
    assert pnl_per_contract("yes", 0.4, "") is None
    assert market_settlement({"result": "scalar"}) == ""  # no value yet


# ---------- buckets ----------

@pytest.mark.parametrize(
    "line,bucket",
    [(1.5, "<=4.5"), (4.5, "<=4.5"), (5.5, "5.5-7.5"), (7.5, "5.5-7.5"), (8.5, "8.5-10.5"),
     (10.5, "8.5-10.5"), (11.5, ">=11.5"), (19.5, ">=11.5")],
)
def test_nba_line_buckets(line, bucket):
    assert line_bucket(line) == bucket


# ---------- report ----------

def _placed(order_id="ord-1", contracts=10, price=0.40, side="yes", ticker=TICKER):
    return {"note": "post-submit", "success": True, "order_id": order_id, "ticker": ticker, "side": side,
            "action": "buy", "contracts": contracts, "limit_price": price, "player_name": "Player A",
            "kalshi_line": 7.5, "p_model": 0.5, "edge": 0.06, "book_spread": 0.02}


def _fill(order_id="ord-1", filled=10, price=0.40, **kw):
    row = {"note": "fill", "order_id": order_id, "filled_contracts": filled, "avg_fill_price": price,
           "fee_per_contract": 0.0}
    row.update(kw)
    return row


def _write(tmp_path, rows):
    (tmp_path / f"trades_{DATE}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def journal_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)
    return tmp_path


def _report(monkeypatch, markets):
    class _C:
        def get_market(self, t):
            return markets.get(t, {"result": ""})

    monkeypatch.setattr("kalshi_bridge.get_client", lambda: _C())
    from pipeline.commands import report

    report(game_date=DATE, reconcile_first=False, include_resting=False)


def test_report_resolves_scalar_and_excludes_unresolved_cost(journal_dir, monkeypatch, capsys):
    t2 = TICKER.replace("8-8", "6-6")
    _write(journal_dir, [_placed(), _fill(), _placed("ord-2", ticker=t2), _fill("ord-2")])
    # ord-1 settled scalar at 0.05 (player sat): -0.35 x 10; ord-2 still open.
    _report(monkeypatch, {TICKER: {"result": "scalar", "settlement_value_dollars": "0.0500"}})
    out = capsys.readouterr().out
    assert "Realized P&L (resolved)   : $-3.50" in out
    assert "ROI -87.50%" in out  # -3.50 / 4.00 resolved cost, not / 8.00


# ---------- reconcile ----------

class _Orders:
    def __init__(self, orders, fills):
        self.orders, self._fills = orders, fills

    def get_order(self, oid):
        return self.orders[oid]

    def get_fills(self, min_ts, max_ts, limit=1000, max_pages=20):
        return self._fills


def test_reconcile_updates_partial_fill_once_order_completes(journal_dir):
    _write(journal_dir, [_placed()])
    resting = _Orders({"ord-1": {"status": "resting"}},
                      [{"order_id": "ord-1", "count_fp": "4.00", "yes_price_dollars": "0.4000"}])
    reconcile_fills_for_date(DATE, client=resting)
    done = _Orders({"ord-1": {"status": "executed"}},
                   [{"order_id": "ord-1", "count_fp": "4.00", "yes_price_dollars": "0.4000"},
                    {"order_id": "ord-1", "count_fp": "6.00", "yes_price_dollars": "0.4000"}])
    reconcile_fills_for_date(DATE, client=done)
    fills = index_fills_by_order_id(load_jsonl_rows(journal_dir / f"trades_{DATE}.jsonl"))
    assert fills["ord-1"]["filled_contracts"] == 10
    # terminal now: a third run writes nothing new
    n = len(load_jsonl_rows(journal_dir / f"trades_{DATE}.jsonl"))
    reconcile_fills_for_date(DATE, client=done)
    assert len(load_jsonl_rows(journal_dir / f"trades_{DATE}.jsonl")) == n


def test_reconcile_records_actual_fees_from_order(journal_dir):
    _write(journal_dir, [_placed()])
    c = _Orders({"ord-1": {"status": "executed", "taker_fees_dollars": "0.1700", "maker_fees_dollars": "0.0000"}},
                [{"order_id": "ord-1", "count_fp": "10.00", "yes_price_dollars": "0.4000"}])
    reconcile_fills_for_date(DATE, client=c)
    fill = index_fills_by_order_id(load_jsonl_rows(journal_dir / f"trades_{DATE}.jsonl"))["ord-1"]
    assert fill["fee_per_contract"] == pytest.approx(0.017)
