from __future__ import annotations

import pytest

from edge_detector import EdgeSignal
from journal_reader import load_jsonl_rows, placed_with_order_id
from paper_trading import paper_fill_rows, record_paper_orders
from trade_journal import journal_path

DATE = "2026-10-20"
TICKER = "KXNBAREB-26OCT20BOSNYK-BOSPLAYERA8-8"


def _sig(side="yes", limit=0.41, ask=0.42, contracts=10, ticker=TICKER):
    return EdgeSignal(
        player_name="Player A", player_id=1, game_date=DATE, ticker=ticker, kalshi_line=7.5,
        predicted_lambda=8.0, p_model=0.5, p_model_raw=0.5, p_market=ask, edge=0.06, ev=0.1, kelly_f=0.02,
        recommended_contracts=contracts, recommended_side=side, bet_dollars=round(contracts * limit, 2),
        limit_price=limit, book_bid=0.40, book_ask=ask, book_spread=round(ask - 0.40, 2),
    )


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)
    return tmp_path


def test_paper_orders_go_to_paper_journal_once_per_ticker_side(data_dir):
    assert record_paper_orders(DATE, [_sig()]) == 1
    assert record_paper_orders(DATE, [_sig(), _sig(side="no", limit=0.57)]) == 1  # later scan: only new key
    placed = placed_with_order_id(load_jsonl_rows(journal_path(DATE, paper=True)))
    assert [(r["ticker"], r["side"]) for r in placed] == [(TICKER, "yes"), (TICKER, "no")]
    assert all(r["order_id"].startswith("paper-") for r in placed)
    assert not journal_path(DATE).exists()  # live journal untouched


def _placed(oid, side="yes", limit=0.41, ask=0.42):
    return {"order_id": oid, "side": side, "limit_price": limit, "book_ask": ask, "contracts": 10,
            "ts": "2026-10-20T22:00:00+00:00"}


def _mark(oid, yes_bid, yes_ask, ts="2026-10-20T22:30:00+00:00"):
    return {"order_id": oid, "mark_label": "30m", "mark_yes_bid": yes_bid, "mark_yes_ask": yes_ask, "ts": ts}


def test_paper_resting_order_fills_only_when_a_mark_shows_the_quote_through_it():
    placed = [_placed("p1"), _placed("p2")]
    marks = [_mark("p1", 0.38, 0.40), _mark("p2", 0.41, 0.43)]
    fills = {f["order_id"]: f for f in paper_fill_rows(placed, marks)}
    assert fills["p1"]["filled_contracts"] == 10 and fills["p1"]["avg_fill_price"] == 0.41
    assert fills["p2"]["filled_contracts"] == 0


def test_paper_taker_order_fills_immediately():
    (f,) = paper_fill_rows([_placed("p3", limit=0.42, ask=0.42)], [])
    assert f["filled_contracts"] == 10


def test_report_range_paper_reads_paper_journal(data_dir, monkeypatch, capsys):
    record_paper_orders(DATE, [_sig(limit=0.42, ask=0.42)])  # taker: fills

    class _C:
        def get_market(self, t):
            return {"result": "yes"}

    monkeypatch.setattr("kalshi_bridge.get_client", lambda: _C())
    from pipeline.commands import report_range

    report_range(start=DATE, end=DATE, per_day=False, paper=True)
    out = capsys.readouterr().out
    assert "PAPER" in out
    assert "Realized P&L (resolved)   : $5.80" in out  # 10 x (1 - 0.42)
