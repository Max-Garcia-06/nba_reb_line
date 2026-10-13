from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from journal_reader import CLOSE_MARK_LABEL, due_mark_labels, load_jsonl_rows
from segment_health import aggregate_segments

DATE = "2026-10-20"
TICKER = "KXNBAREB-26OCT20BOSNYK-BOSPLAYER8-8"
T0 = datetime(2026, 10, 20, 22, 0, tzinfo=timezone.utc)  # order placed
TIP = T0 + timedelta(minutes=90)


# ---------- due_mark_labels ----------

def test_offsets_are_relative_to_the_orders_own_placement():
    assert due_mark_labels(T0, TIP, T0 + timedelta(minutes=16), set()) == ["15m"]
    assert due_mark_labels(T0, TIP, T0 + timedelta(minutes=31), {"15m"}) == ["30m"]


def test_late_run_does_not_backfill_mislabeled_marks():
    # 45 min after placement, the 15m slot is long gone: don't label a 45m price "15m".
    assert due_mark_labels(T0, TIP, T0 + timedelta(minutes=45), set()) == []


def test_close_mark_only_in_final_minutes_before_tip():
    assert CLOSE_MARK_LABEL in due_mark_labels(T0, TIP, TIP - timedelta(minutes=3), {"15m", "30m", "60m"})
    assert CLOSE_MARK_LABEL not in due_mark_labels(T0, TIP, TIP - timedelta(minutes=30), set())


def test_nothing_is_marked_once_the_game_has_started():
    assert due_mark_labels(T0, TIP, TIP + timedelta(minutes=1), set()) == []


def test_unknown_tip_marks_nothing():
    assert due_mark_labels(T0, None, T0 + timedelta(minutes=16), set()) == []


# ---------- mark command ----------

def _placed(order_id="o1", ts=T0):
    return {"note": "post-submit", "success": True, "order_id": order_id, "ticker": TICKER, "side": "yes",
            "contracts": 10, "limit_price": 0.40, "player_name": "Player A", "kalshi_line": 7.5,
            "ts": ts.isoformat()}


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)
    monkeypatch.setattr("data_engine.tip_epoch_by_ticker",
                        lambda lines, d, **kw: {ml.ticker: int(TIP.timestamp()) for ml in lines})

    class _C:
        def get_market(self, t):
            return {"yes_bid_dollars": "0.4400", "yes_ask_dollars": "0.4600",
                    "no_bid_dollars": "0.5400", "no_ask_dollars": "0.5600"}

    monkeypatch.setattr("kalshi_bridge.get_client", lambda: _C())
    path = tmp_path / f"trades_{DATE}.jsonl"
    path.write_text(json.dumps(_placed()) + "\n")
    return path


def _run_mark(monkeypatch, now):
    from pipeline import commands

    monkeypatch.setattr(commands, "_utc_now", lambda: now)
    commands.mark(game_date=DATE, paper=False)


def _marks(path):
    return [r for r in load_jsonl_rows(path) if r.get("note") == "mark"]


def test_mark_writes_labelled_rows_once(journal, monkeypatch):
    _run_mark(monkeypatch, T0 + timedelta(minutes=16))
    _run_mark(monkeypatch, T0 + timedelta(minutes=17))
    (m,) = _marks(journal)
    assert m["mark_label"] == "15m"
    assert m["mark_yes_mid"] == pytest.approx(0.45)


def test_mark_after_tip_writes_nothing(journal, monkeypatch):
    _run_mark(monkeypatch, TIP + timedelta(minutes=5))
    assert _marks(journal) == []


def test_mark_writes_close_before_tip(journal, monkeypatch):
    _run_mark(monkeypatch, TIP - timedelta(minutes=2))
    assert [m["mark_label"] for m in _marks(journal)] == [CLOSE_MARK_LABEL]


# ---------- segment health ----------

def test_segment_clv_uses_close_mark_and_ignores_unlabelled_legacy_marks():
    placed = [_placed()]
    fills = [{"note": "fill", "order_id": "o1", "filled_contracts": 10, "avg_fill_price": 0.40}]
    marks = [
        {"note": "mark", "order_id": "o1", "mark_label": "", "mark_yes_mid": 0.95, "ts": "2026-10-21T03:00:00"},
        {"note": "mark", "order_id": "o1", "mark_label": CLOSE_MARK_LABEL, "mark_yes_mid": 0.45,
         "ts": "2026-10-20T23:27:00"},
        {"note": "mark", "order_id": "o1", "mark_label": "60m", "mark_yes_mid": 0.42, "ts": "2026-10-20T23:01:00"},
    ]
    segs = aggregate_segments(placed, fills, marks, outcome_for=lambda t: "")
    (seg,) = [s for s in segs.values() if s.clv_contracts]
    assert seg.clv_sum / seg.clv_contracts == pytest.approx(0.05)
