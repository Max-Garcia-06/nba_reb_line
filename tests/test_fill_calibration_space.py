from __future__ import annotations

import json

import pytest

from calibration import fill_calibration_rows, fit_segmented, segment_key

STAMP = "2026-10-11T00:00:00"


def _fill(side, p_raw, ticker="T", stamp=STAMP, filled=5, line=7.5):
    return {"note": "fill", "order_id": f"{side}{p_raw}", "ticker": ticker, "side": side,
            "p_model_raw": p_raw, "p_model": p_raw, "filled_contracts": filled, "kalshi_line": line,
            "games_played": 60, "model_trained_on": stamp}


def test_no_fill_is_mapped_into_over_space():
    (r,) = fill_calibration_rows([_fill("no", 0.70)], outcome_for=lambda t: "no", model_trained_on=STAMP)
    # P(under)=0.70 and the under won  =>  P(over)=0.30 and the over lost.
    assert r["p"] == pytest.approx(0.30) and r["y"] == 0.0


def test_yes_fill_stays_in_over_space():
    (r,) = fill_calibration_rows([_fill("yes", 0.60)], outcome_for=lambda t: "yes", model_trained_on=STAMP)
    assert r["p"] == pytest.approx(0.60) and r["y"] == 1.0


def test_rows_from_other_or_unknown_models_are_dropped():
    rows = [_fill("yes", 0.6), _fill("yes", 0.6, stamp="2026-09-01T00:00:00"), _fill("yes", 0.6, stamp=None)]
    out = fill_calibration_rows(rows, outcome_for=lambda t: "yes", model_trained_on=STAMP)
    assert len(out) == 1


def test_unsettled_and_scalar_rows_are_dropped():
    out = fill_calibration_rows([_fill("yes", 0.6)], outcome_for=lambda t: "", model_trained_on=STAMP)
    assert out == []


def test_segment_key_has_no_side():
    assert segment_key(line=7.5, games_played=60) == "7.5|gp50plus"


def test_bundle_transform_is_side_free():
    rows = [{"p": 0.6, "y": float(i % 2), "line": 7.5, "games_played": 60} for i in range(40)]
    b = fit_segmented(rows, min_global=10, min_segment=10)
    assert b.transform(0.6, line=7.5, games_played=60) == b.transform(0.6, line=7.5, games_played=60)
    assert "7.5|gp50plus" in b.segments


def test_execute_signals_journals_model_stamp(tmp_path, monkeypatch):
    import edge_detector
    from execution_engine import ExecutionLedger
    from tests.test_live_execution import EXP, _Exchange, _sig

    rows = []
    monkeypatch.setattr(edge_detector, "ExecutionLedger", lambda: ExecutionLedger(tmp_path / "l.json"))
    monkeypatch.setattr(edge_detector, "append_row", lambda d, row, **kw: rows.append(row))
    monkeypatch.setattr("risk_manager.KILL_SWITCH_PATH", tmp_path / "KILL")
    monkeypatch.setattr("model.get_model_trained_on", lambda: STAMP)
    edge_detector.execute_signals([_sig()], client=_Exchange(), expiration_by_ticker=EXP)
    assert rows and all(r["model_trained_on"] == STAMP for r in rows)


def test_reconcile_copies_model_stamp(tmp_path, monkeypatch):
    from journal_reader import index_fills_by_order_id, load_jsonl_rows
    from reconcile_fills import reconcile_fills_for_date

    monkeypatch.setattr("trade_journal.DATA_DIR", tmp_path)
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)
    placed = {"note": "post-submit", "success": True, "order_id": "o1", "ticker": "T", "side": "yes",
              "contracts": 5, "limit_price": 0.4, "model_trained_on": STAMP}
    (tmp_path / "trades_2026-10-20.jsonl").write_text(json.dumps(placed) + "\n")

    class _C:
        def get_order(self, oid):
            return {"status": "executed"}

        def get_fills(self, min_ts, max_ts, limit=1000, max_pages=20):
            return [{"order_id": "o1", "count_fp": "5.00", "yes_price_dollars": "0.4000"}]

    reconcile_fills_for_date("2026-10-20", client=_C())
    fill = index_fills_by_order_id(load_jsonl_rows(tmp_path / "trades_2026-10-20.jsonl"))["o1"]
    assert fill["model_trained_on"] == STAMP
