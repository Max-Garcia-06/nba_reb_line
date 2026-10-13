from __future__ import annotations

import pytest

import backtest
import market_blend
from backtest import Candle, day_bootstrap_roi_ci, simulate_fill
from market_blend import BlendParams

DEC, TIP = 1_000, 10_000


def _c(end, bid=0.40, ask=0.42, last=0.0, vol=0.0):
    return Candle(end_ts=end, yes_bid=bid, yes_ask=ask, last_price=last, volume=vol)


# ---------- fill model ----------

def test_taker_order_fills_in_full():
    assert simulate_fill("yes", 0.42, 10, is_taker=True, later=[], decision_ts=DEC, tip_ts=TIP) == 10


def test_maker_yes_never_reached_does_not_fill():
    later = [_c(2000, last=0.43, vol=50), _c(3000, bid=0.41, ask=0.43)]
    assert simulate_fill("yes", 0.41, 10, is_taker=False, later=later, decision_ts=DEC, tip_ts=TIP) == 0


def test_maker_yes_trade_through_fills_up_to_traded_volume():
    later = [_c(2000, last=0.40, vol=4), _c(3000, last=0.39, vol=3)]
    assert simulate_fill("yes", 0.41, 10, is_taker=False, later=later, decision_ts=DEC, tip_ts=TIP) == 7


def test_maker_yes_ask_through_limit_fills_in_full():
    later = [_c(2000, bid=0.38, ask=0.40)]
    assert simulate_fill("yes", 0.41, 10, is_taker=False, later=later, decision_ts=DEC, tip_ts=TIP) == 10


def test_maker_no_mirrors_yes_prices():
    # Buying NO at 0.55 fills when YES trades at >= 0.45.
    later = [_c(2000, last=0.46, vol=20)]
    assert simulate_fill("no", 0.55, 10, is_taker=False, later=later, decision_ts=DEC, tip_ts=TIP) == 10
    later = [_c(2000, last=0.44, vol=20)]
    assert simulate_fill("no", 0.55, 10, is_taker=False, later=later, decision_ts=DEC, tip_ts=TIP) == 0


def test_candles_after_tip_or_before_decision_are_ignored():
    later = [_c(DEC, last=0.30, vol=99), _c(TIP + 3600, last=0.30, vol=99)]
    assert simulate_fill("yes", 0.41, 10, is_taker=False, later=later, decision_ts=DEC, tip_ts=TIP) == 0


# ---------- walk-forward ----------

def test_walk_forward_blend_fits_only_on_earlier_weeks(monkeypatch):
    from model_vs_market import ScoreRow

    scored_days = []

    def fake_eval(d, **kw):
        scored_days.append(d)
        return [ScoreRow(d, "a", "T", 7.5, 0.6, 0.6, 0.5, 8, 1.0)]

    fits = []
    monkeypatch.setattr("model_vs_market.evaluate_day", fake_eval)
    monkeypatch.setattr(backtest, "fit_capped_blend",
                        lambda rows: fits.append(sorted({r["date"] for r in rows})) or (BlendParams(0.3, 0.2), {}))
    monkeypatch.setattr(backtest, "run_backtest_day", lambda d, **kw: None)
    backtest.run_backtest_range("2026-04-06", "2026-04-14", blend="walk_forward", blend_since="2026-03-30")
    # Two weeks (Mon 04-06, Mon 04-13); each fit sees only days before its Monday.
    assert len(fits) == 2
    assert max(fits[0]) < "2026-04-06" and max(fits[1]) < "2026-04-13"
    assert "2026-04-12" in fits[1]


# ---------- bootstrap ----------

def test_day_bootstrap_roi_ci_brackets_point_estimate():
    days = [(1.0, 10.0), (-0.5, 10.0), (2.0, 10.0), (0.2, 10.0), (-1.0, 10.0)] * 4
    point, lo, hi = day_bootstrap_roi_ci(days, n_boot=500, seed=1)
    assert lo <= point <= hi
    assert point == pytest.approx(sum(p for p, _ in days) / sum(c for _, c in days) * 100)


def test_day_bootstrap_dll_ci_brackets_point_and_is_paired():
    from model_vs_market import ScoreRow, day_bootstrap_dll_ci

    rows = []
    for i in range(30):
        d = f"2026-04-{i % 28 + 1:02d}"
        y = 1.0 if i % 2 else 0.0
        rows.append(ScoreRow(d, "a", f"T{i}", 7.5, 0.6, 0.55 if y else 0.45, 0.5, 8, y))
    point, lo, hi = day_bootstrap_dll_ci(rows, n_boot=400, seed=2)
    assert point < 0  # model beat the 0.5 market on every row
    assert lo <= point <= hi and hi < 0
