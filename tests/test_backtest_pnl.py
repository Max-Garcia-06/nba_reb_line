"""Side-correct settlement: a NO signal's limit_price is a NO-contract price."""

import pytest

from backtest import pnl_per_contract, settle


def test_no_side_pnl_uses_no_price():
    # Bought NO at 0.30; NO wins -> +0.70, loses -> -0.30 (mlb_tb_line had these inverted).
    assert pnl_per_contract(True, 0.30) == pytest.approx(0.70)
    assert pnl_per_contract(False, 0.30) == pytest.approx(-0.30)


def test_fee_reduces_both_outcomes():
    assert pnl_per_contract(True, 0.40, fee=0.02) == pytest.approx(0.58)
    assert pnl_per_contract(False, 0.40, fee=0.02) == pytest.approx(-0.42)


def test_settle_half_lines():
    assert settle("yes", 7.5, 8) and not settle("yes", 7.5, 7)
    assert settle("no", 7.5, 7) and not settle("no", 7.5, 8)
