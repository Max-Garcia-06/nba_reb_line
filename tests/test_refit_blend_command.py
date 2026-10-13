from __future__ import annotations

import json
import random

import pytest
import typer

import market_blend
from model_vs_market import ScoreRow


def _rows_for(day: str, n: int, seed: int):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        truth = rng.uniform(0.2, 0.8)
        m = min(0.97, max(0.03, truth + rng.gauss(0, 0.05)))
        y = 1.0 if rng.random() < truth else 0.0
        out.append(ScoreRow(day, "a", f"T{i}", 7.5, truth, truth, m, 8, y))
    return out


@pytest.fixture
def meta(tmp_path, monkeypatch):
    path = tmp_path / "blend_meta.json"
    monkeypatch.setattr(market_blend, "BLEND_META_PATH", path)
    monkeypatch.setattr(market_blend, "MIN_BLEND_ROWS", 200)
    return path


def test_refit_writes_capped_fit_over_expanding_window(meta, monkeypatch):
    days = []

    def fake_eval(d, **kw):
        days.append(d)
        return _rows_for(d, 20, hash(d) % 1000)

    monkeypatch.setattr("model_vs_market.evaluate_day", fake_eval)
    from pipeline.commands import refit_blend

    refit_blend(days=None, start=None, end="2026-03-31", dry_run=False)
    saved = json.loads(meta.read_text())
    assert "cap" in saved and 0 <= saved["w"] <= 1
    assert saved["end"] == "2026-03-31"
    assert min(days) <= "2025-04-01"  # default window is a season, not 30 days


def test_refit_keeps_previous_fit_when_too_few_rows(meta, monkeypatch):
    meta.write_text(json.dumps({"w": 0.33, "cap": 0.3, "end": "2026-03-01"}))
    monkeypatch.setattr("model_vs_market.evaluate_day", lambda d, **kw: _rows_for(d, 1, 1) if d == "2026-03-31" else [])
    from pipeline.commands import refit_blend

    with pytest.raises(typer.Exit):
        refit_blend(days=10, start=None, end="2026-03-31", dry_run=False)
    assert json.loads(meta.read_text())["w"] == 0.33
