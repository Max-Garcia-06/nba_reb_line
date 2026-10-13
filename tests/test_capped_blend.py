from __future__ import annotations

import json
import math
import random

import pytest

import market_blend
from market_blend import BlendParams, _logit, blend_probability, fit_capped_blend, load_blend_weight, pinned_blend


def _pb(params: BlendParams, p: float, m: float) -> float:
    return blend_probability(p, m, params.offset_weight(p, m))


def test_capped_blend_is_monotone_in_model_probability():
    params = BlendParams(w=0.6, cap=0.3)
    ps = [i / 100 for i in range(2, 99)]
    out = [_pb(params, p, 0.50) for p in ps]
    assert all(b > a for a, b in zip(out, out[1:]))


def test_capped_blend_never_moves_more_than_cap_from_market():
    params = BlendParams(w=1.0, cap=0.25)
    for p in (0.02, 0.2, 0.8, 0.98):
        assert abs(_logit(_pb(params, p, 0.5)) - _logit(0.5)) <= 0.25 + 1e-9


def test_infinite_cap_is_the_plain_logit_blend():
    params = BlendParams(w=0.4)
    assert _pb(params, 0.7, 0.5) == pytest.approx(blend_probability(0.7, 0.5, 0.4))


def _synthetic(model_informative: bool, n=3000, seed=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        truth = rng.uniform(0.2, 0.8)
        noisy = min(0.97, max(0.03, truth + rng.gauss(0, 0.12)))
        p, m = (truth, noisy) if model_informative else (noisy, truth)
        rows.append({"p": p, "m": m, "y": 1.0 if rng.random() < truth else 0.0})
    return rows


def test_fit_trusts_an_informative_model_and_not_a_noisy_one(monkeypatch):
    monkeypatch.setattr(market_blend, "MIN_BLEND_ROWS", 100)
    good, _ = fit_capped_blend(_synthetic(True))
    bad, _ = fit_capped_blend(_synthetic(False))
    assert good.w > 0.6 and bad.w <= 0.2


def test_fit_returns_none_below_min_rows(monkeypatch):
    monkeypatch.setattr(market_blend, "MIN_BLEND_ROWS", 200)
    assert fit_capped_blend(_synthetic(True, n=50)) is None


def test_load_blend_weight_uses_capped_meta_and_ignores_segments(tmp_path, monkeypatch):
    meta = tmp_path / "blend_meta.json"
    meta.write_text(json.dumps({"w": 0.6, "cap": 0.3, "end": "2026-06-13"}))
    seg = tmp_path / "seg.json"
    seg.write_text(json.dumps({"segments": {">=0.15": {"w": 0.99, "n_rows": 999}}}))
    monkeypatch.setattr(market_blend, "BLEND_META_PATH", meta)
    monkeypatch.setattr(market_blend, "SEGMENT_BLEND_META_PATH", seg)
    monkeypatch.setattr(market_blend, "USE_MARKET_BLEND", True)
    monkeypatch.setattr(market_blend, "BLEND_WEIGHT_OVERRIDE", None)
    market_blend.reset_blend_cache()
    d = abs(_logit(0.80) - _logit(0.50))
    assert load_blend_weight(0.80, 0.50) == pytest.approx(0.6 / (1 + 0.6 * d / 0.3))
    market_blend.reset_blend_cache()


def test_pinned_blend_scoped(monkeypatch):
    monkeypatch.setattr(market_blend, "USE_MARKET_BLEND", True)
    monkeypatch.setattr(market_blend, "BLEND_WEIGHT_OVERRIDE", None)
    with pinned_blend(BlendParams(w=0.5, cap=0.2)):
        assert load_blend_weight(0.5, 0.5) == pytest.approx(0.5)
        assert load_blend_weight(0.9, 0.5) < 0.5
    assert market_blend._CACHED_WEIGHT is None


def test_fit_keeps_the_configured_cap(monkeypatch):
    monkeypatch.setattr(market_blend, "MIN_BLEND_ROWS", 100)
    monkeypatch.setattr(market_blend, "BLEND_CAP", 0.3)
    params, _ = fit_capped_blend(_synthetic(True))
    assert params.cap == 0.3
    params2, _ = fit_capped_blend(_synthetic(True), cap=0.2)
    assert params2.cap == 0.2
