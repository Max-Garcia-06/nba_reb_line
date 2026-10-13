from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta, timezone

import pytest
import typer

from calibrate_preflight import check_calibrate_preflight, write_calibrator_meta

MODEL = "2026-10-11T00:00:00"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Healthy baseline: segmented calibrator fresh and matching, everything else off."""
    seg = tmp_path / "seg.pkl"
    seg.write_bytes(b"")
    meta = tmp_path / "calibrator_meta.json"
    monkeypatch.setattr("calibrate_preflight.CALIBRATOR_META_PATH", meta)
    monkeypatch.setattr("calibrate_preflight.PLATT_META_PATH", tmp_path / "platt_meta.json")
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", seg)
    monkeypatch.setattr("calibrate_preflight.OOF_CALIB_PATH", tmp_path / "oof.pkl")
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight.USE_PLATT_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight.CALIBRATE_MAX_AGE_DAYS", 14)
    monkeypatch.setattr("calibrate_preflight.model_data_lag_days", lambda: 0)
    monkeypatch.setattr("calibrate_preflight.SEGMENT_BLEND_META_PATH", tmp_path / "blend_seg.json")
    monkeypatch.setattr("calibrate_preflight.BLEND_META_PATH", tmp_path / "blend.json")
    monkeypatch.setattr("model.get_model_trained_on", lambda: MODEL)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    write_calibrator_meta(n_rows=100, n_segments=2, start=today, end=today, model_trained_on=MODEL)
    (tmp_path / "blend.json").write_text(json.dumps({"w": 0.4, "end": "2026-06-13"}))
    return tmp_path


def test_baseline_is_ok(env):
    r = check_calibrate_preflight(live=True)
    assert r.ok, r.errors


def test_platt_enabled_without_meta_blocks_live(env, monkeypatch):
    monkeypatch.setattr("calibrate_preflight.USE_PLATT_CALIBRATION", True)
    r = check_calibrate_preflight(live=True)
    assert not r.ok and any("Platt" in e for e in r.errors)


def test_platt_fit_on_other_model_blocks_live(env, monkeypatch):
    monkeypatch.setattr("calibrate_preflight.USE_PLATT_CALIBRATION", True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    write_calibrator_meta(n_rows=50, n_segments=0, start=today, end=today, model_trained_on="2026-01-01T00:00:00",
                          path=env / "platt_meta.json")
    r = check_calibrate_preflight(live=True)
    assert not r.ok and any("Platt" in e and "trained_on" in e for e in r.errors)


def test_oof_fit_on_other_model_blocks_live(env, monkeypatch):
    from calibration import ProbabilityCalibrator

    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", True)
    (env / "oof.pkl").write_bytes(b"x")
    monkeypatch.setattr("calibrate_preflight.load_oof",
                        lambda: type("C", (), {"model_trained_on": "2026-01-01T00:00:00"})())
    r = check_calibrate_preflight(live=True)
    assert not r.ok and any("OOF" in e for e in r.errors)
    assert check_calibrate_preflight(live=False).ok


def test_unknown_model_lag_blocks_live_only(env, monkeypatch):
    monkeypatch.setattr("calibrate_preflight.model_data_lag_days", lambda: None)
    monkeypatch.setattr("calibrate_preflight.MODEL_MAX_DATA_LAG_DAYS", 14)
    assert not check_calibrate_preflight(live=True).ok
    assert check_calibrate_preflight(live=False).ok


def test_stale_blend_weights_block_live(env, monkeypatch):
    (env / "blend.json").write_text(json.dumps({"w": 0.4, "end": "2026-11-01"}))
    monkeypatch.setattr("calibrate_preflight._latest_game_date", lambda: "2026-12-15")
    monkeypatch.setattr("calibrate_preflight.BLEND_MAX_AGE_DAYS", 21)
    r = check_calibrate_preflight(live=True)
    assert not r.ok and any("blend" in e.lower() for e in r.errors)


def test_offseason_blend_weights_are_current_on_opening_night(env, monkeypatch):
    (env / "blend.json").write_text(json.dumps({"w": 0.4, "end": "2026-06-13"}))
    monkeypatch.setattr("calibrate_preflight._latest_game_date", lambda: "2026-06-13")
    assert check_calibrate_preflight(live=True).ok


def test_train_command_fails_when_oof_refit_fails(monkeypatch):
    monkeypatch.setattr("model.train", lambda **kw: (None, {
        "family": "c7", "train_rows": 1, "data_start": "a", "data_end": "b", "trained_on": MODEL,
        "oof_calibrator_fit": False}))
    monkeypatch.setattr("config.USE_OOF_CALIBRATION", True)
    from pipeline import commands

    with pytest.raises(typer.Exit) as e:
        commands.train()
    assert e.value.exit_code == 1


def test_train_ignores_oof_refit_when_oof_calibration_is_off(monkeypatch):
    monkeypatch.setattr("model.train", lambda **kw: (None, {
        "family": "c7", "train_rows": 1, "data_start": "a", "data_end": "b", "trained_on": MODEL,
        "oof_calibrator_fit": False}))
    monkeypatch.setattr("config.USE_OOF_CALIBRATION", False)
    from pipeline import commands

    commands.train()  # no exit


def test_pit_scoring_matches_live_when_oof_calibration_is_off(monkeypatch):
    import model_vs_market as mvm
    from probability_engine import ProbabilityResult

    monkeypatch.setattr(mvm, "USE_OOF_CALIBRATION", False)
    monkeypatch.setattr(mvm, "_pit_calibrator", lambda d: pytest.fail("PIT calibrator used while live has none"))
    pr = ProbabilityResult(player_id=1, player_name="A", game_date="2026-04-01", kalshi_line=7.5,
                           predicted_lambda=8.0, p_over=0.61, p_under=0.39, distribution="pmf", variance=4.0)
    mvm.calibrate_results([pr], "2026-04-01", pit_train=True)
    assert pr.p_over_calibrated == pytest.approx(0.61)


def test_unused_bucket_file_does_not_gate_when_capped_blend_is_active(env, monkeypatch):
    (env / "blend.json").write_text(json.dumps({"w": 1.0, "cap": 0.3, "end": "2026-06-13"}))
    (env / "blend_seg.json").write_text(json.dumps({"end": "2026-03-31", "segments": {}}))
    assert check_calibrate_preflight(live=True).ok


def test_bucket_file_still_gates_in_legacy_mode(env, monkeypatch):
    (env / "blend.json").write_text(json.dumps({"w": 0.4, "end": "2026-06-13"}))
    (env / "blend_seg.json").write_text(json.dumps({"end": "2026-03-31", "segments": {}}))
    assert not check_calibrate_preflight(live=True).ok


def test_pit_cache_key_changes_with_feature_version(monkeypatch, tmp_path):
    import model_vs_market as mvm

    monkeypatch.setattr(mvm, "PIT_CACHE", tmp_path)
    monkeypatch.setattr("feature_store.FEATURE_VERSION", "vA")
    a = mvm._pit_cache_path("2026-04-06")
    monkeypatch.setattr("feature_store.FEATURE_VERSION", "vB")
    assert mvm._pit_cache_path("2026-04-06") != a
