from pathlib import Path

from calibrate_preflight import check_calibrate_preflight, write_calibrator_meta, load_calibrator_meta


def test_write_and_load_meta(tmp_path, monkeypatch):
    p = tmp_path / "calibrator_meta.json"
    monkeypatch.setattr("calibrate_preflight.CALIBRATOR_META_PATH", p)
    write_calibrator_meta(n_rows=50, n_segments=3, start="2026-05-01", end="2026-05-20")
    meta = load_calibrator_meta()
    assert meta is not None
    assert meta["n_rows"] == 50


def test_preflight_missing_segmented(tmp_path, monkeypatch):
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr("calibrate_preflight.OOF_CALIB_PATH", tmp_path / "oof.pkl")
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.REQUIRE_FILL_CALIB_FOR_LIVE", False)
    r = check_calibrate_preflight(live=True)
    assert r.ok
    assert any("missing" in w.lower() for w in r.warnings)


def test_preflight_flags_fresh_fit_over_stale_data(tmp_path, monkeypatch):
    """A refit over old journals resets trained_at; the data window must still gate."""
    meta_path = tmp_path / "calibrator_meta.json"
    seg_path = tmp_path / "seg.pkl"
    seg_path.write_bytes(b"")
    monkeypatch.setattr("calibrate_preflight.CALIBRATOR_META_PATH", meta_path)
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", seg_path)
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight.REQUIRE_FILL_CALIB_FOR_LIVE", False)
    monkeypatch.setattr("calibrate_preflight.CALIBRATE_MAX_AGE_DAYS", 14)

    # trained_at is now (fresh), but the newest slate it saw is years back.
    write_calibrator_meta(n_rows=50, n_segments=3, start="2020-05-01", end="2020-05-20", path=meta_path)
    r = check_calibrate_preflight(live=True)

    assert any("newest training slate" in w for w in r.warnings), r.warnings


def test_preflight_accepts_recent_data(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    meta_path = tmp_path / "calibrator_meta.json"
    seg_path = tmp_path / "seg.pkl"
    seg_path.write_bytes(b"")
    monkeypatch.setattr("calibrate_preflight.CALIBRATOR_META_PATH", meta_path)
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", seg_path)
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight.REQUIRE_FILL_CALIB_FOR_LIVE", False)
    monkeypatch.setattr("calibrate_preflight.CALIBRATE_MAX_AGE_DAYS", 14)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    write_calibrator_meta(n_rows=800, n_segments=3, start="2026-05-25", end=today, path=meta_path)
    r = check_calibrate_preflight(live=True)

    assert not any("stale" in w.lower() for w in r.warnings), r.warnings


def test_preflight_flags_bundle_without_meta(tmp_path, monkeypatch):
    """A bundle with no meta cannot be dated; that must not read as healthy."""
    seg_path = tmp_path / "seg.pkl"
    seg_path.write_bytes(b"")
    monkeypatch.setattr("calibrate_preflight.CALIBRATOR_META_PATH", tmp_path / "absent.json")
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", seg_path)
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight.REQUIRE_FILL_CALIB_FOR_LIVE", False)

    r = check_calibrate_preflight(live=True)

    assert any("provenance unknown" in w for w in r.warnings), r.warnings


def test_preflight_blocks_when_required(tmp_path, monkeypatch):
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.REQUIRE_FILL_CALIB_FOR_LIVE", True)
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    r = check_calibrate_preflight(live=True)
    assert not r.ok
