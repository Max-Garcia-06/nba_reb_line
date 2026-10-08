"""A calibrator of unknown or stale provenance must stop live trading, not just warn.

The warning path already existed and fired 49 times in `logs/scan.log` while the
scan kept placing orders against a bundle with no `calibrator_meta.json` behind
it. A check nobody can act on in time is not a check; live runs have to fail
closed by default.
"""

from calibrate_preflight import check_calibrate_preflight, write_calibrator_meta


def _clean_calibrator_env(tmp_path, monkeypatch):
    seg_path = tmp_path / "seg.pkl"
    seg_path.write_bytes(b"")
    monkeypatch.setattr("calibrate_preflight.CALIBRATOR_META_PATH", tmp_path / "calibrator_meta.json")
    monkeypatch.setattr("calibrate_preflight.SEGMENTED_CALIB_PATH", seg_path)
    monkeypatch.setattr("calibrate_preflight.OOF_CALIB_PATH", tmp_path / "oof.pkl")
    monkeypatch.setattr("calibrate_preflight.USE_SEGMENTED_CALIBRATION", True)
    monkeypatch.setattr("calibrate_preflight.USE_OOF_CALIBRATION", False)
    monkeypatch.setattr("calibrate_preflight.CALIBRATE_MAX_AGE_DAYS", 14)


def test_unknown_provenance_blocks_live_by_default(tmp_path, monkeypatch):
    """No meta file => nothing dates the bundle. Default config must fail closed."""
    _clean_calibrator_env(tmp_path, monkeypatch)

    result = check_calibrate_preflight(live=True)

    assert not result.ok
    assert any("provenance unknown" in e for e in result.errors), result.errors


def test_unknown_provenance_still_only_warns_when_not_live(tmp_path, monkeypatch):
    """Backtests and reports must stay runnable against an undated calibrator."""
    _clean_calibrator_env(tmp_path, monkeypatch)

    result = check_calibrate_preflight(live=False)

    assert result.ok
    assert any("provenance unknown" in w for w in result.warnings), result.warnings


def test_fresh_documented_calibrator_passes_live(tmp_path, monkeypatch):
    """The gate must not block a bundle that is actually current."""
    from datetime import datetime, timezone

    _clean_calibrator_env(tmp_path, monkeypatch)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    write_calibrator_meta(
        n_rows=200,
        n_segments=4,
        start=today,
        end=today,
        path=tmp_path / "calibrator_meta.json",
    )

    result = check_calibrate_preflight(live=True)

    assert result.ok, (result.errors, result.warnings)
