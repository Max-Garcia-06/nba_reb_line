"""
Preflight checks for probability calibrators before live scan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CALIBRATE_MAX_AGE_DAYS,
    MODEL_DIR,
    REQUIRE_FILL_CALIB_FOR_LIVE,
    USE_OOF_CALIBRATION,
    USE_SEGMENTED_CALIBRATION,
)
from calibration import OOF_CALIB_PATH, SEGMENTED_CALIB_PATH, load_oof

log = logging.getLogger(__name__)

CALIBRATOR_META_PATH = Path(MODEL_DIR) / "calibrator_meta.json"
PLATT_META_PATH = Path(MODEL_DIR) / "calibrator_meta_platt.json"


@dataclass
class CalibratePreflightResult:
    ok: bool
    warnings: list[str]
    errors: list[str]


def write_calibrator_meta(
    *,
    n_rows: int,
    n_segments: int,
    start: str,
    end: str,
    model_trained_on: str | None = None,
    path: Path | None = None,
) -> None:
    # Resolved at call time: a module-level default binds at import and silently
    # ignores monkeypatching, which let the test suite overwrite the real meta.
    path = path if path is not None else CALIBRATOR_META_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(n_rows),
        "n_segments": int(n_segments),
        "start": str(start),
        "end": str(end),
        "model_trained_on": model_trained_on,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibrator_meta(path: Path | None = None) -> dict | None:
    path = path if path is not None else CALIBRATOR_META_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _data_age_days(end: str) -> float | None:
    """Days since the newest slate the calibrator was fit on."""
    try:
        dt = datetime.strptime(str(end).strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def _age_days(trained_at: str) -> float | None:
    try:
        s = trained_at.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0
    except Exception:
        return None


def check_calibrate_preflight(*, live: bool = False) -> CalibratePreflightResult:
    warnings: list[str] = []
    errors: list[str] = []

    from model import get_model_trained_on

    current_model = get_model_trained_on()

    if USE_OOF_CALIBRATION and not OOF_CALIB_PATH.exists():
        warnings.append(
            f"OOF calibrator missing ({OOF_CALIB_PATH.name}); run train with --fit-oof or disable USE_OOF_CALIBRATION"
        )
    elif USE_OOF_CALIBRATION and current_model is not None:
        oof_cal = load_oof()
        oof_model = getattr(oof_cal, "model_trained_on", None) if oof_cal is not None else None
        if oof_model is not None and oof_model != current_model:
            warnings.append(
                f"OOF calibrator was fit against model trained_on={oof_model!r}, current model is "
                f"trained_on={current_model!r} — stale; re-run train (now refits OOF automatically)."
            )

    seg_missing = USE_SEGMENTED_CALIBRATION and not SEGMENTED_CALIB_PATH.exists()
    seg_stale = False
    stale_reasons: list[str] = []
    seg_model_mismatch = False
    meta = load_calibrator_meta()
    if USE_SEGMENTED_CALIBRATION and not seg_missing and meta is None:
        # Without meta there is nothing to date the bundle against, and every check
        # below is skipped — which reads as "healthy" when it means "unknown".
        stale_reasons.append(f"no {CALIBRATOR_META_PATH.name}: provenance unknown")
        seg_stale = True
    if meta and CALIBRATE_MAX_AGE_DAYS > 0:
        age = _age_days(str(meta.get("trained_at", "") or ""))
        if age is not None and age > float(CALIBRATE_MAX_AGE_DAYS):
            seg_stale = True
            stale_reasons.append(f"fit ran {age:.0f}d ago ({meta.get('trained_at')})")
        # Gating on trained_at alone lets a refit over old journals reset the clock
        # without making the calibrator any fresher — that is exactly how a bundle
        # fit on 50 rows from May stayed "current" into August. Gate on the data.
        data_age = _data_age_days(str(meta.get("end", "") or ""))
        if data_age is not None and data_age > float(CALIBRATE_MAX_AGE_DAYS):
            seg_stale = True
            stale_reasons.append(f"newest training slate {meta.get('end')} is {data_age:.0f}d old")
    if meta and current_model is not None:
        seg_model = meta.get("model_trained_on")
        if seg_model is not None and seg_model != current_model:
            seg_model_mismatch = True

    if seg_missing:
        msg = (
            f"Fill-based segmented calibrator missing ({SEGMENTED_CALIB_PATH.name}); "
            "run reconcile then calibrate on your fills"
        )
        if live and REQUIRE_FILL_CALIB_FOR_LIVE:
            errors.append(msg)
        else:
            warnings.append(msg)
    elif seg_stale:
        msg = (
            f"Fill calibrator stale (limit {CALIBRATE_MAX_AGE_DAYS}d): "
            f"{'; '.join(stale_reasons)}; run reconcile + calibrate"
        )
        if live and REQUIRE_FILL_CALIB_FOR_LIVE:
            errors.append(msg)
        else:
            warnings.append(msg)
    elif seg_model_mismatch:
        msg = (
            f"Fill-based segmented calibrator was fit against model trained_on={meta.get('model_trained_on')!r}, "
            f"current model is trained_on={current_model!r} — a retrain invalidated it; "
            "re-run calibrate once enough new fills exist under the new model."
        )
        if live and REQUIRE_FILL_CALIB_FOR_LIVE:
            errors.append(msg)
        else:
            warnings.append(msg)

    ok = len(errors) == 0
    return CalibratePreflightResult(ok=ok, warnings=warnings, errors=errors)
