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
    BLEND_MAX_AGE_DAYS,
    BLEND_META_PATH,
    BLEND_WEIGHT_OVERRIDE,
    CALIBRATE_MAX_AGE_DAYS,
    MODEL_DIR,
    MODEL_MAX_DATA_LAG_DAYS,
    REQUIRE_FILL_CALIB_FOR_LIVE,
    SEGMENT_BLEND_META_PATH,
    USE_OOF_CALIBRATION,
    USE_PLATT_CALIBRATION,
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


def _model_data_end() -> str | None:
    import pickle

    from model import META_PATH

    try:
        with open(META_PATH, "rb") as f:
            return pickle.load(f).get("data_end")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None


def _latest_game_date() -> str | None:
    from sqlalchemy import text

    from data_engine import _get_engine

    try:
        with _get_engine().connect() as conn:
            d = conn.execute(text("SELECT MAX(GAME_DATE) FROM player_gamelogs")).scalar()
    except Exception:
        return None
    return str(d)[:10] if d else None


def model_data_lag_days() -> int | None:
    """Days between the newest game in the DB and the newest game the model trained on."""
    end, latest = _model_data_end(), _latest_game_date()
    if not end or not latest:
        return None
    try:
        return (datetime.strptime(latest, "%Y-%m-%d") - datetime.strptime(end[:10], "%Y-%m-%d")).days
    except ValueError:
        return None


def _meta_stale_reasons(meta: dict) -> list[str]:
    """Why a calibrator's meta is too old: fit time and, separately, the age of its newest data."""
    reasons: list[str] = []
    if CALIBRATE_MAX_AGE_DAYS <= 0:
        return reasons
    age = _age_days(str(meta.get("trained_at", "") or ""))
    if age is not None and age > float(CALIBRATE_MAX_AGE_DAYS):
        reasons.append(f"fit ran {age:.0f}d ago ({meta.get('trained_at')})")
    # Gating on trained_at alone lets a refit over old journals reset the clock
    # without making the calibrator any fresher — that is exactly how a bundle
    # fit on 50 rows from May stayed "current" into August. Gate on the data.
    data_age = _data_age_days(str(meta.get("end", "") or ""))
    if data_age is not None and data_age > float(CALIBRATE_MAX_AGE_DAYS):
        reasons.append(f"newest training slate {meta.get('end')} is {data_age:.0f}d old")
    return reasons


def check_calibrate_preflight(*, live: bool = False) -> CalibratePreflightResult:
    warnings: list[str] = []
    errors: list[str] = []

    def _flag(msg: str, *, blocks_live: bool = True) -> None:
        (errors if (live and blocks_live) else warnings).append(msg)

    # mlb_tb_line ran 19 days on a model that predated an ETL fix, and the only
    # signal was P&L. Retraining is cheap here, so a lagging model means the
    # Sunday retrain stopped running: fail closed for live, like the calibrators.
    if MODEL_MAX_DATA_LAG_DAYS > 0:
        lag = model_data_lag_days()
        if lag is None:
            _flag("Model data lag unknown (model meta or player_gamelogs unreadable); cannot verify the model is current")
        elif lag > MODEL_MAX_DATA_LAG_DAYS:
            _flag(
                f"Model is {lag}d behind the data (trained through {_model_data_end()}, "
                f"limit {MODEL_MAX_DATA_LAG_DAYS}d); run train"
            )

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
            _flag(
                f"OOF calibrator was fit against model trained_on={oof_model!r}, current model is "
                f"trained_on={current_model!r} — stale; re-run train (it refits OOF)."
            )

    # Platt takes priority over every other calibrator in edge_detector._calibrate,
    # so when it is on it needs the same provenance checks as the segmented bundle.
    if USE_PLATT_CALIBRATION:
        pmeta = load_calibrator_meta(PLATT_META_PATH)
        if pmeta is None:
            _flag(f"Platt calibrator enabled but {PLATT_META_PATH.name} is missing: provenance unknown; run calibrate --family platt")
        else:
            reasons = _meta_stale_reasons(pmeta)
            if reasons:
                _flag(f"Platt calibrator stale (limit {CALIBRATE_MAX_AGE_DAYS}d): {'; '.join(reasons)}")
            pm = pmeta.get("model_trained_on")
            if current_model is not None and pm != current_model:
                _flag(f"Platt calibrator was fit against model trained_on={pm!r}, current is trained_on={current_model!r}")

    # Like the model-lag gate, measure blend weights against the newest game in the
    # DB, not the wall clock: on opening night last season's fit is still current.
    latest_game = _latest_game_date()
    if BLEND_MAX_AGE_DAYS > 0 and BLEND_WEIGHT_OVERRIDE is None and latest_game:
        global_meta = load_calibrator_meta(BLEND_META_PATH) or {}
        # A capped fit (``cap`` key) is the whole blend; the legacy bucket file is unused then.
        blend_files = (BLEND_META_PATH,) if global_meta.get("cap") is not None else (BLEND_META_PATH, SEGMENT_BLEND_META_PATH)
        for bpath in blend_files:
            bmeta = load_calibrator_meta(bpath)
            if not bmeta or not bmeta.get("end"):
                continue
            try:
                b_lag = (datetime.strptime(latest_game, "%Y-%m-%d")
                         - datetime.strptime(str(bmeta["end"])[:10], "%Y-%m-%d")).days
            except ValueError:
                continue
            if b_lag > BLEND_MAX_AGE_DAYS:
                _flag(
                    f"Market-blend weights in {bpath.name} were fit on data ending {bmeta.get('end')}, "
                    f"{b_lag}d behind the newest game ({latest_game}, limit {BLEND_MAX_AGE_DAYS}d); run refit-blend"
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
    if meta:
        stale_reasons.extend(_meta_stale_reasons(meta))
        seg_stale = seg_stale or bool(stale_reasons)
    if meta and current_model is not None:
        seg_model = meta.get("model_trained_on")
        if seg_model is not None and seg_model != current_model:
            seg_model_mismatch = True

    if seg_missing:
        msg = (
            f"Fill-based segmented calibrator missing ({SEGMENTED_CALIB_PATH.name}); "
            "run reconcile then calibrate on your fills"
        )
        _flag(msg, blocks_live=REQUIRE_FILL_CALIB_FOR_LIVE)
    elif seg_stale:
        msg = (
            f"Fill calibrator stale (limit {CALIBRATE_MAX_AGE_DAYS}d): "
            f"{'; '.join(stale_reasons)}; run reconcile + calibrate"
        )
        _flag(msg, blocks_live=REQUIRE_FILL_CALIB_FOR_LIVE)
    elif seg_model_mismatch:
        msg = (
            f"Fill-based segmented calibrator was fit against model trained_on={meta.get('model_trained_on')!r}, "
            f"current model is trained_on={current_model!r} — a retrain invalidated it; "
            "re-run calibrate once enough new fills exist under the new model."
        )
        _flag(msg, blocks_live=REQUIRE_FILL_CALIB_FOR_LIVE)

    ok = len(errors) == 0
    return CalibratePreflightResult(ok=ok, warnings=warnings, errors=errors)
