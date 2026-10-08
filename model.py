"""
model.py
--------
Production wrapper around the model_zoo family chosen by the bake-off
(config.MODEL_FAMILY). Persists the fitted model plus a meta record whose
`trained_on` stamp identifies it; calibrators record the same stamp and
calibrate_preflight refuses live trading on a mismatch.

Lessons carried over from mlb_tb_line (SYSTEM.md §9):
  - A retrain silently invalidates calibrators fit against the old model's
    probability distribution, so `train` refits the OOF calibrator itself.
  - PIT backtests need `train_as_of(date)`: only rows strictly before `date`.

Usage
-----
  python model.py train
  python model.py evaluate      # same walk-forward as bakeoff, current family only
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import EVAL_LINES, MODEL_DIR, MODEL_FAMILY
from feature_store import MODEL_FEATURES, build_feature_table
from model_zoo import CANDIDATES, Model, TOP, prob_over

log = logging.getLogger(__name__)

MODEL_PATH = MODEL_DIR / "rebound_model.pkl"
META_PATH = MODEL_DIR / "model_meta.pkl"
OOF_MONTHS = 4
OOF_LINE_WINDOW = 6.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepare_data(df: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if df is None:
        df = build_feature_table()
    if "_live" in df.columns:
        df = df[~df["_live"]]
    df = df[df["REB"].notna()].sort_values("GAME_DATE").reset_index(drop=True)
    return df[MODEL_FEATURES], df["REB"].to_numpy(), df


def prepare_data_as_of(as_of_date: str, df: Optional[pd.DataFrame] = None):
    """Training rows strictly before `as_of_date` (point-in-time)."""
    X, y, dff = prepare_data(df)
    keep = pd.to_datetime(dff["GAME_DATE"]) < pd.Timestamp(as_of_date)
    return X[keep.values], y[keep.values], dff[keep.values].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train / load
# ---------------------------------------------------------------------------

def _fit(dff: pd.DataFrame, family: str) -> tuple[Model, dict]:
    m = CANDIDATES[family]().fit(dff)
    meta = {
        "family": family,
        "features": list(MODEL_FEATURES),
        "train_rows": int(len(dff)),
        "data_start": str(pd.to_datetime(dff["GAME_DATE"]).min().date()),
        "data_end": str(pd.to_datetime(dff["GAME_DATE"]).max().date()),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
    }
    # Identity stamp: data end date + fit time, so two fits on the same data differ.
    meta["trained_on"] = f"{meta['data_end']}@{meta['fitted_at'][:19]}"
    return m, meta


def train(save: bool = True, family: str = MODEL_FAMILY, df: Optional[pd.DataFrame] = None,
          fit_oof: bool = True) -> tuple[Model, dict]:
    _, _, dff = prepare_data(df)
    log.info(f"Training {family} on {len(dff):,} rows ({len(MODEL_FEATURES)} features)")
    m, meta = _fit(dff, family)
    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(m, f)
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)
        log.info(f"Saved {MODEL_PATH.name} (trained_on={meta['trained_on']})")
        if fit_oof:
            fit_and_save_oof_calibrator(dff, family=family, model_trained_on=meta["trained_on"])
    return m, meta


def train_as_of(as_of_date: str, family: str = MODEL_FAMILY, df: Optional[pd.DataFrame] = None):
    _, _, dff = prepare_data_as_of(as_of_date, df)
    return _fit(dff, family)


def load_model() -> tuple[Model, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run: python run_pipeline.py train")
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return m, meta


def get_model_trained_on() -> str | None:
    try:
        with open(META_PATH, "rb") as f:
            return pickle.load(f).get("trained_on")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None


def predict_pmf(df: pd.DataFrame, model: Optional[Model] = None) -> np.ndarray:
    if model is None:
        model, _ = load_model()
    return model.pmf(df)


# ---------------------------------------------------------------------------
# OOF calibration rows (walk-forward, last OOF_MONTHS months)
# ---------------------------------------------------------------------------

def collect_oof_calibration_rows(dff: pd.DataFrame, family: str = MODEL_FAMILY,
                                 months: int = OOF_MONTHS) -> list[dict]:
    dff = dff.sort_values("GAME_DATE")
    periods = sorted(pd.to_datetime(dff["GAME_DATE"]).dt.to_period("M").unique())[-months:]
    rows: list[dict] = []
    for per in periods:
        start, end = per.start_time, per.end_time
        tr = dff[dff["GAME_DATE"] < start]
        te = dff[(dff["GAME_DATE"] >= start) & (dff["GAME_DATE"] <= end)]
        if te.empty or len(tr) < 5000:
            continue
        pmf = CANDIDATES[family]().fit(tr).pmf(te)
        ref = te["reb_roll"].fillna(te["reb_ewm"]).fillna(5.0).to_numpy()
        for k in EVAL_LINES + [x + 1.0 for x in EVAL_LINES]:
            m = np.abs(ref - k) <= OOF_LINE_WINDOW
            if not m.any():
                continue
            p = prob_over(pmf[m], k)
            y = (te["REB"].to_numpy()[m] > k).astype(float)
            rows.extend({"p": float(a), "y": float(b), "line": float(k)} for a, b in zip(p, y))
        log.info(f"  OOF {per}: train={len(tr):,} test={len(te):,}")
    return rows


def fit_and_save_oof_calibrator(dff: Optional[pd.DataFrame] = None, family: str = MODEL_FAMILY,
                                model_trained_on: Optional[str] = None) -> bool:
    from calibration import fit_oof_from_rows, save_oof

    if dff is None:
        _, _, dff = prepare_data()
    rows = collect_oof_calibration_rows(dff, family)
    cal = fit_oof_from_rows(rows, model_trained_on=model_trained_on or get_model_trained_on())
    if cal is None:
        log.warning(f"OOF calibrator not fit ({len(rows)} rows)")
        return False
    save_oof(cal)
    log.info(f"Saved OOF calibrator from {len(rows):,} rows")
    return True


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        _, meta = train()
        print(meta)
    else:
        print("usage: python model.py train  (evaluation lives in bakeoff.py)")
