"""
model.py
--------
XGBoost rebound regressor with walk-forward (time-series) cross-validation.

Training target : REB (actual total rebounds)
Objective       : Minimize MAE
Validation      : Walk-forward — train on months 1–N, test on month N+1

Usage
-----
  python model.py train     # train and save model
  python model.py evaluate  # print walk-forward CV results
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

from config import MODEL_DIR, ROLLING_WINDOW
from feature_store import build_feature_table, MODEL_FEATURES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_PATH = MODEL_DIR / "xgb_rebound_model.pkl"
META_PATH = MODEL_DIR / "model_meta.pkl"


# ---------------------------------------------------------------------------
# XGBoost hyper-parameters
# ---------------------------------------------------------------------------

XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.7,
    colsample_bylevel=0.7,
    reg_alpha=0.05,
    reg_lambda=1.5,
    objective="reg:absoluteerror",   # directly minimises MAE
    tree_method="hist",
    random_state=42,
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and clean feature table. Returns (X, y) with no NaN rows.
    Rows are sorted by GAME_DATE to preserve temporal order.
    """
    if df is None:
        df = build_feature_table()

    df = df.dropna(subset=["REB"] + MODEL_FEATURES).copy()
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    X = df[MODEL_FEATURES]
    y = df["REB"]
    return X, y


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """
    TimeSeriesSplit cross-validation.
    Returns dict with per-fold MAE and overall mean/std.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        maes.append(mae)
        log.info(f"  Fold {fold + 1}/{n_splits} — MAE: {mae:.3f}")

    result = {
        "fold_maes": maes,
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
    }
    log.info(f"CV MAE: {result['mean_mae']:.3f} ± {result['std_mae']:.3f}")
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: Optional[pd.DataFrame] = None, save: bool = True) -> xgb.XGBRegressor:
    """Train on all available data and optionally persist."""
    X, y = prepare_data(df)
    log.info(f"Training on {len(X):,} player-game rows, {len(MODEL_FEATURES)} features.")

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    # Store residual stats for Negative Binomial dispersion estimation
    preds = model.predict(X)
    residuals = y.values - preds
    meta = {
        "features": MODEL_FEATURES,
        "train_mean_y": float(y.mean()),
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "residual_var": float(residuals.var()),
        "train_rows": len(X),
    }

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)
        log.info(f"Model saved to {MODEL_PATH}")

    return model, meta


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model() -> tuple[xgb.XGBRegressor, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No trained model at {MODEL_PATH}. Run: python model.py train")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return model, meta


def predict_lambda(
    features: dict | pd.DataFrame,
    model: Optional[xgb.XGBRegressor] = None,
) -> float | np.ndarray:
    """
    Predict the expected rebound mean (λ) for one or more player-game rows.

    Parameters
    ----------
    features : dict (single row) or DataFrame (batch)
    model    : pre-loaded model; loads from disk if None

    Returns
    -------
    float (single) or np.ndarray (batch)
    """
    if model is None:
        model, _ = load_model()

    if isinstance(features, dict):
        X = pd.DataFrame([features])[MODEL_FEATURES]
        return float(max(0.0, model.predict(X)[0]))

    X = features[MODEL_FEATURES]
    return np.maximum(0.0, model.predict(X))


def get_feature_importance(model: Optional[xgb.XGBRegressor] = None) -> pd.DataFrame:
    if model is None:
        model, _ = load_model()
    scores = model.get_booster().get_score(importance_type="gain")
    return (
        pd.DataFrame(scores.items(), columns=["feature", "importance"])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"

    if cmd == "train":
        model, meta = train()
        print(f"\nTrain residual std (σ): {meta['residual_std']:.3f}")
        print(f"Feature importance:")
        print(get_feature_importance(model).to_string(index=False))

    elif cmd == "evaluate":
        X, y = prepare_data()
        results = walk_forward_cv(X, y, n_splits=5)
        print(f"\nWalk-forward CV  —  MAE: {results['mean_mae']:.3f} ± {results['std_mae']:.3f}")
        league_avg = y.mean()
        print(f"League avg rebounds: {league_avg:.2f}")
        print(f"MAE as % of league avg: {results['mean_mae'] / league_avg * 100:.1f}%")

    else:
        print(f"Unknown command: {cmd}. Use 'train' or 'evaluate'.")
