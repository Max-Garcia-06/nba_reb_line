"""
bakeoff.py
----------
Walk-forward, point-in-time comparison of the candidate rebound models in
model_zoo.py.

Protocol
--------
For each test block (monthly in 2025-26 so every Kalshi market month has a
PIT prediction; two-month blocks in earlier seasons), every model is trained
on ALL rows dated before the block and predicts the block. Nothing from the
block or later is visible at fit time — the same discipline as
mlb_tb_line's `--pit-train` backtests, applied to every fold.

Metrics (per model, on the union of test rows)
---------------------------------------------
nll        full-distribution negative log-likelihood of actual REB (capped 20+)
ll_line    mean log-loss of P(REB > k) at half-integer lines k in 1.5..17.5
           within 6 of the player's trailing average (the lines Kalshi lists)
brier_line Brier score on the same line set
ece_line   expected calibration error (10 bins) on the same line set
bias       mean(predicted mean) - mean(actual)

Per-row PMFs are saved to data/bakeoff/<model>.npz so ensembles and the
model-vs-Kalshi scoring (market_eval.py) can reuse them without refitting.

Usage
-----
  python bakeoff.py run [--models c0_naive_nb,c3_xgb_nb] [--since 2023-10-01]
  python bakeoff.py summary [--ensembles]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR
from model_zoo import CANDIDATES, TOP, prob_over

log = logging.getLogger("bakeoff")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUT_DIR = Path(DATA_DIR) / "bakeoff"
FEAT_CACHE = Path(DATA_DIR) / "features.pkl"
LINES = np.arange(1.5, 18.0, 1.0)
LINE_WINDOW = 6.0


def load_features(refresh: bool = False) -> pd.DataFrame:
    if FEAT_CACHE.exists() and not refresh:
        return pd.read_pickle(FEAT_CACHE)
    from feature_store import build_feature_table
    df = build_feature_table()
    df = df[~df["_live"] & df["REB"].notna()].reset_index(drop=True)
    df.to_pickle(FEAT_CACHE)
    return df


def test_blocks(df: pd.DataFrame, since: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Monthly blocks in the latest season, bi-monthly before it; only months with games."""
    months = sorted(df.loc[df["GAME_DATE"] >= since, "GAME_DATE"].dt.to_period("M").unique())
    latest = df["SEASON"].max()
    latest_start = df.loc[df["SEASON"] == latest, "GAME_DATE"].min().to_period("M")
    blocks, i = [], 0
    while i < len(months):
        step = 1 if months[i] >= latest_start else 2
        grp = months[i:i + step]
        # Don't let a bi-monthly block straddle into the latest season.
        grp = [m for m in grp if (m >= latest_start) == (grp[0] >= latest_start)]
        blocks.append((grp[0].start_time, grp[-1].end_time))
        i += len(grp)
    return blocks


def run(models: list[str], since: str) -> None:
    df = load_features()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = test_blocks(df, since)
    log.info(f"{len(df):,} rows, {len(blocks)} test blocks, models={models}")

    for name in models:
        keys, pmfs, folds = [], [], []
        t0 = time.time()
        for b, (start, end) in enumerate(blocks):
            train = df[df["GAME_DATE"] < start]
            test = df[(df["GAME_DATE"] >= start) & (df["GAME_DATE"] <= end)]
            if test.empty:
                continue
            m = CANDIDATES[name]().fit(train)
            pmfs.append(m.pmf(test))
            keys.append(test[["PLAYER_ID", "GAME_ID"]].to_numpy())
            folds.append(np.full(len(test), b))
            log.info(f"  {name} block {b + 1}/{len(blocks)} {start.date()}..{end.date()} "
                     f"train={len(train):,} test={len(test):,} ({time.time() - t0:.0f}s)")
        np.savez_compressed(OUT_DIR / f"{name}.npz", keys=np.vstack(keys).astype(str),
                            pmf=np.vstack(pmfs), fold=np.concatenate(folds))
        log.info(f"saved {name} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _load_pmfs() -> dict[str, pd.DataFrame]:
    out = {}
    for f in sorted(OUT_DIR.glob("*.npz")):
        z = np.load(f)
        idx = pd.MultiIndex.from_arrays([z["keys"][:, 0].astype(int), z["keys"][:, 1]],
                                        names=["PLAYER_ID", "GAME_ID"])
        out[f.stem] = pd.DataFrame(z["pmf"], index=idx)
    return out


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return tot / len(p)


def line_frame(feat: pd.DataFrame) -> pd.DataFrame:
    """(row, line) pairs for lines within LINE_WINDOW of the player's trailing average."""
    rows, lines = [], []
    ref = feat["reb_roll"].fillna(feat["reb_ewm"]).fillna(5.0).to_numpy()
    for k in LINES:
        m = np.abs(k - ref) <= LINE_WINDOW
        rows.append(np.nonzero(m)[0]); lines.append(np.full(m.sum(), k))
    return pd.DataFrame({"row": np.concatenate(rows), "line": np.concatenate(lines)})


def score(pmf: np.ndarray, y: np.ndarray, lf: pd.DataFrame) -> dict:
    yc = np.minimum(y, TOP).astype(int)
    eps = 1e-12
    nll = -np.log(np.maximum(pmf[np.arange(len(yc)), yc], eps)).mean()
    p = np.clip(prob_over(pmf[lf["row"].values], lf["line"].values), 1e-6, 1 - 1e-6)
    o = (y[lf["row"].values] > lf["line"].values).astype(float)
    ll = -(o * np.log(p) + (1 - o) * np.log(1 - p)).mean()
    mean_pred = (pmf * np.arange(TOP + 1)).sum(axis=1)
    return {"n": len(y), "nll": nll, "ll_line": ll, "brier_line": ((p - o) ** 2).mean(),
            "ece_line": _ece(p, o), "bias": float(mean_pred.mean() - y.mean())}


def summary(ensembles: bool = False, by_season: bool = True) -> pd.DataFrame:
    feat = load_features().set_index(["PLAYER_ID", "GAME_ID"])
    pm = _load_pmfs()
    common = None
    for d in pm.values():
        common = d.index if common is None else common.intersection(d.index)
    feat = feat.loc[common]
    if ensembles:
        names = [n for n in pm if not n.startswith("c0") and not n.startswith("c1")]
        for a, b in combinations(names, 2):
            pm[f"ens[{a[:2]}+{b[:2]}]"] = (pm[a].loc[common] + pm[b].loc[common]) / 2
        for a, b, c in combinations(names, 3):
            pm[f"ens[{a[:2]}+{b[:2]}+{c[:2]}]"] = (pm[a].loc[common] + pm[b].loc[common] + pm[c].loc[common]) / 3

    y = feat["REB"].to_numpy()
    lf = line_frame(feat)
    rows = [{"model": n, "season": "ALL", **score(d.loc[common].to_numpy(), y, lf)} for n, d in pm.items()]
    if by_season:
        for s in sorted(feat["SEASON"].unique()):
            m = (feat["SEASON"] == s).to_numpy()
            lfs = line_frame(feat[m])
            for n, d in pm.items():
                rows.append({"model": n, "season": s, **score(d.loc[common].to_numpy()[m], y[m], lfs)})
    out = pd.DataFrame(rows).sort_values(["season", "ll_line"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "features"])
    ap.add_argument("--models", default=",".join(CANDIDATES))
    ap.add_argument("--since", default="2023-10-01")
    ap.add_argument("--ensembles", action="store_true")
    a = ap.parse_args()
    if a.cmd == "features":
        load_features(refresh=True)
    elif a.cmd == "run":
        run(a.models.split(","), a.since)
    else:
        pd.set_option("display.width", 200)
        print(summary(a.ensembles).round(4).to_string(index=False))
