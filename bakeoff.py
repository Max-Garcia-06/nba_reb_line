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
  FEATURE_GROUPS=travel python bakeoff.py run --models c7_ens_c4_c5 --tag travel
  python bakeoff.py ablation --model c7_ens_c4_c5 [--since-score 2025-12-01]
"""

from __future__ import annotations

import argparse
import logging
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
LINES = np.arange(1.5, 18.0, 1.0)
LINE_WINDOW = 6.0


def _feat_cache() -> Path:
    from feature_store import FEATURE_VERSION
    return Path(DATA_DIR) / f"features_{FEATURE_VERSION}.pkl"


def load_features(refresh: bool = False) -> pd.DataFrame:
    path = _feat_cache()
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    from feature_store import build_feature_table
    df = build_feature_table()
    df = df[~df["_live"] & df["REB"].notna()].reset_index(drop=True)
    df.to_pickle(path)
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


def run(models: list[str], since: str, tag: str = "") -> None:
    from feature_store import FEATURE_VERSION, MODEL_FEATURES

    df = load_features()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = test_blocks(df, since)
    log.info(f"{len(df):,} rows, {len(blocks)} test blocks, models={models}, features={FEATURE_VERSION}")

    for name in models:
        keys, pmfs, folds = [], [], []
        t0 = time.time()
        for b, (start, end) in enumerate(blocks):
            train = df[df["GAME_DATE"] < start]
            test = df[(df["GAME_DATE"] >= start) & (df["GAME_DATE"] <= end)]
            if test.empty:
                continue
            m = CANDIDATES[name]().fit(train, features=MODEL_FEATURES)
            pmfs.append(m.pmf(test))
            keys.append(test[["PLAYER_ID", "GAME_ID"]].to_numpy())
            folds.append(np.full(len(test), b))
            log.info(f"  {name} block {b + 1}/{len(blocks)} {start.date()}..{end.date()} "
                     f"train={len(train):,} test={len(test):,} ({time.time() - t0:.0f}s)")
        out = OUT_DIR / (f"{name}__{tag}.npz" if tag else f"{name}.npz")
        np.savez_compressed(out, keys=np.vstack(keys).astype(str),
                            pmf=np.vstack(pmfs), fold=np.concatenate(folds))
        log.info(f"saved {out.name} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _load_npz(f: Path) -> pd.DataFrame:
    z = np.load(f)
    idx = pd.MultiIndex.from_arrays([z["keys"][:, 0].astype(int), z["keys"][:, 1]], names=["PLAYER_ID", "GAME_ID"])
    return pd.DataFrame(z["pmf"], index=idx)


def _load_pmfs() -> dict[str, pd.DataFrame]:
    return {f.stem: _load_npz(f) for f in sorted(OUT_DIR.glob("*.npz"))}


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


def line_ll_rows(pmf: np.ndarray, y: np.ndarray, lf: pd.DataFrame) -> np.ndarray:
    """Per-(row, line) log-loss of P(REB > line), same clipping as score()."""
    p = np.clip(prob_over(pmf[lf["row"].values], lf["line"].values), 1e-6, 1 - 1e-6)
    o = (y[lf["row"].values] > lf["line"].values).astype(float)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def date_bootstrap(diff: np.ndarray, dates: np.ndarray, n_boot: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of ``diff`` with a CI that resamples whole game dates (players on one slate share news)."""
    s = pd.DataFrame({"d": diff, "date": dates}).groupby("date")["d"].agg(["sum", "count"])
    sums, cnts = s["sum"].to_numpy(), s["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boots = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return (float(sums.sum() / cnts.sum()), float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def score(pmf: np.ndarray, y: np.ndarray, lf: pd.DataFrame) -> dict:
    yc = np.minimum(y, TOP).astype(int)
    eps = 1e-12
    nll = -np.log(np.maximum(pmf[np.arange(len(yc)), yc], eps)).mean()
    p = np.clip(prob_over(pmf[lf["row"].values], lf["line"].values), 1e-6, 1 - 1e-6)
    o = (y[lf["row"].values] > lf["line"].values).astype(float)
    ll = line_ll_rows(pmf, y, lf).mean()
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


def ablation(model: str, base_tag: str = "base", since: str | None = None) -> pd.DataFrame:
    """ΔLL (tag − base; negative = tag better) per tagged run of ``model``, on shared rows."""
    feat = load_features().set_index(["PLAYER_ID", "GAME_ID"])
    runs = {f.stem.split("__", 1)[1]: _load_npz(f) for f in sorted(OUT_DIR.glob(f"{model}__*.npz"))}
    base = runs.pop(base_tag)
    rows = []
    for tag, d in runs.items():
        common = base.index.intersection(d.index).intersection(feat.index)
        f = feat.loc[common]
        if since:
            keep = (f["GAME_DATE"] >= pd.Timestamp(since)).to_numpy()
            common, f = common[keep], f[keep]
        y, lf = f["REB"].to_numpy(), line_frame(f)
        ll_b = line_ll_rows(base.loc[common].to_numpy(), y, lf)
        ll_t = line_ll_rows(d.loc[common].to_numpy(), y, lf)
        point, lo, hi = date_bootstrap(ll_t - ll_b, f["GAME_DATE"].to_numpy()[lf["row"].values])
        rows.append({"tag": tag, "rows": len(common), "ll_base": ll_b.mean(), "ll_tag": ll_t.mean(),
                     "d_ll": point, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values("d_ll")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "features", "ablation"])
    ap.add_argument("--models", default=",".join(CANDIDATES))
    ap.add_argument("--since", default="2023-10-01")
    ap.add_argument("--ensembles", action="store_true")
    ap.add_argument("--tag", default="", help="run: save as <model>__<tag>.npz")
    ap.add_argument("--model", default="c7_ens_c4_c5", help="ablation: model whose tagged runs to compare")
    ap.add_argument("--base-tag", default="base")
    ap.add_argument("--since-score", default=None, help="ablation: score only rows on/after this date")
    a = ap.parse_args()
    if a.cmd == "features":
        load_features(refresh=True)
    elif a.cmd == "run":
        run(a.models.split(","), a.since, a.tag)
    elif a.cmd == "ablation":
        pd.set_option("display.width", 200)
        print(ablation(a.model, a.base_tag, a.since_score).round(5).to_string(index=False))
    else:
        pd.set_option("display.width", 200)
        print(summary(a.ensembles).round(4).to_string(index=False))
