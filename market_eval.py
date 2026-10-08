"""
market_eval.py
--------------
Full-slate model-vs-market scoring on the 2025-26 KXNBAREB history.

Port of mlb_tb_line's `model-vs-market` idea (SYSTEM.md §5, §8): score the
model against the book on EVERY market, not only the ones it would trade,
because trade-selected samples are biased toward the model's worst errors
(the winner's curse that cost MLB -5.4% ROI). Unlike the MLB run, which used
the saved model (look-ahead in the model's favour), predictions here come
from bakeoff.py's walk-forward folds, so they are point-in-time.

Steps
-----
1. identity: map Kalshi "Player Name" -> NBA PLAYER_ID within the game's two
   teams on that date (accent/suffix-insensitive), GAME_ID from schedule.
2. snapshot: market mid from the last hourly candle closing >= LEAD_MIN
   before tip. Requires a two-sided book and spread <= MAX_SPREAD.
3. score: log-loss / Brier for model, market, and logit blends; per line,
   per disagreement bucket (market_blend buckets), per month; plus a naive
   taker-at-ask P&L check with Kalshi fees.

Usage
-----
  python market_eval.py [--lead 60] [--models c3_xgb_nb,c5_minutes_rate]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from bakeoff import _load_pmfs
from config import DB_PATH
from identity_bridge import norm_player_name as norm_name
from model_zoo import prob_over

MAX_SPREAD = 0.25
TAKER_FEE = 0.07

def _eng():
    return create_engine(f"sqlite:///{DB_PATH}")


# ---------------------------------------------------------------------------
# 1. Identity
# ---------------------------------------------------------------------------

def map_markets() -> pd.DataFrame:
    e = _eng()
    mk = pd.read_sql("SELECT * FROM kalshi_markets WHERE result IN ('yes','no')", e)
    sch = pd.read_sql("SELECT GAME_ID, GAME_DATE_EST, TIP_UTC, HOME_TEAM_ID, AWAY_TEAM_ID, "
                      "HOME_TRICODE, AWAY_TRICODE FROM schedule", e)
    sch["teams_code"] = sch["AWAY_TRICODE"] + sch["HOME_TRICODE"]
    mk = mk.merge(sch.rename(columns={"GAME_DATE_EST": "game_date"}), on=["game_date", "teams_code"], how="left")

    gl = pd.read_sql("SELECT PLAYER_ID, PLAYER_NAME, TEAM_ID, GAME_ID FROM player_gamelogs WHERE SEASON='2025-26'", e)
    gl["nk"] = gl["PLAYER_NAME"].map(norm_name)
    mk["nk"] = mk["player_name"].map(norm_name)
    out = mk.merge(gl[["GAME_ID", "nk", "PLAYER_ID", "TEAM_ID"]], on=["GAME_ID", "nk"], how="left")

    # Fallback: last name + first initial within the same game.
    miss = out["PLAYER_ID"].isna() & out["GAME_ID"].notna()
    if miss.any():
        gl["nk2"] = gl["nk"].str.split().map(lambda t: f"{t[0][0]} {t[-1]}" if t else "")
        fb = out.loc[miss, ["ticker", "GAME_ID", "nk"]].copy()
        fb["nk2"] = fb["nk"].str.split().map(lambda t: f"{t[0][0]} {t[-1]}" if t else "")
        g2 = gl.drop_duplicates(["GAME_ID", "nk2"], keep=False)
        fb = fb.merge(g2[["GAME_ID", "nk2", "PLAYER_ID", "TEAM_ID"]], on=["GAME_ID", "nk2"], how="left")
        out = out.set_index("ticker")
        out.loc[fb["ticker"], ["PLAYER_ID", "TEAM_ID"]] = fb[["PLAYER_ID", "TEAM_ID"]].values
        out = out.reset_index()
    return out


# ---------------------------------------------------------------------------
# 2. Snapshot
# ---------------------------------------------------------------------------

def snapshot(mk: pd.DataFrame, lead_min: int = 60) -> pd.DataFrame:
    c = pd.read_sql("SELECT ticker, end_ts, yes_bid, yes_ask, volume FROM kalshi_candles", _eng())
    tip = pd.to_datetime(mk.set_index("ticker")["TIP_UTC"], utc=True, errors="coerce")
    tip_ts = tip.dt.as_unit("s").astype("int64").rename("tip_ts")  # unit-safe (pandas 3 infers [s])
    c = c.merge(tip_ts, left_on="ticker", right_index=True)
    c = c[c["end_ts"] <= c["tip_ts"] - lead_min * 60]
    c = c[(c["yes_bid"] > 0) & (c["yes_ask"] < 1) & (c["yes_ask"] > c["yes_bid"])]
    snap = c.sort_values("end_ts").groupby("ticker").tail(1)
    snap = snap.assign(mid=(snap["yes_bid"] + snap["yes_ask"]) / 2, spread=snap["yes_ask"] - snap["yes_bid"],
                       mins_before_tip=(snap["tip_ts"] - snap["end_ts"]) / 60)
    return snap[snap["spread"] <= MAX_SPREAD]


# ---------------------------------------------------------------------------
# 3. Score
# ---------------------------------------------------------------------------

def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def blend(p_model, p_mkt, w):
    return 1 / (1 + np.exp(-(w * _logit(p_model) + (1 - w) * _logit(p_mkt))))


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_w(p_model, p_mkt, y) -> float:
    grid = np.linspace(0, 1, 101)
    return float(grid[np.argmin([_ll(blend(p_model, p_mkt, w), y) for w in grid])])


def disagreement_bucket(d: np.ndarray) -> np.ndarray:
    return np.select([d < 0.05, d < 0.10, d < 0.15], ["<0.05", "0.05-0.10", "0.10-0.15"], ">=0.15")


def build_eval_frame(lead_min: int, models: list[str] | None) -> pd.DataFrame:
    mk = map_markets()
    snap = snapshot(mk, lead_min)
    df = mk.merge(snap[["ticker", "mid", "spread", "yes_bid", "yes_ask", "mins_before_tip"]], on="ticker")
    df = df[df["PLAYER_ID"].notna()].copy()
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
    df["y"] = (df["result"] == "yes").astype(float)
    pm = _load_pmfs()
    for name in (models or list(pm)):
        d = pm[name]
        idx = pd.MultiIndex.from_arrays([df["PLAYER_ID"], df["GAME_ID"]])
        have = idx.isin(d.index)
        p = np.full(len(df), np.nan)
        p[have] = prob_over(d.loc[idx[have]].to_numpy(), df.loc[have, "line"].to_numpy())
        df[f"p_{name}"] = p
    return df


def report(df: pd.DataFrame, models: list[str]) -> None:
    cols = [f"p_{m}" for m in models]
    base = df.dropna(subset=cols).copy()
    base["month"] = base["game_date"].str[:7]
    months = sorted(base["month"].unique())
    half = months[: len(months) // 2]
    first, second = base[base["month"].isin(half)], base[~base["month"].isin(half)]
    print(f"\nMarkets scored: {len(base):,} (of {len(df):,} mapped w/ snapshot)  "
          f"months={months[0]}..{months[-1]}  base rate={base['y'].mean():.3f}")
    print(f"market log-loss {_ll(base['mid'], base['y']):.4f}  brier {((base['mid'] - base['y']) ** 2).mean():.4f}\n")

    print(f"{'model':<22}{'LL':>8}{'ΔLL vs mkt':>12}{'w_fit(all)':>11}{'w_fit(H1)':>10}{'LL blend H2':>12}{'mkt H2':>8}")
    for m, c in zip(models, cols):
        ll = _ll(base[c], base["y"])
        w_all = fit_w(base[c].values, base["mid"].values, base["y"].values)
        w1 = fit_w(first[c].values, first["mid"].values, first["y"].values)
        llb = _ll(blend(second[c].values, second["mid"].values, w1), second["y"].values)
        print(f"{m:<22}{ll:>8.4f}{ll - _ll(base['mid'], base['y']):>+12.4f}{w_all:>11.2f}{w1:>10.2f}"
              f"{llb:>12.4f}{_ll(second['mid'], second['y']):>8.4f}")

    best = min(cols, key=lambda c: _ll(base[c], base["y"]))
    print(f"\nSlices for best model {best[2:]} (ΔLL = model − market; negative = model better):")
    base["dis"] = disagreement_bucket(np.abs(base[best] - base["mid"]))
    for key in ["line", "dis", "month"]:
        g = base.groupby(key)
        t = pd.DataFrame({
            "n": g.size(),
            "LL_model": g.apply(lambda x: _ll(x[best], x["y"])),
            "LL_mkt": g.apply(lambda x: _ll(x["mid"], x["y"])),
            "w_fit": g.apply(lambda x: fit_w(x[best].values, x["mid"].values, x["y"].values) if len(x) > 50 else np.nan),
        })
        t["ΔLL"] = t["LL_model"] - t["LL_mkt"]
        print(f"\nby {key}:\n{t.round(4).to_string()}")

    # Naive taker P&L at the ask, using the H1-fitted blend on H2 (no look-ahead in w).
    w1 = fit_w(first[best].values, first["mid"].values, first["y"].values)
    s = second.copy()
    s["p_yes"] = blend(s[best].values, s["mid"].values, w1)
    print(f"\nTaker-at-ask P&L check on H2 (blend w={w1:.2f} fit on H1), 1 contract per signal:")
    for thr in (0.02, 0.04, 0.06, 0.08):
        pnl, n = 0.0, 0
        for side, p, px in (("yes", s["p_yes"], s["yes_ask"]), ("no", 1 - s["p_yes"], 1 - s["yes_bid"])):
            fee = TAKER_FEE * px * (1 - px)
            sig = (p - px - fee) > thr
            win = (s["y"] == 1) if side == "yes" else (s["y"] == 0)
            pnl += float(((win.astype(float) - px - fee)[sig]).sum())
            n += int(sig.sum())
        print(f"  edge>{thr:.2f}: {n:5d} bets  P&L ${pnl:8.2f}  per bet {pnl / max(n, 1):+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", type=int, default=60, help="minutes before tip for the market snapshot")
    ap.add_argument("--models", default=None)
    a = ap.parse_args()
    models = a.models.split(",") if a.models else sorted(_load_pmfs())
    df = build_eval_frame(a.lead, models)
    report(df, models)
