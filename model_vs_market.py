"""
model_vs_market.py
------------------
Score model probabilities against the Kalshi book on FULL slates — every
market, no trade filter — settled against actual rebounds.

Port of mlb_tb_line/model_vs_market.py (SYSTEM.md §3.9, §5, §8): fills are a
selection-biased subset (only where the model disagreed with the market), so
the honest test of whether the model earns blend weight is the full slate,
sliced by line and by model-vs-market disagreement. `refit-blend` uses
evaluate_day() to refit the per-bucket blend weights weekly.

Differences from MLB
--------------------
* pit_train defaults to True: each day is scored with a model trained only on
  data before that day's week (cached per week under models/pit_cache). MLB's
  default used the saved model — look-ahead in the model's favour, which
  inflates the fitted w that `refit-blend` then trades on.
* Market source: live snapshots (data/snapshots, captured by `snapshot`) when
  present, else the Kalshi historical candles (kalshi_history.py).
* Players are resolved to box scores by identity_bridge within the game;
  DNPs drop out (Kalshi settles those at the pregame fair price).
"""

from __future__ import annotations

import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

import data_engine as de
from config import DB_PATH, MODEL_DIR, MODEL_FAMILY
from edge_detector import fill_calibrated_probabilities
from feature_store import build_feature_table
from identity_bridge import PlayerIndex
from market_blend import fit_blend_weight
from market_snapshots import load_snapshots
from probability_engine import calculate_probabilities

log = logging.getLogger(__name__)

MAX_BOOK_SPREAD = 0.25
HISTORY_LEAD_MIN = 60
PIT_CACHE = MODEL_DIR / "pit_cache"
PIT_RETRAIN_DAYS = 7

_FEATURES: pd.DataFrame | None = None


@dataclass(frozen=True)
class ScoreRow:
    game_date: str
    player_name: str
    ticker: str
    line: float
    p_model_raw: float
    p_model_cal: float
    p_market_mid: float
    actual_reb: int
    y: float  # 1.0 if REB > line


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _features() -> pd.DataFrame:
    global _FEATURES
    if _FEATURES is None:
        _FEATURES = build_feature_table()
    return _FEATURES


def _market_rows_snapshots(game_date: str, earliest: bool) -> pd.DataFrame:
    snaps = load_snapshots(game_date, latest_only=not earliest, earliest_only=earliest)
    return pd.DataFrame([{
        "ticker": s.ticker, "event_ticker": s.event_ticker, "player_name": s.player_name,
        "line": float(s.line), "yes_bid": float(s.yes_bid), "yes_ask": float(s.yes_ask),
    } for s in snaps])


def _market_rows_history(game_date: str, lead_min: int = HISTORY_LEAD_MIN) -> pd.DataFrame:
    """Pregame book from Kalshi hourly candles: last candle closing >= lead_min before tip."""
    eng = create_engine(f"sqlite:///{DB_PATH}")
    try:
        mk = pd.read_sql(text("SELECT ticker, event_ticker, player_name, line, teams_code FROM kalshi_markets "
                              "WHERE game_date = :d AND result IN ('yes','no')"), eng, params={"d": game_date})
        if mk.empty:
            return mk
        sch = pd.read_sql(text("SELECT AWAY_TRICODE || HOME_TRICODE AS teams_code, TIP_UTC FROM schedule "
                               "WHERE GAME_DATE_EST = :d"), eng, params={"d": game_date})
        c = pd.read_sql(text("SELECT ticker, end_ts, yes_bid, yes_ask FROM kalshi_candles WHERE ticker IN "
                             "(SELECT ticker FROM kalshi_markets WHERE game_date = :d)"), eng, params={"d": game_date})
    except Exception as e:
        log.debug("history rows unavailable for %s: %s", game_date, e)
        return pd.DataFrame()
    mk = mk.merge(sch, on="teams_code", how="inner")
    tip = pd.to_datetime(mk["TIP_UTC"], utc=True).dt.as_unit("s").astype("int64")  # unit-safe
    c = c.merge(pd.DataFrame({"ticker": mk["ticker"], "tip_ts": tip}), on="ticker")
    c = c[(c["end_ts"] <= c["tip_ts"] - lead_min * 60)].sort_values("end_ts").groupby("ticker").tail(1)
    return mk.drop(columns=["TIP_UTC", "teams_code"]).merge(c[["ticker", "yes_bid", "yes_ask"]], on="ticker")


def _resolve(rows: pd.DataFrame, game_date: str) -> pd.DataFrame:
    """Attach PLAYER_ID / GAME_ID / REB from the box scores of the game in each event ticker."""
    gl = de.load_gamelogs()
    gl = gl[pd.to_datetime(gl["GAME_DATE"]).dt.strftime("%Y-%m-%d") == game_date]
    if gl.empty:
        return pd.DataFrame()
    idx = de.slate_schedule_index(game_date)
    out = []
    for et, grp in rows.groupby("event_ticker"):
        mu = de.parse_kalshi_event_matchup(et)
        g = idx.get(de.matchup_slug(*mu)) if mu else None
        if not g:
            continue
        box = gl[gl["GAME_ID"].astype(str) == g["game_id"]]
        pi = PlayerIndex(zip(box["PLAYER_ID"], box["PLAYER_NAME"]))
        reb = dict(zip(box["PLAYER_ID"].astype(int), pd.to_numeric(box["REB"]).astype(int)))
        for r in grp.itertuples():
            pid = pi.resolve(r.player_name)
            if pid:
                out.append({**r._asdict(), "PLAYER_ID": pid, "GAME_ID": g["game_id"], "REB": reb[pid]})
    return pd.DataFrame(out).drop(columns=["Index"], errors="ignore")


def _pit_model(game_date: str, pit_train: bool):
    from model import load_model, train_as_of

    if not pit_train:
        return load_model()[0]
    d = pd.Timestamp(game_date)
    as_of = (d - timedelta(days=d.weekday() % PIT_RETRAIN_DAYS)).strftime("%Y-%m-%d")
    path = PIT_CACHE / f"{MODEL_FAMILY}_{as_of}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    log.info("PIT-training %s as of %s", MODEL_FAMILY, as_of)
    m, _ = train_as_of(as_of, df=_features())
    PIT_CACHE.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(m, f)
    return m


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def evaluate_day(game_date: str, *, pit_train: bool = True, earliest: bool = False) -> list[ScoreRow]:
    """Score one day's full slate. Returns [] when markets/outcomes are missing."""
    rows = _market_rows_snapshots(game_date, earliest)
    if rows.empty:
        rows = _market_rows_history(game_date)
    if rows.empty:
        return []
    rows = rows[(rows["yes_bid"] > 0) & (rows["yes_ask"] < 1) & (rows["yes_ask"] > rows["yes_bid"])
                & (rows["yes_ask"] - rows["yes_bid"] <= MAX_BOOK_SPREAD)]
    rows = _resolve(rows, game_date)
    if rows.empty:
        log.warning("No resolvable markets/box scores for %s — run etl first", game_date)
        return []

    feats = _features()
    feats = feats[pd.to_datetime(feats["GAME_DATE"]).dt.strftime("%Y-%m-%d") == game_date]
    feats = feats.set_index(["PLAYER_ID", "GAME_ID"])
    keys = list(zip(rows["PLAYER_ID"], rows["GAME_ID"]))
    have = [k in feats.index for k in keys]
    rows = rows[have].reset_index(drop=True)
    if rows.empty:
        return []
    model = _pit_model(game_date, pit_train)
    uniq = feats.loc[sorted(set(zip(rows["PLAYER_ID"], rows["GAME_ID"])))]
    pmf = model.pmf(uniq.reset_index())
    pmf_by = {k: pmf[i] for i, k in enumerate(uniq.index)}

    preds = [{"player_id": r.PLAYER_ID, "player_name": r.player_name, "game_date": game_date,
              "kalshi_line": r.line, "pmf": pmf_by[(r.PLAYER_ID, r.GAME_ID)],
              "games_played": int(feats.loc[(r.PLAYER_ID, r.GAME_ID), "games_played"])}
             for r in rows.itertuples()]
    prs = calculate_probabilities(preds)
    fill_calibrated_probabilities(prs)
    return [ScoreRow(
        game_date=game_date, player_name=r.player_name, ticker=r.ticker, line=float(r.line),
        p_model_raw=float(pr.p_over),
        p_model_cal=float(pr.p_over_calibrated if pr.p_over_calibrated is not None else pr.p_over),
        p_market_mid=(float(r.yes_bid) + float(r.yes_ask)) / 2, actual_reb=int(r.REB),
        y=1.0 if r.REB > r.line else 0.0,
    ) for pr, r in zip(prs, rows.itertuples())]


def _logloss(rows: list[ScoreRow], key: str) -> float:
    p = np.clip([getattr(r, key) for r in rows], 1e-4, 1 - 1e-4)
    y = np.array([r.y for r in rows])
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _brier(rows: list[ScoreRow], key: str) -> float:
    return sum((getattr(r, key) - r.y) ** 2 for r in rows) / len(rows)


def disagreement_bucket(r: ScoreRow) -> str:
    d = abs(r.p_model_cal - r.p_market_mid)
    if d < 0.05:
        return "<0.05"
    if d < 0.10:
        return "0.05-0.10"
    if d < 0.15:
        return "0.10-0.15"
    return ">=0.15"


def summarize_slice(rows: list[ScoreRow]) -> dict:
    """n, log-losses/Briers for model vs market, and the fitted blend weight for this slice."""
    out = {
        "n": len(rows),
        "ll_model": _logloss(rows, "p_model_cal"),
        "ll_market": _logloss(rows, "p_market_mid"),
        "brier_model": _brier(rows, "p_model_cal"),
        "brier_market": _brier(rows, "p_market_mid"),
        "base_rate": sum(r.y for r in rows) / len(rows),
    }
    fit = [{"p": r.p_model_cal, "m": r.p_market_mid, "y": r.y, "weight": 1.0} for r in rows]
    try:
        w, diag = fit_blend_weight(fit)
        out["w_fit"] = w
        out["ll_blend"] = diag["logloss_best"]
    except ValueError:
        out["w_fit"] = float("nan")
        out["ll_blend"] = float("nan")
    return out


def summarize(rows: list[ScoreRow]) -> dict[str, dict[str, dict]]:
    """Slices: overall, by line, by model-market disagreement, by date."""
    groups: dict[str, dict[str, list[ScoreRow]]] = {
        "overall": {"all": rows},
        "line": defaultdict(list),
        "disagreement": defaultdict(list),
        "date": defaultdict(list),
    }
    for r in rows:
        groups["line"][f"{r.line:g}"].append(r)
        groups["disagreement"][disagreement_bucket(r)].append(r)
        groups["date"][r.game_date].append(r)
    return {
        gname: {k: summarize_slice(v) for k, v in sorted(g.items()) if v}
        for gname, g in groups.items()
    }
