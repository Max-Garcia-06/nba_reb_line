"""
model_zoo.py
------------
Candidate rebound models for the bake-off (bakeoff.py). Every model maps a
feature frame to a full probability mass function over rebounds:

    pmf[:, j] = P(REB = j)  for j = 0..TOP-1,   pmf[:, TOP] = P(REB >= TOP)

Lines are scored from the PMF (P(REB > k) = sum of pmf above floor(k)), so
over/under stay coherent. mlb_tb_line moved from mean + Poisson/NB to a full
ordinal PMF for the same reason; here several shapes compete instead of
assuming one.

Training protocol (inside each model): the last CALIB_FRAC of the training
window, by date, is held out to pick the boosting round count and to fit any
dispersion / residual parameters on out-of-sample predictions. Then the mean
model is refit on the full window with that round count.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import optimize, special, stats

from feature_store import MODEL_FEATURES

TOP = 20                 # top bucket = 20+ rebounds (Kalshi max line 17.5 seen)
CALIB_FRAC = 0.15
GRID = np.arange(TOP + 1)

XGB_BASE = dict(
    n_estimators=3000, learning_rate=0.03, max_depth=5, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, tree_method="hist",
    early_stopping_rounds=100, random_state=42, n_jobs=-1,
)


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def _cap(pmf_full: np.ndarray) -> np.ndarray:
    """Collapse a PMF over 0..N (N > TOP) into 0..TOP-1 + TOP+ bucket, renormalised."""
    out = np.empty((pmf_full.shape[0], TOP + 1))
    out[:, :TOP] = pmf_full[:, :TOP]
    out[:, TOP] = np.clip(1.0 - out[:, :TOP].sum(axis=1), 0.0, None)
    return out / out.sum(axis=1, keepdims=True)


def poisson_pmf(mu: np.ndarray) -> np.ndarray:
    mu = np.maximum(np.asarray(mu, float), 1e-3)[:, None]
    return _cap(stats.poisson.pmf(GRID[None, :], mu))


def nb_pmf(mu: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
    """NB2: Var = mu + alpha*mu^2. alpha -> 0 is Poisson."""
    mu = np.maximum(np.asarray(mu, float), 1e-3)
    alpha = np.broadcast_to(np.maximum(np.asarray(alpha, float), 1e-5), mu.shape)
    n = 1.0 / alpha
    p = n / (n + mu)
    return _cap(stats.nbinom.pmf(GRID[None, :], n[:, None], p[:, None]))


def nb_logpmf(y: np.ndarray, mu: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    n = 1.0 / alpha
    return (special.gammaln(y + n) - special.gammaln(n) - special.gammaln(y + 1)
            + n * np.log(n / (n + mu)) + y * np.log(mu / (n + mu)))


def fit_nb_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-3)
    res = optimize.minimize_scalar(
        lambda la: -nb_logpmf(y, mu, np.full_like(mu, math.exp(la))).sum(),
        bounds=(-8, 2), method="bounded")
    return float(math.exp(res.x))


def prob_over(pmf: np.ndarray, line: float | np.ndarray) -> np.ndarray:
    """P(REB > line) for half-integer lines; vectorised over rows."""
    tail = np.cumsum(pmf[:, ::-1], axis=1)[:, ::-1]          # tail[:, j] = P(REB >= j)
    start = np.floor(np.asarray(line, float)).astype(int) + 1
    start = np.broadcast_to(start, (pmf.shape[0],))
    out = np.zeros(pmf.shape[0])
    ok = start <= TOP
    out[ok] = tail[np.arange(pmf.shape[0])[ok], start[ok]]
    return out


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.sort(df["GAME_DATE"].unique())
    cut = dates[int(len(dates) * (1 - CALIB_FRAC))]
    return df[df["GAME_DATE"] < cut], df[df["GAME_DATE"] >= cut]


def _xgb_fit(objective: str, tr: pd.DataFrame, ca: pd.DataFrame, full: pd.DataFrame,
             target: str = "REB", margin: str | None = None, **over) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    """Fit on tr with early stop on ca; refit on full with best round count.
    Returns (holdout_model, full_model)."""
    params = {**XGB_BASE, "objective": objective, **over}
    kw_tr = {"base_margin": tr[margin].values} if margin else {}
    kw_ca = {"base_margin_eval_set": [ca[margin].values]} if margin else {}
    m_ho = xgb.XGBRegressor(**params)
    m_ho.fit(tr[MODEL_FEATURES], tr[target], eval_set=[(ca[MODEL_FEATURES], ca[target])],
             verbose=False, **kw_tr, **kw_ca)
    best = max(50, int((m_ho.best_iteration or 0) + 1))
    p_full = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    p_full["n_estimators"] = int(best * 1.1)
    m_full = xgb.XGBRegressor(**p_full)
    m_full.fit(full[MODEL_FEATURES], full[target], verbose=False,
               **({"base_margin": full[margin].values} if margin else {}))
    return m_ho, m_full


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

class Model:
    name = "base"

    def fit(self, df: pd.DataFrame) -> "Model":
        raise NotImplementedError

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class NaiveNB(Model):
    """C0: mean = trailing EWM rebounds, one global NB alpha. The bar to clear."""
    name = "c0_naive_nb"

    def fit(self, df):
        mu = df["reb_ewm"].fillna(df["REB"].mean()).values
        self.alpha = fit_nb_alpha(df["REB"].values, mu)
        self.fill = float(df["REB"].mean())
        return self

    def pmf(self, df):
        return nb_pmf(df["reb_ewm"].fillna(self.fill).values, self.alpha)


class LegacyMAENB(Model):
    """C1: what nba_reb_line shipped — XGB MAE (a MEDIAN model) + NB from global
    in-sample residual variance. Kept to measure what the fixes are worth."""
    name = "c1_legacy_mae_nb"

    def fit(self, df):
        params = {**XGB_BASE, "objective": "reg:absoluteerror"}
        params.pop("early_stopping_rounds")
        params["n_estimators"] = 1000
        params["learning_rate"] = 0.02
        self.m = xgb.XGBRegressor(**params).fit(df[MODEL_FEATURES], df["REB"], verbose=False)
        resid = df["REB"].values - self.m.predict(df[MODEL_FEATURES])
        self.var = float(resid.var())
        return self

    def pmf(self, df):
        mu = np.maximum(self.m.predict(df[MODEL_FEATURES]), 0.01)
        # probability_engine._nb_params: var = max(global_var, mu+eps); alpha = (var-mu)/mu^2
        var = np.maximum(self.var, mu + 1e-6)
        return nb_pmf(mu, (var - mu) / mu ** 2)


class XGBPoisson(Model):
    """C2: XGB count:poisson mean, Poisson PMF."""
    name = "c2_xgb_poisson"

    def fit(self, df):
        tr, ca = _split(df)
        _, self.m = _xgb_fit("count:poisson", tr, ca, df)
        return self

    def pmf(self, df):
        return poisson_pmf(self.m.predict(df[MODEL_FEATURES]))


class XGBNB(Model):
    """C3: XGB Poisson-objective mean + NB alpha fit by MLE on held-out predictions."""
    name = "c3_xgb_nb"

    def fit(self, df):
        tr, ca = _split(df)
        ho, self.m = _xgb_fit("count:poisson", tr, ca, df)
        self.alpha = fit_nb_alpha(ca["REB"].values, ho.predict(ca[MODEL_FEATURES]))
        return self

    def pmf(self, df):
        return nb_pmf(self.m.predict(df[MODEL_FEATURES]), self.alpha)


DISP_FEATURES = ["min_cv", "reb_cv", "exp_margin_abs", "vacuum_min", "is_playoffs", "is_b2b", "log_mu"]


def _disp_X(df: pd.DataFrame, mu: np.ndarray) -> np.ndarray:
    z = pd.DataFrame({
        "min_cv": (df["min_std_roll"] / df["min_roll"].replace(0, np.nan)).fillna(0.2),
        "reb_cv": (df["reb_std_roll"] / df["reb_roll"].replace(0, np.nan)).fillna(0.5).clip(0, 3),
        "exp_margin_abs": df["exp_margin_abs"].fillna(3.5) / 10,
        "vacuum_min": df["vacuum_min"].fillna(0) / 100,
        "is_playoffs": df["is_playoffs"].fillna(0),
        "is_b2b": df["is_b2b"].fillna(0),
        "log_mu": np.log(np.maximum(mu, 0.1)),
    })
    return np.column_stack([np.ones(len(z)), z[DISP_FEATURES].values])


class XGBNBHetero(Model):
    """C4: XGB mean + NB with log(alpha) linear in volatility/context features (MLE on holdout)."""
    name = "c4_xgb_nb_hetero"

    def fit(self, df):
        tr, ca = _split(df)
        ho, self.m = _xgb_fit("count:poisson", tr, ca, df)
        mu = np.maximum(ho.predict(ca[MODEL_FEATURES]), 1e-3)
        Z, y = _disp_X(ca, mu), ca["REB"].values
        a0 = math.log(fit_nb_alpha(y, mu))
        x0 = np.zeros(Z.shape[1]); x0[0] = a0
        obj = lambda b: -nb_logpmf(y, mu, np.exp(np.clip(Z @ b, -9, 3))).sum() + 1.0 * (b[1:] ** 2).sum()
        self.beta = optimize.minimize(obj, x0, method="L-BFGS-B").x
        return self

    def pmf(self, df):
        mu = np.maximum(self.m.predict(df[MODEL_FEATURES]), 1e-3)
        return nb_pmf(mu, np.exp(np.clip(_disp_X(df, mu) @ self.beta, -9, 3)))


class MinutesRate(Model):
    """C5: REB = rate x minutes.
    Minutes: XGB mean + empirical holdout residuals by predicted-minutes bin
    (captures the fat left tail: early injury exits, foul trouble, blowouts).
    Rate: XGB Poisson with log(minutes) offset. REB | minutes ~ NB(rate*min, alpha).
    PMF = average over N_Q minutes quantiles."""
    name = "c5_minutes_rate"
    N_Q = 25
    BINS = np.array([0, 20, 26, 32, 48])

    def fit(self, df):
        df = df.assign(log_min=np.log(np.maximum(pd.to_numeric(df["MIN"], errors="coerce").fillna(0), 0.5)))
        tr, ca = _split(df)
        ho_m, self.m_min = _xgb_fit("reg:squarederror", tr, ca, df, target="MIN")
        pred_ca = ho_m.predict(ca[MODEL_FEATURES])
        resid = ca["MIN"].values - pred_ca
        qs = (np.arange(self.N_Q) + 0.5) / self.N_Q
        b = np.digitize(pred_ca, self.BINS[1:-1])
        self.resid_q = np.vstack([np.quantile(resid[b == i] if (b == i).sum() > 200 else resid, qs)
                                  for i in range(len(self.BINS) - 1)])
        pos_tr, pos_ca, pos = tr[tr["MIN"] > 0], ca[ca["MIN"] > 0], df[df["MIN"] > 0]
        ho_r, self.m_rate = _xgb_fit("count:poisson", pos_tr, pos_ca, pos, margin="log_min")
        rate_ca = ho_r.predict(pos_ca[MODEL_FEATURES], base_margin=np.zeros(len(pos_ca)))
        self.alpha = fit_nb_alpha(pos_ca["REB"].values, rate_ca * pos_ca["MIN"].values)
        return self

    def minutes_grid(self, df) -> np.ndarray:
        mu_m = self.m_min.predict(df[MODEL_FEATURES])
        b = np.digitize(mu_m, self.BINS[1:-1])
        return np.clip(mu_m[:, None] + self.resid_q[b], 0.0, 53.0)   # (n, N_Q)

    def pmf(self, df):
        rate = self.m_rate.predict(df[MODEL_FEATURES], base_margin=np.zeros(len(df)))
        mins = self.minutes_grid(df)
        out = np.zeros((len(df), TOP + 1))
        for q in range(self.N_Q):
            out += nb_pmf(rate * mins[:, q] + 1e-3, self.alpha)
        return out / self.N_Q


class LGBMulticlass(Model):
    """C6: LightGBM multiclass on REB clipped to 0..TOP -> PMF directly."""
    name = "c6_lgbm_multiclass"
    PARAMS = dict(objective="multiclass", num_class=TOP + 1, learning_rate=0.05, num_leaves=31,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, verbose=-1, seed=42)

    def fit(self, df):
        tr, ca = _split(df)
        yc = lambda d: np.minimum(d["REB"].values, TOP).astype(int)
        dtr = lgb.Dataset(tr[MODEL_FEATURES], yc(tr))
        dca = lgb.Dataset(ca[MODEL_FEATURES], yc(ca), reference=dtr)
        ho = lgb.train(self.PARAMS, dtr, 2000, valid_sets=[dca],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        rounds = max(50, int(ho.best_iteration * 1.1))
        self.m = lgb.train(self.PARAMS, lgb.Dataset(df[MODEL_FEATURES], yc(df)), rounds)
        return self

    def pmf(self, df):
        p = self.m.predict(df[MODEL_FEATURES])
        return p / p.sum(axis=1, keepdims=True)


CANDIDATES: dict[str, type[Model]] = {c.name: c for c in
                                      [NaiveNB, LegacyMAENB, XGBPoisson, XGBNB, XGBNBHetero,
                                       MinutesRate, LGBMulticlass]}
