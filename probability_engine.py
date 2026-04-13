"""
probability_engine.py
---------------------
Converts the model's predicted mean (λ) into P(X > k) — the probability
a player exceeds the Kalshi rebound line.

Distributions
-------------
Poisson      : P(X > k | λ) — baseline, assumes mean == variance.
Negative Binomial (default) : accounts for over-dispersion common in
                               rebounding (clumpy possession sequences).

The dispersion parameter (r) is estimated from training residuals stored
in model_meta.pkl.  A higher variance-to-mean ratio → stronger NB effect.
"""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import poisson, nbinom

from config import DISTRIBUTION


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _nb_params(mu: float, var: float) -> tuple[float, float]:
    """
    Convert (mean, variance) → Negative Binomial (r, p) parameterisation.

    NB: E[X] = r*(1-p)/p, Var[X] = r*(1-p)/p^2
    => p = mu/var, r = mu*p/(1-p)
    Clamp var > mu to ensure valid NB (Poisson limit when var == mu).
    """
    var = max(var, mu + 1e-6)  # NB requires var > mean
    p = mu / var
    p = min(max(p, 1e-6), 1 - 1e-6)
    r = mu * p / (1 - p)
    r = max(r, 1e-3)
    return r, p


def prob_exceed_poisson(lam: float, k: float) -> float:
    """P(X > k) under Poisson(λ). k may be a half-integer (e.g. 7.5)."""
    floor_k = int(math.floor(k))
    return float(1.0 - poisson.cdf(floor_k, mu=lam))


def prob_exceed_nbinom(lam: float, k: float, variance: float) -> float:
    """
    P(X > k) under NegativeBinomial(r, p).
    variance: residual variance from model training.
    """
    r, p = _nb_params(lam, variance)
    floor_k = int(math.floor(k))
    return float(1.0 - nbinom.cdf(floor_k, n=r, p=p))


def prob_exceed(
    lam: float,
    k: float,
    variance: float,
    distribution: Literal["poisson", "nbinom"] = DISTRIBUTION,
) -> float:
    """
    Unified P(X > k) interface.

    Parameters
    ----------
    lam          : predicted mean rebounds (from model)
    k            : Kalshi line threshold (e.g. 7.5)
    variance     : residual variance — used only for NB; ignored for Poisson
    distribution : "poisson" or "nbinom"
    """
    lam = max(lam, 0.01)
    if distribution == "poisson":
        return prob_exceed_poisson(lam, k)
    return prob_exceed_nbinom(lam, k, variance)


# ---------------------------------------------------------------------------
# Probability result container
# ---------------------------------------------------------------------------

@dataclass
class ProbabilityResult:
    player_id: int
    player_name: str
    game_date: str
    kalshi_line: float          # The "k" threshold
    predicted_lambda: float     # Model's μ
    p_over: float               # P(X > k)
    p_under: float              # 1 - P(X > k)
    distribution: str
    variance: float

    @property
    def model_yes_price(self) -> float:
        """What this probability implies as a fair contract price."""
        return round(self.p_over, 4)

    def __repr__(self) -> str:
        return (
            f"ProbabilityResult({self.player_name} | line={self.kalshi_line} | "
            f"λ={self.predicted_lambda:.2f} | P(over)={self.p_over:.3f})"
        )


# ---------------------------------------------------------------------------
# Batch calculation
# ---------------------------------------------------------------------------

def calculate_probabilities(
    predictions: list[dict],
    variance: float,
    distribution: str = DISTRIBUTION,
) -> list[ProbabilityResult]:
    """
    Convert a list of prediction dicts to ProbabilityResult objects.

    Each dict must have:
      player_id, player_name, game_date, kalshi_line, predicted_lambda
    """
    results = []
    for pred in predictions:
        lam = pred["predicted_lambda"]
        k = pred["kalshi_line"]
        p_over = prob_exceed(lam, k, variance, distribution)
        results.append(
            ProbabilityResult(
                player_id=pred["player_id"],
                player_name=pred["player_name"],
                game_date=pred.get("game_date", ""),
                kalshi_line=k,
                predicted_lambda=lam,
                p_over=p_over,
                p_under=1.0 - p_over,
                distribution=distribution,
                variance=variance,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Distribution diagnostics (offline calibration check)
# ---------------------------------------------------------------------------

def brier_score(actuals: np.ndarray, probabilities: np.ndarray) -> float:
    """Lower is better. 0 = perfect. 0.25 = no-skill baseline for binary."""
    return float(np.mean((probabilities - actuals) ** 2))


def calibration_table(
    actuals: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> "pd.DataFrame":
    """Group predictions by probability bucket and check actual hit rate."""
    import pandas as pd

    df = pd.DataFrame({"prob": probabilities, "actual": actuals})
    df["bucket"] = pd.cut(df["prob"], bins=bins)
    cal = df.groupby("bucket", observed=False).agg(
        predicted_mean=("prob", "mean"),
        actual_rate=("actual", "mean"),
        count=("actual", "count"),
    ).reset_index()
    cal["calibration_error"] = (cal["predicted_mean"] - cal["actual_rate"]).abs()
    return cal


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario: model says λ=8.2, Kalshi line is 7.5, variance = 9.0
    lam, k, var = 8.2, 7.5, 9.0
    p_pois = prob_exceed_poisson(lam, k)
    p_nb = prob_exceed_nbinom(lam, k, var)
    print(f"λ={lam}, line={k}, var={var}")
    print(f"  Poisson  P(X > {k}) = {p_pois:.4f}")
    print(f"  NegBinom P(X > {k}) = {p_nb:.4f}")

    # Edge scenario: λ very close to k
    for lam2 in [7.0, 7.5, 8.0, 9.0, 10.0]:
        p = prob_exceed(lam2, 7.5, var)
        print(f"  λ={lam2:4.1f} → P(over 7.5) = {p:.3f}")
