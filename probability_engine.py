"""
probability_engine.py
---------------------
Turns a model's rebound PMF into P(REB > k) for each Kalshi line.

Every model in model_zoo emits a full PMF (0..19, 20+), so P(over) and
P(under) come from the same distribution and stay coherent. The old engine
built an NB from a single global residual variance around a MEDIAN
prediction (MAE objective) — see bakeoff.py's c1_legacy_mae_nb for what that
cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from model_zoo import TOP, prob_over


@dataclass
class ProbabilityResult:
    player_id: int
    player_name: str
    game_date: str
    kalshi_line: float
    predicted_lambda: float          # mean of the PMF (kept for journal continuity)
    p_over: float
    p_under: float
    distribution: str
    variance: float
    pmf: tuple[float, ...] | None = None
    p_over_calibrated: float | None = None
    p_under_calibrated: float | None = None
    games_played: int | None = None


def pmf_mean_var(pmf: np.ndarray) -> tuple[float, float]:
    k = np.arange(TOP + 1)
    m = float((pmf * k).sum())
    return m, float((pmf * (k - m) ** 2).sum())


def calculate_probabilities(
    predictions: Sequence[dict[str, Any]],
    distribution: str = "pmf",
) -> list[ProbabilityResult]:
    """
    predictions: dicts with player_id, player_name, game_date, kalshi_line,
    pmf (length TOP+1), optional games_played.
    """
    out = []
    for pred in predictions:
        pmf = np.asarray(pred["pmf"], dtype=float)
        k = float(pred["kalshi_line"])
        p_over = float(prob_over(pmf[None, :], k)[0])
        mean, var = pmf_mean_var(pmf)
        out.append(ProbabilityResult(
            player_id=int(pred.get("player_id", 0)),
            player_name=str(pred.get("player_name", "")),
            game_date=str(pred.get("game_date", "")),
            kalshi_line=k,
            predicted_lambda=mean,
            p_over=p_over,
            p_under=1.0 - p_over,
            distribution=distribution,
            variance=var,
            pmf=tuple(float(x) for x in pmf),
            games_played=pred.get("games_played"),
        ))
    return out


def brier_score(actuals: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - actuals) ** 2))
