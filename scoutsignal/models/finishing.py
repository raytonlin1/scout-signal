"""Shooter finishing talent with partial pooling (Gamma-Poisson hierarchical model).

    goals_i ~ Poisson(theta_i * xG_i)       theta_i: goals per expected goal
    theta_i ~ Gamma(alpha, alpha)            league prior, mean 1

The prior strength alpha is fitted by maximising the marginal (negative binomial)
likelihood, so the data decide how much a 40-shot sample should be trusted. The posterior
is conjugate, Gamma(alpha + goals, alpha + xG), which gives exact credible intervals with
no MCMC, fast enough to recompute inside an API request.

Raw goals/xG is the number fans quote and it is mostly noise; the shrunk estimate is the
one to trade on. backtest_finishing() shows the difference on next-season goals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


def _marginal_loglik(alpha: float, goals: np.ndarray, xg: np.ndarray) -> float:
    # Negative binomial with mean xG and shape alpha (constant terms in goals dropped).
    return float(np.sum(
        special.gammaln(alpha + goals) - special.gammaln(alpha)
        + alpha * np.log(alpha / (alpha + xg)) + goals * np.log(xg / (alpha + xg))
    ))


@dataclass
class FinishingModel:
    alpha: float = 50.0

    def fit(self, goals, xg) -> "FinishingModel":
        goals, xg = np.asarray(goals, float), np.asarray(xg, float)
        res = optimize.minimize_scalar(lambda la: -_marginal_loglik(np.exp(la), goals, xg),
                                       bounds=(np.log(1.0), np.log(5000.0)), method="bounded")
        self.alpha = float(np.exp(res.x))
        return self

    def posterior(self, goals, xg, level: float = 0.8) -> pd.DataFrame:
        a = self.alpha + np.asarray(goals, float)
        b = self.alpha + np.asarray(xg, float)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return pd.DataFrame({
            "theta_mean": a / b,
            "theta_lo": stats.gamma.ppf(lo, a, scale=1 / b),
            "theta_hi": stats.gamma.ppf(hi, a, scale=1 / b),
            "p_above_avg": stats.gamma.sf(1.0, a, scale=1 / b),
        })


def leaderboard(seasons: pd.DataFrame, through_season: int, min_shots: int = 50) -> pd.DataFrame:
    """Pool every season up to through_season, fit the prior, rank shooters.

    'buy_low' flags shooters the model likes (P(theta > 1) >= 0.7) whose goals in the latest
    season fell short of their xG: the profile a market that reads raw goal totals undervalues.
    """
    hist = seasons[seasons.season <= through_season]
    agg = hist.groupby(["player_id", "name", "position"], dropna=False).agg(
        shots=("shots", "sum"), goals=("goals", "sum"), xg=("xg", "sum")).reset_index()
    latest = hist[hist.season == through_season].set_index("player_id")
    agg = agg[agg.shots >= min_shots].reset_index(drop=True)
    model = FinishingModel().fit(agg.goals, agg.xg)
    post = model.posterior(agg.goals, agg.xg)
    out = pd.concat([agg, post], axis=1)
    out["raw_ratio"] = out.goals / out.xg
    out["goals_above_xg_per_100"] = 100 * (out.theta_mean - 1) * out.xg / out.shots
    out["latest_goals_minus_xg"] = out.player_id.map(latest.goals - latest.xg)
    out["buy_low"] = (out.p_above_avg >= 0.7) & (out.latest_goals_minus_xg < 0)
    out.attrs["alpha"] = model.alpha
    return out.sort_values("theta_mean", ascending=False).reset_index(drop=True)


def backtest_finishing(seasons: pd.DataFrame, min_shots: int = 30) -> dict:
    """Predict next-season goals from xG times each finishing estimate.

    For each target season s, estimates use seasons < s and are scored on s. 'league' assumes
    every shooter finishes at xG (theta = 1); 'raw' uses past goals/xG; 'shrunk' uses the
    posterior mean.
    """
    rows = []
    for s in sorted(seasons.season.unique())[1:]:
        past = seasons[seasons.season < s].groupby("player_id")[["goals", "xg", "shots"]].sum()
        past = past[past.shots >= min_shots]
        model = FinishingModel().fit(past.goals, past.xg)
        post = model.posterior(past.goals, past.xg).set_index(past.index)
        nxt = seasons[(seasons.season == s) & seasons.player_id.isin(past.index)].set_index("player_id")
        raw = (past.goals / past.xg).reindex(nxt.index)
        for name, theta in [("league", pd.Series(1.0, index=nxt.index)), ("raw", raw),
                            ("shrunk", post.theta_mean.reindex(nxt.index))]:
            pred = np.maximum(theta * nxt.xg, 1e-3)  # raw goals/xG of 0 would predict 0
            rows.append(dict(season=int(s), method=name, alpha=model.alpha,
                             rmse=float(np.sqrt(np.mean((nxt.goals - pred) ** 2))),
                             poisson_dev=float(2 * np.mean(
                                 special.xlogy(nxt.goals, nxt.goals / pred) - (nxt.goals - pred)))))
    df = pd.DataFrame(rows)
    summary = df.groupby("method")[["rmse", "poisson_dev"]].mean().round(4)
    return {"by_season": df.round(4).to_dict("records"), "mean": summary.to_dict("index")}
