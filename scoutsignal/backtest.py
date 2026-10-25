"""Model audit: rolling-origin backtests, drift checks and recalibration after a rule change.

A front office keeps using a model long after the season it was trained on. This module
answers the three questions to ask before trusting it again:

1. How does it score on seasons it never saw?           rolling_origin()
2. Did the league change under it, and how?              rule_change_audit()
   Feature drift (PSI) says whether the shots look different; calibration-in-the-large
   (observed / expected goals) says whether the same shot now scores more often.
3. What is the cheapest fix that restores calibration?   recalibrate()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from . import config as C
from .features import PBP_NUMERIC, TRACKING_NUMERIC
from .metrics import binary_report
from .models.xg import XGModel


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index; > 0.1 is worth a look, > 0.25 is a real shift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0] / len(expected)
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def recalibrate(p: np.ndarray, y: np.ndarray) -> float:
    """Logit offset d such that sum(sigmoid(logit(p) + d)) equals observed goals."""
    z = np.log(p / (1 - p))
    f = lambda d: (1 / (1 + np.exp(-(z + d)))).sum() - y.sum()
    return float(optimize.brentq(f, -3, 3))


def rolling_origin(shots: pd.DataFrame, feature_sets=("pbp", "tracking"), min_train: int = 2) -> list[dict]:
    seasons = sorted(shots.season.unique())
    rows = []
    for s in seasons[min_train:]:
        train, test = shots[shots.season < s], shots[shots.season == s]
        for fs in feature_sets:
            if fs == "tracking" and test[TRACKING_NUMERIC[0]].isna().all():
                continue
            p = XGModel(fs).fit(train).predict(test)
            rows.append({"test_season": int(s), "model": fs, **binary_report(test.is_goal, p)})
        if "p_true" in test and test.p_true.notna().all():
            rows.append({"test_season": int(s), "model": "oracle (true p)",
                         **binary_report(test.is_goal, test.p_true)})
    return rows


def rule_change_audit(shots: pd.DataFrame, change_season: int = C.RULE_CHANGE_SEASON,
                      feature_set: str = "tracking") -> dict:
    pre = shots[shots.season < change_season]
    model = XGModel(feature_set).fit(pre)
    numeric = PBP_NUMERIC + (TRACKING_NUMERIC if feature_set == "tracking" else [])

    by_season = []
    for s in sorted(shots.season.unique()):
        cur = shots[shots.season == s]
        p = model.predict(cur)
        drift = {c: round(psi(pre[c].to_numpy(float), cur[c].to_numpy(float)), 4) for c in numeric}
        by_season.append({
            "season": int(s), "in_training": bool(s < change_season),
            "obs_over_exp": round(float(cur.is_goal.sum() / p.sum()), 3),
            "goal_rate": round(float(cur.is_goal.mean()), 4),
            "max_feature_psi": max(drift.values()), "psi": drift,
        })

    # Remedies, scored on the season after the change: keep the frozen model, shift its
    # intercept using the first post-change season, or retrain with recent seasons upweighted.
    first, second = change_season, change_season + 1
    remedies = {}
    if second in shots.season.values:
        y1, d1 = shots[shots.season == first], shots[shots.season == second]
        p_frozen = model.predict(d1)
        offset = recalibrate(model.predict(y1), y1.is_goal.to_numpy())
        model.offset = offset
        p_recal = model.predict(d1)
        model.offset = 0.0
        hist = shots[shots.season <= first]
        w = np.where(hist.season == first, 3.0, 1.0)
        p_weighted = XGModel(feature_set).fit(hist, sample_weight=w).predict(d1)
        remedies = {
            "evaluated_on": int(second),
            "intercept_offset": round(offset, 3),
            "frozen": binary_report(d1.is_goal, p_frozen),
            "recalibrated": binary_report(d1.is_goal, p_recal),
            "retrained_recency_weighted": binary_report(d1.is_goal, p_weighted),
        }

    post = [r for r in by_season if not r["in_training"]]
    flagged = [r["season"] for r in post if abs(r["obs_over_exp"] - 1) > 0.08]
    feature_shift = any(r["max_feature_psi"] > 0.1 for r in post)
    verdict = ("no material drift" if not flagged else
               "feature drift: shots look different" if feature_shift else
               "concept drift: same shots, more goals; recalibrate rather than re-engineer features")
    return {"change_season": change_season, "by_season": by_season, "flagged_seasons": flagged,
            "verdict": verdict, "remedies": remedies}
