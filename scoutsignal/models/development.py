"""Prospect development: time from draft to NHL debut, with right-censoring.

Most recent draft classes have not finished developing, so "made the NHL" is unknown for
them, not "no". A discrete-time hazard model handles that directly: each prospect
contributes one row per development year they were observed, with a 1 in the year they
debuted. A logistic regression on those rows estimates the yearly hazard; survival is the
running product of (1 - hazard).

The features come from the scouting-report extractor, so this is where the information
extraction pays off: does what scouts wrote predict who makes it, beyond draft round?
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as C
from ..extract import extract

BASELINE = ["draft_round"]
TEXT = [f"g_{s}" for s in C.SKILLS] + [f"m_{s}" for s in C.SKILLS] + ["role_tier", "height_in", "is_d"]


def report_features(texts, draft_round=None) -> pd.DataFrame:
    """Extract grades from reports. Unmentioned skills get the league-average 50 plus a
    'mentioned' flag, since a scout skipping a skill is itself mildly informative."""
    rows = []
    for t in texts:
        r = extract(t)
        g = r.grades()
        row = {f"g_{s}": g.get(s, 50) for s in C.SKILLS}
        row.update({f"m_{s}": int(s in g) for s in C.SKILLS})
        row["role_tier"] = r.role_tier or 3
        row["height_in"] = r.height_in or 72
        row["is_d"] = int(r.position == "D")
        rows.append(row)
    df = pd.DataFrame(rows)
    if draft_round is not None:
        df["draft_round"] = np.asarray(draft_round)
    return df


def person_period(df: pd.DataFrame) -> pd.DataFrame:
    """One row per prospect per observed development year."""
    pp = df.loc[df.index.repeat(df.dev_years_observed)].copy()
    pp["year"] = pp.groupby(level=0).cumcount() + 1
    pp["debut"] = ((pp.year == pp.dev_years_observed) & (pp.made_nhl == 1)).astype(int)
    return pp.reset_index(drop=True)


def _design(df: pd.DataFrame, features: list[str], year: np.ndarray) -> np.ndarray:
    dummies = (year[:, None] == np.arange(1, C.MAX_DEV_YEARS + 1)[None, :]).astype(float)
    return np.hstack([df[features].to_numpy(float), dummies])


@dataclass
class DevelopmentModel:
    features: list[str] = field(default_factory=lambda: BASELINE + TEXT)
    C_reg: float = 1.0

    def fit(self, df: pd.DataFrame) -> "DevelopmentModel":
        pp = person_period(df)
        X = _design(pp, self.features, pp.year.to_numpy())
        self.clf_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C_reg, max_iter=2000))
        self.clf_.fit(X, pp.debut)
        return self

    def hazard(self, df: pd.DataFrame) -> np.ndarray:
        """(n_prospects, MAX_DEV_YEARS) yearly debut hazard."""
        n = len(df)
        rep = df.loc[df.index.repeat(C.MAX_DEV_YEARS)]
        year = np.tile(np.arange(1, C.MAX_DEV_YEARS + 1), n)
        return self.clf_.predict_proba(_design(rep, self.features, year))[:, 1].reshape(n, -1)

    def debut_curve(self, df: pd.DataFrame) -> np.ndarray:
        """P(debuted by end of year k) for k = 1..MAX_DEV_YEARS."""
        return 1 - np.cumprod(1 - self.hazard(df), axis=1)


def kaplan_meier(time, event) -> pd.DataFrame:
    time, event = np.asarray(time), np.asarray(event)
    out, surv = [], 1.0
    for k in range(1, C.MAX_DEV_YEARS + 1):
        at_risk = (time >= k).sum()
        d = ((time == k) & (event == 1)).sum()
        if at_risk:
            surv *= 1 - d / at_risk
        out.append({"year": k, "at_risk": int(at_risk), "debuts": int(d), "p_debut": 1 - surv})
    return pd.DataFrame(out)
