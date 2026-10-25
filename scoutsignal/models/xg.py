"""Expected-goals model: gradient-boosted trees on play-by-play and tracking features.

Two feature sets share one class. "pbp" uses only what public play-by-play has; "tracking"
adds the spatio-temporal features. Monotonic constraints keep the model consistent with
what coaches already know (farther shots are worse, open shooters are better), which
matters more for buy-in than the last 0.001 of log loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .. import config as C
from ..features import PBP_CATEGORICAL, PBP_NUMERIC, TRACKING_NUMERIC

CATEGORIES = {
    "shot_type": list(C.SHOT_TYPES) + ["other"],
    "strength": ["EV", "PP", "SH"],
    "last_event_type": ["none", "faceoff", "hit", "giveaway", "takeaway", "shot-on-goal",
                        "missed-shot", "blocked-shot", "goal", "other"],
}
MONOTONIC = {"distance": -1, "angle": -1, "def_dist": 1, "goalie_err": 1, "is_rebound": 1}


def model_frame(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    cols = PBP_NUMERIC + (TRACKING_NUMERIC if feature_set == "tracking" else [])
    X = df[cols].astype(float).copy()
    for c in PBP_CATEGORICAL:
        values = df[c].where(df[c].isin(CATEGORIES[c]), "other" if "other" in CATEGORIES[c] else "EV")
        X[c] = pd.Categorical(values, categories=CATEGORIES[c])
    return X


@dataclass
class XGModel:
    feature_set: str = "pbp"
    params: dict = field(default_factory=lambda: dict(
        learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=0))
    offset: float = 0.0  # logit shift from recalibration, see backtest.recalibrate

    def fit(self, df: pd.DataFrame, sample_weight=None) -> "XGModel":
        X = model_frame(df, self.feature_set)
        cst = {c: MONOTONIC.get(c, 0) for c in X.columns if c not in PBP_CATEGORICAL}
        self.clf_ = HistGradientBoostingClassifier(
            categorical_features="from_dtype", monotonic_cst=cst, **self.params)
        self.clf_.fit(X, df.is_goal.to_numpy(), sample_weight=sample_weight)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self.clf_.predict_proba(model_frame(df, self.feature_set))[:, 1]
        if self.offset:
            z = np.log(p / (1 - p)) + self.offset
            p = 1 / (1 + np.exp(-z))
        return p


def modelling_shots(df: pd.DataFrame) -> pd.DataFrame:
    """Unblocked shots against a goalie: the population every xG number refers to."""
    return df[df.strength.isin(["EV", "PP", "SH"])].reset_index(drop=True)
