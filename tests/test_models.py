import numpy as np
import pandas as pd
import pytest

from scoutsignal.metrics import concordance_index
from scoutsignal.models.development import DevelopmentModel, kaplan_meier, person_period
from scoutsignal.models.finishing import FinishingModel


def test_finishing_recovers_prior_and_shrinks():
    rng = np.random.default_rng(0)
    alpha = 25.0
    xg = rng.uniform(5, 40, 3000)
    theta = rng.gamma(alpha, 1 / alpha, 3000)
    goals = rng.poisson(theta * xg)
    m = FinishingModel().fit(goals, xg)
    assert 15 < m.alpha < 40
    post = m.posterior(goals, xg)
    raw = goals / xg
    # Shrunk estimates sit between raw and 1 and track the truth better.
    assert np.all((post.theta_mean - 1) * (raw - 1) >= 0)
    assert np.mean((post.theta_mean - theta) ** 2) < np.mean((raw - theta) ** 2)
    assert np.mean((post.theta_lo <= theta) & (theta <= post.theta_hi)) == pytest.approx(0.8, abs=0.04)


def test_person_period_and_kaplan_meier():
    df = pd.DataFrame({"dev_years_observed": [2, 3, 1], "made_nhl": [1, 0, 1], "draft_round": [1, 2, 1]})
    pp = person_period(df)
    assert len(pp) == 6 and pp.debut.sum() == 2
    km = kaplan_meier(df.dev_years_observed, df.made_nhl)
    assert km.loc[0, "p_debut"] == pytest.approx(1 / 3)
    assert km.loc[1, "p_debut"] == pytest.approx(1 - (2 / 3) * (1 / 2))


def test_development_model_ranks_by_feature():
    rng = np.random.default_rng(1)
    n = 2000
    skill = rng.normal(0, 1, n)
    years = np.zeros(n, int)
    event = np.zeros(n, int)
    for i in range(n):
        for k in range(1, 9):
            if rng.random() < 1 / (1 + np.exp(-(-2.5 + 1.2 * skill[i]))):
                years[i], event[i] = k, 1
                break
        else:
            years[i] = 8
    df = pd.DataFrame({"draft_round": skill, "dev_years_observed": years, "made_nhl": event})
    m = DevelopmentModel(["draft_round"]).fit(df)
    curve = m.debut_curve(df)
    assert np.all(np.diff(curve, axis=1) >= 0)
    assert concordance_index(years, event, curve[:, 3]) > 0.7


def test_concordance_index_perfect_and_reversed():
    t, e = np.array([1, 2, 3, 4]), np.array([1, 1, 1, 0])
    assert concordance_index(t, e, -t) == 1.0
    assert concordance_index(t, e, t) == 0.0
