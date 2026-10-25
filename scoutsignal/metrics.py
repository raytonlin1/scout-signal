"""Probabilistic forecast metrics used across the models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error over equal-count bins."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return err / len(p)


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    order = np.argsort(p)
    return [{"predicted": float(p[c].mean()), "observed": float(y[c].mean()), "n": int(len(c))}
            for c in np.array_split(order, bins) if len(c)]


def binary_report(y, p) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "log_loss": round(float(log_loss(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
        "ece": round(float(ece(y, p)), 4),
        # Calibration-in-the-large: observed goals / expected goals. Drifts from 1 when the
        # league changes under a frozen model.
        "obs_over_exp": round(float(y.sum() / p.sum()), 3),
    }


def concordance_index(time, event, risk) -> float:
    """Harrell's C: share of comparable pairs where the higher-risk subject fails first."""
    time, event, risk = map(np.asarray, (time, event, risk))
    num = den = 0.0
    for i in np.flatnonzero(event == 1):
        later = time > time[i]
        den += later.sum()
        num += (risk[later] < risk[i]).sum() + 0.5 * (risk[later] == risk[i]).sum()
    return float(num / den) if den else float("nan")
