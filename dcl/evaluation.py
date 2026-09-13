"""Evaluation: the quantity you can measure vs. the quantity you care about."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .auc_bounds import sharp_auc_interval
from .objectives import get_loss, worstcase_risk
from .sensitivity import expit

__all__ = ["deployment_metrics", "observed_metrics", "bounds_report",
           "contraction_failure_rate"]


def _auc(scores, y):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def deployment_metrics(scores: np.ndarray, y_full: np.ndarray,
                       loss: str = "logistic") -> Dict[str, float]:
    """Ground truth on **all** units -- the target, available only in benchmarks."""
    L = get_loss(loss)
    scores = np.asarray(scores, float); y = np.asarray(y_full, float)
    pos, neg = y == 1, y == 0
    bal = float("nan")
    if pos.any() and neg.any():
        bal = float(0.5 * (np.mean(scores[pos] > 0) + np.mean(scores[neg] <= 0)))
    return {
        "risk": float(np.mean(L(scores, y))),
        "auroc": _auc(scores, y),
        "accuracy": float(np.mean((scores > 0).astype(float) == y)),
        "balanced_accuracy": bal,
        "brier": float(np.mean((expit(scores) - y) ** 2)),
    }


def observed_metrics(scores: np.ndarray, T: np.ndarray, Y_obs: np.ndarray,
                     loss: str = "logistic") -> Dict[str, float]:
    """What a practitioner reports: metrics on the labelled sub-population."""
    sel = np.asarray(T, float) == 1
    return {f"obs_{k}": v for k, v in
            deployment_metrics(np.asarray(scores)[sel],
                               np.asarray(Y_obs, float)[sel], loss).items()}


def bounds_report(scores: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  loss: str = "logistic") -> Dict[str, float]:
    """What DCL reports: sharp AUROC interval and worst-case risk."""
    res = sharp_auc_interval(scores, lo, hi)
    return {
        "auroc_lo": res.lower, "auroc_hi": res.upper,
        "auroc_width": res.width,
        "worstcase_risk": worstcase_risk(scores, lo, hi, loss),
    }


def contraction_failure_rate(scores: np.ndarray, T: np.ndarray, Y_obs: np.ndarray,
                             Z: np.ndarray, acceptance_rate: float) -> float:
    """Lakkaraju et al. (2017) contraction estimate of a model's failure rate.

    Uses the most lenient decision maker's caseload as a quasi-random sample,
    then "contracts" it to the target acceptance rate by removing the units the
    model scores worst.  Requires (i) a leniency instrument, (ii) that the most
    lenient decision maker accepts at least ``acceptance_rate``, and (iii) random
    case assignment.  It estimates a *point*, not a bound, and it is silent about
    what happens outside the most lenient decision maker's accepted set -- which
    is exactly the region DCL bounds.
    """
    scores = np.asarray(scores, float); T = np.asarray(T, float)
    Y = np.asarray(Y_obs, float); Z = np.asarray(Z)
    groups = np.unique(Z)
    rates = {g: float(np.mean(T[Z == g])) for g in groups}
    most = max(rates, key=rates.get)
    idx = np.flatnonzero((Z == most) & (T == 1))
    if idx.size == 0:
        return float("nan")
    order = idx[np.argsort(-scores[idx])]          # worst-scoring removed first
    n_keep = int(round(acceptance_rate * np.sum(Z == most)))
    n_keep = max(0, min(n_keep, order.size))
    kept = order[order.size - n_keep:] if n_keep else order[:0]
    if kept.size == 0:
        return float("nan")
    return float(np.mean(Y[kept]))
