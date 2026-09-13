"""Worst-case *ranking* learning: where decision censoring is genuinely hard.

Theorem 2 shows that robust *risk* minimisation under DCSM(Gamma) is an exactly
convex program.  Robust *ranking* is not.  Maximising the worst-case AUROC

    max_{f in F}  min_{P in I_Gamma(P_obs)}  AUC_P(f)                        (5)

is a max-min over a fractional objective: by identity (1) of
:mod:`dcl.auc_bounds`, AUROC is linear in ``p`` only *after* conditioning on the
prevalence ``pi``, and the prevalence is itself chosen by the adversary.  The
inner minimisation is therefore not concave in ``p`` and does not decouple.

What saves us is that Theorem 1 gives an **exact, O(n log n) oracle** for the
inner problem.  We can therefore run a Danskin / best-response scheme:

    1. adversary:  p_t  <- argmin_{p in box} AUC(f_t, p)      (Theorem 1, exact)
    2. learner:    f_{t+1} <- gradient step on a smooth pairwise surrogate of
                   AUC(f, p_t), with pair weights p_t(x)(1 - p_t(x')).

Because step 1 is solved exactly, the gradient in step 2 is a valid (super)
gradient of the inner minimum wherever the argmin is unique (Danskin), and the
iterates converge to a stationary point of (5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .auc_bounds import auc_from_regression, midrank, sharp_auc_interval

__all__ = ["DCLRanker", "soft_auc", "worst_case_auc"]


def worst_case_auc(scores, lo, hi, weights=None) -> float:
    """``min_{p in box} AUC(f, p)`` -- the exact inner value of (5)."""
    return sharp_auc_interval(scores, lo, hi, weights).lower


def soft_auc(scores: np.ndarray, p: np.ndarray, tau: float = 1.0,
             weights: np.ndarray | None = None) -> float:
    """Sigmoid-smoothed pairwise AUROC with soft labels ``p`` (for diagnostics)."""
    z = np.asarray(scores, float)
    p = np.asarray(p, float)
    n = z.shape[0]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, float) / np.sum(weights)
    d = (z[:, None] - z[None, :]) / tau
    s = 1.0 / (1.0 + np.exp(-np.clip(d, -50, 50)))
    wp = w * p
    wn = w * (1.0 - p)
    num = float(wp @ s @ wn)
    pi = float(w @ p)
    return num / max(pi * (1.0 - pi), 1e-12)


@dataclass
class DCLRanker:
    """Worst-case-AUROC linear scorer trained by the Danskin scheme above.

    Parameters
    ----------
    tau : float
        Temperature of the pairwise sigmoid surrogate.
    n_pairs : int
        Pairs sampled per step (the exact pairwise objective is ``O(n^2)``).
    n_steps, lr : int, float
        Adam-free projected gradient ascent schedule.
    oracle_every : int
        How often to re-solve the exact inner problem (Theorem 1).
    l2 : float
        Ridge penalty, also fixing the scale invariance of AUROC.
    """

    tau: float = 0.5
    n_pairs: int = 4096
    n_steps: int = 400
    lr: float = 0.2
    oracle_every: int = 10
    l2: float = 1e-3
    seed: int = 0
    coef_: np.ndarray | None = field(default=None, init=False)
    history_: list = field(default_factory=list, init=False)

    def _scores(self, X):
        return X @ self.coef_

    def fit(self, X: np.ndarray, lo: np.ndarray, hi: np.ndarray,
            weights: np.ndarray | None = None) -> "DCLRanker":
        X = np.asarray(X, float)
        lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, float) / np.sum(weights)

        # initialise at the plug-in midpoint ranking (a strong warm start)
        mid = 0.5 * (lo + hi)
        XtX = X.T @ X + 1e-3 * np.eye(d)
        self.coef_ = np.linalg.solve(XtX, X.T @ (mid - mid.mean()))
        nrm = np.linalg.norm(self.coef_)
        if nrm > 0:
            self.coef_ /= nrm

        p_adv = mid.copy()
        for t in range(self.n_steps):
            if t % self.oracle_every == 0:
                res = sharp_auc_interval(self._scores(X), lo, hi, w)
                p_adv = res.p_lower                      # exact inner argmin
                self.history_.append(res.lower)

            i = rng.integers(0, n, self.n_pairs)
            j = rng.integers(0, n, self.n_pairs)
            zi, zj = X[i] @ self.coef_, X[j] @ self.coef_
            dz = (zi - zj) / self.tau
            s = 1.0 / (1.0 + np.exp(-np.clip(dz, -50, 50)))
            # weight of an ordered pair: P(Y_i = 1, Y_j = 0) under p_adv
            pw = p_adv[i] * (1.0 - p_adv[j])
            g_coef = (pw * s * (1.0 - s) / self.tau)[:, None] * (X[i] - X[j])
            grad = g_coef.mean(axis=0) - self.l2 * self.coef_
            self.coef_ = self.coef_ + self.lr * grad / (np.linalg.norm(grad) + 1e-12)
            nrm = np.linalg.norm(self.coef_)
            if nrm > 0:
                self.coef_ /= nrm
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) @ self.coef_
