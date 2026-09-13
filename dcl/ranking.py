"""Worst-case *ranking* learning: where decision censoring is genuinely hard.

The DCL theorem shows that robust *risk* minimisation under DCSM(Gamma) is an exactly
convex program.  Robust *ranking* is not.  Maximising the worst-case AUROC

    max_{f in F}  min_{P in I_Gamma(P_obs)}  AUC_P(f)                        (5)

is a max-min over a fractional objective: by identity (1) of
:mod:`dcl.auc_bounds`, AUROC is linear in ``p`` only *after* conditioning on the
prevalence ``pi``, and the prevalence is itself chosen by the adversary.  The
inner minimisation is therefore not concave in ``p`` and does not decouple.

What saves us is that the sharp-interval theorem gives an **exact,
O(n log n) oracle** for the
inner problem.  We can therefore run a Danskin / best-response scheme:

    1. adversary:  p_t  <- argmin_{p in box} AUC(f_t, p)      (exact)
    2. learner:    f_{t+1} <- gradient step on a smooth pairwise surrogate of
                   AUC(f, p_t), with pair weights p_t(x)(1 - p_t(x')).

Because step 1 is solved exactly, the gradient in step 2 is a valid (super)
gradient of the inner minimum wherever the argmin is unique (Danskin), and the
iterates converge to a stationary point of (5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .auc_bounds import sharp_auc_interval

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
        How often to re-solve the exact inner problem.
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
    best_worst_case_auc_: float = field(default=float('nan'), init=False)
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

        # Warm start at the plug-in midpoint ranking.
        mid = 0.5 * (lo + hi)
        XtX = X.T @ X + 1e-3 * np.eye(d)
        self.coef_ = np.linalg.solve(XtX, X.T @ (mid - mid.mean()))
        nrm = np.linalg.norm(self.coef_)
        if nrm > 0:
            self.coef_ /= nrm

        # Best-responding to the *latest* adversary oscillates: the exact inner
        # argmin inverts the soft labels relative to the current ranking, the
        # learner flips, and the adversary flips back.  We therefore (i) do
        # fictitious play -- gradient steps against the running AVERAGE of the
        # adversary's responses -- and (ii) keep the best iterate measured by the
        # exact inner value, which the sharp-interval theorem lets us evaluate in
        # O(n log n).  So
        # the returned model can never be worse than the warm start.
        p_bar = mid.copy()
        n_resp = 0
        best_coef = self.coef_.copy()
        best_val = sharp_auc_interval(self._scores(X), lo, hi, w).lower
        self.history_.append(best_val)

        for t in range(self.n_steps):
            if t % self.oracle_every == 0:
                p_new = sharp_auc_interval(self._scores(X), lo, hi, w).p_lower
                n_resp += 1
                p_bar += (p_new - p_bar) / n_resp

            i = rng.integers(0, n, self.n_pairs)
            j = rng.integers(0, n, self.n_pairs)
            zi, zj = X[i] @ self.coef_, X[j] @ self.coef_
            dz = (zi - zj) / self.tau
            s = 1.0 / (1.0 + np.exp(-np.clip(dz, -50, 50)))
            pw = p_bar[i] * (1.0 - p_bar[j])          # P(Y_i = 1, Y_j = 0)
            g = (pw * s * (1.0 - s) / self.tau)[:, None] * (X[i] - X[j])
            grad = g.mean(axis=0) - self.l2 * self.coef_
            lr = self.lr / (1.0 + t / max(self.n_steps / 8.0, 1.0))
            self.coef_ = self.coef_ + lr * grad / (np.linalg.norm(grad) + 1e-12)
            nrm = np.linalg.norm(self.coef_)
            if nrm > 0:
                self.coef_ /= nrm

            if (t + 1) % self.oracle_every == 0:
                val = sharp_auc_interval(self._scores(X), lo, hi, w).lower
                self.history_.append(val)
                if val > best_val:
                    best_val, best_coef = val, self.coef_.copy()

        self.coef_ = best_coef
        self.best_worst_case_auc_ = best_val
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) @ self.coef_
