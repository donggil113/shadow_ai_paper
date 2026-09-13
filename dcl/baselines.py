"""Baselines: what the field does today under selective labels.

* ``ERMObserved``    -- fit on the labelled (selected) units.  The default in
  practice, and the thing that silently optimises the wrong risk.
* ``IPWLearner``     -- inverse-propensity weighting.  Valid *only* under
  selection on observables, i.e. DCSM(Gamma = 1).
* ``AIPWLearner``    -- augmented / doubly-robust weighting.  Same identifying
  assumption, better variance.
* ``ImputationLearner`` -- impute censored labels with ``p1_hat`` and fit on
  everything (the "pseudo-label the rejects" heuristic).  Also MAR.
* ``HeckmanProbit``  -- bivariate probit with partial observability (Greene):
  a parametric selection model that *does* allow selection on unobservables, but
  buys identification with a joint-normality assumption and, optionally, an
  exclusion restriction.  The classical econometric answer to this problem.
* ``ManskiLearner``  -- DCL with ``Gamma = inf``: assumption-free worst case.
* ``OracleLearner``  -- fits on the uncensored ground truth.  Not attainable;
  it marks the ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .nuisance import NuisanceEstimates
from .objectives import dcl_bayes_score
from .sensitivity import expit, identified_set, logit

__all__ = ["ERMObserved", "IPWLearner", "AIPWLearner", "ImputationLearner",
           "HeckmanProbit", "ManskiLearner", "OracleLearner", "bvn_cdf"]


def _clf(model: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if model == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.07,
                                              min_samples_leaf=30,
                                              l2_regularization=1.0,
                                              random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


@dataclass
class ERMObserved:
    """Standard practice: train on the units whose labels the policy revealed."""

    model: str = "gbm"
    seed: int = 0

    def fit(self, X, T, Y_obs, **kw):
        X = np.asarray(X, float); T = np.asarray(T, float)
        sel = T == 1
        self.clf_ = _clf(self.model, self.seed).fit(X[sel],
                                                    np.asarray(Y_obs, float)[sel])
        return self

    def decision_function(self, X):
        return logit(self.clf_.predict_proba(np.asarray(X, float))[:, 1])


@dataclass
class IPWLearner:
    """Weight the selected units by ``1 / e_hat(x)``; valid iff ``Gamma = 1``."""

    model: str = "gbm"
    seed: int = 0
    clip: float = 0.02

    def fit(self, X, T, Y_obs, nuisance: NuisanceEstimates = None, **kw):
        X = np.asarray(X, float); T = np.asarray(T, float)
        sel = T == 1
        w = 1.0 / np.clip(nuisance.e[sel], self.clip, 1.0)
        self.clf_ = _clf(self.model, self.seed).fit(
            X[sel], np.asarray(Y_obs, float)[sel], sample_weight=w)
        return self

    def decision_function(self, X):
        return logit(self.clf_.predict_proba(np.asarray(X, float))[:, 1])


@dataclass
class AIPWLearner:
    """Doubly-robust pseudo-outcome regression; still assumes ``Gamma = 1``."""

    model: str = "gbm"
    seed: int = 0
    clip: float = 0.02

    def fit(self, X, T, Y_obs, nuisance: NuisanceEstimates = None, **kw):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = np.asarray(X, float); T = np.asarray(T, float)
        Y = np.nan_to_num(np.asarray(Y_obs, float))
        e = np.clip(nuisance.e, self.clip, 1.0)
        psi = nuisance.p1 + T * (Y - nuisance.p1) / e     # AIPW pseudo-outcome
        self.reg_ = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.07, min_samples_leaf=30,
            random_state=self.seed).fit(X, psi)
        return self

    def decision_function(self, X):
        return logit(np.clip(self.reg_.predict(np.asarray(X, float)), 1e-4, 1 - 1e-4))


@dataclass
class ImputationLearner:
    """Pseudo-label the censored units with ``p1_hat`` and fit on everything."""

    model: str = "gbm"
    seed: int = 0

    def fit(self, X, T, Y_obs, nuisance: NuisanceEstimates = None, **kw):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = np.asarray(X, float); T = np.asarray(T, float)
        target = np.where(T == 1, np.nan_to_num(np.asarray(Y_obs, float)),
                          nuisance.p1)
        self.reg_ = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.07, min_samples_leaf=30,
            random_state=self.seed).fit(X, target)
        return self

    def decision_function(self, X):
        return logit(np.clip(self.reg_.predict(np.asarray(X, float)), 1e-4, 1 - 1e-4))


@dataclass
class ManskiLearner:
    """DCL at ``Gamma = inf``: the assumption-free worst case."""

    seed: int = 0

    def fit(self, X, T, Y_obs, nuisance: NuisanceEstimates = None, **kw):
        self.box_ = identified_set(nuisance.p1, nuisance.e, np.inf)
        self.train_scores_ = dcl_bayes_score(self.box_.lo, self.box_.hi)
        self._nu = nuisance
        return self

    def decision_function(self, X=None):
        return self.train_scores_


@dataclass
class OracleLearner:
    """Ceiling: fits on the ground-truth labels of *all* units."""

    model: str = "gbm"
    seed: int = 0

    def fit(self, X, T, Y_obs, Y_full=None, **kw):
        if Y_full is None:
            raise ValueError("OracleLearner needs Y_full (evaluation only)")
        self.clf_ = _clf(self.model, self.seed).fit(np.asarray(X, float),
                                                    np.asarray(Y_full, float))
        return self

    def decision_function(self, X):
        return logit(self.clf_.predict_proba(np.asarray(X, float))[:, 1])


# --------------------------------------------------------------------------- #
# Bivariate probit with partial observability (Heckman / Greene)
# --------------------------------------------------------------------------- #
def bvn_cdf(a: np.ndarray, b: np.ndarray, rho: float, n_nodes: int = 24) -> np.ndarray:
    """Standard bivariate normal CDF via Plackett's integral over the correlation.

    ``Phi2(a, b, rho) = Phi(a) Phi(b) + int_0^rho phi2(a, b, r) dr``, evaluated by
    Gauss-Legendre quadrature.  Vectorised over ``a``, ``b``; accurate to ~3e-12
    for ``|rho| <= 0.99`` at the default 24 nodes.
    """
    from scipy.stats import norm

    a = np.asarray(a, float); b = np.asarray(b, float)
    base = norm.cdf(a) * norm.cdf(b)
    if abs(rho) < 1e-12:
        return base
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    r = 0.5 * rho * (x + 1.0)                      # map [-1,1] -> [0, rho]
    jac = 0.5 * rho
    om = np.clip(1.0 - r**2, 1e-12, None)
    dens = (1.0 / (2 * np.pi * np.sqrt(om)))[None, :] * np.exp(
        -(a[:, None]**2 - 2 * r[None, :] * a[:, None] * b[:, None] + b[:, None]**2)
        / (2 * om[None, :]))
    return base + jac * (dens @ w)


@dataclass
class HeckmanProbit:
    """Bivariate probit with partial observability.

        T* = w'g + u,   T = 1{T* > 0}
        Y* = x'b + v,   Y = 1{Y* > 0},  observed only when T = 1
        (u, v) ~ BVN(0, 0, 1, 1, rho)

    ``rho != 0`` is exactly selection on unobservables, so unlike IPW/AIPW this
    baseline is *not* restricted to ``Gamma = 1``.  Its identification comes from
    joint normality (plus an exclusion restriction if ``Z`` is supplied), which is
    an assumption of a completely different kind from DCSM(Gamma): untestable and
    not indexed by an interpretable strength parameter.  That contrast is the
    point of including it.
    """

    max_iter: int = 150
    seed: int = 0
    ridge: float = 1e-3
    n_restarts: int = 2
    max_fit_n: int = 8000

    def fit(self, X, T, Y_obs, Z=None, **kw):
        from scipy.optimize import minimize
        from scipy.stats import norm

        X = np.asarray(X, float); T = np.asarray(T, float)
        Y = np.nan_to_num(np.asarray(Y_obs, float))
        # The likelihood needs a bivariate-normal CDF at every unit and is
        # optimised with numerical gradients, so cost is linear in n with a
        # large constant.  Subsample for the fit; the MLE is root-n consistent
        # and 8k units is far more than this 20-odd parameter model needs.
        if X.shape[0] > self.max_fit_n:
            sub = np.random.default_rng(self.seed).choice(
                X.shape[0], self.max_fit_n, replace=False)
            X, T, Y = X[sub], T[sub], Y[sub]
            if Z is not None:
                Z = np.asarray(Z, float).reshape(-1, 1)[sub]
        n, d = X.shape
        Xo = np.hstack([np.ones((n, 1)), X])
        W = Xo if Z is None else np.hstack([Xo, np.asarray(Z, float).reshape(n, -1)])
        dw = W.shape[1]

        def nll(theta):
            g, b = theta[:dw], theta[dw:dw + d + 1]
            rho = np.tanh(theta[-1])
            wg, xb = W @ g, Xo @ b
            s11 = bvn_cdf(wg, xb, rho)
            s10 = bvn_cdf(wg, -xb, -rho)
            s0 = norm.cdf(-wg)
            eps = 1e-10
            ll = (T * Y * np.log(np.clip(s11, eps, None))
                  + T * (1 - Y) * np.log(np.clip(s10, eps, None))
                  + (1 - T) * np.log(np.clip(s0, eps, None)))
            pen = self.ridge * (np.sum(g[1:]**2) + np.sum(b[1:]**2))
            return -ll.mean() + pen

        # Warm start from the two separate (mis-specified but cheap) probits:
        # a good initial point cuts L-BFGS iterations several-fold.
        from sklearn.linear_model import LogisticRegression
        g0 = np.zeros(dw); b0 = np.zeros(d + 1)
        try:
            lt = LogisticRegression(max_iter=2000, C=1.0).fit(W[:, 1:], T)
            g0 = np.concatenate([[lt.intercept_[0]], lt.coef_[0]]) / 1.6
            sel = T == 1
            if len(np.unique(Y[sel])) > 1:
                ly = LogisticRegression(max_iter=2000, C=1.0).fit(X[sel], Y[sel])
                b0 = np.concatenate([[ly.intercept_[0]], ly.coef_[0]]) / 1.6
        except Exception:                                  # pragma: no cover
            pass

        rng = np.random.default_rng(self.seed)
        starts = [np.concatenate([g0, b0, [0.0]])]
        for _ in range(max(0, self.n_restarts - 1)):
            starts.append(np.concatenate([g0 + rng.normal(0, .05, dw),
                                          b0 + rng.normal(0, .05, d + 1),
                                          [rng.normal(0, .3)]]))
        best, best_val = starts[0], np.inf
        for x0 in starts:
            res = minimize(nll, x0, method="L-BFGS-B",
                           options=dict(maxiter=self.max_iter))
            if res.fun < best_val:
                best, best_val = res.x, res.fun
        self.theta_ = best
        self.rho_ = float(np.tanh(best[-1]))
        self.beta_ = best[dw:dw + d + 1]
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        Xo = np.hstack([np.ones((X.shape[0], 1)), X])
        from scipy.stats import norm
        return logit(np.clip(norm.cdf(Xo @ self.beta_), 1e-6, 1 - 1e-6))
