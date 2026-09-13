"""Semi-synthetic censoring: real covariates, real outcomes, a known decision policy.

The evaluation problem in selective labels is that on real data the censored
outcomes are, by definition, missing -- so the deployment risk you want to
report cannot be computed.  The standard remedy (Lakkaraju et al., 2017;
De-Arteaga et al., 2018) is to start from a dataset with *complete* outcomes and
impose a censoring mechanism yourself.

We do that, but we keep the confounding **real**: the decision maker's private
information ``S`` is an actual held-out variable of the dataset that genuinely
predicts the outcome -- for Lending Club, the underwriter's assigned interest
rate, which encodes risk assessment not present in the raw applicant features.
So the induced ``Gamma_0`` comes from real covariate/outcome structure rather
than from a hand-chosen number, and the ground truth ``Y`` needed to evaluate
deployment risk is real and available for every unit.

Oracle nuisances.  ``S`` is residualised on ``X`` and standardised so that
``S ind. X`` holds approximately; the outcome model ``q(x, s) = P(Y=1|X=x,S=s)``
is fit on the *complete* data.  Population quantities then follow by the same
Gauss-Hermite quadrature over ``S`` used in :mod:`dcl.data.synthetic`, which is
what lets us calibrate ``Gamma_0`` to a target and report it exactly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..sensitivity import expit, logit
from .base import SelectiveLabelsDataset
from .synthetic import _gauss_hermite

__all__ = ["residualise", "censor_with_hidden_signal"]


def residualise(S: np.ndarray, X: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Remove the linear part of ``S`` explained by ``X``; return a z-score.

    Makes the "``S`` independent of ``X``" idealisation approximately true, which
    is what licenses integrating ``S`` out by quadrature.  Any residual
    dependence only makes the reported ``Gamma_0`` an approximation, never
    invalidates the evaluation (``Y`` is real and observed for all units).
    """
    S = np.asarray(S, float).ravel()
    X = np.asarray(X, float)
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ S)
    r = S - A @ coef
    return (r - r.mean()) / (r.std() + 1e-12)


def _fit_outcome_model(X: np.ndarray, S: np.ndarray, Y: np.ndarray, seed: int):
    """``q(x, s) = P(Y = 1 | X = x, S = s)`` on the complete data (gradient boosting)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV

    Z = np.column_stack([X, S])
    base = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06,
                                          max_leaf_nodes=31, random_state=seed)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(Z, Y)

    def q(Xg: np.ndarray, s_val: float) -> np.ndarray:
        return clf.predict_proba(np.column_stack([Xg, np.full(Xg.shape[0], s_val)]))[:, 1]

    return q


def censor_with_hidden_signal(
    X: np.ndarray,
    S_raw: np.ndarray,
    Y: np.ndarray,
    feature_names,
    name: str,
    target_gamma: float = 2.0,
    selection_rate: float = 0.5,
    n_judges: int = 5,
    leniency_spread: float = 0.8,
    kappa_heterogeneity: float = 0.25,
    seed: int = 0,
    align_sign: bool = True,
    notes: str = "",
    calib_n: int = 4000,
) -> SelectiveLabelsDataset:
    """Impose a ``Gamma_0``-calibrated censoring policy driven by the hidden ``S``.

    The policy is ``T | X, S, Z ~ Bern(sigmoid(a0 + c_Z + alpha'X + kappa_Z S))``
    where ``alpha`` is fit so the *observable* part of the policy mimics a
    sensible risk-based rule, and ``kappa`` is bisected so that the realised
    ``Gamma_0`` matches ``target_gamma``.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n = X.shape[0]
    rng = np.random.default_rng(seed)

    S = residualise(S_raw, X)
    if align_sign:
        # orient S so that larger S means higher outcome risk
        if np.corrcoef(S, Y)[0, 1] < 0:
            S = -S

    q = _fit_outcome_model(X, S, Y, seed)

    # observable part of the policy: deny the applicants a simple risk model flags
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(X, Y)
    risk = lr.decision_function(X)
    risk = (risk - risk.mean()) / (risk.std() + 1e-12)
    core_t_obs = -risk                     # higher risk -> less likely selected

    Zi = rng.integers(0, n_judges, size=n)
    leniency = rng.normal(scale=leniency_spread, size=n_judges)
    leniency -= leniency.mean()
    kappa_mult = 1.0 + kappa_heterogeneity * rng.normal(size=n_judges)

    s_nodes, s_w = _gauss_hermite()
    # Evaluate the outcome model on the quadrature grid ONCE, then do the whole
    # Gamma_0 calibration on the full sample: calibrating on a subsample makes
    # the realised supremum drift upward, because the sup is taken over more
    # units than the probe contains.
    q_full = np.stack([q(X, sv) for sv in s_nodes], axis=1)      # (n, K)
    core_t = core_t_obs + leniency[Zi]

    def _solve_a0(kap_i, iters=40):
        lo, hi = -14.0, 14.0
        for _ in range(iters):
            a0 = 0.5 * (lo + hi)
            st = expit(core_t[:, None] + a0 - kap_i[:, None] * s_nodes[None, :])
            if (st @ s_w).mean() < selection_rate:
                lo = a0
            else:
                hi = a0
        return 0.5 * (lo + hi)

    def _nuisances(kap_i, a0):
        """Z-marginal oracle nuisances: a learner sees only X, so the relevant
        propensity averages over the random assignment of cases to decision
        makers rather than conditioning on the realised one."""
        p_true = q_full @ s_w
        es, j1s = [], []
        for j in range(n_judges):
            kj = (kap_i / np.clip(kappa_mult[Zi], 1e-12, None)) * kappa_mult[j]
            stj = expit((core_t_obs + leniency[j] + a0)[:, None]
                        - kj[:, None] * s_nodes[None, :])
            ej = stj @ s_w
            es.append(ej); j1s.append((q_full * stj) @ s_w)
        e = np.mean(np.stack(es, 0), 0)
        j1 = np.mean(np.stack(j1s, 0), 0)
        p1 = j1 / np.clip(e, 1e-12, None)
        p0 = (p_true - j1) / np.clip(1 - e, 1e-12, None)
        return e, p1, p0

    def realise(kappa):
        kap_i = kappa * kappa_mult[Zi]
        a0 = _solve_a0(kap_i, iters=30)
        _, p1, p0 = _nuisances(kap_i, a0)
        return float(np.exp(np.max(np.abs(logit(p0) - logit(p1))))), a0

    if target_gamma <= 1.0 + 1e-9:
        kappa = 0.0
    else:
        lo_k, hi_k = 0.0, 1.0
        while realise(hi_k)[0] < target_gamma and hi_k < 64:
            hi_k *= 2.0
        for _ in range(34):
            mid = 0.5 * (lo_k + hi_k)
            if realise(mid)[0] < target_gamma:
                lo_k = mid
            else:
                hi_k = mid
        kappa = 0.5 * (lo_k + hi_k)

    kap_i = kappa * kappa_mult[Zi]
    a0 = _solve_a0(kap_i, iters=60)

    T = (rng.uniform(size=n) < expit(core_t + a0 - kap_i * S)).astype(float)
    Y_obs = np.where(T == 1, Y, np.nan)

    e, p1, p0 = _nuisances(kap_i, a0)
    p = q_full @ s_w
    lor = np.abs(logit(p0) - logit(p1))

    p1_by_z, e_by_z, lor_z = [], [], []
    for j in range(n_judges):
        stj = expit((core_t_obs + leniency[j] + a0)[:, None]
                    - (kappa * kappa_mult[j]) * s_nodes[None, :])
        ej = stj @ s_w
        p1j = ((q_full * stj) @ s_w) / np.clip(ej, 1e-12, None)
        p0j = (p - ej * p1j) / np.clip(1.0 - ej, 1e-12, None)
        p1_by_z.append(p1j); e_by_z.append(ej)
        lor_z.append(np.abs(logit(np.clip(p0j, 1e-12, 1 - 1e-12)) - logit(p1j)))
    # conditional (within decision maker) sensitivity parameter -- see the note
    # in dcl/data/synthetic.py; this is the quantity Proposition 9 bounds below.
    gamma0_cond = float(np.exp(np.max(np.stack(lor_z, 0))))

    return SelectiveLabelsDataset(
        X=X, T=T, Y_obs=Y_obs, Y_full=Y, Z=Zi,
        feature_names=list(feature_names), name=name,
        oracle=dict(p=p, e=e, p1=p1, p0=p0,
                    gamma0=float(np.exp(np.max(lor))),
                    gamma0_q99=float(np.exp(np.quantile(lor, 0.99))),
                    gamma0_cond=gamma0_cond,
                    kappa=kappa, S=S, p1_by_z=p1_by_z, e_by_z=e_by_z,
                    leniency=leniency),
        notes=notes or "Semi-synthetic censoring on real covariates and real outcomes.",
    )
