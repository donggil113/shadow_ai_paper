"""SL-Bench: a synthetic selective-labels benchmark with exact oracle quantities.

Generative model
----------------
    X ~ N(0, I_d)                      covariates the learner sees
    S ~ N(0, 1)                        the decision maker's *private* information
    Z ~ Uniform{1..K}                  decision-maker identity (leniency instrument)
    Y | X, S      ~ Bern( sigmoid(b0 + beta'X + gamma_S * S) )
    T | X, S, Z   ~ Bern( sigmoid(a0 + c_Z + alpha'X + kappa_Z * S) )
    Y observed iff T = 1.

``S`` enters both the outcome and the decision, which is exactly what makes the
problem *not* missing-at-random.  ``Z`` shifts only the decision (exclusion
restriction ``Z ind. (Y, S) | X`` holds by construction), so the falsification
test of :mod:`dcl.falsify` is applicable and can be checked against the truth.

Every population quantity -- ``p(x)``, ``e_z(x)``, ``p1_z(x)``, ``p0_z(x)`` and the
realised sensitivity parameter ``Gamma_0`` -- is computed exactly by Gauss-Hermite
quadrature over ``S``, so experiments can separate *identification* error from
*estimation* error.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..sensitivity import expit, logit
from .base import SelectiveLabelsDataset

__all__ = ["make_sl_bench", "oracle_quantities"]

_GH_DEG = 48


def _gauss_hermite(deg: int = _GH_DEG) -> Tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for ``E_{S ~ N(0,1)}[h(S)] = sum_k w_k h(x_k)``."""
    x, w = np.polynomial.hermite_e.hermegauss(deg)
    return x, w / np.sum(w)


def oracle_quantities(
    lin_y: np.ndarray, lin_t: np.ndarray, gamma_s: float, kappa_s: np.ndarray,
    deg: int = _GH_DEG,
):
    """Exact ``p``, ``e``, ``p1``, ``p0`` by quadrature over the latent ``S``.

    ``lin_y`` / ``lin_t`` are the ``S``-free parts of the two linear predictors,
    each of shape ``(n,)``; ``kappa_s`` is per-unit (it varies with the assigned
    decision maker).
    """
    s, w = _gauss_hermite(deg)
    sy = expit(lin_y[:, None] + gamma_s * s[None, :])          # (n, K)
    st = expit(lin_t[:, None] + kappa_s[:, None] * s[None, :])  # (n, K)
    p = sy @ w
    e = st @ w
    num1 = (sy * st) @ w
    num0 = (sy * (1.0 - st)) @ w
    p1 = num1 / np.clip(e, 1e-12, None)
    p0 = num0 / np.clip(1.0 - e, 1e-12, None)
    return p, e, p1, p0


def _solve_intercept(core, coef_s, target, kappa_i=None, is_outcome=True, iters=28):
    """Bisect an intercept so that the marginal rate hits ``target``."""
    n = core.shape[0]
    zeros = np.zeros(n)
    lo, hi = -14.0, 14.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if is_outcome:
            rate = oracle_quantities(core + mid, zeros, coef_s, zeros)[0].mean()
        else:
            rate = oracle_quantities(zeros, core + mid, 0.0, kappa_i)[1].mean()
        if rate < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_sl_bench(
    n: int = 20000,
    d: int = 10,
    target_gamma: float | None = 3.0,
    confounding_scale: float | None = None,
    n_judges: int = 5,
    leniency_spread: float = 1.0,
    kappa_heterogeneity: float = 0.0,
    selection_rate: float = 0.5,
    prevalence: float = 0.25,
    signal: float = 1.0,
    seed: int = 0,
    quantile: float = 1.0,
    calib_n: int = 2000,
) -> SelectiveLabelsDataset:
    """Generate an SL-Bench instance with an accurately calibrated ``Gamma_0``.

    Confounding strength is a single scalar ``c`` scaling the latent coefficient
    in *both* equations (``gamma_S = c`` in the outcome, ``kappa = c`` in the
    decision).  Scaling only one of the two saturates: if the decision barely
    uses ``S`` no amount of outcome dependence can move the censored group.
    ``c`` is bisected **on the instance itself**, so the realised ``Gamma_0``
    matches ``target_gamma`` to three decimals rather than drifting.

    Parameters
    ----------
    target_gamma
        Desired realised ``Gamma_0 = sup_x |logit p0(x) - logit p1(x)|`` (or the
        ``quantile``-th percentile when ``quantile < 1``).
    confounding_scale
        Set ``c`` directly and skip calibration.
    kappa_heterogeneity
        Cross-decision-maker variation in reliance on ``S``.  Zero makes the
        leniency instrument maximally informative (``Gamma_min = Gamma_0``);
        positive values make ``Gamma_min < Gamma_0``, the realistic case.
    quantile
        ``1.0`` (default) uses the supremum, so the identified set provably
        covers the truth for *every* unit.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    beta = signal * rng.normal(scale=1.0 / np.sqrt(d), size=d)
    alpha = rng.normal(scale=1.0 / np.sqrt(d), size=d)
    Z = rng.integers(0, n_judges, size=n)
    leniency = rng.normal(scale=leniency_spread, size=n_judges)
    leniency -= leniency.mean()
    kappa_mult = 1.0 + kappa_heterogeneity * rng.normal(size=n_judges)

    core_y_all = X @ beta
    core_t_all = X @ alpha + leniency[Z]
    probe = rng.choice(n, size=min(calib_n, n), replace=False)

    def realise(c, idx):
        """Realised (Z-marginal) Gamma_0 at confounding scale ``c`` on ``idx``."""
        cy, ct = core_y_all[idx], core_t_all[idx]
        kap = c * kappa_mult[Z[idx]]
        b0 = _solve_intercept(cy, c, prevalence, is_outcome=True)
        a0 = _solve_intercept(ct, None, selection_rate, kappa_i=kap, is_outcome=False)
        ly = cy + b0
        es, j1s = [], []
        for j in range(n_judges):
            lt = X[idx] @ alpha + leniency[j] + a0
            _, e_j, p1_j, _ = oracle_quantities(ly, lt, c, np.full(len(idx), c * kappa_mult[j]))
            es.append(e_j); j1s.append(e_j * p1_j)
        pp = oracle_quantities(ly, ct + a0, c, kap)[0]
        e_m = np.mean(np.stack(es, 0), 0)
        j1 = np.mean(np.stack(j1s, 0), 0)
        p1_m = j1 / np.clip(e_m, 1e-12, None)
        p0_m = (pp - j1) / np.clip(1.0 - e_m, 1e-12, None)
        lor = np.abs(logit(p0_m) - logit(p1_m))
        g = np.exp(np.max(lor) if quantile >= 1.0 else np.quantile(lor, quantile))
        return float(g), b0, a0

    if confounding_scale is not None:
        c = float(confounding_scale)
    elif target_gamma is None or target_gamma <= 1.0 + 1e-9:
        c = 0.0
    else:
        lo_c, hi_c = 0.0, 1.0
        while realise(hi_c, probe)[0] < target_gamma and hi_c < 64:
            hi_c *= 2.0
        for _ in range(30):
            mid = 0.5 * (lo_c + hi_c)
            if realise(mid, probe)[0] < target_gamma:
                lo_c = mid
            else:
                hi_c = mid
        c = 0.5 * (lo_c + hi_c)

    gamma_s = c
    kappa_i = c * kappa_mult[Z]
    b0 = _solve_intercept(core_y_all, gamma_s, prevalence, is_outcome=True, iters=60)
    a0 = _solve_intercept(core_t_all, None, selection_rate, kappa_i=kappa_i,
                          is_outcome=False, iters=60)
    lin_y = core_y_all + b0
    lin_t = core_t_all + a0

    S = rng.normal(size=n)
    Y_full = (rng.uniform(size=n) < expit(lin_y + gamma_s * S)).astype(float)
    T = (rng.uniform(size=n) < expit(lin_t + kappa_i * S)).astype(float)
    Y_obs = np.where(T == 1, Y_full, np.nan)

    # Oracle nuisances must be MARGINAL over the decision maker Z: a learner sees
    # only X, so the relevant propensity is P(T=1|X=x) averaged over the random
    # assignment of cases to decision makers, not P(T=1|X=x, Z=z).  (The
    # Z-conditional versions are kept separately for the falsification test.)
    p1_by_z, e_by_z, joint1_by_z = [], [], []
    for j in range(n_judges):
        lt = X @ alpha + leniency[j] + a0
        _, e_j, p1_j, _ = oracle_quantities(lin_y, lt, gamma_s,
                                            np.full(n, c * kappa_mult[j]))
        p1_by_z.append(p1_j); e_by_z.append(e_j); joint1_by_z.append(e_j * p1_j)
    p = oracle_quantities(lin_y, lin_t, gamma_s, kappa_i)[0]
    e = np.mean(np.stack(e_by_z, 0), 0)
    joint1 = np.mean(np.stack(joint1_by_z, 0), 0)
    p1 = joint1 / np.clip(e, 1e-12, None)
    p0 = (p - joint1) / np.clip(1.0 - e, 1e-12, None)
    lor = np.abs(logit(p0) - logit(p1))
    gamma0 = float(np.exp(np.max(lor)))
    gamma0_q99 = float(np.exp(np.quantile(lor, 0.99)))
    # The *conditional* sensitivity parameter, i.e. DCSM within each decision
    # maker.  This -- not the Z-marginal gamma0 -- is what the leniency-instrument
    # assumption posits, and what the falsification bound must respect:
    # marginalising over Z averages
    # the tilts and can shrink the log odds ratio, so gamma0 <= gamma0_cond.
    lor_z = []
    for e_j, p1_j in zip(e_by_z, p1_by_z):
        p0_j = (p - e_j * p1_j) / np.clip(1.0 - e_j, 1e-12, None)
        lor_z.append(np.abs(logit(np.clip(p0_j, 1e-12, 1 - 1e-12)) - logit(p1_j)))
    gamma0_cond = float(np.exp(np.max(np.stack(lor_z, 0))))

    return SelectiveLabelsDataset(
        X=X, T=T, Y_obs=Y_obs, Y_full=Y_full, Z=Z,
        feature_names=[f"x{i}" for i in range(d)],
        name=f"SL-Bench(d={d}, Gamma0={gamma0:.2f})",
        oracle=dict(p=p, e=e, p1=p1, p0=p0, gamma0=gamma0, gamma0_q99=gamma0_q99,
                    gamma0_cond=gamma0_cond,
                    confounding_scale=c, beta=beta, alpha=alpha, S=S,
                    p1_by_z=p1_by_z, e_by_z=e_by_z, leniency=leniency),
        notes="Synthetic; oracle nuisances exact by Gauss-Hermite quadrature.",
    )
