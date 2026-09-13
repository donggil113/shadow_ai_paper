"""Theorem 3, constructive half: Gamma is not identified, but it *is* falsifiable.

Theorem 3 says two things.  (i) No estimator can be consistent for the
deployment risk uniformly over censoring mechanisms -- so some sensitivity
parameter must be supplied by the analyst.  (ii) ``Gamma`` itself is not point
identified: DCSM(Gamma) and DCSM(Gamma') induce exactly the same observed-data
law for any ``Gamma' > Gamma``.

That is only half the story.  When the decision was taken by *several* decision
makers whose leniency varies for reasons unrelated to the outcome -- judges,
clinicians, loan officers, moderation queues -- DCSM(Gamma) acquires testable
implications, and small values of ``Gamma`` can be **refuted**.

The test.  Let ``Z`` index decision makers and assume the exclusion restriction
``Z ind. (Y, S) | X``, so ``p(x) = P(Y = 1 | X = x)`` does not depend on ``z``,
while ``e_z(x)`` and ``p1_z(x)`` do.  Each ``z`` produces its own identified
interval ``[lo_z^Gamma(x), hi_z^Gamma(x)]``.  Since the *same* ``p(x)`` must lie in
all of them,

    Gamma_min(x) = min { Gamma >= 1 : intersect_z [lo_z^Gamma(x), hi_z^Gamma(x)] != empty }

is an identified **lower bound** on the true Gamma, and DCSM(Gamma) is refuted
for every ``Gamma < Gamma_min(x)``.  The intervals are nested and monotone in
Gamma, so bisection is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .sensitivity import expit, logit, outcome_bounds

__all__ = ["gamma_lower_bound", "GammaFalsification", "falsification_curve"]


def _intersection_nonempty(p1_by_z, e_by_z, gamma: float) -> np.ndarray:
    """Per-unit indicator that the DCSM(Gamma) intervals across z intersect."""
    los, his = [], []
    for p1, e in zip(p1_by_z, e_by_z):
        lo, hi = outcome_bounds(p1, e, gamma)
        los.append(lo); his.append(hi)
    lo_max = np.max(np.stack(los, axis=0), axis=0)
    hi_min = np.min(np.stack(his, axis=0), axis=0)
    return lo_max <= hi_min + 1e-12


def gamma_lower_bound(
    p1_by_z: Sequence[np.ndarray],
    e_by_z: Sequence[np.ndarray],
    gamma_hi: float = 1e4,
    n_bisect: int = 60,
    quantile: float = 1.0,
) -> float:
    """Identified lower bound on ``Gamma`` from leniency variation.

    Parameters
    ----------
    p1_by_z, e_by_z
        Per-decision-maker nuisances evaluated on a common grid of units,
        each an array of shape ``(n,)``.
    quantile
        ``1.0`` returns ``max_x Gamma_min(x)`` (any single unit can refute).
        A value like ``0.95`` returns a robust version less sensitive to the
        noisiest unit -- report both.
    """
    p1_by_z = [np.asarray(a, float) for a in p1_by_z]
    e_by_z = [np.asarray(a, float) for a in e_by_z]
    if len(p1_by_z) < 2:
        return 1.0
    n = p1_by_z[0].shape[0]

    # Per-unit bisection, vectorised: the feasibility indicator is monotone in
    # Gamma (intervals grow), so a single bisection per unit suffices.
    lo = np.ones(n)
    hi = np.full(n, gamma_hi)
    feasible_at_hi = _intersection_nonempty(p1_by_z, e_by_z, gamma_hi)
    etas = [logit(p1) for p1 in p1_by_z]
    for _ in range(n_bisect):
        mid = np.sqrt(lo * hi)                       # geometric bisection
        # Each unit is evaluated at its OWN current midpoint, so the bounds are
        # formed here with a per-unit Gamma rather than through outcome_bounds.
        lg = np.log(mid)
        los, his = [], []
        for eta, p1, e in zip(etas, p1_by_z, e_by_z):
            los.append(e * p1 + (1 - e) * expit(eta - lg))
            his.append(e * p1 + (1 - e) * expit(eta + lg))
        ok = np.max(np.stack(los, 0), 0) <= np.min(np.stack(his, 0), 0) + 1e-12
        hi = np.where(ok, mid, hi)       # feasible -> search lower
        lo = np.where(ok, lo, mid)       # infeasible -> search higher
    gmin = np.where(feasible_at_hi, hi, np.inf)
    finite = gmin[np.isfinite(gmin)]
    if finite.size == 0:
        return float("inf")
    if quantile >= 1.0:
        return float(np.max(finite))
    return float(np.quantile(finite, quantile))


@dataclass
class GammaFalsification:
    gamma_min: float
    gamma_min_q95: float
    gamma_min_boot_lo: float
    n_units: int
    n_groups: int

    def refutes(self, gamma: float) -> bool:
        """Is DCSM(``gamma``) refuted at the bootstrap lower confidence limit?"""
        return gamma < self.gamma_min_boot_lo


def falsification_curve(
    p1_by_z, e_by_z, n_boot: int = 200, alpha: float = 0.05, seed: int = 0
) -> GammaFalsification:
    """Point estimate + bootstrap lower confidence limit for ``Gamma_min``."""
    p1_by_z = [np.asarray(a, float) for a in p1_by_z]
    e_by_z = [np.asarray(a, float) for a in e_by_z]
    n = p1_by_z[0].shape[0]
    rng = np.random.default_rng(seed)
    point = gamma_lower_bound(p1_by_z, e_by_z)
    q95 = gamma_lower_bound(p1_by_z, e_by_z, quantile=0.95)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(gamma_lower_bound([a[idx] for a in p1_by_z],
                                       [a[idx] for a in e_by_z], quantile=0.95))
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo = float(np.quantile(boots, alpha)) if boots.size else 1.0
    return GammaFalsification(gamma_min=point, gamma_min_q95=q95,
                              gamma_min_boot_lo=lo, n_units=n, n_groups=len(p1_by_z))
