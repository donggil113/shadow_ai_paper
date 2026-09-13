"""Decision-Censoring Sensitivity Model (DCSM).

Setting (selective labels).  For each unit we observe covariates ``X``.  A
decision maker -- who additionally sees private information ``S`` that we never
observe -- takes a binary decision ``T ~ pi(. | X, S)``.  The outcome ``Y`` is
recorded **only** when ``T = 1``.  The observed data are therefore

    (X_i, T_i, T_i * Y_i),   i = 1..n,

and the deployment target -- the risk of a predictor on *all* units, including
those the incumbent policy censored -- is not point identified.

Identifiable ("observed") nuisances
-----------------------------------
    e(x)   = P(T = 1 | X = x)                 propensity / selection rate
    p1(x)  = P(Y = 1 | X = x, T = 1)          outcome among the selected

Unidentifiable target
---------------------
    p(x)   = P(Y = 1 | X = x)
           = e(x) p1(x) + (1 - e(x)) p0(x),
    p0(x)  = P(Y = 1 | X = x, T = 0).

The DCSM
--------
``DCSM(Gamma)`` bounds how far the censored sub-population may differ from the
selected one, on the log-odds scale:

    1/Gamma  <=  odds(p0(x)) / odds(p1(x))  <=  Gamma        for a.e. x,       (*)

equivalently ``|logit p0(x) - logit p1(x)| <= log Gamma``.  ``Gamma = 1``
recovers selection-on-observables (MAR / "no unmeasured judgement"), and
``Gamma = inf`` recovers the assumption-free Manski bounds.

Relation to the marginal sensitivity model.  If the decision propensity obeys
Tan's MSM with parameter ``Lambda``,

    1/Lambda <= odds(e(x)) / odds(e(x, s)) <= Lambda,   e(x, s) = P(T=1|X=x,S=s),

then (see Lemma 1 in the paper) the induced constraint on ``p0`` is exactly (*)
with ``Gamma = Lambda^2``; the two models are equivalent reparameterisations for
every functional of ``P(Y | X, T = 0)``.  Use :func:`gamma_from_lambda` /
:func:`lambda_from_gamma` to translate.

Because the bound (*) is imposed *pointwise in x* and the tilt ``a(x)`` is an
otherwise unrestricted measurable function, the identified set for the whole
regression function ``p(.)`` is an axis-aligned **box**

    I_Gamma = { p : p_lo(x) <= p(x) <= p_hi(x)  for a.e. x },

which is what makes both the sharp AUROC interval (Theorem 1) and the minimax
learning problem (Theorem 2) tractable.  This module computes that box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "logit",
    "expit",
    "IdentifiedSet",
    "identified_set",
    "outcome_bounds",
    "tilt_to_outcome",
    "realised_gamma",
    "gamma_from_lambda",
    "lambda_from_gamma",
    "width",
]

_EPS = 1e-12


def expit(z: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p: np.ndarray | float, eps: float = _EPS) -> np.ndarray:
    """Log-odds with clipping; ``logit(0) = -inf`` is avoided by ``eps``."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def gamma_from_lambda(lam: float) -> float:
    """MSM parameter ``Lambda`` -> DCSM parameter ``Gamma = Lambda ** 2``."""
    if lam < 1.0:
        raise ValueError("Lambda must be >= 1")
    return float(lam) ** 2


def lambda_from_gamma(gamma: float) -> float:
    """DCSM parameter ``Gamma`` -> MSM parameter ``Lambda = sqrt(Gamma)``."""
    if gamma < 1.0:
        raise ValueError("Gamma must be >= 1")
    return float(np.sqrt(gamma))


def tilt_to_outcome(p1: np.ndarray, e: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Marginal outcome probability induced by a tilt ``a(x)``.

    ``p^a(x) = e(x) p1(x) + (1 - e(x)) * expit(logit p1(x) + log a(x))``.

    ``a`` is the odds ratio ``odds(p0) / odds(p1)``; ``a = 1`` is MAR.
    """
    p1 = np.asarray(p1, dtype=float)
    e = np.asarray(e, dtype=float)
    a = np.asarray(a, dtype=float)
    p0 = expit(logit(p1) + np.log(np.clip(a, _EPS, None)))
    return e * p1 + (1.0 - e) * p0


def outcome_bounds(
    p1: np.ndarray, e: np.ndarray, gamma: float, direction: str = "two-sided"
) -> Tuple[np.ndarray, np.ndarray]:
    """Sharp pointwise bounds ``[p_lo(x), p_hi(x)]`` on ``p(x)`` under DCSM(Gamma).

    ``p^a(x)`` is strictly increasing in the tilt ``a``, so the extremes are
    attained at ``a = 1/Gamma`` and ``a = Gamma``.  ``gamma = np.inf`` returns
    the Manski bounds ``[e p1, e p1 + (1 - e)]``.

    Directional DCSM
    ----------------
    In most applications the analyst knows the *sign* of the selection effect
    even when its magnitude is unknown: clinicians test the patients they
    already suspect are sick (so the untested are lower risk, ``p0 <= p1``);
    lenders fund the applicants they judge safest (so the rejected are higher
    risk, ``p0 >= p1``).  Imposing the known sign is a genuinely weaker
    assumption than picking a number for ``Gamma``, and it halves the identified
    set.  ``direction`` takes:

    ``"two-sided"``      ``1/Gamma <= a <= Gamma``      (default, agnostic)
    ``"censored-lower"`` ``1/Gamma <= a <= 1``          (censored units safer)
    ``"censored-higher"`` ``1 <= a <= Gamma``           (censored units riskier)
    """
    p1 = np.asarray(p1, dtype=float)
    e = np.asarray(e, dtype=float)
    if np.any(p1 < 0) or np.any(p1 > 1):
        raise ValueError("p1 must lie in [0, 1]")
    if np.any(e < 0) or np.any(e > 1):
        raise ValueError("e must lie in [0, 1]")
    if gamma < 1.0:
        raise ValueError("Gamma must be >= 1")

    if direction not in ("two-sided", "censored-lower", "censored-higher"):
        raise ValueError(f"unknown direction {direction!r}")

    if not np.isfinite(gamma):
        lo_all, hi_all = e * p1, e * p1 + (1.0 - e)
        if direction == "censored-lower":
            return lo_all, p1
        if direction == "censored-higher":
            return p1, hi_all
        return lo_all, hi_all

    log_g = np.log(gamma)
    eta1 = logit(p1)
    # The log-odds scale is where Gamma acts as a pure translation; this keeps
    # the map from the estimated nuisance to the bounds 1-Lipschitz (Theorem 4).
    lo = e * p1 + (1.0 - e) * expit(eta1 - log_g)
    hi = e * p1 + (1.0 - e) * expit(eta1 + log_g)
    if direction == "censored-lower":
        return lo, p1.copy() if isinstance(p1, np.ndarray) else p1
    if direction == "censored-higher":
        return (p1.copy() if isinstance(p1, np.ndarray) else p1), hi
    return lo, hi


@dataclass(frozen=True)
class IdentifiedSet:
    """The box ``{p : lo <= p <= hi}`` evaluated on a finite sample.

    Attributes
    ----------
    lo, hi : arrays of shape (n,)
        Sharp pointwise lower/upper bounds on ``P(Y = 1 | X = x_i)``.
    p1, e : arrays of shape (n,)
        The identifiable nuisances the box was built from.
    gamma : float
        Sensitivity parameter used.
    """

    lo: np.ndarray
    hi: np.ndarray
    p1: np.ndarray
    e: np.ndarray
    gamma: float

    def __post_init__(self) -> None:
        if np.any(self.lo > self.hi + 1e-9):
            raise ValueError("lower bound exceeds upper bound")

    @property
    def mid(self) -> np.ndarray:
        """Midpoint ``(lo + hi) / 2`` -- the 'p-tilde' of Theorem 2."""
        return 0.5 * (self.lo + self.hi)

    @property
    def delta(self) -> np.ndarray:
        """Identification width ``hi - lo`` -- the censoring-uncertainty signal."""
        return self.hi - self.lo

    def contains(self, p: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return (p >= self.lo - tol) & (p <= self.hi + tol)

    def clip(self, p: np.ndarray) -> np.ndarray:
        """Project a candidate regression function onto the identified set."""
        return np.clip(np.asarray(p, dtype=float), self.lo, self.hi)

    def sample(self, rng: np.random.Generator, kind: str = "uniform") -> np.ndarray:
        """Draw a member of the identified set (for Monte-Carlo sharpness checks)."""
        n = self.lo.shape[0]
        if kind == "uniform":
            u = rng.uniform(size=n)
        elif kind == "vertex":
            u = rng.integers(0, 2, size=n).astype(float)
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown kind {kind!r}")
        return self.lo + u * self.delta


def identified_set(p1: np.ndarray, e: np.ndarray, gamma: float,
                   direction: str = "two-sided") -> IdentifiedSet:
    """Convenience constructor: nuisances + Gamma -> :class:`IdentifiedSet`."""
    lo, hi = outcome_bounds(p1, e, gamma, direction)
    return IdentifiedSet(
        lo=lo, hi=hi, p1=np.asarray(p1, float), e=np.asarray(e, float), gamma=float(gamma)
    )


def width(p1: np.ndarray, e: np.ndarray, gamma: float,
          direction: str = "two-sided") -> np.ndarray:
    """Identification width ``hi - lo`` without materialising the box."""
    lo, hi = outcome_bounds(p1, e, gamma, direction)
    return hi - lo


def realised_gamma(p0: np.ndarray, p1: np.ndarray, quantile: float = 1.0) -> float:
    """The Gamma actually realised by a known (oracle) censoring mechanism.

    Only available in synthetic / semi-synthetic designs where ``p0`` is known.
    ``quantile < 1`` reports a robust version (e.g. the 99th percentile of the
    absolute log odds ratio) which is useful when a handful of units with
    ``p1`` near 0 or 1 dominate the supremum.
    """
    lor = np.abs(logit(p0) - logit(p1))
    if quantile >= 1.0:
        return float(np.exp(np.max(lor)))
    return float(np.exp(np.quantile(lor, quantile)))
