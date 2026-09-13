"""Theorem 1: sharp partial identification of *ranking* metrics under DCSM(Gamma).

Why ranking metrics are harder than threshold metrics
-----------------------------------------------------
Sensitivity, specificity, PPV and NPV at a *fixed score threshold* are
linear-fractional in the outcome regression ``p(.)``, so their sharp bounds
under a pointwise sensitivity model follow from a one-dimensional search; this
is what the existing selective-labels / sensitivity-analysis literature
delivers.  AUROC is not of that form: it is a *ratio of a bilinear form in p to
a product of linear forms*, and the naive "plug the pointwise extremes in"
recipe is **not** sharp -- and, worse, is not even valid (it under-covers).

The rank identity that makes it tractable
-----------------------------------------
Write ``psi(u, v) = 1{u > v} + 1{u = v}/2`` and, for a score ``f`` and covariate
law ``mu``, define the **mid-rank function**

    R(x) = E_{X' ~ mu}[ psi(f(x), f(X')) ]                                   (R)

which is *identified* (it depends only on ``f`` and on ``mu = P_X``, both
observed).  Let ``pi = E_mu[p]``.  Because ``psi(u, v) + psi(v, u) = 1`` the
operator ``K h = E[psi(f(.), f(X')) h(X')]`` satisfies ``K + K^* = 1 (x) 1``,
hence ``<p, K p> = pi^2 / 2`` for *every* ``p``.  Expanding the Mann-Whitney
representation of AUROC therefore gives the exact identity

    AUC(p) = ( <p, R>_mu - pi^2 / 2 ) / ( pi (1 - pi) ).                     (1)

Given ``pi``, AUROC is **linear** in ``p``.  So the sharp interval is obtained by
(i) a continuous-knapsack optimisation of ``<p, R>`` over the box at fixed
prevalence, whose value function is piecewise linear and concave/convex, and
(ii) an outer one-dimensional optimisation over ``pi`` of a ratio of quadratics,
solvable in closed form on each linear piece.  Total cost ``O(n log n)``, exact.

This module implements (1), the exact algorithm, a unifying solver for the whole
class of "prevalence-linear" functionals (which recovers the known threshold
metric bounds as special cases), and independent brute-force verifiers used in
the sharpness experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "midrank",
    "auc_from_regression",
    "auc_direct",
    "MetricSpec",
    "AUROC_SPEC",
    "sensitivity_spec",
    "specificity_spec",
    "ppv_spec",
    "BoundResult",
    "sharp_bounds",
    "sharp_auc_interval",
    "naive_corner_interval",
    "separate_bounds_interval",
    "bruteforce_auc_interval",
]

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Rank machinery
# --------------------------------------------------------------------------- #
def _normalise_weights(w: np.ndarray | None, n: int) -> np.ndarray:
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError("weights must have shape (n,)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must sum to a positive number")
    return w / s


def midrank(scores: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Mid-rank function ``R(x_i) = E_mu[psi(f(x_i), f(X'))]``; see (R).

    Ties contribute one half.  ``O(n log n)``.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    w = _normalise_weights(weights, n)

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    w_sorted = w[order]
    cum = np.concatenate([[0.0], np.cumsum(w_sorted)])

    # Group boundaries of equal scores.
    new_group = np.concatenate([[True], s_sorted[1:] != s_sorted[:-1]])
    group_id = np.cumsum(new_group) - 1
    n_groups = group_id[-1] + 1
    start = np.zeros(n_groups, dtype=int)
    start[group_id[new_group]] = np.flatnonzero(new_group)
    end = np.concatenate([start[1:], [n]])

    below = cum[start]                       # total weight strictly below group
    equal = cum[end] - cum[start]            # total weight tied with group
    r_sorted = (below + 0.5 * equal)[group_id]

    out = np.empty(n, dtype=float)
    out[order] = r_sorted
    return out


def auc_from_regression(
    p: np.ndarray, ranks: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """AUROC via the rank identity (1).  ``ranks`` comes from :func:`midrank`."""
    p = np.asarray(p, dtype=float)
    w = _normalise_weights(weights, p.shape[0])
    pi = float(w @ p)
    if pi <= _EPS or pi >= 1.0 - _EPS:
        return float("nan")
    return float((w @ (p * ranks) - 0.5 * pi**2) / (pi * (1.0 - pi)))


def auc_direct(
    p: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """AUROC from the defining double integral -- ``O(n^2)``, used to verify (1)."""
    p = np.asarray(p, dtype=float)
    scores = np.asarray(scores, dtype=float)
    w = _normalise_weights(weights, p.shape[0])
    psi = (scores[:, None] > scores[None, :]).astype(float)
    psi += 0.5 * (scores[:, None] == scores[None, :])
    num = (w * p) @ psi @ (w * (1.0 - p))
    pi = float(w @ p)
    den = pi * (1.0 - pi)
    if den <= _EPS:
        return float("nan")
    return float(num / den)


# --------------------------------------------------------------------------- #
# The unifying functional class
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricSpec:
    """A *prevalence-linear* functional of the outcome regression.

    ``M(p) = ( <p, phi>_mu + c0 + c1 pi + c2 pi^2 ) / ( d0 + d1 pi + d2 pi^2 )``

    with ``pi = <p, 1>_mu`` and ``phi`` identified.  AUROC, Somers' D, and every
    fixed-threshold confusion-matrix metric are members (see the constructors
    below), so Theorem 1 subsumes the known threshold-metric bounds.
    """

    phi: np.ndarray
    c0: float = 0.0
    c1: float = 0.0
    c2: float = 0.0
    d0: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    name: str = "metric"

    def value(self, p: np.ndarray, weights: np.ndarray | None = None) -> float:
        p = np.asarray(p, dtype=float)
        w = _normalise_weights(weights, p.shape[0])
        pi = float(w @ p)
        num = float(w @ (p * self.phi)) + self.c0 + self.c1 * pi + self.c2 * pi**2
        den = self.d0 + self.d1 * pi + self.d2 * pi**2
        if abs(den) <= _EPS:
            return float("nan")
        return num / den


def AUROC_SPEC(scores: np.ndarray, weights: np.ndarray | None = None) -> MetricSpec:
    """AUROC as a :class:`MetricSpec`:  phi = mid-rank, c2 = -1/2, den = pi(1-pi)."""
    return MetricSpec(
        phi=midrank(scores, weights), c2=-0.5, d1=1.0, d2=-1.0, name="AUROC"
    )


def sensitivity_spec(scores: np.ndarray, tau: float) -> MetricSpec:
    """``P(f(X) > tau | Y = 1)``."""
    return MetricSpec(phi=(np.asarray(scores) > tau).astype(float), d1=1.0,
                      name=f"TPR@{tau:g}")


def specificity_spec(
    scores: np.ndarray, tau: float, weights: np.ndarray | None = None
) -> MetricSpec:
    """``P(f(X) <= tau | Y = 0)``."""
    scores = np.asarray(scores, dtype=float)
    w = _normalise_weights(weights, scores.shape[0])
    ind = (scores <= tau).astype(float)
    return MetricSpec(phi=-ind, c0=float(w @ ind), d0=1.0, d1=-1.0,
                      name=f"TNR@{tau:g}")


def ppv_spec(
    scores: np.ndarray, tau: float, weights: np.ndarray | None = None
) -> MetricSpec:
    """``P(Y = 1 | f(X) > tau)``."""
    scores = np.asarray(scores, dtype=float)
    w = _normalise_weights(weights, scores.shape[0])
    ind = (scores > tau).astype(float)
    return MetricSpec(phi=ind, d0=float(w @ ind), name=f"PPV@{tau:g}")


# --------------------------------------------------------------------------- #
# Continuous-knapsack value function
# --------------------------------------------------------------------------- #
@dataclass
class _Profile:
    """Piecewise-linear value function of ``max/min <p, phi>`` at fixed prevalence."""

    pis: np.ndarray        # breakpoints, increasing, length m + 1
    vals: np.ndarray       # value at each breakpoint
    slopes: np.ndarray     # slope on segment k = (pis[k], pis[k+1]); length m
    order: np.ndarray      # unit ordering used by the greedy fill
    lo: np.ndarray
    cap: np.ndarray        # w_i * (hi_i - lo_i)
    gap: np.ndarray        # hi_i - lo_i

    def argmin_p(self, pi: float) -> np.ndarray:
        """Reconstruct the extremal ``p`` achieving the profile at prevalence ``pi``."""
        budget = max(0.0, pi - self.pis[0])
        p = self.lo.copy()
        for idx in self.order:
            c = self.cap[idx]
            if c <= 0:
                continue
            take = min(c, budget)
            if take > 0:
                # cap = w * (hi - lo); take/cap is the fraction of the gap used.
                p[idx] = self.lo[idx] + (take / c) * self.gap[idx]
                budget -= take
            if budget <= 1e-15:
                break
        return p



def _build_profile(
    lo: np.ndarray, hi: np.ndarray, phi: np.ndarray, w: np.ndarray, maximise: bool
) -> _Profile:
    cap = w * (hi - lo)
    order = np.argsort(-phi if maximise else phi, kind="mergesort")
    pi0 = float(w @ lo)
    v0 = float(w @ (lo * phi))
    caps = cap[order]
    gains = caps * phi[order]
    pis = np.concatenate([[pi0], pi0 + np.cumsum(caps)])
    vals = np.concatenate([[v0], v0 + np.cumsum(gains)])
    return _Profile(pis=pis, vals=vals, slopes=phi[order], order=order,
                    lo=lo, cap=cap, gap=hi - lo)


# --------------------------------------------------------------------------- #
# Outer 1-D optimisation: ratio of quadratics on each linear piece
# --------------------------------------------------------------------------- #
def _stationary_points(n0, n1, n2, d0, d1, d2) -> list[float]:
    """Roots of ``d/dpi [ (n0+n1 pi+n2 pi^2) / (d0+d1 pi+d2 pi^2) ] = 0``.

    The cubic terms cancel identically, leaving ``A pi^2 + B pi + C = 0`` with
    ``A = n2 d1 - n1 d2``, ``B = 2(n2 d0 - n0 d2)``, ``C = n1 d0 - n0 d1``.
    """
    A = n2 * d1 - n1 * d2
    B = 2.0 * (n2 * d0 - n0 * d2)
    C = n1 * d0 - n0 * d1
    if abs(A) < 1e-14:
        if abs(B) < 1e-14:
            return []
        return [-C / B]
    disc = B * B - 4.0 * A * C
    if disc < 0:
        return []
    sq = np.sqrt(disc)
    return [(-B + sq) / (2.0 * A), (-B - sq) / (2.0 * A)]


@dataclass(frozen=True)
class BoundResult:
    """Sharp identified interval for a metric, plus the least-favourable members."""

    lower: float
    upper: float
    pi_lower: float
    pi_upper: float
    p_lower: np.ndarray
    p_upper: np.ndarray
    metric: str = "metric"

    @property
    def interval(self) -> Tuple[float, float]:
        return (self.lower, self.upper)

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float, tol: float = 1e-9) -> bool:
        return bool(self.lower - tol <= value <= self.upper + tol)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"BoundResult({self.metric}: [{self.lower:.4f}, {self.upper:.4f}], "
                f"width={self.width:.4f})")


def _optimise(
    spec: MetricSpec,
    lo: np.ndarray,
    hi: np.ndarray,
    w: np.ndarray,
    maximise: bool,
    pi_floor: float,
    pi_ceil: float,
) -> Tuple[float, float, np.ndarray]:
    prof = _build_profile(lo, hi, spec.phi, w, maximise=maximise)
    best_val = -np.inf if maximise else np.inf
    best_pi = float("nan")

    for k in range(len(prof.slopes)):
        a_pi, b_pi = prof.pis[k], prof.pis[k + 1]
        seg_lo = max(a_pi, pi_floor)
        seg_hi = min(b_pi, pi_ceil)
        if seg_hi < seg_lo:
            continue
        beta = prof.slopes[k]
        alpha = prof.vals[k] - beta * a_pi           # <p, phi> = alpha + beta * pi
        n0, n1, n2 = alpha + spec.c0, beta + spec.c1, spec.c2
        d0, d1, d2 = spec.d0, spec.d1, spec.d2

        cands = [seg_lo, seg_hi]
        cands += [r for r in _stationary_points(n0, n1, n2, d0, d1, d2)
                  if seg_lo < r < seg_hi]
        for pi in cands:
            den = d0 + d1 * pi + d2 * pi**2
            if abs(den) <= 1e-14:
                continue
            val = (n0 + n1 * pi + n2 * pi**2) / den
            if (maximise and val > best_val) or ((not maximise) and val < best_val):
                best_val, best_pi = float(val), float(pi)

    if not np.isfinite(best_val):
        return float("nan"), float("nan"), lo.copy()
    return best_val, best_pi, prof.argmin_p(best_pi)


def sharp_bounds(
    spec: MetricSpec,
    lo: np.ndarray,
    hi: np.ndarray,
    weights: np.ndarray | None = None,
    pi_margin: float = 1e-6,
) -> BoundResult:
    """Exact sharp identified interval for any :class:`MetricSpec`.

    Parameters
    ----------
    spec
        The functional (e.g. ``AUROC_SPEC(scores)``).
    lo, hi
        The identified box for ``p`` from :mod:`dcl.sensitivity`.
    weights
        Sampling weights (defaults to uniform ``1/n``).
    pi_margin
        Prevalence is restricted to ``[pi_margin, 1 - pi_margin]``: at exactly
        ``pi in {0, 1}`` a ranking metric is undefined (one class is empty).

    Complexity ``O(n log n)``.  The returned ``p_lower`` / ``p_upper`` are the
    least-favourable members of the identified set, so the bounds are attained
    (this is what "sharp" means) -- not merely valid.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if lo.shape != hi.shape:
        raise ValueError("lo and hi must have the same shape")
    if np.any(lo > hi + 1e-9):
        raise ValueError("lo must be <= hi")
    w = _normalise_weights(weights, lo.shape[0])

    pi_lo_raw, pi_hi_raw = float(w @ lo), float(w @ hi)
    pi_floor = max(pi_lo_raw, pi_margin)
    pi_ceil = min(pi_hi_raw, 1.0 - pi_margin)
    if pi_ceil < pi_floor:  # the whole identified set is degenerate
        pi_floor = pi_ceil = float(np.clip(0.5 * (pi_lo_raw + pi_hi_raw),
                                           pi_margin, 1.0 - pi_margin))

    up, pi_up, p_up = _optimise(spec, lo, hi, w, True, pi_floor, pi_ceil)
    dn, pi_dn, p_dn = _optimise(spec, lo, hi, w, False, pi_floor, pi_ceil)
    return BoundResult(lower=dn, upper=up, pi_lower=pi_dn, pi_upper=pi_up,
                       p_lower=p_dn, p_upper=p_up, metric=spec.name)


def sharp_auc_interval(
    scores: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    weights: np.ndarray | None = None,
    pi_margin: float = 1e-6,
) -> BoundResult:
    """Theorem 1: the sharp identified interval for the deployment AUROC of ``f``."""
    return sharp_bounds(AUROC_SPEC(scores, weights), lo, hi, weights, pi_margin)


# --------------------------------------------------------------------------- #
# Comparators used in the experiments
# --------------------------------------------------------------------------- #
def naive_corner_interval(
    scores: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    weights: np.ndarray | None = None, include_mid: bool = True,
) -> Tuple[float, float]:
    """The recipe a practitioner would reach for: evaluate at the box corners.

    Not valid: it ignores that AUROC couples units through the prevalence, so it
    systematically *under-covers* the true deployment AUROC.
    """
    r = midrank(scores, weights)
    cands = [auc_from_regression(lo, r, weights), auc_from_regression(hi, r, weights)]
    if include_mid:
        cands.append(auc_from_regression(0.5 * (np.asarray(lo) + np.asarray(hi)),
                                         r, weights))
    cands = [c for c in cands if np.isfinite(c)]
    return (min(cands), max(cands)) if cands else (float("nan"), float("nan"))


def separate_bounds_interval(
    scores: np.ndarray, lo: np.ndarray, hi: np.ndarray,
    weights: np.ndarray | None = None, pi_margin: float = 1e-6,
) -> Tuple[float, float]:
    """Valid-but-loose: bound numerator and denominator of (1) independently.

    A strict superset of the sharp interval; quantifies what sharpness buys.
    """
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    w = _normalise_weights(weights, lo.shape[0])
    r = midrank(scores, weights)
    num_lo = float(w @ (np.where(r >= 0, lo, hi) * r))
    num_hi = float(w @ (np.where(r >= 0, hi, lo) * r))
    pi_l = max(float(w @ lo), pi_margin)
    pi_h = min(float(w @ hi), 1.0 - pi_margin)
    if pi_h < pi_l:
        pi_l = pi_h = float(np.clip(0.5 * (pi_l + pi_h), pi_margin, 1 - pi_margin))
    grid = np.linspace(pi_l, pi_h, 2001)
    den = grid * (1.0 - grid)
    lower = float(np.min((num_lo - 0.5 * grid**2) / den))
    upper = float(np.max((num_hi - 0.5 * grid**2) / den))
    return max(lower, 0.0), min(upper, 1.0)


def bruteforce_auc_interval(
    scores: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    weights: np.ndarray | None = None,
    n_restarts: int = 200,
    n_steps: int = 400,
    step: float = 0.35,
    seed: int = 0,
) -> Tuple[float, float]:
    """Independent verifier: projected-gradient search for extremal AUROC.

    Optimises the *direct* AUROC functional over the box without using identity
    (1) at all, from many random starts (uniform, vertex, and the corners).  Used
    in the sharpness experiment: the analytic interval must contain every value
    found here, and the gap must vanish.
    """
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    n = lo.shape[0]
    w = _normalise_weights(weights, n)
    rng = np.random.default_rng(seed)
    r = midrank(scores, weights)
    gap = hi - lo

    def value_and_grad(p):
        pi = float(w @ p)
        if pi <= 1e-9 or pi >= 1 - 1e-9:
            return -np.inf, np.zeros(n)
        num = float(w @ (p * r)) - 0.5 * pi**2
        den = pi * (1.0 - pi)
        val = num / den
        dnum = w * r - pi * w
        dden = w * (1.0 - 2.0 * pi)
        return val, (dnum * den - num * dden) / den**2

    best_hi, best_lo = -np.inf, np.inf
    starts = [lo.copy(), hi.copy(), 0.5 * (lo + hi)]
    for _ in range(n_restarts):
        kind = rng.integers(0, 2)
        u = rng.uniform(size=n) if kind == 0 else rng.integers(0, 2, n).astype(float)
        starts.append(lo + u * gap)

    for p0 in starts:
        for sign in (1.0, -1.0):
            p = p0.copy()
            for t in range(n_steps):
                val, g = value_and_grad(p)
                if not np.isfinite(val):
                    break
                lr = step / (1.0 + 0.01 * t)
                p = np.clip(p + sign * lr * g / (np.abs(g).max() + 1e-12) * gap, lo, hi)
            val, _ = value_and_grad(p)
            if np.isfinite(val):
                best_hi = max(best_hi, val)
                best_lo = min(best_lo, val)
    return best_lo, best_hi
