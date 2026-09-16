"""Coverage-valid identified intervals when the nuisances are *estimated*.

The box of :mod:`dcl.sensitivity` is exact when ``e`` and ``p1`` are known.
With cross-fitted estimates it inherits their error, and on our benchmarks the
plug-in box covers the true ``p(x)`` for only about half of the units pointwise
(``results/exp3_uq_identification.csv``).  This module provides the two
constructions behind the nuisance theorem in the paper.

(a) Aggregate functionals -- one-step / doubly-robust estimation.
    Every bound endpoint used by the sharp AUROC interval is a *linear*
    functional of the outcome regression at a fixed vertex pattern, i.e. of
    the form

        psi_w = E[ w(X) * ( T Y + (1 - T) g(p1(X)) ) ],
        g(p) = tilt of p by Gamma:  g_Gamma(p) = Gamma p / (1 + (Gamma - 1) p),

    (``w = 1`` gives the prevalence bounds ``pi_lo``, ``pi_hi``; ``w = R`` the
    numerator pieces of the AUROC identity).  ``E[w T Y]`` needs no nuisance.
    The second term has the efficient influence function

        (1 - T) g(p1(X)) + (1 - e(X)) g'(p1(X)) T (Y - p1(X)) / e(X) - psi,

    which is Neyman-orthogonal in both nuisances: the plug-in bias is second
    order, ``O(||e_hat - e|| ||p1_hat - p1|| + ||p1_hat - p1||^2)``, so with
    cross-fitting and n^{1/4}-rate nuisances the estimator is root-n and a
    Wald interval from the influence values is asymptotically valid.

(b) Pointwise -- exact binomial intervals plus monotone interval arithmetic.
    No assumption-free confidence interval for a regression function at a
    point exists, so pointwise statements are made at the level of
    *calibration bins* of the cross-fitted scores: within a bin, ``e_b`` and
    ``p1_b`` have exact Clopper-Pearson intervals from the binomial counts.
    Because ``lo(e, eta)`` is increasing in both arguments and ``hi(e, eta)``
    is decreasing in ``e`` and increasing in ``eta``, the widest box
    compatible with ``e in [e_L, e_U]``, ``eta_1 in [eta_L, eta_U]`` is

        [ lo(e_L, eta_L),  hi(e_L, eta_U) ],

    which contains the true box whenever both nuisance intervals do
    (Bonferroni: alpha/2 each).  The guarantee is for bin-level averages;
    it transfers to units under bin-homogeneity, which is what the
    experiments test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .sensitivity import expit, logit

__all__ = [
    "g_tilt", "g_tilt_prime",
    "DRResult", "dr_functional", "prevalence_bounds_dr",
    "binned_nuisance_intervals", "inflate_box", "InflatedBox",
    "clopper_pearson",
]


# --------------------------------------------------------------------------- #
# The tilt map and its derivative
# --------------------------------------------------------------------------- #
def g_tilt(p1: np.ndarray, gamma: float) -> np.ndarray:
    """``sigma(logit p1 + log gamma) = gamma p1 / (1 + (gamma - 1) p1)``.

    ``gamma > 1`` tilts up (the ``hi`` endpoint uses ``g_tilt(p1, Gamma)``),
    ``gamma < 1`` tilts down (``lo`` uses ``g_tilt(p1, 1 / Gamma)``).
    """
    p1 = np.asarray(p1, float)
    return gamma * p1 / (1.0 + (gamma - 1.0) * p1)


def g_tilt_prime(p1: np.ndarray, gamma: float) -> np.ndarray:
    """``d/dp g_tilt(p, gamma) = gamma / (1 + (gamma - 1) p)^2``."""
    p1 = np.asarray(p1, float)
    return gamma / (1.0 + (gamma - 1.0) * p1) ** 2


# --------------------------------------------------------------------------- #
# (a) One-step / DR estimation of aggregate bound functionals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DRResult:
    estimate: float
    se: float
    influence: np.ndarray
    plugin: float

    def ci(self, alpha: float = 0.05) -> Tuple[float, float]:
        z = stats.norm.ppf(1 - alpha / 2)
        return self.estimate - z * self.se, self.estimate + z * self.se


def dr_functional(
    T: np.ndarray, Y_obs: np.ndarray, e_hat: np.ndarray, p1_hat: np.ndarray,
    gamma_tilt: float, w: Optional[np.ndarray] = None, clip: float = 0.02,
) -> DRResult:
    """One-step estimator of ``psi_w = E[w(X)(T Y + (1-T) g(p1(X)))]``.

    ``gamma_tilt`` is the tilt applied inside ``g`` (``1/Gamma`` for the lower
    endpoint, ``Gamma`` for the upper; ``1`` gives the MAR value).  ``Y_obs``
    may be NaN where ``T == 0``.  Nuisances must be cross-fitted.
    """
    T = np.asarray(T, float)
    Y = np.nan_to_num(np.asarray(Y_obs, float))
    e = np.clip(np.asarray(e_hat, float), clip, 1.0)
    p1 = np.clip(np.asarray(p1_hat, float), 1e-6, 1 - 1e-6)
    w = np.ones_like(T) if w is None else np.asarray(w, float)
    g, gp = g_tilt(p1, gamma_tilt), g_tilt_prime(p1, gamma_tilt)
    plugin_terms = w * (T * Y + (1.0 - T) * g)
    correction = w * (1.0 - e) * gp * T * (Y - p1) / e
    phi = plugin_terms + correction
    n = phi.shape[0]
    return DRResult(estimate=float(phi.mean()), se=float(phi.std(ddof=1) / np.sqrt(n)),
                    influence=phi, plugin=float(plugin_terms.mean()))


def prevalence_bounds_dr(
    T, Y_obs, e_hat, p1_hat, gamma: float, alpha: float = 0.05, clip: float = 0.02,
) -> dict:
    """DR point estimates and CIs for the prevalence bounds ``pi_lo``, ``pi_hi``.

    Returns a dict with the two :class:`DRResult` objects and an outer interval
    ``[pi_lo_L, pi_hi_U]`` that covers the identified prevalence interval with
    asymptotic probability ``>= 1 - alpha`` (Bonferroni over the two ends).
    """
    lo = dr_functional(T, Y_obs, e_hat, p1_hat, 1.0 / gamma, clip=clip)
    hi = dr_functional(T, Y_obs, e_hat, p1_hat, gamma, clip=clip)
    z = stats.norm.ppf(1 - alpha / 4)
    return {"pi_lo": lo, "pi_hi": hi,
            "outer": (float(np.clip(lo.estimate - z * lo.se, 0, 1)),
                      float(np.clip(hi.estimate + z * hi.se, 0, 1))),
            "plugin": (lo.plugin, hi.plugin)}


# --------------------------------------------------------------------------- #
# (b) Pointwise: binned exact intervals + monotone interval arithmetic
# --------------------------------------------------------------------------- #
def clopper_pearson(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """Exact two-sided ``1 - alpha`` binomial interval (0 or 1 when degenerate)."""
    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def _quantile_bins(v: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency bin ids in ``0..n_bins-1`` (ties broken by rank)."""
    order = np.argsort(v, kind="mergesort")
    ids = np.empty(v.shape[0], dtype=int)
    ids[order] = np.minimum((np.arange(v.shape[0]) * n_bins) // max(v.shape[0], 1),
                            n_bins - 1)
    return ids


def binned_nuisance_intervals(
    e_hat: np.ndarray, p1_hat: np.ndarray, T: np.ndarray, Y_obs: np.ndarray,
    alpha: float = 0.10, n_bins_e: int = 10, n_bins_p: int = 10,
    smooth_margin: bool = False, fit_mask: Optional[np.ndarray] = None,
) -> dict:
    """Per-unit intervals for ``e`` and ``eta_1 = logit p1`` from calibration bins.

    Units are binned by quantiles of the cross-fitted ``e_hat`` (for ``e``) and
    of ``p1_hat`` (for ``p1``, counts taken from the selected units); each bin
    gets an exact Clopper-Pearson interval at level ``1 - alpha/2`` so that the
    pair holds jointly at ``1 - alpha``.

    What is guaranteed: each bin interval covers the bin *average* of the true
    nuisance (exact, finite sample, no assumptions beyond i.i.d.).  What is
    NOT guaranteed: coverage of an individual unit's ``e(x)`` -- no
    assumption-free confidence interval for a regression function at a point
    exists.  ``smooth_margin=True`` widens each bin interval by the bin's own
    range of the cross-fitted score (``max - min`` of ``e_hat`` / ``logit
    p1_hat`` inside the bin), which covers a unit whenever the true nuisance
    lies within the bin's score range -- a smoothness idealisation that the
    experiments test rather than assume.
    """
    e_hat = np.asarray(e_hat, float); p1_hat = np.asarray(p1_hat, float)
    T = np.asarray(T, float); Y = np.asarray(Y_obs, float)
    n = e_hat.shape[0]
    a = alpha / 2.0
    # ``fit_mask`` selects the CALIBRATION units whose (T, Y) supply the binomial
    # counts; every unit (calibration or not) still receives its bin's interval.
    fm = np.ones(n, dtype=bool) if fit_mask is None else np.asarray(fit_mask, bool)

    eb = _quantile_bins(e_hat, n_bins_e)
    e_L = np.empty(n); e_U = np.empty(n); e_table = []
    for b in range(n_bins_e):
        m = eb == b
        mc = m & fm
        k, nb = int(T[mc].sum()), int(mc.sum())
        l, u = clopper_pearson(k, nb, a)
        if smooth_margin and nb:
            rng_ = float(e_hat[m].max() - e_hat[m].min())
            l, u = max(0.0, l - rng_), min(1.0, u + rng_)
        e_L[m], e_U[m] = l, u
        e_table.append(dict(bin=b, n=nb, k=k, lo=l, hi=u,
                            e_hat_mean=float(e_hat[m].mean()) if nb else np.nan))

    pb = _quantile_bins(p1_hat, n_bins_p)
    eta_L = np.empty(n); eta_U = np.empty(n); p_table = []
    for b in range(n_bins_p):
        m = pb == b
        sel = m & fm & (T == 1)
        k, nb = int(np.nansum(Y[sel])), int(sel.sum())
        l, u = clopper_pearson(k, nb, a)
        el, eu = logit(l), logit(u)
        if smooth_margin and m.sum():
            lp = logit(np.clip(p1_hat[m], 1e-6, 1 - 1e-6))
            rng_ = float(lp.max() - lp.min())
            el, eu = el - rng_, eu + rng_
        eta_L[m], eta_U[m] = el, eu
        p_table.append(dict(bin=b, n_selected=nb, k=k, lo=l, hi=u,
                            p1_hat_mean=float(p1_hat[m].mean()) if m.sum() else np.nan))
    return dict(e_L=e_L, e_U=e_U, eta_L=eta_L, eta_U=eta_U,
                e_bins=eb, p_bins=pb, e_table=e_table, p_table=p_table, alpha=alpha,
                smooth_margin=smooth_margin)


@dataclass(frozen=True)
class InflatedBox:
    lo: np.ndarray
    hi: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.hi - self.lo

    def covers(self, p: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        p = np.asarray(p, float)
        return (p >= self.lo - tol) & (p <= self.hi + tol)


def inflate_box(
    e_L: np.ndarray, e_U: np.ndarray, eta_L: np.ndarray, eta_U: np.ndarray,
    gamma: float, direction: str = "two-sided",
) -> InflatedBox:
    """Widest DCSM box compatible with ``e in [e_L,e_U]``, ``eta_1 in [eta_L,eta_U]``.

    Monotonicity (proved in the paper): ``lo`` increases in ``e`` and in
    ``eta_1``; ``hi`` decreases in ``e`` and increases in ``eta_1``.  Hence the
    extremes are ``lo(e_L, eta_L)`` and ``hi(e_L, eta_U)``; ``e_U`` is not
    needed for the two-sided box (it would be for a *lower* bound on ``hi``).
    """
    e_L = np.clip(np.asarray(e_L, float), 0, 1)
    eta_L = np.asarray(eta_L, float); eta_U = np.asarray(eta_U, float)
    p_L, p_U = expit(eta_L), expit(eta_U)
    log_g = np.log(gamma) if np.isfinite(gamma) else np.inf
    if np.isfinite(gamma):
        lo = e_L * p_L + (1 - e_L) * expit(eta_L - log_g)
        hi = e_L * p_U + (1 - e_L) * expit(eta_U + log_g)
    else:
        lo = e_L * p_L
        hi = e_L * p_U + (1 - e_L)
    if direction == "censored-lower":
        hi = p_U
    elif direction == "censored-higher":
        lo = p_L
    elif direction != "two-sided":
        raise ValueError(f"unknown direction {direction!r}")
    return InflatedBox(lo=np.clip(lo, 0, 1), hi=np.clip(hi, 0, 1))
