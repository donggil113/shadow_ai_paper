"""Theorem 2: the decision-censored minimax learning objective.

We do not want to minimise the *observed* risk -- the risk on the sub-population
the incumbent policy happened to label.  We want the risk on the deployment
population, which is only partially identified.  DCL therefore minimises the
worst case over the identified set:

    f_hat = argmin_{f in F}   sup_{P in I_Gamma(P_obs)}  R_P(f).             (2)

(a) Exact reduction.  For a pointwise loss ``l(z, y)`` the risk is *affine* in
``p(x)``, so the inner supremum decouples across ``x`` and is attained at a
vertex of the box.  Using ``max(a,b) = (a+b)/2 + |a-b|/2``,

    Rbar_Gamma(f) = E[ ptilde(x) l1 + (1 - ptilde(x)) l0 ]                   (3)
                  + (1/2) E[ delta(x) * g_l(f(x)) ],

    ptilde = (p_lo + p_hi)/2,  delta = p_hi - p_lo,  g_l(z) = |l(z,1) - l(z,0)|,

i.e. **worst-case risk = risk at the midpoint model + a margin penalty weighted
by the local identification width**.  For the logistic loss ``l(z,1) - l(z,0) =
-z`` exactly, so the penalty is ``(1/2) E[delta(x) |f(x)|]``: an L1 penalty on
the *score*, weighted by how censored each region is.

(b) Convexity -- no relaxation is needed.  ``Rbar_Gamma`` is a pointwise
supremum of functions that are convex in ``z``, hence convex in ``z``; over any
convex class ``F`` (linear models, RKHS balls, convex combinations) problem (2)
is therefore an *exactly convex* program.  This is in sharp contrast to
f-divergence / Wasserstein DRO, where the inner supremum is tractable only after
a duality step, and to the non-convexity that appears as soon as the metric is a
*ranking* metric (Theorem 1 / :mod:`dcl.ranking`).
   What is *not* convex in general is the penalty term of (3) taken on its own:
that term is convex iff the label gap ``g_l`` is convex.  Logistic, squared and
exponential losses have convex gaps (so (3) can be implemented modularly as
"any convex risk + a weighted L1 score penalty"); the hinge loss does not, and
must be implemented through the max form.  :func:`gap_convex_envelope` provides
the envelope used to quantify that modular-implementation gap.

(c) Saddle point.  ``R_P(f)`` is convex in ``f`` and linear -- hence concave --
in ``p``, over a convex compact box, so Sion's minimax theorem gives
``inf sup = sup inf`` and a saddle point ``(f_hat, P*)``: the DCL solution is the
Bayes predictor for the *least favourable* member of the identified set, whose
closed form is :func:`least_favourable_p`.

(d) Bayes act.  Minimising (3) pointwise for the logistic loss gives

    f*(x) = logit(p_lo(x))   if p_lo(x) > 1/2
          = logit(p_hi(x))   if p_hi(x) < 1/2
          = 0                otherwise,                                      (4)

the **interval-shrunk logit**: log-odds pulled toward the decision boundary by
the identification width, and pinned to 0 on the *abstention band*
``{x : p_lo(x) <= 1/2 <= p_hi(x)}`` where the data cannot resolve the sign of the
optimal decision.  ``Gamma = 1`` recovers ``logit p(x)``.

(e) Budgeted DCSM.  Pointwise-Gamma is conservative: it lets *every* unit be
maximally tilted at once.  :func:`budgeted_worstcase_risk` implements the
variant that additionally caps the average tilt, ``E|log a(X)| <= B``.  There the
inner supremum no longer decouples and is solved by its one-dimensional dual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from .sensitivity import IdentifiedSet, expit, logit

__all__ = [
    "Loss", "LOSSES", "get_loss",
    "worstcase_risk", "worstcase_risk_decomposition", "worstcase_risk_pointwise",
    "least_favourable_p", "dcl_bayes_score", "minimax_regret", "abstention_band",
    "gap_convex_envelope", "gap_is_convex",
    "budgeted_worstcase_risk",
    "minimax_value_alternating",
]


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


def _weights(w, n):
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, float)
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Loss registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Loss:
    """A margin loss ``l(z, y)`` together with metadata used by Theorem 2."""

    name: str
    l1: Callable[[np.ndarray], np.ndarray]    # l(z, 1)
    l0: Callable[[np.ndarray], np.ndarray]    # l(z, 0)
    gap_convex: bool      # is the *penalty term* |l(.,1) - l(.,0)| convex?
    gap_affine: bool      # is l(.,1) - l(.,0) affine in z?  (=> penalty is c*|z|)

    def __call__(self, z, y):
        z = np.asarray(z, float); y = np.asarray(y, float)
        return y * self.l1(z) + (1.0 - y) * self.l0(z)

    def gap(self, z):
        z = np.asarray(z, float)
        return np.abs(self.l1(z) - self.l0(z))


LOSSES = {
    "logistic": Loss("logistic", lambda z: _softplus(-z), lambda z: _softplus(z),
                     gap_convex=True, gap_affine=True),
    "squared": Loss("squared", lambda z: (z - 1.0) ** 2, lambda z: z ** 2,
                    gap_convex=True, gap_affine=True),
    "exponential": Loss("exponential",
                        lambda z: np.exp(-np.clip(z, -30, 30)),
                        lambda z: np.exp(np.clip(z, -30, 30)),
                        gap_convex=True, gap_affine=False),
    "hinge": Loss("hinge", lambda z: np.maximum(0.0, 1.0 - z),
                  lambda z: np.maximum(0.0, 1.0 + z),
                  gap_convex=False, gap_affine=False),
}


def get_loss(loss):
    if isinstance(loss, Loss):
        return loss
    if loss not in LOSSES:
        raise KeyError(f"unknown loss {loss!r}; available: {sorted(LOSSES)}")
    return LOSSES[loss]


def gap_is_convex(loss, zgrid=None, tol=1e-8) -> bool:
    """Numerically certify convexity of the penalty term ``g_l`` (part (b))."""
    L = get_loss(loss)
    z = np.linspace(-6, 6, 4001) if zgrid is None else np.asarray(zgrid, float)
    g = L.gap(z)
    sec = g[2:] - 2 * g[1:-1] + g[:-2]
    return bool(np.all(sec >= -tol * max(1.0, float(np.abs(g).max()))))


# --------------------------------------------------------------------------- #
# Worst-case risk (3)
# --------------------------------------------------------------------------- #
def worstcase_risk_pointwise(scores, lo, hi, loss="logistic") -> np.ndarray:
    """Per-unit worst-case loss ``max_{p in [lo, hi]} p l1 + (1-p) l0``."""
    L = get_loss(loss)
    z = np.asarray(scores, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    a = hi * L.l1(z) + (1.0 - hi) * L.l0(z)
    b = lo * L.l1(z) + (1.0 - lo) * L.l0(z)
    return np.maximum(a, b)


def worstcase_risk_decomposition(scores, lo, hi, loss="logistic", weights=None):
    """``(midpoint_risk, censoring_penalty)`` -- the two terms of (3)."""
    L = get_loss(loss)
    z = np.asarray(scores, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    w = _weights(weights, z.shape[0])
    mid = 0.5 * (lo + hi)
    base = float(w @ (mid * L.l1(z) + (1.0 - mid) * L.l0(z)))
    pen = float(0.5 * (w @ ((hi - lo) * L.gap(z))))
    return base, pen


def worstcase_risk(scores, lo, hi, loss="logistic", weights=None) -> float:
    """``Rbar_Gamma(f)`` -- the exact worst-case risk over the identified set."""
    w = _weights(weights, np.asarray(scores).shape[0])
    return float(w @ worstcase_risk_pointwise(scores, lo, hi, loss))


def least_favourable_p(scores, lo, hi, loss="logistic") -> np.ndarray:
    """Adversary's best response: the box vertex attaining the sup in (2).

    ``p*(x) = p_hi(x)`` wherever the positive-label loss is the larger one (for
    the logistic loss: wherever ``f(x) < 0``), and ``p_lo(x)`` elsewhere.  The
    adversary raises prevalence exactly where the model predicts *negative* --
    precisely the failure mode selective labelling induces in practice.
    """
    L = get_loss(loss)
    z = np.asarray(scores, float)
    return np.where(L.l1(z) >= L.l0(z), np.asarray(hi, float), np.asarray(lo, float))


# --------------------------------------------------------------------------- #
# Bayes act (4)
# --------------------------------------------------------------------------- #
def dcl_bayes_score(lo, hi, criterion: str = "risk") -> np.ndarray:
    """Closed-form pointwise DCL score for the logistic loss.

    ``criterion="risk"`` -- minimax **risk**, equation (4): the interval-shrunk
    logit with an abstention band.  This is the right rule when the loss itself
    is what must be controlled.

    ``criterion="regret"`` -- minimax **excess** risk (regret) relative to the
    best predictor for whichever member of the identified set turns out to be
    true.  Since ``R_p(f) - min_g R_p(g) = E[KL(p || sigma(f))]`` for the
    logistic loss, the rule minimises ``max_{p in [lo, hi]} KL(p || q)``.
    ``KL(. || q)`` is convex in ``q``, so the optimum equalises the two endpoints
    and admits the closed form (Proposition 3)

        logit q*(x) = ( H(p_lo(x)) - H(p_hi(x)) ) / ( p_hi(x) - p_lo(x) ),

    a difference quotient of the binary entropy -- a discrete version of
    ``-H'(p) = logit(p)``, to which it degenerates as the box collapses.  Unlike
    (4) it has **no abstention band**: it stays a genuine probability estimate,
    which is usually what a downstream decision rule wants.
    """
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    if criterion == "risk":
        out = np.zeros_like(lo)
        pos = lo > 0.5
        neg = hi < 0.5
        out[pos] = logit(lo[pos])
        out[neg] = logit(hi[neg])
        return out
    if criterion == "regret":
        from .uq import binary_entropy
        d = hi - lo
        out = np.where(d > 1e-9,
                       (binary_entropy(lo) - binary_entropy(hi)) / np.where(d > 1e-9, d, 1.0),
                       logit(np.where(d > 1e-9, 0.5, 0.5 * (lo + hi))))
        return out
    raise ValueError("criterion must be 'risk' or 'regret'")


def minimax_regret(scores, lo, hi, weights=None) -> float:
    """``max_{p in box} [ R_p(f) - inf_g R_p(g) ] = E[ max_p KL(p || sigma(f)) ]``.

    The excess risk DCL is certified against, in nats.  ``KL(. || q)`` is convex
    in its first argument, so the inner max is at an endpoint of the interval.
    """
    z = np.asarray(scores, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    w = _weights(weights, z.shape[0])
    q = np.clip(expit(z), 1e-12, 1 - 1e-12)

    def kl(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    return float(w @ np.maximum(kl(lo), kl(hi)))


def abstention_band(lo, hi) -> np.ndarray:
    """Mask of units whose identified set straddles 1/2 (DCL predicts 0 there)."""
    return (np.asarray(lo, float) <= 0.5) & (np.asarray(hi, float) >= 0.5)


# --------------------------------------------------------------------------- #
# Convex envelope of the penalty term (part (b))
# --------------------------------------------------------------------------- #
def gap_convex_envelope(loss, radius: float = 6.0, n_grid: int = 4001):
    """Convex envelope ``g**`` of the penalty ``g_l`` on ``[-radius, radius]``.

    Returns ``(g_star_star, max_gap)``.  ``(1/2) E[delta] * max_gap`` bounds the
    error incurred by implementing (3) *modularly* -- convex risk plus convex
    penalty -- instead of through the exact max form, for losses whose gap is
    non-convex (the hinge).  For logistic / squared / exponential the gap is
    already convex and ``max_gap = 0``.
    """
    L = get_loss(loss)
    z = np.linspace(-radius, radius, n_grid)
    g = L.gap(z)
    hull: list[int] = []
    for i in range(n_grid):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (g[b] - g[a]) * (z[i] - z[a]) >= (g[i] - g[a]) * (z[b] - z[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    env = np.interp(z, z[hull], g[hull])
    max_gap = float(np.max(g - env))

    def g_star_star(zz):
        return np.interp(np.clip(np.asarray(zz, float), -radius, radius), z, env)

    return g_star_star, max_gap


# --------------------------------------------------------------------------- #
# Budgeted DCSM (part (e))
# --------------------------------------------------------------------------- #
def budgeted_worstcase_risk(
    scores, p1, e, gamma: float, budget: float,
    loss: str = "logistic", weights=None, n_tilt: int = 129,
    lam_hi: float = 1e4, n_bisect: int = 60,
):
    """Worst-case risk under DCSM(Gamma) *plus* an average-tilt budget.

    Constraint set: ``|log a(x)| <= log Gamma`` for all x **and**
    ``E|log a(X)| <= budget``.  The supremum no longer decouples, so we dualise:

        sup_a  E[ r(x, a) ]  s.t.  E|log a| <= B
          = min_{lam >= 0}  lam B + E[ sup_t { r(x, t) - lam |t| } ],

    which decouples pointwise given ``lam``; ``lam`` is then found by bisection on
    the budget constraint.  Strong duality holds (the primal is a concave-valued
    sup over a convex compact set with a Slater point ``a == 1``).

    Returns ``(risk, lam, mean_abs_tilt)``.
    """
    L = get_loss(loss)
    z = np.asarray(scores, float)
    p1 = np.asarray(p1, float); e = np.asarray(e, float)
    w = _weights(weights, z.shape[0])
    log_g = np.log(gamma)
    if budget >= log_g - 1e-12:          # budget inactive -> pointwise problem
        from .sensitivity import outcome_bounds
        lo, hi = outcome_bounds(p1, e, gamma)
        return worstcase_risk(z, lo, hi, loss, weights), 0.0, log_g

    t_grid = np.linspace(-log_g, log_g, n_tilt)                    # (K,)
    eta1 = logit(p1)
    # p(x, t) for every unit and tilt level: (n, K)
    p_mat = e[:, None] * p1[:, None] + (1.0 - e[:, None]) * expit(eta1[:, None] + t_grid[None, :])
    l1 = L.l1(z)[:, None]; l0 = L.l0(z)[:, None]
    r_mat = p_mat * l1 + (1.0 - p_mat) * l0                        # (n, K)
    abs_t = np.abs(t_grid)[None, :]

    def dual(lam):
        obj = r_mat - lam * abs_t
        k = np.argmax(obj, axis=1)
        idx = (np.arange(len(z)), k)
        return float(w @ r_mat[idx]), float(w @ np.abs(t_grid)[k])

    lo_l, hi_l = 0.0, lam_hi
    for _ in range(n_bisect):
        mid = 0.5 * (lo_l + hi_l)
        _, used = dual(mid)
        if used > budget:
            lo_l = mid
        else:
            hi_l = mid
    lam = 0.5 * (lo_l + hi_l)
    risk, used = dual(lam)
    return risk, lam, used


# --------------------------------------------------------------------------- #
# Minimax verification (Sion saddle point)
# --------------------------------------------------------------------------- #
def minimax_value_alternating(iset: IdentifiedSet, n_iter: int = 400, loss: str = "logistic"):
    """Fictitious play on (2); must agree with (4) + :func:`worstcase_risk`."""
    L = get_loss(loss)
    lo, hi = iset.lo, iset.hi
    p = iset.mid.copy()
    p_bar = np.zeros_like(p)
    for t in range(1, n_iter + 1):
        z = logit(p)
        p_new = least_favourable_p(z, lo, hi, L)
        p = p + (p_new - p) / (t + 1)
        p_bar += (p_new - p_bar) / t
    z_star = dcl_bayes_score(lo, hi)
    return worstcase_risk(z_star, lo, hi, L), z_star, p_bar
