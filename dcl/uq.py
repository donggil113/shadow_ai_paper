"""Corollary: aleatoric / epistemic decomposition is systematically biased
under decision censoring.

The standard decomposition of predictive uncertainty for an ensemble or
posterior ``{p_m(x)}`` is

    Total(x)      = H( mean_m p_m(x) )
    Aleatoric(x)  = mean_m H( p_m(x) )
    Epistemic(x)  = Total - Aleatoric   ( = mutual information I(Y; theta | x) ).

Under selective labels every one of those quantities is computed from models fit
to the **selected** sub-population, so they concentrate on ``p1(x) = P(Y=1|x,T=1)``
rather than on the deployment quantity ``p(x) = P(Y=1|x)``.  Two consequences:

1. **The bias is invisible.**  ``p1(x)`` always lies inside the identified set
   ``[p_lo(x), p_hi(x)]`` (Lemma: the box is built by tilting ``p1`` symmetrically
   on the log-odds scale and mixing with weight ``1 - e(x)``).  So no
   goodness-of-fit or calibration test on observed data can refute the naive
   decomposition -- it is not *wrong on the data*, it is *answering a different
   question*.

2. **Epistemic uncertainty vanishes; identification uncertainty does not.**  As
   ``n -> inf`` the ensemble spread collapses and ``Epistemic(x) -> 0``, while the
   identified set retains width ``delta(x) = p_hi(x) - p_lo(x) > 0`` whenever
   ``Gamma > 1`` and ``e(x) < 1``.  Any two-way decomposition is therefore
   asymptotically overconfident by a non-vanishing amount.

We therefore use the credal-set (imprecise-probability) decomposition on the
*union* of the sampling credal set and the identification box:

    Total(x)      = max_{p in C(x)} H(p)          (upper entropy)
    Aleatoric(x)  = min_{p in C(x)} H(p)          (lower entropy)
    Epistemic(x)  = Total - Aleatoric,

and split ``Epistemic`` into the part attributable to finite data (which shrinks)
and the part attributable to decision censoring (which does not):

    Censoring(x)  = entropy range over [p_lo(x), p_hi(x)]                    (6)
    Sampling(x)   = Epistemic(x) - Censoring(x)  (>= 0 by construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "binary_entropy", "entropy_range", "naive_decomposition",
    "UncertaintyDecomposition", "three_way_decomposition",
    "aleatoric_identified_interval",
]

_EPS = 1e-12


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """Binary Shannon entropy in nats."""
    p = np.clip(np.asarray(p, float), _EPS, 1.0 - _EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def entropy_range(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sharp ``[min H, max H]`` over an interval ``[lo, hi]``.

    ``H`` is concave and maximised at ``1/2``, so the maximum is at the point of
    the interval closest to ``1/2`` and the minimum at whichever endpoint is
    further from it.
    """
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    closest = np.clip(0.5, lo, hi)
    h_max = binary_entropy(closest)
    h_min = np.minimum(binary_entropy(lo), binary_entropy(hi))
    return h_min, h_max


@dataclass
class UncertaintyDecomposition:
    total: np.ndarray
    aleatoric: np.ndarray
    epistemic: np.ndarray
    censoring: np.ndarray | None = None
    sampling: np.ndarray | None = None

    def summary(self) -> dict:
        out = {"total": float(np.mean(self.total)),
               "aleatoric": float(np.mean(self.aleatoric)),
               "epistemic": float(np.mean(self.epistemic))}
        if self.censoring is not None:
            out["censoring"] = float(np.mean(self.censoring))
        if self.sampling is not None:
            out["sampling"] = float(np.mean(self.sampling))
        return out


def naive_decomposition(member_probs: np.ndarray) -> UncertaintyDecomposition:
    """Standard two-way decomposition from ensemble member probabilities.

    ``member_probs`` has shape ``(M, n)``.
    """
    P = np.asarray(member_probs, float)
    mean_p = P.mean(axis=0)
    total = binary_entropy(mean_p)
    aleatoric = binary_entropy(P).mean(axis=0)
    return UncertaintyDecomposition(total=total, aleatoric=aleatoric,
                                    epistemic=total - aleatoric)


def three_way_decomposition(
    member_probs: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> UncertaintyDecomposition:
    """Credal decomposition over the union of ensemble spread and the DCSM box.

    The credal set at ``x`` is the interval hull of the ensemble members *and* the
    identified set ``[p_lo(x), p_hi(x)]``.
    """
    P = np.asarray(member_probs, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    c_lo = np.minimum(P.min(axis=0), lo)
    c_hi = np.maximum(P.max(axis=0), hi)
    h_min, h_max = entropy_range(c_lo, c_hi)
    cen_min, cen_max = entropy_range(lo, hi)
    censoring = cen_max - cen_min
    epistemic = h_max - h_min
    return UncertaintyDecomposition(
        total=h_max, aleatoric=h_min, epistemic=epistemic,
        censoring=censoring, sampling=np.maximum(epistemic - censoring, 0.0),
    )


def aleatoric_identified_interval(lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float]:
    """Sharp identified interval for the *population mean* aleatoric entropy.

    ``E[H(p(X))]`` with ``p`` ranging over the box.  Because the objective is
    separable and the box is a product, the sharp interval is just the mean of
    the pointwise sharp interval.
    """
    h_min, h_max = entropy_range(lo, hi)
    return float(np.mean(h_min)), float(np.mean(h_max))
