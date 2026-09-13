"""Common container for selective-labels datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["SelectiveLabelsDataset", "standardise"]


def standardise(X: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    ref = X if ref is None else ref
    mu = np.nanmean(ref, axis=0)
    sd = np.nanstd(ref, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


@dataclass
class SelectiveLabelsDataset:
    """A selective-labels problem instance.

    Attributes
    ----------
    X : (n, d)
        Covariates available to the *learner*.  Never contains the decision
        maker's private information ``S``, and never contains post-decision
        variables.
    T : (n,)
        Decision / label-revelation indicator.  ``Y`` is observed iff ``T == 1``.
    Y_obs : (n,)
        Outcome where ``T == 1``, ``np.nan`` elsewhere.  **This is all a method is
        allowed to see.**
    Y_full : (n,) or None
        Ground-truth outcome for every unit, including censored ones.  Available
        only in synthetic / semi-synthetic designs and in COMPAS; used *solely*
        for evaluation, never for fitting.
    Z : (n,) or None
        Decision-maker identity (leniency instrument) when available.
    oracle : dict
        Known population quantities (``p``, ``p1``, ``p0``, ``e``, ``gamma0``) where
        the design makes them available.
    """

    X: np.ndarray
    T: np.ndarray
    Y_obs: np.ndarray
    feature_names: List[str]
    name: str
    Y_full: Optional[np.ndarray] = None
    Z: Optional[np.ndarray] = None
    oracle: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        n = self.X.shape[0]
        for nm, arr in [("T", self.T), ("Y_obs", self.Y_obs)]:
            if arr.shape[0] != n:
                raise ValueError(f"{nm} has {arr.shape[0]} rows, X has {n}")
        if not np.all(np.isnan(self.Y_obs[self.T == 0])):
            raise ValueError("Y_obs must be NaN wherever T == 0 (labels are censored)")
        if np.any(np.isnan(self.Y_obs[self.T == 1])):
            raise ValueError("Y_obs must be observed wherever T == 1")

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d(self) -> int:
        return self.X.shape[1]

    @property
    def selection_rate(self) -> float:
        return float(np.mean(self.T))

    @property
    def observed_prevalence(self) -> float:
        return float(np.nanmean(self.Y_obs[self.T == 1]))

    @property
    def true_prevalence(self) -> Optional[float]:
        return None if self.Y_full is None else float(np.mean(self.Y_full))

    def split(self, test_size: float = 0.3, seed: int = 0
              ) -> Tuple["SelectiveLabelsDataset", "SelectiveLabelsDataset"]:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n)
        cut = int(round(self.n * (1 - test_size)))
        return self._subset(idx[:cut]), self._subset(idx[cut:])

    def _subset(self, idx: np.ndarray) -> "SelectiveLabelsDataset":
        oracle = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (self.n,) else v)
                  for k, v in self.oracle.items()}
        return SelectiveLabelsDataset(
            X=self.X[idx], T=self.T[idx], Y_obs=self.Y_obs[idx],
            feature_names=self.feature_names, name=self.name,
            Y_full=None if self.Y_full is None else self.Y_full[idx],
            Z=None if self.Z is None else self.Z[idx],
            oracle=oracle, notes=self.notes,
        )

    def describe(self) -> str:
        lines = [f"{self.name}: n={self.n}, d={self.d}",
                 f"  selection rate P(T=1)   = {self.selection_rate:.3f}",
                 f"  observed prevalence     = {self.observed_prevalence:.4f}"]
        if self.Y_full is not None:
            tp = self.true_prevalence
            cen = self.Y_full[self.T == 0]
            lines.append(f"  TRUE prevalence         = {tp:.4f}  "
                         f"(censored group: {np.mean(cen):.4f})")
            sel = self.Y_full[self.T == 1]
            o1 = np.mean(sel) / max(1e-9, 1 - np.mean(sel))
            o0 = np.mean(cen) / max(1e-9, 1 - np.mean(cen))
            lines.append(f"  marginal odds ratio p0/p1 = {o0 / max(o1,1e-9):.3f}")
        if "gamma0" in self.oracle:
            lines.append(f"  realised Gamma_0        = {self.oracle['gamma0']:.3f}")
        if self.Z is not None:
            lines.append(f"  decision makers         = {len(np.unique(self.Z))}")
        return "\n".join(lines)
