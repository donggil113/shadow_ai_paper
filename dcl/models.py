"""The DCL learner.

Two variants, matching the two theorems:

``DCLPlugin``
    Non-parametric.  Estimate the nuisances, form the identified set, and apply
    the closed-form Bayes act (4).  This is the population-optimal rule of
    Theorem 2(d) with estimated nuisances, and is what we recommend in practice.

``DCLParametric``
    Minimise the *empirical* worst-case risk over a restricted class ``F``
    (linear or MLP scores).  This is the object Theorem 4 bounds: the excess
    worst-case risk of the empirical minimiser over ``F``.  For convex ``F`` the
    program is convex (Theorem 2(b)), so L-BFGS finds the global optimum; the
    MLP variant is there to show the objective is a drop-in replacement for the
    log-loss in a standard training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .nuisance import CrossFitNuisance, NuisanceEstimates
from .objectives import dcl_bayes_score
from .sensitivity import IdentifiedSet

__all__ = ["DCLPlugin", "DCLParametric"]


@dataclass
class DCLPlugin:
    """Plug-in DCL: nuisances -> identified set -> interval-shrunk logit (4)."""

    gamma: float = 2.0
    nuisance_model: str = "gbm"
    n_folds: int = 5
    seed: int = 0
    nuisance_: Optional[NuisanceEstimates] = field(default=None, init=False)
    box_: Optional[IdentifiedSet] = field(default=None, init=False)
    _cf: Optional[CrossFitNuisance] = field(default=None, init=False)

    def fit(self, X, T, Y_obs, nuisance: NuisanceEstimates | None = None,
            cross_fitter: CrossFitNuisance | None = None):
        if cross_fitter is not None:
            self._cf = cross_fitter
        if nuisance is None:
            self._cf = CrossFitNuisance(model=self.nuisance_model,
                                        n_folds=self.n_folds, seed=self.seed)
            nuisance = self._cf.fit_predict(X, T, Y_obs)
        self.nuisance_ = nuisance
        self.box_ = nuisance.box(self.gamma)
        self.train_scores_ = dcl_bayes_score(self.box_.lo, self.box_.hi)
        return self

    def decision_function(self, X=None):
        """In-sample scores, or out-of-sample scores when ``X`` is given."""
        if X is None:
            return self.train_scores_
        if self._cf is None:
            raise RuntimeError(
                "out-of-sample scoring needs the fitted nuisance models: pass "
                "cross_fitter=... to fit(), or call fit() without `nuisance`.")
        nu = self._cf.predict(X)
        box = nu.box(self.gamma)
        return dcl_bayes_score(box.lo, box.hi)

    def predict_proba_bounds(self, X=None):
        """Return the identified interval ``[p_lo, p_hi]`` -- the honest output."""
        box = self.box_ if X is None else self._cf.predict(X).box(self.gamma)
        return box.lo, box.hi

    @property
    def abstention_rate(self) -> float:
        """Fraction of units whose identified set straddles 1/2."""
        return float(np.mean((self.box_.lo <= 0.5) & (self.box_.hi >= 0.5)))


@dataclass
class DCLParametric:
    """Empirical worst-case risk minimisation over a restricted class."""

    gamma: float = 2.0
    arch: str = "linear"          # "linear" | "mlp"
    loss: str = "logistic"
    hidden: tuple = (64, 32)
    l2: float = 1e-4
    max_iter: int = 500
    lr: float = 0.05
    seed: int = 0
    score_clip: Optional[float] = None   # enforce ||f||_inf <= clip (Theorem 4)

    def fit(self, X, box: IdentifiedSet):
        import torch

        torch.manual_seed(self.seed)
        Xt = torch.tensor(np.asarray(X, float), dtype=torch.float32)
        lo = torch.tensor(box.lo, dtype=torch.float32)
        hi = torch.tensor(box.hi, dtype=torch.float32)
        mid = 0.5 * (lo + hi)
        delta = hi - lo

        if self.arch == "linear":
            self.net_ = torch.nn.Linear(Xt.shape[1], 1)
        else:
            layers, d = [], Xt.shape[1]
            for h in self.hidden:
                layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
                d = h
            layers += [torch.nn.Linear(d, 1)]
            self.net_ = torch.nn.Sequential(*layers)

        sp = torch.nn.functional.softplus

        def objective():
            z = self.net_(Xt).squeeze(-1)
            if self.score_clip is not None:
                z = torch.clamp(z, -self.score_clip, self.score_clip)
            if self.loss == "logistic":
                base = mid * sp(-z) + (1 - mid) * sp(z)
                pen = 0.5 * delta * z.abs()
            else:
                raise NotImplementedError(
                    "the torch path implements the logistic loss; for other "
                    "losses use the numpy objective in dcl.objectives, which "
                    "evaluates the exact max form for every registered loss")
            reg = self.l2 * sum((p ** 2).sum() for p in self.net_.parameters())
            return (base + pen).mean() + reg

        if self.arch == "linear":
            # The program is convex (Theorem 6b), so L-BFGS reaches the global
            # optimum quickly; the previous 1e-9 gradient tolerance made it burn
            # the full iteration budget chasing noise.
            opt = torch.optim.LBFGS(self.net_.parameters(), lr=1.0,
                                    max_iter=self.max_iter,
                                    tolerance_grad=1e-7, tolerance_change=1e-10,
                                    line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad(); loss = objective(); loss.backward(); return loss

            opt.step(closure)
        else:
            opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
            for _ in range(self.max_iter):
                opt.zero_grad(); loss = objective(); loss.backward(); opt.step()

        with torch.no_grad():
            self.train_scores_ = self.net_(Xt).squeeze(-1).numpy().astype(float)
            if self.score_clip is not None:
                self.train_scores_ = np.clip(self.train_scores_,
                                             -self.score_clip, self.score_clip)
        return self

    def decision_function(self, X=None):
        if X is None:
            return self.train_scores_
        import torch
        with torch.no_grad():
            z = self.net_(torch.tensor(np.asarray(X, float),
                                       dtype=torch.float32)).squeeze(-1).numpy()
        return np.clip(z, -self.score_clip, self.score_clip) if self.score_clip else z
