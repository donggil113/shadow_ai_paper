"""Cross-fitted estimation of the two identifiable nuisances.

    e(x)  = P(T = 1 | X = x)              from all units
    p1(x) = P(Y = 1 | X = x, T = 1)       from the selected units only

Both feed the identified set, so *calibration matters more than discrimination*:
the bounds live on the probability scale, and a miscalibrated ``p1`` shifts the
whole box.  We therefore wrap every learner in isotonic calibration and use
K-fold cross-fitting, so no unit's bound is built from a model that saw its own
label -- the same reason cross-fitting is standard in double machine learning.

The generalisation bound is stated on the *logit* scale because ``Gamma`` acts there as a pure
translation: ``logit p_hi = logit p1 + log Gamma``.  Estimation error in ``p1``
therefore propagates to the bounds without amplification, which is what keeps
the generalisation bound linear (rather than exponential) in ``log Gamma``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sensitivity import IdentifiedSet, identified_set

__all__ = ["NuisanceEstimates", "CrossFitNuisance"]


def _make_base(model: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if model == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.07, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0, random_state=seed)
    if model == "logistic":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=1.0))
    if model == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=600, random_state=seed))
    raise KeyError(f"unknown nuisance model {model!r}")


@dataclass
class NuisanceEstimates:
    """Out-of-fold ``e_hat`` and ``p1_hat`` plus the derived identified set."""

    e: np.ndarray
    p1: np.ndarray

    def box(self, gamma: float, direction: str = "two-sided") -> IdentifiedSet:
        return identified_set(self.p1, self.e, gamma, direction)


@dataclass
class CrossFitNuisance:
    """K-fold cross-fitted, isotonically calibrated nuisance estimator."""

    model: str = "gbm"
    n_folds: int = 5
    calibrate: bool = True
    clip: float = 1e-3
    seed: int = 0
    _fitted_e: list = field(default_factory=list, init=False)
    _fitted_p1: list = field(default_factory=list, init=False)

    def _wrap(self, base):
        if not self.calibrate:
            return base
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(base, method="isotonic", cv=3)

    def fit_predict(
        self, X: np.ndarray, T: np.ndarray, Y_obs: np.ndarray
    ) -> NuisanceEstimates:
        """Return out-of-fold estimates of ``e`` and ``p1`` for every row of ``X``."""
        from sklearn.model_selection import StratifiedKFold

        X = np.asarray(X, float)
        T = np.asarray(T, float)
        n = X.shape[0]
        e_hat = np.full(n, np.nan)
        p1_hat = np.full(n, np.nan)

        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed)
        for tr, te in skf.split(X, T):
            m = self._wrap(_make_base(self.model, self.seed))
            m.fit(X[tr], T[tr])
            e_hat[te] = m.predict_proba(X[te])[:, 1]
            self._fitted_e.append(m)

        sel = np.flatnonzero(T == 1)
        ysel = np.asarray(Y_obs, float)[sel]
        if len(np.unique(ysel)) < 2:
            p1_hat[:] = float(np.mean(ysel))
        else:
            skf2 = StratifiedKFold(self.n_folds, shuffle=True, random_state=self.seed + 1)
            # fit on folds of the SELECTED units, predict for ALL units
            preds = np.zeros((self.n_folds, n))
            for k, (tr, _te) in enumerate(skf2.split(X[sel], ysel)):
                m = self._wrap(_make_base(self.model, self.seed + k))
                m.fit(X[sel][tr], ysel[tr])
                preds[k] = m.predict_proba(X)[:, 1]
                self._fitted_p1.append(m)
            # out-of-fold for selected units, fold-average for censored ones
            p1_hat = preds.mean(axis=0)
            for k, (_tr, te) in enumerate(skf2.split(X[sel], ysel)):
                others = [j for j in range(self.n_folds) if j != k]
                p1_hat[sel[te]] = preds[others][:, sel[te]].mean(axis=0)

        c = self.clip
        return NuisanceEstimates(e=np.clip(e_hat, c, 1 - c),
                                 p1=np.clip(p1_hat, c, 1 - c))

    def predict(self, X: np.ndarray) -> NuisanceEstimates:
        """Average the fitted fold models on new data."""
        X = np.asarray(X, float)
        e = np.mean([m.predict_proba(X)[:, 1] for m in self._fitted_e], axis=0)
        p1 = np.mean([m.predict_proba(X)[:, 1] for m in self._fitted_p1], axis=0)
        c = self.clip
        return NuisanceEstimates(e=np.clip(e, c, 1 - c), p1=np.clip(p1, c, 1 - c))
