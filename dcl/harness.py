"""Experiment harness: fit on train, evaluate honestly on held-out units.

Two traps this file exists to avoid.

1. **In-sample evaluation.**  Under selective labels, a model fit on the
   selected units and scored on the same units looks *better* on deployment
   metrics than an oracle fit on the complete labels, purely through
   memorisation.  Every number in the paper is out-of-sample.

2. **Leaking the decision maker.**  The learner sees ``X`` only.  The relevant
   propensity is therefore ``P(T=1|X)`` marginal over the random assignment of
   cases to decision makers; the decision-maker identity is used *only* by the
   falsification test, which is explicitly an instrument-based procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .baselines import (AIPWLearner, ERMObserved, HeckmanProbit,
                        ImputationLearner, IPWLearner, OracleLearner)
from .data.base import SelectiveLabelsDataset
from .evaluation import bounds_report, deployment_metrics, observed_metrics
from .nuisance import CrossFitNuisance, NuisanceEstimates
from .objectives import dcl_bayes_score, minimax_regret

__all__ = ["Fold", "prepare_fold", "run_methods", "DEFAULT_GAMMAS"]

DEFAULT_GAMMAS = (1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass
class Fold:
    train: SelectiveLabelsDataset
    test: SelectiveLabelsDataset
    nu_train: NuisanceEstimates
    nu_test: NuisanceEstimates
    cf: CrossFitNuisance


def prepare_fold(ds: SelectiveLabelsDataset, test_size: float = 0.35,
                 seed: int = 0, nuisance_model: str = "gbm",
                 n_folds: int = 5) -> Fold:
    tr, te = ds.split(test_size=test_size, seed=seed)
    cf = CrossFitNuisance(model=nuisance_model, n_folds=n_folds, seed=seed)
    nu_tr = cf.fit_predict(tr.X, tr.T, tr.Y_obs)
    nu_te = cf.predict(te.X)
    return Fold(train=tr, test=te, nu_train=nu_tr, nu_test=nu_te, cf=cf)


def _row(name: str, scores_te: np.ndarray, fold: Fold, gamma_report: float,
         extra: Optional[Dict] = None) -> Dict:
    te = fold.test
    row: Dict[str, object] = {"method": name}
    if te.Y_full is not None:
        row.update(deployment_metrics(scores_te, te.Y_full))
    row.update(observed_metrics(scores_te, te.T, te.Y_obs))
    box = fold.nu_test.box(gamma_report)
    row.update(bounds_report(scores_te, box.lo, box.hi))
    row["minimax_regret"] = minimax_regret(scores_te, box.lo, box.hi)
    if extra:
        row.update(extra)
    return row


def run_methods(
    ds: SelectiveLabelsDataset,
    gammas=DEFAULT_GAMMAS,
    gamma_report: float = 2.0,
    seed: int = 0,
    test_size: float = 0.35,
    nuisance_model: str = "gbm",
    include_heckman: bool = True,
    include_oracle: bool = True,
    direction: str = "two-sided",
) -> List[Dict]:
    """Fit every method on the training split and evaluate on the test split."""
    fold = prepare_fold(ds, test_size=test_size, seed=seed,
                        nuisance_model=nuisance_model)
    tr, te = fold.train, fold.test
    rows: List[Dict] = []

    def fit_eval(name, learner, **kw):
        learner.fit(tr.X, tr.T, tr.Y_obs, **kw)
        rows.append(_row(name, learner.decision_function(te.X), fold, gamma_report))

    fit_eval("ERM-observed", ERMObserved(seed=seed))
    fit_eval("IPW", IPWLearner(seed=seed), nuisance=fold.nu_train)
    fit_eval("AIPW", AIPWLearner(seed=seed), nuisance=fold.nu_train)
    fit_eval("Imputation", ImputationLearner(seed=seed), nuisance=fold.nu_train)

    if include_heckman:
        try:
            Z = None
            if tr.Z is not None and "leniency" in tr.oracle:
                Z = np.asarray(tr.oracle["leniency"])[tr.Z].reshape(-1, 1)
            h = HeckmanProbit(seed=seed).fit(tr.X, tr.T, tr.Y_obs, Z=Z)
            rows.append(_row(f"Heckman(rho={h.rho_:+.2f})",
                             h.decision_function(te.X), fold, gamma_report))
        except Exception as exc:                      # pragma: no cover
            rows.append({"method": "Heckman", "error": repr(exc)})

    # DCL is a plug-in rule: build the test-fold box and apply a closed-form act.
    for g in gammas:
        box_te = fold.nu_test.box(g, direction)
        for crit, tag in (("risk", ""), ("regret", "-reg")):
            s = dcl_bayes_score(box_te.lo, box_te.hi, crit)
            rows.append(_row(f"DCL{tag}(G={g:g})", s, fold, gamma_report,
                             extra={"abstention_rate": float(np.mean(s == 0.0)),
                                    "gamma": g, "criterion": crit,
                                    "direction": direction}))

    box_inf = fold.nu_test.box(np.inf)
    rows.append(_row("Manski(G=inf)", dcl_bayes_score(box_inf.lo, box_inf.hi),
                     fold, gamma_report))

    if include_oracle and tr.Y_full is not None:
        o = OracleLearner(seed=seed).fit(tr.X, tr.T, tr.Y_obs, Y_full=tr.Y_full)
        rows.append(_row("ORACLE(uncensored)", o.decision_function(te.X),
                         fold, gamma_report))
    return rows
