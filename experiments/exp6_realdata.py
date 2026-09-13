"""Experiment 6 -- real decisions.

COMPAS is the rare case where the censored outcomes were recorded anyway: judges
made real pretrial release decisions, and two-year rearrest is known for
released *and* detained defendants.  So we can hide the detained defendants'
labels from every method, exactly as a real deployment would, and then check the
bounds against the truth.  No simulated policy is involved anywhere.

AER-CreditCard is the opposite extreme and is included for that reason: the
accept/reject decisions are real and the outcome for rejected applicants does
not exist and cannot be recovered.  Bounds are the only reportable object, and
we show what the pipeline produces when ground truth is permanently unavailable.

Caveat, repeated from ``dcl/data/compas.py``: pretrial detention is partly
incapacitating, so the detained group's rearrest rate mixes selection with a
mechanical suppression.  We exclude custody spells over 180 days and read the
realised odds ratio as a rough anchor for plausible ``Gamma``, not as an
estimate of ``Gamma_0``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Timer, save, save_table
from dcl.auc_bounds import (auc_from_regression, midrank, naive_corner_interval,
                            sharp_auc_interval)
from dcl.baselines import ERMObserved
from dcl.data import make_compas, make_creditcard
from dcl.harness import run_methods
from dcl.nuisance import CrossFitNuisance
from dcl.objectives import dcl_bayes_score, minimax_regret, worstcase_risk
from dcl.sensitivity import outcome_bounds

GAMMAS = (1.0, 1.25, 1.5, 1.885, 2.5, 4.0, 8.0)


def compas_coverage(n_seeds=5, seed=0):
    """Does the interval cover the TRUE deployment AUROC on real decisions?"""
    from sklearn.metrics import roc_auc_score
    rows = []
    ds = make_compas()
    or_marg = ds.oracle["marginal_odds_ratio"]
    for sd in range(n_seeds):
        tr, te = ds.split(test_size=0.4, seed=seed + sd)
        cf = CrossFitNuisance(seed=seed + sd)
        cf.fit_predict(tr.X, tr.T, tr.Y_obs)
        nte = cf.predict(te.X)
        f = ERMObserved(seed=seed + sd).fit(tr.X, tr.T, tr.Y_obs).decision_function(te.X)
        true_emp = float(roc_auc_score(te.Y_full, f))
        obs_emp = float(roc_auc_score(te.Y_obs[te.T == 1], f[te.T == 1]))
        for g in GAMMAS:
            lo, hi = outcome_bounds(nte.p1, nte.e, g)
            res = sharp_auc_interval(f, lo, hi)
            nc = naive_corner_interval(f, lo, hi)
            rows.append(dict(
                seed=sd, gamma=g, marginal_odds_ratio=or_marg,
                true_auc=true_emp, observed_auc=obs_emp,
                gap_observed_minus_true=obs_emp - true_emp,
                sharp_lo=res.lower, sharp_hi=res.upper, width=res.width,
                covers=bool(res.contains(true_emp)),
                naive_lo=nc[0], naive_hi=nc[1],
                naive_covers=bool(nc[0] - 1e-9 <= true_emp <= nc[1] + 1e-9),
            ))
    return pd.DataFrame(rows)


def compas_methods(n_seeds=3, seed=0):
    ds = make_compas()
    rows = []
    for sd in range(n_seeds):
        res = run_methods(ds, gammas=(1.0, 1.5, 2.0, 3.0), gamma_report=1.885,
                          seed=seed + sd, direction="two-sided")
        for r in res:
            r.update(dataset="compas", seed=sd)
        rows += res
    return pd.DataFrame(rows)


def creditcard_bounds(seed=0):
    """Bounds-only reporting where ground truth does not and cannot exist."""
    from sklearn.metrics import roc_auc_score
    ds = make_creditcard()
    tr, te = ds.split(test_size=0.4, seed=seed)
    cf = CrossFitNuisance(model="gbm", n_folds=5, seed=seed)
    cf.fit_predict(tr.X, tr.T, tr.Y_obs)
    nte = cf.predict(te.X)
    f = ERMObserved(seed=seed).fit(tr.X, tr.T, tr.Y_obs).decision_function(te.X)
    obs = float(roc_auc_score(te.Y_obs[te.T == 1], f[te.T == 1]))
    rows = []
    for g in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0):
        lo, hi = outcome_bounds(nte.p1, nte.e, g)
        res = sharp_auc_interval(f, lo, hi)
        f_dcl = dcl_bayes_score(lo, hi, "regret")
        rows.append(dict(gamma=g, observed_auc=obs,
                         auroc_lo=res.lower, auroc_hi=res.upper, width=res.width,
                         wc_risk_incumbent=worstcase_risk(f, lo, hi),
                         wc_risk_dcl=worstcase_risk(f_dcl, lo, hi),
                         regret_incumbent=minimax_regret(f, lo, hi),
                         regret_dcl=minimax_regret(f_dcl, lo, hi),
                         selection_rate=ds.selection_rate))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with Timer("exp6a COMPAS coverage"):
        a = compas_coverage()
    with Timer("exp6b COMPAS methods"):
        b = compas_methods()
    with Timer("exp6c CreditCard bounds"):
        c = creditcard_bounds()
    save_table("exp6a_compas_coverage", a)
    save_table("exp6b_compas_methods", b)
    save_table("exp6c_creditcard_bounds", c)

    g = a.groupby("gamma")[["true_auc", "observed_auc", "sharp_lo", "sharp_hi",
                            "width", "covers", "naive_covers"]].mean()
    print("\n--- COMPAS: real decisions, real censored labels ---")
    print(f"  marginal odds ratio (detained vs released): {a.marginal_odds_ratio.iloc[0]:.3f}")
    print(f"  observed AUROC overstates the truth by {a.gap_observed_minus_true.mean():+.4f}")
    print(g.to_string(float_format=lambda x: f"{x:8.4f}"))

    print("\n--- COMPAS: methods (mean over seeds) ---")
    s = b.groupby("method")[["risk", "auroc", "accuracy", "obs_auroc",
                             "worstcase_risk", "minimax_regret"]].mean()
    print(s.sort_values("minimax_regret").to_string(float_format=lambda x: f"{x:8.4f}"))

    print("\n--- AER-CreditCard: bounds only (no ground truth exists) ---")
    print(c.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    save("exp6_summary", dict(
        compas_marginal_odds_ratio=float(a.marginal_odds_ratio.iloc[0]),
        compas_observed_minus_true=float(a.gap_observed_minus_true.mean()),
        compas_coverage_at_gamma_1=float(a[a.gamma == 1.0].covers.mean()),
        compas_coverage_at_or=float(a[np.isclose(a.gamma, 1.885)].covers.mean()),
        compas_coverage_at_gamma_25=float(a[a.gamma == 2.5].covers.mean()),
        compas_naive_coverage_at_or=float(a[np.isclose(a.gamma, 1.885)].naive_covers.mean()),
        compas_width_at_or=float(a[np.isclose(a.gamma, 1.885)].width.mean()),
    ))
