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

from _common import Timer, ci95, save, save_table, stamp_provenance
from dcl.auc_bounds import (naive_corner_interval, sharp_auc_interval)
from dcl.baselines import ERMObserved
from dcl.data import make_compas, make_creditcard
from dcl.harness import run_methods
from dcl.nuisance import CrossFitNuisance
from dcl.objectives import dcl_bayes_score, minimax_regret, worstcase_risk
from dcl.sensitivity import outcome_bounds

GAMMAS = (1.0, 1.25, 1.5, 1.885, 2.5, 4.0, 8.0)


def compas_coverage(n_seeds=5, seed=0, outcome="recorded"):
    """Does the interval cover the RECORDED deployment AUROC on real decisions?

    ``outcome="recorded"`` scores two-year rearrest as published;
    ``"exposure_adjusted"`` holds time at risk fixed (see dcl/data/compas.py).
    The realised detained/released odds ratio of the chosen outcome is added to
    the Gamma grid so that "coverage at the data's own odds ratio" is defined
    for both.
    """
    from sklearn.metrics import roc_auc_score
    rows = []
    ds = make_compas(outcome=outcome)
    or_marg = ds.oracle["marginal_odds_ratio"]
    gammas = tuple(sorted(set(GAMMAS) | {round(or_marg, 3)}))
    for sd in range(n_seeds):
        tr, te = ds.split(test_size=0.4, seed=seed + sd)
        cf = CrossFitNuisance(seed=seed + sd)
        cf.fit_predict(tr.X, tr.T, tr.Y_obs)
        nte = cf.predict(te.X)
        f = ERMObserved(seed=seed + sd).fit(tr.X, tr.T, tr.Y_obs).decision_function(te.X)
        true_emp = float(roc_auc_score(te.Y_full, f))
        obs_emp = float(roc_auc_score(te.Y_obs[te.T == 1], f[te.T == 1]))
        for g in gammas:
            lo, hi = outcome_bounds(nte.p1, nte.e, g)
            res = sharp_auc_interval(f, lo, hi)
            nc = naive_corner_interval(f, lo, hi)
            rows.append(dict(
                seed=sd, outcome=outcome, gamma=g, marginal_odds_ratio=or_marg,
                at_own_or=bool(abs(g - or_marg) < 5e-3), n=ds.n,
                true_auc=true_emp, observed_auc=obs_emp,
                gap_observed_minus_true=obs_emp - true_emp,
                sharp_lo=res.lower, sharp_hi=res.upper, width=res.width,
                covers=bool(res.contains(true_emp)),
                naive_lo=nc[0], naive_hi=nc[1],
                naive_covers=bool(nc[0] - 1e-9 <= true_emp <= nc[1] + 1e-9),
            ))
    return pd.DataFrame(rows)


def compas_methods(n_seeds=5, seed=0, outcome="recorded"):
    ds = make_compas(outcome=outcome)
    gamma_report = round(ds.oracle["marginal_odds_ratio"], 3)
    rows = []
    for sd in range(n_seeds):
        res = run_methods(ds, gammas=(1.0, 1.5, 2.0, 3.0), gamma_report=gamma_report,
                          seed=seed + sd, direction="two-sided")
        for r in res:
            r.update(dataset="compas", outcome=outcome, seed=sd, gamma_report=gamma_report)
        rows += res
    return pd.DataFrame(rows)


METRICS = ("risk", "auroc", "accuracy", "obs_auroc", "worstcase_risk", "minimax_regret")


def _method_family(name: str) -> str:
    return "Heckman" if name.startswith("Heckman") else name


def methods_summary(b: pd.DataFrame) -> dict:
    """Per-method mean and 95% CI over seeds (Heckman's rho-tagged runs pooled)."""
    out = {}
    b = b[b.get("error").isna()] if "error" in b else b
    b = b.assign(family=b.method.map(_method_family))
    for fam, g in b.groupby("family"):
        d = {}
        for m in METRICS:
            if m in g and g[m].notna().any():
                mean, hw, n = ci95(g[m].to_numpy(float))
                d[m] = dict(mean=mean, ci95=hw, n=n, min=float(g[m].min()), max=float(g[m].max()))
        out[fam] = d
    return out


def coverage_summary(a: pd.DataFrame) -> dict:
    """Coverage / width by Gamma with 95% CIs over seeds, plus the two anchors."""
    out = dict(by_gamma={}, marginal_odds_ratio=float(a.marginal_odds_ratio.iloc[0]), n=int(a.n.iloc[0]),
               n_seeds=int(a.seed.nunique()))
    for g, gg in a.groupby("gamma"):
        d = {}
        for k in ("covers", "naive_covers", "width", "true_auc", "observed_auc", "gap_observed_minus_true"):
            mean, hw, n = ci95(gg[k].to_numpy(float))
            d[k] = dict(mean=mean, ci95=hw, n=n)
        out["by_gamma"][f"{g:g}"] = d
    own = a[a.at_own_or]
    one = a[np.isclose(a.gamma, 1.0)]
    for tag, sub in (("at_own_or", own), ("at_gamma_1", one)):
        d = {}
        for k in ("covers", "naive_covers", "width"):
            mean, hw, n = ci95(sub[k].to_numpy(float))
            d[k] = dict(mean=mean, ci95=hw, n=n)
        out[tag] = d
    m, hw, n = ci95(a.groupby("seed").gap_observed_minus_true.mean().to_numpy(float))
    out["observed_minus_true"] = dict(mean=m, ci95=hw, n=n)
    return out


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
    N_SEEDS = 5
    with Timer("exp6a COMPAS coverage (recorded outcome)"):
        a = compas_coverage(n_seeds=N_SEEDS, outcome="recorded")
    with Timer("exp6a' COMPAS coverage (exposure-adjusted outcome)"):
        a_exp = compas_coverage(n_seeds=N_SEEDS, outcome="exposure_adjusted")
    with Timer("exp6b COMPAS methods (recorded outcome)"):
        b = compas_methods(n_seeds=N_SEEDS, outcome="recorded")
    with Timer("exp6b' COMPAS methods (exposure-adjusted outcome)"):
        b_exp = compas_methods(n_seeds=N_SEEDS, outcome="exposure_adjusted")
    with Timer("exp6c CreditCard bounds"):
        c = creditcard_bounds()
    save_table("exp6a_compas_coverage", pd.concat([a, a_exp], ignore_index=True))
    save_table("exp6b_compas_methods", pd.concat([b, b_exp], ignore_index=True))
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

    or_rec = float(a.marginal_odds_ratio.iloc[0])
    save("exp6_summary", dict(
        data_source="real",
        n_seeds=N_SEEDS,
        compas_marginal_odds_ratio=or_rec,
        compas_marginal_odds_ratio_exposure=float(a_exp.marginal_odds_ratio.iloc[0]),
        compas_n_recorded=int(a.n.iloc[0]), compas_n_exposure=int(a_exp.n.iloc[0]),
        # legacy scalar keys (means over seeds) kept for verify_paper_numbers.py
        compas_observed_minus_true=float(a.gap_observed_minus_true.mean()),
        compas_coverage_at_gamma_1=float(a[np.isclose(a.gamma, 1.0)].covers.mean()),
        compas_coverage_at_or=float(a[a.at_own_or].covers.mean()),
        compas_coverage_at_gamma_25=float(a[np.isclose(a.gamma, 2.5)].covers.mean()),
        compas_naive_coverage_at_or=float(a[a.at_own_or].naive_covers.mean()),
        compas_width_at_or=float(a[a.at_own_or].width.mean()),
        coverage=dict(recorded=coverage_summary(a), exposure_adjusted=coverage_summary(a_exp)),
        methods=dict(recorded=methods_summary(b), exposure_adjusted=methods_summary(b_exp)),
        gamma_report=dict(recorded=float(b.gamma_report.iloc[0]), exposure_adjusted=float(b_exp.gamma_report.iloc[0])),
        creditcard=dict(selection_rate=float(c.selection_rate.iloc[0]), observed_auc=float(c.observed_auc.iloc[0]),
                        n_gammas=int(len(c))),
    ))
    stamp_provenance("exp6_summary", "real")
