"""Experiment 4 -- Gamma is not identified, but it is falsifiable.

Part A (falsification).  With leniency variation across decision makers and the
exclusion restriction ``Z ind. (Y, S) | X``, the same ``p(x)`` must lie in every
decision maker's identified interval.  The smallest ``Gamma`` making those
intervals intersect is an identified **lower bound** ``Gamma_min``, and
DCSM(Gamma) is refuted below it.  We check ``Gamma_min <= Gamma_0`` always, and
measure how much of ``Gamma_0`` the test recovers as leniency spread grows and
as reliance on private information becomes more heterogeneous across decision
makers.

Part B (break-even Gamma).  The deliverable a practitioner actually wants: how
much hidden judgement would it take to overturn the conclusion?  We report the
largest ``Gamma`` at which the AUROC lower bound still clears a threshold, and
the ``Gamma`` at which a candidate model's certified worst-case risk stops
beating the incumbent's.

Part C (mis-specification).  What happens when the assumed ``Gamma`` is too
small: coverage degrades gracefully and monotonically rather than catastrophically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Timer, save, save_table
from dcl.auc_bounds import sharp_auc_interval
from dcl.baselines import ERMObserved
from dcl.data import make_lending_club, make_mimic_sim, make_sl_bench
from dcl.falsify import falsification_curve
from dcl.nuisance import CrossFitNuisance
from dcl.objectives import dcl_bayes_score, worstcase_risk
from dcl.sensitivity import outcome_bounds


def part_a_falsification(n=16000, seed=0):
    rows = []
    for target in (1.0, 1.5, 2.0, 3.0, 5.0):
        for spread in (0.5, 1.0, 2.0):
            for het in (0.0, 0.3):
                ds = make_sl_bench(n=n, d=8, target_gamma=target,
                                   leniency_spread=spread, kappa_heterogeneity=het,
                                   n_judges=5, seed=seed)
                o = ds.oracle
                r = falsification_curve(o["p1_by_z"], o["e_by_z"], n_boot=80, seed=seed)
                rows.append(dict(
                    target=target, gamma0=o["gamma0"],
                    gamma0_cond=o["gamma0_cond"],
                    leniency_spread=spread, kappa_heterogeneity=het,
                    gamma_min=r.gamma_min, gamma_min_q95=r.gamma_min_q95,
                    gamma_min_boot_lo=r.gamma_min_boot_lo,
                    recovered_fraction=float(np.log(max(r.gamma_min_q95, 1.0))
                                             / max(np.log(o["gamma0_cond"]), 1e-9))
                    if o["gamma0_cond"] > 1 + 1e-9 else np.nan,
                    # The falsification bound is on the CONDITIONAL parameter.
                    valid_lower_bound=bool(r.gamma_min <= o["gamma0_cond"] + 1e-6),
                    valid_lower_bound_q95=bool(
                        r.gamma_min_q95 <= o["gamma0_cond"] + 1e-6),
                    refutes_MAR=bool(r.refutes(1.05)),
                ))
                print(f"    G0={o['gamma0']:5.2f} G0cond={o['gamma0_cond']:5.2f} spread={spread} het={het} -> "
                      f"Gmin={r.gamma_min_q95:5.2f} (boot-lo {r.gamma_min_boot_lo:5.2f})",
                      flush=True)
    return pd.DataFrame(rows)


def part_b_breakeven(seed=0):
    """Largest Gamma at which a conclusion still survives."""
    rows = []
    makers = {
        "sl_bench": lambda: make_sl_bench(n=16000, d=10, target_gamma=3.0, seed=seed),
        "lending_club": lambda: make_lending_club(target_gamma=2.0, seed=seed),
        "mimic_sim": lambda: make_mimic_sim(n=12000, target_gamma=2.5, seed=seed),
    }
    grid = np.concatenate([[1.0], np.exp(np.linspace(np.log(1.05), np.log(40), 60))])
    for name, mk in makers.items():
        ds = mk()
        tr, te = ds.split(test_size=0.4, seed=seed)
        cf = CrossFitNuisance(seed=seed)
        cf.fit_predict(tr.X, tr.T, tr.Y_obs)
        nte = cf.predict(te.X)
        f_inc = ERMObserved(seed=seed).fit(tr.X, tr.T, tr.Y_obs).decision_function(te.X)
        for g in grid:
            lo, hi = outcome_bounds(nte.p1, nte.e, g)
            res = sharp_auc_interval(f_inc, lo, hi)
            f_dcl = dcl_bayes_score(lo, hi, "regret")
            rows.append(dict(
                dataset=name, gamma=g, gamma0=ds.oracle.get("gamma0", np.nan),
                auroc_lo=res.lower, auroc_hi=res.upper,
                wc_risk_incumbent=worstcase_risk(f_inc, lo, hi),
                wc_risk_dcl=worstcase_risk(f_dcl, lo, hi),
            ))
    df = pd.DataFrame(rows)
    summary = []
    for name, sub in df.groupby("dataset"):
        sub = sub.sort_values("gamma")
        def last_true(mask):
            ok = sub.gamma[mask]
            return float(ok.max()) if len(ok) else 1.0
        summary.append(dict(
            dataset=name, gamma0=float(sub.gamma0.iloc[0]),
            breakeven_auroc_60=last_true(sub.auroc_lo >= 0.60),
            breakeven_auroc_55=last_true(sub.auroc_lo >= 0.55),
            breakeven_auroc_50=last_true(sub.auroc_lo >= 0.50),
            gamma_dcl_beats_incumbent_upto=last_true(
                sub.wc_risk_dcl <= sub.wc_risk_incumbent),
        ))
    return df, pd.DataFrame(summary)


def part_c_misspecification(n=16000, seed=0, n_seeds=3):
    rows = []
    for sd in range(n_seeds):
        ds = make_sl_bench(n=n, d=10, target_gamma=3.0, seed=seed + sd)
        tr, te = ds.split(test_size=0.5, seed=seed + sd)
        f = ERMObserved(seed=seed).fit(tr.X, tr.T, tr.Y_obs).decision_function(te.X)
        from dcl.auc_bounds import auc_from_regression, midrank
        true_pop = auc_from_regression(te.oracle["p"], midrank(f))
        g0 = ds.oracle["gamma0"]
        for g in np.exp(np.linspace(0, np.log(12), 30)):
            lo, hi = outcome_bounds(te.oracle["p1"], te.oracle["e"], g)
            res = sharp_auc_interval(f, lo, hi)
            rows.append(dict(seed=sd, gamma=g, gamma0=g0, ratio=g / g0,
                             width=res.width, covers=bool(res.contains(true_pop)),
                             true_auc=true_pop, lo=res.lower, hi=res.upper))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with Timer("exp4a falsification"):
        a = part_a_falsification()
    with Timer("exp4b break-even"):
        b_raw, b_sum = part_b_breakeven()
    with Timer("exp4c misspecification"):
        c = part_c_misspecification()
    save_table("exp4a_falsification", a)
    save_table("exp4b_breakeven_raw", b_raw)
    save_table("exp4b_breakeven", b_sum)
    save_table("exp4c_misspecification", c)
    print("\n--- falsification: Gamma_min is always a valid lower bound? "
          f"{bool(a.valid_lower_bound.all())}")
    print(a.groupby(["target", "kappa_heterogeneity"])[
        ["gamma0", "gamma0_cond", "gamma_min", "gamma_min_q95",
         "recovered_fraction"]].mean().to_string(
        float_format=lambda x: f"{x:7.3f}"))
    print("\n--- break-even Gamma ---")
    print(b_sum.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))
    print("\n--- coverage vs Gamma/Gamma_0 ---")
    c["bucket"] = pd.cut(c.ratio, [0, .5, .8, 1.0, 1.5, 3, 100])
    print(c.groupby("bucket", observed=True)[["covers", "width"]].mean().to_string(
        float_format=lambda x: f"{x:7.3f}"))
    save("exp4_summary", dict(
        falsification_always_valid=bool(a.valid_lower_bound.all()),
        refutes_MAR_when_confounded=float(
            a[a.gamma0 > 1.2].refutes_MAR.mean()),
        false_alarm_when_MAR=float(a[a.gamma0 <= 1.001].refutes_MAR.mean())
        if (a.gamma0 <= 1.001).any() else float("nan"),
    ))
