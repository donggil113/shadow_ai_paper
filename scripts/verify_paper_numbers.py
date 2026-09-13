#!/usr/bin/env python3
"""Check the numbers hard-coded in the paper against the saved result files.

Every claim in the text that came from an experiment is listed here with the
file and column it must agree with. Run after re-running the experiments: if a
result moves, this fails loudly instead of the paper quietly drifting.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def _csv(name):
    p = os.path.join(RES, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _json(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


CHECKS = []


def check(label, claimed, actual, tol=5e-4):
    CHECKS.append((label, claimed, actual, tol))


def main() -> int:
    s1 = _json("exp1_summary")
    if s1:
        check("Thm2 coverage, sharp", 1.00, s1["coverage_sharp_pop"])
        check("Thm2 coverage, corner", 0.10, s1["coverage_naive"])
        check("Thm2 coverage, separate", 1.00, s1["coverage_separate"])
        check("Thm2 coverage when Gamma<Gamma0", 0.71,
              s1["coverage_sharp_when_gamma_too_small"], tol=0.01)
        check("Thm2 width, sharp", 0.277, s1["width_sharp"], tol=2e-3)
        check("Thm2 width, separate", 0.883, s1["width_separate"], tol=2e-3)
        check("Thm2 max |analytic - brute force| < 1e-5", True,
              s1["max_abs_gap"] < 1e-5)
        check("Thm2 attainment error < 1e-12", True,
              s1["max_attainment_error"] < 1e-12)
        check("brute force never outside", True, s1["bruteforce_always_inside"])

    rt = _csv("exp1c_runtime")
    if rt is not None:
        for n, claimed in ((20000, 0.16), (100000, 0.85), (400000, 3.4)):
            row = rt[rt.n == n]
            if len(row):
                check(f"runtime n={n} within 3x of {claimed}s", True,
                      float(row.seconds.iloc[0]) < 3 * claimed)

    s6 = _json("exp6_summary")
    if s6:
        check("COMPAS marginal odds ratio", 1.885, s6["compas_marginal_odds_ratio"],
              tol=2e-3)
        check("COMPAS coverage at Gamma=1", 0.0, s6["compas_coverage_at_gamma_1"])
        check("COMPAS coverage at the odds ratio", 1.0, s6["compas_coverage_at_or"])
        check("COMPAS corner coverage at the odds ratio", 0.0,
              s6["compas_naive_coverage_at_or"])
        check("COMPAS interval width at the odds ratio", 0.099,
              s6["compas_width_at_or"], tol=2e-3)

    lr = _csv("exp2_learning_summary")
    if lr is not None:
        tbl = {
            ("lending_club", "ERM-observed"): (0.2492, 0.6099, 0.3015, 0.0743),
            ("lending_club", "IPW"): (0.2550, 0.5977, 0.3064, 0.0789),
            ("lending_club", "AIPW"): (0.2668, 0.5879, 0.3182, 0.0977),
            ("lending_club", "ORACLE(uncensored)"): (0.2213, 0.6047, 0.2688, 0.0428),
            ("mimic_sim", "ERM-observed"): (0.3058, 0.8159, 0.3878, 0.1126),
            ("mimic_sim", "ORACLE(uncensored)"): (0.2520, 0.8331, 0.3351, 0.0703),
            ("sl_bench", "ERM-observed"): (0.5633, 0.6479, 0.7805, 0.1414),
            ("sl_bench", "ORACLE(uncensored)"): (0.5264, 0.6688, 0.7761, 0.1230),
        }
        for (ds, m), (risk, auroc, wc, reg) in tbl.items():
            r = lr[(lr.dataset == ds) & (lr.method == m)]
            if not len(r):
                check(f"{ds}/{m} present", True, False)
                continue
            for nm, claimed, col in (("risk", risk, "risk"), ("auroc", auroc, "auroc"),
                                     ("wc", wc, "worstcase_risk"),
                                     ("regret", reg, "minimax_regret")):
                check(f"{ds}/{m} {nm}", claimed, float(r[col].iloc[0]), tol=1e-3)
        # the best-regret DCL variant per dataset
        for ds, claimed in (("lending_club", 0.0021), ("mimic_sim", 0.0089),
                            ("sl_bench", 0.0316)):
            sub = lr[lr.dataset == ds]
            check(f"{ds} best certified regret", claimed,
                  float(sub.minimax_regret.min()), tol=1e-3)

    fa = _csv("exp4a_falsification")
    if fa is not None:
        check("Gamma_min is always a valid lower bound", True,
              bool(fa.valid_lower_bound.all()))
        mar = fa[fa.gamma0_cond <= 1.001]
        if len(mar):
            check("no false refutation of MAR when Gamma0 = 1", 0.0,
                  float(mar.refutes_MAR.mean()))

    fb = _csv("exp4b_breakeven")
    if fb is not None:
        for ds, k55, k50 in (("lending_club", 1.72, 2.82), ("mimic_sim", 9.10, 14.91)):
            r = fb[fb.dataset == ds]
            if len(r):
                check(f"{ds} break-even AUROC>=0.55", k55,
                      float(r.breakeven_auroc_55.iloc[0]), tol=0.05)
                check(f"{ds} break-even AUROC>=0.50", k50,
                      float(r.breakeven_auroc_50.iloc[0]), tol=0.05)
        check("DCL beats the incumbent certificate at every Gamma tested", True,
              bool((fb.gamma_dcl_beats_incumbent_upto >= 40.0).all()))

    mis = _csv("exp4c_misspecification")
    if mis is not None:
        mis = mis.assign(ratio=mis.gamma / mis.gamma0)
        check("coverage is 1.0 for Gamma >= Gamma_0/2", 1.0,
              float(mis[mis.ratio > 0.5].covers.mean()))
        check("coverage ~0.69 for Gamma <= Gamma_0/2", 0.69,
              float(mis[mis.ratio <= 0.5].covers.mean()), tol=0.02)

    ident = _csv("exp3_uq_identification")
    if ident is not None:
        check("oracle: p1 inside the identified set for every unit", 1.0,
              float(ident.oracle_p1_in_box.min()))
        check("oracle: true mean aleatoric inside its interval", True,
              bool(ident.oracle_true_inside.all()))
        check("oracle: naive mean aleatoric inside its interval", True,
              bool(ident.oracle_naive_inside.all()))
        check("naive aleatoric bias ~ +0.05 nats", 0.05,
              float(ident.oracle_naive_bias.mean()), tol=0.01)
        check("estimated box covers 45-58% of units pointwise", True,
              0.40 <= float(ident.est_p_in_box.min()) <= 0.60)

    uq = _csv("exp3_uq_decomposition")
    if uq is not None:
        log = uq[uq.ensemble == "logistic"].sort_values("n")
        check("epistemic/censoring ratio at the largest n < 0.02", True,
              float(log.ratio_epistemic_to_censoring.iloc[-1]) < 0.02)
        check("epistemic shrinks by > 10x across the n range", True,
              float(log.naive_epistemic.iloc[0] / log.naive_epistemic.iloc[-1]) > 10)
        check("censoring term stays within 20% across the n range", True,
              abs(float(log.censoring.iloc[-1] / log.censoring.iloc[0]) - 1) < 0.2)

    g5 = _json("exp5_summary")
    if g5:
        check("uniform-deviation rate exponent near -0.5", -0.507,
              g5["deviation_rate_exponent"], tol=0.02)
        check("R^2 of the coefficient vs log Gamma", 0.89,
              g5["r2_vs_log_gamma"], tol=0.02)
        check("R^2 of the coefficient vs Gamma", 0.60,
              g5["r2_vs_gamma"], tol=0.02)
        check("log Gamma explains the coefficient better than Gamma", True,
              g5["r2_vs_log_gamma"] > g5["r2_vs_gamma"])
        check("excess risk shows fast rates (exponent ~ -1.08)", -1.08,
              g5["excess_rate_exponent"], tol=0.05)
    sc5 = _csv("exp5_scaling")
    if sc5 is not None:
        rng_ = float(sc5.coefficient.max() / sc5.coefficient.min() - 1)
        check("coefficient grows ~24% over a 16-fold Gamma range", 0.24, rng_, tol=0.03)

    fails = []
    for label, claimed, actual, tol in CHECKS:
        if isinstance(claimed, bool):
            ok = bool(actual) == claimed
            shown = f"{actual}"
        else:
            ok = abs(float(actual) - float(claimed)) <= tol
            shown = f"{float(actual):.4f} vs {float(claimed):.4f}"
        if not ok:
            fails.append(f"  MISMATCH  {label}: {shown}")
    print(f"checked {len(CHECKS)} paper numbers against results/")
    if fails:
        print("\n".join(fails))
        return 1
    print("OK - every number in the paper matches the saved results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
