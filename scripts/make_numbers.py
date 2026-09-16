#!/usr/bin/env python3
"""Generate paper/numbers.tex and paper/numbers_provenance.json from results/.

CLAUDE.md rule 2: every numeric value in the paper body is a macro defined
here, never typed by hand.  Rule 3: every macro carries the data_source of the
results file it came from (per-dataset where a file mixes sources).

The REGISTRY is explicit and auditable: one entry per macro.  Each entry names
the results file, a dotted path into its JSON (or a CSV expression), a format,
and -- for dataset-specific values -- the dataset whose provenance applies.
Macro names are letters only (LaTeX), CamelCase, prefixed with 'n'.

    python3 scripts/make_numbers.py          # write paper/numbers.tex
    python3 scripts/make_numbers.py --check  # exit 1 if numbers.tex is stale

Formats: 'pct0' (37\%), 'pct1' (37.4\%), 'f2'/'f3'/'f4' (fixed decimals),
'g3' (3 significant), 'int', 'x1' (multiplicative factor like 14.6x -> "14.6"),
'ci:f3' (value with a companion \\<name>CI macro giving \\pm CI).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT_TEX = os.path.join(ROOT, "paper", "numbers.tex")
OUT_JSON = os.path.join(ROOT, "paper", "numbers_provenance.json")

# --------------------------------------------------------------------------- #
# Registry: (macro, file, path, fmt, dataset)
#   file   : results/<file>.json  (or 'csv:<file>.csv' with path = pandas expr)
#   path   : dotted JSON path; a trailing '.mean' picks the mean of a CI cell
#   fmt    : see module docstring
#   dataset: key into data_source_by_dataset (None -> top-level data_source)
# --------------------------------------------------------------------------- #
R = []


_SEEN = set()


def reg(macro, file, path, fmt, dataset=None):
    if macro in _SEEN or not re.fullmatch(r"n[A-Z][A-Za-z]*", macro):
        raise ValueError(f"macro {macro!r} duplicated or not letters-only")
    _SEEN.add(macro)
    R.append(dict(macro=macro, file=file, path=path, fmt=fmt, dataset=dataset))


# ---- exp1: Theorem (sharp AUROC interval) validity ------------------------
reg("nExpOneCovSharp", "exp1_summary.json", "coverage_sharp_pop", "pct0")
reg("nExpOneCovSharpEmp", "exp1_summary.json", "coverage_sharp_emp", "pct0")
reg("nExpOneCovCorner", "exp1_summary.json", "coverage_naive", "pct0")
reg("nExpOneCovSeparate", "exp1_summary.json", "coverage_separate", "pct0")
reg("nExpOneCovTooSmall", "exp1_summary.json", "coverage_sharp_when_gamma_too_small", "pct0")
reg("nExpOneWidthSharp", "exp1_summary.json", "width_sharp", "f3")
reg("nExpOneWidthSharpCI", "exp1_summary.json", "width_sharp_ci95", "f3")
reg("nExpOneWidthSeparate", "exp1_summary.json", "width_separate", "f3")
reg("nExpOneGap", "exp1_summary.json", "max_abs_gap", "sci1")
reg("nExpOneAttain", "exp1_summary.json", "max_attainment_error", "sci1")
reg("nExpOneSeeds", "exp1_summary.json", "n_seeds", "int", dataset="n/a")
reg("nExpOneWidthRatioSep", "exp1_summary.json", "width_separate/width_sharp", "x1")
reg("nExpOneWidthRatioCorner", "csv:exp1b_validity.csv",
    "df[df.gamma_sufficient].sharp_w.mean()/df[df.gamma_sufficient].naive_w.mean()", "x1")
reg("nExpOneWidthCorner", "csv:exp1b_validity.csv", "df[df.gamma_sufficient].naive_w.mean()", "f3")
reg("nExpOneRuntimeTwentyK", "csv:exp1c_runtime.csv", "df[df.n==20000].seconds.iloc[0]", "f2")
reg("nExpOneRuntimeHundredK", "csv:exp1c_runtime.csv", "df[df.n==100000].seconds.iloc[0]", "f2")
reg("nExpOneRuntimeFourHundredK", "csv:exp1c_runtime.csv", "df[df.n==400000].seconds.iloc[0]", "f1")

# ---- exp6: COMPAS coverage (real) ----------------------------------------
# (COMPAS coverage / methods macros are registered below from the 5-seed exp6 summary.)

# ---- exp2: learning comparison (per dataset, 5 seeds) --------------------
for ds, tag in (("lending_club", "Lending"), ("mimic_sim", "MimicSim"), ("sl_bench", "SlBench")):
    for m, mtag in (("ERM-observed", "Erm"), ("IPW", "Ipw"), ("AIPW", "Aipw"), ("Imputation", "Imp"),
                    ("Manski(G=inf)", "Manski"), ("ORACLE(uncensored)", "Oracle"),
                    ("DCL(G=2)", "DclTwo"), ("DCL-reg(G=2)", "DclRegTwo"), ("DCL(G=3)", "DclThree"),
                    ("DCL-reg(G=3)", "DclRegThree"), ("DCL-reg(G=1.5)", "DclRegOneFive"),
                    ("DCL-dir-reg(G=1.5)", "DclDirRegOneFive")):
        for k, ktag in (("risk", "Risk"), ("auroc", "Auroc"), ("obs_auroc", "ObsAuroc"),
                        ("worstcase_risk", "Wc"), ("minimax_regret", "Regret")):
            reg(f"nExpTwo{tag}{mtag}{ktag}", "exp2_summary.json",
                f"datasets.{ds}.{m}.{k}", "ci:f4", dataset=ds)
reg("nExpTwoSeeds", "exp2_summary.json", "n_seeds", "int", dataset="n/a")

# ---- exp8: nuisance theorem coverage (per dataset, 5 seeds) --------------
for ds, tag in (("sl_bench", "SlBench"), ("lending_club", "Lending"), ("compas", "Compas")):
    for k, ktag in (("cov_plugin", "CovPlugin"), ("cov_bins10", "CovBinsTen"), ("cov_bins20", "CovBinsTwenty"),
                    ("cov_bins10_smooth", "CovBinsTenSmooth"), ("cov_bins20_smooth", "CovBinsTwentySmooth"),
                    ("cov_oracle_e_est_p1", "CovOracleE"), ("cov_est_e_oracle_p1", "CovOracleP"),
                    ("width_plugin", "WidthPlugin"), ("width_bins20", "WidthBinsTwenty"),
                    ("width_bins20_smooth", "WidthBinsTwentySmooth"),
                    ("mae_e", "MaeE"), ("mae_p1", "MaeP"),
                    ("dr_covers_prevalence_interval", "DrCovPrev"), ("plug_covers_prevalence_interval", "PlugCovPrev"),
                    ("dr_outer_covers_realised", "DrCovRealised"),
                    ("auc_cov_plugin", "AucCovPlugin"), ("auc_cov_bins20", "AucCovBinsTwenty"),
                    ("auc_cov_emp_plugin", "AucCovEmpPlugin"), ("auc_cov_emp_bins20", "AucCovEmpBinsTwenty"),
                    ("auc_width_plugin", "AucWidthPlugin"), ("auc_width_bins20", "AucWidthBinsTwenty"),
                    ("binlevel_cov_plugin", "BinCovPlugin"), ("binlevel_cov_bins20", "BinCovBinsTwenty"),
                    ("dr_abs_err_lo", "DrErrLo"), ("plug_abs_err_lo", "PlugErrLo"),
                    ("dr_abs_err_hi", "DrErrHi"), ("plug_abs_err_hi", "PlugErrHi")):
        fmt = "ci:pct1" if k.startswith(("cov_", "dr_covers", "plug_covers", "dr_outer", "auc_cov", "binlevel")) else "ci:f3"
        reg(f"nExpEight{tag}{ktag}", "exp8_summary.json", f"datasets.{ds}.{k}", fmt, dataset=ds)
reg("nExpEightSeeds", "exp8_summary.json", "n_seeds", "int", dataset="n/a")
reg("nExpEightAlphaPct", "exp8_summary.json", "alpha*100", "int", dataset="n/a")

# ---- exp3: uncertainty decomposition (5 seeds) --------------------------
reg("nExpThreeSeeds", "exp3_summary.json", "n_seeds", "int", dataset="n/a")
NTAG = {2000: "TwoK", 5000: "FiveK", 12000: "TwelveK", 30000: "ThirtyK", 70000: "SeventyK"}   # LaTeX macro names: letters only
for n, nt in NTAG.items():
    reg(f"nExpThreeEpiLog{nt}", "exp3_summary.json", f"by_cell_ci.logistic_n{n}.naive_epistemic", "ci:f4")
    reg(f"nExpThreeEpiMlp{nt}", "exp3_summary.json", f"by_cell_ci.mlp_n{n}.naive_epistemic", "ci:f3")
    reg(f"nExpThreeCens{nt}", "exp3_summary.json", f"by_cell_ci.logistic_n{n}.censoring", "ci:f4")
    reg(f"nExpThreeRatio{nt}", "exp3_summary.json", f"by_cell_ci.logistic_n{n}.ratio_epistemic_to_censoring", "ci:f3")
reg("nExpThreeSlopeLog", "exp3_summary.json", "epistemic_decay_exponent_logistic", "f2")
reg("nExpThreeSlopeMlp", "exp3_summary.json", "epistemic_decay_exponent_mlp", "f2")
reg("nExpThreeAleaBias", "exp3_summary.json", "oracle_mean_aleatoric_bias", "f3s")
reg("nExpThreeEstBoxMin", "exp3_summary.json", "est_p_in_box_min", "pct0")
reg("nExpThreeEpiDrop", "exp3_summary.json", "epistemic_first_logistic/epistemic_last_logistic", "x1")

# ---- exp4: falsification and break-even ----------------------------------
reg("nExpFourValid", "exp4_summary.json", "falsification_always_valid", "bool")
reg("nExpFourRefutesMar", "exp4_summary.json", "refutes_MAR_when_confounded", "pct0")
reg("nExpFourFalseAlarm", "exp4_summary.json", "false_alarm_when_MAR", "pct0")
reg("nExpFourSeedsMis", "exp4_summary.json", "n_seeds_misspecification", "int", dataset="n/a")
reg("nExpFourRecoverMin", "csv:exp4a_falsification.csv", "df.recovered_fraction.min()", "pct0")
reg("nExpFourRecoverMax", "csv:exp4a_falsification.csv", "df.recovered_fraction.max()", "pct0")
reg("nExpFourNCells", "csv:exp4a_falsification.csv", "len(df)", "int", dataset="n/a")
reg("nExpFourGminAtThree", "csv:exp4a_falsification.csv",
    "df[(df.target==3.0)&(df.kappa_heterogeneity==0)].gamma_min_q95.mean()", "f2")
for ds, tag in (("lending_club", "Lending"), ("mimic_sim", "MimicSim"), ("sl_bench", "SlBench")):
    for k, ktag in (("breakeven_auroc_55", "BreakEvenFiftyFive"), ("breakeven_auroc_50", "BreakEvenFifty"),
                    ("breakeven_auroc_60", "BreakEvenSixty"), ("gamma_dcl_beats_incumbent_upto", "DclBeatsUpto")):
        reg(f"nExpFour{tag}{ktag}", "csv:exp4b_breakeven.csv", f"df[df.dataset=='{ds}'].{k}.iloc[0]", "f2", dataset=ds)
reg("nExpFourMisCovLow", "csv:exp4c_misspecification.csv", "df[df.gamma/df.gamma0<=0.5].covers.mean()", "pct0")
reg("nExpFourMisCovHigh", "csv:exp4c_misspecification.csv", "df[df.gamma/df.gamma0>0.5].covers.mean()", "pct0")
reg("nExpFourMisWidthLow", "csv:exp4c_misspecification.csv", "df[df.gamma/df.gamma0<=0.5].width.mean()", "f2")
reg("nExpFourMisWidthHigh", "csv:exp4c_misspecification.csv", "df[df.gamma/df.gamma0>3].width.mean()", "f2")

# ---- exp5: generalisation scaling ---------------------------------------
reg("nExpFiveDevRate", "exp5_summary.json", "deviation_rate_exponent", "f3s")
reg("nExpFiveRtwoLog", "exp5_summary.json", "r2_vs_log_gamma", "f2")
reg("nExpFiveRtwoLin", "exp5_summary.json", "r2_vs_gamma", "f2")
reg("nExpFiveExcessRate", "exp5_summary.json", "excess_rate_exponent", "f2s")
reg("nExpFiveSeedsDev", "exp5_summary.json", "n_seeds_deviation", "int", dataset="n/a")
reg("nExpFiveCoefGrowthPct", "csv:exp5_scaling.csv", "100*(df.coefficient.max()/df.coefficient.min()-1)", "int")


# ---- exp9: exhaustive ranking search (synthetic; audit/t3_ranking_search.py)
reg("nExpNineEqualInstances", "exp9_ranking_search.json", "equal_n_instances", "int", dataset="n/a")
reg("nExpNineEqualViolations", "exp9_ranking_search.json", "equal_violations", "int")
reg("nExpNineUnequalInstances", "exp9_ranking_search.json", "unequal_n_instances", "int", dataset="n/a")
reg("nExpNineMidBeatenPct", "exp9_ranking_search.json", "unequal_mid_beaten_frac", "pct0")
reg("nExpNineLoBeatenPct", "exp9_ranking_search.json", "unequal_lo_beaten_frac", "pct0")
reg("nExpNineMidGapMean", "exp9_ranking_search.json", "unequal_mid_gap_mean", "f3")
reg("nExpNineMidGapMax", "exp9_ranking_search.json", "unequal_mid_gap_max", "f2")
reg("nExpNineFamilyPct", "exp9_ranking_search.json", "convex_family_attains_optimum_frac", "pct0")
reg("nExpNineMidBeatenSevenPct", "exp9_ranking_search.json", "unequal_mid_beaten_frac_by_n.7", "pct0")
reg("nExpNineNMax", "exp9_ranking_search.json", "n_max", "int", dataset="n/a")
reg("nExpTenRankerGain", "exp10_summary.json", "ranker_max_gain_over_midpoint", "f4")
reg("nExpTenRankerGainTrain", "exp10_summary.json", "ranker_max_gain_train", "f4")
reg("nExpTenSpreadPlugin", "exp10_summary.json", "max_spread_plugin_orderings", "f4")
reg("nExpTenSeeds", "exp10_summary.json", "n_seeds", "int", dataset="n/a")
for ds, tag in (("sl_bench", "SlBench"), ("lending_club", "Lending")):
    for k, ktag in (("wc_mid_linear", "WcMid"), ("wc_ranker", "WcRanker"), ("wc_width_only", "WcWidth"), ("gain_ranker_over_mid_linear", "Gain")):
        reg(f"nExpTen{tag}{ktag}", "exp10_summary.json", f"datasets.{ds}.{k}", "ci:f4", dataset=ds)

# ---- COMPAS lineage / time-at-risk audit (real; audit/compas_time_at_risk.py)
reg("nCompasTarNCohort", "compas_time_at_risk.json", "n_cohort", "int")
reg("nCompasTarNRaw", "compas_time_at_risk.json", "n_raw", "int")
reg("nCompasTarNPropublica", "compas_time_at_risk.json", "n_after_propublica_filter", "int")
reg("nCompasTarSelectionPct", "compas_time_at_risk.json", "selection_rate", "pct0")
reg("nCompasTarORRecorded", "compas_time_at_risk.json", "recorded.odds_ratio", "f2")
reg("nCompasTarORExposure", "compas_time_at_risk.json", "exposure_adjusted.730.odds_ratio", "f2")
reg("nCompasTarKeptPct", "compas_time_at_risk.json", "exposure_adjusted.730.frac_kept", "pct0")
reg("nCompasTarKeptDetainedPct", "compas_time_at_risk.json", "exposure_adjusted.730.frac_kept_detained", "pct0")
reg("nCompasTarCustodyDetainedMedian", "compas_time_at_risk.json", "custody_days_detained.median", "x1")
reg("nCompasTarCustodyDetainedQNinety", "compas_time_at_risk.json", "custody_days_detained.q90", "int")
reg("nCompasTarCustodyReleasedMedian", "compas_time_at_risk.json", "custody_days_released.median", "x1")
reg("nCompasTarPZeroRecorded", "compas_time_at_risk.json", "recorded.p0", "pct0")
reg("nCompasTarPOneRecorded", "compas_time_at_risk.json", "recorded.p1", "pct0")
reg("nCompasTarPZeroExposure", "compas_time_at_risk.json", "exposure_adjusted.730.p0", "pct0")
reg("nCompasTarPOneExposure", "compas_time_at_risk.json", "exposure_adjusted.730.p1", "pct0")

# ---- exp6 (5 seeds, both COMPAS outcome definitions) -------------------
for oc, otag in (("recorded", ""), ("exposure_adjusted", "Exp")):
    for k, ktag in (("covers", "Cov"), ("naive_covers", "CornerCov"), ("width", "Width")):
        fmt = "ci:pct0" if k.endswith("covers") else "ci:f3"
        reg(f"nCompas{otag}{ktag}AtOR", "exp6_summary.json", f"coverage.{oc}.at_own_or.{k}", fmt)
        reg(f"nCompas{otag}{ktag}AtOne", "exp6_summary.json", f"coverage.{oc}.at_gamma_1.{k}", fmt)
    reg(f"nCompas{otag}OR", "exp6_summary.json", f"coverage.{oc}.marginal_odds_ratio", "f3")
    reg(f"nCompas{otag}N", "exp6_summary.json", f"coverage.{oc}.n", "int")
    reg(f"nCompas{otag}ObsMinusTrue", "exp6_summary.json", f"coverage.{oc}.observed_minus_true", "ci:f4s")
    for m, mtag in (("ERM-observed", "Erm"), ("IPW", "Ipw"), ("AIPW", "Aipw"), ("Imputation", "Imp"),
                    ("Heckman", "Heckman"), ("Manski(G=inf)", "Manski"), ("ORACLE(uncensored)", "Oracle"),
                    ("DCL(G=2)", "DclTwo"), ("DCL-reg(G=2)", "DclRegTwo"), ("DCL(G=1.5)", "DclOneFive"),
                    ("DCL-reg(G=1.5)", "DclRegOneFive"), ("DCL(G=3)", "DclThree"), ("DCL-reg(G=3)", "DclRegThree")):
        for k, ktag in (("risk", "Risk"), ("auroc", "Auroc"), ("obs_auroc", "ObsAuroc"),
                        ("worstcase_risk", "Wc"), ("minimax_regret", "Regret")):
            reg(f"nCompas{otag}M{mtag}{ktag}", "exp6_summary.json", f"methods.{oc}.{m}.{k}", "ci:f4")
            if m == "Heckman":
                reg(f"nCompas{otag}M{mtag}{ktag}Min", "exp6_summary.json", f"methods.{oc}.{m}.{k}.min", "f3")
                reg(f"nCompas{otag}M{mtag}{ktag}Max", "exp6_summary.json", f"methods.{oc}.{m}.{k}.max", "f3")
reg("nExpSixSeeds", "exp6_summary.json", "n_seeds", "int", dataset="n/a")
for oc, otag in (("recorded", ""), ("exposure_adjusted", "Exp")):
    reg(f"nCompas{otag}RegretRatioErm", "exp6_summary.json",
        f"expr:d['methods']['{oc}']['ERM-observed']['minimax_regret']['mean']/d['methods']['{oc}']['DCL-reg(G=2)']['minimax_regret']['mean']", "x1")
    reg(f"nCompas{otag}RegretRatioHeckman", "exp6_summary.json",
        f"expr:d['methods']['{oc}']['Heckman']['minimax_regret']['mean']/d['methods']['{oc}']['DCL-reg(G=2)']['minimax_regret']['mean']", "x1")


# ---- exp0: three-point counterexample (synthetic; appendix) ---------------
for i, tag in enumerate(("One", "Two", "Three")):
    reg(f"nCexPlo{tag}", "exp0_counterexample.json", f"expr:d['lo'][{i}]", "f2")
    reg(f"nCexPup{tag}", "exp0_counterexample.json", f"expr:d['hi'][{i}]", "f2")
    reg(f"nCexPStar{tag}", "exp0_counterexample.json", f"expr:d['p_star_upper'][{i}]", "f2")
for k, tag, fmt in (("auc_lo", "AucLo", "f4"), ("auc_mid", "AucMid", "f4"), ("auc_hi", "AucHi", "f4"),
                    ("corner_width", "CornerWidth", "f3"), ("sharp_lo", "SharpLo", "f4"), ("sharp_hi", "SharpHi", "f4"),
                    ("sharp_width", "SharpWidth", "f3"), ("auc_p_star", "AucPStar", "f4"), ("miss", "Miss", "f3")):
    reg(f"nCex{tag}", "exp0_counterexample.json", k, fmt)

# ---- exp1 extras ---------------------------------------------------------
reg("nExpOneNCells", "csv:exp1b_validity.csv", "int(df[df.seed==df.seed.min()].shape[0])", "int", dataset="n/a")
reg("nExpOneNInstances", "csv:exp1b_validity.csv", "len(df)", "int", dataset="n/a")

# ---- exp2 extras: pooled Heckman, blind spot, regret ratios ---------------
for ds, tag in (("lending_club", "Lending"), ("mimic_sim", "MimicSim"), ("sl_bench", "SlBench")):
    for k, ktag in (("risk", "Risk"), ("auroc", "Auroc"), ("obs_auroc", "ObsAuroc"),
                    ("worstcase_risk", "Wc"), ("minimax_regret", "Regret")):
        reg(f"nExpTwo{tag}Heckman{ktag}", "csv:exp2_learning_raw.csv",
            f"df[(df.dataset=='{ds}')&df.method.str.startswith('Heckman')].{k}.mean()", "f4", dataset=ds)
        reg(f"nExpTwo{tag}Heckman{ktag}Min", "csv:exp2_learning_raw.csv",
            f"df[(df.dataset=='{ds}')&df.method.str.startswith('Heckman')].{k}.min()", "f3", dataset=ds)
        reg(f"nExpTwo{tag}Heckman{ktag}Max", "csv:exp2_learning_raw.csv",
            f"df[(df.dataset=='{ds}')&df.method.str.startswith('Heckman')].{k}.max()", "f3", dataset=ds)
    reg(f"nExpTwo{tag}ErmObsMinusTrue", "exp2_summary.json",
        f"expr:d['datasets']['{ds}']['ERM-observed']['obs_auroc']['mean']-d['datasets']['{ds}']['ERM-observed']['auroc']['mean']",
        "f3s", dataset=ds)
    gds = {"lending_club": "2", "sl_bench": "3", "mimic_sim": "3"}[ds]     # Gamma used for DCL on each testbed (>= Gamma_0)
    reg(f"nExpTwo{tag}RegretRatioErm", "exp2_summary.json",
        f"expr:d['datasets']['{ds}']['ERM-observed']['minimax_regret']['mean']/d['datasets']['{ds}']['DCL-reg(G={gds})']['minimax_regret']['mean']",
        "x1", dataset=ds)
    reg(f"nExpTwo{tag}WcRatioErmOverDcl", "exp2_summary.json",
        f"expr:d['datasets']['{ds}']['ERM-observed']['worstcase_risk']['mean']/d['datasets']['{ds}']['DCL(G={gds})']['worstcase_risk']['mean']",
        "x1", dataset=ds)
    reg(f"nExpTwo{tag}GammaZero", "csv:exp2_learning_raw.csv", f"df[df.dataset=='{ds}'].gamma0.iloc[0]", "f2", dataset=ds)
    reg(f"nExpTwo{tag}ManskiRiskRatioErm", "exp2_summary.json",
        f"expr:d['datasets']['{ds}']['Manski(G=inf)']['risk']['mean']/d['datasets']['{ds}']['ERM-observed']['risk']['mean']",
        "x1", dataset=ds)

# ---- exp3 extras ---------------------------------------------------------
reg("nExpThreeCensChangePct", "exp3_summary.json", "censoring_last_logistic/censoring_first_logistic", "pchg0")
reg("nExpThreeRatioLast", "exp3_summary.json", "ratio_last_logistic", "f3")
reg("nExpThreeOracleInBox", "exp3_summary.json", "oracle_p1_always_in_box", "pct0")
for n, nt in NTAG.items():
    reg(f"nExpThreeCalLog{nt}", "csv:exp3_uq_decomposition.csv",
        f"df[(df.ensemble=='logistic')&(df.n=={n})].mean_abs_calibration_error.mean()", "f3")
    reg(f"nExpThreeCalMlp{nt}", "csv:exp3_uq_decomposition.csv",
        f"df[(df.ensemble=='mlp')&(df.n=={n})].mean_abs_calibration_error.mean()", "f3")
reg("nExpThreeNuisMaeMin", "csv:exp3_uq_identification.csv", "min(df.nuis_mae_e.min(), df.nuis_mae_p1.min())", "f2")
reg("nExpThreeNuisMaeMax", "csv:exp3_uq_identification.csv", "max(df.nuis_mae_e.max(), df.nuis_mae_p1.max())", "f2")
reg("nExpThreeEstBoxMax", "csv:exp3_uq_identification.csv", "df.est_p_in_box.max()", "pct0")
reg("nExpThreeAleaBiasRelPct", "csv:exp3_uq_identification.csv", "100*(df.oracle_naive_bias/df.true_aleatoric).mean()", "int")
reg("nExpThreeCalRatioMlpOverLog", "csv:exp3_uq_decomposition.csv",
    "df[df.ensemble=='mlp'].mean_abs_calibration_error.mean()/df[df.ensemble=='logistic'].mean_abs_calibration_error.mean()", "x1")

# ---- exp4 table rows -----------------------------------------------------
for t, ttag in ((1.0, "One"), (1.5, "OneFive"), (2.0, "Two"), (3.0, "Three"), (5.0, "Five")):
    for hcond, htag in (("==0", "Hom"), (">0", "Het")):
        cond = f"(np.isclose(df.target,{t}))&(df.kappa_heterogeneity{hcond})"
        for k, ktag in (("gamma0_cond", "GzCond"), ("gamma_min", "Gmin"), ("gamma_min_q95", "GminQ"),
                        ("recovered_fraction", "Rec")):
            reg(f"nExpFour{ttag}{htag}{ktag}", "csv:exp4a_falsification.csv", f"df[{cond}].{k}.mean()", "f2")
reg("nExpFourMisRange", "csv:exp4c_misspecification.csv", "df.gamma.max()/df.gamma.min()", "int", dataset="n/a")
reg("nExpFourMisWidthMin", "csv:exp4c_misspecification.csv", "df.groupby('gamma').width.mean().min()", "f2")
reg("nExpFourMisWidthMax", "csv:exp4c_misspecification.csv", "df.groupby('gamma').width.mean().max()", "f2")
reg("nExpFourMisSeeds", "csv:exp4c_misspecification.csv", "df.seed.nunique()", "int", dataset="n/a")

# ---- exp5 table ----------------------------------------------------------
for g, gtag in ((1.5, "OneFive"), (2, "Two"), (3, "Three"), (6, "Six"), (12, "Twelve"), (24, "TwentyFour")):
    for k, ktag, fmt in (("B", "B", "f2"), ("rate_exponent", "Rate", "f3"), ("coefficient", "Coef", "f3")):
        reg(f"nExpFive{gtag}{ktag}", "csv:exp5_scaling.csv", f"df[np.isclose(df.gamma,{g})].{k}.iloc[0]", fmt)
reg("nExpFiveRateMin", "csv:exp5_scaling.csv", "df.rate_exponent.min()", "f3")
reg("nExpFiveRateMax", "csv:exp5_scaling.csv", "df.rate_exponent.max()", "f3")
reg("nExpFiveGammaRange", "csv:exp5_scaling.csv", "df.gamma.max()/df.gamma.min()", "int", dataset="n/a")


# --------------------------------------------------------------------------- #
def _get(d, path):
    """Dotted path with arithmetic on the last hop ('a/b', 'a*100'), or a python
    expression over the JSON when the path starts with 'expr:' (d = the JSON)."""
    if path.startswith("expr:"):
        return eval(path[5:], {"d": d, "np": np, "len": len, "min": min, "max": max})
    if re.search(r"[*/]", path.split(".")[-1]) and not path.startswith("csv:"):
        base, expr = path.rsplit(".", 1) if "." in path else ("", path)
        toks = re.split(r"([*/])", expr)
        def val(tok):
            tok = tok.strip()
            try:
                return float(tok)
            except ValueError:
                return _get(d, (base + "." + tok) if base else tok)
        acc = val(toks[0])
        for op, tok in zip(toks[1::2], toks[2::2]):
            v = val(tok)
            acc = acc / v if op == "/" else acc * v
        return acc
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise KeyError(path)
    return cur


def fmt_value(v, fmt):
    if isinstance(v, dict) and "mean" in v:
        v = v["mean"]
    if fmt == "bool":
        return "yes" if bool(v) else "no"
    if fmt == "int":
        return f"{int(round(float(v))):,}".replace(",", "{,}")
    if fmt == "pct0":
        return f"{100*float(v):.0f}\\%"
    if fmt == "pct1":
        return f"{100*float(v):.1f}\\%"
    if fmt == "pchg0":                    # ratio -> signed percent change
        return f"{100*(float(v)-1):+.0f}\\%"
    if fmt == "x1":
        return f"{float(v):.1f}"
    if fmt == "sci1":
        m, e = f"{float(v):.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    if fmt.endswith("s"):          # signed
        n = int(fmt[1:-1]); return f"{float(v):+.{n}f}"
    if fmt[0] == "f":
        return f"{float(v):.{int(fmt[1:])}f}"
    if fmt[0] == "g":
        return f"{float(v):.{int(fmt[1:])}g}"
    raise ValueError(fmt)


def build():
    lines, prov, warnings = [], {}, []
    cache = {}
    for e in R:
        f = e["file"]
        src_desc = f"{f}:{e['path']}"
        try:
            if f.startswith("csv:"):
                path = os.path.join(RES, f[4:])
                if path not in cache:
                    cache[path] = pd.read_csv(path)
                df = cache[path]
                v = eval(e["path"], {"df": df, "np": np, "len": len})
                # provenance of CSVs: inherit from the sibling summary JSON if any
                mm = re.match(r"^(exp\d+)[a-z]?_", f[4:])
                sib = f"{mm.group(1)}_summary.json" if mm else f[4:]
                ds_src = _provenance(sib, e["dataset"])
                src_desc = f"csv:{f[4:]}:{e['path']}"
                ci = None
            else:
                path = os.path.join(RES, f)
                if path not in cache:
                    cache[path] = json.load(open(path))
                d = cache[path]
                v = _get(d, e["path"])
                ds_src = _provenance(f, e["dataset"], d)
                src_desc = f"{f}:{e['path']}"
                ci = v.get("ci95") if isinstance(v, dict) else None
                if isinstance(v, dict) and e["fmt"].startswith("ci:") is False and "mean" in v:
                    v = v["mean"]
        except (FileNotFoundError, KeyError, IndexError, AttributeError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
            warnings.append(f"{e['macro']}: {src_desc} -> {type(exc).__name__}: {exc}")
            # PENDING placeholder: lets the paper compile while a result is being
            # (re)computed; check_provenance.py fails on any 'pending' macro, so a
            # placeholder can never survive into a verified build.
            lines.append(f"\\newcommand{{\\{e['macro']}}}{{\\textbf{{??}}}}")
            if e["fmt"].startswith("ci:"):
                lines.append(f"\\newcommand{{\\{e['macro']}CI}}{{\\textbf{{??}}}}")
            prov[e["macro"]] = dict(source=src_desc, data_source="pending", n_seeds=None)
            continue
        base_fmt = e["fmt"][3:] if e["fmt"].startswith("ci:") else e["fmt"]
        lines.append(f"\\newcommand{{\\{e['macro']}}}{{{fmt_value(v, base_fmt)}}}")
        if e["fmt"].startswith("ci:") and ci is not None and np.isfinite(ci):
            lines.append(f"\\newcommand{{\\{e['macro']}CI}}{{{fmt_value(ci, base_fmt)}}}")
        prov[e["macro"]] = dict(source=src_desc, data_source=ds_src,
                                n_seeds=(v.get("n") if isinstance(v, dict) else None))
    header = ["% AUTO-GENERATED by scripts/make_numbers.py -- do not edit.",
              "% Every number in the paper body is one of these macros (CLAUDE.md rule 2).",
              "% Provenance for each macro: paper/numbers_provenance.json (rule 3)."]
    return "\n".join(header + sorted(lines)) + "\n", prov, warnings


def _provenance(json_name, dataset, d=None):
    if dataset == "n/a":                  # design constants (seed counts, grid sizes) carry no data provenance
        return "n/a"
    path = os.path.join(RES, json_name)
    if d is None:
        if not os.path.exists(path):
            return "unknown"
        d = json.load(open(path))
    if "data_source" not in d:
        raise ValueError(f"{json_name} has no data_source field (CLAUDE.md rule 3)")
    if dataset and "data_source_by_dataset" in d:
        return d["data_source_by_dataset"].get(dataset, d["data_source"])
    return d["data_source"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if numbers.tex is stale")
    a = ap.parse_args()
    tex, prov, warnings = build()
    for w in warnings:
        print("  warning:", w)
    if a.check:
        cur = open(OUT_TEX).read() if os.path.exists(OUT_TEX) else ""
        if cur != tex:
            print("numbers.tex is STALE: run scripts/make_numbers.py"); return 1
        print(f"numbers.tex up to date ({len(prov)} macros)"); return 0
    open(OUT_TEX, "w").write(tex)
    json.dump(prov, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_TEX} ({len(prov)} macros, {len(warnings)} warnings) and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
