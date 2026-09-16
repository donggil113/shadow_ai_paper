"""Experiment 8 -- the nuisance theorem in practice: coverage under ESTIMATED nuisances.

Protocol (per dataset, per seed): 50% train / 25% calibration / 25% test.
Nuisances are cross-fitted on train and predicted on calibration + test.
The calibration fold supplies the binomial counts for the binned intervals; the
test fold is where every coverage number is measured.

Reported, with 95% CIs over 5 seeds:
  * pointwise coverage of the TRUE p(x) (available on SL-Bench exactly and on
    Lending Club through its semi-synthetic oracle) by the plug-in box and by
    the inflated boxes (10/20 bins, with and without the smoothness margin),
    and their mean widths;
  * on COMPAS (no pointwise truth) bin-level coverage of the realised outcome
    rate by the bin-average box, with the binomial noise of the check stated;
  * DR (one-step) confidence intervals for the prevalence bounds and whether
    they cover the identified prevalence interval (oracle on SL-Bench/Lending;
    realised prevalence on COMPAS);
  * the sharp AUROC interval of the incumbent model over the plug-in box vs.
    over the inflated box (the latter is a valid superset), widths and coverage
    of the true deployment AUROC;
  * the diagnostic of the theorem: coverage when only e_hat is replaced by the
    oracle e, and when only p1_hat is replaced by the oracle p1 -- which
    nuisance term dominates the miss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from _common import Timer, ci95, save, save_table, stamp_provenance
from dcl.auc_bounds import auc_from_regression, midrank, sharp_auc_interval
from dcl.baselines import ERMObserved
from dcl.data import make_compas, make_lending_club, make_sl_bench
from dcl.intervals import (binned_nuisance_intervals, g_tilt, inflate_box,
                           prevalence_bounds_dr)
from dcl.nuisance import CrossFitNuisance
from dcl.sensitivity import outcome_bounds

DATASETS = {
    "sl_bench": (lambda sd: make_sl_bench(n=16000, d=10, target_gamma=3.0, seed=sd), "synthetic"),
    "lending_club": (lambda sd: make_lending_club(target_gamma=2.0, seed=sd), "semi-synthetic"),
    "compas": (lambda sd: make_compas(), "real"),
}
COMPAS_GAMMA = 1.885          # the data's own detained/released odds ratio (exp6)
CONFIGS = [("bins10", 10, False), ("bins20", 20, False),
           ("bins10_smooth", 10, True), ("bins20_smooth", 20, True)]


def _three_way_split(n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a, b = int(0.5 * n), int(0.75 * n)
    return idx[:a], idx[a:b], idx[b:]


def run_one(name, maker, seed, alpha=0.10):
    ds = maker(seed)
    gamma = float(ds.oracle.get("gamma0", COMPAS_GAMMA))
    has_oracle = "p" in ds.oracle
    itr, ica, ite = _three_way_split(ds.n, seed)
    Xtr, Ttr, Ytr = ds.X[itr], ds.T[itr], ds.Y_obs[itr]
    cf = CrossFitNuisance(seed=seed)
    cf.fit_predict(Xtr, Ttr, Ytr)
    rest = np.concatenate([ica, ite])
    nu = cf.predict(ds.X[rest])
    e_hat, p1_hat = nu.e, nu.p1
    T_r, Yo_r = ds.T[rest], ds.Y_obs[rest]
    calib = np.zeros(rest.size, bool); calib[:ica.size] = True
    test = ~calib
    row = dict(dataset=name, seed=seed, gamma=gamma, n_test=int(test.sum()))

    # ------------------------------------------------------------ plug-in box
    lo_p, hi_p = outcome_bounds(p1_hat, e_hat, gamma)
    if has_oracle:
        p_true = ds.oracle["p"][rest]
        e_true, p1_true = ds.oracle["e"][rest], ds.oracle["p1"][rest]
        row["cov_plugin"] = float(np.mean(((p_true >= lo_p) & (p_true <= hi_p))[test]))
        # diagnostic: which nuisance drives the miss
        lo_e, hi_e = outcome_bounds(p1_hat, e_true, gamma)       # oracle e, estimated p1
        lo_q, hi_q = outcome_bounds(p1_true, e_hat, gamma)       # estimated e, oracle p1
        row["cov_oracle_e_est_p1"] = float(np.mean(((p_true >= lo_e) & (p_true <= hi_e))[test]))
        row["cov_est_e_oracle_p1"] = float(np.mean(((p_true >= lo_q) & (p_true <= hi_q))[test]))
        row["mae_e"] = float(np.mean(np.abs(e_hat - e_true)[test]))
        row["mae_p1"] = float(np.mean(np.abs(p1_hat - p1_true)[test]))
    row["width_plugin"] = float(np.mean((hi_p - lo_p)[test]))

    # ------------------------------------------------------- inflated boxes
    boxes = {}
    for tag, nb, sm in CONFIGS:
        iv = binned_nuisance_intervals(e_hat, p1_hat, T_r, Yo_r, alpha=alpha,
                                       n_bins_e=nb, n_bins_p=nb, smooth_margin=sm,
                                       fit_mask=calib)
        box = inflate_box(iv["e_L"], iv["e_U"], iv["eta_L"], iv["eta_U"], gamma)
        boxes[tag] = box
        row[f"width_{tag}"] = float(np.mean(box.width[test]))
        if has_oracle:
            row[f"cov_{tag}"] = float(np.mean(box.covers(p_true)[test]))

    # --------------------------------------- COMPAS: bin-level realised-rate check
    if not has_oracle and ds.Y_full is not None:
        yf = ds.Y_full[rest]
        # bins of the TEST units by plug-in midpoint, >= 150 units each
        mid = 0.5 * (lo_p + hi_p)
        te_idx = np.flatnonzero(test)
        order = te_idx[np.argsort(mid[te_idx])]
        nbins = max(1, order.size // 150)
        chunks = np.array_split(order, nbins)
        for tag, box in [("plugin", None)] + list(boxes.items()):
            lo_b = lo_p if box is None else box.lo
            hi_b = hi_p if box is None else box.hi
            ok = 0
            for ch in chunks:
                rate = yf[ch].mean()
                # binomial half-width of the realised rate itself (the check is noisy)
                hw = 1.96 * np.sqrt(rate * (1 - rate) / max(len(ch), 1))
                ok += (rate + hw >= lo_b[ch].mean()) and (rate - hw <= hi_b[ch].mean())
            row[f"binlevel_cov_{tag}"] = ok / len(chunks)
        row["binlevel_n_bins"] = len(chunks)

    # ------------------------------------------------ DR prevalence bounds (test)
    dr = prevalence_bounds_dr(T_r[test], Yo_r[test], e_hat[test], p1_hat[test], gamma, alpha=alpha)
    row["dr_pi_lo"], row["dr_pi_hi"] = dr["pi_lo"].estimate, dr["pi_hi"].estimate
    row["dr_outer_lo"], row["dr_outer_hi"] = dr["outer"]
    row["dr_se_lo"], row["dr_se_hi"] = dr["pi_lo"].se, dr["pi_hi"].se
    row["plug_pi_lo"], row["plug_pi_hi"] = dr["plugin"]
    if has_oracle:
        pi_lo_true = float(np.mean(e_true * p1_true + (1 - e_true) * g_tilt(p1_true, 1 / gamma))[test].mean()) if False else \
            float(np.mean((e_true * p1_true + (1 - e_true) * g_tilt(p1_true, 1 / gamma))[test]))
        pi_hi_true = float(np.mean((e_true * p1_true + (1 - e_true) * g_tilt(p1_true, gamma))[test]))
        row["pi_lo_true"], row["pi_hi_true"] = pi_lo_true, pi_hi_true
        row["dr_covers_prevalence_interval"] = float(dr["outer"][0] <= pi_lo_true and dr["outer"][1] >= pi_hi_true)
        row["plug_covers_prevalence_interval"] = float(dr["plugin"][0] <= pi_lo_true and dr["plugin"][1] >= pi_hi_true)
        row["dr_abs_err_lo"] = abs(dr["pi_lo"].estimate - pi_lo_true)
        row["plug_abs_err_lo"] = abs(dr["plugin"][0] - pi_lo_true)
        row["dr_abs_err_hi"] = abs(dr["pi_hi"].estimate - pi_hi_true)
        row["plug_abs_err_hi"] = abs(dr["plugin"][1] - pi_hi_true)
    if ds.Y_full is not None:
        prev = float(ds.Y_full[rest][test].mean())
        row["realised_prevalence"] = prev
        row["dr_outer_covers_realised"] = float(dr["outer"][0] <= prev <= dr["outer"][1])
        row["plug_covers_realised"] = float(dr["plugin"][0] <= prev <= dr["plugin"][1])

    # ----------------------------------- AUROC interval: plug-in vs inflated box
    f = ERMObserved(seed=seed).fit(Xtr, Ttr, Ytr).decision_function(ds.X[rest])
    f_te = f[test]
    res_p = sharp_auc_interval(f_te, lo_p[test], hi_p[test])
    row["auc_lo_plugin"], row["auc_hi_plugin"] = res_p.lower, res_p.upper
    for tag, box in boxes.items():
        r = sharp_auc_interval(f_te, box.lo[test], box.hi[test])
        row[f"auc_lo_{tag}"], row[f"auc_hi_{tag}"] = r.lower, r.upper
        row[f"auc_width_{tag}"] = r.width
    row["auc_width_plugin"] = res_p.width
    if has_oracle:
        true_auc = auc_from_regression(p_true[test], midrank(f_te))
        row["true_auc_pop"] = true_auc
        row["auc_cov_plugin"] = float(res_p.contains(true_auc))
        for tag in boxes:
            row[f"auc_cov_{tag}"] = float(row[f"auc_lo_{tag}"] - 1e-9 <= true_auc <= row[f"auc_hi_{tag}"] + 1e-9)
    if ds.Y_full is not None:
        emp = float(roc_auc_score(ds.Y_full[rest][test], f_te))
        row["true_auc_emp"] = emp
        row["auc_cov_emp_plugin"] = float(res_p.contains(emp))
        for tag in boxes:
            row[f"auc_cov_emp_{tag}"] = float(row[f"auc_lo_{tag}"] - 1e-9 <= emp <= row[f"auc_hi_{tag}"] + 1e-9)
    return row


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for name, sub in df.groupby("dataset"):
        d = {}
        for col in sub.columns:
            if col in ("dataset", "seed"):
                continue
            v = sub[col].dropna()
            if len(v) == 0 or not np.issubdtype(v.dtype, np.number):
                continue
            m, hw, n = ci95(v)
            d[col] = {"mean": m, "ci95": hw, "n": n}
        out[name] = d
    return out


if __name__ == "__main__":
    rows = []
    with Timer("exp8 nuisance coverage (5 seeds)"):
        for name, (maker, _src) in DATASETS.items():
            for sd in range(5):
                rows.append(run_one(name, maker, sd))
                r = rows[-1]
                msg = " ".join(f"{k}={r[k]:.3f}" for k in ("cov_plugin", "cov_bins20", "cov_bins20_smooth") if k in r)
                print(f"    {name} seed={sd} {msg} dr_cov={r.get('dr_covers_prevalence_interval', r.get('dr_outer_covers_realised'))}", flush=True)
    df = pd.DataFrame(rows)
    save_table("exp8_nuisance_coverage_raw", df)
    summary = summarise(df)
    save("exp8_summary", dict(datasets=summary, n_seeds=int(df.seed.nunique()), alpha=0.10,
                              configs=[c[0] for c in CONFIGS]))
    stamp_provenance("exp8_summary", "real",
                     per_dataset={k: v[1] for k, v in DATASETS.items()})
    for name, d in summary.items():
        print(f"\n=== {name} ===")
        for k in sorted(d):
            if k.startswith(("cov_", "width_", "dr_covers", "plug_covers", "auc_width", "auc_cov", "binlevel", "dr_abs", "plug_abs", "mae_")):
                print(f"  {k:32s} {d[k]['mean']:8.4f} +- {d[k]['ci95']:.4f}")
