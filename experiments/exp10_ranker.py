"""Experiment 10 -- does robust-ranking training beat the plug-in midpoint ordering?

For each benchmark and seed: cross-fit the nuisances, form the box at the
instance's Gamma_0, and compare the exact worst-case AUROC (Theorem: sharp
interval, lower endpoint) on held-out units of (i) the plug-in orderings by
lo / midpoint / hi / p1, (ii) DCLRanker (Danskin best response with fictitious
play, warm-started at the midpoint), (iii) ranking by the width alone (sanity).
The quantity the paper cites is the largest gain of DCLRanker over the midpoint
ordering across datasets and seeds.  5 seeds, 95% CIs; provenance per dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Timer, ci95, save, save_table, stamp_provenance
from dcl.auc_bounds import sharp_auc_interval
from dcl.data import make_lending_club, make_sl_bench
from dcl.nuisance import CrossFitNuisance
from dcl.ranking import DCLRanker
from dcl.sensitivity import outcome_bounds

DATASETS = {
    "sl_bench": (lambda seed: make_sl_bench(n=6000, target_gamma=2.5, seed=seed), "synthetic"),
    "lending_club": (lambda seed: make_lending_club(seed=seed, target_gamma=2.0), "semi-synthetic"),
}


def wc(scores, lo, hi):
    return float(sharp_auc_interval(scores, lo, hi).lower)


def run(n_seeds=5):
    rows = []
    for name, (maker, _) in DATASETS.items():
        for sd in range(n_seeds):
            ds = maker(sd)
            tr, te = ds.split(test_size=0.4, seed=sd)
            cf = CrossFitNuisance(seed=sd)
            cf.fit_predict(tr.X, tr.T, tr.Y_obs)
            ntr, nte = cf.predict(tr.X), cf.predict(te.X)
            g = float(ds.oracle.get("gamma0", 2.0))
            lo_tr, hi_tr = outcome_bounds(ntr.p1, ntr.e, g)
            lo_te, hi_te = outcome_bounds(nte.p1, nte.e, g)
            mid_tr = 0.5 * (lo_tr + hi_tr)
            # linear plug-in scorer fit to the midpoint (same class as DCLRanker)
            XtX = tr.X.T @ tr.X + 1e-3 * np.eye(tr.X.shape[1])
            coef_mid = np.linalg.solve(XtX, tr.X.T @ (mid_tr - mid_tr.mean()))
            r = DCLRanker(n_steps=300, oracle_every=10, seed=sd).fit(tr.X, lo_tr, hi_tr)
            res = dict(dataset=name, seed=sd, gamma0=g, n_test=te.n,
                       wc_mid_plugin=wc(0.5 * (lo_te + hi_te), lo_te, hi_te),
                       wc_lo_plugin=wc(lo_te, lo_te, hi_te),
                       wc_hi_plugin=wc(hi_te, lo_te, hi_te),
                       wc_p1_plugin=wc(nte.p1, lo_te, hi_te),
                       wc_width_only=wc(hi_te - lo_te, lo_te, hi_te),
                       wc_mid_linear=wc(te.X @ coef_mid, lo_te, hi_te),
                       wc_ranker=wc(r.decision_function(te.X), lo_te, hi_te),
                       wc_ranker_train=float(r.best_worst_case_auc_),
                       wc_mid_linear_train=wc(tr.X @ coef_mid, lo_tr, hi_tr))
            res["gain_ranker_over_mid_linear"] = res["wc_ranker"] - res["wc_mid_linear"]
            res["gain_ranker_over_mid_plugin"] = res["wc_ranker"] - res["wc_mid_plugin"]
            res["gain_ranker_train"] = res["wc_ranker_train"] - res["wc_mid_linear_train"]
            res["spread_plugin_orderings"] = max(res["wc_mid_plugin"], res["wc_lo_plugin"], res["wc_hi_plugin"], res["wc_p1_plugin"]) \
                - min(res["wc_mid_plugin"], res["wc_lo_plugin"], res["wc_hi_plugin"], res["wc_p1_plugin"])
            rows.append(res)
            print(f"  {name} seed {sd}: mid {res['wc_mid_linear']:.4f} ranker {res['wc_ranker']:.4f} "
                  f"gain {res['gain_ranker_over_mid_linear']:+.4f} (train {res['gain_ranker_train']:+.4f})", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with Timer("exp10 ranker vs midpoint"):
        df = run()
    save_table("exp10_ranker_raw", df)
    out = dict(n_seeds=int(df.seed.nunique()), datasets={})
    for name, g in df.groupby("dataset"):
        d = {}
        for k in ("wc_mid_plugin", "wc_lo_plugin", "wc_hi_plugin", "wc_p1_plugin", "wc_width_only", "wc_mid_linear",
                  "wc_ranker", "gain_ranker_over_mid_linear", "gain_ranker_over_mid_plugin", "gain_ranker_train",
                  "spread_plugin_orderings"):
            m, hw, n = ci95(g[k].to_numpy(float)); d[k] = dict(mean=m, ci95=hw, n=n, max=float(g[k].max()))
        out["datasets"][name] = d
    out["ranker_max_gain_over_midpoint"] = float(df.gain_ranker_over_mid_linear.max())
    out["ranker_max_gain_train"] = float(df.gain_ranker_train.max())
    out["max_spread_plugin_orderings"] = float(df.spread_plugin_orderings.max())
    save("exp10_summary", out)
    stamp_provenance("exp10_summary", "synthetic", {k: v[1] for k, v in DATASETS.items()})
    print(df.groupby("dataset")[["wc_mid_linear", "wc_ranker", "gain_ranker_over_mid_linear", "wc_width_only"]].mean())
