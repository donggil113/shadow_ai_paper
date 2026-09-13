"""Experiment 2 -- Theorem 2: decision-censored learning against the field.

Every number is out of sample.  For each testbed we report, per method:

* ``risk`` / ``auroc`` / ``accuracy`` -- the **realised deployment** metrics on
  ground truth for *all* units, censored ones included.  Available only because
  the benchmarks retain the hidden labels; unavailable in any real deployment.
* ``obs_auroc`` -- what a practitioner would actually report, computed on the
  labelled sub-population.  The gap between this and ``auroc`` is the blind spot.
* ``worstcase_risk`` and ``minimax_regret`` -- the *certified* quantities:
  guarantees that hold for every member of the identified set, and therefore the
  only ones a deployer can rely on.

The expected pattern, and the honest trade-off: the minimax-*risk* rule buys the
best certificate at some cost in realised performance (that is what robustness
costs); the minimax-*regret* rule of Proposition 3 gives up almost nothing
realised while still cutting the certified regret several-fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Timer, save, save_table
from dcl.data import make_compas, make_lending_club, make_mimic_sim, make_sl_bench
from dcl.harness import run_methods

TESTBEDS = {
    "sl_bench": lambda sd: make_sl_bench(n=16000, d=10, target_gamma=3.0, seed=sd),
    "lending_club": lambda sd: make_lending_club(target_gamma=2.0, seed=sd),
    "mimic_sim": lambda sd: make_mimic_sim(n=12000, target_gamma=2.5, seed=sd),
}


def run(n_seeds: int = 3, gammas=(1.0, 1.5, 2.0, 3.0, 5.0)):
    rows = []
    for name, maker in TESTBEDS.items():
        for sd in range(n_seeds):
            ds = maker(sd)
            g0 = ds.oracle.get("gamma0", np.nan)
            res = run_methods(ds, gammas=gammas, gamma_report=g0, seed=sd,
                              direction="two-sided")
            for r in res:
                r.update(dataset=name, seed=sd, gamma0=g0,
                         selection_rate=ds.selection_rate,
                         true_prevalence=ds.true_prevalence)
            rows += res
            # the directional variant, where the sign of selection is known
            cen = float(np.mean(ds.Y_full[ds.T == 0]))
            sel = float(np.mean(ds.Y_full[ds.T == 1]))
            direction = "censored-lower" if cen < sel else "censored-higher"
            res2 = run_methods(ds, gammas=gammas, gamma_report=g0, seed=sd,
                               direction=direction, include_heckman=False,
                               include_oracle=False)
            for r in res2:
                if str(r.get("method", "")).startswith("DCL"):
                    r["method"] = r["method"].replace("DCL", "DCL-dir")
                    r.update(dataset=name, seed=sd, gamma0=g0,
                             selection_rate=ds.selection_rate,
                             true_prevalence=ds.true_prevalence)
                    rows.append(r)
            print(f"    {name} seed={sd} Gamma0={g0:.2f} done", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["risk", "auroc", "accuracy", "obs_auroc", "worstcase_risk",
            "minimax_regret", "auroc_lo", "auroc_hi", "abstention_rate"]
    g = df.groupby(["dataset", "method"])[keep]
    out = g.mean().join(g.std(), rsuffix="_sd").reset_index()
    return out


if __name__ == "__main__":
    with Timer("exp2 learning"):
        df = run()
    save_table("exp2_learning_raw", df)
    s = summarise(df)
    save_table("exp2_learning_summary", s)
    for ds_name in s.dataset.unique():
        sub = s[s.dataset == ds_name].copy()
        sub = sub.sort_values("minimax_regret")
        print(f"\n=== {ds_name} (mean over seeds) ===")
        cols = ["method", "risk", "auroc", "accuracy", "obs_auroc",
                "worstcase_risk", "minimax_regret"]
        print(sub[cols].to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
