#!/usr/bin/env python3
"""COMPAS lineage and time-at-risk audit (A_AUDIT.md section 2).

Quantifies what the incapacitation caveat does to the numbers the paper reports
on COMPAS:

* how custody time is distributed in the released (T=1) and detained (T=0)
  groups after the paper's cohort filter;
* the detained/released odds ratio under the RECORDED outcome
  (``two_year_recid``) and under the EXPOSURE-ADJUSTED outcome (rearrest within
  W days at risk in the community, W in {365, 540, 730});
* how many units the exposure adjustment drops, by group.

Writes results/compas_time_at_risk.json (data_source: real) so the paper cites
these numbers through macros only.  Exits 2 if the ProPublica file is missing
(CLAUDE.md rule 1: never substitute a simulator).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PATH = os.path.join(ROOT, "data", "raw", "compas_two_years.csv")


def main() -> int:
    if not os.path.exists(PATH):
        print(f"ERROR: {PATH} missing; run scripts/download_data.py (exit 2)", file=sys.stderr)
        return 2
    df = pd.read_csv(PATH, low_memory=False)
    n_raw = len(df)
    df = df[(df.days_b_screening_arrest.abs() <= 30) & (df.is_recid != -1)
            & (df.c_charge_degree != "O") & (df.score_text != "N/A")].copy()
    n_propublica = len(df)
    jin = pd.to_datetime(df.c_jail_in, errors="coerce"); jout = pd.to_datetime(df.c_jail_out, errors="coerce")
    days = (jout - jin).dt.total_seconds() / 86400.0
    df = df[days.notna()].copy(); days = days[days.notna()]
    n_with_jail_dates = len(df)
    keep = days <= 180.0
    df, days = df[keep].copy(), days[keep]
    n_cohort = len(df)
    T = (days.to_numpy() <= 2.0).astype(int)
    at_risk = (df["end"] - df["start"]).to_numpy(float)
    event = df["event"].to_numpy(float)
    y_rec = df["two_year_recid"].to_numpy(float)

    def odds_ratio(y, t):
        a = y[t == 1].mean(); b = y[t == 0].mean()
        return float((b / (1 - b)) / (a / (1 - a)))

    out = dict(
        data_source="real",
        source_file="compas-scores-two-years.csv (ProPublica, Broward County FL)",
        n_raw=int(n_raw), n_after_propublica_filter=int(n_propublica),
        n_with_jail_dates=int(n_with_jail_dates), n_cohort=int(n_cohort),
        n_released=int((T == 1).sum()), n_detained=int((T == 0).sum()),
        selection_rate=float(T.mean()),
        custody_days_released=dict(median=float(np.median(days[T == 1])), q90=float(np.quantile(days[T == 1], .9))),
        custody_days_detained=dict(median=float(np.median(days[T == 0])), q90=float(np.quantile(days[T == 0], .9)),
                                   max=float(days[T == 0].max())),
        start_days_detained_median=float(np.median(df["start"].to_numpy(float)[T == 0])),
        corr_start_custody=float(np.corrcoef(df["start"].to_numpy(float), days.to_numpy())[0, 1]),
        recorded=dict(p1=float(y_rec[T == 1].mean()), p0=float(y_rec[T == 0].mean()),
                      odds_ratio=odds_ratio(y_rec, T)),
        exposure_adjusted={},
    )
    for W in (365, 540, 730):
        ok = (at_risk >= W) | (event == 1)
        y = ((event == 1) & (at_risk <= W)).astype(float)
        out["exposure_adjusted"][str(W)] = dict(
            exposure_days=W, frac_kept=float(ok.mean()),
            frac_kept_released=float(ok[T == 1].mean()), frac_kept_detained=float(ok[T == 0].mean()),
            n_kept=int(ok.sum()),
            p1=float(y[ok & (T == 1)].mean()), p0=float(y[ok & (T == 0)].mean()),
            odds_ratio=odds_ratio(y[ok], T[ok]),
        )
    out["odds_ratio_change_730"] = out["exposure_adjusted"]["730"]["odds_ratio"] - out["recorded"]["odds_ratio"]
    out["incapacitation_masks_selection"] = bool(out["odds_ratio_change_730"] > 0)
    path = os.path.join(ROOT, "results", "compas_time_at_risk.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "exposure_adjusted"}, indent=1))
    for W, r in out["exposure_adjusted"].items():
        print(f"W={W}: kept {r['frac_kept']:.3f} (released {r['frac_kept_released']:.3f}, detained {r['frac_kept_detained']:.3f}); "
              f"p1={r['p1']:.3f} p0={r['p0']:.3f} OR={r['odds_ratio']:.3f}")
    print("->", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
