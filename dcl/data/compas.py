"""COMPAS: a *real* decision with *real* outcomes on both sides of it.

Source.  ProPublica's ``compas-scores-two-years.csv`` (Broward County, FL;
n = 7,214 pretrial defendants).

Why this dataset matters here.  It is the rare selective-labels instance where
the censored outcomes are actually recorded.  The judge's pretrial release
decision is real, and two-year rearrest is recorded for detained and released
defendants alike -- so we can impose the *real* censoring pattern, hide the
detained defendants' labels from every method, and still evaluate deployment
risk against the truth.  No simulated policy is involved.

    T = 1  <=>  released quickly (custody spell <= ``release_days``)
    Y      =  ``two_year_recid``   (ground truth for all units)

The empirical odds ratio between detained and released rearrest rates is about
1.8, which is where the ``Gamma`` values we report as "realistic" come from.

Caveat, stated plainly.  Detention is partly *incapacitating*: a defendant held
for much of the two-year window has less opportunity to be rearrested, so the
observed detained-group rate mixes selection with a mechanical effect.  This
biases the realised odds ratio *downward* (detention suppresses rearrest) while
selection on dangerousness pushes it up.  We therefore report COMPAS as a
coverage check for the bounds, not as a precise estimate of ``Gamma_0``, and we
exclude spells longer than ``max_custody_days`` where incapacitation dominates.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .base import SelectiveLabelsDataset

__all__ = ["make_compas"]


def make_compas(
    data_dir: str = "data/raw",
    release_days: float = 2.0,
    max_custody_days: float = 180.0,
    seed: int = 0,
) -> SelectiveLabelsDataset:
    path = os.path.join(data_dir, "compas_two_years.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path)

    # ProPublica's published cohort filter.
    df = df[(df["days_b_screening_arrest"].abs() <= 30)
            & (df["is_recid"] != -1)
            & (df["c_charge_degree"] != "O")
            & (df["score_text"] != "N/A")].copy()

    jin = pd.to_datetime(df["c_jail_in"], errors="coerce")
    jout = pd.to_datetime(df["c_jail_out"], errors="coerce")
    days = (jout - jin).dt.total_seconds() / 86400.0
    df = df[days.notna()].copy()
    days = days[days.notna()]
    keep = days <= max_custody_days
    df, days = df[keep].copy(), days[keep]

    T = (days.to_numpy() <= release_days).astype(float)
    Y = df["two_year_recid"].astype(float).to_numpy()

    num = df[["age", "priors_count", "juv_fel_count", "juv_misd_count",
              "juv_other_count"]].astype(float)
    cat = pd.get_dummies(
        df[["c_charge_degree", "sex", "race"]].astype(str), drop_first=True
    ).astype(float)
    Xdf = pd.concat([num, cat], axis=1).fillna(0.0)
    X = Xdf.to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))

    Y_obs = np.where(T == 1, Y, np.nan)

    p1m = Y[T == 1].mean(); p0m = Y[T == 0].mean()
    or_marg = (p0m / (1 - p0m)) / (p1m / (1 - p1m))

    return SelectiveLabelsDataset(
        X=X, T=T, Y_obs=Y_obs, Y_full=Y, Z=None,
        feature_names=list(Xdf.columns), name="COMPAS(real decisions)",
        oracle=dict(marginal_odds_ratio=float(or_marg)),
        notes=("Real pretrial release decisions; two-year rearrest recorded for "
               "released AND detained defendants, so deployment risk is "
               "evaluable. Custody spells > %g days excluded (incapacitation)."
               % max_custody_days),
    )
