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

Time-at-risk correction (``outcome="exposure_adjusted"``).  ProPublica's file
carries the survival fields it used for its Cox model: ``start`` (days from
screening to the start of the at-risk period, i.e. release from the index
custody spell), ``end`` (days from screening to the recidivism offence or to the
end of follow-up) and ``event``.  ``end - start`` is therefore time *at risk in
the community*.  The exposure-adjusted outcome is

    Y_exp = 1{event == 1 and end - start <= exposure_days},

and units with neither an event nor ``exposure_days`` of at-risk follow-up are
dropped (they cannot be classified).  This holds exposure fixed across the
released and detained groups and removes the mechanical incapacitation effect;
``audit/compas_time_at_risk.py`` quantifies what it changes (on the ProPublica
cohort the detained/released odds ratio *rises* from about 1.9 to about 2.1,
i.e. incapacitation was masking selection, not creating it).  The recorded
outcome ``two_year_recid`` remains the primary target in the paper because it
is the quantity a deployed model is scored on; the exposure-adjusted variant is
the sensitivity analysis.  Neither is a "ground truth" in the causal sense: we
call them *recorded* outcomes.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import SelectiveLabelsDataset

__all__ = ["make_compas"]


OUTCOMES = ("recorded", "exposure_adjusted")


def make_compas(
    data_dir: str = "data/raw",
    release_days: float = 2.0,
    max_custody_days: float = 180.0,
    seed: int = 0,
    outcome: str = "recorded",
    exposure_days: int = 730,
) -> SelectiveLabelsDataset:
    """COMPAS as a selective-labels instance.

    Parameters
    ----------
    outcome : {"recorded", "exposure_adjusted"}
        ``"recorded"`` scores ``two_year_recid`` as published; ``"exposure_adjusted"``
        scores rearrest within ``exposure_days`` of *time at risk* (see module
        docstring) and drops units with insufficient at-risk follow-up.
    """
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
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
    Y_rec = df["two_year_recid"].astype(float).to_numpy()
    n_cohort = int(len(df))
    at_risk = (df["end"].astype(float) - df["start"].astype(float)).to_numpy()
    event = df["event"].astype(float).to_numpy()
    Y_exp_all = ((event == 1) & (at_risk <= exposure_days)).astype(float)
    exposed = (at_risk >= exposure_days) | (event == 1)
    if outcome == "exposure_adjusted":
        df, days = df[exposed].copy(), days[exposed]
        T, Y = T[exposed], Y_exp_all[exposed]
    else:
        Y = Y_rec

    num = df[["age", "priors_count", "juv_fel_count", "juv_misd_count",
              "juv_other_count"]].astype(float)
    cat = pd.get_dummies(
        df[["c_charge_degree", "sex", "race"]].astype(str), drop_first=True
    ).astype(float)
    Xdf = pd.concat([num, cat], axis=1).fillna(0.0)
    X = Xdf.to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))

    Y_obs = np.where(T == 1, Y, np.nan)

    def _or(y, t):
        a = y[t == 1].mean(); b = y[t == 0].mean()
        return float((b / (1 - b)) / (a / (1 - a)))

    or_marg = _or(Y, T)
    return SelectiveLabelsDataset(
        X=X, T=T, Y_obs=Y_obs, Y_full=Y, Z=None,
        feature_names=list(Xdf.columns),
        name="COMPAS(real decisions, %s outcome)" % outcome,
        oracle=dict(marginal_odds_ratio=or_marg, outcome=outcome,
                    exposure_days=int(exposure_days), n_cohort=n_cohort,
                    n_kept=int(len(Y)), custody_days=days.to_numpy(),
                    marginal_odds_ratio_recorded=_or(Y_rec, (days.to_numpy() <= release_days).astype(float))
                    if outcome == "recorded" else None),
        notes=("Real pretrial release decisions; two-year rearrest RECORDED for "
               "released AND detained defendants, so a recorded deployment outcome "
               "is available for every unit (not a causal ground truth: detention "
               "also incapacitates). Custody spells > %g days excluded; outcome=%s."
               % (max_custody_days, outcome)),
    )
