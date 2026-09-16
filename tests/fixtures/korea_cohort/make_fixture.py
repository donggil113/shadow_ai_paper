#!/usr/bin/env python3
"""SYNTHETIC fixture for the Korean health-screening cohort pipeline (T7).

*** EVERYTHING THIS SCRIPT WRITES IS SYNTHETIC. ***  No Korean screening data
were used, seen or approximated from individual records.  The values are drawn
from a deterministic numpy generator (seed ``SEED``) with hand-picked
population-level parameters that merely look like an adult screening panel.
The fixture exists ONLY so that ``pytest`` can exercise
``scripts/korea_cohort/prepare_korea_cohort.py`` and
``scripts/korea_cohort/analyze_korea_cohort.py`` end-to-end.  It is never
written to ``results/`` and no number derived from it may be reported anywhere
(CLAUDE.md rules 1 and 3).

Design (so that every rule of the pipeline is exercised at least once):

* ~400 rows, 5 screening centres ``KC01``..``KC05`` (the leniency instrument
  ``Z``), pseudonymised subject ids ``SYN-000001`` .. (clearly synthetic);
* a hidden signal ``s`` (what the physician sees but the panel does not) enters
  both the follow-up-test decision ``T`` and the outcome ``Y``, so the true
  ``Gamma`` is > 1 and the centres differ in leniency (``ALPHA``) and in how
  much they rely on ``s`` (``KAPPA``); centre assignment is independent of
  everything else, so the exclusion restriction holds by construction;
* the outcome is recorded for EVERY row (truth-batch cohort);
* ``public_model_score`` is a noisy logistic risk score of the kind a model
  trained on a US population would produce;
* duplicates: some subjects have two exams in the same calendar year (dedup
  keeps the first), some have exams in consecutive years (both kept);
* window rule: a few outcome-0 rows have ``followup_days < 365`` (dropped);
* a small cell: only a handful of exams are dated 2021, so the per-year count
  for 2021 is suppressed by the ``>= 10 units per cell`` rule;
* a little missingness in ``bmi``, ``ldl``, ``ggt`` and ``exercise_freq``.

Usage: python3 tests/fixtures/korea_cohort/make_fixture.py [--out PATH]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "screening_exams_fixture.csv")
SEED = 20260916

N_BASE = 345          # subjects with one exam
N_SAME_YEAR_DUP = 35  # subjects with a second exam in the SAME year (dedup drops it)
N_NEXT_YEAR = 20      # subjects with a second exam in the NEXT year (kept)
N_YEAR_2021 = 5       # exams dated 2021 -> small cell in the per-year table
N_ROWS = N_BASE + N_SAME_YEAR_DUP + N_NEXT_YEAR   # = 400

CENTRES = ["KC01", "KC02", "KC03", "KC04", "KC05"]
ALPHA = np.array([-0.2, 0.0, 0.3, 0.6, 1.1])     # centre leniency (logit scale)
KAPPA = np.array([0.6, 1.0, 1.4, 0.8, 1.2])       # reliance on the hidden signal
WINDOW_DAYS = 365


def _expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def _draw_panel(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """One screening panel per row; population-level parameters only."""
    age = rng.integers(20, 80, size=n)
    male = rng.random(n) < 0.52
    bmi = np.clip(rng.normal(23.8 + 0.6 * male, 3.3, n), 14, 45)
    sbp = np.clip(rng.normal(118 + 0.35 * (age - 50) + 4 * male, 14, n), 80, 210)
    dbp = np.clip(rng.normal(74 + 0.1 * (age - 50) + 3 * male, 9, n), 45, 130)
    glucose = np.clip(np.exp(rng.normal(np.log(93) + 0.004 * (age - 50), 0.14, n)), 55, 400)
    tc = np.clip(rng.normal(192 + 0.2 * (age - 50), 34, n), 90, 400)
    hdl = np.clip(rng.normal(58 - 8 * male, 12, n), 20, 120)
    tg = np.clip(np.exp(rng.normal(np.log(110) + 0.25 * male, 0.5, n)), 30, 1200)
    ldl = np.clip(tc - hdl - tg / 5.0 + rng.normal(0, 8, n), 20, 300)
    creat = np.clip(rng.normal(0.72 + 0.25 * male + 0.003 * (age - 50), 0.14, n), 0.3, 4.0)
    kappa = np.where(male, 0.9, 0.7)                       # CKD-EPI-style shape
    egfr = np.clip(142 * np.minimum(creat / kappa, 1.0) ** -0.302
                   * np.maximum(creat / kappa, 1.0) ** -1.2 * 0.9938 ** age
                   * np.where(male, 1.0, 1.012), 15, 150)
    ast = np.clip(np.exp(rng.normal(np.log(23) + 0.1 * male, 0.35, n)), 5, 600)
    alt = np.clip(np.exp(rng.normal(np.log(21) + 0.35 * male + 0.03 * (bmi - 24), 0.5, n)), 5, 800)
    ggt = np.clip(np.exp(rng.normal(np.log(22) + 0.6 * male + 0.04 * (bmi - 24), 0.7, n)), 5, 1500)
    hb = np.clip(rng.normal(13.0 + 2.2 * male, 1.1, n), 7, 19)
    smoking = np.where(male, rng.choice([0, 1, 2], n, p=[0.35, 0.30, 0.35]),
                       rng.choice([0, 1, 2], n, p=[0.90, 0.05, 0.05]))
    alcohol = np.where(male, rng.choice(np.arange(5), n, p=[0.25, 0.25, 0.25, 0.15, 0.10]),
                       rng.choice(np.arange(5), n, p=[0.55, 0.25, 0.12, 0.05, 0.03]))
    exercise = rng.choice(np.arange(8), n, p=[0.35, 0.12, 0.13, 0.13, 0.10, 0.07, 0.05, 0.05])
    fh_dm = rng.random(n) < 0.22
    fh_htn = rng.random(n) < 0.28
    fh_cvd = rng.random(n) < 0.10
    fh_ca = rng.random(n) < 0.18
    return pd.DataFrame({
        "age": age, "sex": np.where(male, "M", "F"),
        "bmi": np.round(bmi, 1), "sbp": np.round(sbp), "dbp": np.round(dbp),
        "fasting_glucose": np.round(glucose), "total_cholesterol": np.round(tc),
        "ldl": np.round(ldl), "hdl": np.round(hdl), "triglycerides": np.round(tg),
        "creatinine": np.round(creat, 2), "egfr": np.round(egfr, 1),
        "ast": np.round(ast), "alt": np.round(alt), "ggt": np.round(ggt),
        "haemoglobin": np.round(hb, 1),
        "smoking_status": smoking, "alcohol_freq": alcohol, "exercise_freq": exercise,
        "fh_diabetes": fh_dm.astype(int), "fh_hypertension": fh_htn.astype(int),
        "fh_cvd": fh_cvd.astype(int), "fh_cancer": fh_ca.astype(int),
    })


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_ROWS

    # --- subject / exam structure -------------------------------------------
    subj = np.arange(1, N_BASE + 1)
    year = np.where(rng.random(N_BASE) < 0.75, 2019, 2020)
    year[:N_YEAR_2021] = 2021                                   # the small cell
    # duplicates never involve the 2021 subjects (keeps that cell at 5 rows) and
    # next-year exams are drawn from 2019 subjects only (so they fall in 2020)
    dup_same = rng.choice(subj[year < 2021], N_SAME_YEAR_DUP, replace=False)
    eligible_next = np.setdiff1d(subj[year == 2019], dup_same)
    dup_next = rng.choice(eligible_next, N_NEXT_YEAR, replace=False)
    subject_id = np.concatenate([subj, dup_same, dup_next])
    year_all = np.concatenate([year, year[dup_same - 1], year[dup_next - 1] + 1])
    day = rng.integers(0, 365, size=n)
    # second exam in the same year: strictly later in that year
    day[N_BASE:N_BASE + N_SAME_YEAR_DUP] = np.minimum(day[dup_same - 1] + rng.integers(20, 200, N_SAME_YEAR_DUP), 364)
    exam_date = pd.to_datetime(year_all.astype(str) + "-01-01") + pd.to_timedelta(day, unit="D")

    centre = rng.choice(len(CENTRES), size=n)
    centre[N_BASE:N_BASE + N_SAME_YEAR_DUP] = centre[dup_same - 1]   # same centre as first exam

    panel = _draw_panel(rng, n)
    # subject-level fields are copied from the first exam of the subject
    for block, src in ((slice(N_BASE, N_BASE + N_SAME_YEAR_DUP), dup_same - 1),
                       (slice(N_BASE + N_SAME_YEAR_DUP, n), dup_next - 1)):
        for c in ("sex", "age", "fh_diabetes", "fh_hypertension", "fh_cvd", "fh_cancer"):
            panel.iloc[block, panel.columns.get_loc(c)] = panel[c].to_numpy()[src]
    age_col = panel.columns.get_loc("age")
    panel.iloc[N_BASE + N_SAME_YEAR_DUP:, age_col] += 1             # a year older

    # --- hidden signal, outcome, decision ------------------------------------
    s = rng.normal(size=n)
    age = panel["age"].to_numpy(float); bmi = panel["bmi"].to_numpy(float)
    sbp = panel["sbp"].to_numpy(float); glu = panel["fasting_glucose"].to_numpy(float)
    cur_smoker = (panel["smoking_status"].to_numpy() == 2).astype(float)
    logit_p = (-1.2 + 0.03 * (age - 50) + 0.08 * (bmi - 24) + 0.02 * (sbp - 120)
               + 0.01 * (glu - 95) + 0.5 * cur_smoker + 0.9 * s)
    y = (rng.random(n) < _expit(logit_p)).astype(int)
    logit_e = (ALPHA[centre] + 0.02 * (age - 50) + 0.01 * (sbp - 120)
               + 0.01 * (glu - 95) + KAPPA[centre] * s)
    t = (rng.random(n) < _expit(logit_e)).astype(int)

    # a "US-trained" public risk score: right direction, foreign calibration
    us_logit = (-1.0 + 0.025 * (age - 50) + 0.10 * (bmi - 27) + 0.015 * (sbp - 125)
                + 0.008 * (glu - 100) + 0.4 * cur_smoker + rng.normal(0, 0.7, n))
    score = np.round(_expit(us_logit), 4)

    # follow-up: events inside the window; non-events mostly reach the window,
    # a few are lost to follow-up (window rule drops them)
    fu = np.where(y == 1, rng.integers(10, WINDOW_DAYS + 1, n), rng.integers(WINDOW_DAYS, 731, n))
    lost = (y == 0) & (rng.random(n) < 0.04)
    fu = np.where(lost, rng.integers(30, WINDOW_DAYS, n), fu)

    df = pd.DataFrame({
        "subject_id": [f"SYN-{k:06d}" for k in subject_id],
        "exam_date": exam_date.strftime("%Y-%m-%d"),
        "centre_id": np.asarray(CENTRES)[centre],
    })
    df = pd.concat([df, panel], axis=1)
    df["followup_test_ordered"] = t
    df["outcome"] = y
    df["public_model_score"] = score
    df["followup_days"] = fu.astype(int)

    # a little missingness (never in the decision, outcome or keys)
    for col, frac in (("bmi", 0.02), ("ldl", 0.04), ("ggt", 0.03), ("exercise_freq", 0.03)):
        m = rng.random(n) < frac
        df.loc[m, col] = np.nan

    # shuffle row order (dedup must not rely on sorted input)
    df = df.iloc[rng.permutation(n)].reset_index(drop=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    df = build(a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[make_fixture] SYNTHETIC fixture: {len(df)} rows, {df.subject_id.nunique()} subjects, "
          f"{df.centre_id.nunique()} centres -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
