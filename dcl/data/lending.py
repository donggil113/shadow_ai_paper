"""Lending Club: real credit applications with the underwriter's signal held out.

Source.  ``modeldata::lending_club`` (n = 9,857, 5.2% adverse) and
``openintro::loans_full_schema`` (n = 10,000, 55 columns), both redistributions
of Lending Club's public loan book, mirrored in the Rdatasets archive.  See
``scripts/download_data.py`` for the exact URLs.

Selective-labels construction.  Lending Club's book contains only *funded*
loans, so the genuinely rejected applications are not in the file and their
repayment behaviour is unobservable -- the selective labels problem in its
purest form, and precisely why the deployment risk of a credit model cannot be
measured from it.  To obtain a benchmark with ground truth we use the
semi-synthetic protocol of :mod:`dcl.data.semisynthetic`:

* ``X`` = pre-decision applicant characteristics only (income, employment,
  delinquency history, utilisation, inquiry counts, ...);
* ``S`` = the **assigned interest rate / sub-grade**, i.e. the underwriter's own
  risk assessment.  This is a real variable that genuinely predicts default and
  is genuinely unavailable to a model built from application data, so the
  confounding it induces is real rather than invented.  It is excluded from
  ``X``;
* ``Y`` = the real repayment outcome (``Class == "bad"``), available for every
  unit, which is what makes exact evaluation of the deployment risk possible;
* ``T`` = a funding decision drawn from a policy in ``(X, S)`` calibrated to a
  target ``Gamma_0``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import SelectiveLabelsDataset
from .semisynthetic import censor_with_hidden_signal

__all__ = ["load_lending_club_frame", "make_lending_club"]

_NUM = ["funded_amnt", "annual_inc", "delinq_2yrs", "inq_last_6mths",
        "revol_util", "acc_now_delinq", "open_il_6m", "open_il_12m",
        "open_il_24m", "total_bal_il", "all_util", "inq_fi", "inq_last_12m",
        "delinq_amnt", "num_il_tl", "total_il_high_credit_limit"]

_EMP_ORDER = {"emp_lt_1": 0.0, "emp_1": 1.0, "emp_2": 2.0, "emp_3": 3.0,
              "emp_4": 4.0, "emp_5": 5.0, "emp_6": 6.0, "emp_7": 7.0,
              "emp_8": 8.0, "emp_9": 9.0, "emp_ge_10": 10.0, "emp_unk": np.nan}


def load_lending_club_frame(data_dir: str = "data/raw") -> pd.DataFrame:
    path = os.path.join(data_dir, "modeldata_lending_club.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def make_lending_club(
    data_dir: str = "data/raw",
    target_gamma: float = 2.0,
    selection_rate: float = 0.6,
    seed: int = 0,
    n_judges: int = 5,
    **kwargs,
) -> SelectiveLabelsDataset:
    df = load_lending_club_frame(data_dir)

    y = (df["Class"].astype(str).str.lower() == "bad").astype(float).to_numpy()

    num = df[_NUM].apply(pd.to_numeric, errors="coerce").astype(float).copy()
    # `term` and `emp_length` ship as labelled strings in this redistribution
    num["term_months"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
    emp = df["emp_length"].astype(str).map(_EMP_ORDER)
    num["emp_length_years"] = emp
    num["emp_length_unknown"] = emp.isna().astype(float)
    # heavy-tailed money columns -> log1p
    for c in ["annual_inc", "total_bal_il", "total_il_high_credit_limit",
              "funded_amnt", "delinq_amnt"]:
        num[c] = np.log1p(np.clip(num[c], 0, None))
    ver = pd.get_dummies(df["verification_status"].astype(str),
                         prefix="verif", drop_first=True).astype(float)
    Xdf = pd.concat([num, ver], axis=1)
    Xdf = Xdf.fillna(Xdf.median(numeric_only=True))

    X = Xdf.to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))

    # The underwriter's private risk assessment: assigned rate + letter sub-grade.
    sub = df["sub_grade"].astype(str)
    grade_ord = sub.str[0].map({g: i for i, g in enumerate("ABCDEFG")}).astype(float)
    sub_num = pd.to_numeric(sub.str[1:], errors="coerce").fillna(3.0)
    S_raw = (df["int_rate"].astype(float).to_numpy()
             + 0.5 * (grade_ord.to_numpy() * 5 + sub_num.to_numpy()))

    return censor_with_hidden_signal(
        X=X, S_raw=S_raw, Y=y, feature_names=list(Xdf.columns),
        name=f"LendingClub(Gamma0~{target_gamma:g})",
        target_gamma=target_gamma, selection_rate=selection_rate,
        n_judges=n_judges, seed=seed,
        notes=("Real Lending Club applicant features and repayment outcomes; "
               "hidden signal S = assigned interest rate / sub-grade (the "
               "underwriter's own risk assessment, excluded from X); funding "
               "decision simulated with a Gamma_0-calibrated policy."),
        **kwargs,
    )
