"""AER CreditCard: a fully real accept/reject decision with a genuinely
unobservable outcome.

Source.  ``AER::CreditCard`` (Greene), n = 1,319 credit card applications;
``card`` records whether the application was **accepted** (77.6%).  Post-decision
behaviour (``expenditure``, ``share``) exists only for accepted applicants -- for
rejected ones there is no card and therefore no usage, so the outcome is
*structurally* censored by the decision.  There is no ground truth for the
rejected group and there never can be.

That is exactly the point: this dataset is the setting where our bounds are the
*only* thing one can report.  We use it to demonstrate the end-to-end pipeline
on fully real decisions, not to validate coverage (which is impossible here).

    T = 1  <=>  application accepted
    Y = 1  <=>  heavy utilisation (``share`` above the median among card holders)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import SelectiveLabelsDataset

__all__ = ["make_creditcard"]


def make_creditcard(data_dir: str = "data/raw") -> SelectiveLabelsDataset:
    path = os.path.join(data_dir, "AER_CreditCard.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path)

    T = (df["card"].astype(str).str.lower() == "yes").astype(float).to_numpy()
    share = df["share"].astype(float).to_numpy()
    thr = float(np.median(share[T == 1]))
    Y_obs = np.where(T == 1, (share > thr).astype(float), np.nan)

    num = df[["reports", "age", "income", "dependents", "months",
              "majorcards", "active"]].astype(float)
    cat = pd.get_dummies(df[["owner", "selfemp"]].astype(str),
                         drop_first=True).astype(float)
    Xdf = pd.concat([num, cat], axis=1).fillna(0.0)
    X = Xdf.to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))

    return SelectiveLabelsDataset(
        X=X, T=T, Y_obs=Y_obs, Y_full=None, Z=None,
        feature_names=list(Xdf.columns), name="AER-CreditCard(real accept/reject)",
        notes=("Fully real application decisions; the outcome is structurally "
               "unobservable for rejected applicants, so only bounds are "
               "reportable -- no ground truth exists or can exist."),
    )
