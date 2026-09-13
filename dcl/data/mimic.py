"""MIMIC-IV: diagnostic testing as the censoring decision.

Clinical task (``acs``).  Adults presenting to the emergency department with a
cardiac chief complaint.  The clinician decides whether to order a cardiac
troponin assay; the acute-coronary-syndrome label exists only for the patients
who were tested.  Untested patients are not a random sample -- the clinician
integrates gestalt, history and examination findings that never reach the
structured record.  That gestalt is exactly the ``S`` of the DCSM.

    T = 1  <=>  cardiac troponin ordered within ``window_hours`` of ED arrival
    Y = 1  <=>  ACS at that encounter (ICD-10 I20.0, I21.*, I22.*)
    X      =  age, sex, triage vitals and acuity, chief-complaint flags

Second task (``lvef``) uses MIMIC-IV-ECG: ``T`` = transthoracic echocardiogram
ordered after an abnormal ECG, ``Y`` = reduced ejection fraction.

Data access.  MIMIC-IV is a **credentialed** PhysioNet resource: it requires a
CITI training certificate and a signed data use agreement, and it may not be
redistributed.  ``physionet.org`` is also unreachable from the sandbox this code
was developed in (blocked by the egress policy), so **the MIMIC numbers in the
paper come from :func:`make_mimic_sim`, a simulator, unless you place the real
files and pass ``source="real"``.**  The loader below implements the full
cohort-building pipeline against the real file layout; running it is a matter of
downloading the archives and pointing ``data_dir`` at them.

Required files under ``data_dir``::

    mimic-iv-ed/2.2/ed/{edstays,triage,diagnosis}.csv.gz
    mimic-iv/3.1/hosp/{labevents,d_labitems,diagnoses_icd,patients}.csv.gz

Download (after credentialing) with, e.g.::

    wget -r -N -c -np --user <user> --ask-password \\
        https://physionet.org/files/mimic-iv-ed/2.2/
    wget -r -N -c -np --user <user> --ask-password \\
        https://physionet.org/files/mimic-iv/3.1/
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..sensitivity import expit, logit
from .base import SelectiveLabelsDataset
from .semisynthetic import censor_with_hidden_signal
from .synthetic import _gauss_hermite

__all__ = ["make_mimic", "make_mimic_sim", "MIMIC_FILES"]

MIMIC_FILES = {
    "edstays": "mimic-iv-ed/2.2/ed/edstays.csv.gz",
    "triage": "mimic-iv-ed/2.2/ed/triage.csv.gz",
    "ed_diagnosis": "mimic-iv-ed/2.2/ed/diagnosis.csv.gz",
    "labevents": "mimic-iv/3.1/hosp/labevents.csv.gz",
    "d_labitems": "mimic-iv/3.1/hosp/d_labitems.csv.gz",
    "diagnoses_icd": "mimic-iv/3.1/hosp/diagnoses_icd.csv.gz",
    "patients": "mimic-iv/3.1/hosp/patients.csv.gz",
}

_TROPONIN_PATTERNS = ("troponin",)
_ACS_ICD10 = ("I21", "I22", "I200")
_CARDIAC_COMPLAINTS = ("chest pain", "chest pressure", "cp", "sob",
                       "shortness of breath", "dyspnea", "palpitation",
                       "cardiac", "arm pain", "epigastric")


def _missing(data_dir: str) -> list[str]:
    return [p for p in MIMIC_FILES.values() if not os.path.exists(os.path.join(data_dir, p))]


def make_mimic(
    data_dir: str = "data/raw",
    task: str = "acs",
    window_hours: float = 6.0,
    min_age: int = 18,
) -> SelectiveLabelsDataset:
    """Build the real MIMIC-IV cohort.  Raises with instructions if files are absent."""
    if task != "acs":
        raise NotImplementedError("only task='acs' is implemented for the real loader")
    miss = _missing(data_dir)
    if miss:
        raise FileNotFoundError(
            "MIMIC-IV is a credentialed PhysioNet resource and is not bundled.\n"
            f"Missing under {data_dir!r}:\n  " + "\n  ".join(miss) +
            "\n\nObtain credentials at https://physionet.org/settings/credentialing/ "
            "then download mimic-iv-ed/2.2 and mimic-iv/3.1.\n"
            "For a runnable stand-in use dcl.data.mimic.make_mimic_sim()."
        )

    j = lambda k: os.path.join(data_dir, MIMIC_FILES[k])
    ed = pd.read_csv(j("edstays"), parse_dates=["intime", "outtime"])
    tri = pd.read_csv(j("triage"))
    pat = pd.read_csv(j("patients"))
    dlab = pd.read_csv(j("d_labitems"))

    trop_ids = dlab.loc[
        dlab["label"].str.lower().str.contains("|".join(_TROPONIN_PATTERNS), na=False),
        "itemid"].unique()

    lab = pd.read_csv(j("labevents"), usecols=["subject_id", "itemid", "charttime"],
                      parse_dates=["charttime"])
    lab = lab[lab["itemid"].isin(trop_ids)]

    dx = pd.read_csv(j("diagnoses_icd"), usecols=["subject_id", "hadm_id",
                                                  "icd_code", "icd_version"])
    dx = dx[dx["icd_version"] == 10]
    dx["acs"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False) \
                              .str.upper().str.startswith(_ACS_ICD10)
    acs = dx.groupby("hadm_id")["acs"].max()

    coh = ed.merge(tri, on="stay_id", how="left").merge(
        pat[["subject_id", "anchor_age", "gender"]], on="subject_id", how="left")
    cc = coh["chiefcomplaint"].astype(str).str.lower()
    coh = coh[cc.str.contains("|".join(_CARDIAC_COMPLAINTS), na=False)]
    coh = coh[coh["anchor_age"] >= min_age]

    # T: troponin charted within the window after ED arrival
    lab_m = coh[["stay_id", "subject_id", "intime"]].merge(lab, on="subject_id",
                                                           how="left")
    dt = (lab_m["charttime"] - lab_m["intime"]).dt.total_seconds() / 3600.0
    ordered = lab_m.assign(ok=((dt >= -0.5) & (dt <= window_hours))) \
                   .groupby("stay_id")["ok"].max()
    coh["T"] = coh["stay_id"].map(ordered).fillna(False).astype(float)
    coh["Y"] = coh["hadm_id"].map(acs).fillna(False).astype(float)

    vit = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
           "pain", "acuity"]
    for v in vit:
        coh[v] = pd.to_numeric(coh[v], errors="coerce")
    feats = coh[["anchor_age"] + vit].copy()
    feats["female"] = (coh["gender"].astype(str).str.upper() == "F").astype(float)
    cc = coh["chiefcomplaint"].astype(str).str.lower()
    for kw in ("chest", "dyspnea", "sob", "palpitation", "arm", "epigastric"):
        feats[f"cc_{kw}"] = cc.str.contains(kw, na=False).astype(float)
    feats = feats.fillna(feats.median(numeric_only=True))

    X = feats.to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))
    T = coh["T"].to_numpy()
    Y_obs = np.where(T == 1, coh["Y"].to_numpy(), np.nan)

    return SelectiveLabelsDataset(
        X=X, T=T, Y_obs=Y_obs, Y_full=None, Z=None,
        feature_names=list(feats.columns), name="MIMIC-IV-ED/ACS(real)",
        notes=("Real MIMIC-IV-ED cohort; troponin ordering is the censoring "
               "decision. No ground truth for untested patients -- bounds only."),
    )


def make_mimic_sim(
    n: int = 12000,
    target_gamma: float = 2.5,
    selection_rate: float = 0.55,
    prevalence: float = 0.09,
    seed: int = 0,
    n_judges: int = 6,
) -> SelectiveLabelsDataset:
    """Simulator standing in for the credentialed MIMIC-IV ED cohort.

    **This is simulated data, not MIMIC.**  Marginal characteristics (age
    distribution, vital-sign means and dispersions, troponin-ordering rate,
    ACS prevalence among tested patients, clinician-to-clinician variation in
    ordering) are set to values typical of published MIMIC-IV-ED chest-pain
    cohorts, so the pipeline exercises realistic scales; nothing here is a
    substitute for running :func:`make_mimic` on the real files.

    ``S`` -- clinician gestalt -- is a latent variable that drives both testing
    and the true diagnosis, and is deliberately *not* in ``X``.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(58, 18, n), 18, 95)
    female = (rng.uniform(size=n) < 0.47).astype(float)
    hr = np.clip(rng.normal(84, 18, n), 35, 190)
    sbp = np.clip(rng.normal(134, 23, n), 60, 240)
    dbp = np.clip(sbp * 0.58 + rng.normal(0, 8, n), 30, 140)
    rr = np.clip(rng.normal(18, 4, n), 6, 45)
    o2 = np.clip(rng.normal(97, 2.4, n), 70, 100)
    temp = np.clip(rng.normal(98.2, 0.9, n), 93, 105)
    pain = np.clip(rng.integers(0, 11, n) + rng.normal(0, 1, n), 0, 10)
    acuity = np.clip(rng.normal(2.6, 0.8, n).round(), 1, 5)
    cc_chest = (rng.uniform(size=n) < 0.62).astype(float)
    cc_sob = (rng.uniform(size=n) < 0.28).astype(float)
    cc_palp = (rng.uniform(size=n) < 0.12).astype(float)

    feature_names = ["age", "female", "heartrate", "sbp", "dbp", "resprate",
                     "o2sat", "temperature", "pain", "acuity",
                     "cc_chest", "cc_sob", "cc_palpitation"]
    Xraw = np.column_stack([age, female, hr, sbp, dbp, rr, o2, temp, pain,
                            acuity, cc_chest, cc_sob, cc_palp])
    X = (Xraw - Xraw.mean(0)) / np.where(Xraw.std(0) < 1e-9, 1.0, Xraw.std(0))

    # clinician gestalt: correlated with risk factors but far from determined by them
    S_raw = (0.55 * X[:, 0] + 0.35 * X[:, 2] - 0.30 * X[:, 6] + 0.40 * X[:, 8]
             + 0.45 * X[:, 10] + rng.normal(0, 1.0, n))

    # true ACS status generated from X and gestalt
    lin = (0.9 * X[:, 0] + 0.35 * X[:, 2] - 0.35 * X[:, 6] + 0.30 * X[:, 8]
           + 0.55 * X[:, 10] + 0.25 * X[:, 11] - 0.20 * female + 0.8 * S_raw)
    # solve the intercept so that E[P(Y=1)] matches the target prevalence
    lo_b, hi_b = -20.0, 20.0
    for _ in range(80):
        b0 = 0.5 * (lo_b + hi_b)
        if expit(lin + b0).mean() < prevalence:
            lo_b = b0
        else:
            hi_b = b0
    b0 = 0.5 * (lo_b + hi_b)
    Y = (rng.uniform(size=n) < expit(lin + b0)).astype(float)

    ds = censor_with_hidden_signal(
        X=X, S_raw=S_raw, Y=Y, feature_names=feature_names,
        name=f"MIMIC-sim/ACS(Gamma0~{target_gamma:g})",
        target_gamma=target_gamma, selection_rate=selection_rate,
        n_judges=n_judges, seed=seed, kappa_heterogeneity=0.3,
        notes=("SIMULATED ED cohort standing in for credentialed MIMIC-IV-ED. "
               "Not real patient data. S = clinician gestalt, excluded from X."),
    )
    return ds
