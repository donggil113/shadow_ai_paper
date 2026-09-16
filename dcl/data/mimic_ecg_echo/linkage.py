"""ECG -> echo linkage, coverage restriction, de-duplication, decision-maker.

Implements the "Unit of analysis and decision" and "Decision-maker" sections of
docs/mimic_ecg_echo_spec.md on plain DataFrames (as returned by
:meth:`dcl.data.mimic_ecg_echo.io.DataRoot.load_table`).  Nothing here touches
the file system and nothing here knows about simulators.

Conventions
-----------
* ``records`` is the MIMIC-IV-ECG ``record_list`` (one row per ECG) with
  ``subject_id``, ``study_id``, ``ecg_time`` (datetime64).
* ``echo_study_list`` has ``subject_id``, ``study_id``, ``study_datetime``.
* All functions return a **new** DataFrame (inputs are never mutated) and keep
  the input row order unless stated otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "link_ecg_to_echo",
    "compute_echo_coverage_window",
    "restrict_to_coverage",
    "dedup_ecgs",
    "attach_decision_maker",
    "Z_UNKNOWN",
]

Z_UNKNOWN = "unknown"
_DAY = pd.Timedelta(days=1)


def _as_datetime(s: pd.Series) -> pd.Series:
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")


def _check(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what} is missing columns {missing}; has {list(df.columns)}")


# ----------------------------------------------------------------- linkage
def link_ecg_to_echo(record_list: pd.DataFrame, echo_study_list: pd.DataFrame,
                     window_days: float = 365) -> pd.DataFrame:
    """Attach the decision ``T`` to every ECG.

    ``T = 1`` iff the same ``subject_id`` has at least one echo study with
    ``|study_datetime - ecg_time| <= window_days`` (inclusive: a gap of exactly
    ``window_days`` counts; EchoNext protocol).

    Added columns
    -------------
    T : int (0/1)
    nearest_echo_study_id : Int64 (NA when the subject has no echo at all)
    days_to_nearest_echo : float, signed ``(study_datetime - ecg_time)`` in days
        of the nearest echo (NaN when none).  Ties are broken towards the
        earlier echo study_id.
    """
    _check(record_list, ("subject_id", "study_id", "ecg_time"), "record_list")
    _check(echo_study_list, ("subject_id", "study_id", "study_datetime"), "echo_study_list")
    if record_list["study_id"].duplicated().any():
        raise ValueError("record_list.study_id must be unique (one row per ECG)")
    out = record_list.copy()
    out["ecg_time"] = _as_datetime(out["ecg_time"])

    echo = echo_study_list[["subject_id", "study_id", "study_datetime"]].rename(
        columns={"study_id": "nearest_echo_study_id"}).copy()
    echo["study_datetime"] = _as_datetime(echo["study_datetime"])
    echo = echo.dropna(subset=["study_datetime"])

    key = out[["study_id", "subject_id", "ecg_time"]].reset_index(drop=True)
    key["_row"] = np.arange(len(key))
    m = key.merge(echo, on="subject_id", how="inner")
    m["_delta"] = (m["study_datetime"] - m["ecg_time"]) / _DAY
    m["_abs"] = m["_delta"].abs()
    m = m.sort_values(["_row", "_abs", "nearest_echo_study_id"], kind="mergesort")
    best = m.drop_duplicates("_row", keep="first").set_index("_row")

    days = pd.Series(np.nan, index=key["_row"], dtype=float)
    days.loc[best.index] = best["_delta"].to_numpy()
    nearest = pd.Series(pd.array([pd.NA] * len(key), dtype="Int64"), index=key["_row"])
    nearest.loc[best.index] = best["nearest_echo_study_id"].astype("Int64").to_numpy()

    out["T"] = (np.abs(days.to_numpy()) <= float(window_days)).astype(int)
    out.loc[np.isnan(days.to_numpy()), "T"] = 0
    out["nearest_echo_study_id"] = nearest.to_numpy()
    out["days_to_nearest_echo"] = days.to_numpy()
    return out


# --------------------------------------------------------------- coverage
def compute_echo_coverage_window(echo_study_list: pd.DataFrame
                                 ) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """``(min, max)`` of ``study_datetime`` -- the period in which an echo could
    have been recorded at all.  Raises if there is no valid datetime."""
    _check(echo_study_list, ("study_datetime",), "echo_study_list")
    t = _as_datetime(echo_study_list["study_datetime"]).dropna()
    if t.empty:
        raise ValueError("echo_study_list has no valid study_datetime")
    return pd.Timestamp(t.min()), pd.Timestamp(t.max())


def restrict_to_coverage(records: pd.DataFrame,
                         window: Tuple[pd.Timestamp, pd.Timestamp],
                         pad_days: float = 365,
                         ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Keep ECGs with ``ecg_time in [min - pad, max + pad]`` (inclusive).

    Outside that band ``T = 0`` is guaranteed by data availability rather than by
    a clinical decision and would contaminate e(x) (spec).  Returns
    ``(records, provenance)`` with before/after counts.
    """
    _check(records, ("ecg_time",), "records")
    lo, hi = (pd.Timestamp(window[0]), pd.Timestamp(window[1]))
    pad = pd.Timedelta(days=float(pad_days))
    t = _as_datetime(records["ecg_time"])
    keep = (t >= lo - pad) & (t <= hi + pad)
    out = records.loc[keep].copy()
    prov = {
        "rule": "ecg_time in [echo_min - pad, echo_max + pad] (inclusive)",
        "echo_coverage_window": [lo.isoformat(), hi.isoformat()],
        "pad_days": float(pad_days),
        "ecg_window": [(lo - pad).isoformat(), (hi + pad).isoformat()],
        "n_before": int(len(records)),
        "n_after": int(len(out)),
        "n_dropped": int(len(records) - len(out)),
        "n_subjects_before": int(records["subject_id"].nunique()) if "subject_id" in records else None,
        "n_subjects_after": int(out["subject_id"].nunique()) if "subject_id" in out else None,
    }
    return out, prov


# ------------------------------------------------------------------ dedup
def dedup_ecgs(records: pd.DataFrame, min_gap_days: float = 30
               ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """At most one ECG per subject per ``min_gap_days``; keep the first.

    Rule (greedy, per subject, in time order): the subject's first ECG is kept;
    every subsequent ECG is kept iff it is at least ``min_gap_days`` after the
    *last kept* ECG.  Consequently any two kept ECGs of one subject are
    >= ``min_gap_days`` apart.  Ties in ``ecg_time`` are broken by ``study_id``.
    Output keeps the input row order.  Returns ``(records, provenance)``.
    """
    _check(records, ("subject_id", "study_id", "ecg_time"), "records")
    t = _as_datetime(records["ecg_time"])
    order = np.lexsort((records["study_id"].to_numpy(), t.to_numpy().astype("datetime64[ns]"),
                        records["subject_id"].to_numpy()))
    subj = records["subject_id"].to_numpy()[order]
    tt = t.to_numpy().astype("datetime64[ns]")[order].astype("int64")
    gap = int(round(float(min_gap_days) * 86_400 * 1_000_000_000))
    keep_sorted = np.zeros(len(order), dtype=bool)
    last_subj = None
    last_t = 0
    for i in range(len(order)):
        if subj[i] != last_subj or tt[i] - last_t >= gap:
            keep_sorted[i] = True
            last_subj, last_t = subj[i], tt[i]
    keep = np.zeros(len(order), dtype=bool)
    keep[order] = keep_sorted
    out = records.loc[keep].copy()
    prov = {
        "rule": (f"greedy per subject in time order: keep the first ECG, then keep an ECG "
                 f"iff >= {float(min_gap_days)} days after the last kept ECG"),
        "min_gap_days": float(min_gap_days),
        "n_before": int(len(records)),
        "n_after": int(len(out)),
        "n_dropped": int(len(records) - len(out)),
        "n_subjects": int(records["subject_id"].nunique()),
    }
    return out, prov


# --------------------------------------------------------- decision maker
def _asof_backward(left: pd.DataFrame, right: pd.DataFrame, right_on: str,
                   cols) -> pd.DataFrame:
    """For every ECG (``left``, columns subject_id/ecg_time/_row) find the row of
    ``right`` of the same subject with the latest ``right_on <= ecg_time``."""
    r = right.dropna(subset=[right_on]).sort_values(right_on, kind="mergesort")
    lft = left.sort_values("ecg_time", kind="mergesort")
    r = r[["subject_id", right_on] + [c for c in cols if c != right_on]]
    r["subject_id"] = r["subject_id"].astype("int64")
    lft = lft.copy()
    lft["subject_id"] = lft["subject_id"].astype("int64")
    m = pd.merge_asof(lft, r, left_on="ecg_time", right_on=right_on,
                      by="subject_id", direction="backward", allow_exact_matches=True)
    return m.set_index("_row")


def attach_decision_maker(records: pd.DataFrame, transfers: pd.DataFrame,
                          services: Optional[pd.DataFrame] = None,
                          admissions: Optional[pd.DataFrame] = None,
                          ) -> pd.DataFrame:
    """Attach ``Z`` (ordering unit) and ``Z_source`` to every ECG.

    Fallback order (first that applies):

    1. ``careunit`` of the ``transfers`` row active at ``ecg_time``
       (``intime <= ecg_time < outtime``; open ``outtime`` counts as active;
       rows without a careunit, e.g. ``eventtype == "discharge"``, never match).
       With several active rows the latest ``intime`` wins.
    2. ``curr_service`` of the latest ``services`` row with
       ``transfertime <= ecg_time``; when ``admissions`` is given the ECG must
       also fall on or before that admission's ``dischtime``.
    3. ``admission_location`` of the admission containing ``ecg_time``
       (``admittime <= ecg_time <= dischtime``).
    4. ``"unknown"``.
    """
    _check(records, ("subject_id", "ecg_time"), "records")
    _check(transfers, ("subject_id", "careunit", "intime", "outtime"), "transfers")
    out = records.copy()
    n = len(out)
    left = pd.DataFrame({
        "subject_id": out["subject_id"].to_numpy(),
        "ecg_time": _as_datetime(out["ecg_time"]).to_numpy(),
        "_row": np.arange(n),
    })
    Z = np.full(n, None, dtype=object)
    src = np.full(n, Z_UNKNOWN, dtype=object)
    valid = ~pd.isna(left["ecg_time"]).to_numpy()

    # 1. careunit active at ecg_time
    tr = transfers.copy()
    tr["intime"] = _as_datetime(tr["intime"])
    tr["outtime"] = _as_datetime(tr["outtime"])
    tr = tr[tr["careunit"].notna() & (tr["careunit"].astype(str).str.strip() != "")]
    if len(tr) and valid.any():
        m = _asof_backward(left[valid], tr, "intime", ["careunit", "outtime"])
        active = m["careunit"].notna() & (m["outtime"].isna() | (m["ecg_time"] < m["outtime"]))
        idx = m.index[active.to_numpy()]
        Z[idx] = m.loc[idx, "careunit"].astype(str).to_numpy()
        src[idx] = "careunit"

    adm = None
    if admissions is not None:
        _check(admissions, ("subject_id", "hadm_id", "admittime", "dischtime",
                            "admission_location"), "admissions")
        adm = admissions.copy()
        adm["admittime"] = _as_datetime(adm["admittime"])
        adm["dischtime"] = _as_datetime(adm["dischtime"])

    # 2. curr_service
    todo = valid & (Z == None)  # noqa: E711  (elementwise on object array)
    if services is not None and todo.any():
        _check(services, ("subject_id", "hadm_id", "transfertime", "curr_service"), "services")
        sv = services.copy()
        sv["transfertime"] = _as_datetime(sv["transfertime"])
        sv = sv[sv["curr_service"].notna()]
        if adm is not None:
            sv = sv.merge(adm[["hadm_id", "dischtime"]].drop_duplicates("hadm_id"),
                          on="hadm_id", how="left")
        else:
            sv["dischtime"] = pd.NaT
        if len(sv):
            m = _asof_backward(left[todo], sv, "transfertime", ["curr_service", "dischtime"])
            ok = m["curr_service"].notna() & (m["dischtime"].isna() | (m["ecg_time"] <= m["dischtime"]))
            idx = m.index[ok.to_numpy()]
            Z[idx] = m.loc[idx, "curr_service"].astype(str).to_numpy()
            src[idx] = "curr_service"

    # 3. admission_location
    todo = valid & (Z == None)  # noqa: E711
    if adm is not None and todo.any():
        a = adm[adm["admission_location"].notna()]
        if len(a):
            m = _asof_backward(left[todo], a, "admittime", ["admission_location", "dischtime"])
            ok = m["admission_location"].notna() & (m["dischtime"].isna() | (m["ecg_time"] <= m["dischtime"]))
            idx = m.index[ok.to_numpy()]
            Z[idx] = m.loc[idx, "admission_location"].astype(str).to_numpy()
            src[idx] = "admission_location"

    # 4. unknown
    Z[Z == None] = Z_UNKNOWN  # noqa: E711
    out["Z"] = Z.astype(str)
    out["Z_source"] = src.astype(str)
    return out
