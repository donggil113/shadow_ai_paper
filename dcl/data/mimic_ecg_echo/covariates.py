"""Pre-decision covariates X for the MIMIC-IV-ECG x MIMIC-IV-ECHO arm (T1).

Implements the "Covariates X" section of docs/mimic_ecg_echo_spec.md on plain
DataFrames as returned by :meth:`dcl.data.mimic_ecg_echo.io.DataRoot.load_table`.
Nothing here reads the file system and nothing here knows about simulators
(``dcl.data.mimic.make_mimic_sim`` is never imported).

Every feature function takes ``records`` (one row per ECG: ``subject_id``,
``study_id``, ``ecg_time``) and returns a **new** DataFrame of float features
aligned to ``records.index`` in the same row order.  Feature frames contain
*only* features -- never ``subject_id``/``study_id`` -- so that they can be
concatenated straight into the design matrix by :func:`assemble_X`.

Nothing downstream of the decision may enter X.  :data:`FORBIDDEN` lists the
column-name patterns that are refused (echo-derived quantities, any outcome
``Y``, the decision ``T``, ``study_id``, the decision-maker ``Z`` and its
sources, discharge-time fields); :func:`assemble_X` asserts against it and
:func:`assert_no_forbidden` can be called on any column list.

Column conventions of the produced features
-------------------------------------------
``age_at_ecg``, ``sex_male``                       (demographics)
``lab_<analyte>``, ``lab_<analyte>_missing``      (labs within ``hours`` before the ECG)
``setting_ed``, ``setting_inpatient``, ``setting_outpatient``  (one-hot)
``ecg_rr``, ``ecg_pr``, ``ecg_qrs``, ``ecg_qt``, ``ecg_qtc``,
``ecg_p_axis``, ``ecg_qrs_axis``, ``ecg_t_axis``   (machine measurements)
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

__all__ = [
    "FORBIDDEN",
    "ForbiddenFeatureError",
    "DEFAULT_ANALYTES",
    "ANALYTE_PATTERNS",
    "PLAUSIBLE_RANGES",
    "is_forbidden",
    "assert_no_forbidden",
    "demographics",
    "resolve_lab_itemids",
    "labs_before_ecg",
    "setting",
    "tabular_ecg_features",
    "assemble_X",
    "covariate_blocks",
    "describe_blocks",
]

# --------------------------------------------------------------- forbidden
#: Case-insensitive regular expressions.  A column whose name matches any of
#: them is refused by :func:`assemble_X`.  Kept deliberately broad: a false
#: positive costs a rename, a false negative leaks the decision into X.
FORBIDDEN: Tuple[str, ...] = (
    # the decision, the outcome(s), the decision maker -- exact or as a prefix
    r"^(y|t|z)(_|$)",
    r"^(y|t|z)_(obs|hat|pred|true|note|icd|echo|struct|source)",
    r"^(outcome|label|target|decision|treat|treated|selected|censor)",
    r"^(z_source|careunit|curr_service|prev_service|admission_location)$",
    # identifiers of the unit / of downstream records
    r"^(study_id|hadm_id|stay_id|note_id|transfer_id|specimen_id)$",
    # anything echo-derived
    r"echo", r"lvef", r"ejection", r"ivsd", r"lvpwd", r"wall_thick", r"valve",
    r"\bshd\b", r"^shd", r"_shd", r"dicom", r"study_datetime",
    r"nearest_echo", r"days_to",
    # ICD / note derived outcomes
    r"icd", r"discharge", r"note_",
    # post-ECG time fields that encode the future
    r"^(dod|deathtime|dischtime|outtime)$",
)
_FORBIDDEN_RE = tuple(re.compile(p, re.IGNORECASE) for p in FORBIDDEN)


class ForbiddenFeatureError(AssertionError):
    """A covariate column is (or may be) downstream of the decision."""


def is_forbidden(name: str) -> Optional[str]:
    """The first pattern of :data:`FORBIDDEN` that ``name`` matches, else None."""
    s = str(name)
    for pat, rx in zip(FORBIDDEN, _FORBIDDEN_RE):
        if rx.search(s):
            return pat
    return None


def assert_no_forbidden(names: Iterable[str]) -> None:
    """Raise :class:`ForbiddenFeatureError` if any name matches :data:`FORBIDDEN`."""
    bad = [(str(n), is_forbidden(n)) for n in names]
    bad = [(n, p) for n, p in bad if p is not None]
    if bad:
        raise ForbiddenFeatureError(
            "covariate columns downstream of the decision are forbidden in X: "
            + ", ".join(f"{n!r} (matches {p!r})" for n, p in bad))


# ----------------------------------------------------------------- helpers
def _check(df: pd.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what} is missing columns {missing}; has {list(df.columns)}")


def _as_datetime(s: pd.Series) -> pd.Series:
    return s if pd.api.types.is_datetime64_any_dtype(s) else pd.to_datetime(s, errors="coerce")


def _left_frame(records: pd.DataFrame) -> pd.DataFrame:
    """``subject_id`` (int64), ``ecg_time`` (datetime), ``_row`` for every ECG."""
    _check(records, ("subject_id", "ecg_time"), "records")
    return pd.DataFrame({
        "subject_id": pd.to_numeric(records["subject_id"]).astype("int64").to_numpy(),
        "ecg_time": _as_datetime(records["ecg_time"]).to_numpy(),
        "_row": np.arange(len(records)),
    })


def _asof_backward(left: pd.DataFrame, right: pd.DataFrame, right_on: str,
                   cols: Sequence[str], tolerance: Optional[pd.Timedelta] = None
                   ) -> pd.DataFrame:
    """For every ECG in ``left`` (``subject_id``/``ecg_time``/``_row``) the row of
    ``right`` of the same subject with the latest ``right_on <= ecg_time``
    (optionally within ``tolerance``).  Result is indexed by ``_row``; ECGs
    without a match carry NaN in ``cols``."""
    lft = left.dropna(subset=["ecg_time"]).sort_values("ecg_time", kind="mergesort")
    r = right.dropna(subset=[right_on]).sort_values(right_on, kind="mergesort")
    r = r[["subject_id", right_on] + [c for c in cols if c != right_on]].copy()
    r["subject_id"] = pd.to_numeric(r["subject_id"]).astype("int64")
    if lft.empty or r.empty:
        return pd.DataFrame(index=pd.Index(left["_row"], name="_row"),
                            columns=list(cols), dtype=object)
    m = pd.merge_asof(lft, r, left_on="ecg_time", right_on=right_on, by="subject_id",
                      direction="backward", allow_exact_matches=True, tolerance=tolerance)
    return m.set_index("_row").reindex(left["_row"])


# ------------------------------------------------------------ demographics
def demographics(records: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Age at the ECG and sex.

    ``age_at_ecg = anchor_age + (year(ecg_time) - anchor_year)`` (MIMIC-IV
    convention: ``anchor_age`` is the age in ``anchor_year``; ages above 89 are
    stored as 91 by the de-identification and stay so).  ``sex_male`` is 1.0 for
    ``gender == "M"``, 0.0 for ``"F"``, NaN otherwise.  Subjects absent from
    ``patients`` get NaN in both columns (imputed later, never dropped here).
    """
    _check(records, ("subject_id", "ecg_time"), "records")
    _check(patients, ("subject_id", "gender", "anchor_age", "anchor_year"), "patients")
    pt = patients.drop_duplicates("subject_id").set_index("subject_id")
    sid = records["subject_id"].to_numpy()
    anchor_age = pd.to_numeric(pt["anchor_age"], errors="coerce").reindex(sid).to_numpy(float)
    anchor_year = pd.to_numeric(pt["anchor_year"], errors="coerce").reindex(sid).to_numpy(float)
    year = _as_datetime(records["ecg_time"]).dt.year.to_numpy(float)
    gender = pt["gender"].astype("string").str.upper().str.strip().reindex(sid)
    sex = np.where(gender == "M", 1.0, np.where(gender == "F", 0.0, np.nan))
    return pd.DataFrame({
        "age_at_ecg": anchor_age + (year - anchor_year),
        "sex_male": sex.astype(float),
    }, index=records.index)


# -------------------------------------------------------------------- labs
DEFAULT_ANALYTES: Tuple[str, ...] = ("troponin", "bnp", "creatinine", "hemoglobin")

#: analyte -> (case-insensitive regex on ``d_labitems.label``, required fluid or
#: None).  Written against MIMIC-IV v3.1 ``d_labitems`` labels ("Troponin T",
#: "Troponin I", "NTproBNP", "Creatinine", "Creatinine, Whole Blood",
#: "Hemoglobin").  Anchored so that "% Hemoglobin A1c", "Carboxyhemoglobin",
#: "Creatinine, Urine" (fluid Urine) and "Creatine Kinase" are NOT matched.
ANALYTE_PATTERNS: Dict[str, Tuple[str, Optional[str]]] = {
    "troponin": (r"troponin", "blood"),
    "bnp": (r"(^|[^a-z])bnp([^a-z]|$)|ntprobnp|nt-probnp|natriuretic", "blood"),
    "creatinine": (r"^creatinine(\s*,\s*whole blood)?$", "blood"),
    "hemoglobin": (r"^hemoglobin$", "blood"),
}


def resolve_lab_itemids(d_labitems: pd.DataFrame,
                        analytes: Sequence[str] = DEFAULT_ANALYTES,
                        verbose: bool = True, strict: bool = True,
                        ) -> Dict[str, List[int]]:
    """Map every analyte to the ``itemid`` values whose ``d_labitems.label``
    matches its pattern (case-insensitive; :data:`ANALYTE_PATTERNS`).  An
    analyte without an entry is matched as a plain case-insensitive substring of
    the label.  The resolved ``(itemid, label, fluid)`` triples are printed
    when ``verbose``.  With ``strict``, an analyte that resolves to nothing
    raises ``ValueError`` (the dictionary differs from the version the pattern
    was written against: fix the pattern, do not paper over)."""
    _check(d_labitems, ("itemid", "label"), "d_labitems")
    lab = d_labitems["label"].astype("string").fillna("").str.strip().str.lower()
    fluid = (d_labitems["fluid"].astype("string").fillna("").str.strip().str.lower()
             if "fluid" in d_labitems.columns else pd.Series("", index=d_labitems.index))
    out: Dict[str, List[int]] = {}
    for a in analytes:
        pat, req_fluid = ANALYTE_PATTERNS.get(a.lower(), (re.escape(a.lower()), None))
        hit = lab.str.contains(pat, regex=True, case=False, na=False).fillna(False).to_numpy(bool)
        if req_fluid is not None and "fluid" in d_labitems.columns:
            hit &= (fluid == req_fluid).fillna(False).to_numpy(bool)
        rows = d_labitems.loc[hit]
        ids = sorted(int(i) for i in pd.to_numeric(rows["itemid"]).dropna().unique())
        out[a] = ids
        if verbose:
            desc = "; ".join(f"{int(r.itemid)}={str(r.label)!r}"
                             + (f"/{r.fluid}" if "fluid" in rows.columns else "")
                             for r in rows.itertuples())
            print(f"[covariates.labs] analyte {a!r}: pattern {pat!r}"
                  f"{' fluid=' + req_fluid if req_fluid else ''} -> itemids {ids} [{desc}]",
                  flush=True)
        if strict and not ids:
            near = lab.str.contains(a.lower(), regex=False, na=False).fillna(False).to_numpy(bool)
            candidates = sorted(set(d_labitems.loc[near, "label"].astype(str)))[:20]
            raise ValueError(
                f"analyte {a!r} (pattern {pat!r}, fluid {req_fluid!r}) resolves to no itemid in "
                f"d_labitems ({len(d_labitems)} rows); labels containing the name: {candidates}. "
                "Fix ANALYTE_PATTERNS in dcl/data/mimic_ecg_echo/covariates.py.")
    return out


def labs_before_ecg(records: pd.DataFrame, labevents: pd.DataFrame,
                    d_labitems: pd.DataFrame, hours: float = 24,
                    analytes: Sequence[str] = DEFAULT_ANALYTES,
                    verbose: bool = True, strict: bool = True) -> pd.DataFrame:
    """Last numeric value of each analyte charted in ``[ecg_time - hours, ecg_time]``.

    For every ECG and analyte the ``labevents`` row of the same subject with the
    latest ``charttime <= ecg_time`` and ``ecg_time - charttime <= hours`` is
    taken (both ends inclusive); rows charted after the ECG are never used.
    Rows with a non-numeric ``valuenum`` are ignored.  Missing => NaN in
    ``lab_<analyte>`` and 1.0 in ``lab_<analyte>_missing``.  Item ids are
    resolved by :func:`resolve_lab_itemids` (printed).
    """
    _check(labevents, ("subject_id", "itemid", "charttime", "valuenum"), "labevents")
    itemids = resolve_lab_itemids(d_labitems, analytes, verbose=verbose, strict=strict)
    left = _left_frame(records)
    lab = pd.DataFrame({
        "subject_id": pd.to_numeric(labevents["subject_id"], errors="coerce"),
        "itemid": pd.to_numeric(labevents["itemid"], errors="coerce"),
        "charttime": _as_datetime(labevents["charttime"]),
        "valuenum": pd.to_numeric(labevents["valuenum"], errors="coerce"),
    }).dropna(subset=["subject_id", "itemid", "charttime", "valuenum"])
    tol = pd.Timedelta(hours=float(hours))
    out: Dict[str, np.ndarray] = {}
    for a in analytes:
        sub = lab[lab["itemid"].isin(itemids[a])]
        m = _asof_backward(left, sub, "charttime", ["valuenum"], tolerance=tol)
        v = pd.to_numeric(m["valuenum"], errors="coerce").to_numpy(float)
        out[f"lab_{a}"] = v
        out[f"lab_{a}_missing"] = np.isnan(v).astype(float)
    return pd.DataFrame(out, index=records.index)


# ----------------------------------------------------------------- setting
def setting(records: pd.DataFrame, edstays: pd.DataFrame,
            admissions: pd.DataFrame) -> pd.DataFrame:
    """One-hot care setting at the ECG: ED / inpatient / outpatient.

    * ``setting_ed``: an ``edstays`` row of the subject with
      ``intime <= ecg_time <= outtime`` (open ``outtime`` counts as ongoing);
    * else ``setting_inpatient``: an admission with
      ``admittime <= ecg_time <= dischtime`` (open ``dischtime`` likewise);
    * else ``setting_outpatient``.

    ED wins over inpatient when both overlap (the ED stay of an admitted patient
    precedes the ward).  With several candidate stays the one with the latest
    start is checked (stays of one subject do not overlap in MIMIC-IV-ED).
    Exactly one column is 1.0 per ECG; ECGs without a valid ``ecg_time`` are
    outpatient.
    """
    _check(edstays, ("subject_id", "intime", "outtime"), "edstays")
    _check(admissions, ("subject_id", "admittime", "dischtime"), "admissions")
    left = _left_frame(records)

    ed = edstays[["subject_id", "intime", "outtime"]].copy()
    ed["intime"], ed["outtime"] = _as_datetime(ed["intime"]), _as_datetime(ed["outtime"])
    m = _asof_backward(left, ed, "intime", ["intime", "outtime"])
    in_ed = (m["intime"].notna()
             & (m["outtime"].isna() | (left["ecg_time"].to_numpy() <= m["outtime"].to_numpy()))
             ).to_numpy(bool)

    adm = admissions[["subject_id", "admittime", "dischtime"]].copy()
    adm["admittime"], adm["dischtime"] = _as_datetime(adm["admittime"]), _as_datetime(adm["dischtime"])
    m = _asof_backward(left, adm, "admittime", ["admittime", "dischtime"])
    in_hosp = (m["admittime"].notna()
               & (m["dischtime"].isna() | (left["ecg_time"].to_numpy() <= m["dischtime"].to_numpy()))
               ).to_numpy(bool)

    is_ed = in_ed
    is_inpt = in_hosp & ~in_ed
    is_out = ~(is_ed | is_inpt)
    assert (is_ed.astype(int) + is_inpt.astype(int) + is_out.astype(int) == 1).all()
    return pd.DataFrame({
        "setting_ed": is_ed.astype(float),
        "setting_inpatient": is_inpt.astype(float),
        "setting_outpatient": is_out.astype(float),
    }, index=records.index)


# ------------------------------------------------------- machine features
#: Physiologically plausible (inclusive) ranges; values outside are set to NaN.
#: MIMIC-IV-ECG machine measurements use out-of-range sentinels (e.g. 29999)
#: for unavailable intervals.  Units assumed: intervals in ms, axes in degrees
#: (MIMIC-IV-ECG v1.0 ``machine_measurements.csv``; verify on the real file).
PLAUSIBLE_RANGES: Dict[str, Tuple[float, float]] = {
    "ecg_rr": (200.0, 3000.0),
    "ecg_pr": (40.0, 500.0),
    "ecg_qrs": (20.0, 300.0),
    "ecg_qt": (100.0, 800.0),
    "ecg_qtc": (100.0, 1000.0),
    "ecg_p_axis": (-180.0, 360.0),
    "ecg_qrs_axis": (-180.0, 360.0),
    "ecg_t_axis": (-180.0, 360.0),
}
_MM_COLS = ("rr_interval", "p_onset", "qrs_onset", "qrs_end", "t_end",
            "p_axis", "qrs_axis", "t_axis")


def tabular_ecg_features(machine_measurements: pd.DataFrame,
                         records: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Scalar ECG features from ``machine_measurements`` (floats, NaN if absent).

    ``ecg_rr = rr_interval``; ``ecg_pr = qrs_onset - p_onset``;
    ``ecg_qrs = qrs_end - qrs_onset``; ``ecg_qt = t_end - qrs_onset``;
    ``ecg_qtc = ecg_qt / sqrt(ecg_rr / 1000)`` (Bazett, RR in ms);
    ``ecg_p_axis``, ``ecg_qrs_axis``, ``ecg_t_axis`` copied.  Values outside
    :data:`PLAUSIBLE_RANGES` (sentinels, non-positive intervals) become NaN.

    Without ``records`` the result is indexed by ``study_id`` (duplicates: first
    row kept, with a warning).  With ``records`` it is aligned to
    ``records.index`` through ``records["study_id"]`` (ECGs without a
    measurement row are all-NaN).  The textual ``report_*`` columns are not
    used: they are machine interpretations that may mention the echo request.
    """
    _check(machine_measurements, ("study_id",) + _MM_COLS, "machine_measurements")
    mm = machine_measurements
    if mm["study_id"].duplicated().any():
        warnings.warn("machine_measurements has duplicated study_id; keeping the first row")
        mm = mm.drop_duplicates("study_id", keep="first")
    num = {c: pd.to_numeric(mm[c], errors="coerce").to_numpy(float) for c in _MM_COLS}
    rr = num["rr_interval"]
    qt = num["t_end"] - num["qrs_onset"]
    feats = {
        "ecg_rr": rr,
        "ecg_pr": num["qrs_onset"] - num["p_onset"],
        "ecg_qrs": num["qrs_end"] - num["qrs_onset"],
        "ecg_qt": qt,
        "ecg_p_axis": num["p_axis"],
        "ecg_qrs_axis": num["qrs_axis"],
        "ecg_t_axis": num["t_axis"],
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in ("ecg_rr", "ecg_pr", "ecg_qrs", "ecg_qt", "ecg_p_axis", "ecg_qrs_axis", "ecg_t_axis"):
            lo, hi = PLAUSIBLE_RANGES[k]
            v = feats[k]
            feats[k] = np.where((v >= lo) & (v <= hi), v, np.nan)
        qtc = feats["ecg_qt"] / np.sqrt(feats["ecg_rr"] / 1000.0)
        lo, hi = PLAUSIBLE_RANGES["ecg_qtc"]
        feats["ecg_qtc"] = np.where((qtc >= lo) & (qtc <= hi), qtc, np.nan)
    cols = ["ecg_rr", "ecg_pr", "ecg_qrs", "ecg_qt", "ecg_qtc",
            "ecg_p_axis", "ecg_qrs_axis", "ecg_t_axis"]
    out = pd.DataFrame({c: feats[c] for c in cols},
                       index=pd.Index(pd.to_numeric(mm["study_id"]).astype("int64"), name="study_id"))
    if records is None:
        return out
    _check(records, ("study_id",), "records")
    aligned = out.reindex(pd.to_numeric(records["study_id"]).astype("int64").to_numpy())
    aligned.index = records.index
    return aligned


# ------------------------------------------------------------ assemble X
Blocks = Union[Mapping[str, pd.DataFrame], Sequence[pd.DataFrame], pd.DataFrame]


def assemble_X(blocks: Blocks, train_mask: Sequence[bool],
               forbidden: Sequence[str] = FORBIDDEN,
               imputer: Optional[SimpleImputer] = None,
               ) -> Tuple[np.ndarray, List[str], SimpleImputer]:
    """Concatenate feature blocks into ``X`` with median imputation fit on
    TRAIN rows only.

    Parameters
    ----------
    blocks : DataFrame, sequence or mapping of DataFrames
        Feature frames with identical indices (as produced by the functions of
        this module).  All columns must be numeric.
    train_mask : bool array of length n
        Rows the imputer may see.  Medians are computed on these rows only and
        applied to every row (train and test alike).
    forbidden : sequence of regex
        Column-name patterns that are refused (:data:`FORBIDDEN`).  The check is
        an explicit assertion (:class:`ForbiddenFeatureError`, an
        ``AssertionError``) and runs before anything else.
    imputer : fitted SimpleImputer, optional
        Reuse a previously returned imputer (e.g. to transform a later batch
        with the same train medians) instead of fitting a new one.

    Returns ``(X, feature_names, imputer)`` with ``X`` a float64 array of shape
    ``(n, p)``.  A column that is entirely NaN on the train rows is imputed with
    0.0 and a warning is issued (``keep_empty_features``), so ``feature_names``
    always matches ``X.shape[1]``.
    """
    if isinstance(blocks, pd.DataFrame):
        frames = [blocks]
    elif isinstance(blocks, Mapping):
        frames = list(blocks.values())
    else:
        frames = list(blocks)
    if not frames:
        raise ValueError("assemble_X needs at least one feature block")
    idx = frames[0].index
    for f in frames[1:]:
        if len(f) != len(idx) or not f.index.equals(idx):
            raise ValueError("feature blocks must share the same index (row alignment)")
    df = pd.concat(frames, axis=1)
    names = [str(c) for c in df.columns]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicated feature names across blocks: {dup}")

    # explicit assertion: nothing downstream of the decision
    rxs = [re.compile(p, re.IGNORECASE) for p in forbidden]
    bad = [(n, p) for n in names for p, rx in zip(forbidden, rxs) if rx.search(n)]
    if bad:  # an explicit ``raise`` so that ``python -O`` cannot strip the check
        raise ForbiddenFeatureError(
            "covariate columns downstream of the decision are forbidden in X: "
            + ", ".join(f"{n!r} (matches {p!r})" for n, p in bad))

    non_num = [n for n, c in zip(names, df.columns) if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise TypeError(f"non-numeric feature columns: {non_num}")
    V = df.to_numpy(dtype=float)
    mask = np.asarray(train_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != V.shape[0]:
        raise ValueError(f"train_mask has length {mask.shape[0]}, expected {V.shape[0]}")
    if imputer is None:
        if not mask.any():
            raise ValueError("train_mask selects no rows; cannot fit the imputer")
        empty = [n for n, j in zip(names, range(V.shape[1])) if np.isnan(V[mask, j]).all()]
        if empty:
            warnings.warn(f"features entirely missing on the train rows are imputed with 0: {empty}")
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputer.fit(V[mask])
    else:
        if getattr(imputer, "n_features_in_", V.shape[1]) != V.shape[1]:
            raise ValueError("the supplied imputer was fit on a different number of features")
    X = np.asarray(imputer.transform(V), dtype=float)
    assert X.shape == V.shape and not np.isnan(X).any()
    return X, names, imputer


def covariate_blocks(records: pd.DataFrame, *, patients: pd.DataFrame,
                     labevents: Optional[pd.DataFrame] = None,
                     d_labitems: Optional[pd.DataFrame] = None,
                     edstays: Optional[pd.DataFrame] = None,
                     admissions: Optional[pd.DataFrame] = None,
                     machine_measurements: Optional[pd.DataFrame] = None,
                     lab_hours: float = 24, analytes: Sequence[str] = DEFAULT_ANALYTES,
                     verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Convenience: every block whose inputs were supplied, keyed
    ``demographics`` / ``labs`` / ``setting`` / ``ecg_tabular``, all aligned to
    ``records.index`` and ready for :func:`assemble_X`."""
    blocks: Dict[str, pd.DataFrame] = {"demographics": demographics(records, patients)}
    if labevents is not None and d_labitems is not None:
        blocks["labs"] = labs_before_ecg(records, labevents, d_labitems, hours=lab_hours,
                                         analytes=analytes, verbose=verbose)
    if edstays is not None and admissions is not None:
        blocks["setting"] = setting(records, edstays, admissions)
    if machine_measurements is not None:
        blocks["ecg_tabular"] = tabular_ecg_features(machine_measurements, records)
    return blocks


def describe_blocks(blocks: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    """Provenance-style summary (n, columns, missing fraction per column)."""
    out: Dict[str, Any] = {}
    for k, df in blocks.items():
        out[k] = {"n": int(len(df)), "columns": [str(c) for c in df.columns],
                  "missing_frac": {str(c): float(df[c].isna().mean()) for c in df.columns}}
    return out
