"""Outcome construction for the MIMIC-IV-ECG x MIMIC-IV-ECHO arm.

Three outcome routes, in the priority order of ``docs/mimic_ecg_echo_spec.md``:

(a) ``Y_echo_struct`` -- the EchoNext 11-component SHD composite computed from
    structured echo measurements, **if** MIMIC-IV-ECHO ships such a table.
    v0.1 is images only, so :func:`detect_structured_echo_measurements` is
    expected to report ``available=False``; the experiment records that in
    provenance.  :func:`compute_structured_shd` is implemented anyway so that
    a future release (or a user-supplied measurement export) plugs in without
    code changes.
(b) ``Y_note`` -- discharge-summary extraction (:func:`label_from_notes`)
    using :mod:`.notes_regex` and, optionally, a local LLM through
    :mod:`.notes_llm`.
(c) ``Y_icd`` -- ICD-10 codes (:func:`label_from_icd`), auxiliary only.  ICD
    assignment is downstream of whether an echo was performed, so
    :func:`icd_circularity_diagnostic` reports the ICD-SHD rate by ``T`` and by
    echo linkage; Y_icd goes to the appendix sensitivity table only.

EchoNext reference: Poterucha, T. J. et al. "Detecting structural heart
disease from electrocardiograms using AI", *Nature* (2025),
doi:10.1038/s41586-025-09227-0; label definitions as released with the
PhysioNet ``echonext`` dataset v1.1.1 (and the EchoNext-Mini release).  The
11 component flags are transcribed in :data:`ECHONEXT_SHD_COMPONENTS`.  The
development sandbox could not open physionet.org / nature.com, so the
thresholds of the RV-dysfunction and pericardial-effusion components are the
categorical grades implied by the flag names and must be re-checked against
the paper's Methods before a structured composite is ever reported.

Nothing here reads files from disk except
:func:`detect_structured_echo_measurements`; loaders live in the sibling
modules and are responsible for the exit-code-2 policy on missing files.
"""

from __future__ import annotations

import ast
import glob
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .notes_llm import LLMExtractor, NullExtractor, merge_extractions
from .notes_regex import extract_note

__all__ = [
    "ECHONEXT_SHD_COMPONENTS",
    "MEASUREMENT_COLUMN_ALIASES",
    "SHD_ICD10_PREFIXES",
    "SHD_ICD10_DESCRIPTIONS",
    "detect_structured_echo_measurements",
    "compute_structured_shd",
    "label_from_notes",
    "label_from_icd",
    "icd_circularity_diagnostic",
    "normalise_icd10",
]

# ---------------------------------------------------------------------------
# (a) EchoNext composite from structured measurements
# ---------------------------------------------------------------------------
#: The 11 EchoNext component labels.  ``shd`` = OR over all of them.
#: ``kind``: "numeric_le" / "numeric_ge" compare a measurement with
#: ``threshold``; "grade_ge_moderate" requires a categorical grade in
#: {moderate, moderate-severe, severe} (or numeric grade >= 2 on the 0-4 scale;
#: "moderate_large" for pericardial effusion: {moderate, large}).
ECHONEXT_SHD_COMPONENTS: List[Dict[str, Any]] = [
    {"name": "lvef_lte_45", "measurement": "lvef", "unit": "%",
     "kind": "numeric_le", "threshold": 45.0,
     "description": "LV ejection fraction <= 45 %", "note_extractable": True},
    {"name": "lvwt_gte_13", "measurement": "lvwt", "unit": "cm",
     "kind": "numeric_ge", "threshold": 1.3,
     "description": "max(IVSd, LVPWd) >= 1.3 cm", "note_extractable": True},
    {"name": "aortic_stenosis_moderate_severe", "measurement": "as_grade", "unit": "grade",
     "kind": "grade_ge_moderate", "threshold": "moderate",
     "description": "aortic stenosis moderate or severe", "note_extractable": True},
    {"name": "aortic_regurgitation_moderate_severe", "measurement": "ar_grade", "unit": "grade",
     "kind": "grade_ge_moderate", "threshold": "moderate",
     "description": "aortic regurgitation moderate or severe", "note_extractable": True},
    {"name": "mitral_regurgitation_moderate_severe", "measurement": "mr_grade", "unit": "grade",
     "kind": "grade_ge_moderate", "threshold": "moderate",
     "description": "mitral regurgitation moderate or severe", "note_extractable": True},
    {"name": "tricuspid_regurgitation_moderate_severe", "measurement": "tr_grade", "unit": "grade",
     "kind": "grade_ge_moderate", "threshold": "moderate",
     "description": "tricuspid regurgitation moderate or severe", "note_extractable": True},
    {"name": "pulmonary_regurgitation_moderate_severe", "measurement": "pr_grade", "unit": "grade",
     "kind": "grade_ge_moderate", "threshold": "moderate",
     "description": "pulmonic regurgitation moderate or severe", "note_extractable": True},
    {"name": "rv_systolic_dysfunction_moderate_severe", "measurement": "rv_function_grade",
     "unit": "grade", "kind": "grade_ge_moderate", "threshold": "moderate",
     "description": "RV systolic dysfunction moderate or severe", "note_extractable": False},
    {"name": "pericardial_effusion_moderate_large", "measurement": "pericardial_effusion_grade",
     "unit": "grade", "kind": "grade_moderate_large", "threshold": "moderate",
     "description": "pericardial effusion moderate or large", "note_extractable": False},
    {"name": "pasp_gte_45", "measurement": "pasp", "unit": "mmHg",
     "kind": "numeric_ge", "threshold": 45.0,
     "description": "PA systolic pressure (RVSP) >= 45 mmHg", "note_extractable": False},
    {"name": "tr_max_gte_32", "measurement": "tr_vmax", "unit": "m/s",
     "kind": "numeric_ge", "threshold": 3.2,
     "description": "TR peak velocity >= 3.2 m/s", "note_extractable": False},
]
assert len(ECHONEXT_SHD_COMPONENTS) == 11

#: lower-cased column-name aliases used to recognise a measurements table.
MEASUREMENT_COLUMN_ALIASES: Dict[str, List[str]] = {
    "lvef": ["lvef", "ef", "lv_ef", "ejection_fraction", "lvef_pct", "lvef_percent", "ef_percent"],
    "ivsd": ["ivsd", "ivs", "ivs_d", "septal_thickness", "interventricular_septum", "ivsd_cm"],
    "lvpwd": ["lvpwd", "lvpw", "lvpw_d", "posterior_wall_thickness", "posterior_wall", "lvpwd_cm"],
    "lvwt": ["lvwt", "lv_wall_thickness", "wall_thickness", "max_wall_thickness"],
    "as_grade": ["as_grade", "aortic_stenosis", "aortic_stenosis_grade", "as_severity", "aortic_stenosis_severity"],
    "ar_grade": ["ar_grade", "aortic_regurgitation", "aortic_regurgitation_grade", "ar_severity", "aortic_insufficiency"],
    "mr_grade": ["mr_grade", "mitral_regurgitation", "mitral_regurgitation_grade", "mr_severity"],
    "tr_grade": ["tr_grade", "tricuspid_regurgitation", "tricuspid_regurgitation_grade", "tr_severity"],
    "pr_grade": ["pr_grade", "pulmonic_regurgitation", "pulmonary_regurgitation", "pulmonary_regurgitation_grade", "pr_severity"],
    "rv_function_grade": ["rv_function", "rv_systolic_function", "rv_dysfunction", "rv_function_grade", "rv_systolic_dysfunction"],
    "pericardial_effusion_grade": ["pericardial_effusion", "pericardial_effusion_grade", "effusion"],
    "pasp": ["pasp", "rvsp", "pa_systolic_pressure", "pasp_mmhg", "rvsp_mmhg"],
    "tr_vmax": ["tr_vmax", "tr_max_velocity", "tr_velocity", "tr_peak_velocity", "trv_max"],
}

_GRADE_TEXT_RE = re.compile(
    r"(?P<sev>moderate[- ]to[- ]severe|mod(?:erate)?[-/]severe|moderately\s+severe|"
    r"severe|critical|moderate|mild[- ]to[- ]moderate|mild|trivial|trace|none|absent|"
    r"large|small|normal|reduced|depressed)", re.IGNORECASE)


def _grade_rank(x: Any) -> float:
    """Categorical / numeric grade -> 0 (none) .. 4 (severe); NaN if unknown."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        v = float(x)
        return v if 0 <= v <= 4 else np.nan
    s = str(x).strip().lower()
    if not s:
        return np.nan
    m = re.fullmatch(r"([0-4])\+?", s)
    if m:
        return float(m.group(1))
    m = _GRADE_TEXT_RE.search(s)
    if not m:
        return np.nan
    sev = re.sub(r"[-/]", " ", m.group("sev")).lower()
    sev = re.sub(r"\s+", " ", sev)
    table = {"none": 0, "absent": 0, "normal": 0, "trivial": 1, "trace": 1, "small": 1,
             "mild": 2, "mild to moderate": 2, "reduced": 2, "depressed": 2,
             "moderate": 3, "moderate to severe": 3.5, "mod severe": 3.5,
             "moderately severe": 3.5, "severe": 4, "critical": 4, "large": 4}
    return float(table.get(sev, np.nan))


def _match_columns(columns: Iterable[str]) -> Dict[str, str]:
    lower = {c.lower().strip(): c for c in columns}
    hit = {}
    for meas, aliases in MEASUREMENT_COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                hit[meas] = lower[a]
                break
    return hit


def detect_structured_echo_measurements(data_root: str,
                                        echo_subdir: str = "mimic-iv-echo/0.1") -> Dict[str, Any]:
    """Look for a structured measurements table in MIMIC-IV-ECHO.

    Returns ``{"available": bool, "path": str|None, "columns_recognised":
    {measurement: column}, "searched": [...], "provenance_note": str}``.
    Never exits: the absence of this table is the *expected* state of v0.1
    (images only) and is recorded in provenance as route (a) ``unavailable``.
    """
    base = os.path.join(data_root, echo_subdir)
    searched: List[str] = []
    if not os.path.isdir(base):
        return {"available": False, "path": None, "columns_recognised": {}, "searched": [base],
                "provenance_note": (f"MIMIC-IV-ECHO directory {base} not found; route (a) "
                                    "unavailable (the loader for echo-study-list.csv "
                                    "enforces the exit-2 policy).")}
    patterns = ["*measur*", "*report*", "*param*", "*.csv", "*.csv.gz"]
    cands: List[str] = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(base, pat))) + sorted(glob.glob(os.path.join(base, "*", pat))):
            if os.path.isfile(p) and p.lower().endswith((".csv", ".csv.gz")) and p not in cands:
                cands.append(p)
    for p in cands:
        searched.append(p)
        try:
            header = list(pd.read_csv(p, nrows=0).columns)
        except Exception as exc:  # unreadable candidate: report, keep looking
            searched.append(f"{p} (unreadable: {exc})")
            continue
        hit = _match_columns(header)
        if "lvef" in hit or "ivsd" in hit or "lvwt" in hit:
            return {"available": True, "path": p, "columns_recognised": hit, "searched": searched,
                    "provenance_note": (f"structured echo measurements found at {p}; "
                                        f"recognised {sorted(hit)}; EchoNext composite via "
                                        "compute_structured_shd")}
    return {"available": False, "path": None, "columns_recognised": {}, "searched": searched,
            "provenance_note": ("MIMIC-IV-ECHO v0.1 ships DICOM images only: no structured "
                                "measurements table (searched " + ", ".join(searched or [base])
                                + "); route (a) Y_echo_struct unavailable, as expected by the spec.")}


def compute_structured_shd(measurements: pd.DataFrame,
                           column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """EchoNext 11-component composite from a structured measurements table.

    One row per echo study.  Returns the 11 component flags (float 0/1, NaN
    where the measurement is absent), ``shd_echo_struct`` (1 if any flag is 1;
    0 if no flag is 1 and at least one is 0; NaN otherwise) and
    ``n_components_evaluated``.  ``column_map`` overrides alias matching.
    Wall thickness in mm (values > 5) is converted to cm.
    """
    cols = _match_columns(measurements.columns)
    if column_map:
        cols.update(column_map)
    out = pd.DataFrame(index=measurements.index)

    def _numeric(meas: str) -> pd.Series:
        if meas not in cols:
            return pd.Series(np.nan, index=measurements.index)
        return pd.to_numeric(measurements[cols[meas]], errors="coerce")

    lvef = _numeric("lvef")
    lvef = lvef.where(lvef > 1.0, lvef * 100.0)
    ivsd, lvpwd, lvwt = _numeric("ivsd"), _numeric("lvpwd"), _numeric("lvwt")
    wall = pd.concat([ivsd, lvpwd, lvwt], axis=1).max(axis=1, skipna=True)
    wall = wall.where(wall <= 5.0, wall / 10.0)
    numeric = {"lvef": lvef, "lvwt": wall, "pasp": _numeric("pasp"), "tr_vmax": _numeric("tr_vmax")}
    for comp in ECHONEXT_SHD_COMPONENTS:
        name, meas, kind = comp["name"], comp["measurement"], comp["kind"]
        if kind in ("numeric_le", "numeric_ge"):
            v = numeric[meas]
            flag = (v <= comp["threshold"]) if kind == "numeric_le" else (v >= comp["threshold"])
            out[name] = flag.astype(float).where(v.notna(), np.nan)
        else:
            if meas not in cols:
                out[name] = np.nan
                continue
            rank = measurements[cols[meas]].map(_grade_rank).astype(float)
            out[name] = (rank >= 3.0).astype(float).where(rank.notna(), np.nan)
    flags = out[[c["name"] for c in ECHONEXT_SHD_COMPONENTS]]
    any_pos = (flags == 1).any(axis=1)
    any_eval = flags.notna().any(axis=1)
    out["shd_echo_struct"] = np.where(any_pos, 1.0, np.where(any_eval, 0.0, np.nan))
    out["n_components_evaluated"] = flags.notna().sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# (b) notes
# ---------------------------------------------------------------------------
def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _require(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing columns {missing}; has {list(df.columns)}")


def label_from_notes(records: pd.DataFrame, discharge_notes: pd.DataFrame,
                     admissions: Optional[pd.DataFrame], window_days: int = 365,
                     extractor: Optional[LLMExtractor] = None,
                     merge_policy: str = "llm_overrides") -> pd.DataFrame:
    """Route (b): ``Y_note`` per ECG record from linked discharge summaries.

    Linking (spec): the discharge summaries of the admission containing the
    linked echo (``records["echo_time"]`` when present and inside an
    admission), otherwise of every admission overlapping
    ``[ecg_time - window_days, ecg_time + window_days]``.  With
    ``admissions=None`` notes are linked by ``charttime`` inside that window
    instead.

    Aggregation over several linked notes: ``Y_note = 1`` if any note is
    composite-positive, ``0`` if none is positive and at least one note has a
    documented composite, ``NaN`` if no note could be labelled.  The reported
    LVEF / wall thickness come from the highest-confidence note.

    Returns a DataFrame aligned with ``records`` (same index) with columns
    ``Y_note, note_lvef, note_wall_cm, note_valve_modsev, note_confidence,
    note_n_notes, note_ids, note_hadm_ids, note_link``.
    """
    _require(records, ["subject_id", "study_id", "ecg_time"], "records")
    _require(discharge_notes, ["note_id", "subject_id", "hadm_id", "charttime", "text"], "discharge_notes")
    if admissions is not None:
        _require(admissions, ["subject_id", "hadm_id", "admittime", "dischtime"], "admissions")
    extractor = extractor or NullExtractor()
    w = pd.Timedelta(days=window_days)

    rec = records[["subject_id", "study_id", "ecg_time"]].copy()
    rec["ecg_time"] = _to_dt(rec["ecg_time"])
    rec["echo_time"] = _to_dt(records["echo_time"]) if "echo_time" in records.columns else pd.NaT
    rec["_row"] = np.arange(len(rec))

    notes = discharge_notes[["note_id", "subject_id", "hadm_id", "charttime", "text"]].copy()
    notes["charttime"] = _to_dt(notes["charttime"])

    if admissions is not None:
        adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
        adm["admittime"] = _to_dt(adm["admittime"])
        adm["dischtime"] = _to_dt(adm["dischtime"])
        m = rec.merge(adm, on="subject_id", how="inner")
        overlap = (m["admittime"] <= m["ecg_time"] + w) & (m["dischtime"] >= m["ecg_time"] - w)
        contains_echo = m["echo_time"].notna() & (m["admittime"] <= m["echo_time"]) & \
            (m["echo_time"] <= m["dischtime"])
        m = m[overlap | contains_echo].copy()
        m["_link"] = np.where(contains_echo[m.index], "echo_admission", "ecg_window")
        has_echo_adm = m.groupby("_row")["_link"].transform(lambda s: (s == "echo_admission").any())
        m = m[~has_echo_adm | (m["_link"] == "echo_admission")]
        linked = m[["_row", "subject_id", "hadm_id", "_link"]].merge(
            notes, on=["subject_id", "hadm_id"], how="inner")
    else:
        m = rec.merge(notes, on="subject_id", how="inner")
        inwin = (m["charttime"] >= m["ecg_time"] - w) & (m["charttime"] <= m["ecg_time"] + w)
        linked = m[inwin].copy()
        linked["_link"] = "ecg_window"

    # extract each note once
    uniq = linked.drop_duplicates("note_id")[["note_id", "text"]]
    cache: Dict[Any, Dict[str, Any]] = {}
    for nid, text in zip(uniq["note_id"], uniq["text"]):
        r = extract_note(text)
        if not extractor.is_null:
            r = merge_extractions(r, extractor.extract(text if isinstance(text, str) else ""),
                                  policy=merge_policy)
        cache[nid] = r

    out = pd.DataFrame({
        "Y_note": np.nan, "note_lvef": np.nan, "note_wall_cm": np.nan,
        "note_valve_modsev": np.nan, "note_confidence": np.nan, "note_n_notes": 0,
        "note_ids": "", "note_hadm_ids": "", "note_link": "none",
    }, index=records.index)
    out["note_n_notes"] = out["note_n_notes"].astype(int)
    if linked.empty:
        return out

    for row, grp in linked.groupby("_row"):
        idx = records.index[row]
        recs = [cache[n] for n in grp["note_id"]]
        shd = [r["shd_composite"] for r in recs]
        if any(s == 1 for s in shd):
            y = 1.0
        elif any(s == 0 for s in shd):
            y = 0.0
        else:
            y = np.nan
        best = max(recs, key=lambda r: (r["confidence"], r["n_fields_found"]))
        valve = [r["components"]["valve_moderate_or_severe"] for r in recs]
        out.at[idx, "Y_note"] = y
        out.at[idx, "note_lvef"] = best["lvef"] if best["lvef"] is not None else np.nan
        out.at[idx, "note_wall_cm"] = (best["wall_thickness_cm"]
                                       if best["wall_thickness_cm"] is not None else np.nan)
        out.at[idx, "note_valve_modsev"] = (1.0 if any(v is True for v in valve)
                                            else 0.0 if any(v is False for v in valve) else np.nan)
        out.at[idx, "note_confidence"] = max(r["confidence"] for r in recs)
        out.at[idx, "note_n_notes"] = int(len(recs))
        out.at[idx, "note_ids"] = ";".join(str(n) for n in grp["note_id"])
        out.at[idx, "note_hadm_ids"] = ";".join(sorted({str(h) for h in grp["hadm_id"]}))
        out.at[idx, "note_link"] = "echo_admission" if (grp["_link"] == "echo_admission").any() else "ecg_window"
    return out


# ---------------------------------------------------------------------------
# (c) ICD-10
# ---------------------------------------------------------------------------
#: ICD-10 prefixes (dots removed, as stored in MIMIC ``diagnoses_icd``).
SHD_ICD10_PREFIXES: List[str] = [
    "I50",   # heart failure (all I50.x)
    "I42",   # cardiomyopathy
    "I34", "I35", "I36", "I37",   # non-rheumatic mitral / aortic / tricuspid / pulmonary valve disorders
    "I05", "I06", "I07", "I08",   # rheumatic mitral / aortic / tricuspid / multiple valve disease
    "I110",  # hypertensive heart disease with heart failure
    "I255",  # ischaemic cardiomyopathy
]
SHD_ICD10_DESCRIPTIONS: Dict[str, str] = {
    "I50": "heart failure", "I42": "cardiomyopathy",
    "I34": "non-rheumatic mitral valve disorders", "I35": "non-rheumatic aortic valve disorders",
    "I36": "non-rheumatic tricuspid valve disorders", "I37": "non-rheumatic pulmonary valve disorders",
    "I05": "rheumatic mitral valve diseases", "I06": "rheumatic aortic valve diseases",
    "I07": "rheumatic tricuspid valve diseases", "I08": "multiple valve diseases",
    "I110": "hypertensive heart disease with heart failure", "I255": "ischaemic cardiomyopathy",
}


def normalise_icd10(code: Any) -> str:
    return re.sub(r"[.\s]", "", str(code)).upper() if code is not None else ""


def _is_shd_code(code: str) -> bool:
    return any(code.startswith(p) for p in SHD_ICD10_PREFIXES)


def _parse_code_list(x: Any) -> List[str]:
    """``"['I5023', 'E119']"`` (MIMIC-IV-ECG-ICD) or a list -> normalised codes."""
    if isinstance(x, (list, tuple, set)):
        return [normalise_icd10(c) for c in x]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple, set)):
            return [normalise_icd10(c) for c in v]
    except (ValueError, SyntaxError):
        pass
    return [normalise_icd10(c) for c in re.findall(r"[A-Za-z]\d[0-9A-Za-z.]*", s)]


def label_from_icd(records: pd.DataFrame, diagnoses_icd: Optional[pd.DataFrame],
                   ecg_icd_table: Optional[pd.DataFrame] = None,
                   admissions: Optional[pd.DataFrame] = None,
                   ecg_icd_column: str = "all_diag_all") -> pd.DataFrame:
    """Route (c): ``Y_icd`` per ECG record.

    Source priority: the MIMIC-IV-ECG-ICD table (``records_w_diag_icd10.csv``,
    Strodthoff et al. 2024; matched on ``study_id``, codes in column
    ``ecg_icd_column``) when given; otherwise the ICD-10 codes of the hospital
    admission containing the ECG (``admittime <= ecg_time <= dischtime``),
    which needs ``admissions``.  Records with neither, or whose admission has
    ICD-9 codes only, get ``NaN`` (not assessable).

    Returns a DataFrame aligned with ``records``: ``Y_icd`` (0/1/NaN),
    ``icd_source`` in {"mimic_ecg_icd", "hosp_admission", "hosp_admission_icd9_only",
    "none"}, ``icd_matched`` (';'-joined matching codes) and ``icd_hadm_id``.
    """
    _require(records, ["subject_id", "study_id", "ecg_time"], "records")
    if ecg_icd_table is None and (diagnoses_icd is None or admissions is None):
        raise ValueError("label_from_icd needs ecg_icd_table, or diagnoses_icd together with admissions")
    out = pd.DataFrame({"Y_icd": np.nan, "icd_source": "none", "icd_matched": "",
                        "icd_hadm_id": np.nan}, index=records.index)

    done = pd.Series(False, index=records.index)
    if ecg_icd_table is not None:
        _require(ecg_icd_table, ["study_id", ecg_icd_column], "ecg_icd_table")
        tab = ecg_icd_table.drop_duplicates("study_id").set_index("study_id")[ecg_icd_column]
        for idx, sid in zip(records.index, records["study_id"]):
            if sid in tab.index:
                codes = _parse_code_list(tab.loc[sid])
                hit = [c for c in codes if _is_shd_code(c)]
                out.at[idx, "Y_icd"] = 1.0 if hit else 0.0
                out.at[idx, "icd_source"] = "mimic_ecg_icd"
                out.at[idx, "icd_matched"] = ";".join(hit)
                done.at[idx] = True

    if diagnoses_icd is not None and admissions is not None and not done.all():
        _require(diagnoses_icd, ["subject_id", "hadm_id", "icd_code", "icd_version"], "diagnoses_icd")
        _require(admissions, ["subject_id", "hadm_id", "admittime", "dischtime"], "admissions")
        rec = records.loc[~done, ["subject_id", "study_id", "ecg_time"]].copy()
        rec["ecg_time"] = _to_dt(rec["ecg_time"])
        rec["_idx"] = rec.index
        adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
        adm["admittime"] = _to_dt(adm["admittime"])
        adm["dischtime"] = _to_dt(adm["dischtime"])
        m = rec.merge(adm, on="subject_id", how="inner")
        m = m[(m["admittime"] <= m["ecg_time"]) & (m["ecg_time"] <= m["dischtime"])]
        # one admission per ECG: the earliest-admitted containing one
        m = m.sort_values("admittime").drop_duplicates("_idx")
        dx = diagnoses_icd[["hadm_id", "icd_code", "icd_version"]].copy()
        dx["icd_version"] = pd.to_numeric(dx["icd_version"], errors="coerce")
        dx["code"] = dx["icd_code"].map(normalise_icd10)
        dx10 = dx[dx["icd_version"] == 10]
        dx10 = dx10[dx10["code"].map(_is_shd_code)]
        hits = dx10.groupby("hadm_id")["code"].apply(lambda s: ";".join(sorted(set(s))))
        has10 = set(dx.loc[dx["icd_version"] == 10, "hadm_id"])
        has_any = set(dx["hadm_id"])
        for _, r in m.iterrows():
            idx, h = r["_idx"], r["hadm_id"]
            out.at[idx, "icd_hadm_id"] = h
            if h in has10:
                hit = hits.get(h, "")
                out.at[idx, "Y_icd"] = 1.0 if hit else 0.0
                out.at[idx, "icd_source"] = "hosp_admission"
                out.at[idx, "icd_matched"] = hit
            elif h in has_any:
                out.at[idx, "icd_source"] = "hosp_admission_icd9_only"
            else:
                out.at[idx, "icd_source"] = "hosp_admission"  # admission without any code
                out.at[idx, "Y_icd"] = 0.0
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(centre - half), float(centre + half)]


def _rate_by(y: np.ndarray, g: np.ndarray, labels: Dict[Any, str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for val, name in labels.items():
        sel = (g == val) & ~np.isnan(y)
        n, k = int(sel.sum()), int(np.nansum(y[sel]))
        res[name] = {"n": n, "n_pos": k, "rate": (k / n if n else float("nan")),
                     "ci95": _wilson(k, n)}
    keys = list(labels.values())
    r0, r1 = res[keys[0]]["rate"], res[keys[1]]["rate"]
    res["risk_ratio"] = float(r1 / r0) if r0 and not np.isnan(r0) else float("nan")
    res["risk_difference"] = float(r1 - r0) if not (np.isnan(r0) or np.isnan(r1)) else float("nan")
    return res


def icd_circularity_diagnostic(y_icd: Sequence[float], T: Sequence[int],
                               echo_linked: Optional[Sequence[bool]] = None) -> Dict[str, Any]:
    """Quantify the circularity of the ICD outcome.

    ``icd_rate_by_T``: P(Y_icd = 1 | T = 1) vs P(Y_icd = 1 | T = 0) with Wilson
    95% CIs and the risk ratio.  ``icd_rate_by_echo_linked``: the same split by
    whether the subject has *any* echo study in MIMIC-IV-ECHO (regardless of
    the 365-day window), when supplied.  A risk ratio far above what the true
    SHD prevalence difference could explain signals that ICD coding is itself
    downstream of the echo decision; that is why Y_icd is appendix-only.
    """
    y = np.asarray(y_icd, dtype=float)
    t = np.asarray(T).astype(int)
    if y.shape != t.shape:
        raise ValueError("y_icd and T must have the same length")
    res: Dict[str, Any] = {
        "n": int(len(y)), "n_assessable": int((~np.isnan(y)).sum()),
        "icd_rate_by_T": _rate_by(y, t, {0: "T0", 1: "T1"}),
        "icd_rate_by_echo_linked": None,
        "interpretation": ("ICD-SHD rate among tested (T=1) vs untested (T=0) ECGs; a large "
                           "ratio reflects both true prevalence differences and coding that "
                           "depends on the echo having been done (circularity)."),
    }
    if echo_linked is not None:
        e = np.asarray(echo_linked).astype(bool).astype(int)
        if e.shape != t.shape:
            raise ValueError("echo_linked must have the same length as T")
        res["icd_rate_by_echo_linked"] = _rate_by(y, e, {0: "not_linked", 1: "linked"})
    return res
