#!/usr/bin/env python3
"""Prepare the Korean health-screening cohort (T7) for the DCL analyses.

Input: one row per screening exam (``scripts/korea_cohort/schema.md``).
Steps: schema validation -> one exam per subject per calendar year (keep the
first) -> follow-up window rule -> derived arrays T / Y / Z -> standardised
covariates -> an analysis-ready unit-level file plus an aggregate JSON sidecar.

Binding rules (CLAUDE.md):

* Real path: ``--data-root/<screening_exams.csv>``.  If the file is missing
  the script prints the path it looked for and exits with code 2.  NO
  simulator and NO fixture is ever substituted.
* Fixture path: ``--fixture PATH`` (pytest only).  Everything written is
  labelled ``data_source = "synthetic"`` and may never be reported; the script
  refuses to write fixture-derived output into the repository's ``results/``.
* Schema violations exit with code 3 and list every offending rule.
* Privacy: every count in the sidecar obeys the ``>= MIN_CELL`` units per
  cell rule (small cells are ``null`` with ``"suppressed": true``); the
  unit-level prepared file is pseudonymised intermediate data and must stay on
  the data custodian's server (never commit it).

Exit codes: 0 ok · 2 missing input file · 3 schema violation ·
4 refused to write synthetic output into results/.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS_DIR = os.path.join(REPO, "results")

SCHEMA_VERSION = "1.0"
INPUT_FILE = "screening_exams.csv"
DEFAULT_DATA_ROOT = "/data/korea_screening"
DEFAULT_OUT = os.path.join(RESULTS_DIR, "korea_cohort_prepared.parquet")
EXIT_MISSING_DATA = 2
EXIT_SCHEMA = 3
EXIT_REFUSED = 4
MIN_CELL = 10
WINDOW_DAYS = 365
DATA_SOURCES = ("real", "synthetic")

# ----------------------------------------------------------------- schema
KEY_COLUMNS = ("subject_id", "exam_date", "centre_id")
AGE_RANGE = (19, 110)
SEX_VALUES = ("F", "M")
# name -> (unit, plausible lower bound, plausible upper bound); missing allowed
NUMERIC_COVARIATES: Dict[str, Tuple[str, float, float]] = {
    "bmi": ("kg/m^2", 10.0, 80.0),
    "sbp": ("mmHg", 50.0, 300.0),
    "dbp": ("mmHg", 30.0, 200.0),
    "fasting_glucose": ("mg/dL", 20.0, 1000.0),
    "total_cholesterol": ("mg/dL", 50.0, 1000.0),
    "ldl": ("mg/dL", 0.0, 800.0),
    "hdl": ("mg/dL", 5.0, 200.0),
    "triglycerides": ("mg/dL", 10.0, 5000.0),
    "creatinine": ("mg/dL", 0.1, 30.0),
    "egfr": ("mL/min/1.73m^2", 0.0, 300.0),
    "ast": ("IU/L", 1.0, 5000.0),
    "alt": ("IU/L", 1.0, 5000.0),
    "ggt": ("IU/L", 1.0, 5000.0),
    "haemoglobin": ("g/dL", 3.0, 25.0),
}
ORDINAL_COVARIATES: Dict[str, Tuple[int, ...]] = {
    "smoking_status": (0, 1, 2),            # never / former / current
    "alcohol_freq": (0, 1, 2, 3, 4),        # none / <=1 per month / 2-4 per month / 2-3 per week / >=4 per week
    "exercise_freq": tuple(range(8)),       # days per week with >= 30 min of exercise
}
FLAG_COVARIATES = ("fh_diabetes", "fh_hypertension", "fh_cvd", "fh_cancer")
COVARIATES: List[str] = list(NUMERIC_COVARIATES) + list(ORDINAL_COVARIATES) + list(FLAG_COVARIATES)
DECISION = "followup_test_ordered"
OUTCOME = "outcome"
SCORE = "public_model_score"
FOLLOWUP = "followup_days"
REQUIRED_COLUMNS: List[str] = [*KEY_COLUMNS, "age", "sex", *COVARIATES, DECISION, OUTCOME, FOLLOWUP]
OPTIONAL_COLUMNS: List[str] = [SCORE]
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# columns of the prepared (unit-level) file
UNIT_COLUMNS = ["subject_id", "exam_date", "exam_year", "centre_id", "Z", "T",
                "Y_full", "Y_obs", SCORE, FOLLOWUP]
FEATURE_PREFIX = "x_"
MISSING_PREFIX = "m_"


def _die(msg: str, code: int) -> None:
    print(msg, file=sys.stderr, flush=True)
    sys.exit(code)


# ------------------------------------------------------ privacy: suppression
def cell(n: int, min_cell: int = MIN_CELL) -> Dict[str, Any]:
    """A published count: ``0`` and ``>= min_cell`` are reported, else suppressed."""
    n = int(n)
    if n == 0 or n >= min_cell:
        return {"n": n, "suppressed": False}
    return {"n": None, "suppressed": True}


def partition_2d(table: Dict[str, Dict[str, int]], min_cell: int = MIN_CELL
                 ) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Primary + complementary suppression for a table whose margins are published.

    Primary: every cell in ``(0, min_cell)`` is suppressed.  Complementary: as
    long as some row or column contains exactly one suppressed cell (which the
    published margin would reveal), the smallest non-zero unsuppressed cell of
    that row / column is suppressed too (``"reason": "complementary"``).
    """
    rows = list(table)
    cols = sorted({c for r in rows for c in table[r]})
    out = {r: {c: cell(table[r].get(c, 0), min_cell) for c in cols} for r in rows}

    def _fix(groups):
        changed = False
        for cells in groups:
            supp = [(r, c) for r, c in cells if out[r][c]["suppressed"]]
            if len(supp) == 1:
                cand = sorted((table[r].get(c, 0), r, c) for r, c in cells
                              if not out[r][c]["suppressed"] and table[r].get(c, 0) > 0)
                if cand:
                    _, r, c = cand[0]
                    out[r][c] = {"n": None, "suppressed": True, "reason": "complementary"}
                    changed = True
        return changed

    for _ in range(10 * (len(rows) + len(cols)) + 1):
        by_row = [[(r, c) for c in cols] for r in rows]
        by_col = [[(r, c) for r in rows] for c in cols]
        if not (_fix(by_row) | _fix(by_col)):
            break
    return out


def partition(counts: Dict[str, int], min_cell: int = MIN_CELL) -> Dict[str, Dict[str, Any]]:
    """Suppression for counts that partition a published total (one row table)."""
    return partition_2d({"_": dict(counts)}, min_cell)["_"]


def rate(k: int, n: int, min_cell: int = MIN_CELL) -> Dict[str, Any]:
    """A proportion ``k / n``; published only when the denominator, the
    numerator and its complement are each ``0`` or ``>= min_cell``."""
    k, n = int(k), int(n)
    ok = n >= min_cell and (k == 0 or k >= min_cell) and (n - k == 0 or n - k >= min_cell)
    if not ok:
        return {"value": None, "n": None, "suppressed": True}
    return {"value": k / n, "n": n, "suppressed": False}


# --------------------------------------------------------------- validation
def validate(df: pd.DataFrame, window_days: int = WINDOW_DAYS) -> List[str]:
    """Return a list of schema violations (empty list = valid)."""
    problems: List[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")
        return problems
    if len(df) == 0:
        problems.append("the file has a header but no rows")
        return problems

    def bad(mask, msg: str) -> None:
        idx = np.flatnonzero(np.asarray(mask, dtype=bool))
        if idx.size:
            problems.append(f"{msg}: {idx.size} row(s), e.g. rows {idx[:5].tolist()}")

    for c in KEY_COLUMNS:
        s = df[c]
        bad(s.isna() | (s.astype(str).str.strip() == ""), f"{c} must be a non-empty string")
    d = df["exam_date"].astype(str)
    fmt_ok = d.str.match(DATE_RE).fillna(False).to_numpy(dtype=bool)
    bad(~fmt_ok, "exam_date must be formatted YYYY-MM-DD")
    parsed = pd.to_datetime(d.where(fmt_ok), format="%Y-%m-%d", errors="coerce")
    bad(fmt_ok & parsed.isna().to_numpy(), "exam_date is not a valid calendar date")

    age = pd.to_numeric(df["age"], errors="coerce")
    bad(age.isna() | (age != np.floor(age)) | (age < AGE_RANGE[0]) | (age > AGE_RANGE[1]),
        f"age must be an integer in [{AGE_RANGE[0]}, {AGE_RANGE[1]}] (no missing)")
    bad(~df["sex"].isin(SEX_VALUES), f"sex must be one of {list(SEX_VALUES)} (no missing)")

    for c, (unit, lo, hi) in NUMERIC_COVARIATES.items():
        v = pd.to_numeric(df[c], errors="coerce")
        bad(df[c].notna() & v.isna(), f"{c} must be numeric ({unit}) or empty")
        bad(v.notna() & ((v < lo) | (v > hi)), f"{c} outside the plausible range [{lo:g}, {hi:g}] {unit}")
    for c, allowed in ORDINAL_COVARIATES.items():
        v = pd.to_numeric(df[c], errors="coerce")
        bad(df[c].notna() & ~v.isin(allowed), f"{c} must be one of {list(allowed)} or empty")
    for c in FLAG_COVARIATES:
        v = pd.to_numeric(df[c], errors="coerce")
        bad(df[c].notna() & ~v.isin([0, 1]), f"{c} must be 0 or 1 or empty")

    t = pd.to_numeric(df[DECISION], errors="coerce")
    bad(~t.isin([0, 1]), f"{DECISION} must be 0 or 1 (no missing)")
    y = pd.to_numeric(df[OUTCOME], errors="coerce")
    bad(df[OUTCOME].notna() & ~y.isin([0, 1]), f"{OUTCOME} must be 0, 1 or empty")
    bad((t == 1) & y.isna(), f"{OUTCOME} must be recorded wherever {DECISION} == 1")
    fu = pd.to_numeric(df[FOLLOWUP], errors="coerce")
    bad(fu.isna() | (fu != np.floor(fu)) | (fu < 0), f"{FOLLOWUP} must be a non-negative integer (no missing)")
    bad((y == 1) & (fu > window_days),
        f"{OUTCOME} == 1 requires {FOLLOWUP} <= {window_days} (the outcome is defined within the window)")
    if SCORE in df.columns:
        v = pd.to_numeric(df[SCORE], errors="coerce")
        bad(df[SCORE].notna() & v.isna(), f"{SCORE} must be numeric or empty")
    return problems


# ------------------------------------------------------------------ rules
def dedup_first_per_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """One exam per subject per calendar year: keep the earliest ``exam_date``
    (ties broken by file order).  Output keeps the input row order."""
    d = df.copy()
    d["_date"] = pd.to_datetime(d["exam_date"], format="%Y-%m-%d")
    d["exam_year"] = d["_date"].dt.year.astype(int)
    d["_row"] = np.arange(len(d))
    d = d.sort_values(["subject_id", "exam_year", "_date", "_row"], kind="mergesort")
    dup = d.duplicated(["subject_id", "exam_year"], keep="first").to_numpy()
    d = d[~dup].sort_values("_row", kind="mergesort").drop(columns=["_date", "_row"])
    return d.reset_index(drop=True), int(dup.sum())


def apply_window(df: pd.DataFrame, window_days: int) -> Tuple[pd.DataFrame, int]:
    """Drop units that cannot be classified within the window: no recorded
    event and fewer than ``window_days`` of follow-up."""
    y = pd.to_numeric(df[OUTCOME], errors="coerce")
    fu = pd.to_numeric(df[FOLLOWUP], errors="coerce")
    drop = ((y != 1) & (fu < window_days)).to_numpy(dtype=bool)
    return df[~drop].reset_index(drop=True), int(drop.sum())


# ------------------------------------------------------------ derivation
def derive(df: pd.DataFrame, min_cell: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Derived arrays, standardised covariates and the standardisation record."""
    n = len(df)
    centres = sorted(df["centre_id"].astype(str).unique().tolist())
    z_code = {c: i for i, c in enumerate(centres)}
    t = pd.to_numeric(df[DECISION]).astype(int).to_numpy()
    y_full = pd.to_numeric(df[OUTCOME], errors="coerce").astype(float).to_numpy()
    y_obs = np.where(t == 1, y_full, np.nan)
    score = (pd.to_numeric(df[SCORE], errors="coerce").astype(float).to_numpy()
             if SCORE in df.columns else np.full(n, np.nan))
    out = pd.DataFrame({
        "subject_id": df["subject_id"].astype(str).to_numpy(),
        "exam_date": df["exam_date"].astype(str).to_numpy(),
        "exam_year": df["exam_year"].astype(int).to_numpy(),
        "centre_id": df["centre_id"].astype(str).to_numpy(),
        "Z": df["centre_id"].astype(str).map(z_code).astype(int).to_numpy(),
        "T": t,
        "Y_full": y_full,
        "Y_obs": y_obs,
        SCORE: score,
        FOLLOWUP: pd.to_numeric(df[FOLLOWUP]).astype(int).to_numpy(),
    })

    raw: Dict[str, np.ndarray] = {
        "age": pd.to_numeric(df["age"]).astype(float).to_numpy(),
        "sex_male": (df["sex"].astype(str) == "M").astype(float).to_numpy(),
    }
    for c in COVARIATES:
        raw[c] = pd.to_numeric(df[c], errors="coerce").astype(float).to_numpy()

    stand: Dict[str, Any] = {}
    features: List[str] = []
    for name, v in raw.items():
        miss = np.isnan(v)
        median = float(np.nanmedian(v)) if (~miss).any() else 0.0
        filled = np.where(miss, median, v)
        mean = float(filled.mean())
        std = float(filled.std())
        std = std if std > 1e-9 else 1.0
        col = FEATURE_PREFIX + name
        out[col] = (filled - mean) / std
        features.append(col)
        stand[col] = {"mean": mean, "std": std, "impute_median": median,
                      "n_missing": cell(int(miss.sum()), min_cell)}
        if miss.any():
            mcol = MISSING_PREFIX + name
            out[mcol] = miss.astype(float)
            features.append(mcol)
    if n < min_cell:                      # aggregates of a tiny cohort are not published
        stand = {k: {"mean": None, "std": None, "impute_median": None, "suppressed": True,
                     "n_missing": v["n_missing"]} for k, v in stand.items()}
    record = {"features": features, "standardisation": stand, "Z_levels": centres}
    return out, record


def sidecar_counts(raw_n: int, n_dedup_dropped: int, n_window_dropped: int,
                   prepared: pd.DataFrame, min_cell: int) -> Dict[str, Any]:
    n = len(prepared)
    t = prepared["T"].to_numpy()
    y = prepared["Y_full"].to_numpy()
    obs = ~np.isnan(y)
    counts = {
        "n_raw": raw_n,
        "n_dropped_dedup": n_dedup_dropped,
        "n_after_dedup": raw_n - n_dedup_dropped,
        "n_dropped_window": n_window_dropped,
        "n_final": n,
        "n_subjects": int(prepared["subject_id"].nunique()),
        "n_centres": int(prepared["centre_id"].nunique()),
        "n_years": int(prepared["exam_year"].nunique()),
    }
    by_t = partition({"T0": int((t == 0).sum()), "T1": int((t == 1).sum())}, min_cell)
    cohort = {
        "n_T1": by_t["T1"], "n_T0": by_t["T0"],
        "selection_rate": rate(int((t == 1).sum()), n, min_cell),
        "T_all_one": bool(n > 0 and (t == 1).all()),
        "n_outcome_observed": cell(int(obs.sum()), min_cell),
        "n_outcome_missing": cell(int((~obs).sum()), min_cell),
        "outcome_observed_all": bool(n > 0 and obs.all()),
        "outcome_prevalence_observed": rate(int(np.nansum(y)), int(obs.sum()), min_cell),
        "has_public_model_score": bool(np.isfinite(prepared[SCORE].to_numpy()).any()),
        "n_public_model_score": cell(int(np.isfinite(prepared[SCORE].to_numpy()).sum()), min_cell),
    }
    per_centre = partition(prepared["centre_id"].value_counts().sort_index().astype(int).to_dict(), min_cell)
    per_year = partition({str(k): int(v) for k, v in
                          prepared["exam_year"].value_counts().sort_index().items()}, min_cell)
    ct = pd.crosstab(prepared["centre_id"], prepared["T"])
    per_centre_by_t = partition_2d(
        {str(c): {"T0": int(ct.loc[c].get(0, 0)), "T1": int(ct.loc[c].get(1, 0))} for c in ct.index},
        min_cell)
    return {"counts": counts, "cohort": cohort, "per_centre": per_centre, "per_year": per_year,
            "per_centre_by_T": per_centre_by_t}


# ------------------------------------------------------------------ io
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_input(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={c: str for c in (*KEY_COLUMNS, "sex")})


def write_prepared(df: pd.DataFrame, out: str) -> Tuple[str, str]:
    """Write parquet when pyarrow is importable, else CSV.  Returns (path, format)."""
    root, ext = os.path.splitext(out)
    have_parquet = importlib.util.find_spec("pyarrow") is not None
    if ext.lower() == ".parquet" and not have_parquet:
        print(f"[korea_cohort] pyarrow not available; writing CSV instead of {out}", flush=True)
        out, ext = root + ".csv", ".csv"
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if ext.lower() == ".parquet":
        df.to_parquet(out, index=False, engine="pyarrow")
        return out, "parquet"
    df.to_csv(out, index=False)
    return out, "csv"


def sidecar_path(prepared_path: str) -> str:
    return os.path.splitext(prepared_path)[0] + ".json"


def read_prepared(path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a prepared file and its sidecar (used by analyze_korea_cohort.py)."""
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype={"subject_id": str, "centre_id": str, "exam_date": str})
    with open(sidecar_path(path)) as f:
        side = json.load(f)
    return df, side


# ---------------------------------------------------------------- main
def prepare(path: str, source: str, out: str, window_days: int, min_cell: int) -> Dict[str, Any]:
    if source not in DATA_SOURCES:
        raise ValueError(f"source must be one of {DATA_SOURCES}")
    try:
        df = read_input(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        _die(f"[korea_cohort] SCHEMA VIOLATION: {path} is not a readable CSV ({exc})", EXIT_SCHEMA)
    problems = validate(df, window_days)
    if problems:
        lines = ["[korea_cohort] SCHEMA VIOLATION (see scripts/korea_cohort/schema.md)",
                 f"  file: {path}"] + [f"  - {p}" for p in problems]
        lines.append(f"  {len(problems)} rule(s) violated. Exiting with code {EXIT_SCHEMA}.")
        _die("\n".join(lines), EXIT_SCHEMA)
    n_raw = int(len(df))
    df, n_dedup = dedup_first_per_year(df)
    df, n_window = apply_window(df, window_days)
    prepared, record = derive(df, min_cell)
    prepared = prepared[UNIT_COLUMNS + record["features"]]
    prepared_path, fmt = write_prepared(prepared, out)
    side: Dict[str, Any] = {
        "data_source": source,
        "schema_version": SCHEMA_VERSION,
        "input": {"path": os.path.abspath(path), "sha256": sha256_file(path), "n_rows": n_raw},
        "prepared": {"path": os.path.abspath(prepared_path), "format": fmt, "n_rows": int(len(prepared)),
                     "unit_columns": UNIT_COLUMNS, "features": record["features"]},
        "rules": {
            "dedup": "one exam per subject per calendar year: keep the earliest exam_date (ties: file order)",
            "window_days": int(window_days),
            "window": "units with no recorded event and followup_days < window_days are dropped "
                      "(cannot be classified within the window); outcome == 1 requires followup_days <= window_days",
            "order": "validate -> dedup -> window -> derive",
            "min_cell": int(min_cell),
            "suppression": "counts in (0, min_cell) are null with suppressed=true; complementary suppression "
                           "inside tables whose margins are published; a rate is published only when its "
                           "denominator, numerator and complement are each 0 or >= min_cell",
        },
        "Z_levels": record["Z_levels"],
        "standardisation": record["standardisation"],
    }
    side.update(sidecar_counts(n_raw, n_dedup, n_window, prepared, min_cell))
    with open(sidecar_path(prepared_path), "w") as f:
        json.dump(side, f, indent=2)
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help=f"directory containing {INPUT_FILE} (real path; default %(default)s)")
    ap.add_argument("--fixture", default=None,
                    help="path to a synthetic fixture CSV (tests only); output is labelled data_source='synthetic'")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="prepared file (.parquet if pyarrow is available, else .csv); the JSON sidecar "
                         "is written next to it with the same stem")
    ap.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    ap.add_argument("--min-cell", type=int, default=MIN_CELL,
                    help="smallest count that may be published (default %(default)s)")
    a = ap.parse_args(argv)

    if a.fixture:
        path, source = os.path.abspath(a.fixture), "synthetic"
    else:
        path, source = os.path.join(os.path.abspath(a.data_root), INPUT_FILE), "real"
    if not os.path.isfile(path):
        _die("\n".join([
            "[korea_cohort] REQUIRED DATA FILE IS MISSING",
            f"  looked for : {path}",
            f"  data root  : {os.path.abspath(a.data_root) if not a.fixture else os.path.dirname(path)}",
            "  obtain from: the screening-cohort data custodian (IRB approval pending); place "
            f"{INPUT_FILE} in the data root (see scripts/korea_cohort/README.md)",
            f"  No simulator or fixture is substituted (CLAUDE.md rule 1). Exiting with code {EXIT_MISSING_DATA}.",
        ]), EXIT_MISSING_DATA)
    out = os.path.abspath(a.out)
    if source == "synthetic" and os.path.commonpath([out, RESULTS_DIR]) == RESULTS_DIR:
        _die(f"[korea_cohort] REFUSED: fixture-derived output may not be written under {RESULTS_DIR} "
             f"(CLAUDE.md rules 1 and 3). Choose another --out. Exiting with code {EXIT_REFUSED}.",
             EXIT_REFUSED)

    side = prepare(path, source, out, a.window_days, a.min_cell)
    c = side["counts"]
    print(f"[korea_cohort] data_source={source}  raw={c['n_raw']}  dedup-dropped={c['n_dropped_dedup']}  "
          f"window-dropped={c['n_dropped_window']}  final={c['n_final']}  centres={c['n_centres']}")
    print(f"[korea_cohort] T_all_one={side['cohort']['T_all_one']}  "
          f"outcome_observed_all={side['cohort']['outcome_observed_all']}")
    print(f"  -> {side['prepared']['path']}")
    print(f"  -> {sidecar_path(side['prepared']['path'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
