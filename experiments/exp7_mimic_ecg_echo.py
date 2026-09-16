#!/usr/bin/env python3
"""Experiment 7 -- the real medical arm: MIMIC-IV-ECG x MIMIC-IV-ECHO (T1).

Design: docs/mimic_ecg_echo_spec.md (binding).  Unit = one ECG; decision
T = 1 iff the subject has an echo within +-365 d; outcome Y = structural heart
disease (structured echo measurements when the release ships them, otherwise
the discharge-summary extraction; ICD codes are an appendix-only sensitivity
outcome because they are downstream of the echo decision).

Pipeline (in this order)
  1. load every table through ``DataRoot`` (schema verified; a missing file
     prints its path and the PhysioNet page and exits 2 -- never a simulator);
  2. link ECG -> echo (T), compute and apply the echo coverage window, dedup
     ECGs (one per subject per 30 d), attach the ordering unit Z;
  3. outcomes: (a) structured echo -> EchoNext composite, (b) notes regex
     (+ optional local LLM), (c) ICD auxiliary with the circularity diagnostic;
  4. pre-decision covariates X (``assert_no_forbidden``), waveforms when
     ``--features`` includes ``ecg``;
  5. per seed: subject-level 70/30 split; outcome model f on labelled T = 1
     train units (ECGEncoder or gradient boosting); propensity and outcome
     nuisances with Z as an input, 5-fold cross-fitted on the train split and
     **marginalised over the empirical distribution of Z** (as the
     semi-synthetic oracle in dcl/data/semisynthetic.py does); calibration
     (10 bins, ECE) and AUROC of e(x) on the test split; Gamma_min from the
     per-Z nuisances (dcl.falsify.falsification_curve) with the H_A1 check
     "Gamma_min in [1.5, 3.0]"; observed AUROC; sharp AUROC interval [L, U]
     over the Gamma grid (dcl.auc_bounds.sharp_auc_interval); break-even
     Gamma* where L crosses 0.5 / 0.6 / 0.7;
  6. mean and t-based 95% CI over seeds (experiments/_common.ci95).

Outputs (in ``--out``): ``exp7_mimic_ecg_echo_summary.json`` (``data_source``
"real" on the real path, "synthetic" on ``--fixture-root``),
``exp7_mimic_ecg_echo_seeds.csv`` (one row per seed),
``exp7_mimic_ecg_echo_gamma.csv`` (one row per seed x Gamma) and
``exp7_note_extraction_for_review.csv`` (manual-validation hook).

Exit codes: 0 ok; 2 a required data file is missing or a header does not
match (the message names the path); 3 schema / argument / provenance
violation (e.g. ``--fixture-root`` with ``--out`` inside results/).

Fixture path.  ``--fixture-root tests/fixtures/mimic_ecg_echo`` runs the whole
pipeline on the synthetic pytest fixture; ``--out`` must then lie outside
results/.  The fixture has no waveform files: on that path (and only there)
``fixture_waveforms`` synthesises 12 x 5000 arrays in memory, deterministic
per study_id.  On the real path a missing WFDB record exits 2.

    python3 experiments/exp7_mimic_ecg_echo.py --data-root /data --features both \\
        --echonext-weights /data/echonext/weights.pt --seeds 5 --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _preset_threads(argv: Sequence[str]) -> int:
    """OpenMP / BLAS thread count for sklearn's gradient boosting.  Must be in
    the environment *before* numpy / sklearn are imported, hence the pre-scan
    of ``--threads`` here.  Default 1: some sandboxes stall HistGradientBoosting
    with several OpenMP threads; raise it on the GPU server (``--threads 16``).
    An explicit ``--threads`` overrides the environment; otherwise an existing
    OMP_NUM_THREADS is respected."""
    n = None
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith("--threads="):
            n = a.split("=", 1)[1]
    explicit = n is not None
    n = str(int(n)) if n is not None else os.environ.get("OMP_NUM_THREADS", "1")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if explicit:
            os.environ[var] = n
        else:
            os.environ.setdefault(var, n)
    return int(n)


THREADS = _preset_threads(sys.argv[1:] if __name__ == "__main__" else ())

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import ci95  # noqa: E402
from dcl.auc_bounds import sharp_auc_interval  # noqa: E402
from dcl.data.mimic_ecg_echo import (DataRoot, attach_decision_maker,  # noqa: E402
                                     compute_echo_coverage_window, dedup_ecgs,
                                     link_ecg_to_echo, restrict_to_coverage)
from dcl.data.mimic_ecg_echo import covariates as cov  # noqa: E402
from dcl.data.mimic_ecg_echo import outcomes as outc  # noqa: E402
from dcl.data.mimic_ecg_echo.io import sha256_prefix  # noqa: E402
from dcl.data.mimic_ecg_echo.notes_llm import get_extractor  # noqa: E402
from dcl.falsify import falsification_curve  # noqa: E402
from dcl.objectives import dcl_bayes_score, minimax_regret, worstcase_risk  # noqa: E402
from dcl.sensitivity import logit, outcome_bounds  # noqa: E402

EXIT_OK, EXIT_MISSING_DATA, EXIT_SCHEMA = 0, 2, 3
NAME = "exp7_mimic_ecg_echo"
H_A1_RANGE = (1.5, 3.0)
DEFAULT_GAMMA_GRID = "1,1.25,1.5,2,2.5,3,4,6"
BREAKEVEN_THRESHOLDS = (0.5, 0.6, 0.7)
Z_OTHER = "other"
RESULTS_DIR = os.path.join(ROOT, "results")


# =============================================================== utilities
def die(code: int, msg: str) -> None:
    print(f"[{NAME}] {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(out_dir: str, name: str, payload: Dict[str, Any]) -> str:
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"  -> {path}", flush=True)
    return path


def save_csv(out_dir: str, name: str, df: pd.DataFrame) -> str:
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  -> {path}", flush=True)
    return path


def ci_dict(values) -> Dict[str, Any]:
    mean, hw, n = ci95(list(values))
    return {"mean": mean, "ci95": hw, "n": n}


def inside_results_dir(path: str) -> bool:
    """True if ``path`` is under the repository's results/ or has a path
    component named ``results`` (fixture outputs must never land there)."""
    p = os.path.abspath(os.path.expanduser(path))
    if p == RESULTS_DIR or p.startswith(RESULTS_DIR + os.sep):
        return True
    return "results" in os.path.normpath(p).split(os.sep)


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on bad arguments; here 2 means 'data missing', so
    argument violations exit 3 instead."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        die(EXIT_SCHEMA, f"argument error: {message}")


# ================================================================== config
@dataclass
class Config:
    data_root: Optional[str]
    fixture_root: Optional[str]
    features: str
    echonext_weights: Optional[str]
    echonext_weights_note: str
    n_seeds: int
    out: str
    gamma_grid: List[float]
    max_records: Optional[int]
    n_boot: int
    test_frac: float
    n_folds: int
    propensity_model: str
    outcome_model: str
    min_z_count: int
    llm_extractor: str
    note_validation_json: Optional[str]
    ecg_epochs: int
    ecg_patience: int
    ecg_batch_size: int
    ecg_lr: float
    ecg_base_width: int
    ecg_layers: Tuple[int, ...]
    ecg_kernel_size: int
    device: Optional[str]
    num_workers: int
    breakeven_max: float
    lab_hours: float
    echo_window_days: float
    dedup_days: float
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fixture(self) -> bool:
        return self.fixture_root is not None

    @property
    def use_ecg(self) -> bool:
        return self.features in ("ecg", "both")

    @property
    def use_tabular_ecg(self) -> bool:
        return self.features in ("tabular", "both")

    @property
    def root_dir(self) -> str:
        return self.fixture_root if self.is_fixture else (self.data_root or "/data")

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d["ecg_layers"] = list(self.ecg_layers)
        d.update(self.extra)
        return d


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(description=__doc__.splitlines()[0],
                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None,
                   help="real data root (default /data); mutually exclusive with --fixture-root")
    p.add_argument("--fixture-root", default=None,
                   help="run on a pytest fixture root; --out must lie outside results/")
    p.add_argument("--features", choices=("tabular", "ecg", "both"), default="both")
    p.add_argument("--echonext-weights", default=None,
                   help="state dict of an ECGEncoder; absent -> train the same architecture on T=1 units")
    p.add_argument("--echonext-weights-note", default="",
                   help="source / licence of the weights (recorded in provenance)")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1)")
    p.add_argument("--out", default=RESULTS_DIR)
    p.add_argument("--gamma-grid", default=DEFAULT_GAMMA_GRID)
    p.add_argument("--max-records", type=int, default=None, help="smoke runs: keep the first N ECGs")
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap replicates for Gamma_min")
    p.add_argument("--test-frac", type=float, default=0.3, help="subject-level hold-out fraction")
    p.add_argument("--n-folds", type=int, default=5, help="cross-fitting folds")
    p.add_argument("--propensity-model", choices=("gbm", "logistic"), default="gbm")
    p.add_argument("--outcome-model", choices=("gbm", "logistic"), default="gbm",
                   help="tabular outcome model (used when --features excludes ecg)")
    p.add_argument("--min-z-count", type=int, default=None,
                   help="ordering units with fewer ECGs are merged into 'other' "
                        "(default: max(2, min(100, n_analysed // 100)), i.e. 100 on the real data)")
    p.add_argument("--llm-extractor", default="null",
                   help="'null' or 'module:factory' returning an LLMExtractor (local only)")
    p.add_argument("--note-validation-json", default=None,
                   help="scored manual-review JSON (scripts/note_extraction_manual_review.py --score)")
    p.add_argument("--ecg-epochs", type=int, default=30)
    p.add_argument("--ecg-patience", type=int, default=5)
    p.add_argument("--ecg-batch-size", type=int, default=64)
    p.add_argument("--ecg-lr", type=float, default=1e-3)
    p.add_argument("--ecg-base-width", type=int, default=64, help="64 = ResNet-18 width; 4 for tests")
    p.add_argument("--ecg-layers", default="2,2,2,2", help="blocks per stage; '1,1,1,1' for tests")
    p.add_argument("--ecg-kernel-size", type=int, default=7)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=THREADS,
                   help="OpenMP/BLAS/torch CPU threads (default 1; raise on the GPU server)")
    p.add_argument("--breakeven-max", type=float, default=40.0, help="upper end of the fine Gamma grid")
    p.add_argument("--lab-hours", type=float, default=24.0)
    p.add_argument("--echo-window-days", type=float, default=365.0)
    p.add_argument("--dedup-days", type=float, default=30.0)
    return p


def parse_config(argv: Optional[Sequence[str]] = None) -> Config:
    a = build_parser().parse_args(argv)
    try:
        grid = sorted({float(g) for g in a.gamma_grid.split(",") if g.strip()})
        layers = tuple(int(x) for x in a.ecg_layers.split(","))
    except ValueError as exc:
        die(EXIT_SCHEMA, f"could not parse --gamma-grid / --ecg-layers: {exc}")
    if not grid or min(grid) < 1.0:
        die(EXIT_SCHEMA, "--gamma-grid must be non-empty with every Gamma >= 1")
    if len(layers) != 4:
        die(EXIT_SCHEMA, "--ecg-layers needs 4 comma-separated integers")
    if a.seeds < 1:
        die(EXIT_SCHEMA, "--seeds must be >= 1")
    if not (0.0 < a.test_frac < 1.0):
        die(EXIT_SCHEMA, "--test-frac must be in (0, 1)")
    if a.fixture_root is not None and a.data_root is not None:
        die(EXIT_SCHEMA, "--data-root and --fixture-root are mutually exclusive")
    if a.fixture_root is not None and inside_results_dir(a.out):
        die(EXIT_SCHEMA, f"--fixture-root given but --out={a.out!r} lies inside results/: "
                         "fixture outputs never go to results/ (CLAUDE.md rule 1/3)")
    return Config(
        data_root=a.data_root, fixture_root=a.fixture_root, features=a.features,
        echonext_weights=a.echonext_weights, echonext_weights_note=a.echonext_weights_note,
        n_seeds=a.seeds, out=os.path.abspath(os.path.expanduser(a.out)), gamma_grid=grid,
        max_records=a.max_records, n_boot=a.n_boot, test_frac=a.test_frac, n_folds=a.n_folds,
        propensity_model=a.propensity_model, outcome_model=a.outcome_model,
        min_z_count=a.min_z_count, llm_extractor=a.llm_extractor,
        note_validation_json=a.note_validation_json, ecg_epochs=a.ecg_epochs,
        ecg_patience=a.ecg_patience, ecg_batch_size=a.ecg_batch_size, ecg_lr=a.ecg_lr,
        ecg_base_width=a.ecg_base_width, ecg_layers=layers, ecg_kernel_size=a.ecg_kernel_size,
        device=a.device, num_workers=a.num_workers, breakeven_max=a.breakeven_max,
        lab_hours=a.lab_hours, echo_window_days=a.echo_window_days, dedup_days=a.dedup_days,
        extra={"threads": int(a.threads)},
    )


# ================================================================= loading
def make_root(cfg: Config) -> DataRoot:
    if cfg.is_fixture:
        return DataRoot(cfg.fixture_root, source="fixture")
    try:
        return DataRoot(cfg.root_dir, source="real")
    except ValueError as exc:            # a fixtures directory passed as --data-root
        die(EXIT_SCHEMA, f"provenance violation: {exc}")
        raise


def load_filtered(root: DataRoot, name: str, keep, usecols: Optional[Sequence[str]] = None,
                  chunksize: int = 500_000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Schema-verified chunked read of a large table keeping ``keep(chunk)`` rows
    (bounded memory for labevents / discharge notes on the real server)."""
    _, prov = root.load_table(name, nrows=0)          # header check + provenance + exit-2 policy
    parts = []
    n_total = 0
    for chunk in pd.read_csv(prov["path"], usecols=list(usecols) if usecols else None,
                             chunksize=chunksize, low_memory=False):
        n_total += len(chunk)
        sub = chunk[keep(chunk)]
        if len(sub):
            parts.append(sub)
    df = pd.concat(parts, ignore_index=True) if parts else pd.read_csv(
        prov["path"], usecols=list(usecols) if usecols else None, nrows=0)
    from dcl.data.mimic_ecg_echo.io import DATETIME_COLUMNS
    for c in DATETIME_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    prov = dict(prov, n_rows=int(len(df)), n_rows_file=int(n_total), filtered=True)
    root.loaded[name] = prov
    return df, prov


def load_tables(root: DataRoot, cfg: Config) -> Dict[str, pd.DataFrame]:
    """Every required table (exit 2 on the first missing one), then the
    optional ICD sources.  Large tables are filtered to the cohort subjects."""
    t: Dict[str, pd.DataFrame] = {}
    for name in ("record_list", "echo_study_list", "patients", "admissions", "transfers",
                 "services", "d_labitems", "edstays"):
        t[name], _ = root.load_table(name)
    if cfg.use_tabular_ecg:
        t["machine_measurements"], _ = root.load_table("machine_measurements")
    subjects = set(pd.to_numeric(t["record_list"]["subject_id"]).astype("int64").tolist())
    itemids = set()
    for ids in cov.resolve_lab_itemids(t["d_labitems"], verbose=True, strict=True).values():
        itemids |= set(ids)
    t["labevents"], _ = load_filtered(
        root, "labevents",
        lambda c: c["subject_id"].isin(subjects) & c["itemid"].isin(itemids),
        usecols=("subject_id", "hadm_id", "specimen_id", "itemid", "charttime", "value", "valuenum"))
    t["discharge"], _ = load_filtered(
        root, "discharge", lambda c: c["subject_id"].isin(subjects),
        usecols=("note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime",
                 "storetime", "text"), chunksize=50_000)
    # ICD: MIMIC-IV-ECG-ICD if present, hosp diagnoses if present; at least one required
    have_ecg_icd = os.path.isfile(root.path("records_w_diag_icd10"))
    have_dx = os.path.isfile(root.path("diagnoses_icd"))
    if have_ecg_icd:
        t["records_w_diag_icd10"], _ = root.load_table("records_w_diag_icd10")
    if have_dx or not have_ecg_icd:
        t["diagnoses_icd"], _ = root.load_table("diagnoses_icd")   # exits 2 when absent
    return t


# ================================================================= cohort
def build_cohort(t: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rec = t["record_list"]
    echo = t["echo_study_list"]
    prov: Dict[str, Any] = {"n_records_raw": int(len(rec)), "n_subjects_raw": int(rec["subject_id"].nunique()),
                            "n_echo_studies": int(len(echo)), "n_echo_subjects": int(echo["subject_id"].nunique())}
    linked = link_ecg_to_echo(rec, echo, window_days=cfg.echo_window_days)
    lo, hi = compute_echo_coverage_window(echo)
    prov["echo_coverage_window"] = [lo.isoformat(), hi.isoformat()]
    kept, cov_prov = restrict_to_coverage(linked, (lo, hi), pad_days=cfg.echo_window_days)
    prov["coverage_restriction"] = cov_prov
    kept, dd_prov = dedup_ecgs(kept, min_gap_days=cfg.dedup_days)
    prov["dedup"] = dd_prov
    kept = attach_decision_maker(kept, t["transfers"], services=t.get("services"), admissions=t["admissions"])
    prov["decision_maker_source_counts"] = {k: int(v) for k, v in kept["Z_source"].value_counts().items()}
    if cfg.max_records is not None and len(kept) > cfg.max_records:
        kept = kept.head(int(cfg.max_records)).copy()
        prov["max_records"] = int(cfg.max_records)
    kept = kept.reset_index(drop=True)
    # echo_time of the linked echo (T = 1 only): used by the note linkage
    et = echo.drop_duplicates("study_id").set_index("study_id")["study_datetime"]
    nearest = kept["nearest_echo_study_id"].astype("float").to_numpy()
    echo_time = pd.Series(pd.NaT, index=kept.index, dtype="datetime64[ns]")
    ok = (kept["T"].to_numpy() == 1) & ~np.isnan(nearest)
    echo_time[ok] = pd.to_datetime(et.reindex(nearest[ok].astype("int64")).to_numpy())
    kept["echo_time"] = echo_time
    prov["n_analysed"] = int(len(kept))
    prov["n_subjects_analysed"] = int(kept["subject_id"].nunique())
    prov["n_T1"] = int((kept["T"] == 1).sum())
    prov["n_T0"] = int((kept["T"] == 0).sum())
    prov["selection_rate"] = float(kept["T"].mean()) if len(kept) else float("nan")
    return kept, prov


def merge_rare_z(z: pd.Series, min_count: int) -> Tuple[np.ndarray, List[str], np.ndarray]:
    counts = z.value_counts()
    keep = set(counts[counts >= int(min_count)].index)
    zz = z.where(z.isin(keep), Z_OTHER)
    levels = sorted(zz.unique().tolist())
    idx = np.array([levels.index(v) for v in zz], dtype=int)
    pi = np.bincount(idx, minlength=len(levels)).astype(float)
    return idx, levels, pi / pi.sum()


# =============================================================== outcomes
def build_outcomes(records: pd.DataFrame, t: Dict[str, pd.DataFrame], root: DataRoot, cfg: Config,
                   ) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Adds Y_echo_struct / Y_note / Y_icd and Y (primary).  Returns
    ``(records, provenance, per-record note table for manual review)``."""
    prov: Dict[str, Any] = {}
    rec = records.copy()
    # (a) structured echo measurements
    det = outc.detect_structured_echo_measurements(root.root, echo_subdir=("." if cfg.is_fixture
                                                                          else "mimic-iv-echo/0.1"))
    rec["Y_echo_struct"] = np.nan
    route_a = {"available": bool(det["available"]), "path": det["path"],
               "columns_recognised": det["columns_recognised"], "note": det["provenance_note"],
               "joined": False}
    if det["available"]:
        meas = pd.read_csv(det["path"], low_memory=False)
        if "study_id" in meas.columns:
            shd = outc.compute_structured_shd(meas)
            shd.index = pd.to_numeric(meas["study_id"], errors="coerce")
            shd = shd[~shd.index.isna()]
            shd = shd[~shd.index.duplicated(keep="first")]
            key = rec["nearest_echo_study_id"].astype("float")
            vals = shd["shd_echo_struct"].reindex(key.to_numpy()).to_numpy(float)
            vals[(rec["T"].to_numpy() != 1)] = np.nan
            rec["Y_echo_struct"] = vals
            route_a["joined"] = True
            route_a["n_labelled_T1"] = int(np.sum(~np.isnan(vals)))
        else:
            route_a["note"] += " -- table has no study_id column: cannot be joined to the echo; route (b) used"
            print(f"[{NAME}] WARNING: {route_a['note']}", flush=True)
    prov["route_a_echo_struct"] = route_a
    # (b) notes
    extractor = get_extractor(cfg.llm_extractor)
    notes = outc.label_from_notes(rec, t["discharge"], t["admissions"], window_days=int(cfg.echo_window_days),
                                  extractor=extractor)
    for c in notes.columns:
        rec[c] = notes[c].to_numpy()
    T1 = rec["T"].to_numpy() == 1
    route_b = {"extractor": extractor.name, "llm_extractor_spec": cfg.llm_extractor,
               "n_records_with_notes": int((rec["note_n_notes"] > 0).sum()),
               "n_T1": int(T1.sum()), "n_T1_with_note_label": int((T1 & rec["Y_note"].notna()).sum()),
               "n_T1_undocumented": int((T1 & (rec["note_n_notes"] > 0) & rec["Y_note"].isna()).sum()),
               "n_T1_without_note": int((T1 & (rec["note_n_notes"] == 0)).sum()),
               "note_link_counts": {k: int(v) for k, v in rec.loc[T1, "note_link"].value_counts().items()},
               "composite_rule": "LVEF <= 45 OR max(IVSd, LVPWd) >= 1.3 cm OR moderate/severe AS/AR/MS/MR/TR/PR "
                                 "(subset of the EchoNext composite; deviations in notes_regex.py)"}
    review = rec[["study_id", "subject_id", "T", "note_ids", "note_hadm_ids", "note_link", "Y_note",
                  "note_lvef", "note_wall_cm", "note_valve_modsev", "note_confidence", "note_n_notes"]].copy()
    review = review[review["note_n_notes"] > 0]
    for c in ("truth_lvef", "truth_wall_cm", "truth_valve_modsev", "truth_shd", "reviewer_comment"):
        review[c] = ""
    review["extractor"] = extractor.name
    review["data_source"] = "synthetic" if cfg.is_fixture else "real"
    validation = None
    if cfg.note_validation_json:
        if not os.path.isfile(cfg.note_validation_json):
            die(EXIT_MISSING_DATA, f"--note-validation-json not found: {cfg.note_validation_json}")
        validation = json.load(open(cfg.note_validation_json))
    route_b["manual_validation"] = {
        "per_record_csv": os.path.join(cfg.out, "exp7_note_extraction_for_review.csv"),
        "sample_command": (f"python3 scripts/note_extraction_manual_review.py --data-root {cfg.root_dir} "
                           "--n 200 --seed 0 --out audit/note_review_sample.csv"),
        "score_command": ("python3 scripts/note_extraction_manual_review.py --score "
                          "audit/note_review_sample.csv --out results/note_extraction_validation.json"),
        "validation_json": cfg.note_validation_json,
        "validation": validation,
    }
    prov["route_b_notes"] = route_b
    # (c) ICD, auxiliary only
    icd = outc.label_from_icd(rec, t.get("diagnoses_icd"), t.get("records_w_diag_icd10"), t["admissions"])
    for c in icd.columns:
        rec[c] = icd[c].to_numpy()
    echo_linked = rec["nearest_echo_study_id"].notna().to_numpy()
    circ = outc.icd_circularity_diagnostic(rec["Y_icd"].to_numpy(float), rec["T"].to_numpy(int), echo_linked)
    prov["route_c_icd"] = {"circularity": circ,
                           "icd_source_counts": {k: int(v) for k, v in rec["icd_source"].value_counts().items()},
                           "codes": outc.SHD_ICD10_PREFIXES,
                           "status": "auxiliary; appendix sensitivity table only (circular with the echo decision)"}
    # primary outcome
    if route_a["joined"]:
        rec["Y"] = rec["Y_echo_struct"].to_numpy(float)
        prov["primary_outcome_route"] = "a_echo_struct"
    else:
        rec["Y"] = rec["Y_note"].to_numpy(float)
        prov["primary_outcome_route"] = "b_notes"
    rec.loc[rec["T"] != 1, "Y"] = np.nan
    lab = T1 & rec["Y"].notna().to_numpy()
    prov["n_T1_labelled"] = int(lab.sum())
    prov["label_recovery_rate_T1"] = float(lab.sum() / max(T1.sum(), 1))
    prov["prevalence_among_labelled_T1"] = float(rec.loc[lab, "Y"].mean()) if lab.any() else float("nan")
    prov["unlabelled_T1_policy"] = ("T = 1 ECGs whose outcome could not be recovered are kept for the "
                                    "propensity (T is known) and excluded from the outcome model and from "
                                    "the observed AUROC; reported as label_recovery_rate_T1")
    return rec, prov, review


# ============================================================= covariates
def build_blocks(records: pd.DataFrame, t: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    try:
        blocks = cov.covariate_blocks(
            records, patients=t["patients"], labevents=t["labevents"], d_labitems=t["d_labitems"],
            edstays=t["edstays"], admissions=t["admissions"],
            machine_measurements=t.get("machine_measurements") if cfg.use_tabular_ecg else None,
            lab_hours=cfg.lab_hours, verbose=False)
    except ValueError as exc:            # an analyte pattern resolves to nothing: fix, do not paper over
        die(EXIT_SCHEMA, f"covariate schema problem: {exc}")
        raise
    for b in blocks.values():
        cov.assert_no_forbidden(b.columns)
    return blocks


# ============================================================== waveforms
def fixture_waveforms(study_ids: Sequence[int], n_samples: int = 5000, n_leads: int = 12,
                      fs: int = 500) -> np.ndarray:
    """Synthetic 12-lead arrays for the pytest fixture ONLY (deterministic per
    study_id; no physiological content).  Never called on the real path."""
    t = np.arange(int(n_samples)) / float(fs)
    out = np.empty((len(study_ids), int(n_leads), int(n_samples)), dtype=np.float32)
    for i, sid in enumerate(study_ids):
        rng = np.random.default_rng(int(sid))
        hr = rng.uniform(50.0, 110.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        base = np.sin(2 * np.pi * hr / 60.0 * t + phase) + 0.3 * np.sin(2 * np.pi * 2 * hr / 60.0 * t)
        gain = rng.uniform(0.3, 1.5, size=int(n_leads))
        out[i] = gain[:, None] * base[None, :] + 0.1 * rng.standard_normal((int(n_leads), int(n_samples)))
    return out


def load_waveforms(records: pd.DataFrame, root: DataRoot, cfg: Config):
    """Real path: ``ECGDataset`` over the WFDB files (missing -> exit 2).
    Fixture path: in-memory synthetic arrays (documented in the spec)."""
    import torch
    from dcl import models_ecg
    torch.set_num_threads(max(1, int(cfg.extra.get("threads", 1))))
    if cfg.is_fixture:
        print(f"[{NAME}] fixture path: synthesising {len(records)} waveforms in memory "
              "(fixture_waveforms; no waveform files in the fixture)", flush=True)
        return fixture_waveforms(records["study_id"].to_numpy())
    ecg_root = os.path.join(root.root, "mimic-iv-ecg", "1.0")
    if not os.path.isdir(ecg_root):
        ecg_root = root.root
    try:
        return models_ecg.ECGDataset(records["file_name"].tolist(), ecg_root, on_missing="exit")
    except ImportError as exc:
        die(EXIT_SCHEMA, str(exc))
        raise


def _subset_waves(waves, idx: np.ndarray):
    if isinstance(waves, np.ndarray):
        return waves[idx]
    from torch.utils.data import Subset
    return Subset(waves, [int(i) for i in idx])


def encoder_kwargs(cfg: Config) -> Dict[str, Any]:
    return dict(base_width=cfg.ecg_base_width, layers=tuple(cfg.ecg_layers),
                kernel_size=cfg.ecg_kernel_size)


def load_echonext(cfg: Config) -> Tuple[Any, Dict[str, Any]]:
    import torch
    from dcl.models_ecg import ECGEncoder
    path = cfg.echonext_weights
    if not os.path.isfile(path):
        die(EXIT_MISSING_DATA, f"--echonext-weights not found: {path}\n  (EchoNext weights are a required "
                               "input when the flag is given; no substitute is trained silently)")
    obj = torch.load(path, map_location="cpu")
    sd = obj
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
    enc = ECGEncoder(n_outputs=1, **encoder_kwargs(cfg))
    try:
        enc.load_state_dict(sd, strict=True)
    except (RuntimeError, TypeError, AttributeError) as exc:
        die(EXIT_SCHEMA, f"--echonext-weights do not match ECGEncoder({encoder_kwargs(cfg)}): {exc}")
    enc.eval()
    prov = {"path": os.path.abspath(path), "sha256_first_mib": sha256_prefix(path),
            "note": cfg.echonext_weights_note or "source/licence not stated (pass --echonext-weights-note)",
            "architecture": encoder_kwargs(cfg)}
    return enc, prov


# ================================================================ models
class ConstantModel:
    """predict_proba for a degenerate label set (one class only)."""

    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def make_clf(kind: str, seed: int, n_train: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "gbm":
        msl = int(min(30, max(2, n_train // 8)))    # 30 as dcl.nuisance; shrinks only on tiny data
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.07, max_leaf_nodes=31,
                                              min_samples_leaf=msl, l2_regularization=1.0,
                                              random_state=int(seed))
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    raise KeyError(kind)


def _calibrated(base, y: np.ndarray, min_per_class: int = 15):
    """Isotonic calibration (cv=3) as dcl.nuisance does; skipped on tiny data."""
    from sklearn.calibration import CalibratedClassifierCV
    counts = np.bincount(np.asarray(y, int), minlength=2)
    if counts.min() >= min_per_class:
        return CalibratedClassifierCV(base, method="isotonic", cv=3), True
    return base, False


def _design(X: np.ndarray, z_idx: np.ndarray, K: int) -> np.ndarray:
    oh = np.zeros((X.shape[0], K), dtype=float)
    oh[np.arange(X.shape[0]), np.asarray(z_idx, int)] = 1.0
    return np.hstack([np.asarray(X, float), oh])


@dataclass
class NuisanceModels:
    e_models: List[Any]
    p1_models: List[Any]
    e_oof: np.ndarray
    p1_oof: np.ndarray
    K: int
    meta: Dict[str, Any]


def fit_nuisances_with_z(X: np.ndarray, z_idx: np.ndarray, K: int, T: np.ndarray, Y: np.ndarray,
                         seed: int, n_folds: int, kind: str) -> NuisanceModels:
    """Cross-fitted e(x, z) = P(T=1 | X, Z) on all units and p1(x, z) =
    P(Y=1 | X, Z, T=1) on the labelled T = 1 units (StratifiedKFold; fold
    models are kept for out-of-sample prediction on the test split)."""
    from sklearn.model_selection import StratifiedKFold
    Xz = _design(X, z_idx, K)
    n = len(T)
    T = np.asarray(T, int)
    meta: Dict[str, Any] = {"n_train": int(n)}
    # --- propensity
    e_oof = np.full(n, np.nan)
    e_models: List[Any] = []
    kf = int(min(n_folds, np.bincount(T, minlength=2).min()))
    meta["n_folds_e"] = kf
    if kf >= 2:
        for k, (tr, te) in enumerate(StratifiedKFold(kf, shuffle=True, random_state=seed).split(Xz, T)):
            m, cal = _calibrated(make_clf(kind, seed + k, len(tr)), T[tr])
            m.fit(Xz[tr], T[tr])
            e_oof[te] = m.predict_proba(Xz[te])[:, 1]
            e_models.append(m)
            meta["calibrated_e"] = cal
    else:
        m = ConstantModel(float(T.mean()))
        e_oof[:] = T.mean()
        e_models.append(m)
        meta["calibrated_e"] = False
    # --- outcome among labelled T = 1
    sel = np.flatnonzero((T == 1) & ~np.isnan(Y))
    ysel = np.asarray(Y, float)[sel].astype(int)
    p1_oof = np.full(n, np.nan)
    p1_models: List[Any] = []
    if len(sel) == 0 or len(np.unique(ysel)) < 2:
        p = float(ysel.mean()) if len(sel) else 0.5
        p1_models.append(ConstantModel(p))
        p1_oof[:] = p
        meta["n_folds_p1"] = 0
        meta["calibrated_p1"] = False
    else:
        kf2 = int(min(n_folds, np.bincount(ysel, minlength=2).min()))
        meta["n_folds_p1"] = kf2
        if kf2 >= 2:
            preds = np.zeros((kf2, n))
            splits = list(StratifiedKFold(kf2, shuffle=True, random_state=seed + 1).split(Xz[sel], ysel))
            for k, (tr, _te) in enumerate(splits):
                m, cal = _calibrated(make_clf(kind, seed + 10 + k, len(tr)), ysel[tr])
                m.fit(Xz[sel][tr], ysel[tr])
                preds[k] = m.predict_proba(Xz)[:, 1]
                p1_models.append(m)
                meta["calibrated_p1"] = cal
            p1_oof = preds.mean(axis=0)
            for k, (_tr, te) in enumerate(splits):
                others = [j for j in range(kf2) if j != k]
                p1_oof[sel[te]] = preds[others][:, sel[te]].mean(axis=0)
        else:
            m, cal = _calibrated(make_clf(kind, seed + 10, len(sel)), ysel)
            m.fit(Xz[sel], ysel)
            p1_oof[:] = m.predict_proba(Xz)[:, 1]
            p1_models.append(m)
            meta["calibrated_p1"] = cal
    return NuisanceModels(e_models, p1_models, e_oof, p1_oof, K, meta)


def predict_by_z(nm: NuisanceModels, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``e_z(x)`` and ``p1_z(x)`` for every unit and every ordering unit z
    (common grid of units), shape ``(n, K)``."""
    n = np.asarray(X).shape[0]
    e_z = np.zeros((n, nm.K))
    p1_z = np.zeros((n, nm.K))
    for k in range(nm.K):
        Xz = _design(X, np.full(n, k), nm.K)
        e_z[:, k] = np.mean([m.predict_proba(Xz)[:, 1] for m in nm.e_models], axis=0)
        p1_z[:, k] = np.mean([m.predict_proba(Xz)[:, 1] for m in nm.p1_models], axis=0)
    return e_z, p1_z


def marginalise_over_z(e_z: np.ndarray, p1_z: np.ndarray, pi_z: np.ndarray, clip: float = 1e-3,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """``e(x) = sum_z pi_z e_z(x)``, ``p1(x) = sum_z pi_z e_z(x) p1_z(x) / e(x)``
    -- the Z-marginal nuisances of dcl/data/semisynthetic.py."""
    e = e_z @ pi_z
    j1 = (e_z * p1_z) @ pi_z
    p1 = j1 / np.clip(e, 1e-12, None)
    return np.clip(e, clip, 1 - clip), np.clip(p1, clip, 1 - clip)


def calibration_10bins(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    edges = np.linspace(0.0, 1.0, 11)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
    n = np.bincount(b, minlength=10).astype(float)
    mean_pred = np.full(10, np.nan)
    frac_pos = np.full(10, np.nan)
    for k in range(10):
        if n[k] > 0:
            mean_pred[k] = float(np.mean(p[b == k]))
            frac_pos[k] = float(np.mean(y[b == k]))
    ok = n > 0
    ece = float(np.sum(n[ok] / n.sum() * np.abs(mean_pred[ok] - frac_pos[ok]))) if n.sum() else float("nan")
    return {"bin_edges": edges.tolist(), "n": n.tolist(), "mean_pred": mean_pred.tolist(),
            "frac_pos": frac_pos.tolist(), "ece": ece}


def _safe_auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    ok = ~np.isnan(y) & ~np.isnan(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def subject_holdout(sid: np.ndarray, T: np.ndarray, Y: np.ndarray, test_frac: float, seed: int,
                    max_tries: int = 10) -> Tuple[np.ndarray, int]:
    """Subject-level train mask.  Re-drawn (seed + 1000 k) when a split cannot
    support the analysis (tiny data): test needs both T classes, train needs
    both Y classes among labelled T = 1 units.  Returns ``(train_mask, seed_used)``."""
    uniq = np.unique(sid)
    n_test = int(min(max(1, round(test_frac * len(uniq))), len(uniq) - 1))
    for k in range(max_tries):
        s = seed + 1000 * k
        rng = np.random.default_rng(s)
        test_subj = set(rng.permutation(uniq)[:n_test].tolist())
        train = np.array([x not in test_subj for x in sid.tolist()], dtype=bool)
        lab = train & (T == 1) & ~np.isnan(Y)
        if len(np.unique(T[~train])) == 2 and lab.sum() >= 2 and len(np.unique(Y[lab])) == 2:
            return train, s
    return train, s


def breakeven(f: np.ndarray, p1: np.ndarray, e: np.ndarray, gamma_max: float) -> Dict[str, Any]:
    """Largest Gamma on a fine geometric grid at which the sharp lower AUROC
    bound still clears each threshold."""
    grid = np.concatenate([[1.0], np.exp(np.linspace(np.log(1.05), np.log(gamma_max), 60))])
    lows = []
    for g in grid:
        lo, hi = outcome_bounds(p1, e, g)
        lows.append(sharp_auc_interval(f, lo, hi).lower)
    lows = np.asarray(lows)
    out: Dict[str, Any] = {"grid_max": float(gamma_max)}
    for thr in BREAKEVEN_THRESHOLDS:
        key = f"{int(round(thr * 100))}"
        ok = lows >= thr
        out[f"gamma_star_{key}"] = float(grid[ok].max()) if ok.any() else 1.0
        out[f"holds_at_gamma_1_{key}"] = bool(ok[0])
        out[f"censored_at_grid_max_{key}"] = bool(ok[-1])
    return out


# =============================================================== one seed
def run_seed(seed: int, cfg: Config, rec: pd.DataFrame, blocks_tab: Dict[str, pd.DataFrame],
             z_idx: np.ndarray, K: int, pi_z: np.ndarray, waves, echonext) -> Dict[str, Any]:
    t0 = time.time()
    sid = rec["subject_id"].to_numpy()
    T = rec["T"].to_numpy(int)
    Y = rec["Y"].to_numpy(float)
    Y_icd = rec["Y_icd"].to_numpy(float)
    train, split_seed = subject_holdout(sid, T, Y, cfg.test_frac, seed)
    test = ~train
    labelled = (T == 1) & ~np.isnan(Y)
    row: Dict[str, Any] = {"seed": seed, "split_seed_used": split_seed, "n_train": int(train.sum()),
                           "n_test": int(test.sum()), "n_train_T1_labelled": int((train & labelled).sum()),
                           "n_test_T1_labelled": int((test & labelled).sum())}
    blocks = dict(blocks_tab)
    f_all = None
    # --- outcome model f
    if cfg.use_ecg:
        from dcl.models_ecg import embed_records, predict_proba, train_ecg_classifier
        if echonext is not None:
            encoders = [echonext]
            row["outcome_model"] = "echonext_weights"
            row["outcome_val_auc"] = float("nan")
        else:
            tr_idx = np.flatnonzero(train & labelled)
            res = train_ecg_classifier(_subset_waves(waves, tr_idx), Y[tr_idx], sid[tr_idx], seeds=(seed,),
                                       epochs=cfg.ecg_epochs, patience=cfg.ecg_patience,
                                       batch_size=cfg.ecg_batch_size, lr=cfg.ecg_lr, device=cfg.device,
                                       num_workers=cfg.num_workers, **encoder_kwargs(cfg))
            encoders = res.models
            row["outcome_model"] = "ecg_encoder_trained"
            row["outcome_val_auc"] = float(res.val_auc[0])
            row["outcome_best_epoch"] = int(res.best_epoch[0])
        f_all = logit(predict_proba(encoders, waves, batch_size=cfg.ecg_batch_size, device=cfg.device))
        emb = embed_records(encoders[0], waves, batch_size=cfg.ecg_batch_size, device=cfg.device)
        blocks["ecg_embedding"] = pd.DataFrame(emb, index=rec.index,
                                               columns=[f"ecg_emb_{j:03d}" for j in range(emb.shape[1])])
        cov.assert_no_forbidden(blocks["ecg_embedding"].columns)
    try:
        X, names, _ = cov.assemble_X(blocks, train)
    except cov.ForbiddenFeatureError as exc:
        die(EXIT_SCHEMA, str(exc))
        raise
    row["n_features"] = int(X.shape[1])
    if not cfg.use_ecg or cfg.features == "both":
        X_tab, _, _ = cov.assemble_X(blocks_tab, train)
        lab_tr = train & labelled
        clf = make_clf(cfg.outcome_model, seed, int(lab_tr.sum())).fit(X_tab[lab_tr], Y[lab_tr].astype(int))
        f_tab = logit(clf.predict_proba(X_tab)[:, 1])
        row["observed_auroc_tabular"] = _safe_auc(Y[test & labelled], f_tab[test & labelled])
        if f_all is None:
            f_all = f_tab
            row["outcome_model"] = f"{cfg.outcome_model}_tabular"
            row["outcome_val_auc"] = float("nan")
    # --- nuisances with Z on the train split, marginalised over Z on the test split
    nm = fit_nuisances_with_z(X[train], z_idx[train], K, T[train], Y[train], seed, cfg.n_folds,
                              cfg.propensity_model)
    e_z, p1_z = predict_by_z(nm, X[test])
    e_te, p1_te = marginalise_over_z(e_z, p1_z, pi_z)
    T_te, f_te = T[test], f_all[test]
    row.update({f"nuisance_{k}": v for k, v in nm.meta.items()})
    row["propensity_auroc"] = _safe_auc(T_te, e_te)
    row["propensity_auroc_train_oof_realised_z"] = _safe_auc(T[train], nm.e_oof)
    cal = calibration_10bins(T_te.astype(float), e_te)
    row["propensity_ece"] = cal["ece"]
    row["propensity_mean"] = float(e_te.mean())
    row["selection_rate_test"] = float(T_te.mean())
    # --- Gamma_min from per-Z nuisances on the common grid of test units
    if K >= 2:
        fal = falsification_curve([p1_z[:, k] for k in range(K)], [e_z[:, k] for k in range(K)],
                                  n_boot=cfg.n_boot, seed=seed)
        row["gamma_min"], row["gamma_min_q95"], row["gamma_min_boot_lo"] = fal.gamma_min, fal.gamma_min_q95, fal.gamma_min_boot_lo
    else:
        row["gamma_min"] = row["gamma_min_q95"] = row["gamma_min_boot_lo"] = float("nan")
    row["H_A1_point"] = bool(H_A1_RANGE[0] <= row["gamma_min"] <= H_A1_RANGE[1])
    row["H_A1_q95"] = bool(H_A1_RANGE[0] <= row["gamma_min_q95"] <= H_A1_RANGE[1])
    # --- observed AUROC and sharp interval over the Gamma grid
    lab_te = test & labelled
    row["observed_auroc"] = _safe_auc(Y[lab_te], f_all[lab_te])
    gamma_rows = []
    gammas = list(cfg.gamma_grid)
    if np.isfinite(row["gamma_min"]):
        gammas.append(("gamma_min", float(max(1.0, row["gamma_min"]))))
    for g in gammas:
        tag, gv = (g if isinstance(g, tuple) else (f"{g:g}", float(g)))
        lo, hi = outcome_bounds(p1_te, e_te, gv)
        res = sharp_auc_interval(f_te, lo, hi)
        f_dcl = dcl_bayes_score(lo, hi, "regret")
        gamma_rows.append(dict(seed=seed, gamma_tag=tag, gamma=gv, lower=res.lower, upper=res.upper,
                               width=res.width, wc_risk_incumbent=worstcase_risk(f_te, lo, hi),
                               wc_risk_dcl=worstcase_risk(f_dcl, lo, hi),
                               regret_incumbent=minimax_regret(f_te, lo, hi),
                               regret_dcl=minimax_regret(f_dcl, lo, hi),
                               mean_width_pointwise=float(np.mean(hi - lo))))
    row.update(breakeven(f_te, p1_te, e_te, cfg.breakeven_max))
    # --- ICD sensitivity (appendix only)
    row["icd_auroc_T1_test"] = _safe_auc(Y_icd[test & (T == 1)], f_te[T_te == 1])
    row["icd_auroc_T0_test"] = _safe_auc(Y_icd[test & (T == 0)], f_te[T_te == 0])
    row["seconds"] = float(time.time() - t0)
    return {"row": row, "gamma_rows": gamma_rows, "calibration": cal}


# ============================================================== aggregate
def aggregate(results: List[Dict[str, Any]], cfg: Config) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame([r["row"] for r in results])
    grows = pd.DataFrame([g for r in results for g in r["gamma_rows"]])
    summ: Dict[str, Any] = {}
    scalar_keys = [c for c in rows.columns if pd.api.types.is_numeric_dtype(rows[c])
                   and c not in ("seed", "split_seed_used")]
    summ["per_seed_ci"] = {c: ci_dict(rows[c].to_numpy(float)) for c in scalar_keys}
    gm = ci_dict(rows["gamma_min"].to_numpy(float))
    summ["gamma_min"] = {"point": gm, "q95": ci_dict(rows["gamma_min_q95"].to_numpy(float)),
                         "boot_lo": ci_dict(rows["gamma_min_boot_lo"].to_numpy(float)),
                         "per_seed": rows["gamma_min"].tolist()}
    lo_ci = gm["mean"] - gm["ci95"] if np.isfinite(gm["ci95"]) else float("nan")
    hi_ci = gm["mean"] + gm["ci95"] if np.isfinite(gm["ci95"]) else float("nan")
    summ["H_A1"] = {
        "statement": "Gamma_min in [1.5, 3.0]", "range": list(H_A1_RANGE),
        "gamma_min_mean": gm["mean"], "gamma_min_ci95": gm["ci95"],
        "holds": bool(H_A1_RANGE[0] <= gm["mean"] <= H_A1_RANGE[1]) if np.isfinite(gm["mean"]) else False,
        "holds_per_seed": [bool(x) for x in rows["H_A1_point"].tolist()],
        "fraction_of_seeds": float(rows["H_A1_point"].astype(float).mean()),
        "ci95_entirely_within": bool(np.isfinite(lo_ci) and H_A1_RANGE[0] <= lo_ci and hi_ci <= H_A1_RANGE[1]),
        "holds_q95_mean": bool(H_A1_RANGE[0] <= summ["gamma_min"]["q95"]["mean"] <= H_A1_RANGE[1])
        if np.isfinite(summ["gamma_min"]["q95"]["mean"]) else False,
    }
    cal_n = np.nansum([r["calibration"]["n"] for r in results], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # bins empty in every seed -> NaN
        cal_pred = np.nanmean([r["calibration"]["mean_pred"] for r in results], axis=0).tolist()
        cal_pos = np.nanmean([r["calibration"]["frac_pos"] for r in results], axis=0).tolist()
    summ["propensity"] = {
        "auroc": ci_dict(rows["propensity_auroc"]), "ece": ci_dict(rows["propensity_ece"]),
        "auroc_train_oof_realised_z": ci_dict(rows["propensity_auroc_train_oof_realised_z"]),
        "calibration_curve_10bins": {
            "bin_edges": results[0]["calibration"]["bin_edges"],
            "n_total": cal_n.tolist(),
            "mean_pred": cal_pred,
            "frac_pos": cal_pos,
        },
        "z_marginalisation": ("e(x) = sum_z pi_z e_z(x), p1(x) = sum_z pi_z e_z(x) p1_z(x) / e(x); pi_z = "
                              "empirical distribution of the ordering unit over the analysed ECGs "
                              "(dcl/data/semisynthetic.py oracle convention); e_z, p1_z from the "
                              "cross-fitted models with Z as an input"),
    }
    summ["outcome_model"] = {"kind": rows["outcome_model"].iloc[0], "observed_auroc": ci_dict(rows["observed_auroc"]),
                             "val_auc_inner_split": ci_dict(rows["outcome_val_auc"])}
    if "observed_auroc_tabular" in rows:
        summ["outcome_model"]["observed_auroc_tabular"] = ci_dict(rows["observed_auroc_tabular"])
    intervals: Dict[str, Any] = {}
    for tag, gg in grows.groupby("gamma_tag", sort=False):
        d: Dict[str, Any] = {"gamma": ci_dict(gg["gamma"]) if tag == "gamma_min" else float(gg["gamma"].iloc[0])}
        for k in ("lower", "upper", "width", "wc_risk_incumbent", "wc_risk_dcl", "regret_incumbent",
                  "regret_dcl", "mean_width_pointwise"):
            d[k] = ci_dict(gg[k])
        intervals[tag] = d
    summ["intervals"] = intervals
    summ["breakeven"] = {f"auroc_{k}": {"gamma_star": ci_dict(rows[f"gamma_star_{k}"]),
                                        "holds_at_gamma_1_fraction": float(rows[f"holds_at_gamma_1_{k}"].astype(float).mean()),
                                        "censored_at_grid_max_fraction": float(rows[f"censored_at_grid_max_{k}"].astype(float).mean())}
                         for k in ("50", "60", "70")}
    summ["breakeven"]["grid_max"] = float(cfg.breakeven_max)
    summ["icd_sensitivity_per_seed"] = {"auroc_vs_Y_icd_T1_test": ci_dict(rows["icd_auroc_T1_test"]),
                                        "auroc_vs_Y_icd_T0_test": ci_dict(rows["icd_auroc_T0_test"]),
                                        "status": "appendix-only sensitivity; Y_icd is circular with the echo decision"}
    return summ, rows, grows


# =================================================================== main
def main(argv: Optional[Sequence[str]] = None) -> int:
    cfg = parse_config(argv)
    t_start = time.time()
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(int(cfg.extra["threads"]))
    except ImportError:
        pass
    root = make_root(cfg)
    print(f"[{NAME}] mode={'FIXTURE (synthetic)' if cfg.is_fixture else 'REAL'} root={root.root} "
          f"features={cfg.features} seeds={cfg.n_seeds} out={cfg.out}", flush=True)
    tables = load_tables(root, cfg)
    if not cfg.is_fixture and not root.all_real():
        die(EXIT_SCHEMA, "provenance violation: not every table came from --data-root as real data")
    rec, cohort_prov = build_cohort(tables, cfg)
    print(f"[{NAME}] echo coverage window {cohort_prov['echo_coverage_window']}; ECGs "
          f"{cohort_prov['n_records_raw']} -> coverage {cohort_prov['coverage_restriction']['n_after']} "
          f"-> dedup {cohort_prov['dedup']['n_after']} -> analysed {cohort_prov['n_analysed']} "
          f"(T=1: {cohort_prov['n_T1']})", flush=True)
    if cohort_prov["n_analysed"] < 4 or cohort_prov["n_T1"] == 0 or cohort_prov["n_T0"] == 0:
        die(EXIT_SCHEMA, "cohort too small or single-valued T after restriction; nothing to estimate")
    rec, out_prov, review = build_outcomes(rec, tables, root, cfg)
    print(f"[{NAME}] primary outcome route: {out_prov['primary_outcome_route']}; labelled T=1: "
          f"{out_prov['n_T1_labelled']}/{cohort_prov['n_T1']} (prevalence "
          f"{out_prov['prevalence_among_labelled_T1']:.3f})", flush=True)
    if out_prov["n_T1_labelled"] < 2:
        die(EXIT_SCHEMA, "fewer than 2 labelled T=1 ECGs; the outcome model cannot be fit")
    blocks = build_blocks(rec, tables, cfg)
    min_z = (int(cfg.min_z_count) if cfg.min_z_count is not None
             else int(max(2, min(100, len(rec) // 100))))
    cfg.extra["min_z_count_effective"] = min_z
    z_idx, z_levels, pi_z = merge_rare_z(rec["Z"].astype(str), min_z)
    print(f"[{NAME}] ordering units after merging (< {min_z} -> other): {len(z_levels)} "
          f"{dict(zip(z_levels, np.round(pi_z, 3).tolist()))}", flush=True)
    waves, echonext, echonext_prov = None, None, None
    if cfg.use_ecg:
        waves = load_waveforms(rec, root, cfg)
        if cfg.echonext_weights:
            echonext, echonext_prov = load_echonext(cfg)
    os.makedirs(cfg.out, exist_ok=True)
    results = []
    for seed in range(cfg.n_seeds):
        r = run_seed(seed, cfg, rec, blocks, z_idx, len(z_levels), pi_z, waves, echonext)
        results.append(r)
        row = r["row"]
        print(f"[{NAME}] seed {seed}: e-AUROC {row['propensity_auroc']:.3f} ECE {row['propensity_ece']:.3f} | "
              f"Gamma_min {row['gamma_min']:.3f} (q95 {row['gamma_min_q95']:.3f}) H_A1={row['H_A1_point']} | "
              f"obs AUROC {row['observed_auroc']:.3f} | Gamma*_50 {row['gamma_star_50']:.2f} "
              f"Gamma*_70 {row['gamma_star_70']:.2f} | {row['seconds']:.1f}s", flush=True)
    summ, rows, grows = aggregate(results, cfg)
    data_source = "synthetic" if cfg.is_fixture else "real"
    payload: Dict[str, Any] = {
        "data_source": data_source,
        "experiment": NAME,
        "mode": "fixture" if cfg.is_fixture else "real",
        "n_seeds": int(cfg.n_seeds),
        "seeds": list(range(cfg.n_seeds)),
        "features": cfg.features,
        "echo_coverage_window": cohort_prov["echo_coverage_window"],
        "gamma_min": summ["gamma_min"],
        "H_A1": summ["H_A1"],
        "propensity": summ["propensity"],
        "outcome_model": summ["outcome_model"],
        "intervals": summ["intervals"],
        "breakeven": summ["breakeven"],
        "per_seed_ci": summ["per_seed_ci"],
        "counts": {k: cohort_prov[k] for k in ("n_records_raw", "n_subjects_raw", "n_echo_studies",
                                                "n_echo_subjects", "n_analysed", "n_subjects_analysed",
                                                "n_T1", "n_T0", "selection_rate")}
        | {k: out_prov[k] for k in ("n_T1_labelled", "label_recovery_rate_T1", "prevalence_among_labelled_T1")},
        "cohort": cohort_prov,
        "decision_maker": {"levels": z_levels, "pi_z": pi_z.tolist(), "min_z_count": min_z,
                           "source_counts": cohort_prov["decision_maker_source_counts"],
                           "exclusion_restriction": "ordering unit Z is independent of (Y, S) given X; can fail "
                                                    "when sicker patients are routed to cardiology (limitations)"},
        "outcomes": out_prov,
        "icd_sensitivity": {"circularity": out_prov["route_c_icd"]["circularity"],
                            **summ["icd_sensitivity_per_seed"]},
        "covariates": cov.describe_blocks(blocks),
        "echonext_weights": echonext_prov,
        "config": cfg.as_dict(),
        "provenance_tables": root.loaded,
        "fixture_waveforms": (("synthetic in-memory arrays (fixture_waveforms), deterministic per study_id"
                               if cfg.is_fixture and cfg.use_ecg else None)),
        "runtime_seconds": float(time.time() - t_start),
    }
    if cfg.is_fixture:
        payload["WARNING"] = ("synthetic fixture run: no number in this file may be reported anywhere "
                              "(docs/mimic_ecg_echo_spec.md)")
    if cfg.is_fixture and inside_results_dir(cfg.out):      # belt and braces
        die(EXIT_SCHEMA, "refusing to write fixture output into results/")
    save_json(cfg.out, f"{NAME}_summary", payload)
    save_csv(cfg.out, f"{NAME}_seeds", rows)
    save_csv(cfg.out, f"{NAME}_gamma", grows)
    save_csv(cfg.out, "exp7_note_extraction_for_review", review)
    gm = summ["gamma_min"]["point"]
    print(f"\n[{NAME}] CONCLUSIONS ({data_source}; {cfg.n_seeds} seeds, 95% CI)")
    print(f"  Gamma_min = {gm['mean']:.3f} +- {gm['ci95']:.3f}  ->  H_A1 (in [1.5, 3.0]): {summ['H_A1']['holds']}")
    oa = summ["outcome_model"]["observed_auroc"]
    print(f"  observed AUROC ({summ['outcome_model']['kind']}) = {oa['mean']:.3f} +- {oa['ci95']:.3f}")
    for tag, d in summ["intervals"].items():
        print(f"  Gamma {tag:>9}: [L, U] = [{d['lower']['mean']:.3f}, {d['upper']['mean']:.3f}] "
              f"(width {d['width']['mean']:.3f} +- {d['width']['ci95']:.3f})")
    be = summ["breakeven"]
    print(f"  break-even Gamma*: L>=0.5 at {be['auroc_50']['gamma_star']['mean']:.2f}, "
          f"L>=0.7 at {be['auroc_70']['gamma_star']['mean']:.2f}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
