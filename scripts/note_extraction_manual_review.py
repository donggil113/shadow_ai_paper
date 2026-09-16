#!/usr/bin/env python3
"""Manual validation of the discharge-note SHD extraction (route (b)).

Two modes.

**Sample** (default)::

    python3 scripts/note_extraction_manual_review.py --data-root /data \\
        --n 200 --seed 0 --out audit/note_review_sample.csv [--include-text]

reads ``<data-root>/mimic-iv-note/2.2/note/discharge.csv.gz`` (missing file
=> exit 2 with the path and the PhysioNet page), runs the regex extractor on
a random pool of notes (``--max-notes``), and writes ``--n`` notes stratified
by the *extracted* label (positive / negative / missing, equal allocation,
shortfalls redistributed) with the extracted values, evidence snippets and
**blank truth columns**:

    truth_lvef          numeric LVEF in % as documented; "NA" if the note
                        documents no LVEF; blank = not reviewed for this field
    truth_wall_cm       max septal/posterior wall thickness in cm; "NA"/blank as above
    truth_valve_modsev  1 if any moderate/severe AS/AR/MS/MR/TR/PR, else 0; blank = not reviewed
    truth_shd           1 / 0 by the composite rule; "NA" if the note documents
                        nothing about the echo; blank = not reviewed
    reviewer_comment    free text

Note text is *not* written unless ``--include-text`` is given (the reviewer
can look notes up by ``note_id``).  A file with note text is governed by the
PhysioNet DUA: keep it out of version control.

**Score**::

    python3 scripts/note_extraction_manual_review.py --score audit/note_review_sample.csv \\
        [--out results/note_extraction_validation.json]

computes, over the rows whose truth columns are filled, precision / recall /
F1 (with Wilson 95% CIs on precision and recall and a bootstrap 95% CI on F1)
for the binary fields ``lvef_le_45``, ``wall_ge_1p3``, ``valve_modsev`` and
``shd``, plus numeric agreement for LVEF and wall thickness, and writes the
JSON with ``data_source: "real"``.  Recall is computed two ways: with a
missing extraction counted as a false negative (``recall``) and restricted to
notes where the extractor produced a value (``recall_documented``).

Guard: a sample whose ``data_source`` column is not ``real`` (e.g. the pytest
fixture) can never be scored into ``results/``; the script exits 3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dcl.data.mimic_ecg_echo.notes_regex import (LVEF_LOW_THRESHOLD,  # noqa: E402
                                                 WALL_THICK_THRESHOLD_CM, extract_note)

NOTES_REL = os.path.join("mimic-iv-note", "2.2", "note", "discharge.csv.gz")
PHYSIONET_PAGE = "https://physionet.org/content/mimic-iv-note/2.2/"
NOTE_COLUMNS = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq",
                "charttime", "storetime", "text"]
TRUTH_COLUMNS = ["truth_lvef", "truth_wall_cm", "truth_valve_modsev", "truth_shd", "reviewer_comment"]
STRATA = ["positive", "negative", "missing"]
FIXTURE_SUBJECT_MIN = 90_000_000


# --------------------------------------------------------------------------- sample
def _load_notes(path: str, max_notes: Optional[int], seed: int) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[note_extraction_manual_review] required file missing: {path}\n"
              f"  obtain MIMIC-IV-Note v2.2 (credentialed) from {PHYSIONET_PAGE}\n"
              f"  expected columns: {NOTE_COLUMNS}", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(path, compression="infer", dtype={"note_id": str})
    missing = [c for c in ("note_id", "subject_id", "hadm_id", "charttime", "text") if c not in df.columns]
    if missing:
        print(f"[note_extraction_manual_review] schema mismatch in {path}\n"
              f"  actual:   {list(df.columns)}\n  expected: {NOTE_COLUMNS} (MIMIC-IV-Note v2.2)\n"
              f"  missing:  {missing}", file=sys.stderr)
        sys.exit(2)
    if max_notes is not None and len(df) > max_notes:
        df = df.sample(n=max_notes, random_state=seed)
    return df.reset_index(drop=True)


def _stratum(shd: Optional[int]) -> str:
    return "missing" if shd is None else ("positive" if shd == 1 else "negative")


def _allocate(counts: Dict[str, int], n: int) -> Dict[str, int]:
    """Equal allocation over strata; shortfalls go to the strata with spare."""
    alloc = {s: 0 for s in STRATA}
    remaining = n
    open_strata = [s for s in STRATA if counts.get(s, 0) > 0]
    while remaining > 0 and open_strata:
        share = max(1, remaining // len(open_strata))
        progressed = False
        for s in list(open_strata):
            take = min(share, counts[s] - alloc[s], remaining)
            if take > 0:
                alloc[s] += take
                remaining -= take
                progressed = True
            if alloc[s] >= counts[s]:
                open_strata.remove(s)
            if remaining == 0:
                break
        if not progressed:
            break
    return alloc


def run_sample(args: argparse.Namespace) -> int:
    notes_path = args.notes_csv or os.path.join(args.data_root, NOTES_REL)
    data_source = "real" if args.notes_csv is None else "fixture"
    if args.notes_csv is not None and os.path.abspath(args.notes_csv).startswith(
            os.path.join(os.path.abspath(args.data_root), "")):
        data_source = "real"
    df = _load_notes(notes_path, args.max_notes, args.seed)
    if df["subject_id"].min() >= FIXTURE_SUBJECT_MIN:
        data_source = "fixture"
    rows: List[Dict[str, Any]] = []
    for r in df.itertuples(index=False):
        ex = extract_note(getattr(r, "text"))
        rows.append({
            "note_id": getattr(r, "note_id"), "subject_id": getattr(r, "subject_id"),
            "hadm_id": getattr(r, "hadm_id"), "charttime": getattr(r, "charttime"),
            "stratum": _stratum(ex["shd_composite"]),
            "extracted_shd": ex["shd_composite"],
            "extracted_lvef": ex["lvef"], "extracted_lvef_qualitative": ex["lvef_qualitative"],
            "extracted_wall_cm": ex["wall_thickness_cm"], "extracted_lvh_qualitative": ex["lvh_qualitative"],
            "extracted_valve_modsev": ex["components"]["valve_moderate_or_severe"],
            "extracted_valve_lesions": json.dumps([
                {"valve": v["valve"], "severity": v["severity"], "negated": v["negated"]}
                for v in ex["valve_lesions"]]),
            "confidence": ex["confidence"],
            "lvef_evidence": ex["lvef_evidence"], "wall_evidence": ex["wall_evidence"],
            "valve_evidence": " | ".join(ex["valve_evidence"]),
            "text": getattr(r, "text") if args.include_text else None,
        })
    ext = pd.DataFrame(rows)
    counts = ext["stratum"].value_counts().to_dict()
    alloc = _allocate(counts, args.n)
    rng = np.random.default_rng(args.seed)
    parts = []
    for s in STRATA:
        pool = ext[ext["stratum"] == s]
        if alloc[s] > 0:
            parts.append(pool.iloc[rng.permutation(len(pool))[: alloc[s]]])
    sample = pd.concat(parts) if parts else ext.iloc[0:0]
    sample = sample.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    for c in TRUTH_COLUMNS:
        sample[c] = ""
    sample["data_source"] = data_source
    sample["extractor"] = "regex"
    if not args.include_text:
        sample = sample.drop(columns=["text"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"[sample] pool={len(ext)} extracted-label counts={counts} allocation={alloc}")
    print(f"[sample] wrote {len(sample)} rows -> {args.out} (data_source={data_source})")
    if args.include_text:
        print("[sample] WARNING: the file contains note text; do not commit it.")
    print("[sample] fill truth_lvef / truth_wall_cm / truth_valve_modsev / truth_shd "
          "(NA = not documented, blank = not reviewed), then run --score")
    return 0


# --------------------------------------------------------------------------- score
def _wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (tp + fn) and (prec + rec) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "precision_ci95": _wilson(tp, tp + fp), "recall_ci95": _wilson(tp, tp + fn),
            "accuracy": (tp + tn) / len(y_true) if len(y_true) else float("nan")}


def _bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, reps: int = 1000, seed: int = 0) -> List[float]:
    if len(y_true) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y_true)
    for _ in range(reps):
        i = rng.integers(0, n, n)
        vals.append(_prf(y_true[i], y_pred[i])["f1"])
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _parse_truth(col: pd.Series) -> pd.Series:
    """blank -> NaN (not reviewed); 'NA' -> -1 sentinel (documented absent);
    otherwise float."""
    s = col.astype(str).str.strip()
    out = pd.Series(np.nan, index=col.index, dtype=float)
    na_mask = s.str.upper().isin(["NA", "N/A", "NONE", "NOT DOCUMENTED", "ND"])
    out[na_mask] = -1.0
    num = pd.to_numeric(s.where(~na_mask & (s != "") & (s.str.lower() != "nan")), errors="coerce")
    out[num.notna()] = num[num.notna()]
    return out


def _binary_field(truth: pd.Series, pred: pd.Series, rule) -> Dict[str, Any]:
    """Binary metrics with missing prediction counted as negative, plus the
    documented-only variant."""
    reviewed = truth.notna()
    t = truth[reviewed]
    p = pred[reviewed]
    y_true = np.where(t < 0, 0, rule(t)).astype(int)
    pred_missing = p.isna().to_numpy()
    y_pred = np.where(pred_missing, 0, rule(p.fillna(0))).astype(int)
    res = _prf(y_true, y_pred)
    res["f1_ci95_bootstrap"] = _bootstrap_f1(y_true, y_pred)
    res["n_reviewed"] = int(reviewed.sum())
    res["n_pred_missing"] = int(pred_missing.sum())
    doc = ~pred_missing
    res["recall_documented"] = _prf(y_true[doc], y_pred[doc])["recall"] if doc.any() else float("nan")
    res["n_truth_positive"] = int(y_true.sum())
    return res


def _numeric_agreement(truth: pd.Series, pred: pd.Series, tol: float) -> Dict[str, Any]:
    both = truth.notna() & (truth >= 0) & pred.notna()
    detect_truth = truth.notna() & (truth >= 0)
    d = (pred[both] - truth[both]).abs()
    return {
        "n_both_numeric": int(both.sum()),
        "n_truth_numeric": int(detect_truth.sum()),
        "detection_recall": float(both.sum() / detect_truth.sum()) if detect_truth.sum() else float("nan"),
        "mae": float(d.mean()) if both.any() else float("nan"),
        "within_tolerance": float((d <= tol).mean()) if both.any() else float("nan"),
        "tolerance": tol,
    }


def run_score(args: argparse.Namespace) -> int:
    path = args.score
    if not os.path.exists(path):
        print(f"[score] sample file missing: {path}", file=sys.stderr)
        return 2
    df = pd.read_csv(path, dtype={"note_id": str}, keep_default_na=False)
    missing = [c for c in TRUTH_COLUMNS + ["extracted_shd", "extracted_lvef", "extracted_wall_cm",
                                            "extracted_valve_modsev"] if c not in df.columns]
    if missing:
        print(f"[score] {path} lacks columns {missing}", file=sys.stderr)
        return 2
    data_source = str(df["data_source"].iloc[0]) if "data_source" in df.columns and len(df) else "unknown"
    out_path = os.path.abspath(args.out)
    if data_source != "real" and out_path.startswith(os.path.join(ROOT, "results") + os.sep):
        print(f"[score] refusing to write a non-real sample (data_source={data_source}) into results/",
              file=sys.stderr)
        return 3

    t_lvef = _parse_truth(df["truth_lvef"])
    t_wall = _parse_truth(df["truth_wall_cm"])
    t_valve = _parse_truth(df["truth_valve_modsev"])
    t_shd = _parse_truth(df["truth_shd"])
    p_lvef = pd.to_numeric(df["extracted_lvef"], errors="coerce")
    p_wall = pd.to_numeric(df["extracted_wall_cm"], errors="coerce")
    p_valve = pd.to_numeric(df["extracted_valve_modsev"].replace({"True": 1, "False": 0}), errors="coerce")
    p_shd = pd.to_numeric(df["extracted_shd"], errors="coerce")

    n_reviewed = int(t_shd.notna().sum())
    if n_reviewed == 0:
        print("[score] no filled truth_shd rows: nothing to score", file=sys.stderr)
        return 2
    fields = {
        "lvef_le_45": _binary_field(t_lvef, p_lvef, lambda v: (v <= LVEF_LOW_THRESHOLD)),
        "wall_ge_1p3": _binary_field(t_wall, p_wall, lambda v: (v >= WALL_THICK_THRESHOLD_CM)),
        "valve_modsev": _binary_field(t_valve, p_valve, lambda v: (v >= 0.5)),
        "shd": _binary_field(t_shd, p_shd, lambda v: (v >= 0.5)),
    }
    numeric = {
        "lvef_percent": _numeric_agreement(t_lvef, p_lvef, tol=5.0),
        "wall_cm": _numeric_agreement(t_wall, p_wall, tol=0.1),
    }
    by_stratum = {}
    if "stratum" in df.columns:
        for s, g in df.groupby("stratum"):
            ts, ps = _parse_truth(g["truth_shd"]), pd.to_numeric(g["extracted_shd"], errors="coerce")
            by_stratum[s] = {"n": int(len(g)), "n_reviewed": int(ts.notna().sum()),
                             "agreement": float(((ts >= 0.5).astype(int) == (ps.fillna(0) >= 0.5).astype(int))[ts.notna()].mean())
                             if ts.notna().any() else float("nan")}
    payload = {
        "data_source": data_source,
        "task": "manual validation of discharge-note SHD extraction (route b)",
        "sample_csv": os.path.relpath(path, ROOT) if path.startswith(ROOT) else path,
        "extractor": str(df["extractor"].iloc[0]) if "extractor" in df.columns else "regex",
        "n_sampled": int(len(df)), "n_reviewed": n_reviewed,
        "thresholds": {"lvef_le": LVEF_LOW_THRESHOLD, "wall_ge_cm": WALL_THICK_THRESHOLD_CM},
        "fields": fields, "numeric": numeric, "by_stratum": by_stratum,
        "notes": ("recall counts a missing extraction as a false negative; recall_documented "
                  "restricts to notes where the extractor returned a value. CIs: Wilson for "
                  "precision/recall, 1000-rep bootstrap for F1. Single reviewer pass; not a "
                  "seed-based comparison."),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[score] n_reviewed={n_reviewed} shd: P={fields['shd']['precision']:.3f} "
          f"R={fields['shd']['recall']:.3f} F1={fields['shd']['f1']:.3f} -> {out_path}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


# --------------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="/data")
    p.add_argument("--notes-csv", default=None,
                   help="override the discharge notes file (tests); anything outside --data-root "
                        "is marked data_source=fixture")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-notes", type=int, default=20000,
                   help="size of the random pool that is extracted before stratified sampling")
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--score", default=None, metavar="FILLED_CSV",
                   help="score a reviewed sample instead of drawing one")
    p.add_argument("--out", default=None,
                   help="sample: CSV path (default audit/note_review_sample.csv); "
                        "score: JSON path (default results/note_extraction_validation.json)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.score is not None:
        args.out = args.out or os.path.join(ROOT, "results", "note_extraction_validation.json")
        return run_score(args)
    args.out = args.out or os.path.join(ROOT, "audit", "note_review_sample.csv")
    return run_sample(args)


if __name__ == "__main__":
    sys.exit(main())
