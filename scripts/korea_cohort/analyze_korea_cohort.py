#!/usr/bin/env python3
"""Analyse the prepared Korean health-screening cohort (T7): the medical COMPAS.

Two analyses (details in scripts/korea_cohort/README.md):

(i)  Containment.  A model developed on public / US data carries an identified
     AUROC interval [L, U] on the US population under DCSM(Gamma) (the sharp
     interval of ``dcl.auc_bounds``).  On the Korean truth-batch cohort every
     unit's outcome is recorded, so the realised AUROC of ``public_model_score``
     is point-identified.  Report: realised AUROC with a bootstrap 95% CI
     (n_seeds x n_boot resamples), the containment flag, and the smallest Gamma
     at which the US interval contains it (recomputed with ``outcome_bounds`` +
     ``sharp_auc_interval`` when the US JSON carries per-unit nuisances, else
     from the supplied endpoints only).
(ii) Gamma comparison.  On the subset where the downstream decision
     ``T = followup_test_ordered`` varies: Gamma_min from leniency variation
     across ``centre_id`` (``dcl.falsify.falsification_curve``, the routine of
     ``experiments/exp4_gamma.py``) and, because the outcome is recorded for
     T = 0 units too, the realised odds ratio odds(P(Y=1|T=0)) / odds(P(Y=1|T=1))
     -- the COMPAS "detained / released" anchor -- both compared with the US
     values passed via ``--us-gamma-min`` / ``--us-odds-ratio``.

Outputs are aggregates only and every count obeys the ``>= min_cell`` rule.
``data_source`` is set from the ``--fixture`` flag ("synthetic") or its absence
("real") and cross-checked against the prepared file's sidecar; it is never
guessed from the data.  ``--write-pending`` writes the IRB placeholder
(``data_source: "pending"``, analysis keys null) and touches no data.

Exit codes: 0 ok · 2 missing input, or a US-bounds file without the required
keys · 3 provenance mismatch / malformed prepared file · 4 refused to write
synthetic output into results/.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple


def _pin_threads(argv: List[str]) -> None:
    """BLAS / OpenMP threads (default 1).  The per-centre nuisance fits are
    tiny and become many times slower under thread oversubscription; pass
    ``--threads N`` to raise it for the real cohort.  Must run before numpy /
    sklearn are imported; explicit environment settings are respected."""
    n = "1"
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif a.startswith("--threads="):
            n = a.split("=", 1)[1]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, n)


_pin_threads(sys.argv)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from prepare_korea_cohort import (MIN_CELL, RESULTS_DIR, SCORE, UNIT_COLUMNS,  # noqa: E402
                                  cell, partition, partition_2d, rate,
                                  read_prepared, sha256_file, sidecar_path)

import dcl  # noqa: E402
from dcl.auc_bounds import sharp_auc_interval  # noqa: E402
from dcl.falsify import falsification_curve  # noqa: E402
from dcl.nuisance import CrossFitNuisance  # noqa: E402
from dcl.sensitivity import outcome_bounds  # noqa: E402

DEFAULT_OUT = os.path.join(RESULTS_DIR, "korea_cohort_analysis.json")
DEFAULT_PENDING = os.path.join(RESULTS_DIR, "korea_cohort_pending.json")
EXIT_MISSING = 2
EXIT_PROVENANCE = 3
EXIT_REFUSED = 4
GAMMA_MAX = 1e3
MAR_GAMMA = 1.05          # "refutes MAR" threshold used by experiments/exp4_gamma.py
US_BOUNDS_FORMAT = """expected US-bounds JSON (see scripts/korea_cohort/README.md):
  {
    "data_source": "real",                       # provenance of the US result
    "gamma": 1.885, "auroc_lo": 0.63, "auroc_hi": 0.73,   # the interval [L, U] at gamma
    "by_gamma": {"1": {"auroc_lo": .., "auroc_hi": ..}, "1.5": {...}},   # optional table
    "nuisances": {"scores": [...], "p1": [...], "e": [...]}              # optional, per US test unit
  }
Use --us-gamma to pick an entry of "by_gamma"."""


def _die(msg: str, code: int) -> None:
    print(msg, file=sys.stderr, flush=True)
    sys.exit(code)


# ------------------------------------------------------------- statistics
def ci95(values) -> Dict[str, Any]:
    """Mean and t-based 95% half-width over seed values (experiments/_common.ci95)."""
    from scipy import stats
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    n = int(v.size)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    if n == 1:
        return {"mean": float(v[0]), "ci95": None, "n": 1, "min": float(v[0]), "max": float(v[0])}
    hw = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return {"mean": float(v.mean()), "ci95": hw, "n": n, "min": float(v.min()), "max": float(v.max())}


def bootstrap(fn: Callable[[np.ndarray], float], n: int, n_seeds: int, n_boot: int,
              base_seed: int = 0) -> Tuple[List[Dict[str, Any]], Optional[List[float]], int]:
    """Unit bootstrap: ``fn(idx)`` on ``n_seeds x n_boot`` resamples.

    Returns per-seed summaries (mean and percentile interval), the pooled
    percentile 95% interval and the number of finite resamples."""
    per_seed, pooled = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        vals = []
        for _ in range(n_boot):
            v = fn(rng.integers(0, n, n))
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        vals = np.asarray(vals, float)
        if vals.size:
            per_seed.append({"seed": s, "mean": float(vals.mean()),
                             "lo": float(np.quantile(vals, 0.025)),
                             "hi": float(np.quantile(vals, 0.975)), "n_boot": int(vals.size)})
            pooled.append(vals)
    pooled_arr = np.concatenate(pooled) if pooled else np.array([])
    ci = ([float(np.quantile(pooled_arr, 0.025)), float(np.quantile(pooled_arr, 0.975))]
          if pooled_arr.size else None)
    return per_seed, ci, int(pooled_arr.size)


# --------------------------------------------------------- (i) containment
def load_us_bounds(path: str, us_gamma: Optional[float]) -> Dict[str, Any]:
    """Read the US interval; exit 2 when the file or the required keys are missing."""
    if not os.path.isfile(path):
        _die(f"[korea_cohort] US-bounds file not found: {path}\n{US_BOUNDS_FORMAT}", EXIT_MISSING)
    try:
        with open(path) as f:
            d = json.load(f)
    except json.JSONDecodeError as exc:
        _die(f"[korea_cohort] US-bounds file is not valid JSON: {path} ({exc})", EXIT_MISSING)
    if not isinstance(d, dict):
        _die(f"[korea_cohort] US-bounds file must contain a JSON object: {path}\n{US_BOUNDS_FORMAT}", EXIT_MISSING)

    table: List[Tuple[float, float, float]] = []
    if isinstance(d.get("by_gamma"), dict):
        for k, v in d["by_gamma"].items():
            if isinstance(v, dict) and "auroc_lo" in v and "auroc_hi" in v:
                try:
                    table.append((float(k), float(v["auroc_lo"]), float(v["auroc_hi"])))
                except (TypeError, ValueError):
                    continue
        table.sort()
    top = all(k in d for k in ("gamma", "auroc_lo", "auroc_hi"))

    chosen: Optional[Tuple[float, float, float]] = None
    if us_gamma is not None:
        cands = [row for row in table if abs(row[0] - us_gamma) < 1e-6]
        if cands:
            chosen = cands[0]
        elif top and abs(float(d["gamma"]) - us_gamma) < 1e-6:
            chosen = (float(d["gamma"]), float(d["auroc_lo"]), float(d["auroc_hi"]))
        else:
            _die(f"[korea_cohort] --us-gamma {us_gamma:g} not found in {path} "
                 f"(available: {[r[0] for r in table] + ([float(d['gamma'])] if top else [])})", EXIT_MISSING)
    elif top:
        chosen = (float(d["gamma"]), float(d["auroc_lo"]), float(d["auroc_hi"]))
    elif len(table) == 1:
        chosen = table[0]
    elif len(table) > 1:
        _die(f"[korea_cohort] {path} has a by_gamma table but no top-level gamma; pass --us-gamma "
             f"(available: {[r[0] for r in table]})", EXIT_MISSING)
    if chosen is None:
        _die(f"[korea_cohort] US-bounds file lacks the required keys auroc_lo / auroc_hi at a gamma: {path}\n"
             f"{US_BOUNDS_FORMAT}", EXIT_MISSING)
    g, lo, hi = chosen
    if not (g >= 1.0 and 0.0 <= lo <= hi <= 1.0):
        _die(f"[korea_cohort] malformed US interval in {path}: gamma={g}, [L, U]=[{lo}, {hi}]", EXIT_MISSING)

    nuis = None
    if isinstance(d.get("nuisances"), dict):
        nd = d["nuisances"]
        try:
            scores = np.asarray(nd["scores"], float)
            p1 = np.asarray(nd["p1"], float)
            e = np.asarray(nd["e"], float)
        except (KeyError, TypeError, ValueError):
            _die(f"[korea_cohort] 'nuisances' in {path} must carry numeric arrays scores, p1, e", EXIT_MISSING)
        if not (scores.shape == p1.shape == e.shape and scores.ndim == 1 and scores.size >= 2):
            _die(f"[korea_cohort] 'nuisances' arrays in {path} must be 1-D of equal length >= 2", EXIT_MISSING)
        if np.any(~np.isfinite(scores)) or np.any((p1 < 0) | (p1 > 1)) or np.any((e < 0) | (e > 1)):
            _die(f"[korea_cohort] 'nuisances' in {path}: scores must be finite and p1, e in [0, 1]", EXIT_MISSING)
        nuis = (scores, p1, e)
    return {"path": os.path.abspath(path), "sha256": sha256_file(path),
            "data_source": d.get("data_source"), "model": d.get("model"),
            "population": d.get("population"), "gamma": g, "auroc_lo": lo, "auroc_hi": hi,
            "table": table, "nuisances": nuis, "n_us_units": int(nuis[0].size) if nuis else None}


def realised_auroc(y: np.ndarray, score: np.ndarray, n_seeds: int, n_boot: int,
                   min_cell: int) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(y) & np.isfinite(score)
    y, s = y[m], score[m]
    n, k = int(m.sum()), int(y.sum())
    counts = {"n_units": cell(n, min_cell),
              "events": partition({"n_events": k, "n_non_events": n - k}, min_cell)}
    if n < min_cell or k == 0 or k == n:
        return {"available": False, "point": None, "ci95": None,
                "reason": "too few scored units with a recorded outcome, or a single outcome class",
                **counts}
    point = float(roc_auc_score(y, s))

    def fn(idx):
        yy = y[idx]
        return roc_auc_score(yy, s[idx]) if yy.min() != yy.max() else np.nan

    per_seed, ci, n_res = bootstrap(fn, n, n_seeds, n_boot)
    return {"available": True, "point": point, "ci95": ci,
            "ci_method": "percentile bootstrap over units; n_seeds x n_boot resamples pooled",
            "n_resamples": n_res, "per_seed": per_seed,
            "over_seeds": ci95([p["mean"] for p in per_seed]), **counts}


def smallest_gamma_containing(auroc: float, us: Dict[str, Any], gamma_max: float = GAMMA_MAX
                              ) -> Dict[str, Any]:
    """Smallest Gamma at which the US interval contains the realised AUROC."""
    if us["nuisances"] is not None:
        scores, p1, e = us["nuisances"]

        def interval(g: float):
            lo, hi = outcome_bounds(p1, e, g)
            return sharp_auc_interval(scores, lo, hi)

        at_stated = interval(us["gamma"])
        base = {"method": "recomputed_from_nuisances", "gamma_max": gamma_max,
                "n_us_units": int(scores.size),
                "recomputed_interval_at_stated_gamma": [at_stated.lower, at_stated.upper],
                "stated_interval": [us["auroc_lo"], us["auroc_hi"]],
                "recomputed_matches_stated": bool(abs(at_stated.lower - us["auroc_lo"]) < 5e-3
                                                  and abs(at_stated.upper - us["auroc_hi"]) < 5e-3)}
        if interval(1.0).contains(auroc):
            r = interval(1.0)
            return {"gamma": 1.0, "interval_at_gamma": [r.lower, r.upper], **base}
        if not interval(gamma_max).contains(auroc):
            return {"gamma": None, "contained_at_gamma_max": False, **base}
        lo_g, hi_g = 1.0, gamma_max
        for _ in range(50):                      # nested intervals -> monotone -> bisection
            mid = float(np.sqrt(lo_g * hi_g))
            if interval(mid).contains(auroc):
                hi_g = mid
            else:
                lo_g = mid
        r = interval(hi_g)
        return {"gamma": float(hi_g), "interval_at_gamma": [r.lower, r.upper],
                "contained_at_gamma_max": True, **base}
    if len(us["table"]) > 1:
        for g, lo, hi in us["table"]:
            if lo - 1e-9 <= auroc <= hi + 1e-9:
                return {"gamma": float(g), "method": "table_lookup", "interval_at_gamma": [lo, hi],
                        "gammas_available": [r[0] for r in us["table"]]}
        return {"gamma": None, "method": "table_lookup", "gammas_available": [r[0] for r in us["table"]],
                "note": "no tabulated Gamma contains the realised AUROC"}
    contained = us["auroc_lo"] - 1e-9 <= auroc <= us["auroc_hi"] + 1e-9
    return {"gamma": float(us["gamma"]) if contained else None, "method": "single_endpoint",
            "note": "only one interval supplied; the smallest containing Gamma is at most the stated one"
                    if contained else "the supplied interval does not contain the realised AUROC"}


def containment_analysis(df: pd.DataFrame, us: Dict[str, Any], n_seeds: int, n_boot: int,
                         min_cell: int) -> Dict[str, Any]:
    y = df["Y_full"].to_numpy(float)
    t = df["T"].to_numpy(int)
    score = pd.to_numeric(df[SCORE], errors="coerce").to_numpy(float)
    out: Dict[str, Any] = {
        "batch_cohort_outcome_observed_all": bool(np.isfinite(y).all()),
        "T_all_one": bool((t == 1).all()),
        "us_bounds": {k: us[k] for k in ("path", "sha256", "data_source", "model", "population",
                                          "gamma", "auroc_lo", "auroc_hi", "n_us_units")},
        "us_interval_width": us["auroc_hi"] - us["auroc_lo"],
    }
    if not np.isfinite(score).any():
        out.update({"available": False, "reason": f"no {SCORE} in the prepared file",
                    "realised_auroc": None, "contains_point": None, "contains_ci": None,
                    "overlaps_ci": None, "min_gamma_containing": None})
        return out
    ra = realised_auroc(y, score, n_seeds, n_boot, min_cell)
    out["realised_auroc"] = ra
    if not ra["available"]:
        out.update({"available": False, "reason": ra["reason"], "contains_point": None,
                    "contains_ci": None, "overlaps_ci": None, "min_gamma_containing": None})
        return out
    lo, hi, a = us["auroc_lo"], us["auroc_hi"], ra["point"]
    ci_lo, ci_hi = ra["ci95"] if ra["ci95"] else (a, a)
    out.update({
        "available": True,
        "contains_point": bool(lo - 1e-9 <= a <= hi + 1e-9),
        "contains_ci": bool(lo - 1e-9 <= ci_lo and ci_hi <= hi + 1e-9),
        "overlaps_ci": bool(ci_hi >= lo - 1e-9 and ci_lo <= hi + 1e-9),
        "realised_minus_us_lo": a - lo, "realised_minus_us_hi": a - hi,
        "min_gamma_containing": smallest_gamma_containing(a, us),
        "note": ("the realised AUROC is the deployment AUROC of the public model on the Korean cohort "
                 "(outcome recorded for every unit)" if out["batch_cohort_outcome_observed_all"] else
                 "outcome NOT recorded for every unit: the realised AUROC is an observed-units AUROC, "
                 "not the deployment AUROC"),
    })
    return out


# ---------------------------------------------------- (ii) Gamma comparison
def realised_odds_ratio(y: np.ndarray, t: np.ndarray, n_seeds: int, n_boot: int,
                        min_cell: int) -> Dict[str, Any]:
    m = np.isfinite(y)
    y, t = y[m], t[m]
    n = int(m.sum())
    tab = {f"T{tv}": {f"Y{yv}": int(((t == tv) & (y == yv)).sum()) for yv in (0, 1)} for tv in (0, 1)}
    counts = partition_2d(tab, min_cell)
    n0, n1 = int((t == 0).sum()), int((t == 1).sum())
    k0, k1 = int(y[t == 0].sum()) if n0 else 0, int(y[t == 1].sum()) if n1 else 0
    base = {"definition": "odds(P(Y=1 | T=0)) / odds(P(Y=1 | T=1)) on units with a recorded outcome "
                          "(COMPAS convention: censored group over selected group)",
            "counts_2x2": counts,
            "rate_Y_given_T0": rate(k0, n0, min_cell), "rate_Y_given_T1": rate(k1, n1, min_cell)}
    if min(n0, n1) < min_cell:
        return {"available": False, "point": None, "ci95": None,
                "reason": "outcome not recorded for enough T = 0 and T = 1 units", **base}
    if k0 in (0, n0) or k1 in (0, n1):
        return {"available": False, "point": None, "ci95": None,
                "reason": "a T group has a single outcome class (odds ratio undefined)", **base}

    def _or(yy, tt):
        s1, s0 = tt == 1, tt == 0
        if not (s1.any() and s0.any()):
            return np.nan
        a, b = yy[s1].mean(), yy[s0].mean()
        if a in (0.0, 1.0) or b in (0.0, 1.0):
            return np.nan
        return (b / (1 - b)) / (a / (1 - a))

    point = float(_or(y, t))
    per_seed, ci, n_res = bootstrap(lambda idx: _or(y[idx], t[idx]), n, n_seeds, n_boot)
    return {"available": True, "point": point, "log_point": float(np.log(point)), "ci95": ci,
            "gamma_scale_magnitude": float(max(point, 1.0 / point)),
            "direction": "censored-lower (untested units are lower risk)" if point < 1
            else "censored-higher (untested units are higher risk)",
            "ci_method": "percentile bootstrap over units; n_seeds x n_boot resamples pooled",
            "n_resamples": n_res, "per_seed": per_seed,
            "over_seeds": ci95([p["mean"] for p in per_seed]), **base}


def gamma_min_from_leniency(df: pd.DataFrame, features: List[str], n_seeds: int, n_boot: int,
                            n_folds: int, model: str, grid_size: int, min_cell: int,
                            calibrate: bool = True) -> Dict[str, Any]:
    X = df[features].to_numpy(float)
    T = df["T"].to_numpy(int)
    Y = df["Y_obs"].to_numpy(float)
    Zc = df["centre_id"].astype(str).to_numpy()
    centres = sorted(set(Zc.tolist()))
    per_centre: Dict[str, Any] = {}
    usable: List[str] = []
    for c in centres:
        m = Zc == c
        n_c, n_t1, n_t0 = int(m.sum()), int((T[m] == 1).sum()), int((T[m] == 0).sum())
        ysel = Y[m][T[m] == 1]
        n_pos, n_neg = int(np.nansum(ysel)), int((ysel == 0).sum())
        reason = None
        if n_c < max(min_cell, 2 * n_folds):
            reason = "too few units"
        elif min(n_t1, n_t0) < n_folds:
            reason = "decision does not vary enough within the centre"
        elif min(n_pos, n_neg) < n_folds:
            reason = "too few outcomes of each class among the centre's T = 1 units"
        per_centre[c] = {"n": cell(n_c, min_cell), "selection_rate": rate(n_t1, n_c, min_cell),
                         "used_for_gamma_min": reason is None, "excluded_because": reason}
        if reason is None:
            usable.append(c)
    base = {"per_centre": per_centre, "n_centres": len(centres), "n_centres_used": len(usable),
            "exclusion_restriction": "centre_id independent of (Y, S) given X -- assumed, not testable here",
            "nuisance_model": model, "n_folds": n_folds, "calibrated": bool(calibrate)}
    if len(usable) < 2:
        return {"available": False, "reason": "fewer than two centres with usable decision variation",
                **base}

    n = X.shape[0]
    grid_idx = (np.random.default_rng(0).choice(n, grid_size, replace=False) if n > grid_size
                else np.arange(n))
    X_grid = X[grid_idx]
    rows, failures = [], []
    for seed in range(n_seeds):
        p1_by_z, e_by_z = [], []
        for c in usable:
            m = Zc == c
            cf = CrossFitNuisance(model=model, n_folds=n_folds, calibrate=calibrate, seed=seed)
            try:
                cf.fit_predict(X[m], T[m], Y[m])
                est = cf.predict(X_grid)
            except ValueError as exc:                # a fold without both classes, etc.
                failures.append({"seed": seed, "centre": c, "error": str(exc)[:200]})
                continue
            p1_by_z.append(est.p1)
            e_by_z.append(est.e)
        if len(p1_by_z) < 2:
            failures.append({"seed": seed, "centre": None, "error": "fewer than two centres fitted"})
            continue
        r = falsification_curve(p1_by_z, e_by_z, n_boot=n_boot, seed=seed)
        rows.append({"seed": seed, "gamma_min": r.gamma_min, "gamma_min_q95": r.gamma_min_q95,
                     "gamma_min_boot_lo": r.gamma_min_boot_lo, "n_units_grid": r.n_units,
                     "n_groups": r.n_groups, "refutes_MAR": bool(r.refutes(MAR_GAMMA))})
    if not rows:
        return {"available": False, "reason": "nuisance fitting failed for every seed",
                "failures": failures, **base}
    return {"available": True,
            "gamma_min": ci95([r["gamma_min"] for r in rows]),
            "gamma_min_q95": ci95([r["gamma_min_q95"] for r in rows]),
            "gamma_min_boot_lo": ci95([r["gamma_min_boot_lo"] for r in rows]),
            "refutes_MAR_fraction": float(np.mean([r["refutes_MAR"] for r in rows])),
            "n_seeds_completed": len(rows), "n_units_grid": int(X_grid.shape[0]),
            "per_seed": rows, "failures": failures, "n_boot_falsification": n_boot, **base}


def gamma_comparison(df: pd.DataFrame, features: List[str], us_gamma_min: Optional[float],
                     us_odds_ratio: Optional[float], n_seeds: int, n_boot: int, n_folds: int,
                     model: str, grid_size: int, min_cell: int, calibrate: bool = True
                     ) -> Dict[str, Any]:
    t = df["T"].to_numpy(int)
    y = df["Y_full"].to_numpy(float)
    out: Dict[str, Any] = {
        "T_varies": bool(t.min() != t.max()),
        "n_T": partition({"T0": int((t == 0).sum()), "T1": int((t == 1).sum())}, min_cell),
        "selection_rate": rate(int((t == 1).sum()), int(t.size), min_cell),
        "outcome_recorded_for_T0": bool(np.isfinite(y[t == 0]).any()) if (t == 0).any() else False,
        "us_gamma_min": us_gamma_min, "us_odds_ratio": us_odds_ratio,
    }
    if not out["T_varies"]:
        out.update({"available": False,
                    "reason": "the decision does not vary (batch cohort: followup_test_ordered identical "
                              "for every unit), so neither Gamma_min nor the odds ratio is defined",
                    "gamma_min": None, "odds_ratio": None, "gamma_min_vs_us": None,
                    "odds_ratio_vs_us": None})
        return out
    gm = gamma_min_from_leniency(df, features, n_seeds, n_boot, n_folds, model, grid_size, min_cell,
                                 calibrate)
    orr = realised_odds_ratio(y, t, n_seeds, n_boot, min_cell)
    out.update({"available": bool(gm["available"] or orr["available"]), "gamma_min": gm, "odds_ratio": orr})

    def _vs(korea_point, korea_ci, us_value, label):
        if us_value is None or korea_point is None:
            return None
        d = {f"korea_{label}": korea_point, f"us_{label}": us_value,
             "korea_minus_us": korea_point - us_value,
             "korea_over_us": korea_point / us_value if us_value else None}
        if korea_ci is not None:
            d["korea_ci95_contains_us"] = bool(korea_ci[0] - 1e-9 <= us_value <= korea_ci[1] + 1e-9)
        return d

    if gm["available"]:
        g = gm["gamma_min_q95"]
        g_ci = ([g["mean"] - g["ci95"], g["mean"] + g["ci95"]]
                if g["ci95"] is not None and g["mean"] is not None else None)
        out["gamma_min_vs_us"] = _vs(g["mean"], g_ci, us_gamma_min, "gamma_min_q95")
    else:
        out["gamma_min_vs_us"] = None
    if orr["available"]:
        v = _vs(orr["point"], orr["ci95"], us_odds_ratio, "odds_ratio")
        if v is not None and us_odds_ratio:
            v["gamma_scale_magnitude_korea"] = orr["gamma_scale_magnitude"]
            v["gamma_scale_magnitude_us"] = float(max(us_odds_ratio, 1.0 / us_odds_ratio))
            v["same_direction"] = bool((orr["point"] < 1) == (us_odds_ratio < 1))
        out["odds_ratio_vs_us"] = v
    else:
        out["odds_ratio_vs_us"] = None
    return out


# ------------------------------------------------------------- output
def _clean(o: Any) -> Any:
    """JSON-safe: numpy scalars -> Python, non-finite floats -> null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _assert_aggregate(o: Any, max_len: int = 64, path: str = "$") -> None:
    """Refuse to write anything that looks unit-level (long arrays, subject ids)."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("subject_id", "subject_ids"):
                raise ValueError(f"unit-level key {k!r} at {path}")
            _assert_aggregate(v, max_len, f"{path}.{k}")
    elif isinstance(o, list):
        if len(o) > max_len:
            raise ValueError(f"array of length {len(o)} at {path} looks unit-level")
        for i, v in enumerate(o):
            _assert_aggregate(v, max_len, f"{path}[{i}]")


def write_json(payload: Dict[str, Any], out: str) -> None:
    payload = _clean(payload)
    _assert_aggregate(payload)
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> {out}")


def pending_payload() -> Dict[str, Any]:
    return {
        "data_source": "pending",
        "status": "pending IRB",
        "cohort": "Korean national health-screening batch cohort (T7)",
        "n_seeds": None,
        "containment": None,
        "gamma_comparison": None,
        "note": ("Placeholder written by scripts/korea_cohort/analyze_korea_cohort.py --write-pending; "
                 "no data were accessed. It lets the paper's macros resolve to a visible PENDING marker "
                 "(scripts/check_provenance.py refuses pending macros in a verified build). Replace it by "
                 "running the real path once IRB approval and data access are granted."),
        "written_by": "scripts/korea_cohort/analyze_korea_cohort.py --write-pending",
    }


# ---------------------------------------------------------------- main
def analyze(prepared: str, us_bounds: str, us_gamma: Optional[float], us_gamma_min: Optional[float],
            us_odds_ratio: Optional[float], fixture: bool, n_seeds: int, n_boot: int, n_folds: int,
            model: str, grid_size: int, min_cell: int, out: str, calibrate: bool = True,
            threads: int = 1) -> Dict[str, Any]:
    source = "synthetic" if fixture else "real"
    if not os.path.isfile(prepared):
        _die(f"[korea_cohort] prepared file not found: {prepared}\n  run prepare_korea_cohort.py first",
             EXIT_MISSING)
    if not os.path.isfile(sidecar_path(prepared)):
        _die(f"[korea_cohort] sidecar not found next to the prepared file: {sidecar_path(prepared)}",
             EXIT_MISSING)
    if source == "synthetic" and os.path.commonpath([os.path.abspath(out), RESULTS_DIR]) == RESULTS_DIR:
        _die(f"[korea_cohort] REFUSED: fixture-derived output may not be written under {RESULTS_DIR} "
             f"(CLAUDE.md rules 1 and 3). Exiting with code {EXIT_REFUSED}.", EXIT_REFUSED)
    try:
        df, side = read_prepared(prepared)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        _die(f"[korea_cohort] cannot read the prepared file / sidecar: {exc}", EXIT_PROVENANCE)
    if side.get("data_source") != source:
        _die(f"[korea_cohort] PROVENANCE MISMATCH: sidecar says data_source={side.get('data_source')!r} "
             f"but the flag implies {source!r} ({'--fixture given' if fixture else 'no --fixture'}). "
             f"Exiting with code {EXIT_PROVENANCE}.", EXIT_PROVENANCE)
    features = list(side.get("prepared", {}).get("features", []))
    missing = [c for c in UNIT_COLUMNS + features if c not in df.columns]
    if missing or not features:
        _die(f"[korea_cohort] malformed prepared file: missing columns {missing or 'features list'}",
             EXIT_PROVENANCE)
    us = load_us_bounds(us_bounds, us_gamma)

    if n_seeds < 5:
        print(f"[korea_cohort] WARNING: {n_seeds} seed(s) < 5; CLAUDE.md rule 4 forbids claiming a "
              "difference from this run", flush=True)
    print(f"[korea_cohort] data_source={source}  n={len(df)}  seeds={n_seeds}  n_boot={n_boot}", flush=True)
    cont = containment_analysis(df, us, n_seeds, n_boot, min_cell)
    print(f"[korea_cohort] containment: realised AUROC={cont.get('realised_auroc', {}) and cont['realised_auroc'].get('point')}"
          f"  US [L, U]=[{us['auroc_lo']:.4f}, {us['auroc_hi']:.4f}] at Gamma={us['gamma']:g}"
          f"  contains_point={cont.get('contains_point')}", flush=True)
    gcmp = gamma_comparison(df, features, us_gamma_min, us_odds_ratio, n_seeds, n_boot, n_folds,
                            model, grid_size, min_cell, calibrate)
    gm = gcmp.get("gamma_min") or {}
    orr = gcmp.get("odds_ratio") or {}
    print(f"[korea_cohort] Gamma_min(q95)={gm.get('gamma_min_q95', {}).get('mean') if gm.get('available') else None}"
          f"  odds ratio={orr.get('point') if orr.get('available') else None}", flush=True)

    payload: Dict[str, Any] = {
        "data_source": source,
        "status": "synthetic fixture (pytest only; never report)" if fixture else "real cohort",
        "n_seeds": int(n_seeds),
        "n_boot_per_seed": int(n_boot),
        "rule4_compliant": bool(n_seeds >= 5),
        "inputs": {
            "prepared": {"path": os.path.abspath(prepared), "sha256": sha256_file(prepared),
                         "sidecar_data_source": side.get("data_source"),
                         "input_sha256": side.get("input", {}).get("sha256"),
                         "schema_version": side.get("schema_version")},
            "us_bounds": {k: us[k] for k in ("path", "sha256", "data_source", "gamma", "auroc_lo", "auroc_hi")},
            "us_gamma_min": us_gamma_min, "us_odds_ratio": us_odds_ratio,
        },
        "cohort": {"counts": side.get("counts"), "summary": side.get("cohort"),
                   "per_centre": side.get("per_centre"), "rules": side.get("rules")},
        "containment": cont,
        "gamma_comparison": gcmp,
        "privacy": {"min_cell": int(min_cell),
                    "rule": "aggregates only; counts in (0, min_cell) are null with suppressed=true "
                            "(complementary suppression inside published tables); rates need denominator, "
                            "numerator and complement each 0 or >= min_cell"},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "dcl": dcl.__version__, "threads": int(threads),
                     "nuisance_model": model, "calibrated": bool(calibrate), "n_folds": int(n_folds)},
    }
    write_json(payload, out)
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", default=None, help="prepared file from prepare_korea_cohort.py "
                                                    "(its JSON sidecar must sit next to it)")
    ap.add_argument("--us-bounds", default=None, help="JSON with the US interval (auroc_lo, auroc_hi at gamma)")
    ap.add_argument("--us-gamma", type=float, default=None, help="entry of the US 'by_gamma' table to use")
    ap.add_argument("--us-gamma-min", type=float, default=None, help="US Gamma_min to compare with (optional)")
    ap.add_argument("--us-odds-ratio", type=float, default=None,
                    help="US realised odds ratio (e.g. COMPAS detained/released) to compare with (optional)")
    ap.add_argument("--fixture", action="store_true",
                    help="the prepared file came from the synthetic fixture: label data_source='synthetic'")
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (rule 4: >= 5)")
    ap.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples per seed")
    ap.add_argument("--n-folds", type=int, default=5, help="cross-fitting folds for the per-centre nuisances")
    ap.add_argument("--nuisance-model", default="gbm", choices=("gbm", "logistic", "mlp"))
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip isotonic calibration of the per-centre nuisances (small centres)")
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS / OpenMP threads (default 1; raise for the real cohort)")
    ap.add_argument("--grid-size", type=int, default=20000,
                    help="max units on the common grid for Gamma_min (subsampled deterministically)")
    ap.add_argument("--min-cell", type=int, default=MIN_CELL, help="smallest count that may be published")
    ap.add_argument("--out", default=None,
                    help=f"output JSON (default {DEFAULT_OUT}; {DEFAULT_PENDING} with --write-pending)")
    ap.add_argument("--write-pending", action="store_true",
                    help="write the IRB placeholder JSON (data_source 'pending', analysis keys null) and exit")
    a = ap.parse_args(argv)

    if a.write_pending:
        out = os.path.abspath(a.out or DEFAULT_PENDING)
        write_json(pending_payload(), out)
        print("[korea_cohort] wrote the pending-IRB placeholder; no data were accessed")
        return 0
    if not a.prepared or not a.us_bounds:
        ap.error("--prepared and --us-bounds are required (or use --write-pending)")
    if a.seeds < 1 or a.n_boot < 1:
        ap.error("--seeds and --n-boot must be >= 1")
    analyze(os.path.abspath(a.prepared), os.path.abspath(a.us_bounds), a.us_gamma, a.us_gamma_min,
            a.us_odds_ratio, a.fixture, a.seeds, a.n_boot, a.n_folds, a.nuisance_model, a.grid_size,
            a.min_cell, os.path.abspath(a.out or DEFAULT_OUT), calibrate=not a.no_calibrate,
            threads=a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
