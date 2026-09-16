"""CLI / end-to-end tests for experiments/exp7_mimic_ecg_echo.py on the fixture.

Every run writes into ``tmp_path``; nothing is ever written to results/
(the driver itself refuses ``--fixture-root`` with ``--out`` inside results/).
"""
from __future__ import annotations

import filecmp
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(REPO, "experiments", "exp7_mimic_ecg_echo.py")
FIXTURE = os.path.join(REPO, "tests", "fixtures", "mimic_ecg_echo")
RESULTS = os.path.join(REPO, "results")
GAMMA_KEYS = ["1", "1.25", "1.5", "2", "2.5", "3", "4", "6"]


def _run(*args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, DRIVER, *args], cwd=REPO, capture_output=True,
                          text=True, timeout=timeout)


def _load(path: str):
    name = "_exp7_test_" + os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def _results_listing():
    return sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []


# ------------------------------------------------------------ exit codes
def test_exit_2_when_data_root_missing(tmp_path):
    before = _results_listing()
    r = _run("--data-root", str(tmp_path / "no_such_root"), "--seeds", "1", "--out", str(tmp_path / "o"))
    assert r.returncode == 2, r.stderr
    assert "record_list.csv" in r.stderr and "physionet.org" in r.stderr
    assert "simulator" in r.stderr.lower()
    assert not (tmp_path / "o").exists()
    assert _results_listing() == before


def test_exit_3_when_fixture_output_would_land_in_results():
    before = _results_listing()
    r = _run("--fixture-root", FIXTURE, "--seeds", "1", "--out", os.path.join("results", "exp7_fixture_test"))
    assert r.returncode == 3, r.stderr
    assert "results/" in r.stderr and "fixture" in r.stderr
    assert not os.path.exists(os.path.join(RESULTS, "exp7_fixture_test"))
    assert _results_listing() == before
    # a nested 'results' component is refused too, and so are both roots at once
    r = _run("--fixture-root", FIXTURE, "--seeds", "1", "--out", os.path.join("/tmp", "x", "results", "y"))
    assert r.returncode == 3
    r = _run("--fixture-root", FIXTURE, "--data-root", "/data", "--seeds", "1", "--out", "/tmp/x")
    assert r.returncode == 3 and "mutually exclusive" in r.stderr


def test_exit_3_on_argument_violations(tmp_path):
    out = str(tmp_path / "o")
    assert _run("--fixture-root", FIXTURE, "--features", "waveform", "--out", out).returncode == 3
    assert _run("--fixture-root", FIXTURE, "--gamma-grid", "0.5,2", "--out", out).returncode == 3
    assert _run("--fixture-root", FIXTURE, "--ecg-layers", "1,1", "--out", out).returncode == 3
    assert _run("--fixture-root", FIXTURE, "--seeds", "0", "--out", out).returncode == 3
    assert _run("--fixture-root", FIXTURE, "--features", "ecg", "--echonext-weights",
                str(tmp_path / "missing.pt"), "--out", out).returncode == 2       # required input missing
    assert not (tmp_path / "o").exists()


# ----------------------------------------------------------- end-to-end
def _check_common(d: dict, n_seeds: int, features: str) -> None:
    assert d["data_source"] == "synthetic" and d["mode"] == "fixture" and "WARNING" in d
    assert d["n_seeds"] == n_seeds and d["seeds"] == list(range(n_seeds)) and d["features"] == features
    assert d["echo_coverage_window"] == ["2017-03-01T00:00:00", "2019-10-31T00:00:00"]
    assert d["cohort"]["coverage_restriction"]["n_before"] == 36
    assert d["cohort"]["coverage_restriction"]["n_after"] == 33 and d["cohort"]["dedup"]["n_after"] == 30
    c = d["counts"]
    assert c["n_analysed"] == 30 and c["n_T1"] == 19 and c["n_T0"] == 11 and c["n_T1_labelled"] == 16
    gm = d["gamma_min"]
    assert gm["point"]["n"] == n_seeds and len(gm["per_seed"]) == n_seeds
    assert np.isfinite(gm["point"]["mean"]) and gm["point"]["mean"] >= 1.0
    for k in ("q95", "boot_lo"):
        assert gm[k]["mean"] >= 1.0
    h = d["H_A1"]
    assert h["statement"] == "Gamma_min in [1.5, 3.0]" and h["range"] == [1.5, 3.0]
    assert isinstance(h["holds"], bool) and len(h["holds_per_seed"]) == n_seeds
    assert h["holds"] == (1.5 <= h["gamma_min_mean"] <= 3.0)
    assert 0.0 <= h["fraction_of_seeds"] <= 1.0
    iv = d["intervals"]
    assert list(iv)[: len(GAMMA_KEYS)] == GAMMA_KEYS and "gamma_min" in iv
    prev_w = -1.0
    for k in GAMMA_KEYS:
        e = iv[k]
        assert e["gamma"] == float(k) and e["lower"]["n"] == n_seeds
        assert 0.0 <= e["lower"]["mean"] <= e["upper"]["mean"] <= 1.0
        assert e["width"]["mean"] >= prev_w - 1e-12          # wider with Gamma
        prev_w = e["width"]["mean"]
        for kk in ("wc_risk_incumbent", "wc_risk_dcl", "regret_incumbent", "regret_dcl"):
            assert np.isfinite(e[kk]["mean"])
    assert iv["1"]["width"]["mean"] == pytest.approx(0.0, abs=1e-9)   # Gamma = 1 is point identified
    p = d["propensity"]
    assert p["auroc"]["n"] == n_seeds and 0.0 <= p["ece"]["mean"] <= 1.0
    curve = p["calibration_curve_10bins"]
    assert len(curve["bin_edges"]) == 11 and len(curve["n_total"]) == 10 and len(curve["mean_pred"]) == 10
    assert sum(curve["n_total"]) > 0
    be = d["breakeven"]
    for k in ("auroc_50", "auroc_60", "auroc_70"):
        assert be[k]["gamma_star"]["mean"] >= 1.0
    assert be["auroc_70"]["gamma_star"]["mean"] <= be["auroc_50"]["gamma_star"]["mean"] + 1e-9
    o = d["outcomes"]
    assert o["primary_outcome_route"] == "b_notes" and o["route_a_echo_struct"]["available"] is False
    assert o["route_b_notes"]["n_T1_with_note_label"] == 16
    circ = d["icd_sensitivity"]["circularity"]["icd_rate_by_T"]
    assert circ["T1"]["n"] > 0 and circ["T0"]["n"] > 0 and "risk_ratio" in circ
    assert d["decision_maker"]["min_z_count"] == 2 and len(d["decision_maker"]["levels"]) >= 2
    assert all(v["source"] == "fixture" for v in d["provenance_tables"].values())
    assert d["config"]["threads"] == 1


def test_fixture_run_tabular(tmp_path):
    before = _results_listing()
    out = tmp_path / "exp7"
    r = _run("--fixture-root", FIXTURE, "--features", "tabular", "--seeds", "2", "--max-records", "60",
             "--out", str(out))
    assert r.returncode == 0, r.stderr[-4000:]
    assert _results_listing() == before
    d = json.load(open(out / "exp7_mimic_ecg_echo_summary.json"))
    _check_common(d, n_seeds=2, features="tabular")
    assert d["outcome_model"]["kind"] == "gbm_tabular" and d["outcome_model"]["observed_auroc"]["n"] == 2
    assert d["fixture_waveforms"] is None
    assert set(d["covariates"]) == {"demographics", "labs", "setting", "ecg_tabular"}
    seeds = pd.read_csv(out / "exp7_mimic_ecg_echo_seeds.csv")
    assert len(seeds) == 2 and seeds["seed"].tolist() == [0, 1]
    assert (seeds["n_train"] + seeds["n_test"] == 30).all()
    assert seeds["gamma_min"].notna().all() and seeds["H_A1_point"].dtype == bool
    gam = pd.read_csv(out / "exp7_mimic_ecg_echo_gamma.csv")
    assert len(gam) == 2 * (len(GAMMA_KEYS) + 1) and (gam["lower"] <= gam["upper"] + 1e-12).all()
    rev = pd.read_csv(out / "exp7_note_extraction_for_review.csv")
    assert {"truth_lvef", "truth_wall_cm", "truth_valve_modsev", "truth_shd", "reviewer_comment"} <= set(rev.columns)
    assert (rev["data_source"] == "synthetic").all() and rev["note_n_notes"].min() >= 1
    assert "CONCLUSIONS (synthetic" in r.stdout


def test_fixture_run_ecg_tiny_encoder(tmp_path):
    before = _results_listing()
    out = tmp_path / "exp7ecg"
    r = _run("--fixture-root", FIXTURE, "--features", "ecg", "--seeds", "2", "--max-records", "60",
             "--n-boot", "10", "--ecg-base-width", "4", "--ecg-layers", "1,1,1,1", "--ecg-epochs", "2",
             "--ecg-batch-size", "8", "--out", str(out))
    assert r.returncode == 0, r.stderr[-4000:]
    assert _results_listing() == before
    assert "synthesising 30 waveforms in memory" in r.stdout
    d = json.load(open(out / "exp7_mimic_ecg_echo_summary.json"))
    _check_common(d, n_seeds=2, features="ecg")
    assert d["outcome_model"]["kind"] == "ecg_encoder_trained"
    assert d["fixture_waveforms"] is not None and d["echonext_weights"] is None
    assert d["config"]["ecg_base_width"] == 4 and d["config"]["ecg_layers"] == [1, 1, 1, 1]
    seeds = pd.read_csv(out / "exp7_mimic_ecg_echo_seeds.csv")
    assert (seeds["outcome_model"] == "ecg_encoder_trained").all()
    assert seeds["outcome_best_epoch"].between(0, 1).all()
    assert (seeds["n_features"] == 13 + 256).all()      # tabular blocks (no machine features) + embedding


# --------------------------------------------------------- driver helpers
def test_driver_helpers_and_fixture_waveforms():
    mod = _load(DRIVER)
    w = mod.fixture_waveforms([40_000_001, 40_000_002], n_samples=500)
    assert w.shape == (2, 12, 500) and w.dtype == np.float32
    assert np.array_equal(w[0], mod.fixture_waveforms([40_000_001], n_samples=500)[0])   # deterministic
    assert not np.array_equal(w[0], w[1])
    assert mod.inside_results_dir(os.path.join(REPO, "results")) and mod.inside_results_dir("results/x")
    assert mod.inside_results_dir("/tmp/a/results/b") and not mod.inside_results_dir("/tmp/a/res/b")
    cal = mod.calibration_10bins(np.array([0, 1, 1, 0.0]), np.array([0.05, 0.95, 0.85, 0.15]))
    assert len(cal["n"]) == 10 and sum(cal["n"]) == 4 and 0.0 <= cal["ece"] <= 1.0
    e_z = np.array([[0.2, 0.6], [0.5, 0.5]])
    p1_z = np.array([[0.1, 0.3], [0.4, 0.4]])
    e, p1 = mod.marginalise_over_z(e_z, p1_z, np.array([0.5, 0.5]))
    assert e[0] == pytest.approx(0.4) and p1[0] == pytest.approx((0.2 * 0.1 + 0.6 * 0.3) / 0.8)
    assert p1[1] == pytest.approx(0.4)
    idx, levels, pi = mod.merge_rare_z(pd.Series(["a", "a", "a", "b", "c"]), 2)
    assert levels == ["a", "other"] and pi.tolist() == [0.6, 0.4] and idx.tolist() == [0, 0, 0, 1, 1]


def test_extra_fixture_tables_are_deterministic_and_checked_in(tmp_path):
    gen = _load(os.path.join(REPO, "scripts", "make_mimic_fixture.py"))
    base = gen.build_tables()
    extra = gen.build_extra_tables(base)
    assert set(extra) == {"machine_measurements.csv", "hosp/d_labitems.csv.gz", "hosp/labevents.csv.gz",
                          "hosp/services.csv.gz", "hosp/diagnoses_icd.csv.gz", "ed/edstays.csv.gz",
                          "note/discharge.csv.gz", "records_w_diag_icd10.csv"}
    for name, df in extra.items():
        assert len(df) <= 50, name
        if "subject_id" in df:
            assert (pd.to_numeric(df["subject_id"]) >= 90_000_000).all(), name
    assert len(base["hosp/admissions.csv.gz"]) == 19
    gen.write_tables(gen.build_all_tables(), str(tmp_path))
    for name in list(base) + list(extra):
        assert filecmp.cmp(os.path.join(FIXTURE, name), os.path.join(str(tmp_path), name), shallow=False), \
            f"{name}: regenerate with scripts/make_mimic_fixture.py"
