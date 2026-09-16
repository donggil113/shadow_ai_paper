"""Guards that the paper's data constructions rely on (CLAUDE.md rules 1, 3, 5).

* the hidden decision-maker signal S (and deterministic derivatives) is never
  in the learner's X -- enforced in code, not by convention;
* COMPAS exposure-adjusted outcome is well formed and changes the odds ratio in
  the direction the incapacitation argument predicts;
* results files carry a data_source field.
"""
import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dcl.data.semisynthetic import (HiddenSignalLeak,  # noqa: E402
                                    assert_hidden_signal_excluded)

RAW = os.path.join(ROOT, "data", "raw")
HAVE_COMPAS = os.path.exists(os.path.join(RAW, "compas_two_years.csv"))
HAVE_LENDING = os.path.exists(os.path.join(RAW, "modeldata_lending_club.csv"))


def test_guard_rejects_name_leak():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)); S = rng.normal(size=200)
    with pytest.raises(HiddenSignalLeak):
        assert_hidden_signal_excluded(X, S, ["age", "int_rate_bucket", "inc"], forbidden_names=("int_rate",))


def test_guard_rejects_exact_and_monotone_copy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 3)); S = rng.normal(size=500)
    X_leak = np.column_stack([X, S])
    with pytest.raises(HiddenSignalLeak):
        assert_hidden_signal_excluded(X_leak, S, ["a", "b", "c", "d"])
    X_mono = np.column_stack([X, np.exp(S)])          # monotone transform
    with pytest.raises(HiddenSignalLeak):
        assert_hidden_signal_excluded(X_mono, S, ["a", "b", "c", "d"])


def test_guard_rejects_linear_reconstruction():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 4)); S = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 1e-3 * rng.normal(size=500)
    with pytest.raises(HiddenSignalLeak):
        assert_hidden_signal_excluded(X, S, list("abcd"))


def test_guard_accepts_dependent_but_not_derived():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 4)); S = 0.5 * X[:, 0] + rng.normal(size=500)
    d = assert_hidden_signal_excluded(X, S, list("abcd"))
    assert d["linear_r2"] < 0.5 and d["max_abs_pearson"] < 0.95


@pytest.mark.skipif(not HAVE_LENDING, reason="Lending Club raw file not downloaded")
def test_lending_hidden_signal_not_in_X():
    import pandas as pd
    from dcl.data.lending import FORBIDDEN_FEATURES, _NUM, load_lending_club_frame
    df = load_lending_club_frame(RAW)
    for f in FORBIDDEN_FEATURES:
        assert f not in _NUM
    # the raw hidden-signal columns exist in the file (so the guard is not vacuous)
    assert {"int_rate", "sub_grade"} <= set(df.columns)
    # the guard runs on the actual feature matrix the paper uses
    num = df[_NUM].apply(pd.to_numeric, errors="coerce").astype(float).fillna(0.0)
    X = num.to_numpy(); S = df["int_rate"].astype(float).to_numpy()
    d = assert_hidden_signal_excluded(X, S, list(num.columns), forbidden_names=FORBIDDEN_FEATURES)
    assert d["max_abs_pearson"] < 0.95 and d["linear_r2"] < 0.95


@pytest.mark.skipif(not HAVE_COMPAS, reason="COMPAS raw file not downloaded")
def test_compas_exposure_adjusted_outcome():
    from dcl.data import make_compas
    rec = make_compas(outcome="recorded")
    exp = make_compas(outcome="exposure_adjusted", exposure_days=730)
    assert exp.n < rec.n and exp.n > 0.7 * rec.n
    assert set(np.unique(exp.Y_full)) <= {0.0, 1.0}
    # incapacitation suppresses the detained group's recorded rearrest, so
    # holding exposure fixed must RAISE the detained/released odds ratio
    assert exp.oracle["marginal_odds_ratio"] > rec.oracle["marginal_odds_ratio"]
    with pytest.raises(ValueError):
        make_compas(outcome="ground_truth")


def test_every_results_json_has_data_source():
    missing = []
    for p in glob.glob(os.path.join(ROOT, "results", "*.json")):
        try:
            d = json.load(open(p))
        except json.JSONDecodeError:
            continue                       # being rewritten by a running experiment
        if isinstance(d, dict) and d.get("data_source") not in ("real", "simulator", "semi-synthetic", "synthetic"):
            missing.append(os.path.basename(p))
    assert not missing, f"results files without data_source: {missing}"
