"""Tests for dcl.intervals: the nuisance-robust interval machinery."""
import numpy as np

from dcl.intervals import (binned_nuisance_intervals, clopper_pearson, dr_functional,
                           g_tilt, g_tilt_prime, inflate_box, prevalence_bounds_dr)
from dcl.sensitivity import expit, outcome_bounds


def test_tilt_matches_sensitivity_module():
    rng = np.random.default_rng(0)
    p1 = rng.uniform(0.02, 0.98, 500); e = rng.uniform(0.1, 0.9, 500)
    for gamma in (1.5, 3.0):
        lo, hi = outcome_bounds(p1, e, gamma)
        assert np.allclose(lo, e * p1 + (1 - e) * g_tilt(p1, 1 / gamma))
        assert np.allclose(hi, e * p1 + (1 - e) * g_tilt(p1, gamma))
    # derivative by finite differences
    p = np.linspace(0.05, 0.95, 19); h = 1e-6
    fd = (g_tilt(p + h, 2.5) - g_tilt(p - h, 2.5)) / (2 * h)
    assert np.allclose(fd, g_tilt_prime(p, 2.5), atol=1e-6)


def test_box_monotonicity_claims():
    """lo increases in e and eta; hi decreases in e and increases in eta."""
    gamma = 3.0
    for eta in (-2.0, 0.0, 1.5):
        for e in (0.2, 0.5, 0.8):
            p1 = expit(eta)
            lo0, hi0 = outcome_bounds(np.array([p1]), np.array([e]), gamma)
            lo1, hi1 = outcome_bounds(np.array([p1]), np.array([e + 0.05]), gamma)
            assert lo1 >= lo0 - 1e-12 and hi1 <= hi0 + 1e-12
            lo2, hi2 = outcome_bounds(np.array([expit(eta + 0.1)]), np.array([e]), gamma)
            assert lo2 >= lo0 - 1e-12 and hi2 >= hi0 - 1e-12


def test_inflated_box_contains_true_box_when_nuisance_intervals_cover():
    rng = np.random.default_rng(1)
    n = 2000
    e = rng.uniform(0.15, 0.85, n); eta = rng.normal(0, 1.5, n)
    gamma = 2.5
    lo, hi = outcome_bounds(expit(eta), e, gamma)
    e_L, e_U = np.clip(e - rng.uniform(0, .1, n), 0, 1), np.clip(e + rng.uniform(0, .1, n), 0, 1)
    eta_L, eta_U = eta - rng.uniform(0, .5, n), eta + rng.uniform(0, .5, n)
    for d in ("two-sided", "censored-lower", "censored-higher"):
        box = inflate_box(e_L, e_U, eta_L, eta_U, gamma, d)
        lo_d, hi_d = outcome_bounds(expit(eta), e, gamma, d)
        assert np.all(box.lo <= lo_d + 1e-12) and np.all(box.hi >= hi_d - 1e-12)
    # degenerate intervals recover the exact box
    box = inflate_box(e, e, eta, eta, gamma)
    assert np.allclose(box.lo, lo) and np.allclose(box.hi, hi)


def test_clopper_pearson_basic():
    lo, hi = clopper_pearson(0, 20, 0.05); assert lo == 0.0 and 0.1 < hi < 0.2
    lo, hi = clopper_pearson(20, 20, 0.05); assert hi == 1.0 and 0.8 < lo < 0.9
    lo, hi = clopper_pearson(10, 20, 0.05); assert lo < 0.5 < hi


def _simulate(n, seed, gamma0=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = expit(0.4 * X[:, 0] - 0.3 * X[:, 1])
    eta1 = 0.8 * X[:, 0] + 0.5 * X[:, 2] - 0.3
    p1 = expit(eta1)
    p0 = expit(eta1 - np.log(gamma0))            # censored units are safer
    T = (rng.uniform(size=n) < e).astype(float)
    Y = np.where(T == 1, rng.uniform(size=n) < p1, rng.uniform(size=n) < p0).astype(float)
    Y_obs = np.where(T == 1, Y, np.nan)
    return X, T, Y, Y_obs, e, p1, p0


def test_dr_functional_is_unbiased_with_true_nuisances_and_robust_to_perturbation():
    n = 40000
    X, T, Y, Y_obs, e, p1, p0 = _simulate(n, 0)
    gamma = 2.0
    truth_lo = float(np.mean(e * p1 + (1 - e) * g_tilt(p1, 1 / gamma)))
    res = dr_functional(T, Y_obs, e, p1, 1 / gamma)
    assert abs(res.estimate - truth_lo) < 4 * res.se
    # perturb the nuisances: the DR estimate moves by second order, the plug-in by first
    rng = np.random.default_rng(5)
    e_bad = np.clip(e + 0.08 * rng.normal(size=n), 0.02, 0.98)
    p1_bad = np.clip(p1 + 0.08 * rng.normal(size=n), 0.02, 0.98)
    res_bad = dr_functional(T, Y_obs, e_bad, p1_bad, 1 / gamma)
    assert abs(res_bad.estimate - truth_lo) < abs(res_bad.plugin - truth_lo) + 3 * res_bad.se


def test_prevalence_bounds_dr_outer_interval_covers_identified_interval():
    covered = 0
    for seed in range(20):
        _, T, Y, Y_obs, e, p1, p0 = _simulate(6000, seed)
        gamma = 2.0
        pi_lo = float(np.mean(e * p1 + (1 - e) * g_tilt(p1, 1 / gamma)))
        pi_hi = float(np.mean(e * p1 + (1 - e) * g_tilt(p1, gamma)))
        out = prevalence_bounds_dr(T, Y_obs, e, p1, gamma, alpha=0.1)["outer"]
        covered += (out[0] <= pi_lo) and (out[1] >= pi_hi)
    assert covered >= 17          # nominal 90% joint coverage


def test_binned_intervals_cover_bin_means_and_inflation_helps():
    """Guaranteed: bin-mean coverage.  Measured: pointwise coverage rises, and
    the smoothness margin raises it further at the price of width."""
    n = 20000
    X, T, Y, Y_obs, e, p1, p0 = _simulate(n, 2)
    rng = np.random.default_rng(3)
    e_hat = np.clip(e + 0.05 * rng.normal(size=n), 0.02, 0.98)
    p1_hat = np.clip(p1 + 0.05 * rng.normal(size=n), 0.02, 0.98)
    iv = binned_nuisance_intervals(e_hat, p1_hat, T, Y_obs, alpha=0.1)
    ok = sum(row["lo"] <= e[iv["e_bins"] == row["bin"]].mean() <= row["hi"]
             for row in iv["e_table"])
    assert ok >= 8                                   # bin-mean coverage ~ 95% per bin
    p_true = e * p1 + (1 - e) * p0
    plug = outcome_bounds(p1_hat, e_hat, 2.0)
    cov_plug = np.mean((p_true >= plug[0]) & (p_true <= plug[1]))
    box = inflate_box(iv["e_L"], iv["e_U"], iv["eta_L"], iv["eta_U"], 2.0)
    cov_infl = np.mean(box.covers(p_true))
    iv2 = binned_nuisance_intervals(e_hat, p1_hat, T, Y_obs, alpha=0.1, smooth_margin=True)
    box2 = inflate_box(iv2["e_L"], iv2["e_U"], iv2["eta_L"], iv2["eta_U"], 2.0)
    cov_smooth = np.mean(box2.covers(p_true))
    assert cov_infl > cov_plug
    assert cov_smooth > cov_infl and cov_smooth >= 0.9
    assert box2.width.mean() > box.width.mean() > (plug[1] - plug[0]).mean()
