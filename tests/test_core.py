"""Tests for the DCL core: every claim the paper makes that can be checked."""
from __future__ import annotations

import numpy as np
import pytest

from dcl.auc_bounds import (auc_direct, auc_from_regression, bruteforce_auc_interval,
                            midrank, naive_corner_interval, ppv_spec,
                            sensitivity_spec, separate_bounds_interval, sharp_bounds,
                            sharp_auc_interval, specificity_spec)
from dcl.falsify import gamma_lower_bound
from dcl.objectives import (LOSSES, dcl_bayes_score, gap_is_convex, minimax_regret,
                            worstcase_risk, worstcase_risk_decomposition,
                            budgeted_worstcase_risk, least_favourable_p)
from dcl.sensitivity import (expit, identified_set, logit, outcome_bounds,
                             realised_gamma, gamma_from_lambda, lambda_from_gamma)
from dcl.uq import binary_entropy, entropy_range, naive_decomposition


@pytest.fixture
def box():
    rng = np.random.default_rng(0)
    n = 400
    p1 = np.clip(rng.beta(2, 4, n), 0.02, 0.98)
    e = rng.uniform(0.15, 0.9, n)
    s = rng.normal(size=n)
    return s, p1, e


# ---------------------------------------------------------------- sensitivity
def test_gamma_one_collapses(box):
    _, p1, e = box
    lo, hi = outcome_bounds(p1, e, 1.0)
    assert np.allclose(lo, p1) and np.allclose(hi, p1)


def test_manski_limit(box):
    _, p1, e = box
    lo, hi = outcome_bounds(p1, e, np.inf)
    assert np.allclose(lo, e * p1) and np.allclose(hi, e * p1 + (1 - e))


def test_p1_always_inside_box(box):
    """Uncertainty corollary (i): the observed-data regression is never refuted."""
    _, p1, e = box
    for g in (1.0, 1.5, 3.0, 10.0, np.inf):
        lo, hi = outcome_bounds(p1, e, g)
        assert np.all(lo <= p1 + 1e-12) and np.all(hi >= p1 - 1e-12)


def test_box_nested_in_gamma(box):
    _, p1, e = box
    prev = outcome_bounds(p1, e, 1.0)
    for g in (1.5, 2.0, 5.0, 20.0):
        cur = outcome_bounds(p1, e, g)
        assert np.all(cur[0] <= prev[0] + 1e-12) and np.all(cur[1] >= prev[1] - 1e-12)
        prev = cur


def test_directional_halves_and_contains(box):
    _, p1, e = box
    lo2, hi2 = outcome_bounds(p1, e, 3.0, "two-sided")
    lo_l, hi_l = outcome_bounds(p1, e, 3.0, "censored-lower")
    lo_h, hi_h = outcome_bounds(p1, e, 3.0, "censored-higher")
    for lo, hi in ((lo_l, hi_l), (lo_h, hi_h)):
        assert np.all(lo >= lo2 - 1e-12) and np.all(hi <= hi2 + 1e-12)
        assert np.all(lo <= p1 + 1e-12) and np.all(hi >= p1 - 1e-12)
    assert np.allclose(hi_l, p1) and np.allclose(lo_h, p1)


def test_msm_translation():
    assert np.isclose(gamma_from_lambda(lambda_from_gamma(7.0)), 7.0)
    assert np.isclose(gamma_from_lambda(2.0), 4.0)


def test_realised_gamma_roundtrip(box):
    _, p1, _ = box
    p0 = expit(logit(p1) + np.log(2.5))
    assert np.isclose(realised_gamma(p0, p1), 2.5, rtol=1e-6)


# -------------------------------------------------------------------- ranking
@pytest.mark.parametrize("seed", range(5))
def test_rank_identity_matches_definition(seed):
    """The rank identity, including ties and non-uniform weights."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 90))
    s = np.round(rng.normal(size=n), 1)          # deliberate ties
    p = rng.uniform(size=n)
    w = rng.uniform(0.2, 2.0, n)
    assert np.isclose(auc_from_regression(p, midrank(s, w), w),
                      auc_direct(p, s, w), atol=1e-12)


def test_rank_identity_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    s = rng.normal(size=600)
    y = (rng.uniform(size=600) < 0.3).astype(float)
    assert np.isclose(auc_from_regression(y, midrank(s)), roc_auc_score(y, s),
                      atol=1e-12)


@pytest.mark.parametrize("gamma", [1.5, 3.0, 8.0])
def test_sharp_bounds_match_bruteforce(box, gamma):
    """Sharpness: the analytic interval is what an independent search finds."""
    s, p1, e = box
    s, p1, e = s[:120], p1[:120], e[:120]
    lo, hi = outcome_bounds(p1, e, gamma)
    res = sharp_auc_interval(s, lo, hi)
    bl, bh = bruteforce_auc_interval(s, lo, hi, n_restarts=80, seed=1)
    assert bl >= res.lower - 1e-7 and bh <= res.upper + 1e-7
    assert abs(bl - res.lower) < 1e-4 and abs(res.upper - bh) < 1e-4


@pytest.mark.parametrize("gamma", [1.5, 4.0])
def test_bounds_are_attained(box, gamma):
    """Sharpness: the extremal p is in the box and realises the bound."""
    s, p1, e = box
    lo, hi = outcome_bounds(p1, e, gamma)
    res = sharp_auc_interval(s, lo, hi)
    for p, target in ((res.p_upper, res.upper), (res.p_lower, res.lower)):
        assert np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9)
        assert np.isclose(auc_direct(p, s), target, atol=1e-9)


def test_random_members_inside_interval(box):
    s, p1, e = box
    iset = identified_set(p1, e, 3.0)
    res = sharp_auc_interval(s, iset.lo, iset.hi)
    rng = np.random.default_rng(5)
    for kind in ("uniform", "vertex"):
        for _ in range(40):
            p = iset.sample(rng, kind)
            assert res.lower - 1e-9 <= auc_direct(p, s) <= res.upper + 1e-9


def test_corner_interval_is_inside_and_can_be_strict(box):
    """Corner evaluation under-covers."""
    s, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    res = sharp_auc_interval(s, lo, hi)
    c = naive_corner_interval(s, lo, hi)
    assert res.lower - 1e-9 <= c[0] and c[1] <= res.upper + 1e-9
    assert (c[1] - c[0]) < res.width


def test_separate_bounds_is_a_superset(box):
    s, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    res = sharp_auc_interval(s, lo, hi)
    sb = separate_bounds_interval(s, lo, hi)
    assert sb[0] <= res.lower + 1e-9 and sb[1] >= res.upper - 1e-9


def test_gamma_one_gives_point_interval(box):
    s, p1, e = box
    lo, hi = outcome_bounds(p1, e, 1.0)
    res = sharp_auc_interval(s, lo, hi)
    assert res.width < 1e-9
    assert np.isclose(res.lower, auc_from_regression(p1, midrank(s)), atol=1e-9)


def test_threshold_metrics_are_special_cases(box):
    """The unified metric class recovers fixed-threshold metrics."""
    s, p1, e = box
    lo, hi = outcome_bounds(p1, e, 2.5)
    rng = np.random.default_rng(0)
    for spec in (sensitivity_spec(s, 0.0), specificity_spec(s, 0.0), ppv_spec(s, 0.0)):
        res = sharp_bounds(spec, lo, hi)
        iset = identified_set(p1, e, 2.5)
        for _ in range(60):
            p = iset.sample(rng, "uniform")
            v = spec.value(p)
            assert res.lower - 1e-8 <= v <= res.upper + 1e-8


# ------------------------------------------------------------------ objective
@pytest.mark.parametrize("loss", list(LOSSES))
def test_decomposition_equals_direct_max(box, loss):
    """The exact reduction of the inner supremum."""
    s, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    L = LOSSES[loss]
    direct = np.mean(np.maximum(hi * L.l1(s) + (1 - hi) * L.l0(s),
                                lo * L.l1(s) + (1 - lo) * L.l0(s)))
    base, pen = worstcase_risk_decomposition(s, lo, hi, loss)
    assert np.isclose(base + pen, direct, atol=1e-12)


@pytest.mark.parametrize("loss", list(LOSSES))
def test_worstcase_risk_is_convex_in_scores(box, loss):
    """Convex for EVERY convex loss -- no relaxation needed."""
    s, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(len(s), 4))
    for _ in range(60):
        a, b = rng.normal(size=4) * 2, rng.normal(size=4) * 2
        t = rng.uniform()
        mid = worstcase_risk(X @ (t * a + (1 - t) * b), lo, hi, loss)
        ends = t * worstcase_risk(X @ a, lo, hi, loss) + \
            (1 - t) * worstcase_risk(X @ b, lo, hi, loss)
        assert mid <= ends + 1e-9


def test_penalty_convexity_dichotomy():
    """The PENALTY term alone is convex iff the label gap is."""
    assert gap_is_convex("logistic") and gap_is_convex("squared")
    assert gap_is_convex("exponential")
    assert not gap_is_convex("hinge")


def test_bayes_act_matches_grid(box):
    """The closed-form minimax-risk Bayes act."""
    _, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    grid = np.linspace(-14, 14, 20001)
    z = dcl_bayes_score(lo, hi, "risk")
    rng = np.random.default_rng(2)
    for i in rng.choice(len(lo), 25, replace=False):
        m, d = 0.5 * (lo[i] + hi[i]), hi[i] - lo[i]
        J = m * np.logaddexp(0, -grid) + (1 - m) * np.logaddexp(0, grid) \
            + 0.5 * d * np.abs(grid)
        assert abs(grid[np.argmin(J)] - z[i]) < 5e-3


def test_regret_rule_matches_grid(box):
    """The closed-form minimax-regret rule."""
    _, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    q = expit(dcl_bayes_score(lo, hi, "regret"))
    grid = np.linspace(1e-4, 1 - 1e-4, 40001)

    def kl(p, qq):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return p * np.log(p / qq) + (1 - p) * np.log((1 - p) / (1 - qq))

    rng = np.random.default_rng(4)
    for i in rng.choice(len(lo), 15, replace=False):
        obj = np.maximum(kl(lo[i], grid), kl(hi[i], grid))
        assert abs(grid[np.argmin(obj)] - q[i]) < 1e-3


def test_each_rule_wins_its_own_criterion(box):
    _, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    zr, zg = dcl_bayes_score(lo, hi, "risk"), dcl_bayes_score(lo, hi, "regret")
    assert worstcase_risk(zr, lo, hi) <= worstcase_risk(zg, lo, hi) + 1e-12
    assert minimax_regret(zg, lo, hi) <= minimax_regret(zr, lo, hi) + 1e-12


def test_certificate_is_valid(box):
    """The certified risk upper-bounds the realised risk."""
    s, p1, e = box
    iset = identified_set(p1, e, 3.0)
    cert = worstcase_risk(s, iset.lo, iset.hi)
    rng = np.random.default_rng(6)
    L = LOSSES["logistic"]
    for _ in range(80):
        p = iset.sample(rng, "uniform")
        assert np.mean(p * L.l1(s) + (1 - p) * L.l0(s)) <= cert + 1e-12


def test_least_favourable_attains_the_sup(box):
    s, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    p = least_favourable_p(s, lo, hi)
    L = LOSSES["logistic"]
    assert np.isclose(np.mean(p * L.l1(s) + (1 - p) * L.l0(s)),
                      worstcase_risk(s, lo, hi), atol=1e-12)


def test_budgeted_interpolates(box):
    """Remark: budgeted DCSM lies between Gamma=1 and pointwise Gamma."""
    s, p1, e = box
    g = 3.0
    full = worstcase_risk(s, *outcome_bounds(p1, e, g))
    mar = worstcase_risk(s, *outcome_bounds(p1, e, 1.0))
    prev = mar
    for B in (0.05, 0.3, 0.6, np.log(g)):
        r, _, _ = budgeted_worstcase_risk(s, p1, e, g, B)
        assert mar - 1e-6 <= r <= full + 1e-6
        assert r >= prev - 1e-6
        prev = r


# ------------------------------------------------------------------------- uq
def test_entropy_range_is_sharp(box):
    _, p1, e = box
    lo, hi = outcome_bounds(p1, e, 3.0)
    h_min, h_max = entropy_range(lo, hi)
    rng = np.random.default_rng(7)
    for _ in range(50):
        p = lo + rng.uniform(size=len(lo)) * (hi - lo)
        h = binary_entropy(p)
        assert np.all(h >= h_min - 1e-9) and np.all(h <= h_max + 1e-9)


def test_naive_decomposition_adds_up():
    rng = np.random.default_rng(8)
    P = np.clip(rng.uniform(size=(12, 300)), 1e-4, 1 - 1e-4)
    d = naive_decomposition(P)
    assert np.allclose(d.total, d.aleatoric + d.epistemic)
    assert np.all(d.epistemic >= -1e-12)


# -------------------------------------------------------------------- falsify
def test_gamma_min_is_a_valid_lower_bound():
    """Gamma_min <= Gamma_0, and it is tight in the ideal case."""
    rng = np.random.default_rng(0)
    n = 2000
    p = expit(rng.normal(0, 1.2, n))
    for g0 in (1.0, 2.0, 4.0):
        p1s, es = [], []
        for lenient, tilt in ((0.35, +1.0), (0.75, -1.0)):
            e_z = np.full(n, lenient)
            t = np.log(g0) * tilt
            a, b = np.full(n, 1e-6), np.full(n, 1 - 1e-6)
            for _ in range(70):
                m = 0.5 * (a + b)
                v = e_z * m + (1 - e_z) * expit(logit(m) + t)
                a = np.where(v < p, m, a)
                b = np.where(v < p, b, m)
            p1s.append(0.5 * (a + b)); es.append(e_z)
        gmin = gamma_lower_bound(p1s, es, quantile=0.95)
        assert gmin <= g0 * 1.05 + 1e-6
        assert gmin >= g0 * 0.95 - 1e-6      # tight when tilts are opposed


# ------------------------------------------------------------------- learners
def test_dcl_ranker_never_worse_than_warm_start():
    """DCLRanker uses the exact sharp-interval oracle for best-iterate selection, so
    it can never return something worse than its plug-in initialisation."""
    from sklearn.linear_model import Ridge

    from dcl.ranking import DCLRanker, worst_case_auc
    rng = np.random.default_rng(0)
    n, d = 1200, 6
    X = rng.normal(size=(n, d))
    p1 = np.clip(expit(X @ rng.normal(size=d) / np.sqrt(d)), 0.02, 0.98)
    e = np.clip(0.5 + 0.2 * X[:, 0], 0.1, 0.9)
    lo, hi = outcome_bounds(p1, e, 3.0)
    mid = 0.5 * (lo + hi)
    w0 = Ridge(alpha=1e-3).fit(X, mid).coef_
    w0 = w0 / np.linalg.norm(w0)
    warm = worst_case_auc(X @ w0, lo, hi)

    r = DCLRanker(n_steps=60, oracle_every=10, seed=0).fit(X, lo, hi)
    assert worst_case_auc(r.decision_function(X), lo, hi) >= warm - 1e-9
    assert np.isfinite(r.best_worst_case_auc_)


def test_dcl_parametric_matches_closed_form_on_a_rich_class():
    """A flexible parametric fit should approach the pointwise optimum."""
    pytest.importorskip("torch")
    from dcl.models import DCLParametric
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 4))
    p1 = np.clip(expit(X @ np.array([1.0, -0.5, 0.3, 0.2])), 0.02, 0.98)
    e = np.full(n, 0.5)
    iset = identified_set(p1, e, 2.0)
    star = worstcase_risk(dcl_bayes_score(iset.lo, iset.hi), iset.lo, iset.hi)
    m = DCLParametric(gamma=2.0, arch="mlp", max_iter=400, seed=0).fit(X, iset)
    got = worstcase_risk(m.decision_function(), iset.lo, iset.hi)
    assert got >= star - 1e-6          # the closed form is the pointwise optimum
    assert got <= star + 0.05          # and a rich class gets close to it


def test_dcl_plugin_out_of_sample():
    from dcl.data import make_sl_bench
    from dcl.models import DCLPlugin
    ds = make_sl_bench(n=3000, d=6, target_gamma=2.0, seed=0)
    tr, te = ds.split(test_size=0.3, seed=0)
    m = DCLPlugin(gamma=2.0, n_folds=3).fit(tr.X, tr.T, tr.Y_obs)
    s = m.decision_function(te.X)
    assert s.shape == (te.n,)
    lo, hi = m.predict_proba_bounds(te.X)
    assert np.all(lo <= hi + 1e-12) and np.all((lo >= 0) & (hi <= 1))
