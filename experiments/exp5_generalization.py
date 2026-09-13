"""Experiment 5 -- Theorem 11 and its log-Gamma scaling.

What the theorem actually bounds.  \\Cref{thm:generalization} controls the excess
worst-case risk through the *uniform deviation*
``sup_{f in F} |Rhat_Gamma(f) - Rbar_Gamma(f)|``, and it is there that the
``Gamma`` dependence lives: the bound's second term is
``ell_max(B) sqrt(log(2/delta)/2n)`` with ``B = B_0 + log Gamma``.

So we measure two different things, and keep them apart.

**Panel A -- uniform deviation.**  The quantity the bound is about.  Estimated by
maximising ``|Rhat - Rbar|`` over a large random sample of the score class (plus
the empirical and population minimisers, which are the scores that matter).  No
optimisation is involved, so it is essentially noise-free, and it should scale
like ``(B_0 + log Gamma) n^{-1/2}``.

**Panel B -- excess risk of the empirical minimiser.**  The quantity a
practitioner cares about.  It decays *faster* than the bound guarantees --
roughly ``n^{-1}`` -- because the objective is strongly convex near the optimum
and the class is well specified; that is the usual fast-rate phenomenon and is
consistent with, not evidence for, an ``n^{-1/2}`` upper bound.  Reporting it as
a confirmation of the rate would be wrong, so we report it as what it is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Timer, save, save_table
from dcl.data import make_sl_bench
from dcl.models import DCLParametric
from dcl.objectives import worstcase_risk
from dcl.sensitivity import identified_set, outcome_bounds


def _score_class(X, rng, n_draws, B):
    """Random directions in the linear class, scores clipped to ``[-B, B]``."""
    d = X.shape[1]
    W = rng.normal(size=(n_draws, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    scale = rng.uniform(0.2, 3.0, size=(n_draws, 1))
    return np.clip(X @ (W * scale).T, -B, B)          # (n, n_draws)


def panel_a_deviation(ns=(250, 500, 1000, 2000, 4000, 8000),
                      gammas=(1.5, 2.0, 3.0, 6.0, 12.0, 24.0),
                      n_seeds=8, d=10, n_pop=60000, n_draws=400, seed=0):
    """Uniform deviation over the score class -- the object Theorem 11 bounds."""
    pop = make_sl_bench(n=n_pop, d=d, target_gamma=3.0, seed=seed)
    o = pop.oracle
    B0 = float(np.max(np.abs(np.log(o["p1"] / (1 - o["p1"])))))
    rows = []
    for gamma in gammas:
        B = B0 + np.log(gamma)
        lo_all, hi_all = outcome_bounds(o["p1"], o["e"], gamma)
        rng = np.random.default_rng(hash((seed, gamma)) % (2**31))
        S_pop = _score_class(pop.X, rng, n_draws, B)          # (n_pop, n_draws)
        # population worst-case risk of each candidate score
        Rbar = np.array([worstcase_risk(S_pop[:, k], lo_all, hi_all)
                         for k in range(n_draws)])
        for n in ns:
            for sd in range(n_seeds):
                r2 = np.random.default_rng(10_000 * sd + n)
                idx = r2.choice(n_pop, n, replace=False)
                lo_s, hi_s = lo_all[idx], hi_all[idx]
                Rhat = np.array([worstcase_risk(S_pop[idx, k], lo_s, hi_s)
                                 for k in range(n_draws)])
                rows.append(dict(gamma=gamma, log_gamma=float(np.log(gamma)),
                                 B=B, n=n, seed=sd,
                                 deviation=float(np.max(np.abs(Rhat - Rbar)))))
        print(f"    Gamma={gamma:5.1f}  B={B:5.2f}  "
              f"dev(n=250)={np.mean([r['deviation'] for r in rows if r['gamma']==gamma and r['n']==250]):.4f}"
              f"  dev(n=8000)={np.mean([r['deviation'] for r in rows if r['gamma']==gamma and r['n']==8000]):.4f}",
              flush=True)
    return pd.DataFrame(rows), B0


def panel_b_excess(ns=(250, 500, 1000, 2000, 4000, 8000),
                   gammas=(1.5, 2.0, 3.0, 6.0, 12.0), n_seeds=5, d=10,
                   n_pop=40000, seed=0):
    rows = []
    for gi, gamma in enumerate(gammas):
        pop = make_sl_bench(n=n_pop, d=d, target_gamma=3.0, seed=1000 + gi)
        o = pop.oracle
        lo_all, hi_all = outcome_bounds(o["p1"], o["e"], gamma)
        star = DCLParametric(gamma=gamma, arch="linear", max_iter=300).fit(
            pop.X, identified_set(o["p1"], o["e"], gamma))
        risk_star = worstcase_risk(star.decision_function(pop.X), lo_all, hi_all)
        for n in ns:
            for sd in range(n_seeds):
                rng = np.random.default_rng(10_000 * gi + 100 * sd + n)
                idx = rng.choice(n_pop, n, replace=False)
                m = DCLParametric(gamma=gamma, arch="linear", max_iter=300,
                                  seed=sd).fit(
                    pop.X[idx], identified_set(o["p1"][idx], o["e"][idx], gamma))
                rows.append(dict(
                    gamma=gamma, log_gamma=float(np.log(gamma)), n=n, seed=sd,
                    excess=float(worstcase_risk(m.decision_function(pop.X),
                                                lo_all, hi_all) - risk_star)))
        print(f"    Gamma={gamma}: R*={risk_star:.5f}", flush=True)
    return pd.DataFrame(rows)


def _loglog_slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def _r2(x, y):
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    return float(b), float(1 - np.sum((y - pred) ** 2)
                 / max(np.sum((y - y.mean()) ** 2), 1e-18))


if __name__ == "__main__":
    with Timer("exp5a uniform deviation"):
        dev, B0 = panel_a_deviation()
    with Timer("exp5b excess risk"):
        exc = panel_b_excess()
    save_table("exp5a_deviation", dev)
    save_table("exp5_generalization_raw", exc)

    # Panel A: rate in n, and the Gamma dependence of the coefficient
    rows = []
    for gamma, sub in dev.groupby("gamma"):
        m = sub.groupby("n").deviation.mean()
        slope = _loglog_slope(m.index.values, m.values)
        # coefficient of the n^{-1/2} fit: dev ~ C * n^{-1/2}
        C = float(np.mean(m.values * np.sqrt(m.index.values)))
        rows.append(dict(gamma=float(gamma), log_gamma=float(np.log(gamma)),
                         B=float(sub.B.iloc[0]), rate_exponent=slope,
                         coefficient=C))
    sc = pd.DataFrame(rows)
    save_table("exp5_scaling", sc)
    b_log, r2_log = _r2(sc.log_gamma.values, sc.coefficient.values)
    b_lin, r2_lin = _r2(sc.gamma.values, sc.coefficient.values)
    b_B, r2_B = _r2(sc.B.values, sc.coefficient.values)

    exc_rows = []
    for gamma, sub in exc.groupby("gamma"):
        m = sub.groupby("n").excess.mean()
        m = m[m > 0]
        if len(m) >= 3:
            exc_rows.append(dict(gamma=float(gamma),
                                 rate_exponent=_loglog_slope(m.index.values, m.values)))
    exc_sc = pd.DataFrame(exc_rows)
    save_table("exp5b_excess_scaling", exc_sc)

    print("\n=== Panel A: uniform deviation (what Theorem 11 bounds) ===")
    print(sc.to_string(index=False, float_format=lambda x: f"{x:9.5f}"))
    print(f"\n  mean rate exponent in n : {sc.rate_exponent.mean():.3f}   (theory -0.5)")
    print(f"  coefficient vs log Gamma: slope {b_log:.4f}  R^2 {r2_log:.4f}")
    print(f"  coefficient vs Gamma    : slope {b_lin:.4f}  R^2 {r2_lin:.4f}")
    print(f"  coefficient vs B = B0+logGamma: slope {b_B:.4f}  R^2 {r2_B:.4f}")
    print("\n=== Panel B: excess risk of the empirical minimiser ===")
    print(exc_sc.to_string(index=False, float_format=lambda x: f"{x:9.5f}"))
    print(f"  mean rate exponent: {exc_sc.rate_exponent.mean():.3f}"
          "   (faster than the n^-1/2 the bound guarantees: fast rates)")

    save("exp5_summary", dict(
        B0=float(B0),
        deviation_rate_exponent=float(sc.rate_exponent.mean()),
        slope_vs_log_gamma=b_log, r2_vs_log_gamma=r2_log,
        slope_vs_gamma=b_lin, r2_vs_gamma=r2_lin,
        slope_vs_B=b_B, r2_vs_B=r2_B,
        excess_rate_exponent=float(exc_sc.rate_exponent.mean()),
    ))
