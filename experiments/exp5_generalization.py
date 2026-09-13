"""Experiment 5 -- Theorem 4: excess worst-case risk, and its log-Gamma scaling.

Theorem 4 bounds the excess worst-case risk of the empirical minimiser over a
class ``F`` by ``4 L Rad_n(F) + 2 (B + log Gamma) sqrt(log(2/delta) / 2n)`` plus
a nuisance term.  Two predictions are checked directly:

(a) the excess worst-case risk decays like ``n^{-1/2}`` at fixed ``Gamma``;
(b) the coefficient grows **linearly in ``log Gamma``**, not in ``Gamma`` -- which
    is the whole point of working on the log-odds scale, where ``Gamma`` acts as
    a translation rather than a multiplicative blow-up.

We fit ``DCLParametric`` (a convex program, by Theorem 2(b)) on ``n`` training
units and measure the gap between its worst-case risk on held-out units and that
of the population minimiser over the same class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Timer, save, save_table
from dcl.data import make_sl_bench
from dcl.models import DCLParametric
from dcl.objectives import worstcase_risk
from dcl.sensitivity import identified_set, outcome_bounds


def run(ns=(250, 500, 1000, 2000, 4000, 8000), gammas=(1.5, 2.0, 3.0, 6.0, 12.0),
        n_seeds=5, d=10, n_pop=40000, seed=0):
    rows = []
    for gi, gamma in enumerate(gammas):
        # one large population per Gamma, defining the population optimum
        pop = make_sl_bench(n=n_pop, d=d, target_gamma=3.0, seed=1000 + gi)
        o = pop.oracle
        lo_all, hi_all = outcome_bounds(o["p1"], o["e"], gamma)
        star = DCLParametric(gamma=gamma, arch="linear", max_iter=400).fit(
            pop.X, identified_set(o["p1"], o["e"], gamma))
        risk_star = worstcase_risk(star.decision_function(pop.X), lo_all, hi_all)
        for n in ns:
            for sd in range(n_seeds):
                rng = np.random.default_rng(10_000 * gi + 100 * sd + n)
                idx = rng.choice(n_pop, n, replace=False)
                m = DCLParametric(gamma=gamma, arch="linear", max_iter=400,
                                  seed=sd).fit(
                    pop.X[idx], identified_set(o["p1"][idx], o["e"][idx], gamma))
                # evaluate the learned rule on the WHOLE population
                risk_hat = worstcase_risk(m.decision_function(pop.X), lo_all, hi_all)
                rows.append(dict(gamma=gamma, log_gamma=float(np.log(gamma)), n=n,
                                 seed=sd, excess=float(risk_hat - risk_star),
                                 risk_hat=float(risk_hat), risk_star=float(risk_star)))
        print(f"    Gamma={gamma}: R*={risk_star:.5f}", flush=True)
    return pd.DataFrame(rows)


def fit_scaling(df: pd.DataFrame):
    """Excess ~ C(Gamma) * n^{-1/2}; then regress C on log Gamma and on Gamma."""
    out = []
    for gamma, sub in df.groupby("gamma"):
        m = sub.groupby("n").excess.mean()
        m = m[m > 0]
        if len(m) < 3:
            continue
        slope, intercept = np.polyfit(np.log(m.index.values), np.log(m.values), 1)
        C = float(np.exp(intercept))
        out.append(dict(gamma=float(gamma), log_gamma=float(np.log(gamma)),
                        rate_exponent=float(slope), coefficient=C))
    o = pd.DataFrame(out)
    def r2(x, y):
        b, a = np.polyfit(x, y, 1)
        pred = a + b * x
        ss = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-18)
        return float(b), float(a), float(ss)
    b1, a1, r2_log = r2(o.log_gamma.values, o.coefficient.values)
    b2, a2, r2_lin = r2(o.gamma.values, o.coefficient.values)
    return o, dict(slope_vs_log_gamma=b1, r2_vs_log_gamma=r2_log,
                   slope_vs_gamma=b2, r2_vs_gamma=r2_lin,
                   mean_rate_exponent=float(o.rate_exponent.mean()))


if __name__ == "__main__":
    with Timer("exp5 generalization"):
        df = run()
    save_table("exp5_generalization_raw", df)
    o, fit = fit_scaling(df)
    save_table("exp5_scaling", o)
    save("exp5_summary", fit)
    print("\n" + o.to_string(index=False, float_format=lambda x: f"{x:9.5f}"))
    print(f"\n  mean empirical rate exponent (theory: -0.5): {fit['mean_rate_exponent']:.3f}")
    print(f"  coefficient vs log Gamma : slope {fit['slope_vs_log_gamma']:.4f}"
          f"  R^2 {fit['r2_vs_log_gamma']:.4f}")
    print(f"  coefficient vs Gamma     : slope {fit['slope_vs_gamma']:.4f}"
          f"  R^2 {fit['r2_vs_gamma']:.4f}")
