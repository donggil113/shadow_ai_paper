"""Experiment 1 -- Theorem 1: the AUROC bounds are sharp, valid, and fast.

Three claims, three checks.

(a) *Sharpness.*  The analytic ``O(n log n)`` interval agrees with an
    independent projected-gradient search over the identified box, to machine
    precision, and the extremal ``p`` it returns is a genuine member of the box
    (so the bounds are attained, not merely valid).

(b) *The naive recipe is invalid.*  Evaluating AUROC at the corners of the box --
    what a practitioner would do -- produces an interval that fails to contain
    the true deployment AUROC, because AUROC couples units through the
    prevalence.  Bounding numerator and denominator separately is valid but
    strictly loose.

(c) *Cost.*  Runtime is ``O(n log n)``.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from _common import Timer, save, save_table
from dcl.auc_bounds import (auc_direct, auc_from_regression, bruteforce_auc_interval,
                            midrank, naive_corner_interval, separate_bounds_interval,
                            sharp_auc_interval)
from dcl.data import make_sl_bench
from dcl.nuisance import CrossFitNuisance
from dcl.sensitivity import outcome_bounds


def part_a_sharpness(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in (50, 150, 500):
        for gamma in (1.5, 2.0, 4.0, 8.0):
            for rep in range(3):
                s = rng.normal(size=n)
                p1 = np.clip(rng.beta(2, 5, n), 0.01, 0.99)
                e = rng.uniform(0.15, 0.9, n)
                lo, hi = outcome_bounds(p1, e, gamma)
                res = sharp_auc_interval(s, lo, hi)
                bl, bh = bruteforce_auc_interval(s, lo, hi, n_restarts=100, seed=rep)
                rows.append(dict(
                    n=n, gamma=gamma, rep=rep,
                    sharp_lo=res.lower, sharp_hi=res.upper,
                    bf_lo=bl, bf_hi=bh,
                    gap_lo=bl - res.lower, gap_hi=res.upper - bh,
                    bf_inside=bool(bl >= res.lower - 1e-8 and bh <= res.upper + 1e-8),
                    attained_hi=abs(auc_direct(res.p_upper, s) - res.upper),
                    attained_lo=abs(auc_direct(res.p_lower, s) - res.lower),
                    in_box_hi=bool(np.all(res.p_upper >= lo - 1e-9)
                                   and np.all(res.p_upper <= hi + 1e-9)),
                ))
    df = pd.DataFrame(rows)
    print(f"  brute force never escaped the interval: {df.bf_inside.all()}")
    print(f"  max |analytic - bruteforce|            : {df[['gap_lo','gap_hi']].abs().max().max():.3e}")
    print(f"  max |AUC(p*) - bound| (attainment)     : {df[['attained_lo','attained_hi']].max().max():.3e}")
    print(f"  extremal p always inside the box       : {df.in_box_hi.all()}")
    return df


def part_b_validity(n: int = 24000, seed: int = 0, n_seeds: int = 3):
    """Do the intervals contain the TRUE deployment AUROC?

    Two targets, because they are different quantities and the distinction
    matters.  The identified set bounds the **population** AUROC functional
    ``AUC(p) = (<p, R> - pi^2/2) / (pi(1-pi))`` -- a functional of the outcome
    *regression*.  What an evaluator actually computes is the **empirical**
    AUROC from realised 0/1 labels, which estimates that functional with
    ``O(n^{-1/2})`` Monte-Carlo error.  We report coverage of both.

    Everything is out of sample: the scorer is fit on a training split and both
    the bounds and the targets are computed on a disjoint test split.  (In
    sample, a flexible model memorises realised labels and its empirical AUROC
    detaches from any population functional -- it exceeded the sharp upper bound
    by 0.23 in an earlier version of this script, which is a property of
    memorisation, not a failure of the bound.)
    """
    from sklearn.metrics import roc_auc_score

    from dcl.baselines import ERMObserved
    rows = []
    for sd in range(n_seeds):
        for target in (1.5, 2.0, 3.0, 5.0):
            ds = make_sl_bench(n=n, d=10, target_gamma=target, seed=seed + sd)
            tr, te = ds.split(test_size=0.5, seed=seed + sd)
            g0 = ds.oracle["gamma0"]
            f = ERMObserved(seed=seed).fit(tr.X, tr.T, tr.Y_obs).decision_function(te.X)
            o = te.oracle
            true_pop = auc_from_regression(o["p"], midrank(f))
            true_emp = float(roc_auc_score(te.Y_full, f))
            obs_emp = float(roc_auc_score(te.Y_obs[te.T == 1], f[te.T == 1]))
            for gamma in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
                lo, hi = outcome_bounds(o["p1"], o["e"], gamma)
                sh = sharp_auc_interval(f, lo, hi)
                nc = naive_corner_interval(f, lo, hi)
                sb = separate_bounds_interval(f, lo, hi)
                rows.append(dict(
                    seed=sd, gamma0=g0, gamma=gamma,
                    true_auc_pop=true_pop, true_auc_emp=true_emp, observed_auc=obs_emp,
                    sharp_lo=sh.lower, sharp_hi=sh.upper, sharp_w=sh.width,
                    sharp_covers_pop=bool(sh.lower - 1e-9 <= true_pop <= sh.upper + 1e-9),
                    sharp_covers_emp=bool(sh.lower - 1e-9 <= true_emp <= sh.upper + 1e-9),
                    naive_lo=nc[0], naive_hi=nc[1], naive_w=nc[1] - nc[0],
                    naive_covers_pop=bool(nc[0] - 1e-9 <= true_pop <= nc[1] + 1e-9),
                    sep_lo=sb[0], sep_hi=sb[1], sep_w=sb[1] - sb[0],
                    sep_covers_pop=bool(sb[0] - 1e-9 <= true_pop <= sb[1] + 1e-9),
                    observed_covers=bool(sh.lower - 1e-9 <= obs_emp <= sh.upper + 1e-9),
                    gamma_sufficient=bool(gamma >= g0),
                ))
    df = pd.DataFrame(rows)
    ok = df[df.gamma_sufficient]
    bad = df[~df.gamma_sufficient]
    print(f"\n  Gamma >= Gamma_0  ({len(ok)} cells):")
    print(f"    sharp covers population AUROC   : {ok.sharp_covers_pop.mean():.1%}")
    print(f"    sharp covers empirical AUROC    : {ok.sharp_covers_emp.mean():.1%}")
    print(f"    naive corner covers             : {ok.naive_covers_pop.mean():.1%}")
    print(f"    separate-bounds covers          : {ok.sep_covers_pop.mean():.1%}")
    print(f"    widths: sharp {ok.sharp_w.mean():.4f} | naive {ok.naive_w.mean():.4f}"
          f" | separate {ok.sep_w.mean():.4f}")
    print(f"    sharp inside separate-bounds    : "
          f"{bool(((ok.sharp_lo >= ok.sep_lo - 1e-9) & (ok.sharp_hi <= ok.sep_hi + 1e-9)).all())}")
    print(f"\n  Gamma <  Gamma_0  ({len(bad)} cells, assumption too strong):")
    print(f"    sharp covers population AUROC   : {bad.sharp_covers_pop.mean():.1%}")
    print(f"    |observed - true| (all cells)   : "
          f"{(df.observed_auc - df.true_auc_emp).abs().mean():.4f}")
    return df


def part_c_runtime(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in (1000, 5000, 20000, 100000, 400000):
        s = rng.normal(size=n)
        p1 = np.clip(rng.beta(2, 5, n), 0.01, 0.99)
        e = rng.uniform(0.15, 0.9, n)
        lo, hi = outcome_bounds(p1, e, 3.0)
        t = time.time(); sharp_auc_interval(s, lo, hi); dt = time.time() - t
        rows.append(dict(n=n, seconds=dt, per_n_log_n=dt / (n * np.log2(n))))
    df = pd.DataFrame(rows)
    print("\n  runtime:")
    for _, r in df.iterrows():
        print(f"    n={int(r.n):>7,d}  {r.seconds:7.3f}s")
    return df


if __name__ == "__main__":
    with Timer("exp1a sharpness"):
        a = part_a_sharpness()
    with Timer("exp1b validity"):
        b = part_b_validity()
    with Timer("exp1c runtime"):
        c = part_c_runtime()
    save_table("exp1a_sharpness", a)
    save_table("exp1b_validity", b)
    save_table("exp1c_runtime", c)
    save("exp1_summary", dict(
        max_abs_gap=float(a[["gap_lo", "gap_hi"]].abs().max().max()),
        bruteforce_always_inside=bool(a.bf_inside.all()),
        max_attainment_error=float(a[["attained_lo", "attained_hi"]].max().max()),
        coverage_sharp_pop=float(b[b.gamma_sufficient].sharp_covers_pop.mean()),
        coverage_sharp_emp=float(b[b.gamma_sufficient].sharp_covers_emp.mean()),
        coverage_naive=float(b[b.gamma_sufficient].naive_covers_pop.mean()),
        coverage_separate=float(b[b.gamma_sufficient].sep_covers_pop.mean()),
        coverage_sharp_when_gamma_too_small=float(
            b[~b.gamma_sufficient].sharp_covers_pop.mean()),
        width_sharp=float(b[b.gamma_sufficient].sharp_w.mean()),
        width_separate=float(b[b.gamma_sufficient].sep_w.mean()),
    ))
