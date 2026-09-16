#!/usr/bin/env python3
"""T3: exhaustive search for rankings that beat the midpoint ranking on
worst-case AUROC.

For small n every permutation of the units is scored by the exact inner value
min_{p in box} AUC(sigma, p) from dcl.auc_bounds.sharp_auc_interval (scores =
position in the permutation). Two regimes:
  A) equal widths  -- the rearrangement argument predicts the lo (= mid = hi)
                      ranking is optimal for every instance: zero violations;
  B) unequal widths -- collect every instance where some permutation beats the
                      midpoint ranking by > tol, and characterise the winner.
Writes audit/t3_counterexamples.csv and prints a summary. Provenance: produced
in this session (2026-09-16); no prior Codex audit file existed in the repo.
"""
import itertools, sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dcl.auc_bounds import sharp_auc_interval

TOL = 1e-9


def wc_auc(order, lo, hi):
    scores = np.empty(len(order)); scores[np.asarray(order)] = np.arange(len(order), dtype=float)
    return sharp_auc_interval(scores, lo, hi).lower


def rank_by(v):
    return tuple(np.argsort(v, kind="mergesort"))       # ascending: position 0 = lowest score


def best_perm(lo, hi):
    n = len(lo); best, arg = -np.inf, None
    for perm in itertools.permutations(range(n)):
        v = wc_auc(perm, lo, hi)
        if v > best + 1e-15:
            best, arg = v, perm
    return best, arg


def run(regime, n_list, n_inst, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_list:
        for k in range(n_inst[n]):
            lo = rng.uniform(0.02, 0.9, n)
            if regime == "A":
                d = rng.uniform(0.02, min(0.5, 0.98 - lo.max()))
                hi = lo + d
            else:
                hi = np.minimum(lo + rng.uniform(0.02, 0.6, n), 0.98)
            mid = 0.5 * (lo + hi)
            v_mid, v_lo, v_hi = (wc_auc(rank_by(x), lo, hi) for x in (mid, lo, hi))
            v_best, p_best = best_perm(lo, hi)
            rows.append(dict(regime=regime, n=n, inst=k, wc_mid=v_mid, wc_lo=v_lo, wc_hi=v_hi,
                             wc_best=v_best, gap_vs_mid=v_best - v_mid, gap_vs_lo=v_best - v_lo,
                             best_order=" ".join(map(str, p_best)), lo=" ".join(repr(float(x)) for x in lo),
                             hi=" ".join(repr(float(x)) for x in hi), width_cv=float(np.std(hi - lo) / np.mean(hi - lo))))
    return pd.DataFrame(rows)


def summarise(A, B, family_hits, out_path):
    """Write results/exp9_ranking_search.json (data_source: synthetic) for the paper macros."""
    import json
    byn = B.groupby("n").gap_vs_mid.agg(["mean", "max", lambda g: float((g > TOL).mean())])
    byn.columns = ["mean", "max", "frac_beaten"]
    summary = dict(
        data_source="synthetic",
        description="exhaustive permutation search for the worst-case-AUROC-optimal ranking; audit/t3_ranking_search.py",
        equal_n_instances=int(len(A)), equal_violations=int((A.gap_vs_mid > TOL).sum()),
        equal_max_gap=float(A.gap_vs_mid.max()),
        unequal_n_instances=int(len(B)), unequal_mid_beaten_frac=float((B.gap_vs_mid > TOL).mean()),
        unequal_lo_beaten_frac=float((B.gap_vs_lo > TOL).mean()),
        unequal_mid_gap_mean=float(B.gap_vs_mid.mean()), unequal_mid_gap_max=float(B.gap_vs_mid.max()),
        unequal_lo_gap_mean=float(B.gap_vs_lo.mean()), unequal_lo_gap_max=float(B.gap_vs_lo.max()),
        unequal_mid_beaten_frac_by_n={str(int(n)): float(r.frac_beaten) for n, r in byn.iterrows()},
        unequal_mid_gap_mean_by_n={str(int(n)): float(r["mean"]) for n, r in byn.iterrows()},
        convex_family_attains_optimum_frac=float(family_hits),
        n_max=int(B.n.max()), n_min=int(B.n.min()),
    )
    json.dump(summary, open(out_path, "w"), indent=1)
    print("->", out_path)


if __name__ == "__main__":
    t = time.time()
    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(here, "t3_counterexamples.csv")
    if "--summarise-only" in sys.argv:
        df = pd.read_csv(csv_path)
        A, B = df[df.regime == "A"], df[df.regime == "B"]
    else:
        n_inst = {3: 1500, 4: 1500, 5: 800, 6: 300, 7: 60}
        A = run("A", [3, 4, 5, 6, 7], n_inst, seed=1)
        B = run("B", [3, 4, 5, 6, 7], n_inst, seed=2)
        df = pd.concat([A, B], ignore_index=True)
        df.to_csv(csv_path, index=False)
    vA = int((A.gap_vs_mid > TOL).sum()); vB = int((B.gap_vs_mid > TOL).sum())
    print(f"regime A (equal widths): {len(A)} instances, violations of mid ranking: {vA}, max gap {A.gap_vs_mid.max():.2e}")
    print(f"regime B (unequal):      {len(B)} instances, mid ranking beaten in {vB} ({vB/len(B):.1%}), max gap {B.gap_vs_mid.max():.4f}")
    print(f"regime B: lo ranking beaten in {int((B.gap_vs_lo > TOL).sum())} ({(B.gap_vs_lo > TOL).mean():.1%}); "
          f"mid better than lo in {int((B.wc_mid > B.wc_lo + TOL).sum())}, lo better than mid in {int((B.wc_lo > B.wc_mid + TOL).sum())}")
    top = B.sort_values("gap_vs_mid", ascending=False).head(5)
    print("\ntop-5 counterexamples (unequal widths):")
    print(top[["n", "wc_mid", "wc_lo", "wc_best", "gap_vs_mid", "best_order", "lo", "hi"]].to_string(index=False))
    # characterise: does 'lo - c*width' explain the best permutation?
    hits = {}
    for c in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        h = 0
        for _, r in B.iterrows():
            lo = np.array(r.lo.split(), float); hi = np.array(r.hi.split(), float)
            cand = rank_by(lo - c * (hi - lo))
            h += abs(wc_auc(cand, lo, hi) - r.wc_best) <= 1e-9
        hits[c] = h / len(B)
    print("\nfraction of instances where ranking by (lo - c*width) attains the optimum:", {k: round(v, 3) for k, v in hits.items()})
    # is the optimum always attained by SOME score-based (threshold-free) ranking of the box?
    # test the family  s_c = lo + c*(hi-lo), c in [0,1]  (c=0: lo, c=1/2: mid, c=1: hi)
    grid = np.linspace(0, 1, 41)
    fam_hits = 0
    for _, r in B.iterrows():
        lo = np.array(r.lo.split(), float); hi = np.array(r.hi.split(), float)
        fam_hits += any(abs(wc_auc(rank_by(lo + c * (hi - lo)), lo, hi) - r.wc_best) <= 1e-9 for c in grid)
    print(f"regime B: optimum attained by some convex-combination ranking lo+c*(hi-lo), c in [0,1]: {fam_hits/len(B):.1%}")
    summarise(A, B, fam_hits / len(B), os.path.join(os.path.dirname(here), "results", "exp9_ranking_search.json"))
    # size of the gap in AUC units, by n
    print("regime B gap_vs_mid by n (mean, max):")
    print(B.groupby("n").gap_vs_mid.agg(["mean", "max", lambda g: (g > TOL).mean()]).rename(columns={"<lambda_0>": "frac_beaten"}).to_string())
    print("regime B gap_vs_lo by n (mean, max):")
    print(B.groupby("n").gap_vs_lo.agg(["mean", "max", lambda g: (g > TOL).mean()]).rename(columns={"<lambda_0>": "frac_beaten"}).to_string())
    print(f"done in {time.time()-t:.0f}s")
