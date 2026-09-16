"""Experiment 0 -- the three-point counterexample to corner evaluation (Appendix).

Reproduces the numbers quoted in the appendix from the code path the paper
uses (sharp interval vs. corner evaluation), so they are macros, not typed.
"""
from __future__ import annotations

import numpy as np

from _common import save, stamp_provenance
from dcl.auc_bounds import naive_corner_interval, sharp_auc_interval


def auc_soft(scores, p):
    """Population AUROC with soft labels p on the uniform measure (ratio of expectations)."""
    s = np.asarray(scores, float); p = np.asarray(p, float); n = len(s)
    psi = (s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :])
    num = float((p[:, None] * (1 - p)[None, :] * psi).sum()) / n ** 2
    pi = p.mean()
    return num / (pi * (1 - pi))


if __name__ == "__main__":
    f = np.array([0.0, 1.0, 2.0])
    lo = np.array([0.59, 0.85, 0.48]); hi = np.array([0.98, 0.98, 0.97])
    mid = 0.5 * (lo + hi)
    corners = dict(lo=auc_soft(f, lo), mid=auc_soft(f, mid), hi=auc_soft(f, hi))
    nc = naive_corner_interval(f, lo, hi)
    res = sharp_auc_interval(f, lo, hi)
    p_star = np.asarray(res.p_upper)
    out = dict(
        scores=f.tolist(), lo=lo.tolist(), hi=hi.tolist(),
        auc_lo=corners["lo"], auc_mid=corners["mid"], auc_hi=corners["hi"],
        corner_lo=float(nc[0]), corner_hi=float(nc[1]), corner_width=float(nc[1] - nc[0]),
        sharp_lo=float(res.lower), sharp_hi=float(res.upper), sharp_width=float(res.width),
        p_star_upper=p_star.tolist(), auc_p_star=auc_soft(f, p_star),
        miss=float(res.upper - nc[1]),
    )
    save("exp0_counterexample", out)
    stamp_provenance("exp0_counterexample", "synthetic")
    print(out)
