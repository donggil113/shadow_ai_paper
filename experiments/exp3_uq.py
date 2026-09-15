"""Experiment 3 -- the uncertainty-decomposition corollary.

Three panels, deliberately separated, because an earlier version of this script
conflated them and produced a misleading answer.

**Panel A (identification, oracle nuisances).**  The parts of the uncertainty corollary that
are theorems: with the true ``e`` and ``p1``, the observed-data regression
``p1(x)`` lies inside the identified set for every ``x``, and the true mean
aleatoric entropy lies inside its sharp identified interval.  Both must hold
exactly; this panel is a check on the implementation, not an experiment.

**Panel B (estimation).**  The same quantities with cross-fitted nuisances.  The
guarantees are now only as good as the nuisance estimates, and we report how
often they survive -- this is the honest cost of not knowing ``e`` and ``p1``.

**Panel C (the decomposition itself).**  How the reported epistemic term behaves
as ``n`` grows, against the censoring term (6) that no amount of data reduces.
We use two ensembles on purpose:

* a **converged, calibrated bootstrap ensemble** (logistic regression run to
  convergence), whose spread really does reflect sampling uncertainty and so
  contracts at the rate theory predicts;
* a **deep ensemble** of MLPs trained with a fixed iteration budget, the way
  practitioners actually build them.

The second is a cautionary panel: its "epistemic" term does *not* contract,
because it is dominated by optimisation noise rather than posterior
concentration, and its aleatoric term can fall outside the identified interval
entirely because the ensemble is overconfident.  That is a *different* pathology
from the one the uncertainty corollary describes -- and, unlike that one, it is
detectable from observed data by a calibration check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import Timer, save, save_table, stamp_provenance, ci95
from dcl.data import make_sl_bench
from dcl.nuisance import CrossFitNuisance
from dcl.sensitivity import outcome_bounds
from dcl.uq import (aleatoric_identified_interval, binary_entropy,
                    naive_decomposition, three_way_decomposition)


def _bootstrap_ensemble(Xtr, ytr, Xte, kind="logistic", n_members=20, seed=0):
    """Bagged, calibrated probability estimates."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    out = []
    for m in range(n_members):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        if len(np.unique(ytr[idx])) < 2:
            continue
        if kind == "logistic":
            base = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=5000, C=1.0))
            clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        else:
            base = make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                              random_state=seed * 100 + m))
            clf = base
        clf.fit(Xtr[idx], ytr[idx])
        out.append(clf.predict_proba(Xte)[:, 1])
    return np.clip(np.stack(out), 1e-6, 1 - 1e-6)


def panel_ab(sizes=(2000, 5000, 12000, 30000), gamma=2.5, seed=0):
    rows = []
    for n in sizes:
        ds = make_sl_bench(n=n, d=10, target_gamma=gamma, seed=seed)
        tr, te = ds.split(test_size=0.4, seed=seed)
        g0 = ds.oracle["gamma0"]
        o = te.oracle

        # --- Panel A: oracle nuisances (pure identification) ---
        lo, hi = outcome_bounds(o["p1"], o["e"], g0)
        a_lo, a_hi = aleatoric_identified_interval(lo, hi)
        true_al = float(np.mean(binary_entropy(o["p"])))
        naive_al_oracle = float(np.mean(binary_entropy(o["p1"])))

        # --- Panel B: cross-fitted nuisances (identification + estimation) ---
        cf = CrossFitNuisance(seed=seed)
        cf.fit_predict(tr.X, tr.T, tr.Y_obs)
        nte = cf.predict(te.X)
        lo_h, hi_h = outcome_bounds(nte.p1, nte.e, g0)
        ah_lo, ah_hi = aleatoric_identified_interval(lo_h, hi_h)

        rows.append(dict(
            n=n, gamma0=g0,
            # A
            oracle_p1_in_box=float(np.mean((o["p1"] >= lo - 1e-12)
                                           & (o["p1"] <= hi + 1e-12))),
            oracle_p_in_box=float(np.mean((o["p"] >= lo - 1e-12)
                                          & (o["p"] <= hi + 1e-12))),
            oracle_alea_lo=a_lo, oracle_alea_hi=a_hi,
            true_aleatoric=true_al,
            naive_aleatoric_oracle=naive_al_oracle,
            oracle_true_inside=bool(a_lo - 1e-9 <= true_al <= a_hi + 1e-9),
            oracle_naive_inside=bool(a_lo - 1e-9 <= naive_al_oracle <= a_hi + 1e-9),
            oracle_naive_bias=naive_al_oracle - true_al,
            # B
            est_p_in_box=float(np.mean((o["p"] >= lo_h - 1e-12)
                                       & (o["p"] <= hi_h + 1e-12))),
            est_alea_lo=ah_lo, est_alea_hi=ah_hi,
            est_true_inside=bool(ah_lo - 1e-9 <= true_al <= ah_hi + 1e-9),
            nuis_mae_e=float(np.mean(np.abs(nte.e - o["e"]))),
            nuis_mae_p1=float(np.mean(np.abs(nte.p1 - o["p1"]))),
        ))
        print(f"    [AB] n={n:>6,d} oracle p1-in-box={rows[-1]['oracle_p1_in_box']:.4f} "
              f"true-alea-inside={rows[-1]['oracle_true_inside']} "
              f"est-p-in-box={rows[-1]['est_p_in_box']:.4f}", flush=True)
    return pd.DataFrame(rows)


def panel_c(sizes=(2000, 5000, 12000, 30000, 70000), gamma=2.5, seed=0,
            kinds=("logistic", "mlp")):
    rows = []
    for n in sizes:
        ds = make_sl_bench(n=n, d=10, target_gamma=gamma, seed=seed)
        tr, te = ds.split(test_size=0.4, seed=seed)
        o = te.oracle
        lo, hi = outcome_bounds(o["p1"], o["e"], ds.oracle["gamma0"])
        sel = tr.T == 1
        for kind in kinds:
            P = _bootstrap_ensemble(tr.X[sel], tr.Y_obs[sel], te.X, kind=kind,
                                    n_members=20, seed=seed)
            nv = naive_decomposition(P)
            tw = three_way_decomposition(P, lo, hi)
            a_lo, a_hi = aleatoric_identified_interval(lo, hi)
            naive_al = float(np.mean(nv.aleatoric))
            rows.append(dict(
                n=n, ensemble=kind,
                naive_total=float(np.mean(nv.total)),
                naive_aleatoric=naive_al,
                naive_epistemic=float(np.mean(nv.epistemic)),
                censoring=float(np.mean(tw.censoring)),
                sampling=float(np.mean(tw.sampling)),
                credal_total=float(np.mean(tw.total)),
                ratio_epistemic_to_censoring=float(
                    np.mean(nv.epistemic) / max(np.mean(tw.censoring), 1e-12)),
                naive_inside_identified=bool(a_lo - 1e-9 <= naive_al <= a_hi + 1e-9),
                mean_abs_calibration_error=float(np.mean(np.abs(P.mean(0) - o["p1"]))),
            ))
            print(f"    [C ] n={n:>6,d} {kind:8s} epistemic={rows[-1]['naive_epistemic']:.5f} "
                  f"censoring={rows[-1]['censoring']:.5f} "
                  f"ratio={rows[-1]['ratio_epistemic_to_censoring']:.3f} "
                  f"calib_err={rows[-1]['mean_abs_calibration_error']:.4f}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with Timer("exp3 panels A/B"):
        ab = panel_ab()
    with Timer("exp3 panel C (5 seeds)"):
        cs = []
        for sd in range(5):
            ci_ = panel_c(seed=sd)
            ci_["seed"] = sd
            cs.append(ci_)
        c = pd.concat(cs, ignore_index=True)
    save_table("exp3_uq_identification", ab)
    save_table("exp3_uq_decomposition", c)

    print("\n=== Panel A: identification (oracle nuisances) ===")
    print(ab[["n", "gamma0", "oracle_p1_in_box", "oracle_p_in_box",
              "oracle_alea_lo", "oracle_alea_hi", "true_aleatoric",
              "naive_aleatoric_oracle", "oracle_naive_inside",
              "oracle_true_inside", "oracle_naive_bias"]].to_string(
        index=False, float_format=lambda x: f"{x:8.4f}"))
    print("\n=== Panel B: with cross-fitted nuisances ===")
    print(ab[["n", "est_p_in_box", "est_alea_lo", "est_alea_hi",
              "est_true_inside", "nuis_mae_e", "nuis_mae_p1"]].to_string(
        index=False, float_format=lambda x: f"{x:8.4f}"))
    print("\n=== Panel C: decomposition vs n ===")
    print(c.to_string(index=False, float_format=lambda x: f"{x:9.5f}"))

    agg = c.groupby(["ensemble", "n"], as_index=False)[
        ["naive_epistemic", "censoring", "ratio_epistemic_to_censoring",
         "mean_abs_calibration_error"]].mean()
    log = agg[agg.ensemble == "logistic"].sort_values("n")
    mlp = agg[agg.ensemble == "mlp"].sort_values("n")
    cis = {}
    for (ens, n_), g in c.groupby(["ensemble", "n"]):
        cis[f"{ens}_n{int(n_)}"] = {k: dict(zip(("mean", "ci95", "n"), ci95(g[k])))
                                    for k in ("naive_epistemic", "censoring",
                                              "ratio_epistemic_to_censoring")}
    def decay(sub):
        if len(sub) < 3 or (sub.naive_epistemic <= 0).any():
            return float("nan")
        return float(np.polyfit(np.log(sub.n.values),
                                np.log(sub.naive_epistemic.values), 1)[0])
    save("exp3_summary", dict(
        oracle_p1_always_in_box=float(ab.oracle_p1_in_box.min()),
        oracle_naive_always_inside=bool(ab.oracle_naive_inside.all()),
        oracle_true_always_inside=bool(ab.oracle_true_inside.all()),
        oracle_mean_aleatoric_bias=float(ab.oracle_naive_bias.mean()),
        est_p_in_box_min=float(ab.est_p_in_box.min()),
        epistemic_decay_exponent_logistic=decay(log),
        epistemic_decay_exponent_mlp=decay(mlp),
        epistemic_first_logistic=float(log.naive_epistemic.iloc[0]),
        epistemic_last_logistic=float(log.naive_epistemic.iloc[-1]),
        censoring_first_logistic=float(log.censoring.iloc[0]),
        censoring_last_logistic=float(log.censoring.iloc[-1]),
        ratio_last_logistic=float(log.ratio_epistemic_to_censoring.iloc[-1]),
        mlp_inside_identified=bool(c[c.ensemble == "mlp"].naive_inside_identified.all()),
        logistic_inside_identified=bool(c[c.ensemble == "logistic"].naive_inside_identified.all()),
        n_seeds=int(c.seed.nunique()),
        by_cell_ci=cis,
    ))
    stamp_provenance("exp3_summary", "synthetic")
