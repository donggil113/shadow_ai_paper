"""Regenerate every figure in the paper from the saved result tables."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from _common import FIGURES, RESULTS
from _style import C, DASHES, INK, INK2, MARKERS, MUTED, SLOTS, apply_style, grid, sequential

import matplotlib.pyplot as plt

apply_style()


def _load(name):
    p = os.path.join(RESULTS, f"{name}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  -> paper/figures/{name}.pdf")


# --------------------------------------------------------------------------- #
def fig1_sharpness():
    """Interval position/width vs Gamma, and coverage -- the sharp-interval figure."""
    df = _load("exp1b_validity")
    if df is None:
        return
    # Panel (a) shows ONE instance family so that a single true-AUROC reference
    # line is meaningful; panel (b) aggregates over all of them.
    targets = sorted(df.gamma0.unique(), key=lambda v: abs(v - 3.0))
    one = df[np.isclose(df.gamma0, targets[0])]
    g = one.groupby("gamma").agg(
        true_auc=("true_auc_pop", "mean"),
        sharp_lo=("sharp_lo", "mean"), sharp_hi=("sharp_hi", "mean"),
        naive_lo=("naive_lo", "mean"), naive_hi=("naive_hi", "mean"),
        sep_lo=("sep_lo", "mean"), sep_hi=("sep_hi", "mean")).reset_index()
    gam = g["gamma"].values
    x = np.arange(len(gam), dtype=float)
    g0 = float(one.gamma0.iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), width_ratios=[1.5, 1])
    ax = axes[0]
    for key, lab, col, off in [("sep", "separate bounds", SLOTS[2], -0.26),
                               ("sharp", "sharp bound", SLOTS[0], 0.0),
                               ("naive", "corner evaluation", SLOTS[1], +0.26)]:
        lo, hi = g[f"{key}_lo"].values, g[f"{key}_hi"].values
        for i in range(len(gam)):
            ax.plot([x[i] + off] * 2, [lo[i], hi[i]], color=col, lw=3.4,
                    solid_capstyle="round", zorder=4)
            if hi[i] - lo[i] < 0.004:
                ax.plot(x[i] + off, lo[i], marker="o", ms=3.4, color=col, zorder=5)
        ax.plot([], [], color=col, lw=3.4, solid_capstyle="round", label=lab)

    ax.axhline(g["true_auc"].iloc[0], color=INK, lw=1.1, zorder=6)
    ax.annotate("true deployment AUROC",
                (len(gam) - 0.45, g["true_auc"].iloc[0]),
                xytext=(0, 3), textcoords="offset points", ha="right",
                fontsize=7.2, color=INK, annotation_clip=False)
    ax.axvspan(-0.6, np.interp(g0, gam, x) , color=C["red"], alpha=.045,
               zorder=0, linewidth=0)
    ax.annotate(rf"$\Gamma < \Gamma_0 = {g0:.1f}$" "\nassumption too strong",
                (-0.5, 0.055), fontsize=6.8, color=MUTED, ha="left")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in gam])
    ax.set_xlim(-0.6, len(gam) - 0.4); ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"assumed $\Gamma$")
    ax.set_ylabel("AUROC")
    grid(ax)
    ax.legend(loc="lower left", ncol=3, fontsize=7.2, borderaxespad=0,
              bbox_to_anchor=(0, 1.005))
    ax.set_title("a", loc="left", pad=20)

    ax = axes[1]
    ok = df[df.gamma_sufficient]
    names = ["sharp\nbound", "separate\nbounds", "corner\neval."]
    vals = [ok.sharp_covers_pop.mean(), ok.sep_covers_pop.mean(),
            ok.naive_covers_pop.mean()]
    widths = [ok.sharp_w.mean(), ok.sep_w.mean(), ok.naive_w.mean()]
    b_ = ax.bar(names, vals, color=[SLOTS[0], SLOTS[2], SLOTS[1]], width=.55,
                linewidth=0, zorder=3)
    for r, v, w in zip(b_, vals, widths):
        ax.annotate(f"{v:.0%}", (r.get_x() + r.get_width() / 2, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK, weight="bold")
        ax.annotate(f"width\n{w:.2f}", (r.get_x() + r.get_width() / 2,
                                        max(v - 0.06, 0.06)),
                    ha="center", va="top", fontsize=6.8, color="white",
                    weight="bold")
    ax.axhline(1.0, color=MUTED, lw=.8, ls=":")
    ax.set_ylim(0, 1.17); ax.set_yticks([0, .5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_ylabel("coverage of the truth")
    ax.set_title(r"b   coverage when $\Gamma \geq \Gamma_0$", loc="left", pad=20)
    grid(ax)
    _save(fig, "fig1_sharpness")


def fig2_compas():
    """Real decisions: coverage on COMPAS as a function of the assumed Gamma."""
    df = _load("exp6a_compas_coverage")
    if df is None:
        return
    if "outcome" in df:                       # 5-seed rerun stores both outcome definitions
        df = df[df.outcome == "recorded"]
    or_marg = float(df.marginal_odds_ratio.iloc[0]) if "marginal_odds_ratio" in df else float("nan")
    g = df.groupby("gamma").agg(
        lo=("sharp_lo", "mean"), hi=("sharp_hi", "mean"),
        true_auc=("true_auc", "mean"), obs_auc=("observed_auc", "mean"),
        cover_rate=("covers", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    x = np.arange(len(g))
    covered = g['cover_rate'].values >= 1.0
    for i in range(len(g)):
        col = SLOTS[0] if covered[i] else MUTED
        ax.plot([x[i], x[i]], [g['lo'][i], g['hi'][i]], color=col,
                lw=5, solid_capstyle="round", alpha=.9 if covered[i] else .5,
                zorder=3)
        if g['hi'][i] - g['lo'][i] < 0.004:      # Gamma = 1 collapses to a point
            ax.plot(x[i], g['lo'][i], marker="o", ms=5, color=col, zorder=4)
    ax.axhline(g["true_auc"].iloc[0], color=INK, lw=1.2, zorder=5,
               label="recorded deployment AUROC")
    ax.axhline(g["obs_auc"].iloc[0], color=SLOTS[1], lw=1.2, ls="--", zorder=5,
               label="AUROC on labelled units only")
    for i, c in enumerate(covered):
        ax.annotate("covers" if c else "misses", (x[i], g['hi'][i]),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=6.4, color=INK2 if c else MUTED)
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in g["gamma"]])
    ax.set_xlabel(rf"assumed $\Gamma$   ({or_marg:.3f} = the data's own odds ratio)")
    ax.set_ylabel("AUROC")
    ax.set_title("COMPAS: real decisions, recorded censored outcomes")
    ax.set_ylim(0.44, 0.86)
    grid(ax)
    ax.legend(loc="lower left", fontsize=7.0)
    _save(fig, "fig2_compas")


def fig3_learning():
    """The robustness trade-off: realised risk against certified regret."""
    df = _load("exp2_learning_summary")
    if df is None:
        return
    sets = sorted(df.dataset.unique())

    def fam(m):
        if m.startswith("DCL-dir"):
            return 3
        if m.startswith("DCL-reg"):
            return 2
        if m.startswith("DCL") or m.startswith("Manski"):
            return 1
        if m.startswith("ORACLE"):
            return 4
        return 0

    labels = {0: "baselines (ERM / IPW / AIPW / imputation / Heckman)",
              1: "DCL, minimax risk", 2: "DCL, minimax regret",
              3: "DCL, directional", 4: "oracle (uncensored labels)"}
    fig, axes = plt.subplots(1, len(sets), figsize=(2.55 * len(sets), 2.85))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, sets):
        sub = df[df.dataset == ds].copy()
        sub["fam"] = sub.method.map(fam)
        for k in sorted(sub.fam.unique()):
            t = sub[sub.fam == k]
            ax.scatter(t.risk, t.minimax_regret, s=28,
                       color=SLOTS[k] if k < len(SLOTS) else INK,
                       marker=MARKERS[k % len(MARKERS)], linewidths=0,
                       alpha=.9, zorder=4, label=labels[k] if ax is axes[0] else None)
        best = sub.loc[sub.minimax_regret.idxmin()]
        ax.annotate(best.method, (best.risk, best.minimax_regret),
                    xytext=(6, -1), textcoords="offset points",
                    fontsize=6.8, color=INK2, ha="left")
        base = sub[sub.fam == 0]
        if len(base):
            w = base.loc[base.minimax_regret.idxmax()]
            ax.annotate(w.method, (w.risk, w.minimax_regret),
                        xytext=(-6, 2), textcoords="offset points",
                        fontsize=6.8, color=INK2, ha="right")
        ax.set_yscale("log")
        ax.set_title({"sl_bench": "SL-Bench (synthetic)",
                      "lending_club": "Lending Club",
                      "mimic_sim": "MIMIC-sim"}.get(ds, ds))
        ax.set_xlabel("realised deployment risk")
        grid(ax, axis="both")
    axes[0].set_ylabel("certified minimax regret (nats)")
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.5, -0.20))
    fig.suptitle("Lower-left is better: low realised risk AND a tight certificate",
                 x=0.5, y=1.04, fontsize=9, color=INK2)
    _save(fig, "fig3_learning")


def fig4_uq():
    """Epistemic uncertainty vanishes; identification uncertainty does not (5 seeds)."""
    df = _load("exp3_uq_decomposition")
    if df is None:
        return
    if "seed" not in df.columns:
        df = df.assign(seed=0)
    fig, ax = plt.subplots(figsize=(4.1, 2.8))

    def band(sub, col, color, marker, ls, label, lw=1.6):
        g = sub.groupby("n")[col]
        m, sd, k = g.mean(), g.std(ddof=1).fillna(0.0), g.count()
        from scipy import stats
        hw = stats.t.ppf(0.975, np.maximum(k - 1, 1)) * sd / np.sqrt(k)
        ax.plot(m.index, m.values, color=color, marker=marker, ls=ls, lw=lw,
                zorder=4, label=label)
        ax.fill_between(m.index, np.maximum(m - hw, 1e-6), m + hw, color=color,
                        alpha=.15, linewidth=0, zorder=3)
        return m

    log = df[df.ensemble == "logistic"]
    m_log = band(log, "naive_epistemic", SLOTS[0], MARKERS[0], DASHES[0],
                 "reported epistemic (converged ensemble)")
    band(log, "censoring", SLOTS[1], MARKERS[1], DASHES[1], "censoring term")
    if (df.ensemble == "mlp").any():
        band(df[df.ensemble == "mlp"], "naive_epistemic", MUTED, MARKERS[2], DASHES[2],
             "reported epistemic (fixed-budget deep ensemble)", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(list(m_log.index.values))
    ax.set_xticklabels([f"{int(v):,}" for v in m_log.index.values], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("training units $n$"); ax.set_ylabel("nats  (mean $\\pm$ 95% CI, 5 seeds)")
    ax.set_title("More data cannot shrink what censoring hides")
    grid(ax, axis="both")
    ax.legend(loc="lower left", fontsize=6.8, bbox_to_anchor=(-0.01, -0.02))
    _save(fig, "fig4_uq")


def fig8_nuisance():
    """Coverage under estimated nuisances: plug-in vs inflated boxes (Thm 5)."""
    df = _load("exp8_nuisance_coverage_raw")
    if df is None:
        return
    from scipy import stats
    cfgs = [("plugin", "plug-in"), ("bins10", "10 bins"), ("bins20", "20 bins"),
            ("bins10_smooth", "10 bins\n+ margin"), ("bins20_smooth", "20 bins\n+ margin")]
    sets = [d for d in ("sl_bench", "lending_club") if d in df.dataset.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    x = np.arange(len(cfgs)); w = 0.36
    for j, ds in enumerate(sets):
        sub = df[df.dataset == ds]
        means, hws = [], []
        for key, _ in cfgs:
            v = sub[f"cov_{key}"].dropna().values
            means.append(v.mean())
            hws.append(stats.t.ppf(0.975, max(len(v) - 1, 1)) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        ax.bar(x + (j - 0.5) * w, means, width=w, color=SLOTS[j], linewidth=0, zorder=3,
               label={"sl_bench": "SL-Bench (exact $p$)", "lending_club": "Lending Club (oracle $p$)"}[ds])
        ax.errorbar(x + (j - 0.5) * w, means, yerr=hws, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
    ax.axhline(0.9, color=MUTED, ls=":", lw=1)
    ax.annotate("target 90%", (len(cfgs) - 0.55, 0.905), ha="right", fontsize=7, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in cfgs], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("pointwise coverage of true $p(x)$")
    ax.set_title("a  Coverage")
    grid(ax); ax.legend(loc="upper left", fontsize=6.8)

    ax = axes[1]
    for j, ds in enumerate(sets):
        sub = df[df.dataset == ds]
        wid = [sub[f"width_{k}"].mean() for k, _ in cfgs]
        cov = [sub[f"cov_{k}"].mean() for k, _ in cfgs]
        ax.plot(wid, cov, color=SLOTS[j], marker=MARKERS[j], ls=DASHES[j], zorder=4)
        for (k, lab), wv, cv in zip(cfgs, wid, cov):
            if k in ("plugin", "bins20_smooth"):
                ax.annotate(lab.replace("\n", " "), (wv, cv), xytext=(4, -9 if k == "plugin" else 4),
                            textcoords="offset points", fontsize=6.5, color=INK2)
    ax.axhline(0.9, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("mean box width"); ax.set_ylabel("coverage")
    ax.set_title("b  The price of validity")
    grid(ax, axis="both")
    _save(fig, "fig8_nuisance")


def fig5_generalization():
    """Generalisation: n^{-1/2} decay of the uniform deviation, coefficient ~ log Gamma."""
    dev = _load("exp5a_deviation")
    sc = _load("exp5_scaling")
    if dev is None or sc is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    gammas = sorted(dev.gamma.unique())
    ramp = sequential(len(gammas))
    n = np.array(sorted(dev.n.unique()), dtype=float)
    # Six nearly-coincident curves: label only the extremes of the ramp and say
    # which way it runs, rather than crowding six end labels together.
    for col, gm in zip(ramp, gammas):
        s = dev[dev.gamma == gm].groupby("n").deviation.mean()
        ax.plot(s.index, s.values, color=col, marker="o", ms=3.2, lw=1.5,
                zorder=4 if gm in (gammas[0], gammas[-1]) else 3)
        if gm in (gammas[0], gammas[-1]):
            ax.annotate(rf"$\Gamma={gm:g}$", (s.index[-1], s.values[-1]),
                        xytext=(5, -3 if gm == gammas[0] else 3),
                        textcoords="offset points", va="center", fontsize=7,
                        color=INK2)
    ref = dev[dev.gamma == gammas[-1]].groupby("n").deviation.mean().values[0]
    ax.plot(n, ref * (n / n[0]) ** -0.5, color=MUTED, ls=":", lw=1.1, zorder=2)
    ax.annotate(r"$n^{-1/2}$", (n[len(n) // 2], ref * (n[len(n) // 2] / n[0]) ** -0.5),
                xytext=(2, -11), textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(list(n)); ax.set_xticklabels([f"{int(v):,}" for v in n], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel(r"training units $n$   (lighter $=$ smaller $\Gamma$)")
    ax.set_ylabel(r"$\sup_{f}|\hat{R}_\Gamma(f)-\bar{R}_\Gamma(f)|$")
    ax.set_title("a  deviation vs $n$", loc="left")
    ax.set_xlim(right=n[-1] * 3.0)
    grid(ax, axis="both")

    ax = axes[1]
    b, a_ = np.polyfit(sc.log_gamma, sc.coefficient, 1)
    xs = np.linspace(sc.log_gamma.min(), sc.log_gamma.max(), 50)
    ax.plot(xs, a_ + b * xs, color=SLOTS[0], lw=1.3, ls="--", zorder=3)
    ax.scatter(sc.log_gamma, sc.coefficient, s=36, color=SLOTS[0],
               linewidths=0, zorder=4)
    for _, r in sc.iterrows():
        ax.annotate(rf"${r.gamma:g}$", (r.log_gamma, r.coefficient),
                    xytext=(3, -8), textcoords="offset points",
                    fontsize=6.5, color=INK2)
    import json
    p_ = os.path.join(RESULTS, "exp5_summary.json")
    if os.path.exists(p_):
        j = json.load(open(p_))
        ax.annotate(rf"vs $\log\Gamma$:  $R^2={j['r2_vs_log_gamma']:.2f}$"
                    "\n"
                    rf"vs $\Gamma$:  $R^2={j['r2_vs_gamma']:.2f}$",
                    (0.04, 0.96), xycoords="axes fraction", va="top",
                    fontsize=7.5, color=INK2)
    ax.set_xlabel(r"$\log \Gamma$   (point labels give $\Gamma$)")
    ax.set_ylabel(r"coefficient of $n^{-1/2}$")
    ax.set_title(r"b  coefficient vs $\log\Gamma$", loc="left")
    grid(ax, axis="both")
    _save(fig, "fig5_generalization")


def fig6_falsification():
    """Falsification: Gamma_min is a valid, informative lower bound."""
    df = _load("exp4a_falsification")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    hets = sorted(df.kappa_heterogeneity.unique())
    for i, het in enumerate(hets):
        s = df[df.kappa_heterogeneity == het]
        ax.scatter(s.gamma0_cond, s.gamma_min, s=30, color=SLOTS[i],
                   marker=MARKERS[i], linewidths=0, alpha=.85, zorder=4,
                   label=("homogeneous" if het == 0 else "heterogeneous")
                         + " reliance on $S$")
    lim = [1.0, max(df.gamma0_cond.max(), df.gamma_min.max()) * 1.05]
    ax.plot(lim, lim, color=MUTED, ls=":", lw=1.1, zorder=2)
    mid = 0.55 * lim[1]
    ax.annotate(r"$\Gamma_{\min}=\Gamma_0$", (mid, mid), xytext=(-3, 6),
                textcoords="offset points", ha="right", rotation=45,
                rotation_mode="anchor", fontsize=7, color=MUTED)
    ax.fill_between(lim, lim, [lim[1] * 1.3] * 2, color=C["red"], alpha=.05,
                    zorder=1, linewidth=0)
    ax.annotate("infeasible region:\n$\\Gamma_{\\min}$ would not be a lower bound",
                (0.05, 0.94), xycoords="axes fraction", fontsize=7,
                color=MUTED, va="top")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"true conditional $\Gamma_0$")
    ax.set_ylabel(r"identified lower bound $\Gamma_{\min}$")
    ax.set_title(r"Leniency variation falsifies small $\Gamma$")
    grid(ax, axis="both")
    # The upper-left is empty by construction (it is the infeasible region), so
    # the legend costs no data ink there.
    ax.legend(loc="upper left", fontsize=6.8, bbox_to_anchor=(0.02, 0.84))
    _save(fig, "fig6_falsification")


def fig7_bayes_act():
    """The two closed-form DCL rules, drawn."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dcl.objectives import dcl_bayes_score
    from dcl.sensitivity import expit, outcome_bounds

    p1 = np.linspace(0.01, 0.99, 601)
    e = np.full_like(p1, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, gamma in zip(axes, (2.0, 5.0)):
        lo, hi = outcome_bounds(p1, e, gamma)
        ax.fill_between(p1, lo, hi, color=SLOTS[0], alpha=.13, linewidth=0,
                        zorder=2, label="identified set $[\\underline{p},\\overline{p}]$")
        ax.plot(p1, expit(dcl_bayes_score(lo, hi, "risk")), color=SLOTS[0],
                ls=DASHES[0], zorder=5, label="minimax-risk rule")
        ax.plot(p1, expit(dcl_bayes_score(lo, hi, "regret")), color=SLOTS[1],
                ls=DASHES[1], zorder=5, label="minimax-regret rule")
        ax.plot(p1, p1, color=MUTED, ls=":", lw=1.1, zorder=3,
                label="naive ($\\Gamma=1$)")
        band = (lo <= .5) & (hi >= .5)
        if band.any():
            ax.axvspan(p1[band][0], p1[band][-1], color=C["yellow"], alpha=.10,
                       zorder=1, linewidth=0)
            ax.annotate("abstention band", (p1[band].mean(), 0.03),
                        ha="center", fontsize=6.8, color=INK2)
        ax.set_title(rf"$\Gamma={gamma:g}$,  $e(x)=0.5$")
        ax.set_xlabel(r"observed-data regression $p_1(x)$")
        grid(ax, axis="both")
    axes[0].set_ylabel("predicted probability")
    axes[0].legend(loc="upper left", fontsize=6.8)
    fig.suptitle("The DCL rules shrink toward the decision boundary by the identification width",
                 x=0.5, y=1.03, fontsize=9, color=INK2)
    _save(fig, "fig7_rules")


if __name__ == "__main__":
    for fn in (fig7_bayes_act, fig1_sharpness, fig2_compas, fig3_learning,
               fig4_uq, fig5_generalization, fig6_falsification, fig8_nuisance):
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {exc}")
