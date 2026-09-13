# Decision-Censored Learning

Sharp ranking bounds and minimax prediction under selective labels.

Research code for the paper in [`paper/`](paper/). When the label `Y` is
recorded only where a decision maker chose to look — loans repaid only if
granted, diagnoses recorded only if a test was ordered, recidivism observed
only for defendants who were released — the deployment risk of a model is **not
identified**. This repository implements a sensitivity model for that regime and
the evaluation and learning machinery that goes with it.

```
        X ──────────────► T ∈ {0,1}  ──────► Y observed iff T = 1
        ▲                  ▲
        │                  │
     observed          S (the decision maker's private
     covariates          information — never observed)
```

## The sensitivity model

`DCSM(Γ)` bounds how far the censored units' outcome log-odds may differ from
the labelled units':

```
1/Γ  ≤  odds( P(Y=1 | X=x, T=0) ) / odds( P(Y=1 | X=x, T=1) )  ≤  Γ
```

`Γ = 1` is selection on observables (what IPW and AIPW assume); `Γ = ∞` is
Manski. In between, the identified set for the outcome regression `p(x)` is an
axis-aligned box, and everything becomes computable.

## What's here

| Result | Where | Verified by |
|---|---|---|
| **Rank identity** — `AUC(p) = (⟨p,R⟩ − π²/2) / (π(1−π))`, so AUROC is *linear* in the outcome regression at fixed prevalence | `dcl/auc_bounds.py` | exact match to the defining double sum and to `sklearn`, incl. ties and weights |
| **Sharp AUROC interval** in exact `O(n log n)` — continuous knapsack inside, ratio-of-quadratics line search outside | `dcl/auc_bounds.py` | agrees with independent projected-gradient search to `3.6e-6`; bounds *attained* to `1.2e-15` |
| **Corner evaluation is invalid** (not merely loose) | `dcl/auc_bounds.py` | covers the truth in 10% of benchmark cells vs 100% |
| **Decision-censored ERM** — worst case = midpoint risk + width-weighted margin penalty; *exactly convex*, no relaxation | `dcl/objectives.py` | convexity probe over all four losses; Sion saddle point matches closed form |
| **Interval-shrunk logit** (minimax risk) and the **entropy-difference-quotient rule** (minimax regret), both closed form | `dcl/objectives.py` | match grid minimisation to grid resolution |
| **Impossibility** — no estimator is consistent for deployment risk, at any `n` | `paper/sections/A_proofs.tex` | two-point construction |
| **Γ is falsifiable from below** given leniency variation | `dcl/falsify.py` | recovers the true `Γ₀` to 3 decimals in the ideal case; never exceeds it |
| **Three-way uncertainty decomposition** (aleatoric / epistemic / *censoring*) | `dcl/uq.py` | `p₁(x)` proved and tested to lie inside the identified set always |

## Install and run

```bash
pip install -e .                       # or: pip install numpy scipy pandas scikit-learn torch matplotlib
python scripts/download_data.py        # public data -> data/raw/ (prints sha256)
pytest tests/ -q                       # 41 tests
python experiments/run_all.py          # every experiment -> results/
python experiments/make_figures.py     # every figure -> paper/figures/
```

## Quick start

```python
from dcl.data import make_sl_bench
from dcl.nuisance import CrossFitNuisance
from dcl.objectives import dcl_bayes_score, worstcase_risk, minimax_regret
from dcl.auc_bounds import sharp_auc_interval

ds = make_sl_bench(n=16000, target_gamma=3.0)     # T, Y_obs (NaN where T == 0)

nu  = CrossFitNuisance().fit_predict(ds.X, ds.T, ds.Y_obs)   # e_hat, p1_hat
box = nu.box(gamma=2.0)                                      # identified set

scores = dcl_bayes_score(box.lo, box.hi, criterion="regret") # the DCL model
print(sharp_auc_interval(scores, box.lo, box.hi))            # sharp AUROC interval
print(worstcase_risk(scores, box.lo, box.hi))                # certified risk
print(minimax_regret(scores, box.lo, box.hi))                # certified regret
```

`box.lo`/`box.hi` are the honest output: an interval for `P(Y=1|X=x)`, not a
point. `box.delta` is the per-unit identification width — how much the incumbent
policy could be hiding at `x`.

## Datasets

| name | decisions | censored labels | ground truth |
|---|---|---|---|
| `sl_bench` | simulated | simulated | exact, by quadrature |
| `lending_club` | simulated policy on **real** features | **real** outcomes | yes |
| `mimic_sim` | simulated | simulated | yes |
| `mimic` | **real** (credentialed) | **real** | no |
| `compas` | **real** judges | **real**, recorded anyway | **yes** |
| `creditcard` | **real** accept/reject | structurally unobservable | no, and never |

`compas` is the important one: real decisions *and* recorded outcomes on both
sides of them, so the bounds can be checked against the truth.

`mimic` requires credentialed PhysioNet access. `dcl/data/mimic.py` implements
the full cohort-building pipeline against the real file layout and documents the
download; `make_mimic_sim()` is the simulator used for the reported numbers.
**The medical results in the paper come from the simulator**, and are labelled
as such.

## Layout

```
dcl/
  sensitivity.py   DCSM(Γ); the identified box; directional variants
  auc_bounds.py    Theorem 2: rank identity, sharp O(n log n) interval, verifiers
  objectives.py    Theorem 6: worst-case risk, closed-form Bayes acts, budgeted DCSM
  ranking.py       worst-case-AUROC training via the Theorem 2 oracle
  uq.py            three-way uncertainty decomposition
  falsify.py       Γ lower bound from leniency variation
  nuisance.py      cross-fitted, calibrated ê and p̂₁
  models.py        plug-in and parametric DCL learners
  baselines.py     ERM / IPW / AIPW / imputation / Heckman bivariate probit / Manski
  evaluation.py    deployment vs observed metrics; Lakkaraju contraction
  harness.py       train/test protocol (out-of-sample by construction)
  data/            SL-Bench, Lending Club, MIMIC, COMPAS, AER-CreditCard
experiments/       exp1..exp6 + figures
paper/             LaTeX source, full proofs in sections/A_proofs.tex
```

## Notes on correctness

Three bugs found during development are worth repeating because they are easy to
reproduce:

1. **In-sample evaluation inverts the results.** A flexible model fit on the
   labelled units and scored on the same units beats an oracle fit on the
   complete labels, and its empirical AUROC can exceed the sharp upper bound by
   0.23 — a property of memorisation, not a failure of the bound. Everything here
   is out of sample.
2. **The relevant propensity is marginal over the decision maker.** A learner
   sees only `X`, so the nuisance is `P(T=1|X)` averaged over case assignment,
   not `P(T=1|X,Z)`. Conditioning on the realised decision maker made `corr(ê,e)`
   look like 0.42 when it was really 0.93.
3. **`Γ_min ≤ Γ₀` is about the *conditional* `Γ₀`.** Marginalising over decision
   makers averages the tilts and shrinks the log odds ratio, so the falsification
   bound must be compared against the within-decision-maker parameter.
