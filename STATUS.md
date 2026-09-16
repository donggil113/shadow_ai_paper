# STATUS

Claim ledger (CLAUDE.md rule 7). Evidence keys of the form `results/<file>.json:<path>` are resolved at generation time, so the value shown is the one in the repository.

| claim_id | statement | type | evidence | data_source | seeds | status |
|---|---|---|---|---|---|---|
| C1 | DCSM(Gamma) is equivalent to Tan's MSM with Gamma = Lambda^2; identified set is a box of per-unit intervals (Lemmas msm, box) | thm | `paper/sections/A_proofs.tex: proofs of lem:msm, lem:box` | n/a | n/a | **proved** |
| C2 | Rank identity: AUC(p) = (E[pR] - pi^2/2)/(pi(1-pi)); AUROC is affine in p at fixed prevalence (Thm rank-identity) | thm | `paper/sections/A_proofs.tex: proof of thm:rank-identity` | n/a | n/a | **proved** |
| C3 | Sharp AUROC interval computed exactly in O(n log n); endpoints attained (Thm sharp-auc) | thm | `results/exp1_summary.json:max_abs_gap` = 3.578e-06 | synthetic | 5 | **verified** |
| C4 | Sharp interval covers the deployment AUROC whenever Gamma >= Gamma_0; corner evaluation does not (Prop corner) | exp | `results/exp1_summary.json:coverage_sharp_pop` = 1 | synthetic | 5 | **verified** |
| C5 | Worst-case risk over the identified set = midpoint risk + width-weighted margin penalty; exactly convex; saddle point; closed-form Bayes act (Thm dcl) | thm | `paper/sections/A_proofs.tex: proof of thm:dcl` | n/a | n/a | **proved** |
| C6 | Minimax-regret rule logit q* = (H(lo)-H(hi))/(hi-lo) (Prop regret) | prop | `paper/sections/A_proofs.tex: proof of prop:regret` | n/a | n/a | **proved** |
| C7 | Minimax-regret rule has lower realised risk than ERM/IPW/AIPW and the oracle on Lending Club, with certified regret an order of magnitude tighter | exp | `results/exp2_summary.json:datasets.lending_club.DCL-reg(G=2).risk` = 0.2005±0.013 (n=5) | semi-synthetic | 5 | **verified** |
| C8 | On COMPAS (real decisions, recorded outcomes) the sharp interval covers the recorded AUROC at the data's own odds ratio and misses at Gamma = 1 | exp | `results/exp6_summary.json:coverage.recorded.at_own_or.covers` = 1±0 (n=5) | real | 5 | **verified** |
| C9 | No estimator is uniformly consistent for deployment risk over Gamma; Gamma is not identified from (X,T,TY) (Thm impossible, Cor) | thm | `paper/sections/A_proofs.tex: proof of thm:impossible` | n/a | n/a | **proved** |
| C10 | Leniency variation gives an identified lower bound Gamma_min <= Gamma_0^cond; valid in every configuration and never refutes MAR (Prop falsify) | exp | `results/exp4_summary.json:falsification_always_valid` = True | synthetic | 5 | **verified** |
| C11 | Excess worst-case risk bound with complexity term linear in log Gamma; empirically the uniform deviation decays at n^-1/2 and its coefficient scales with log Gamma (Thm generalization, Cor loggamma) | thm | `results/exp5_summary.json:r2_vs_log_gamma` = 0.8916 | synthetic | 8 | **verified** |
| C12 | Under decision censoring the aleatoric/epistemic decomposition is biased by at least the censoring term minus the epistemic term, the bias is undetectable from observed data, and the epistemic term vanishes with n while the censoring term does not (Cor uq) | prop | `results/exp3_summary.json:ratio_last_logistic` = 0.005168 | synthetic | 5 | **verified** |
| C13 | Theorem 5(a): one-step estimator of the identified prevalence endpoints has second-order bias (product term + squared outcome term); outer interval covers the identified interval | thm | `results/exp8_summary.json:datasets.lending_club.dr_covers_prevalence_interval` = 1±0 (n=5) | semi-synthetic | 5 | **verified** |
| C14 | Theorem 5(b,c): inflated per-unit boxes from binned exact nuisance intervals cover p(x) at the nominal level; the plug-in box does not | exp | `results/exp8_summary.json:datasets.lending_club.cov_bins20` = 0.9716±0.009 (n=5) | semi-synthetic | 5 | **verified** |
| C15 | Outcome model, not propensity, is the binding nuisance (oracle e barely helps; oracle p1 restores coverage) | exp | `results/exp8_summary.json:datasets.sl_bench.cov_est_e_oracle_p1` = 0.8968±0.057 (n=5) | synthetic | 5 | **verified** |
| C16 | Plug-in (midpoint) ordering maximises worst- and best-case AUROC when all identified intervals have equal width (Prop midrank) | prop | `results/exp9_ranking_search.json:equal_violations` = 0 | synthetic | n/a | **proved** |
| C17 | The midpoint ordering is minimax-optimal for AUROC in general | prop | `results/exp9_ranking_search.json:unequal_mid_beaten_frac` = 0.2978 | synthetic | n/a | **refuted** |
| C18 | DCLRanker (best response against the exact oracle) does not improve on the midpoint ordering by more than a negligible margin on the benchmarks | exp | `results/exp10_summary.json:ranker_max_gain_over_midpoint` = 0.0002598 | synthetic | 5 | **verified** |
| C19 | Lending Club hidden signal S (interest rate / sub-grade) and its deterministic derivatives are excluded from X, enforced in code | exp | `tests/test_data_guards.py: test_lending_hidden_signal_not_in_X` | semi-synthetic | n/a | **verified** |
| C20 | COMPAS time-at-risk: holding exposure fixed raises the detained/released odds ratio (incapacitation masks selection) | exp | `results/compas_time_at_risk.json:exposure_adjusted.730.odds_ratio` = 2.135 | real | n/a | **verified** |
| C21 | H_A1: Gamma_min in [1.5, 3.0] on MIMIC-IV-ECG x ECHO; EchoNext-architecture observed AUROC has sharp [L,U] and break-even Gamma* on real data | exp | `experiments/exp7_mimic_ecg_echo.py (pending run on the GPU server; exits 2 here)` | real | 5 | **open** |
| C22 | Korean screening cohort: (i) US-model interval contains Korean realised AUROC, (ii) Gamma differs Korea vs US | exp | `scripts/korea_cohort/analyze_korea_cohort.py (pending IRB)` | real | 5 | **open** |

## Changed claims (before → after)
- **C8** — before: COMPAS provides ground truth; coverage 100% at Gamma = 1.885 (3 seeds)  →  after: COMPAS provides recorded outcomes; exposure-adjusted odds ratio 2.135 (incapacitation masked selection); coverage reported under both outcome definitions at 5 seeds
- **C14** — before: the identified box with cross-fitted nuisances contains the true p(x) (implicitly assumed)  →  after: plug-in box covers ~50% of units; inflated box reaches >=90% with 20 bins; outcome model is the binding nuisance
- **C17** — before: conjectured optimal in general (evidence suggested, unproved)  →  after: false for unequal widths (exhaustive counterexamples, mean gap 0.006 AUROC); proved for equal widths; midpoint remains the recommended default

