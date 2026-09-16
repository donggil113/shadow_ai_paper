# A_AUDIT.md — audit record for branch `claude/busy-allen-ws1ttc`

**Provenance.** The task brief referenced a Codex-produced `audit/A_AUDIT.md`
(sections 2 and 3) and `REVIEW.md`. Neither file existed in the repository at
commit `87d8f3e` or at any later commit before this one. Everything below was
produced in this session (2026-09-15/16) by the paper's authoring agent, from
the repository's own code, data and results. Numbers quoted here are copied
from `results/*.json` produced by the named scripts; the paper cites the same
values only through `paper/numbers.tex` macros.

## §1 Scope

| item | status |
|---|---|
| Real-data medical arm (MIMIC-IV-ECG × ECHO) | pipeline implemented and fixture-tested; **not run** (data root `/data/*` and PhysioNet credentials exist only on the user's server; physionet.org unreachable here). Exits 2 without data. Paper reports it as pending. |
| Simulator (MIMIC-sim) | appendix "Simulator validation" only; `check_provenance.py` fails the build if a `simulator` macro appears in the main text. |
| COMPAS | real decisions, recorded outcomes; lineage and time-at-risk audit in §2. |
| Lending Club | semi-synthetic (real X, Y; simulated funding decision); code-level guard that S ∉ X (`dcl/data/semisynthetic.py:assert_hidden_signal_excluded`, `tests/test_data_guards.py`). |
| SL-Bench | synthetic with exact oracle quantities. |
| Seeds | every comparison in the paper: 5 seeds, t-based 95% CI (`experiments/_common.py:ci95`); `verify_paper_numbers.py` fails if a CI-bearing macro with n<5 is cited. |

## §2 COMPAS lineage and time-at-risk correction

Script: `audit/compas_time_at_risk.py` → `results/compas_time_at_risk.json`
(`data_source: real`).

**Lineage.** `compas-scores-two-years.csv` (ProPublica, Broward County FL;
7,214 rows) → ProPublica's published filter (|days_b_screening_arrest| ≤ 30,
is_recid ≠ −1, charge degree ≠ 'O', score_text ≠ 'N/A'; 6,172) → rows with
parseable jail dates and index custody ≤ 180 days (6,054 = the paper's cohort).
T = 1 iff custody ≤ 2 days (62.5% released). The learner's X: age, priors,
juvenile counts, charge degree, sex, race (one-hot). Y: `two_year_recid`.

**Why "ground truth" was the wrong word.** Detention incapacitates. Median
index custody is 0.9 days for the released and 10.1 days (90th percentile 35
days) for the detained; `start` (ProPublica's start of the at-risk period)
correlates 0.75 with custody length. A detained defendant therefore has less
time in the community during the two-year window, which mechanically lowers
the detained group's recorded rearrest rate. The recorded outcome is what a
deployed model is scored on, so it stays primary, but it is a *recorded*
outcome, not a causal ground truth. The paper's wording was changed
accordingly (abstract, §1, §8, appendix).

**Correction.** Exposure-adjusted outcome `Y_exp = 1{event = 1 and end − start
≤ W}` with W = 730 days at risk; units with neither an event nor W days of
at-risk follow-up are dropped.

| W (days at risk) | kept | kept (detained) | p1 (released) | p0 (detained) | odds ratio |
|---|---|---|---|---|---|
| recorded (`two_year_recid`) | 100% | 100% | 0.392 | 0.548 | **1.885** |
| 365 | 86.4% | 83.5% | 0.280 | 0.437 | 1.999 |
| 540 | 83.5% | 80.4% | 0.344 | 0.510 | 1.981 |
| 730 | 81.9% | 78.6% | 0.380 | 0.566 | **2.135** |

**Diagnosis.** Holding exposure fixed *raises* the detained/released odds
ratio (1.885 → 2.135). Incapacitation was masking selection on unobservables,
not creating it; the recorded anchor Γ ≈ 1.9 is therefore conservative rather
than inflated. The paper reports both anchors and the coverage of the AUROC
interval under both outcomes (`experiments/exp6_realdata.py`, 5 seeds, both
`outcome="recorded"` and `outcome="exposure_adjusted"`).

**Changed claims.** "COMPAS lets us check the bounds against ground truth" →
"against recorded outcomes on decisions no one simulated; detention also
incapacitates, and the exposure-adjusted odds ratio is 2.1".

## §3 Ranking search (T3): is the midpoint ordering minimax-optimal?

Script: `audit/t3_ranking_search.py` → `audit/t3_counterexamples.csv`,
`results/exp9_ranking_search.json` (`data_source: synthetic`). Every
permutation of n ∈ {3,…,7} units is scored by the exact inner value
min_{p ∈ box} AUC(σ, p) from `dcl.auc_bounds.sharp_auc_interval`.

| regime | instances | midpoint ordering beaten | mean gap | max gap | lo ordering beaten |
|---|---|---|---|---|---|
| A: equal widths | 4,160 | **0** | 0 | 0 | 0 |
| B: unequal widths | 4,160 | 1,239 (29.8%) | 0.006 | 0.155 | 1,718 (41.3%) |

By n (regime B, fraction of instances where the midpoint ordering is beaten):
n=3: 16%, n=4: 25%, n=5: 49%, n=6: 59%, n=7: 83%. Mean gap stays ≈ 0.006 AUROC.
No fixed c makes the ordering by lo − c·width optimal on every instance
(best: c = 0, 58.7%); some member of the family lo + c(hi − lo), c ∈ [0,1],
attains the optimum in 94.3% of instances.

**Result.** (a) Equal widths: the midpoint ordering is optimal in every
instance, matching the proof (rank identity + rearrangement inequality;
`paper/sections/A_proofs.tex`, proof of Prop. midrank). (b) Unequal widths:
the conjecture "the midpoint ordering is optimal for worst-case AUROC" is
**refuted** by explicit counterexamples (largest gaps occur when several
intervals are clipped at the same upper endpoint). (c) On the benchmark
instances, whose widths vary smoothly with e(x), the measured gain of a
best-response scheme over the midpoint ordering is reported by
`experiments/exp10_ranker.py` (5 seeds).

**Changed claims.** "Whether x ↦ p̃(x) is exactly optimal for worst-case AUROC
— as the evidence suggests but we have not been able to prove — is ... open"
→ Proposition (equal widths, proved) + counterexample (unequal widths) +
recommendation unchanged (rank by the midpoint; the exact oracle checks any
proposed improvement).

## §4 Other refutations and corrections recorded in this session

| claim (before) | finding | claim (after) |
|---|---|---|
| Plug-in identified set is a per-unit confidence set | with cross-fitted nuisances it contains p(x) for ~44–55% of units (exp8, 5 seeds) | Theorem 5: aggregate endpoints are √n-estimable (DR); per-unit coverage requires nuisance intervals; inflated boxes reach ≥90% with 20 bins; the outcome model is the binding nuisance |
| Falsification recovers 34–61% of log Γ₀ | unchanged qualitatively; re-measured on 5-seed reruns; numbers are macros | (see `paper/numbers.tex`) |
| Medical column = MIMIC-sim | simulator numbers may not appear in real-data sections | medical column pending real run; simulator in appendix only |
