# T5 — manuscript restructure plan (ICML 2027, 9 pages main text)

## Framing
Title: **Decision-Censored Learning: Sharp Ranking Bounds and Minimax
Prediction When the Label Exists Only Where Someone Decided to Look**
(short: "Decision-Censored Learning"). Abstract states the general problem
(decision-censored labels: credit, diagnostics, bail, moderation); medicine is
ONE of three domains, never the framing.

## Main text order (target page budget in parentheses)
1. Introduction (1.25) — problem, three failure modes of current practice, five
   contributions as bullets, Figure 1 = COMPAS coverage (real decisions, real
   censored labels) as the representative figure.
2. Setup (0.75) — DCSM(Γ), Lemma (MSM equivalence), Lemma (box), directional
   remark (2 lines).
3. Sharp ranking bounds (1.5) — rank identity (Thm), sharp interval + O(n log n)
   (Thm), unified class (Prop, 4 lines), corner evaluation is invalid (Prop,
   3 lines + pointer to the 3-point counterexample in the appendix).
4. Minimax rules (1.25) — DCL theorem (a)–(d) compressed, minimax-regret
   proposition, validity corollary (1 line), Remark: budgeted DCSM → appendix,
   ranking-optimality proposition (equal widths) + conjecture/counterexample
   (from T3), DCLRanker negative result in 3 lines.
5. Thm 5: estimated nuisances (1.0) — the two-part theorem (aggregate DR with
   product-rate bias; pointwise distribution-free via monotone interval
   arithmetic) + the diagnostic decomposition. (from T2)
6. Impossibility and falsifiability (0.5) — Thm (two-point, every n),
   Prop (Γ_min from leniency). Generalisation bound → one sentence + appendix.
7. UQ corollary (0.4) — compact cor:uq; figure in the experiments.
8. Experiments (2.0) — testbeds paragraph (Lending explicitly semi-synthetic;
   MIMIC arm: real column pending GPU-server run, simulator table moved to the
   appendix "Simulator validation"); Table: sharpness/validity; Table:
   learning (5 seeds ± CI); Table: Thm 5 coverage; Figure: falsification +
   UQ (combined 2-panel); break-even paragraph.
9. Negative results and limitations (0.35) — ranking scheme never beats the
   midpoint warm start; pointwise coverage with plug-in nuisances 45–58% →
   fixed by Thm 5 inflation (numbers by macro); MIMIC pending; exclusion
   restriction for Γ_min; time-at-risk on COMPAS.
Related work → 0.5 page in main text with the differentiation matrix moved to
the appendix (tab:related-matrix).

## Appendix
A Proofs (all). B Details: counterexample, budgeted DCSM, DCLRanker algorithm
and negative result, nuisance estimation, related-work matrix, simulator
validation (former MIMIC-sim numbers, clearly labelled `data_source: simulator`),
Korean cohort protocol (pending IRB), MIMIC ECG×ECHO protocol (pending run),
full COMPAS time-at-risk audit, reproducibility.

## Mechanics
- Every number via `\num...` macros from `paper/numbers.tex` (rule 2).
- `% PROVENANCE: real` markers on the COMPAS/MIMIC-real subsections; the
  checker refuses simulator macros there (rule 3).
- 5 seeds + 95% CI wherever a difference is claimed (rule 4).
- `\label{lastmainpage}` immediately before `\appendix` for the page-budget
  check in scripts/build_paper.sh.

## Outcome (2026-09-16)

Final main-text order (9.0 pages, ICML substitute style, 0 undefined
references, 0 overfull boxes; `bash scripts/build_paper.sh`):

1. Introduction (0.9 p) — general framing, COMPAS figure first, contributions.
2. Setup (0.7 p) — DCSM(Γ), Lemmas msm/box, directional remark.
3. Sharp partial identification of ranking metrics (1.2 p) — rank identity,
   sharp interval, corner evaluation; unified class moved to App. B.
4. Decision-censored learning (1.1 p) — Thm dcl, Prop regret, Prop midrank
   (equal widths) + counterexample summary; DRO/budget remarks and the validity
   corollary moved to App. B.
5. The nuisance bottleneck (1.0 p) — Thm 5 (a)-(c) + coverage table.
6. Γ is unavoidable — and falsifiable (0.6 p).
7. Generalisation (0.3 p; full statements in App. B_generalization).
8. Uncertainty quantification (0.6 p) — Cor uq + three-way decomposition.
9. Experiments (2.0 p) — COMPAS first (merged results table), sharpness,
   learning, falsification/break-even/generalisation/uncertainty paragraphs.
10. Related work (0.3 p; matrix + extended text in App. D).
11. Negative results, limitations and outlook (0.4 p).

Appendices (one-column): A proofs; B generalisation statements; B details
(counterexample, unified class, learning remarks, budgeted DCSM, ranking
algorithm, ranking search audit, nuisance estimation, reproducibility);
C additional results (full tables with CIs, COMPAS audit, hidden-signal
guard, falsification/misspecification, scaling, UQ table, nuisance variants,
simulator validation, medical-arm status); D related-work matrix.
