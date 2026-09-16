# Related-work matrix (produced in this session, 2026-09-16)

Provenance note: the task brief referred to a Codex-produced `REVIEW.md`. No such
file existed in the repository at commit 87d8f3e or later; this matrix was
produced by the paper's authoring agent from the cited works' published
statements and is the source of `paper/sections/09b_related_matrix.tex`.

| work | assumption | target | sharp | ranking metric | learns predictor | Gamma testable | nuisance coverage | relation to this paper |
|---|---|---|---|---|---|---|---|---|
| Manski (1990, 2003) | none | mean outcome / conditional mean | yes | no | no | n/a | no | our Gamma -> infinity endpoint (Lemma box) |
| Rosenbaum (1983, 2002) | Gamma on treatment odds | randomisation tests, effects | yes | no | no | no | no | DCSM(Gamma) is the decision-censoring form; equivalence to Tan's MSM with Gamma = Lambda^2 (Lemma msm) |
| Lakkaraju et al. (2017) | leniency instrument, random assignment | contraction: failure rate of a model at a target acceptance rate | point estimate | no | no | n/a | no | uses leniency to identify a point; we use it to falsify Gamma (Prop falsify) |
| Kleinberg et al. (2018) | as above, at scale | bail outcomes | point | no | no | n/a | no | application of contraction |
| De-Arteaga et al. (2018) | expert consistency | labels for censored units | no | no | yes (label augmentation) | n/a | no | complementary: we make no consistency assumption |
| Kallus & Zhou (2018); Kallus, Mao & Zhou (2021) | MSM(Lambda) | confounding-robust policy value | yes | no | policy learning | no | partial (asymptotics of policy value) | closest antecedent to Sec. learning; policy value under treatment vs predictive risk under label censoring; our objective is exactly convex with closed-form Bayes act |
| Coston, Rambachan & Chouldechova (2021) | MSM / IV | fairness metrics under selective labels | yes | threshold metrics | no | no | no | threshold metrics are the fixed-threshold special case of Prop unified |
| Rambachan, Coston & Kennedy (2022) | IV / partial identification | counterfactual risk assessment | yes | no | no | partial (IV tests) | no | complementary target |
| Dorn & Guo (2023); Dorn, Guo & Kallus (2024) | MSM(Lambda) | sharp bounds on weighted means / ATE | yes | no | no | no | yes (DR, quantile balancing) | Thm 5(a) is the DR product-rate argument applied to the bound map; we add per-unit inflation (b,c) |
| Zhao, Small & Bhattacharya (2019) | MSM(Lambda) | IPW functionals, percentile bootstrap | yes | no | no | no | bootstrap | fixed-threshold metrics only |
| Yadlowsky et al. (2018) | Rosenbaum-type | conditional ATE bounds | yes | no | no | no | partial | no ranking metric |
| Jung et al. (2020) | sensitivity | decision making | yes | no | no | no | no | related bounds |
| this paper | DCSM(Gamma) = MSM(Lambda^2) | AUROC (sharp), risk, regret, uncertainty decomposition | yes | yes (Thm sharp-auc) | yes (Thm dcl, Prop regret) | lower bound (Prop falsify) | Thm 5 | |

Items the matrix resolves for the manuscript: (i) contraction (Lakkaraju) is a
point-identification method requiring the most lenient decision maker to accept
at least the target rate; our use of leniency is falsification, not
identification; (ii) the closest learning antecedent (Kallus & Zhou) targets
policy value, not predictive risk, and needs a relaxation where our inner
supremum is closed-form; (iii) no prior work bounds a ranking metric sharply;
(iv) the DR argument in Thm 5(a) is standard (Chernozhukov et al. 2018; Dorn
et al.) and is claimed only for the bound map, not as a new estimator class.
