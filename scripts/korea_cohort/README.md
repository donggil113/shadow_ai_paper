# Korean health-screening cohort (T7) — the medical analogue of COMPAS

**Status: pending IRB. No Korean data exist in this repository or in this
environment.** Everything here runs on a small **synthetic** fixture for
`pytest` and exits with code 2 on the real path until
`/data/korea_screening/screening_exams.csv` is present on the GPU server.

## Purpose

COMPAS is the paper's real-decision test bed because the outcome was recorded
for the censored units too, so an interval computed from censored data can be
checked against a recorded deployment quantity. A Korean national
health-screening **batch cohort** is the medical counterpart: everyone in the
batch is screened, so the screening decision is `T ≡ 1` and the outcome is
recorded for every unit ("truth batch"). Two things become checkable that
are not checkable on MIMIC:

1. **Containment.** A model developed on public / US data (e.g. the COMPAS
   ERM score of `experiments/exp6_realdata.py`) has an identified deployment
   AUROC interval `[L, U]` on the US population under DCSM(Γ) — the sharp
   interval of `dcl.auc_bounds`. On the Korean batch cohort the realised
   AUROC of that model's score (`public_model_score`) is **point-identified**.
   Does `[L, U]` contain it? Report the realised AUROC with a bootstrap 95% CI
   (5 seeds), the containment flag, and the smallest Γ at which the US
   interval contains it. Containment is a *necessary* consistency check: a
   miss means either that Γ was too small or that the population differs
   (transportability); the two are confounded and the script says so.
2. **Γ comparison.** Within the batch, the downstream decision
   `followup_test_ordered` (a confirmatory test ordered by the examining
   centre / physician) does vary. On that subset the leniency variation across
   `centre_id` gives `Γ_min` through the repository's falsification routine
   (`dcl.falsify.falsification_curve`, exactly as in
   `experiments/exp4_gamma.py`), and — because the outcome is recorded for the
   `T = 0` units as well — the realised odds ratio
   `odds(P(Y=1 | T=0)) / odds(P(Y=1 | T=1))` (the COMPAS detained / released
   anchor, `results/exp6_summary.json:compas_marginal_odds_ratio`). Both are
   compared with the US values passed on the command line.

Every reported number is an aggregate over ≥ 5 seeds with a 95% CI wherever a
difference is claimed (CLAUDE.md rule 4).

## Files

| file | role |
|---|---|
| `schema.md` | unit-level input schema, validation rules, processing rules, derived variables |
| `prepare_korea_cohort.py` | validate → dedup (one exam per subject per year, keep the first) → follow-up window → derive `T`, `Y`, `Z`, standardised covariates → prepared file + JSON sidecar |
| `analyze_korea_cohort.py` | the two analyses above; `--write-pending` writes the IRB placeholder |
| `../../tests/fixtures/korea_cohort/make_fixture.py` | deterministic generator of the **synthetic** fixture (`screening_exams_fixture.csv`, 400 rows, 5 centres) |
| `../../tests/test_korea_cohort.py` | pytest (fixture path, exit codes, suppression, output keys, placeholder) |

## Input (summary; full schema in `schema.md`)

One row per screening exam: `subject_id` (pseudonymised), `exam_date`
(`YYYY-MM-DD`), `centre_id` (screening centre / physician = leniency
instrument `Z`), `age`, `sex`, the screening panel (`bmi`, `sbp`, `dbp`,
`fasting_glucose`, `total_cholesterol`, `ldl`, `hdl`, `triglycerides`,
`creatinine`, `egfr`, `ast`, `alt`, `ggt`, `haemoglobin`, `smoking_status`,
`alcohol_freq`, `exercise_freq`, `fh_diabetes`, `fh_hypertension`, `fh_cvd`,
`fh_cancer`), `followup_test_ordered` (`T`), `outcome` (`Y`, within the
follow-up window), `public_model_score` (optional), `followup_days`.

## Running

Real path (GPU server; `--data-root` holds `screening_exams.csv`):

```bash
python3 scripts/korea_cohort/prepare_korea_cohort.py \
    --data-root /data/korea_screening --out results/korea_cohort_prepared.parquet
python3 scripts/korea_cohort/analyze_korea_cohort.py \
    --prepared results/korea_cohort_prepared.parquet \
    --us-bounds results/exp6_summary.json \
    --us-gamma-min 1.5 --us-odds-ratio 1.885 \
    --out results/korea_cohort_analysis.json
```

The prepared file (`.parquet` when `pyarrow` is importable, else `.csv`) is
**unit-level pseudonymised data**: it stays on the server and is never
committed (add `results/korea_cohort_prepared.*` to `.gitignore` before the
first real run). The sidecar `results/korea_cohort_prepared.json` and the
analysis JSON are aggregates only.

Fixture path (what `pytest` runs; output must not go under `results/` — the
scripts refuse with exit code 4):

```bash
python3 scripts/korea_cohort/prepare_korea_cohort.py \
    --fixture tests/fixtures/korea_cohort/screening_exams_fixture.csv --out /tmp/kc/prepared.parquet
python3 scripts/korea_cohort/analyze_korea_cohort.py \
    --prepared /tmp/kc/prepared.parquet --us-bounds /tmp/kc/us_bounds.json --fixture \
    --nuisance-model logistic --n-boot 50 --out /tmp/kc/analysis.json
```

IRB placeholder (no data touched; `data_source: "pending"`, analysis keys
`null`):

```bash
python3 scripts/korea_cohort/analyze_korea_cohort.py --write-pending      # -> results/korea_cohort_pending.json
```

`results/korea_cohort_pending.json` is **not** written by the tests (they pass
`--out` inside `tmp_path`) and was not created by this task. Note that
`tests/test_data_guards.py::test_every_results_json_has_data_source`
currently accepts only `{real, simulator, semi-synthetic, synthetic}`; the
literal `pending` must be added to that allowed set before the placeholder is
committed. `scripts/check_provenance.py` already refuses any `pending` macro
in a verified paper build, so the placeholder can never leak into a reported
number.

Exit codes (both scripts): `0` ok · `2` required input missing (data file,
prepared file, sidecar, or a US-bounds file without the required keys) ·
`3` schema violation / provenance mismatch · `4` refused to write
fixture-derived output under `results/`.

### `prepare_korea_cohort.py` options

`--data-root` (default `/data/korea_screening`), `--fixture PATH`,
`--out PATH`, `--window-days` (365), `--min-cell` (10).

### `analyze_korea_cohort.py` options

`--prepared`, `--us-bounds`, `--us-gamma` (entry of the US `by_gamma`
table), `--us-gamma-min`, `--us-odds-ratio` (floats; default `None` → the
comparison is reported as `null`), `--fixture` (labels the output
`synthetic`; refused if the sidecar disagrees), `--seeds` (5), `--n-boot`
(200 bootstrap resamples per seed), `--n-folds` (5), `--nuisance-model`
(`gbm` | `logistic` | `mlp`; `gbm` is the repository default, `logistic` is
the sensible choice when centres are small), `--no-calibrate` (skip isotonic
calibration of the per-centre nuisances; on small centres calibration yields
step functions with extreme values that inflate `Γ_min`), `--grid-size`
(20000; the common grid of units for `Γ_min` is subsampled deterministically
above this), `--threads` (BLAS / OpenMP threads, default 1 — the tiny
per-centre fits are several times slower under oversubscription; raise it
for the real cohort), `--min-cell` (10), `--out`, `--write-pending`.

`data_source` in the output is `real` without `--fixture` and `synthetic`
with it, cross-checked against the sidecar written by `prepare`; it is never
inferred from the data.

## US-bounds JSON (`--us-bounds`)

```json
{
  "data_source": "real",
  "model": "ERM on COMPAS released defendants",
  "population": "COMPAS (Broward County FL)",
  "gamma": 1.885, "auroc_lo": 0.6286, "auroc_hi": 0.7279,
  "by_gamma": {"1": {"auroc_lo": 0.6834, "auroc_hi": 0.6834}, "1.5": {"auroc_lo": 0.6491, "auroc_hi": 0.7133}},
  "nuisances": {"scores": [...], "p1": [...], "e": [...]}
}
```

* Required: `auroc_lo`, `auroc_hi` at a `gamma` — either top-level or as an
  entry of `by_gamma` selected with `--us-gamma`. If neither is present the
  script exits 2 (this is what happens today with
  `results/exp6_summary.json`, which stores coverage summaries but not the
  interval endpoints).
* Optional `by_gamma`: with several entries the "smallest Γ containing the
  realised AUROC" is a table lookup.
* Optional `nuisances` (per US test unit: score `f(x)`, `p1(x)`, `e(x)`): the
  interval is recomputed with `dcl.sensitivity.outcome_bounds` +
  `dcl.auc_bounds.sharp_auc_interval` and the smallest containing Γ is found
  by bisection (nested intervals ⇒ monotone). The recomputed interval at the
  stated Γ is compared with the stated endpoints (`recomputed_matches_stated`).

To build the COMPAS file from the paper's own 5-seed run
(`results/exp6a_compas_coverage.csv`, recorded outcome, mean over seeds):

```python
import json, pandas as pd
a = pd.read_csv("results/exp6a_compas_coverage.csv"); a = a[a.outcome == "recorded"]
g = a.groupby("gamma")[["sharp_lo", "sharp_hi"]].mean()
o = float(a.marginal_odds_ratio.iloc[0])
json.dump({"data_source": "real",
           "model": "ERMObserved on COMPAS (exp6a, 5 seeds, recorded outcome)",
           "population": "COMPAS (Broward County FL)", "gamma": o,
           "auroc_lo": float(g.loc[round(o, 3), "sharp_lo"]),
           "auroc_hi": float(g.loc[round(o, 3), "sharp_hi"]),
           "by_gamma": {f"{k:g}": {"auroc_lo": float(v.sharp_lo), "auroc_hi": float(v.sharp_hi)}
                        for k, v in g.iterrows()},
           "n_seeds": int(a.seed.nunique())},
          open("results/us_bounds_compas.json", "w"), indent=2)
```

## Output (`results/korea_cohort_analysis.json`)

Top level: `data_source` (`real` | `synthetic`), `status`, `n_seeds`,
`n_boot_per_seed`, `rule4_compliant`, `inputs` (paths, SHA-256 of the
prepared file and of the US-bounds file, the US interval used), `cohort`
(counts and rules copied from the sidecar), `containment`,
`gamma_comparison`, `privacy`, `software`.

`containment`: `batch_cohort_outcome_observed_all`, `T_all_one`,
`us_bounds`, `realised_auroc` (`point`; `ci95` = pooled percentile bootstrap
over `n_seeds × n_boot` resamples; `per_seed`; `over_seeds` = mean, t-based
95% half-width `ci95` and `n`, the convention used elsewhere in `results/`),
`contains_point` (**the containment flag**, a bool), `contains_ci`,
`overlaps_ci`, `realised_minus_us_lo` / `realised_minus_us_hi`,
`min_gamma_containing` (`gamma`, `method` ∈ {`recomputed_from_nuisances`,
`table_lookup`, `single_endpoint`}).

`gamma_comparison`: `T_varies`, `n_T`, `selection_rate`,
`outcome_recorded_for_T0`, `gamma_min` (`gamma_min`, `gamma_min_q95`,
`gamma_min_boot_lo` each summarised over seeds by mean / `ci95` / `n` /
min / max, `refutes_MAR_fraction`, `per_centre` with selection rates and
whether the centre was usable, `failures`), `odds_ratio` (`point`, `ci95`,
`gamma_scale_magnitude = max(OR, 1/OR)`, `direction`, the 2×2 counts),
`gamma_min_vs_us` and `odds_ratio_vs_us` (`null` unless the US value was
passed; otherwise differences, ratios, `korea_ci95_contains_us`,
`same_direction`). If `T` does not vary (pure batch cohort) the block reports
`available: false` with the reason and `null` values.

## Privacy

Only aggregated outputs are ever written by `analyze` and in the sidecar of
`prepare`; every count with fewer than `--min-cell` (10) units is suppressed:
it is reported as `{"n": null, "suppressed": true}`. Zero counts are
reported. Inside a table whose margins are published (per centre, per year,
centre × T, the 2×2 outcome table) complementary suppression hides the
smallest remaining cell of any row / column that would otherwise reveal a
suppressed one (`"reason": "complementary"`). A rate is published only when
its denominator, numerator and complement are each 0 or ≥ `min-cell`. The
analysis script refuses to write any array longer than 64 entries or any
`subject_id` key. The prepared unit-level file is the only non-aggregate
artefact and never leaves the server.

## Assumptions and limitations (to be stated in the paper)

* `Γ_min` needs the exclusion restriction `centre_id ⟂ (Y, S) | X`: people
  attend the centre nearest to them rather than the one matching their
  hidden risk. It fails if sicker people choose hospital-based centres.
* With few units per centre the cross-fitted per-centre nuisances are noisy
  and `Γ_min` (a maximum over units) is inflated; report `gamma_min_q95` and
  the bootstrap lower limit, and use `--nuisance-model logistic`
  (optionally `--no-calibrate`) unless centres are large. On the synthetic
  fixture the estimates are meaningless by construction and are never
  reported.
* Clinically the untested are usually *lower* risk (`OR < 1`,
  `censored-lower`), the opposite sign to COMPAS (`OR ≈ 1.9`); the script
  reports the sign and compares magnitudes on the Γ scale (`max(OR, 1/OR)`).
* Containment across populations conflates censoring (Γ) with
  transportability; a miss is informative, containment is not a proof.

## Fixture (synthetic)

`tests/fixtures/korea_cohort/screening_exams_fixture.csv` is generated by
`tests/fixtures/korea_cohort/make_fixture.py` (numpy, seed 20260916): 400
rows, 345 subjects (`SYN-…`), 5 centres `KC01`–`KC05` with different
leniency and different reliance on a hidden signal, outcome recorded for
every row, a `public_model_score`, same-year duplicates (dedup), a few rows
lost to follow-up (window rule), five exams in 2021 (a suppressed cell) and
sparse missingness. **It is synthetic; no number derived from it may appear
anywhere.**

```bash
python3 -m pytest tests/test_korea_cohort.py -q
```
