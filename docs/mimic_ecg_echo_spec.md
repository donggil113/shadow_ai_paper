# T1 — MIMIC-IV-ECG × MIMIC-IV-ECHO real medical arm: design spec

Binding for every module under `dcl/data/mimic_ecg_echo/` and
`experiments/exp7_mimic_ecg_echo.py`. Read CLAUDE.md rule 1 first.

## Non-negotiables
- **Data root:** `--data-root /data` (each dataset in its own subfolder, see
  layout). A missing required file ⇒ print exactly which path is missing, the
  PhysioNet page to obtain it from, and `sys.exit(2)`. **Never** fall back to a
  simulator. `dcl.data.mimic.make_mimic_sim` may not be imported anywhere in
  this package.
- **Schema is verified at load time, not assumed.** Each loader declares
  `EXPECTED_COLUMNS` and, on mismatch, prints the actual header, the expected
  header and the dataset version it was written against, then exits 2. We
  could not open the PhysioNet pages from the development sandbox (egress
  blocked); the expected schemas below are written from the public
  documentation of MIMIC-IV-ECG v1.0, MIMIC-IV v2.2/3.1, MIMIC-IV-ECHO v0.1,
  MIMIC-IV-Note v2.2 and the ECG-MIMIC linkage code
  (`AI4HealthUOL/ECG-MIMIC/src/full_preprocessing.py`, fetched and cached in
  the scratchpad). Any column that turns out to differ must be fixed in the
  loader, not papered over.
- **Fixtures.** Every module ships a tiny synthetic fixture under
  `tests/fixtures/mimic_ecg_echo/` (≤ 50 rows per table, random values,
  clearly fake subject_ids ≥ 90_000_000) used ONLY by pytest to exercise the
  code path end-to-end. Fixtures are never written to `results/` and no number
  derived from a fixture may be reported anywhere.
- **Provenance.** `experiments/exp7_mimic_ecg_echo.py` writes
  `results/exp7_mimic_ecg_echo_summary.json` with `data_source: "real"` — and
  refuses to write at all unless every input came from `--data-root` (the
  loader returns a `provenance` dict; the experiment asserts
  `DataRoot.all_real()` and exits 3 otherwise). On `--fixture-root` the same
  file is written with `data_source: "synthetic"` into `--out`, which must lie
  outside `results/` (exit 3 otherwise).
- **Seeds.** 5 seeds for every reported number, with 95% CIs via
  `experiments/_common.ci95`.

## Expected layout under `--data-root`
```
/data/mimic-iv-ecg/1.0/
    record_list.csv            subject_id, study_id, file_name, ecg_time
    machine_measurements.csv   subject_id, study_id, ecg_time, rr_interval, p_onset, ..., report_0..17
    files/pNNNN/pNNNNNNNN/sNNNNNNNN/NNNNNNNN.{hea,dat}   (WFDB, 12 leads, 500 Hz, 10 s)
/data/mimic-iv/3.1/hosp/
    patients.csv.gz            subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod
    admissions.csv.gz          subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admission_location, ...
    transfers.csv.gz           subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime
    services.csv.gz            subject_id, hadm_id, transfertime, prev_service, curr_service
    diagnoses_icd.csv.gz       subject_id, hadm_id, seq_num, icd_code, icd_version
    labevents.csv.gz           subject_id, hadm_id, specimen_id, itemid, charttime, value, valuenum, ...
    d_labitems.csv.gz          itemid, label, fluid, category
/data/mimic-iv-ed/2.2/ed/
    edstays.csv.gz             subject_id, hadm_id, stay_id, intime, outtime, gender, race, arrival_transport, disposition
    triage.csv.gz              subject_id, stay_id, temperature, heartrate, resprate, o2sat, sbp, dbp, pain, acuity, chiefcomplaint
/data/mimic-iv-echo/0.1/
    echo-record-list.csv       subject_id, study_id, dicom_filepath   (expected; verify)
    echo-study-list.csv        subject_id, study_id, study_datetime   (expected; verify)
    files/...                  DICOM (not needed unless route (a) measurements exist)
/data/mimic-iv-note/2.2/note/
    discharge.csv.gz           note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text
/data/mimic-iv-ecg-ext-icd/   (MIMIC-IV-ECG-ICD, Strodthoff et al. 2024)
    records_w_diag_icd10.csv   file_name, study_id, subject_id, ecg_time, ..., all_diag_all (list of ICD-10) (expected; verify)
```

## Unit of analysis and decision
- Unit = one ECG record (`study_id` in MIMIC-IV-ECG).
- **Decision T = 1** iff the same `subject_id` has an echo study with
  `|study_datetime − ecg_time| ≤ 365 days` (EchoNext protocol). The echo
  linkage uses `echo-study-list.csv`.
- **Population restriction:** MIMIC-IV-ECHO covers a limited date range
  (expected 2017–2019; the loader MUST compute the actual min/max
  `study_datetime` and write it to the provenance dict and to the results
  JSON as `echo_coverage_window`). Restrict ECGs to
  `ecg_time ∈ [echo_min − 365 d, echo_max + 365 d]` and additionally to the
  window where an echo could have been recorded on either side; report the
  ECG count before/after restriction. Reason (state it in the paper): outside
  the ECHO coverage window T = 0 is guaranteed by data availability, not by a
  clinical decision, and would contaminate e(x).
- Deduplicate: at most one ECG per subject per 30 days (keep the first), to
  avoid a subject with many ECGs dominating. Record the rule in provenance.

## Outcome Y (priority order; implement all three as separate columns)
- (a) `Y_echo_struct`: if MIMIC-IV-ECHO ships structured measurements
  (LVEF, IVSd/LVPWd, valve grades), reproduce the EchoNext 11-component SHD
  composite. **Expected: NOT available in v0.1 (images only)** — the loader
  detects this and marks route (a) as `unavailable` in provenance.
- (b) `Y_note`: discharge summaries (`discharge.csv.gz`) of admissions
  overlapping the linked echo (`hadm_id` of the echo's admission, or the
  admission containing the ECG ±365 d) parsed for LVEF (%), septal/posterior
  wall thickness (cm), and valve severity (moderate/severe AS/AR/MS/MR/TR),
  by regex first (`dcl/data/mimic_ecg_echo/notes_regex.py`) with a pluggable
  LLM extractor interface (`notes_llm.py`, provider-agnostic, off by
  default). SHD composite = LVEF ≤ 45 OR wall thickness ≥ 1.3 cm OR any
  moderate/severe valve lesion (mirror EchoNext's definition as closely as the
  notes allow; document deviations). Ship
  `scripts/note_extraction_manual_review.py` that samples 200 notes
  (stratified by extracted label) into a CSV for manual verification and
  computes precision/recall once the reviewer fills in the truth column.
- (c) `Y_icd`: SHD from ICD-10 codes (I50.x heart failure, I42 cardiomyopathy,
  I34–I37 valve disorders, I05–I08 rheumatic valve, I11.0 hypertensive HD with
  HF, I25.5 ischaemic cardiomyopathy; list them in code) taken from
  `diagnoses_icd` of the admission containing the ECG, or from
  MIMIC-IV-ECG-ICD if present. **Auxiliary only**: ICD assignment depends on
  whether an echo was performed (circularity). The experiment must quantify
  it: report the ICD-SHD rate among T=1 vs T=0 ECGs and among echo-linked vs
  not, and place Y_icd analyses in the appendix sensitivity table only.

## Covariates X (never include anything downstream of the decision)
- ECG: 12-lead waveform (500 Hz, 10 s) → `ECGEncoder` (1-D ResNet-18-style,
  the EchoNext-family architecture; `dcl/models_ecg.py`), plus the
  machine-measurement scalars (RR, PR, QRS, QT/QTc, axes) as a tabular
  fallback when waveforms are unavailable (`--features tabular|waveform|both`).
- Demographics: age at ECG (anchor_age + year offset), sex.
- Labs within 24 h before the ECG (troponin, BNP/NT-proBNP, creatinine, Hb):
  `labs.py` with itemids resolved through `d_labitems` by label match; missing
  ⇒ indicator + median.
- Setting: ED vs inpatient vs outpatient (from edstays/admissions overlap).

## Decision-maker for marginalisation and Γ_min (T1.3–1.4)
- Decision-maker `Z` = ordering unit = `careunit` from `transfers` active at
  `ecg_time` (fallback: `curr_service` from `services`; else `admission_location`).
- Propensity e(x) = P(T=1 | X) fit **without** Z (marginal over ordering unit),
  5-fold cross-fitted, isotonic-calibrated (`dcl.nuisance.CrossFitNuisance`
  extended to accept precomputed ECG embeddings). Report calibration curve
  (10 bins, ECE) and AUROC with 95% CI over 5 seeds.
- Γ_min: per-Z nuisances (e_z, p1_z) on a common grid of units →
  `dcl.falsify.falsification_curve`. Hypothesis H_A1: Γ_min ∈ [1.5, 3.0];
  report the point estimate, q95 variant and bootstrap lower limit. State the
  exclusion restriction assumption (ordering unit ⊥ (Y, S) | X) and the
  obvious ways it can fail (sicker patients go to cardiology) — this goes in
  the limitations.

## Bounds for the medical column (T1.5)
- Score f = EchoNext weights if provided via `--echonext-weights PATH`
  (state the source and licence in provenance); otherwise retrain the same
  architecture on the T=1 subset (5 seeds), holding out 30% of subjects.
- Report: observed AUROC (T=1 test units), sharp interval [L, U] at
  Γ ∈ {1, 1.5, Γ_min, 2, 3, 5} via `dcl.auc_bounds.sharp_auc_interval`,
  break-even Γ* where L drops below 0.5 / 0.6 / 0.7, worst-case risk and
  minimax regret of the DCL rules. All with 5-seed CIs.

## Command
```
python3 experiments/exp7_mimic_ecg_echo.py --data-root /data --features both \
    --echonext-weights /data/echonext/weights.pt --seeds 5 --out results/
```
Exit codes: 0 ok · 2 missing data / schema mismatch (the message names the
file and the PhysioNet page) · 3 argument / schema / provenance violation
(fixture output aimed at `results/`, both roots given, weights that do not fit
the architecture, provenance not real).

## Implementation status (2026-09-16)

### Implemented and exercised end-to-end on the pytest fixture
| piece | file | status |
|---|---|---|
| loaders, schema check, exit-2 policy, provenance | `dcl/data/mimic_ecg_echo/io.py` | done; 15 tests in `tests/test_mimic_io_linkage.py` |
| ECG→echo linkage (±365 d), coverage window, dedup (30 d), decision maker Z | `linkage.py` | done; same tests |
| route (a) detection + EchoNext 11-component composite | `outcomes.py: detect_structured_echo_measurements, compute_structured_shd` | done; v0.1 is expected to report `unavailable` |
| route (b) notes: regex extractor, local-LLM hook, per-record labels | `notes_regex.py`, `notes_llm.py`, `outcomes.label_from_notes` | done; unit tests on sentences with negations |
| route (c) ICD-10 auxiliary + circularity diagnostic | `outcomes.label_from_icd`, `icd_circularity_diagnostic` | done |
| covariates (demographics, labs ≤24 h, setting, machine scalars), `assemble_X` + `assert_no_forbidden` | `covariates.py` | done; `tests/test_mimic_outcomes_covariates.py` (16 tests) |
| ECG encoder (1-D ResNet-18 style), WFDB/.npy dataset, subject-level training | `dcl/models_ecg.py` | done |
| experiment driver (pipeline of §"Command", seeds + CIs, JSON/CSV outputs) | `experiments/exp7_mimic_ecg_echo.py` | done; `tests/test_exp7_cli.py` (7 tests, two end-to-end fixture runs) |
| manual-validation hook | `scripts/note_extraction_manual_review.py`; the driver writes `exp7_note_extraction_for_review.csv` and accepts `--note-validation-json` | done |
| fixture generator (14 tables, ≤50 rows each, subject_id ≥ 90 000 000) | `scripts/make_mimic_fixture.py` | done; regenerate with `python3 scripts/make_mimic_fixture.py` |

The driver on the fixture: 36 ECGs → 33 (coverage) → 30 (dedup), T=1 for 19,
16 of them note-labelled, three ordering units after merging. Waveforms are
**not** shipped in the fixture (8.6 MB); on `--fixture-root` only, the driver
synthesises deterministic `(12, 5000)` arrays in memory
(`exp7_mimic_ecg_echo.fixture_waveforms`) so that the encoder path runs. On
the real path a missing WFDB record exits 2 through `ECGDataset`. No fixture
number is ever reported; the fixture JSON carries `data_source: "synthetic"`
and a `WARNING` key.

### What the driver computes (per seed, then mean ± t-based 95% CI over seeds)
1. subject-level 70/30 split (`--test-frac`), re-drawn deterministically when
   a split cannot support the analysis (tiny data only; `split_seed_used`);
2. outcome model f on labelled T=1 train units: `ECGEncoder` when
   `--features` includes `ecg` (or the supplied `--echonext-weights`),
   gradient boosting on the tabular X otherwise (`--features both` reports
   the tabular AUROC as well);
3. nuisances e(x,z)=P(T=1|X,Z) and p1(x,z)=P(Y=1|X,Z,T=1), 5-fold
   cross-fitted on the train split (isotonic calibration as in
   `dcl.nuisance`; skipped only when a class has <15 units), fold models
   averaged on the test split; **marginalised over the empirical distribution
   of Z**: e(x)=Σ_z π_z e_z(x), p1(x)=Σ_z π_z e_z(x)p1_z(x)/e(x) (the
   convention of the semi-synthetic oracle in `dcl/data/semisynthetic.py`);
   propensity AUROC, 10-bin calibration curve and ECE on the test split;
4. Γ_min from the per-Z nuisances on the common grid of test units
   (`dcl.falsify.falsification_curve`: point, q95 and bootstrap lower limit,
   `--n-boot`), and H_A1 = "Γ_min ∈ [1.5, 3.0]" as a boolean on the seed
   mean, per seed, and whether the CI lies entirely inside;
5. observed AUROC on labelled T=1 test units; sharp interval [L, U]
   (`dcl.auc_bounds.sharp_auc_interval`) at every Γ of `--gamma-grid`
   (default 1, 1.25, 1.5, 2, 2.5, 3, 4, 6) and at the seed's Γ_min, with the
   worst-case risk and minimax regret of f and of the DCL regret rule;
6. break-even Γ* = largest Γ on a 61-point geometric grid up to
   `--breakeven-max` at which L ≥ 0.5 / 0.6 / 0.7;
7. ICD sensitivity (appendix only): circularity diagnostic and AUROC of f
   against Y_icd among T=1 and T=0 test units.

Outputs in `--out`: `exp7_mimic_ecg_echo_summary.json`,
`exp7_mimic_ecg_echo_seeds.csv`, `exp7_mimic_ecg_echo_gamma.csv`,
`exp7_note_extraction_for_review.csv`.

### Deviations from the text above (deliberate; keep in sync)
- `--features` takes `tabular|ecg|both` (not `waveform`).
- The propensity is fit **with** Z as an input and then averaged over the
  empirical Z distribution rather than fit without Z; the out-of-fold AUROC
  with the realised Z on the train split is reported next to it
  (`propensity.auroc_train_oof_realised_z`).
- Rare ordering units are merged into `other` below `--min-z-count`
  (default 100 on the real data; `max(2, min(100, n//100))` otherwise).
- T=1 ECGs whose outcome cannot be recovered from the notes stay in the
  propensity fit and are excluded from the outcome model / observed AUROC;
  `counts.label_recovery_rate_T1` reports the loss.
- Route (a) needs a `study_id` column in the measurements table to be joined;
  otherwise it is recorded as unavailable and route (b) is used.
- Gradient boosting uses `min_samples_leaf = min(30, max(2, n_train // 8))`
  (30 on the real data); `--threads` defaults to 1 because several OpenMP
  threads stall `HistGradientBoosting` in the development sandbox — raise it
  on the server.

### Needs the GPU server (cannot happen here: `/data` absent, physionet.org blocked)
- the real run itself (`--data-root /data`), i.e. every number of the medical
  column; the real `echo_coverage_window`, counts and Z levels;
- verification of the expected headers against the actual files — the loader
  exits 2 on any mismatch and the fix goes into `io.EXPECTED_COLUMNS`;
- `pip install wfdb` for the waveform path; the first run may convert records
  to `.npy` next to the `.hea` files (documented fallback of `read_waveform`)
  to avoid re-reading WFDB for every epoch and seed;
- EchoNext weights (`--echonext-weights`, `--echonext-weights-note` for source
  and licence) if available; otherwise the encoder is trained on T=1 units;
- the 200-note manual review (`scripts/note_extraction_manual_review.py`)
  whose scored JSON is passed back through `--note-validation-json`;
- optionally a local LLM extractor (`--llm-extractor module:factory`).

### Run commands
```
# real data, full run (GPU server)
python3 experiments/exp7_mimic_ecg_echo.py --data-root /data --features both --seeds 5 --out results/ --threads 16
# with EchoNext weights
python3 experiments/exp7_mimic_ecg_echo.py --data-root /data --features both \
    --echonext-weights /data/echonext/weights.pt --echonext-weights-note "<source, licence>" --seeds 5 --out results/
# tabular fallback (no waveforms), and a smoke run on real data
python3 experiments/exp7_mimic_ecg_echo.py --data-root /data --features tabular --seeds 5 --out results/
python3 experiments/exp7_mimic_ecg_echo.py --data-root /data --features tabular --seeds 1 --max-records 2000 --n-boot 20 --out /tmp/exp7_smoke
# after the manual note review
python3 scripts/note_extraction_manual_review.py --data-root /data --n 200 --seed 0 --out audit/note_review_sample.csv
python3 scripts/note_extraction_manual_review.py --score audit/note_review_sample.csv --out results/note_extraction_validation.json
python3 experiments/exp7_mimic_ecg_echo.py --data-root /data --features both --seeds 5 --out results/ \
    --note-validation-json results/note_extraction_validation.json
# fixture (synthetic; output outside results/)
python3 experiments/exp7_mimic_ecg_echo.py --fixture-root tests/fixtures/mimic_ecg_echo --features tabular --seeds 2 --max-records 60 --out /tmp/exp7_fixture
python3 experiments/exp7_mimic_ecg_echo.py --fixture-root tests/fixtures/mimic_ecg_echo --features ecg --seeds 2 \
    --ecg-base-width 4 --ecg-layers 1,1,1,1 --ecg-epochs 2 --ecg-batch-size 8 --n-boot 10 --out /tmp/exp7_fixture_ecg
# tests and fixture regeneration
python3 -m pytest tests/test_exp7_cli.py tests/test_mimic_outcomes_covariates.py tests/test_mimic_io_linkage.py -q
python3 scripts/make_mimic_fixture.py
```
