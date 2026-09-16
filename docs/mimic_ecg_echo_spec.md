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
  `results/exp7_mimic_ecg_echo.json` with `data_source: "real"` — and refuses
  to write at all unless every input came from `--data-root` (the loader
  returns a `provenance` dict; the experiment asserts `provenance["source"] == "real"`).
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
python experiments/exp7_mimic_ecg_echo.py --data-root /data --features both \
    --echonext-weights /data/echonext/weights.pt --seeds 5 --out results/
```
Exit codes: 0 ok · 2 missing data / schema mismatch · 3 provenance not real.
