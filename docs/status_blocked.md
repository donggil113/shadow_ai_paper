## Blocked (needs the user's environment)

| item | what is needed | where it lands |
|---|---|---|
| T1 real run (claim C21, the paper's medical column) | credentialed MIMIC-IV-ECG v1.0, MIMIC-IV v3.1 hosp/ED, MIMIC-IV-ECHO v0.1 (+ MIMIC-IV-Note, MIMIC-IV-ECG-Ext-ICD) under `/data/<dataset>` on the GPU server, `pip install wfdb`, optionally EchoNext weights; then `python3 experiments/exp7_mimic_ecg_echo.py --data-root /data --features both --seeds 5 --out results/` (`--threads 16`), the 200-note manual review via `scripts/note_extraction_manual_review.py` fed back with `--note-validation-json` | `results/exp7_mimic_ecg_echo_summary.json` (`data_source: real`); then register the macros in `scripts/make_numbers.py` and fill the medical paragraph of `paper/sections/08_experiments.tex` (scope `sec:medical` requires `real` provenance) |
| T7 Korean cohort (claim C22) | IRB approval and `/data/korea_screening/screening_exams.csv` in the schema of `scripts/korea_cohort/schema.md`; a US-bounds JSON with `auroc_lo/auroc_hi` (snippet in `scripts/korea_cohort/README.md`) | `results/korea_cohort_analysis.json` replaces `results/korea_cohort_pending.json` |
| Echo coverage period | verify the 2017–2019 assumption against the real `echo-study-list.csv`; the driver computes and reports the window (`echo_coverage_window`) and restricts to it | reported in the summary JSON and in `paper/sections/C_more_results.tex` (medical-arm status) |

Everything else in the ledger is reproducible here: `python3 experiments/run_all.py`
re-runs exp0–exp6, exp8, exp10, the audits, the number macros, the verifier and
this file's parent.
