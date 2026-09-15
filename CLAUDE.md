# Project rules (binding for every agent working in this repository)

Target: complete branch `claude/busy-allen-ws1ttc` as an ICML 2027 submission
on decision-censored learning (selective labels).

1. **Data.** Execution environment for real data is the user's local GPU
   server. Data root is `--data-root=/data/<dataset>`; PhysioNet credentials
   live in `~/.netrc` there. If a required data file is missing the script must
   **fail immediately with exit code 2** and must **never** substitute a
   simulator silently. If a substitute is needed, ask the user first.
2. **Numbers in the paper.** Every numeric value in the paper body is generated
   from `results/*.json` through LaTeX macros (`paper/numbers.tex`, produced by
   `scripts/make_numbers.py`). Hand-typed numbers are forbidden.
   `scripts/verify_paper_numbers.py` must always pass.
3. **Provenance.** Every `results/*.json` carries a top-level
   `data_source` field with value in `{real, simulator, semi-synthetic}`
   (per-dataset entries carry their own where a file mixes sources). A checker
   fails CI if the real-data / clinical sections of the paper cite a macro whose
   provenance is `simulator`.
4. **Statistics.** Any comparison that claims a difference uses **at least 5
   seeds with a 95% confidence interval**. Single-seed numbers may not enter
   the `.tex`.
5. **Refuted hypotheses** are recorded, not hidden: (a) the fact of refutation,
   (b) the diagnosed cause, (c) the corrected claim — in both README.md and the
   paper.
6. **Commits** are per task; the first line carries a `[T#]` or `[P#]` tag.
7. **STATUS.md** is updated at session end with the fixed schema:
   `| claim_id | statement | type(thm/prop/exp) | evidence(file:line or results key) | data_source | seeds | status(proved/verified/refuted/open) |`
8. **Reporting** leads with conclusions and always includes a
   "changed claims (before → after)" list.

Conventions: `python3`, numpy/scipy/pandas/sklearn/torch; tests in `tests/`
(pytest); experiment scripts in `experiments/` write to `results/`; the paper
lives in `paper/` (`main.tex` + `sections/*.tex`); fixtures for schema tests live
in `tests/fixtures/` and are **never** written to `results/`.
