# STATUS

Session-end claim ledger (CLAUDE.md rule 7). One row per claim that appears in
the paper. `evidence` is a file:line in `paper/sections/` for theorems and a
`results/<file>.json:<key>` for experiments. `status` ∈ {proved, verified,
refuted, open}. Rows marked *pending* depend on a run that has not happened in
this environment (see "Blocked" at the bottom).

| claim_id | statement | type | evidence | data_source | seeds | status |
|---|---|---|---|---|---|---|
| (filled at session end by scripts/make_status.py; see below) | | | | | | |

## Blocked
- **T1 (MIMIC-IV-ECG × ECHO)**: `/data/*` absent and `physionet.org` blocked by
  the organisation egress policy in this sandbox; `~/.netrc` absent. The
  pipeline is built and fixture-tested and exits 2 without data. The medical
  column of the main table is **pending a run on the GPU server**. No
  simulator result is used for it (rule 1).

## Changed claims (before → after)
(filled at session end)
