#!/usr/bin/env python3
"""Write the tiny synthetic MIMIC-IV-ECG/ECHO/hosp fixture used ONLY by pytest.

Deterministic (seed 0, gzip mtime fixed to 0), clearly fake subject_ids
(>= 90_000_000), <= 50 rows per table.  Output goes to
``tests/fixtures/mimic_ecg_echo/`` (flat layout understood by
``dcl.data.mimic_ecg_echo.io.DataRoot``).  Never write this into ``results/``
and never report a number derived from it.

Hand-checkable cases (referenced by tests/test_mimic_io_linkage.py):

* subject 90000001: ECG 2018-01-01 00:00, echo 2019-01-01 00:00 -> exactly
  365 d -> T = 1; transfers careunit "Medical Intensive Care Unit (MICU)"
  active at the ECG -> Z from careunit.
* subject 90000002: ECG 2018-01-01 00:00, echo 2019-01-02 00:00 -> 366 d ->
  T = 0; transfer ended before the ECG, admission spans it with
  admission_location "EMERGENCY ROOM" -> Z falls back to admission_location
  (or to curr_service when a services table is supplied).
* subject 90000003: ECG 2018-06-01, echo 2017-06-01 (-365 d) -> T = 1;
  a second ECG at 2016-02-01 lies before echo_min - 365 d and is dropped by
  the coverage restriction.
* subject 90000004: no echo, no transfers, no admission -> T = 0, Z = unknown.
* subject 90000005: ECGs on days 0, 10, 29, 30, 45, 61 from 2018-03-01 ->
  dedup keeps days 0, 30, 61.
* subject 90000006: ECGs at 2015-06-01 and 2021-01-01 (outside coverage +-365 d)
  and one at 2016-03-01 00:00 == echo_min - 365 d (boundary, kept).
* echo coverage sentinels: earliest echo 2017-03-01 00:00 (subject 90000007),
  latest echo 2019-10-31 00:00 (subject 90000008).

exp7 extension (``build_extra_tables``; tests/test_exp7_cli.py): admissions
without transfers for every T = 1 subject that had none (hadm 20000010..19,
appended after all random draws so that the other base tables are unchanged),
``machine_measurements.csv`` (one row per ECG, sentinel 29999 every 9th row),
``hosp/d_labitems.csv.gz`` + ``hosp/labevents.csv.gz`` (troponin / NT-proBNP /
creatinine within 20 h before 12 ECGs, plus edge rows for subject 90000001:
after the ECG, 30 h before, 5 h before, and an HbA1c distractor),
``ed/edstays.csv.gz`` (one stay per ED transfer + a stay containing ECG
40000025), ``hosp/services.csv.gz``, ``note/discharge.csv.gz`` (one summary per
admission from NOTE_TEMPLATES, cycled), ``records_w_diag_icd10.csv`` (every
7th ECG absent; SHD codes more frequent for echo subjects, on purpose) and
``hosp/diagnoses_icd.csv.gz`` (4th admission ICD-9 only).

ECG waveforms are NOT part of the fixture: on ``--fixture-root`` the exp7
driver synthesises them in memory (see experiments/exp7_mimic_ecg_echo.py,
``fixture_waveforms``); on the real path missing WFDB files exit 2.

Usage: python3 scripts/make_mimic_fixture.py [--out DIR]
"""
from __future__ import annotations

import argparse
import gzip
import io
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "tests", "fixtures", "mimic_ecg_echo")
SEED = 0
BASE_SUBJECT = 90_000_000
ECG_BASE = 40_000_000
ECHO_BASE = 50_000_000
HADM_BASE = 20_000_000
TRANSFER_BASE = 30_000_000
CAREUNITS = ["Emergency Department", "Medicine", "Cardiology",
             "Medical Intensive Care Unit (MICU)", "Coronary Care Unit (CCU)",
             "Med/Surg", "Cardiac Surgery"]
ADM_LOC = ["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER FROM HOSPITAL",
           "CLINIC REFERRAL", "WALK-IN/SELF REFERRAL"]


def _fmt(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _ecg_file(subject_id: int, study_id: int) -> str:
    return f"files/p{str(subject_id)[:4]}/p{subject_id}/s{study_id}/{study_id}"


def build_tables(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    S = lambda k: BASE_SUBJECT + k  # noqa: E731
    ecgs: list[tuple[int, pd.Timestamp]] = []
    echos: list[tuple[int, pd.Timestamp]] = []

    # --- hand-checkable cases
    ecgs.append((S(1), pd.Timestamp("2018-01-01 00:00:00")))
    echos.append((S(1), pd.Timestamp("2019-01-01 00:00:00")))          # +365 d -> T=1
    ecgs.append((S(2), pd.Timestamp("2018-01-01 00:00:00")))
    echos.append((S(2), pd.Timestamp("2019-01-02 00:00:00")))          # +366 d -> T=0
    ecgs.append((S(3), pd.Timestamp("2018-06-01 00:00:00")))
    ecgs.append((S(3), pd.Timestamp("2016-02-01 00:00:00")))           # dropped by coverage
    echos.append((S(3), pd.Timestamp("2017-06-01 00:00:00")))          # -365 d -> T=1
    ecgs.append((S(4), pd.Timestamp("2018-09-15 12:00:00")))           # no echo
    d0 = pd.Timestamp("2018-03-01 08:00:00")
    for d in (0, 10, 29, 30, 45, 61):
        ecgs.append((S(5), d0 + pd.Timedelta(days=d)))
    echos.append((S(5), pd.Timestamp("2018-04-20 00:00:00")))
    ecgs.append((S(6), pd.Timestamp("2015-06-01 00:00:00")))           # dropped
    ecgs.append((S(6), pd.Timestamp("2021-01-01 00:00:00")))           # dropped
    ecgs.append((S(6), pd.Timestamp("2016-03-01 00:00:00")))           # == echo_min - 365 d, kept
    echos.append((S(7), pd.Timestamp("2017-03-01 00:00:00")))          # coverage min
    ecgs.append((S(7), pd.Timestamp("2017-02-10 09:30:00")))
    echos.append((S(8), pd.Timestamp("2019-10-31 00:00:00")))          # coverage max
    ecgs.append((S(8), pd.Timestamp("2019-11-20 16:45:00")))

    # --- random filler, all inside [2017-04-01, 2019-09-30]
    lo = pd.Timestamp("2017-04-01").value
    hi = pd.Timestamp("2019-09-30").value
    filler_subjects = [S(k) for k in range(9, 21)]
    for sid in filler_subjects:
        for _ in range(int(rng.integers(1, 3))):
            ecgs.append((sid, pd.Timestamp(int(rng.integers(lo, hi))).floor("min")))
    for sid in rng.choice(filler_subjects, size=8, replace=False):
        echos.append((int(sid), pd.Timestamp(int(rng.integers(lo, hi))).floor("min")))

    record_list = pd.DataFrame({
        "subject_id": [s for s, _ in ecgs],
        "study_id": [ECG_BASE + 1 + i for i in range(len(ecgs))],
        "ecg_time": [_fmt(t) for _, t in ecgs],
    })
    record_list["file_name"] = [_ecg_file(s, st) for s, st in
                                zip(record_list["subject_id"], record_list["study_id"])]
    record_list = record_list[["subject_id", "study_id", "file_name", "ecg_time"]]

    echo_study_list = pd.DataFrame({
        "subject_id": [s for s, _ in echos],
        "study_id": [ECHO_BASE + 1 + i for i in range(len(echos))],
        "study_datetime": [_fmt(t) for _, t in echos],
    })
    rows = []
    for s, st in zip(echo_study_list["subject_id"], echo_study_list["study_id"]):
        for k in range(int(rng.integers(1, 3))):
            rows.append((s, st, f"files/p{str(s)[:3]}/p{s}/s{st}/{st}_{k + 1:04d}.dcm"))
    echo_record_list = pd.DataFrame(rows, columns=["subject_id", "study_id", "dicom_filepath"])

    subjects = sorted({s for s, _ in ecgs} | {s for s, _ in echos})
    patients = pd.DataFrame({
        "subject_id": subjects,
        "gender": rng.choice(["F", "M"], size=len(subjects)),
        "anchor_age": rng.integers(25, 89, size=len(subjects)),
        "anchor_year": rng.integers(2017, 2020, size=len(subjects)),
        "anchor_year_group": "2017 - 2019",
        "dod": "",
    })

    # --- admissions / transfers (hand-crafted cases first)
    adm_rows = [
        (S(1), HADM_BASE + 1, "2017-12-30 18:00:00", "2018-01-05 12:00:00", "", "EW EMER.",
         "EMERGENCY ROOM", "HOME"),
        (S(2), HADM_BASE + 2, "2017-12-28 10:00:00", "2018-01-03 09:00:00", "", "URGENT",
         "EMERGENCY ROOM", "HOME"),
        (S(3), HADM_BASE + 3, "2018-05-30 07:00:00", "2018-06-04 15:00:00", "", "ELECTIVE",
         "PHYSICIAN REFERRAL", "HOME"),
    ]
    tr_rows = [
        # S1: MICU active at the ECG (2018-01-01 00:00)
        (S(1), HADM_BASE + 1, TRANSFER_BASE + 1, "ED", "Emergency Department",
         "2017-12-30 18:00:00", "2017-12-31 02:00:00"),
        (S(1), HADM_BASE + 1, TRANSFER_BASE + 2, "admit", "Medical Intensive Care Unit (MICU)",
         "2017-12-31 02:00:00", "2018-01-02 10:00:00"),
        (S(1), HADM_BASE + 1, TRANSFER_BASE + 3, "transfer", "Medicine",
         "2018-01-02 10:00:00", "2018-01-05 12:00:00"),
        (S(1), HADM_BASE + 1, TRANSFER_BASE + 4, "discharge", "",
         "2018-01-05 12:00:00", ""),
        # S2: only a transfer that ENDED before the ECG (2018-01-01 00:00)
        (S(2), HADM_BASE + 2, TRANSFER_BASE + 5, "admit", "Cardiology",
         "2017-12-28 10:00:00", "2017-12-31 20:00:00"),
        (S(2), HADM_BASE + 2, TRANSFER_BASE + 6, "discharge", "",
         "2018-01-03 09:00:00", ""),
        # S3: open-ended stay (outtime empty) covering the 2018-06-01 ECG
        (S(3), HADM_BASE + 3, TRANSFER_BASE + 7, "admit", "Cardiac Surgery",
         "2018-05-30 07:00:00", ""),
    ]
    k_h, k_t = 4, 8
    for sid in filler_subjects[:6]:
        t0 = pd.Timestamp(int(rng.integers(lo, hi))).floor("h")
        t1 = t0 + pd.Timedelta(hours=int(rng.integers(6, 48)))
        t2 = t1 + pd.Timedelta(hours=int(rng.integers(12, 120)))
        adm_rows.append((sid, HADM_BASE + k_h, _fmt(t0), _fmt(t2), "", "EW EMER.",
                         str(rng.choice(ADM_LOC)), "HOME"))
        tr_rows.append((sid, HADM_BASE + k_h, TRANSFER_BASE + k_t, "ED", "Emergency Department",
                        _fmt(t0), _fmt(t1)))
        tr_rows.append((sid, HADM_BASE + k_h, TRANSFER_BASE + k_t + 1, "admit",
                        str(rng.choice(CAREUNITS[1:])), _fmt(t1), _fmt(t2)))
        tr_rows.append((sid, HADM_BASE + k_h, TRANSFER_BASE + k_t + 2, "discharge", "",
                        _fmt(t2), ""))
        k_h += 1
        k_t += 3
    # --- admissions (no transfers) for the T=1 subjects that had none, so that
    # discharge summaries can be linked (exp7 fixture run).  Hand-crafted, no
    # rng draws, appended after every random draw: the other tables are
    # byte-identical to the pre-extension fixture.  Z for these ECGs comes
    # from admission_location (ECG inside the stay) or stays "unknown".
    adm_rows += [
        (S(5), HADM_BASE + 10, "2018-04-18 09:00:00", "2018-04-23 14:00:00", "", "EW EMER.",
         "EMERGENCY ROOM", "HOME"),                                    # contains echo 2018-04-20
        (S(7), HADM_BASE + 11, "2017-02-27 11:00:00", "2017-03-03 16:00:00", "", "URGENT",
         "PHYSICIAN REFERRAL", "HOME"),                                # contains echo 2017-03-01
        (S(8), HADM_BASE + 12, "2019-10-29 08:00:00", "2019-11-02 12:00:00", "", "EW EMER.",
         "EMERGENCY ROOM", "HOME"),                                    # contains echo 2019-10-31
        (S(12), HADM_BASE + 13, "2018-09-02 10:00:00", "2018-09-06 15:00:00", "", "ELECTIVE",
         "PHYSICIAN REFERRAL", "HOME"),                                # contains echo 2018-09-04
        (S(15), HADM_BASE + 14, "2018-07-22 07:00:00", "2018-07-27 13:00:00", "", "EW EMER.",
         "EMERGENCY ROOM", "HOME"),                                    # contains echo 2018-07-24
        (S(16), HADM_BASE + 15, "2019-06-19 18:00:00", "2019-06-24 11:00:00", "", "URGENT",
         "TRANSFER FROM HOSPITAL", "SNF"),                             # contains echo 2019-06-21
        (S(18), HADM_BASE + 16, "2018-01-06 21:00:00", "2018-01-11 10:00:00", "", "EW EMER.",
         "EMERGENCY ROOM", "HOME"),                                    # contains echo 2018-01-08
        (S(19), HADM_BASE + 17, "2018-11-11 20:00:00", "2018-11-15 12:00:00", "", "EW EMER.",
         "WALK-IN/SELF REFERRAL", "HOME"),                             # contains ECG 2018-11-12 05:41
        (S(20), HADM_BASE + 18, "2018-02-19 08:00:00", "2018-02-24 17:00:00", "", "ELECTIVE",
         "CLINIC REFERRAL", "HOME"),                                   # contains echo 2018-02-21
        (S(17), HADM_BASE + 19, "2017-04-26 15:00:00", "2017-04-29 09:00:00", "", "EW EMER.",
         "EMERGENCY ROOM", "HOME"),                                    # T=0 subject, contains ECG
    ]
    admissions = pd.DataFrame(adm_rows, columns=[
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type",
        "admission_location", "discharge_location"])
    transfers = pd.DataFrame(tr_rows, columns=[
        "subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime"])

    tables = {
        "record_list.csv": record_list,
        "echo-study-list.csv": echo_study_list,
        "echo-record-list.csv": echo_record_list,
        "hosp/patients.csv.gz": patients,
        "hosp/admissions.csv.gz": admissions,
        "hosp/transfers.csv.gz": transfers,
    }
    for name, df in tables.items():
        assert len(df) <= 50, f"{name} has {len(df)} rows (> 50)"
        if "subject_id" in df:
            assert (df["subject_id"] >= BASE_SUBJECT).all(), name
    return tables


# ----------------------------------------------------------------------------
# Extension (exp7): labs, discharge notes, ICD, ED stays, services, machine
# measurements.  Generated from the base tables with a *separate* rng so the
# base tables stay byte-identical.  <= 50 rows per table, fake ids only.
# ----------------------------------------------------------------------------
NOTE_BASE = "DS"
STAY_BASE = 60_000_000
SPECIMEN_BASE = 70_000_000

#: (label, fluid, category, itemid) -- ids follow MIMIC-IV d_labitems; the last
#: three rows are distractors that ANALYTE_PATTERNS must NOT match.
D_LABITEMS = [
    (51003, "Troponin T", "Blood", "Chemistry"),
    (51002, "Troponin I", "Blood", "Chemistry"),
    (50963, "NTproBNP", "Blood", "Chemistry"),
    (50912, "Creatinine", "Blood", "Chemistry"),
    (51222, "Hemoglobin", "Blood", "Hematology"),
    (51221, "Hematocrit", "Blood", "Hematology"),
    (50852, "% Hemoglobin A1c", "Blood", "Chemistry"),
    (51082, "Creatinine, Urine", "Urine", "Chemistry"),
]
_ANALYTE_ITEMS = {"troponin": 51003, "bnp": 50963, "creatinine": 50912, "hemoglobin": 51222}
_ANALYTE_VALUES = {"troponin": (0.01, 0.9), "bnp": (50.0, 9000.0),
                   "creatinine": (0.5, 3.5), "hemoglobin": (8.0, 16.0)}

#: (text, expected composite) -- hand-checkable by the regex extractor.
NOTE_TEMPLATES = [
    ("Brief Hospital Course: TTE on hospital day 2 showed LVEF 30% with global "
     "hypokinesis, mild MR and no AS. Started on GDMT.", 1),
    ("Echocardiogram: EF 55-60%, IVSd 0.9 cm, LVPWd 0.9 cm, trivial TR, no evidence "
     "of moderate or severe AS or MR.", 0),
    ("TTE demonstrated preserved EF (60%) but severe aortic stenosis (valve area 0.8 cm2) "
     "and moderate mitral regurgitation. Cardiac surgery consulted.", 1),
    ("Echo: LVEF 65%, IVSd 1.5 cm consistent with concentric LVH; mild AR; RVSP 30 mmHg.", 1),
    ("Repeat echo this admission showed normal LVEF, no LVH and no moderate or severe "
     "valvular disease.", 0),
    ("Patient admitted with community-acquired pneumonia, treated with ceftriaxone and "
     "azithromycin. No cardiac workup was performed.", None),
    ("Echo showed severely depressed LV systolic function and moderate to severe TR. "
     "Diuresed with IV furosemide.", 1),
    ("TTE: EF 62%, no valvular abnormalities, normal wall thickness (septum 1.0 cm).", 0),
    ("Transthoracic echo: LVEF 40%, mild MR, mild TR, no AS. Nephrology followed for AKI.", 1),
    ("Limited echo: EF 58%, IVS 1.1 cm, trace MR. Chest pain ruled out by serial troponins.", 0),
]

#: ICD-10 codes (dots removed).  SHD codes first, then non-SHD filler.
ICD_SHD = ["I5022", "I5023", "I4220", "I350", "I340", "I110", "I255", "I080"]
ICD_OTHER = ["E119", "I10", "J189", "N179", "K219", "Z87891", "E785", "F329", "R079"]
ICD9_FILLER = ["4280", "4019", "25000"]


def _study_time(record_list: pd.DataFrame) -> dict[int, pd.Timestamp]:
    return {int(s): pd.Timestamp(t) for s, t in zip(record_list["study_id"], record_list["ecg_time"])}


def build_extra_tables(base: dict[str, pd.DataFrame], seed: int = SEED + 1) -> dict[str, pd.DataFrame]:
    """The tables added for exp7 (labs, notes, ICD, ED stays, services, machine
    measurements).  ``base`` is the output of :func:`build_tables`."""
    rng = np.random.default_rng(seed)
    rec = base["record_list.csv"]
    adm = base["hosp/admissions.csv.gz"]
    echo = base["echo-study-list.csv"]
    ecg_time = _study_time(rec)
    subjects_with_echo = set(int(s) for s in echo["subject_id"])

    # --- machine_measurements.csv (one row per ECG; a few sentinel rows)
    mm_rows = []
    for i, (sid, st) in enumerate(zip(rec["subject_id"], rec["study_id"])):
        rr = float(np.clip(rng.normal(850, 120), 400, 1600))
        p_on = float(rng.integers(180, 260))
        pr = float(rng.integers(120, 210))
        qrs = float(rng.integers(76, 130))
        qt = float(rng.integers(340, 460))
        row = {"subject_id": int(sid), "study_id": int(st), "ecg_time": _fmt(ecg_time[int(st)]),
               "rr_interval": rr, "p_onset": p_on, "p_end": p_on + 100.0, "qrs_onset": p_on + pr,
               "qrs_end": p_on + pr + qrs, "t_end": p_on + pr + qt,
               "p_axis": float(rng.integers(-20, 80)), "qrs_axis": float(rng.integers(-40, 100)),
               "t_axis": float(rng.integers(-10, 90))}
        if i % 9 == 4:                      # MIMIC-IV-ECG sentinel for "unavailable"
            row["p_onset"] = 29999.0
            row["p_axis"] = 29999.0
        for k in range(18):
            row[f"report_{k}"] = ""
        mm_rows.append(row)
    machine_measurements = pd.DataFrame(mm_rows)

    # --- d_labitems / labevents (<= 50 rows: 12 ECGs x 3 analytes + edge rows)
    d_labitems = pd.DataFrame(D_LABITEMS, columns=["itemid", "label", "fluid", "category"])
    lab_rows = []
    k_spec = 1
    chosen = rec.iloc[::3].head(12)
    for sid, st in zip(chosen["subject_id"], chosen["study_id"]):
        t0 = ecg_time[int(st)]
        hadm = adm.loc[adm["subject_id"] == sid, "hadm_id"]
        h = int(hadm.iloc[0]) if len(hadm) else ""
        for a in ("troponin", "bnp", "creatinine"):
            lo_v, hi_v = _ANALYTE_VALUES[a]
            v = float(np.round(rng.uniform(lo_v, hi_v), 2))
            t = t0 - pd.Timedelta(hours=float(rng.uniform(1, 20)))
            lab_rows.append((int(sid), h, SPECIMEN_BASE + k_spec, _ANALYTE_ITEMS[a], _fmt(t), str(v), v))
            k_spec += 1
    # edge rows: charted AFTER the ECG, > 24 h BEFORE the ECG, a distractor item
    st0 = int(rec["study_id"].iloc[0]); sid0 = int(rec["subject_id"].iloc[0])
    lab_rows.append((sid0, HADM_BASE + 1, SPECIMEN_BASE + k_spec, 51222,
                     _fmt(ecg_time[st0] + pd.Timedelta(hours=2)), "13.1", 13.1))       # after -> ignored
    lab_rows.append((sid0, HADM_BASE + 1, SPECIMEN_BASE + k_spec + 1, 51222,
                     _fmt(ecg_time[st0] - pd.Timedelta(hours=30)), "12.4", 12.4))      # too early
    lab_rows.append((sid0, HADM_BASE + 1, SPECIMEN_BASE + k_spec + 2, 51222,
                     _fmt(ecg_time[st0] - pd.Timedelta(hours=5)), "11.8", 11.8))       # used
    lab_rows.append((sid0, HADM_BASE + 1, SPECIMEN_BASE + k_spec + 3, 50852,
                     _fmt(ecg_time[st0] - pd.Timedelta(hours=5)), "7.2", 7.2))         # A1c: never an analyte
    labevents = pd.DataFrame(lab_rows, columns=["subject_id", "hadm_id", "specimen_id", "itemid",
                                                "charttime", "value", "valuenum"])

    # --- ed/edstays.csv.gz: one ED stay per "ED" transfer + one stay without admission
    ed_rows = []
    tr = base["hosp/transfers.csv.gz"]
    k_stay = 1
    for r in tr[tr["eventtype"] == "ED"].itertuples(index=False):
        ed_rows.append((int(r.subject_id), int(r.hadm_id), STAY_BASE + k_stay, r.intime, r.outtime,
                        "F" if k_stay % 2 else "M", "WHITE", "AMBULANCE", "ADMITTED"))
        k_stay += 1
    ed_rows.append((BASE_SUBJECT + 13, "", STAY_BASE + k_stay, "2017-04-03 09:00:00",
                    "2017-04-03 15:00:00", "F", "ASIAN", "WALK IN", "HOME"))  # contains ECG 40000025
    edstays = pd.DataFrame(ed_rows, columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime",
                                             "gender", "race", "arrival_transport", "disposition"])

    # --- hosp/services.csv.gz
    sv_rows = [
        (BASE_SUBJECT + 2, HADM_BASE + 2, "2017-12-28 10:00:00", "", "MED"),
        (BASE_SUBJECT + 2, HADM_BASE + 2, "2017-12-30 09:00:00", "MED", "CMED"),
        (BASE_SUBJECT + 9, HADM_BASE + 4, "2018-05-25 13:00:00", "", "MED"),
        (BASE_SUBJECT + 10, HADM_BASE + 5, "2018-07-27 17:00:00", "", "MED"),
        (BASE_SUBJECT + 19, HADM_BASE + 17, "2018-11-11 20:00:00", "", "CMED"),
        (BASE_SUBJECT + 17, HADM_BASE + 19, "2017-04-26 15:00:00", "", "MED"),
    ]
    services = pd.DataFrame(sv_rows, columns=["subject_id", "hadm_id", "transfertime",
                                              "prev_service", "curr_service"])

    # --- note/discharge.csv.gz: one summary per admission, templates cycled
    note_rows = []
    for i, r in enumerate(adm.sort_values("hadm_id").itertuples(index=False)):
        text, _ = NOTE_TEMPLATES[i % len(NOTE_TEMPLATES)]
        chart = pd.Timestamp(r.dischtime)
        note_rows.append((f"{int(r.subject_id)}-{NOTE_BASE}-{i + 1}", int(r.subject_id), int(r.hadm_id),
                          "DS", 1, _fmt(chart), _fmt(chart + pd.Timedelta(hours=3)), text))
    discharge = pd.DataFrame(note_rows, columns=["note_id", "subject_id", "hadm_id", "note_type",
                                                 "note_seq", "charttime", "storetime", "text"])

    # --- records_w_diag_icd10.csv: most ECGs; SHD codes more frequent when the
    # subject has an echo (deliberate circularity for the diagnostic)
    icd_rows = []
    for i, r in enumerate(rec.itertuples(index=False)):
        if i % 7 == 6:
            continue                                     # absent -> hosp fallback / none
        linked = int(r.subject_id) in subjects_with_echo
        codes = list(rng.choice(ICD_OTHER, size=2, replace=False))
        if rng.uniform() < (0.7 if linked else 0.2):
            codes.append(str(rng.choice(ICD_SHD)))
        icd_rows.append((r.file_name, int(r.study_id), int(r.subject_id), r.ecg_time, str(codes)))
    records_w_diag_icd10 = pd.DataFrame(icd_rows, columns=["file_name", "study_id", "subject_id",
                                                           "ecg_time", "all_diag_all"])

    # --- hosp/diagnoses_icd.csv.gz: codes per admission (one ICD-9-only admission)
    dx_rows = []
    for i, r in enumerate(adm.sort_values("hadm_id").itertuples(index=False)):
        if i == 3:                                       # ICD-9 only
            for k, c in enumerate(ICD9_FILLER[:2]):
                dx_rows.append((int(r.subject_id), int(r.hadm_id), k + 1, c, 9))
            continue
        codes = list(rng.choice(ICD_OTHER, size=1))
        if i % 2 == 0:
            codes.append(str(rng.choice(ICD_SHD)))
        for k, c in enumerate(codes):
            dx_rows.append((int(r.subject_id), int(r.hadm_id), k + 1, c, 10))
    diagnoses_icd = pd.DataFrame(dx_rows, columns=["subject_id", "hadm_id", "seq_num", "icd_code",
                                                   "icd_version"])

    extra = {
        "machine_measurements.csv": machine_measurements,
        "hosp/d_labitems.csv.gz": d_labitems,
        "hosp/labevents.csv.gz": labevents,
        "hosp/services.csv.gz": services,
        "hosp/diagnoses_icd.csv.gz": diagnoses_icd,
        "ed/edstays.csv.gz": edstays,
        "note/discharge.csv.gz": discharge,
        "records_w_diag_icd10.csv": records_w_diag_icd10,
    }
    for name, df in extra.items():
        assert len(df) <= 50, f"{name} has {len(df)} rows (> 50)"
        if "subject_id" in df:
            assert (pd.to_numeric(df["subject_id"]) >= BASE_SUBJECT).all(), name
    return extra


def build_all_tables(seed: int = SEED) -> dict[str, pd.DataFrame]:
    base = build_tables(seed)
    out = dict(base)
    out.update(build_extra_tables(base, seed + 1))
    return out


def write_tables(tables: dict[str, pd.DataFrame], out_dir: str) -> list[str]:
    written = []
    for name, df in tables.items():
        path = os.path.join(out_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        csv = df.to_csv(index=False, lineterminator="\n").encode()
        if name.endswith(".gz"):
            buf = io.BytesIO()
            with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as g:
                g.write(csv)
            data = buf.getvalue()
        else:
            data = csv
        with open(path, "wb") as f:
            f.write(data)
        written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    if os.sep + "results" + os.sep in out + os.sep:
        raise SystemExit("refusing to write fixtures under results/")
    for p in write_tables(build_all_tables(), out):
        print(p)


if __name__ == "__main__":
    main()
