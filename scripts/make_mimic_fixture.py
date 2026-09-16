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
    for p in write_tables(build_tables(), out):
        print(p)


if __name__ == "__main__":
    main()
