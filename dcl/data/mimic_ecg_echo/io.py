"""File-system access for the MIMIC-IV-ECG x MIMIC-IV-ECHO arm (T1).

Binding rules (CLAUDE.md rule 1, docs/mimic_ecg_echo_spec.md):

* every table is read from ``--data-root`` (``/data`` on the GPU server);
* a missing file prints the absolute path that was looked for and the
  PhysioNet page it must be obtained from, then ``sys.exit(2)``;
* the header of every table is verified against ``EXPECTED_COLUMNS`` at load
  time; on mismatch the actual header, the expected header and the dataset
  version the loader was written against are printed and the process exits 2;
* no simulator is ever substituted.  ``dcl.data.mimic.make_mimic_sim`` is not
  imported anywhere in this package.

Every loader returns ``(DataFrame, provenance)`` where ``provenance`` is a plain
dict with at least the keys ``source`` (``"real"`` or ``"fixture"``), ``path``,
``n_rows``, ``sha256`` (of the first 1 MiB of the file), ``columns``.  A
provenance whose ``source`` is ``"fixture"`` may never reach ``results/``.

Layout.  Two layouts are accepted.  The production layout is the versioned
PhysioNet tree (``<root>/mimic-iv-ecg/1.0/record_list.csv``,
``<root>/mimic-iv/3.1/hosp/patients.csv.gz`` ...).  The flat layout used by the
pytest fixture (``<root>/record_list.csv``, ``<root>/hosp/patients.csv.gz``) is
tried second.  The error message on a missing file names the production path.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

__all__ = [
    "DATASETS",
    "DATETIME_COLUMNS",
    "EXPECTED_COLUMNS",
    "TABLES",
    "DataRoot",
    "EXIT_MISSING_DATA",
    "HASH_BYTES",
    "sha256_prefix",
]

EXIT_MISSING_DATA = 2
HASH_BYTES = 1 << 20  # first 1 MiB

# dataset key -> (version the schemas were written against, PhysioNet page)
DATASETS: Dict[str, Dict[str, str]] = {
    "mimic-iv-ecg": {"version": "1.0",
                     "url": "https://physionet.org/content/mimic-iv-ecg/1.0/"},
    "mimic-iv": {"version": "3.1",
                 "url": "https://physionet.org/content/mimiciv/3.1/"},
    "mimic-iv-ed": {"version": "2.2",
                    "url": "https://physionet.org/content/mimic-iv-ed/2.2/"},
    "mimic-iv-echo": {"version": "0.1",
                      "url": "https://physionet.org/content/mimic-iv-echo/0.1/"},
    "mimic-iv-note": {"version": "2.2",
                      "url": "https://physionet.org/content/mimic-iv-note/2.2/"},
    "mimic-iv-ecg-ext-icd": {"version": "1.0.1",
                             "url": "https://physionet.org/content/mimic-iv-ecg-ext-icd/1.0.1/"},
}

# Columns every table MUST contain (a superset in the file is tolerated; the
# spec lists "..." for several tables).  Written from the public documentation
# of the versions in ``DATASETS``; any discrepancy found on the real server must
# be fixed here, never papered over.
EXPECTED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "record_list": ("subject_id", "study_id", "file_name", "ecg_time"),
    "machine_measurements": (
        "subject_id", "study_id", "ecg_time",
        "rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end",
        "p_axis", "qrs_axis", "t_axis",
    ) + tuple(f"report_{i}" for i in range(18)),
    "patients": ("subject_id", "gender", "anchor_age", "anchor_year",
                 "anchor_year_group", "dod"),
    "admissions": ("subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                   "admission_type", "admission_location"),
    "transfers": ("subject_id", "hadm_id", "transfer_id", "eventtype", "careunit",
                  "intime", "outtime"),
    "services": ("subject_id", "hadm_id", "transfertime", "prev_service",
                 "curr_service"),
    "diagnoses_icd": ("subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"),
    "labevents": ("subject_id", "hadm_id", "specimen_id", "itemid", "charttime",
                  "value", "valuenum"),
    "d_labitems": ("itemid", "label", "fluid", "category"),
    "edstays": ("subject_id", "hadm_id", "stay_id", "intime", "outtime", "gender",
                "race", "arrival_transport", "disposition"),
    "triage": ("subject_id", "stay_id", "temperature", "heartrate", "resprate",
               "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"),
    "echo_record_list": ("subject_id", "study_id", "dicom_filepath"),
    "echo_study_list": ("subject_id", "study_id", "study_datetime"),
    "discharge": ("note_id", "subject_id", "hadm_id", "note_type", "note_seq",
                  "charttime", "storetime", "text"),
    "records_w_diag_icd10": ("file_name", "study_id", "subject_id", "ecg_time",
                             "all_diag_all"),
}

# table -> (dataset key, production relative path, flat fixture relative path)
TABLES: Dict[str, Tuple[str, str, str]] = {
    "record_list": ("mimic-iv-ecg", "mimic-iv-ecg/1.0/record_list.csv",
                    "record_list.csv"),
    "machine_measurements": ("mimic-iv-ecg", "mimic-iv-ecg/1.0/machine_measurements.csv",
                             "machine_measurements.csv"),
    "patients": ("mimic-iv", "mimic-iv/3.1/hosp/patients.csv.gz", "hosp/patients.csv.gz"),
    "admissions": ("mimic-iv", "mimic-iv/3.1/hosp/admissions.csv.gz",
                   "hosp/admissions.csv.gz"),
    "transfers": ("mimic-iv", "mimic-iv/3.1/hosp/transfers.csv.gz",
                  "hosp/transfers.csv.gz"),
    "services": ("mimic-iv", "mimic-iv/3.1/hosp/services.csv.gz", "hosp/services.csv.gz"),
    "diagnoses_icd": ("mimic-iv", "mimic-iv/3.1/hosp/diagnoses_icd.csv.gz",
                      "hosp/diagnoses_icd.csv.gz"),
    "labevents": ("mimic-iv", "mimic-iv/3.1/hosp/labevents.csv.gz",
                  "hosp/labevents.csv.gz"),
    "d_labitems": ("mimic-iv", "mimic-iv/3.1/hosp/d_labitems.csv.gz",
                   "hosp/d_labitems.csv.gz"),
    "edstays": ("mimic-iv-ed", "mimic-iv-ed/2.2/ed/edstays.csv.gz", "ed/edstays.csv.gz"),
    "triage": ("mimic-iv-ed", "mimic-iv-ed/2.2/ed/triage.csv.gz", "ed/triage.csv.gz"),
    "echo_record_list": ("mimic-iv-echo", "mimic-iv-echo/0.1/echo-record-list.csv",
                         "echo-record-list.csv"),
    "echo_study_list": ("mimic-iv-echo", "mimic-iv-echo/0.1/echo-study-list.csv",
                        "echo-study-list.csv"),
    "discharge": ("mimic-iv-note", "mimic-iv-note/2.2/note/discharge.csv.gz",
                  "note/discharge.csv.gz"),
    "records_w_diag_icd10": ("mimic-iv-ecg-ext-icd",
                             "mimic-iv-ecg-ext-icd/1.0.1/records_w_diag_icd10.csv",
                             "records_w_diag_icd10.csv"),
}

# Columns parsed as datetimes when present.
DATETIME_COLUMNS: Tuple[str, ...] = (
    "ecg_time", "study_datetime", "admittime", "dischtime", "deathtime",
    "intime", "outtime", "transfertime", "charttime", "storetime", "dod",
)

_FIXTURE_DIR_NAMES = ("fixtures", "fixture")


def sha256_prefix(path: str, n_bytes: int = HASH_BYTES) -> str:
    """SHA-256 of the first ``n_bytes`` of ``path`` (raw bytes, gz included)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()


def _die(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    sys.exit(EXIT_MISSING_DATA)


@dataclass
class DataRoot:
    """Resolves and loads tables under ``--data-root``.

    Parameters
    ----------
    root : str
        The ``--data-root`` directory.
    source : {"real", "fixture"} or None
        Provenance label attached to every table.  ``None`` auto-detects: a root
        that has a ``fixtures``/``fixture`` path component is a fixture,
        everything else is real.  A root under a fixtures directory can never be
        labelled ``"real"`` (``ValueError``).
    """

    root: str
    source: Optional[str] = None
    loaded: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.root = os.path.abspath(os.path.expanduser(str(self.root)))
        parts = set(os.path.normpath(self.root).split(os.sep))
        looks_fixture = bool(parts & set(_FIXTURE_DIR_NAMES))
        if self.source is None:
            self.source = "fixture" if looks_fixture else "real"
        if self.source not in ("real", "fixture"):
            raise ValueError(f"source must be 'real' or 'fixture', got {self.source!r}")
        if self.source == "real" and looks_fixture:
            raise ValueError(f"refusing to label a fixture directory as real data: {self.root}")

    # ------------------------------------------------------------- paths
    def candidates(self, name: str) -> List[str]:
        if name not in TABLES:
            raise KeyError(f"unknown table {name!r}; known: {sorted(TABLES)}")
        _, prod, flat = TABLES[name]
        return [os.path.join(self.root, prod), os.path.join(self.root, flat)]

    def path(self, name: str) -> str:
        """Absolute path of table ``name`` (first existing candidate, else the
        production path so that the error message is precise)."""
        cands = self.candidates(name)
        for c in cands:
            if os.path.isfile(c):
                return c
        return cands[0]

    def require(self, path: str, dataset: Optional[str] = None) -> str:
        """Return ``path`` if it exists, else print where it should come from
        and ``sys.exit(2)``.  Never substitutes anything."""
        path = os.path.abspath(path)
        if os.path.isfile(path):
            return path
        lines = ["[dcl.data.mimic_ecg_echo] REQUIRED DATA FILE IS MISSING",
                 f"  looked for : {path}",
                 f"  data root  : {self.root}"]
        if dataset in DATASETS:
            d = DATASETS[dataset]
            lines.append(f"  obtain from: {d['url']}  ({dataset} v{d['version']}, "
                         "credentialed PhysioNet resource; credentials in ~/.netrc)")
        else:
            lines.append("  obtain from: https://physionet.org/ (credentialed resource)")
        lines.append("  No simulator is substituted (CLAUDE.md rule 1). Exiting with code 2.")
        _die("\n".join(lines))
        return path  # unreachable

    # ------------------------------------------------------------- loading
    def load_table(self, name: str, usecols: Optional[Sequence[str]] = None,
                   nrows: Optional[int] = None, parse_dates: bool = True,
                   ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Read table ``name`` and verify its header.

        Returns ``(df, provenance)``.  ``usecols`` restricts the columns that are
        materialised (the header check always runs on the full header).
        """
        dataset, _, _ = TABLES[name]
        path = self.require(self.path(name), dataset)
        expected = EXPECTED_COLUMNS[name]
        header = list(pd.read_csv(path, nrows=0).columns)
        missing = [c for c in expected if c not in header]
        if missing:
            d = DATASETS[dataset]
            _die("\n".join([
                "[dcl.data.mimic_ecg_echo] SCHEMA MISMATCH",
                f"  table    : {name}",
                f"  file     : {path}",
                f"  written against: {dataset} v{d['version']} ({d['url']})",
                f"  expected columns (required subset): {list(expected)}",
                f"  actual header                     : {header}",
                f"  missing                           : {missing}",
                "  Fix the loader in dcl/data/mimic_ecg_echo/io.py (do not paper over). "
                "Exiting with code 2.",
            ]))
        cols = None
        if usecols is not None:
            unknown = [c for c in usecols if c not in header]
            if unknown:
                _die(f"[dcl.data.mimic_ecg_echo] requested columns {unknown} absent from "
                     f"{path}; header = {header}. Exiting with code 2.")
            cols = list(usecols)
        df = pd.read_csv(path, usecols=cols, nrows=nrows, low_memory=False)
        if parse_dates:
            for c in DATETIME_COLUMNS:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        prov = {
            "source": self.source,
            "table": name,
            "dataset": dataset,
            "dataset_version": DATASETS[dataset]["version"],
            "path": path,
            "n_rows": int(len(df)),
            "sha256": sha256_prefix(path),
            "sha256_bytes": HASH_BYTES,
            "columns": list(df.columns),
            "header": header,
        }
        self.loaded[name] = prov
        return df, prov

    def all_real(self) -> bool:
        """True iff every table loaded so far carries ``source == "real"``."""
        return bool(self.loaded) and all(p["source"] == "real" for p in self.loaded.values())
