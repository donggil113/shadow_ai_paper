"""MIMIC-IV-ECG x MIMIC-IV-ECHO real medical arm (spec: docs/mimic_ecg_echo_spec.md).

Real data only.  This package reads from ``--data-root`` through
:class:`~dcl.data.mimic_ecg_echo.io.DataRoot`, exits with code 2 when a file is
missing or a header does not match, and never imports
``dcl.data.mimic.make_mimic_sim`` (tests/test_mimic_io_linkage.py enforces it).
"""

from .io import (DATASETS, DATETIME_COLUMNS, EXIT_MISSING_DATA, EXPECTED_COLUMNS,
                 HASH_BYTES, TABLES, DataRoot, sha256_prefix)
from .linkage import (Z_UNKNOWN, attach_decision_maker, compute_echo_coverage_window,
                      dedup_ecgs, link_ecg_to_echo, restrict_to_coverage)

__all__ = [
    "DATASETS", "DATETIME_COLUMNS", "EXIT_MISSING_DATA", "EXPECTED_COLUMNS",
    "HASH_BYTES", "TABLES", "DataRoot", "sha256_prefix",
    "Z_UNKNOWN", "attach_decision_maker", "compute_echo_coverage_window",
    "dedup_ecgs", "link_ecg_to_echo", "restrict_to_coverage",
]
