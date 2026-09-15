"""Shared plumbing for the experiment scripts."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
FIGURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "paper", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)


def save(name: str, payload: Any) -> str:
    path = os.path.join(RESULTS, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_default)
    print(f"  -> {path}")
    return path


def save_table(name: str, df) -> str:
    path = os.path.join(RESULTS, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  -> {path}")
    return path


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"[{self.label}] start", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"[{self.label}] done in {time.time() - self.t:.1f}s", flush=True)


DATA_SOURCES = ("real", "simulator", "semi-synthetic", "synthetic")


def ci95(values):
    """Mean and a 95% confidence interval (t-based) for a list of seed values.

    Returns ``(mean, half_width, n)``; ``half_width`` is NaN when ``n < 2``.
    """
    from scipy import stats
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(v[0]), float("nan"), 1
    hw = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n))
    return float(v.mean()), hw, int(n)


def stamp_provenance(name: str, data_source: str, per_dataset: dict | None = None) -> None:
    """Add the mandatory ``data_source`` field to an existing results JSON."""
    if data_source not in DATA_SOURCES:
        raise ValueError(f"data_source must be one of {DATA_SOURCES}, got {data_source!r}")
    path = os.path.join(RESULTS, f"{name}.json")
    d = json.load(open(path))
    d["data_source"] = data_source
    if per_dataset:
        d["data_source_by_dataset"] = dict(per_dataset)
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=_default)


def save_provenanced(name: str, payload: dict, data_source: str) -> str:
    """Like :func:`save` but stamps the mandatory ``data_source`` field (rule 3).

    ``data_source`` is one of ``DATA_SOURCES``; use ``"mixed"``-free per-dataset
    entries inside ``payload`` when a file combines sources, and set the
    top-level value to the *least* trustworthy source present.
    """
    if data_source not in DATA_SOURCES:
        raise ValueError(f"data_source must be one of {DATA_SOURCES}, got {data_source!r}")
    payload = dict(payload)
    payload["data_source"] = data_source
    return save(name, payload)
