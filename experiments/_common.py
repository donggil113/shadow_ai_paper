"""Shared plumbing for the experiment scripts."""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

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
