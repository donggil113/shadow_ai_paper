#!/usr/bin/env python3
"""Run every experiment in order, then regenerate the figures."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["exp1_sharpness.py", "exp2_learning.py", "exp3_uq.py",
           "exp4_gamma.py", "exp5_generalization.py", "exp6_realdata.py",
           "make_figures.py"]

if __name__ == "__main__":
    only = sys.argv[1:]
    for s in SCRIPTS:
        if only and not any(o in s for o in only):
            continue
        print(f"\n{'=' * 70}\n== {s}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE)
        if r.returncode != 0:
            print(f"!! {s} exited with {r.returncode}")
