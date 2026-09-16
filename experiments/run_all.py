#!/usr/bin/env python3
"""Run every experiment in order, then regenerate the figures."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# exp7 (real medical arm) is deliberately absent: it needs the credentialed
# MIMIC-IV files on the GPU server and exits 2 without them (CLAUDE.md rule 1);
# run it explicitly with `python3 experiments/exp7_mimic_ecg_echo.py --data-root /data`.
SCRIPTS = ["exp0_counterexample.py", "exp1_sharpness.py", "exp2_learning.py", "exp3_uq.py",
           "exp4_gamma.py", "exp5_generalization.py", "exp6_realdata.py",
           "exp8_nuisance_coverage.py", "exp10_ranker.py",
           "make_figures.py"]
POST = [  # regenerate the paper's numbers and check them after the experiments
    ["python3", "audit/t3_ranking_search.py"],
    ["python3", "audit/compas_time_at_risk.py"],
    ["python3", "scripts/make_numbers.py"],
    ["python3", "scripts/verify_paper_numbers.py"],
    ["python3", "scripts/make_status.py"],
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for s in SCRIPTS:
        if only and not any(o in s for o in only):
            continue
        print(f"\n{'=' * 70}\n== {s}\n{'=' * 70}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(HERE, s)], cwd=HERE)
        if r.returncode != 0:
            print(f"!! {s} exited with {r.returncode}")
    if not only:
        root = os.path.dirname(HERE)
        for cmd in POST:
            print(f"\n== {' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd, cwd=root)
            if r.returncode != 0:
                print(f"!! {' '.join(cmd)} exited with {r.returncode}")
