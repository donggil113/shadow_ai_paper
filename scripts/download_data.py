#!/usr/bin/env python3
"""Fetch the public datasets used in the paper into ``data/raw/``.

Everything here is public and redistributable.  MIMIC-IV is **not** downloaded:
it is a credentialed PhysioNet resource, and ``dcl/data/mimic.py`` documents how
to obtain it and what file layout the loader expects.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

RDATASETS = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv"
SOURCES = {
    "modeldata_lending_club.csv": f"{RDATASETS}/modeldata/lending_club.csv",
    "openintro_loans_full_schema.csv": f"{RDATASETS}/openintro/loans_full_schema.csv",
    "AER_CreditCard.csv": f"{RDATASETS}/AER/CreditCard.csv",
    "AER_HMDA.csv": f"{RDATASETS}/AER/HMDA.csv",
    "modeldata_credit_data.csv": f"{RDATASETS}/modeldata/credit_data.csv",
    "compas_two_years.csv": ("https://raw.githubusercontent.com/propublica/"
                             "compas-analysis/master/compas-scores-two-years.csv"),
}


def main(out_dir: str = "data/raw") -> int:
    os.makedirs(out_dir, exist_ok=True)
    for name, url in SOURCES.items():
        dest = os.path.join(out_dir, name)
        if os.path.exists(dest):
            print(f"  [skip] {name}")
            continue
        print(f"  [get ] {name} <- {url}")
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
    print("\nsha256:")
    for name in SOURCES:
        dest = os.path.join(out_dir, name)
        h = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        print(f"  {h[:16]}...  {os.path.getsize(dest):>9,d}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
