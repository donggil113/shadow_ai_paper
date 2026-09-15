#!/usr/bin/env python3
"""Snapshot / verify that formal statements survive editing unchanged.

    python3 scripts/check_theorem_preservation.py snapshot   # writes docs/theorem_snapshot.json
    python3 scripts/check_theorem_preservation.py verify     # compares current tex to the snapshot

A statement is every theorem/proposition/lemma/corollary/definition/assumption
environment, keyed by its \\label. Whitespace is normalised; anything else that
differs is reported. Statements present in the snapshot but missing now are
reported as REMOVED (allowed only if listed in docs/theorem_moves.json under
"removed" with a reason); new labels are reported as ADDED (informational).
"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(ROOT, "docs", "theorem_snapshot.json")
MOVES = os.path.join(ROOT, "docs", "theorem_moves.json")
ENVS = "theorem|proposition|lemma|corollary|definition|assumption"


def collect():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "paper", "sections", "*.tex"))):
        s = open(f).read()
        for m in re.finditer(r"\\begin\{(%s)\}(\[[^\]]*\])?(.*?)\\end\{\1\}" % ENVS, s, re.S):
            body = m.group(3)
            lab = re.search(r"\\label\{([^}]+)\}", body)
            key = lab.group(1) if lab else f"{os.path.basename(f)}:{m.start()}"
            norm = re.sub(r"\s+", " ", re.sub(r"\\label\{[^}]+\}", "", body)).strip()
            out[key] = {"env": m.group(1), "title": (m.group(2) or "").strip("[]"),
                        "file": os.path.basename(f), "text": norm}
    return out


def main(cmd):
    cur = collect()
    if cmd == "snapshot":
        json.dump(cur, open(SNAP, "w"), indent=1)
        print(f"snapshot: {len(cur)} statements -> {SNAP}")
        return 0
    snap = json.load(open(SNAP))
    allowed = json.load(open(MOVES)) if os.path.exists(MOVES) else {"removed": {}, "changed": {}}
    bad = 0
    for k, v in snap.items():
        if k not in cur:
            if k in allowed.get("removed", {}):
                print(f"  removed (allowed): {k} -- {allowed['removed'][k]}")
            else:
                print(f"REMOVED: {k} ({v['env']} in {v['file']})"); bad += 1
        elif cur[k]["text"] != v["text"]:
            if k in allowed.get("changed", {}):
                print(f"  changed (allowed): {k} -- {allowed['changed'][k]}")
            else:
                print(f"CHANGED: {k} ({v['env']}, {v['file']} -> {cur[k]['file']})")
                bad += 1
    for k in cur:
        if k not in snap:
            print(f"  added: {k} ({cur[k]['env']} in {cur[k]['file']})")
    print(f"{len(snap)} snapshot statements checked; {bad} unexplained differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "verify"))
