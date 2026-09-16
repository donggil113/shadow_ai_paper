#!/usr/bin/env python3
"""Fail if a real-data / clinical section of the paper cites a simulator number.

CLAUDE.md rule 3.  Every number in the paper is a macro from paper/numbers.tex
whose provenance (real / simulator / semi-synthetic / synthetic) is recorded in
paper/numbers_provenance.json by scripts/make_numbers.py.  This checker reads
paper/PROVENANCE_SECTIONS.json, locates every macro use inside each configured
scope and checks its provenance against the scope's rule.  It also refuses
macros that are used in the .tex files but not defined in numbers.tex.

    python3 scripts/check_provenance.py            # exit 1 on any violation
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "paper", "PROVENANCE_SECTIONS.json")
PROV = os.path.join(ROOT, "paper", "numbers_provenance.json")
NUMS = os.path.join(ROOT, "paper", "numbers.tex")

MACRO_RE = re.compile(r"\\(n[A-Z][A-Za-z]*)")
HEAD_RE = re.compile(r"\\(sub)*section\*?\{")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")


def strip_comments(tex: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())


def blocks(tex: str):
    """Yield (label, text) for each \\section / \\subsection block (label may be None)."""
    lines = strip_comments(tex).splitlines()
    starts = [i for i, ln in enumerate(lines) if HEAD_RE.search(ln)]
    bounds = [0] + starts + [len(lines)]
    seen = set()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a == b:
            continue
        chunk = "\n".join(lines[a:b])
        m = LABEL_RE.search(chunk)
        label = m.group(1) if m else None
        if (a, b) in seen:
            continue
        seen.add((a, b))
        yield label, chunk


def macros_in(text: str, defined: set) -> list:
    return [m for m in MACRO_RE.findall(text) if m in defined or m.rstrip("CI") in defined]


def main() -> int:
    cfg = json.load(open(CFG))
    prov = json.load(open(PROV)) if os.path.exists(PROV) else {}
    defined = set(re.findall(r"\\newcommand\{\\(n[A-Za-z]+)\}", open(NUMS).read())) if os.path.exists(NUMS) else set()
    problems, undefined = [], []

    def provenance(macro: str) -> str:
        base = macro[:-2] if macro.endswith("CI") and macro not in prov else macro
        return prov.get(base, {}).get("data_source", "unknown")

    def check_scope(path: str, label: str | None, text: str, forbid=(), require=(), why=""):
        for m in sorted(set(MACRO_RE.findall(text))):
            if m not in defined:
                undefined.append((path, label, m)); continue
            src = provenance(m)
            if src == "n/a":
                continue
            if src == "pending":
                problems.append(f"{path}#{label or '-'}: \\{m} is a PENDING placeholder (result not yet computed)"); continue
            if (forbid and src in forbid) or (require and src not in require):
                problems.append(f"{path}#{label or '-'}: \\{m} has provenance '{src}' ({why or 'rule'})")

    # 1. main text: no simulator macros except in allowed scopes
    allowed = set(cfg.get("simulator_allowed", []))
    for rel in cfg.get("main_text", []):
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        for label, chunk in blocks(open(path).read()):
            if f"{rel}#{label}" in allowed:
                continue
            check_scope(rel, label, chunk, forbid=tuple(cfg.get("main_text_forbid", [])), why="main text may not cite simulator results")
    # 2. explicit scopes
    for scope, rule in cfg.get("scopes", {}).items():
        rel, _, lab = scope.partition("#")
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        tex = open(path).read()
        if lab:
            for label, chunk in blocks(tex):
                if label == lab:
                    check_scope(rel, label, chunk, tuple(rule.get("forbid", [])), tuple(rule.get("require", [])), rule.get("why", ""))
        else:
            check_scope(rel, None, strip_comments(tex), tuple(rule.get("forbid", [])), tuple(rule.get("require", [])), rule.get("why", ""))
    # 3. undefined macros anywhere in paper/
    for dirpath, _, files in os.walk(os.path.join(ROOT, "paper")):
        for fn in files:
            if fn.endswith(".tex") and fn != "numbers.tex":
                rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
                for m in sorted(set(MACRO_RE.findall(strip_comments(open(os.path.join(dirpath, fn)).read())))):
                    if m not in defined and (rel, None, m) not in undefined:
                        undefined.append((rel, None, m))
    for dirpath, _, files in os.walk(os.path.join(ROOT, "paper")):
        for fn in files:
            if fn.endswith(".tex") and fn != "numbers.tex":
                rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
                for m in sorted(set(MACRO_RE.findall(strip_comments(open(os.path.join(dirpath, fn)).read())))):
                    if m in defined and provenance(m) == "pending" and not any(m in p for p in problems):
                        problems.append(f"{rel}: \\{m} is a PENDING placeholder (result not yet computed)")
    seen = set()
    for rel, label, m in undefined:
        if m in seen:
            continue
        seen.add(m)
        problems.append(f"{rel}: \\{m} is used but not defined in paper/numbers.tex (results missing or stale)")
    for p in problems:
        print("PROVENANCE:", p)
    if problems:
        print(f"{len(problems)} provenance problem(s)")
        return 1
    print("provenance check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
