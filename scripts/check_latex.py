#!/usr/bin/env python3
"""Structural sanity check for the paper (no LaTeX toolchain required).

Checks environment nesting, brace balance, undefined/duplicate labels, missing
bib entries, missing \\input files and missing figures.
"""
from __future__ import annotations

import collections
import glob
import os
import re
import sys


def strip(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", s)      # comments, but not an escaped \%
    return re.sub(r"\\[{}%&_#$]", "", s)  # escaped specials


def main(root: str = "paper") -> int:
    files = [os.path.join(root, "main.tex")] + sorted(
        glob.glob(os.path.join(root, "sections", "*.tex")))
    src = {f: open(f).read() for f in files}
    allsrc = "\n".join(src.values())
    problems = []

    for f, s in src.items():
        stack = []
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
            line = s[: m.start()].count("\n") + 1
            if m.group(1) == "begin":
                stack.append((m.group(2), line))
            elif not stack or stack[-1][0] != m.group(2):
                problems.append(f"{f}:{line} stray \\end{{{m.group(2)}}}")
            else:
                stack.pop()
        problems += [f"{f}:{ln} unclosed \\begin{{{nm}}}" for nm, ln in stack]
        t = strip(s)
        d = t.count("{") - t.count("}")
        if d:
            problems.append(f"{f} brace imbalance {d:+d}")

    labels = re.findall(r"\\label\{([^}]+)\}", allsrc)
    refs = set(re.findall(r"\\[cC]?ref\{([^}]+)\}", allsrc)) \
        | set(re.findall(r"\\eqref\{([^}]+)\}", allsrc))
    problems += [f"undefined ref: {r}" for r in sorted(refs - set(labels))]
    problems += [f"duplicate label: {k}" for k, v
                 in collections.Counter(labels).items() if v > 1]

    cites = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", allsrc):
        cites |= {c.strip() for c in m.group(1).split(",")}
    bib = set(re.findall(r"@\w+\{([^,]+),",
                         open(os.path.join(root, "refs.bib")).read()))
    problems += [f"missing bib entry: {c}" for c in sorted(cites - bib)]

    for i in re.findall(r"\\input\{([^}]+)\}", allsrc):
        if not os.path.exists(os.path.join(root, i + ".tex")):
            problems.append(f"missing \\input: {i}.tex")
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", allsrc):
        if not os.path.exists(os.path.join(root, g + ".pdf")):
            problems.append(f"missing figure: {g}.pdf")

    print(f"{len(files)} files, {sum(s.count(chr(10)) for s in src.values())} lines, "
          f"{len(set(labels))} labels, {len(cites)} citations")
    unused = sorted(set(labels) - refs)
    if unused:
        print(f"note: {len(unused)} labels never referenced: {', '.join(unused)}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  " + p)
        return 1
    print("OK - no structural problems")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
