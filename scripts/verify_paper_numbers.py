#!/usr/bin/env python3
"""The paper's numbers gate (CLAUDE.md rules 2-4).  Must always pass.

1. paper/numbers.tex is exactly what scripts/make_numbers.py would generate from
   the current results/ (no stale macros);
2. no hand-typed numeric literal appears in the main text or appendix prose:
   every experimental number is a \\n... macro.  Structural numbers (fractions
   in theorem statements, design constants such as the Gamma grid, figure
   widths, table layout) are matched by an explicit, justified allowlist in
   paper/NUMBER_ALLOWLIST.txt -- anything else fails;
3. scripts/check_provenance.py passes (no simulator number in a real-data /
   clinical section; every macro used is defined);
4. every macro cited in the paper that carries a CI companion comes from at
   least 5 seeds (rule 4) -- checked through numbers_provenance.json.

    python3 scripts/verify_paper_numbers.py           # exit 1 on failure
    python3 scripts/verify_paper_numbers.py --list    # print every flagged literal
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
PAPER = os.path.join(ROOT, "paper")
ALLOW = os.path.join(PAPER, "NUMBER_ALLOWLIST.txt")

SCAN_FILES = sorted(
    [os.path.join("paper", "sections", f) for f in os.listdir(os.path.join(PAPER, "sections")) if f.endswith(".tex")]
    + ["paper/main.tex"])

# things removed before scanning (they legitimately contain digits)
STRIP_PATTERNS = [
    r"(?<!\\)%.*$",                                   # comments
    r"\\(label|ref|cref|Cref|eqref|pageref|nameref|autoref)\*?\{[^}]*\}",
    r"\\cite[a-z]*\*?(\[[^\]]*\])*\{[^}]*\}",
    r"\\includegraphics(\[[^\]]*\])?\{[^}]*\}",
    r"\\begin\{tabular\}\{[^}]*\}",
    r"\\multicolumn\{\d+\}\{[^}]*\}", r"\\multirow\{\d+\}\{[^}]*\}", r"\\rotatebox\{\d+\}",
    r"\\(vskip|hskip|vspace\*?|hspace\*?|setlength|addtolength)\s*\{?[^\n{}]*\}?",
    r"\\\\\[[^\]]*\]", r"\\item\[[^\]]*\]", r"\\(input|include|usepackage|documentclass|bibliographystyle)(\[[^\]]*\])?\{[^}]*\}",
    r"\\(newcommand|renewcommand|def|DeclareMathOperator\*?)\{?\\[A-Za-z]+\}?(\[\d+\])?",
    r"\\n[A-Z][A-Za-z]*",                             # the number macros themselves
    r"\\[a-zA-Z]+(\d)",                                # \tfrac12 style
    r"\\(t|d)?frac\{[^}]*\}\{[^}]*\}",                 # explicit fractions
    r"\^\{?-?\d+(/\d+)?\}?", r"_\{?\d+\}?",             # exponents / subscripts
    r"\\(sqrt|log|exp|sigma|Gamma|Lambda|alpha|delta|eta|pi|mu|sup|inf|min|max|argmin|argmax)\b",
    r"\\texttt\{[^}]*\}", r"\\url\{[^}]*\}", r"\\href\{[^}]*\}",
    r"\\icml[a-z]*\{[^}]*\}",
    r"\\(begin|end)\{[^}]*\}",
]

LITERAL_RE = re.compile(
    r"(?<![A-Za-z\\{_^])("
    r"\d+\.\d+"                                        # decimals
    r"|\d{1,3}(?:\{,\}\d{3})+"                         # 6{,}054 style thousands
    r"|\d+(?:\.\d+)?\s*\\%"                            # percentages
    r"|\d+(?:\.\d+)?\s*\\times(?!\s*10)"               # 14\times factors
    r"|\d+(?:\.\d+)?\s*\\times\s*10"                    # scientific
    r"|\b\d+\s+(?:seeds?|cells?|configurations?|splits?|units?|bins?|members?|epochs?|folds?|directions?|decision makers?|instances?|seconds?|steps?|orderings?|notes?|records?|subjects?|studies|patients?)\b"
    r"|\b[1-9]\d{2,}\b"                                # bare integers >= 100
    r")")


def strip(tex: str) -> str:
    out = []
    for ln in tex.splitlines():
        for pat in STRIP_PATTERNS:
            ln = re.sub(pat, " ", ln)
        out.append(ln)
    return "\n".join(out)


def load_allowlist():
    allow = set()
    if os.path.exists(ALLOW):
        for ln in open(ALLOW):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) < 3:
                raise SystemExit(f"NUMBER_ALLOWLIST.txt: each line is 'file | token | reason': {ln!r}")
            allow.add((parts[0], parts[1]))
    return allow


def scan(list_all=False):
    allow = load_allowlist()
    flagged = []
    for rel in SCAN_FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            continue
        for i, ln in enumerate(strip(open(path).read()).splitlines(), 1):
            for m in LITERAL_RE.finditer(ln):
                tok = re.sub(r"\s+", " ", m.group(1)).strip()
                ok = (rel, tok) in allow or ("*", tok) in allow
                if not ok or list_all:
                    flagged.append((rel, i, tok, ok, ln.strip()[:90]))
    return flagged


def main() -> int:
    list_all = "--list" in sys.argv
    failures = 0
    # 1. stale macros
    import make_numbers
    tex, prov, warnings = make_numbers.build()
    cur = open(make_numbers.OUT_TEX).read() if os.path.exists(make_numbers.OUT_TEX) else ""
    if cur != tex:
        print("FAIL: paper/numbers.tex is stale -- run scripts/make_numbers.py"); failures += 1
    else:
        print(f"ok: numbers.tex current ({len(prov)} macros, {len(warnings)} registry warnings)")
    # 2. literals
    flagged = scan(list_all)
    bad = [f for f in flagged if not f[3]]
    for rel, i, tok, ok, ctx in (flagged if list_all else bad):
        print(f"{'allowed' if ok else 'LITERAL'}: {rel}:{i}: {tok!r}   {ctx}")
    if bad:
        print(f"FAIL: {len(bad)} hand-typed numeric literal(s); add a macro or justify in paper/NUMBER_ALLOWLIST.txt"); failures += 1
    else:
        print("ok: no hand-typed numeric literals outside the allowlist")
    # 3. provenance
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "check_provenance.py")], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        failures += 1
    # 4. seeds behind CI macros used in the paper
    used = set()
    for rel in SCAN_FILES:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            used |= set(re.findall(r"\\(n[A-Z][A-Za-z]*)", open(p).read()))
    prov_json = json.load(open(make_numbers.OUT_JSON)) if os.path.exists(make_numbers.OUT_JSON) else {}
    few = sorted(m for m in used if m in prov_json and prov_json[m].get("n_seeds") not in (None,) and prov_json[m]["n_seeds"] < 5)
    if few:
        print("FAIL: macros with fewer than 5 seeds cited in the paper:", ", ".join(few)); failures += 1
    else:
        print("ok: every CI-bearing macro cited in the paper has >= 5 seeds")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
