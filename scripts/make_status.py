#!/usr/bin/env python3
"""Generate STATUS.md from docs/claims.json (CLAUDE.md rule 7).

docs/claims.json is a list of objects with the fixed schema:
  claim_id, statement, type (thm|prop|exp), evidence (file:line or results key),
  data_source (real|simulator|semi-synthetic|synthetic|n/a), seeds (int or "n/a"),
  status (proved|verified|refuted|open), and optional "before" / "after" for the
  changed-claims list and "note".

Evidence keys of the form results/<file>.json:<dotted.path> are resolved and the
value is appended to the evidence column so the table is self-checking; a key
that does not resolve is flagged as MISSING and the script exits 1.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(ev: str):
    m = re.match(r"results/([\w\-.]+\.json):(.+)$", ev)
    if not m:
        return None
    path = os.path.join(ROOT, "results", m.group(1))
    if not os.path.exists(path):
        return "MISSING-FILE"
    d = json.load(open(path))
    for part in m.group(2).split("."):
        if isinstance(d, dict) and part in d:
            d = d[part]
        else:
            return "MISSING-KEY"
    if isinstance(d, float):
        return f"{d:.4g}"
    if isinstance(d, dict) and "mean" in d:
        return f"{d['mean']:.4g}±{d.get('ci95', float('nan')):.2g} (n={d.get('n','?')})"
    return str(d)


def main() -> int:
    claims = json.load(open(os.path.join(ROOT, "docs", "claims.json")))
    rows, bad, changed = [], 0, []
    for c in claims:
        ev = c["evidence"]
        val = resolve(ev)
        if val in ("MISSING-FILE", "MISSING-KEY"):
            bad += 1
            ev = f"{ev} **{val}**"
        elif val is not None:
            ev = f"`{ev}` = {val}"
        else:
            ev = f"`{ev}`"
        rows.append(f"| {c['claim_id']} | {c['statement']} | {c['type']} | {ev} | "
                    f"{c['data_source']} | {c['seeds']} | **{c['status']}** |")
        if c.get("before"):
            changed.append(f"- **{c['claim_id']}** — before: {c['before']}  →  after: {c['after']}")
    blocked = open(os.path.join(ROOT, "docs", "status_blocked.md")).read() \
        if os.path.exists(os.path.join(ROOT, "docs", "status_blocked.md")) else ""
    out = ["# STATUS", "",
           "Claim ledger (CLAUDE.md rule 7). Evidence keys of the form "
           "`results/<file>.json:<path>` are resolved at generation time, so the value "
           "shown is the one in the repository.", "",
           "| claim_id | statement | type | evidence | data_source | seeds | status |",
           "|---|---|---|---|---|---|---|", *rows, "",
           "## Changed claims (before → after)", *(changed or ["(none)"]), "",
           blocked]
    open(os.path.join(ROOT, "STATUS.md"), "w").write("\n".join(out))
    print(f"STATUS.md: {len(rows)} claims, {len(changed)} changed, {bad} unresolved evidence keys")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
