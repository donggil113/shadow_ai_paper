#!/usr/bin/env bash
# Build the paper and enforce the submission constraints:
#   main text <= 9 pages, 0 undefined references, 0 overfull boxes.
set -u
cd "$(dirname "$0")/../paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1
rc=$?
pages=$(python3 -c "import re;s=open('main.log',errors='ignore').read();m=re.findall(r'Output written on main.pdf \((\d+) pages',s);print(m[-1] if m else 0)")
mainpages=$(python3 -c "import re;s=open('main.aux',errors='ignore').read();m=re.search(r'\\\\newlabel\{lastmainpage\}\{\{[^}]*\}\{(\d+)\}',s);print(m.group(1) if m else 'n/a')")
undef=$(grep -c "Reference .* undefined\|Citation .* undefined" main.log)
overfull=$(grep -c "^Overfull" main.log)
echo "latexmk exit=$rc  total pages=$pages  main-text pages=$mainpages  undefined refs=$undef  overfull boxes=$overfull"
fail=0
[ "$rc" -ne 0 ] && { echo "FAIL: compile error"; grep -m3 "^!" main.log; fail=1; }
[ "$undef" -ne 0 ] && { echo "FAIL: undefined references"; grep "undefined" main.log | head -5; fail=1; }
[ "$overfull" -ne 0 ] && { echo "FAIL: overfull boxes"; grep -A1 "^Overfull" main.log | head -12; fail=1; }
if [ "$mainpages" != "n/a" ] && [ "$mainpages" -gt 9 ]; then echo "FAIL: main text is $mainpages pages (> 9)"; fail=1; fi
exit $fail
