"""Regex extraction of echo findings from MIMIC-IV discharge summaries.

Route (b) of ``docs/mimic_ecg_echo_spec.md``: MIMIC-IV-ECHO v0.1 ships images
only, so the structural-heart-disease (SHD) outcome has to be recovered from
the free-text discharge summary of the admission linked to the echo.  This
module extracts, per note,

* **LVEF** in percent (``"EF 35%"``, ``"LVEF of 40 percent"``, ``"ejection
  fraction ... 25%"``, ranges ``"55-60%"`` -> midpoint, comparators ``">55%"``)
  and qualitative statements (``"normal LVEF"``, ``"severely depressed EF"``);
* **wall thickness** of the interventricular septum / posterior wall in cm
  (mm is converted) and qualitative LVH statements;
* **valve lesions** with a severity for AS, AR, MS, MR, TR, PR, including
  ``"3+ MR"`` grades, post-positioned severities (``"AS is severe"``) and
  negations (``"no evidence of moderate or severe AS"``).

Composite rule (task definition, see :func:`shd_composite`)::

    SHD = LVEF <= 45  OR  max(IVSd, LVPWd) >= 1.3 cm  OR  any moderate/severe
          lesion among {AS, AR, MS, MR, TR, PR}

Deviations from the EchoNext composite (Poterucha et al., *Nature* 2025;
PhysioNet ``echonext`` v1.1.1, 11 component flags) -- these must be stated in
the paper:

1. EchoNext's components ``rv_systolic_dysfunction_moderate_severe``,
   ``pericardial_effusion_moderate_large``, ``pasp_gte_45`` and
   ``tr_max_gte_32`` are **not** extracted here, so the note-based composite is
   a *subset* of EchoNext SHD (lower prevalence, biased towards LV disease).
2. Mitral stenosis (MS) is **not** an EchoNext component but is included in
   the task rule; it is extracted and counted (``COUNT_MITRAL_STENOSIS``).
3. EchoNext uses structured report grades.  Here a "mild-to-moderate" lesion
   is treated as *below* threshold and "moderate-to-severe" as *above*.
4. EchoNext uses the measured wall thickness.  Here a qualitative
   "moderate"/"severe" LVH statement (not attributed to ECG voltage criteria)
   also satisfies the wall-thickness component when no measurement is quoted
   (``COUNT_QUALITATIVE_LVH``); "mild"/unspecified LVH does not.
5. Qualitative LVEF is mapped as: normal / preserved / hyperdynamic /
   low-normal -> not low; severely or moderately reduced -> low; unqualified
   "reduced" / "depressed" -> low (with a confidence penalty); "mildly
   reduced" -> undetermined.  A numeric value always overrides.
6. A note that documents *some* echo finding but no positive component gets
   composite 0 ("not documented" == absent); a note without any extractable
   field gets ``None`` (missing).  This is the standard note-phenotyping
   convention and is one reason the manual review
   (``scripts/note_extraction_manual_review.py``) is mandatory before the
   note label is used in the paper.

When several LVEF values are quoted the primary value is the *last*
non-historical mention in echo context (then the last non-historical mention,
then the last mention).  Mentions preceded by "prior", "previous", "baseline",
"history of" or a year are marked historical.

The returned ``confidence`` is a heuristic ordering in [0, 1] (not a
calibrated probability) used to stratify the manual-review sample.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

__all__ = [
    "LVEF_LOW_THRESHOLD",
    "WALL_THICK_THRESHOLD_CM",
    "COUNT_MITRAL_STENOSIS",
    "COUNT_QUALITATIVE_LVH",
    "VALVE_CODES",
    "SEVERITY_ORDER",
    "extract_lvef",
    "extract_wall_thickness",
    "extract_valve_lesions",
    "shd_composite",
    "extract_note",
    "empty_extraction",
    "EXTRACTION_KEYS",
]

LVEF_LOW_THRESHOLD = 45.0
WALL_THICK_THRESHOLD_CM = 1.3
COUNT_MITRAL_STENOSIS = True      # deviation 2 (task rule includes MS)
COUNT_QUALITATIVE_LVH = True      # deviation 4

VALVE_CODES = ["AS", "AR", "MS", "MR", "TR", "PR"]

# canonical severity -> rank; >= moderate counts for the composite
SEVERITY_ORDER = {
    "none": 0, "trace": 1, "mild": 2, "mild_moderate": 2.5,
    "moderate": 3, "moderate_severe": 3.5, "severe": 4,
}
_GE_MODERATE = {"moderate", "moderate_severe", "severe"}

EXTRACTION_KEYS = [
    "lvef", "lvef_comparator", "lvef_qualitative", "lvef_all", "lvef_evidence",
    "wall_thickness_cm", "septal_cm", "posterior_cm", "lvh_qualitative",
    "lvh_negated", "wall_evidence", "valve_lesions", "valve_evidence",
    "components", "shd_composite", "confidence", "n_fields_found",
    "echo_context", "extractor",
]

# ----------------------------------------------------------------------------
# shared fragments
# ----------------------------------------------------------------------------
_NUM = r"\d{1,2}(?:\.\d+)?"
_RANGE_SEP = r"\s*(?:-|–|—|to|/)\s*"
_PCT = r"\s*(?:%|percent|per\s+cent)?"
_CMP = r"(?P<cmp>>=|<=|≥|≤|>|<|greater\s+than|less\s+than|at\s+least|up\s+to|above|below)?\s*"

_ECHO_CONTEXT_RE = re.compile(
    r"\b(?:echo(?:cardiogram|cardiography|cardiographic)?|TTE|TEE|transthoracic|"
    r"transesophageal|transoesophageal|LVEF|ejection\s+fraction|valve|valvular|"
    r"regurgitation|stenosis|IVSd|LVPWd)\b",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"\b(?:prior|previous(?:ly)?|old|baseline|remote|historical|history\s+of|h/o|"
    r"in\s+(?:19|20)\d\d|(?:19|20)\d\d)\b[^.\n]{0,40}$",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|without|w/o|no\s+evidence\s+(?:of|for)|absence\s+of|absent|"
    r"negative\s+for|rule\s+out|r/o|denies|denied|free\s+of|resolved|resolution\s+of)\b",
    re.IGNORECASE,
)
_NEGATION_BREAK_RE = re.compile(r"[,;:.\n]|\b(?:but|however|although|with|and)\b",
                                re.IGNORECASE)


def _snippet(text: str, start: int, end: int, pad: int = 60) -> str:
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    return re.sub(r"\s+", " ", text[s:e]).strip()


def _in_echo_context(text: str, pos: int, back: int = 200, fwd: int = 60) -> bool:
    return bool(_ECHO_CONTEXT_RE.search(text[max(0, pos - back): pos + fwd]))


def _is_historical(text: str, pos: int) -> bool:
    return bool(_HISTORICAL_RE.search(text[max(0, pos - 60): pos]))


def _is_negated(text: str, pos: int, back: int = 45) -> bool:
    """True if a negation cue precedes ``pos`` inside the same clause."""
    window = text[max(0, pos - back): pos]
    cue = None
    for m in _NEGATION_RE.finditer(window):
        cue = m
    if cue is None:
        return False
    # a clause breaker between the cue and the finding cancels the negation
    between = window[cue.end():]
    return not _NEGATION_BREAK_RE.search(between)


# ----------------------------------------------------------------------------
# LVEF
# ----------------------------------------------------------------------------
_EF_CUE = (r"(?<![A-Za-z])(?P<cue>LV\s*EF|LVEF|EF|"
           r"(?:left\s+ventricular\s+|LV\s+)?ejection\s+fraction)(?![A-Za-z])")
_EF_NUMERIC_RE = re.compile(
    _EF_CUE
    + r"(?P<gap>[^.\n\d%]{0,45}?)"
    + _CMP
    + r"(?P<n1>" + _NUM + r")(?!\d)"
    + r"(?:" + _RANGE_SEP + r"(?P<n2>" + _NUM + r")(?!\d))?"
    + _PCT
    + r"(?!\s*(?:cm|mm|mmhg|bpm|ms|msec|m/s|kg|mg|/min|min|hours?|hrs?|days?|weeks?|months?|years?|yo|y/o)\b)",
    re.IGNORECASE,
)
_QUAL_WORDS = (r"(?P<qual>hyperdynamic|low[- ]normal|normal|preserved|"
               r"(?:severely|moderately|mildly|markedly|profoundly|globally)\s+"
               r"(?:reduced|depressed|decreased|diminished|impaired|dysfunction)|"
               r"reduced|depressed|decreased|diminished|impaired)")
_EF_QUAL_PRE_RE = re.compile(_QUAL_WORDS + r"\s+(?:LV\s+|global\s+|systolic\s+)?" + _EF_CUE,
                             re.IGNORECASE)
_EF_QUAL_POST_RE = re.compile(
    _EF_CUE + r"\s*(?:is|was|appears?|remains?|noted\s+to\s+be|:|,|-)?\s*(?:grossly\s+|overall\s+)?"
    + _QUAL_WORDS + r"(?![A-Za-z])",
    re.IGNORECASE,
)
_GAP_BAD_RE = re.compile(r"\b(?:cm|mm|IVS|LVPW|septum|wall|LA|RA|RV|BNP|troponin|HR|BP|creatinine)\b",
                         re.IGNORECASE)


def _canon_qual(q: str) -> str:
    q = re.sub(r"\s+", " ", q.lower())
    if q in ("normal", "preserved", "low normal", "low-normal"):
        return "normal"
    if q == "hyperdynamic":
        return "hyperdynamic"
    if q.startswith(("severely", "markedly", "profoundly")):
        return "severely_reduced"
    if q.startswith("moderately"):
        return "moderately_reduced"
    if q.startswith("mildly"):
        return "mildly_reduced"
    return "reduced"


def _canon_cmp(c: Optional[str]) -> Optional[str]:
    if not c:
        return None
    c = re.sub(r"\s+", " ", c.lower())
    return {"≥": ">=", "≤": "<=", "greater than": ">", "less than": "<",
            "at least": ">=", "up to": "<=", "above": ">", "below": "<"}.get(c, c)


def extract_lvef(text: str) -> List[Dict[str, Any]]:
    """All LVEF mentions in ``text`` (numeric and qualitative), in order."""
    out: List[Dict[str, Any]] = []
    for m in _EF_NUMERIC_RE.finditer(text):
        gap = m.group("gap") or ""
        if _GAP_BAD_RE.search(gap):
            continue
        v1 = float(m.group("n1"))
        v2 = float(m.group("n2")) if m.group("n2") else None
        # fraction notation 0.35 -> 35
        if v1 < 1.0:
            v1 *= 100.0
        if v2 is not None and v2 < 1.0:
            v2 *= 100.0
        if v2 is not None and v2 < v1:
            v1, v2 = v2, v1
        value = v1 if v2 is None else (v1 + v2) / 2.0
        if not (5.0 <= value <= 90.0):
            continue
        has_pct = bool(re.search(r"%|percent", m.group(0), re.IGNORECASE))
        out.append({
            "kind": "numeric", "value": float(value),
            "range": (v1, v2) if v2 is not None else None,
            "comparator": _canon_cmp(m.group("cmp")), "has_percent": has_pct,
            "qualitative": None, "pos": m.start(),
            "historical": _is_historical(text, m.start()),
            "echo_context": _in_echo_context(text, m.start()),
            "evidence": _snippet(text, m.start(), m.end()),
        })
    numeric_pos = [o["pos"] for o in out]
    for rx in (_EF_QUAL_PRE_RE, _EF_QUAL_POST_RE):
        for m in rx.finditer(text):
            # a qualitative mention next to a numeric one is redundant
            # ("EF is depressed at 25%" -> the number wins)
            if any(abs(p - m.start()) < 60 for p in numeric_pos):
                continue
            out.append({
                "kind": "qualitative", "value": None, "range": None,
                "comparator": None, "has_percent": False,
                "qualitative": _canon_qual(m.group("qual")), "pos": m.start(),
                "historical": _is_historical(text, m.start()),
                "echo_context": _in_echo_context(text, m.start()),
                "negated": _is_negated(text, m.start()),
                "evidence": _snippet(text, m.start(), m.end()),
            })
    out.sort(key=lambda o: o["pos"])
    return out


def _pick_primary(findings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not findings:
        return None
    for pred in (lambda f: f["echo_context"] and not f["historical"],
                 lambda f: not f["historical"],
                 lambda f: True):
        cands = [f for f in findings if pred(f)]
        if cands:
            return cands[-1]
    return None


# ----------------------------------------------------------------------------
# wall thickness / LVH
# ----------------------------------------------------------------------------
_SITE = (r"(?P<site>IVSd|IVS|inter-?ventricular\s+sept(?:um|al)(?:\s+(?:wall\s+)?thickness)?|"
         r"septal(?:\s+wall)?(?:\s+thickness)?|septum(?:\s+thickness)?|LVPWd|LVPW|PWd|PW|"
         r"(?:LV\s+|left\s+ventricular\s+)?posterior\s+wall(?:\s+thickness)?|"
         r"(?:LV\s+|left\s+ventricular\s+)?wall\s+thickness)")
_WALL_RE = re.compile(
    r"(?<![A-Za-z])" + _SITE + r"(?![A-Za-z])"
    + r"(?P<gap>[^.\n\d]{0,30}?)"
    + r"(?P<n1>\d{1,2}(?:\.\d+)?)(?!\d)"
    + r"(?:" + _RANGE_SEP + r"(?P<n2>\d{1,2}(?:\.\d+)?)(?!\d))?"
    + r"\s*(?P<unit>cm|mm|centimet(?:er|re)s?|millimet(?:er|re)s?)?(?![A-Za-z])",
    re.IGNORECASE,
)
_LVH_RE = re.compile(
    r"(?P<sev>mild|moderate|severe|marked|mild[- ]to[- ]moderate|moderate[- ]to[- ]severe)?\s*"
    r"(?:concentric|asymmetric|eccentric|septal|symmetric)?\s*"
    r"(?P<lvh>LVH|left\s+ventricular\s+hypertrophy|LV\s+hypertrophy)(?![A-Za-z])",
    re.IGNORECASE,
)
_VOLTAGE_RE = re.compile(r"\b(?:voltage|ECG|EKG|electrocardiogram|criteria)\b", re.IGNORECASE)
_UNIT_REQUIRED_SITES = {"pw", "pwd", "septum"}


def _site_kind(site: str) -> str:
    s = site.lower()
    if "post" in s or s.startswith("lvpw") or s in ("pw", "pwd"):
        return "posterior"
    if "sept" in s or s.startswith("ivs"):
        return "septal"
    return "wall"


def extract_wall_thickness(text: str) -> Dict[str, Any]:
    """Numeric septal / posterior wall thickness (cm) and qualitative LVH."""
    meas: List[Dict[str, Any]] = []
    for m in _WALL_RE.finditer(text):
        site = m.group("site")
        unit = (m.group("unit") or "").lower()
        v1 = float(m.group("n1"))
        v2 = float(m.group("n2")) if m.group("n2") else None
        raw = max(v1, v2) if v2 is not None else v1
        if unit.startswith("mm") or unit.startswith("milli"):
            val = raw / 10.0
        elif unit.startswith("cm") or unit.startswith("centi"):
            val = raw
        else:
            # no unit: accept a decimal cm-looking number for the unambiguous
            # abbreviations only
            if site.lower() in _UNIT_REQUIRED_SITES or "." not in m.group("n1"):
                continue
            val = raw
        if not (0.4 <= val <= 3.5):
            continue
        meas.append({"site": _site_kind(site), "value_cm": round(val, 2),
                     "pos": m.start(), "evidence": _snippet(text, m.start(), m.end())})
    septal = [x["value_cm"] for x in meas if x["site"] == "septal"]
    posterior = [x["value_cm"] for x in meas if x["site"] == "posterior"]
    generic = [x["value_cm"] for x in meas if x["site"] == "wall"]
    all_vals = septal + posterior + generic

    lvh_q: Optional[str] = None
    lvh_neg: Optional[bool] = None
    lvh_ev: Optional[str] = None
    for m in _LVH_RE.finditer(text):
        ctx = text[max(0, m.start() - 60): m.end() + 40]
        if _VOLTAGE_RE.search(ctx):
            continue  # ECG voltage criteria are not echo wall thickness
        sev = (m.group("sev") or "").lower().replace("-", " ")
        if sev in ("severe", "marked", "moderate to severe"):
            q = "severe" if sev != "moderate to severe" else "moderate"
        elif sev == "moderate":
            q = "moderate"
        elif sev in ("mild", "mild to moderate"):
            q = "mild"
        else:
            q = "unspecified"
        neg = _is_negated(text, m.start())
        # keep the most severe non-negated statement; else the negated one
        rank = {"unspecified": 0.5, "mild": 1, "moderate": 2, "severe": 3}
        if lvh_q is None or (not neg and (lvh_neg or rank[q] > rank[lvh_q])):
            lvh_q, lvh_neg, lvh_ev = q, neg, _snippet(text, m.start(), m.end())

    ev = None
    if meas:
        best = max(meas, key=lambda x: x["value_cm"])
        ev = best["evidence"]
    return {
        "wall_thickness_cm": max(all_vals) if all_vals else None,
        "septal_cm": max(septal) if septal else None,
        "posterior_cm": max(posterior) if posterior else None,
        "measurements": meas,
        "lvh_qualitative": lvh_q, "lvh_negated": lvh_neg,
        "wall_evidence": ev if ev is not None else lvh_ev,
    }


# ----------------------------------------------------------------------------
# valve lesions
# ----------------------------------------------------------------------------
_SEV = (r"(?P<sev>mild[- ]to[- ]moderate|mild[-/]mod(?:erate)?|"
        r"moderate[- ]to[- ]severe|mod(?:erate)?[-/]severe|moderate[- ]severe|"
        r"moderately[- ]severe|"
        r"severe|critical|moderate|mild|trivial|trace|minimal|physiologic(?:al)?|"
        r"significant|none|absent)")
_ADJ = (r"(?:(?:calcific|calcified|functional|eccentric|central|ischemic|ischaemic|"
        r"degenerative|secondary|primary|rheumatic|bicuspid|chronic|acute|valvular|"
        r"native|paravalvular|residual|new|persistent|worsening|stable|"
        r"low[- ]flow|low[- ]gradient|paradoxical|posteriorly[- ]directed|"
        r"anteriorly[- ]directed|\(\s*[1-4]\+\s*\))\s+){0,3}")
# long forms are case-insensitive; abbreviations must be upper-case
_LESION_LONG = {
    "AS": r"aortic(?:\s+valve)?\s+stenosis|aortic\s+valvular\s+stenosis",
    "AR": r"aortic(?:\s+valve)?\s+(?:regurgitation|insufficiency|regurg|incompetence)",
    "MS": r"mitral(?:\s+valve)?\s+stenosis",
    "MR": r"mitral(?:\s+valve)?\s+(?:regurgitation|insufficiency|regurg|incompetence)",
    "TR": r"tricuspid(?:\s+valve)?\s+(?:regurgitation|insufficiency|regurg)",
    "PR": r"pulmon(?:ic|ary)(?:\s+valve)?\s+(?:regurgitation|insufficiency|regurg)",
}
_LESION_ABBR = {"AS": r"AS|AoS", "AR": r"AR|AI", "MS": r"MS", "MR": r"MR",
                "TR": r"TR", "PR": r"PR|PI"}
_ABBR_NEEDS_CONTEXT = {"MS", "PR"}   # multiple sclerosis / PR interval
# "no MR", "without significant AS", "no evidence of moderate or severe TR"
_NEG_PREFIX = (r"(?P<sev>no|without|w/o|absence\s+of|no\s+evidence\s+(?:of|for)|negative\s+for|"
               r"free\s+of)\s+(?:significant\s+|any\s+|evidence\s+of\s+|hemodynamically\s+significant\s+)?"
               r"(?:(?:mild|moderate|severe)(?:\s+(?:or|to|-)\s+(?:mild|moderate|severe))?\s+)?")


def _lesion_regexes():
    rx = []
    for code, longf in _LESION_LONG.items():
        rx.append((code, "long", re.compile(
            _SEV + r"\s+" + _ADJ + r"(?P<lesion>" + longf + r")(?![A-Za-z])", re.IGNORECASE)))
        rx.append((code, "long", re.compile(
            r"(?<![A-Za-z])(?P<lesion>" + longf + r")\s*(?:is|was|were|appears?|appeared|"
            r"noted\s+to\s+be|remains?|graded\s+as|,|:|\(|-)?\s*(?:at\s+least\s+)?"
            + _SEV + r"(?![A-Za-z])", re.IGNORECASE)))
        rx.append((code, "long", re.compile(
            r"(?P<grade>[1-4])\+\s*" + _ADJ + r"(?P<lesion>" + longf + r")(?![A-Za-z])",
            re.IGNORECASE)))
        rx.append((code, "long", re.compile(
            r"(?<![A-Za-z])(?P<lesion>" + longf + r")\s*\(?\s*(?P<grade>[1-4])\+\s*\)?",
            re.IGNORECASE)))
        rx.append((code, "long", re.compile(
            _NEG_PREFIX + _ADJ + r"(?P<lesion>" + longf + r")(?![A-Za-z])", re.IGNORECASE)))
    for code, abbr in _LESION_ABBR.items():
        sev_ci = _SEV
        rx.append((code, "abbr", re.compile(
            r"(?i:" + _NEG_PREFIX + _ADJ + r")(?<![A-Za-z])(?P<lesion>" + abbr + r")(?![A-Za-z])")))
        rx.append((code, "abbr", re.compile(
            r"(?i:" + sev_ci + r")\s+(?i:" + _ADJ + r")(?<![A-Za-z])(?P<lesion>" + abbr + r")(?![A-Za-z])")))
        rx.append((code, "abbr", re.compile(
            r"(?<![A-Za-z])(?P<lesion>" + abbr + r")(?![A-Za-z])\s*(?i:(?:is|was|appears?|noted\s+to\s+be|"
            r"remains?|graded\s+as|,|:|\(|-)?\s*(?:at\s+least\s+)?" + sev_ci + r")(?![A-Za-z])")))
        rx.append((code, "abbr", re.compile(
            r"(?P<grade>[1-4])\+\s*(?i:" + _ADJ + r")(?<![A-Za-z])(?P<lesion>" + abbr + r")(?![A-Za-z])")))
        rx.append((code, "abbr", re.compile(
            r"(?<![A-Za-z])(?P<lesion>" + abbr + r")(?![A-Za-z])\s*\(?\s*(?P<grade>[1-4])\+\s*\)?")))
    return rx


_LESION_RES = _lesion_regexes()
_VALVE_CONTEXT_RE = re.compile(
    r"\b(?:echo\w*|TTE|TEE|valve|valvular|regurg\w*|stenosis|LVEF|EF|aortic|mitral|"
    r"tricuspid|pulmonic|IVSd|LVPWd|gradient|leaflets?)\b", re.IGNORECASE)


def _canon_sev(sev: Optional[str], grade: Optional[str]) -> str:
    if grade:
        return {"1": "mild", "2": "moderate", "3": "moderate_severe", "4": "severe"}[grade]
    s = re.sub(r"[-/]", " ", (sev or "").lower()).strip()
    s = re.sub(r"\s+", " ", s)
    if s in ("no", "without", "w/o", "absence of", "no evidence of", "no evidence for",
             "negative for", "free of"):
        return "none"
    if s in ("severe", "critical"):
        return "severe"
    if s in ("moderate to severe", "mod severe", "moderate severe", "moderately severe"):
        return "moderate_severe"
    if s in ("moderate", "significant"):
        return "moderate"
    if s in ("mild to moderate", "mild mod", "mild moderate"):
        return "mild_moderate"
    if s == "mild":
        return "mild"
    if s in ("trivial", "trace", "minimal", "physiologic", "physiological"):
        return "trace"
    return "none"


def _looks_allcaps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.8


def extract_valve_lesions(text: str) -> List[Dict[str, Any]]:
    """Valve lesion statements with canonical severity and negation flag."""
    found: List[Dict[str, Any]] = []
    taken: List[tuple] = []
    # context-free forms first so that they can vouch for ambiguous abbreviations
    ordered = sorted(_LESION_RES, key=lambda t: t[1] == "abbr" and t[0] in _ABBR_NEEDS_CONTEXT)
    for code, form, rx in ordered:
        for m in rx.finditer(text):
            span = (m.start(), m.end())
            if any(a < span[1] and span[0] < b for a, b in taken):
                continue
            if form == "abbr":
                ctx = text[max(0, m.start() - 250): m.end() + 100]
                has_ctx = bool(_VALVE_CONTEXT_RE.search(ctx)) or any(
                    abs(f["pos"] - m.start()) < 300 for f in found)
                if code in _ABBR_NEEDS_CONTEXT and not has_ctx:
                    continue
                # in ALL-CAPS prose "AS"/"AR"/"MR" are ordinary words: demand context
                if not has_ctx and _looks_allcaps(text[max(0, m.start() - 40): m.end() + 40]):
                    continue
            gd = m.groupdict()
            sev = _canon_sev(gd.get("sev"), gd.get("grade"))
            neg = _is_negated(text, m.start())
            if sev == "none":
                neg = True
            taken.append(span)
            found.append({
                "valve": code, "severity": sev,
                "ge_moderate": sev in _GE_MODERATE, "negated": bool(neg),
                "historical": _is_historical(text, m.start()),
                "pos": m.start(), "evidence": _snippet(text, m.start(), m.end()),
            })
    found.sort(key=lambda o: o["pos"])
    return found


# ----------------------------------------------------------------------------
# composite
# ----------------------------------------------------------------------------
def _lvef_component(primary: Optional[Dict[str, Any]]) -> Optional[bool]:
    if primary is None:
        return None
    if primary["kind"] == "numeric":
        v, c = primary["value"], primary["comparator"]
        if c in (">", ">="):
            return None if v < LVEF_LOW_THRESHOLD else False
        if c in ("<", "<="):
            return True if v <= LVEF_LOW_THRESHOLD else None
        return bool(v <= LVEF_LOW_THRESHOLD)
    q = primary["qualitative"]
    if primary.get("negated"):
        return None
    if q in ("normal", "hyperdynamic"):
        return False
    if q in ("severely_reduced", "moderately_reduced", "reduced"):
        return True
    return None  # mildly_reduced: undetermined


def shd_composite(lvef_component: Optional[bool], wall_component: Optional[bool],
                  valve_component: Optional[bool]) -> Optional[int]:
    """1 if any component is True; 0 if none True and at least one documented
    False; None if nothing documented."""
    comps = [lvef_component, wall_component, valve_component]
    if any(c is True for c in comps):
        return 1
    if any(c is False for c in comps):
        return 0
    return None


def empty_extraction(extractor: str = "none") -> Dict[str, Any]:
    d: Dict[str, Any] = {k: None for k in EXTRACTION_KEYS}
    d.update({"lvef_all": [], "valve_lesions": [], "valve_evidence": [],
              "components": {"lvef_le_45": None, "wall_ge_1p3": None,
                             "valve_moderate_or_severe": None},
              "confidence": 0.0, "n_fields_found": 0, "echo_context": False,
              "extractor": extractor})
    return d


def extract_note(text: Optional[str]) -> Dict[str, Any]:
    """Full per-note extraction (see module docstring for the schema)."""
    res = empty_extraction("regex")
    if not isinstance(text, str) or not text.strip():
        return res
    # --- LVEF
    ef = extract_lvef(text)
    numeric = [f for f in ef if f["kind"] == "numeric"]
    primary = _pick_primary(numeric) or _pick_primary(ef)
    res["lvef_all"] = [f["value"] for f in numeric]
    if primary is not None:
        res["lvef"] = primary["value"]
        res["lvef_comparator"] = primary["comparator"]
        res["lvef_qualitative"] = primary["qualitative"]
        res["lvef_evidence"] = primary["evidence"]
    lvef_c = _lvef_component(primary)
    # --- wall
    w = extract_wall_thickness(text)
    for k in ("wall_thickness_cm", "septal_cm", "posterior_cm", "lvh_qualitative",
              "lvh_negated", "wall_evidence"):
        res[k] = w[k]
    if w["wall_thickness_cm"] is not None:
        wall_c: Optional[bool] = w["wall_thickness_cm"] >= WALL_THICK_THRESHOLD_CM
    elif w["lvh_qualitative"] is not None:
        if w["lvh_negated"]:
            wall_c = False
        elif COUNT_QUALITATIVE_LVH and w["lvh_qualitative"] in ("moderate", "severe"):
            wall_c = True
        else:
            wall_c = None
    else:
        wall_c = None
    # --- valves
    lesions = extract_valve_lesions(text)
    counted = [x for x in lesions if COUNT_MITRAL_STENOSIS or x["valve"] != "MS"]
    res["valve_lesions"] = [{k: v for k, v in x.items() if k != "pos"} for x in lesions]
    res["valve_evidence"] = [x["evidence"] for x in counted if x["ge_moderate"] and not x["negated"]]
    if any(x["ge_moderate"] and not x["negated"] for x in counted):
        valve_c: Optional[bool] = True
    elif counted:
        valve_c = False
    else:
        valve_c = None
    res["components"] = {"lvef_le_45": lvef_c, "wall_ge_1p3": wall_c,
                         "valve_moderate_or_severe": valve_c}
    res["shd_composite"] = shd_composite(lvef_c, wall_c, valve_c)
    res["echo_context"] = bool(_ECHO_CONTEXT_RE.search(text))
    # --- confidence (heuristic ordering)
    conf = 0.0
    n_fields = 0
    if primary is not None:
        n_fields += 1
        conf += 0.35 if primary["kind"] == "numeric" else 0.2
        if primary["kind"] == "numeric":
            vals = [f["value"] for f in numeric if not f["historical"]] or res["lvef_all"]
            if vals and (min(vals) <= LVEF_LOW_THRESHOLD < max(vals)):
                conf -= 0.2  # conflicting values straddle the threshold
            if primary["comparator"] and lvef_c is None:
                conf -= 0.1
        elif primary["qualitative"] == "reduced":
            conf -= 0.1
    if w["wall_thickness_cm"] is not None:
        n_fields += 1
        conf += 0.2
    elif w["lvh_qualitative"] is not None:
        n_fields += 1
        conf += 0.1
    if counted:
        n_fields += 1
        conf += 0.25
        if any(x["negated"] for x in counted):
            conf -= 0.05
    if res["echo_context"]:
        conf += 0.2
    res["confidence"] = float(min(1.0, max(0.0, conf))) if n_fields else 0.0
    res["n_fields_found"] = n_fields
    return res
