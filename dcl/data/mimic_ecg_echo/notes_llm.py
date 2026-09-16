"""Provider-agnostic LLM hook for note extraction (OFF by default).

The regex extractor (:mod:`.notes_regex`) is the default route (b) labeller.
This module defines the *interface* through which a user may plug in any
local model (llama.cpp, vLLM, HF transformers, an in-house service ...) to
produce the same per-note dictionary, and a :class:`NullExtractor` that is
used when nothing is configured.

**Network policy.** Nothing in this module (or anywhere in
``dcl.data.mimic_ecg_echo``) opens a network connection.  MIMIC notes are
covered by the PhysioNet data-use agreement, which forbids sending them to
third-party services; the hook is therefore *local-only by construction*:
the only way to run a model is to hand this module a Python callable, and it
is the user's responsibility that the callable stays on the machine.

How to plug in a local model
----------------------------
1. Write a function ``generate(prompt: str) -> str`` that runs your model
   and returns its raw text output, e.g.::

       from llama_cpp import Llama                       # any local backend
       llm = Llama(model_path="/models/med-8b.gguf")
       def generate(prompt):
           return llm(prompt, max_tokens=400)["choices"][0]["text"]

2. Wrap it::

       from dcl.data.mimic_ecg_echo.notes_llm import CallableExtractor
       extractor = CallableExtractor(generate, name="med-8b-local")

   or subclass :class:`LLMExtractor` directly and implement ``extract``.

3. Pass it to :func:`dcl.data.mimic_ecg_echo.outcomes.label_from_notes`
   (``extractor=extractor``).  Results are merged with the regex result by
   :func:`merge_extractions` (LLM values override regex values field by
   field; regex fills what the LLM left ``None``).

4. Alternatively, from the command line / experiment config, give a dotted
   path ``"my_module:make_extractor"`` to :func:`get_extractor`; the named
   callable must return an :class:`LLMExtractor`.  ``"null"`` (default)
   returns :class:`NullExtractor`.

The LLM is asked to return a JSON object with the fields of
:data:`LLM_JSON_FIELDS`; :func:`parse_llm_json` converts it into the regex
schema (``notes_regex.EXTRACTION_KEYS``) so that both routes are audited by
the same manual-review script.
"""

from __future__ import annotations

import importlib
import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

from .notes_regex import (EXTRACTION_KEYS, LVEF_LOW_THRESHOLD, VALVE_CODES,
                          WALL_THICK_THRESHOLD_CM, empty_extraction, shd_composite)

__all__ = [
    "LLMExtractor", "NullExtractor", "CallableExtractor", "get_extractor",
    "merge_extractions", "parse_llm_json", "PROMPT_TEMPLATE", "LLM_JSON_FIELDS",
]

LLM_JSON_FIELDS = {
    "lvef": "number or null -- left ventricular ejection fraction in percent (midpoint of a range)",
    "lvef_qualitative": "one of normal, mildly_reduced, moderately_reduced, severely_reduced, null",
    "wall_thickness_cm": "number or null -- largest of septal (IVSd) and posterior (LVPWd) wall thickness in cm",
    "lvh_qualitative": "one of mild, moderate, severe, unspecified, null",
    "valve_lesions": "list of {valve: AS|AR|MS|MR|TR|PR, severity: trace|mild|mild_moderate|moderate|"
                     "moderate_severe|severe|none, negated: bool, evidence: str}",
    "evidence": "object mapping lvef / wall to the verbatim sentence used",
}

PROMPT_TEMPLATE = (
    "You are extracting echocardiography findings from a hospital discharge summary.\n"
    "Return ONLY a JSON object with these fields:\n{fields}\n"
    "Use null when a field is not documented. Do not guess. Prefer the echo performed\n"
    "during this admission over historical values.\n\nNOTE:\n\"\"\"\n{text}\n\"\"\"\nJSON:"
)


class LLMExtractor:
    """Interface: ``extract(text) -> dict`` with the ``notes_regex`` schema."""

    name: str = "abstract"

    def extract(self, text: str) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError

    def extract_many(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        return [self.extract(t) for t in texts]

    @property
    def is_null(self) -> bool:
        return False


class NullExtractor(LLMExtractor):
    """Default: contributes nothing, so the regex route is used unchanged."""

    name = "null"

    def extract(self, text: str) -> Dict[str, Any]:
        return empty_extraction("null")

    @property
    def is_null(self) -> bool:
        return True


class CallableExtractor(LLMExtractor):
    """Wrap a local ``generate(prompt) -> str`` callable.

    The callable receives :data:`PROMPT_TEMPLATE` filled with the note and
    must return the model's text; the first JSON object in that text is
    parsed by :func:`parse_llm_json`.  Parsing failures give an empty
    extraction with ``extractor == "<name>:parse_error"`` instead of raising,
    so a flaky model never silently changes labels: the failure is visible in
    the per-note record.
    """

    def __init__(self, generate: Callable[[str], str], name: str = "local-llm",
                 prompt_template: str = PROMPT_TEMPLATE, max_chars: int = 12000):
        if not callable(generate):
            raise TypeError("generate must be a callable(prompt: str) -> str")
        self._generate = generate
        self.name = name
        self.prompt_template = prompt_template
        self.max_chars = max_chars

    def build_prompt(self, text: str) -> str:
        fields = "\n".join(f"- {k}: {v}" for k, v in LLM_JSON_FIELDS.items())
        return self.prompt_template.format(fields=fields, text=text[: self.max_chars])

    def extract(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return empty_extraction(self.name)
        raw = self._generate(self.build_prompt(text))
        return parse_llm_json(raw, extractor=self.name)


def _first_json_object(raw: str) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, str):
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def parse_llm_json(raw: str, extractor: str = "llm") -> Dict[str, Any]:
    """Convert a model's JSON answer into the ``notes_regex`` schema."""
    obj = _first_json_object(raw)
    if obj is None:
        return empty_extraction(f"{extractor}:parse_error")
    res = empty_extraction(extractor)
    lvef = _num(obj.get("lvef"))
    if lvef is not None and lvef < 1.0:
        lvef *= 100.0
    if lvef is not None and not (5.0 <= lvef <= 90.0):
        lvef = None
    res["lvef"] = lvef
    res["lvef_all"] = [lvef] if lvef is not None else []
    q = obj.get("lvef_qualitative")
    res["lvef_qualitative"] = q if q in ("normal", "mildly_reduced", "moderately_reduced",
                                          "severely_reduced", "reduced", "hyperdynamic") else None
    wall = _num(obj.get("wall_thickness_cm"))
    if wall is not None and wall > 5.0:      # the model answered in mm
        wall /= 10.0
    if wall is not None and not (0.4 <= wall <= 3.5):
        wall = None
    res["wall_thickness_cm"] = wall
    lvh = obj.get("lvh_qualitative")
    res["lvh_qualitative"] = lvh if lvh in ("mild", "moderate", "severe", "unspecified") else None
    res["lvh_negated"] = False if res["lvh_qualitative"] else None
    ev = obj.get("evidence") or {}
    if isinstance(ev, dict):
        res["lvef_evidence"] = ev.get("lvef")
        res["wall_evidence"] = ev.get("wall")
    lesions = []
    for item in obj.get("valve_lesions") or []:
        if not isinstance(item, dict):
            continue
        valve = str(item.get("valve", "")).upper()
        sev = str(item.get("severity", "none")).lower().replace("-", "_").replace(" ", "_")
        if valve not in VALVE_CODES:
            continue
        if sev not in ("trace", "mild", "mild_moderate", "moderate", "moderate_severe",
                       "severe", "none"):
            continue
        neg = bool(item.get("negated", False)) or sev == "none"
        lesions.append({"valve": valve, "severity": sev,
                        "ge_moderate": sev in ("moderate", "moderate_severe", "severe"),
                        "negated": neg, "historical": False,
                        "evidence": item.get("evidence")})
    res["valve_lesions"] = lesions
    res["valve_evidence"] = [x["evidence"] for x in lesions if x["ge_moderate"] and not x["negated"]]
    _finalise(res)
    return res


def _finalise(res: Dict[str, Any]) -> None:
    """Recompute components / composite / n_fields from the field values."""
    if res["lvef"] is not None:
        lvef_c: Optional[bool] = res["lvef"] <= LVEF_LOW_THRESHOLD
    else:
        lvef_c = {"normal": False, "hyperdynamic": False, "severely_reduced": True,
                  "moderately_reduced": True, "reduced": True}.get(res["lvef_qualitative"])
    if res["wall_thickness_cm"] is not None:
        wall_c: Optional[bool] = res["wall_thickness_cm"] >= WALL_THICK_THRESHOLD_CM
    elif res["lvh_qualitative"] in ("moderate", "severe") and not res.get("lvh_negated"):
        wall_c = True
    elif res["lvh_qualitative"] is not None and res.get("lvh_negated"):
        wall_c = False
    else:
        wall_c = None
    lesions = res["valve_lesions"] or []
    if any(x["ge_moderate"] and not x["negated"] for x in lesions):
        valve_c: Optional[bool] = True
    elif lesions:
        valve_c = False
    else:
        valve_c = None
    res["components"] = {"lvef_le_45": lvef_c, "wall_ge_1p3": wall_c,
                         "valve_moderate_or_severe": valve_c}
    res["shd_composite"] = shd_composite(lvef_c, wall_c, valve_c)
    n = int(lvef_c is not None or res["lvef"] is not None) \
        + int(res["wall_thickness_cm"] is not None or res["lvh_qualitative"] is not None) \
        + int(bool(lesions))
    res["n_fields_found"] = n
    if res.get("confidence") is None:
        res["confidence"] = 0.0


def merge_extractions(regex_result: Dict[str, Any], llm_result: Dict[str, Any],
                      policy: str = "llm_overrides") -> Dict[str, Any]:
    """Combine a regex and an LLM extraction into one record.

    ``policy="llm_overrides"`` (default): LLM field values replace regex values
    wherever the LLM produced something; regex fills the rest.
    ``policy="regex_overrides"``: the reverse.  Components and the composite
    are recomputed from the merged fields.  A null LLM result returns the
    regex record unchanged.
    """
    if llm_result is None or llm_result.get("extractor") in (None, "null") \
            or str(llm_result.get("extractor", "")).endswith(":parse_error"):
        out = dict(regex_result)
        if llm_result is not None and str(llm_result.get("extractor", "")).endswith(":parse_error"):
            out["extractor"] = f"{regex_result.get('extractor')}+{llm_result['extractor']}"
        return out
    if policy not in ("llm_overrides", "regex_overrides"):
        raise ValueError(f"unknown merge policy {policy!r}")
    first, second = (llm_result, regex_result) if policy == "llm_overrides" else (regex_result, llm_result)
    out = empty_extraction(f"{regex_result.get('extractor')}+{llm_result.get('extractor')}")
    for k in EXTRACTION_KEYS:
        if k in ("components", "shd_composite", "n_fields_found", "extractor", "confidence"):
            continue
        v = first.get(k)
        if v is None or v == [] :
            v = second.get(k)
        out[k] = v
    out["confidence"] = max(float(regex_result.get("confidence") or 0.0),
                            float(llm_result.get("confidence") or 0.0))
    out["valve_lesions"] = out["valve_lesions"] or []
    out["lvef_all"] = out["lvef_all"] or []
    _finalise(out)
    return out


def get_extractor(spec: Optional[str] = None) -> LLMExtractor:
    """``None``/``"null"`` -> :class:`NullExtractor`; ``"pkg.mod:factory"`` ->
    ``factory()`` which must return an :class:`LLMExtractor`.  No other
    values are accepted, so nothing can be reached by accident."""
    if spec is None or spec.strip().lower() in ("", "null", "none", "off"):
        return NullExtractor()
    if ":" not in spec:
        raise ValueError("extractor spec must be 'null' or 'module.path:factory_callable'")
    mod_name, attr = spec.split(":", 1)
    factory = getattr(importlib.import_module(mod_name), attr)
    ext = factory() if callable(factory) and not isinstance(factory, LLMExtractor) else factory
    if not isinstance(ext, LLMExtractor):
        raise TypeError(f"{spec} did not produce an LLMExtractor (got {type(ext).__name__})")
    return ext
