"""Structured-outputs (guided decoding) helpers shared by the GRPO and OPD rollout paths.

TrainSpec.structured_outputs carries the constraint as canonical JSON — exactly the kwargs of
vLLM's ``StructuredOutputsParams`` (one of json/regex/choice/json_object plus backend options),
normalized once at TOML parse time (schema/fields.py). The helpers here
decode that string and describe it for worker logs; they import no GPU deps so every caller
stays CPU-importable.
"""

from __future__ import annotations

import json

_CONSTRAINT_KEYS = ("json", "regex", "choice", "json_object")


def parse_structured_outputs(spec_json: str | None) -> dict | None:
    """Decode a TrainSpec.structured_outputs string to StructuredOutputsParams kwargs.

    Returns None when unset (""/None). Raises ValueError on a corrupt payload — the spec is
    platform-normalized before it reaches the worker, so anything unparseable here is a wiring
    bug, not user input, and must fail loudly rather than silently train unconstrained.
    """
    if not spec_json:
        return None
    try:
        spec = json.loads(spec_json)
    except ValueError as exc:
        raise ValueError(f"corrupt train.structured_outputs payload: {spec_json!r} ({exc})") from exc
    if not isinstance(spec, dict) or not any(spec.get(k) is not None for k in _CONSTRAINT_KEYS):
        raise ValueError(f"corrupt train.structured_outputs payload (no constraint): {spec_json!r}")
    return spec


def describe_structured_outputs(spec: dict) -> str:
    """One-line summary for worker logs, e.g. ``json (3 schema keys)`` or ``choice (4 options)``."""
    for kind in _CONSTRAINT_KEYS:
        val = spec.get(kind)
        if val is None:
            continue
        if kind == "json":
            return f"json ({len(val)} schema keys)" if isinstance(val, dict) else "json"
        if kind == "choice":
            return f"choice ({len(val)} options)"
        if kind == "json_object":
            return "json_object"
        return kind
    return "unconstrained"
