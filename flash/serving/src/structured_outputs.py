"""Structured-outputs (guided decoding) spec normalization. Pure python — no vllm import — so
the CPU router can validate specs at the API edge and the GPU engine can trust what it receives.

Every entry point (``/generate``, ``/adapters/{id}/generate``, ``/v1/chat/completions``,
``POST /adapters``) funnels caller specs through :func:`normalize_structured_outputs`, which maps
the accepted forms (a raw JSON schema or the
canonical constraint dict) onto ONE canonical shape: a dict of ``StructuredOutputsParams`` kwargs.
The engine then only ever does ``StructuredOutputsParams(**spec)`` — all flexibility and error
reporting lives here, on the CPU, where a bad spec is a clean 422 instead of a GPU-side 500.

Normalization is IDEMPOTENT on its own outputs: the router validates a ``GenerateRequest`` (which
normalizes), forwards it over Modal RPC as a dict, and the engine re-validates that dict — so a
canonical spec must survive a second pass unchanged (including ``{}``, the explicit-off marker).
"""

from __future__ import annotations

import json
from typing import Any

from flash.serve.contract import reject_non_finite_json_constant

# The vLLM ``StructuredOutputsParams`` constraint fields we support — exactly ONE must be set (its
# __post_init__ raises ValueError otherwise), which normalization enforces up front.
CONSTRAINT_KEYS = ("json", "regex", "choice", "json_object")
# vLLM also offers grammar/structural_tag, but this server does not support them; reject them
# explicitly so a `{"grammar": ...}` table isn't silently swallowed by the raw-JSON-schema fallback.
REMOVED_KEYS = frozenset({"grammar", "structural_tag"})
# Non-constraint option fields, passed through alongside the single constraint.
OPTION_KEYS = ("disable_any_whitespace", "disable_additional_properties", "whitespace_pattern")

# Caller-friendly spellings mapped onto the canonical constraint keys.
_CONSTRAINT_ALIASES = {"json_schema": "json", "schema": "json", "choices": "choice"}
# String forms meaning "explicitly unconstrained" (compared case-insensitively after stripping).
# "text" mirrors OpenAI's response_format {"type": "text"} sentinel — and the error messages below
# already advertise it as an off-marker, so it must normalize like the others rather than be rejected.
_OFF_STRINGS = frozenset({"", "none", "text"})
# String shorthands for "any valid JSON object".
_JSON_OBJECT_STRINGS = frozenset({"json", "json_object"})

_ALLOWED_KEYS_HINT = (
    f"allowed keys: {', '.join(CONSTRAINT_KEYS)} "
    f"(aliases: {', '.join(sorted(_CONSTRAINT_ALIASES))}), options: {', '.join(OPTION_KEYS)}"
)


class StructuredOutputsError(ValueError):
    """Invalid structured-outputs spec. A ``ValueError`` subclass so pydantic validators surface
    it as a 422 with the message intact."""


def _decode_json(value: str) -> Any:
    return json.loads(value, parse_constant=reject_non_finite_json_constant)


def normalize_structured_outputs(value: Any) -> dict[str, Any] | None:
    """Normalize any accepted structured-output spec to a canonical dict of
    ``StructuredOutputsParams`` kwargs.

    Returns ``None`` when ``value`` is ``None`` (not specified — inherit the adapter default, or
    unconstrained). Returns ``{}`` for the explicit "unconstrained" forms (``false``, ``""``,
    ``"none"``, ``{}``) — distinct from ``None`` so a per-call
    ``{}`` can override a per-adapter default. Raises :class:`StructuredOutputsError` with a clear
    message on invalid input.
    """
    if value is None:
        return None
    if value is False:
        return {}
    if value is True:
        raise StructuredOutputsError(
            "structured outputs spec cannot be `true`: pass a constraint "
            f'(e.g. {{"json": <schema>}}); {_ALLOWED_KEYS_HINT}'
        )
    if isinstance(value, str):
        return _normalize_str(value)
    if isinstance(value, dict):
        return _normalize_dict(value)
    raise StructuredOutputsError(
        "structured outputs spec must be a dict, a JSON string, or an off marker "
        f"(null/false/'none'/'text'), got {type(value).__name__}"
    )


def _normalize_str(value: str) -> dict[str, Any]:
    keyword = value.strip().lower()
    if keyword in _OFF_STRINGS:
        return {}
    if keyword in _JSON_OBJECT_STRINGS:
        return {"json_object": True}
    # Any other string must be a JSON document describing a dict form (a schema or a spec dict) —
    # convenient for callers whose transport can only carry strings.
    try:
        parsed = _decode_json(value)
    except ValueError as exc:
        raise StructuredOutputsError(
            "structured outputs string must be 'json'/'json_object', an off marker "
            f"(''/'none'/'text'), or valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputsError(
            "structured outputs JSON string must decode to an object (a JSON schema or a "
            f"constraint spec), got {type(parsed).__name__}"
        )
    return _normalize_dict(parsed)


def _normalize_dict(value: dict[str, Any]) -> dict[str, Any]:
    # For spec-shaped dicts, keys explicitly set to null are "unset" (matching the
    # StructuredOutputsParams defaults), so drop them before dispatching — {"json": None} is the
    # same as not constraining at all.
    data = {k: v for k, v in value.items() if v is not None}
    if not data:
        # {} (or a dict of only nulls) is the explicit-off marker. It must round-trip unchanged:
        # the engine re-validates the router-normalized payload, so {} -> {"json": {}} here would
        # silently turn "explicitly unconstrained" into "any valid JSON" on the second pass.
        return {}

    removed = sorted(k for k in data if k in REMOVED_KEYS)
    if removed:
        raise StructuredOutputsError(
            f"structured outputs {', '.join(removed)} constraint(s) are not supported; "
            f"use one of {', '.join(CONSTRAINT_KEYS)}"
        )

    if any(k in CONSTRAINT_KEYS or k in _CONSTRAINT_ALIASES or k in OPTION_KEYS for k in data):
        return _normalize_canonical(data)
    # No constraint keys: treat the whole dict as a raw JSON schema (e.g.
    # {"type": "object", "properties": ...}). Wrap the ORIGINAL dict, not the null-stripped copy:
    # null values inside a schema (e.g. ``"default": null``) are schema content, not spec keys to drop.
    return {"json": value}


def _normalize_canonical(data: dict[str, Any]) -> dict[str, Any]:
    """A dict carrying canonical constraint/option keys (or their aliases)."""
    constraints: dict[str, Any] = {}
    options: dict[str, Any] = {}
    for key, raw in data.items():
        canonical = _CONSTRAINT_ALIASES.get(key, key)
        if canonical in CONSTRAINT_KEYS:
            if canonical in constraints:
                raise StructuredOutputsError(
                    f"structured outputs spec sets {canonical!r} twice "
                    f"(via {key!r} and an alias); {_ALLOWED_KEYS_HINT}"
                )
            constraints[canonical] = _validate_constraint(canonical, raw)
        elif canonical in OPTION_KEYS:
            options[canonical] = _validate_option(canonical, raw)
        else:
            raise StructuredOutputsError(
                f"unknown structured outputs key {key!r}; {_ALLOWED_KEYS_HINT}"
            )
    if len(constraints) != 1:
        found = ", ".join(sorted(constraints)) or "none"
        raise StructuredOutputsError(
            f"structured outputs spec must set exactly one constraint of "
            f"{', '.join(CONSTRAINT_KEYS)}; got {found}"
        )
    # Canonical key order (constraint first, then options) so the dict serializes stably.
    return {**constraints, **{k: options[k] for k in OPTION_KEYS if k in options}}


def _validate_constraint(key: str, value: Any) -> Any:
    if key == "json":
        return _coerce_json_schema(value, key="json")
    if key == "regex":
        if not isinstance(value, str) or not value.strip():
            raise StructuredOutputsError(
                f"structured outputs {key!r} must be a non-empty string, got {type(value).__name__}"
            )
        return value
    if key == "choice":
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or not value:
            raise StructuredOutputsError(
                f"structured outputs 'choice' must be a non-empty list of strings, got {value!r}"
            )
        if not all(isinstance(item, str) for item in value):
            raise StructuredOutputsError(
                f"structured outputs 'choice' entries must all be strings, got {value!r}"
            )
        return value
    assert key == "json_object"
    if value is not True:
        # json_object=false is not a constraint (and StructuredOutputsParams would count it as one
        # anyway); to disable structured outputs send null/false/'none' instead.
        raise StructuredOutputsError(
            "structured outputs 'json_object' must be true (to disable structured outputs "
            "send null, false, or 'none')"
        )
    return True


def _validate_option(key: str, value: Any) -> Any:
    if key == "whitespace_pattern":
        if not isinstance(value, str) or not value:
            raise StructuredOutputsError(
                f"structured outputs option {key!r} must be a non-empty string, "
                f"got {type(value).__name__}"
            )
        return value
    if not isinstance(value, bool):
        raise StructuredOutputsError(
            f"structured outputs option {key!r} must be a boolean, got {type(value).__name__}"
        )
    return value


def _coerce_json_schema(value: Any, *, key: str) -> dict[str, Any]:
    """A JSON schema given as a dict, or as a string that parses to one."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = _decode_json(value)
        except ValueError as exc:
            raise StructuredOutputsError(
                f"structured outputs {key!r} string is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise StructuredOutputsError(
                f"structured outputs {key!r} string must decode to a JSON schema object, "
                f"got {type(parsed).__name__}"
            )
        return parsed
    raise StructuredOutputsError(
        f"structured outputs {key!r} must be a JSON schema object (or a JSON string of one), "
        f"got {type(value).__name__}"
    )
