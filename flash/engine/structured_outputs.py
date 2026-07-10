"""Structured-outputs (guided decoding) helpers shared by the GRPO and OPD rollout paths.

TrainSpec.structured_outputs carries the constraint as canonical JSON — exactly the kwargs of
vLLM's ``StructuredOutputsParams`` (one of json/regex/choice/json_object plus backend options),
normalized once at TOML parse time (schema/fields.py). The helpers here
decode that string and describe it for worker logs; they import no GPU deps so every caller
stays CPU-importable.
"""

from __future__ import annotations

import json

# The vLLM StructuredOutputsParams constraint fields Flash supports. Single source of truth: the
# schema-time normalizer (schema/fields.py) imports this so the two layers can never drift.
CONSTRAINT_KEYS = ("json", "regex", "choice", "json_object")

# vLLM reasoning parser for Flash's <think>...</think> thinking format. Set as the rollout engine's
# EngineArgs.reasoning_parser, it makes vLLM's V1 structured-output gate hold the guided grammar
# until the reasoning block closes (</think>) instead of binding from the very first token — so a
# thinking model reasons freely, then only its post-</think> answer is constrained. It is
# format-based, not weight-specific: the deepseek_r1 parser resolves the </think> boundary from the
# model's own tokenizer, so it fits every Flash thinking model (Qwen3.x, GLM, ...), all of which
# delimit reasoning with <think>. Without it, a json/regex/choice constraint would forbid the free
# <think> phase entirely (it forces the first token to open the schema, e.g. `{`).
THINKING_REASONING_PARSER = "deepseek_r1"


def reasoning_parser_for(*, thinking: bool, structured_outputs: dict | None) -> str | None:
    """The vLLM reasoning-parser name that defers a guided grammar past </think>, or None.

    Only meaningful when BOTH thinking is on AND a structured-outputs constraint is configured: with
    no constraint the grammar gate never runs, and with thinking off there is no reasoning phase to
    protect. When both hold, returning the parser lets vLLM apply the schema only to the answer that
    follows the reasoning block, so structured outputs and thinking are no longer mutually exclusive.

    Single-turn (the common structured-output case) is fully covered: the prompt carries no closing
    </think>, so the grammar stays off until the model emits one. In a multi-turn rollout the parser
    is engine-global and each later turn's prompt already contains a prior </think>, so vLLM treats
    reasoning as already ended and constrains that turn from its first token — no worse than the
    unparsed baseline, and turn one still reasons freely."""
    if thinking and structured_outputs:
        return THINKING_REASONING_PARSER
    return None


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
    if not isinstance(spec, dict) or not any(spec.get(k) is not None for k in CONSTRAINT_KEYS):
        raise ValueError(f"corrupt train.structured_outputs payload (no constraint): {spec_json!r}")
    return spec


def describe_structured_outputs(spec: dict) -> str:
    """One-line summary for worker logs, e.g. ``json (3 schema keys)`` or ``choice (4 options)``."""
    for kind in CONSTRAINT_KEYS:
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
    # Defensive default: validated specs always carry a constraint key (parse_structured_outputs
    # guarantees it), so this is unreachable in practice — it just keeps the return type total.
    return "unconstrained"
