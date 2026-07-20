"""Pure serving-path validation shared by training preflight and deployment smoke."""

from __future__ import annotations

from jsonschema import SchemaError
from jsonschema.validators import validator_for

from flash.engine.recipe import RECIPE
from flash.engine.structured_outputs import parse_structured_outputs
from flash.lora_rank import preflight_train_context_within_serving
from flash.spec import JobSpec

SERVING_PROMPT_TOKEN_ALLOWANCE = 256


def resolve_effective_completion_tokens(spec: JobSpec) -> int:
    """Resolve the run's explicit or recipe-default completion-token budget."""
    explicit = spec.train.max_completion_tokens
    if explicit is not None:
        return int(explicit)
    recipe = RECIPE.opd if spec.algorithm == "opd" else RECIPE.rl
    return int(recipe.max_completion_len_thinking if spec.thinking else recipe.max_completion_len)


def preflight_serving_path(spec: JobSpec) -> None:
    """Validate structured-output serving inputs without contacting a live server."""
    constraint = parse_structured_outputs(spec.train.structured_outputs)
    if constraint is None:
        return
    schema = constraint.get("json")
    if schema is not None:
        try:
            validator_for(schema).check_schema(schema)
        except SchemaError as exc:
            raise ValueError(
                f"train.structured_outputs JSON schema is invalid: {exc.message}"
            ) from exc
    preflight_train_context_within_serving(
        spec,
        completion_tokens=resolve_effective_completion_tokens(spec),
        prompt_allowance=SERVING_PROMPT_TOKEN_ALLOWANCE,
    )
