"""Pure serving-path validation shared by training preflight and deployment smoke."""

from __future__ import annotations

from collections.abc import Mapping

import regex as safe_regex
from jsonschema import SchemaError
from jsonschema.validators import validator_for
from referencing.exceptions import NoSuchResource
from referencing.jsonschema import specification_with

from flash.engine.recipe import RECIPE
from flash.engine.structured_outputs import parse_structured_outputs
from flash.lora_rank import preflight_train_context_within_serving
from flash.spec import JobSpec

SERVING_PROMPT_TOKEN_ALLOWANCE = 256


class ExternalSchemaReference(ValueError):
    def __init__(self, ref: str) -> None:
        self.ref = ref
        super().__init__(ref)


def reject_external_schema_reference(uri: str):
    raise NoSuchResource(ref=uri)


def _external_schema_reference(schema: dict, validator_class) -> str | None:
    specification = specification_with(validator_class.META_SCHEMA["$id"])
    pending = [specification.create_resource(schema)]
    while pending:
        resource = pending.pop()
        contents = resource.contents
        if isinstance(contents, Mapping):
            ref = contents.get("$ref")
            if isinstance(ref, str) and ref and not ref.startswith("#"):
                return ref
        pending.extend(resource.subresources())
    return None


def validate_local_json_schema(schema: dict, *, validator_factory=validator_for):
    """Validate schema syntax and reject references requiring external retrieval."""
    validator_class = validator_factory(schema)
    validator_class.check_schema(schema)
    external = _external_schema_reference(schema, validator_class)
    if external is not None:
        raise ExternalSchemaReference(external)
    return validator_class


def validate_structured_output_patterns(constraint: dict) -> None:
    """Compile every accepted regex field with the serving smoke's regex engine."""
    for field in ("regex", "whitespace_pattern"):
        pattern = constraint.get(field)
        if pattern is None:
            continue
        try:
            safe_regex.compile(str(pattern))
        except safe_regex.error as exc:
            raise ValueError(f"{field} is invalid: {exc}") from exc


def resolve_effective_completion_tokens(spec: JobSpec) -> int:
    """Resolve the run's positive explicit or recipe-default completion-token budget."""
    explicit = spec.train.max_completion_tokens
    if explicit is not None and int(explicit) > 0:
        return int(explicit)
    recipe = RECIPE.opd if spec.algorithm == "opd" else RECIPE.rl
    return int(recipe.max_completion_len_thinking if spec.thinking else recipe.max_completion_len)


def preflight_serving_path(spec: JobSpec) -> None:
    """Validate structured-output serving inputs without contacting a live server."""
    constraint = parse_structured_outputs(spec.train.structured_outputs)
    if constraint is None:
        return
    try:
        validate_structured_output_patterns(constraint)
    except ValueError as exc:
        raise ValueError(f"train.structured_outputs {exc}") from exc
    schema = constraint.get("json")
    if schema is not None:
        try:
            validate_local_json_schema(schema)
        except ExternalSchemaReference as exc:
            raise ValueError(
                "train.structured_outputs JSON schema uses external $ref "
                f"{exc.ref!r}; external schema retrieval is unsupported, so use a local "
                "fragment reference beginning with '#/'"
            ) from exc
        except SchemaError as exc:
            raise ValueError(
                f"train.structured_outputs JSON schema is invalid: {exc.message}"
            ) from exc
    preflight_train_context_within_serving(
        spec,
        completion_tokens=resolve_effective_completion_tokens(spec),
        prompt_allowance=SERVING_PROMPT_TOKEN_ALLOWANCE,
    )
