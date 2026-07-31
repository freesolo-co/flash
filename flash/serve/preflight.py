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
from flash.engine.vram import opd_completion_len
from flash.lora_rank import ServingPreflightError, preflight_train_context_within_serving
from flash.spec import JobSpec

SERVING_PROMPT_TOKEN_ALLOWANCE = 256


class ExternalSchemaReference(ValueError):
    def __init__(self, ref: str) -> None:
        self.ref = ref
        super().__init__(ref)


def reject_external_schema_reference(uri: str):
    raise NoSuchResource(ref=uri)


def _external_schema_reference(schema: dict, validator_class) -> str | None:
    meta_schema = validator_class.META_SCHEMA
    specification_id = meta_schema.get("$id") or meta_schema.get("id")
    specification = specification_with(specification_id)
    pending = [specification.create_resource(schema)]
    while pending:
        resource = pending.pop()
        contents = resource.contents
        if isinstance(contents, Mapping):
            for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                ref = contents.get(keyword)
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


def resolve_smoke_completion_tokens(spec: JobSpec) -> int:
    """Resolve the deployment smoke's completion budget for one generation.

    This sizes a single smoke request, not the run's training context. sft is the reason the two
    cannot share a number: an sft run has no completion budget of its own, because the worker
    trains one packed prompt+completion block bounded by ``max_context_tokens``. Feeding that
    packed limit back as a completion budget double-counts the prompt, so sft derives its smoke
    budget from the sft recipe instead and ``max_context_tokens`` stays a context check.
    """
    explicit = spec.train.max_completion_tokens
    positive_explicit = int(explicit) if explicit is not None and int(explicit) > 0 else None
    if spec.algorithm == "opd":
        return opd_completion_len(positive_explicit, spec.thinking)
    if spec.algorithm == "sft":
        # sft ignores max_completion_tokens (it is a rollout-only knob), so honouring it here would
        # cap the smoke below what the run can legitimately generate: a 2048-context sft run with a
        # leftover max_completion_tokens=320 trains valid long reasoning it could never deploy.
        #
        # the recipe value is the DEFAULT, not the budget. the worker bounds its packed block by an
        # explicit max_context_tokens and only falls back to the recipe when none is set
        # (flash/engine/worker/sft.py), so resolving to the recipe unconditionally sized the smoke
        # for a run that was never trained: an 8192-context adapter got 2048 tokens, came back
        # finish_reason="length", and the truncation guard rejected a checkpoint that answered
        # correctly (codex[bot]). mirror the worker's own resolution instead.
        #
        # not subtracting a prompt allowance is deliberate. the packed limit spans prompt and
        # completion, so this over-allocates by the length of _SMOKE_PROMPT -- the safe direction,
        # since only an under-allocation can fail a good checkpoint, and the caller already clamps
        # to what serving can actually deliver.
        context = spec.train.max_context_tokens
        if context is not None and int(context) > 0:
            return int(context)
        return int(RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len)
    if positive_explicit is not None:
        return positive_explicit
    recipe = RECIPE.rl
    return int(recipe.max_completion_len_thinking if spec.thinking else recipe.max_completion_len)


def resolve_effective_completion_tokens(spec: JobSpec) -> int | None:
    """Resolve the completion budget the serving context guard must fit, if the run has one.

    Returns ``None`` for sft: its ``max_context_tokens`` already spans prompt and completion, and
    the guard checks that separately. Returning a number here would compare a whole-sequence limit
    against completion-only capacity and reject a spec that fits the serving cap exactly.
    """
    if spec.algorithm == "sft":
        return None
    explicit = spec.train.max_completion_tokens
    positive_explicit = int(explicit) if explicit is not None and int(explicit) > 0 else None
    if spec.algorithm == "opd":
        return opd_completion_len(positive_explicit, spec.thinking)
    if positive_explicit is not None:
        return positive_explicit
    recipe = RECIPE.rl
    return int(recipe.max_completion_len_thinking if spec.thinking else recipe.max_completion_len)


def preflight_serving_path(spec: JobSpec) -> None:
    """Validate structured-output serving inputs without contacting a live server."""
    try:
        constraint = parse_structured_outputs(spec.train.structured_outputs)
    except ValueError as exc:
        raise ServingPreflightError(f"train.structured_outputs {exc}") from exc
    if constraint is not None:
        try:
            validate_structured_output_patterns(constraint)
        except ValueError as exc:
            raise ServingPreflightError(f"train.structured_outputs {exc}") from exc
        schema = constraint.get("json")
        if schema is not None:
            try:
                validate_local_json_schema(schema)
            except ExternalSchemaReference as exc:
                raise ServingPreflightError(
                    "train.structured_outputs JSON schema uses external reference "
                    f"{exc.ref!r}; external schema retrieval is unsupported, so use a local "
                    "fragment reference beginning with '#/'"
                ) from exc
            except SchemaError as exc:
                raise ServingPreflightError(
                    f"train.structured_outputs JSON schema is invalid: {exc.message}"
                ) from exc
    # always run the serving context guard, even when structured-outputs parses to
    # None (a truthy-but-empty value), so it can never skip the context check.
    try:
        preflight_train_context_within_serving(
            spec,
            completion_tokens=resolve_effective_completion_tokens(spec),
            prompt_allowance=SERVING_PROMPT_TOKEN_ALLOWANCE,
        )
    except ValueError as exc:
        raise ServingPreflightError(str(exc)) from exc
