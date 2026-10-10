"""Pure serving-path validation shared by training preflight and deployment smoke."""

from __future__ import annotations

from collections.abc import Mapping

import regex as safe_regex
from jsonschema import SchemaError
from jsonschema.validators import validator_for
from referencing.exceptions import NoSuchResource
from referencing.jsonschema import specification_with

from flash.adapters.lora_rank import ServingPreflightError, preflight_train_context_within_serving
from flash.content.structured_outputs import parse_structured_outputs
from flash.core.spec import JobSpec
from flash.engine.plan.recipe import RECIPE
from flash.engine.plan.vram import opd_completion_len

SERVING_PROMPT_TOKEN_ALLOWANCE = 256

# ceiling on what one deployment smoke may generate, independent of whatever the run trains at.
# the smoke asks a fixed trivial question under a wall clock that must ALSO cover cold-starting the
# base model and loading the adapter, so a budget inherited from training spends that deadline on
# tokens nobody reads: an sft run raised to an 8192 context asked a thinking 27B for 8192 tokens and
# timed out the deployment. that also coupled the knobs the wrong way, since raising
# max_context_tokens to avoid training truncation made the run harder to deploy.
# sized at the largest thinking budget any algorithm resolves to by default, so no
# default-configured run is smoked at less than it is today, and a generation that still hits this
# has failed to answer a trivial prompt rather than run out of room.
SMOKE_COMPLETION_TOKEN_CEILING = 2048


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


def _run_completion_budget(spec: JobSpec) -> int:
    """The completion budget the run itself trains at, before the smoke's own ceiling applies."""
    explicit = spec.train.max_completion_tokens
    positive_explicit = int(explicit) if explicit is not None and int(explicit) > 0 else None
    if spec.algorithm == "opd":
        return opd_completion_len(positive_explicit, spec.thinking)
    if spec.algorithm == "sft":
        # SFT ignores max_completion_tokens, so mirror the worker: use explicit max_context_tokens
        # with the recipe fallback.
        context = spec.train.max_context_tokens
        if context is not None and int(context) > 0:
            return int(context)
        return int(RECIPE.sft.max_seq_len_thinking if spec.thinking else RECIPE.sft.max_seq_len)
    if positive_explicit is not None:
        return positive_explicit
    recipe = RECIPE.rl
    return int(recipe.max_completion_len_thinking if spec.thinking else recipe.max_completion_len)


def resolve_smoke_completion_tokens(spec: JobSpec, *, constrained: bool = False) -> int:
    """Resolve the deployment smoke's completion budget for one generation.

    This sizes a single smoke request, not the run's training context, and the two are deliberately
    decoupled: the smoke's question is fixed and trivial, so what a run trains at says nothing about
    how many tokens answering it needs. Taking the run's number verbatim let a long training context
    spend the smoke's whole wall-clock budget generating, which is why the ceiling exists.

    A configured grammar is exempt. Deployment registers it as the adapter's serving default, so the
    smoke generates under it too, and the shortest string it admits can exceed the ceiling -- a long
    ``choice``, a fixed-repetition ``regex``, a schema with a large ``minLength``. Capping there
    would truncate the only legal answer and reject an adapter that serves correctly. Deciding that
    from the constraint needs the tokenizer and a minimum-length analysis of three grammar dialects,
    so the run's own budget stands in for it: the run trains within that number, so it bounds the
    grammar it was trained to satisfy.
    """
    budget = _run_completion_budget(spec)
    if constrained:
        return budget
    return min(budget, SMOKE_COMPLETION_TOKEN_CEILING)


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
