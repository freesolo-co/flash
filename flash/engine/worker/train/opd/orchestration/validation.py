"""cpu-only fail-closed validation for structured-output verl OPD."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from flash.content.structured_outputs import parse_structured_outputs

_MISTRAL_TOKENIZER_FILE = re.compile(r"(?:^|/)(?:tekken\.json|tokenizer(?:\.mm)?\.model\.v[^/]*)$")


@dataclass(frozen=True)
class OpdVerlStructuredValidation:
    constraint: dict[str, Any] | None
    model_vocab_size: int = 0


def _xgrammar_unsupported_json_feature(schema: dict[str, Any]) -> bool:
    """mirror vllm 0.11.0's xgrammar-to-guidance fallback predicate."""
    if not isinstance(schema, dict):
        return False
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        # array-valued type: the keyword applies if ANY member type would trigger the fallback;
        # normalize by re-running the predicate per member.
        return any(
            _xgrammar_unsupported_json_feature({**schema, "type": member}) for member in schema_type
        )
    # keywords trigger the guidance fallback whether or not "type" is spelled out (json-schema
    # keywords apply implicitly); an untyped schema with these keywords must also be rejected.
    if schema_type in {"integer", "number", None} and "multipleOf" in schema:
        return True
    if schema_type in {"array", None} and any(
        key in schema for key in ("uniqueItems", "contains", "minContains", "maxContains")
    ):
        return True
    if schema_type in {"string", None} and "format" in schema:
        return True
    if schema_type in {"object", None} and any(
        key in schema
        for key in ("minProperties", "maxProperties", "propertyNames", "patternProperties")
    ):
        return True
    for value in schema.values():
        if isinstance(value, dict) and _xgrammar_unsupported_json_feature(value):
            return True
        if isinstance(value, list) and any(
            isinstance(item, dict) and _xgrammar_unsupported_json_feature(item) for item in value
        ):
            return True
    return False


def _require_exact_replay_constraint(constraint: dict[str, Any]) -> None:
    if "whitespace_pattern" in constraint:
        raise ValueError(
            "train.structured_outputs whitespace_pattern is unsupported by vLLM 0.11.0 V1"
        )
    if "disable_additional_properties" in constraint:
        raise ValueError(
            "train.structured_outputs disable_additional_properties is guidance-backend only; "
            "verl OPD pins xgrammar"
        )
    schema = constraint.get("json")
    if isinstance(schema, dict) and _xgrammar_unsupported_json_feature(schema):
        raise ValueError(
            "train.structured_outputs JSON schema requires vLLM's guidance fallback, but verl OPD "
            "pins xgrammar for exact forced-token replay"
        )


def _resolve_compiler_vocab_size(*, model_id: str, model_revision: str) -> int:
    from flash.core.catalog import resolve_model

    return int(resolve_model(model_id, "opd", model_revision=model_revision).vocab_size)


def _resolve_structured_model_metadata(
    model_id: str, model_revision: str
) -> tuple[int, tuple[str, ...]]:
    from huggingface_hub import HfApi

    from flash.engine.plan.vram import fetch_hf_model_geometry

    _params_b, vocab_size, _hidden, _layers, _heads = fetch_hf_model_geometry(
        model_id,
        model_revision,
        strict=True,
    )
    try:
        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(
            model_id,
            revision=model_revision or None,
        )
    except Exception as exc:
        raise ValueError(
            f"could not resolve structured OPD tokenizer metadata for model {model_id!r}"
        ) from exc
    return int(vocab_size), tuple(str(path) for path in files)


def _uses_vllm_mistral_tokenizer(repo_files: tuple[str, ...]) -> bool:
    return any(_MISTRAL_TOKENIZER_FILE.search(path) is not None for path in repo_files)


def _require_non_mistral_tokenizer(repo_files: tuple[str, ...]) -> None:
    if _uses_vllm_mistral_tokenizer(repo_files):
        raise ValueError(
            "structured OPD does not support vLLM MistralTokenizer models; exact replay parity is "
            "only proven for the non-Mistral xgrammar tokenizer path"
        )


def preflight_opd_structured_outputs(
    structured_outputs: str | None,
    *,
    model_id: str,
    model_revision: str = "",
) -> None:
    """Reject the structured-OPD inputs that need no allocation to judge.

    Split out of the full validation below so submission can run it: both of these are decided by
    the constraint and the model id alone, and both reject the run permanently, so leaving them to
    the worker charged the user for a gpu to learn the job could never have run.

    Deliberately NOT the whole validation. The vocab-size comparison depends on the allocated card
    count, and applying it here -- before the allocator has placed the run -- would re-raise "does
    not fit" for a shardable shape submission legitimately accepted.
    """
    constraint = parse_structured_outputs(structured_outputs)
    if constraint is None:
        return
    _require_exact_replay_constraint(constraint)
    _, repo_files = _resolve_structured_model_metadata(model_id, model_revision)
    _require_non_mistral_tokenizer(repo_files)


def validate_opd_structured_outputs(
    structured_outputs: str | None,
    *,
    model_id: str,
    model_revision: str = "",
    compiler_vocab_size: int | None = None,
) -> OpdVerlStructuredValidation:
    """validate every assumption required for exact vllm generation and cpu replay."""
    constraint = parse_structured_outputs(structured_outputs)
    if constraint is None:
        return OpdVerlStructuredValidation(constraint=None)

    _require_exact_replay_constraint(constraint)
    if compiler_vocab_size is None:
        compiler_vocab_size = _resolve_compiler_vocab_size(
            model_id=model_id,
            model_revision=model_revision,
        )
    compiler_vocab_size = int(compiler_vocab_size)
    actual_vocab_size, repo_files = _resolve_structured_model_metadata(model_id, model_revision)
    _require_non_mistral_tokenizer(repo_files)
    if actual_vocab_size <= 0 or compiler_vocab_size != actual_vocab_size:
        raise ValueError(
            "structured OPD model vocabulary size does not match the xgrammar compiler vocabulary"
        )
    return OpdVerlStructuredValidation(
        constraint=constraint,
        model_vocab_size=compiler_vocab_size,
    )
