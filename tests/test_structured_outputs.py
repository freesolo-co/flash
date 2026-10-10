"""[train] structured_outputs: TOML parsing/normalization, spec roundtrip, and worker decode.

The knob must accept every reasonable spelling (canonical vLLM constraint table, aliases, bare
JSON-schema table, JSON strings, shorthands), normalize ALL of them to
one canonical StructuredOutputsParams-kwargs JSON string at parse time, survive the
schema -> JobSpec -> worker to_dict/from_dict hops byte-for-byte, and decode losslessly in the
worker. Anything ambiguous or contradictory must be a parse-time ConfigError — a mis-specified
constraint that trained unconstrained would silently poison the reward.
"""

from __future__ import annotations

import json

import pytest

from flash.content.structured_outputs import (
    THINKING_REASONING_PARSER,
    describe_structured_outputs,
    parse_structured_outputs,
    reasoning_parser_for,
)
from flash.core.spec import JobSpec
from flash.schema import ConfigError, spec_from_dict

_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}


def _raw(structured_outputs=None, algorithm="grpo", **top):
    raw = {
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": algorithm,
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {},
        **top,
    }
    if algorithm == "sft":
        raw["train"]["max_examples"] = 2
    if structured_outputs is not None:
        raw["train"]["structured_outputs"] = structured_outputs
    return raw


def _canonical(structured_outputs, **top) -> dict:
    spec = spec_from_dict(_raw(structured_outputs, **top), run_id="so-x")
    assert spec.train.structured_outputs, "expected a normalized constraint"
    return json.loads(spec.train.structured_outputs)


# ---------------------------------------------------------------- accepted forms


def test_canonical_table_forms_normalize():
    assert _canonical({"json": _SCHEMA}) == {"json": _SCHEMA}
    assert _canonical({"regex": r"yes|no"}) == {"regex": r"yes|no"}
    assert _canonical({"choice": ["yes", "no"]}) == {"choice": ["yes", "no"]}
    assert _canonical({"json_object": True}) == {"json_object": True}


def test_grammar_and_structural_tag_are_rejected():
    # grammar / structural_tag are intentionally NOT supported -> rejected, not silently treated
    # as a bare JSON schema.
    for bad in ({"grammar": "root ::= digit+"}, {"structural_tag": '{"triggers": []}'}):
        with pytest.raises(ConfigError, match="not supported"):
            spec_from_dict(_raw(bad), run_id="so-drop")


def test_aliases_fold_to_vllm_field_names():
    assert _canonical({"json_schema": _SCHEMA}) == {"json": _SCHEMA}
    assert _canonical({"schema": _SCHEMA}) == {"json": _SCHEMA}
    assert _canonical({"choices": ["a", "b"]}) == {"choice": ["a", "b"]}


def test_bare_json_schema_table_is_a_json_constraint():
    # No constraint key at all -> the table IS the schema ("type"/"properties" are schema vocab).
    assert _canonical(_SCHEMA) == {"json": _SCHEMA}


def test_enum_schema_is_a_json_constraint():
    # An enum inside the schema (the common "one of these values" case) is ordinary schema content:
    # it rides through untouched as a json constraint for vLLM/xgrammar to enforce at decode time.
    enum_schema = {
        "type": "object",
        "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        "required": ["color"],
    }
    assert _canonical(enum_schema) == {"json": enum_schema}
    # A bare enum ({"enum": [...]}) is itself a valid JSON Schema (no constraint keys -> raw schema).
    assert _canonical({"enum": ["yes", "no"]}) == {"json": {"enum": ["yes", "no"]}}


def test_former_openai_json_object_is_now_a_raw_schema():
    # {"type": "json_object"} is no longer an OpenAI mode; with no constraint keys it is a raw schema.
    assert _canonical({"type": "json_object"}) == {"json": {"type": "json_object"}}


def test_string_forms():
    # inline JSON text (how a TOML single-quoted string arrives), shorthands, and JSON-of-canonical
    assert _canonical(json.dumps(_SCHEMA)) == {"json": _SCHEMA}
    assert _canonical(json.dumps({"regex": "a+"})) == {"regex": "a+"}
    assert _canonical("json") == {"json_object": True}
    assert _canonical("JSON_OBJECT") == {"json_object": True}
    # json constraint given as an embedded JSON string still lands as a parsed schema table
    assert _canonical({"json": json.dumps(_SCHEMA)}) == {"json": _SCHEMA}


def test_backend_options_pass_through():
    got = _canonical(
        {"json": _SCHEMA, "disable_any_whitespace": True, "whitespace_pattern": r"[\n ]?"}
    )
    assert got == {
        "json": _SCHEMA,
        "disable_any_whitespace": True,
        "whitespace_pattern": r"[\n ]?",
    }


def test_explicit_off_forms_are_unset():
    # A user can always opt out of structured outputs: false, "", "none", or omitting the key all
    # leave the rollout unconstrained (the canonical "" spec).
    for off in (False, "", "none"):
        spec = spec_from_dict(_raw(off), run_id="so-off")
        assert spec.train.structured_outputs == ""
    # and omitting the key entirely
    assert spec_from_dict(_raw(), run_id="so-none").train.structured_outputs == ""


# ---------------------------------------------------------------- rejected forms


@pytest.mark.parametrize(
    ("bad", "hint"),
    [
        ({"json": _SCHEMA, "regex": "a+"}, "exactly ONE constraint"),
        ({"json": _SCHEMA, "schema": _SCHEMA}, "aliases"),
        ({"json": _SCHEMA, "backend": "xgrammar"}, "unknown key"),
        ({"choice": []}, "non-empty list"),
        ({"choice": ["a", 1]}, "non-empty list"),
        ({"regex": 3}, "non-empty string"),
        ({"json_object": False}, "must be true"),
        ({"json": 42}, "schema table"),
        ({"json": "{not json"}, "not valid JSON"),
        ({"type": "json_schema", "schema": _SCHEMA}, "unknown key"),
        ({"disable_any_whitespace": True}, "without a constraint"),
        ({"whitespace_pattern": r"[\n ]?"}, "without a constraint"),
        ({"json": _SCHEMA, "disable_any_whitespace": "yes"}, "boolean"),
        ({"json": _SCHEMA, "whitespace_pattern": 1}, "whitespace_pattern"),
        ("{not json", "unparseable"),
        (42, "must be a table"),
        ([_SCHEMA], "must be a table"),
        ("[1, 2]", "decode to an object"),
    ],
)
def test_invalid_forms_are_config_errors(bad, hint):
    with pytest.raises(ConfigError, match=r"train\.structured_outputs") as exc:
        spec_from_dict(_raw(bad), run_id="so-bad")
    assert hint in str(exc.value)


def test_sft_rejects_structured_outputs():
    # SFT never generates; a constraint there is a silent no-op -> parse-time rejection.
    with pytest.raises(ConfigError, match="only applies to rollout algorithms"):
        spec_from_dict(_raw({"json": _SCHEMA}, algorithm="sft"), run_id="so-sft")


def test_opd_accepts_structured_outputs():
    spec = spec_from_dict(_raw({"json": _SCHEMA}, algorithm="opd"), run_id="so-opd")
    assert json.loads(spec.train.structured_outputs) == {"json": _SCHEMA}


# ---------------------------------------------------------------- roundtrip + worker decode


def test_roundtrips_schema_then_worker_reparse():
    # server parse -> worker JobSpec.from_dict must carry the canonical string byte-for-byte,
    # and the worker decode must return exactly the vLLM kwargs.
    spec = spec_from_dict(_raw({"schema": _SCHEMA, "disable_any_whitespace": True}), run_id="so-rt")
    rehydrated = JobSpec.from_dict(spec.to_dict())
    assert rehydrated.train.structured_outputs == spec.train.structured_outputs
    assert parse_structured_outputs(rehydrated.train.structured_outputs) == {
        "json": _SCHEMA,
        "disable_any_whitespace": True,
    }
    # canonical form is deterministic (sorted, compact) so spec hashing/dedup stays stable
    assert spec.train.structured_outputs == json.dumps(
        {"json": _SCHEMA, "disable_any_whitespace": True}, sort_keys=True, separators=(",", ":")
    )


def test_worker_parse_handles_unset_and_corrupt():
    assert parse_structured_outputs("") is None
    assert parse_structured_outputs(None) is None
    with pytest.raises(ValueError, match="corrupt"):
        parse_structured_outputs("{not json")
    with pytest.raises(ValueError, match="no constraint"):
        parse_structured_outputs('{"disable_any_whitespace": true}')


def test_describe_structured_outputs():
    assert describe_structured_outputs({"json": _SCHEMA}) == "json (3 schema keys)"
    assert describe_structured_outputs({"choice": ["a", "b"]}) == "choice (2 options)"
    assert describe_structured_outputs({"json_object": True}) == "json_object"
    assert describe_structured_outputs({"regex": "a+"}) == "regex"


def test_thinking_plus_constraint_notes_deferred_grammar_on_stderr(capsys):
    # thinking + a constraint is supported: the worker defers the grammar past </think>, so the CLI
    # notes the deferred behavior instead of warning about first-token constraint.
    thinking_raw = _raw({"json": _SCHEMA}, model="Qwen/Qwen3.5-9B", thinking=True)
    spec_from_dict(thinking_raw, run_id="so-warn")
    err = capsys.readouterr().err
    assert "after the </think> reasoning phase" in err
    # a non-thinking constrained run says nothing about </think>
    plain_raw = _raw({"json": _SCHEMA}, model="Qwen/Qwen3.5-9B", thinking=False)
    spec_from_dict(plain_raw, run_id="so-warn2")
    assert "</think>" not in capsys.readouterr().err


def test_reasoning_parser_for_gates_on_thinking_and_constraint():
    # Only thinking AND a constraint together defer the grammar past </think>.
    assert reasoning_parser_for(thinking=True, structured_outputs={"json": _SCHEMA}) == (
        THINKING_REASONING_PARSER
    )
    # no constraint -> the grammar gate never runs; thinking off -> no reasoning phase to protect.
    assert reasoning_parser_for(thinking=True, structured_outputs=None) is None
    assert reasoning_parser_for(thinking=False, structured_outputs={"json": _SCHEMA}) is None
    assert reasoning_parser_for(thinking=False, structured_outputs=None) is None
