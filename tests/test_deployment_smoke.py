"""The bounded deployment smoke: provenance, thinking, structured output, and its hard deadlines.

Everything here runs against untrusted model output, so the point of most of these is that a
hostile or broken answer is REFUSED rather than accepted or allowed to hang.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flash.serve.deploy import AliasThinkingSilent, RetryableServingUnavailable, ServingError
from flash.server.domain import deployment_smoke as smoke

_SHA = "b" * 40
_RUN = "flash-1-abcd"
_REVISION = f"{_RUN}@final.{_SHA}"


def _response(
    content: str,
    *,
    finish: str = "stop",
    revision: str = _REVISION,
    checkpoint: str = _RUN,
    message_extra: dict | None = None,
) -> dict:
    message = {"content": content}
    if message_extra:
        message.update(message_extra)
    hf_revision = revision.rsplit(".", 1)[-1]
    return {
        "choices": [{"message": message, "finish_reason": finish}],
        "freesolo": {
            "adapter_revision": revision,
            "checkpoint": checkpoint,
            "hf_revision": hf_revision,
        },
        "_freesolo_headers": {
            "adapter_revision": revision,
            "checkpoint": checkpoint,
            "hf_revision": hf_revision,
        },
    }


class _Gateway:
    """A serving gateway that answers the smoke with whatever it was configured to return."""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(result, Exception):
            raise result
        return result

    # the smoke never uses these, but the protocol carries them
    def deployment_record(self, **_kwargs):
        raise AssertionError("the smoke never registers a record")

    def deploy_adapter(self, **_kwargs):
        raise AssertionError("the smoke never deploys")

    def undeploy_adapter(self, _run_id):
        raise AssertionError("the smoke never undeploys")

    def adapter_alias_target(self, _run_id):
        raise AssertionError("the smoke never reads the alias")

    def chat_stream(self, **_kwargs):
        raise AssertionError("the smoke never streams")


def _spec(**overrides):
    from flash.core.spec import JobSpec

    raw = {
        "run_id": _RUN,
        "model": "Qwen/Qwen3.5-4B",
        "algorithm": "grpo",
        "project": "11111111-1111-1111-1111-111111111111",
        "train": {"hf_repo": "owner/runs", "lora_rank": 32},
        "environment": {"id": "ns/proj/env"},
    }
    raw.update(overrides)
    return JobSpec.from_dict(raw)


# ---- provenance ------------------------------------------------------------------------


def test_a_healthy_smoke_reports_its_verification_metadata():
    gateway = _Gateway(_response("4"))

    result = smoke.run_deployment_smoke(
        _RUN, _spec(), serving=gateway, serving_model=_REVISION, expected_checkpoint=_RUN
    )

    assert result["verify_kind"] == "fixed_prompt"
    assert result["verify_turns"] == 1
    assert result["verify_sample"] == "4"
    assert result["verify_finish_reason"] == "stop"
    assert gateway.calls[0]["expected_adapter_revision"] == _REVISION
    assert gateway.calls[0]["temperature"] == 0.0


def test_an_empty_answer_fails_the_smoke():
    gateway = _Gateway(_response("   "))

    with pytest.raises(ServingError, match="returned no content"):
        smoke.run_deployment_smoke(
            _RUN, _spec(), serving=gateway, serving_model=_REVISION, expected_checkpoint=_RUN
        )


def test_a_response_without_provenance_fails_the_smoke():
    response = _response("4")
    del response["freesolo"]
    gateway = _Gateway(response)

    with pytest.raises(ServingError, match="omitted immutable revision provenance"):
        smoke.run_deployment_smoke(
            _RUN, _spec(), serving=gateway, serving_model=_REVISION, expected_checkpoint=_RUN
        )


def test_a_response_for_the_wrong_revision_fails_the_smoke():
    gateway = _Gateway(_response("4", revision=f"{_RUN}@final.{'c' * 40}"))

    with pytest.raises(ServingError, match="wrong adapter revision"):
        smoke.run_deployment_smoke(
            _RUN, _spec(), serving=gateway, serving_model=_REVISION, expected_checkpoint=_RUN
        )


def test_headers_that_disagree_with_the_body_fail_the_smoke():
    response = _response("4")
    response["_freesolo_headers"]["checkpoint"] = "some-other-run"
    gateway = _Gateway(response)

    with pytest.raises(ServingError, match="mismatched provenance headers"):
        smoke.run_deployment_smoke(
            _RUN, _spec(), serving=gateway, serving_model=_REVISION, expected_checkpoint=_RUN
        )


# ---- thinking --------------------------------------------------------------------------


def test_a_catalog_thinking_model_must_close_its_reasoning():
    gateway = _Gateway(_response("<think>still reasoning"))

    with pytest.raises(ServingError, match="never closed its reasoning"):
        smoke.run_deployment_smoke(
            _RUN,
            _spec(thinking=True),
            serving=gateway,
            serving_model=_REVISION,
            expected_checkpoint=_RUN,
        )


def test_a_thinking_answer_is_taken_from_after_the_close_tag():
    assert smoke.thinking_answer("<think>2+2</think>The answer is 4") == "The answer is 4"


def test_a_thinking_response_with_nothing_after_the_tag_is_refused():
    with pytest.raises(ServingError, match="no answer after"):
        smoke.thinking_answer("<think>reasoned</think>   ")


def test_a_doubled_close_tag_is_not_an_answer():
    """A backend that keeps only the sampled close emits the delimiter twice; that is not an
    answer, and neither is an adapter whose answer literally IS the tag."""
    with pytest.raises(ServingError, match="only a close tag"):
        smoke.thinking_answer("<think>why</think></think>")


def test_an_uncataloged_model_is_not_required_to_emit_a_thinking_tag():
    assert smoke.thinking_answer("just an answer", require_tag=False) == "just an answer"


def test_truncation_fails_a_thinking_smoke_even_though_the_tags_balance():
    """Reasoning goes to reasoning_content, so a cut-off run still arrives balanced."""
    gateway = _Gateway(_response("<think>reasoned</think>The ans", finish="length"))

    with pytest.raises(ServingError, match="truncated at the maximum token length"):
        smoke.run_deployment_smoke(
            _RUN,
            _spec(thinking=True),
            serving=gateway,
            serving_model=_REVISION,
            expected_checkpoint=_RUN,
        )


def test_the_catalog_is_asked_directly_whether_a_thinking_tag_is_guaranteed():
    assert smoke.thinking_tag_is_guaranteed(SimpleNamespace(model="Qwen/Qwen3.5-4B")) is True
    assert smoke.thinking_tag_is_guaranteed(SimpleNamespace(model="some/unknown-model")) is False
    assert smoke.thinking_tag_is_guaranteed(SimpleNamespace(model=None)) is False


# ---- alias thinking verification -------------------------------------------------------


def test_an_alias_that_returns_no_reasoning_channel_is_reported_as_silent():
    gateway = _Gateway(_response("<think>x</think>4"))

    with pytest.raises(AliasThinkingSilent):
        smoke.verify_alias_thinking(_RUN, _spec(thinking=True), _REVISION, _RUN, serving=gateway)


def test_an_alias_that_reasons_reports_its_verification():
    gateway = _Gateway(_response("<think>x</think>4", message_extra={"reasoning_content": "x"}))

    result = smoke.verify_alias_thinking(
        _RUN, _spec(thinking=True), _REVISION, _RUN, serving=gateway
    )

    assert result["alias_thinking_tag"] is True
    assert "alias_thinking_verified_at" in result


def test_a_non_string_reasoning_channel_is_refused():
    gateway = _Gateway(
        _response("<think>x</think>4", message_extra={"reasoning_content": {"not": "a string"}})
    )

    with pytest.raises(ServingError, match="non-string reasoning_content"):
        smoke.verify_alias_thinking(_RUN, _spec(thinking=True), _REVISION, _RUN, serving=gateway)


# ---- deadlines -------------------------------------------------------------------------


def test_an_exhausted_budget_refuses_before_the_generation_starts():
    class _NeverCalled(_Gateway):
        def chat(self, **kwargs):
            raise AssertionError("an expired budget must not start a generation")

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        smoke.bounded_smoke_chat(
            serving=_NeverCalled(),
            serving_model=_REVISION,
            thinking=False,
            expected_checkpoint=_RUN,
            expected_adapter_revision=_REVISION,
            max_tokens=16,
            stop_sequences=None,
            deadline=0.0,
            budget_s=1.0,
        )


def test_a_retryable_unavailable_backend_is_retried_within_the_budget():
    gateway = _Gateway(
        RetryableServingUnavailable("cold start", retry_after_seconds=0.01),
        _response("4"),
    )

    result = smoke.run_deployment_smoke(
        _RUN, _spec(), serving=gateway, serving_model=_REVISION, expected_checkpoint=_RUN
    )

    assert result["verify_sample"] == "4"
    assert len(gateway.calls) == 2


def test_an_expired_budget_refuses_before_regex_evaluation(monkeypatch):
    monkeypatch.setattr(
        smoke.safe_regex,
        "fullmatch",
        lambda *a, **k: pytest.fail("expired budgets must not evaluate regexes"),
    )
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 10.0)

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        smoke.bounded_regex_fullmatch("a", "a", deadline=10.0, budget_s=2.0)


def test_a_regex_timeout_bounded_by_the_global_deadline_reports_the_smoke_budget(monkeypatch):
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 10.0)

    def slow(*_args, **_kwargs):
        raise TimeoutError("catastrophic backtracking")

    monkeypatch.setattr(smoke.safe_regex, "fullmatch", slow)

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        smoke.bounded_regex_fullmatch("(a+)+$", "a!", deadline=10.01, budget_s=1.0)


def test_a_regex_timeout_within_the_global_deadline_reports_the_regex_deadline(monkeypatch):
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 10.0)

    def slow(*_args, **_kwargs):
        raise TimeoutError("catastrophic backtracking")

    monkeypatch.setattr(smoke.safe_regex, "fullmatch", slow)

    with pytest.raises(ServingError, match="regex evaluation exceeded"):
        smoke.bounded_regex_fullmatch("(a+)+$", "a!", deadline=100.0, budget_s=600.0)


def test_a_call_that_outlasts_the_deadline_is_abandoned():
    import time as real_time

    with pytest.raises(ServingError, match="deployment_smoke_timeout"):
        smoke.bounded_call(
            lambda: real_time.sleep(5),
            deadline=real_time.monotonic() + 0.05,
            budget_s=0.05,
        )


# ---- structured output -----------------------------------------------------------------


def test_a_choice_constraint_rejects_an_answer_outside_the_set():
    with pytest.raises(ServingError, match="is not one of"):
        smoke.validate_structured_smoke(
            "maybe", {"choice": ["yes", "no"]}, deadline=1e9, budget_s=600.0
        )


def test_a_choice_constraint_accepts_a_member_of_the_set():
    smoke.validate_structured_smoke("yes", {"choice": ["yes", "no"]}, deadline=1e9, budget_s=600.0)


def test_a_json_object_constraint_rejects_a_bare_scalar():
    with pytest.raises(ServingError, match="not a JSON object"):
        smoke.validate_structured_smoke("4", {"json_object": True}, deadline=1e9, budget_s=600.0)


def test_non_finite_json_constants_are_refused():
    with pytest.raises(ServingError, match="not valid JSON"):
        smoke.strict_json_loads('{"x": NaN}')


def test_malformed_json_is_refused():
    with pytest.raises(ServingError, match="not valid JSON"):
        smoke.strict_json_loads("{not json")


def test_schema_diagnostics_are_collapsed_and_bounded():
    """Untrusted schema text reaches users and logs, so it must be single-line and capped."""
    exc = SimpleNamespace(message="  unsafe\n\ttext  " * 100)

    message = smoke.sanitized_schema_error(exc)

    assert "\n" not in message
    assert "\t" not in message
    assert len(message) == 500
    # the shape assertions above are all satisfied by 500 spaces, or by text that was stripped
    # rather than collapsed. this is the one that pins the CONTENT: whitespace runs collapse to
    # single spaces and the surviving prefix is the real message.
    assert message.startswith("unsafe text unsafe text")


@pytest.mark.parametrize(
    ("instance", "schema", "status"),
    [
        ({"answer": 4}, {"type": "object", "required": ["answer"]}, "ok"),
        ({}, {"type": "object", "required": ["answer"]}, "validation"),
        ({}, {"type": 4}, "schema"),
        ({}, {"$ref": "https://schemas.invalid/missing.json"}, "reference"),
    ],
)
def test_the_schema_worker_classifies_each_outcome_and_always_closes_its_connection(
    instance, schema, status
):
    class _Connection:
        def __init__(self) -> None:
            self.sent = []
            self.closed = False

        def send(self, value):
            self.sent.append(value)

        def close(self):
            self.closed = True

    connection = _Connection()

    smoke.json_schema_validation_worker(connection, instance, schema)

    assert connection.closed is True
    assert connection.sent[0][0] == status


def test_an_exploding_validator_becomes_an_inert_error_instead_of_escaping(monkeypatch):
    class _Connection:
        def __init__(self) -> None:
            self.sent = []
            self.closed = False

        def send(self, value):
            self.sent.append(value)

        def close(self):
            self.closed = True

    def boom(_schema):
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(smoke, "validator_for", boom)
    connection = _Connection()

    smoke.json_schema_validation_worker(connection, {}, {})

    assert connection.sent == [("error", "RuntimeError: validator exploded")]
    assert connection.closed is True


def test_a_real_json_schema_violation_fails_the_validation():
    """Exercises the real spawned-process path end to end."""
    with pytest.raises(ServingError, match="violates the configured JSON schema"):
        smoke.validate_json_schema(
            {}, {"type": "object", "required": ["answer"]}, deadline=1e9, budget_s=600.0
        )


def test_a_valid_instance_passes_the_real_spawned_validation():
    smoke.validate_json_schema(
        {"answer": 4}, {"type": "object", "required": ["answer"]}, deadline=1e9, budget_s=600.0
    )
