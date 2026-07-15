from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from flash.engine.worker.forward_teacher import (
    FORWARD_TEACHER_TERMINAL,
    ForwardTeacherRecordKind,
    ForwardTeacherSemanticRecord,
    ForwardTeacherTopLogprob,
)
from flash.engine.worker.opd import OpdKnobs, _resolve_hybrid_eligibility
from flash.engine.worker.opd_soft_targets import (
    REPORTED_MASS_TOLERANCE,
    ProjectedPosition,
    ProjectedTarget,
    ProjectionDropCounts,
    SoftTargetProjectionError,
    project_visible_records,
    projected_row_is_active,
    sparse_projected_conditional_cross_entropy,
)


def test_final_successful_opd_heartbeat_retains_cumulative_forward_teacher_telemetry(monkeypatch):
    from flash.engine.worker import opd

    telemetry = {
        "forward_teacher_logical_accepted_targets": 7,
        "forward_teacher_supervised_positions": 41,
        "forward_teacher_provider_requests": 5,
        "forward_teacher_provider_generations": 4,
        "forward_teacher_provider_failures": 1,
        "forward_teacher_prompt_tokens": 101,
        "forward_teacher_completion_tokens": 202,
        "forward_teacher_attempts": 6,
        "forward_teacher_retries": 1,
        "forward_teacher_latency_seconds": 12.5,
        "forward_teacher_ambiguous_paid_requests": 0,
    }
    captured = {}

    class _Totals:
        def runtime_telemetry(self):
            return dict(telemetry)

    class _Accounting:
        totals = _Totals()

    monkeypatch.setattr(opd, "gpu_diagnostics", lambda: {"gpu": "ok"})
    monkeypatch.setattr(
        opd._w,
        "heartbeat",
        lambda stage, **fields: captured.update(stage=stage, **fields),
    )

    opd._emit_opd_trained_heartbeat(
        opt_steps=9,
        train_wall=123.0,
        forward_teacher_accounting=_Accounting(),
        hybrid_enabled=True,
    )

    assert captured == {
        "stage": "opd_trained",
        "step": 9,
        "train_wall": 123.0,
        "gpu": {"gpu": "ok"},
        **telemetry,
    }


def test_reverse_only_opd_heartbeat_omits_forward_teacher_telemetry(monkeypatch):
    from flash.engine.worker import opd

    captured = {}

    class _Totals:
        def runtime_telemetry(self):
            return {"forward_teacher_provider_requests": 0}

    class _Accounting:
        totals = _Totals()

    monkeypatch.setattr(opd, "gpu_diagnostics", lambda: {"gpu": "ok"})
    monkeypatch.setattr(
        opd._w,
        "heartbeat",
        lambda stage, **fields: captured.update(stage=stage, **fields),
    )

    opd._emit_opd_trained_heartbeat(
        opt_steps=9,
        train_wall=123.0,
        forward_teacher_accounting=_Accounting(),
        hybrid_enabled=False,
    )

    assert captured == {
        "stage": "opd_trained",
        "step": 9,
        "train_wall": 123.0,
        "gpu": {"gpu": "ok"},
    }


def test_terminal_success_heartbeats_retain_sanitized_forward_teacher_telemetry(monkeypatch):
    from flash.engine import worker
    from flash.engine.worker import finalize

    telemetry = {
        "forward_teacher_provider_requests": 5,
        "forward_teacher_provider_generations": 4,
        "forward_teacher_provider_failures": 1,
        "forward_teacher_prompt_tokens": 101,
        "forward_teacher_completion_tokens": 202,
        "forward_teacher_attempts": 6,
        "forward_teacher_retries": 1,
        "forward_teacher_latency_seconds": 12.5,
        "forward_teacher_ambiguous_paid_requests": 0,
    }
    beats = []
    monkeypatch.setattr(worker, "require_active_env", lambda: type("Env", (), {"id": "env"})())
    monkeypatch.setattr(worker, "hf_upload_file", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(worker, "heartbeat", lambda stage, **fields: beats.append((stage, fields)))
    monkeypatch.setattr(worker, "gpu_diagnostics", lambda: {"gpu": "ok"})
    monkeypatch.setattr(finalize, "gpu_diagnostics", lambda: {"gpu": "ok"})

    finalize.write_train_meta(
        phase="opd",
        step=9,
        adapter_dir="/tmp/adapter",
        model_id="model",
        train_wall=123.0,
        setup_seconds=4.0,
        train_tokens=0,
        generated_tokens=202,
        notes={},
        forward_teacher_runtime_telemetry={
            **telemetry,
            "private_prompt": "must not escape",
        },
    )

    assert [stage for stage, _fields in beats] == ["opd_train_done", "done"]
    for _stage, fields in beats:
        assert telemetry.items() <= fields.items()
        assert "private_prompt" not in fields
    assert "must not escape" not in repr(beats)

    beats.clear()
    finalize.write_train_meta(
        phase="opd",
        step=9,
        adapter_dir="/tmp/adapter",
        model_id="model",
        train_wall=123.0,
        setup_seconds=4.0,
        train_tokens=0,
        generated_tokens=202,
        notes={},
    )
    assert [stage for stage, _fields in beats] == ["opd_train_done", "done"]
    assert all(
        not any(key.startswith("forward_teacher_") for key in fields)
        for _stage, fields in beats
    )


def test_hybrid_eligibility_is_narrow_and_reports_sanitized_reason():
    eligible = OpdKnobs(teacher_model="accounts/fireworks/models/deepseek-v4-pro")
    assert _resolve_hybrid_eligibility(
        multi_turn=False, knobs=eligible, thinking=False, gpu_count=1, activated=True
    ).enabled
    assert (
        _resolve_hybrid_eligibility(
            multi_turn=False, knobs=eligible, thinking=False, gpu_count=1, activated=False
        ).reason
        == "not_activated"
    )

    cases = [
        ({"multi_turn": True}, "multi_turn"),
        ({"thinking": True}, "thinking"),
        (
            {"knobs": OpdKnobs(teacher_model=eligible.teacher_model, structured_outputs="{}")},
            "structured_outputs",
        ),
        (
            {"knobs": OpdKnobs(teacher_model=eligible.teacher_model, stop_sequences=("x",))},
            "explicit_stop_sequences",
        ),
        ({"gpu_count": 2}, "multi_gpu"),
        ({"knobs": OpdKnobs(teacher_model="accounts/fireworks/models/glm-5p2")}, "teacher_pairing"),
    ]
    defaults = {
        "multi_turn": False,
        "knobs": eligible,
        "thinking": False,
        "gpu_count": 1,
        "activated": True,
    }
    for override, reason in cases:
        result = _resolve_hybrid_eligibility(**{**defaults, **override})
        assert result.enabled is False
        assert result.reason == reason
        unactivated = _resolve_hybrid_eligibility(**{**defaults, **override, "activated": False})
        assert unactivated.reason == reason


def test_forward_teacher_prepare_wrapper_sanitizes_expected_error(monkeypatch):
    from flash.engine.worker import opd

    stats = opd._ForwardTeacherBatchStats(provider_requests=1, attempts=2)
    private_cause = RuntimeError("private provider response")
    expected = opd._ForwardTeacherPreparationError(
        "opd hybrid target preparation failed",
        stats=stats,
        retriable=True,
    )

    def _raise_expected(*_args, **_kwargs):
        raise expected from private_cause

    monkeypatch.setattr(opd, "_prepare_forward_teacher_targets_impl", _raise_expected)

    with pytest.raises(
        opd._ForwardTeacherPreparationError,
        match="opd hybrid target preparation failed",
    ) as caught:
        opd._prepare_forward_teacher_targets(object(), object(), [], max_length=8)

    assert caught.value is not expected
    assert caught.value.stats is stats
    assert caught.value.retriable is True
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "private provider response" not in str(caught.value)
    assert "private provider response" not in repr(caught.value.runtime_telemetry)


def test_forward_teacher_prepare_wrapper_preserves_unexpected_exception(monkeypatch):
    from flash.engine.worker import opd

    class _UnexpectedPreparationError(RuntimeError):
        pass

    unexpected = _UnexpectedPreparationError("privacy-safe internal diagnostic")

    def _raise_unexpected(*_args, **_kwargs):
        raise unexpected

    monkeypatch.setattr(opd, "_prepare_forward_teacher_targets_impl", _raise_unexpected)

    with pytest.raises(_UnexpectedPreparationError) as caught:
        opd._prepare_forward_teacher_targets(object(), object(), [], max_length=8)

    assert caught.value is unexpected
    assert caught.value.args == ("privacy-safe internal diagnostic",)
    assert caught.traceback[-1].name == "_raise_unexpected"


def test_forward_teacher_local_validation_does_not_count_provider_request():
    from flash.engine.worker.forward_teacher import ForwardTeacherClient
    from flash.engine.worker.opd import (
        _ForwardTeacherPreparationError,
        _prepare_forward_teacher_targets,
    )

    client = ForwardTeacherClient(
        "key",
        seed=123,
        opener=lambda *_args, **_kwargs: pytest.fail("transport must not open"),
    )
    batch = [(None, [{"role": "user", "content": ["not a string"]}], [1])]

    with pytest.raises(_ForwardTeacherPreparationError, match="messages are invalid") as caught:
        _prepare_forward_teacher_targets(client, object(), batch, max_length=8)

    assert caught.value.stats.provider_requests == 0
    assert caught.value.stats.provider_failures == 0
    assert caught.value.stats.attempts == 0
    assert caught.value.stats.ambiguous_paid_requests == 0


def test_forward_teacher_resource_transient_preserves_opd_accounting():
    from flash.engine.worker.forward_teacher import ForwardTeacherTransientError
    from flash.engine.worker.opd import (
        _ForwardTeacherPreparationError,
        _prepare_forward_teacher_targets,
    )

    class _Client:
        def generate(self, _messages):
            raise ForwardTeacherTransientError(
                "forward_teacher provider reported insufficient system resources",
                attempts=2,
                latency_seconds=3.5,
                ambiguous_paid_requests=2,
            )

    batch = [(None, [{"role": "user", "content": "x"}], [1])]
    with pytest.raises(
        _ForwardTeacherPreparationError, match="insufficient system resources"
    ) as caught:
        _prepare_forward_teacher_targets(_Client(), object(), batch, max_length=8)

    assert caught.value.retriable is True
    assert caught.value.stats.provider_requests == 1
    assert caught.value.stats.provider_failures == 1
    assert caught.value.stats.attempts == 2
    assert caught.value.stats.retries == 1
    assert caught.value.stats.latency_seconds == pytest.approx(3.5)
    assert caught.value.stats.ambiguous_paid_requests == 2


def test_forward_teacher_target_preparation_reports_partial_success_then_transient_failure(monkeypatch):
    from types import SimpleNamespace

    from flash.engine.worker.forward_teacher import ForwardTeacherTransientError
    from flash.engine.worker.opd import (
        _ForwardTeacherPreparationError,
        _prepare_forward_teacher_targets,
    )

    private = "private provider content must not leak"
    calls = []

    class _Client:
        def generate(self, messages):
            calls.append(messages)
            if len(calls) == 3:
                raise ForwardTeacherTransientError(
                    "forward_teacher transport failure",
                    attempts=2,
                    latency_seconds=1.25,
                    ambiguous_paid_requests=1,
                )
            return SimpleNamespace(
                content=private,
                parsed_completion=SimpleNamespace(visible_content_records=()),
                prompt_tokens=5,
                completion_tokens=2,
                attempts=1,
                latency_seconds=0.25,
            )

    target = _loss_target(_loss_row(0, (2,), (1.0,)))
    monkeypatch.setattr(
        "flash.engine.worker.opd._build_projected_target",
        lambda *_args, **_kwargs: target,
    )
    batch = [(None, [{"role": "user", "content": value}], [1]) for value in ("a", "b", "c")]

    with pytest.raises(_ForwardTeacherPreparationError, match="transport failure") as caught:
        _prepare_forward_teacher_targets(_Client(), object(), batch, max_length=8)

    stats = caught.value.stats
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.retriable is True
    assert stats.logical_accepted_targets == 2
    assert stats.supervised_positions == 2
    assert stats.provider_requests == 3
    assert stats.provider_generations == 2
    assert stats.provider_failures == 1
    assert stats.prompt_tokens == 10
    assert stats.completion_tokens == 4
    assert stats.attempts == 4
    assert stats.retries == 1
    assert stats.latency_seconds == pytest.approx(1.75)
    assert stats.ambiguous_paid_requests == 1
    assert not hasattr(caught.value, "targets")
    assert private not in str(caught.value)
    assert private not in repr(caught.value.runtime_telemetry)


def test_forward_teacher_target_preparation_accounts_provider_before_local_target_failure(monkeypatch):
    from types import SimpleNamespace

    from flash.engine.worker.opd import (
        _ForwardTeacherPreparationError,
        _prepare_forward_teacher_targets,
    )

    private = "private successful response"

    class _Client:
        def generate(self, _messages):
            return SimpleNamespace(
                content=private,
                parsed_completion=SimpleNamespace(visible_content_records=()),
                prompt_tokens=7,
                completion_tokens=3,
                attempts=2,
                latency_seconds=0.75,
            )

    monkeypatch.setattr(
        "flash.engine.worker.opd._build_projected_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("opd hybrid target boundary mismatch")
        ),
    )

    with pytest.raises(_ForwardTeacherPreparationError, match="boundary mismatch") as caught:
        _prepare_forward_teacher_targets(
            _Client(),
            object(),
            [(None, [{"role": "user", "content": "prompt"}], [1])],
            max_length=8,
        )

    stats = caught.value.stats
    assert caught.value.retriable is False
    assert stats.logical_accepted_targets == 0
    assert stats.supervised_positions == 0
    assert stats.provider_requests == 1
    assert stats.provider_generations == 1
    assert stats.provider_failures == 0
    assert stats.prompt_tokens == 7
    assert stats.completion_tokens == 3
    assert stats.attempts == 2
    assert stats.retries == 1
    assert stats.latency_seconds == pytest.approx(0.75)
    assert private not in str(caught.value)
    assert private not in repr(caught.value.runtime_telemetry)


class _ProjectionTokenizer:
    def __init__(
        self,
        encodings,
        decodings,
        *,
        vocab_size=128,
        tokenizer_size=None,
        special_ids=(),
    ):
        self.encodings = {text: tuple(ids) for text, ids in encodings.items()}
        self.decodings = {tuple(ids): text for ids, text in decodings.items()}
        self.vocab_size = vocab_size
        self.tokenizer_size = vocab_size if tokenizer_size is None else tokenizer_size
        self.all_special_ids = tuple(special_ids)

    def __len__(self):
        return self.tokenizer_size

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text not in self.encodings:
            raise ValueError("unknown synthetic text")
        return type("Encoded", (), {"input_ids": list(self.encodings[text])})()

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        key = tuple(int(value) for value in ids)
        if key not in self.decodings:
            raise ValueError("unknown synthetic ids")
        return self.decodings[key]


def _visible_record(index, token, alternatives):
    return ForwardTeacherSemanticRecord(
        kind=ForwardTeacherRecordKind.VISIBLE_CONTENT,
        index=index,
        token=token,
        logprob=math.log(0.5),
        top_logprobs=tuple(
            ForwardTeacherTopLogprob(token=text, logprob=math.log(probability))
            for text, probability in alternatives
        ),
    )


def test_projected_target_preparation_uses_visible_records_only_and_preserves_logical_weight():
    from types import SimpleNamespace

    from flash.engine.worker.opd import _prepare_forward_teacher_targets

    visible_records = (_visible_record(3, " visible", ((" visible", 0.6), (" shown", 0.2))),)

    class _Tok(_ProjectionTokenizer):
        pad_token_id = 0

        def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
            if add_generation_prompt:
                return "P"
            return "P" + messages[-1]["content"] + "E"

    tokenizer = _Tok(
        {
            "P": [1],
            "P visible": [1, 2],
            "P shown": [1, 3],
        },
        {
            (1,): "P",
            (1, 2): "P visible",
            (1, 3): "P shown",
        },
    )
    calls = []

    class _Client:
        def generate(self, messages):
            calls.append(messages)
            return SimpleNamespace(
                content=" visible",
                reasoning="hidden reasoning must not project",
                parsed_completion=SimpleNamespace(
                    hidden_reasoning_records=(object(),),
                    boundary_record=object(),
                    visible_content_records=visible_records,
                    terminal_record=object(),
                ),
                prompt_tokens=4,
                completion_tokens=4,
                attempts=1,
                latency_seconds=0.5,
            )

    messages = [{"role": "user", "content": "private prompt"}]
    rollout_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    rollout_prompt_ids = tokenizer(rollout_prompt, add_special_tokens=False).input_ids
    assert rollout_prompt_ids == [1]
    targets, stats = _prepare_forward_teacher_targets(
        _Client(),
        tokenizer,
        [(None, messages, rollout_prompt_ids), (None, messages, rollout_prompt_ids)],
        max_length=8,
    )

    assert calls == [messages]
    assert len(targets) == 2
    assert targets[0] is targets[1]
    assert targets[0].input_ids == (1,)
    assert targets[0].eligible_row_count == 1
    assert targets[0].rows[0].token_ids == (2, 3)
    assert targets[0].rows[0].probabilities == pytest.approx((0.75, 0.25))
    assert stats.logical_accepted_targets == 2
    assert stats.supervised_positions == 2
    assert stats.visible_provider_positions == 2
    assert stats.eligible_projected_rows == 2
    assert stats.retained_support_entries == 4
    assert stats.reported_mass_sum == pytest.approx(1.6)
    assert stats.retained_mass_sum == pytest.approx(1.6)
    assert stats.dropped_mass_sum == pytest.approx(0.4)
    assert stats.provider_requests == 1
    assert stats.provider_generations == 1


def test_projected_target_telemetry_and_backward_count_only_entropy_active_rows(monkeypatch):
    from types import SimpleNamespace

    import flash.engine.worker.opd as opd

    rows = tuple(
        _loss_row(index, (index,), (1.0,), provider_entropy=entropy)
        for index, entropy in enumerate((0.4, 0.5, 0.6))
    )
    target = ProjectedTarget(
        input_ids=(1, 2, 3),
        positions=rows,
        rows=rows,
        visible_position_count=3,
        eligible_row_count=3,
        drop_counts=ProjectionDropCounts(),
    )
    monkeypatch.setattr(opd, "_build_projected_target", lambda *args, **kwargs: target)

    class _Client:
        def generate(self, messages):
            return SimpleNamespace(
                content="visible",
                parsed_completion=SimpleNamespace(visible_content_records=()),
                prompt_tokens=1,
                completion_tokens=1,
                attempts=1,
                latency_seconds=0.1,
            )

    targets, stats = opd._prepare_forward_teacher_targets(
        _Client(),
        object(),
        [(None, [{"role": "user", "content": "private prompt"}], [1])],
        max_length=8,
        entropy_tau=0.5,
    )

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.values = torch.nn.Parameter(torch.zeros(1, 3, 4, dtype=torch.float64))

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(logits=self.values.expand(input_ids.shape[0], -1, -1))

    model = _Model()
    _loss, backward_supervised = opd._backward_projected_targets(
        model,
        targets,
        torch.device("cpu"),
        SimpleNamespace(pad_token_id=0),
        coef=0.03,
        microbatch_size=1,
        entropy_tau=0.5,
    )

    assert stats.supervised_positions == 1
    assert stats.eligible_projected_rows == 3
    assert backward_supervised == stats.supervised_positions
    assert model.values.grad is not None
    assert model.values.grad[0, 2].abs().sum() > 0
    torch.testing.assert_close(model.values.grad[0, :2], torch.zeros(2, 4, dtype=torch.float64))


def test_projected_target_preparation_accepts_all_unprojectable_visible_target_with_telemetry():
    from types import SimpleNamespace

    from flash.engine.worker.opd import _prepare_forward_teacher_targets

    class _Tok(_ProjectionTokenizer):
        pad_token_id = 0

        def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
            if add_generation_prompt:
                return "P"
            return "P" + messages[-1]["content"] + "E"

    tokenizer = _Tok(
        {"P": [1], "Pa": [1, 2], "Pxy": [1, 3, 4]},
        {(1,): "P", (1, 2): "Pa", (1, 3, 4): "Pxy"},
    )
    visible_records = (_visible_record(0, "a", (("xy", 0.4),)),)

    class _Client:
        def generate(self, messages):
            return SimpleNamespace(
                content="a",
                parsed_completion=SimpleNamespace(visible_content_records=visible_records),
                prompt_tokens=2,
                completion_tokens=1,
                attempts=1,
                latency_seconds=0.25,
            )

    messages = [{"role": "user", "content": "private prompt"}]
    targets, stats = _prepare_forward_teacher_targets(
        _Client(),
        tokenizer,
        [(None, messages, [1])],
        max_length=8,
    )

    assert len(targets) == 1
    target = targets[0]
    assert target.eligible_row_count == 0
    assert target.visible_position_count == 1
    assert target.rows == ()
    assert target.drop_counts.multi_token == 1
    assert stats.logical_accepted_targets == 1
    assert stats.supervised_positions == 0
    assert stats.visible_provider_positions == 1
    assert stats.eligible_projected_rows == 0
    assert stats.projected_drop_multi_token == 1
    assert stats.dropped_mass_sum == pytest.approx(1.0)


def _loss_target(*rows):
    return ProjectedTarget(
        input_ids=(1,),
        positions=tuple(rows),
        rows=tuple(rows),
        visible_position_count=len(rows),
        eligible_row_count=len(rows),
        drop_counts=ProjectionDropCounts(),
    )


def _loss_row(
    position,
    token_ids,
    probabilities,
    *,
    provider_entropy=0.0,
    projected_entropy=0.0,
):
    return ProjectedPosition(
        provider_record_index=position,
        logits_index=position,
        token_ids=tuple(token_ids),
        probabilities=tuple(probabilities),
        reported_top_k_mass=1.0,
        retained_projected_mass=1.0,
        rejected_reported_mass=0.0,
        unreported_mass=0.0,
        total_dropped_mass=0.0,
        support_size=len(token_ids),
        collision_count=0,
        conditional_entropy_nats=projected_entropy,
        drop_counts=ProjectionDropCounts(),
        provider_top_k_entropy_nats=provider_entropy,
    )


@pytest.mark.parametrize(
    ("entropy", "active"),
    [
        pytest.param(0.4999999, False, id="below"),
        pytest.param(0.5, False, id="equal"),
        pytest.param(0.5000001, True, id="above"),
    ],
)
def test_projected_row_activation_uses_strict_provider_entropy_threshold(entropy, active):
    row = _loss_row(0, (1,), (1.0,), provider_entropy=entropy, projected_entropy=10.0)

    assert projected_row_is_active(row, 0.5) is active


def test_exact_prefix_projection_records_mass_conditional_normalization_and_entropy():
    tokenizer = _ProjectionTokenizer(
        {"P": [1], "Pa": [1, 2], "Pb": [1, 3]},
        {(1,): "P", (1, 2): "Pa", (1, 3): "Pb", (2,): "a", (3,): "b"},
    )
    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[_visible_record(4, "a", (("a", 0.6), ("b", 0.3)))],
    )

    assert target.input_ids == (1,)
    assert target.visible_position_count == target.eligible_row_count == 1
    row = target.rows[0]
    assert row.logits_index == 0
    assert row.token_ids == (2, 3)
    assert row.probabilities == pytest.approx((2 / 3, 1 / 3))
    assert row.reported_top_k_mass == pytest.approx(0.9)
    assert row.retained_projected_mass == pytest.approx(0.9)
    assert row.rejected_reported_mass == pytest.approx(0.0)
    assert row.unreported_mass == pytest.approx(0.1)
    assert row.total_dropped_mass == pytest.approx(0.1)
    assert row.support_size == 2
    assert row.collision_count == 0
    assert row.conditional_entropy_nats == pytest.approx(
        -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
    )
    assert (
        row.retained_projected_mass + row.rejected_reported_mass + row.unreported_mass
        == pytest.approx(1.0)
    )
    assert row.total_dropped_mass == pytest.approx(row.rejected_reported_mass + row.unreported_mass)


@pytest.mark.parametrize("support_size", [2, 20])
def test_projection_uniform_support_entropy_is_shannon_entropy_in_nats(support_size):
    alternatives = tuple((f"t{index}", 1.0 / support_size) for index in range(support_size))
    encodings = {"P": [1]}
    decodings = {(1,): "P"}
    for index, (text, _probability) in enumerate(alternatives, start=2):
        encodings["P" + text] = [1, index]
        decodings[(1, index)] = "P" + text
    tokenizer = _ProjectionTokenizer(encodings, decodings)

    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[_visible_record(0, alternatives[0][0], alternatives)],
    )

    row = target.rows[0]
    assert row.support_size == support_size
    assert row.conditional_entropy_nats == pytest.approx(math.log(support_size), abs=1e-12)


def test_projection_uses_len_for_added_token_ids_above_base_vocabulary():
    tokenizer = _ProjectionTokenizer(
        {"P": [1], "Pa": [1, 7]},
        {(1,): "P", (1, 7): "Pa"},
        vocab_size=5,
        tokenizer_size=8,
    )

    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[_visible_record(0, "a", (("a", 0.8),))],
    )

    assert target.rows[0].token_ids == (7,)
    assert target.rows[0].retained_projected_mass == pytest.approx(0.8)


def test_projection_accepts_contextual_metaspace_decode_without_isolated_surface_match():
    cleanup_flags = []

    class _MetaspaceTokenizer(_ProjectionTokenizer):
        def decode(
            self,
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=None,
        ):
            cleanup_flags.append(clean_up_tokenization_spaces)
            assert clean_up_tokenization_spaces is False
            key = tuple(int(value) for value in ids)
            if key not in self.decodings:
                raise ValueError("unknown synthetic ids")
            return self.decodings[key]

    tokenizer = _MetaspaceTokenizer(
        {"P": [1], "P word": [1, 2]},
        {(1,): "P", (1, 2): "P word", (2,): "word"},
    )

    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[_visible_record(0, " word", ((" word", 0.8),))],
    )

    assert tokenizer.decodings[(2,)] == "word"
    assert cleanup_flags
    assert all(flag is False for flag in cleanup_flags)
    assert target.rows[0].token_ids == (2,)
    assert target.rows[0].probabilities == (1.0,)


def test_validated_forward_teacher_semantic_records_compose_with_projection():
    from flash.engine.recipe import FORWARD_TEACHER_MODEL_ID
    from flash.engine.worker.forward_teacher import ForwardTeacherClient

    realized_logprob = math.log(0.6)
    result = ForwardTeacherClient._validate(
        {
            "model": FORWARD_TEACHER_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"content": " visible", "reasoning": ""},
                    "logprobs": {
                        "content": [
                            {
                                "token": "</think>",
                                "logprob": -0.01,
                                "top_logprobs": [{"token": "</think>", "logprob": -0.01}],
                            },
                            {
                                "token": " visible",
                                "logprob": realized_logprob,
                                "top_logprobs": [
                                    {
                                        "token": " visible",
                                        "logprob": realized_logprob,
                                    },
                                    {"token": " shown", "logprob": math.log(0.2)},
                                ],
                            },
                            {
                                "token": FORWARD_TEACHER_TERMINAL,
                                "logprob": -0.01,
                                "top_logprobs": [{"token": FORWARD_TEACHER_TERMINAL, "logprob": -0.01}],
                            },
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
        },
        attempts=1,
        latency=0.0,
    )
    tokenizer = _ProjectionTokenizer(
        {
            "P": [1],
            "P visible": [1, 2],
            "P shown": [1, 3],
        },
        {
            (1,): "P",
            (1, 2): "P visible",
            (1, 3): "P shown",
        },
    )

    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=result.parsed_completion.visible_content_records,
    )

    assert target.eligible_row_count == 1
    assert target.rows[0].token_ids == (2, 3)
    assert target.rows[0].probabilities == pytest.approx((0.75, 0.25))


def test_projection_drops_each_invalid_alternative_category_and_keeps_fewer_than_twenty():
    decomposed = "é"
    tokenizer = _ProjectionTokenizer(
        {
            "P": [1],
            "Pa": [1, 2],
            "Pxy": [1, 4, 5],
            "Pr": [7],
            "Ps": [1, 9],
            "Pbad": [1, 50],
            "P" + decomposed: [1, 8],
        },
        {
            (1,): "P",
            (1, 2): "Pa",
            (2,): "a",
            (1, 8): "Pé",
            (8,): "é",
        },
        vocab_size=10,
        special_ids=(9,),
    )
    alternatives = (
        ("a", 0.2),
        ("", 0.1),
        ("xy", 0.1),
        ("r", 0.1),
        ("s", 0.1),
        ("bad", 0.1),
        (decomposed, 0.1),
    )
    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[_visible_record(0, "a", alternatives)],
    )

    row = target.rows[0]
    assert row.support_size == 1 < 20
    assert row.token_ids == (2,)
    assert row.probabilities == (1.0,)
    assert row.conditional_entropy_nats == pytest.approx(0.0)
    assert row.provider_top_k_entropy_nats == pytest.approx(
        -(0.25 * math.log(0.25) + 6 * 0.125 * math.log(0.125))
    )
    assert row.reported_top_k_mass == pytest.approx(0.8)
    assert row.retained_projected_mass == pytest.approx(0.2)
    assert row.rejected_reported_mass == pytest.approx(0.6)
    assert row.unreported_mass == pytest.approx(0.2)
    assert row.total_dropped_mass == pytest.approx(0.8)
    assert row.drop_counts == ProjectionDropCounts(
        zero_token=1,
        multi_token=1,
        prefix_retokenization=1,
        special_token=1,
        invalid_token_id=1,
        round_trip_mismatch=1,
    )
    assert target.drop_counts == row.drop_counts


def test_projection_keeps_zero_retained_mass_as_ineligible_position_telemetry():
    tokenizer = _ProjectionTokenizer(
        {"P": [1], "Pa": [1, 2], "Pxy": [1, 3, 4]},
        {(1,): "P", (1, 2): "Pa"},
    )
    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[_visible_record(0, "a", (("xy", 0.4),))],
    )

    assert target.visible_position_count == 1
    assert target.eligible_row_count == 0
    assert target.input_ids == (1,)
    assert target.rows == ()
    position = target.positions[0]
    assert position.eligible is False
    assert position.retained_projected_mass == 0.0
    assert position.rejected_reported_mass == pytest.approx(0.4)
    assert position.unreported_mass == pytest.approx(0.6)
    assert position.total_dropped_mass == 1.0
    assert position.drop_counts.multi_token == 1


def test_projection_rejects_invalid_record_kind_with_sanitized_error():
    private = "private boundary payload"
    record = ForwardTeacherSemanticRecord(
        kind=ForwardTeacherRecordKind.THINK_BOUNDARY,
        index=0,
        token=private,
        logprob=math.log(0.5),
        top_logprobs=(ForwardTeacherTopLogprob(token=private, logprob=math.log(0.5)),),
    )
    tokenizer = _ProjectionTokenizer({"P": [1]}, {(1,): "P"})

    with pytest.raises(SoftTargetProjectionError, match="record kind") as caught:
        project_visible_records(tokenizer, prefix_text="P", visible_records=[record])

    assert private not in str(caught.value)


def test_projection_defensive_collision_bypasses_provider_validation_and_is_order_invariant():
    tokenizer = _ProjectionTokenizer(
        {"P": [1], "Pa": [1, 2], "Pb": [1, 3]},
        {(1,): "P", (1, 2): "Pa", (1, 3): "Pb", (2,): "a", (3,): "b"},
    )
    first = _visible_record(0, "a", (("a", 0.2), ("a", 0.3), ("b", 0.1)))
    second = _visible_record(0, "a", (("b", 0.1), ("a", 0.3), ("a", 0.2)))
    projected_first = project_visible_records(tokenizer, prefix_text="P", visible_records=[first])
    projected_second = project_visible_records(tokenizer, prefix_text="P", visible_records=[second])

    row = projected_first.rows[0]
    assert row.collision_count == 1
    assert row.retained_projected_mass == pytest.approx(0.6)
    by_id = dict(zip(row.token_ids, row.probabilities, strict=True))
    assert by_id == pytest.approx({2: 5 / 6, 3: 1 / 6})
    logits = torch.tensor([[[0.1, -0.2, 0.7, -0.4]]], dtype=torch.float64)
    first_loss = sparse_projected_conditional_cross_entropy(logits, [projected_first])
    second_loss = sparse_projected_conditional_cross_entropy(logits, [projected_second])
    torch.testing.assert_close(first_loss, second_loss, rtol=0, atol=1e-15)


def test_projection_accepts_float_noise_above_one_but_rejects_material_mass_and_sanitizes():
    tokenizer = _ProjectionTokenizer(
        {"P": [1], "Pa": [1, 2], "Pb": [1, 3]},
        {(1,): "P", (1, 2): "Pa", (1, 3): "Pb", (2,): "a", (3,): "b"},
    )
    noisy = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[
            _visible_record(
                0,
                "a",
                (("a", 0.5), ("b", 0.5 + REPORTED_MASS_TOLERANCE / 2)),
            )
        ],
    )
    noisy_row = noisy.rows[0]
    assert noisy_row.reported_top_k_mass == 1.0
    assert noisy_row.retained_projected_mass == 1.0
    assert noisy_row.rejected_reported_mass == 0.0
    assert noisy_row.unreported_mass == 0.0
    assert noisy_row.total_dropped_mass == 0.0
    assert (
        noisy_row.retained_projected_mass
        + noisy_row.rejected_reported_mass
        + noisy_row.unreported_mass
        == pytest.approx(1.0)
    )

    private = "private impossible alternative"
    with pytest.raises(SoftTargetProjectionError, match="reported top-k mass") as caught:
        project_visible_records(
            tokenizer,
            prefix_text="P",
            visible_records=[
                _visible_record(
                    0,
                    "a",
                    (("a", 0.5), (private, 0.5 + 2 * REPORTED_MASS_TOLERANCE)),
                )
            ],
        )
    assert private not in str(caught.value)


def test_realized_zero_token_and_prefix_retokenization_reject_target_without_text_leakage():
    zero_tokenizer = _ProjectionTokenizer({"P": [1]}, {(1,): "P"})
    with pytest.raises(SoftTargetProjectionError, match="realized extension is empty"):
        project_visible_records(
            zero_tokenizer,
            prefix_text="P",
            visible_records=[_visible_record(0, "", (("", 1.0),))],
        )

    private = "private retokenizing realized text"
    retokenizing = _ProjectionTokenizer(
        {"P": [1, 2], "P" + private: [7]},
        {(1, 2): "P"},
    )
    with pytest.raises(SoftTargetProjectionError, match="realized prefix retokenization") as caught:
        project_visible_records(
            retokenizing,
            prefix_text="P",
            visible_records=[_visible_record(0, private, ((private, 1.0),))],
        )
    assert private not in str(caught.value)


def test_multi_token_realized_extension_advances_prefix_without_creating_row():
    tokenizer = _ProjectionTokenizer(
        {
            "P": [1],
            "Pxy": [1, 2, 3],
            "Pxyz": [1, 2, 3, 4],
            "Pxyw": [1, 2, 3, 5],
        },
        {
            (1,): "P",
            (1, 2, 3): "Pxy",
            (1, 2, 3, 4): "Pxyz",
            (1, 2, 3, 5): "Pxyw",
            (4,): "z",
            (5,): "w",
        },
    )
    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[
            _visible_record(0, "xy", (("xy", 0.8),)),
            _visible_record(1, "z", (("z", 0.6), ("w", 0.2))),
        ],
    )

    assert target.input_ids == (1, 2, 3)
    assert target.visible_position_count == 2
    assert target.eligible_row_count == 1
    assert target.positions[0].eligible is False
    assert target.positions[0].drop_counts.realized_multi_token == 1
    assert target.rows[0].provider_record_index == 1
    assert target.rows[0].logits_index == 2


def test_projection_trims_trailing_ineligible_extension_to_exact_latest_context():
    tokenizer = _ProjectionTokenizer(
        {
            "P": [1, 2],
            "Pa": [1, 2, 3],
            "Paxy": [1, 2, 3, 4, 5],
        },
        {
            (1, 2): "P",
            (1, 2, 3): "Pa",
            (1, 2, 3, 4, 5): "Paxy",
        },
    )
    target = project_visible_records(
        tokenizer,
        prefix_text="P",
        visible_records=[
            _visible_record(0, "a", (("a", 0.7),)),
            _visible_record(1, "xy", (("xy", 0.6),)),
        ],
    )

    assert target.rows[0].logits_index == 1
    assert target.input_ids == (1, 2)
    assert target.positions[1].eligible is False
    assert target.positions[1].drop_counts.realized_multi_token == 1


def test_sparse_projected_conditional_ce_matches_manual_dense_reference_and_reductions():
    logits = torch.tensor(
        [
            [[0.2, -0.1, 0.7, 0.0], [0.1, 0.3, -0.2, 0.4]],
            [[-0.4, 0.5, 0.2, 0.1], [9.0, -9.0, 3.0, -3.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    first_row = _loss_row(0, (1, 2), (0.25, 0.75))
    repeated_row = _loss_row(1, (0, 3), (0.4, 0.6))
    second_target_row = _loss_row(0, (1,), (1.0,))
    targets = [_loss_target(first_row, repeated_row), _loss_target(second_target_row)]

    loss = sparse_projected_conditional_cross_entropy(logits, targets)
    logp = logits.log_softmax(dim=-1)
    first_loss = (
        -(0.25 * logp[0, 0, 1] + 0.75 * logp[0, 0, 2] + 0.4 * logp[0, 1, 0] + 0.6 * logp[0, 1, 3])
        / 2
    )
    second_loss = -logp[1, 0, 1]
    expected = (first_loss + second_loss) / 2
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-15)

    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[1, 0].abs().sum() > 0
    torch.testing.assert_close(logits.grad[1, 1], torch.zeros(4, dtype=torch.float64))


def test_sparse_projected_ce_tau_unset_preserves_uniform_loss_exactly():
    logits = torch.tensor([[[0.2, -0.4, 0.7]]], dtype=torch.float64)
    target = _loss_target(
        _loss_row(0, (1, 2), (0.4, 0.6), provider_entropy=0.7),
    )

    implicit = sparse_projected_conditional_cross_entropy(logits, [target])
    explicit = sparse_projected_conditional_cross_entropy(logits, [target], entropy_tau=None)

    torch.testing.assert_close(explicit, implicit, rtol=0, atol=0)


def test_sparse_projected_ce_routes_on_strict_provider_entropy_with_fixed_denominator():
    logits = torch.tensor(
        [[[0.2, -0.4, 0.7], [0.3, 0.8, -0.2]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    active = _loss_row(
        0,
        (1, 2),
        (0.4, 0.6),
        provider_entropy=0.5000001,
        projected_entropy=0.0,
    )
    boundary = _loss_row(
        1,
        (0, 1),
        (0.5, 0.5),
        provider_entropy=0.5,
        projected_entropy=10.0,
    )
    active_only = sparse_projected_conditional_cross_entropy(
        logits[:, :1],
        [_loss_target(active)],
    )

    gated = sparse_projected_conditional_cross_entropy(
        logits,
        [_loss_target(active, boundary)],
        entropy_tau=0.5,
    )

    torch.testing.assert_close(gated, active_only / 2, rtol=0, atol=1e-15)
    gated.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0].abs().sum() > 0
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros(3, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_projected_ce_validates_python_probabilities_and_computes_half_logits_in_fp32(
    dtype,
):
    probabilities = (0.12345, 0.23456, 0.64199)
    quantized = torch.tensor(probabilities, dtype=dtype)
    assert math.fsum(quantized.float().tolist()) != 1.0
    logits = torch.tensor([[[0.2, -0.1, 0.7, 0.0]]], dtype=dtype, requires_grad=True)
    row = _loss_row(0, (0, 1, 2), probabilities)

    loss = sparse_projected_conditional_cross_entropy(logits, [_loss_target(row)])
    expected = -(
        torch.tensor(probabilities, dtype=torch.float32)
        * logits.float().log_softmax(dim=-1)[0, 0, :3]
    ).sum()

    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected, rtol=0, atol=1e-6)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_projected_ce_mixed_active_inactive_and_empty_targets_use_fp32_graph_zeros(
    dtype,
):
    logits = torch.tensor(
        [
            [[0.2, -0.1, 0.7], [0.3, 0.8, -0.2]],
            [[9.0, -9.0, 3.0], [2.0, -2.0, 1.0]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    active = _loss_row(0, (1, 2), (0.4, 0.6), provider_entropy=0.6)
    inactive = _loss_row(1, (0, 1), (0.5, 0.5), provider_entropy=0.5)
    empty = ProjectedTarget(
        input_ids=(1,),
        positions=(),
        rows=(),
        visible_position_count=0,
        eligible_row_count=0,
        drop_counts=ProjectionDropCounts(),
    )
    active_only = sparse_projected_conditional_cross_entropy(
        logits[:1, :1],
        [_loss_target(active)],
    )

    mixed = sparse_projected_conditional_cross_entropy(
        logits,
        [_loss_target(active, inactive), empty],
        entropy_tau=0.5,
    )

    assert mixed.dtype == torch.float32
    torch.testing.assert_close(mixed, active_only / 4, rtol=0, atol=1e-6)
    mixed.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[0, 1], torch.zeros_like(logits.grad[0, 1]))
    torch.testing.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_projected_ce_rejects_invalid_python_mass_before_half_quantization(dtype):
    probabilities = (0.3334, 0.3334, 0.3334)
    assert math.fsum(probabilities) > 1.0 + 1e-6
    assert torch.tensor(probabilities, dtype=dtype).sum().item() == 1.0
    logits = torch.zeros((1, 1, 3), dtype=dtype)

    with pytest.raises(ValueError, match="not normalized"):
        sparse_projected_conditional_cross_entropy(
            logits,
            [_loss_target(_loss_row(0, (0, 1, 2), probabilities))],
        )


def test_sparse_projected_ce_is_invariant_to_padding_batching_and_repeated_position_count():
    base = torch.tensor([[[0.2, -0.4, 0.7], [0.1, 0.3, -0.2]]], dtype=torch.float64)
    row = _loss_row(0, (1, 2), (0.4, 0.6))
    single = sparse_projected_conditional_cross_entropy(base[:, :1], [_loss_target(row)])
    padded = torch.cat((base[:, :1], torch.tensor([[[100.0, -100.0, 50.0]]])), dim=1)
    padded_loss = sparse_projected_conditional_cross_entropy(padded, [_loss_target(row)])
    repeated = sparse_projected_conditional_cross_entropy(
        base,
        [_loss_target(row, _loss_row(1, (1, 2), (0.4, 0.6)))],
    )
    expected_repeated = (
        single
        + sparse_projected_conditional_cross_entropy(
            base[:, 1:2], [_loss_target(_loss_row(0, (1, 2), (0.4, 0.6)))]
        )
    ) / 2

    torch.testing.assert_close(single, padded_loss, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected_repeated, rtol=0, atol=1e-15)
    batched = sparse_projected_conditional_cross_entropy(
        torch.cat((base[:, :1], base[:, :1]), dim=0),
        [_loss_target(row), _loss_target(row)],
    )
    torch.testing.assert_close(batched, single, rtol=0, atol=0)


def test_sparse_projected_ce_empty_target_halves_loss_and_eligible_gradient():
    eligible_logits = torch.tensor([[[0.2, -0.4, 0.7]]], dtype=torch.float64, requires_grad=True)
    row = _loss_row(0, (1, 2), (0.4, 0.6))
    eligible_loss = sparse_projected_conditional_cross_entropy(eligible_logits, [_loss_target(row)])
    eligible_loss.backward()
    assert eligible_logits.grad is not None

    mixed_logits = torch.tensor(
        [[[0.2, -0.4, 0.7]], [[9.0, -9.0, 3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    empty = ProjectedTarget(
        input_ids=(1,),
        positions=(),
        rows=(),
        visible_position_count=0,
        eligible_row_count=0,
        drop_counts=ProjectionDropCounts(),
    )
    mixed_loss = sparse_projected_conditional_cross_entropy(
        mixed_logits, [_loss_target(row), empty]
    )
    mixed_loss.backward()
    assert mixed_logits.grad is not None

    torch.testing.assert_close(mixed_loss, eligible_loss.detach() / 2, rtol=0, atol=0)
    torch.testing.assert_close(mixed_logits.grad[0], eligible_logits.grad[0] / 2, rtol=0, atol=0)
    torch.testing.assert_close(
        mixed_logits.grad[1], torch.zeros_like(mixed_logits.grad[1]), rtol=0, atol=0
    )


def test_sparse_projected_ce_empty_fp16_support_uses_bounded_finite_graph_zero():
    logits = torch.full(
        (2, 3, 5),
        torch.finfo(torch.float16).max,
        dtype=torch.float16,
        requires_grad=True,
    )
    empty = ProjectedTarget(
        input_ids=(1,),
        positions=(),
        rows=(),
        visible_position_count=0,
        eligible_row_count=0,
        drop_counts=ProjectionDropCounts(),
    )

    loss = sparse_projected_conditional_cross_entropy(logits, [empty, empty])

    assert loss.item() == 0.0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_sparse_projected_ce_empty_batch_support_is_graph_attached_exact_zero():
    logits = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    empty = ProjectedTarget(
        input_ids=(1,),
        positions=(),
        rows=(),
        visible_position_count=0,
        eligible_row_count=0,
        drop_counts=ProjectionDropCounts(),
    )
    loss = sparse_projected_conditional_cross_entropy(logits, [empty, empty])

    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert logits.grad is not None
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
