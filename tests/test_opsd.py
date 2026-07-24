"""CPU-only mocked smoke coverage for on-policy self-distillation."""

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace
from typing import ClassVar

import pytest


def _opsd_module_for_knobs(monkeypatch):
    from flash.engine.worker import opsd as opsd_mod
    from flash.spec import JobSpec

    monkeypatch.setattr(
        opsd_mod,
        "_w",
        SimpleNamespace(JOB_SPEC=JobSpec(), THINKING=False),
    )
    return opsd_mod


def test_opsd_knobs_use_paper_recipe_defaults(monkeypatch):
    # opsd hardcodes the released recipe (siyan-zhao/OPSD, scripts/run_opsd_4b.sh): forward kl,
    # lr 5e-6, gradient clip 0.1, softmax/rollout temperature 1.1, top_p 0.95, per-vocab clip 0.05.
    # a regression on any of these silently changes training away from the paper.
    opsd_mod = _opsd_module_for_knobs(monkeypatch)

    knobs = opsd_mod._resolve_opsd_knobs()

    assert knobs.learning_rate == pytest.approx(5e-6)
    assert knobs.temperature == pytest.approx(1.1)
    assert knobs.top_p == pytest.approx(0.95)
    assert pytest.approx(5e-6) == opsd_mod._OPSD_LEARNING_RATE
    assert pytest.approx(0.1) == opsd_mod._OPSD_MAX_GRAD_NORM
    assert pytest.approx(1.1) == opsd_mod._OPSD_TEMPERATURE
    assert pytest.approx(0.05) == opsd_mod._OPSD_CLIP_TAU


def test_opsd_forward_kl_pointwise_clip_can_go_negative_with_student_only_gradient():
    # paper-faithful upper-only clip (clamp_max) caps the large positive vocab summand at tau while
    # keeping the negative summand, so the summed loss is negative by design. asymmetric probs mean
    # a forward/reverse branch swap would change the value and fail this test.
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opsd import _opsd_kl_loss

    student = torch.tensor(
        [[[math.log(0.2), math.log(0.8)], [float("-inf"), float("-inf")]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[math.log(0.6), math.log(0.4)], [float("-inf"), float("-inf")]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, False]])

    loss = _opsd_kl_loss(student, teacher, mask, temperature=1.0, clip_tau=0.05)

    # coord0 = 0.6*log(0.6/0.2) clamped to tau=0.05; coord1 = 0.4*log(0.4/0.8) kept negative
    expected = 0.05 + 0.4 * math.log(0.5)
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert loss.item() < 0  # intended: upper-only clipping lets the loss go negative
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_opsd_forward_completion_logits_slices_shifted_completion_positions():
    # teacher-forcing: the logits that predict completion token i live at sequence index
    # len(prefix)-1+i. distinct-per-position logits catch an off-by-one in the slice bounds.
    torch = pytest.importorskip("torch")
    from flash.engine.worker.opsd import _forward_completion_logits

    vocab = 3

    class _PosModel:
        def __call__(self, input_ids, attention_mask=None, position_ids=None):
            seqlen = int(input_ids.shape[1])
            rows = torch.stack(
                [torch.tensor([float(p), float(p) + 0.1, float(p) + 0.2]) for p in range(seqlen)]
            )
            return SimpleNamespace(logits=rows.view(1, seqlen, vocab))

    prefix = [11, 12, 13, 14]
    completion = [21, 22]
    out = _forward_completion_logits(_PosModel(), prefix, completion, "cpu")

    assert tuple(out.shape) == (1, len(completion), vocab)
    # returned rows must be sequence positions [len(prefix)-1, len(prefix)] == [3, 4]
    assert out[0, 0, 0].item() == pytest.approx(3.0)
    assert out[0, 1, 0].item() == pytest.approx(4.0)


def _patch_opsd_run(
    monkeypatch, *, gold="worked solution: add one and one to get two", max_steps=1
):
    torch = pytest.importorskip("torch")
    import flash.engine.worker.hf as hf_mod
    from flash.engine.worker import opsd as opsd_mod
    from flash.engine.worker.opd_vllm import OpdVllmOutput

    class _Tokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ):
            return "\n".join(str(message.get("content", "")) for message in messages)

        def __call__(self, text, add_special_tokens=False):
            ids = [1, 2, 5] if "=== Reference Solution Begin ===" in text else [1, 2]
            return SimpleNamespace(input_ids=ids)

        def decode(self, ids, skip_special_tokens=True):
            return "".join({3: "a", 4: "b"}.get(int(token), "") for token in ids)

        def save_pretrained(self, path):
            return None

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(
                torch.tensor([0.3, -0.2, 0.1, 0.0, -0.1]), requires_grad=False
            )
            self.w = torch.nn.Parameter(torch.tensor([0.2, -0.1, 0.05, -0.05, 0.1]))
            self.config = SimpleNamespace(use_cache=True, eos_token_id=9)
            self.generation_config = SimpleNamespace(eos_token_id=9)
            self.adapter_enabled = True
            self.disable_entries = 0
            self.forward_events = []

        @contextlib.contextmanager
        def disable_adapter(self):
            self.disable_entries += 1
            previous = self.adapter_enabled
            self.adapter_enabled = False
            try:
                yield
            finally:
                self.adapter_enabled = previous

        def forward(self, input_ids, attention_mask=None, position_ids=None):
            logits = self.base
            if self.adapter_enabled:
                logits = logits + self.w
            logits = logits.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
            self.forward_events.append(
                (self.adapter_enabled, torch.is_grad_enabled(), logits.requires_grad)
            )
            return SimpleNamespace(logits=logits)

        def save_pretrained(self, path):
            return None

    class _FakeRollout:
        instances: ClassVar[list] = []

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.sync_count = 0
            self.closed = False
            _FakeRollout.instances.append(self)

        def sync_from_model(self, model):
            self.model = model
            self.sync_count += 1

        def generate(
            self,
            prompt_ids_batch,
            *,
            max_tokens,
            request_seeds=None,
            multi_modal_data_batch=None,
        ):
            return [
                OpdVllmOutput([3, 4], "ab", finish_reason="stop")
                for _prompt_ids in prompt_ids_batch
            ]

        def close(self):
            self.closed = True

    class _Env:
        is_tool_env = False
        multi_turn = False

        def dataset(self):
            return [{"input": "what is one plus one?", "output": gold}]

        def prompt_messages(self, example):
            return [{"role": "user", "content": example["input"]}]

        def sft_completion(self, example):
            return [{"role": "assistant", "content": example["output"]}]

    model = _Model()
    metadata = {}
    fake_w = SimpleNamespace(
        require_active_env=lambda: _Env(),
        JOB_SPEC=SimpleNamespace(
            model="fake/model",
            model_revision="",
            train=SimpleNamespace(max_examples=0, lora_rank=4),
            gpu=SimpleNamespace(type=None),
        ),
        THINKING=False,
        SEED=7,
        heartbeat=lambda stage, **kwargs: None,
        prefetch_model=lambda model_id, **kwargs: 0.0,
        write_train_meta=lambda **kwargs: metadata.update(kwargs),
    )
    monkeypatch.setattr(opsd_mod, "_w", fake_w)
    monkeypatch.setattr(
        opsd_mod,
        "_resolve_opsd_knobs",
        lambda: opsd_mod.OpsdKnobs(
            epochs=1,
            learning_rate=0.1,
            temperature=1.0,
            top_p=1.0,
            max_completion=4,
            prompts_per_step=1,
            max_steps=max_steps,
            max_length=32,
            stop_sequences=(),
        ),
    )
    monkeypatch.setattr(opsd_mod, "_student_model", lambda *args, **kwargs: (model, "fake/model"))
    monkeypatch.setattr(opsd_mod, "OpdVllmRolloutEngine", _FakeRollout)
    monkeypatch.setattr(
        opsd_mod,
        "_opd_vllm_kwargs",
        lambda *args, **kwargs: {
            "gpu_memory_utilization": 0.1,
            "kv_cache_dtype": None,
            "max_num_seqs": None,
            "max_num_batched_tokens": None,
            "rollout_batch_size": None,
            "attention_backend": None,
            "mm_encoder_attn_backend": None,
            "enforce_eager": None,
            "compilation_config": None,
        },
    )
    monkeypatch.setattr(opsd_mod, "_save_adapter", lambda *args, **kwargs: None)
    monkeypatch.setattr(opsd_mod, "_publish_opsd_deployable", lambda *args, **kwargs: None)
    monkeypatch.setattr(opsd_mod, "wait_for_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(opsd_mod, "setup_perf_backends", lambda: None)
    monkeypatch.setattr(opsd_mod, "optimal_attn_impl", lambda: None)
    monkeypatch.setattr(opsd_mod, "grad_checkpointing_on", lambda *args, **kwargs: False)
    monkeypatch.setattr(opsd_mod, "gpu_diagnostics", lambda *args, **kwargs: {})
    monkeypatch.setattr(opsd_mod, "free_gpu", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_mod, "load_tokenizer", lambda *args, **kwargs: _Tokenizer())
    monkeypatch.setattr(hf_mod, "model_revision_kwargs", lambda revision: {})
    return opsd_mod, model, metadata, _FakeRollout


def test_opsd_mocked_run_uses_adapter_off_teacher_and_updates_only_lora(monkeypatch):
    torch = pytest.importorskip("torch")
    opsd_mod, model, metadata, fake_rollout = _patch_opsd_run(monkeypatch)
    before_lora = model.w.detach().clone()
    before_base = model.base.detach().clone()

    opsd_mod.run_opsd()

    assert model.disable_entries == 1
    teacher_events = [event for event in model.forward_events if event[0] is False]
    assert teacher_events == [(False, False, False)]
    assert any(adapter_on and grad_on and requires_grad for adapter_on, grad_on, requires_grad in model.forward_events)
    assert not torch.equal(before_lora, model.w.detach())
    assert torch.equal(before_base, model.base.detach())
    assert metadata["phase"] == "opsd"
    assert metadata["step"] == 1
    assert metadata["notes"]["teacher"] == "frozen_base_adapter_disabled"
    assert metadata["notes"]["objective"] == "forward_kl"
    assert fake_rollout.instances[-1].sync_count == 2
    assert fake_rollout.instances[-1].closed is True


def test_opsd_empty_gold_fails_loudly(monkeypatch):
    opsd_mod, _model, _metadata, _fake_rollout = _patch_opsd_run(monkeypatch, gold="   ")

    with pytest.raises(RuntimeError, match="requires a nonempty gold completion"):
        opsd_mod.run_opsd()


def test_opsd_truncated_step_is_skipped_not_fatal(monkeypatch):
    # a step whose student rollouts all hit the max-token cap (truncated) has no naturally
    # terminated completion to teacher-force, so opsd skips it (opsd_step_skipped heartbeat, no
    # optimizer update) and continues to the next step instead of aborting. long-trace thinking
    # envs emit occasional all-truncate steps, and the paid run must survive them while still
    # training on the steps that do terminate.
    pytest.importorskip("torch")
    from flash.engine.worker.opd_vllm import OpdVllmOutput

    opsd_mod, _model, metadata, fake_rollout = _patch_opsd_run(monkeypatch, max_steps=2)

    class _FirstStepTruncates(fake_rollout):
        calls = 0

        def generate(
            self,
            prompt_ids_batch,
            *,
            max_tokens,
            request_seeds=None,
            multi_modal_data_batch=None,
        ):
            _FirstStepTruncates.calls += 1
            # step 1 truncates (finish_reason="length" -> truncated=True, skipped); step 2 terminates
            finish_reason = "length" if _FirstStepTruncates.calls == 1 else "stop"
            return [
                OpdVllmOutput([3, 4], "ab", finish_reason=finish_reason)
                for _prompt_ids in prompt_ids_batch
            ]

    monkeypatch.setattr(opsd_mod, "OpdVllmRolloutEngine", _FirstStepTruncates)

    events: list[str] = []
    monkeypatch.setattr(opsd_mod._w, "heartbeat", lambda stage, **kwargs: events.append(stage))

    opsd_mod.run_opsd()  # must not raise: the skipped step is tolerated and the next step trains

    assert "opsd_step_skipped" in events  # the all-truncate step was skipped
    assert "opsd_step" in events  # a later terminating step still trained
    # only the terminating step contributed to the loss curve; the skipped step did not
    assert len(metadata["notes"]["loss_curve"]) == 1


def test_opsd_all_steps_skipped_fails_loud(monkeypatch):
    # if every step's student rollouts truncate, no optimizer update ever runs and the adapter is
    # untrained. opsd must fail loud rather than silently publish an untrained adapter, which would
    # otherwise happen on a grossly misconfigured env (e.g. max_completion_tokens far too low).
    pytest.importorskip("torch")
    from flash.engine.worker.opd_vllm import OpdVllmOutput

    opsd_mod, _model, _metadata, fake_rollout = _patch_opsd_run(monkeypatch, max_steps=2)

    class _AlwaysTruncates(fake_rollout):
        def generate(
            self,
            prompt_ids_batch,
            *,
            max_tokens,
            request_seeds=None,
            multi_modal_data_batch=None,
        ):
            return [
                OpdVllmOutput([3, 4], "ab", finish_reason="length")
                for _prompt_ids in prompt_ids_batch
            ]

    monkeypatch.setattr(opsd_mod, "OpdVllmRolloutEngine", _AlwaysTruncates)

    with pytest.raises(RuntimeError, match="trained no step"):
        opsd_mod.run_opsd()
