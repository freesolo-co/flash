"""CPU coverage for the optional Chalk GRPO selected-token path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from flash.engine.worker.grpo_fused import (
    ChalkGrpoRuntime,
    _chunked_entropy,
    make_chalk_grpo_trainer,
    probe_chalk_grpo,
    resolve_chalk_grpo_trainer,
)


def _selective_log_softmax(hidden, weight, target_ids, *, bias=None, temperature=1.0):
    logits = hidden @ weight.t()
    if bias is not None:
        logits = logits + bias
    return torch.log_softmax(logits.float() / temperature, dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)


def test_probe_chalk_grpo_accepts_functional_cpu_op_without_advancing_rng():
    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state()

    runtime = probe_chalk_grpo(importer=lambda: _selective_log_softmax, torch_module=torch)

    assert runtime is not None
    assert runtime.selective_log_softmax is _selective_log_softmax
    assert torch.equal(torch.random.get_rng_state(), rng_state)


@pytest.mark.parametrize(
    "importer",
    [
        lambda: (_ for _ in ()).throw(ImportError("chalk absent")),
        lambda: (lambda hidden, weight, target_ids, **kwargs: torch.zeros_like(target_ids)),
    ],
)
def test_probe_chalk_grpo_rejects_absent_or_broken_op(importer):
    assert probe_chalk_grpo(importer=importer, torch_module=torch) is None


def test_resolve_chalk_grpo_trainer_keeps_exact_base_fallback(monkeypatch):
    class BaseTrainer:
        pass

    monkeypatch.setattr("flash.engine.worker.grpo_fused.probe_chalk_grpo", lambda: None)

    trainer_class, fused = resolve_chalk_grpo_trainer(BaseTrainer, "Qwen/Qwen3.5-0.8B")

    assert fused is False
    assert trainer_class is BaseTrainer


def test_resolve_chalk_grpo_trainer_rejects_unvalidated_model_before_probe(monkeypatch):
    class BaseTrainer:
        pass

    def unexpected_probe():
        raise AssertionError("unsupported models must not probe or engage the fused path")

    monkeypatch.setattr("flash.engine.worker.grpo_fused.probe_chalk_grpo", unexpected_probe)

    trainer_class, fused = resolve_chalk_grpo_trainer(BaseTrainer, "google/gemma-3-1b-it")

    assert fused is False
    assert trainer_class is BaseTrainer


def test_resolve_chalk_grpo_trainer_rejects_pinned_revision_before_probe(monkeypatch):
    class BaseTrainer:
        pass

    def unexpected_probe():
        raise AssertionError("pinned revisions must not probe or engage the fused path")

    monkeypatch.setattr("flash.engine.worker.grpo_fused.probe_chalk_grpo", unexpected_probe)

    trainer_class, fused = resolve_chalk_grpo_trainer(
        BaseTrainer,
        "Qwen/Qwen3.5-0.8B",
        model_revision="0123456789abcdef",
    )

    assert fused is False
    assert trainer_class is BaseTrainer


def test_resolve_chalk_grpo_trainer_engages_validated_runtime(monkeypatch):
    class BaseTrainer:
        pass

    runtime = ChalkGrpoRuntime(_selective_log_softmax)
    monkeypatch.setattr("flash.engine.worker.grpo_fused.probe_chalk_grpo", lambda: runtime)

    trainer_class, fused = resolve_chalk_grpo_trainer(BaseTrainer, "Qwen/Qwen3.5-0.8B")

    assert fused is True
    assert issubclass(trainer_class, BaseTrainer)
    assert trainer_class._flash_chalk_runtime is runtime


def test_fused_trainer_chunks_persistent_and_runtime_fallbacks_to_one_completion():
    class BaseTrainer:
        def _get_per_token_logps_and_entropies(self, *args, **kwargs):
            self.fallback_batch_sizes.append(args[4])
            return "fallback"

    trainer_class = make_chalk_grpo_trainer(
        BaseTrainer, ChalkGrpoRuntime(_selective_log_softmax)
    )
    trainer = trainer_class.__new__(trainer_class)
    trainer.fallback_batch_sizes = []
    trainer.selected_batch_sizes = []
    trainer._flash_chalk_failed = True

    assert trainer._get_per_token_logps_and_entropies(None, None, None, 4, 8) == "fallback"

    trainer._flash_chalk_failed = False

    def fail_selected_path(*args, **kwargs):
        trainer.selected_batch_sizes.append(args[4])
        raise RuntimeError("shape-specific chalk failure")

    trainer._flash_selected_logps_and_entropies = fail_selected_path

    assert trainer._get_per_token_logps_and_entropies(None, None, None, 4, 8) == "fallback"
    assert trainer._flash_chalk_failed is True
    assert trainer.selected_batch_sizes == [8]
    assert trainer.fallback_batch_sizes == [1, 1]


def test_fused_trainer_falls_back_before_bypassing_multimodal_preprocessing():
    class BaseTrainer:
        def _get_per_token_logps_and_entropies(self, *args, **kwargs):
            self.fallback_batch_sizes.append(args[4])
            return "fallback"

    trainer_class = make_chalk_grpo_trainer(
        BaseTrainer, ChalkGrpoRuntime(_selective_log_softmax)
    )
    trainer = trainer_class.__new__(trainer_class)
    trainer.fallback_batch_sizes = []
    trainer._flash_chalk_failed = False

    result = trainer._get_per_token_logps_and_entropies(
        None,
        None,
        None,
        4,
        8,
        pixel_values=object(),
    )

    assert result == "fallback"
    assert trainer._flash_chalk_failed is True
    assert trainer.fallback_batch_sizes == [1]


def test_chunked_entropy_matches_full_logits():
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, 6)
    weight = torch.randn(13, 6)
    bias = torch.randn(13)
    temperature = 0.8

    logits = (hidden @ weight.t() + bias) / temperature
    probabilities = torch.softmax(logits.float(), dim=-1)
    expected = -(probabilities * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
    actual = _chunked_entropy(
        hidden,
        weight,
        bias,
        temperature,
        seq_chunk_size=3,
        vocab_chunk_size=4,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_fused_trainer_delegates_masks_and_normalization_to_base_compute_loss():
    class BaseTrainer:
        def _compute_loss(self, model, inputs):
            self.seen_model = model
            self.seen_inputs = inputs
            mask = inputs["completion_mask"] * inputs["tool_mask"]
            return (inputs["per_token_loss"] * mask).sum() / (
                mask.size(0) * self.max_completion_length
            )

    trainer_class = make_chalk_grpo_trainer(
        BaseTrainer, ChalkGrpoRuntime(_selective_log_softmax)
    )
    trainer = trainer_class.__new__(trainer_class)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model: model)
    trainer._flash_chalk_failed = False
    trainer._flash_forward_redirection = (
        lambda wrapper, original, method, *args, **kwargs: method(*args, **kwargs)
    )
    trainer.max_completion_length = 6
    model = object()
    inputs = {
        "completion_mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
        "tool_mask": torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
        "per_token_loss": torch.tensor([[2.0, 50.0, 70.0], [3.0, 5.0, 90.0]]),
    }

    loss = trainer.compute_loss(model, inputs)

    assert "_compute_loss" not in trainer_class.__dict__
    assert trainer.seen_model is model
    assert trainer.seen_inputs is inputs
    assert loss.item() == pytest.approx((2.0 + 3.0 + 5.0) / (2 * 6))
