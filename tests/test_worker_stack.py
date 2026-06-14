"""Worker stack selection + TRL config compat + LoRA exclusion unit tests (CPU-only)."""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest

from autoslm.flash.train import (
    WORKER_DEPS_LEGACY,
    WORKER_DEPS_MODERN,
    resolve_worker_deps,
)


def test_resolve_worker_deps_default(monkeypatch):
    monkeypatch.delenv("AUTOSLM_WORKER_DEPS", raising=False)
    monkeypatch.delenv("AUTOSLM_WORKER_STACK", raising=False)
    # The modern stack is the validated default (bench/results/phase1 matrix).
    assert resolve_worker_deps() == WORKER_DEPS_MODERN


@pytest.mark.parametrize(
    ("stack", "expected"),
    [
        ("legacy", WORKER_DEPS_LEGACY),
        ("modern", WORKER_DEPS_MODERN),
        ("MODERN", WORKER_DEPS_MODERN),
    ],
)
def test_resolve_worker_deps_named_stack(monkeypatch, stack, expected):
    monkeypatch.delenv("AUTOSLM_WORKER_DEPS", raising=False)
    monkeypatch.setenv("AUTOSLM_WORKER_STACK", stack)
    assert resolve_worker_deps() == expected


def test_resolve_worker_deps_explicit_list_wins(monkeypatch):
    # Whitespace-separated; a comma is part of a PEP 440 range, not a delimiter.
    monkeypatch.setenv("AUTOSLM_WORKER_DEPS", "torch==2.99  vllm==9.9.9   transformers>=5.6,<5.11")
    monkeypatch.setenv("AUTOSLM_WORKER_STACK", "modern")
    assert resolve_worker_deps() == ["torch==2.99", "vllm==9.9.9", "transformers>=5.6,<5.11"]


def test_resolve_worker_deps_json_list_supports_comma_specs(monkeypatch):
    monkeypatch.setenv(
        "AUTOSLM_WORKER_DEPS", '["torch==2.10.0", "transformers>=5.6,<5.11", "fla==0.5.0"]'
    )
    assert resolve_worker_deps() == ["torch==2.10.0", "transformers>=5.6,<5.11", "fla==0.5.0"]


def test_modern_stack_pins_qwen35_capable_versions():
    joined = " ".join(WORKER_DEPS_MODERN)
    assert "vllm==0.19" in joined  # first transformers-5-compatible vllm line
    assert "transformers>=5" in joined  # qwen3_5 model types need transformers 5.x
    assert "trl>=1.5" in joined  # DistillationTrainer + colocate default
    assert "bitsandbytes" in joined  # QLoRA tier for the 35B-A3B MoE


# ---------------------------------------------------------------------------
# supported_config_kwargs (worker module imports lazily heavy deps; safe on CPU)
# ---------------------------------------------------------------------------
def _import_worker(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("autoslm.engine.worker", None)
    import autoslm.engine.worker as worker

    return worker


def test_supported_config_kwargs_filters_unknown(monkeypatch):
    worker = _import_worker(monkeypatch)

    @dataclasses.dataclass
    class FakeConfig:
        output_dir: str = "x"
        learning_rate: float = 1e-5

    kept = worker.supported_config_kwargs(
        FakeConfig, {"output_dir": "y", "learning_rate": 3e-4, "max_prompt_length": 512}
    )
    assert kept == {"output_dir": "y", "learning_rate": 3e-4}


# ---------------------------------------------------------------------------
# lora_exclude_modules: vision tower excluded for qwen3_5*, none for text models
# ---------------------------------------------------------------------------
def _fake_transformers(monkeypatch, model_type: str):
    fake_cfg = types.SimpleNamespace(model_type=model_type)
    fake_auto = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: fake_cfg,
    )
    fake_mod = types.ModuleType("transformers")
    fake_mod.AutoConfig = fake_auto
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)


def test_lora_exclude_modules_qwen35(monkeypatch):
    import re

    worker = _import_worker(monkeypatch)
    _fake_transformers(monkeypatch, "qwen3_5")
    excl = worker.lora_exclude_modules("Qwen/Qwen3.5-4B")
    assert excl is not None
    assert "visual" in excl
    # peft applies exclude_modules regex with fullmatch on the module path: leaf
    # modules under the vision tower MUST match (the earlier suffix-list form didn't,
    # which let LoRA onto visual.* and broke vLLM adapter loading).
    assert re.fullmatch(excl, "visual.blocks.0.attn.qkv")
    assert re.fullmatch(excl, "model.visual.blocks.3.mlp.linear_fc1")
    assert not re.fullmatch(excl, "model.layers.0.self_attn.q_proj")


def test_lora_exclude_modules_text_model(monkeypatch):
    worker = _import_worker(monkeypatch)
    _fake_transformers(monkeypatch, "llama")
    assert worker.lora_exclude_modules("openbmb/MiniCPM5-1B") is None
