from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import os
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

if os.environ.get("FLASH_TEST_PINNED_VERL") != "1":
    pytest.skip("requires the exact pinned Verl source and interpreter", allow_module_level=True)

from verl.base_config import BaseConfig
from verl.workers.config import model as verl_model_config
from verl.workers.config.model import HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

from flash.engine.worker.backend_common import VERL_REQUIREMENT_URL
from flash.engine.worker.train.core.child import runtime as child_runtime

_PINNED_COMMIT = "32d6200de81dc484893baf8b9cf30297ebe7fa49"
_PINNED_BLOBS = {
    "base_config.py": (BaseConfig, "f425dd1464b0f13c83a0944249cd84d55903f120"),
    "model.py": (HFModelConfig, "95814663dcaa91f7e2984b1d4c39ca042712c485"),
    "transformer_impl.py": (FSDPEngine, "2c66c6a3fad912d674bcf5f0a6454743725f33cd"),
}


class _FakeLinear:
    pass


class _FakeModule:
    def named_modules(self):
        return iter(
            [
                ("", object()),
                ("module.model.language_model", object()),
                ("module.model.language_model.layers.0.self_attn.q_proj", _FakeLinear()),
                ("module.model.language_model.layers.0.mlp.down_proj", _FakeLinear()),
                ("module.model.visual.blocks.0.attn.proj", _FakeLinear()),
            ]
        )


def _git_blob_sha(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(
        f"blob {len(payload)}\0".encode() + payload, usedforsecurity=False
    ).hexdigest()


def _build_model_config(monkeypatch, fused_targets, override_config):
    monkeypatch.setattr(verl_model_config, "import_external_libs", lambda _value: None)
    monkeypatch.setattr(
        verl_model_config,
        "copy_to_local",
        lambda path, use_shm=False: path,
    )
    monkeypatch.setattr(
        verl_model_config,
        "hf_tokenizer",
        lambda path, trust_remote_code=False: SimpleNamespace(
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            chat_template="template",
        ),
    )
    monkeypatch.setattr(
        verl_model_config,
        "hf_processor",
        lambda path, trust_remote_code=False: SimpleNamespace(chat_template="template"),
    )
    monkeypatch.setattr(
        verl_model_config,
        "get_generation_config",
        lambda path, trust_remote_code=False: SimpleNamespace(source=path),
    )
    monkeypatch.setattr(
        verl_model_config.AutoConfig,
        "from_pretrained",
        lambda path, **kwargs: SimpleNamespace(
            architectures=["PinnedNoGpuModel"],
            tie_word_embeddings=False,
            model_type="pinned_no_gpu",
            source=path,
        ),
    )
    monkeypatch.setattr(verl_model_config, "update_model_config", lambda config, **kwargs: None)
    return HFModelConfig(
        path="/models/pinned",
        lora_rank=8,
        lora_alpha=16,
        target_modules="all-linear",
        target_parameters=fused_targets,
        exclude_modules=None,
        lora_adapter_path=None,
        override_config=override_config,
    )


def test_exact_pinned_verl_frozen_config_is_replaced_without_gpu_or_field_loss(monkeypatch):
    import torch

    assert importlib.metadata.version("verl") == "0.8.0"
    assert VERL_REQUIREMENT_URL.endswith("@" + _PINNED_COMMIT)
    assert torch.cuda.is_available() is False
    for symbol, expected in _PINNED_BLOBS.values():
        source = inspect.getsourcefile(symbol)
        assert source is not None
        assert _git_blob_sha(Path(source)) == expected

    calls = []

    def original_builder(self, module):
        calls.append(self.model_config)
        return SimpleNamespace(
            base_model=SimpleNamespace(targeted_module_names=list(self.model_config.target_modules))
        )

    monkeypatch.setattr(FSDPEngine, "_build_lora_module", original_builder)
    monkeypatch.setattr(child_runtime, "_text_lora_module_types", lambda: (_FakeLinear,))
    child_runtime.install_text_lora_targeting("model.language_model")

    fused_targets = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    override_config = {"sentinel": object()}
    original = _build_model_config(monkeypatch, fused_targets, override_config)
    original_values = {item.name: getattr(original, item.name) for item in fields(original)}
    with pytest.raises(FrozenInstanceError, match="target_modules"):
        original.target_modules = ["would-reproduce-the-live-failure"]

    engine = FSDPEngine.__new__(FSDPEngine)
    engine.model_config = original
    result = engine._build_lora_module(_FakeModule())

    replacement = calls[-1]
    expected_targets = [
        "module.model.language_model.layers.0.mlp.down_proj",
        "module.model.language_model.layers.0.self_attn.q_proj",
    ]
    assert replacement is not original
    assert type(replacement) is type(original)
    assert replacement.target_modules == expected_targets
    assert result.base_model.targeted_module_names == expected_targets
    assert replacement.target_parameters is fused_targets
    assert replacement.override_config is override_config
    assert engine.model_config is original
    assert {item.name: getattr(original, item.name) for item in fields(original)} == original_values
    for item in fields(original):
        if item.name != "target_modules":
            assert getattr(replacement, item.name) == getattr(original, item.name)
    with pytest.raises(FrozenInstanceError, match="target_modules"):
        replacement.target_modules = []
