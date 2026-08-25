from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

if os.environ.get("FLASH_TEST_PINNED_VERL") != "1":
    pytest.skip("requires the exact pinned Verl source and interpreter", allow_module_level=True)

from verl.base_config import BaseConfig
from verl.workers.config import model as verl_model_config
from verl.workers.config.model import HFModelConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

from flash.engine.worker.train.core.child import runtime as child_runtime
from flash.engine.worker.train.entry.backend_common import VERL_REQUIREMENT_URL

_PINNED_COMMIT = "f71a02ddb32a9c6a6915f7519bda6dede92e9dd0"
_PINNED_BLOBS = {
    "base_config.py": (BaseConfig, "f425dd1464b0f13c83a0944249cd84d55903f120"),
    "model.py": (HFModelConfig, "95814663dcaa91f7e2984b1d4c39ca042712c485"),
    "transformer_impl.py": (FSDPEngine, "2c66c6a3fad912d674bcf5f0a6454743725f33cd"),
}


class _FakeLinear:
    pass


class _FakeModule:
    def __init__(self, wrapper="module."):
        self.wrapper = wrapper

    def named_modules(self):
        root = f"{self.wrapper}model.language_model"
        return iter(
            [
                ("", object()),
                (root, object()),
                (f"{root}.layers.0.self_attn.q_proj", _FakeLinear()),
                (f"{root}.layers.0.mlp.down_proj", _FakeLinear()),
                (f"{self.wrapper}model.visual.blocks.0.attn.proj", _FakeLinear()),
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


def test_exact_pinned_verl_builder_isolates_same_engine_concurrent_calls(monkeypatch):
    barrier = Barrier(2)
    live_engine = []
    isolated_engines = []

    def original_builder(self, module):
        isolated_engines.append(self)
        assert self is not live_engine[0]
        assert live_engine[0].model_config is original
        module.attached_targets = tuple(self.model_config.target_modules)
        barrier.wait(timeout=5)
        assert live_engine[0].model_config is original
        return SimpleNamespace(
            base_model=SimpleNamespace(targeted_module_names=list(module.attached_targets))
        )

    monkeypatch.setattr(FSDPEngine, "_build_lora_module", original_builder)
    monkeypatch.setattr(child_runtime, "_text_lora_module_types", lambda: (_FakeLinear,))
    child_runtime.install_text_lora_targeting("model.language_model")
    original = _build_model_config(monkeypatch, [], {"concurrent": True})
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.model_config = original
    live_engine.append(engine)
    modules = [_FakeModule("module."), _FakeModule("_orig_mod.")]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(engine._build_lora_module, modules))

    assert engine.model_config is original
    assert len({id(item) for item in isolated_engines}) == 2
    assert [tuple(result.base_model.targeted_module_names) for result in results] == [
        module.attached_targets for module in modules
    ]
    assert modules[0].attached_targets[0].startswith("module.")
    assert modules[1].attached_targets[0].startswith("_orig_mod.")


def test_exact_pinned_verl_builder_isolates_same_engine_reentry_and_exceptions(monkeypatch):
    live_engine = []
    isolated_engines = []
    nested_module = _FakeModule("module.")

    def original_builder(self, module):
        isolated_engines.append(self)
        assert self is not live_engine[0]
        assert live_engine[0].model_config is original
        if len(isolated_engines) == 1:
            live_engine[0]._build_lora_module(nested_module)
        if module.wrapper == "raise.":
            raise RuntimeError("pinned builder failed")
        module.attached_targets = tuple(self.model_config.target_modules)
        return SimpleNamespace(
            base_model=SimpleNamespace(targeted_module_names=list(module.attached_targets))
        )

    monkeypatch.setattr(FSDPEngine, "_build_lora_module", original_builder)
    monkeypatch.setattr(child_runtime, "_text_lora_module_types", lambda: (_FakeLinear,))
    child_runtime.install_text_lora_targeting("model.language_model")
    original = _build_model_config(monkeypatch, [], {"reentrant": True})
    engine = FSDPEngine.__new__(FSDPEngine)
    engine.model_config = original
    live_engine.append(engine)
    outer_module = _FakeModule("_orig_mod.")

    result = engine._build_lora_module(outer_module)

    assert engine.model_config is original
    assert len({id(item) for item in isolated_engines}) == 2
    assert tuple(result.base_model.targeted_module_names) == outer_module.attached_targets
    assert nested_module.attached_targets[0].startswith("module.")

    with pytest.raises(RuntimeError, match="pinned builder failed"):
        engine._build_lora_module(_FakeModule("raise."))
    assert engine.model_config is original
    assert isolated_engines[-1] is not engine
