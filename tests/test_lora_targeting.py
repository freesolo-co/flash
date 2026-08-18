from __future__ import annotations

import json
import re
import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save

from flash.adapters.targets import LoraTargeting, resolve_lora_targeting
from flash.core.catalog import ALGORITHMS, MODELS
from flash.engine.worker.train.core.child import runtime as child_runtime
from flash.engine.worker.train.core.child.runtime import (
    install_text_lora_targeting,
    resolve_text_lora_target_modules,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "qwen35_08b_target_metadata.json"
_CAMPAIGN_MODELS = tuple(MODELS)


def _metadata() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _excluded(targeting: LoraTargeting, module_name: str) -> bool:
    return bool(targeting.exclude_modules and re.fullmatch(targeting.exclude_modules, module_name))


class _FakeLinear:
    pass


class _FakeModule:
    def __init__(self, names: list[tuple[str, object]]):
        self._names = names

    def named_modules(self):
        return iter(self._names)


def _runtime_topology(wrapper: str = "") -> _FakeModule:
    root = f"{wrapper}model.language_model"
    return _FakeModule(
        [
            ("", object()),
            (root, object()),
            (f"{root}.layers.0.self_attn.q_proj", _FakeLinear()),
            (f"{root}.layers.0.mlp.down_proj", _FakeLinear()),
            (f"{root}.mtp.layers.0.proj", _FakeLinear()),
            (f"{wrapper}model.visual.blocks.0.attn.proj", _FakeLinear()),
            (f"{wrapper}model.visual.patch_embed.proj", _FakeLinear()),
            (f"{wrapper}model.multi_modal_projector.linear", _FakeLinear()),
        ]
    )


def test_actual_qwen35_08b_metadata_resolves_the_language_subtree_without_torch():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    metadata = _metadata()
    config = Qwen3_5Config.from_dict(metadata["config"])
    targeting = resolve_lora_targeting(metadata["model_id"], algorithm="sft", multimodal=False)

    assert config.architectures == ["Qwen3_5ForConditionalGeneration"]
    assert config.model_type == "qwen3_5"
    assert config.text_config.model_type == "qwen3_5_text"
    assert config.text_config.mtp_num_hidden_layers == 1
    assert config.vision_config.model_type == "qwen3_5"
    assert not _excluded(targeting, metadata["module_names"]["language_linear"])
    for key in ("visual_linear", "patch_linear", "projector_linear", "mtp_linear"):
        assert _excluded(targeting, metadata["module_names"][key]), key


@pytest.mark.parametrize("wrapper", ["", "module.", "_orig_mod.", "base_model.model."])
def test_runtime_target_resolution_uses_exact_wrapper_qualified_language_paths(wrapper):
    targets = resolve_text_lora_target_modules(
        _runtime_topology(wrapper),
        "model.language_model",
        module_types=(_FakeLinear,),
    )

    assert targets == [
        f"{wrapper}model.language_model.layers.0.mlp.down_proj",
        f"{wrapper}model.language_model.layers.0.self_attn.q_proj",
    ]
    assert not any(
        segment in target
        for target in targets
        for segment in ("visual", "patch_embed", "multi_modal_projector", "mtp")
    )


@pytest.mark.parametrize(
    "names",
    [
        [("", object()), ("model.language_model", object())],
        [("", object()), ("model.text_model.layers.0.q_proj", _FakeLinear())],
    ],
)
def test_runtime_target_resolution_fails_instead_of_returning_an_empty_set(names):
    with pytest.raises(RuntimeError, match=r"no supported language modules|exactly one runtime"):
        resolve_text_lora_target_modules(
            _FakeModule(names),
            "model.language_model",
            module_types=(_FakeLinear,),
        )


def test_runtime_target_resolution_rejects_ambiguous_wrapper_roots():
    names = list(_runtime_topology("module.").named_modules())
    names.extend(
        [
            ("other.model.language_model", object()),
            ("other.model.language_model.layers.0.q_proj", _FakeLinear()),
        ]
    )
    with pytest.raises(RuntimeError, match="exactly one runtime language root"):
        resolve_text_lora_target_modules(
            _FakeModule(names),
            "model.language_model",
            module_types=(_FakeLinear,),
        )


@pytest.mark.parametrize("model_id", _CAMPAIGN_MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_campaign_text_trainer_targets_only_the_cataloged_language_subtree(
    model_id, algorithm
):
    targeting = resolve_lora_targeting(model_id, algorithm=algorithm, multimodal=False)
    prefix = MODELS[model_id].lora_language_prefix

    assert targeting.target_modules == "all-linear"
    assert not _excluded(targeting, f"{prefix}.layers.0.self_attn.q_proj")
    assert resolve_text_lora_target_modules(
        _runtime_topology("module."),
        prefix,
        module_types=(_FakeLinear,),
    ) == [
        "module.model.language_model.layers.0.mlp.down_proj",
        "module.model.language_model.layers.0.self_attn.q_proj",
    ]
    for module_name in (
        "model.visual.blocks.0.attn.proj",
        "model.visual.patch_embed.proj",
        "model.multi_modal_projector.linear",
        "model.vision_tower.blocks.0.proj",
        "model.mtp.layers.0.proj",
    ):
        assert _excluded(targeting, module_name)
    if model_id == "Qwen/Qwen3.6-35B-A3B":
        assert targeting.target_parameters == [
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ]
    else:
        assert targeting.target_parameters is None


@pytest.mark.parametrize("model_id", _CAMPAIGN_MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_campaign_multimodal_trainer_keeps_the_existing_all_linear_surface(
    model_id, algorithm
):
    targeting = resolve_lora_targeting(model_id, algorithm=algorithm, multimodal=True)

    assert targeting.target_modules == "all-linear"
    assert targeting.exclude_modules is None
    if model_id == "Qwen/Qwen3.6-35B-A3B":
        assert targeting.target_parameters == [
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ]
    else:
        assert targeting.target_parameters is None


def _install_fake_verl_engine(monkeypatch):
    calls = []

    class _Engine:
        def __init__(self, config):
            self.model_config = config

        def _build_lora_module(self, module):
            calls.append(
                (
                    self.model_config.target_modules,
                    self.model_config.exclude_modules,
                    self.model_config.lora_adapter_path,
                )
            )
            targets = self.model_config.target_modules
            return SimpleNamespace(
                base_model=SimpleNamespace(
                    targeted_module_names=list(targets) if isinstance(targets, list) else []
                )
            )

    impl = types.ModuleType("verl.workers.engine.fsdp.transformer_impl")
    impl.FSDPEngine = _Engine
    modules = {
        "verl": types.ModuleType("verl"),
        "verl.workers": types.ModuleType("verl.workers"),
        "verl.workers.engine": types.ModuleType("verl.workers.engine"),
        "verl.workers.engine.fsdp": types.ModuleType("verl.workers.engine.fsdp"),
        "verl.workers.engine.fsdp.transformer_impl": impl,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(child_runtime, "_text_lora_module_types", lambda: (_FakeLinear,))
    install_text_lora_targeting("model.language_model")
    return _Engine, calls


def test_runtime_installer_replaces_only_fresh_text_targeting(monkeypatch):
    engine_cls, calls = _install_fake_verl_engine(monkeypatch)
    broken_hydra_regex = r"^(?!model\\.language_model(?:\\.|$)).*$"

    text_config = SimpleNamespace(
        target_modules="all-linear",
        exclude_modules=broken_hydra_regex,
        lora_adapter_path=None,
    )
    engine_cls(text_config)._build_lora_module(_runtime_topology("module."))
    assert calls[-1] == (
        [
            "module.model.language_model.layers.0.mlp.down_proj",
            "module.model.language_model.layers.0.self_attn.q_proj",
        ],
        None,
        None,
    )

    multimodal_config = SimpleNamespace(
        target_modules="all-linear",
        exclude_modules=None,
        lora_adapter_path=None,
    )
    engine_cls(multimodal_config)._build_lora_module(_runtime_topology())
    assert calls[-1] == ("all-linear", None, None)

    warmstart_config = SimpleNamespace(
        target_modules="all-linear",
        exclude_modules=broken_hydra_regex,
        lora_adapter_path="/adapter",
    )
    engine_cls(warmstart_config)._build_lora_module(_runtime_topology())
    assert calls[-1] == ("all-linear", broken_hydra_regex, "/adapter")


def test_sft_and_opd_child_configs_install_text_targeting_only_for_text_jobs(tmp_path):
    from flash.engine.worker import sft_train_runner
    from flash.engine.worker.train.opd.overrides import _build_opd_plugin_config

    shim_dir = tmp_path / "sft-shim"
    shim_dir.mkdir()
    _marker, expected, raw_sft = sft_train_runner._write_sft_child_shims(
        SimpleNamespace(model_id="Qwen/Qwen3.5-0.8B", save_at_steps=()),
        SimpleNamespace(
            update_horizon=1,
            reentrant_gradient_checkpointing=False,
            exclude_modules="text-only",
        ),
        shim_dir=str(shim_dir),
        custom_dataset_path=str(shim_dir / "dataset.py"),
        seed=42,
        loggers=[],
        gdn_reset_arch=None,
    )
    assert child_runtime.TEXT_LORA_TARGET_SHIM in expected
    assert json.loads(raw_sft)["lora_language_prefix"] == "model.language_model"

    raw_opd = _build_opd_plugin_config(
        shim_dir=str(tmp_path),
        save_at_steps=(),
        total_steps=1,
        gdn_model_type=None,
        loggers=[],
        lora_language_prefix="model.language_model",
    )
    assert json.loads(raw_opd)["lora_language_prefix"] == "model.language_model"


def test_exact_verl_hf_model_reproduces_old_peft_failure_and_attaches_new_targets():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    hydra = pytest.importorskip("hydra.core.override_parser.overrides_parser")
    from transformers import AutoModelForImageTextToText
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    from flash.engine.worker.train.sft.config import _hydra_val

    config = Qwen3_5Config.from_dict(_metadata()["config"])
    intended_regex = resolve_lora_targeting(
        "Qwen/Qwen3.5-0.8B", algorithm="sft", multimodal=False
    ).exclude_modules
    override = f"model.exclude_modules={_hydra_val(intended_regex)}"
    broken_hydra_regex = hydra.OverridesParser.create().parse_override(override).value()
    assert broken_hydra_regex == r"^(?!model\\.language_model(?:\\.|$)).*$"
    with torch.device("meta"):
        broken_model = AutoModelForImageTextToText.from_config(config)
    assert type(broken_model).__name__ == "Qwen3_5ForConditionalGeneration"
    with pytest.raises(ValueError, match="All modules were excluded"):
        peft.get_peft_model(
            broken_model,
            peft.LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=16,
                target_modules="all-linear",
                exclude_modules=broken_hydra_regex,
                bias="none",
            ),
        )

    with torch.device("meta"):
        repaired_model = AutoModelForImageTextToText.from_config(config)
    targets = resolve_text_lora_target_modules(repaired_model, "model.language_model")
    attached = peft.get_peft_model(
        repaired_model,
        peft.LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            target_modules=targets,
            exclude_modules=None,
            bias="none",
        ),
    )
    assert set(attached.base_model.targeted_module_names) == set(targets)
    assert len(targets) == 186
    assert not any(
        set(target.lower().split(".")) & child_runtime._NON_LANGUAGE_LORA_SEGMENTS
        for target in targets
    )


def _write_adapter_from_targeting(path: Path, targeting: LoraTargeting) -> set[str]:
    candidates = {
        "model.language_model.layers.0.self_attn.q_proj": (
            "base_model.model.model.language_model.layers.0.self_attn.q_proj"
        ),
        "model.visual.blocks.0.attn.proj": ("base_model.model.model.visual.blocks.0.attn.proj"),
    }
    tensors = {}
    targets = []
    for runtime_module, export_module in candidates.items():
        if _excluded(targeting, runtime_module):
            continue
        targets.append(export_module.rsplit(".", 1)[-1])
        tensors[f"{export_module}.lora_A.weight"] = np.array([[1.0, 0.0]], dtype=np.float16)
        tensors[f"{export_module}.lora_B.weight"] = np.array([[1.0], [0.0]], dtype=np.float16)
    path.mkdir()
    (path / "adapter_model.safetensors").write_bytes(save(tensors))
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 1,
                "lora_alpha": 2,
                "target_modules": sorted(set(targets)),
            }
        ),
        encoding="utf-8",
    )
    return set(tensors)


def test_text_export_passes_only_when_training_targeting_omits_the_visual_tensor(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    targeting = resolve_lora_targeting("Qwen/Qwen3.5-0.8B", algorithm="sft", multimodal=False)
    good = tmp_path / "good"
    good_keys = _write_adapter_from_targeting(good, targeting)

    stamp_adapter_dir_provenance(
        str(good),
        "Qwen/Qwen3.5-0.8B",
        exclude_modules=targeting.exclude_modules,
    )

    assert all("visual" not in key for key in good_keys)
    saved = json.loads((good / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"
    assert saved["exclude_modules"] == targeting.exclude_modules

    sabotaged = replace(targeting, exclude_modules=None)
    bad = tmp_path / "bad"
    bad_keys = _write_adapter_from_targeting(bad, sabotaged)
    assert any("visual" in key for key in bad_keys)
    with pytest.raises(RuntimeError, match="contains non-language tensor"):
        stamp_adapter_dir_provenance(str(bad), "Qwen/Qwen3.5-0.8B")
