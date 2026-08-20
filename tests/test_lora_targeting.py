from __future__ import annotations

import importlib
import json
import re
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass, field, fields, replace
from pathlib import Path
from threading import Barrier
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


@dataclass(frozen=True)
class _FrozenModelConfig:
    target_modules: object = "all-linear"
    target_parameters: list[str] | None = field(default_factory=list)
    exclude_modules: str | None = None
    lora_adapter_path: str | None = None
    preserved: object = None


def _install_fake_verl_engine(
    monkeypatch, *, attached_targets=None, builder_error=None, builder_observer=None
):
    calls = []

    class _Engine:
        def __init__(self, config):
            self.model_config = config

        def _build_lora_module(self, module):
            calls.append(self.model_config)
            if builder_observer is not None:
                builder_observer(self, module)
            if builder_error is not None:
                raise builder_error
            targets = self.model_config.target_modules
            attached = targets if attached_targets is None else attached_targets
            return SimpleNamespace(
                base_model=SimpleNamespace(
                    targeted_module_names=list(attached) if isinstance(attached, list) else []
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


@pytest.mark.parametrize("wrapper", ["", "module.", "_orig_mod.", "base_model.model."])
def test_runtime_installer_replaces_only_target_modules_and_restores_the_caller(
    monkeypatch, wrapper
):
    engine_cls, calls = _install_fake_verl_engine(monkeypatch)
    fused_targets = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    preserved = object()
    original = _FrozenModelConfig(target_parameters=fused_targets, preserved=preserved)
    original_values = {item.name: getattr(original, item.name) for item in fields(original)}
    engine = engine_cls(original)

    engine._build_lora_module(_runtime_topology(wrapper))

    replacement = calls[-1]
    expected_targets = [
        f"{wrapper}model.language_model.layers.0.mlp.down_proj",
        f"{wrapper}model.language_model.layers.0.self_attn.q_proj",
    ]
    assert replacement is not original
    assert replacement.target_modules == expected_targets
    assert replacement.target_parameters is fused_targets
    assert replacement.preserved is preserved
    assert engine.model_config is original
    assert {item.name: getattr(original, item.name) for item in fields(original)} == original_values
    for item in fields(original):
        if item.name != "target_modules":
            assert getattr(replacement, item.name) is getattr(original, item.name)
    with pytest.raises(FrozenInstanceError):
        replacement.target_modules = []
    with pytest.raises(FrozenInstanceError):
        original.target_modules = []


def test_runtime_installer_bypasses_warm_start_without_copying_or_resolving(monkeypatch):
    engine_cls, calls = _install_fake_verl_engine(monkeypatch)
    original = _FrozenModelConfig(lora_adapter_path="/adapter", exclude_modules="preserved")
    engine = engine_cls(original)

    engine._build_lora_module(_FakeModule([]))

    assert calls == [original]
    assert engine.model_config is original


def test_runtime_installer_isolates_the_original_config_when_the_builder_fails(monkeypatch):
    class _BuilderError(RuntimeError):
        pass

    live_engine = []

    def observe(isolated_engine, _module):
        assert isolated_engine is not live_engine[0]
        assert live_engine[0].model_config is original

    error = _BuilderError("builder failed")
    engine_cls, calls = _install_fake_verl_engine(
        monkeypatch, builder_error=error, builder_observer=observe
    )
    original = _FrozenModelConfig()
    engine = engine_cls(original)
    live_engine.append(engine)

    with pytest.raises(_BuilderError, match="builder failed"):
        engine._build_lora_module(_runtime_topology())
    assert calls[-1] is not original
    assert engine.model_config is original


def test_runtime_installer_isolates_same_engine_concurrent_calls(monkeypatch):
    barrier = Barrier(2)
    live_engine = []
    isolated_engines = []

    def observe(isolated_engine, module):
        isolated_engines.append(isolated_engine)
        assert isolated_engine is not live_engine[0]
        assert live_engine[0].model_config is original
        module.attached_targets = tuple(isolated_engine.model_config.target_modules)
        barrier.wait(timeout=5)
        assert live_engine[0].model_config is original

    engine_cls, calls = _install_fake_verl_engine(monkeypatch, builder_observer=observe)
    original = _FrozenModelConfig()
    engine = engine_cls(original)
    live_engine.append(engine)
    modules = [_runtime_topology("module."), _runtime_topology("_orig_mod.")]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(engine._build_lora_module, modules))

    assert engine.model_config is original
    assert len({id(item) for item in isolated_engines}) == 2
    assert all(item is not engine for item in isolated_engines)
    assert {tuple(config.target_modules) for config in calls} == {
        (
            "module.model.language_model.layers.0.mlp.down_proj",
            "module.model.language_model.layers.0.self_attn.q_proj",
        ),
        (
            "_orig_mod.model.language_model.layers.0.mlp.down_proj",
            "_orig_mod.model.language_model.layers.0.self_attn.q_proj",
        ),
    }
    assert [tuple(result.base_model.targeted_module_names) for result in results] == [
        module.attached_targets for module in modules
    ]


def test_runtime_installer_isolates_same_engine_reentrant_calls(monkeypatch):
    live_engine = []
    nested_module = _runtime_topology("module.")
    observed = []

    def observe(isolated_engine, module):
        observed.append(isolated_engine)
        assert isolated_engine is not live_engine[0]
        assert live_engine[0].model_config is original
        module.attached_targets = tuple(isolated_engine.model_config.target_modules)
        if len(observed) == 1:
            live_engine[0]._build_lora_module(nested_module)
            assert live_engine[0].model_config is original

    engine_cls, calls = _install_fake_verl_engine(monkeypatch, builder_observer=observe)
    original = _FrozenModelConfig()
    engine = engine_cls(original)
    live_engine.append(engine)
    outer_module = _runtime_topology("_orig_mod.")

    result = engine._build_lora_module(outer_module)

    assert engine.model_config is original
    assert len({id(item) for item in observed}) == 2
    assert calls[0].target_modules[0].startswith("_orig_mod.")
    assert calls[1].target_modules[0].startswith("module.")
    assert tuple(result.base_model.targeted_module_names) == outer_module.attached_targets
    assert nested_module.attached_targets == tuple(calls[1].target_modules)


def test_runtime_installer_rejects_stale_or_non_dataclass_configs_and_incomplete_attachment(
    monkeypatch,
):
    engine_cls, _calls = _install_fake_verl_engine(monkeypatch)
    with pytest.raises(RuntimeError, match="requires exclude_modules=null"):
        engine_cls(_FrozenModelConfig(exclude_modules="stale"))._build_lora_module(
            _runtime_topology()
        )
    with pytest.raises(RuntimeError, match="requires Verl's dataclass model config"):
        engine_cls(
            SimpleNamespace(
                target_modules="all-linear",
                exclude_modules=None,
                lora_adapter_path=None,
            )
        )._build_lora_module(_runtime_topology())

    engine_cls, _calls = _install_fake_verl_engine(monkeypatch, attached_targets=[])
    with pytest.raises(RuntimeError, match="did not attach the complete exact"):
        engine_cls(_FrozenModelConfig())._build_lora_module(_runtime_topology())


def test_sft_grpo_and_opd_plugins_install_text_targeting_only_for_text_jobs(tmp_path):
    importlib.import_module("flash.engine.worker.sft_train")
    from flash.engine.worker import sft_train_runner
    from flash.engine.worker.train.opd.overrides import _build_opd_plugin_config
    from flash.engine.worker.train.rl.child.plugin import required_patch_names

    shim_dir = tmp_path / "sft-text-shim"
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

    multimodal_dir = tmp_path / "sft-multimodal-shim"
    multimodal_dir.mkdir()
    _marker, expected, raw_sft = sft_train_runner._write_sft_child_shims(
        SimpleNamespace(model_id="Qwen/Qwen3.5-0.8B", save_at_steps=()),
        SimpleNamespace(
            update_horizon=1,
            reentrant_gradient_checkpointing=False,
            exclude_modules=None,
        ),
        shim_dir=str(multimodal_dir),
        custom_dataset_path=str(multimodal_dir / "dataset.py"),
        seed=42,
        loggers=[],
        gdn_reset_arch=None,
    )
    assert child_runtime.TEXT_LORA_TARGET_SHIM not in expected
    assert json.loads(raw_sft)["lora_language_prefix"] == ""

    grpo_text = required_patch_names(
        {"lora_language_prefix": "model.language_model", "total_steps": 1}
    )
    grpo_multimodal = required_patch_names({"lora_language_prefix": "", "total_steps": 1})
    assert child_runtime.TEXT_LORA_TARGET_SHIM in grpo_text
    assert child_runtime.TEXT_LORA_TARGET_SHIM not in grpo_multimodal

    raw_opd = _build_opd_plugin_config(
        shim_dir=str(tmp_path),
        save_at_steps=(),
        total_steps=1,
        gdn_model_type=None,
        loggers=[],
        lora_language_prefix="model.language_model",
    )
    assert json.loads(raw_opd)["lora_language_prefix"] == "model.language_model"
    raw_opd = _build_opd_plugin_config(
        shim_dir=str(tmp_path),
        save_at_steps=(),
        total_steps=1,
        gdn_model_type=None,
        loggers=[],
        lora_language_prefix="",
    )
    assert "lora_language_prefix" not in json.loads(raw_opd)


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
        # the artifact is sabotaged, not the run: this is still a text-only export, so it is
        # stamped with the same real exclude regex the good case used. dropping it here would
        # make the call announce a multimodal run and legitimize the vision tensors.
        stamp_adapter_dir_provenance(
            str(bad), "Qwen/Qwen3.5-0.8B", exclude_modules=targeting.exclude_modules
        )


def test_multimodal_export_publishes_the_vision_tensors_it_was_told_to_train(tmp_path):
    """a multimodal run trains vision linears on purpose, so publish must not reject them.

    `resolve_lora_targeting(multimodal=True)` emits no exclude regex, so `all-linear` covers the
    vision tower and the merger writes those tensors. rejecting them at the export boundary would
    fail a healthy image run at publish for weights its own targeting selected.
    """
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    targeting = resolve_lora_targeting("Qwen/Qwen3.5-0.8B", algorithm="sft", multimodal=True)
    assert targeting.exclude_modules is None

    adapter = tmp_path / "multimodal"
    keys = _write_adapter_from_targeting(adapter, targeting)
    assert any("visual" in key for key in keys), "fixture must carry a real vision tensor"
    assert any("language_model" in key for key in keys), "fixture must carry a language tensor"

    stamp_adapter_dir_provenance(
        str(adapter),
        "Qwen/Qwen3.5-0.8B",
        exclude_modules=targeting.exclude_modules,
    )

    saved = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"
    assert "exclude_modules" not in saved


def test_multimodal_export_still_requires_a_trained_language_tensor(tmp_path):
    """a payload that is entirely vision would serve as a text model with no learned text delta."""
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    adapter = tmp_path / "vision-only"
    adapter.mkdir()
    module = "base_model.model.model.visual.blocks.0.attn.proj"
    (adapter / "adapter_model.safetensors").write_bytes(
        save(
            {
                f"{module}.lora_A.weight": np.array([[1.0, 0.0]], dtype=np.float16),
                f"{module}.lora_B.weight": np.array([[1.0], [0.0]], dtype=np.float16),
            }
        )
    )
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 1, "lora_alpha": 2, "target_modules": ["proj"]}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no language-stack LoRA pair"):
        stamp_adapter_dir_provenance(str(adapter), "Qwen/Qwen3.5-0.8B", exclude_modules=None)
