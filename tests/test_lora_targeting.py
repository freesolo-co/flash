from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save

from flash.adapters.targets import LoraTargeting, resolve_lora_targeting
from flash.core.catalog import ALGORITHMS, MODELS

_FIXTURE = Path(__file__).parent / "fixtures" / "qwen35_08b_target_metadata.json"
_CAMPAIGN_MODELS = tuple(MODELS)


def _metadata() -> dict:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _excluded(targeting: LoraTargeting, module_name: str) -> bool:
    return bool(targeting.exclude_modules and re.fullmatch(targeting.exclude_modules, module_name))


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


@pytest.mark.parametrize("model_id", _CAMPAIGN_MODELS)
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_campaign_text_trainer_targets_only_the_cataloged_language_subtree(
    model_id, algorithm
):
    targeting = resolve_lora_targeting(model_id, algorithm=algorithm, multimodal=False)
    prefix = MODELS[model_id].lora_language_prefix

    assert targeting.target_modules == "all-linear"
    assert not _excluded(targeting, f"{prefix}.layers.0.self_attn.q_proj")
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
