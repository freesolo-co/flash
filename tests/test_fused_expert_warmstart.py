"""Fused-expert LoRA export and strict warm-start validation tests."""

import copy
import json
import struct
import sys

import pytest

_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
_TARGETS = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]


def _import_worker(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as worker

    return worker


def _wrapper_tensors(prefix):
    """Return both LoRA factors for one PEFT wrapper."""
    return [f"{prefix}.lora_A.default.weight", f"{prefix}.lora_B.default.weight"]


def _complete_expert_keys():
    keys = []
    for layer in range(40):
        owner = f"base_model.model.layers.{layer}.mlp.experts"
        keys += _wrapper_tensors(owner) + _wrapper_tensors(f"{owner}.base_layer")
    return keys


def _write_safetensors(path, keys):
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x01\x02")


def _write_expert_adapter(directory, *, config, tensor_mode="complete"):
    """Write the config and the requested tensor shape without importing torch."""
    if tensor_mode == "complete":
        _write_safetensors(directory / "adapter_model.safetensors", _complete_expert_keys())
    elif tensor_mode == "incomplete":
        _write_safetensors(
            directory / "adapter_model.safetensors",
            ["base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"],
        )
    elif tensor_mode == "unreadable":
        (directory / "adapter_model.safetensors").write_bytes(b"bad")
    else:
        raise AssertionError(f"unknown tensor mode {tensor_mode}")
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def _valid_config(**overrides):
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj"],
        "target_parameters": list(_TARGETS),
    }
    config.update(overrides)
    return config


def test_verl_export_normalizes_fused_targets_without_reordering_other_modules(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["v_proj", "experts", "q_proj", "v_proj", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)

    stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "c" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_parameters"] == _TARGETS
    assert saved["target_modules"] == ["v_proj", "q_proj", "v_proj"]
    assert saved["base_model_name_or_path"] == _MODEL_ID
    assert saved["revision"] == "c" * 40


def test_export_refuses_incomplete_expert_weights_before_changing_config(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "experts", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode="incomplete")
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(RuntimeError, match="complete fused expert LoRA weights"):
        stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "d" * 40)

    assert config_path.read_bytes() == before


def test_export_rejects_malformed_modules_after_normalization_without_writing(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["experts", {"invalid": True}, "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(ValueError, match="target_modules"):
        stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "d" * 40)

    assert config_path.read_bytes() == before


def test_non_moe_export_keeps_adapter_targeting_unchanged(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {"peft_type": "LORA", "r": 32, "target_modules": ["q_proj", "v_proj"]}
    _write_expert_adapter(tmp_path, config=config, tensor_mode="incomplete")

    stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == ["q_proj", "v_proj"]
    assert "target_parameters" not in saved


def test_strict_worker_accepts_current_config_without_changing_memory_or_disk(
    monkeypatch, tmp_path
):
    worker = _import_worker(monkeypatch)
    config = _valid_config()
    _write_expert_adapter(tmp_path, config=config)
    before_config = copy.deepcopy(config)
    before_file = (tmp_path / "adapter_config.json").read_bytes()

    worker.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path))

    assert config == before_config
    assert (tmp_path / "adapter_config.json").read_bytes() == before_file


def test_strict_worker_rejects_missing_targets_without_changing_memory_or_disk(
    monkeypatch, tmp_path
):
    worker = _import_worker(monkeypatch)
    config = _valid_config(target_parameters=None)
    _write_expert_adapter(tmp_path, config=config)
    before_config = copy.deepcopy(config)
    before_file = (tmp_path / "adapter_config.json").read_bytes()

    with pytest.raises(ValueError, match="omits required expert targets"):
        worker.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path))

    assert config == before_config
    assert (tmp_path / "adapter_config.json").read_bytes() == before_file


@pytest.mark.parametrize("synthetic", ["experts", "base_layer"])
def test_strict_worker_rejects_synthetic_modules_without_changing_memory_or_disk(
    monkeypatch, tmp_path, synthetic
):
    worker = _import_worker(monkeypatch)
    config = _valid_config(target_modules=["q_proj", synthetic])
    _write_expert_adapter(tmp_path, config=config)
    before_config = copy.deepcopy(config)
    before_file = (tmp_path / "adapter_config.json").read_bytes()

    with pytest.raises(ValueError, match="invalid synthetic target_modules"):
        worker.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path))

    assert config == before_config
    assert (tmp_path / "adapter_config.json").read_bytes() == before_file


@pytest.mark.parametrize(
    ("tensor_mode", "message"),
    [
        ("incomplete", "complete fused expert LoRA weights"),
        ("unreadable", "safetensors file"),
    ],
)
def test_strict_worker_rejects_bad_tensor_artifacts_without_changing_config_or_files(
    monkeypatch, tmp_path, tensor_mode, message
):
    worker = _import_worker(monkeypatch)
    config = _valid_config()
    _write_expert_adapter(tmp_path, config=config, tensor_mode=tensor_mode)
    before_config = copy.deepcopy(config)
    before_config_file = (tmp_path / "adapter_config.json").read_bytes()
    before_weights = (tmp_path / "adapter_model.safetensors").read_bytes()

    with pytest.raises(ValueError, match=message):
        worker.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path))

    assert config == before_config
    assert (tmp_path / "adapter_config.json").read_bytes() == before_config_file
    assert (tmp_path / "adapter_model.safetensors").read_bytes() == before_weights


def test_tensor_analyzer_requires_every_catalog_layer_and_both_lora_factors():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    complete = _complete_expert_keys()
    assert has_complete_fused_expert_tensors(complete, _MODEL_ID)
    assert not has_complete_fused_expert_tensors(
        [key for key in complete if "layers.1.mlp.experts" not in key], _MODEL_ID
    )
    assert not has_complete_fused_expert_tensors(
        [key for key in complete if not key.endswith("lora_B.default.weight")], _MODEL_ID
    )


def test_tensor_analyzer_matches_the_full_owner_path():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    keys = []
    for layer in range(40):
        owner = f"base_model.model.layers.{layer}.router.experts"
        keys += _wrapper_tensors(owner) + _wrapper_tensors(f"{owner}.base_layer")
    assert not has_complete_fused_expert_tensors(keys, _MODEL_ID)


def test_tensor_analyzer_requires_the_nested_peft_wrapper_ladder():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    keys = []
    for layer in range(40):
        owner = f"base_model.model.layers.{layer}.mlp.experts"
        keys += _wrapper_tensors(f"{owner}.foo") + _wrapper_tensors(f"{owner}.bar")
    assert not has_complete_fused_expert_tensors(keys, _MODEL_ID)
