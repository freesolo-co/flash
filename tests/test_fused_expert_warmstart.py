"""Fused-expert LoRA export and strict warm-start validation tests."""

import copy
import ctypes
import json
import sys

import numpy as np
import pytest
from safetensors.numpy import save

_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
_TARGETS = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]
_EXPECTED_PAIRS = (
    ((8192, 512), (2048, 8192)),
    ((8192, 2048), (1024, 8192)),
)


def _import_worker(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as worker

    return worker


def _wrapper_tensors(prefix, pair, factor_leaves=("default.weight", "default.weight")):
    shape_a, shape_b = pair
    leaf_a, leaf_b = factor_leaves
    return {
        f"{prefix}.lora_A.{leaf_a}": shape_a,
        f"{prefix}.lora_B.{leaf_b}": shape_b,
    }


def _ordinary_tensors(
    target="q_proj",
    pair=((32, 2048), (2048, 32)),
    factor_leaves=("default.weight", "default.weight"),
):
    prefix = f"base_model.model.layers.0.self_attn.{target}"
    return _wrapper_tensors(prefix, pair, factor_leaves)


def _complete_expert_tensors(
    *,
    owner_segment="mlp.experts",
    pairs=_EXPECTED_PAIRS,
    swap_rungs=False,
    factor_leaves=("default.weight", "default.weight"),
    include_ordinary=True,
):
    tensors = {}
    for layer in range(40):
        owner = f"base_model.model.layers.{layer}.{owner_segment}"
        first, second = reversed(pairs) if swap_rungs and layer % 2 else pairs
        tensors.update(_wrapper_tensors(owner, first, factor_leaves))
        tensors.update(_wrapper_tensors(f"{owner}.base_layer", second, factor_leaves))
    if include_ordinary:
        tensors.update(_ordinary_tensors(factor_leaves=factor_leaves))
    return tensors


def _write_small_safetensors(path, tensors=None):
    arrays = tensors or {"placeholder": np.zeros((1,), dtype=np.float16)}
    path.write_bytes(save(arrays))


def _write_bf16_safetensors(path, tensors):
    from safetensors import TensorSpec, serialize

    buffers = []
    specs = {}
    for key, value in tensors.items():
        array = np.asarray(value, dtype=np.float32)
        raw = (array.view(np.uint32) >> 16).astype("<u2").tobytes()
        buffer = ctypes.create_string_buffer(raw)
        buffers.append(buffer)
        specs[key] = TensorSpec(
            dtype="bfloat16",
            shape=list(array.shape),
            data_ptr=ctypes.addressof(buffer),
            data_len=len(raw),
        )
    path.write_bytes(serialize(specs))


def _write_sharded_bf16_text_adapter(directory, factor_a, factor_b, *, config=None):
    pair = _text_pair("self_attn.q_proj", factor_a, factor_b)
    a_key, b_key = pair
    first = "adapter_model-00001-of-00002.safetensors"
    second = "adapter_model-00002-of-00002.safetensors"
    _write_bf16_safetensors(directory / first, {a_key: pair[a_key]})
    _write_bf16_safetensors(directory / second, {b_key: pair[b_key]})
    (directory / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {a_key: first, b_key: second}}), encoding="utf-8"
    )
    adapter_config = config or {
        "peft_type": "LORA",
        "r": 1,
        "lora_alpha": 2,
        "target_modules": ["q_proj"],
    }
    (directory / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    return (directory / first, directory / second)


def _text_pair(module, factor_a, factor_b):
    prefix = f"base_model.model.layers.0.{module}"
    return {
        f"{prefix}.lora_A.weight": np.asarray(factor_a, dtype=np.float16),
        f"{prefix}.lora_B.weight": np.asarray(factor_b, dtype=np.float16),
    }


def _text_adapter_tensors(mode, rank=1):
    zero_a = np.zeros((rank, 2), dtype=np.float16)
    zero_b = np.zeros((2, rank), dtype=np.float16)
    nonzero_a = zero_a.copy()
    nonzero_b = zero_b.copy()
    nonzero_a[0, 0] = 1.0
    nonzero_b[0, 0] = 1.0
    zero_pair = _text_pair("self_attn.q_proj", zero_a, zero_b)
    nonzero_pair = _text_pair("self_attn.v_proj", nonzero_a, nonzero_b)
    if mode == "mixed":
        return {**zero_pair, **nonzero_pair}
    if mode == "orphan_a":
        key = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
        return {**nonzero_pair, key: np.ones((rank, 2), dtype=np.float16)}
    if mode == "orphan_b":
        key = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
        return {**nonzero_pair, key: np.ones((2, rank), dtype=np.float16)}
    if mode == "nonfinite":
        nonfinite_a = nonzero_a.copy()
        nonfinite_a[0, 0] = np.nan
        return _text_pair("self_attn.q_proj", nonfinite_a, nonzero_b)
    if mode == "all_zero":
        return zero_pair
    if mode == "vision":
        return {
            **nonzero_pair,
            **_text_pair("visual.patch_embed.proj", nonzero_a, nonzero_b),
        }
    if mode == "vision_saved":
        return {
            **nonzero_pair,
            "base_model.model.visual.proj.modules_to_save.default.weight": np.ones(
                (2, 2), dtype=np.float16
            ),
        }
    if mode == "projector_lora":
        return {
            **nonzero_pair,
            **_text_pair("multi_modal_projector.linear", nonzero_a, nonzero_b),
        }
    if mode == "mtp_saved":
        return {
            **nonzero_pair,
            "base_model.model.mtp.proj.modules_to_save.default.weight": np.ones(
                (2, 2), dtype=np.float16
            ),
        }
    if mode == "text_saved":
        return {
            **nonzero_pair,
            "base_model.model.layers.0.embed_tokens.modules_to_save.default.weight": np.ones(
                (2, 2), dtype=np.float16
            ),
        }
    if mode == "legacy_default_leaf":
        return {
            key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"
            ): value
            for key, value in nonzero_pair.items()
        }
    if mode == "bias_leaf":
        return {key.replace(".weight", ".bias"): value for key, value in nonzero_pair.items()}
    if mode == "arbitrary_namespace":
        return {
            "attacker.layers.0.q_proj.lora_A.weight": nonzero_a,
            "attacker.layers.0.q_proj.lora_B.weight": nonzero_b,
        }
    if mode == "rank_mismatch":
        return _text_pair(
            "self_attn.q_proj",
            np.ones((rank + 1, 2), dtype=np.float16),
            np.ones((2, rank + 1), dtype=np.float16),
        )
    if mode == "zero_dimension":
        return {
            **nonzero_pair,
            **_text_pair(
                "self_attn.q_proj",
                np.ones((rank, 0), dtype=np.float16),
                np.ones((2, rank), dtype=np.float16),
            ),
        }
    raise AssertionError(f"unknown text tensor mode {mode}")


def _write_expert_adapter(directory, *, config, tensor_mode="complete", text_rank=1):
    """Write config plus a physically valid small artifact or a deliberate corruption."""
    if tensor_mode == "complete":
        _write_small_safetensors(directory / "adapter_model.safetensors")
    elif tensor_mode == "incomplete":
        _write_small_safetensors(
            directory / "adapter_model.safetensors",
            _text_adapter_tensors("orphan_a", text_rank),
        )
    elif tensor_mode in {
        "arbitrary_namespace",
        "all_zero",
        "bias_leaf",
        "legacy_default_leaf",
        "mixed",
        "mtp_saved",
        "nonfinite",
        "orphan_a",
        "orphan_b",
        "projector_lora",
        "rank_mismatch",
        "text_saved",
        "vision",
        "vision_saved",
        "zero_dimension",
    }:
        _write_small_safetensors(
            directory / "adapter_model.safetensors",
            _text_adapter_tensors(tensor_mode, text_rank),
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


def _patch_export_metadata(monkeypatch):
    import flash.engine.worker.verl.checkpoints as checkpoints

    tensors = _complete_expert_tensors()
    tensors.update(_ordinary_tensors(target="v_proj"))
    monkeypatch.setattr(
        checkpoints,
        "_read_adapter_tensor_metadata",
        lambda _path: tensors,
    )
    return checkpoints.stamp_adapter_dir_provenance


def _patch_worker_metadata(monkeypatch):
    import flash.engine.worker.model.adapter as adapter

    monkeypatch.setattr(
        adapter,
        "_read_adapter_tensor_metadata",
        lambda _path: _complete_expert_tensors(),
    )


@pytest.mark.parametrize(
    "target_modules",
    [
        ["v_proj", "experts", "q_proj", "v_proj", "base_layer"],
        ["base_layer", "q_proj", "experts", "v_proj"],
    ],
)
def test_verl_fused_export_canonicalizes_suffix_order_to_all_linear(
    monkeypatch, tmp_path, target_modules
):
    stamp_adapter_dir_provenance = _patch_export_metadata(monkeypatch)
    config = {
        "peft_type": "LORA",
        "r": 32,
        "lora_alpha": 64,
        "target_modules": target_modules,
        "target_parameters": None,
        "flash_provenance": {"source": "verl"},
    }
    _write_expert_adapter(tmp_path, config=config)

    stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "c" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_parameters"] == _TARGETS
    assert saved["target_modules"] == "all-linear"
    assert saved["r"] == 32
    assert saved["lora_alpha"] == 64
    assert saved["base_model_name_or_path"] == _MODEL_ID
    assert saved["revision"] == "c" * 40
    assert saved["flash_provenance"] == {"source": "verl"}


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


def test_export_rejects_debris_only_modules_without_writing(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["experts", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(ValueError, match="non-empty"):
        stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "d" * 40)

    assert config_path.read_bytes() == before


def test_export_rejects_direct_parameter_only_config_without_writing(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": None,
        "target_parameters": list(_TARGETS),
    }
    _write_expert_adapter(tmp_path, config=config)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(ValueError, match="ordinary LoRA target_modules"):
        stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "d" * 40)

    assert config_path.read_bytes() == before


@pytest.mark.parametrize(
    ("rank", "alpha", "dropout", "use_rslora"),
    [
        (1, 1, 0.0, False),
        (2, 4, 0.05, True),
        (4, 16, 0.1, False),
    ],
)
def test_non_moe_export_canonicalizes_targeting_without_changing_other_fields(
    tmp_path, rank, alpha, dropout, use_rslora
):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "use_rslora": use_rslora,
        "target_modules": ["q_proj", "v_proj"],
        "rank_pattern": {"q_proj": rank},
        "flash_provenance": {"source": "verl"},
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode="mixed", text_rank=rank)

    stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"
    assert saved["r"] == rank
    assert saved["lora_alpha"] == alpha
    assert saved["lora_dropout"] == dropout
    assert saved["use_rslora"] is use_rslora
    assert saved["rank_pattern"] == {"q_proj": rank}
    assert saved["base_model_name_or_path"] == "Qwen/Qwen3.5-9B"
    assert saved["revision"] == "d" * 40
    assert saved["flash_provenance"] == {"source": "verl"}
    assert "target_parameters" not in saved


def test_non_moe_export_enforces_applicable_rank_pattern(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    q_a = np.ones((1, 2), dtype=np.float16)
    q_b = np.ones((2, 1), dtype=np.float16)
    v_a = np.ones((3, 2), dtype=np.float16)
    v_b = np.ones((2, 3), dtype=np.float16)
    tensors = {
        **_text_pair("self_attn.q_proj", q_a, q_b),
        **_text_pair("self_attn.v_proj", v_a, v_b),
    }
    config = {
        "peft_type": "LORA",
        "r": 1,
        "rank_pattern": {"v_proj": 3},
        "target_modules": ["q_proj", "v_proj"],
    }
    _write_small_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["rank_pattern"] == {"v_proj": 3}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"r": "1"}, id="numeric-string-scalar"),
        pytest.param({"rank_pattern": {"q_proj": "1"}}, id="applicable-string-override"),
    ],
)
def test_non_moe_export_uses_strict_shared_rank_declarations(tmp_path, overrides):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj", "v_proj"],
        **overrides,
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode="mixed", text_rank=1)
    config_path = tmp_path / "adapter_config.json"
    before_config = config_path.read_bytes()
    before_weights = (tmp_path / "adapter_model.safetensors").read_bytes()

    with pytest.raises(RuntimeError, match="positive integer"):
        stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    assert config_path.read_bytes() == before_config
    assert (tmp_path / "adapter_model.safetensors").read_bytes() == before_weights


@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize(
    ("tensor_mode", "message"),
    [
        ("orphan_a", "no orphan factors"),
        ("orphan_b", "no orphan factors"),
        ("nonfinite", "contains non-finite values"),
        ("all_zero", "no nonzero composed LoRA delta"),
        ("vision", "contains non-language tensor"),
        ("vision_saved", "contains non-language tensor"),
        ("projector_lora", "contains non-language tensor"),
        ("mtp_saved", "contains non-language tensor"),
        ("text_saved", "non-canonical tensor key"),
        ("legacy_default_leaf", "non-canonical tensor key"),
        ("bias_leaf", "non-canonical tensor key"),
        ("arbitrary_namespace", "non-canonical tensor key"),
        ("rank_mismatch", "disagrees with its configured LoRA rank"),
        ("zero_dimension", "not positive and 2-D"),
    ],
)
def test_non_moe_export_rejects_invalid_actual_tensor_artifacts(
    tmp_path, tensor_mode, message, rank
):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank * 2,
        "target_modules": ["q_proj", "v_proj"],
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode=tensor_mode, text_rank=rank)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(RuntimeError, match=message):
        stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    assert config_path.read_bytes() == before


def test_non_moe_export_rejects_declared_multimodal_projector_lora(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 2,
        "target_modules": ["v_proj", "multi_modal_projector.linear"],
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode="projector_lora", text_rank=2)

    with pytest.raises(RuntimeError, match="contains non-language tensor"):
        stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)


def test_exported_targeting_validates_the_actual_mixed_delta_artifact(tmp_path):
    from safetensors import safe_open

    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode="mixed", text_rank=2)

    stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"
    with safe_open(tmp_path / "adapter_model.safetensors", framework="numpy") as weights:
        keys = set(weights.keys())
        a_keys = sorted(key for key in keys if ".lora_A." in key)
        assert len(a_keys) == 2
        assert all(key.endswith(".lora_A.weight") for key in a_keys)
        assert not any(".default.weight" in key for key in keys)
        nonzero = []
        for a_key in a_keys:
            b_key = a_key.replace(".lora_A.", ".lora_B.", 1)
            assert b_key in keys
            delta = weights.get_tensor(b_key) @ weights.get_tensor(a_key)
            assert np.isfinite(delta).all()
            nonzero.append(bool(np.count_nonzero(delta)))
        assert sorted(nonzero) == [False, True]
        assert not any("visual" in key or "patch_embed" in key for key in keys)


def test_non_moe_export_accepts_sharded_bf16_without_torch(monkeypatch, tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    factor_a = np.array([[1.0, 0.0]], dtype=np.float32)
    factor_b = np.array([[1.0], [0.0]], dtype=np.float32)
    weight_paths = _write_sharded_bf16_text_adapter(tmp_path, factor_a, factor_b)
    before_weights = [path.read_bytes() for path in weight_paths]
    monkeypatch.setitem(sys.modules, "torch", None)

    stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"
    assert [path.read_bytes() for path in weight_paths] == before_weights


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_moe_export_rejects_nonfinite_bf16_without_writing(tmp_path, bad_value):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    factor_a = np.array([[bad_value, 1.0]], dtype=np.float32)
    factor_b = np.array([[1.0], [0.0]], dtype=np.float32)
    weight_paths = _write_sharded_bf16_text_adapter(tmp_path, factor_a, factor_b)
    config_path = tmp_path / "adapter_config.json"
    before_config = config_path.read_bytes()
    before_weights = [path.read_bytes() for path in weight_paths]

    with pytest.raises(RuntimeError, match="contains non-finite values"):
        stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    assert config_path.read_bytes() == before_config
    assert [path.read_bytes() for path in weight_paths] == before_weights


def test_non_moe_export_rejects_zero_bf16_delta_without_writing(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    factor_a = np.zeros((1, 2), dtype=np.float32)
    factor_b = np.ones((2, 1), dtype=np.float32)
    weight_paths = _write_sharded_bf16_text_adapter(tmp_path, factor_a, factor_b)
    config_path = tmp_path / "adapter_config.json"
    before_config = config_path.read_bytes()
    before_weights = [path.read_bytes() for path in weight_paths]

    with pytest.raises(RuntimeError, match="no nonzero composed LoRA delta"):
        stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    assert config_path.read_bytes() == before_config
    assert [path.read_bytes() for path in weight_paths] == before_weights


def test_strict_worker_accepts_current_config_without_changing_memory_or_disk(
    monkeypatch, tmp_path
):
    worker = _import_worker(monkeypatch)
    _patch_worker_metadata(monkeypatch)
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


def test_strict_worker_rejects_structurally_compatible_wrong_shapes(monkeypatch, tmp_path):
    import flash.engine.worker.model.adapter as adapter

    worker = _import_worker(monkeypatch)
    counterexample = _complete_expert_tensors(pairs=(((7, 1), (2, 7)), ((7, 1), (2, 7))))
    monkeypatch.setattr(adapter, "_read_adapter_tensor_metadata", lambda _path: counterexample)
    config = _valid_config()
    _write_expert_adapter(tmp_path, config=config)
    before = (tmp_path / "adapter_config.json").read_bytes()

    with pytest.raises(ValueError, match="complete fused expert LoRA weights"):
        worker.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path))

    assert (tmp_path / "adapter_config.json").read_bytes() == before


def test_export_rejects_structurally_compatible_wrong_shapes(monkeypatch, tmp_path):
    import flash.engine.worker.verl.checkpoints as checkpoints

    counterexample = _complete_expert_tensors(pairs=(((7, 1), (2, 7)), ((7, 1), (2, 7))))
    monkeypatch.setattr(checkpoints, "_read_adapter_tensor_metadata", lambda _path: counterexample)
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "experts", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(RuntimeError, match="complete fused expert LoRA weights"):
        checkpoints.stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "d" * 40)

    assert config_path.read_bytes() == before


def test_tensor_analyzer_accepts_canonical_qwen36_rung_geometry():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    assert has_complete_fused_expert_tensors(_complete_expert_tensors(), _valid_config(), _MODEL_ID)


def test_tensor_analyzer_rejects_swapped_qwen36_rungs():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    assert not has_complete_fused_expert_tensors(
        _complete_expert_tensors(swap_rungs=True), _valid_config(), _MODEL_ID
    )


def test_tensor_analyzer_requires_matching_adapter_namespaces_across_all_exact_owners():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    config = _valid_config()
    assert has_complete_fused_expert_tensors(
        _complete_expert_tensors(factor_leaves=("foo.weight", "foo.weight")),
        config,
        _MODEL_ID,
    )
    assert not has_complete_fused_expert_tensors(
        _complete_expert_tensors(factor_leaves=("foo.weight", "bar.weight")),
        config,
        _MODEL_ID,
    )


def test_tensor_analyzer_rejects_namespace_changes_across_wrapper_rungs():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors(factor_leaves=("foo.weight", "foo.weight"))
    tensors = {
        key.replace(".foo.weight", ".bar.weight") if ".base_layer.lora_" in key else key: shape
        for key, shape in tensors.items()
    }

    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


def test_tensor_analyzer_rejects_namespace_changes_across_catalog_layers():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors(factor_leaves=("foo.weight", "foo.weight"))
    tensors = {
        key.replace(".foo.weight", ".bar.weight") if ".layers.17." in key else key: shape
        for key, shape in tensors.items()
    }

    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


@pytest.mark.parametrize(
    "factor_leaves",
    [
        pytest.param(("not_peft", "default.weight"), id="invalid-a-leaf"),
        pytest.param(("default.weight", "not_peft"), id="invalid-b-leaf"),
        pytest.param(("default.weight.extra", "default.weight"), id="extra-segment"),
        pytest.param(("default.weight", ".weight"), id="empty-adapter-name"),
    ],
)
def test_tensor_analyzer_rejects_invalid_factor_leaves_across_all_exact_owners(
    factor_leaves,
):
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors(factor_leaves=factor_leaves)

    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


def test_tensor_analyzer_requires_every_catalog_layer_and_both_lora_factors():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    complete = _complete_expert_tensors()
    config = _valid_config()
    assert not has_complete_fused_expert_tensors(
        {key: shape for key, shape in complete.items() if "layers.1.mlp.experts" not in key},
        config,
        _MODEL_ID,
    )
    assert not has_complete_fused_expert_tensors(
        {
            key: shape
            for key, shape in complete.items()
            if not key.endswith("lora_B.default.weight")
        },
        config,
        _MODEL_ID,
    )


def test_tensor_analyzer_requires_the_exact_owner_after_the_layer_prefix():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    config = _valid_config()
    assert not has_complete_fused_expert_tensors(
        _complete_expert_tensors(owner_segment="junk.mlp.experts"), config, _MODEL_ID
    )
    assert not has_complete_fused_expert_tensors(
        _complete_expert_tensors(owner_segment="router.experts"), config, _MODEL_ID
    )


def test_tensor_analyzer_requires_the_nested_peft_wrapper_ladder():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = {}
    for layer in range(40):
        owner = f"base_model.model.layers.{layer}.mlp.experts"
        tensors.update(_wrapper_tensors(f"{owner}.foo", _EXPECTED_PAIRS[0]))
        tensors.update(_wrapper_tensors(f"{owner}.bar", _EXPECTED_PAIRS[1]))
    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


def _expert_namespace_mismatch(rung):
    tensors = _complete_expert_tensors()
    suffix = "" if rung == "outer" else ".base_layer"
    prefix = f"base_model.model.layers.0.mlp.experts{suffix}"
    default_key = f"{prefix}.lora_B.default.weight"
    tensors[f"{prefix}.lora_B.other.weight"] = tensors.pop(default_key)
    return tensors


@pytest.mark.parametrize("boundary", ["export", "warmstart"])
@pytest.mark.parametrize("rung", ["outer", "nested"])
def test_boundaries_reject_cross_namespace_fused_pairs(monkeypatch, tmp_path, boundary, rung):
    tensors = _expert_namespace_mismatch(rung)
    if boundary == "export":
        import flash.engine.worker.verl.checkpoints as checkpoints

        monkeypatch.setattr(checkpoints, "_read_adapter_tensor_metadata", lambda _path: tensors)
        config = {
            "peft_type": "LORA",
            "r": 32,
            "target_modules": ["q_proj", "experts", "base_layer"],
            "target_parameters": None,
        }

        def validate():
            checkpoints.stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "d" * 40)

        expected_error = RuntimeError
    else:
        import flash.engine.worker.model.adapter as adapter

        worker = _import_worker(monkeypatch)
        monkeypatch.setattr(adapter, "_read_adapter_tensor_metadata", lambda _path: tensors)
        config = _valid_config()

        def validate():
            worker.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path))

        expected_error = ValueError

    _write_expert_adapter(tmp_path, config=config)
    before = (tmp_path / "adapter_config.json").read_bytes()
    with pytest.raises(expected_error, match="complete fused expert LoRA weights"):
        validate()
    assert (tmp_path / "adapter_config.json").read_bytes() == before


def test_tensor_analyzer_requires_each_declared_ordinary_target():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    config = _valid_config(target_modules=["q_proj", "v_proj"])
    assert not has_complete_fused_expert_tensors(_complete_expert_tensors(), config, _MODEL_ID)


def test_tensor_analyzer_uses_anchored_ordinary_target_suffixes():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    config = _valid_config(target_modules=["proj"])
    assert not has_complete_fused_expert_tensors(_complete_expert_tensors(), config, _MODEL_ID)


def test_tensor_analyzer_requires_ordinary_evidence_for_all_linear():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    config = _valid_config(target_modules="all-linear")
    assert not has_complete_fused_expert_tensors(
        _complete_expert_tensors(include_ordinary=False), config, _MODEL_ID
    )


def test_tensor_analyzer_rejects_unrecognized_fused_descendant_as_all_linear_evidence():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors(include_ordinary=False)
    tensors.update(
        _wrapper_tensors(
            "base_model.model.layers.0.mlp.experts.base_layer.base_layer",
            ((32, 2048), (2048, 32)),
        )
    )
    assert not has_complete_fused_expert_tensors(
        tensors, _valid_config(target_modules="all-linear"), _MODEL_ID
    )


def test_tensor_analyzer_rejects_cross_namespace_ordinary_pair():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors(include_ordinary=False)
    prefix = "base_model.model.layers.0.self_attn.q_proj"
    tensors.update(
        _wrapper_tensors(
            prefix,
            ((32, 2048), (2048, 32)),
            factor_leaves=("default.weight", "other.weight"),
        )
    )
    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


def test_tensor_analyzer_rejects_extra_ordinary_namespace():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors()
    prefix = "base_model.model.layers.0.self_attn.q_proj"
    tensors.update(
        _wrapper_tensors(
            prefix,
            ((32, 2048), (2048, 32)),
            factor_leaves=("other.weight", "other.weight"),
        )
    )
    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


def test_tensor_analyzer_rejects_ordinary_rank_mismatch():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors(include_ordinary=False)
    tensors.update(_ordinary_tensors(pair=((16, 2048), (2048, 16))))
    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"r": True},
            "r must be a positive integer",
            id="invalid-scalar",
        ),
        pytest.param(
            {"rank_pattern": []},
            "rank_pattern must be an object",
            id="invalid-pattern-container",
        ),
        pytest.param(
            {"rank_pattern": {"(": 16}},
            "rank_pattern contains invalid regex",
            id="invalid-pattern-regex",
        ),
    ],
)
def test_fused_config_uses_strict_shared_rank_declarations(overrides, message):
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    with pytest.raises(ValueError, match=message):
        validate_fused_expert_adapter_config(_valid_config(**overrides), _MODEL_ID)


def test_tensor_analyzer_accepts_per_target_fused_rank_overrides():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    config = _valid_config(
        rank_pattern={
            "mlp.experts.gate_up_proj": 16,
            "mlp.experts.down_proj": 8,
        }
    )
    pairs = (
        ((2048, 512), (2048, 2048)),
        ((4096, 2048), (1024, 4096)),
    )
    assert has_complete_fused_expert_tensors(
        _complete_expert_tensors(pairs=pairs), config, _MODEL_ID
    )


def test_tensor_analyzer_accepts_ordinary_override_with_fused_scalar_fallback():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors(include_ordinary=False)
    tensors.update(_ordinary_tensors(pair=((16, 2048), (2048, 16))))
    assert has_complete_fused_expert_tensors(
        tensors, _valid_config(rank_pattern={"q_proj": 16}), _MODEL_ID
    )


def test_tensor_analyzer_rejects_scalar_shapes_when_fused_override_applies():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    config = _valid_config(rank_pattern={"mlp.experts.gate_up_proj": 16})
    assert not has_complete_fused_expert_tensors(_complete_expert_tensors(), config, _MODEL_ID)


@pytest.mark.parametrize(
    ("tensors", "config"),
    [
        pytest.param(
            _complete_expert_tensors(pairs=(((7, 1), (2, 7)), ((7, 1), (2, 7)))),
            _valid_config(),
            id="structurally-compatible-counterexample",
        ),
        pytest.param(_complete_expert_tensors(), _valid_config(r=16), id="wrong-rank"),
        pytest.param(
            _complete_expert_tensors(pairs=(((8192, 2047), (1024, 8192)), _EXPECTED_PAIRS[1])),
            _valid_config(),
            id="wrong-target-width",
        ),
        pytest.param(
            _complete_expert_tensors(
                pairs=(((1, 8192, 2048), (1, 1024, 8192)), _EXPECTED_PAIRS[1])
            ),
            _valid_config(),
            id="three-dimensional-factors",
        ),
        pytest.param(
            _complete_expert_tensors(pairs=(_EXPECTED_PAIRS[0], _EXPECTED_PAIRS[0])),
            _valid_config(),
            id="wrong-rung-multiset",
        ),
    ],
)
def test_tensor_analyzer_rejects_non_peft_fused_shapes(tensors, config):
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    assert not has_complete_fused_expert_tensors(tensors, config, _MODEL_ID)
