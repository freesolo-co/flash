"""Fused-expert LoRA export and strict warm-start validation tests."""

import copy
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


_SERIALIZED_LEAVES = ("weight", "weight")


def _wrapper_tensors(prefix, pair, factor_leaves=_SERIALIZED_LEAVES):
    shape_a, shape_b = pair
    leaf_a, leaf_b = factor_leaves
    return {
        f"{prefix}.lora_A.{leaf_a}": shape_a,
        f"{prefix}.lora_B.{leaf_b}": shape_b,
    }


def _ordinary_tensors(
    target="q_proj",
    pair=((32, 2048), (2048, 32)),
    factor_leaves=_SERIALIZED_LEAVES,
):
    prefix = f"base_model.model.layers.0.self_attn.{target}"
    return _wrapper_tensors(prefix, pair, factor_leaves)


def _complete_expert_tensors(
    *,
    owner_segment="mlp.experts",
    pairs=_EXPECTED_PAIRS,
    swap_rungs=False,
    factor_leaves=_SERIALIZED_LEAVES,
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


def _write_expert_adapter(directory, *, config, tensor_mode="complete"):
    """Write config plus a physically valid small artifact or a deliberate corruption."""
    if tensor_mode == "complete":
        _write_small_safetensors(directory / "adapter_model.safetensors")
    elif tensor_mode == "incomplete":
        key = "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        _write_small_safetensors(
            directory / "adapter_model.safetensors",
            {key: np.zeros((1, 1), dtype=np.float16)},
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


def test_verl_export_normalizes_fused_targets_without_reordering_other_modules(
    monkeypatch, tmp_path
):
    stamp_adapter_dir_provenance = _patch_export_metadata(monkeypatch)
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


def test_tensor_analyzer_rejects_every_adapter_namespace_on_disk():
    """No namespaced spelling is a real serialized artifact, uniform or not.

    PEFT strips the namespace on save and re-inserts it on load, so a file that still carries one
    loads with the namespace doubled and matches no module.
    """
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    config = _valid_config()
    for leaves in (("foo.weight", "foo.weight"), ("foo.weight", "bar.weight")):
        assert not has_complete_fused_expert_tensors(
            _complete_expert_tensors(factor_leaves=leaves), config, _MODEL_ID
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
        {key: shape for key, shape in complete.items() if not key.endswith("lora_B.weight")},
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
    tensors[f"{prefix}.lora_B.other.weight"] = tensors.pop(f"{prefix}.lora_B.weight")
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


def test_tensor_analyzer_accepts_the_grammar_peft_and_verl_actually_write():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    stripped = _complete_expert_tensors()

    assert not any(".default." in key for key in stripped)
    assert has_complete_fused_expert_tensors(stripped, _valid_config(), _MODEL_ID)


def test_export_stamps_an_adapter_serialized_without_the_adapter_namespace(monkeypatch, tmp_path):
    import flash.engine.worker.verl.checkpoints as checkpoints

    tensors = _complete_expert_tensors()
    tensors.update(_ordinary_tensors(target="v_proj"))
    monkeypatch.setattr(checkpoints, "_read_adapter_tensor_metadata", lambda _path: tensors)
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["v_proj", "experts", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)

    checkpoints.stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "e" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_parameters"] == _TARGETS
    assert saved["base_model_name_or_path"] == _MODEL_ID


def test_warmstart_accepts_an_adapter_serialized_without_the_adapter_namespace(
    monkeypatch, tmp_path
):
    import flash.engine.worker.model.adapter as adapter

    worker = _import_worker(monkeypatch)
    tensors = _complete_expert_tensors()
    monkeypatch.setattr(adapter, "_read_adapter_tensor_metadata", lambda _path: tensors)

    worker.validate_warmstart_adapter(_valid_config(), _MODEL_ID, str(tmp_path))


@pytest.mark.parametrize("rung", ["outer", "nested"])
def test_tensor_analyzer_still_rejects_cross_namespace_pairs_when_serialized(rung):
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors()
    suffix = "" if rung == "outer" else ".base_layer"
    prefix = f"base_model.model.layers.0.mlp.experts{suffix}"
    tensors[f"{prefix}.lora_B.other.weight"] = tensors.pop(f"{prefix}.lora_B.weight")

    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


@pytest.mark.parametrize(
    ("leaf", "expected"),
    [
        pytest.param("weight", "module", id="serialized-bare-leaf"),
        pytest.param("default.weight", None, id="explicit-default-does-not-roundtrip"),
        pytest.param("other.weight", None, id="explicit-namespace-does-not-roundtrip"),
        pytest.param("not_peft", None, id="wrong-parameter-name"),
        pytest.param("default.not_peft", None, id="namespaced-wrong-parameter-name"),
        pytest.param("default.weight.extra", None, id="extra-segment"),
        pytest.param(".weight", None, id="empty-adapter-name"),
        pytest.param("", None, id="empty-leaf"),
    ],
)
def test_parse_lora_tensor_pins_the_exact_accepted_leaf_grammar(leaf, expected):
    """Pin the grammar directly.

    Routing these through the whole-artifact analyzer used to be inert: a malformed key merely
    failed to parse, and the surviving tensors still satisfied every downstream check, so the
    assertion passed for the wrong reason. Sabotaging the parser to accept an empty adapter name
    left the artifact-level version green.
    """
    from flash.adapters.fused_experts import _parse_lora_tensor

    parsed = _parse_lora_tensor(f"module.lora_A.{leaf}", (8, 16))

    if expected is None:
        assert parsed is None
    else:
        module_path, factor, key, shape = parsed
        assert (module_path, factor) == (expected, "A")
        assert (key, shape) == (f"module.lora_A.{leaf}", (8, 16))


def test_parse_lora_tensor_requires_exactly_one_factor_infix():
    from flash.adapters.fused_experts import _parse_lora_tensor

    assert _parse_lora_tensor("m.lora_A.lora_B.weight", (8, 16)) is None
    assert _parse_lora_tensor("lora_A.weight", (8, 16)) is None


def test_tensor_analyzer_rejects_an_artifact_carrying_the_adapter_namespace():
    """Only the stripped grammar round-trips through PEFT, so a namespaced file must not validate.

    Verified against the locked peft 0.19.1 ``_insert_adapter_name_into_state_dict``: a stripped
    ``...lora_A.weight`` loads as ``...lora_A.default.weight`` and matches the live module, while
    ``...lora_A.default.weight`` loads as ``...lora_A.default.default.weight`` and matches nothing.
    """
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    for leaves in (("default.weight", "default.weight"), ("other.weight", "other.weight")):
        tensors = _complete_expert_tensors(factor_leaves=leaves)

        assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)


def test_tensor_analyzer_rejects_an_unparseable_lora_key():
    """An unrecognized lora key must fail the artifact rather than be skipped.

    Dropping it would let a malformed tensor ride along beside a complete canonical set, and a
    trailing-dot module alias would then collapse onto the outer rung and make the verdict depend
    on mapping iteration order.
    """
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    owner = "base_model.model.layers.0.mlp.experts"
    for alias in (f"{owner}..lora_A.weight", f"{owner}.lora_A.not_peft"):
        for alias_first in (True, False):
            tensors = _complete_expert_tensors()
            if alias_first:
                tensors = {alias: (1, 1), **tensors}
            else:
                tensors[alias] = (1, 1)

            assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)
