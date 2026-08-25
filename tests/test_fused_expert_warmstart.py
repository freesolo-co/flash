"""Fused-expert LoRA export and strict warm-start validation tests."""

import copy
import ctypes
import json
import sys

import numpy as np
import pytest
from safetensors.numpy import save

import flash.engine.worker.model.adapter as worker_adapter
from flash.adapters.targets import resolve_lora_targeting

_MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
# every export in this file is a text-only run, and the exporter reads the modality off
# `exclude_modules`: the language-prefix regex `resolve_lora_targeting` emits for a text-only run,
# None for a multimodal one. stating it keeps these artifacts under the strict text contract.
_TEXT_ONLY_EXCLUDE = r"^(?!model\.language_model(?:\.|$)).*$"
_TEXT_TARGETING = resolve_lora_targeting(_MODEL_ID, algorithm="sft", multimodal=False)
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
        "exclude_modules": _TEXT_ONLY_EXCLUDE,
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
    monkeypatch.setattr(
        checkpoints, "_validate_adapter_tensor_values", lambda *args, **kwargs: None
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

    stamp_adapter_dir_provenance(
        str(tmp_path), _MODEL_ID, "c" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_parameters"] == _TARGETS
    assert saved["target_modules"] == "all-linear"
    assert saved["r"] == 32
    assert saved["lora_alpha"] == 64
    assert saved["base_model_name_or_path"] == _MODEL_ID
    assert saved["revision"] == "c" * 40
    assert saved["flash_provenance"] == {"source": "verl"}


def _fused_artifact_with_visual_pair(monkeypatch, tmp_path):
    """stage a fused-expert artifact that also carries one complete visual LoRA pair."""
    import flash.engine.worker.verl.checkpoints as checkpoints

    tensors = _complete_expert_tensors()
    tensors.update(
        _wrapper_tensors(
            "base_model.model.visual.blocks.0.attn.proj",
            ((32, 2048), (2048, 32)),
        )
    )
    monkeypatch.setattr(checkpoints, "_read_adapter_tensor_metadata", lambda _path: tensors)
    monkeypatch.setattr(
        checkpoints, "_validate_adapter_tensor_values", lambda *args, **kwargs: None
    )
    # peft expands `all-linear` into the concrete module list at model-creation time, and the
    # exporter re-canonicalizes it back to the shorthand only after validation. so the config a
    # real multimodal export presents names the vision suffix alongside the language modules.
    # declaring only the language ones would make the validator reject for an undeclared ordinary
    # target rather than for the modality property these two tests are about.
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "experts", "base_layer", "proj"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)
    return checkpoints


def test_fused_export_rejects_complete_visual_pair_before_stamping(monkeypatch, tmp_path):
    checkpoints = _fused_artifact_with_visual_pair(monkeypatch, tmp_path)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    # a text-only run carries the language-prefix exclude regex, so it never targeted the vision
    # tower: a visual pair in its artifact is contamination and must not be stamped.
    with pytest.raises(RuntimeError, match="contains non-language tensor"):
        checkpoints.stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


def test_fused_export_accepts_the_visual_pair_a_multimodal_run_trained(monkeypatch, tmp_path):
    """the same artifact is a healthy export when the run actually targeted the vision tower.

    a multimodal run gets no exclude regex, so `all-linear` covers the vision linears and the
    merger writes them. the fused topology check describes the language stack only, so those
    tensors are separated out rather than read as an incomplete expert export.
    """
    checkpoints = _fused_artifact_with_visual_pair(monkeypatch, tmp_path)

    checkpoints.stamp_adapter_dir_provenance(
        str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=None
    )

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"
    # the key is written as an explicit null rather than omitted: its presence is the modality
    # marker warm-start validation reads, and peft 0.19.1 resolves an explicit null and an absent
    # key to the same `exclude_modules=None`, so stating it costs nothing at load time.
    assert saved["exclude_modules"] is None
    assert saved["base_model_name_or_path"] == _MODEL_ID


def test_fused_export_rejects_a_tensor_no_lora_pair_claims(monkeypatch, tmp_path):
    """an unpaired tensor is published without ever being value-checked.

    `fused_expert_lora_tensor_pairs` filters to canonical `.lora_A/.lora_B.` keys, so a
    `modules_to_save` entry or a bare bias key rides along unvalidated. peft loads the saved state
    dict with `strict=False`, so such a key whose name matches the base model is restored OVER the
    base weights at warm start and serve.
    """
    import flash.engine.worker.verl.checkpoints as checkpoints

    tensors = _complete_expert_tensors()
    tensors["base_model.model.layers.0.mlp.gate_proj.weight"] = (2048, 2048)
    monkeypatch.setattr(checkpoints, "_read_adapter_tensor_metadata", lambda _path: tensors)
    monkeypatch.setattr(
        checkpoints, "_validate_adapter_tensor_values", lambda *args, **kwargs: None
    )
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "experts", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(RuntimeError, match="unpaired tensor"):
        checkpoints.stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


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
        stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


@pytest.mark.parametrize(
    ("tensor_mode", "message"),
    [
        ("nonfinite", "contains non-finite values"),
        ("all_zero", "no nonzero composed LoRA delta"),
    ],
)
def test_fused_export_validates_actual_tensor_values_before_stamping(
    monkeypatch, tmp_path, tensor_mode, message
):
    import flash.engine.worker.verl.checkpoints as checkpoints

    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode=tensor_mode, text_rank=1)

    def pair_keys(tensors, _config, _model_id):
        a_key = next(key for key in tensors if ".lora_A." in key)
        b_key = next(key for key in tensors if ".lora_B." in key)
        return {"q_proj": (a_key, b_key)}

    monkeypatch.setattr(checkpoints, "fused_expert_lora_tensor_pairs", pair_keys)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(RuntimeError, match=message):
        checkpoints.stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


def test_fused_export_accepts_a_finite_nonzero_actual_payload(monkeypatch, tmp_path):
    import flash.engine.worker.verl.checkpoints as checkpoints

    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj", "v_proj"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode="mixed", text_rank=1)

    def pair_keys(tensors, _config, _model_id):
        groups = {}
        for key in tensors:
            module, factor = key.rsplit(".lora_", 1)
            groups.setdefault(module, {})[factor[0]] = key
        return {module: (factors["A"], factors["B"]) for module, factors in groups.items()}

    monkeypatch.setattr(checkpoints, "fused_expert_lora_tensor_pairs", pair_keys)

    checkpoints.stamp_adapter_dir_provenance(
        str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"
    assert saved["target_parameters"] == _TARGETS


def _merger_expert_tensors(*, include_ordinary=True, drop_last_layer_rung=False):
    """Build the tensor map the pinned verl merger actually writes.

    Pinned verl's merger runs ``name.replace(".default.weight", ".weight")`` before ``save_file``,
    and PEFT's own ``save_and_load`` strips the adapter name the same way, so no adapter namespace
    reaches disk.

    Also models the `shared_expert*` linears an `all-linear` run picks up, which is what makes the
    real artifact's expert-ish tensor count 480 (160 routed-fused + 240 shared MLP + 80 gate)
    rather than the 160 routed-fused rungs the topology check counts.
    """
    leaves = ("weight", "weight")
    tensors = {}
    for layer in range(40):
        owner = f"base_model.model.layers.{layer}.mlp.experts"
        first, second = _EXPECTED_PAIRS
        if drop_last_layer_rung and layer == 39:
            continue
        tensors.update(_wrapper_tensors(owner, first, leaves))
        tensors.update(_wrapper_tensors(f"{owner}.base_layer", second, leaves))
        if include_ordinary:
            for shared in ("gate_proj", "up_proj", "down_proj"):
                tensors.update(
                    _wrapper_tensors(
                        f"base_model.model.layers.{layer}.mlp.shared_expert.{shared}",
                        ((32, 2048), (2048, 32)),
                        leaves,
                    )
                )
    if include_ordinary:
        tensors.update(_ordinary_tensors(factor_leaves=leaves))
    return tensors


def _merger_config(**overrides):
    """The config that accompanies `_merger_expert_tensors()` on disk.

    peft resolves ``target_modules="all-linear"`` into the concrete module list at model-creation
    time (`_maybe_include_all_linear_layers` assigns `peft_config.target_modules`), and the exporter
    only re-canonicalizes it back to the shorthand *after* validation
    (`checkpoints.stamp_adapter_dir_provenance`). So the config the validator sees names every
    targeted module -- including the `shared_expert` linears this fixture models. Declaring only
    `q_proj` against a 402-tensor artifact is a pair no real run produces, and it makes the
    validator reject for a config mismatch rather than for the property under test.
    """
    config = _valid_config(
        target_modules=["q_proj", "gate_proj", "up_proj", "down_proj"],
    )
    config.update(overrides)
    return config


def test_fused_completeness_rejects_mixed_namespaced_and_bare_keys():
    """A namespaced key is invalid even when canonical bare keys are also present."""
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _merger_expert_tensors()
    stale = {
        key.replace(".lora_A.weight", ".lora_A.other.weight").replace(
            ".lora_B.weight", ".lora_B.other.weight"
        ): shape
        for key, shape in _ordinary_tensors(
            target="v_proj", factor_leaves=("weight", "weight")
        ).items()
    }
    tensors.update(stale)

    assert not has_complete_fused_expert_tensors(tensors, _merger_config(), _MODEL_ID)


@pytest.mark.parametrize(
    "smuggled",
    [
        {"base_model.model.layers.0.self_attn.q_proj.lora_A.bias": (32, 2048)},
        {"attacker.layers.0.q_proj.lora_A.weight": (32, 2048)},
    ],
)
def test_fused_completeness_still_rejects_unparseable_keys(smuggled):
    """Any tensor outside the canonical LoRA grammar rejects the whole adapter."""
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _merger_expert_tensors()
    assert has_complete_fused_expert_tensors(tensors, _merger_config(), _MODEL_ID)
    tensors.update(smuggled)

    assert not has_complete_fused_expert_tensors(tensors, _merger_config(), _MODEL_ID)


def test_fused_completeness_rejects_a_genuinely_incomplete_merger_artifact():
    """SABOTAGE GUARD. The fix must not turn the check into one that accepts anything: a real
    merger artifact missing a layer's rungs is still incomplete."""
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    assert has_complete_fused_expert_tensors(_merger_expert_tensors(), _merger_config(), _MODEL_ID)
    tensors = _merger_expert_tensors(drop_last_layer_rung=True)

    assert not has_complete_fused_expert_tensors(tensors, _merger_config(), _MODEL_ID)


def test_fused_export_stamps_a_real_namespace_free_artifact_end_to_end(tmp_path):
    """The reported failure, reproduced through the real export boundary.

    Unlike `_patch_export_metadata`, this writes an ACTUAL safetensors file and lets
    `stamp_adapter_dir_provenance` read its real header -- no monkeypatched metadata reader and no
    no-op value validator. Before the namespace fix this raised
    `does not contain complete fused expert LoRA weights`.
    """
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    payload = {}
    for key, shape in _merger_expert_tensors().items():
        array = np.zeros(shape, dtype=np.float16)
        array[0, 0] = 1.0
        payload[key] = array
    _write_small_safetensors(tmp_path / "adapter_model.safetensors", payload)
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 32,
                "lora_alpha": 64,
                # the synthetic `experts`/`base_layer` entries are stripped by
                # `normalize_verl_fused_expert_export`; the rest is the resolved all-linear list
                # peft wrote, which must name every ordinary module the artifact carries.
                "target_modules": [
                    "q_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "experts",
                    "base_layer",
                ],
                "target_parameters": None,
            }
        ),
        encoding="utf-8",
    )

    stamp_adapter_dir_provenance(
        str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_parameters"] == _TARGETS
    assert saved["target_modules"] == "all-linear"
    assert saved["base_model_name_or_path"] == _MODEL_ID


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
        stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

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
        stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


def test_export_rejects_empty_exclude_modules_without_writing(tmp_path):
    """an empty regex would validate as text-only and then persist as multimodal.

    modality is read as ``exclude_modules is None`` but was written as ``exclude_modules or None``,
    so `""` passed the text-only tensor checks and then stamped the config a warm start reads back
    as multimodal. the two readings must not be able to disagree.
    """
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": list(_TARGETS),
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(RuntimeError, match="non-empty regex or None"):
        stamp_adapter_dir_provenance(str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules="")

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
        stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

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

    stamp_adapter_dir_provenance(
        str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

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


def test_non_moe_export_rejects_a_fully_absent_declared_target(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    tensors = _text_pair(
        "self_attn.q_proj",
        np.ones((1, 2), dtype=np.float16),
        np.ones((2, 1), dtype=np.float16),
    )
    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj", "k_proj"],
    }
    _write_small_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    before = config_path.read_bytes()

    with pytest.raises(
        RuntimeError,
        match=r"has no tensors for declared target_modules \['k_proj'\]",
    ):
        stamp_adapter_dir_provenance(
            str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


def test_non_moe_export_accepts_complete_pairs_for_every_declared_target(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    factor_a = np.ones((1, 2), dtype=np.float16)
    factor_b = np.ones((2, 1), dtype=np.float16)
    tensors = {
        **_text_pair("self_attn.q_proj", factor_a, factor_b),
        **_text_pair("self_attn.k_proj", factor_a, factor_b),
    }
    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj", "k_proj"],
    }
    _write_small_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    stamp_adapter_dir_provenance(
        str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"


def test_non_moe_export_rejects_a_pair_whose_outer_dimension_disagrees(tmp_path):
    """rank and composability do not constrain the base module's own width.

    both pairs below are internally consistent -- rank 1, and B @ A composes -- so every existing
    check passes. but two tensors targeting the same `q_proj` suffix claim different widths, so at
    most one of them can match the base module. without this check the export publishes and only
    fails later at peft or vllm load, with no provenance back to the run that wrote it.

    scope: this is cross-pair consistency, not a base-model check. a suffix whose pairs are
    UNIFORMLY wrong, or one carrying a single pair, still passes -- catching those needs the base
    model's own dims, which this boundary does not have.
    """
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    tensors = {
        **_text_pair(
            "self_attn.q_proj",
            np.ones((1, 2), dtype=np.float16),
            np.ones((2, 1), dtype=np.float16),
        ),
    }
    wide = "base_model.model.layers.1.self_attn.q_proj"
    tensors[f"{wide}.lora_A.weight"] = np.ones((1, 4), dtype=np.float16)
    tensors[f"{wide}.lora_B.weight"] = np.ones((4, 1), dtype=np.float16)
    config = {"peft_type": "LORA", "r": 1, "target_modules": ["q_proj"]}
    _write_small_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    before = config_path.read_bytes()

    with pytest.raises(RuntimeError, match="outer dimension"):
        stamp_adapter_dir_provenance(
            str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


def test_multimodal_export_accepts_a_suffix_shared_across_both_stacks(tmp_path):
    """a VL model's vision tower and language model share leaf names at DIFFERENT widths.

    a multimodal run carries no exclude regex, so `all-linear` covers both stacks and the merger
    writes `...language_model...mlp.down_proj` (text intermediate) alongside
    `...visual.blocks.N.mlp.down_proj` (vision intermediate). those are two different base modules
    that legitimately disagree on outer dimension. keying the width check by the bare target suffix
    read that as corruption and failed a HEALTHY image export at publish time, after the paid run
    had already finished -- worse than the load-time failure the check exists to prevent.

    the synthetic widths deliberately differ between the text and vision stacks.
    """
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    def stack_pair(prefix, hidden, intermediate):
        return {
            f"{prefix}.lora_A.weight": np.ones((1, intermediate), dtype=np.float16),
            f"{prefix}.lora_B.weight": np.ones((hidden, 1), dtype=np.float16),
        }

    tensors = {
        **stack_pair("base_model.model.model.language_model.layers.0.mlp.down_proj", 1024, 3584),
        **stack_pair("base_model.model.model.language_model.layers.1.mlp.down_proj", 1024, 3584),
        **stack_pair("base_model.model.model.visual.blocks.0.mlp.down_proj", 768, 3072),
    }
    config = {"peft_type": "LORA", "r": 1, "target_modules": ["down_proj"]}
    _write_small_safetensors(tmp_path / "adapter_model.safetensors", tensors)
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=None)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == "all-linear"
    assert saved["exclude_modules"] is None


def test_non_moe_export_preserves_the_orphan_pair_rejection_message(tmp_path):
    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 1,
        "target_modules": ["q_proj", "v_proj"],
    }
    _write_expert_adapter(tmp_path, config=config, tensor_mode="orphan_a", text_rank=1)

    with pytest.raises(RuntimeError) as error:
        stamp_adapter_dir_provenance(
            str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert str(error.value) == (
        "exported text adapter must contain at least one complete LoRA A/B pair and no orphan "
        "factors; incomplete_modules="
        "['base_model.model.layers.0.self_attn.q_proj']"
    )


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

    stamp_adapter_dir_provenance(
        str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

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
        stamp_adapter_dir_provenance(
            str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

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
        stamp_adapter_dir_provenance(
            str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

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
        stamp_adapter_dir_provenance(
            str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )


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

    stamp_adapter_dir_provenance(
        str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

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

    stamp_adapter_dir_provenance(
        str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

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
        stamp_adapter_dir_provenance(
            str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

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
        stamp_adapter_dir_provenance(
            str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before_config
    assert [path.read_bytes() for path in weight_paths] == before_weights


def test_strict_worker_accepts_current_config_without_changing_memory_or_disk(
    monkeypatch, tmp_path
):
    _import_worker(monkeypatch)
    _patch_worker_metadata(monkeypatch)
    config = _valid_config()
    _write_expert_adapter(tmp_path, config=config)
    before_config = copy.deepcopy(config)
    before_file = (tmp_path / "adapter_config.json").read_bytes()

    worker_adapter.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path), _TEXT_TARGETING)

    assert config == before_config
    assert (tmp_path / "adapter_config.json").read_bytes() == before_file


def test_strict_worker_rejects_missing_targets_without_changing_memory_or_disk(
    monkeypatch, tmp_path
):
    _import_worker(monkeypatch)
    config = _valid_config(target_parameters=None)
    _write_expert_adapter(tmp_path, config=config)
    before_config = copy.deepcopy(config)
    before_file = (tmp_path / "adapter_config.json").read_bytes()

    with pytest.raises(ValueError, match="omits required expert targets"):
        worker_adapter.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path), _TEXT_TARGETING)

    assert config == before_config
    assert (tmp_path / "adapter_config.json").read_bytes() == before_file


@pytest.mark.parametrize("synthetic", ["experts", "base_layer"])
def test_strict_worker_rejects_synthetic_modules_without_changing_memory_or_disk(
    monkeypatch, tmp_path, synthetic
):
    _import_worker(monkeypatch)
    config = _valid_config(target_modules=["q_proj", synthetic])
    _write_expert_adapter(tmp_path, config=config)
    before_config = copy.deepcopy(config)
    before_file = (tmp_path / "adapter_config.json").read_bytes()

    with pytest.raises(ValueError, match="invalid synthetic target_modules"):
        worker_adapter.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path), _TEXT_TARGETING)

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
    _import_worker(monkeypatch)
    config = _valid_config()
    _write_expert_adapter(tmp_path, config=config, tensor_mode=tensor_mode)
    before_config = copy.deepcopy(config)
    before_config_file = (tmp_path / "adapter_config.json").read_bytes()
    before_weights = (tmp_path / "adapter_model.safetensors").read_bytes()

    with pytest.raises(ValueError, match=message):
        worker_adapter.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path), _TEXT_TARGETING)

    assert config == before_config
    assert (tmp_path / "adapter_config.json").read_bytes() == before_config_file
    assert (tmp_path / "adapter_model.safetensors").read_bytes() == before_weights


def test_strict_worker_rejects_structurally_compatible_wrong_shapes(monkeypatch, tmp_path):
    import flash.engine.worker.model.adapter as adapter

    _import_worker(monkeypatch)
    counterexample = _complete_expert_tensors(pairs=(((7, 1), (2, 7)), ((7, 1), (2, 7))))
    monkeypatch.setattr(adapter, "_read_adapter_tensor_metadata", lambda _path: counterexample)
    config = _valid_config()
    _write_expert_adapter(tmp_path, config=config)
    before = (tmp_path / "adapter_config.json").read_bytes()

    with pytest.raises(ValueError, match="complete fused expert LoRA weights"):
        worker_adapter.validate_warmstart_adapter(config, _MODEL_ID, str(tmp_path), _TEXT_TARGETING)

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
        checkpoints.stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


def test_tensor_analyzer_accepts_canonical_qwen36_rung_geometry():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    assert has_complete_fused_expert_tensors(_complete_expert_tensors(), _valid_config(), _MODEL_ID)


@pytest.mark.parametrize(
    "module",
    [
        "visual.blocks.0.attn.proj",
        "vision_tower.blocks.0.proj",
        "multi_modal_projector.linear",
        "patch_embed.proj",
        "mtp.layers.0.proj",
    ],
)
def test_text_only_export_rejects_complete_non_language_pairs(monkeypatch, tmp_path, module):
    """a text-only run never targets these modules, so their tensors must not reach publish.

    the check lives at the export boundary rather than in `has_complete_fused_expert_tensors`,
    because only the boundary knows the run's modality: the same tensor is contamination in a
    text-only export and a trained weight in a multimodal one.
    """
    import flash.engine.worker.verl.checkpoints as checkpoints

    tensors = _complete_expert_tensors()
    tensors.update(
        _wrapper_tensors(
            f"base_model.model.{module}",
            ((32, 2048), (2048, 32)),
        )
    )
    monkeypatch.setattr(checkpoints, "_read_adapter_tensor_metadata", lambda _path: tensors)
    monkeypatch.setattr(checkpoints, "_validate_adapter_tensor_values", lambda *a, **k: None)
    _write_expert_adapter(tmp_path, config=_valid_config())
    config_path = tmp_path / "adapter_config.json"
    before = config_path.read_bytes()

    with pytest.raises(RuntimeError, match="contains non-language tensor"):
        checkpoints.stamp_adapter_dir_provenance(
            str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
        )

    assert config_path.read_bytes() == before


def test_fused_multimodal_export_keeps_vision_tensors_as_ordinary_evidence():
    """a declared vision target must be satisfiable, and its tensors must still be validated.

    a vision module never matches a fused expert owner, so it falls through the fused-rung walk on
    its own and reaches `_has_ordinary_evidence` -- which requires evidence for every declared
    ordinary target and applies the A/B, rank, and shape checks. skipping such tensors anywhere in
    this function would fail a healthy multimodal export whose merger config names a vision suffix,
    and would leave those tensors unchecked.
    """
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    vision = "base_model.model.visual.blocks.0.attn.qkv"
    config = _merger_config(target_modules=["q_proj", "gate_proj", "up_proj", "down_proj", "qkv"])

    healthy = _merger_expert_tensors()
    healthy.update(_wrapper_tensors(vision, ((32, 2048), (2048, 32))))
    assert has_complete_fused_expert_tensors(healthy, config, _MODEL_ID)

    # the vision pair is ordinary evidence, so a rank that disagrees with the config rejects.
    bad_rank = _merger_expert_tensors()
    bad_rank.update(_wrapper_tensors(vision, ((7, 2048), (2048, 7))))
    assert not has_complete_fused_expert_tensors(bad_rank, config, _MODEL_ID)

    # and an orphan factor rejects, exactly as it would for a language module.
    orphan = _merger_expert_tensors()
    orphan[f"{vision}.lora_A.weight"] = (32, 2048)
    assert not has_complete_fused_expert_tensors(orphan, config, _MODEL_ID)


@pytest.mark.parametrize(
    "module",
    [
        "visual.blocks.0.attn.proj",
        "vision_tower.blocks.0.proj",
        "multi_modal_projector.linear",
        "patch_embed.proj",
        "mtp.layers.0.proj",
    ],
)
def test_fused_topology_check_requires_a_declared_target_for_a_non_language_pair(module):
    """a non-language pair is ordinary evidence, so it must be declared like any other module.

    these modules are not expert rungs, so the fused walk classifies them as "elsewhere" and they
    go through `_has_ordinary_evidence`. an undeclared one therefore rejects -- the same verdict a
    stray undeclared language module gets -- while a declared one is accepted and fully validated
    (see `test_fused_multimodal_export_keeps_vision_tensors_as_ordinary_evidence`).
    """
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    tensors = _complete_expert_tensors()
    tensors.update(_wrapper_tensors(f"base_model.model.{module}", ((32, 2048), (2048, 32))))

    # `_valid_config()` declares only `q_proj`, so this pair is undeclared.
    assert not has_complete_fused_expert_tensors(tensors, _valid_config(), _MODEL_ID)

    # declaring its suffix makes the same artifact a healthy multimodal export.
    declared = _valid_config(target_modules=["q_proj", module.rsplit(".", 1)[-1]])
    assert has_complete_fused_expert_tensors(tensors, declared, _MODEL_ID)


def test_tensor_analyzer_rejects_unparsed_and_undeclared_ordinary_tensors():
    from flash.adapters.fused_experts import has_complete_fused_expert_tensors

    malformed = _complete_expert_tensors()
    malformed["base_model.model.layers.0.self_attn.q_proj.lora_A.bias"] = (1,)
    assert not has_complete_fused_expert_tensors(malformed, _valid_config(), _MODEL_ID)

    arbitrary_root = _complete_expert_tensors()
    arbitrary_root.update(_wrapper_tensors("attacker.layers.0.q_proj", ((32, 2048), (2048, 32))))
    assert not has_complete_fused_expert_tensors(arbitrary_root, _valid_config(), _MODEL_ID)

    undeclared = _complete_expert_tensors()
    undeclared.update(_ordinary_tensors(target="v_proj"))
    assert not has_complete_fused_expert_tensors(undeclared, _valid_config(), _MODEL_ID)


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
            checkpoints.stamp_adapter_dir_provenance(
                str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
            )

        expected_error = RuntimeError
    else:
        import flash.engine.worker.model.adapter as adapter

        _import_worker(monkeypatch)
        monkeypatch.setattr(adapter, "_read_adapter_tensor_metadata", lambda _path: tensors)
        config = _valid_config()

        def validate():
            worker_adapter.validate_warmstart_adapter(
                config, _MODEL_ID, str(tmp_path), _TEXT_TARGETING
            )

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


@pytest.mark.parametrize(
    ("exclude_modules", "stamped_multimodal"),
    [(_TEXT_ONLY_EXCLUDE, False), (None, True)],
)
def test_stamped_modality_marker_survives_into_warmstart(
    monkeypatch, tmp_path, exclude_modules, stamped_multimodal
):
    """The stamper's explicit modality marker must drive warm-start validation."""
    import flash.engine.worker.model.adapter as adapter_module

    stamp_adapter_dir_provenance = _patch_export_metadata(monkeypatch)
    _write_expert_adapter(
        tmp_path,
        config={
            "peft_type": "LORA",
            "r": 32,
            "lora_alpha": 64,
            "target_modules": "all-linear",
            "target_parameters": None,
            "flash_provenance": {"source": "verl"},
        },
    )

    stamp_adapter_dir_provenance(
        str(tmp_path), _MODEL_ID, "d" * 40, exclude_modules=exclude_modules
    )

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    # present in both directions so every artifact states its modality.
    assert "exclude_modules" in saved
    assert (saved["exclude_modules"] is None) is stamped_multimodal

    # and the reader classifies it from the marker, never by inspecting tensor values.
    def unexpected_tensor_read(_path):
        raise AssertionError("a stamped adapter must not infer modality from tensor values")

    monkeypatch.setattr(adapter_module, "_read_adapter_tensor_metadata", unexpected_tensor_read)
    _patch_worker_metadata(monkeypatch)
    matching = resolve_lora_targeting(_MODEL_ID, algorithm="sft", multimodal=stamped_multimodal)
    adapter_module.validate_warmstart_adapter(saved, _MODEL_ID, str(tmp_path), matching)

    # the opposite-modality run must be rejected rather than silently allowed through.
    opposing = resolve_lora_targeting(_MODEL_ID, algorithm="sft", multimodal=not stamped_multimodal)
    with pytest.raises(ValueError, match="warm-start modality mismatch"):
        adapter_module.validate_warmstart_adapter(saved, _MODEL_ID, str(tmp_path), opposing)


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
    monkeypatch.setattr(
        checkpoints, "_validate_adapter_tensor_values", lambda *args, **kwargs: None
    )
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "v_proj", "experts", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)

    checkpoints.stamp_adapter_dir_provenance(
        str(tmp_path), _MODEL_ID, "e" * 40, exclude_modules=_TEXT_ONLY_EXCLUDE
    )

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_parameters"] == _TARGETS
    assert saved["base_model_name_or_path"] == _MODEL_ID


def test_warmstart_accepts_an_adapter_serialized_without_the_adapter_namespace(
    monkeypatch, tmp_path
):
    import flash.engine.worker.model.adapter as adapter

    _import_worker(monkeypatch)
    tensors = _complete_expert_tensors()
    monkeypatch.setattr(adapter, "_read_adapter_tensor_metadata", lambda _path: tensors)

    worker_adapter.validate_warmstart_adapter(
        _valid_config(), _MODEL_ID, str(tmp_path), _TEXT_TARGETING
    )


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
