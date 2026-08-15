"""Fused-expert LoRA export and warm-start recovery tests."""

import sys

import pytest


def _import_worker(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as worker

    return worker


def _write_expert_adapter(directory, *, config, expert_tensors=True):
    """write an adapter dir shaped like one of verl's exports.

    the safetensors header is written by hand rather than through `safetensors.torch`: the offline
    test job has no torch, and the code under test only ever reads tensor KEYS out of the header.
    """
    import json
    import struct

    keys = ["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    if expert_tensors:
        # peft names a wrapped nn.parameter after its owning module, never after the parameter, and
        # nests the second wrapper under `base_layer`. both lora factors are written for each: a
        # delta is `b @ a`, so a wrapper carrying only `lora_a` never trained.
        keys += [
            "base_model.model.layers.0.mlp.experts.lora_A.default.weight",
            "base_model.model.layers.0.mlp.experts.lora_B.default.weight",
            "base_model.model.layers.0.mlp.experts.base_layer.lora_A.default.weight",
            "base_model.model.layers.0.mlp.experts.base_layer.lora_B.default.weight",
        ]
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    encoded = json.dumps(header).encode("utf-8")
    (directory / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\x01\x02"
    )
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")


def test_verl_export_restores_the_fused_expert_targeting_it_drops(tmp_path):
    """the exported adapter must be loadable back, not merely present.

    verl rebuilds adapter_config.json from rank/alpha/target_modules, so it drops
    `target_parameters` AND flattens the wrapped expert modules into `target_modules` as
    `experts`/`base_layer`. peft cannot bind those names, so an unrepaired export fails to load
    even before the warm-start validator sees it.
    """
    import json

    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    _write_expert_adapter(
        tmp_path,
        config={
            "peft_type": "LORA",
            "r": 32,
            "target_modules": ["q_proj", "experts", "base_layer"],
            "target_parameters": None,
        },
    )
    stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.6-35B-A3B", "c" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_parameters"] == [
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    ]
    # the synthesized names are gone; the real module peft targeted survives.
    assert saved["target_modules"] == ["q_proj"]
    assert saved["base_model_name_or_path"] == "Qwen/Qwen3.6-35B-A3B"


def test_export_refuses_to_stamp_incomplete_expert_weights_as_compatible(tmp_path):
    """the exporter must prove coverage before writing the full target list.

    otherwise a partial merger output is mislabeled as complete, and the worker's early return for
    an already-declared target list has no evidence left that the declaration was synthesized.
    """
    import json
    import struct

    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "experts", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)
    path = tmp_path / "adapter_model.safetensors"
    # leave both factors of the outer wrapper but delete the nested wrapper entirely.
    keys = _wrapper_tensors("base_model.model.layers.0.mlp.experts")
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x01\x02")

    with pytest.raises(RuntimeError, match="complete fused expert LoRA weights"):
        stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.6-35B-A3B", "d" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_parameters"] is None


def test_non_moe_export_is_untouched_by_the_expert_repair(tmp_path):
    import json

    from flash.engine.worker.verl.checkpoints import stamp_adapter_dir_provenance

    config = {"peft_type": "LORA", "r": 32, "target_modules": ["q_proj", "v_proj"]}
    _write_expert_adapter(tmp_path, config=config, expert_tensors=False)
    stamp_adapter_dir_provenance(str(tmp_path), "Qwen/Qwen3.5-9B", "d" * 40)

    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == ["q_proj", "v_proj"]
    assert saved.get("target_parameters") is None


def test_warmstart_recovers_expert_targets_from_the_adapters_own_weights(monkeypatch, tmp_path):
    """an adapter exported before the fix trained the experts; its config just forgot to say so.

    rejecting it stranded every 35B adapter this pipeline had ever produced, including ones that
    deploy and serve, behind advice ("retrain with the current Flash version") that reproduced the
    same file. the weights are the ground truth.
    """
    import json

    worker = _import_worker(monkeypatch)
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "experts", "base_layer"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config)

    worker.prepare_warmstart_adapter_config(config, "Qwen/Qwen3.6-35B-A3B", str(tmp_path))

    assert set(config["target_parameters"]) == {
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    }
    # verl re-reads the directory itself (model.lora_adapter_path -> peftmodel.from_pretrained),
    # so the repair is worthless unless it reaches the file on disk.
    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert set(saved["target_parameters"]) == {
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
    }
    assert saved["target_modules"] == ["q_proj"]


def test_warmstart_still_rejects_an_adapter_that_never_trained_the_experts(monkeypatch, tmp_path):
    """the recovery must not become a rubber stamp: no expert tensors, no warm start."""
    worker = _import_worker(monkeypatch)
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj"],
        "target_parameters": None,
    }
    _write_expert_adapter(tmp_path, config=config, expert_tensors=False)

    with pytest.raises(ValueError, match="omits required expert targets"):
        worker.prepare_warmstart_adapter_config(config, "Qwen/Qwen3.6-35B-A3B", str(tmp_path))

    # and with no directory to read, there is no evidence to recover from either.
    with pytest.raises(ValueError, match="omits required expert targets"):
        worker.prepare_warmstart_adapter_config(dict(config), "Qwen/Qwen3.6-35B-A3B", None)


@pytest.mark.parametrize(
    "present",
    [
        pytest.param(
            [
                "base_model.model.layers.0.mlp.experts.lora_A.default.weight",
                "base_model.model.layers.0.mlp.experts.lora_B.default.weight",
            ],
            id="outer-only",
        ),
        pytest.param(
            [
                "base_model.model.layers.0.mlp.experts.base_layer.lora_A.default.weight",
                "base_model.model.layers.0.mlp.experts.base_layer.lora_B.default.weight",
            ],
            id="inner-only",
        ),
        pytest.param(
            [
                "base_model.model.layers.0.mlp.experts.lora_A.default.weight",
                "base_model.model.layers.0.mlp.experts.base_layer.lora_A.default.weight",
            ],
            id="both-wrappers-but-no-lora-B",
        ),
    ],
)
def test_warmstart_rejects_an_adapter_that_trained_only_one_expert_parameter(
    monkeypatch, tmp_path, present
):
    """a truncated adapter must not be recovered into claiming it trained both parameters.

    peft NESTS its wrappers, so targeting two parameters on one module yields two distinct lora
    paths. accepting a single one as proof of both would rewrite the config to declare the full
    target set, and peft would then freshly initialize the parameter that never trained -- a
    silently wrong warm start rather than a failed one.

    the third case is the same failure one level down: both wrappers are present but neither has a
    `lora_B`. a delta is `B @ A`, so peft initializes the missing matrix to zero and the parameter
    contributes nothing, which the wrapper count alone cannot see.
    """
    import json
    import struct

    worker = _import_worker(monkeypatch)
    keys = ["base_model.model.layers.0.self_attn.q_proj.lora_A.weight", *present]
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    encoded = json.dumps(header).encode("utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\x01\x02"
    )
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "experts", "base_layer"],
        "target_parameters": None,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="omits required expert targets"):
        worker.prepare_warmstart_adapter_config(config, "Qwen/Qwen3.6-35B-A3B", str(tmp_path))

    # the rejected adapter's config is left exactly as found: no partial repair written back.
    on_disk = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert on_disk["target_parameters"] is None


def _expert_keys(tmp_path, keys):
    import json
    import struct

    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    encoded = json.dumps(header).encode("utf-8")
    (tmp_path / "adapter_model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\x01\x02"
    )


def _wrapper_tensors(prefix):
    """both lora factors for one wrapper, spelled the way peft saves them."""
    return [f"{prefix}.lora_A.default.weight", f"{prefix}.lora_B.default.weight"]


def test_expert_tensor_coverage_is_counted_per_layer_not_pooled(monkeypatch, tmp_path):
    """a wrapper count pooled across the adapter hides a layer that trained only one parameter.

    layer 0 holding the outer wrapper and layer 1 the inner one unions to a complete-looking set
    while NEITHER layer trained both fused parameters. recovering that adapter would declare both
    targets and let peft initialize the untrained half.
    """
    worker = _import_worker(monkeypatch)
    _expert_keys(
        tmp_path,
        _wrapper_tensors("base_model.model.layers.0.mlp.experts")
        + _wrapper_tensors("base_model.model.layers.1.mlp.experts.base_layer"),
    )
    assert not worker.adapter_has_fused_expert_tensors(str(tmp_path), "Qwen/Qwen3.6-35B-A3B")

    # every layer complete is accepted, and one degraded layer is enough to reject.
    complete = []
    for layer in range(3):
        owner = f"base_model.model.layers.{layer}.mlp.experts"
        complete += _wrapper_tensors(owner) + _wrapper_tensors(f"{owner}.base_layer")
    _expert_keys(tmp_path, complete)
    assert worker.adapter_has_fused_expert_tensors(str(tmp_path), "Qwen/Qwen3.6-35B-A3B")

    _expert_keys(
        tmp_path,
        [k for k in complete if "layers.1.mlp.experts.base_layer" not in k],
    )
    assert not worker.adapter_has_fused_expert_tensors(str(tmp_path), "Qwen/Qwen3.6-35B-A3B")

    # an entire expert instance can be absent while the rest of that layer's adapter tensors remain.
    only_layer_zero_experts = [k for k in complete if "layers.1.mlp.experts" not in k]
    only_layer_zero_experts += _wrapper_tensors("base_model.model.layers.1.self_attn.q_proj")
    _expert_keys(tmp_path, only_layer_zero_experts)
    assert not worker.adapter_has_fused_expert_tensors(str(tmp_path), "Qwen/Qwen3.6-35B-A3B")


def test_declared_targets_still_get_the_unloadable_module_names_stripped(monkeypatch, tmp_path):
    """declaring the targets is not enough: peft cannot bind `experts` either way.

    a config carrying the full target list AND verl's synthesized module names passed validation
    and then died inside verl's own `PeftModel.from_pretrained` on `Target module Experts() is not
    supported`. our exports are stripped at publish, so this is about a config that reached the
    worker by some other route -- accepting it here just moves the failure somewhere less legible.
    """
    import json
    import os

    worker = _import_worker(monkeypatch)
    config = {
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "experts", "base_layer"],
        "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    }
    _write_expert_adapter(tmp_path, config=config)

    worker.prepare_warmstart_adapter_config(config, "Qwen/Qwen3.6-35B-A3B", str(tmp_path))

    assert config["target_modules"] == ["q_proj"]
    saved = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    assert saved["target_modules"] == ["q_proj"]
    # the rewrite is atomic: no temporary file is left beside the config.
    assert not [name for name in os.listdir(tmp_path) if name.endswith(".tmp")]


def test_expert_tensor_coverage_matches_the_full_owner_path(monkeypatch, tmp_path):
    """`experts` as a bare segment also matches an unrelated module that merely ends in it."""
    worker = _import_worker(monkeypatch)
    _expert_keys(
        tmp_path,
        _wrapper_tensors("base_model.model.layers.0.router.experts")
        + _wrapper_tensors("base_model.model.layers.0.router.experts.base_layer"),
    )
    assert not worker.adapter_has_fused_expert_tensors(str(tmp_path), "Qwen/Qwen3.6-35B-A3B")


def test_expert_wrappers_must_be_peft_nesting_not_arbitrary_children(monkeypatch, tmp_path):
    """two unrelated children of the owner are not two nested wrappers.

    peft's nesting is a fixed ladder -- the Nth wrapper sits under N-1 `base_layer` levels -- so
    counting any distinct suffix would let `experts.foo` and `experts.bar` satisfy a two-parameter
    contract that neither of them represents.
    """
    worker = _import_worker(monkeypatch)
    _expert_keys(
        tmp_path,
        _wrapper_tensors("base_model.model.layers.0.mlp.experts.foo")
        + _wrapper_tensors("base_model.model.layers.0.mlp.experts.bar"),
    )
    assert not worker.adapter_has_fused_expert_tensors(str(tmp_path), "Qwen/Qwen3.6-35B-A3B")
