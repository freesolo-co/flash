"""offline tests: sharded adapter weights count only as a complete, index-referenced set.

peft discovers the sharded representation solely through ``adapter_model.<ext>.index.json``. every
gate that asks "is this a loadable adapter" therefore has to ask it of the file SET, and all of them
have to answer the same way -- serving accepting what the worker rejects is how an artifact gets
deployed that ``init_from_adapter`` then refuses to load.
"""

from __future__ import annotations

import json
import struct

import pytest

from flash.adapters.artifacts import (
    has_loadable_adapter_weights,
    is_adapter_weight_filename,
    loadable_adapter_weight_files,
)

_SHARDS = (
    "adapter_model-00001-of-00002.safetensors",
    "adapter_model-00002-of-00002.safetensors",
)
_INDEX = "adapter_model.safetensors.index.json"


def _safetensors_bytes(keys: list[str]) -> bytes:
    header = {key: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for key in keys}
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + b"\x01\x02"


def test_an_orphan_shard_is_not_loadable_weights():
    """the exact incomplete upload the gate exists to catch: shards, no index.

    the filename predicate still accepts the name -- it answers a different question -- so the two
    must not be confused for each other.
    """
    assert is_adapter_weight_filename(_SHARDS[0])
    assert not has_loadable_adapter_weights([_SHARDS[0]])
    assert not has_loadable_adapter_weights(_SHARDS)
    assert loadable_adapter_weight_files(_SHARDS) == []


def test_a_complete_indexed_shard_set_is_loadable():
    assert has_loadable_adapter_weights([*_SHARDS, _INDEX])
    assert loadable_adapter_weight_files([*_SHARDS, _INDEX]) == list(_SHARDS)


def test_a_complete_indexed_set_ignores_a_stale_shard_from_a_larger_total():
    """retry residue is expected, so an orphan from another total cannot mask the live set."""
    stale = "adapter_model-00001-of-00003.safetensors"
    assert loadable_adapter_weight_files([*_SHARDS, _INDEX, stale]) == list(_SHARDS)


def test_a_complete_larger_set_ignores_a_stale_shard_from_a_smaller_total():
    """candidate totals are independent: completeness, not listing order or size, selects the set."""
    live = tuple(f"adapter_model-{index:05d}-of-00003.safetensors" for index in range(1, 4))
    stale = "adapter_model-00001-of-00002.safetensors"
    assert loadable_adapter_weight_files([*live, _INDEX, stale]) == list(live)


def test_two_complete_indexed_sets_are_rejected_as_ambiguous():
    """the listing cannot reveal which complete set the index names, so serving either may be stale."""
    larger = tuple(f"adapter_model-{index:05d}-of-00003.safetensors" for index in range(1, 4))
    assert not has_loadable_adapter_weights([*_SHARDS, *larger, _INDEX])
    assert loadable_adapter_weight_files([*_SHARDS, *larger, _INDEX]) == []


def test_an_index_missing_one_of_its_shards_is_not_loadable():
    """the index names the complete set, so a listing short one member cannot be loaded either."""
    assert not has_loadable_adapter_weights([_SHARDS[0], _INDEX])


def test_a_single_file_adapter_wins_over_leftover_shards():
    """peft binds the single file when it is present, so that is what the readers must read.

    a retry uploads over the previous attempt without deleting what it no longer writes, so a stale
    shard beside the live single file is expected rather than an error.
    """
    names = ["adapter_model.safetensors", _SHARDS[0]]
    assert loadable_adapter_weight_files(names) == ["adapter_model.safetensors"]


def test_safetensors_is_preferred_over_a_stale_bin():
    names = ["adapter_model.safetensors", "adapter_model.bin"]
    assert loadable_adapter_weight_files(names) == ["adapter_model.safetensors"]


def test_an_orphan_safetensors_shard_falls_through_to_a_complete_bin():
    """an unloadable representation must not mask the loadable one in the other suffix."""
    names = [_SHARDS[0], "adapter_model.bin"]
    assert loadable_adapter_weight_files(names) == ["adapter_model.bin"]


def test_paths_are_matched_by_basename():
    """serving validates against repo paths, the worker against bare directory entries."""
    prefixed = [f"sft/run/adapter/{name}" for name in (*_SHARDS, _INDEX)]
    assert has_loadable_adapter_weights(prefixed)


def test_worker_deployable_check_accepts_a_sharded_save(tmp_path):
    """the worker publishes per-step adapters; a sharded save must not be silently skipped."""
    from flash.engine.worker.io.hf import _has_deployable_adapter

    (tmp_path / "adapter_config.json").write_text("{}")
    for name in _SHARDS:
        (tmp_path / name).write_bytes(b"\x00")
    (tmp_path / _INDEX).write_text(json.dumps({"weight_map": {}}))

    assert _has_deployable_adapter(str(tmp_path))


def test_worker_deployable_check_rejects_an_orphan_shard(tmp_path):
    from flash.engine.worker.io.hf import _has_deployable_adapter

    (tmp_path / "adapter_config.json").write_text("{}")
    (tmp_path / _SHARDS[0]).write_bytes(b"\x00")

    assert not _has_deployable_adapter(str(tmp_path))


def test_worker_key_reader_reads_every_shard(tmp_path):
    """keys live spread across the shards, so reading one file reports a partial namespace."""
    import flash.engine.worker.model.lora as lora

    first = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    second = "base_model.model.model.layers.1.mlp.up_proj.lora_B.default.weight"
    (tmp_path / _SHARDS[0]).write_bytes(_safetensors_bytes([first]))
    (tmp_path / _SHARDS[1]).write_bytes(_safetensors_bytes([second]))
    (tmp_path / _INDEX).write_text(
        json.dumps({"weight_map": {first: _SHARDS[0], second: _SHARDS[1]}})
    )

    assert lora._read_adapter_tensor_keys(str(tmp_path)) == [first, second]


def test_worker_key_reader_reports_nothing_for_an_orphan_shard(tmp_path):
    import flash.engine.worker.model.lora as lora

    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    (tmp_path / _SHARDS[0]).write_bytes(_safetensors_bytes([key]))

    assert lora._read_adapter_tensor_keys(str(tmp_path)) is None


def test_checkpoint_listing_offers_a_sharded_step():
    """listing a step the deploy path would refuse is worse than not listing it, and vice versa."""
    from flash.runner.results.checkpoints import _has_required_adapter_files

    assert _has_required_adapter_files({"adapter_config.json", *_SHARDS, _INDEX})
    assert not _has_required_adapter_files({"adapter_config.json", _SHARDS[0]})


def test_opd_resume_gate_rejects_an_orphan_shard():
    from flash.teacher.retry_contract import (
        OPD_RESUME_STATE_REQUIRED_FILES,
        opd_resume_checkpoint_complete,
    )

    state = set(OPD_RESUME_STATE_REQUIRED_FILES)
    assert opd_resume_checkpoint_complete({*state, *_SHARDS, _INDEX})
    assert not opd_resume_checkpoint_complete({*state, _SHARDS[0]})


def test_export_refuses_shards_without_an_index(tmp_path):
    """export fell back to shipping every matching shard, producing weights peft loads as a no-op."""
    from flash.serve import export

    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    (tmp_path / _SHARDS[0]).write_bytes(_safetensors_bytes([key]))

    assert export._adapter_weight_paths(tmp_path) == []


def test_serving_validation_refuses_an_orphan_shard(monkeypatch):
    """registering it deploys a base model the user benchmarks as their adapter."""
    from types import SimpleNamespace

    import huggingface_hub

    from flash.serve.adapter_check import _verify_adapter_artifact_tensors
    from flash.serve.errors import AdapterTensorMissing

    class FakeApi:
        def list_repo_tree(self, **_kwargs):
            return [SimpleNamespace(path=f"sft/run/adapter/{_SHARDS[0]}", size=456)]

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: FakeApi())

    with pytest.raises(AdapterTensorMissing, match="no complete index-referenced set"):
        _verify_adapter_artifact_tensors("org/repo", "sft/run/adapter", hf_revision="sha")


def test_serving_validation_accepts_a_complete_shard_set(monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    from flash.serve.adapter_check import _verify_adapter_artifact_tensors

    class FakeApi:
        def list_repo_tree(self, **_kwargs):
            return [
                SimpleNamespace(path=f"sft/run/adapter/{name}", size=456)
                for name in (*_SHARDS, _INDEX)
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: FakeApi())

    _verify_adapter_artifact_tensors("org/repo", "sft/run/adapter", hf_revision="sha")


def test_warmstart_identity_binds_every_shard(monkeypatch):
    """a sharded source could not be seen at all, so warm-start failed at submission.

    binding only the first shard would let a later attempt rewrite the rest and still compare equal,
    which is the change this identity exists to detect.
    """
    from types import SimpleNamespace

    import huggingface_hub

    from flash.adapters.lora_rank import adapter_artifact_identity

    oids = {name: f"sha256:{name}-v1" for name in _SHARDS}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_tree(self, **_kwargs):
            return [
                SimpleNamespace(path=f"sft/run/adapter/{name}", blob_id=None, size=8, lfs=None)
                if name == _INDEX
                else SimpleNamespace(
                    path=f"sft/run/adapter/{name}",
                    blob_id=None,
                    size=123,
                    lfs={"sha256": oids[name], "size": 123},
                )
                for name in (*_SHARDS, _INDEX)
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    config = {"peft_type": "LORA", "r": 16, "lora_alpha": 32}
    first = adapter_artifact_identity("owner/runs:sft/run", config, token="t")
    assert first.weight_filename == ",".join(_SHARDS)

    oids[_SHARDS[1]] = "sha256:rewritten"
    changed = adapter_artifact_identity("owner/runs:sft/run", config, token="t")
    assert changed.digest != first.digest
