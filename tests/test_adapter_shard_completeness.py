"""offline tests: sharded adapter weights count only as a complete, index-referenced set.

peft discovers the sharded representation solely through ``adapter_model.<ext>.index.json``. every
gate that asks "is this a loadable adapter" therefore has to ask it of the file SET, and all of them
have to answer the same way -- serving accepting what the worker rejects is how an artifact gets
deployed that ``init_from_adapter`` then refuses to load.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from safetensors.numpy import save

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
    return save({key: np.zeros((1,), dtype=np.float16) for key in keys})


def _write_sharded_adapter(tmp_path, shard_keys, weight_map):
    for name, keys in zip(_SHARDS, shard_keys, strict=True):
        (tmp_path / name).write_bytes(_safetensors_bytes(keys))
    (tmp_path / _INDEX).write_text(json.dumps({"weight_map": weight_map}))


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


def test_a_complete_indexed_set_with_a_partial_larger_total_fails_closed():
    """the listing cannot reveal whether the index names the complete old set or partial new set."""
    stale_or_live = "adapter_model-00001-of-00003.safetensors"
    assert loadable_adapter_weight_files([*_SHARDS, _INDEX, stale_or_live]) == []


def test_a_complete_larger_set_with_a_partial_smaller_total_fails_closed():
    """any second total is ambiguous regardless of which candidate happens to be complete."""
    complete = tuple(f"adapter_model-{index:05d}-of-00003.safetensors" for index in range(1, 4))
    stale_or_live = "adapter_model-00001-of-00002.safetensors"
    assert loadable_adapter_weight_files([*complete, _INDEX, stale_or_live]) == []


def test_an_old_complete_set_beside_an_incomplete_live_index_set_fails_closed():
    """a retry can leave complete stale weights while its newly indexed upload is still partial."""
    incomplete_live = "adapter_model-00001-of-00003.safetensors"
    names = [*_SHARDS, incomplete_live, _INDEX]

    assert not has_loadable_adapter_weights(names)
    assert loadable_adapter_weight_files(names) == []


def test_an_oversized_total_fails_closed_without_materializing_its_candidate_set(monkeypatch):
    """untrusted residue cannot force allocation proportional to a filename's claimed total."""
    import flash.adapters.artifacts as artifacts

    names = [*_SHARDS, _INDEX, "adapter_model-00001-of-99999999.safetensors"]
    builtin_range = range

    def bounded_range(*args):
        stop = args[-1]
        assert stop <= len(names), f"materialized oversized shard range ending at {stop}"
        return builtin_range(*args)

    monkeypatch.setattr(artifacts, "range", bounded_range, raising=False)
    started = time.monotonic()

    assert loadable_adapter_weight_files(names) == []
    assert time.monotonic() - started < 1.0


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


def test_bin_adapter_weights_are_rejected_beside_safetensors():
    names = ["adapter_model.safetensors", "adapter_model.bin"]
    assert not is_adapter_weight_filename("adapter_model.bin")
    assert loadable_adapter_weight_files(names) == ["adapter_model.safetensors"]


def test_an_orphan_safetensors_shard_does_not_fall_through_to_bin():
    names = [_SHARDS[0], "adapter_model.bin"]
    assert not has_loadable_adapter_weights(names)
    assert loadable_adapter_weight_files(names) == []


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


def test_worker_metadata_reader_rejects_duplicate_keys_across_shards(tmp_path):
    import flash.engine.worker.model.lora as lora

    first = "base_model.model.layers.0.q_proj.lora_A.default.weight"
    second = "base_model.model.layers.0.q_proj.lora_B.default.weight"
    _write_sharded_adapter(
        tmp_path,
        ([first], [first, second]),
        {first: _SHARDS[0], second: _SHARDS[1]},
    )

    with pytest.raises(ValueError, match="duplicate tensor keys"):
        lora._read_adapter_tensor_metadata(str(tmp_path))


def test_worker_metadata_reader_rejects_missing_mapped_key(tmp_path):
    import flash.engine.worker.model.lora as lora

    first = "base_model.model.layers.0.q_proj.lora_A.default.weight"
    missing = "base_model.model.layers.0.q_proj.lora_B.default.weight"
    _write_sharded_adapter(
        tmp_path,
        ([first], []),
        {first: _SHARDS[0], missing: _SHARDS[1]},
    )

    with pytest.raises(ValueError, match=r"missing=.*lora_B"):
        lora._read_adapter_tensor_metadata(str(tmp_path))


def test_worker_metadata_reader_rejects_unmapped_header_key(tmp_path):
    import flash.engine.worker.model.lora as lora

    first = "base_model.model.layers.0.q_proj.lora_A.default.weight"
    second = "base_model.model.layers.0.q_proj.lora_B.default.weight"
    extra = "base_model.model.layers.1.q_proj.lora_A.default.weight"
    _write_sharded_adapter(
        tmp_path,
        ([first], [second, extra]),
        {first: _SHARDS[0], second: _SHARDS[1]},
    )

    with pytest.raises(ValueError, match="disagrees with weight_map"):
        lora._read_adapter_tensor_metadata(str(tmp_path))


def test_worker_metadata_reader_rejects_weight_map_shard_disagreement(tmp_path):
    import flash.engine.worker.model.lora as lora

    first = "base_model.model.layers.0.q_proj.lora_A.default.weight"
    second = "base_model.model.layers.0.q_proj.lora_B.default.weight"
    _write_sharded_adapter(
        tmp_path,
        ([first], [second]),
        {first: _SHARDS[1], second: _SHARDS[0]},
    )

    with pytest.raises(ValueError, match="disagrees with weight_map"):
        lora._read_adapter_tensor_metadata(str(tmp_path))


def test_worker_metadata_reader_rejects_duplicate_weight_map_key(tmp_path):
    import flash.engine.worker.model.lora as lora

    first = "base_model.model.layers.0.q_proj.lora_A.default.weight"
    for name in _SHARDS:
        (tmp_path / name).write_bytes(_safetensors_bytes([first]))
    (tmp_path / _INDEX).write_text(
        '{"weight_map":{"duplicate":"adapter_model-00001-of-00002.safetensors",'
        '"duplicate":"adapter_model-00002-of-00002.safetensors"}}'
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        lora._read_adapter_tensor_metadata(str(tmp_path))


def test_worker_metadata_reader_rejects_index_shard_set_disagreement(tmp_path):
    import flash.engine.worker.model.lora as lora

    first = "base_model.model.layers.0.q_proj.lora_A.default.weight"
    _write_sharded_adapter(tmp_path, ([first], []), {first: _SHARDS[0]})

    with pytest.raises(ValueError, match="do not match selected shards"):
        lora._read_adapter_tensor_metadata(str(tmp_path))


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
    from flash.serve.deployment import export

    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    (tmp_path / _SHARDS[0]).write_bytes(_safetensors_bytes([key]))

    assert export._adapter_weight_paths(tmp_path) == []


def test_serving_validation_refuses_an_orphan_shard(monkeypatch):
    """registering it deploys a base model the user benchmarks as their adapter."""
    from types import SimpleNamespace

    import huggingface_hub

    from flash.serve.contract.errors import AdapterTensorMissing
    from flash.serve.deployment.adapter_check import _verify_adapter_artifact_tensors

    class FakeApi:
        def list_repo_tree(self, **_kwargs):
            return [SimpleNamespace(path=f"sft/run/adapter/{_SHARDS[0]}", size=456)]

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: FakeApi())

    with pytest.raises(AdapterTensorMissing, match="no complete index-referenced set"):
        _verify_adapter_artifact_tensors("org/repo", "sft/run/adapter", artifact_revision="sha")


def test_serving_validation_accepts_a_complete_shard_set(monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    from flash.serve.deployment.adapter_check import _verify_adapter_artifact_tensors

    class FakeApi:
        def list_repo_tree(self, **_kwargs):
            return [
                SimpleNamespace(path=f"sft/run/adapter/{name}", size=456)
                for name in (*_SHARDS, _INDEX)
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: FakeApi())

    _verify_adapter_artifact_tensors("org/repo", "sft/run/adapter", artifact_revision="sha")


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
                    # attribute-style `BlobLfsInfo`, the only shape `list_repo_tree` produces.
                    lfs=SimpleNamespace(sha256=oids[name], size=123),
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
