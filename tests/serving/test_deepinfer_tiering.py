from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.prefix_cache import Prefix
from flash.serving.deepinfer.tiered_cache import TieredPrefixCache


def cache():
    freed = []
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(free_seq=freed.append, get_num_free_gpu_blocks=lambda: 1000),
        cache_config=SimpleNamespace(block_size=128),
    )
    value = TieredPrefixCache(engine, cpu_bytes=0, max_entries=1)
    value.cpu_limit = 100
    value.snapshots = SimpleNamespace(
        size=lambda seq: 60,
        capture=lambda seq: SimpleNamespace(nbytes=60, source=seq),
        restore=lambda snapshot, params: snapshot.source + "-restored",
    )
    return value, freed


def insert(value, key, namespace="tenant"):
    value.entries[key] = Prefix(namespace, (1,) * 256, key, 100, 384)


def test_spill_is_bounded_and_promotes_only_exact_tenant_match(monkeypatch):
    monkeypatch.setattr("flash.serving.deepinfer.tiered_cache.time.monotonic", lambda: 100)
    value, freed = cache()
    insert(value, "a")
    value._evict("a")
    insert(value, "b")
    value._evict("b")
    assert list(value.cpu) == ["b"]
    assert value.cpu_bytes == 60
    assert freed == ["a", "b"]
    assert value._match("other", [1] * 300, None) is None
    assert value._match("tenant", [2] * 300, None) is None
    key, prefix = value._match("tenant", [1] * 300, None)
    assert key == "b"
    assert prefix.sequence == "b-restored"
    assert not value.cpu
    assert value.cpu_bytes == 0
    value.close()
    assert freed[-1] == "b-restored"


def test_expiry_and_close_do_not_spill_and_failed_restore_is_a_miss(monkeypatch):
    monkeypatch.setattr("flash.serving.deepinfer.tiered_cache.time.monotonic", lambda: 100)
    value, freed = cache()
    insert(value, "a")
    value._evict("a")
    value.snapshots.restore = lambda snapshot, params: None
    assert value._match("tenant", [1] * 300, None) is None
    assert not value.entries
    assert not value.cpu
    assert value.cpu_bytes == 0
    insert(value, "b")
    value._evict("b")
    insert(value, "c")
    value.ttl = 1
    monkeypatch.setattr("flash.serving.deepinfer.tiered_cache.time.monotonic", lambda: 102)
    value.prune()
    assert not value.entries
    assert not value.cpu
    assert value.cpu_bytes == 0
    assert freed == ["a", "b", "c"]


def test_invalid_budget_rejected():
    with pytest.raises(ValueError, match="CPU KV budget"):
        TieredPrefixCache(None, cpu_bytes=9 * 1024**3)


def test_restore_copy_failure_releases_native_allocation(monkeypatch):
    import sys

    from flash.serving.deepinfer.snapshots import Snapshots

    freed = []
    seq = SimpleNamespace(block_table=[1], state_id=0, multi_modal_data={"segment_input": "text"})
    prepared = {"mm_token_type": [0, 0], "images_seq_mask": [False, False]}

    def map_metadata(raw):
        assert raw == {"segment_input": "text"}
        return prepared

    def allocate(value):
        assert value.multi_modal_data is prepared
        return True

    def fail_copy(*args):
        raise RuntimeError("copy failed")

    blocks = SimpleNamespace(shape=(10, 8), dtype="bytes", index_copy_=fail_copy)
    state = SimpleNamespace(shape=(4,), dtype="bytes")

    class States(list):
        dtype = "bytes"

    engine = SimpleNamespace(
        driver_worker=SimpleNamespace(
            cache_engine=SimpleNamespace(gpu_blocks=blocks, gpu_states=States([state]))
        ),
        add_request=lambda **kwargs: seq,
        multi_modal_processor=SimpleNamespace(multi_modal_mapper=map_metadata),
        scheduler=SimpleNamespace(try_allocate_seq=allocate, free_seq=freed.append),
    )
    store = Snapshots(engine)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(synchronize=lambda: None),
            tensor=lambda *args, **kwargs: [1],
            long="long",
        ),
    )
    snapshot = SimpleNamespace(
        owner=store.owner,
        tokens=(1, 2),
        kv=SimpleNamespace(shape=(1, 8), dtype="bytes", to=lambda device: None),
        state=state,
    )
    with pytest.raises(RuntimeError, match="copy failed"):
        store.restore(snapshot, None)
    assert freed == [seq]
    snapshot.owner = "another-model-instance"
    with pytest.raises(ValueError, match="another engine"):
        store.restore(snapshot, None)
    assert freed == [seq]


def test_continuation_rebuilds_text_masks_for_a_different_length_question():
    from flash.serving.deepinfer.prefix_cache import PrefixCache

    seq = SimpleNamespace()
    donor = SimpleNamespace(
        data=SimpleNamespace(get_last_token_id=lambda: 2, get_len=lambda: 3),
        so_matcher=None,
    )
    metadata = SimpleNamespace(
        data="new tokens", compressed_data="new compressed tokens", multi_modal_data="raw"
    )
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(fork_seq=lambda *args: seq),
        seq_counter=iter([1]),
        add_request=lambda **kwargs: metadata,
        multi_modal_processor=SimpleNamespace(multi_modal_mapper=lambda raw: {"processed": raw}),
    )
    result = PrefixCache(engine)._fork(donor, "request", None, [1, 2, 3, 4, 5])
    assert result.request_id == "request"
    assert result.data == "new tokens"
    assert result.multi_modal_data == {"processed": "raw"}
