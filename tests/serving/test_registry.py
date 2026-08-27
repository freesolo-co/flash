from __future__ import annotations

from pathlib import Path

from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.store.registry import AdapterRegistry, lora_int_id
from tests.serving.checkpoint_fixtures import checkpoint_record


def _record(
    adapter_id: str,
    *,
    updated_at: str | None = None,
    repo_id: str | None = None,
    subfolder: str | None = None,
    repo_type: str = "model",
) -> AdapterRecord:
    return checkpoint_record(
        adapter_id,
        "openai/gpt-oss-20b",
        repo_id=repo_id or f"Freesolo-Co/{adapter_id}",
        updated_at=updated_at,
        subfolder=subfolder,
        repo_type=repo_type,
        thinking=True,
    )


def _id(run_id: str) -> str:
    return f"{run_id}/final"


def test_registry_hydrates_lists_and_upserts_records() -> None:
    registry = AdapterRegistry()
    registry.hydrate([_record("b"), _record("a")])

    assert [record.adapter_id for record in registry.list_ready()] == [_id("a"), _id("b")]

    registry.upsert(_record("c"))
    assert registry.get("org-1", _id("c")) is not None

    record_c = _record("c")
    path = Path("/tmp/adapter-c")
    registry.set_local_path(record_c, path)
    assert registry.local_path(record_c) == path


def test_upsert_does_not_downgrade_to_an_older_record() -> None:
    # A stale router can forward an OLDER record to the engine via the lazy (revive=False) upsert
    # after a redeploy that reused the id with a new source. That must NOT overwrite the engine's
    # newer record (which would evict the fresh LoRA and re-download the old source).
    registry = AdapterRegistry()
    new = _record("x", updated_at="2026-06-26T12:00:00Z", repo_id="org/new", subfolder="v2")
    old = _record("x", updated_at="2026-06-26T11:00:00Z", repo_id="org/old", subfolder="v1")

    registry.upsert(new)
    returned = registry.upsert(old)  # stale forward, revive defaults to False
    assert returned.repo_id == "org/new"  # caller gets the newer record back
    assert registry.get("org-1", _id("x")).repo_id == "org/new"  # newer record kept


def test_upsert_newer_and_explicit_revive_win() -> None:
    registry = AdapterRegistry()
    registry.upsert(_record("y", updated_at="2026-06-26T11:00:00Z", repo_id="org/old"))
    # A strictly newer record overwrites.
    registry.upsert(_record("y", updated_at="2026-06-26T12:00:00Z", repo_id="org/newer"))
    assert registry.get("org-1", _id("y")).repo_id == "org/newer"
    # An explicit redeploy (revive=True) wins even with an OLDER timestamp.
    registry.upsert(
        _record("y", updated_at="2026-06-26T10:00:00Z", repo_id="org/redeploy"), revive=True
    )
    assert registry.get("org-1", _id("y")).repo_id == "org/redeploy"


def test_upsert_overwrites_when_timestamps_unknown_or_equal() -> None:
    registry = AdapterRegistry()
    # No timestamps -> can't prove older -> overwrite (preserves legitimate lazy-load forwards).
    registry.upsert(_record("z", repo_id="org/a"))
    registry.upsert(_record("z", repo_id="org/b"))
    assert registry.get("org-1", _id("z")).repo_id == "org/b"
    # EQUAL timestamps -> incoming is not STRICTLY older -> overwrite (a same-instant forward of the
    # same-or-newer record must still win; the guard only blocks strictly-older incoming records).
    registry.upsert(_record("e", updated_at="2026-06-26T12:00:00Z", repo_id="org/a"))
    registry.upsert(_record("e", updated_at="2026-06-26T12:00:00Z", repo_id="org/b"))
    assert registry.get("org-1", _id("e")).repo_id == "org/b"


def test_revive_stamps_timestamp_so_post_deploy_resists_stale_forward() -> None:
    # POST /adapters upserts the request body with revive=True, but that body carries NO updated_at
    # (persist_adapter stamps the DB row's `now` without copying it back). Without a stamp the
    # deployed record would have updated_at=None, so a stale router's lazy forward of the OLDER DB
    # record (parseable ts, existing_at None) would slip past the do-not-downgrade guard and
    # re-download stale weights. The revive stamp gives the deploy a fresh updated_at so the older
    # forward is correctly rejected.
    registry = AdapterRegistry()
    deployed = registry.upsert(
        _record("a", updated_at=None, repo_id="org/new", subfolder="v2"), revive=True
    )
    assert deployed.updated_at is not None  # stamped on revive
    stale = _record("a", updated_at="2026-06-18T00:00:00Z", repo_id="org/old", subfolder="v1")
    returned = registry.upsert(stale)  # stale router lazy-forwards the OLD record (revive=False)
    assert returned.repo_id == "org/new"  # guard rejects the older forward
    assert (
        registry.get("org-1", _id("a")).repo_id == "org/new"
    )  # fresh source kept, no stale rollback


def test_lora_int_id_is_stable_positive() -> None:
    # a literal, not a second call: comparing the function to itself passes for ANY implementation,
    # so it could not catch a changed hash that silently repoints every served adapter. this value
    # is sha1("adapter")[:4] masked to 31 bits, derived independently of the code under test.
    assert lora_int_id("adapter") == 2145909947
    assert lora_int_id("adapter") > 0


def test_lora_int_id_fits_int32() -> None:
    # vLLM stores lora_int_id in an int32 array; a full uint32 hash (>2^31-1) overflows
    # and crashes the engine. Every id must stay in [1, 2^31-1] across many adapter ids.
    for i in range(2000):
        assert 1 <= lora_int_id(f"adapter-{i}-serving-demo") <= 0x7FFFFFFF
    assert lora_int_id("q4b") <= 0x7FFFFFFF


def test_remove_tombstones_block_hydrate_and_lazy_upsert() -> None:
    # After remove(), neither a hydrate (reload reading a stale ready row) nor a lazy upsert
    # (default revive=False, e.g. the generate-path forward) may resurrect the adapter.
    registry = AdapterRegistry()
    registry.hydrate([_record("a")])
    removed = registry.remove("org-1", _id("a"))
    assert removed is not None
    assert registry.get("org-1", _id("a")) is None
    assert not registry.has("org-1", _id("a"))

    registry.hydrate([_record("a")])  # reload still has the not-yet-disabled ready row
    assert registry.get("org-1", _id("a")) is None
    assert not registry.has("org-1", _id("a"))

    registry.upsert(_record("a"))  # generate-path lazy adopt
    assert registry.get("org-1", _id("a")) is None
    assert not registry.has("org-1", _id("a"))


def test_explicit_revive_clears_tombstone() -> None:
    # An explicit redeploy (revive=True, the POST /adapters path) must clear the tombstone.
    registry = AdapterRegistry()
    registry.hydrate([_record("a")])
    registry.remove("org-1", _id("a"))
    registry.upsert(_record("a"), revive=True)
    assert registry.has("org-1", _id("a"))
    assert registry.get("org-1", _id("a")) is not None
    # And once revived, a later reload keeps it (tombstone is gone).
    registry.hydrate([_record("a")])
    assert registry.has("org-1", _id("a"))


def test_durably_newer_record_expires_tombstone_on_hydrate() -> None:
    # Redeploy-elsewhere: this container undeployed "a", but a POST landed on ANOTHER container
    # and persisted a row with a NEWER updated_at. A later reload here carries that newer
    # timestamp, which must expire the local tombstone so this warm container stops 404-ing.
    registry = AdapterRegistry()
    registry.hydrate([_record("a", updated_at="2026-06-18T00:00:00Z")])
    registry.remove("org-1", _id("a"))  # tombstone at the removed row's updated_at
    # Stale ready row at the SAME timestamp must stay suppressed (no redeploy yet).
    registry.hydrate([_record("a", updated_at="2026-06-18T00:00:00Z")])
    assert not registry.has("org-1", _id("a"))
    # Durably newer row (a redeploy elsewhere) expires the tombstone -> back in the registry.
    registry.hydrate([_record("a", updated_at="2026-06-18T01:00:00Z")])
    assert registry.has("org-1", _id("a"))


def test_post_added_record_undeploy_is_durable_against_timestamped_reload() -> None:
    # Records added via POST /adapters carry no in-memory updated_at, so remove() must stamp the
    # tombstone with ``now`` (not leave it None). Otherwise a stale status=ready reload or a
    # generate-path forward carrying ANY updated_at would resurrect the just-undeployed adapter.
    registry = AdapterRegistry()
    registry.upsert(_record("a", updated_at=None), revive=True)  # POST: no in-memory timestamp
    registry.remove("org-1", _id("a"))
    assert not registry.has("org-1", _id("a"))
    # A stale ready row from shared storage (which always has updated_at) must NOT revive it.
    registry.hydrate([_record("a", updated_at="2026-06-18T00:00:00Z")])
    assert not registry.has("org-1", _id("a"))
    # Nor a generate-path lazy forward carrying a timestamp.
    registry.upsert(_record("a", updated_at="2026-06-18T00:00:00Z"))
    assert not registry.has("org-1", _id("a"))
    # An explicit redeploy still revives it.
    registry.upsert(_record("a"), revive=True)
    assert registry.has("org-1", _id("a"))


def test_tombstone_same_instant_mixed_iso_does_not_resurrect() -> None:
    # The corrupting case from the review: a POST-added record undeploys, so remove() stamps the
    # tombstone via _utc_now_iso() => "...+00:00". A later stale reload reads the not-yet-disabled
    # row from PostgREST at the SAME instant but rendered with a trailing "Z". Lexically
    # "...Z" > "...+00:00" (ord("Z")=0x5A > ord("+")=0x2B), so a raw-string ``>`` would WRONGLY
    # expire the tombstone and resurrect the just-undeployed adapter. Comparing instants must not.
    registry = AdapterRegistry()
    registry.upsert(_record("a", updated_at=None), revive=True)  # POST: no in-memory timestamp
    registry.remove(
        "org-1", _id("a")
    )  # tombstone backfilled with _utc_now_iso() => "+00:00" rendering
    tombstone_iso = registry._tombstones[("org-1", _id("a"))]
    assert tombstone_iso is not None
    assert tombstone_iso.endswith("+00:00")
    same_instant_z = tombstone_iso.replace("+00:00", "Z")  # PostgREST rendering, same instant
    registry.hydrate([_record("a", updated_at=same_instant_z)])
    assert not registry.has("org-1", _id("a"))


def test_tombstone_compares_instants_across_mixed_iso_renderings() -> None:
    # updated_at arrives in mixed ISO-8601 forms: PostgREST emits a trailing ``Z`` while
    # datetime.isoformat() emits ``+00:00``. A same-instant row in the other rendering must not
    # expire the tombstone, but a strictly-NEWER row in either rendering must.
    registry = AdapterRegistry()
    registry.hydrate([_record("a", updated_at="2026-06-18T00:00:00Z")])
    registry.remove("org-1", _id("a"))  # tombstone stamped at the removed row's "...Z" timestamp
    # Same instant rendered as "+00:00" must NOT expire the tombstone.
    registry.hydrate([_record("a", updated_at="2026-06-18T00:00:00+00:00")])
    assert not registry.has("org-1", _id("a"))
    # A strictly newer "+00:00" redeploy must expire the "...Z" tombstone even though "+" < "Z".
    registry.upsert(_record("a", updated_at="2026-06-18T01:00:00+00:00"))
    assert registry.has("org-1", _id("a"))


def test_tombstone_unparseable_timestamp_stays_blocked() -> None:
    # An unparseable updated_at can't prove a newer redeploy, so the safe default holds: the
    # adapter stays suppressed (only an explicit revive=True clears it).
    registry = AdapterRegistry()
    registry.hydrate([_record("a", updated_at="2026-06-18T00:00:00Z")])
    registry.remove("org-1", _id("a"))
    registry.upsert(_record("a", updated_at="not-a-timestamp"))
    assert not registry.has("org-1", _id("a"))
    registry.upsert(_record("a"), revive=True)
    assert registry.has("org-1", _id("a"))


def test_redeploy_with_new_source_invalidates_cached_local_path() -> None:
    # A redeploy reusing the same adapter id but pointing at a DIFFERENT repo/subfolder/repo_type
    # must not keep serving the previously-downloaded weights: local_path returns None (and reports
    # the stale entry) so the engine evicts the old vLLM LoRA and re-downloads.
    registry = AdapterRegistry()
    v1 = _record("a", repo_id="org/run-1", subfolder="rl/r1/seed0/adapter", repo_type="dataset")
    registry.upsert(v1, revive=True)
    path_v1 = Path("/tmp/adapter-a-v1")
    registry.set_local_path(v1, path_v1)
    assert registry.local_path(v1) == path_v1
    assert not registry.local_path_is_stale(v1)

    # Redeploy: same id, new subfolder (new weights). Adopt it (lazy generate-path upsert).
    v2 = _record("a", repo_id="org/run-1", subfolder="rl/r2/seed0/adapter", repo_type="dataset")
    registry.upsert(v2)
    assert registry.local_path_is_stale(v2)  # engine must evict the old LoRA
    assert registry.local_path(v2) is None  # forces a fresh download
    # After re-download under the new identity, the new path is cached and no longer stale.
    path_v2 = Path("/tmp/adapter-a-v2")
    registry.set_local_path(v2, path_v2)
    assert registry.local_path(v2) == path_v2
    assert not registry.local_path_is_stale(v2)

    # An unchanged record (same source) keeps serving the cached path with no eviction.
    assert registry.local_path(v2) == path_v2
    assert not registry.local_path_is_stale(v2)


def test_durably_newer_record_expires_tombstone_on_lazy_upsert() -> None:
    # Same redeploy-elsewhere expiry, but via the generate-path lazy upsert (revive=False).
    registry = AdapterRegistry()
    registry.hydrate([_record("a", updated_at="2026-06-18T00:00:00Z")])
    registry.remove("org-1", _id("a"))
    # Old/equal-timestamp forward stays suppressed.
    registry.upsert(_record("a", updated_at="2026-06-18T00:00:00Z"))
    assert not registry.has("org-1", _id("a"))
    # Newer forward (redeploy) revives it without an explicit revive=True.
    registry.upsert(_record("a", updated_at="2026-06-18T02:00:00Z"))
    assert registry.has("org-1", _id("a"))


def test_artifact_revision_participates_in_local_source_identity() -> None:
    first = _record(
        "run",
        repo_id="org/run",
        subfolder="checkpoints/step-1",
    ).model_copy(update={"artifact_revision": "a" * 40})
    second = first.model_copy(update={"artifact_revision": "b" * 40})
    registry = AdapterRegistry()
    registry.upsert(first)
    registry.set_local_path(first, Path("/tmp/source-a"))
    registry.upsert(second)
    assert registry.local_path_is_stale(second)
    assert registry.local_path(second) is None
