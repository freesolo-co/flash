"""Shared, fully-managed, best-effort, multi-region model-weight cache on RunPod network volumes.

The runner attaches ONE platform-wide logical cache name (``flash-weights``) to EVERY run; the RunPod
provider realizes it as a DISTINCT per-DC physical volume (``flash-weights-<dc>``) in every storage
datacenter (distinct names avoid the runpod_flash resource-key collision) and allows the endpoint
across all of them. This gives a cross-run weight cache with NO single-datacenter pin — whichever
region the run lands in, that region's volume mounts at ``/runpod-volume`` and the worker's
``HF_HOME`` points there. On a no-capacity failure the lifecycle drops the volume so a run can never
wedge IN_QUEUE on one full region.

Fully managed: there are NO env knobs (fixed name/size/datacenter set) and NO per-model gating —
these tests pin that down. Everything is offline (the autouse ``_offline`` conftest stubs the RunPod
API); the one SDK contract we lock — that our datacenter list is always a superset of the volume DCs
— is driven through the real ``Endpoint._build_resource_config()`` (a pure, network-free validator).
"""

from __future__ import annotations

import os
import tempfile
import types

from flash.spec import GpuSpec, JobSpec, TrainSpec


def _vol_spec(name="flash-weights", gb=100, **gpu):
    return JobSpec(model="m", gpu=GpuSpec(network_volume=name, network_volume_gb=gb, **gpu))


def _ndc() -> int:
    """Size of the ALLOWED datacenter set (DataCenter.all()) — what the endpoint's `datacenter` list
    spans. Derived from the same source the code uses, so it tracks the SDK's storage-DC set.
    """
    from flash.providers.runpod.jobs import weight_cache_datacenters

    n = len(weight_cache_datacenters())
    assert n > 1
    return n


# ---------------------------------------------------------------------------
# spec carrier round-trips (the field must survive every to_dict()->from_dict() hop)
# ---------------------------------------------------------------------------
def test_network_volume_round_trips():
    spec = _vol_spec(gb=123)
    again = JobSpec.from_dict(spec.to_dict())
    assert again.gpu.network_volume == "flash-weights"
    assert again.gpu.network_volume_gb == 123


def test_default_spec_has_no_volume():
    assert JobSpec(model="m").gpu.network_volume is None


def test_legacy_datacenter_key_tolerated():
    # A stale spec from the reverted single-DC pin may still carry gpu.datacenter; it must be
    # ignored (the DC set is deploy-time policy now), not raise.
    spec = JobSpec.from_dict({"model": "m", "gpu": {"datacenter": "EU-RO-1", "network_volume": "v"}})
    assert spec.gpu.network_volume == "v"
    assert not hasattr(spec.gpu, "datacenter")


def test_network_volume_gb_tolerant_of_bad_values():
    # Platform-managed field: null/empty/"0"/0/negative/non-numeric/missing -> default 100 (never
    # crash int(), never round-trip a nonsensical size). Valid positive sizes pass through.
    for raw in (None, "", 0, "0", -5, "-5", "abc", True, False):
        spec = JobSpec.from_dict({"model": "m", "gpu": {"network_volume": "v", "network_volume_gb": raw}})
        assert spec.gpu.network_volume_gb == 100, f"{raw!r} should default to 100"
    assert JobSpec.from_dict({"model": "m", "gpu": {"network_volume": "v"}}).gpu.network_volume_gb == 100
    for raw in (200, "150"):
        spec = JobSpec.from_dict({"model": "m", "gpu": {"network_volume": "v", "network_volume_gb": raw}})
        assert spec.gpu.network_volume_gb == int(raw)


# ---------------------------------------------------------------------------
# jobs.weight_cache_* — the volume fleet + endpoint kwargs (fully managed, no knobs)
# ---------------------------------------------------------------------------
def test_weight_cache_datacenters_excludes_volume_incapable():
    # DataCenter.all() includes DCs RunPod does NOT back with network volumes (live-found: US-MO-1);
    # creating a volume there 500s the whole deploy, so the fleet must exclude them.
    from runpod_flash.core.resources.datacenter import DataCenter

    from flash.providers.runpod import jobs

    dcs = jobs.weight_cache_datacenters()
    vals = {d.value for d in dcs}
    assert len(set(dcs)) == len(dcs)  # all distinct
    assert vals == {d.value for d in DataCenter.all()} - jobs._VOLUME_INCAPABLE_DATACENTERS
    assert "US-MO-1" not in vals  # the known volume-incapable DC is dropped
    assert not (vals & jobs._VOLUME_INCAPABLE_DATACENTERS)


def test_weight_cache_datacenters_ignores_removed_env_knob(monkeypatch):
    # Regression: the FLASH_WEIGHT_CACHE_DATACENTERS knob is GONE — the fleet is fixed/managed.
    from flash.providers.runpod import jobs

    baseline = len(jobs.weight_cache_datacenters())
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DATACENTERS", "US-CA-2")
    assert len(jobs.weight_cache_datacenters()) == baseline  # env ignored


def test_weight_cache_volumes_distinct_name_per_dc():
    from flash.providers.runpod import jobs

    vols = jobs.weight_cache_volumes(_vol_spec(gb=100))
    # EAGER: one volume in EVERY storage DC (no lazy used-set gating).
    assert len(vols) == _ndc()
    # DISTINCT physical name per DC (the SDK keys resource tracking on name only, so same-named
    # volumes across DCs collide -> a 2nd-volume "replace" -> unimplemented undeploy -> crash).
    assert len({v.name for v in vols}) == len(vols)
    assert all(v.name.startswith("flash-weights-") for v in vols)
    # the name encodes the DC, lowercased
    assert {v.name for v in vols} == {f"flash-weights-{v.dataCenterId.value.lower()}" for v in vols}
    # exactly the full storage-DC set
    assert {v.dataCenterId for v in vols} == set(jobs.weight_cache_datacenters())
    assert {v.size for v in vols} == {100}


def test_weight_cache_volumes_size_tolerant_of_bad_values():
    # weight_cache_volumes builds NetworkVolumes from spec.gpu.network_volume_gb directly (a GpuSpec
    # can carry a raw value that bypassed JobSpec.from_dict's parse). A non-numeric/"0"/negative size
    # must default to 100 via _volume_gb — never raise (best-effort would silently drop the cache) or
    # create a 0-GB volume.
    from flash.providers.runpod import jobs

    for raw in ("0", 0, -5, "abc", None, True):
        vols = jobs.weight_cache_volumes(_vol_spec(gb=raw))
        assert {v.size for v in vols} == {100}, f"{raw!r} should default to 100 GB"
    # a valid positive size still passes through
    assert {v.size for v in jobs.weight_cache_volumes(_vol_spec(gb=250))} == {250}


def test_weight_cache_volume_name_includes_datacenter():
    from runpod_flash.core.resources.datacenter import DataCenter

    from flash.providers.runpod import jobs

    assert jobs.weight_cache_volume_name("flash-weights", DataCenter.US_CA_2) == "flash-weights-us-ca-2"


def test_weight_cache_volumes_empty_without_volume_name():
    from flash.providers.runpod import jobs

    assert jobs.weight_cache_volumes(JobSpec(model="m")) == []


def test_weight_cache_endpoint_kwargs_volume_in_every_dc():
    from flash.providers.runpod import jobs

    kw = jobs.weight_cache_endpoint_kwargs(_vol_spec())
    assert sorted(kw) == ["datacenter", "volume"]
    # EAGER: a volume in EVERY storage DC, and the endpoint allowed across exactly that same set, so
    # whichever DC it lands in is warm. The two lists span the identical storage-DC set.
    assert len(kw["volume"]) == _ndc()
    assert len(kw["datacenter"]) == _ndc()
    assert {v.dataCenterId for v in kw["volume"]} == set(kw["datacenter"])


def test_weight_cache_endpoint_kwargs_empty_without_volume():
    from flash.providers.runpod import jobs

    assert jobs.weight_cache_endpoint_kwargs(JobSpec(model="m")) == {}


def test_weight_cache_endpoint_kwargs_swallows_errors(monkeypatch):
    from flash.providers.runpod import jobs

    monkeypatch.setattr(
        jobs, "weight_cache_volumes", lambda spec: (_ for _ in ()).throw(RuntimeError("sdk boom"))
    )
    # best-effort: ANY failure building the cache -> {} (deploy with no volume), never propagate.
    assert jobs.weight_cache_endpoint_kwargs(_vol_spec()) == {}


def test_weight_cache_satisfies_real_sdk_superset_validation(monkeypatch):
    # The whole no-pin design rests on the SDK accepting N volumes + N datacenters on one endpoint.
    # Drive the REAL (pure, offline) Endpoint validator/builder to lock that contract: every volume
    # DC must be within the endpoint datacenter list (serverless.py superset rule) and the locations
    # string must span all the DCs.
    monkeypatch.setenv("FLASH_IS_LIVE_PROVISIONING", "true")
    from runpod_flash import Endpoint
    from runpod_flash.core.resources.gpu import GpuGroup

    from flash.providers.runpod import jobs

    kw = jobs.weight_cache_endpoint_kwargs(_vol_spec())
    ep = Endpoint(name="wc-test", gpu=GpuGroup.AMPERE_48, gpu_count=1, **kw)
    cfg = ep._build_resource_config()  # raises if the superset rule is violated
    vol_dcs = {v.dataCenterId for v in cfg.networkVolumes}
    assert vol_dcs <= set(cfg.datacenter)  # superset rule holds (eager: the sets are in fact equal)
    assert len(cfg.locations.split(",")) == _ndc()  # endpoint allowed across all storage DCs


def test_deploy_train_endpoint_attaches_volume_kwargs(monkeypatch):
    # End-to-end through the primary deploy path: the multi-volume/multi-DC kwargs reach Endpoint().
    import runpod_flash
    import runpod_flash.core.resources.resource_manager as rm_mod

    from flash.providers.runpod import auth, jobs

    captured: dict = {}

    class RecEndpoint:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def _build_resource_config(self):
            return types.SimpleNamespace()  # no template -> apply_disk_gb(None) early-returns

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            return types.SimpleNamespace(id="ep-abc")

    monkeypatch.setattr(runpod_flash, "Endpoint", RecEndpoint)
    monkeypatch.setattr(rm_mod, "ResourceManager", FakeRM)
    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    monkeypatch.setattr(jobs, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "_patch_runpod_backoff", lambda: None)

    eid, _name = jobs.deploy_train_endpoint("RTX 4090", spec=_vol_spec(), disk_gb=None)
    assert eid == "ep-abc"
    assert len(captured["volume"]) == _ndc()  # EAGER: a volume in every storage DC
    assert len(captured["datacenter"]) == _ndc()  # allowed across all storage DCs
    assert len({v.name for v in captured["volume"]}) == _ndc()  # distinct per-DC names
    assert all(v.name.startswith("flash-weights-") for v in captured["volume"])


def test_deploy_train_endpoint_no_volume_when_spec_has_none(monkeypatch):
    import runpod_flash
    import runpod_flash.core.resources.resource_manager as rm_mod

    from flash.providers.runpod import auth, jobs

    captured: dict = {}

    class RecEndpoint:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def _build_resource_config(self):
            return types.SimpleNamespace()

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            return types.SimpleNamespace(id="ep-xyz")

    monkeypatch.setattr(runpod_flash, "Endpoint", RecEndpoint)
    monkeypatch.setattr(rm_mod, "ResourceManager", FakeRM)
    monkeypatch.setattr(auth, "ensure_auth", lambda: None)
    monkeypatch.setattr(jobs, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "_patch_runpod_backoff", lambda: None)

    jobs.deploy_train_endpoint("RTX 4090", spec=JobSpec(model="m"), disk_gb=None)
    assert "volume" not in captured
    assert "datacenter" not in captured


# ---------------------------------------------------------------------------
# deps.weight_cache_env / build_worker_env redirect
# ---------------------------------------------------------------------------
def test_weight_cache_env_is_base_model_scoped():
    from flash.providers.runpod.train.deps import weight_cache_env

    env = weight_cache_env("/runpod-volume")
    # BASE-MODEL-SCOPED: FLASH_WEIGHT_CACHE_DIR points the base-model prefetch at the mount's HF hub
    # layout. It must NOT set a process-global HF_HOME (that leaked env/reward downloads onto the
    # shared multi-tenant mount — issue #252). The executable kernel-JIT caches are also never on it.
    assert env == {"FLASH_WEIGHT_CACHE_DIR": "/runpod-volume/hf-cache/hub"}
    assert "HF_HOME" not in env
    for k in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TILELANG_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        assert k not in env


def test_weight_cache_env_custom_mount():
    from flash.providers.runpod.train.deps import weight_cache_env

    assert weight_cache_env("/workspace")["FLASH_WEIGHT_CACHE_DIR"] == "/workspace/hf-cache/hub"


def test_build_worker_env_sets_base_model_cache_with_volume():
    from flash.providers.runpod.train.deps import build_worker_env

    env = build_worker_env(_vol_spec(), 0)
    assert env["FLASH_WEIGHT_CACHE_DIR"] == "/runpod-volume/hf-cache/hub"
    # The leak fix: no process-global HF_HOME redirect, so env/reward downloads use the ephemeral cache.
    assert "HF_HOME" not in env


def test_build_worker_env_no_cache_without_volume():
    from flash.providers.runpod.train.deps import build_worker_env

    env = build_worker_env(JobSpec(model="m"), 0)
    # Without a volume the base-model cache var must NOT be set (pointing at a missing mount).
    assert "FLASH_WEIGHT_CACHE_DIR" not in env
    assert "HF_HOME" not in env


def test_build_worker_env_per_run_override_wins():
    from flash.providers.runpod.train.deps import build_worker_env

    spec = _vol_spec()
    spec = JobSpec.from_dict({**spec.to_dict(), "worker_env": {"FLASH_WEIGHT_CACHE_DIR": "/custom/hub"}})
    # a per-run [worker_env] override is merged last and must win over the cache redirect.
    assert build_worker_env(spec, 0)["FLASH_WEIGHT_CACHE_DIR"] == "/custom/hub"


def test_drop_unmounted_cache_env_strips_when_unmounted(monkeypatch):
    import flash.providers.runpod.train.deps as deps

    # volume NOT mounted -> the /runpod-volume cache var is stripped (cold fallback), others kept.
    monkeypatch.setattr(deps.os.path, "isdir", lambda p: False)
    env = {"FLASH_WEIGHT_CACHE_DIR": "/runpod-volume/hf-cache/hub", "OTHER": "x"}
    out = deps.drop_unmounted_cache_env(env)
    assert "FLASH_WEIGHT_CACHE_DIR" not in out
    assert out["OTHER"] == "x"


def test_drop_unmounted_cache_env_keeps_when_mounted(monkeypatch):
    import flash.providers.runpod.train.deps as deps

    monkeypatch.setattr(deps.os.path, "isdir", lambda p: True)
    env = {"FLASH_WEIGHT_CACHE_DIR": "/runpod-volume/hf-cache/hub"}
    assert deps.drop_unmounted_cache_env(env) == {"FLASH_WEIGHT_CACHE_DIR": "/runpod-volume/hf-cache/hub"}


# ---------------------------------------------------------------------------
# worker engine.worker.hf.prefetch_model — base-model-scoped caching (issue #252)
# The shared mount holds ONLY the trusted public base model; the run's env/reward HF downloads use the
# per-worker ephemeral cache and never touch the shared multi-tenant mount.
# ---------------------------------------------------------------------------
def _patch_prefetch_io(monkeypatch, ephemeral_hub):
    """Stub the side effects of prefetch_model: a fake snapshot_download that records its call and
    materializes a repo dir under cache_dir, the heartbeat/gpu probes, and the ephemeral hub cache."""
    import huggingface_hub
    import huggingface_hub.constants

    import flash.engine.worker.hf as hf

    calls = []

    def _fake_snapshot(repo_id, cache_dir=None, ignore_patterns=None, **kw):
        calls.append({"repo_id": repo_id, "cache_dir": cache_dir, "ignore_patterns": ignore_patterns})
        if cache_dir:  # simulate a real download landing on the (mount) cache: create the repo folder
            folder = "models--" + repo_id.replace("/", "--")
            os.makedirs(os.path.join(cache_dir, folder, "snapshots"), exist_ok=True)
        return cache_dir or "/ephemeral"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(ephemeral_hub))
    monkeypatch.setattr(hf, "gpu_diagnostics", lambda: {})
    monkeypatch.setattr(hf._w, "heartbeat", lambda *a, **k: None)
    return hf, calls


def test_prefetch_model_downloads_to_shared_mount_and_links(tmp_path, monkeypatch):
    """With the shared cache attached, the base model is downloaded ONTO the mount (explicit cache_dir)
    and symlinked into the per-worker ephemeral hub cache so the trainer/vLLM hit it without re-download."""
    import os as _os

    mount = tmp_path / "runpod-volume"  # the shared multi-tenant mount (must exist)
    mount.mkdir()
    shared_hub = str(mount / "hf-cache" / "hub")
    ephemeral_hub = tmp_path / "ephemeral" / "hub"
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", shared_hub)
    hf, calls = _patch_prefetch_io(monkeypatch, ephemeral_hub)

    hf.prefetch_model("Qwen/Qwen3.5-0.8B")

    # downloaded straight onto the shared mount, NOT the ephemeral default
    assert calls == [{
        "repo_id": "Qwen/Qwen3.5-0.8B", "cache_dir": shared_hub,
        "ignore_patterns": ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
    }]
    folder = "models--Qwen--Qwen3.5-0.8B"
    dst = ephemeral_hub / folder
    # the base model is now visible in the ephemeral cache as a SYMLINK pointing back to the mount
    assert _os.path.islink(dst)
    assert _os.path.realpath(dst) == _os.path.realpath(_os.path.join(shared_hub, folder))
    # and ONLY the base model is linked — env/reward repos would resolve in (and be written to) the
    # ephemeral cache, never the shared mount.
    assert [p.name for p in ephemeral_hub.iterdir()] == [folder]


def test_prefetch_model_uses_ephemeral_default_without_shared_cache(tmp_path, monkeypatch):
    """No shared cache attached -> cache_dir=None (the worker's ephemeral default) and no symlink."""
    monkeypatch.delenv("FLASH_WEIGHT_CACHE_DIR", raising=False)
    ephemeral_hub = tmp_path / "ephemeral" / "hub"
    hf, calls = _patch_prefetch_io(monkeypatch, ephemeral_hub)

    hf.prefetch_model("Qwen/Qwen3.5-0.8B")

    assert calls[0]["cache_dir"] is None  # ephemeral default cache, a correct cold run
    assert not ephemeral_hub.exists()  # nothing linked


def test_prefetch_model_falls_back_to_ephemeral_when_mount_absent(tmp_path, monkeypatch):
    """FLASH_WEIGHT_CACHE_DIR set but the mount isn't present (failed/absent attach) -> ephemeral cache,
    no write under the missing mount. Defense-in-depth re-check on the worker itself."""
    missing_hub = str(tmp_path / "runpod-volume" / "hf-cache" / "hub")  # parent mount NOT created
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", missing_hub)
    ephemeral_hub = tmp_path / "ephemeral" / "hub"
    hf, calls = _patch_prefetch_io(monkeypatch, ephemeral_hub)

    hf.prefetch_model("Qwen/Qwen3.5-0.8B")

    assert calls[0]["cache_dir"] is None  # mount absent -> ephemeral, never the missing /runpod-volume path


def test_shared_weight_cache_dir_resolves_mount_for_both_substrates(tmp_path, monkeypatch):
    """_shared_weight_cache_dir derives the mount as two levels up from the cache dir (works for the
    RunPod /runpod-volume and instance /weight-cache mounts alike) and requires it to exist."""
    import flash.engine.worker.hf as hf

    monkeypatch.delenv("FLASH_WEIGHT_CACHE_DIR", raising=False)
    assert hf._shared_weight_cache_dir() is None  # unset

    mount = tmp_path / "weight-cache"
    mount.mkdir()
    hub = str(mount / "hf-cache" / "hub")
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", hub)
    assert hf._shared_weight_cache_dir() == hub  # mount present

    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", str(tmp_path / "absent" / "hf-cache" / "hub"))
    assert hf._shared_weight_cache_dir() is None  # mount absent -> ephemeral fallback


# ---------------------------------------------------------------------------
# instance-provider integration (Lambda reuses RunPod's build_worker_env)
# ---------------------------------------------------------------------------
def test_strip_runpod_volume_env_removes_only_mount_rooted_vars():
    from flash.providers.runpod.train.deps import strip_runpod_volume_env

    env = {"FLASH_WEIGHT_CACHE_DIR": "/runpod-volume/hf-cache/hub", "X": "/runpod-volume/foo", "KEEP": "v", "HF_TOKEN": "t"}
    out = strip_runpod_volume_env(env)
    assert "FLASH_WEIGHT_CACHE_DIR" not in out
    assert "X" not in out
    assert out == {"KEEP": "v", "HF_TOKEN": "t"}  # non-/runpod-volume vars preserved


def test_instance_payload_strips_runpod_volume_redirect():
    # The RunPod weight-cache base-model redirect must NOT leak into a Lambda payload —
    # those instances never mount /runpod-volume. (build_worker_env DOES set it; the instance strips.)
    from flash.providers import _instance
    from flash.providers.runpod.train.deps import build_worker_env

    spec = JobSpec.from_dict({**_vol_spec().to_dict(), "run_id": "r", "model": "Qwen/Qwen3.5-0.8B"})
    assert build_worker_env(spec, 0)["FLASH_WEIGHT_CACHE_DIR"].startswith("/runpod-volume")  # leak source
    for arm in ("lambda",):
        env = _instance.build_payload(spec, seed=0, attempt=0, arm=arm)["env"]
        assert not env.get("FLASH_WEIGHT_CACHE_DIR", "").startswith("/runpod-volume"), arm


# ---------------------------------------------------------------------------
# runner._assign_weight_cache_volume — fully managed, no knobs; gated to PUBLIC catalog runs only
# (the shared cross-tenant cache must never hold private/gated weights — confidentiality boundary)
# ---------------------------------------------------------------------------
def test_assign_weight_cache_attaches_to_catalog_run():
    from flash import runner

    out = runner._assign_weight_cache_volume(JobSpec(model="m", run_id="r", model_policy="catalog"))
    assert out.gpu.network_volume == runner.WEIGHT_CACHE_VOLUME_NAME == "flash-weights"
    assert out.gpu.network_volume_gb == runner.WEIGHT_CACHE_VOLUME_GB == 100


def test_assign_weight_cache_default_policy_attaches():
    # The JobSpec default policy is "catalog" (managed runs), so a default-policy run is cached.
    from flash import runner

    out = runner._assign_weight_cache_volume(JobSpec(model="m", run_id="r"))
    assert out.gpu.network_volume == "flash-weights"


def test_assign_weight_cache_skips_open_model_policy():
    # CONFIDENTIALITY GATE: an open-model ("allow") run may target a PRIVATE/GATED HF repo; its
    # weights must NOT enter the shared cross-tenant cache. So the cache is NOT attached and HF_HOME
    # is never redirected onto the shared mount — the weights stay on the worker's ephemeral disk.
    from flash import runner

    out = runner._assign_weight_cache_volume(JobSpec(model="some-org/private-model", run_id="r", model_policy="allow"))
    assert out.gpu.network_volume is None  # cache-less: no shared-mount redirect for a possibly-private model


def test_assign_weight_cache_strips_preset_shared_cache_on_open_model():
    # The gate takes PRECEDENCE over the "honor an existing volume" no-op: a programmatic open-model
    # spec that ALREADY pinned the SHARED cache name must be FORCED cache-less, not bypass the gate.
    from flash import runner

    spec = JobSpec.from_dict({
        "model": "some-org/private-model", "run_id": "r", "model_policy": "allow",
        "gpu": {"type": "A10", "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME, "network_volume_gb": 100},
    })
    out = runner._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume is None  # the pre-set shared cache was stripped


def test_assign_weight_cache_keeps_per_org_volume_on_open_model():
    # A NON-shared (per-org / custom) volume on an open run is the intended escape hatch — left intact.
    from flash import runner

    spec = JobSpec.from_dict({
        "model": "some-org/private-model", "run_id": "r", "model_policy": "allow",
        "gpu": {"type": "A10", "network_volume": "org-123-private-cache", "network_volume_gb": 100},
    })
    out = runner._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume == "org-123-private-cache"  # not the shared cache -> kept


def test_assign_weight_cache_ignores_removed_kill_switch(monkeypatch):
    # Regression: the FLASH_WEIGHT_CACHE=0 kill switch is GONE — fully managed, always on.
    from flash import runner

    monkeypatch.setenv("FLASH_WEIGHT_CACHE", "0")
    out = runner._assign_weight_cache_volume(JobSpec(model="m", run_id="r"))
    assert out.gpu.network_volume == "flash-weights"  # env ignored


def test_assign_weight_cache_does_not_override_existing():
    from flash import runner

    spec = _vol_spec(name="explicit-vol")
    spec = JobSpec.from_dict({**spec.to_dict(), "model_policy": "catalog", "run_id": "r"})
    out = runner._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume == "explicit-vol"  # an explicit/test value is never clobbered


def test_assign_weight_cache_skips_oversized_catalog_model():
    # SIZE GATE: the 35B MoE's ~70 GB checkpoint (peak ~140 GB with Xet temp) can't fit the fixed
    # 100 GB shared cache. Attaching it would redirect HF_HOME onto the undersized mount and overflow
    # mid-download — so the run is left cache-less and downloads to the (200 GB) container disk.
    from flash import runner
    from flash.catalog import MODELS

    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    out = runner._assign_weight_cache_volume(
        JobSpec(model=info.id, run_id="r", model_policy="catalog"), info
    )
    assert out.gpu.network_volume is None  # too big for the shared cache -> cache-less


def test_assign_weight_cache_strips_preset_shared_cache_on_oversized_catalog_model():
    # SIZE GATE re-applies to a pre-set SHARED-cache name: a programmatic/stale catalog spec that
    # already pinned ``flash-weights`` for the 35B MoE must NOT bypass the gate via the "honor an
    # existing volume" no-op and redirect HF_HOME onto the undersized mount. It's stripped cache-less.
    from flash import runner
    from flash.catalog import MODELS

    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    spec = JobSpec.from_dict({
        "model": info.id, "run_id": "r", "model_policy": "catalog",
        "gpu": {"type": "B200", "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME, "network_volume_gb": 100},
    })
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume is None  # oversized -> the pre-set shared cache was stripped


def test_assign_weight_cache_keeps_preset_shared_cache_on_fitting_catalog_model():
    # The re-gate only strips when OVERSIZED: a fitting model that already carries the shared cache
    # keeps it (the pin is correct), exercising the honor-existing path for a shared-name spec.
    from flash import runner
    from flash.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]
    spec = JobSpec.from_dict({
        "model": info.id, "run_id": "r", "model_policy": "catalog",
        "gpu": {"type": "H100", "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME, "network_volume_gb": 100},
    })
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "flash-weights"  # fits -> kept


def test_assign_weight_cache_keeps_preset_custom_volume_on_oversized_catalog_model():
    # The re-gate is scoped to the SHARED name only: a custom/per-org volume on an oversized model is
    # left intact (the caller owns its sizing — it may be a 200 GB org cache that DOES fit).
    from flash import runner
    from flash.catalog import MODELS

    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    spec = JobSpec.from_dict({
        "model": info.id, "run_id": "r", "model_policy": "catalog",
        "gpu": {"type": "B200", "network_volume": "org-123-big-cache", "network_volume_gb": 400},
    })
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "org-123-big-cache"  # custom volume honored despite size


def test_assign_weight_cache_attaches_fitting_catalog_model():
    # A model whose download fits the cache (with temp headroom) is still attached when info is passed.
    from flash import runner
    from flash.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]  # ~19.4 GB download, peak ~39 GB < 100 GB
    out = runner._assign_weight_cache_volume(
        JobSpec(model=info.id, run_id="r", model_policy="catalog"), info
    )
    assert out.gpu.network_volume == "flash-weights"


def test_fits_weight_cache_dense_models_fit_moe_does_not():
    from flash import runner
    from flash.catalog import MODELS

    assert not runner._fits_weight_cache(MODELS["Qwen/Qwen3.6-35B-A3B"])  # 35B MoE overflows
    for mid in ("openbmb/MiniCPM5-1B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"):
        assert runner._fits_weight_cache(MODELS[mid]), mid  # every dense catalog model fits


def test_submit_job_assigns_weight_cache(monkeypatch):
    # Integration: the assignment is wired into submit_job and visible on the dry-run spec.
    from flash import runner

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        spec = JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "sft",
                "environment": {"id": "github:o/r@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 8},
                "run_id": "flash-wc-1",
            }
        )
        out = runner.submit_job(spec, dry_run=True).spec
    assert out["gpu"]["network_volume"] == "flash-weights"
    assert out["gpu"]["network_volume_gb"] == 100


# ---------------------------------------------------------------------------
# lifecycle._drop_weight_cache + the no-capacity fallback
# ---------------------------------------------------------------------------
def test_drop_weight_cache_clears_volume():
    from flash.runner.lifecycle import _drop_weight_cache

    assert _drop_weight_cache(_vol_spec()).gpu.network_volume is None


def test_drop_weight_cache_noop_without_volume():
    from flash.runner.lifecycle import _drop_weight_cache

    spec = JobSpec(model="m")
    assert _drop_weight_cache(spec) is spec  # no copy when there's nothing to drop


def test_drop_weight_cache_preserves_non_shared_escape_hatch_volume():
    # Review (Copilot): the no-capacity cache-drop must NOT strip a non-shared per-org/custom volume —
    # that is the deliberate escape-hatch isolation an open-model run opted into. Only the SHARED
    # platform cache (WEIGHT_CACHE_VOLUME_NAME) is dropped.
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME
    from flash.runner.lifecycle import _drop_weight_cache

    custom = _vol_spec(name="org-1234-private")
    assert custom.gpu.network_volume != WEIGHT_CACHE_VOLUME_NAME
    out = _drop_weight_cache(custom)
    assert out is custom  # untouched (no copy) — the escape-hatch volume survives the retry
    assert out.gpu.network_volume == "org-1234-private"
    # the SHARED cache, by contrast, IS dropped
    assert _drop_weight_cache(_vol_spec(name=WEIGHT_CACHE_VOLUME_NAME)).gpu.network_volume is None


def _supervised_walk(monkeypatch, failures):
    """Run the supervised seed loop, returning per-attempt (gpu.network_volume, gpu.type) tuples.

    ``failures`` maps attempt index -> failure category (absent attempt -> success).
    """
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        orch = fresh_runner(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        seen: list = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            seen.append((spec.gpu.network_volume, spec.gpu.type))
            fail = failures.get(attempt)
            if fail:
                return jobs.PollResult(False, failure=fail, detail="x")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="wc-walk",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)
        assert orch.get_status("wc-walk").state == "done"
        return seen


def test_no_capacity_drops_weight_cache_on_retry(monkeypatch):
    # attempt 0 lands no_capacity (the cache's DC set was starved) -> the retry runs cache-less on
    # the unrestricted all-DC pool. This is what makes the pin impossible: worst case is a cold run.
    seen = _supervised_walk(monkeypatch, {0: "no_capacity"})
    assert seen[0][0] == "flash-weights"  # first attempt used the cache
    assert seen[1][0] is None  # retry dropped it


def test_poll_error_on_volume_attempt_drops_cache(monkeypatch):
    # A deploy/submit poll_error on a volume-backed attempt (e.g. the SDK failing to create/attach a
    # volume) must degrade to a cold no-volume retry, not loop on the same volume-backed spec.
    seen = _supervised_walk(monkeypatch, {0: "poll_error"})
    assert seen[0][0] == "flash-weights"
    assert seen[1][0] is None


def test_cache_drop_does_not_advance_gpu_walk(monkeypatch):
    # The cache-drop transition must retry the SAME (cheapest) GPU cache-less on the wider pool
    # first — the capacity miss may have been the cache DC set, not the GPU class globally.
    seen = _supervised_walk(monkeypatch, {0: "no_capacity"})
    assert seen[1][1] == seen[0][1]  # same GPU class on the cache-drop retry (walk NOT advanced)


def test_cache_drop_then_walks_on_next_failure(monkeypatch):
    # After the cache is dropped, a SUBSEQUENT failure still walks to the next-cheapest GPU.
    seen = _supervised_walk(monkeypatch, {0: "no_capacity", 1: "stalled"})
    assert seen[0][0] == "flash-weights"
    assert seen[1][0] is None
    assert seen[2][0] is None
    assert seen[1][1] == seen[0][1]  # attempt 1: same GPU, cache dropped (no walk)
    assert seen[2][1] != seen[0][1]  # attempt 2: walked to a different GPU class


def test_non_capacity_failure_keeps_weight_cache(monkeypatch):
    # An ordinary infra flake (stall) on a volume attempt must NOT drop the cache — the warm-weights
    # benefit should survive ordinary retries, and the GPU walk advances as usual.
    seen = _supervised_walk(monkeypatch, {0: "stalled"})
    assert seen[0][0] == "flash-weights"
    assert seen[1][0] == "flash-weights"
    assert seen[1][1] != seen[0][1]  # stall walks to the next GPU (cache retained)


# ---------------------------------------------------------------------------
# preload: warm the per-region volumes with the catalog models (operator action)
# ---------------------------------------------------------------------------
def test_catalog_model_ids_are_the_cache_fitting_catalog():
    from flash.catalog import MODELS
    from flash.providers.runpod import preload
    from flash.runner import _fits_weight_cache

    ids = set(preload.catalog_model_ids())
    # The default preload set is the catalog RESTRICTED to models that fit the weight cache (warming a
    # non-fitting model only overflows the fixed mount). Mirrors the submit path's _fits_weight_cache.
    assert ids == {mid for mid, info in MODELS.items() if _fits_weight_cache(info)}
    assert ids <= set(MODELS)
    # The ~70 GB 35B MoE (peak ~140 GB > 100 GB cache) must be excluded from the default warm set.
    assert "Qwen/Qwen3.6-35B-A3B" in MODELS
    assert "Qwen/Qwen3.6-35B-A3B" not in ids


def test_preload_branch_passes_explicit_cache_dir(monkeypatch):
    # BLOCKER regression: _train_body imports huggingface_hub at module load, so HF_HOME set in the
    # preload branch is read too LATE — the download must pass cache_dir=<HF_HOME>/hub explicitly or
    # it lands on the worker's ephemeral default cache and the volume is never warmed.
    import os as _os

    import huggingface_hub

    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(_os, "environ", dict(_os.environ))  # isolate the branch's os.environ.update
    monkeypatch.setattr(_os.path, "isdir", lambda p: True)  # pretend the volume IS mounted
    calls = []

    def fake_snapshot(repo_id, token=None, cache_dir=None, local_files_only=False, ignore_patterns=None):
        calls.append({"repo": repo_id, "cache_dir": cache_dir, "probe": local_files_only,
                      "ignore": ignore_patterns})
        if local_files_only:
            raise FileNotFoundError("not cached yet")  # force the real download path
        return "/somewhere"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    out = endpoints._train_body({
        "mode": "preload",
        "models": ["Qwen/Qwen3.5-0.8B"],
        "env": {"HF_HOME": "/runpod-volume/hf-cache", "HF_TOKEN": "t"},
    })
    assert out["preloaded"] == ["Qwen/Qwen3.5-0.8B"]
    # both the probe and the real download must target the on-volume HF hub dir, not the default
    assert calls
    assert all(c["cache_dir"] == "/runpod-volume/hf-cache/hub" for c in calls)
    # and both must apply the SAME exclusions as the worker prefetch / instance preload (no cache bloat)
    expected_ignore = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
    assert all(c["ignore"] == expected_ignore for c in calls)


def test_preload_rejects_non_volume_hf_home(monkeypatch):
    """Phantom-warm guard: preload's whole purpose is to populate the on-volume HF cache. A missing or
    non-/runpod-volume HF_HOME would make cache_dir fall back to the worker's EPHEMERAL default cache,
    so snapshot_download would report repos preloaded while persisting nothing. The handler must refuse
    such a misconfigured preload (no download) instead of reporting a phantom warm."""
    import os as _os

    import huggingface_hub

    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(_os, "environ", dict(_os.environ))
    monkeypatch.setattr(_os.path, "isdir", lambda p: True)  # even with the mount present...
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda *a, **k: calls.append(k) or "/x",
    )
    for bad in (None, "", "/root/.cache/huggingface", "/tmp/hf"):
        env = {"HF_TOKEN": "t"}
        if bad is not None:
            env["HF_HOME"] = bad
        out = endpoints._train_body({"mode": "preload", "models": ["Qwen/Qwen3.5-0.8B"], "env": env})
        assert out["preloaded"] == []
        assert out["already_cached"] == []
        assert "HF_HOME rooted at /runpod-volume" in out["error"]
    assert not calls  # ...nothing is ever downloaded for a non-volume HF_HOME


def test_teardown_weight_cache_deletes_only_fleet_volumes(monkeypatch):
    from flash.providers.runpod import preload

    vols = {
        "v1": "flash-weights-us-ca-2",
        "v2": "flash-weights-eu-ro-1",
        "v3": "someone-elses-volume",  # must NOT be touched
        "v4": "flash-kernel-cache-abc",  # different fleet, must NOT touch
    }
    deletes = []

    class FakeRest:
        def __init__(self, api_key=None):
            self.api_key = api_key

        async def list_network_volumes(self):
            return {"networkVolumes": [{"name": n, "id": i} for i, n in vols.items()]}

        async def _execute_rest(self, method, url):
            assert method == "DELETE"
            vid = url.rsplit("/", 1)[-1]
            deletes.append(vid)
            vols.pop(vid, None)  # actually delete
            raise Exception("204 No Content")  # empty-body parse error MUST be tolerated

    import runpod_flash.core.api.runpod as rp_api

    from flash.providers.runpod import keys as rp_keys

    monkeypatch.setattr(rp_keys, "keys", lambda: ["k1"])  # single-account pool
    monkeypatch.setattr(rp_api, "RunpodRestClient", FakeRest)
    out = preload.teardown_weight_cache(["US-CA-2", "EU-RO-1"])
    # returns names CONFIRMED gone by re-list (not by trusting the 204-erroring delete response)
    assert sorted(out) == ["flash-weights-eu-ro-1", "flash-weights-us-ca-2"]
    assert sorted(deletes) == ["v1", "v2"]  # only the two fleet volumes were deleted
    assert "v3" in vols  # other accounts' / fleets' volumes untouched
    assert "v4" in vols


def test_teardown_weight_cache_no_runpod_key_is_noop(monkeypatch):
    """No RUNPOD_API_KEY -> RunPod teardown is a best-effort no-op (log + []), never a raise.

    A raise here would abort the chained `--teardown` before the Lambda reclaim runs.
    """
    import runpod_flash.core.api.runpod as rp_api

    from flash.providers.runpod import keys as rp_keys
    from flash.providers.runpod import preload

    def _boom(*a, **k):
        raise AssertionError("RunpodRestClient must not be constructed without a key")

    monkeypatch.setattr(rp_keys, "keys", lambda: [])  # empty pool == RUNPOD_API_KEY unset
    monkeypatch.setattr(rp_api, "RunpodRestClient", _boom)
    assert preload.teardown_weight_cache(["US-CA-2"]) == []


def test_teardown_weight_cache_sweeps_all_pool_accounts(monkeypatch):
    from flash.providers.runpod import preload

    seen_keys = []

    class FakeRest:
        def __init__(self, api_key=None):
            seen_keys.append(api_key)
            self.vols = {"v1": "flash-weights-us-ca-2"}  # each account independently holds it

        async def list_network_volumes(self):
            return {"networkVolumes": [{"name": n, "id": i} for i, n in self.vols.items()]}

        async def _execute_rest(self, method, url):
            self.vols.pop(url.rsplit("/", 1)[-1], None)
            return {}

    import runpod_flash.core.api.runpod as rp_api

    from flash.providers.runpod import keys as rp_keys

    monkeypatch.setattr(rp_keys, "keys", lambda: ["k1", "k2"])  # two-account pool
    monkeypatch.setattr(rp_api, "RunpodRestClient", FakeRest)
    out = preload.teardown_weight_cache(["US-CA-2"])
    # swept BOTH accounts; results are account-prefixed when the pool is multi-account
    assert seen_keys == ["k1", "k2"]
    assert sorted(out) == ["acct0:flash-weights-us-ca-2", "acct1:flash-weights-us-ca-2"]


def test_teardown_does_not_report_failed_deletes(monkeypatch):
    # A real delete failure (auth/5xx/network) leaves the volume present -> it must NOT appear in the
    # "deleted" result (the re-list is the source of truth), and a warning is logged.
    import runpod_flash.core.api.runpod as rp_api

    from flash.providers.runpod import keys as rp_keys
    from flash.providers.runpod import preload

    class FakeRest:
        def __init__(self, api_key=None):
            self.vols = {"v1": "flash-weights-us-ca-2"}  # delete will "fail" -> stays present

        async def list_network_volumes(self):
            return {"networkVolumes": [{"name": n, "id": i} for i, n in self.vols.items()]}

        async def _execute_rest(self, method, url):
            raise Exception("403 Forbidden")  # real failure: volume NOT removed

    monkeypatch.setattr(rp_keys, "keys", lambda: ["k1"])
    monkeypatch.setattr(rp_api, "RunpodRestClient", FakeRest)
    out = preload.teardown_weight_cache(["US-CA-2"])
    assert out == []  # nothing confirmed gone -> not reported as deleted


def test_teardown_works_inside_running_event_loop(monkeypatch):
    # teardown is normally a sync CLI call, but _run_async must also work if invoked from an async
    # context (notebook/server) — asyncio.run() alone would raise "running event loop".
    import asyncio

    import runpod_flash.core.api.runpod as rp_api

    from flash.providers.runpod import keys as rp_keys
    from flash.providers.runpod import preload

    class FakeRest:
        def __init__(self, api_key=None):
            self.vols = {"v1": "flash-weights-us-ca-2"}

        async def list_network_volumes(self):
            return {"networkVolumes": [{"name": n, "id": i} for i, n in self.vols.items()]}

        async def _execute_rest(self, method, url):
            self.vols.pop(url.rsplit("/", 1)[-1], None)
            return {}

    monkeypatch.setattr(rp_keys, "keys", lambda: ["k1"])
    monkeypatch.setattr(rp_api, "RunpodRestClient", FakeRest)

    async def _from_async():
        return preload.teardown_weight_cache(["US-CA-2"])  # sync call from within a live loop

    out = asyncio.run(_from_async())
    assert out == ["flash-weights-us-ca-2"]


def test_teardown_lambda_filesystems_deletes_only_fleet(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.runpod import preload

    fses = [
        {"id": "f1", "name": "flash-weights", "region": {"name": "us-east-1"}},
        {"id": "f2", "name": "flash-weights", "region": {"name": "us-west-2"}},
        {"id": "f3", "name": "someones-data", "region": {"name": "us-east-1"}},  # NOT ours
    ]
    deleted = []
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda: fses)
    monkeypatch.setattr(lambda_api, "delete_filesystem", lambda i: deleted.append(i) or True)

    out = preload.teardown_lambda_filesystems()
    assert sorted(deleted) == ["f1", "f2"]  # only the flash-weights FSes, across regions
    assert sorted(out) == ["lambda:us-east-1/flash-weights", "lambda:us-west-2/flash-weights"]


def test_teardown_lambda_filesystems_no_key_is_noop(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.runpod import preload

    monkeypatch.setattr(
        lambda_api, "list_filesystems",
        lambda: (_ for _ in ()).throw(lambda_api.LambdaApiError("LAMBDA_API_KEY not set")),
    )
    assert preload.teardown_lambda_filesystems() == []  # absent provider -> nothing reclaimed, no raise


def test_teardown_cli_reclaims_all_providers(monkeypatch):
    """`preload --teardown` sweeps RunPod + Lambda in one shot."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "teardown_weight_cache", lambda dcs: ["flash-weights-us-ca-2"])
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", lambda: ["lambda:us-east-1/flash-weights"])
    assert preload.main(["--teardown"]) == 0


def test_teardown_dry_run_deletes_nothing(monkeypatch):
    """`--teardown --dry-run` only PRINTS the plan — it must never call the destructive helpers."""
    from flash.providers.runpod import preload

    def _boom(*a, **k):
        raise AssertionError("--teardown --dry-run must not call any teardown helper")

    monkeypatch.setattr(preload, "teardown_weight_cache", _boom)
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", _boom)
    assert preload.main(["--teardown", "--dry-run"]) == 0


def test_scoped_teardown_rejects_invalid_datacenter(monkeypatch):
    """`--teardown --datacenters <bad-id>` fails non-zero and deletes NOTHING (no silent success)."""
    from flash.providers.runpod import preload

    def _boom(*a, **k):
        raise AssertionError("invalid scoped teardown must not delete anything")

    monkeypatch.setattr(preload, "teardown_weight_cache", _boom)
    assert preload.main(["--teardown", "--datacenters", "NOT-A-REAL-DC"]) == 2


def test_teardown_continues_when_runpod_unconfigured(monkeypatch):
    """A RunPod teardown raise (auth absent / outage) must NOT abort Lambda cleanup."""
    from flash.providers.runpod import preload

    def _boom(dcs):
        raise RuntimeError("RUNPOD_API_KEY not configured")

    lam = []
    monkeypatch.setattr(preload, "teardown_weight_cache", _boom)
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", lambda: lam.append(1) or ["lambda:us-east-1/flash-weights"])
    # RunPod raises but the instance provider still gets cleaned up best-effort; the CLI still exits 0.
    assert preload.main(["--teardown"]) == 0
    assert lam == [1]


def test_scoped_teardown_is_runpod_only(monkeypatch):
    """`--teardown --datacenters ...` scopes to RunPod; instance-provider caches are left intact."""
    from flash.providers.runpod import preload

    seen = {}
    monkeypatch.setattr(preload, "teardown_weight_cache", lambda dcs: seen.setdefault("dcs", dcs) or ["flash-weights-us-ca-2"])
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", lambda: seen.setdefault("lambda", True) or [])
    assert preload.main(["--teardown", "--datacenters", "US-CA-2"]) == 0
    assert seen["dcs"] == ["US-CA-2"]  # the RunPod scope was honored
    assert "lambda" not in seen  # instance providers were NOT touched


def test_teardown_empty_datacenters_scope_is_refused(monkeypatch):
    """`--teardown --datacenters <empty/whitespace>` must ERROR, never silently full-teardown RunPod.

    Regression for the footgun where a present-but-empty scope (a) widened teardown_weight_cache to
    the WHOLE RunPod fleet via its `datacenters or <all>` fallback while (b) skipping the instance
    providers because the flag was present. It must abort (rc != 0) and touch NOTHING.
    """
    from flash.providers.runpod import preload

    called = {}
    monkeypatch.setattr(preload, "teardown_weight_cache", lambda dcs: called.setdefault("runpod", dcs) or [])
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", lambda: called.setdefault("lambda", True) or [])
    for scope in ("", " , , ", "   "):  # empty, all-commas, all-whitespace -> parse to zero ids
        called.clear()
        assert preload.main(["--teardown", "--datacenters", scope]) == 2
        assert called == {}  # no provider teardown ran — not RunPod, not the instance providers


def test_teardown_weight_cache_empty_list_is_noop_not_all(monkeypatch):
    """teardown_weight_cache([]) is a no-op; an EXPLICIT empty scope must not widen to the full fleet."""
    import runpod_flash.core.api.runpod as rp_api

    from flash.providers.runpod import keys as rp_keys
    from flash.providers.runpod import preload

    def _boom(*a, **k):
        raise AssertionError("an empty scope must not list/delete any volumes")

    monkeypatch.setattr(rp_keys, "keys", lambda: ["k1"])
    monkeypatch.setattr(rp_api, "RunpodRestClient", _boom)
    assert preload.teardown_weight_cache([]) == []  # nothing reclaimed, no client constructed


# ---------------------------------------------------------------------------
# Eager PROVISION — create the instance-provider cache storage in every region/env (no GPU)
# ---------------------------------------------------------------------------
def test_provision_lambda_filesystems_covers_every_region(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.runpod import preload

    ensured = []
    monkeypatch.setattr(lambda_api, "all_regions", lambda: ["us-east-1", "us-west-2", "europe-central-1"])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem",
        lambda name, region: ensured.append((name, region)) or f"/lambda/nfs/{name}",
    )
    out = preload.provision_lambda_filesystems()
    # one create-if-absent per region, with the managed cache name
    assert ensured == [("flash-weights", "us-east-1"), ("flash-weights", "us-west-2"), ("flash-weights", "europe-central-1")]
    assert out == ["lambda:us-east-1", "lambda:us-west-2", "lambda:europe-central-1"]


def test_provision_lambda_skips_failed_region(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.runpod import preload

    def flaky(name, region):
        if region == "bad-1":
            raise lambda_api.LambdaApiError("region down")
        return f"/lambda/nfs/{name}"

    monkeypatch.setattr(lambda_api, "all_regions", lambda: ["ok-1", "bad-1", "ok-2"])
    monkeypatch.setattr(lambda_api, "ensure_filesystem", flaky)
    # one bad region never aborts the rest
    assert preload.provision_lambda_filesystems() == ["lambda:ok-1", "lambda:ok-2"]


def test_provision_lambda_no_key_is_noop(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.runpod import preload

    monkeypatch.setattr(
        lambda_api, "all_regions",
        lambda: (_ for _ in ()).throw(lambda_api.LambdaApiError("LAMBDA_API_KEY not set")),
    )
    assert preload.provision_lambda_filesystems() == []


def test_provision_cli_creates_instance_storage(monkeypatch):
    """`preload --provision` creates Lambda storage (GPU-free) and exits 0."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "provision_lambda_filesystems", lambda: ["lambda:us-east-1"])
    assert preload.main(["--provision"]) == 0


def test_provision_cli_dry_run_provisions_nothing(monkeypatch):
    from flash.providers.runpod import preload

    called = {"n": 0}
    monkeypatch.setattr(preload, "provision_lambda_filesystems", lambda: called.__setitem__("n", called["n"] + 1) or [])
    assert preload.main(["--provision", "--dry-run"]) == 0
    assert called["n"] == 0  # dry-run touches no provider


def test_preload_one_dc_deploys_pins_single_dc_and_tears_down(monkeypatch):
    from flash.providers.runpod import preload

    calls = {}

    def fake_deploy(gpu, execution_timeout_ms=None, name_suffix=None, spec=None, endpoint_kwargs=None):
        calls["gpu"] = gpu
        calls["suffix"] = name_suffix
        calls["endpoint_kwargs"] = endpoint_kwargs
        return "ep-1", "name-1"

    submitted = {}

    def fake_submit(eid, payload):
        submitted["eid"] = eid
        submitted["payload"] = payload
        return "job-1"

    deleted = []
    monkeypatch.setattr(preload, "deploy_train_endpoint", fake_deploy)
    monkeypatch.setattr(preload.runpod_api, "submit_job", fake_submit)
    monkeypatch.setattr(preload.runpod_api, "delete_endpoint", lambda eid: deleted.append(eid))
    monkeypatch.setattr(
        preload.runpod_api, "job_status",
        lambda eid, jid: {"status": "COMPLETED", "output": {"preloaded": ["Qwen/Qwen3.5-0.8B"]}},
    )

    out = preload._preload_one_dc(
        "EU-RO-1", ["Qwen/Qwen3.5-0.8B"], token="tok", gpu="RTX 4090",
        timeout_s=60, poll_interval_s=0.0,
    )

    assert out["status"] == "ok"
    assert out["result"]["preloaded"] == ["Qwen/Qwen3.5-0.8B"]
    # endpoint_kwargs is a FACTORY (re-invoked per account on failover) — build a fresh override.
    factory = calls["endpoint_kwargs"]
    assert callable(factory)
    ek = factory()
    # pinned to exactly ONE datacenter, with that DC's single volume
    assert len(ek["volume"]) == 1
    assert [d.value for d in ek["datacenter"]] == ["EU-RO-1"]
    assert ek["volume"][0].dataCenterId.value == "EU-RO-1"
    # preload warms the SAME per-DC physical name a training run in this DC will mount
    assert ek["volume"][0].name == "flash-weights-eu-ro-1"
    # each invocation builds a FRESH NetworkVolume (so a failover account never reuses a stale id)
    assert factory()["volume"][0] is not ek["volume"][0]
    # preload payload: download-only, HF_HOME on the mount, token forwarded
    p = submitted["payload"]
    assert p["mode"] == "preload"
    assert p["models"] == ["Qwen/Qwen3.5-0.8B"]
    assert p["env"]["HF_HOME"] == "/runpod-volume/hf-cache"
    assert p["env"]["HF_TOKEN"] == "tok"
    assert deleted == ["ep-1"]  # endpoint torn down


def test_preload_one_dc_tears_down_on_failure(monkeypatch):
    from flash.providers.runpod import preload

    deleted = []
    monkeypatch.setattr(
        preload, "deploy_train_endpoint",
        lambda *a, **k: ("ep-9", "name-9"),
    )
    monkeypatch.setattr(preload.runpod_api, "submit_job", lambda eid, payload: "job-9")
    monkeypatch.setattr(preload.runpod_api, "delete_endpoint", lambda eid: deleted.append(eid))
    monkeypatch.setattr(
        preload.runpod_api, "job_status",
        lambda eid, jid: {"status": "FAILED", "error": "boom"},
    )

    out = preload._preload_one_dc(
        "US-CA-2", ["m"], token=None, gpu="RTX 4090", timeout_s=60, poll_interval_s=0.0
    )
    assert out["status"] == "error"
    assert deleted == ["ep-9"]  # still torn down on failure


def _stub_preload_deploy(monkeypatch, job_output):
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "deploy_train_endpoint", lambda *a, **k: ("ep", "n"))
    monkeypatch.setattr(preload.runpod_api, "submit_job", lambda eid, p: "job")
    monkeypatch.setattr(preload.runpod_api, "delete_endpoint", lambda eid: None)
    monkeypatch.setattr(
        preload.runpod_api, "job_status",
        lambda eid, jid: {"status": "COMPLETED", "output": job_output},
    )


def test_preload_one_dc_partial_when_a_model_fails(monkeypatch):
    # A COMPLETED job whose handler reports per-model failures is NOT a fully warmed region.
    from flash.providers.runpod import preload

    _stub_preload_deploy(monkeypatch, {
        "preloaded": ["a"], "already_cached": [], "failed": {"b": "gated repo"},
    })
    out = preload._preload_one_dc("US-CA-2", ["a", "b"], token=None, gpu="g", timeout_s=60, poll_interval_s=0.0)
    assert out["status"] == "partial"
    assert out["result"]["failed"] == {"b": "gated repo"}


def test_preload_one_dc_error_when_volume_not_mounted(monkeypatch):
    # The handler's mount-not-mounted hard error must surface as a DC-level error (not silent ok).
    from flash.providers.runpod import preload

    _stub_preload_deploy(monkeypatch, {
        "preloaded": [], "already_cached": [], "failed": {},
        "error": "weight-cache volume not mounted at /runpod-volume",
    })
    out = preload._preload_one_dc("US-CA-2", ["a"], token=None, gpu="g", timeout_s=60, poll_interval_s=0.0)
    assert out["status"] == "error"
    assert "not mounted" in out["error"]


def test_preload_branch_errors_when_volume_not_mounted(monkeypatch):
    # In the worker handler: if /runpod-volume isn't a real mount, preload must NOT silently warm
    # ephemeral disk — it returns an explicit error and downloads nothing.
    import os as _os

    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(_os, "environ", dict(_os.environ))
    monkeypatch.setattr(_os.path, "isdir", lambda p: False)  # /runpod-volume not mounted
    out = endpoints._train_body({
        "mode": "preload", "models": ["Qwen/Qwen3.5-0.8B"],
        "env": {"HF_HOME": "/runpod-volume/hf-cache"},
    })
    assert out["preloaded"] == []
    assert "not mounted" in out["error"]


def test_warm_weight_cache_fans_out_over_datacenters(monkeypatch):
    from flash.providers.runpod import preload

    seen_dcs = []

    def fake_one(dc_id, models, token, gpu, timeout_s, poll_interval_s):
        seen_dcs.append(dc_id)
        return {"datacenter": dc_id, "status": "ok"}

    monkeypatch.setattr(preload, "_preload_one_dc", fake_one)
    results = preload.warm_weight_cache(
        models=["m"], datacenters=["US-CA-2", "EU-RO-1", "US-WA-1"], max_workers=3
    )
    assert {r["datacenter"] for r in results} == {"US-CA-2", "EU-RO-1", "US-WA-1"}
    assert set(seen_dcs) == {"US-CA-2", "EU-RO-1", "US-WA-1"}
    assert all(r["status"] == "ok" for r in results)


def test_warm_weight_cache_defaults_to_full_fleet_and_catalog(monkeypatch):
    from flash.providers.runpod import preload

    captured = {}

    def fake_one(dc_id, models, token, gpu, timeout_s, poll_interval_s):
        captured["models"] = models
        return {"datacenter": dc_id, "status": "ok"}

    monkeypatch.setattr(preload, "_preload_one_dc", fake_one)
    results = preload.warm_weight_cache()  # no args -> all DCs, whole catalog
    assert len(results) == _ndc()
    assert set(captured["models"]) == set(preload.catalog_model_ids())


# ---------------------------------------------------------------------------
# Instance-provider WARM — the baked download-only preload mode + payload plumbing
# ---------------------------------------------------------------------------
def _preload_spec():
    return JobSpec.from_dict({
        "model": "Qwen/Qwen3.5-0.8B", "algorithm": "sft", "run_id": "flash-1700000000-abcd1234",
        "train": {"epochs": 1, "max_examples": 8, "hf_repo": "org/repo"},
        "gpu": {"type": "A10", "max_wall_seconds": 3600,
                "network_volume": "flash-weights", "network_volume_gb": 100},
    })


def test_instance_build_payload_preload_mode():
    from flash.providers import _instance

    p = _instance.build_payload(
        _preload_spec(), 0, 0, arm="lambda",
        cache_host_mount="/lambda/nfs/flash-weights", mode="preload", models=["a/b", "c/d"],
    )
    assert p["mode"] == "preload"
    assert p["models"] == ["a/b", "c/d"]
    # The base-model prefetch (FLASH_WEIGHT_CACHE_DIR) points at the bind-mounted cache so the download
    # persists across runs in the region. It is NOT a process-global HF_HOME (issue #252): env/reward
    # downloads stay on ephemeral disk, off the shared per-region cache.
    assert p["env"]["FLASH_WEIGHT_CACHE_DIR"] == f"{_instance.CACHE_HF_HOME}/hub"
    assert "HF_HOME" not in p["env"]


def test_instance_build_payload_no_mode_by_default():
    from flash.providers import _instance

    p = _instance.build_payload(_preload_spec(), 0, 0, arm="lambda",
                                cache_host_mount="/lambda/nfs/flash-weights")
    assert "mode" not in p  # ordinary train payload
    assert "models" not in p


def test_instance_build_payload_preserves_worker_env_hf_home(monkeypatch):
    """A per-run [worker_env].HF_HOME override is NOT clobbered by the instance cache path (parity
    with RunPod, where the worker_env override wins), and disables the platform cache redirect."""
    from flash.providers import _instance

    spec = JobSpec.from_dict({
        "model": "Qwen/Qwen3.5-0.8B", "algorithm": "sft", "run_id": "flash-1700000000-abcd1234",
        "train": {"max_examples": 8, "hf_repo": "org/repo"},
        "gpu": {"type": "A10", "max_wall_seconds": 3600, "network_volume": "flash-weights"},
        "worker_env": {"HF_HOME": "/custom/hf"},  # user-set override
    })
    p = _instance.build_payload(spec, 0, 0, arm="lambda", cache_host_mount="/lambda/nfs/flash-weights")
    # the user's HF_HOME survives, and the platform cache redirect is NOT installed on top of it.
    assert p["env"]["HF_HOME"] == "/custom/hf"
    assert "FLASH_WEIGHT_CACHE_DIR" not in p["env"]


def test_instance_preload_requires_mounted_cache():
    from flash.providers import _instance_bootstrap as b

    # FLASH_WEIGHT_CACHE_DIR rooted at an UNMOUNTED cache -> refuse (would warm ephemeral disk), no download
    r = b.run_preload({"env": {"FLASH_WEIGHT_CACHE_DIR": "/weight-cache/hf-cache/hub"}, "models": ["x/y"]})
    assert r["preloaded"] == []
    assert r["failed"] == {}
    assert "not mounted" in r["error"]


def test_instance_preload_downloads_into_cache(tmp_path, monkeypatch):
    import sys
    import types

    from flash.providers import _instance_bootstrap as b

    calls = []

    def _snap(**k):
        # local_files_only probe -> not cached yet (force the real download); record the real call
        if k.get("local_files_only"):
            raise FileNotFoundError("not cached")
        calls.append(k)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = _snap
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cache_dir_env = str(tmp_path / "hf-cache" / "hub")
    r = b.run_preload({"env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env, "HF_TOKEN": "t"}, "models": ["a/b"]})
    assert r["preloaded"] == ["a/b"]
    assert not r["failed"]
    assert calls[0]["cache_dir"] == str(tmp_path / "hf-cache" / "hub")  # straight into the mount
    assert calls[0]["token"] == "t"
    assert calls[0]["ignore_patterns"] == ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]


def test_instance_preload_skips_download_when_already_cached(tmp_path, monkeypatch):
    """The local_files_only probe (HF's own resolution, not a dir-name guess) marks an existing
    snapshot already_cached and does NOT re-download it."""
    import sys

    from flash.providers import _instance_bootstrap as b

    real_downloads = []

    def _snap(**k):
        if k.get("local_files_only"):
            return "/cached"  # probe SUCCEEDS -> already on the volume
        real_downloads.append(k)  # must never be reached
        return "/dl"

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = _snap
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cache_dir_env = str(tmp_path / "hf-cache" / "hub")
    r = b.run_preload({"env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env}, "models": ["a/b"]})
    assert r["already_cached"] == ["a/b"]
    assert r["preloaded"] == []
    assert real_downloads == []  # no network re-download for a cache hit


def test_instance_preload_block_device_requires_mount_sentinel(tmp_path, monkeypatch):
    """A block-volume preload with the mount dir present but NO sentinel must refuse.

    Regression: a failed/absent volume attach lets Docker bind an EMPTY host dir, so isdir(mount)
    passes; without requiring the on-device sentinel the worker would warm EPHEMERAL disk and report
    success. The cloud-init preamble writes .flash-cache-mounted only onto a real mount.
    """
    import sys

    from flash.providers import _instance_bootstrap as b

    def _boom(**k):
        raise AssertionError("must not download when the block volume isn't really mounted")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = _boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cache_dir_env = str(tmp_path / "hf-cache" / "hub")  # grandparent (the mount) exists but has no sentinel
    r = b.run_preload({"env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env}, "models": ["a/b"],
                       "cache_block_device": True, "cache_mount_marker": ".flash-cache-mounted"})
    assert r["preloaded"] == []
    assert "not mounted" in r["error"]


def test_instance_preload_block_device_warms_when_sentinel_present(tmp_path, monkeypatch):
    """With the on-device sentinel present, a block-volume preload proceeds to download."""
    import sys

    from flash.providers import _instance_bootstrap as b

    calls = []

    def _snap(**k):
        if k.get("local_files_only"):
            raise FileNotFoundError("not cached")  # force the real download
        calls.append(k)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = _snap
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    mount = tmp_path
    (mount / ".flash-cache-mounted").write_text("")  # preamble's real-mount sentinel
    cache_dir_env = str(mount / "hf-cache" / "hub")
    r = b.run_preload({"env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env}, "models": ["a/b"],
                       "cache_block_device": True, "cache_mount_marker": ".flash-cache-mounted"})
    assert r["preloaded"] == ["a/b"]
    assert not r["failed"]


def test_block_device_preamble_writes_mount_sentinel():
    """The block-device preamble drops the sentinel only on a SUCCESSFUL mount."""
    from flash.providers import _instance

    pre = _instance._cache_block_device_setup(
        {"cache_block_device": True, "cache_host_mount": "/mnt/cache", "cache_size_gb": 100}
    )
    # sentinel is written inside the mount-success branch (after `mount ... &&`), not unconditionally
    assert "touch '/mnt/cache/.flash-cache-mounted'" in pre
    assert _instance.CACHE_MOUNT_MARKER == ".flash-cache-mounted"


def test_nfs_mount_check_verifies_mountpoint_and_writes_sentinel():
    """The NFS (Lambda) preamble drops the sentinel ONLY when the host path is a real mountpoint, so an
    auto-created empty Docker-bind dir (failed/unready NFS) is detectable in-container."""
    from flash.providers import _instance

    pre = _instance._cache_nfs_mount_check(
        {"cache_host_mount": "/lambda/nfs/flash-weights"}  # NFS: no cache_block_device
    )
    assert "mountpoint -q '/lambda/nfs/flash-weights'" in pre  # gates on a REAL mount
    assert "touch '/lambda/nfs/flash-weights/.flash-cache-mounted'" in pre
    # No-op for block-volume (handled by the block preamble) and for cold runs.
    assert _instance._cache_nfs_mount_check(
        {"cache_host_mount": "/mnt/cache", "cache_block_device": True}) == ""
    assert _instance._cache_nfs_mount_check({}) == ""


def test_instance_preload_nfs_requires_mount_sentinel(tmp_path, monkeypatch):
    """A Lambda (NFS) preload whose mount dir exists but has NO sentinel must refuse — Docker's -v bind
    auto-creates a missing host dir, so isdir(mount) alone can't prove the NFS actually mounted."""
    import sys

    from flash.providers import _instance_bootstrap as b

    def _boom(**k):
        raise AssertionError("must not download when the NFS cache isn't really mounted")

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = _boom
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    cache_dir_env = str(tmp_path / "hf-cache" / "hub")  # grandparent (the mount) exists but has no sentinel
    r = b.run_preload({"env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env}, "models": ["a/b"],
                       "cache_mount_marker": ".flash-cache-mounted"})  # NFS: no cache_block_device
    assert r["preloaded"] == []
    assert "not mounted" in r["error"]
    assert "NFS" in r["error"]


def test_instance_preload_nfs_warms_when_sentinel_present(tmp_path, monkeypatch):
    """With the NFS preamble's real-mount sentinel present, a Lambda preload proceeds to download."""
    import sys

    from flash.providers import _instance_bootstrap as b

    calls = []

    def _snap(**k):
        if k.get("local_files_only"):
            raise FileNotFoundError("not cached")
        calls.append(k)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = _snap
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    (tmp_path / ".flash-cache-mounted").write_text("")
    cache_dir_env = str(tmp_path / "hf-cache" / "hub")
    r = b.run_preload({"env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env}, "models": ["a/b"],
                       "cache_mount_marker": ".flash-cache-mounted"})
    assert r["preloaded"] == ["a/b"]


def test_build_payload_carries_mount_marker_for_nfs_cache():
    """A cache-attached preload payload (even NFS) carries cache_mount_marker so the in-container check
    can require the sentinel regardless of substrate."""
    from flash.providers import _instance

    p = _instance.build_payload(
        _preload_spec(), 0, 0, arm="lambda",
        cache_host_mount="/lambda/nfs/flash-weights", mode="preload", models=["a/b"],
    )
    assert p["cache_mount_marker"] == _instance.CACHE_MOUNT_MARKER
    assert "cache_block_device" not in p  # NFS, not a block volume


def test_preload_wall_cap_timer_armed_and_cancellable(monkeypatch):
    """run_preload has no worker subprocess, so the preload branch arms a wall-cap watchdog timer that
    hard-exits the box if a download hangs past max_wall_s. The timer is cancellable on a clean finish."""
    from flash.providers import _instance_bootstrap as b

    cap = b._arm_preload_wall_cap({"max_wall_s": 999})
    assert cap is not None
    timer, done = cap  # (timer, done-event): the caller sets `done` then cancels on a clean finish
    assert timer.is_alive()
    assert not done.is_set()
    # A clean finish sets `done` so a wall expiry racing it no-ops in _fire, then cancels the timer.
    done.set()
    timer.cancel()
    # No cap requested -> no timer (e.g. a 0/absent wall).
    assert b._arm_preload_wall_cap({"max_wall_s": 0}) is None
    assert b._arm_preload_wall_cap({}) is None


def test_lambda_launch_threads_preload_mode_into_payload(monkeypatch):
    """launch_and_submit(mode='preload', models=...) embeds a preload payload in the cache user_data."""
    import base64
    import json as _json

    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    launched = {}
    monkeypatch.setattr(lambda_api, "ensure_filesystem", lambda n, r: f"/lambda/nfs/{n}")
    monkeypatch.setattr(lambda_api, "resolve_ssh_key_names", lambda: ["k"], raising=False)
    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["k"])
    monkeypatch.setattr(
        lambda_api, "launch_instance",
        lambda **kw: launched.update(kw) or "i-123",
    )
    from tests.test_lambda_runner import _inst  # reuse the runner's instance candidate

    jobs.launch_and_submit(
        _preload_spec(), seed=0, instances=[_inst()], attempt=0,
        mode="preload", models=["Qwen/Qwen3.5-0.8B"],
    )
    # decode the base64 payload embedded in the cache user_data
    ud = launched["user_data"]
    b64 = ud.split("FLASH_PAYLOAD_EOF")[1].strip()
    payload = _json.loads(base64.b64decode(b64))
    assert payload["mode"] == "preload"
    assert payload["models"] == ["Qwen/Qwen3.5-0.8B"]


# ---------------------------------------------------------------------------
# Instance-provider WARM orchestrator (warm_instances): launch -> poll marker -> terminate
# ---------------------------------------------------------------------------
def _cand(region):
    return types.SimpleNamespace(region=region)


def _wire_warm(monkeypatch, marker):
    """Stub the warm path: status repo, the provider's usable_instances/launch/terminate, marker poll."""
    import json as _json

    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    launched, terminated = [], []
    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(
        preload, "make_hf_text_reader",
        lambda repo, path, min_interval_s=45.0: (lambda force=False: _json.dumps(marker) if marker else None),
    )

    def fake_launch(spec, seed, instances, attempt=0, mode=None, models=None, **k):
        launched.append((instances[0].region, mode, tuple(models or [])))

    monkeypatch.setattr(lj, "launch_and_submit", fake_launch)
    monkeypatch.setattr(lj, "terminate_run_instances", lambda rid: terminated.append(rid))
    return preload, lj, launched, terminated


def test_warm_instances_one_launch_per_region_and_terminates(monkeypatch):
    preload, lj, launched, terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}})
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [_cand("us-east-1"), _cand("us-east-1"), _cand("us-west-2")])

    res = preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert sorted(r["region"] for r in res) == ["us-east-1", "us-west-2"]  # us-east-1 deduped
    assert all(r["status"] == "ok" for r in res)
    assert all(m == "preload" for _, m, _ in launched)  # download-only launches
    assert len(terminated) == 2  # every launch ALWAYS torn down


def test_warm_instance_partial_when_a_model_failed(monkeypatch):
    preload, lj, _launched, terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {"c/d": "gated"}})
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [_cand("us-east-1")])
    res = preload.warm_instances(models=["a/b", "c/d"], timeout_s=5, poll_interval_s=0.0)
    assert [r["status"] for r in res] == ["partial"]
    assert len(terminated) == 1


def test_warm_instance_times_out_when_no_marker(monkeypatch):
    import types as _types

    preload, lj, _launched, terminated = _wire_warm(monkeypatch, None)  # marker never appears
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [_cand("us-east-1")])
    # Fake clock: jump straight past the deadline so the poll loop exits at once (no real 60s wait,
    # which the effective-budget floor of 60 would otherwise impose). sleep is a no-op.
    clock = {"t": 0.0}
    fake_time = _types.SimpleNamespace(
        time=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + 1e6))
    monkeypatch.setattr(preload, "time", fake_time)
    res = preload.warm_instances(models=["a/b"], timeout_s=0, poll_interval_s=0.0)
    assert [r["status"] for r in res] == ["timeout"]
    assert len(terminated) == 1  # terminated regardless of timeout


def test_warm_poll_budget_matches_worker_wall_cap_below_floor(monkeypatch):
    """`--timeout-s` under the 60s floor: the driver polls for the SAME effective budget (60) the
    worker wall cap is floored to — so a short timeout can't kill a still-running preload early.

    Regression for the mismatch where the spec capped at max(60, timeout_s) but the poll used the raw
    timeout_s (e.g. timeout_s=10 -> worker 60s, driver 10s -> premature terminate)."""
    import types as _types

    preload, lj, _launched, terminated = _wire_warm(monkeypatch, None)
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [_cand("us-east-1")])

    # Capture the wall cap the spec was built with.
    wall_caps = []
    orig_spec = preload._preload_instance_spec
    monkeypatch.setattr(
        preload, "_preload_instance_spec",
        lambda gpu, run_id, wall_s=1800: wall_caps.append(wall_s) or orig_spec(gpu, run_id, wall_s),
    )
    # Fake clock that ticks 1 (virtual) second per sleep, so we can read how long the poll ran.
    clock = {"t": 1000.0}
    start = clock["t"]
    monkeypatch.setattr(
        preload, "time",
        _types.SimpleNamespace(time=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + 1)),
    )

    res = preload.warm_instances(models=["a/b"], timeout_s=10, poll_interval_s=0.0)  # 10 < 60 floor
    assert [r["status"] for r in res] == ["timeout"]
    assert wall_caps == [60]  # worker wall cap floored to 60
    # the poll ran the FULL 60s effective budget (would have stopped at ~10s under the old bug)
    assert clock["t"] - start >= 60
    assert len(terminated) == 1


def test_warm_instance_terminates_on_launch_failure(monkeypatch):
    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    terminated = []
    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(preload, "make_hf_text_reader", lambda *a, **k: (lambda force=False: None))
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [_cand("us-east-1")])
    monkeypatch.setattr(lj, "launch_and_submit",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no capacity")))
    monkeypatch.setattr(lj, "terminate_run_instances", lambda rid: terminated.append(rid))
    res = preload.warm_instances(models=["a/b"], providers=["lambda"], timeout_s=5, poll_interval_s=0.0)
    assert res[0]["status"] == "error"
    assert "no capacity" in res[0]["error"]
    assert len(terminated) == 1  # finally still tears down


def test_warm_instance_stops_early_on_failure_marker(monkeypatch):
    """The box can die BEFORE run_preload uploads preload_result.json (docker/GPU never ready, image
    pull fails, the bootstrap crashes early); it still writes the <arm>_attempt0.json failure marker.
    The driver must watch that marker and free the paid box at once instead of polling the full budget."""
    import types as _types

    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    terminated = []
    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)

    # Two distinct readers by path: the completion file never appears; the attempt-failure marker does.
    def reader_factory(repo, path, min_interval_s=45.0):
        if path.endswith("preload_result.json"):
            return lambda force=False: None
        return lambda force=False: '{"ok": false, "error": "image pull failed"}'

    monkeypatch.setattr(preload, "make_hf_text_reader", reader_factory)
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [_cand("us-east-1")])
    monkeypatch.setattr(lj, "launch_and_submit", lambda *a, **k: None)
    monkeypatch.setattr(lj, "terminate_run_instances", lambda rid: terminated.append(rid))
    # Clock that ticks 1 virtual second per sleep — if the early-out works, the poll exits on the FIRST
    # pass (well under the 60s floor), proving the box wasn't held to the full budget.
    clock = {"t": 1000.0}
    start = clock["t"]
    monkeypatch.setattr(
        preload, "time",
        _types.SimpleNamespace(time=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + 1)),
    )
    res = preload.warm_instances(models=["a/b"], providers=["lambda"], timeout_s=600, poll_interval_s=0.0)
    assert res[0]["status"] == "error"
    assert "image pull failed" in res[0]["error"]
    assert clock["t"] - start < 5  # bailed immediately, did NOT poll the full 600s
    assert len(terminated) == 1


def test_warm_instances_no_capacity_returns_empty(monkeypatch):
    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [])
    assert preload.warm_instances(models=["a/b"]) == []


def test_warm_instances_uses_per_provider_default_gpu(monkeypatch):
    """With no --gpu override, each provider warms with a GPU IT offers (A10 is Lambda's default)."""
    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    seen = {}

    def _rec(provider):
        def _u(gpu):
            seen[provider] = gpu
            return []
        return _u

    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(lj, "usable_instances", _rec("lambda"))
    preload.warm_instances(models=["a/b"])  # gpu=None -> per-provider defaults
    assert seen["lambda"] == preload._PRELOAD_GPU_BY_PROVIDER["lambda"] == "A10"


def test_warm_instances_explicit_gpu_overrides_all_providers(monkeypatch):
    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    seen = {}

    def _rec(provider):
        def _u(gpu):
            seen[provider] = gpu
            return []
        return _u

    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(lj, "usable_instances", _rec("lambda"))
    preload.warm_instances(models=["a/b"], gpu="H100")
    assert seen == {"lambda": "H100"}


def test_warm_instances_requires_status_repo_before_launch(monkeypatch):
    """With targets available, warm must validate the status repo BEFORE launching any paid box.
    Capacity enumeration is cheap/read-only, so it runs first (to decide if there's anything to do);
    the status-repo guard gates the actual LAUNCH, which is what costs money."""
    import types

    import pytest

    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    # Lambda (an unfiltered provider) has capacity -> a real launch target exists.
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [types.SimpleNamespace(region="us-east-1")])
    monkeypatch.setattr(
        lj,
        "launch_and_submit",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch before status repo")),
    )
    monkeypatch.setattr(
        preload,
        "_ensure_status_repo",
        lambda token: (_ for _ in ()).throw(RuntimeError("401 unauthorized")),
    )
    with pytest.raises(RuntimeError, match="status repo"):
        preload.warm_instances(models=["a/b"])


def test_warm_instances_no_targets_is_noop_without_status_repo(monkeypatch):
    """No provider capacity -> documented no-op: warm returns [] and must NOT require the status repo
    (else an empty warm on an unconfigured / at-capacity host would hard-fail on a missing HF_TOKEN)."""
    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [])
    monkeypatch.setattr(
        preload,
        "_ensure_status_repo",
        lambda token: (_ for _ in ()).throw(AssertionError("status repo must not be required with no targets")),
    )
    assert preload.warm_instances(models=["a/b"]) == []


def test_warm_instances_cli_dry_run(monkeypatch):
    from flash.providers.runpod import preload

    called = {"n": 0}
    monkeypatch.setattr(preload, "warm_instances", lambda **k: called.__setitem__("n", called["n"] + 1) or [])
    assert preload.main(["--warm-instances", "--dry-run"]) == 0
    assert called["n"] == 0  # dry-run launches nothing


def test_cli_gpu_default_is_none_per_mode(monkeypatch):
    """--gpu defaults to None; each mode applies its OWN default downstream (no sentinel hack).

    Regression: --gpu used to default to _PRELOAD_GPU ('RTX 4090') and --warm-instances used a
    `args.gpu != _PRELOAD_GPU` comparison — so a user explicitly asking for RTX 4090 on instance
    warming was wrongly treated as 'no override'. None must pass through cleanly.
    """
    from flash.providers.runpod import preload

    # --warm-instances, no --gpu -> warm_instances receives gpu=None (it applies _PRELOAD_INSTANCE_GPU)
    seen = {}
    monkeypatch.setattr(preload, "warm_instances", lambda **k: seen.update(k) or [])
    assert preload.main(["--warm-instances"]) == 0
    assert seen["gpu"] is None

    # --warm-instances --gpu 'RTX 4090' -> passed THROUGH (the previously-broken explicit-default case)
    seen.clear()
    assert preload.main(["--warm-instances", "--gpu", "RTX 4090"]) == 0
    assert seen["gpu"] == "RTX 4090"


def test_cli_runpod_warm_gpu_falls_back_to_preload_default(monkeypatch):
    """RunPod warm path: a None --gpu resolves to _PRELOAD_GPU so None never reaches deploy."""
    from flash.providers.runpod import preload

    seen = {}
    monkeypatch.setattr(preload, "warm_weight_cache", lambda **k: seen.update(k) or [])
    # default (no --gpu) -> the RunPod warm default
    assert preload.main(["--datacenters", "US-CA-2"]) == 0
    assert seen["gpu"] == preload._PRELOAD_GPU
    # explicit override still wins (even a non-default class)
    seen.clear()
    assert preload.main(["--datacenters", "US-CA-2", "--gpu", "H100"]) == 0
    assert seen["gpu"] == "H100"
