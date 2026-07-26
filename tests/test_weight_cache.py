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

import logging
import os
import tempfile
import types

import pytest

from flash.spec import GpuSpec, JobSpec, TrainSpec

_RUNPOD_FINGERPRINT = "rpk-0123456789ab"


def _oversized_model_info():
    """A synthetic catalog entry too large for the shared cache.

    Sized off WEIGHT_CACHE_VOLUME_GB so it stays oversized if the volume grows again — the real
    catalog now fits entirely, so the size gate needs a stand-in to stay covered.
    """
    from flash import runner
    from flash.catalog import ModelInfo

    return ModelInfo(
        id="test/oversized",
        display_name="oversized",
        params="huge",
        algos=("sft",),
        min_vram_gb=80,
        params_b=float(runner.WEIGHT_CACHE_VOLUME_GB),  # peak = 4x params_b GB >> the volume
    )


def _vol_spec(name="flash-weights", gb=100, **gpu):
    return JobSpec(
        model="m",
        gpu=GpuSpec(network_volume=name, network_volume_gb=gb, **gpu),
        seed=0,
    )


def _ndc() -> int:
    """Size of the ALLOWED datacenter set (DataCenter.all()) — what the endpoint's `datacenter` list
    spans. Derived from the same source the code uses, so it tracks the SDK's storage-DC set.
    """
    from flash.providers.runpod.jobs import weight_cache_datacenters

    n = len(weight_cache_datacenters())
    assert n > 1
    return n


# ---------------------------------------------------------------------------
# spec carrier round-trips (network_volume is platform-managed: it must survive every
# to_internal_dict()->from_dict() hop, the control-plane/worker carrier. it is intentionally
# absent from the public to_dict().)
# ---------------------------------------------------------------------------
def test_network_volume_round_trips():
    spec = _vol_spec(gb=123)
    again = JobSpec.from_dict(spec.to_internal_dict())
    assert again.gpu.network_volume == "flash-weights"
    assert again.gpu.network_volume_gb == 123


def test_network_volume_absent_from_public_spec():
    # network_volume / network_volume_gb are platform-managed and must NOT appear in the public spec.
    public = _vol_spec(gb=123).to_dict()
    assert "network_volume" not in public["gpu"]
    assert "network_volume_gb" not in public["gpu"]


def test_default_spec_has_no_volume():
    assert JobSpec(model="m").gpu.network_volume is None


def test_stale_datacenter_key_rejected():
    with pytest.raises(ValueError, match=r"gpu has unknown key\(s\): datacenter"):
        JobSpec.from_dict(
            {"model": "m", "gpu": {"datacenter": "EU-RO-1", "network_volume": "v"}}
        )


def test_network_volume_gb_tolerant_of_bad_values():
    # Platform-managed field: null/empty/"0"/0/negative/non-numeric/missing -> default 100 (never
    # crash int(), never round-trip a nonsensical size). Valid positive sizes pass through.
    for raw in (None, "", 0, "0", -5, "-5", "abc", True, False):
        spec = JobSpec.from_dict(
            {"model": "m", "gpu": {"network_volume": "v", "network_volume_gb": raw}}
        )
        assert spec.gpu.network_volume_gb == 100, f"{raw!r} should default to 100"
    assert (
        JobSpec.from_dict({"model": "m", "gpu": {"network_volume": "v"}}).gpu.network_volume_gb
        == 100
    )
    for raw in (200, "150"):
        spec = JobSpec.from_dict(
            {"model": "m", "gpu": {"network_volume": "v", "network_volume_gb": raw}}
        )
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

    assert (
        jobs.weight_cache_volume_name("flash-weights", DataCenter.US_CA_2)
        == "flash-weights-us-ca-2"
    )


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

    from flash.providers.runpod import auth, jobs, keys

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    keys.reset()
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
    monkeypatch.setattr(auth, "ensure_auth", lambda: "test-key")
    monkeypatch.setattr(jobs, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "_patch_runpod_backoff", lambda: None)

    eid, _name, fingerprint = jobs.deploy_train_endpoint(
        "RTX 4090",
        spec=_vol_spec(),
        disk_gb=None,
    )
    assert eid == "ep-abc"
    assert fingerprint == jobs.runpod_api.key_fingerprint("test-key")
    assert len(captured["volume"]) == _ndc()  # EAGER: a volume in every storage DC
    assert len(captured["datacenter"]) == _ndc()  # allowed across all storage DCs
    assert len({v.name for v in captured["volume"]}) == _ndc()  # distinct per-DC names
    assert all(v.name.startswith("flash-weights-") for v in captured["volume"])


def test_deploy_train_endpoint_no_volume_when_spec_has_none(monkeypatch):
    import runpod_flash
    import runpod_flash.core.resources.resource_manager as rm_mod

    from flash.providers.runpod import auth, jobs, keys

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    keys.reset()
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
    monkeypatch.setattr(auth, "ensure_auth", lambda: "test-key")
    monkeypatch.setattr(jobs, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "_patch_runpod_backoff", lambda: None)

    jobs.deploy_train_endpoint("RTX 4090", spec=JobSpec(model="m"), disk_gb=None)
    assert "volume" not in captured
    assert "datacenter" not in captured


# ---------------------------------------------------------------------------
# worker weight_cache_env / build_worker_env redirect
# ---------------------------------------------------------------------------
def test_weight_cache_env_is_base_model_scoped():
    from flash.providers._worker import weight_cache_env

    env = weight_cache_env("/runpod-volume")
    # BASE-MODEL-SCOPED: FLASH_WEIGHT_CACHE_DIR points the base-model prefetch at the mount's HF hub
    # layout. It must NOT set a process-global HF_HOME (that leaked env/reward downloads onto the
    # shared multi-tenant mount — issue #252). The executable kernel-JIT caches are also never on it.
    assert env == {"FLASH_WEIGHT_CACHE_DIR": "/runpod-volume/hf-cache/hub"}
    assert "HF_HOME" not in env
    for k in (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TILELANG_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
    ):
        assert k not in env


def test_weight_cache_env_custom_mount():
    from flash.providers._worker import weight_cache_env

    assert weight_cache_env("/workspace")["FLASH_WEIGHT_CACHE_DIR"] == "/workspace/hf-cache/hub"


def test_build_worker_env_sets_base_model_cache_with_volume():
    from flash.providers._worker import build_worker_env

    env = build_worker_env(_vol_spec(), 0)
    assert env["FLASH_WEIGHT_CACHE_DIR"] == "/runpod-volume/hf-cache/hub"
    # The leak fix: no process-global HF_HOME redirect, so env/reward downloads use the ephemeral cache.
    assert "HF_HOME" not in env


def test_build_worker_env_no_cache_without_volume():
    from flash.providers._worker import build_worker_env

    env = build_worker_env(JobSpec(model="m", seed=0), 0)
    # Without a volume the base-model cache var must NOT be set (pointing at a missing mount).
    assert "FLASH_WEIGHT_CACHE_DIR" not in env
    assert "HF_HOME" not in env


def test_build_worker_env_per_run_override_wins():
    from flash.providers._worker import build_worker_env

    spec = _vol_spec()
    # network_volume is managed -> carried by the internal dict, so the cache redirect is present
    # and the per-run [worker_env] override (merged last) genuinely wins over it.
    spec = JobSpec.from_dict(
        {**spec.to_internal_dict(), "worker_env": {"FLASH_WEIGHT_CACHE_DIR": "/custom/hub"}}
    )
    assert build_worker_env(spec, 0)["FLASH_WEIGHT_CACHE_DIR"] == "/custom/hub"


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
        calls.append(
            {"repo_id": repo_id, "cache_dir": cache_dir, "ignore_patterns": ignore_patterns}
        )
        if (
            cache_dir
        ):  # simulate a real download landing on the (mount) cache: create the repo folder
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
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3.5-0.8B",
            "cache_dir": shared_hub,
            "ignore_patterns": ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
        }
    ]
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

    assert (
        calls[0]["cache_dir"] is None
    )  # mount absent -> ephemeral, never the missing /runpod-volume path


def test_prefetch_model_starts_no_download_at_deadline(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASH_WEIGHT_CACHE_DIR", raising=False)
    hf, calls = _patch_prefetch_io(monkeypatch, tmp_path / "ephemeral" / "hub")
    monkeypatch.setattr(hf._w, "_remaining_worker_wall_seconds", lambda: 0.0)

    hf.prefetch_model("Qwen/Qwen3.5-0.8B")

    assert calls == []


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
    from flash.providers._worker import strip_runpod_volume_env

    env = {
        "FLASH_WEIGHT_CACHE_DIR": "/runpod-volume/hf-cache/hub",
        "X": "/runpod-volume/foo",
        "KEEP": "v",
        "HF_TOKEN": "t",
    }
    out = strip_runpod_volume_env(env)
    assert "FLASH_WEIGHT_CACHE_DIR" not in out
    assert "X" not in out
    assert out == {"KEEP": "v", "HF_TOKEN": "t"}  # non-/runpod-volume vars preserved


def test_instance_payload_strips_runpod_volume_redirect():
    # The RunPod weight-cache base-model redirect must NOT leak into a Lambda payload —
    # those instances never mount /runpod-volume. (build_worker_env DOES set it; the instance strips.)
    from flash.providers import _instance
    from flash.providers._worker import build_worker_env

    # network_volume is managed -> carried by the internal dict (the leak source that build_worker_env
    # turns into the /runpod-volume redirect).
    spec = JobSpec.from_dict(
        {**_vol_spec().to_internal_dict(), "run_id": "r", "model": "Qwen/Qwen3.5-0.8B"}
    )
    assert build_worker_env(spec, 0)["FLASH_WEIGHT_CACHE_DIR"].startswith(
        "/runpod-volume"
    )  # leak source
    for arm in ("lambda",):
        env = _instance.build_payload(
            spec,
            seed=0,
            attempt=0,
            arm=arm,
            deadline_at=10_000_000_000.0,
        )["env"]
        assert not env.get("FLASH_WEIGHT_CACHE_DIR", "").startswith("/runpod-volume"), arm


# ---------------------------------------------------------------------------
# runner._assign_weight_cache_volume — fully managed, no knobs; gated to PUBLIC catalog runs only
# (the shared cross-tenant cache must never hold private/gated weights — confidentiality boundary)
# ---------------------------------------------------------------------------
def test_assign_weight_cache_attaches_to_catalog_run():
    from flash import runner

    out = runner._assign_weight_cache_volume(JobSpec(model="m", run_id="r", model_policy="catalog"))
    assert out.gpu.network_volume == runner.WEIGHT_CACHE_VOLUME_NAME == "flash-weights"
    assert out.gpu.network_volume_gb == runner.WEIGHT_CACHE_VOLUME_GB


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

    out = runner._assign_weight_cache_volume(
        JobSpec(model="some-org/private-model", run_id="r", model_policy="allow")
    )
    assert (
        out.gpu.network_volume is None
    )  # cache-less: no shared-mount redirect for a possibly-private model


def test_assign_weight_cache_strips_preset_shared_cache_on_open_model():
    # The gate takes PRECEDENCE over the "honor an existing volume" no-op: a programmatic open-model
    # spec that ALREADY pinned the SHARED cache name must be FORCED cache-less, not bypass the gate.
    from flash import runner

    spec = JobSpec.from_dict(
        {
            "model": "some-org/private-model",
            "run_id": "r",
            "model_policy": "allow",
            "gpu": {
                "type": "A10",
                "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume is None  # the pre-set shared cache was stripped


def test_assign_weight_cache_keeps_per_org_volume_on_open_model():
    # A NON-shared (per-org / custom) volume on an open run is the intended escape hatch — left intact.
    from flash import runner

    spec = JobSpec.from_dict(
        {
            "model": "some-org/private-model",
            "run_id": "r",
            "model_policy": "allow",
            "gpu": {
                "type": "A10",
                "network_volume": "org-123-private-cache",
                "network_volume_gb": 100,
            },
        }
    )
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
    # network_volume is managed -> carried by the internal dict; an already-pinned volume is honored.
    spec = JobSpec.from_dict({**spec.to_internal_dict(), "model_policy": "catalog", "run_id": "r"})
    out = runner._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume == "explicit-vol"  # an explicit/test value is never clobbered


def test_assign_weight_cache_skips_oversized_catalog_model():
    # SIZE GATE: a model whose peak download footprint exceeds the shared cache must NOT be
    # attached — that would redirect the base-model download onto an undersized mount and overflow
    # mid-download. It is left cache-less and downloads to the container disk instead. Every model
    # in the current catalog fits (see test_every_catalog_model_fits_the_weight_cache), so the gate
    # is exercised with a synthetic oversized entry rather than a real one.
    from flash import runner

    info = _oversized_model_info()
    out = runner._assign_weight_cache_volume(
        JobSpec(model=info.id, run_id="r", model_policy="catalog"), info
    )
    assert out.gpu.network_volume is None  # too big for the shared cache -> cache-less


def test_assign_weight_cache_strips_preset_shared_cache_on_oversized_catalog_model():
    # SIZE GATE re-applies to a pre-set SHARED-cache name: a programmatic/stale catalog spec that
    # already pinned ``flash-weights`` for an oversized model must NOT bypass the gate via the
    # "honor an existing volume" no-op and redirect the download onto the undersized mount. It is
    # stripped cache-less.
    from flash import runner

    info = _oversized_model_info()
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "model_policy": "catalog",
            "gpu": {
                "type": "B200",
                "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume is None  # oversized -> the pre-set shared cache was stripped


def test_assign_weight_cache_keeps_preset_shared_cache_on_fitting_catalog_model():
    # The re-gate only strips when OVERSIZED: a fitting model that already carries the shared cache
    # keeps it (the pin is correct), exercising the honor-existing path for a shared-name spec.
    from flash import runner
    from flash.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "model_policy": "catalog",
            "gpu": {
                "type": "H100",
                "network_volume": runner.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "flash-weights"  # fits -> kept


def test_assign_weight_cache_keeps_preset_custom_volume_on_oversized_catalog_model():
    # The re-gate is scoped to the SHARED name only: a custom/per-org volume on an oversized model is
    # left intact (the caller owns its sizing — it may be a 200 GB org cache that DOES fit).
    from flash import runner
    from flash.catalog import MODELS

    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "model_policy": "catalog",
            "gpu": {
                "type": "B200",
                "network_volume": "org-123-big-cache",
                "network_volume_gb": 400,
            },
        }
    )
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


def test_fits_weight_cache_is_size_based():
    from flash import runner

    # Oversized models are still excluded: the gate is a size check, not "always true".
    assert not runner._fits_weight_cache(_oversized_model_info())


def test_every_catalog_model_fits_the_weight_cache():
    # The largest models have the slowest cold downloads, so they are exactly the ones the cache
    # must cover. WEIGHT_CACHE_VOLUME_GB must stay >= the peak footprint of the biggest catalog
    # entry; if a larger model is added, grow the volume rather than silently skipping its cache.
    from flash import runner
    from flash.catalog import MODELS

    for mid, info in MODELS.items():
        assert runner._fits_weight_cache(info), (
            f"{mid} ({info.params_b}B) no longer fits the "
            f"{runner.WEIGHT_CACHE_VOLUME_GB} GB weight cache"
        )


def test_submit_job_assigns_weight_cache(monkeypatch):
    # Integration: the assignment is wired into submit_job and visible on the effective worker spec.
    # network_volume is platform-managed -> stripped from the public status.spec, so observe the
    # managed assignment on the effective-preparation worker spec the worker actually runs.
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
        status = runner.submit_job(spec, dry_run=True)
    gpu = status.effective_preparation["worker_spec"]["gpu"]
    assert gpu["network_volume"] == "flash-weights"
    assert gpu["network_volume_gb"] == runner.WEIGHT_CACHE_VOLUME_GB
    # and it must NOT leak into the public spec
    assert "network_volume" not in status.spec["gpu"]


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


def test_effective_spec_persists_managed_cache_removal(monkeypatch):
    # The SHARED platform cache may be dropped on a capacity fallback. network_volume is managed and
    # lives only in the prior preparation snapshot, so the committed shared cache is recorded there;
    # the re-prepared cache-less spec must persist without the removal guard firing.
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        runner = fresh_runner(tmp, monkeypatch)
        public = JobSpec.from_dict(
            {**_vol_spec().to_internal_dict(), "run_id": "managed-cache-fallback"}
        )
        assert public.gpu.network_volume == runner.WEIGHT_CACHE_VOLUME_NAME
        selected_dict = public.to_internal_dict()
        selected_dict["gpu"]["network_volume"] = None
        selected = JobSpec.from_dict(selected_dict)
        runner._save_status(
            runner.RunStatus(
                run_id=public.run_id,
                state="provisioning",
                spec=public.to_dict(),
                effective_preparation={
                    "worker_spec": public.to_internal_dict(),  # committed WITH the shared cache
                    "adapter_identity": None,
                    "preparation_digest": "seed",
                },
            )
        )

        assert runner._persist_effective_worker_spec(selected)

        stored = runner.get_status(public.run_id)
        assert stored.effective_preparation["worker_spec"]["gpu"]["network_volume"] is None
        assert stored.effective_preparation["adapter_identity"] is None
        assert runner.effective_spec_from_status(stored).gpu.network_volume is None


def test_effective_spec_rejects_custom_volume_removal(monkeypatch):
    # A per-org escape-hatch volume an open-model run opted into must never be silently removed.
    # network_volume is managed and no longer travels in the public spec, so the committed custom
    # volume lives only in the prior preparation snapshot; dropping it there must fail closed.
    import pytest

    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        runner = fresh_runner(tmp, monkeypatch)
        committed = JobSpec.from_dict(
            {
                **_vol_spec(name="org-1234-private").to_internal_dict(),
                "run_id": "custom-cache-fallback",
            }
        )
        assert committed.gpu.network_volume != runner.WEIGHT_CACHE_VOLUME_NAME
        runner._save_status(
            runner.RunStatus(
                run_id=committed.run_id,
                state="provisioning",
                spec=committed.to_dict(),
                effective_preparation={
                    "worker_spec": committed.to_internal_dict(),  # committed WITH the custom volume
                    "adapter_identity": None,
                    "preparation_digest": "seed",
                },
            )
        )
        selected_dict = committed.to_internal_dict()
        selected_dict["gpu"]["network_volume"] = None
        selected = JobSpec.from_dict(selected_dict)

        with pytest.raises(ValueError, match="effective preparation"):
            runner._persist_effective_worker_spec(selected)


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
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
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
    # The whole catalog fits the volume, so the large checkpoints — the ones with the
    # slowest cold downloads — are warmed too.
    assert "Qwen/Qwen3.6-27B" in ids
    assert "Qwen/Qwen3.6-35B-A3B" in ids


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

    def fake_snapshot(
        repo_id, token=None, cache_dir=None, local_files_only=False, ignore_patterns=None
    ):
        calls.append(
            {
                "repo": repo_id,
                "cache_dir": cache_dir,
                "probe": local_files_only,
                "ignore": ignore_patterns,
            }
        )
        if local_files_only:
            raise FileNotFoundError("not cached yet")  # force the real download path
        return "/somewhere"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    out = endpoints._train_body(
        {
            "mode": "preload",
            "models": ["Qwen/Qwen3.5-0.8B"],
            "env": {"HF_HOME": "/runpod-volume/hf-cache", "HF_TOKEN": "t"},
        }
    )
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
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: calls.append(k) or "/x",
    )
    for bad in (None, "", "/root/.cache/huggingface", "/tmp/hf"):
        env = {"HF_TOKEN": "t"}
        if bad is not None:
            env["HF_HOME"] = bad
        out = endpoints._train_body(
            {"mode": "preload", "models": ["Qwen/Qwen3.5-0.8B"], "env": env}
        )
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
        lambda_api,
        "list_filesystems",
        lambda: (_ for _ in ()).throw(lambda_api.LambdaApiError("LAMBDA_API_KEY not set")),
    )
    assert (
        preload.teardown_lambda_filesystems() == []
    )  # absent provider -> nothing reclaimed, no raise


def test_teardown_cli_reclaims_all_providers(monkeypatch):
    """`preload --teardown` sweeps RunPod + Lambda in one shot."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "teardown_weight_cache", lambda dcs: ["flash-weights-us-ca-2"])
    monkeypatch.setattr(
        preload, "teardown_lambda_filesystems", lambda: ["lambda:us-east-1/flash-weights"]
    )
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
    monkeypatch.setattr(
        preload,
        "teardown_lambda_filesystems",
        lambda: lam.append(1) or ["lambda:us-east-1/flash-weights"],
    )
    # RunPod raises but the instance provider still gets cleaned up best-effort; the CLI still exits 0.
    assert preload.main(["--teardown"]) == 0
    assert lam == [1]


def test_scoped_teardown_is_runpod_only(monkeypatch):
    """`--teardown --datacenters ...` scopes to RunPod; instance-provider caches are left intact."""
    from flash.providers.runpod import preload

    seen = {}
    monkeypatch.setattr(
        preload,
        "teardown_weight_cache",
        lambda dcs: seen.setdefault("dcs", dcs) or ["flash-weights-us-ca-2"],
    )
    monkeypatch.setattr(
        preload, "teardown_lambda_filesystems", lambda: seen.setdefault("lambda", True) or []
    )
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
    monkeypatch.setattr(
        preload, "teardown_weight_cache", lambda dcs: called.setdefault("runpod", dcs) or []
    )
    monkeypatch.setattr(
        preload, "teardown_lambda_filesystems", lambda: called.setdefault("lambda", True) or []
    )
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
# eager provision: create lambda weight-cache filesystems in every region without a gpu
# ---------------------------------------------------------------------------
def test_provision_lambda_filesystems_covers_every_region(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.runpod import preload

    ensured = []
    monkeypatch.setattr(
        lambda_api, "all_regions", lambda: ["us-east-1", "us-west-2", "europe-central-1"]
    )
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda name, region, deadline_at=None: ensured.append((name, region))
        or f"/lambda/nfs/{name}",
    )
    out = preload.provision_lambda_filesystems()
    # one create-if-absent per region, with the managed cache name
    assert ensured == [
        ("flash-weights", "us-east-1"),
        ("flash-weights", "us-west-2"),
        ("flash-weights", "europe-central-1"),
    ]
    assert out == ["lambda:us-east-1", "lambda:us-west-2", "lambda:europe-central-1"]


def test_provision_lambda_skips_failed_region(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.runpod import preload

    def flaky(name, region, deadline_at=None):
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
        lambda_api,
        "all_regions",
        lambda: (_ for _ in ()).throw(lambda_api.LambdaApiError("LAMBDA_API_KEY not set")),
    )
    assert preload.provision_lambda_filesystems() == []


def test_provision_cli_creates_lambda_filesystems(monkeypatch, capsys):
    """`preload --provision` creates Lambda filesystems (GPU-free) and exits 0."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "provision_lambda_filesystems", lambda: ["lambda:us-east-1"])
    assert preload.main(["--provision"]) == 0
    assert capsys.readouterr().out == "provisioned 1 Lambda filesystem(s): lambda:us-east-1\n"


def test_provision_cli_dry_run_provisions_nothing(monkeypatch):
    from flash.providers.runpod import preload

    called = {"n": 0}
    monkeypatch.setattr(
        preload,
        "provision_lambda_filesystems",
        lambda: called.__setitem__("n", called["n"] + 1) or [],
    )
    assert preload.main(["--provision", "--dry-run"]) == 0
    assert called["n"] == 0  # dry-run touches no provider


def test_preload_one_dc_deploys_pins_single_dc_and_tears_down(monkeypatch):
    from flash.providers.runpod import preload

    calls = {}

    def fake_deploy(
        gpu,
        execution_timeout_ms=None,
        name_suffix=None,
        spec=None,
        endpoint_kwargs=None,
        deadline_at=None,
    ):
        calls["gpu"] = gpu
        calls["suffix"] = name_suffix
        calls["endpoint_kwargs"] = endpoint_kwargs
        return "ep-1", "name-1", _RUNPOD_FINGERPRINT

    submitted = {}

    def fake_submit(eid, payload, **_kw):
        submitted["eid"] = eid
        submitted["payload"] = payload
        return "job-1"

    deleted = []
    monkeypatch.setattr(preload, "deploy_train_endpoint", fake_deploy)
    monkeypatch.setattr(preload.runpod_api, "submit_job", fake_submit)
    monkeypatch.setattr(preload.runpod_api, "delete_endpoint_for_fingerprint", lambda eid, _fingerprint: deleted.append(eid))
    monkeypatch.setattr(
        preload.runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {"status": "COMPLETED", "output": {"preloaded": ["Qwen/Qwen3.5-0.8B"]}},
    )

    out = preload._preload_one_dc(
        "EU-RO-1",
        ["Qwen/Qwen3.5-0.8B"],
        token="tok",
        gpu="RTX 4090",
        timeout_s=60,
        poll_interval_s=0.0,
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
        preload,
        "deploy_train_endpoint",
        lambda *a, **k: ("ep-9", "name-9", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(preload.runpod_api, "submit_job", lambda eid, payload, **_kw: "job-9")
    monkeypatch.setattr(preload.runpod_api, "delete_endpoint_for_fingerprint", lambda eid, _fingerprint: deleted.append(eid))
    monkeypatch.setattr(
        preload.runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {"status": "FAILED", "error": "boom"},
    )

    out = preload._preload_one_dc(
        "US-CA-2", ["m"], token=None, gpu="RTX 4090", timeout_s=60, poll_interval_s=0.0
    )
    assert out["status"] == "error"
    assert deleted == ["ep-9"]  # still torn down on failure


def _stub_preload_deploy(monkeypatch, job_output):
    from flash.providers.runpod import preload

    monkeypatch.setattr(
        preload,
        "deploy_train_endpoint",
        lambda *a, **k: ("ep", "n", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(preload.runpod_api, "submit_job", lambda eid, p, **_kw: "job")
    monkeypatch.setattr(
        preload.runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda eid, _fingerprint: None,
    )
    monkeypatch.setattr(
        preload.runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {"status": "COMPLETED", "output": job_output},
    )


def test_preload_one_dc_partial_when_a_model_fails(monkeypatch):
    # A COMPLETED job whose handler reports per-model failures is NOT a fully warmed region.
    from flash.providers.runpod import preload

    _stub_preload_deploy(
        monkeypatch,
        {
            "preloaded": ["a"],
            "already_cached": [],
            "failed": {"b": "gated repo"},
        },
    )
    out = preload._preload_one_dc(
        "US-CA-2", ["a", "b"], token=None, gpu="g", timeout_s=60, poll_interval_s=0.0
    )
    assert out["status"] == "partial"
    assert out["result"]["failed"] == {"b": "gated repo"}


def test_preload_one_dc_error_when_volume_not_mounted(monkeypatch):
    # The handler's mount-not-mounted hard error must surface as a DC-level error (not silent ok).
    from flash.providers.runpod import preload

    _stub_preload_deploy(
        monkeypatch,
        {
            "preloaded": [],
            "already_cached": [],
            "failed": {},
            "error": "weight-cache volume not mounted at /runpod-volume",
        },
    )
    out = preload._preload_one_dc(
        "US-CA-2", ["a"], token=None, gpu="g", timeout_s=60, poll_interval_s=0.0
    )
    assert out["status"] == "error"
    assert "not mounted" in out["error"]


def test_preload_branch_errors_when_volume_not_mounted(monkeypatch):
    # In the worker handler: if /runpod-volume isn't a real mount, preload must NOT silently warm
    # ephemeral disk — it returns an explicit error and downloads nothing.
    import os as _os

    from flash.providers.runpod.train import endpoints

    monkeypatch.setattr(_os, "environ", dict(_os.environ))
    monkeypatch.setattr(_os.path, "isdir", lambda p: False)  # /runpod-volume not mounted
    out = endpoints._train_body(
        {
            "mode": "preload",
            "models": ["Qwen/Qwen3.5-0.8B"],
            "env": {"HF_HOME": "/runpod-volume/hf-cache"},
        }
    )
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
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": "flash-1700000000-abcd1234",
            "train": {"epochs": 1, "max_examples": 8, "hf_repo": "org/repo"},
            "gpu": {
                "type": "A10",
                "max_wall_seconds": 3600,
                "network_volume": "flash-weights",
                "network_volume_gb": 100,
            },
        }
    )


def test_instance_build_payload_preload_mode():
    from flash.providers import _instance

    spec = _preload_spec()
    p = _instance.build_payload(
        spec,
        spec.seed,
        0,
        arm="lambda",
        deadline_at=10_000_000_000.0,
        cache_host_mount="/lambda/nfs/flash-weights",
        mode="preload",
        models=["a/b", "c/d"],
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

    spec = _preload_spec()
    p = _instance.build_payload(
        spec,
        spec.seed,
        0,
        arm="lambda",
        deadline_at=10_000_000_000.0,
        cache_host_mount="/lambda/nfs/flash-weights",
    )
    assert "mode" not in p  # ordinary train payload
    assert "models" not in p


def test_instance_build_payload_preserves_worker_env_hf_home(monkeypatch):
    """A per-run [worker_env].HF_HOME override is NOT clobbered by the instance cache path (parity
    with RunPod, where the worker_env override wins), and disables the platform cache redirect."""
    from flash.providers import _instance

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": "flash-1700000000-abcd1234",
            "train": {"max_examples": 8, "hf_repo": "org/repo"},
            "gpu": {"type": "A10", "max_wall_seconds": 3600, "network_volume": "flash-weights"},
            "worker_env": {"HF_HOME": "/custom/hf"},  # user-set override
        }
    )
    p = _instance.build_payload(
        spec,
        spec.seed,
        0,
        arm="lambda",
        deadline_at=10_000_000_000.0,
        cache_host_mount="/lambda/nfs/flash-weights",
    )
    # the user's HF_HOME survives, and the platform cache redirect is NOT installed on top of it.
    assert p["env"]["HF_HOME"] == "/custom/hf"
    assert "FLASH_WEIGHT_CACHE_DIR" not in p["env"]


def test_instance_preload_requires_mounted_cache():
    from flash.providers import _instance_bootstrap as b

    # FLASH_WEIGHT_CACHE_DIR rooted at an UNMOUNTED cache -> refuse (would warm ephemeral disk), no download
    r = b.run_preload(
        {"env": {"FLASH_WEIGHT_CACHE_DIR": "/weight-cache/hf-cache/hub"}, "models": ["x/y"]}
    )
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
    r = b.run_preload(
        {"env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env, "HF_TOKEN": "t"}, "models": ["a/b"]}
    )
    assert r["preloaded"] == ["a/b"]
    assert not r["failed"]
    assert calls[0]["cache_dir"] == str(tmp_path / "hf-cache" / "hub")  # straight into the mount
    assert calls[0]["token"] == "t"
    assert calls[0]["ignore_patterns"] == [
        "*.pth",
        "*.gguf",
        "original/*",
        "*.onnx",
        "*.msgpack",
        "*.h5",
    ]


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


def test_nfs_mount_check_verifies_mountpoint_and_writes_sentinel():
    """The NFS (Lambda) preamble drops the sentinel ONLY when the host path is a real mountpoint, so an
    auto-created empty Docker-bind dir (failed/unready NFS) is detectable in-container."""
    from flash.providers import _instance

    pre = _instance._cache_nfs_mount_check({"cache_host_mount": "/lambda/nfs/flash-weights"})
    assert "mountpoint -q '/lambda/nfs/flash-weights'" in pre  # gates on a REAL mount
    assert "touch '/lambda/nfs/flash-weights/.flash-cache-mounted'" in pre
    # no-op for cold runs.
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
    cache_dir_env = str(
        tmp_path / "hf-cache" / "hub"
    )  # grandparent (the mount) exists but has no sentinel
    r = b.run_preload(
        {
            "env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env},
            "models": ["a/b"],
            "cache_mount_marker": ".flash-cache-mounted",
        }
    )
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
    r = b.run_preload(
        {
            "env": {"FLASH_WEIGHT_CACHE_DIR": cache_dir_env},
            "models": ["a/b"],
            "cache_mount_marker": ".flash-cache-mounted",
        }
    )
    assert r["preloaded"] == ["a/b"]


def test_build_payload_carries_mount_marker_for_nfs_cache():
    """a cache-attached Lambda preload payload carries cache_mount_marker so the in-container check
    can require the NFS mount sentinel."""
    from flash.providers import _instance

    spec = _preload_spec()
    p = _instance.build_payload(
        spec,
        spec.seed,
        0,
        arm="lambda",
        deadline_at=10_000_000_000.0,
        cache_host_mount="/lambda/nfs/flash-weights",
        mode="preload",
        models=["a/b"],
    )
    assert p["cache_mount_marker"] == _instance.CACHE_MOUNT_MARKER


def test_preload_wall_cap_timer_armed_and_cancellable(monkeypatch):
    """run_preload has no worker subprocess, so the preload branch arms an absolute-deadline watchdog
    that hard-exits the box if a download hangs past deadline_at. The timer is cancellable on finish."""
    from flash.providers import _instance_bootstrap as b

    now = 1_000.0
    monkeypatch.setattr(b.time, "time", lambda: now)
    timer, done = b._arm_preload_wall_cap(
        {
            "deadline_at": now + 999,
            "run_created_at": now,
            "run_max_wall_seconds": 999,
        }
    )
    assert timer.is_alive()
    assert not done.is_set()
    # A clean finish sets `done` so a wall expiry racing it no-ops in _fire, then cancels the timer.
    done.set()
    timer.cancel()


def test_lambda_launch_threads_preload_mode_into_payload(monkeypatch):
    """launch_and_submit(mode='preload', models=...) embeds a preload payload in the cache user_data."""
    import base64
    import json as _json

    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    launched = {}
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda n, r, deadline_at=None: f"/lambda/nfs/{n}",
    )
    monkeypatch.setattr(lambda_api, "resolve_ssh_key_names", lambda: ["k"], raising=False)
    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["k"])
    monkeypatch.setattr(
        lambda_api,
        "launch_instance",
        lambda **kw: launched.update(kw) or "i-123",
    )
    from tests.test_lambda_runner import _inst  # reuse the runner's instance candidate

    spec = _preload_spec()
    jobs.launch_and_submit(
        spec,
        seed=spec.seed,
        instances=[_inst()],
        attempt=0,
        mode="preload",
        models=["Qwen/Qwen3.5-0.8B"],
        deadline_at=10_000_000_000.0,
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
        preload,
        "make_hf_text_reader",
        lambda repo, path, min_interval_s=45.0: (
            lambda force=False: _json.dumps(marker) if marker else None
        ),
    )

    def fake_launch(spec, seed, instances, attempt=0, mode=None, models=None, **k):
        # the preload launch must thread the spec's authoritative seed: the real
        # build_payload/build_worker_env path calls require_matching_seed, so a stale seed=0
        # against the default spec.seed would crash every real preload launch.
        assert seed == spec.seed, (seed, spec.seed)
        launched.append((instances[0].region, mode, tuple(models or [])))

    monkeypatch.setattr(lj, "launch_and_submit", fake_launch)
    monkeypatch.setattr(lj, "terminate_run_instances", lambda rid: terminated.append(rid))
    return preload, lj, launched, terminated


def test_warm_instances_one_launch_per_region_and_terminates(monkeypatch):
    preload, lj, launched, terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(
        lj,
        "usable_instances",
        lambda gpu: [_cand("us-east-1"), _cand("us-east-1"), _cand("us-west-2")],
    )

    res = preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert sorted(r["region"] for r in res) == ["us-east-1", "us-west-2"]  # us-east-1 deduped
    assert all(r["status"] == "ok" for r in res)
    assert all(m == "preload" for _, m, _ in launched)  # download-only launches
    assert len(terminated) == 2  # every launch ALWAYS torn down


def test_warm_instance_partial_when_a_model_failed(monkeypatch):
    preload, lj, _launched, terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {"c/d": "gated"}}
    )
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
        time=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + 1e6)
    )
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
        preload,
        "_preload_instance_spec",
        lambda gpu, run_id, wall_s=1800: wall_caps.append(wall_s) or orig_spec(gpu, run_id, wall_s),
    )
    # Fake clock that ticks 1 (virtual) second per sleep, so we can read how long the poll ran.
    clock = {"t": 1000.0}
    start = clock["t"]
    monkeypatch.setattr(
        preload,
        "time",
        _types.SimpleNamespace(
            time=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + 1)
        ),
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
    monkeypatch.setattr(preload, "make_hf_text_reader", lambda *a, **k: lambda force=False: None)
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [_cand("us-east-1")])
    monkeypatch.setattr(
        lj, "launch_and_submit", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no capacity"))
    )
    monkeypatch.setattr(lj, "terminate_run_instances", lambda rid: terminated.append(rid))
    res = preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)
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
        preload,
        "time",
        _types.SimpleNamespace(
            time=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + 1)
        ),
    )
    res = preload.warm_instances(models=["a/b"], timeout_s=600, poll_interval_s=0.0)
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


def test_warm_instances_uses_managed_lambda_default_gpu(monkeypatch):
    """With no --gpu override, Lambda warming uses the managed A10 default."""
    import importlib

    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    monkeypatch.setenv("FLASH_PRELOAD_INSTANCE_GPU", "H100")
    importlib.reload(preload)
    seen = {}

    def capture_gpu(gpu):
        seen["lambda"] = gpu
        return []

    monkeypatch.setattr(lj, "usable_instances", capture_gpu)
    preload.warm_instances(models=["a/b"])

    assert seen == {"lambda": "A10"}
    assert preload._LAMBDA_PRELOAD_GPU == "A10"


def test_warm_instances_explicit_gpu_overrides_default(monkeypatch):
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


def test_preload_status_repo_is_managed_across_all_paths(monkeypatch):
    import importlib
    import json

    import huggingface_hub

    from flash.providers.runpod import preload

    monkeypatch.setenv("FLASH_PRELOAD_STATUS_REPO", "other/repo")
    importlib.reload(preload)
    managed_repo = "Freesolo-Co/flash-weight-preload"
    created = []
    readers = []
    submitted = []
    terminated = []

    class FakeHfApi:
        def __init__(self, token):
            self.token = token

        def create_repo(self, repo, **kwargs):
            created.append((repo, self.token, kwargs))

    def reader_factory(repo, path, min_interval_s=45.0):
        readers.append((repo, path))
        if path.endswith("preload_result.json"):
            return lambda force=False: json.dumps({"preloaded": ["a/b"], "failed": {}})
        return lambda force=False: None

    def capture_launch(spec, *args, **kwargs):
        submitted.append(spec)

    jobs_mod = types.SimpleNamespace(
        launch_and_submit=capture_launch,
        terminate_run_instances=lambda run_id: terminated.append(run_id),
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(preload, "make_hf_text_reader", reader_factory)

    preload._ensure_status_repo("token")
    spec = preload._preload_instance_spec("A10", "preload-test")
    result = preload._warm_one_lambda_instance(
        jobs_mod, _cand("us-east-1"), ["a/b"], "A10", 5, 0.0
    )

    assert managed_repo == preload._PRELOAD_STATUS_REPO
    assert created == [
        (managed_repo, "token", {"repo_type": "dataset", "exist_ok": True, "private": True})
    ]
    assert spec.train.hf_repo == managed_repo
    assert len(submitted) == 1
    assert managed_repo == submitted[0].train.hf_repo
    assert [repo for repo, _path in readers] == [managed_repo, managed_repo]
    assert readers[0][1].endswith("/preload_result.json")
    assert readers[1][1].endswith("/lambda_attempt0.json")
    assert result["status"] == "ok"
    assert len(terminated) == 1


def test_warm_instances_requires_status_repo_before_launch(monkeypatch):
    """With targets available, warm must validate the status repo BEFORE launching any paid box.
    Capacity enumeration is cheap/read-only, so it runs first (to decide if there's anything to do);
    the status-repo guard gates the actual LAUNCH, which is what costs money."""
    import types

    import pytest

    from flash.providers.lambdalabs import jobs as lj
    from flash.providers.runpod import preload

    # lambda has capacity -> a real launch target exists.
    monkeypatch.setattr(
        lj, "usable_instances", lambda gpu: [types.SimpleNamespace(region="us-east-1")]
    )
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
        lambda token: (_ for _ in ()).throw(
            AssertionError("status repo must not be required with no targets")
        ),
    )
    assert preload.warm_instances(models=["a/b"]) == []


def test_warm_instances_cli_dry_run(monkeypatch):
    from flash.providers.runpod import preload

    called = {"n": 0}
    monkeypatch.setattr(
        preload, "warm_instances", lambda **k: called.__setitem__("n", called["n"] + 1) or []
    )
    assert preload.main(["--warm-instances", "--dry-run"]) == 0
    assert called["n"] == 0  # dry-run launches nothing


def test_cli_gpu_default_is_none_per_mode(monkeypatch):
    """--gpu defaults to None; each mode applies its OWN default downstream (no sentinel hack).

    Regression: --gpu used to default to _PRELOAD_GPU ('RTX 4090') and --warm-instances used a
    `args.gpu != _PRELOAD_GPU` comparison — so a user explicitly asking for RTX 4090 on instance
    warming was wrongly treated as 'no override'. None must pass through cleanly.
    """
    from flash.providers.runpod import preload

    # --warm-instances, no --gpu -> warm_instances receives gpu=None and applies the managed lambda a10
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


def _poll_env(monkeypatch, statuses, workers):
    """Wire _poll_until_done's two API calls: job status sequence + a fixed worker-health dict."""
    from flash.providers.runpod import preload

    seq = list(statuses)
    monkeypatch.setattr(
        preload.runpod_api,
        "job_status",
        lambda *a, **k: {"status": seq.pop(0) if seq else "IN_QUEUE"},
    )
    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": workers},
    )
    monkeypatch.setattr(preload.time, "sleep", lambda *_: None)


def test_poll_gives_up_when_datacenter_never_allocates_a_worker(monkeypatch):
    """A DC with no GPU of the requested class must fail fast, not burn the whole timeout.

    Regression: US-KS-2/US-MO-2/US-NC-2/US-NE-1/US-WA-1 stock no RTX 4090, so preload jobs there sat
    queued with an all-zero worker set until the 5400s timeout and the DC stayed silently cold.
    """
    from flash.providers.runpod import preload

    clock = {"t": 0.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["t"])
    _poll_env(
        monkeypatch,
        ["IN_QUEUE"] * 50,
        {"initializing": 0, "ready": 0, "running": 0, "idle": 0},
    )

    def _advance(*_a, **_k):
        clock["t"] += 60.0
        return {"workers": {"initializing": 0, "ready": 0, "running": 0, "idle": 0}}

    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", _advance)
    with pytest.raises(preload.NoCapacityError):
        preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=1.0)
    # gave up inside the grace window, nowhere near the 5400s timeout
    assert clock["t"] < preload._NO_CAPACITY_GRACE_S + 120.0


def test_poll_waits_when_a_worker_is_initializing(monkeypatch):
    """An initializing worker proves capacity exists: a slow download must NOT be cut short."""
    from flash.providers.runpod import preload

    clock = {"t": 0.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["t"])
    # advance the clock in sleep(), so time moves even after the health probe stops being called
    monkeypatch.setattr(preload.time, "sleep", lambda *_: clock.__setitem__("t", clock["t"] + 60.0))
    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": {"initializing": 1, "ready": 0, "running": 0, "idle": 0}},
    )
    monkeypatch.setattr(preload, "decode_output", lambda o: {"preloaded": ["m"]})
    # stays queued well past the no-capacity grace window, then completes
    monkeypatch.setattr(
        preload.runpod_api,
        "job_status",
        lambda *a, **k: (
            {"status": "COMPLETED", "output": "x"}
            if clock["t"] > preload._NO_CAPACITY_GRACE_S * 2
            else {"status": "IN_QUEUE"}
        ),
    )
    out = preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=1.0)
    assert out == {"preloaded": ["m"]}
    assert clock["t"] > preload._NO_CAPACITY_GRACE_S  # proves it did not bail out early


def test_poll_restarts_the_grace_window_after_a_requeue(monkeypatch):
    """Leaving the queue must reset the starvation anchor, not let it age through the running spell.

    Regression: the grace anchor survived an IN_PROGRESS interval, so a job that queued briefly, ran
    for longer than the grace window, then got re-queued after an interruption would be declared
    starved on its FIRST zero-worker reading -- and _preload_one_dc deletes the endpoint on that,
    killing a DC that has real capacity and had already been allocated a worker.
    """
    from flash.providers.runpod import preload

    clock = {"t": 0.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["t"])
    monkeypatch.setattr(preload.time, "sleep", lambda *_: clock.__setitem__("t", clock["t"] + 60.0))
    # queued once, allocated (so saw_worker stays False only because health reads zero), then
    # re-queued long after the grace window would have elapsed, then completes.
    statuses = ["IN_QUEUE", "IN_PROGRESS", "IN_QUEUE", "COMPLETED"]
    seq = iter(statuses)
    monkeypatch.setattr(
        preload.runpod_api,
        "job_status",
        lambda *a, **k: {"status": next(seq, "COMPLETED"), "output": "x"},
    )
    # health always reports zero workers: the ONLY thing standing between this job and deletion is
    # the anchor being reset when the status left the queue.
    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": {"initializing": 0, "ready": 0, "running": 0, "idle": 0}},
    )
    monkeypatch.setattr(preload, "decode_output", lambda o: {"preloaded": ["m"]})
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 30.0)  # one sleep outruns it

    out = preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=1.0)
    assert out == {"preloaded": ["m"]}
    # the running interval alone exceeded the grace window, proving the anchor did not carry over
    assert clock["t"] > preload._NO_CAPACITY_GRACE_S


def test_has_worker_treats_health_failure_as_unknown(monkeypatch):
    """Health is a hint: an API error must not be read as 'no capacity' and kill a live download.

    None, not False -- the caller escalates a sustained False into NoCapacityError and deletes the
    endpoint, so conflating "cannot tell" with "no workers" would kill healthy downloads.
    """
    from flash.providers.runpod import preload

    def _boom(*_a, **_k):
        raise RuntimeError("health api down")

    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", _boom)
    assert preload._has_worker("ep", "fp", 0.0) is None

    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", lambda *a, **k: None)
    assert preload._has_worker("ep", "fp", 0.0) is None

    monkeypatch.setattr(
        preload.runpod_api, "endpoint_health_for_fingerprint", lambda *a, **k: {"workers": {}}
    )
    assert preload._has_worker("ep", "fp", 0.0) is False  # a real empty answer IS evidence


def test_a_broken_worker_image_is_not_reported_as_a_starved_datacenter(monkeypatch):
    """unhealthy and throttled workers exist, so they refute starvation.

    Regression: _has_worker counted only initializing/ready/running/idle. A worker that RunPod
    allocated and then marked unhealthy (jobs.py reads that as a failed image pull and retries on a
    fresh endpoint) looked identical to an empty datacenter, so the grace timer fired NoCapacityError
    and told the operator to pick a different GPU class -- which cannot fix a broken image.
    """
    from flash.providers.runpod import preload

    for state in ("unhealthy", "throttled"):
        monkeypatch.setattr(
            preload.runpod_api,
            "endpoint_health_for_fingerprint",
            lambda *a, _s=state, **k: {"workers": {"ready": 0, "running": 0, _s: 1}},
        )
        assert preload._has_worker("ep", "fp", 0.0) is True

    # and end to end: the queued job must time out, never be blamed on the datacenter
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 0.0)  # grace already elapsed
    monkeypatch.setattr(preload.runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})
    with pytest.raises(TimeoutError):  # NOT NoCapacityError
        preload._poll_until_done("ep", "job", "fp", timeout_s=1, poll_interval_s=0.0)


def test_unreadable_health_never_becomes_no_capacity(monkeypatch):
    """A persistently failing health API must not masquerade as a starved datacenter.

    Regression: with health unreadable, the grace timer used to fire NoCapacityError and the caller's
    finally deleted the endpoint mid-download. The job here stays IN_QUEUE well past the grace window
    and must simply time out instead.
    """
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 0.0)  # grace already elapsed
    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("health api down")),
    )
    monkeypatch.setattr(preload.runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})

    with pytest.raises(TimeoutError):  # NOT NoCapacityError
        preload._poll_until_done("ep", "job-1", "fp", timeout_s=1, poll_interval_s=0.0)


def test_unreadable_health_does_not_age_the_grace_timer(monkeypatch):
    """Grace must count an unbroken run of CONFIRMED zero-worker readings, not wall time since launch.

    Regression: the timer started at launch, so an unreadable health API aged it silently and the very
    first definite "no workers" after that fired NoCapacityError instantly -- deleting an endpoint whose
    download may have been progressing the whole time.
    """
    from flash.providers.runpod import preload

    clock = {"t": 0.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["t"])
    monkeypatch.setattr(preload.time, "sleep", lambda *_: None)
    monkeypatch.setattr(preload.runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})

    def _health(*_a, **_k):
        clock["t"] += 60.0
        # unreadable for well past the grace window, then a single definite zero-worker answer
        if clock["t"] <= preload._NO_CAPACITY_GRACE_S * 2:
            raise RuntimeError("health api down")
        return {"workers": {"initializing": 0, "ready": 0, "running": 0, "idle": 0}}

    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", _health)
    # timeout leaves room for the unreadable stretch but NOT for a fresh full grace window after it
    timeout_s = preload._NO_CAPACITY_GRACE_S * 2 + 120.0
    with pytest.raises(TimeoutError):  # NOT NoCapacityError
        preload._poll_until_done("ep", "job", "fp", timeout_s=timeout_s, poll_interval_s=1.0)


def test_no_capacity_guard_only_runs_while_the_job_is_queued(monkeypatch):
    """IN_PROGRESS proves a worker was allocated, so zero-worker health there is a reporting artifact.

    Regression: the guard ran on every nonterminal status. A job already downloading on a worker could
    be declared starved on a stale health reading, and _preload_one_dc would delete the endpoint
    mid-download.
    """
    from flash.providers.runpod import preload

    clock = {"t": 0.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["t"])
    monkeypatch.setattr(preload.time, "sleep", lambda *_: clock.__setitem__("t", clock["t"] + 60.0))
    monkeypatch.setattr(preload, "decode_output", lambda o: {"preloaded": ["m"]})
    # health insists there are no workers for the whole run -- the status is what must be believed
    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": {"initializing": 0, "ready": 0, "running": 0, "idle": 0}},
    )
    monkeypatch.setattr(
        preload.runpod_api,
        "job_status",
        lambda *a, **k: (
            {"status": "COMPLETED", "output": "x"}
            if clock["t"] > preload._NO_CAPACITY_GRACE_S * 2
            else {"status": "IN_PROGRESS"}
        ),
    )
    out = preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=1.0)
    assert out == {"preloaded": ["m"]}
    assert clock["t"] > preload._NO_CAPACITY_GRACE_S  # ran past grace without being killed


def test_no_capacity_is_reported_distinctly_from_error(monkeypatch):
    """A starved DC surfaces as status='no_capacity' so the summary can name it and the GPU class."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(
        preload,
        "deploy_train_endpoint",
        lambda *a, **k: ("ep-1", "name", "fp-1"),
    )
    monkeypatch.setattr(preload.runpod_api, "submit_job", lambda *a, **k: "job-1")
    monkeypatch.setattr(preload.runpod_api, "delete_endpoint_for_fingerprint", lambda *a, **k: True)

    def _starved(*_a, **_k):
        raise preload.NoCapacityError("never allocated")

    monkeypatch.setattr(preload, "_poll_until_done", _starved)
    res = preload._preload_one_dc("US-KS-2", ["m"], None, "RTX 4090", 60, 1.0)
    assert res["status"] == "no_capacity"
    assert res["gpu"] == "RTX 4090"
    assert res["datacenter"] == "US-KS-2"


def test_catalog_is_ordered_largest_first(monkeypatch):
    """Largest-first makes a capacity failure surface first instead of 20 minutes in.

    It is deliberately not the thing that makes the catalog fit -- see
    ``test_volume_holds_whole_catalog_with_largest_model_in_transit`` for that invariant.
    """
    from flash.providers.runpod import preload

    ids = preload.catalog_model_ids()
    from flash.catalog import MODELS

    sizes = [MODELS[m].params_b or 0.0 for m in ids]
    assert sizes == sorted(sizes, reverse=True), (
        f"catalog must be largest-first, got {list(zip(ids, sizes, strict=True))}"
    )


def test_volume_holds_whole_catalog_with_largest_model_in_transit():
    """The volume must fit every model resident PLUS the largest one's download scratch.

    Regression for a real 200 GB failure: the 35B died with "Disk quota exceeded" in every
    datacenter. Download order does not save it -- the volume is persistent and nothing is
    evicted, so the largest model always meets a volume already holding the rest.
    """
    from flash.runner import WEIGHT_CACHE_VOLUME_GB, weight_cache_catalog_peak_gb

    needed = weight_cache_catalog_peak_gb()
    assert needed <= WEIGHT_CACHE_VOLUME_GB, (
        f"catalog needs {needed:.1f} GB at peak but the volume is {WEIGHT_CACHE_VOLUME_GB} GB: "
        "the largest model will fail with Disk quota exceeded on every datacenter"
    )


def test_preload_timeout_covers_a_fully_cold_whole_catalog_warm():
    """The default budget must outlast downloading the ENTIRE catalog, not just one model.

    Raising the volume to 250 GB is what puts the 27B and 35B into the default preload set, so the
    worst case this default has to survive changed with it: a cold volume now pulls ~159 GB in one
    job. Sized off a measured rate rather than a guess -- a real cold 35B pull moved 70 GB in ~870s.

    A too-short budget is not a slow failure, it throws away everything downloaded so far, so this
    asserts the default clears the measured worst case rather than merely approaching it.
    """
    from flash.catalog import MODELS
    from flash.providers.runpod.preload import _PRELOAD_TIMEOUT_S
    from flash.runner import _download_gb, _fits_weight_cache

    measured_gb, measured_s = 70.0, 870.0  # observed cold 35B pull
    rate_gb_s = measured_gb / measured_s
    catalog_gb = sum(_download_gb(i) for i in MODELS.values() if _fits_weight_cache(i))
    needed_s = catalog_gb / rate_gb_s

    assert needed_s < _PRELOAD_TIMEOUT_S, (
        f"a cold warm downloads {catalog_gb:.1f} GB, which needs ~{needed_s:.0f}s at the measured "
        f"{rate_gb_s:.3f} GB/s, but the default budget is {_PRELOAD_TIMEOUT_S}s: the job is killed "
        "mid-catalog and every byte already pulled is discarded"
    )


def test_every_preload_timeout_default_reads_the_shared_constant():
    """No entry point may carry its own literal budget.

    The bug this guards is partial: raising the constant while one call site keeps a stale literal
    leaves that path timing out exactly as before, and it is the CLI default that operators hit.
    """
    import inspect
    import re

    from flash.providers.runpod import preload

    defaults = {
        name: inspect.signature(fn).parameters["timeout_s"].default
        for name, fn in (("warm_weight_cache", preload.warm_weight_cache),
                         ("warm_instances", preload.warm_instances))
    }
    stale = {k: v for k, v in defaults.items() if v != preload._PRELOAD_TIMEOUT_S}
    assert not stale, f"these carry a literal instead of _PRELOAD_TIMEOUT_S: {stale}"

    # The CLI parser is built inline in main(), so there is no parser object to query without
    # running it. Read the source instead -- a numeric literal here is the operator-facing default
    # and would keep timing out at the old budget however high the constant is raised.
    cli = re.search(r'"--timeout-s".*?default=([^,)\s]+)', inspect.getsource(preload.main), re.S)
    assert cli, "--timeout-s no longer declares a default; this test must be updated"
    assert cli.group(1) == "_PRELOAD_TIMEOUT_S", (
        f"--timeout-s defaults to {cli.group(1)} instead of _PRELOAD_TIMEOUT_S"
    )


def test_catalog_peak_counts_others_resident_not_just_the_largest():
    """Pin the shape of the calculation, not just today's numbers.

    The bug was sizing the largest model against an EMPTY volume. Peak must therefore exceed both
    the largest model's own peak and the plain resident total.
    """
    from flash.catalog import MODELS
    from flash.runner import _fits_weight_cache, weight_cache_catalog_peak_gb

    sizes = sorted(
        ((info.params_b or 0.0) * 2.0 for info in MODELS.values() if _fits_weight_cache(info)),
        reverse=True,
    )
    resident_total = sum(sizes)
    largest_alone_peak = 2.0 * sizes[0]
    needed = weight_cache_catalog_peak_gb()

    assert needed > largest_alone_peak, "peak ignores the models already on the volume"
    assert needed > resident_total, "peak ignores the largest model's in-transit scratch"


def test_partial_datacenter_names_the_models_that_failed(monkeypatch, caplog):
    """'partial' alone is unactionable: the summary must name each failed model and its reason."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "catalog_model_ids", lambda: ["m1", "m2"])
    monkeypatch.setattr(preload, "weight_cache_datacenters", lambda: [])
    monkeypatch.setattr(
        preload,
        "_preload_one_dc",
        lambda dc, *a, **k: {
            "datacenter": dc,
            "status": "partial",
            "result": {"preloaded": ["m1"], "failed": {"m2": "429 rate limited"}},
        },
    )
    with caplog.at_level(logging.WARNING):
        results = preload.warm_weight_cache(datacenters=["US-CA-2"], gpu="RTX 4090")

    assert results[0]["status"] == "partial"
    text = caplog.text
    assert "m2" in text
    assert "429 rate limited" in text
    assert "US-CA-2" in text
    # the model that succeeded is not noise in the failure summary
    assert "m1 FAILED" not in text


def test_ok_datacenter_logs_no_per_model_failures(monkeypatch, caplog):
    """A fully warmed DC must not emit failure lines (the summary stays quiet when nothing is wrong)."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "catalog_model_ids", lambda: ["m1"])
    monkeypatch.setattr(preload, "weight_cache_datacenters", lambda: [])
    monkeypatch.setattr(
        preload,
        "_preload_one_dc",
        lambda dc, *a, **k: {
            "datacenter": dc,
            "status": "ok",
            "result": {"preloaded": ["m1"], "failed": {}},
        },
    )
    with caplog.at_level(logging.WARNING):
        preload.warm_weight_cache(datacenters=["US-CA-2"], gpu="RTX 4090")

    assert "FAILED" not in caplog.text
