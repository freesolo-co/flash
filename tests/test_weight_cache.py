"""Shared, fully-managed, best-effort, multi-region model-weight cache on RunPod network volumes.

Each datacenter gets a managed `/runpod-volume`; failover may drop it to avoid a queue wedge. Tests
are offline and pin the fixed cache contract plus the SDK datacenter-superset rule.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
import types

import pytest

import flash.providers._lifecycle.net.worker as provider_worker
import flash.providers.runpod.execution.job_execution as job_execution
import flash.providers.runpod.execution.resources as runpod_resources
import flash.providers.runpod.serverless.endpoints as runpod_endpoints
import flash.runner.accounting.weight_cache as runner_weight_cache
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
from flash.core.spec import GpuSpec, JobSpec, TrainSpec
from tests._helpers.profile import satisfy_sft_profile
from tests._helpers.source_snapshot import valid_source_snapshot

_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64
_SOURCE_SNAPSHOT = valid_source_snapshot()


def _oversized_model_info():
    """A synthetic catalog entry too large for the shared cache.

    Sized off WEIGHT_CACHE_VOLUME_GB so it stays oversized if the volume grows again — the real
    catalog now fits entirely, so the size gate needs a stand-in to stay covered.
    """
    from flash.core.catalog import ModelInfo

    return ModelInfo(
        id="test/oversized",
        display_name="oversized",
        params="huge",
        algos=("sft",),
        min_vram_gb=80,
        params_b=float(
            runner_weight_cache.WEIGHT_CACHE_VOLUME_GB
        ),  # peak = 4x params_b GB >> the volume
    )


def _vol_spec(name="flash-weights", gb=100, **gpu):
    return JobSpec(
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(network_volume=name, network_volume_gb=gb, **gpu),
        seed=0,
    )


def _ndc() -> int:
    """Size of the ALLOWED datacenter set (DataCenter.all()) — what the endpoint's `datacenter` list
    spans. Derived from the same source the code uses, so it tracks the SDK's storage-DC set.
    """
    from flash.providers.runpod.execution.resources import weight_cache_datacenters

    n = len(weight_cache_datacenters())
    assert n > 1
    return n


def test_preload_cli_entrypoint_runs_as_a_subprocess():
    """`python -m ...weight_cache --dry-run` must reach the planner, not die on a NameError.

    MUST be a subprocess: the names `main()` needs were re-exported at the BOTTOM of the module,
    below the `__main__` guard, so `python -m` ran `main()` before the import and every CLI
    invocation died on `NameError: catalog_model_ids`. An in-process import cannot reproduce that --
    importing executes the whole file, bottom import included, so the library path stayed green
    while the only operator-facing entrypoint was dead. --dry-run provisions nothing.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "flash.providers.artifacts.weight_cache", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = proc.stdout + proc.stderr
    assert "NameError" not in combined, f"CLI died on a NameError:\n{combined}"
    assert proc.returncode == 0, f"--dry-run exited {proc.returncode}:\n{combined}"
    # prove it reached the RUNPOD planner specifically. a bare "would warm" also matches the Lambda
    # plan line, which prints from a different branch and would not exercise `catalog_model_ids`.
    assert "datacenter(s):" in proc.stdout, f"runpod planner never ran:\n{combined}"
    # the model list is what `catalog_model_ids` produces -- the name that used to NameError.
    assert "model(s):" in proc.stdout, f"model catalog never resolved:\n{combined}"


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
        JobSpec.from_dict({"model": "m", "gpu": {"datacenter": "EU-RO-1", "network_volume": "v"}})


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

    dcs = runpod_resources.weight_cache_datacenters()
    vals = {d.value for d in dcs}
    assert len(set(dcs)) == len(dcs)  # all distinct
    assert (
        vals == {d.value for d in DataCenter.all()} - runpod_resources._VOLUME_INCAPABLE_DATACENTERS
    )
    assert "US-MO-1" not in vals  # the known volume-incapable DC is dropped
    assert not (vals & runpod_resources._VOLUME_INCAPABLE_DATACENTERS)


def test_weight_cache_datacenters_ignores_removed_env_knob(monkeypatch):
    # Regression: the FLASH_WEIGHT_CACHE_DATACENTERS knob is GONE — the fleet is fixed/managed.

    baseline = len(runpod_resources.weight_cache_datacenters())
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DATACENTERS", "US-CA-2")
    assert len(runpod_resources.weight_cache_datacenters()) == baseline  # env ignored


def test_weight_cache_volumes_distinct_name_per_dc():

    vols = runpod_resources.weight_cache_volumes(_vol_spec(gb=100))
    # EAGER: one volume in EVERY storage DC (no lazy used-set gating).
    assert len(vols) == _ndc()
    # DISTINCT physical name per DC (the SDK keys resource tracking on name only, so same-named
    # volumes across DCs collide -> a 2nd-volume "replace" -> unimplemented undeploy -> crash).
    assert len({v.name for v in vols}) == len(vols)
    assert all(v.name.startswith("flash-weights-") for v in vols)
    # the name encodes the DC, lowercased
    assert {v.name for v in vols} == {f"flash-weights-{v.dataCenterId.value.lower()}" for v in vols}
    # exactly the full storage-DC set
    assert {v.dataCenterId for v in vols} == set(runpod_resources.weight_cache_datacenters())
    # the SHARED cache is platform-managed, so a stale spec size never wins: a spec still carrying
    # the pre-bump 100 must still build 250-GB volumes, or the bump is a no-op for anything that
    # round-tripped a spec.
    assert {v.size for v in vols} == {runner_weight_cache.WEIGHT_CACHE_VOLUME_GB}


def test_weight_cache_volumes_size_tolerant_of_bad_values():
    # weight_cache_volumes builds NetworkVolumes from spec.gpu.network_volume_gb directly (a GpuSpec
    # can carry a raw value that bypassed JobSpec.from_dict's parse). A non-numeric/"0"/negative size
    # must never raise (best-effort would silently drop the cache) or create a 0-GB volume; on the
    # shared cache it lands on the managed size.

    for raw in ("0", 0, -5, "abc", None, True):
        vols = runpod_resources.weight_cache_volumes(_vol_spec(gb=raw))
        assert {v.size for v in vols} == {runner_weight_cache.WEIGHT_CACHE_VOLUME_GB}, (
            f"{raw!r} rejected"
        )
    # an oversized request on the shared cache still passes through: the floor only raises.
    big = runner_weight_cache.WEIGHT_CACHE_VOLUME_GB + 50
    assert {v.size for v in runpod_resources.weight_cache_volumes(_vol_spec(gb=big))} == {big}


def test_custom_volume_keeps_its_spec_size_while_shared_cache_is_managed():
    """The managed floor applies to the SHARED cache only. A caller's own volume is theirs to size.

    Flooring every volume at the cache size would silently over-provision (and over-bill) a custom
    volume that was deliberately asked for small.
    """

    vols = runpod_resources.weight_cache_volumes(_vol_spec(name="my-own-volume", gb=20))
    assert {v.size for v in vols} == {20}


def test_weight_cache_volume_name_includes_datacenter():
    from runpod_flash.core.resources.datacenter import DataCenter

    assert (
        runpod_resources.weight_cache_volume_name("flash-weights", DataCenter.US_CA_2)
        == "flash-weights-us-ca-2"
    )


def test_weight_cache_volumes_empty_without_volume_name():

    assert runpod_resources.weight_cache_volumes(JobSpec(model="m")) == []


def test_weight_cache_endpoint_kwargs_volume_in_every_dc():

    kw = runpod_resources.weight_cache_endpoint_kwargs(_vol_spec())
    assert sorted(kw) == ["datacenter", "volume"]
    # EAGER: a volume in EVERY storage DC, and the endpoint allowed across exactly that same set, so
    # whichever DC it lands in is warm. The two lists span the identical storage-DC set.
    assert len(kw["volume"]) == _ndc()
    assert len(kw["datacenter"]) == _ndc()
    assert {v.dataCenterId for v in kw["volume"]} == set(kw["datacenter"])


def test_weight_cache_endpoint_kwargs_empty_without_volume():

    assert runpod_resources.weight_cache_endpoint_kwargs(JobSpec(model="m")) == {}


def test_weight_cache_endpoint_kwargs_swallows_errors(monkeypatch):

    monkeypatch.setattr(
        runpod_resources,
        "weight_cache_volumes",
        lambda spec: (_ for _ in ()).throw(RuntimeError("sdk boom")),
    )
    # best-effort: ANY failure building the cache -> {} (deploy with no volume), never propagate.
    assert runpod_resources.weight_cache_endpoint_kwargs(_vol_spec()) == {}


def test_weight_cache_satisfies_real_sdk_superset_validation(monkeypatch):
    # The whole no-pin design rests on the SDK accepting N volumes + N datacenters on one endpoint.
    # Drive the REAL (pure, offline) Endpoint validator/builder to lock that contract: every volume
    # DC must be within the endpoint datacenter list (serverless.py superset rule) and the locations
    # string must span all the DCs.
    from runpod_flash import Endpoint
    from runpod_flash.core.resources.gpu import GpuGroup

    kw = runpod_resources.weight_cache_endpoint_kwargs(_vol_spec())
    ep = Endpoint(name="wc-test", gpu=GpuGroup.AMPERE_48, gpu_count=1, **kw)
    cfg = ep._build_resource_config()  # raises if the superset rule is violated
    vol_dcs = {v.dataCenterId for v in cfg.networkVolumes}
    assert vol_dcs <= set(cfg.datacenter)  # superset rule holds (eager: the sets are in fact equal)
    assert len(cfg.locations.split(",")) == _ndc()  # endpoint allowed across all storage DCs


def test_deploy_train_endpoint_attaches_volume_kwargs(monkeypatch):
    # End-to-end through the primary deploy path: the multi-volume/multi-DC kwargs reach Endpoint().
    import runpod_flash
    import runpod_flash.core.resources.resource_manager as rm_mod

    from flash.providers.runpod.client import auth
    from flash.providers.runpod.execution import job_execution

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    auth.reset()
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
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)

    eid, _name, fingerprint = job_execution.deploy_train_endpoint(
        "RTX 4090",
        spec=_vol_spec(),
        disk_gb=None,
    )
    assert eid == "ep-abc"
    assert fingerprint == job_execution.runpod_api.key_fingerprint("test-key")
    assert len(captured["volume"]) == _ndc()  # EAGER: a volume in every storage DC
    assert len(captured["datacenter"]) == _ndc()  # allowed across all storage DCs
    assert len({v.name for v in captured["volume"]}) == _ndc()  # distinct per-DC names
    assert all(v.name.startswith("flash-weights-") for v in captured["volume"])


def test_deploy_train_endpoint_no_volume_when_spec_has_none(monkeypatch):
    import runpod_flash
    import runpod_flash.core.resources.resource_manager as rm_mod

    from flash.providers.runpod.client import auth
    from flash.providers.runpod.execution import job_execution

    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    auth.reset()
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
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)

    job_execution.deploy_train_endpoint("RTX 4090", spec=JobSpec(model="m"), disk_gb=None)
    assert "volume" not in captured
    assert "datacenter" not in captured


# ---------------------------------------------------------------------------
# worker weight_cache_env / build_worker_env redirect
# ---------------------------------------------------------------------------
def test_weight_cache_env_is_base_model_scoped():
    from flash.providers._lifecycle.net.worker import weight_cache_env

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
    from flash.providers._lifecycle.net.worker import weight_cache_env

    assert weight_cache_env("/workspace")["FLASH_WEIGHT_CACHE_DIR"] == "/workspace/hf-cache/hub"


def test_build_worker_env_sets_base_model_cache_with_volume():
    from flash.providers._lifecycle.net.worker import build_worker_env

    env = build_worker_env(_vol_spec())
    assert env["FLASH_WEIGHT_CACHE_DIR"] == "/runpod-volume/hf-cache/hub"
    # The leak fix: no process-global HF_HOME redirect, so env/reward downloads use the ephemeral cache.
    assert "HF_HOME" not in env


def test_build_worker_env_no_cache_without_volume():
    from flash.providers._lifecycle.net.worker import build_worker_env

    env = build_worker_env(JobSpec(model="m", seed=0))
    # Without a volume the base-model cache var must NOT be set (pointing at a missing mount).
    assert "FLASH_WEIGHT_CACHE_DIR" not in env
    assert "HF_HOME" not in env


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

    import flash.engine.worker.io.heartbeat as worker_heartbeat
    import flash.engine.worker.io.prefetch as worker_prefetch
    import flash.engine.worker.perf as worker_perf

    calls = []

    def _fake_snapshot(repo_id, cache_dir=None, ignore_patterns=None, **kw):
        calls.append(
            {"repo_id": repo_id, "cache_dir": cache_dir, "ignore_patterns": ignore_patterns}
        )
        if (
            cache_dir
        ):  # simulate a real download landing on the (mount) cache: create the repo folder
            folder = "models--" + repo_id.replace("/", "--")
            snap = os.path.join(cache_dir, folder, "snapshots", "deadbeef")
            os.makedirs(snap, exist_ok=True)
            # a real download materializes weight files; prefetch validates they exist
            with open(os.path.join(snap, "model.safetensors"), "w") as f:
                f.write("stub")
            return snap
        return "/ephemeral"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(ephemeral_hub))
    monkeypatch.setattr(worker_perf, "gpu_diagnostics", dict)
    monkeypatch.setattr(worker_heartbeat, "heartbeat", lambda *a, **k: None)
    monkeypatch.setattr(
        worker_heartbeat, "liveness_heartbeat", lambda *_args, **_kwargs: contextlib.nullcontext()
    )
    return worker_prefetch, calls


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

    hf.prefetch_model("Qwen/Qwen3.5-9B")

    # downloaded straight onto the shared mount, NOT the ephemeral default
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3.5-9B",
            "cache_dir": shared_hub,
            "ignore_patterns": ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
        }
    ]
    folder = "models--Qwen--Qwen3.5-9B"
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

    hf.prefetch_model("Qwen/Qwen3.5-9B")

    assert calls[0]["cache_dir"] is None  # ephemeral default cache, a correct cold run
    assert not ephemeral_hub.exists()  # nothing linked


def test_prefetch_model_falls_back_to_ephemeral_when_mount_absent(tmp_path, monkeypatch):
    """FLASH_WEIGHT_CACHE_DIR set but the mount isn't present (failed/absent attach) -> ephemeral cache,
    no write under the missing mount. Defense-in-depth re-check on the worker itself."""
    missing_hub = str(tmp_path / "runpod-volume" / "hf-cache" / "hub")  # parent mount NOT created
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", missing_hub)
    ephemeral_hub = tmp_path / "ephemeral" / "hub"
    hf, calls = _patch_prefetch_io(monkeypatch, ephemeral_hub)

    hf.prefetch_model("Qwen/Qwen3.5-9B")

    assert (
        calls[0]["cache_dir"] is None
    )  # mount absent -> ephemeral, never the missing /runpod-volume path


def test_prefetch_model_starts_no_download_at_deadline(tmp_path, monkeypatch):
    monkeypatch.delenv("FLASH_WEIGHT_CACHE_DIR", raising=False)
    hf, calls = _patch_prefetch_io(monkeypatch, tmp_path / "ephemeral" / "hub")
    import flash.engine.worker.runtime.state as worker_state

    monkeypatch.setattr(worker_state, "_remaining_worker_wall_seconds", lambda: 0.0)

    hf.prefetch_model("Qwen/Qwen3.5-9B")

    assert calls == []


def test_shared_weight_cache_dir_resolves_mount_for_both_substrates(tmp_path, monkeypatch):
    """_shared_weight_cache_dir derives the mount as two levels up from the cache dir (works for the
    RunPod /runpod-volume and instance /weight-cache mounts alike) and requires it to exist."""
    import flash.engine.worker.io.prefetch as worker_prefetch

    monkeypatch.delenv("FLASH_WEIGHT_CACHE_DIR", raising=False)
    assert worker_prefetch._shared_weight_cache_dir() is None  # unset

    mount = tmp_path / "weight-cache"
    mount.mkdir()
    hub = str(mount / "hf-cache" / "hub")
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", hub)
    assert worker_prefetch._shared_weight_cache_dir() == hub  # mount present

    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", str(tmp_path / "absent" / "hf-cache" / "hub"))
    assert worker_prefetch._shared_weight_cache_dir() is None  # mount absent -> ephemeral fallback


# ---------------------------------------------------------------------------
# instance-provider integration (Lambda reuses RunPod's build_worker_env)
# ---------------------------------------------------------------------------
def test_strip_runpod_volume_env_removes_only_mount_rooted_vars():
    from flash.providers._lifecycle.net.worker import strip_runpod_volume_env

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
    from flash.providers._lifecycle.instances import instance as _instance
    from flash.providers._lifecycle.net.worker import build_worker_env

    # network_volume is managed -> carried by the internal dict (the leak source that build_worker_env
    # turns into the /runpod-volume redirect).
    spec = JobSpec.from_dict(
        {**_vol_spec().to_internal_dict(), "run_id": "r", "model": "Qwen/Qwen3.5-9B"}
    )
    assert build_worker_env(spec)["FLASH_WEIGHT_CACHE_DIR"].startswith(
        "/runpod-volume"
    )  # leak source
    for arm in ("lambda",):
        env = _instance.build_payload(
            spec,
            attempt=0,
            arm=arm,
            source_snapshot=_SOURCE_SNAPSHOT,
            deadline_at=10_000_000_000.0,
        )["env"]
        assert not env.get("FLASH_WEIGHT_CACHE_DIR", "").startswith("/runpod-volume"), arm


# ---------------------------------------------------------------------------
# runner_weight_cache._assign_weight_cache_volume — fully managed, no knobs. Only curated models are trainable
# and their weights are public, so the shared cross-tenant cache holds nothing private; size
# (_fits_weight_cache) is what gates attachment.
# ---------------------------------------------------------------------------
def test_assign_weight_cache_attaches_to_a_run():

    out = runner_weight_cache._assign_weight_cache_volume(JobSpec(model="m", run_id="r"))
    assert out.gpu.network_volume == runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME == "flash-weights"
    assert out.gpu.network_volume_gb == runner_weight_cache.WEIGHT_CACHE_VOLUME_GB


def test_assign_weight_cache_keeps_a_custom_volume():
    # A NON-shared (per-org / custom) volume is the intended escape hatch — left intact.

    spec = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "run_id": "r",
            "gpu": {
                "type": "A10",
                "network_volume": "org-123-private-cache",
                "network_volume_gb": 100,
            },
        }
    )
    out = runner_weight_cache._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume == "org-123-private-cache"  # not the shared cache -> kept


def test_assign_weight_cache_ignores_removed_kill_switch(monkeypatch):
    # Regression: the FLASH_WEIGHT_CACHE=0 kill switch is GONE — fully managed, always on.

    monkeypatch.setenv("FLASH_WEIGHT_CACHE", "0")
    out = runner_weight_cache._assign_weight_cache_volume(JobSpec(model="m", run_id="r"))
    assert out.gpu.network_volume == "flash-weights"  # env ignored


def test_assign_weight_cache_does_not_override_existing():

    spec = _vol_spec(name="explicit-vol")
    # network_volume is managed -> carried by the internal dict; an already-pinned volume is honored.
    spec = JobSpec.from_dict({**spec.to_internal_dict(), "run_id": "r"})
    out = runner_weight_cache._assign_weight_cache_volume(spec)
    assert out.gpu.network_volume == "explicit-vol"  # an explicit/test value is never clobbered


def test_assign_weight_cache_skips_oversized_catalog_model():
    # SIZE GATE: a model whose peak download footprint exceeds the shared cache must NOT be
    # attached — that would redirect the base-model download onto an undersized mount and overflow
    # mid-download. It is left cache-less and downloads to the container disk instead. Every model
    # in the current catalog fits (see test_every_catalog_model_fits_the_weight_cache), so the gate
    # is exercised with a synthetic oversized entry rather than a real one.

    info = _oversized_model_info()
    out = runner_weight_cache._assign_weight_cache_volume(JobSpec(model=info.id, run_id="r"), info)
    assert out.gpu.network_volume is None  # too big for the shared cache -> cache-less


def test_assign_weight_cache_strips_preset_shared_cache_on_oversized_catalog_model():
    # SIZE GATE re-applies to a pre-set SHARED-cache name: a programmatic/stale catalog spec that
    # already pinned ``flash-weights`` for an oversized model must NOT bypass the gate via the
    # "honor an existing volume" no-op and redirect the download onto the undersized mount. It is
    # stripped cache-less.

    info = _oversized_model_info()
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {
                "type": "B200",
                "network_volume": runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner_weight_cache._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume is None  # oversized -> the pre-set shared cache was stripped


def test_assign_weight_cache_keeps_preset_shared_cache_on_fitting_catalog_model():
    # The re-gate only strips when OVERSIZED: a fitting model that already carries the shared cache
    # keeps it (the pin is correct), exercising the honor-existing path for a shared-name spec.
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {
                "type": "H100",
                "network_volume": runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner_weight_cache._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "flash-weights"  # fits -> kept


def test_assign_weight_cache_keeps_preset_custom_volume_on_oversized_catalog_model():
    # The re-gate is scoped to the SHARED name only: a custom/per-org volume on an oversized model is
    # left intact (the caller owns its sizing — it may be a 200 GB org cache that DOES fit).
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.6-35B-A3B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {
                "type": "B200",
                "network_volume": "org-123-big-cache",
                "network_volume_gb": 400,
            },
        }
    )
    out = runner_weight_cache._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "org-123-big-cache"  # custom volume honored despite size


def test_assign_weight_cache_attaches_fitting_catalog_model():
    # A model whose download fits the cache (with temp headroom) is still attached when info is passed.
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]  # ~19.4 GB download, peak ~39 GB < 100 GB
    out = runner_weight_cache._assign_weight_cache_volume(JobSpec(model=info.id, run_id="r"), info)
    assert out.gpu.network_volume == "flash-weights"


def test_fits_weight_cache_is_size_based():

    # Oversized models are still excluded: the gate is a size check, not "always true".
    assert not runner_weight_cache._fits_weight_cache(_oversized_model_info())


def test_every_catalog_model_fits_the_weight_cache():
    # The largest models have the slowest cold downloads, so they are exactly the ones the cache
    # must cover. WEIGHT_CACHE_VOLUME_GB must stay >= the peak footprint of the biggest catalog
    # entry; if a larger model is added, grow the volume rather than silently skipping its cache.
    from flash.core.catalog import MODELS

    for mid, info in MODELS.items():
        assert runner_weight_cache._fits_weight_cache(info), (
            f"{mid} ({info.params_b}B) no longer fits the "
            f"{runner_weight_cache.WEIGHT_CACHE_VOLUME_GB} GB weight cache"
        )


def test_submit_job_assigns_weight_cache(monkeypatch):
    # Integration: the assignment is wired into submit_job and visible on the effective worker spec.
    # network_volume is platform-managed -> stripped from the public status.spec, so observe the
    # managed assignment on the effective-preparation worker spec the worker actually runs.

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(tmp, "runs"))
        spec = JobSpec.from_dict(
            {
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "sft",
                "environment": {"id": "github:o/r@main:env/environment.py"},
                "train": {"epochs": 1, "max_examples": 8},
                "run_id": "flash-wc-1",
            }
        )
        # sft submission is profile-gated; the cache attachment under test is not, so seed the
        # profile rather than exercising the hub round-trips a real submit performs.
        satisfy_sft_profile(monkeypatch, spec)
        status = runner_submit.submit_job(spec, dry_run=True)
    gpu = status.effective_preparation["worker_spec"]["gpu"]
    assert gpu["network_volume"] == "flash-weights"
    assert gpu["network_volume_gb"] == runner_weight_cache.WEIGHT_CACHE_VOLUME_GB
    # and it must NOT leak into the public spec
    assert "network_volume" not in status.spec["gpu"]


# ---------------------------------------------------------------------------
# lifecycle._drop_weight_cache + the no-capacity fallback
# ---------------------------------------------------------------------------
def test_drop_weight_cache_clears_volume():
    from flash.runner.supervise.retry_decision import _drop_weight_cache

    assert _drop_weight_cache(_vol_spec()).gpu.network_volume is None


def test_drop_weight_cache_noop_without_volume():
    from flash.runner.supervise.retry_decision import _drop_weight_cache

    spec = JobSpec(model="m")
    assert _drop_weight_cache(spec) is spec  # no copy when there's nothing to drop


def test_drop_weight_cache_preserves_non_shared_escape_hatch_volume():
    # Review (Copilot): the no-capacity cache-drop must NOT strip a non-shared per-org/custom volume —
    # that is the deliberate escape-hatch isolation the run opted into. Only the SHARED
    # platform cache (WEIGHT_CACHE_VOLUME_NAME) is dropped.
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME
    from flash.runner.supervise.retry_decision import _drop_weight_cache

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
        fresh_runner(tmp, monkeypatch)
        public = JobSpec.from_dict(
            {**_vol_spec().to_internal_dict(), "run_id": "managed-cache-fallback"}
        )
        assert public.gpu.network_volume == runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME
        selected_dict = public.to_internal_dict()
        selected_dict["gpu"]["network_volume"] = None
        selected = JobSpec.from_dict(selected_dict)
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=public.run_id,
                state="provisioning",
                spec=public.to_dict(),
                effective_preparation={
                    "worker_spec": public.to_internal_dict(),  # committed WITH the shared cache
                    "adapter_identity": None,
                    "version": 1,
                    "preparation_digest": "seed",
                },
            )
        )

        assert runner_submit._persist_effective_worker_spec(selected)

        stored = runner_status.get_status(public.run_id)
        assert stored.effective_preparation["worker_spec"]["gpu"]["network_volume"] is None
        assert stored.effective_preparation["adapter_identity"] is None
        assert runner_status.effective_spec_from_status(stored).gpu.network_volume is None


def test_effective_spec_rejects_custom_volume_removal(monkeypatch):
    # A per-org escape-hatch volume a run opted into must never be silently removed.
    # network_volume is managed and no longer travels in the public spec, so the committed custom
    # volume lives only in the prior preparation snapshot; dropping it there must fail closed.
    import pytest

    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        committed = JobSpec.from_dict(
            {
                **_vol_spec(name="org-1234-private").to_internal_dict(),
                "run_id": "custom-cache-fallback",
            }
        )
        assert committed.gpu.network_volume != runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=committed.run_id,
                state="provisioning",
                spec=committed.to_dict(),
                effective_preparation={
                    "worker_spec": committed.to_internal_dict(),  # committed WITH the custom volume
                    "adapter_identity": None,
                    "version": 1,
                    "preparation_digest": "seed",
                },
            )
        )
        selected_dict = committed.to_internal_dict()
        selected_dict["gpu"]["network_volume"] = None
        selected = JobSpec.from_dict(selected_dict)

        with pytest.raises(ValueError, match="effective preparation"):
            runner_submit._persist_effective_worker_spec(selected)


def _supervised_walk(monkeypatch, failures):
    """Run the supervised seed loop, returning per-attempt (gpu.network_volume, gpu.type) tuples.

    ``failures`` maps attempt index -> failure category (absent attempt -> success).
    """
    from tests._helpers.runner import fresh_runner

    with tempfile.TemporaryDirectory() as tmp:
        fresh_runner(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        seen: list = []

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
            seen.append((spec.gpu.network_volume, spec.gpu.type))
            fail = failures.get(attempt)
            if fail:
                return jobs.PollResult(False, failure=fail, detail="x")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1})

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(
            provider_worker, "publish_source_snapshot", lambda _repo=None: _SOURCE_SNAPSHOT
        )
        monkeypatch.setattr(
            runner_status,
            "validate_terminal_source_metrics",
            lambda _status, metrics, expected_attempt=None: (metrics, expected_attempt),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="wc-walk",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        runner_submit.submit_job(spec, dry_run=False, background=False)
        assert runner_status.get_status("wc-walk").state == "done"
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
    from flash.core.catalog import MODELS
    from flash.providers.artifacts import weight_cache as preload
    from flash.runner.accounting.weight_cache import _fits_weight_cache

    ids = set(preload.catalog_model_ids())
    # The default preload set is the catalog RESTRICTED to models that fit the weight cache (warming a
    # non-fitting model only overflows the fixed mount). Mirrors the submit path's _fits_weight_cache.
    assert ids == {mid for mid, info in MODELS.items() if _fits_weight_cache(info)}
    assert ids <= set(MODELS)
    # The whole catalog fits the volume, so the large checkpoints — the ones with the
    # slowest cold downloads — are warmed too.
    assert "Qwen/Qwen3.8-27B" in ids
    assert "Qwen/Qwen3.6-35B-A3B" in ids


def test_preload_branch_passes_explicit_cache_dir(monkeypatch):
    # BLOCKER regression: _train_body imports huggingface_hub at module load, so HF_HOME set in the
    # preload branch is read too LATE — the download must pass cache_dir=<HF_HOME>/hub explicitly or
    # it lands on the worker's ephemeral default cache and the volume is never warmed.
    import os as _os

    import huggingface_hub

    import flash.providers.runpod.serverless.endpoints as endpoints

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
            "models": ["Qwen/Qwen3.5-9B"],
            "env": {"HF_HOME": "/runpod-volume/hf-cache", "HF_TOKEN": "t"},
        }
    )
    assert out["preloaded"] == ["Qwen/Qwen3.5-9B"]
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

    import flash.providers.runpod.serverless.endpoints as endpoints

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
        out = endpoints._train_body({"mode": "preload", "models": ["Qwen/Qwen3.5-9B"], "env": env})
        assert out["preloaded"] == []
        assert out["already_cached"] == []
        assert "HF_HOME rooted at /runpod-volume" in out["error"]
    assert not calls  # ...nothing is ever downloaded for a non-volume HF_HOME


def test_teardown_weight_cache_deletes_only_fleet_volumes(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload

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

    from flash.providers.runpod.client import auth as rp_keys

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

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.runpod.client import auth as rp_keys

    def _boom(*a, **k):
        raise AssertionError("RunpodRestClient must not be constructed without a key")

    monkeypatch.setattr(rp_keys, "keys", list)  # empty pool == RUNPOD_API_KEY unset
    monkeypatch.setattr(rp_api, "RunpodRestClient", _boom)
    assert preload.teardown_weight_cache(["US-CA-2"]) == []


def test_teardown_weight_cache_sweeps_all_pool_accounts(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload

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

    from flash.providers.runpod.client import auth as rp_keys

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

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.runpod.client import auth as rp_keys

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

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.runpod.client import auth as rp_keys

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
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_.client import api as lambda_api

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
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_.client import api as lambda_api

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
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(preload, "teardown_weight_cache", lambda dcs: ["flash-weights-us-ca-2"])
    monkeypatch.setattr(
        preload, "teardown_lambda_filesystems", lambda: ["lambda:us-east-1/flash-weights"]
    )
    assert preload.main(["--teardown"]) == 0


def test_teardown_dry_run_deletes_nothing(monkeypatch):
    """`--teardown --dry-run` only PRINTS the plan — it must never call the destructive helpers."""
    from flash.providers.artifacts import weight_cache as preload

    def _boom(*a, **k):
        raise AssertionError("--teardown --dry-run must not call any teardown helper")

    monkeypatch.setattr(preload, "teardown_weight_cache", _boom)
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", _boom)
    assert preload.main(["--teardown", "--dry-run"]) == 0


def test_scoped_teardown_rejects_invalid_datacenter(monkeypatch):
    """`--teardown --datacenters <bad-id>` fails non-zero and deletes NOTHING (no silent success)."""
    from flash.providers.artifacts import weight_cache as preload

    def _boom(*a, **k):
        raise AssertionError("invalid scoped teardown must not delete anything")

    monkeypatch.setattr(preload, "teardown_weight_cache", _boom)
    assert preload.main(["--teardown", "--datacenters", "NOT-A-REAL-DC"]) == 2


def test_teardown_continues_when_runpod_unconfigured(monkeypatch):
    """A RunPod teardown raise (auth absent / outage) must NOT abort Lambda cleanup."""
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

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

    A present-but-empty scope must abort without touching the fleet; it cannot fall through to the
    all-datacenters default.
    """
    from flash.providers.artifacts import weight_cache as preload

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

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.runpod.client import auth as rp_keys

    def _boom(*a, **k):
        raise AssertionError("an empty scope must not list/delete any volumes")

    monkeypatch.setattr(rp_keys, "keys", lambda: ["k1"])
    monkeypatch.setattr(rp_api, "RunpodRestClient", _boom)
    assert preload.teardown_weight_cache([]) == []  # nothing reclaimed, no client constructed


# ---------------------------------------------------------------------------
# eager provision: create lambda weight-cache filesystems in every region without a gpu
# ---------------------------------------------------------------------------
def test_provision_lambda_filesystems_covers_every_region(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_.client import api as lambda_api

    ensured = []
    monkeypatch.setattr(
        lambda_api, "all_regions", lambda: ["us-east-1", "us-west-2", "europe-central-1"]
    )
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda name, region, deadline_at=None: (
            ensured.append((name, region)) or f"/lambda/nfs/{name}"
        ),
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
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_.client import api as lambda_api

    def flaky(name, region, deadline_at=None):
        if region == "bad-1":
            raise lambda_api.LambdaApiError("region down")
        return f"/lambda/nfs/{name}"

    monkeypatch.setattr(lambda_api, "all_regions", lambda: ["ok-1", "bad-1", "ok-2"])
    monkeypatch.setattr(lambda_api, "ensure_filesystem", flaky)
    # one bad region never aborts the rest
    assert preload.provision_lambda_filesystems() == ["lambda:ok-1", "lambda:ok-2"]


def test_provision_lambda_no_key_is_noop(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_.client import api as lambda_api

    monkeypatch.setattr(
        lambda_api,
        "all_regions",
        lambda: (_ for _ in ()).throw(lambda_api.LambdaApiError("LAMBDA_API_KEY not set")),
    )
    assert preload.provision_lambda_filesystems() == []


def test_provision_cli_creates_lambda_filesystems(monkeypatch, capsys):
    """`preload --provision` creates Lambda filesystems (GPU-free) and exits 0."""
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(preload, "provision_lambda_filesystems", lambda: ["lambda:us-east-1"])
    assert preload.main(["--provision"]) == 0
    assert capsys.readouterr().out == "provisioned 1 Lambda filesystem(s): lambda:us-east-1\n"


def test_provision_cli_dry_run_provisions_nothing(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload

    called = {"n": 0}
    monkeypatch.setattr(
        preload,
        "provision_lambda_filesystems",
        lambda: called.__setitem__("n", called["n"] + 1) or [],
    )
    assert preload.main(["--provision", "--dry-run"]) == 0
    assert called["n"] == 0  # dry-run touches no provider


def test_preload_one_dc_deploys_pins_single_dc_and_tears_down(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload

    calls = {}

    def fake_deploy(
        gpu,
        execution_timeout_ms=None,
        name_suffix=None,
        spec=None,
        endpoint_kwargs=None,
        deadline_at=None,
        cache_volumes=None,
    ):
        calls["gpu"] = gpu
        calls["suffix"] = name_suffix
        calls["endpoint_kwargs"] = endpoint_kwargs
        calls["cache_volumes"] = cache_volumes
        return "ep-1", "name-1", _RUNPOD_FINGERPRINT

    submitted = {}

    def fake_submit(eid, payload, **_kw):
        submitted["eid"] = eid
        submitted["payload"] = payload
        return "job-1"

    deleted = []
    monkeypatch.setattr(preload, "deploy_train_endpoint", fake_deploy)
    monkeypatch.setattr(preload.runpod_api, "submit_job", fake_submit)
    monkeypatch.setattr(
        preload.runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda eid, _fingerprint: deleted.append(eid),
    )
    monkeypatch.setattr(
        preload.runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {
            "status": "COMPLETED",
            "output": {"preloaded": ["Qwen/Qwen3.5-9B"]},
        },
    )

    out = preload._preload_one_dc(
        "EU-RO-1",
        ["Qwen/Qwen3.5-9B"],
        token="tok",
        gpu="RTX 4090",
        timeout_s=60,
        poll_interval_s=0.0,
    )

    assert out["status"] == "ok"
    assert out["result"]["preloaded"] == ["Qwen/Qwen3.5-9B"]
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
    assert p["models"] == ["Qwen/Qwen3.5-9B"]
    assert p["env"]["HF_HOME"] == "/runpod-volume/hf-cache"
    assert p["env"]["HF_TOKEN"] == "tok"
    assert deleted == ["ep-1"]  # endpoint torn down


def test_preload_one_dc_tears_down_on_failure(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload

    deleted = []
    monkeypatch.setattr(
        preload,
        "deploy_train_endpoint",
        lambda *a, **k: ("ep-9", "name-9", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(preload.runpod_api, "submit_job", lambda eid, payload, **_kw: "job-9")
    monkeypatch.setattr(
        preload.runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda eid, _fingerprint: deleted.append(eid),
    )
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
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

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

    import flash.providers.runpod.serverless.endpoints as endpoints

    monkeypatch.setattr(_os, "environ", dict(_os.environ))
    monkeypatch.setattr(_os.path, "isdir", lambda p: False)  # /runpod-volume not mounted
    out = endpoints._train_body(
        {
            "mode": "preload",
            "models": ["Qwen/Qwen3.5-9B"],
            "env": {"HF_HOME": "/runpod-volume/hf-cache"},
        }
    )
    assert out["preloaded"] == []
    assert "not mounted" in out["error"]


def test_warm_weight_cache_fans_out_over_datacenters(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

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
            "model": "Qwen/Qwen3.5-9B",
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
    from flash.providers._lifecycle.instances import instance as _instance

    spec = _preload_spec()
    p = _instance.build_payload(
        spec,
        0,
        arm="lambda",
        source_snapshot=_SOURCE_SNAPSHOT,
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
    from flash.providers._lifecycle.instances import instance as _instance

    spec = _preload_spec()
    p = _instance.build_payload(
        spec,
        0,
        arm="lambda",
        source_snapshot=_SOURCE_SNAPSHOT,
        deadline_at=10_000_000_000.0,
        cache_host_mount="/lambda/nfs/flash-weights",
    )
    assert "mode" not in p  # ordinary train payload
    assert "models" not in p


def test_instance_preload_requires_mounted_cache():
    from flash.providers._lifecycle.bootstrapping import bootstrap as b

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

    from flash.providers._lifecycle.bootstrapping import bootstrap as b

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

    from flash.providers._lifecycle.bootstrapping import bootstrap as b

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
    from flash.providers._lifecycle.instances import instance as _instance

    pre = _instance._cache_nfs_mount_check({"cache_host_mount": "/lambda/nfs/flash-weights"})
    assert "mountpoint -q '/lambda/nfs/flash-weights'" in pre  # gates on a REAL mount
    assert "touch '/lambda/nfs/flash-weights/.flash-cache-mounted'" in pre
    # no-op for cold runs.
    assert _instance._cache_nfs_mount_check({}) == ""


def test_instance_preload_nfs_requires_mount_sentinel(tmp_path, monkeypatch):
    """A Lambda (NFS) preload whose mount dir exists but has NO sentinel must refuse — Docker's -v bind
    auto-creates a missing host dir, so isdir(mount) alone can't prove the NFS actually mounted."""
    import sys

    from flash.providers._lifecycle.bootstrapping import bootstrap as b

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

    from flash.providers._lifecycle.bootstrapping import bootstrap as b

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
    from flash.providers._lifecycle.instances import instance as _instance

    spec = _preload_spec()
    p = _instance.build_payload(
        spec,
        0,
        arm="lambda",
        source_snapshot=_SOURCE_SNAPSHOT,
        deadline_at=10_000_000_000.0,
        cache_host_mount="/lambda/nfs/flash-weights",
        mode="preload",
        models=["a/b"],
    )
    assert p["cache_mount_marker"] == _instance.CACHE_MOUNT_MARKER


def test_preload_wall_cap_timer_armed_and_cancellable(monkeypatch):
    """run_preload has no worker subprocess, so the preload branch arms an absolute-deadline watchdog
    that hard-exits the box if a download hangs past deadline_at. The timer is cancellable on finish."""
    from flash.providers._lifecycle.bootstrapping import bootstrap as b

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

    from flash.providers.lambda_ import jobs
    from flash.providers.lambda_.client import api as lambda_api

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
        instances=[_inst()],
        attempt=0,
        mode="preload",
        models=["Qwen/Qwen3.5-9B"],
        deadline_at=10_000_000_000.0,
    )
    # decode the base64 payload embedded in the cache user_data
    ud = launched["user_data"]
    b64 = ud.split("FLASH_PAYLOAD_EOF")[1].strip()
    payload = _json.loads(base64.b64decode(b64))
    assert payload["mode"] == "preload"
    assert payload["models"] == ["Qwen/Qwen3.5-9B"]
    assert "source_snapshot" not in payload


def test_lambda_warm_caller_uses_source_independent_preload_payload(monkeypatch):
    import base64
    import json as _json

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs
    from flash.providers.lambda_.client import api as lambda_api
    from tests.test_lambda_runner import _inst

    launched = {}
    terminated = []
    monkeypatch.setattr(preload, "_ensure_region_filesystem", lambda *_args: "listed")
    monkeypatch.setattr(
        preload,
        "make_hf_text_reader",
        lambda _repo, path, **_kwargs: (
            (lambda force=False: _json.dumps({"preloaded": ["a/b"], "failed": {}}))
            if path.endswith("preload_result.json")
            else (lambda force=False: None)
        ),
    )
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda name, region, deadline_at=None: f"/lambda/nfs/{name}",
    )
    monkeypatch.setattr(lambda_api, "resolve_ssh_key_names", lambda: ["key"], raising=False)
    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["key"])
    monkeypatch.setattr(
        lambda_api,
        "launch_instance",
        lambda **kwargs: launched.update(kwargs) or "instance-1",
    )
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda run_id: terminated.append(run_id))

    result = preload._warm_one_lambda_instance(jobs, [_inst()], ["a/b"], 600, 0.0)

    payload_b64 = launched["user_data"].split("FLASH_PAYLOAD_EOF")[1].strip()
    payload = _json.loads(base64.b64decode(payload_b64))
    assert result["status"] == "ok"
    assert payload["mode"] == "preload"
    assert "source_snapshot" not in payload
    assert terminated


# ---------------------------------------------------------------------------
# Instance-provider WARM orchestrator (warm_instances): launch -> poll marker -> terminate
# ---------------------------------------------------------------------------
_LADDER_PRICE = {"A10": 1.29, "A100 SXM 40GB": 1.99, "H100": 3.29, "B200": 6.99}


def _cand(region, gpu="A10"):
    # gpu is read off the candidate now: the class that launches must be the one the region stocks.
    return types.SimpleNamespace(region=region, gpu=gpu, price_usd_hr=_LADDER_PRICE.get(gpu))


def _wire_warm(monkeypatch, marker):
    """Stub the warm path: status repo, the provider's usable_instances/launch/terminate, marker poll."""
    import json as _json

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    launched, terminated = [], []
    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(
        preload,
        "make_hf_text_reader",
        lambda repo, path, min_interval_s=45.0: (
            lambda force=False: _json.dumps(marker) if marker else None
        ),
    )

    def fake_launch(spec, instances, attempt=0, mode=None, models=None, **k):
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
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

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

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

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
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(lj, "usable_instances", lambda gpu: [])
    assert preload.warm_instances(models=["a/b"]) == []


def test_warm_instances_uses_managed_lambda_gpu_ladder(monkeypatch):
    """With no --gpu override, Lambda warming walks the managed ladder cheapest-first."""
    import importlib

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    monkeypatch.setenv("FLASH_PRELOAD_INSTANCE_GPU", "H100")
    importlib.reload(preload)
    asked = []

    def capture_gpu(gpu):
        asked.append(gpu)
        return []

    monkeypatch.setattr(lj, "usable_instances", capture_gpu)
    preload.warm_instances(models=["a/b"])

    # Cheapest first, and the env var is NOT a way to redirect the managed default.
    assert asked == ["A10", "A100 SXM 40GB", "H100", "B200"]
    assert preload._LAMBDA_PRELOAD_GPU_LADDER == ("A10", "A100 SXM 40GB", "H100", "B200")
    assert preload._LAMBDA_PRELOAD_GPU == "A10"


def test_lambda_ladder_is_ordered_cheapest_first():
    """The ladder must stay price-ordered: a warm is a pure download, so paying more never buys speed.

    Guards against someone appending a class without checking where it lands on price.
    """
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_.client.pricing import _STATIC_RATES

    rates = [_STATIC_RATES[c] for c in preload._LAMBDA_PRELOAD_GPU_LADDER]
    assert rates == sorted(rates), f"ladder must be cheapest-first, got {rates}"


def _stocked(**by_class):
    """usable_instances stub: map each GPU class to the regions that stock it right now.

    Carries ``price_usd_hr`` like the real call does, since selection ranks on the live price.
    """
    return lambda gpu: [
        types.SimpleNamespace(region=r, gpu=gpu, price_usd_hr=_LADDER_PRICE.get(gpu))
        for r in by_class.get(gpu.replace(" ", "_"), [])
    ]


def _plan(targets):
    """Flatten warm targets to ``[(region, [class, ...]), ...]`` in cheapest-first order."""
    return [(cands[0].region, [c.gpu for c in cands]) for cands in targets]


def test_lambda_ladder_reaches_regions_the_cheapest_class_cannot(monkeypatch):
    """The whole point: a region with no A10 is still warmed, on the cheapest class that stocks it.

    Regression for the fleet-wide gap where 8 of 10 provisioned filesystems were never warmed because
    A10 -- the only class the warm path ever asked for -- had capacity in just 2 regions.
    """
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    monkeypatch.setattr(
        lj,
        "usable_instances",
        _stocked(
            A10=["us-east-1"],
            # us-east-1 is stocked here too, but A10 already claimed it
            A100_SXM_40GB=["us-east-1", "asia-south-1"],
            H100=["us-west-3"],
            B200=[],
        ),
    )
    targets, planned = preload._lambda_warm_targets(lj, None)

    assert planned is True, "every class answered, so the plan is a complete measurement"
    assert _plan(targets) == [
        ("asia-south-1", ["A100 SXM 40GB"]),  # unreachable on A10 alone
        # contested region: cheapest first, but the pricier class is RETAINED as a launch fallback
        ("us-east-1", ["A10", "A100 SXM 40GB"]),
        ("us-west-3", ["H100"]),  # only H100 stocks it
    ]


def test_lambda_ladder_skips_a_class_whose_capacity_lookup_fails(monkeypatch):
    """One class's API call failing must not strand the regions a later class could still warm."""
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    def flaky(gpu):
        if gpu == "A10":
            raise RuntimeError("instance-types API down")
        return [types.SimpleNamespace(region="asia-south-1", gpu=gpu)] if gpu == "H100" else []

    monkeypatch.setattr(lj, "usable_instances", flaky)
    targets, planned = preload._lambda_warm_targets(lj, None)

    assert _plan(targets) == [("asia-south-1", ["H100"])]
    # the A10 answer never came back, so any region reachable only via A10 is unexamined, not cold
    assert planned is False, "a failed lookup must not be reported as a completed measurement"


def test_lambda_explicit_gpu_pins_the_class_and_skips_the_ladder(monkeypatch):
    """An explicit --gpu is an operator decision: never silently widened to other classes."""
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    asked = []

    def capture(gpu):
        asked.append(gpu)
        return [types.SimpleNamespace(region="us-east-1", gpu=gpu)]

    monkeypatch.setattr(lj, "usable_instances", capture)
    targets, planned = preload._lambda_warm_targets(lj, "B200")

    assert asked == ["B200"]  # ladder not walked
    assert planned is True
    assert _plan(targets) == [("us-east-1", ["B200"])]  # and no fallback class smuggled in


def test_warm_result_reports_the_gpu_that_warmed_each_region(monkeypatch):
    """A mixed-class fleet warm is unreadable unless each result says which class it used."""
    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(lj, "usable_instances", _stocked(H100=["asia-south-1"]))
    res = preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert [(r["region"], r["gpu"], r["status"]) for r in res] == [("asia-south-1", "H100", "ok")]


def test_warm_launches_the_class_the_ladder_claimed_not_the_default(monkeypatch):
    """The launched box must use the ladder's class. Falling back to A10 would relaunch the bug.

    A region reached only because H100 stocks it, launched on A10, fails exactly the way the
    hardcoded class did -- so the class must be threaded through, never re-derived with a default.
    """
    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(lj, "usable_instances", _stocked(H100=["asia-south-1"]))
    specced = []
    real_spec = preload._preload_instance_spec
    monkeypatch.setattr(
        preload,
        "_preload_instance_spec",
        lambda gpu, run_id, wall_s=1800: (specced.append(gpu), real_spec(gpu, run_id, wall_s))[1],
    )
    preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert specced == ["H100"], f"launched on {specced}, not the class that stocks the region"


def test_warm_reports_timeouts_and_partials_as_not_warmed(monkeypatch):
    """A timed-out region left its cache incomplete; reporting only errors would read as success."""
    preload, lj, _launched, _terminated = _wire_warm(monkeypatch, None)  # marker never appears
    monkeypatch.setattr(lj, "usable_instances", _stocked(A10=["us-east-1"]))
    # the poll budget floors at 60s, so expire its deadline instead of waiting. replace the module
    # reference rather than patching `preload.time`, which is the process-wide stdlib module.
    real_time = preload.time
    # Advance on SLEEP, not on a fixed number of time() reads: a counted tick list silently breaks
    # whenever the code under test adds a clock read (it once left the poll loop spinning forever on
    # a deadline it could never reach). One sleep outruns any deadline, so the loop takes exactly one
    # pass and then times out, whatever else reads the clock.
    clock = {"t": 0.0}
    monkeypatch.setattr(
        preload,
        "time",
        types.SimpleNamespace(
            time=lambda: clock["t"],
            sleep=lambda s: clock.__setitem__("t", clock["t"] + 1e9),
            monotonic=real_time.monotonic,
        ),
    )
    res = preload.warm_instances(models=["a/b"], timeout_s=1, poll_interval_s=0.0)

    assert [(r["region"], r["status"]) for r in res] == [("us-east-1", "timeout")]
    assert [r for r in res if r["status"] != "ok"], "timeout must not be counted as warmed"


def test_warm_falls_back_to_a_pricier_class_when_the_cheap_one_is_rejected(monkeypatch):
    """A capacity rejection must climb the ladder, not leave the region cold.

    Preload mode deliberately never refreshes candidates, so handing the launcher only the cheapest
    class means one clean rejection wastes a region whose A100 capacity was in the same snapshot.
    """
    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(
        lj, "usable_instances", _stocked(A10=["us-east-1"], A100_SXM_40GB=["us-east-1"])
    )
    tried = []

    def picky_launch(spec, instances, **k):
        tried.append(instances[0].gpu)
        if instances[0].gpu == "A10":
            raise RuntimeError("no capacity for A10 in us-east-1")

    monkeypatch.setattr(lj, "launch_and_submit", picky_launch)
    res = preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert tried == ["A10", "A100 SXM 40GB"], "must try the next class after a rejection"
    assert [(r["region"], r["gpu"], r["status"]) for r in res] == [
        ("us-east-1", "A100 SXM 40GB", "ok")
    ]


def test_warm_stops_the_ladder_on_an_ambiguous_create(monkeypatch):
    """An ambiguous create must NOT fall through to the next class: that risks paying for two boxes.

    Every class in a region shares one run_id, and UnreconciledCreateError means Lambda may have billed
    an instance we cannot see. The error exists to forbid another create, so it must end the ladder.
    """
    from flash.providers.core.base import UnreconciledCreateError

    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(
        lj, "usable_instances", _stocked(A10=["us-east-1"], A100_SXM_40GB=["us-east-1"])
    )
    tried = []

    def ambiguous_launch(spec, instances, **k):
        tried.append(instances[0].gpu)
        raise UnreconciledCreateError("ambiguous Lambda launch; refusing another create")

    monkeypatch.setattr(lj, "launch_and_submit", ambiguous_launch)
    res = preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert tried == ["A10"], "an ambiguous create must not trigger a second launch for the same run"
    assert [(r["region"], r["status"]) for r in res] == [("us-east-1", "error")]


def test_warm_ensures_the_region_filesystem_once_before_the_class_ladder(monkeypatch):
    """The filesystem must be ensured ONCE up front, not once per class inside the ladder.

    Creation is non-idempotent; pre-ensuring makes later class attempts observe the existing mount
    instead of billing duplicate filesystems.
    """
    from flash.providers.lambda_.client import api as lambda_api

    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(
        lj, "usable_instances", _stocked(A10=["us-east-1"], A100_SXM_40GB=["us-east-1"])
    )
    ensured, tried = [], []
    # already provisioned, which is the steady state once `--provision` has run
    monkeypatch.setattr(
        lambda_api,
        "list_filesystems",
        lambda **k: [{"name": "flash-weights", "region": {"name": "us-east-1"}}],
    )
    monkeypatch.setattr(
        lambda_api,
        "ensure_filesystem",
        lambda name, region, **k: ensured.append((name, region)) or f"/lambda/nfs/{name}",
    )

    def no_capacity(spec, instances, **k):
        tried.append(instances[0].gpu)
        raise RuntimeError("all 1 Lambda region(s) rejected the launch (no capacity): full")

    monkeypatch.setattr(lj, "launch_and_submit", no_capacity)
    preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert ensured == [], "an already-listed filesystem must never reach the create path"
    assert tried == ["A10", "A100 SXM 40GB"], "capacity rejections should still walk the ladder"


def test_warm_skips_a_region_whose_created_filesystem_is_not_yet_listed(monkeypatch):
    """A create that succeeds but has not appeared in the listing must NOT launch.

    Later `ensure_filesystem` calls are safe only after listing visibility; otherwise launch can
    submit a second non-idempotent create.
    """
    from flash.providers.lambda_.client import api as lambda_api

    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(
        lj, "usable_instances", _stocked(A10=["us-east-1"], A100_SXM_40GB=["us-east-1"])
    )
    tried = []
    # the create "succeeds", but the object never shows up in the listing
    monkeypatch.setattr(lambda_api, "list_filesystems", lambda **k: [])
    monkeypatch.setattr(
        lambda_api, "ensure_filesystem", lambda name, region, **k: f"/lambda/nfs/{name}"
    )
    monkeypatch.setattr(
        lj,
        "launch_and_submit",
        lambda spec, instances, **k: tried.append(instances[0].gpu),
    )
    res = preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert tried == [], (
        "an unlisted filesystem must not launch: the launcher would create a duplicate"
    )
    assert [(r["region"], r["status"]) for r in res] == [("us-east-1", "error")]


def test_warm_does_not_launch_while_the_filesystem_is_unconfirmed(monkeypatch):
    """When the filesystem cannot be confirmed, the region must not launch at all.

    Reconciliation failures can lose their sentinel under capacity wrapping. Launching anyway risks
    a second non-idempotent create, so skip the cold region.
    """
    from flash.providers.lambda_.client import api as lambda_api

    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(
        lj, "usable_instances", _stocked(A10=["us-east-1"], A100_SXM_40GB=["us-east-1"])
    )
    tried = []

    def boom(**k):
        # a bare timeout: Lambda WAS reached, so a create may be in flight, and no sentinel text
        # appears anywhere in the message
        raise RuntimeError("read timed out")

    monkeypatch.setattr(lambda_api, "list_filesystems", boom)
    monkeypatch.setattr(
        lj,
        "launch_and_submit",
        lambda spec, instances, **k: tried.append(instances[0].gpu),
    )
    res = preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert tried == [], "an unconfirmed filesystem must not launch at all"
    assert [(r["region"], r["status"]) for r in res] == [("us-east-1", "error")]


def test_ladder_planning_shares_one_deadline_across_every_class(monkeypatch):
    """Four classes must share ONE planning budget, not each get the full retry budget in turn.

    Each class performs retrying price and capacity calls; stop the ladder when the shared budget is
    gone rather than stacking four outage windows.
    """
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    clock = {"t": 0.0}
    monkeypatch.setattr(preload, "time", types.SimpleNamespace(time=lambda: clock["t"]))
    monkeypatch.setattr(preload, "_LAMBDA_PLANNING_BUDGET_S", 100.0)
    asked, deadlines = [], []

    def hang(gpu, *, deadline_at=None):
        asked.append(gpu)
        deadlines.append(deadline_at)
        clock["t"] += 60.0  # each class burns more than half the shared budget
        raise RuntimeError("instance-types timed out")

    monkeypatch.setattr(lj, "usable_instances", hang)
    assert preload._lambda_warm_targets(lj, None) == ([], False)

    # two classes fit in the budget; the ladder stops instead of walking all four
    assert asked == ["A10", "A100 SXM 40GB"]
    # and it is ONE deadline, not a fresh one per class
    assert deadlines == [100.0, 100.0]


def test_warm_still_walks_the_ladder_when_lambda_was_never_reached(monkeypatch):
    """No credentials means no request was sent, so no create can exist and walking stays safe.

    Without this distinction the pre-ensure would fail on every host lacking LAMBDA_API_KEY, pin the
    ladder to a single class, and silently disable the class fallback this module exists to provide.
    """
    from flash.providers.lambda_.client import api as lambda_api

    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(
        lj, "usable_instances", _stocked(A10=["us-east-1"], A100_SXM_40GB=["us-east-1"])
    )
    tried = []

    def unconfigured(**k):
        # the exact text RestClient.missing_key_message builds in flash/providers/_lifecycle/net/http.py
        raise RuntimeError("LAMBDA_API_KEY not configured on the control-plane host")

    def rejected(spec, instances, **k):
        tried.append(instances[0].gpu)
        raise RuntimeError("all 1 Lambda region(s) rejected the launch (no capacity): full")

    monkeypatch.setattr(lambda_api, "list_filesystems", unconfigured)
    monkeypatch.setattr(lj, "launch_and_submit", rejected)
    preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    assert tried == ["A10", "A100 SXM 40GB"], "an unreachable Lambda must not pin the ladder"


def test_warm_raises_when_capacity_could_not_be_measured(monkeypatch):
    """A total lookup failure must raise, not return an empty list.

    Empty means healthy no-capacity to callers; an outage is a different fact and cannot exit 0.
    """
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(preload, "_lambda_provisioned_regions", lambda: set())

    def down(gpu, **k):
        raise RuntimeError("instance-types: read timed out")

    monkeypatch.setattr(lj, "usable_instances", down)
    with pytest.raises(RuntimeError, match="could not determine Lambda capacity"):
        preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)


def test_warm_reports_a_genuine_zero_capacity_fleet_as_success(monkeypatch):
    """Every class answered and none had capacity: that IS a healthy no-op and must not raise.

    The counterpart to the outage case -- without this the fix would turn an ordinary quiet-inventory
    day into a hard failure.
    """
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    monkeypatch.setattr(preload, "_ensure_status_repo", lambda token: None)
    monkeypatch.setattr(preload, "_lambda_provisioned_regions", lambda: set())
    monkeypatch.setattr(lj, "usable_instances", lambda gpu, **k: [])

    assert preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0) == []


def test_warm_does_not_report_a_partial_sweep_as_a_finished_one(monkeypatch):
    """A class going unanswered while ANOTHER still yields targets must not exit 0.

    Reachable-only denominators can report a perfect warm ratio while whole class regions are
    invisible. Preserve completed paid launches on the exception.
    """
    preload, lj, launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(preload, "_lambda_provisioned_regions", lambda: set())

    def flaky(gpu, **k):
        if gpu == "A10":
            raise RuntimeError("instance-types API down")
        return [_cand("asia-south-1", gpu)] if gpu == "H100" else []

    monkeypatch.setattr(lj, "usable_instances", flaky)
    with pytest.raises(preload.IncompleteWarmPlanError) as exc:
        preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    # the reachable region really was warmed, and that work must survive the raise
    assert [r["region"] for r in exc.value.results] == ["asia-south-1"]
    assert [r["status"] for r in exc.value.results] == ["ok"]
    assert [region for region, _mode, _models in launched] == ["asia-south-1"]


def test_warm_cli_exits_nonzero_when_the_fleet_was_not_fully_measured(monkeypatch, capsys):
    """The operator-visible half: unmeasured must not look like success at the shell.

    The launches still print, since they ran and were billed, but the exit code has to say the
    sweep is unfinished -- a green exit here is what let unexamined regions stay cold unnoticed.
    """
    from flash.providers.artifacts import weight_cache as _p

    model = _p.catalog_model_ids()[0]
    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": [model], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(preload, "_lambda_provisioned_regions", lambda: set())

    def flaky(gpu, **k):
        if gpu == "A10":
            raise RuntimeError("instance-types API down")
        return [_cand("asia-south-1", gpu)] if gpu == "H100" else []

    monkeypatch.setattr(lj, "usable_instances", flaky)
    rc = preload.main(["--warm-instances", "--models", model])
    combined = "".join(capsys.readouterr())

    assert rc == 1, f"an unmeasured fleet must not exit 0: {combined!r}"
    # the paid launch is still reported, not swallowed by the failure
    assert "asia-south-1" in combined, combined
    assert "not fully measured" in combined, combined


def test_warm_incomplete_summary_does_not_contradict_the_warmed_count(monkeypatch, capsys):
    """The two closing lines must not disagree about how many caches finished.

    One count is successful warms and the other is regions examined; label each accordingly.
    """
    from flash.providers.artifacts import weight_cache as _p

    model = _p.catalog_model_ids()[0]
    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": [model], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(preload, "_lambda_provisioned_regions", lambda: set())

    def flaky(gpu, **k):
        if gpu == "A10":
            raise RuntimeError("instance-types API down")
        # two regions launch, and one of them fails -- so warmed (1) != examined (2)
        return [_cand("asia-south-1", gpu), _cand("us-west-2", gpu)] if gpu == "H100" else []

    real_launch = lj.launch_and_submit

    def one_region_fails(spec, instances, **k):
        if instances[0].region == "us-west-2":
            raise RuntimeError("all 1 Lambda region(s) rejected the launch (no capacity): full")
        return real_launch(spec, instances, **k)

    monkeypatch.setattr(lj, "usable_instances", flaky)
    monkeypatch.setattr(lj, "launch_and_submit", one_region_fails)
    rc = preload.main(["--warm-instances", "--models", model])
    combined = "".join(capsys.readouterr())

    assert rc == 1, combined
    assert "1/2 regions warmed" in combined, combined
    assert "warmed 2 region(s)" not in combined, (
        f"the summary must not claim more warmed regions than the tally above it: {combined!r}"
    )
    assert "examined 2 region(s)" in combined, combined


def test_provisioned_region_snapshot_is_deadline_bounded(monkeypatch):
    """The reporting snapshot must carry its own deadline.

    It runs after planning budget exhaustion and must not add another retry cycle for a summary
    line.
    """
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_.client import api as lambda_api

    seen = {}

    def capture(**k):
        seen.update(k)
        return []

    monkeypatch.setattr(lambda_api, "list_filesystems", capture)
    preload._lambda_provisioned_regions()

    assert "deadline_at" in seen, "the optional snapshot must not block on an unbounded listing"
    assert seen["deadline_at"] <= time.time() + preload._LAMBDA_SNAPSHOT_BUDGET_S + 1


def test_warm_names_regions_with_no_capacity_in_any_class(monkeypatch, caplog):
    """A region unreachable in every class produces no result, so only the filesystem list names it.

    Reporting from results alone would silently omit exactly the fleet gap this change exposes.
    """
    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": ["a/b"], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(lj, "usable_instances", _stocked(A10=["us-east-1"]))
    monkeypatch.setattr(
        preload, "_lambda_provisioned_regions", lambda: {"us-east-1", "us-south-2", "us-south-3"}
    )
    with caplog.at_level("WARNING"):
        preload.warm_instances(models=["a/b"], timeout_s=5, poll_interval_s=0.0)

    warned = "\n".join(r.getMessage() for r in caplog.records)
    assert "us-south-2" in warned, warned
    assert "us-south-3" in warned, warned
    assert "no capacity" in warned
    assert "us-east-1" not in warned, "a region that warmed fine must not be reported cold"


def test_warm_counts_launch_time_regions_in_the_fleet_total(monkeypatch):
    """The denominator is the union, not the pre-launch snapshot.

    Launch-time provisioning can add regions absent from the snapshot; include result regions to
    avoid impossible counts such as 2 of 1 incomplete.
    """
    from flash.providers.artifacts import weight_cache as preload

    cold, total = preload._cold_lambda_regions(
        {"us-south-2"}, [{"region": "us-west-1", "status": "error"}]
    )
    assert cold == ["us-south-2", "us-west-1"]
    assert total == 2, f"fleet total must count the launch-time region too, got {total}"
    assert total >= len(cold), "the denominator can never be smaller than the numerator"


def test_warm_cli_prints_regions_with_no_capacity(monkeypatch, capsys):
    """The cold-region report must survive to stdout, not die in an unconfigured logger.

    The module entry point may have only a NullHandler, so stdout must carry the no-capacity
    regions.
    """
    from flash.providers.artifacts import weight_cache as _p

    # a real catalog id: --models refuses off-catalog ids before it ever reaches the warm path
    model = _p.catalog_model_ids()[0]
    preload, lj, _launched, _terminated = _wire_warm(
        monkeypatch, {"preloaded": [model], "already_cached": [], "failed": {}}
    )
    monkeypatch.setattr(lj, "usable_instances", _stocked(A10=["us-east-1"]))
    monkeypatch.setattr(preload, "_lambda_provisioned_regions", lambda: {"us-east-1", "us-south-2"})
    rc = preload.main(["--warm-instances", "--models", model])
    out = capsys.readouterr()
    combined = out.out + out.err
    assert "us-south-2" in combined, f"the starved region never reached the operator: {combined!r}"
    assert "no capacity" in combined, combined
    # every launched region did succeed, so the run is still a success -- the point is visibility
    assert rc == 0


def test_warm_ranks_contested_regions_by_live_price_not_ladder_order(monkeypatch):
    """Selection must follow the live rate: a Lambda discount reorders classes the static tuple cannot.

    ``usable_instances`` already attaches the live ``price_usd_hr``; ignoring it would keep claiming
    regions in a stale snapshot order and launch the more expensive box.
    """
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    def discounted(gpu):
        # H100 discounted below A10 -- the ladder tuple still lists A10 first.
        price = {"A10": 1.29, "H100": 0.49}.get(gpu)
        if price is None:
            return []
        return [types.SimpleNamespace(region="us-east-1", gpu=gpu, price_usd_hr=price)]

    monkeypatch.setattr(lj, "usable_instances", discounted)
    assert _plan(preload._lambda_warm_targets(lj, None)[0]) == [("us-east-1", ["H100", "A10"])]


def test_warm_ties_and_missing_prices_keep_ladder_order(monkeypatch):
    """With no usable live price the selection must degrade to the static cheapest-first order.

    An unavailable price is common; it must not randomize which paid class a region gets.
    """
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

    def priceless(gpu):
        return [types.SimpleNamespace(region="us-east-1", gpu=gpu, price_usd_hr=None)]

    monkeypatch.setattr(lj, "usable_instances", priceless)
    assert _plan(preload._lambda_warm_targets(lj, None)[0]) == [
        ("us-east-1", ["A10", "A100 SXM 40GB", "H100", "B200"])
    ]


def test_warm_cli_prints_the_gpu_class_each_region_used(monkeypatch, capsys):
    """Without --gpu the class is per region, so the printed line is the only cost audit an operator gets."""
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(
        preload,
        "warm_instances",
        lambda **k: [{"provider": "lambda", "region": "us-west-3", "gpu": "H100", "status": "ok"}],
    )
    preload.main(["--warm-instances"])

    out = capsys.readouterr().out
    assert "us-west-3" in out, out
    assert "H100" in out, out


def test_warm_cli_help_does_not_promise_the_old_single_class(monkeypatch, capsys):
    """--help must not still advertise A10 for a paid mode that can now launch up to B200."""
    from flash.providers.artifacts import weight_cache as preload

    with pytest.raises(SystemExit):
        preload.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())

    assert "B200" in help_text, "help hides that the default may launch the priciest class"
    for cls in preload._LAMBDA_PRELOAD_GPU_LADDER:
        assert cls in help_text, f"{cls} missing from --gpu help"


def test_warm_instances_explicit_gpu_overrides_default(monkeypatch):
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

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

    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setenv("FLASH_PRELOAD_STATUS_REPO", "other/repo")
    importlib.reload(preload)
    # the repo follows FLASH_HF_NAMESPACE (unset here -> the managed namespace) and nothing else:
    # FLASH_PRELOAD_STATUS_REPO is not a knob, so setting it must change nothing.
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
    result = preload._warm_one_lambda_instance(jobs_mod, [_cand("us-east-1")], ["a/b"], 5, 0.0)

    assert managed_repo == preload._preload_status_repo()
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


def test_preload_status_repo_follows_self_hosted_namespace(monkeypatch):
    """A self-hoster's HF_TOKEN cannot write to Freesolo-Co, so the status repo -- which the warm
    path CREATES before launching paid GPUs -- has to follow FLASH_HF_NAMESPACE like run artifacts
    do. Hardcoded, the warm path fails at create_repo and blames the operator's HF_TOKEN."""
    import huggingface_hub

    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setenv("FLASH_HF_NAMESPACE", "self-hoster")

    assert preload._preload_status_repo() == "self-hoster/flash-weight-preload"
    # the spec the warm box runs writes there too, so the poller and the box agree.
    assert preload._preload_instance_spec("A10", "r1").train.hf_repo == (
        "self-hoster/flash-weight-preload"
    )

    created = []

    class FakeHfApi:
        def __init__(self, token):
            self.token = token

        def create_repo(self, repo, **kwargs):
            created.append(repo)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    preload._ensure_status_repo("token")
    assert created == ["self-hoster/flash-weight-preload"]


def test_warm_instances_requires_status_repo_before_launch(monkeypatch):
    """With targets available, warm must validate the status repo BEFORE launching any paid box.
    Capacity enumeration is cheap/read-only, so it runs first (to decide if there's anything to do);
    the status-repo guard gates the actual LAUNCH, which is what costs money."""
    import types

    import pytest

    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

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


def test_warm_instances_cli_reports_planning_outage_without_traceback(monkeypatch):
    """A total planning outage / unusable status repo raises bare RuntimeError from warm_instances.
    The CLI must turn that into a message and exit 1, not an unhandled traceback: both conditions are
    operator-actionable (Lambda down, HF_TOKEN missing), and a traceback buries the reason."""
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(
        preload,
        "warm_instances",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("could not determine Lambda capacity")),
    )
    assert preload.main(["--warm-instances"]) == 1


def test_precheck_cannot_take_time_from_the_launch_deadline(monkeypatch):
    """The pre-check runs on its OWN budget, so a slow one never shortens the launch/poll deadline.

    Sharing the run deadline can consume create allowance and end polling before a live download.
    """
    from flash.providers.artifacts import weight_cache as preload

    # A slow pre-check that returns "listed" but burns two minutes of wall clock getting there.
    def slow_precheck(region, deadline):
        clock["now"] += 120.0
        return "listed"

    clock = {"now": 1000.0}
    seen = {}
    monkeypatch.setattr(preload.time, "time", lambda: clock["now"])
    monkeypatch.setattr(preload, "_ensure_region_filesystem", slow_precheck)

    def _launch(spec, **kwargs):
        seen["deadline_at"] = kwargs["deadline_at"]
        seen["wall_s"] = spec.gpu.max_wall_seconds
        raise RuntimeError("stop here; the launch deadline is what this test is about")

    jobs_mod = types.SimpleNamespace(
        launch_and_submit=_launch, terminate_run_instances=lambda run_id: None
    )
    preload._warm_one_lambda_instance(jobs_mod, [_cand("us-east-1")], ["a/b"], 600, 0.0)

    # The class WAS tried (the old code returned before the ladder), and it got the full budget:
    # the launch deadline is 600s from the pre-check's END, matching the instance's own wall cap.
    assert seen["wall_s"] == 600
    assert seen["deadline_at"] == clock["now"] + 600


def test_warm_instances_no_targets_is_noop_without_status_repo(monkeypatch):
    """No provider capacity -> documented no-op: warm returns [] and must NOT require the status repo
    (else an empty warm on an unconfigured / at-capacity host would hard-fail on a missing HF_TOKEN)."""
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.lambda_ import jobs as lj

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
    from flash.providers.artifacts import weight_cache as preload

    called = {"n": 0}
    monkeypatch.setattr(
        preload, "warm_instances", lambda **k: called.__setitem__("n", called["n"] + 1) or []
    )
    assert preload.main(["--warm-instances", "--dry-run"]) == 0
    assert called["n"] == 0  # dry-run launches nothing


def test_cli_gpu_default_is_none_per_mode(monkeypatch):
    """--gpu defaults to None; each mode applies its OWN default downstream (no sentinel hack).

    An explicit RTX 4090 must remain distinguishable from no override.
    """
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

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
    # gave up on the grace window, nowhere near the 5400s timeout. Bound allows two 60s polls on
    # top of the grace: GraceTimer arms on the first confirmed reading rather than at launch, and
    # never fires on the poll that armed it, so give-up lands a poll or two past _NO_CAPACITY_GRACE_S
    assert clock["t"] <= preload._NO_CAPACITY_GRACE_S + 120.0


def test_poll_waits_when_a_worker_is_initializing(monkeypatch):
    """An initializing worker proves capacity exists: a slow download must NOT be cut short."""
    from flash.providers.artifacts import weight_cache as preload

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

    A later requeue needs a fresh grace window or its first zero-worker reading can delete a proven
    capacity endpoint.
    """
    from flash.providers.artifacts import weight_cache as preload

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


def test_poll_rechecks_health_after_a_requeue_loses_the_worker(monkeypatch):
    """The worker latch is per queued interval: a requeue with no worker must still be caught.

    Clear it after interruption or later health probes remain disabled for the full 5400s timeout.
    """
    from flash.providers.artifacts import weight_cache as preload

    clock = {"t": 0.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["t"])
    monkeypatch.setattr(preload.time, "sleep", lambda *_: clock.__setitem__("t", clock["t"] + 60.0))
    # queued with a worker, runs, then is re-queued for good with the worker gone.
    seq = iter(["IN_QUEUE", "IN_PROGRESS"])
    monkeypatch.setattr(
        preload.runpod_api, "job_status", lambda *a, **k: {"status": next(seq, "IN_QUEUE")}
    )
    # healthy on the first probe (latching the old bug), zero on every probe after the requeue
    probes = {"n": 0}

    def _health(*_a, **_k):
        probes["n"] += 1
        alive = 1 if probes["n"] == 1 else 0
        return {"workers": {"initializing": alive, "ready": 0, "running": 0, "idle": 0}}

    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", _health)
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 30.0)  # one sleep outruns it

    with pytest.raises(preload.NoCapacityError):
        preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=1.0)
    # health was probed again AFTER the requeue rather than skipped by a stale latch
    assert probes["n"] > 1


def test_poll_rechecks_health_when_a_worker_vanishes_without_leaving_the_queue(monkeypatch):
    """A worker that appears and then disappears while still IN_QUEUE must not latch the probe off.

    Clear the latch when the worker vanishes or starvation detection stays disabled without a queue
    transition.
    """
    from flash.providers.artifacts import weight_cache as preload

    clock = {"t": 0.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["t"])
    monkeypatch.setattr(preload.time, "sleep", lambda *_: clock.__setitem__("t", clock["t"] + 60.0))
    # never leaves the queue: the status stream alone gives the poller no reason to re-probe
    monkeypatch.setattr(preload.runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})
    # one healthy reading, then the box is gone for good
    probes = {"n": 0}

    def _health(*_a, **_k):
        probes["n"] += 1
        alive = 1 if probes["n"] == 1 else 0
        return {"workers": {"initializing": alive, "ready": 0, "running": 0, "idle": 0}}

    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", _health)
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 30.0)  # one sleep outruns it

    with pytest.raises(preload.NoCapacityError):
        preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=1.0)
    # the vanished worker was noticed instead of being masked by the latch
    assert probes["n"] > 1
    assert clock["t"] < 5400
    # and it gave up in the grace window instead of burning the full timeout
    assert clock["t"] < 5400


def test_has_worker_treats_health_failure_as_unknown(monkeypatch):
    """Health is a hint: an API error must not be read as 'no capacity' and kill a live download.

    None, not False -- the caller escalates a sustained False into NoCapacityError and deletes the
    endpoint, so conflating "cannot tell" with "no workers" would kill healthy downloads.
    """
    from flash.providers.artifacts import weight_cache as preload

    def _boom(*_a, **_k):
        raise RuntimeError("health api down")

    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", _boom)
    assert preload._worker_counts("ep", "fp", 0.0) is None
    assert preload._has_worker(None) is None

    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", lambda *a, **k: None)
    assert preload._worker_counts("ep", "fp", 0.0) is None

    monkeypatch.setattr(
        preload.runpod_api, "endpoint_health_for_fingerprint", lambda *a, **k: {"workers": {}}
    )
    # a real empty answer IS evidence
    assert preload._has_worker(preload._worker_counts("ep", "fp", 0.0)) is False


def test_a_broken_worker_image_is_not_reported_as_a_starved_datacenter(monkeypatch):
    """An unhealthy worker still means the datacenter gave us a box, so it refutes starvation.

    Classify it as an image failure, not no capacity; changing GPU class cannot repair a broken
    image.
    """
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": {"ready": 0, "running": 0, "unhealthy": 1}},
    )
    assert preload._has_worker(preload._worker_counts("ep", "fp", 0.0)) is True

    # and end to end: the queued job must NOT be blamed on the datacenter. Within the unhealthy
    # grace it is still just a slow start, so the budget running out is a plain timeout.
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 0.0)  # grace already elapsed
    monkeypatch.setattr(preload.runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})
    with pytest.raises(TimeoutError):  # NOT NoCapacityError
        preload._poll_until_done("ep", "job", "fp", timeout_s=1, poll_interval_s=0.0)


def test_a_persistently_unhealthy_worker_is_reported_as_a_broken_image(monkeypatch):
    """A box allocated and then left unhealthy is a broken image, and nothing but a fix helps.

    Give this state its own timer or it suppresses starvation and holds a paid endpoint for 5400s.
    Match `poll_job`'s 240-second failed-image grace.
    """
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": {"ready": 0, "running": 0, "unhealthy": 1}},
    )
    monkeypatch.setattr(preload.runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})
    monkeypatch.setattr(preload, "_UNHEALTHY_GRACE_S", 0.0)  # grace already elapsed

    with pytest.raises(RuntimeError, match="unhealthy") as exc:
        preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=0.0)
    # and it must be actionable: a broken image, NOT a capacity verdict that sends the operator
    # off to pick a different GPU class.
    assert not isinstance(exc.value, preload.NoCapacityError)
    assert "image failed to start" in str(exc.value)


def test_an_unhealthy_worker_alongside_a_live_one_is_not_a_broken_image(monkeypatch):
    """One bad box among healthy ones is not a failed image, so it must not abort the warm.

    Same predicate ``poll_job`` uses: unhealthy only counts when nothing is usable and nothing is
    still coming up. Firing here would throw away a download that is progressing on the good box.
    """
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(preload, "_UNHEALTHY_GRACE_S", 0.0)  # grace already elapsed
    monkeypatch.setattr(preload.runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})
    for alive in ("running", "ready", "idle", "initializing"):
        monkeypatch.setattr(
            preload.runpod_api,
            "endpoint_health_for_fingerprint",
            lambda *a, _s=alive, **k: {"workers": {"unhealthy": 1, _s: 1}},
        )
        assert preload._only_unhealthy_workers(preload._worker_counts("ep", "fp", 0.0)) is False
        with pytest.raises(TimeoutError):  # ran out of budget, NOT declared broken
            preload._poll_until_done("ep", "job", "fp", timeout_s=1, poll_interval_s=0.0)


def test_a_throttled_worker_still_counts_as_no_capacity(monkeypatch):
    """Throttled is the no-capacity signal itself, so it must NOT suppress the fail-fast path.

    Sustained throttling must follow `jobs.py` into GPU escalation, not wait the full 5400s timeout.
    """
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": {"ready": 0, "running": 0, "throttled": 1}},
    )
    assert preload._has_worker(preload._worker_counts("ep", "fp", 0.0)) is False

    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 0.0)  # grace already elapsed
    monkeypatch.setattr(preload.runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})
    with pytest.raises(preload.NoCapacityError):
        preload._poll_until_done("ep", "job", "fp", timeout_s=60, poll_interval_s=0.0)


def test_unreadable_health_never_becomes_no_capacity(monkeypatch):
    """A persistently failing health API must not masquerade as a starved datacenter.

    Unreadable health cannot prove zero workers or trigger endpoint deletion.
    """
    from flash.providers.artifacts import weight_cache as preload

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

    Health blackouts cannot age the timer; the first later zero-worker reading must start from prior
    confirmed duration only.
    """
    from flash.providers.artifacts import weight_cache as preload

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

    Run starvation checks only while queued or stale health can delete an active download.
    """
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

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
    from flash.providers.artifacts import weight_cache as preload

    ids = preload.catalog_model_ids()
    from flash.core.catalog import MODELS

    sizes = [MODELS[m].params_b or 0.0 for m in ids]
    assert sizes == sorted(sizes, reverse=True), (
        f"catalog must be largest-first, got {list(zip(ids, sizes, strict=True))}"
    )


def test_volume_holds_whole_catalog_with_largest_model_in_transit():
    """The volume must fit every model resident PLUS the largest one's download scratch.

    Persistent volumes evict nothing, so download order cannot prevent the observed 200 GB
    "Disk quota exceeded" failure.
    """
    from flash.runner.accounting.weight_cache import (
        WEIGHT_CACHE_VOLUME_GB,
        weight_cache_catalog_peak_gb,
    )

    needed = weight_cache_catalog_peak_gb()
    assert needed <= WEIGHT_CACHE_VOLUME_GB, (
        f"catalog needs {needed:.1f} GB at peak but the volume is {WEIGHT_CACHE_VOLUME_GB} GB: "
        "the largest model will fail with Disk quota exceeded on every datacenter"
    )


def test_preload_timeout_covers_a_fully_cold_whole_catalog_warm():
    """The default budget must outlast downloading the ENTIRE catalog, not just one model.

    The 250 GB preload pulls about 159 GB cold; size from the measured 35B rate of 70 GB in about
    870 seconds. A timeout discards all progress.
    """
    from flash.core.catalog import MODELS
    from flash.providers.artifacts.weight_cache import _PRELOAD_TIMEOUT_S
    from flash.runner.accounting.weight_cache import _download_gb, _fits_weight_cache

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

    from flash.providers.artifacts import weight_cache as preload

    defaults = {
        name: inspect.signature(fn).parameters["timeout_s"].default
        for name, fn in (
            ("warm_weight_cache", preload.warm_weight_cache),
            ("warm_instances", preload.warm_instances),
        )
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
    from flash.core.catalog import MODELS
    from flash.runner.accounting.weight_cache import (
        _fits_weight_cache,
        weight_cache_catalog_peak_gb,
    )

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
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(preload, "catalog_model_ids", lambda: ["m1", "m2"])
    monkeypatch.setattr(preload, "weight_cache_datacenters", list)
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
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(preload, "catalog_model_ids", lambda: ["m1"])
    monkeypatch.setattr(preload, "weight_cache_datacenters", list)
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


def test_grow_raises_undersized_volumes_and_leaves_the_rest_alone(monkeypatch):
    """A NetworkVolume is sized on CREATE ONLY, so a size bump is a silent no-op on the live fleet.

    Existing 100 GB volumes require REST growth to reach 250 GB or larger models still hit
    "Disk quota exceeded".
    """
    from flash.providers.runpod.client import api

    calls = []

    class _Client:
        def request_with_retries_for_key(self, key, target, **kwargs):
            calls.append((target, kwargs.get("method", "GET"), kwargs.get("body")))
            if target.endswith("/networkvolumes"):
                return [
                    {"id": "v-old", "name": "flash-weights-us-ca-2", "size": 100},
                    {"id": "v-new", "name": "flash-weights-eu-ro-1", "size": 250},
                    {"id": "v-big", "name": "flash-weights-us-tx-3", "size": 400},
                    {"id": "v-other", "name": "somebody-elses-volume", "size": 10},
                ]
            return {}

    monkeypatch.setattr(api, "_CLIENT", _Client())
    grown = api.grow_network_volumes_for_key(
        "k",
        {
            "flash-weights-us-ca-2": 250,
            "flash-weights-eu-ro-1": 250,
            "flash-weights-us-tx-3": 250,
        },
    )

    # only the under-sized managed volume is touched
    assert grown == {"flash-weights-us-ca-2": 250}
    patches = [(t, m, b) for t, m, b in calls if m == "PATCH"]
    assert patches == [(f"{api.REST_BASE}/networkvolumes/v-old", "PATCH", {"size": 250})]
    # RunPod REJECTS a shrink, so an already-larger volume must never be patched down; and a volume
    # that is not ours is not ours to resize.
    assert "v-big" not in str(patches)
    assert "v-other" not in str(calls)


def test_grow_tolerates_a_volume_listing_without_usable_sizes(monkeypatch):
    """A malformed row must be skipped, not crash the warm: this runs on the launch path.

    Raising here would take down a preload for every datacenter over one bad row from the listing
    API, which is strictly worse than attaching a volume at whatever size it already has.
    """
    from flash.providers.runpod.client import api

    patched = []

    class _Client:
        def request_with_retries_for_key(self, key, target, **kwargs):
            if target.endswith("/networkvolumes"):
                return {
                    "networkVolumes": [
                        {"id": "v-1", "name": "flash-weights-us-ca-2", "size": "not-a-number"},
                        {"id": None, "name": "flash-weights-eu-ro-1", "size": 100},
                        {"name": "flash-weights-us-tx-3", "size": 100},
                        {"id": "v-4", "name": "flash-weights-us-ks-2", "size": 100},
                    ]
                }
            patched.append(target)
            return {}

    monkeypatch.setattr(api, "_CLIENT", _Client())
    grown = api.grow_network_volumes_for_key(
        "k",
        dict.fromkeys(
            [
                "flash-weights-us-ca-2",
                "flash-weights-eu-ro-1",
                "flash-weights-us-tx-3",
                "flash-weights-us-ks-2",
            ],
            250,
        ),
    )

    assert grown == {"flash-weights-us-ks-2": 250}  # only the one complete, under-sized row
    assert patched == [f"{api.REST_BASE}/networkvolumes/v-4"]


def test_one_stalling_volume_cannot_starve_the_rest_of_the_fleet(monkeypatch):
    """Regression: a PATCH that burns the shared deadline left every later datacenter unreconciled.

    Bound each failed growth attempt so one timeout cannot consume the fleet-wide reconciliation
    budget and leave later volumes stale.
    """
    import types

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers.runpod.client import api

    clock = {"t": 10_000.0}
    monkeypatch.setattr(_deadline, "time", types.SimpleNamespace(time=lambda: clock["t"]))
    monkeypatch.setattr(api, "time", types.SimpleNamespace(time=lambda: clock["t"]))

    granted = []

    class _Client:
        def request_with_retries_for_key(self, key, target, **kwargs):
            if target.endswith("/networkvolumes"):
                return [
                    {"id": f"v-{i}", "name": f"flash-weights-dc-{i}", "size": 100} for i in range(4)
                ]
            budget = kwargs["deadline_at"] - clock["t"]
            granted.append(round(budget, 1))
            if budget <= 0:
                raise RuntimeError("PATCH deadline exceeded")
            if target.endswith("v-0"):
                clock["t"] += budget  # a stalling volume spends everything it is given
                raise RuntimeError("PATCH timed out")
            clock["t"] += 0.5
            return {}

    monkeypatch.setattr(api, "_CLIENT", _Client())
    grown = api.grow_network_volumes_for_key(
        "k",
        {f"flash-weights-dc-{i}": 250 for i in range(4)},
        deadline_at=clock["t"] + 20.0,
    )

    # the stalling volume gets its own share, not the whole budget, so the rest still reconcile
    assert granted[0] == 5.0
    assert grown == {f"flash-weights-dc-{i}": 250 for i in (1, 2, 3)}


def test_warm_grows_the_volume_before_attaching_it(monkeypatch):
    """The grow must happen BEFORE the endpoint deploys, or the warm attaches the old size.

    Ordering is the whole point: reconciling after the attach leaves this run downloading into the
    under-sized mount it was supposed to fix.
    """
    import inspect

    from flash.providers.runpod.execution import job_execution

    # the grow must sit ahead of the create inside the same serialized attempt
    src = inspect.getsource(job_execution.deploy_train_endpoint)
    assert src.index("grow_weight_cache_volumes(") < src.index("ep = Endpoint(**kwargs)")

    # and it asks for the MANAGED size on this DC's real (per-DC) volume name
    asked = {}
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: asked.update(wanted) or {},
    )
    runpod_resources.grow_weight_cache_volumes(
        None,
        "k1",
        None,
        wanted={"flash-weights-us-ca-2": runner_weight_cache.WEIGHT_CACHE_VOLUME_GB},
    )
    assert asked == {"flash-weights-us-ca-2": runner_weight_cache.WEIGHT_CACHE_VOLUME_GB}


def test_a_bad_account_key_never_blocks_a_warm(monkeypatch):
    """Growing is best-effort: a failing grow must not stop the warm from deploying.

    A volume that cannot be grown still gets attached, which is no worse than not trying at all --
    but a warm aborted over an expired key warms nothing.
    """
    from flash.providers.artifacts import weight_cache as preload

    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: (_ for _ in ()).throw(RuntimeError("401 unauthorized")),
    )
    # the grow raising must not propagate out of the helper the deploy calls
    runpod_resources.grow_weight_cache_volumes(
        None, "bad", None, wanted={"flash-weights-us-ca-2": 250}
    )

    monkeypatch.setattr(preload, "deploy_train_endpoint", lambda *a, **k: ("ep", "name", "fp"))
    monkeypatch.setattr(preload.runpod_api, "submit_job", lambda *a, **k: "job-1")
    monkeypatch.setattr(preload, "_poll_until_done", lambda *a, **k: {"preloaded": ["m1"]})
    monkeypatch.setattr(preload.runpod_api, "delete_endpoint_for_fingerprint", lambda *a, **k: None)

    out = preload._preload_one_dc("US-CA-2", ["m1"], None, "RTX 4090", 60, 0.0)

    assert out["status"] == "ok"  # the warm still ran


def test_assign_refreshes_a_stale_shared_cache_size():
    """A spec already pinned to the shared cache must still be re-sized to the MANAGED size.

    A correct volume name can still carry the stale 100 GB size and admit models that need 250 GB.
    """
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {
                "type": "H100",
                "network_volume": runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME,
                "network_volume_gb": 100,
            },
        }
    )
    out = runner_weight_cache._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME
    assert out.gpu.network_volume_gb == runner_weight_cache.WEIGHT_CACHE_VOLUME_GB


def test_assign_leaves_a_custom_volume_size_alone():
    """The size refresh is scoped to the SHARED cache: a caller's own volume keeps its own size.

    A per-org volume is the caller's to size, and rewriting it to the managed number would silently
    change what they are billed for.
    """
    from flash.core.catalog import MODELS

    info = MODELS["Qwen/Qwen3.5-9B"]
    spec = JobSpec.from_dict(
        {
            "model": info.id,
            "run_id": "r",
            "gpu": {"type": "H100", "network_volume": "my-org-cache", "network_volume_gb": 100},
        }
    )
    out = runner_weight_cache._assign_weight_cache_volume(spec, info)
    assert out.gpu.network_volume == "my-org-cache"
    assert out.gpu.network_volume_gb == 100


def test_an_unknown_job_status_cannot_restart_the_starvation_grace(monkeypatch):
    """Only a status that PROVES the job left the queue may reset the no-capacity timer.

    None or unknown status must not restart grace and hide a permanently starved datacenter.
    """
    from flash.providers.artifacts import weight_cache as preload

    # alternate IN_QUEUE with junk: under the old rule the junk poll cleared the anchor every time
    statuses = iter(["IN_QUEUE", None, "IN_QUEUE", "SOMETHING_NEW", "IN_QUEUE"] * 50)
    monkeypatch.setattr(
        preload.runpod_api, "job_status", lambda *a, **k: {"status": next(statuses)}
    )
    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": {}},  # confirmed zero workers
    )
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 0.0)

    with pytest.raises(preload.NoCapacityError):
        preload._poll_until_done("ep", "job", "fp", timeout_s=60, poll_interval_s=0.0)


def test_a_running_job_still_clears_the_starvation_grace(monkeypatch):
    """The reset must still happen for a status that DOES prove allocation.

    After IN_PROGRESS, a later requeue needs a fresh grace window rather than inherited starvation.
    """
    from flash.providers.artifacts import weight_cache as preload

    clock = {"now": 1000.0}
    statuses = iter(["IN_QUEUE", "IN_PROGRESS", "IN_QUEUE", "COMPLETED"])

    def _status(*_a, **_k):
        # jump well past the grace between the first queued poll and the running one, so a stale
        # anchor carried forward would fire on the SECOND queued poll
        clock["now"] += 200.0
        return {"status": next(statuses), "output": {"preloaded": ["m1"]}}

    monkeypatch.setattr(preload.time, "time", lambda: clock["now"])
    monkeypatch.setattr(preload.time, "sleep", lambda _s: None)
    monkeypatch.setattr(preload.runpod_api, "job_status", _status)
    monkeypatch.setattr(
        preload.runpod_api, "endpoint_health_for_fingerprint", lambda *a, **k: {"workers": {}}
    )
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 100.0)
    monkeypatch.setattr(preload, "decode_output", lambda out: out)

    out = preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=0.0)
    assert out == {"preloaded": ["m1"]}


def test_a_status_blackout_is_not_charged_to_the_starvation_grace(monkeypatch):
    """Time the poller could not observe must not age an armed timer.

    A job may run and requeue during a status blackout. Hold confirmed queued duration, but do not
    charge blind time or fail on the first later reading.
    """
    from flash.providers.artifacts import weight_cache as preload

    clock = {"now": 1000.0}
    # 2 confirmed queued polls, a long unreadable gap, then queued again and done
    statuses = iter(["IN_QUEUE", "IN_QUEUE", None, None, "IN_QUEUE", "COMPLETED"])
    gaps = iter([0.0, 60.0, 150.0, 150.0, 10.0, 10.0])

    def _status(*_a, **_k):
        clock["now"] += next(gaps)
        return {"status": next(statuses), "output": {"preloaded": ["m1"]}}

    monkeypatch.setattr(preload.time, "time", lambda: clock["now"])
    monkeypatch.setattr(preload.time, "sleep", lambda _s: None)
    monkeypatch.setattr(preload.runpod_api, "job_status", _status)
    monkeypatch.setattr(
        preload.runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *a, **k: {"workers": {}},  # confirmed zero workers on every queued poll
    )
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 100.0)
    monkeypatch.setattr(preload, "decode_output", lambda out: out)

    out = preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=0.0)
    assert out == {"preloaded": ["m1"]}


def test_a_blackout_only_pauses_the_grace_and_never_restarts_it(monkeypatch):
    """The pause must not become a reset: a genuinely starved DC still has to fail.

    Preserve confirmed queued duration across blackouts so intermittent errors cannot defer failure
    for the full timeout.
    """
    from flash.providers.artifacts import weight_cache as preload

    clock = {"now": 1000.0}
    # every confirmed queued poll is separated by an unreadable one, so a reset-on-unknown rule
    # would never let the anchor survive long enough to fire
    statuses = iter(["IN_QUEUE", None] * 60)

    def _status(*_a, **_k):
        status = next(statuses)
        # charge the elapsed time to the intervals that END at a confirmed reading; the blackouts
        # themselves are instantaneous, so only genuinely confirmed queued time accumulates
        if status == preload._QUEUED:
            clock["now"] += 40.0
        return {"status": status}

    monkeypatch.setattr(preload.time, "time", lambda: clock["now"])
    monkeypatch.setattr(preload.time, "sleep", lambda _s: None)
    monkeypatch.setattr(preload.runpod_api, "job_status", _status)
    monkeypatch.setattr(
        preload.runpod_api, "endpoint_health_for_fingerprint", lambda *a, **k: {"workers": {}}
    )
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 100.0)

    with pytest.raises(preload.NoCapacityError):
        preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=0.0)


def test_a_health_blackout_cannot_restart_the_starvation_grace(monkeypatch):
    """An unreadable health API must pause the timers, not clear them.

    `_worker_counts` returns None when blind; clearing anchors lets intermittent failures prevent
    starvation, unhealthy, and throttled timers from ever firing.
    """
    from flash.providers.artifacts import weight_cache as preload

    clock = {"now": 1000.0}
    polls = {"n": 0}

    def _blind(n: int) -> bool:
        return n % 2 == 0

    def _status(*_a, **_k):
        # the clock must move BEFORE the loop captures `now`, so charge the advance here rather than
        # in the health stub. Time is charged only to intervals that END at a confirmed reading; the
        # blackouts are instantaneous, so nothing but genuinely observed queued time accumulates.
        if not _blind(polls["n"]):
            clock["now"] += 40.0
        return {"status": "IN_QUEUE"}

    def _health(*_a, **_k):
        # counter-driven, not an iterator: a StopIteration here would be swallowed by
        # _worker_counts into a None reading and the poller would spin on a frozen clock
        n = polls["n"]
        polls["n"] += 1
        return None if _blind(n) else {"workers": {}}

    monkeypatch.setattr(preload.time, "time", lambda: clock["now"])
    monkeypatch.setattr(preload.time, "sleep", lambda _s: None)
    monkeypatch.setattr(preload.runpod_api, "job_status", _status)
    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", _health)
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 100.0)

    with pytest.raises(preload.NoCapacityError):
        preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=0.0)


def test_a_health_blackout_is_not_charged_to_the_grace_either(monkeypatch):
    """The pause must stay a pause: blind time may not age an armed timer.

    A worker may appear during a health blackout. Charge only confirmed state time or the next
    reading can tear down a healthy download.
    """
    from flash.providers.artifacts import weight_cache as preload

    clock = {"now": 1000.0}
    polls = {"n": 0}
    statuses = ["IN_QUEUE"] * 5 + ["COMPLETED"]
    # confirmed zero workers, a long unreadable stretch, then confirmed zero workers again: the
    # post-blackout reading must stay a zero so a charge-the-gap rule actually has a timer to fire
    healths = [{"workers": {}}, {"workers": {}}, None, None, {"workers": {}}]
    gaps = [0.0, 60.0, 150.0, 150.0, 10.0, 10.0]

    def _status(*_a, **_k):
        clock["now"] += gaps[polls["n"]]
        return {"status": statuses[polls["n"]], "output": {"preloaded": ["m1"]}}

    def _health(*_a, **_k):
        # indexed, not an iterator: _worker_counts swallows any exception into a None reading, so a
        # StopIteration would silently become a blackout and spin the poller on a frozen clock
        health = healths[polls["n"]]
        polls["n"] += 1
        return health

    monkeypatch.setattr(preload.time, "time", lambda: clock["now"])
    monkeypatch.setattr(preload.time, "sleep", lambda _s: None)
    monkeypatch.setattr(preload.runpod_api, "job_status", _status)
    monkeypatch.setattr(preload.runpod_api, "endpoint_health_for_fingerprint", _health)
    monkeypatch.setattr(preload, "_NO_CAPACITY_GRACE_S", 100.0)
    monkeypatch.setattr(preload, "decode_output", lambda out: out)

    out = preload._poll_until_done("ep", "job", "fp", timeout_s=5400, poll_interval_s=0.0)
    assert out == {"preloaded": ["m1"]}


def test_a_throttled_worker_is_not_a_broken_image(monkeypatch):
    """unhealthy + throttled is capacity contention, NOT a failed image pull.

    Throttling uses a longer grace; the shorter unhealthy timer would misclassify and tear down a
    potentially runnable box.
    """
    from flash.providers.artifacts import weight_cache as preload

    assert preload._only_unhealthy_workers({"unhealthy": 1, "throttled": 1}) is False
    # with nothing else alive it IS the broken-image case
    assert preload._only_unhealthy_workers({"unhealthy": 1}) is True


def test_an_ordinary_deploy_grows_the_stale_volume_before_attaching(monkeypatch):
    """Regression: growth used to live only in the preload utility.

    Ordinary training must grow existing volumes before attach or newly admitted models mount the
    stale size and fail with "Disk quota exceeded".
    """

    calls = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: calls.append((key, dict(wanted))) or {},
    )

    spec = _vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME)
    runpod_resources.grow_weight_cache_volumes(spec, "owning-key")

    assert len(calls) == 1
    key, wanted = calls[0]
    # scoped to the account actually being deployed under, not the whole pool
    assert key == "owning-key"
    # every storage DC's real per-DC volume name, at the managed size
    assert wanted
    assert set(wanted.values()) == {runner_weight_cache.WEIGHT_CACHE_VOLUME_GB}
    assert all(
        name.startswith(f"{runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME}-") for name in wanted
    )


def test_an_ordinary_deploy_leaves_a_custom_volume_alone(monkeypatch):
    """Only the platform-managed cache is reconciled; a custom volume is the caller's to size."""

    calls = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: calls.append((key, dict(wanted))) or {},
    )

    runpod_resources.grow_weight_cache_volumes(_vol_spec("my-own-volume"), "k")
    runpod_resources.grow_weight_cache_volumes(_vol_spec(None), "k")

    assert calls == []


def test_a_failed_grow_never_blocks_an_ordinary_deploy(monkeypatch):
    """Reconciliation is best-effort: attaching the old size beats failing the deploy outright."""

    def _boom(*_a, **_k):
        raise RuntimeError("runpod unreachable")

    monkeypatch.setattr(runpod_resources.runpod_api, "grow_network_volumes_for_key", _boom)

    # must not raise
    runpod_resources.grow_weight_cache_volumes(
        _vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME), "k"
    )


def test_deploy_side_grow_takes_a_bounded_budget_not_the_run_deadline(monkeypatch):
    """A bad account must not eat the launch allowance the deploy still needs.

    Bound its retrying listing separately so a healthy account retains the 60-second create minimum.
    """

    seen = {}
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: seen.update(kw) or {},
    )

    before = time.time()
    runpod_resources.grow_weight_cache_volumes(
        _vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME), "k"
    )

    assert seen["deadline_at"] - before <= runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S + 1.0


def test_the_warm_names_the_volume_the_deploy_must_reconcile(monkeypatch):
    """The warm attaches its own volume with spec=None, so the deploy cannot derive one.

    Without cache_volumes the deploy-side grow early-returns on spec=None and nothing reconciles
    the warm's mount at all.
    """
    from flash.providers.artifacts import weight_cache as preload

    seen = {}

    def _deploy(gpu, **kw):
        seen.update(kw)
        raise RuntimeError("stop here; the kwargs are what we are asserting on")

    monkeypatch.setattr(preload, "deploy_train_endpoint", _deploy)
    preload._preload_one_dc("US-CA-2", ["m"], None, "RTX 4090", 600, 1.0)

    assert seen["cache_volumes"] == {"flash-weights-us-ca-2": 250}


def test_a_short_timeout_still_reconciles(monkeypatch):
    """Regression: the grow budget is headroom ON TOP of timeout_s, not a slice carved out of it.

    Growth yields the 60-second create allowance, so callers must add headroom or reconciliation
    receives no budget.
    """
    from flash.providers._lifecycle.net.deadline import CREATE_ALLOWANCE_S
    from flash.providers.artifacts import weight_cache as preload
    from flash.providers.runpod.execution.resources import (
        WEIGHT_CACHE_GROW_BUDGET_S,
        weight_cache_grow_headroom_s,
    )

    clock = {"t": 5000.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["t"])

    seen = {}

    def _deploy(gpu, **kw):
        seen.update(kw)
        raise RuntimeError("stop here; the deadline is what we are asserting on")

    monkeypatch.setattr(preload, "deploy_train_endpoint", _deploy)
    preload._preload_one_dc("US-CA-2", ["m"], None, "RTX 4090", 60, 1.0)

    # the grow clamps to remaining - CREATE_ALLOWANCE_S, so headroom must survive that subtraction
    budget = (seen["deadline_at"] - clock["t"]) - CREATE_ALLOWANCE_S
    assert budget > 0
    # and cover a whole deploy, not one attempt: a failover reconciles the account it lands on too
    assert budget >= weight_cache_grow_headroom_s()
    assert weight_cache_grow_headroom_s() >= WEIGHT_CACHE_GROW_BUDGET_S


def test_deploy_side_grow_yields_the_create_allowance_it_sits_in_front_of(monkeypatch):
    """Regression: growing must never be the reason a launchable deploy is rejected.

    Growth must yield the create allowance before `_deploy_once` rechecks it.
    """
    from flash.providers._lifecycle.net.deadline import CREATE_ALLOWANCE_S

    calls = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: calls.append(kw) or {},
    )
    spec = _vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME)

    # only 10s of slack above the allowance: the full budget would overrun it
    tight = time.time() + CREATE_ALLOWANCE_S + 10.0
    before = time.time()
    runpod_resources.grow_weight_cache_volumes(spec, "k", tight)
    assert calls, "a deploy with room to spare should still reconcile"
    assert calls[0]["deadline_at"] <= tight - CREATE_ALLOWANCE_S + 1.0
    assert calls[0]["deadline_at"] - before <= runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S + 1.0

    # no slack at all: skip entirely rather than eat into the allowance
    calls.clear()
    runpod_resources.grow_weight_cache_volumes(spec, "k", time.time() + CREATE_ALLOWANCE_S)
    assert calls == []


def test_deploy_side_grow_is_wired_to_the_run_deadline(monkeypatch):
    """The deadline must actually reach the helper, not just be honoured once it does."""
    import inspect

    from flash.providers.runpod.execution import job_execution

    src = inspect.getsource(job_execution.deploy_train_endpoint)
    compact = "".join(src.split())
    assert (
        "runpod_resources.grow_weight_cache_volumes("
        "spec,owning_key,deadline_at,wanted=cache_volumes)"
    ) in compact


def test_the_account_a_failover_lands_on_still_has_grow_budget(monkeypatch):
    """Regression: headroom for ONE grow is spent by the first attempt, starving every later one.

    Reserve a grow slice for each unreconciled failover account or it can attach a stale volume with
    only create allowance remaining.
    """
    import runpod_flash

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    clock = {"t": 10_000.0}
    monkeypatch.setattr(job_execution.time, "time", lambda: clock["t"])
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["t"])

    account = {"key": "acct-1"}
    # both are imported function-locally by deploy_train_endpoint, so patch them at the source
    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: account["key"])
    monkeypatch.setattr(rp_keys, "key_count", lambda: 2)
    monkeypatch.setattr(rp_keys, "advance_key", lambda: account.update(key="acct-2"))
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(runpod_resources, "_sweep_idle_flash_endpoints", lambda **kw: 0)
    monkeypatch.setattr(job_execution, "_is_balance_error", lambda exc: True)

    events = []

    def _grow(key, wanted, **kw):
        events.append(("grow", key))
        clock["t"] += runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S  # a real grow burns its budget
        return {}

    monkeypatch.setattr(runpod_resources.runpod_api, "grow_network_volumes_for_key", _grow)

    def _endpoint(**kwargs):
        events.append(("attach", account["key"]))
        raise RuntimeError("insufficient balance")

    monkeypatch.setattr(runpod_flash, "Endpoint", _endpoint)

    deadline = clock["t"] + 60 + runpod_resources.weight_cache_grow_headroom_s()
    with pytest.raises(RuntimeError, match="insufficient balance"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=None,
            endpoint_kwargs=dict,
            deadline_at=deadline,
            cache_volumes={"flash-weights-us-ca-2": 250},
        )

    # every account reconciled BEFORE it attached, not just the first
    grown = {k for kind, k in events if kind == "grow"}
    attached = [k for kind, k in events if kind == "attach"]
    assert attached == ["acct-1", "acct-2"]
    assert grown == {"acct-1", "acct-2"}


def test_a_slow_failed_create_cannot_spend_the_failover_grow_budget(monkeypatch):
    """Regression: a create that runs long drains the headroom the NEXT account's grow needs.

    Hold unreconciled accounts' slices back from creates, sweeps, and backoffs; failed creates have
    no smaller bound to size headroom against.
    """
    import runpod_flash

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers._lifecycle.net.deadline import CREATE_ALLOWANCE_S
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    clock = {"t": 10_000.0}
    monkeypatch.setattr(job_execution.time, "time", lambda: clock["t"])
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["t"])

    account = {"key": "acct-1"}
    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: account["key"])
    monkeypatch.setattr(rp_keys, "key_count", lambda: 2)
    monkeypatch.setattr(rp_keys, "advance_key", lambda: account.update(key="acct-2"))
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(runpod_resources, "_sweep_idle_flash_endpoints", lambda **kw: 0)
    monkeypatch.setattr(job_execution, "_is_balance_error", lambda exc: "balance" in str(exc))

    events = []

    def _grow(key, wanted, **kw):
        events.append(("grow", key))
        clock["t"] += runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S
        return {}

    monkeypatch.setattr(runpod_resources.runpod_api, "grow_network_volumes_for_key", _grow)

    # Sized so the failover attempt starts with exactly the create allowance left: enough for
    # require_create_allowance to pass, but `min(budget, remaining - allowance)` is then 0, so the
    # unreserved code proceeded to attach with no reconciliation at all.
    deadline = clock["t"] + 900.0
    burn = {
        "s": deadline
        - clock["t"]
        - runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S
        - CREATE_ALLOWANCE_S
    }

    def _endpoint(**kwargs):
        events.append(("attach", account["key"]))
        clock["t"] += burn["s"]  # a create that runs long before failing
        burn["s"] = 0.0
        raise RuntimeError("insufficient balance")

    monkeypatch.setattr(runpod_flash, "Endpoint", _endpoint)

    with pytest.raises(RuntimeError, match=r"balance|deadline"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=None,
            endpoint_kwargs=dict,
            deadline_at=deadline,
            cache_volumes={"flash-weights-us-ca-2": 250},
        )

    # The invariant, which holds whichever way the deadline falls: an account that reached the
    # create had reconciled first. Out of time is allowed to fail the deploy; it is not allowed to
    # quietly attach a stale volume, which is the failure this PR exists to prevent.
    attached = {k for kind, k in events if kind == "attach"}
    grown = {k for kind, k in events if kind == "grow"}
    assert attached <= grown, f"attached without reconciling: {sorted(attached - grown)}"


def test_a_volume_free_run_reserves_no_grow_time(monkeypatch):
    """Regression: a run that attaches no managed cache must not pay the reconciliation reserve.

    An oversized catalog model, or a spec carrying a custom volume, reconciles nothing --
    ``grow_weight_cache_volumes`` early-returns for both. Reserving for them shortened the deadline
    for a create that was never going to grow anything: with two accounts and 90s left, the create
    failed its allowance check against an effective 50s without ever reaching the provider.
    """
    import runpod_flash

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    clock = {"t": 10_000.0}
    monkeypatch.setattr(job_execution.time, "time", lambda: clock["t"])
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["t"])

    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: "acct-1")
    monkeypatch.setattr(rp_keys, "key_count", lambda: 2)
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)

    def _grow(key, wanted, **kw):  # pragma: no cover - must never run for a volume-free deploy
        raise AssertionError("a volume-free run must not reconcile anything")

    monkeypatch.setattr(runpod_resources.runpod_api, "grow_network_volumes_for_key", _grow)

    reached = []

    def _endpoint(**kwargs):
        reached.append("create")
        raise RuntimeError("reached the provider")

    monkeypatch.setattr(runpod_flash, "Endpoint", _endpoint)

    # exactly Codex's scenario: 2 accounts, 90s left, no managed cache attached
    with pytest.raises(RuntimeError, match=r"reached the provider"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=None,
            endpoint_kwargs=dict,
            deadline_at=clock["t"] + 90.0,
            cache_volumes=None,
        )

    assert reached == ["create"], "volume-free deploy was rejected before reaching the provider"


def test_a_quota_retry_does_not_re_grow_the_same_account(monkeypatch):
    """The headroom is scoped to the account pool, so each account may reconcile at most once.

    Quota retries on the same account must not spend the slice reserved for a later failover.
    """
    import runpod_flash

    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: "only-account")
    monkeypatch.setattr(rp_keys, "key_count", lambda: 1)
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(runpod_resources, "_sweep_idle_flash_endpoints", lambda **kw: 0)
    monkeypatch.setattr(job_execution, "_is_balance_error", lambda exc: False)
    monkeypatch.setattr(job_execution, "_is_workers_quota_error", lambda exc: True)
    monkeypatch.setattr(job_execution.time, "sleep", lambda *_a: None)

    grows = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: grows.append(key) or {},
    )

    def _endpoint(**kwargs):
        raise RuntimeError("no workers available")

    monkeypatch.setattr(runpod_flash, "Endpoint", _endpoint)

    with pytest.raises(RuntimeError, match="no workers available"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=None,
            endpoint_kwargs=dict,
            cache_volumes={"flash-weights-us-ca-2": 250},
        )

    # three quota attempts against one account, but only one grow
    assert grows == ["only-account"]


def test_the_grow_reserve_does_not_reject_a_launchable_deploy(monkeypatch):
    """Regression: the reserve is a spending cap, not an admission test.

    `submit_attempt` cannot add headroom to its wall deadline. Admission charges only this attempt's
    grow; the full pool reserve still limits later spending.
    """
    import runpod_flash

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    clock = {"t": 10_000.0}
    monkeypatch.setattr(job_execution.time, "time", lambda: clock["t"])
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["t"])

    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: "acct-1")
    monkeypatch.setattr(rp_keys, "key_count", lambda: 2)
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(
        runpod_resources.runpod_api, "grow_network_volumes_for_key", lambda key, wanted, **kw: {}
    )

    reached = []

    def _endpoint(**kwargs):
        reached.append("create")
        raise RuntimeError("reached the provider")

    monkeypatch.setattr(runpod_flash, "Endpoint", _endpoint)

    # exactly what submit_attempt passes: the managed cache attached, deadline handed over unpadded
    with pytest.raises(RuntimeError, match=r"reached the provider"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=_vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME),
            endpoint_kwargs=dict,
            deadline_at=clock["t"] + 90.0,
        )

    assert reached == ["create"], "a deploy holding the create allowance was rejected before it ran"


def test_the_grow_reserve_still_caps_what_a_create_may_spend(monkeypatch):
    """The other direction: admission is judged on the real deadline, spending is not.

    Creates still use `_create_deadline()` so they cannot consume a failover account's grow slice.
    """
    import runpod_flash

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    clock = {"t": 10_000.0}
    monkeypatch.setattr(job_execution.time, "time", lambda: clock["t"])
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["t"])

    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: "acct-1")
    monkeypatch.setattr(rp_keys, "key_count", lambda: 2)
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(
        runpod_resources.runpod_api, "grow_network_volumes_for_key", lambda key, wanted, **kw: {}
    )

    timeouts = []

    def _wait_for(coro, timeout=None):
        timeouts.append(timeout)
        coro.close()
        raise RuntimeError("reached the provider")

    monkeypatch.setattr(job_execution.asyncio, "wait_for", _wait_for)
    monkeypatch.setattr(
        runpod_flash,
        "Endpoint",
        lambda **kw: types.SimpleNamespace(_build_resource_config=dict),
    )

    with pytest.raises(RuntimeError, match=r"reached the provider"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=_vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME),
            endpoint_kwargs=dict,
            deadline_at=clock["t"] + 900.0,
        )

    # one account still unreconciled at create time, so its slice stays held back
    assert timeouts == [900.0 - runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S]


def test_the_post_grow_recheck_does_not_recharge_a_paid_slice(monkeypatch):
    """Regression: the re-check after the grow still deducted the slice the grow just spent.

    Once this attempt's account reconciles, release its slice before the pre-create allowance check.
    """
    import runpod_flash

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers._lifecycle.net.deadline import CREATE_ALLOWANCE_S
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    clock = {"t": 10_000.0}
    monkeypatch.setattr(job_execution.time, "time", lambda: clock["t"])
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["t"])

    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: "acct-1")
    monkeypatch.setattr(rp_keys, "key_count", lambda: 2)
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)

    def _grow(key, wanted, **kw):
        clock["t"] += runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S  # a real grow burns its budget
        return {}

    monkeypatch.setattr(runpod_resources.runpod_api, "grow_network_volumes_for_key", _grow)

    reached = []

    def _wait_for(coro, timeout=None):
        reached.append("create")
        coro.close()
        raise RuntimeError("reached the provider")

    monkeypatch.setattr(job_execution.asyncio, "wait_for", _wait_for)
    monkeypatch.setattr(
        runpod_flash,
        "Endpoint",
        lambda **kw: types.SimpleNamespace(_build_resource_config=dict),
    )

    # After the grow burns its slice, exactly the allowance plus half a slice remains: admission
    # passed, the grow ran, and only the stale re-deduction could reject the create from here.
    deadline = clock["t"] + runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S * 1.5 + CREATE_ALLOWANCE_S
    with pytest.raises(RuntimeError, match=r"reached the provider"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=_vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME),
            endpoint_kwargs=dict,
            deadline_at=deadline,
        )

    assert reached == ["create"], "the re-check re-deducted the grow slice the attempt already paid"


def test_admission_is_rejudged_with_the_key_the_attempt_lands_on(monkeypatch):
    """A concurrent deploy's advance_key() between admission and selection must not leak a slice.

    Re-check under the lock against the selected account; if it is unreconciled, fail closed rather
    than attach its stale volume with only create allowance.
    """
    import runpod_flash

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers._lifecycle.net.deadline import CREATE_ALLOWANCE_S
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    clock = {"t": 10_000.0}
    monkeypatch.setattr(job_execution.time, "time", lambda: clock["t"])
    monkeypatch.setattr(job_execution.time, "sleep", lambda s: None)
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["t"])

    monkeypatch.setattr(rp_keys, "key_count", lambda: 2)
    # The global pointer admission reads stays on acct-1 the whole time...
    monkeypatch.setattr(rp_keys, "active_key", lambda: "acct-1")

    # ...but selection under the lock lands on acct-1 first, then -- the race -- on acct-2.
    attempts = {"n": 0}

    def _ensure_auth():
        attempts["n"] += 1
        return "acct-1" if attempts["n"] == 1 else "acct-2"

    monkeypatch.setattr(rp_auth, "ensure_auth", _ensure_auth)
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(runpod_resources, "_sweep_idle_flash_endpoints", lambda **kw: 0)

    grown = []

    def _grow(key, wanted, **kw):
        grown.append(key)
        clock["t"] += runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S
        return {}

    monkeypatch.setattr(runpod_resources.runpod_api, "grow_network_volumes_for_key", _grow)

    reached = []

    def _endpoint(**kwargs):
        reached.append(attempts["n"])
        # attempt 1 hits quota so the loop retries into the raced selection
        raise RuntimeError(
            "GraphQL errors: Max workers across all endpoints must not exceed "
            "your workers quota (30)"
        )

    monkeypatch.setattr(runpod_flash, "Endpoint", _endpoint)

    # One slice plus the allowance: attempt 1's grow burns its slice, leaving exactly the
    # allowance. The retry's pre-lock check reads acct-1 (reconciled, slice released) and passes
    # -- only the under-lock re-check against acct-2 stands between it and a zero-budget grow.
    deadline = clock["t"] + runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S + CREATE_ALLOWANCE_S
    with pytest.raises(RuntimeError, match=r"deadline"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=_vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME),
            endpoint_kwargs=dict,
            deadline_at=deadline,
        )

    # the raced attempt was rejected before it could zero-budget-grow and attach acct-2's volume
    assert reached == [1], "an unreconciled account reached the create without grow budget"
    assert grown == ["acct-1"], "acct-2 must not have been grown with only the allowance left"


def test_a_large_key_pool_does_not_zero_the_create_timeout(monkeypatch):
    """Regression: the reserve must yield the create allowance, like the grow it is reserved for.

    A reserve larger than remaining time must still leave the allowance already proven at admission.
    """
    import runpod_flash

    from flash.providers._lifecycle.net import deadline as _deadline
    from flash.providers._lifecycle.net.deadline import CREATE_ALLOWANCE_S
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    clock = {"t": 10_000.0}
    monkeypatch.setattr(job_execution.time, "time", lambda: clock["t"])
    monkeypatch.setattr(_deadline.time, "time", lambda: clock["t"])

    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: "acct-1")
    # 8 accounts * 20s reserve = 160s, more than the 90s left
    monkeypatch.setattr(rp_keys, "key_count", lambda: 8)
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(
        runpod_resources.runpod_api, "grow_network_volumes_for_key", lambda key, wanted, **kw: {}
    )

    timeouts = []

    def _wait_for(coro, timeout=None):
        timeouts.append(timeout)
        coro.close()
        raise RuntimeError("reached the provider")

    monkeypatch.setattr(job_execution.asyncio, "wait_for", _wait_for)
    monkeypatch.setattr(
        runpod_flash,
        "Endpoint",
        lambda **kw: types.SimpleNamespace(_build_resource_config=dict),
    )

    with pytest.raises(RuntimeError, match=r"reached the provider"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=_vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME),
            endpoint_kwargs=dict,
            deadline_at=clock["t"] + 90.0,
        )

    assert timeouts == [CREATE_ALLOWANCE_S]


def test_grow_headroom_covers_every_account_in_the_pool(monkeypatch):
    """The caller cannot fund a whole deploy from a single grow budget: failover reconciles again."""
    from flash.providers.runpod.client import auth as rp_keys

    monkeypatch.setattr(rp_keys, "key_count", lambda: 3)
    assert (
        runpod_resources.weight_cache_grow_headroom_s()
        == runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S * 3
    )

    # an unconfigured pool still funds the one attempt that runs
    monkeypatch.setattr(rp_keys, "key_count", lambda: 0)
    assert (
        runpod_resources.weight_cache_grow_headroom_s()
        == runpod_resources.WEIGHT_CACHE_GROW_BUDGET_S
    )


def test_one_bad_volume_never_blocks_the_others(monkeypatch):
    """Volumes are independent: a PATCH that fails must not skip every later datacenter.

    Aborting the loop left later DCs' volumes unreconciled, so a run placed in one still hit
    "Disk quota exceeded" -- the failure this whole path exists to prevent.
    """
    from flash.providers.runpod.client import api as runpod_api

    listed = [
        {"name": "flash-weights-us-ca-2", "id": "v1", "size": 100},
        {"name": "flash-weights-eu-ro-1", "id": "v2", "size": 100},
        {"name": "flash-weights-us-tx-3", "id": "v3", "size": 100},
    ]

    def _req(key, url, method="GET", **kw):
        if method == "GET":
            return listed
        if url.endswith("/v2"):  # the middle one is concurrently deleted
            raise RuntimeError("404 volume not found")
        return {}

    monkeypatch.setattr(runpod_api._CLIENT, "request_with_retries_for_key", _req)

    grown = runpod_api.grow_network_volumes_for_key(
        "k",
        {
            "flash-weights-us-ca-2": 250,
            "flash-weights-eu-ro-1": 250,
            "flash-weights-us-tx-3": 250,
        },
    )

    # the failing volume is skipped; the one AFTER it still gets reconciled
    assert grown == {"flash-weights-us-ca-2": 250, "flash-weights-us-tx-3": 250}


def test_a_mixed_unhealthy_and_throttled_endpoint_still_gives_up(monkeypatch):
    """The gap between the two existing timers: neither fires, so nothing ever did.

    Mixed unhealthy and throttled workers reset both exclusive predicates and otherwise burn the
    full 5400-second paid timeout.
    """
    from flash.providers.artifacts import weight_cache as preload

    mixed = {"unhealthy": 1, "throttled": 2}
    assert preload._has_worker(mixed) is True  # starvation timer stays cleared
    assert preload._only_unhealthy_workers(mixed) is False  # broken-image timer stays cleared
    assert preload._throttled_workers(mixed) is True  # ...so this one has to fire

    clock = {"now": 1000.0}
    monkeypatch.setattr(preload.time, "time", lambda: clock["now"])
    monkeypatch.setattr(preload.time, "sleep", lambda _s: None)
    monkeypatch.setattr(preload, "_worker_counts", lambda *a, **kw: dict(mixed))

    def _status(*a, **kw):
        clock["now"] += preload._THROTTLED_GRACE_S + 1.0
        return {"status": "IN_QUEUE"}

    monkeypatch.setattr(preload.runpod_api, "job_status", _status)

    with pytest.raises(preload.NoCapacityError) as err:
        preload._poll_until_done("ep", "job", "fp", 5400, 0.0)
    assert "throttled" in str(err.value)


def test_a_usable_worker_keeps_the_throttled_timer_clear():
    """A throttled box next to a running one is contention, not a dead DC -- never give up on it."""
    from flash.providers.artifacts import weight_cache as preload

    assert preload._throttled_workers({"throttled": 3, "running": 1}) is False
    assert preload._throttled_workers({"throttled": 1, "initializing": 1}) is False
    assert preload._throttled_workers(None) is False
    assert preload._throttled_workers({}) is False


def test_failover_reconciles_the_account_it_lands_on(monkeypatch):
    """The account a quota failover moves to must have its OWN volume grown before the attach.

    Reconcile inside each attempt because a pre-sweep cannot know which account will succeed.
    Drive through `deploy_train_endpoint` so dropped `cache_volumes` passthrough fails the test.
    """
    from flash.providers.runpod.client import auth as rp_auth
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.execution import job_execution

    account = {"key": "first-account"}
    # deploy_train_endpoint imports ensure_auth function-locally, so patch it at the source
    monkeypatch.setattr(rp_auth, "ensure_auth", lambda: account["key"])
    monkeypatch.setattr(rp_keys, "key_count", lambda: 2)
    monkeypatch.setattr(rp_keys, "advance_key", lambda: account.update(key="second-account"))
    monkeypatch.setattr(job_execution.runpod_api, "key_fingerprint", lambda k: f"fp-{k}")
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(runpod_resources, "_sweep_idle_flash_endpoints", lambda **kw: 0)
    monkeypatch.setattr(job_execution, "_is_balance_error", lambda exc: True)

    grown = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: grown.append(key) or {},
    )

    def _endpoint(**kwargs):
        raise RuntimeError("insufficient balance")

    # deploy_train_endpoint imports Endpoint function-locally from runpod_flash, so patch it there
    import runpod_flash

    monkeypatch.setattr(runpod_flash, "Endpoint", _endpoint)

    with pytest.raises(RuntimeError, match="insufficient balance"):
        job_execution.deploy_train_endpoint(
            "RTX 4090",
            spec=None,
            endpoint_kwargs=dict,
            cache_volumes={"flash-weights-us-ca-2": 250},
        )

    # both attempts reconciled the account THEY were about to attach, not just the first
    assert grown == ["first-account", "second-account"]


def test_datacenter_discovery_failure_never_fails_the_deploy(monkeypatch):
    """Regression: discovery sat OUTSIDE the best-effort boundary and could abort the deploy.

    Datacenter discovery failures must be swallowed like growth failures; the helper promises it
    cannot fail a deploy.
    """

    def _boom():
        raise RuntimeError("incompatible SDK: DataCenter.all() is gone")

    monkeypatch.setattr(runpod_resources, "weight_cache_datacenters", _boom)

    # must return, not raise
    runpod_resources.grow_weight_cache_volumes(
        _vol_spec(runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME), "k"
    )


def test_spec_none_without_named_volumes_stays_a_no_op(monkeypatch):
    """Callers that neither carry a spec nor name volumes have nothing to reconcile."""

    grown = []
    monkeypatch.setattr(
        runpod_resources.runpod_api,
        "grow_network_volumes_for_key",
        lambda key, wanted, **kw: grown.append(key) or {},
    )

    runpod_resources.grow_weight_cache_volumes(None, "k", None)

    assert grown == []


def test_the_owning_key_is_read_inside_the_serialized_section():
    """Another thread can advance_key() while this one waits for FLASH_SDK_LOCK.

    A key captured before the wait would grow one account's volume while Endpoint attaches the
    account the env var now names -- leaving the attached volume stale.
    """
    import inspect

    from flash.providers.runpod.execution import job_execution

    src = inspect.getsource(job_execution.deploy_train_endpoint)
    # the key is read under FLASH_SDK_LOCK and that same value is what the grow is charged to, so
    # a key another thread advances after admission cannot grow one account while Endpoint attaches
    # a different one.
    assert "owning_key = ensure_auth()" in src
    compact = "".join(src.split())
    assert "runpod_resources.grow_weight_cache_volumes(spec,owning_key" in compact
