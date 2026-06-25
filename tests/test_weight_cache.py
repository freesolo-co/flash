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
def test_weight_cache_datacenters_is_all_storage_dcs():
    from runpod_flash.core.resources.datacenter import DataCenter

    from flash.providers.runpod import jobs

    dcs = jobs.weight_cache_datacenters()
    assert set(dcs) == set(DataCenter.all())  # exactly the SDK's storage-capable DC set
    assert len(set(dcs)) == len(dcs)  # all distinct


def test_weight_cache_datacenters_ignores_removed_env_knob(monkeypatch):
    # Regression: the FLASH_WEIGHT_CACHE_DATACENTERS knob is GONE — the fleet is fixed/managed.
    from runpod_flash.core.resources.datacenter import DataCenter

    from flash.providers.runpod import jobs

    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DATACENTERS", "US-CA-2")
    assert len(jobs.weight_cache_datacenters()) == len(DataCenter.all())  # env ignored


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
def test_weight_cache_env_is_hf_only():
    from flash.providers.runpod.train.deps import weight_cache_env

    env = weight_cache_env("/runpod-volume")
    # ONLY HF_HOME (inert model blobs). The executable kernel-JIT caches must NOT be redirected onto
    # the shared multi-tenant volume — a poisoned compiled artifact could affect another tenant.
    assert env == {"HF_HOME": "/runpod-volume/hf-cache"}
    for k in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TILELANG_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        assert k not in env


def test_weight_cache_env_custom_mount():
    from flash.providers.runpod.train.deps import weight_cache_env

    assert weight_cache_env("/workspace")["HF_HOME"] == "/workspace/hf-cache"


def test_build_worker_env_redirects_hf_home_with_volume():
    from flash.providers.runpod.train.deps import build_worker_env

    env = build_worker_env(_vol_spec(), 0)
    assert env["HF_HOME"] == "/runpod-volume/hf-cache"


def test_build_worker_env_no_redirect_without_volume():
    from flash.providers.runpod.train.deps import build_worker_env

    env = build_worker_env(JobSpec(model="m"), 0)
    # Without a volume HF_HOME must NOT be set (pointing at a missing mount).
    assert "HF_HOME" not in env


def test_build_worker_env_per_run_override_wins():
    from flash.providers.runpod.train.deps import build_worker_env

    spec = _vol_spec()
    spec = JobSpec.from_dict({**spec.to_dict(), "worker_env": {"HF_HOME": "/custom/hf"}})
    # a per-run [worker_env] override is merged last and must win over the cache redirect.
    assert build_worker_env(spec, 0)["HF_HOME"] == "/custom/hf"


def test_drop_unmounted_cache_env_strips_when_unmounted(monkeypatch):
    import flash.providers.runpod.train.deps as deps

    # volume NOT mounted -> the /runpod-volume HF_HOME is stripped (cold fallback), others kept.
    monkeypatch.setattr(deps.os.path, "isdir", lambda p: False)
    env = {"HF_HOME": "/runpod-volume/hf-cache", "OTHER": "x"}
    out = deps.drop_unmounted_cache_env(env)
    assert "HF_HOME" not in out
    assert out["OTHER"] == "x"


def test_drop_unmounted_cache_env_keeps_when_mounted(monkeypatch):
    import flash.providers.runpod.train.deps as deps

    monkeypatch.setattr(deps.os.path, "isdir", lambda p: True)
    env = {"HF_HOME": "/runpod-volume/hf-cache"}
    assert deps.drop_unmounted_cache_env(env) == {"HF_HOME": "/runpod-volume/hf-cache"}


# ---------------------------------------------------------------------------
# instance-provider integration (Lambda/Hyperstack reuse RunPod's build_worker_env)
# ---------------------------------------------------------------------------
def test_strip_runpod_volume_env_removes_only_mount_rooted_vars():
    from flash.providers.runpod.train.deps import strip_runpod_volume_env

    env = {"HF_HOME": "/runpod-volume/hf-cache", "X": "/runpod-volume/foo", "KEEP": "v", "HF_TOKEN": "t"}
    out = strip_runpod_volume_env(env)
    assert "HF_HOME" not in out
    assert "X" not in out
    assert out == {"KEEP": "v", "HF_TOKEN": "t"}  # non-/runpod-volume vars preserved


def test_instance_payload_strips_runpod_volume_redirect():
    # The RunPod weight-cache HF_HOME redirect must NOT leak into a Lambda/Hyperstack payload — those
    # instances never mount /runpod-volume. (build_worker_env DOES set it; the instance path strips.)
    from flash.providers import _instance
    from flash.providers.runpod.train.deps import build_worker_env

    spec = JobSpec.from_dict({**_vol_spec().to_dict(), "run_id": "r", "model": "Qwen/Qwen3.5-0.8B"})
    assert build_worker_env(spec, 0)["HF_HOME"].startswith("/runpod-volume")  # the leak source
    for arm in ("lambda", "hyperstack"):
        env = _instance.build_payload(spec, seed=0, attempt=0, arm=arm)["env"]
        assert not env.get("HF_HOME", "").startswith("/runpod-volume"), arm


# ---------------------------------------------------------------------------
# runner._assign_weight_cache_volume — fully managed, attaches to EVERY run, no knobs/gating
# ---------------------------------------------------------------------------
def test_assign_weight_cache_attaches_to_catalog_run():
    from flash import runner

    out = runner._assign_weight_cache_volume(JobSpec(model="m", run_id="r", model_policy="catalog"))
    assert out.gpu.network_volume == runner.WEIGHT_CACHE_VOLUME_NAME == "flash-weights"
    assert out.gpu.network_volume_gb == runner.WEIGHT_CACHE_VOLUME_GB == 100


def test_assign_weight_cache_attaches_for_allow_policy_too():
    # Regression: the private-model gating is REMOVED — every run gets the cache, allow-policy too.
    from flash import runner

    out = runner._assign_weight_cache_volume(JobSpec(model="m", run_id="r", model_policy="allow"))
    assert out.gpu.network_volume == "flash-weights"


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
                "train": {"epochs": 1, "seeds": [0]},
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
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="wc-walk",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
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
def test_catalog_model_ids_are_the_catalog():
    from flash.catalog import MODELS
    from flash.providers.runpod import preload

    assert set(preload.catalog_model_ids()) == set(MODELS)


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

    def fake_snapshot(repo_id, token=None, cache_dir=None, local_files_only=False):
        calls.append({"repo": repo_id, "cache_dir": cache_dir, "probe": local_files_only})
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


def test_teardown_hyperstack_volumes_deletes_only_fleet(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.runpod import preload

    vols = [
        {"id": 11, "name": "flash-weights", "environment": {"name": "default-CANADA-1"}},
        {"id": 12, "name": "flash-weights", "environment": {"name": "default-US-1"}},
        {"id": 13, "name": "user-volume", "environment": {"name": "default-US-1"}},  # NOT ours
    ]
    deleted = []
    monkeypatch.setattr(hs_api, "list_volumes", lambda: vols)
    monkeypatch.setattr(hs_api, "delete_volume", lambda i: deleted.append(i) or True)

    out = preload.teardown_hyperstack_volumes()
    assert sorted(deleted) == [11, 12]
    assert sorted(out) == ["hyperstack:default-CANADA-1/flash-weights", "hyperstack:default-US-1/flash-weights"]


def test_teardown_hyperstack_volumes_no_key_is_noop(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.runpod import preload

    monkeypatch.setattr(
        hs_api, "list_volumes",
        lambda: (_ for _ in ()).throw(hs_api.HyperstackApiError("HYPERSTACK_API_KEY not set")),
    )
    assert preload.teardown_hyperstack_volumes() == []


def test_teardown_cli_reclaims_all_three_providers(monkeypatch):
    """`preload --teardown` sweeps RunPod + Lambda + Hyperstack in one shot."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "teardown_weight_cache", lambda dcs: ["flash-weights-us-ca-2"])
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", lambda: ["lambda:us-east-1/flash-weights"])
    monkeypatch.setattr(preload, "teardown_hyperstack_volumes", lambda: ["hyperstack:default-US-1/flash-weights"])
    assert preload.main(["--teardown"]) == 0


def test_teardown_continues_when_runpod_unconfigured(monkeypatch):
    """A RunPod teardown raise (auth absent / outage) must NOT abort Lambda + Hyperstack cleanup."""
    from flash.providers.runpod import preload

    def _boom(dcs):
        raise RuntimeError("RUNPOD_API_KEY not configured")

    lam, hs = [], []
    monkeypatch.setattr(preload, "teardown_weight_cache", _boom)
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", lambda: lam.append(1) or ["lambda:us-east-1/flash-weights"])
    monkeypatch.setattr(preload, "teardown_hyperstack_volumes", lambda: hs.append(1) or ["hyperstack:default-US-1/flash-weights"])
    # RunPod raises but the instance providers still get cleaned up best-effort; the CLI still exits 0.
    assert preload.main(["--teardown"]) == 0
    assert lam == [1]
    assert hs == [1]


def test_scoped_teardown_is_runpod_only(monkeypatch):
    """`--teardown --datacenters ...` scopes to RunPod; instance-provider caches are left intact."""
    from flash.providers.runpod import preload

    seen = {}
    monkeypatch.setattr(preload, "teardown_weight_cache", lambda dcs: seen.setdefault("dcs", dcs) or ["flash-weights-us-ca-2"])
    monkeypatch.setattr(preload, "teardown_lambda_filesystems", lambda: seen.setdefault("lambda", True) or [])
    monkeypatch.setattr(preload, "teardown_hyperstack_volumes", lambda: seen.setdefault("hyperstack", True) or [])
    assert preload.main(["--teardown", "--datacenters", "US-CA-2"]) == 0
    assert seen["dcs"] == ["US-CA-2"]  # the RunPod scope was honored
    assert "lambda" not in seen  # instance providers were NOT touched
    assert "hyperstack" not in seen


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


def test_provision_hyperstack_volumes_per_region_unique_names(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.runpod import preload

    ensured = []
    monkeypatch.setattr(hs_api, "_regions", lambda: ["CANADA-1", "US-1", "NORWAY-1"])
    monkeypatch.setattr(hs_api, "environment_for_region", lambda r: f"default-{r}")
    monkeypatch.setattr(
        hs_api, "ensure_volume",
        lambda name, env, gb: ensured.append((name, env, gb)) or 1,
    )
    out = preload.provision_hyperstack_volumes()
    # DISTINCT per-region name (Hyperstack enforces global name uniqueness) in each region's env
    assert ensured == [
        ("flash-weights-canada-1", "default-CANADA-1", 100),
        ("flash-weights-us-1", "default-US-1", 100),
        ("flash-weights-norway-1", "default-NORWAY-1", 100),
    ]
    assert out == ["hyperstack:CANADA-1", "hyperstack:US-1", "hyperstack:NORWAY-1"]


def test_provision_hyperstack_distinct_name_even_in_shared_env(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.runpod import preload

    ensured = []
    # two regions mapping to the SAME env still get DISTINCT (globally-unique) volume names
    monkeypatch.setattr(hs_api, "_regions", lambda: ["US-1", "US-2"])
    monkeypatch.setattr(hs_api, "environment_for_region", lambda r: "shared-env")
    monkeypatch.setattr(hs_api, "ensure_volume", lambda name, env, gb: ensured.append(name) or 1)
    out = preload.provision_hyperstack_volumes()
    assert ensured == ["flash-weights-us-1", "flash-weights-us-2"]
    assert out == ["hyperstack:US-1", "hyperstack:US-2"]


def test_provision_cli_creates_instance_storage(monkeypatch):
    """`preload --provision` creates Lambda + Hyperstack storage (GPU-free) and exits 0."""
    from flash.providers.runpod import preload

    monkeypatch.setattr(preload, "provision_lambda_filesystems", lambda: ["lambda:us-east-1"])
    monkeypatch.setattr(preload, "provision_hyperstack_volumes", lambda: ["hyperstack:default-US-1"])
    assert preload.main(["--provision"]) == 0


def test_provision_cli_dry_run_provisions_nothing(monkeypatch):
    from flash.providers.runpod import preload

    called = {"n": 0}
    monkeypatch.setattr(preload, "provision_lambda_filesystems", lambda: called.__setitem__("n", called["n"] + 1) or [])
    monkeypatch.setattr(preload, "provision_hyperstack_volumes", lambda: called.__setitem__("n", called["n"] + 1) or [])
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
