"""The run's HF artifact repo is platform-managed, not a user field.

``submit_job`` assigns it server-side — a per-run private dataset under the operator's
namespace — after the run_id is finalized, overwriting any inbound value. This is what
prevents the 403: the operator HF_TOKEN (which the control plane uploads/writes with)
cannot create a dataset under a user-chosen namespace like ``freesolo-founders``.
"""

from __future__ import annotations

import os
import tempfile

from flash.spec import JobSpec


def _spec(**train) -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "seeds": [0], **train},
            "run_id": "flash-managed-1",
        }
    )


def _submit(monkeypatch, spec: JobSpec, platform_context: dict | None = None) -> dict:
    from flash import runner

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        kw = {"platform_context": platform_context} if platform_context is not None else {}
        return runner.submit_job(spec, dry_run=True, **kw).spec


def test_managed_hf_repo_assigned_per_run(monkeypatch):
    spec = _submit(monkeypatch, _spec())
    assert spec["train"]["hf_repo"] == "Freesolo-Co/flashrun-flash-managed-1"


def test_managed_hf_repo_overrides_user_value(monkeypatch):
    # Even if a legacy/old-client spec carries a user namespace, the control plane overrides it.
    spec = _submit(monkeypatch, _spec(hf_repo="freesolo-founders/whatever"))
    assert spec["train"]["hf_repo"] == "Freesolo-Co/flashrun-flash-managed-1"


def test_managed_hf_repo_finalizes_local_run_id(monkeypatch):
    # The JobSpec default run_id "local" is treated as unset: submit_job assigns a real run_id and
    # a matching per-run repo, so default-constructed/programmatic specs never collide on
    # "flashrun-local". (Regression guard for the run_id-finalization review fix.)
    from flash.spec import JobSpec

    base = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "train": {"epochs": 1, "seeds": [0]},
        }
    )
    assert base.run_id == "local"
    spec = _submit(monkeypatch, base)
    assert spec["run_id"] != "local"
    assert spec["train"]["hf_repo"] == f"Freesolo-Co/flashrun-{spec['run_id']}"


# --- Per-org persistent kernel-cache volume (platform-managed; issue B / #258) -----------------
# submit_job attaches a network volume keyed per ORG and stable across that org's runs, so the
# worker's model-weight download AND fused-kernel JIT compile (the ~10-15 min cold-start cost)
# become one-time-per-org instead of per cold worker. Default ON; gated by FLASH_KERNEL_CACHE_*.


def _no_cache_env(monkeypatch):
    """Default-on baseline: clear any operator overrides so the feature's defaults apply."""
    for k in ("FLASH_KERNEL_CACHE_VOLUME", "FLASH_KERNEL_CACHE_DATACENTER", "FLASH_KERNEL_CACHE_VOLUME_GB"):
        monkeypatch.delenv(k, raising=False)


def test_kernel_cache_volume_assigned_per_org(monkeypatch):
    _no_cache_env(monkeypatch)
    gpu = _submit(monkeypatch, _spec(), platform_context={"org_id": "acme"})["gpu"]
    assert gpu["network_volume"] == "flash-kernel-cache-acme"
    assert gpu["datacenter"] == "EU-RO-1"  # default DC pin (volume must co-locate with the worker)
    assert gpu["network_volume_gb"] == 100


def test_kernel_cache_volume_stable_across_runs_same_org(monkeypatch):
    """The warm-cache invariant: two runs for the same org get the SAME volume name (else the
    second run never hits the first's compiled kernels)."""
    _no_cache_env(monkeypatch)
    a = _submit(monkeypatch, _spec(), platform_context={"org_id": "acme"})["gpu"]["network_volume"]
    b = _submit(monkeypatch, _spec(), platform_context={"org_id": "acme"})["gpu"]["network_volume"]
    assert a == b == "flash-kernel-cache-acme"
    # ...and different orgs get different volumes (no cross-org cache sharing).
    c = _submit(monkeypatch, _spec(), platform_context={"org_id": "globex"})["gpu"]["network_volume"]
    assert c == "flash-kernel-cache-globex"


def test_kernel_cache_volume_skipped_without_org(monkeypatch):
    """No org identity (programmatic/test caller) -> no volume: a single-use volume would pay the
    datacenter pin for zero reuse. Falls back to the cold/heartbeat path."""
    _no_cache_env(monkeypatch)
    gpu = _submit(monkeypatch, _spec())["gpu"]  # no platform_context
    assert gpu["network_volume"] is None


def test_kernel_cache_volume_falls_back_to_user_id(monkeypatch):
    _no_cache_env(monkeypatch)
    gpu = _submit(monkeypatch, _spec(), platform_context={"user_id": "u-42"})["gpu"]
    assert gpu["network_volume"] == "flash-kernel-cache-u-42"


def test_kernel_cache_volume_disabled_by_env(monkeypatch):
    monkeypatch.setenv("FLASH_KERNEL_CACHE_VOLUME", "0")
    gpu = _submit(monkeypatch, _spec(), platform_context={"org_id": "acme"})["gpu"]
    assert gpu["network_volume"] is None  # kill switch: every run back to cold/heartbeat path


def test_kernel_cache_volume_sanitizes_org_id(monkeypatch):
    """Org ids become a valid RunPod volume name (lowercase, [a-z0-9-], collapsed/trimmed)."""
    _no_cache_env(monkeypatch)
    gpu = _submit(monkeypatch, _spec(), platform_context={"org_id": "  Acme Corp!! "})["gpu"]
    assert gpu["network_volume"] == "flash-kernel-cache-acme-corp"


def test_kernel_cache_volume_env_overrides_dc_and_size(monkeypatch):
    _no_cache_env(monkeypatch)
    monkeypatch.setenv("FLASH_KERNEL_CACHE_DATACENTER", "US-CA-2")
    monkeypatch.setenv("FLASH_KERNEL_CACHE_VOLUME_GB", "250")
    gpu = _submit(monkeypatch, _spec(), platform_context={"org_id": "acme"})["gpu"]
    assert gpu["datacenter"] == "US-CA-2"
    assert gpu["network_volume_gb"] == 250


def test_kernel_cache_volume_rejects_out_of_range_size(monkeypatch):
    """A fat-fingered size falls back to the default instead of crashing at provision (the RunPod
    SDK requires 10 <= size <= 4096)."""
    _no_cache_env(monkeypatch)
    for bad in ("0", "-5", "999999", "notanint"):
        monkeypatch.setenv("FLASH_KERNEL_CACHE_VOLUME_GB", bad)
        gpu = _submit(monkeypatch, _spec(), platform_context={"org_id": "acme"})["gpu"]
        assert gpu["network_volume_gb"] == 100, bad


def test_assigned_volume_redirects_worker_caches(monkeypatch):
    """End-to-end: org -> assigned volume -> build_worker_env redirects the kernel caches onto it.
    This is the whole chain that turns the cold-start compile into a one-time-per-org cost."""
    _no_cache_env(monkeypatch)
    from flash.providers.runpod.train import build_worker_env

    spec_dict = _submit(monkeypatch, _spec(), platform_context={"org_id": "acme"})
    spec = JobSpec.from_dict(spec_dict)
    env = build_worker_env(spec, 0)
    assert env["HF_HOME"] == "/runpod-volume/hf-cache"
    assert env["TRITON_CACHE_DIR"] == "/runpod-volume/kernel-cache/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/runpod-volume/kernel-cache/inductor"
