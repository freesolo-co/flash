"""Hyperstack run lifecycle: cloud-init/bootstrap, region/stock walk, poll state machine,
guaranteed delete, orphan sweep, capacity-aware allocation (CPU-only; hyperstack API + HF readers
mocked). Mirrors test_lambda_runner.py — the two providers share the instance-based shape."""

from __future__ import annotations

import base64
import io
import itertools
import json

import pytest

from flash.spec import JobSpec


def _spec(gpu_type="L40", **gpu_kw) -> JobSpec:
    gpu = {"type": gpu_type, "max_wall_seconds": 3600, **gpu_kw}
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": "flash-1700000000-abcd1234",
            "train": {"epochs": 1, "seeds": [0], "hf_repo": "org/repo"},
            "gpu": gpu,
        }
    )


def _inst(gpu="L40", region="CANADA-1", flavor="n3-L40x1", price=1.00):
    from flash.providers.hyperstack.jobs.builders import HyperstackInstance

    return HyperstackInstance(
        gpu=gpu, flavor=flavor, region=region, environment=f"default-{region}", vram_gb=48, price_usd_hr=price
    )


def _handle(started_ts=10_000.0, rate=1.00, attempt=0):
    from flash.providers.hyperstack.jobs.builders import HyperstackJobHandle

    return HyperstackJobHandle(
        vm_id="vm-9999", flavor="n3-L40x1", region="CANADA-1", name="flash-x-s0-a0",
        gpu="L40", hourly_usd=rate, attempt=attempt, started_ts=started_ts,
    )


# ---------------------------------------------------------------------------
# cloud-init user_data + payload (shared builder, arm='hyperstack')
# ---------------------------------------------------------------------------
def test_user_data_ships_payload_and_runs_worker_image(monkeypatch):
    from flash.providers.hyperstack.jobs import builders

    monkeypatch.setenv("HYPERSTACK_API_KEY", "hk-supersecret")
    monkeypatch.setenv("HF_TOKEN", "hf-worker-token")
    payload = builders.build_payload(_spec(), seed=0, attempt=1)
    assert payload["phase"] == "sft"
    assert payload["flash_arm"] == "hyperstack"
    assert payload["hf_prefix"] == "sft/flash-1700000000-abcd1234/seed0"
    assert payload["env"]["HF_REPO"] == "org/repo"

    script = builders.build_user_data(payload)
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    assert json.loads(base64.b64decode(b64)) == payload
    assert "FLASH_BOOTSTRAP_EOF" in script
    from flash.providers.runpod.train import WORKER_IMAGE

    assert WORKER_IMAGE in script
    assert "docker run -d" in script
    assert "--gpus all" in script
    assert "/root/flash/bootstrap.py" in script
    assert "waiting for docker+gpu" in script
    # operator key never ships; the worker HF token is inside the base64 payload, not raw shell
    assert "hk-supersecret" not in script
    assert payload["env"]["HF_TOKEN"] == "hf-worker-token"


def test_image_per_sm_opt_in_selects_arch_tag(monkeypatch):
    """Opt-in per-SM warmed images (PR #213) reach Hyperstack too: with FLASH_WORKER_IMAGE_PER_SM
    set, the GPU class picks the matching -smXX tag. Default + FLASH_WORKER_IMAGE override unchanged.
    NB: this is the worker *container* image, distinct from the VM *boot* image (docker_image_for_region)."""
    from flash.providers.hyperstack.jobs import builders
    from flash.providers.runpod.train import WORKER_IMAGE

    for key in ("FLASH_WORKER_IMAGE", "FLASH_WORKER_IMAGE_PER_SM", "FLASH_WORKER_IMAGE_TEMPLATE"):
        monkeypatch.delenv(key, raising=False)

    # default: flat base image, byte-identical to pre-PR behavior
    assert builders.hyperstack_image() == WORKER_IMAGE
    assert builders.hyperstack_image("L40") == WORKER_IMAGE

    # per-SM opt-in: the GPU class appends the arch tag, and it lands in the cloud-init
    monkeypatch.setenv("FLASH_WORKER_IMAGE_PER_SM", "1")
    assert builders.hyperstack_image("L40") == f"{WORKER_IMAGE}-sm89"  # L40 = sm89
    assert builders.hyperstack_image("H100") == f"{WORKER_IMAGE}-sm90"  # H100 = sm90
    payload = builders.build_payload(_spec(gpu_type="L40"), seed=0, attempt=0)
    script = builders.build_user_data(payload, gpu="L40")
    assert f"{WORKER_IMAGE}-sm89" in script

    # absolute override still wins, even with per-SM enabled and a GPU class given
    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/freesolo-co/flash-worker:hotfix")
    assert builders.hyperstack_image("L40") == "ghcr.io/freesolo-co/flash-worker:hotfix"


# ---------------------------------------------------------------------------
# launch_and_submit: region/stock walk
# ---------------------------------------------------------------------------
def test_launch_walks_regions_on_rejection(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "flash-managed")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "Ubuntu Docker CUDA 12.8")
    attempts = []

    def fake_launch(*, name, environment_name, image_name, flavor_name, key_name, user_data):
        attempts.append(environment_name)
        if len(attempts) < 2:
            raise hs_api.HyperstackApiError("POST /core/virtual-machines -> HTTP 400: no stock")
        return "vm-4242"

    monkeypatch.setattr(hs_api, "launch_vm", fake_launch)
    insts = [_inst(region=r) for r in ("CANADA-1", "US-1")]
    h = jobs.launch_and_submit(_spec(), seed=0, instances=insts, attempt=2)
    assert attempts == ["default-CANADA-1", "default-US-1"]
    assert h.vm_id == "vm-4242"
    assert h.region == "US-1"
    assert h.name == "flash-1700000000-abcd1234-s0-a2"


def test_launch_relaxes_to_quarantined_region_when_healthy_exhausted(monkeypatch):
    """If the initial healthy region rejects and the forced refresh finds NO healthy stock, the walk
    relaxes the quarantine (ignore_sick) and launches into a quarantined-but-in-stock region rather than
    failing the attempt while sick stock exists -- relax after the healthy walk is exhausted, not only on
    an initially-empty healthy list. Mirrors Lambda."""
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "flash-managed")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "Ubuntu Docker CUDA 12.8")
    created = []

    def fake_launch(*, name, environment_name, image_name, flavor_name, key_name, user_data):
        if environment_name != "default-SICK-1":
            raise hs_api.HyperstackApiError("POST /core/virtual-machines -> HTTP 400: no stock")
        created.append(environment_name)
        return "vm-99"

    def fake_usable(gpu, force=False, ignore_sick=False):
        # Healthy refresh empty; only the relaxed (quarantine-ignoring) refresh finds stock.
        return [_inst(region="SICK-1")] if ignore_sick else []

    monkeypatch.setattr(hs_api, "launch_vm", fake_launch)
    monkeypatch.setattr(jobs, "usable_instances", fake_usable)
    h = jobs.launch_and_submit(_spec(), seed=0, instances=[_inst(region="US-1")], attempt=0)
    assert created == ["default-SICK-1"]
    assert h.vm_id == "vm-99"


def test_launch_raises_when_no_stock(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "k")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "img")
    monkeypatch.setattr(
        hs_api, "launch_vm", lambda **k: (_ for _ in ()).throw(hs_api.HyperstackApiError("POST /core/virtual-machines -> HTTP 400: no stock"))
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False, ignore_sick=False: [])
    with pytest.raises(hs_api.HyperstackApiError, match="no stock"):
        jobs.launch_and_submit(_spec(), seed=0, instances=[_inst()], attempt=0)
    with pytest.raises(hs_api.HyperstackApiError, match="no Hyperstack stock"):
        jobs.launch_and_submit(_spec(), seed=0, instances=[], attempt=0)


def test_regions_excludes_canada1_by_default(monkeypatch):
    """CANADA-1 (known broken-driver on-demand fleet) is dropped from the region list by default, so
    the allocator + launcher never offer or boot there. The API still returns it; flash filters it."""
    from flash.providers.hyperstack import api as hs_api

    monkeypatch.setattr(
        hs_api,
        "request_with_retries",
        lambda *a, **k: {"regions": [{"name": "NORWAY-1"}, {"name": "CANADA-1"}, {"name": "US-1"}]},
    )
    monkeypatch.delenv("HYPERSTACK_BLOCKED_REGIONS", raising=False)
    regions = hs_api._regions()
    assert "CANADA-1" not in regions
    assert regions == ["NORWAY-1", "US-1"]


def test_regions_blocklist_is_env_overridable(monkeypatch):
    """HYPERSTACK_BLOCKED_REGIONS overrides the default: set to "" re-enables CANADA-1 (operator
    opt-in once the fleet recovers); set to another region blocks that instead."""
    from flash.providers.hyperstack import api as hs_api

    monkeypatch.setattr(
        hs_api,
        "request_with_retries",
        lambda *a, **k: {"regions": [{"name": "NORWAY-1"}, {"name": "CANADA-1"}, {"name": "US-1"}]},
    )
    monkeypatch.setenv("HYPERSTACK_BLOCKED_REGIONS", "")  # explicit empty -> nothing blocked
    assert "CANADA-1" in hs_api._regions()
    monkeypatch.setenv("HYPERSTACK_BLOCKED_REGIONS", "norway-1, us-1")  # case-insensitive
    assert hs_api._regions() == ["CANADA-1"]


def test_throwaway_keypair_missing_ssh_keygen_is_clear_error(monkeypatch):
    """A slim control plane without ssh-keygen must raise an actionable HyperstackApiError (install
    openssh-client / pin HYPERSTACK_KEYPAIR_NAME), not a bare FileNotFoundError that reads like an
    API/stock failure."""
    import subprocess

    from flash.providers.hyperstack import api as hs_api

    def _no_keygen(*a, **k):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'ssh-keygen'")

    monkeypatch.setattr(subprocess, "run", _no_keygen)
    with pytest.raises(hs_api.HyperstackApiError, match=r"openssh-client|HYPERSTACK_KEYPAIR_NAME"):
        hs_api._generate_throwaway_public_key()


def test_launch_skips_region_with_no_boot_image_without_reconciling(monkeypatch):
    """A pre-launch resolution failure (region has stock but no qualifying CUDA image / key) created
    NO VM, so it's a CLEAN region SKIP — walk to the next region, never an ambiguous-phantom abort
    that reconciles + stops the whole attempt (which would needlessly waste an otherwise-good walk)."""
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    # CANADA-1 has stock but no qualifying image (pre-launch raise); US-1 resolves + launches fine.
    def fake_image(region, min_cuda="12.8"):
        if region == "CANADA-1":
            raise hs_api.HyperstackApiError(f"no Docker image in {region} with CUDA >= {min_cuda}")
        return "Ubuntu 24.04 CUDA 13.0 with Docker"

    reconciled = []
    monkeypatch.setattr(hs_api, "docker_image_for_region", fake_image)
    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "k")
    monkeypatch.setattr(hs_api, "launch_vm", lambda **kw: "vm-img")
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda rid: reconciled.append(rid))

    insts = [_inst(region=r) for r in ("CANADA-1", "US-1")]
    h = jobs.launch_and_submit(_spec(), seed=0, instances=insts, attempt=0)
    assert h.vm_id == "vm-img"  # walked PAST CANADA-1's missing image to US-1
    assert h.region == "US-1"
    assert reconciled == []  # NEVER reconciled: a pre-launch failure leaves no phantom VM


# ---------------------------------------------------------------------------
# launch_and_submit: per-region weight cache (Hyperstack block volume)
# ---------------------------------------------------------------------------
def _wire_cache_launch(monkeypatch):
    """Common wiring: image+key resolve, launch records user_data, returns the api module + recorders."""
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "k")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "img")
    launched = []

    def fake_launch(*, name, environment_name, image_name, flavor_name, key_name, user_data):
        launched.append({"env": environment_name, "user_data": user_data})
        return "vm-cache"

    monkeypatch.setattr(hs_api, "launch_vm", fake_launch)
    return hs_api, jobs, launched


def test_cache_ensures_volume_and_attaches_after_launch(monkeypatch):
    hs_api, jobs, launched = _wire_cache_launch(monkeypatch)
    ensured, attached = [], []
    monkeypatch.setattr(hs_api, "ensure_volume", lambda n, env, gb: ensured.append((n, env, gb)) or "vol-7")
    monkeypatch.setattr(hs_api, "attach_volume", lambda vm, vol: attached.append((vm, vol)))

    jobs.launch_and_submit(_spec(network_volume="flash-weights"), seed=0, instances=[_inst()], attempt=0)

    # per-region physical name (Hyperstack volume names are globally unique), created in the env
    assert ensured == [("flash-weights-canada-1", "default-CANADA-1", 100)]
    assert attached == [("vm-cache", "vol-7")]  # attached AFTER launch (block volume can't attach at create)
    ud = launched[0]["user_data"]
    assert "-v '/mnt/flash-weights':/weight-cache" in ud  # bind into the worker (quoted host path)
    assert "blkid" in ud  # format-if-new preamble (guard)
    assert "mkfs.ext4" in ud  # format-if-new preamble (format)


def test_cache_preamble_never_reformats_a_populated_volume(monkeypatch):
    """The block-device preamble guards mkfs behind blkid so a populated cache is never wiped."""
    from flash.providers.hyperstack.jobs import build_payload, build_user_data

    ud = build_user_data(
        build_payload(_spec(network_volume="flash-weights"), 0, 0,
                      cache_host_mount="/mnt/flash-weights", cache_block_device=True)
    )
    # mkfs runs ONLY when blkid finds no filesystem (the `||` short-circuit) — never unconditionally.
    assert 'blkid "$CACHE_DEV" >/dev/null 2>&1 || mkfs.ext4' in ud
    assert "mount \"$CACHE_DEV\" '/mnt/flash-weights'" in ud  # quoted host mount
    # Device is size-matched (never blindly the first unmounted disk) and skips disks with a mounted
    # partition, so the boot disk is never reformatted.
    assert "EXPECT_BYTES=" in ud
    assert "lsblk -pnr -o MOUNTPOINT" in ud


def test_cache_falls_back_cold_when_ensure_fails(monkeypatch):
    hs_api, jobs, launched = _wire_cache_launch(monkeypatch)
    attached = []
    monkeypatch.setattr(hs_api, "ensure_volume", lambda n, env, gb: (_ for _ in ()).throw(RuntimeError("quota")))
    monkeypatch.setattr(hs_api, "attach_volume", lambda vm, vol: attached.append((vm, vol)))

    jobs.launch_and_submit(_spec(network_volume="flash-weights"), seed=0, instances=[_inst()], attempt=0)
    assert attached == []  # nothing attached
    assert "/weight-cache" not in launched[0]["user_data"]  # cold user_data, no bind


def test_cache_falls_back_cold_when_volume_has_no_id(monkeypatch):
    """ensure_volume returning a falsy id (creation returned no id) must launch cold, not a cache
    user_data that waits forever for a device that never attaches."""
    hs_api, jobs, launched = _wire_cache_launch(monkeypatch)
    attached = []
    monkeypatch.setattr(hs_api, "ensure_volume", lambda n, env, gb: None)
    monkeypatch.setattr(hs_api, "attach_volume", lambda vm, vol: attached.append((vm, vol)))

    jobs.launch_and_submit(_spec(network_volume="flash-weights"), seed=0, instances=[_inst()], attempt=0)
    assert attached == []
    assert "/weight-cache" not in launched[0]["user_data"]


def test_no_cache_never_touches_volumes(monkeypatch):
    hs_api, jobs, launched = _wire_cache_launch(monkeypatch)
    monkeypatch.setattr(hs_api, "ensure_volume", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no ensure")))
    monkeypatch.setattr(hs_api, "attach_volume", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no attach")))
    jobs.launch_and_submit(_spec(), seed=0, instances=[_inst()], attempt=0)  # no network_volume
    assert "/weight-cache" not in launched[0]["user_data"]


def test_preload_mode_skips_region_when_cache_unavailable(monkeypatch):
    """In preload mode a cache-ensure failure SKIPS the region — never a cold full-training launch.

    Regression: the cold user_data carries no mode/models, so cold-fallback for a preload would boot
    a full training run (GPU billing, timeout) and warm nothing. The walk must skip and fail if no
    region can host the cache.
    """
    import pytest

    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "k")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "img")
    monkeypatch.setattr(hs_api, "ensure_volume", lambda n, env, gb: (_ for _ in ()).throw(RuntimeError("quota")))
    launched = []
    monkeypatch.setattr(hs_api, "launch_vm", lambda **kw: launched.append(kw) or "vm")

    insts = [_inst(region="CANADA-1"), _inst(region="NORWAY-1")]
    with pytest.raises(hs_api.HyperstackApiError):
        jobs.launch_and_submit(
            _spec(network_volume="flash-weights"), seed=0, instances=insts, attempt=0,
            mode="preload", models=["a/b"],
        )
    assert launched == []  # no region ever launched a cold (training) VM


def test_preload_mode_does_not_refresh_to_a_different_region(monkeypatch):
    """In preload mode the walk must NOT refresh to a NEW region on a target-region miss.

    Regression: warm_instances pins each preload launch to one TARGET region and reports that exact
    region as warmed. If the walk refreshed (usable_instances) to a different region and launched
    there, the caller would report the cold target region as warmed. The walk must stay confined to
    the given candidate(s) and FAIL when none can host the cache.
    """
    import pytest

    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "k")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "img")
    # target region's cache is unavailable
    monkeypatch.setattr(hs_api, "ensure_volume", lambda n, env, gb: (_ for _ in ()).throw(RuntimeError("quota")))
    launched = []
    monkeypatch.setattr(hs_api, "launch_vm", lambda **kw: launched.append(kw) or "vm")
    # the refresh source offers a DIFFERENT region with a working cache — it must NOT be consulted
    refresh_calls = []
    monkeypatch.setattr(
        jobs, "usable_instances",
        lambda gpu, force=False, ignore_sick=False: refresh_calls.append(force) or [_inst(region="ELSEWHERE-9")],
    )

    with pytest.raises(hs_api.HyperstackApiError):
        jobs.launch_and_submit(
            _spec(network_volume="flash-weights"), seed=0, instances=[_inst(region="CANADA-1")],
            attempt=0, mode="preload", models=["a/b"],
        )
    assert launched == []  # never launched anywhere (not in the refreshed region)
    assert refresh_calls == []  # the stale-stock refresh was NOT consulted in preload mode


def test_preload_mode_tears_down_when_attach_fails(monkeypatch):
    """In preload mode a FAILED volume attach (vol busy on another VM / API error) must tear the just-
    launched box down and walk on — not leave it billing while it waits for an absent device.

    Regression: attach_volume() returns False on failure and the call ignored it, so a preload box
    launched, couldn't warm (the sentinel check refuses ephemeral disk), and burned GPU to the wall cap.
    A training run still tolerates a failed attach (degrades cold); only preload hard-fails the region.
    """
    import pytest

    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "k")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "img")
    monkeypatch.setattr(hs_api, "ensure_volume", lambda n, env, gb: "vol-busy")
    launched, terminated = [], []
    monkeypatch.setattr(hs_api, "launch_vm", lambda **kw: launched.append(kw) or "vm-x")
    monkeypatch.setattr(hs_api, "attach_volume", lambda vm, vol: False)  # attach FAILS
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda rid: terminated.append(rid))

    with pytest.raises(hs_api.HyperstackApiError):
        jobs.launch_and_submit(
            _spec(network_volume="flash-weights"), seed=0, instances=[_inst(region="CANADA-1")],
            attempt=0, mode="preload", models=["a/b"],
        )
    assert len(launched) == 1  # the box DID launch (attach is post-launch)...
    assert terminated  # ...and was torn down when the attach failed (not left billing)


def test_training_mode_tolerates_failed_attach(monkeypatch):
    """A TRAINING run survives a failed attach: the cloud-init preamble degrades to a cold run, so the
    box keeps running (no teardown) — only preload is strict about the cache."""
    hs_api, jobs, _launched = _wire_cache_launch(monkeypatch)
    terminated = []
    monkeypatch.setattr(hs_api, "ensure_volume", lambda n, env, gb: "vol-7")
    monkeypatch.setattr(hs_api, "attach_volume", lambda vm, vol: False)  # attach fails
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda rid: terminated.append(rid))
    # mode defaults to None (training) -> returns a handle, no teardown
    h = jobs.launch_and_submit(_spec(network_volume="flash-weights"), seed=0, instances=[_inst()], attempt=0)
    assert h is not None
    assert terminated == []


# ---------------------------------------------------------------------------
# poll_hs_job state machine
# ---------------------------------------------------------------------------
def _wire_poll(monkeypatch, vms, done=None, marker=None, metrics=None, boot=None, error=None, step=10.0, legacy_boot=None):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    seq = iter(vms)
    last = {"vm": None}

    def fake_get(vm_id):
        last["vm"] = next(seq, last["vm"])
        return last["vm"]

    monkeypatch.setattr(hs_api, "get_vm", fake_get)
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=10_000, step=step)
    monkeypatch.setattr(jobs.time, "time", lambda: float(next(clock)))

    def factory(hf_repo, path, min_interval_s=45.0):
        def read(force=False):
            if path.endswith("/DONE"):
                return done() if callable(done) else done
            if "hyperstack_attempt" in path and path.endswith(".json"):  # not the _boot.log
                return marker() if callable(marker) else marker
            if path.endswith("metrics.json"):
                return metrics() if callable(metrics) else metrics
            if path.endswith("_boot.log"):
                if "attempt" in path:  # attempt-scoped: hyperstack_attempt<N>_boot.log
                    return boot() if callable(boot) else boot
                return legacy_boot() if callable(legacy_boot) else legacy_boot  # legacy hyperstack_boot.log
            if "/error_" in path:  # worker crash traceback (error_<phase>.txt)
                return error() if callable(error) else error
            return None

        return read

    monkeypatch.setattr(jobs, "_make_hf_file_reader", factory)
    return jobs


def test_poll_success_stamps_real_cost(monkeypatch):
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}], done="10500.0",
        metrics=json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0}),
    )
    # started_ts precedes the mocked clock (starts 10_000) so wall is positive on the first tick.
    res = jobs.poll_hs_job(_handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["cost_usd"] > 0
    assert res.metrics["notes"]["provider"] == "hyperstack"
    assert res.metrics["notes"]["hyperstack_region"] == "CANADA-1"


def test_poll_missing_started_ts_anchors_to_now_not_epoch(monkeypatch):
    """started_ts coerced to 0.0 (MISSING / corrupt handle) means 'unknown launch'. EVERYTHING (the
    timeout clocks AND done_is_fresh / finish_ok's wall+cost stamping) must anchor to now, NOT the
    1970 epoch — otherwise a booting VM stalls instantly and wall/cost are billed from 1970. DONE
    completes the run normally with a sane (tiny) wall."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}], done="10500.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}), step=10.0,
    )
    res = jobs.poll_hs_job(_handle(started_ts=0.0), _spec(), seed=0, interval_s=0)
    assert res.ok, res  # not instantly stalled by an epoch-anchored clock
    assert res.metrics["cost_usd"] < 1.0, res.metrics["cost_usd"]  # not billed from 1970


def test_poll_marker_failure_is_job_failed(monkeypatch):
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RuntimeError: boom"}),
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"
    assert "boom" in res.detail


def test_poll_retriable_marker_is_job_preempted(monkeypatch):
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "transient"}),
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {"retriable": True, "ts": 10_000.0},  # fresh (ts >= launch)
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert res.host_fault  # fresh retriable crash before any training heartbeat -> sick region


def test_poll_marker_stale_retriable_heartbeat_does_not_quarantine(monkeypatch):
    """On a RETRY, worker_flagged_retriable() can read a PRIOR attempt's RetriableInfraError heartbeat
    (seed-scoped, ts < this attempt's launch). THIS attempt's non-retriable setup failure then still
    retries but must NOT quarantine the healthy region: fresh_retriable_hb() rejects the stale heartbeat
    for the host_fault signal. Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps({"ok": False, "attempt": 1, "error": "bootstrap pip failed"}),  # non-retriable
    )
    res = jobs.poll_hs_job(
        _handle(started_ts=10_000.0, attempt=1), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {"retriable": True, "ts": 5_000.0},  # STALE: ts < launch 10_000
    )
    assert not res.ok
    assert not res.host_fault  # stale prior-attempt retriable heartbeat did not quarantine the region


def test_poll_dead_vm_cancelled_run_not_quarantined(monkeypatch):
    """A user cancel during setup deletes the VM; the poller sees a dead host with no training
    heartbeat. run_cancelled() suppresses host_fault so a HEALTHY region is not quarantined for a
    deliberate teardown. Mirrors the Lambda path."""
    import flash.runner

    monkeypatch.setattr(
        flash.runner, "get_status", lambda run_id: type("S", (), {"state": "cancelled"})()
    )
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        boot="+ docker pull ... (still in setup when cancelled)",
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"
    assert not res.host_fault  # cancelled -> region NOT quarantined


def test_poll_init_stage_heartbeat_quarantines_on_retriable(monkeypatch):
    """rl_initializing/sft_initializing are PRE-training (trainer/vLLM init), so a retriable infra
    fault during init must STILL quarantine the region: reached_training_now() must treat the init
    heartbeat as setup, not training. Regression for the *_initializing stages missing from
    _SETUP_HEARTBEAT_STAGES. Mirrors the Lambda path (uses the rl_ init stage here)."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RetriableInfraError: vLLM init OOM", "retriable": True}),
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {"stage": "rl_initializing", "step": 0, "ts": 10_000.0},
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert res.host_fault  # init-time infra fault is pre-training -> region quarantined


def test_poll_model_prefetched_heartbeat_quarantines_on_retriable(monkeypatch):
    """``model_prefetched`` is emitted by prefetch_model() right after sft_start/rl_start while pulling
    weights -- pre-training cold start, NOT a training step. A retriable infra/GPU fault after prefetch
    but before the first step must STILL quarantine the region: reached_training_now() must treat it as
    setup. Regression: model_prefetched was missing from SETUP_HEARTBEAT_STAGES, so is_training_stage()
    returned True and host_fault was wrongly suppressed. Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RetriableInfraError: cuda init failed", "retriable": True}),
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {"stage": "model_prefetched", "step": 0, "ts": 10_000.0},
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert res.host_fault  # prefetch is pre-training -> region quarantined


def test_poll_midtraining_retriable_marker_does_not_quarantine(monkeypatch):
    """A retriable failure marker can land in the SAME poll iteration the worker first reaches training
    -- the marker branch decides host_fault BEFORE surface_heartbeat() advances seen_training_hb.
    reached_training_now() force-reads the heartbeat and sees the training stage, so a mid-training
    RetriableInfraError retries (job_preempted) WITHOUT quarantining the healthy region."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RetriableInfraError: gpu fell off the bus", "retriable": True}),
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0,
        # sft_step is the worker's real training-step stage -> is_training_stage() True only because
        # it is a genuine step (not just any non-setup string), so this actually guards the classification.
        heartbeat_reader=lambda force=False: {"stage": "sft_step", "step": 7, "ts": 10_000.0},
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert not res.host_fault  # training already reached -> region healthy, do NOT quarantine


def test_poll_trained_completion_upload_retry_not_quarantined(monkeypatch):
    """A retriable marker flagged ``trained`` (the worker REACHED end-of-training but the required
    DONE/metrics UPLOAD failed) is a post-training completion retry on a HEALTHY region. It must retry
    (job_preempted) WITHOUT quarantining the region even on a fast run where no fresh training heartbeat
    was latched and the latest forced hb is the worker's error_* stage. Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps(
            {"ok": False, "attempt": 0, "retriable": True, "trained": True, "error": "DONE upload failed"}
        ),
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {"stage": "error_sft", "ts": 10_000.0},  # no training hb latched
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # retriable upload failure -> retry on a fresh host
    assert not res.host_fault  # trained: post-training completion retry on a healthy region -> NO quarantine


def test_poll_host_neutral_marker_not_quarantined(monkeypatch):
    """A bootstrap-raised retriable marker flagged ``host_neutral`` (the pre-worker spilled-spec HF
    fetch) is a region-INDEPENDENT delivery failure, so it retries (job_preempted) WITHOUT quarantining
    the region even pre-training (no fresh heartbeat). Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps(
            {"ok": False, "attempt": 0, "retriable": True, "host_neutral": True,
             "error": "failed to fetch the spilled job spec from HF"}
        ),
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"  # retriable -> retry on a fresh host
    assert not res.host_fault  # host_neutral: HF-delivery failure, not a sick region -> NO quarantine


def test_poll_host_failmark_still_quarantines(monkeypatch):
    """The HOST cloud-init failmark (docker/GPU never ready) sets retriable=True but OMITS host_neutral
    (it is written by the host, not the in-container bootstrap), so it must STILL quarantine the region
    -- guarding the host_neutral suppression from disabling real broken-region failover (the CANADA-1
    case). Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps(
            {"ok": False, "attempt": 0, "retriable": True, "error": "host: docker run failed"}
        ),
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"  # retriable host fault -> retry on a fresh host
    assert res.host_fault  # docker/GPU never ready, no host_neutral -> region quarantined


def test_poll_marker_path_cancel_requested_not_quarantined(monkeypatch):
    """fail_from_marker must ALSO honor run_cancelled(): a cancel-in-progress (state still 'running',
    cancel_requested true) that lands via a just-written / stale attempt failure marker must NOT
    quarantine a healthy region for our own teardown -- mirroring the dead-VM and first-liveness paths."""
    import flash.runner

    monkeypatch.setattr(
        flash.runner, "get_status",
        lambda run_id: type("S", (), {"state": "running", "cancel_requested": True})(),
    )
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps(
            {"ok": False, "attempt": 0, "retriable": True, "error": "host: docker run failed"}
        ),
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"  # retriable marker -> retry
    assert not res.host_fault  # cancel-in-progress -> NOT quarantined even via the marker path


def test_poll_github_ratelimit_heartbeat_not_quarantined(monkeypatch):
    """A pre-training GitHubRateLimitError makes the worker stamp a FRESH error heartbeat flagged
    retriable + host_neutral; the region-independent fault must retry (job_preempted) WITHOUT
    quarantining a healthy region (fresh_host_neutral_hb suppresses host_fault). Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        marker=json.dumps(
            {"ok": False, "attempt": 0, "retriable": False, "error": "no /tmp/metrics.json (crashed)"}
        ),
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {
            "stage": "error_rl", "ts": 10_000.0, "retriable": True, "host_neutral": True
        },
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # retriable hb -> retry on a fresh host
    assert not res.host_fault  # host_neutral hb: global rate limit, not a sick region -> NO quarantine


def test_poll_dead_vm_neutral_heartbeat_without_marker_not_quarantined(monkeypatch):
    """Same region-INDEPENDENT GitHubRateLimitError, but the attempt marker never landed (its upload
    failed, or the VM was lost before write) so we hit the DEAD-VM branch instead of fail_from_marker.
    That branch must apply the same fresh_host_neutral_hb() guard the marker path does: retry the global
    rate limit (job_preempted via the retriable hb) WITHOUT quarantining an otherwise-healthy region.
    Pre-fix this branch set host_fault unconditionally on a pre-training loss -> false quarantine."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        # NO marker -> falls through terminal_artifact_result() to the dead-VM branch
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {
            "stage": "error_rl", "ts": 10_000.0, "retriable": True, "host_neutral": True
        },
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # retriable hb -> retry on a fresh host
    assert not res.host_fault  # host_neutral hb on the dead-VM path: NO quarantine of a healthy region


def test_poll_dead_vm_cancel_requested_not_quarantined(monkeypatch):
    """cancel_run sets cancel_requested=True BEFORE it deletes the VM and persists the terminal
    'cancelled' state. A poll landing in that teardown window (state still 'running') must honor the
    intent and NOT quarantine a healthy region for our own cancellation."""
    import flash.runner

    monkeypatch.setattr(
        flash.runner, "get_status",
        lambda run_id: type("S", (), {"state": "running", "cancel_requested": True})(),
    )
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        boot="+ docker pull ... (cancel in progress)",
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert not res.host_fault  # cancel-in-progress -> region NOT quarantined


def test_poll_dead_vm_trained_heartbeat_without_marker_not_quarantined(monkeypatch):
    """A post-training required upload fails retriably and BOTH the worker's attempt marker and the VM
    are lost, leaving only a fresh error_* heartbeat flagged trained=True. The dead-VM path must read
    that heartbeat 'trained' flag (the marker is unavailable here) and NOT quarantine a HEALTHY region
    that actually finished training -- mirroring the marker.trained guard."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0,
        heartbeat_reader=lambda force=False: {
            "stage": "error_rl", "ts": 10_000.0, "retriable": True, "trained": True
        },
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # retriable hb -> retry
    assert not res.host_fault  # trained hb: finished training on a healthy region -> NO quarantine


def test_poll_reattach_just_active_floored_by_observed_grace(monkeypatch):
    """On a reattach whose first poll already sees the VM active, active_since is launch-anchored, so
    the launch-relative first_liveness deadline is already blown. The observed-grace floor stops a VM
    that only JUST became active (after a long provision the control plane missed) from being failed
    over before its boot-log uploader's publication window: even with the boot.log absent past
    BOOT_LOG_ABSENT_POLLS and the deadline exceeded, no 'no worker liveness' stall fires until we've
    watched it active for FIRST_LIVENESS_OBSERVED_GRACE_S. Here the VM dies (host loss) inside that
    window -> job_preempted, not the premature liveness stall (which the pre-fix code would return)."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}] * 4 + [{"status": "ERROR"}],
        boot=None,  # uploader has not published yet
        step=0.1,  # tiny steps so the 120s observed-grace floor is NOT reached in these few polls
    )
    # Launch 1_000s ago (clock starts 10_000): the first_liveness deadline (10s) is blown, but the
    # 3_000s setup grace is NOT (else its launch-anchored stall would fire first and mask the floor).
    res = jobs.poll_hs_job(
        _handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0, first_liveness_s=10.0
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # died inside the observed-grace window
    assert "no worker liveness" not in (res.detail or "")


def test_poll_dead_vm_without_marker_is_preempted(monkeypatch):
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        boot="+ docker pull ...\nFLASH: gpu never became ready",
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"
    assert "gpu never became ready" in res.detail


def test_poll_dead_vm_with_error_file_is_job_failed_not_quarantined(monkeypatch):
    """A worker that RAN and crashed early (left error_<phase>.txt) but died before writing the
    attempt marker is a DETERMINISTIC worker-CODE error -> fail fast (job_failed), NOT a host loss.
    Crucially it must NOT set host_fault: the region is healthy, so quarantining it (over-quarantine)
    would needlessly evict a good region for a user-code bug. Mirrors the Lambda dead-host path."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        error="Traceback (most recent call last):\nFileNotFoundError: environment archive did not contain ...",
    )
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0, heartbeat_reader=lambda force=False: {}
    )
    assert not res.ok
    assert res.failure == "job_failed"  # deterministic worker error, fail fast
    assert "environment archive" in res.detail
    assert not res.host_fault  # worker-code crash != region fault -> region NOT quarantined


def test_poll_dead_vm_with_retriable_error_still_preempted(monkeypatch):
    """Even WITH an error_<phase>.txt, a crash the worker flagged retriable (RetriableInfraError,
    stamped in the heartbeat) retries on a fresh host (job_preempted) and DOES quarantine the region
    (a pre-training infra fault). Keeps a stale prior-attempt error file from flipping a genuine
    preemption to job_failed. Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        error="Traceback ...\nRetriableInfraError: cuda device not ready",
    )
    res = jobs.poll_hs_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        # FRESH retriable heartbeat (ts >= launch): this attempt's worker flagged the fault retriable.
        heartbeat_reader=lambda force=False: {"retriable": True, "ts": 10_000.0},
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert res.host_fault  # fresh retriable infra crash before any training heartbeat -> quarantine


def test_poll_dead_vm_stale_retriable_hb_does_not_mask_crash(monkeypatch):
    """A STALE prior-attempt retriable heartbeat (ts < launch) must NOT suppress THIS attempt's
    deterministic crash. The dead branch gates ``worker_crashed`` on fresh_retriable_hb() (launch-fresh),
    NOT bare worker_flagged_retriable(), so a non-retriable error_<phase>.txt this attempt is classified
    terminal job_failed and the healthy region is NOT quarantined. Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        error="Traceback ...\nValueError: bad config",  # deterministic, non-retriable
    )
    res = jobs.poll_hs_job(
        _handle(),  # attempt 0
        _spec(),
        seed=0,
        interval_s=0,
        # retriable BUT stale (ts < launch 10_000): a prior attempt's lingering signal.
        heartbeat_reader=lambda force=False: {"retriable": True, "ts": 5_000.0},
    )
    assert not res.ok
    assert res.failure == "job_failed"  # crash trusted, not masked by the stale retriable hb
    assert not res.host_fault  # deterministic worker crash -> region stays healthy


def test_poll_dead_vm_error_stage_hb_quarantines_pretraining_fault(monkeypatch):
    """An ``error_<phase>`` heartbeat stage is the worker's crash marker -- pre-training, NOT a training
    step. is_training_stage() excludes error_* (and the *_initializing setup stages), so reached_training_now()
    is False and a FRESH retriable infra fault during init still quarantines the region (host_fault). If
    error_* were misread as a training stage the quarantine would be wrongly suppressed. Mirrors Lambda."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        error="Traceback ...\nRetriableInfraError: cuda device not ready",
    )
    res = jobs.poll_hs_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        # fresh, retriable, crash-stage heartbeat -> pre-training infra fault.
        heartbeat_reader=lambda force=False: {"stage": "error_sft", "retriable": True, "ts": 10_000.0},
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # fresh retriable -> retry on a fresh host
    assert res.host_fault  # error_* is pre-training -> quarantine the region


def test_poll_dead_vm_stale_error_with_fresh_boot_heartbeat_is_preempted(monkeypatch):
    """error_<phase>.txt is SEED-scoped, so a retriable PRIOR attempt's traceback can linger on HF. On
    a RETRY (attempt > 0) that posts a boot heartbeat (THIS attempt is still booting) then loses the VM,
    the loss must NOT be classified terminal job_failed off the stale file: fresh_error_heartbeat() is
    false for a setup-stage heartbeat, so without positive current-crash evidence the loss stays
    retriable (job_preempted). Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        error="Traceback ...\nValueError: prior-attempt crash",  # STALE, left by a retriable prior attempt
    )
    res = jobs.poll_hs_job(
        _handle(attempt=1), _spec(), seed=0, interval_s=0,  # a RETRY: a stale prior-attempt file is possible
        heartbeat_reader=lambda force=False: {"stage": "boot", "step": 0, "ts": 10_000.0},
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # stale error file did not flip a host loss to terminal


def test_poll_dead_vm_first_attempt_error_with_setup_heartbeat_is_job_failed(monkeypatch):
    """The inverse guard: on the FIRST attempt (attempt 0) no prior attempt exists, so a present
    error_<phase>.txt is unambiguously THIS attempt's crash. A worker that wrote the error file then
    died in setup BEFORE stamping its terminal error heartbeat (last heartbeat still a setup stage)
    must still fail fast as job_failed -- the stale-file guard must not misread a genuine current crash
    as retriable. Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        error="Traceback ...\nValueError: bad config -> deterministic crash this attempt",
    )
    res = jobs.poll_hs_job(
        _handle(attempt=0), _spec(), seed=0, interval_s=0,  # FIRST attempt: error file can't be stale
        heartbeat_reader=lambda force=False: {"stage": "sft_model_load", "step": 0, "ts": 10_000.0},
    )
    assert not res.ok
    assert res.failure == "job_failed"  # current-attempt crash trusted despite the setup-stage heartbeat


def test_poll_dead_vm_retry_stale_error_no_current_liveness_is_preempted(monkeypatch):
    """On a RETRY (attempt > 0) the seed-scoped error_<phase>.txt may be a PRIOR attempt's, and THIS
    attempt's VM can be lost before producing ANY fresh heartbeat. With no current-attempt liveness,
    there is no proof the worker crashed this attempt, so the loss is a HOST LOSS -> job_preempted, NOT
    a wrongful terminal job_failed off the stale file (over-retry is the documented safe bias on
    retries). Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        error="Traceback ...\nValueError: prior-attempt crash",  # STALE, left by a prior attempt
    )
    res = jobs.poll_hs_job(
        _handle(attempt=1), _spec(), seed=0, interval_s=0,  # a RETRY
        heartbeat_reader=lambda force=False: {},  # NO fresh heartbeat this attempt (cold VM loss)
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # host loss, not the stale file's deterministic crash
    assert res.host_fault  # pre-training host loss with no cancel -> region quarantined


def test_poll_active_no_liveness_fails_over_fast(monkeypatch):
    """Hyperstack CANADA-1 (or any HS region) equivalent of the Lambda sick-region case: a VM that
    reaches ACTIVE but never starts a worker (no boot.log/heartbeat/marker) fails over fast as a
    retriable 'stalled' instead of burning the full ~50 min setup grace."""
    jobs = _wire_poll(monkeypatch, vms=[{"status": "ACTIVE"}], step=100.0)
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=500.0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "no worker liveness" in res.detail
    assert res.host_fault  # region quarantined by submit_run_hyperstack on this result


def test_poll_active_boot_log_protects_slow_cold_start(monkeypatch):
    """A healthy VM still pulling the image emits the boot.log but no heartbeat yet — the
    first-liveness deadline must NOT fire."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ACTIVE"}, {"status": "ERROR"}],
        boot="+ docker pull ... (still pulling)",
        step=100.0,
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"
    assert "no worker liveness" not in (res.detail or "")


def test_poll_legacy_boot_log_satisfies_first_liveness_on_reattach(monkeypatch):
    """Mixed-version reattach: a VM launched by the PREVIOUS user-data wrote the legacy non-attempt-
    scoped hyperstack_boot.log while the attempt-scoped path is absent. The first-liveness gate must
    accept the legacy log as liveness and NOT fast-fail/quarantine a healthy, still-booting old VM
    during the upgrade drain window. Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ACTIVE"}, {"status": "ERROR"}],
        boot=None,  # attempt-scoped boot.log absent
        legacy_boot="+ docker pull ... (old-CP VM, legacy boot.log)",  # legacy path present
        step=100.0,
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"  # died as a host loss, NOT killed by the liveness deadline
    assert "no worker liveness" not in (res.detail or "")  # legacy boot.log satisfied first-liveness


def test_poll_legacy_boot_log_honored_on_attempt1_reattach(monkeypatch):
    """A mixed-version reattach can be polling a PRE-upgrade attempt 1+ VM whose old user-data wrote only
    the legacy boot.log. The legacy fallback must be honored on ANY attempt (gating it to attempt 0 would
    falsely quarantine the healthy old retry); a stale prior-attempt legacy log can't mask a relaunch
    because recovery purges it. Mirrors the Lambda path."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ACTIVE"}, {"status": "ERROR"}],
        boot=None,  # attempt-scoped boot.log absent (old VM wrote only the legacy path)
        legacy_boot="+ docker pull ... (old-CP attempt-1 VM, legacy boot.log)",
        step=100.0,
    )
    res = jobs.poll_hs_job(_handle(attempt=1), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert not res.ok
    assert res.failure == "job_preempted"  # died as a host loss, NOT killed by the liveness deadline
    assert "no worker liveness" not in (res.detail or "")  # legacy boot.log honored on attempt 1


def test_poll_active_empty_boot_log_counts_as_liveness(monkeypatch):
    """An empty ("") boot.log still proves cloud-init ran — its EXISTENCE is liveness. A bare
    ``not boot_log_reader()`` would treat "" as absent and spuriously fail the VM over; the fix uses
    ``is None``. VM later dies -> job_preempted, NOT the 'no worker liveness' stall."""
    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ACTIVE"}, {"status": "ERROR"}],
        boot="",
        step=100.0,
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"
    assert "no worker liveness" not in (res.detail or "")


def test_poll_active_boot_log_seen_once_survives_rate_limited_none(monkeypatch):
    """Regression: make_hf_text_reader returns None for BOTH a missing boot.log AND a rate-limited
    read, so a bare ``not boot_log_reader()`` re-checked each poll would spuriously stall a HEALTHY VM
    on the first throttled read after the log was already seen. The boot.log is read with force=True and
    latched once observed, so a later None can't re-trigger failover."""
    calls = {"n": 0}

    def boot_then_rate_limited():
        calls["n"] += 1
        return "+ docker pull ..." if calls["n"] == 1 else None  # seen once, then "rate-limited"

    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ACTIVE"}, {"status": "ACTIVE"}, {"status": "ERROR"}],
        boot=boot_then_rate_limited,
        step=100.0,
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"  # NOT a spurious 'stalled' from the throttled None
    assert "no worker liveness" not in (res.detail or "")
    # Latched after the first observation: the liveness check reads the boot.log once (not once per
    # poll); the only other read is the terminal-failure-detail surfacer when the VM dies.
    assert calls["n"] <= 2


def test_poll_active_transient_boot_log_error_does_not_fail_over(monkeypatch):
    """make_hf_text_reader returns None for a MISSING boot.log AND a momentary HF/Hub network error,
    so a lone forced-read None at the first-liveness deadline must NOT immediately stall — a transient
    blip clears on the next poll (the absence must persist BOOT_LOG_ABSENT_POLLS times to fail over).
    Here the first forced read errors (None), the next returns the real boot.log -> latched, no
    failover; the VM later dies -> job_preempted, NOT a spurious 'stalled' from the one transient
    None."""
    calls = {"n": 0}

    def transient_then_present():
        calls["n"] += 1
        return None if calls["n"] == 1 else "+ docker pull ..."  # transient error first, then readable

    jobs = _wire_poll(
        monkeypatch,
        vms=[{"status": "ACTIVE"}, {"status": "ACTIVE"}, {"status": "ERROR"}],
        boot=transient_then_present,
        step=100.0,
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "job_preempted"  # the single transient None did not trip a failover
    assert "no worker liveness" not in (res.detail or "")


def test_poll_active_persistent_boot_log_absence_stalls_after_threshold(monkeypatch):
    """The genuine sick-region case: the boot.log is absent on EVERY forced read (cloud-init never
    ran). After BOOT_LOG_ABSENT_POLLS consecutive absent reads the first-liveness check declares the
    region 'stalled' (retriable, escaped cross-provider). Asserts the absence-count threshold is what
    gates the failover, not a single read."""
    from flash.providers._poll import BOOT_LOG_ABSENT_POLLS

    calls = {"n": 0}

    def always_absent():
        calls["n"] += 1  # implicit None: every forced read comes back absent

    jobs = _wire_poll(monkeypatch, vms=[{"status": "ACTIVE"}], boot=always_absent, step=100.0)
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "stalled"
    assert "no worker liveness" in res.detail
    assert calls["n"] >= BOOT_LOG_ABSENT_POLLS  # required the absence to persist, not a lone None


def test_poll_active_done_marker_wins_over_first_liveness_stall(monkeypatch):
    """The boot.log uploader is best-effort and can silently never run even though the worker itself
    ran and uploaded a terminal DONE. The boot.log-absence threshold (~45s) can trip before
    marker_reader()'s NON-forced re-read surfaces the DONE, so before returning a 'stalled' (which would
    mask the real outcome AND quarantine a region that actually completed the run) the poller FORCE-reads
    the terminal artifacts. A fresh DONE must win -> success, not stalled. Mirrors the Lambda path."""
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "get_vm", lambda vm_id: {"status": "ACTIVE"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=10_000, step=100)
    monkeypatch.setattr(jobs.time, "time", lambda: float(next(clock)))
    metrics = json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0})

    def factory(hf_repo, path, min_interval_s=45.0):
        def read(force=False):
            # DONE is rate-limited: the NON-forced loop read never surfaces it; only the FORCED
            # terminal_artifact_result() read at the stall boundary does (mirrors the 45s vs 60s race).
            if path.endswith("/DONE"):
                return "10500.0" if force else None
            if path.endswith("metrics.json"):
                return metrics
            return None  # no boot.log, no marker, no error file

        return read

    monkeypatch.setattr(jobs, "_make_hf_file_reader", factory)
    res = jobs.poll_hs_job(
        _handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0, first_liveness_s=50.0
    )
    assert res.ok  # forced DONE read before the stall -> terminal success, not a wrongful stall/quarantine
    assert res.failure is None


def test_poll_first_liveness_stall_cancelled_run_not_quarantined(monkeypatch):
    """A user cancel can land while the VM is still ACTIVE-but-silent (no boot.log/heartbeat), and the
    first-liveness threshold can fire before the runner observes the cancellation. run_cancelled()
    suppresses host_fault so a deliberate teardown does NOT quarantine a healthy region -- mirrors the
    dead-VM branch's cancel guard, which this liveness-stall path previously lacked. Mirrors Lambda."""
    import flash.runner

    monkeypatch.setattr(
        flash.runner, "get_status", lambda run_id: type("S", (), {"state": "cancelled"})()
    )
    jobs = _wire_poll(monkeypatch, vms=[{"status": "ACTIVE"}], boot=None, step=100.0)
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=50.0)
    assert res.failure == "stalled"
    assert "no worker liveness" in res.detail
    assert not res.host_fault  # cancelled -> region NOT quarantined despite the liveness stall


def test_poll_loading_timeout(monkeypatch):
    jobs = _wire_poll(monkeypatch, vms=[{"status": "BUILD"}], step=100.0)
    monkeypatch.setattr(jobs, "LOAD_TIMEOUT_S", 300.0)
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "never became active" in res.detail


def test_poll_client_deadline(monkeypatch):
    jobs = _wire_poll(monkeypatch, vms=[{"status": "ACTIVE"}], step=100.0)
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0, deadline_s=250.0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


def test_poll_recovered_deadline_persists_done_written_during_outage(monkeypatch):
    """A control-plane outage longer than the launch-anchored deadline must NOT discard a seed the
    worker finished during the downtime: before returning the deadline `stalled`, poll_hs_job reads
    terminal artifacts once and persists a fresh DONE."""
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}], done="10400.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}), step=10.0,
    )
    res = jobs.poll_hs_job(
        _handle(started_ts=5_000.0), _spec(), seed=0, interval_s=0, deadline_s=250.0
    )
    assert res.ok, res  # success persisted, NOT a stalled-retry
    assert res.metrics["cost_usd"] > 0


def test_poll_recovered_deadline_without_artifacts_still_stalls(monkeypatch):
    jobs = _wire_poll(monkeypatch, vms=[{"status": "ACTIVE"}], step=10.0)
    res = jobs.poll_hs_job(
        _handle(started_ts=5_000.0), _spec(), seed=0, interval_s=0, deadline_s=250.0
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


def test_provider_poll_passes_full_launch_relative_deadline(monkeypatch):
    """Reattach must pass the FULL launch-relative budget: poll_hs_job already anchors its deadline
    to handle.started_ts (= launch), so pre-subtracting elapsed too double-counts and deletes a
    still-valid VM once a recovered run is past half its window."""
    from flash.providers.base import JobHandle, PollResult
    from flash.providers.hyperstack import HyperstackProvider
    from flash.providers.hyperstack.jobs import PROVISION_GRACE_S

    captured = {}

    def fake_poll(handle, spec, seed, *, log=None, heartbeat_reader=None, deadline_s=None,
                  first_liveness_s=None, setup_grace_s=None):
        captured["deadline_s"] = deadline_s
        return PollResult(True)

    monkeypatch.setattr("flash.providers.hyperstack.jobs.poll_hs_job", fake_poll)
    monkeypatch.setattr("flash.providers.hyperstack.api.delete_vm", lambda vid: None)
    spec = _spec()  # max_wall_seconds=3600
    handle = JobHandle.from_dict({"provider": "hyperstack", **_handle(started_ts=1.0).to_dict()})
    HyperstackProvider().poll(handle, spec, seed=0)
    assert captured["deadline_s"] == max(60.0, 3600 + PROVISION_GRACE_S)


def test_provider_poll_reuses_on_last_gpu_first_liveness_scaling(monkeypatch):
    """On recovery the reattach must reproduce the SUBMIT path's last-GPU stall tuning: a handle with
    persisted on_last_gpu=True (written by the runner's on_handle) gets the 1.5x-scaled first_liveness /
    setup grace, else a control-plane restart on the LAST candidate would delete an in-flight,
    cold-starting VM early — terminal there, with no GPU left to walk to. Mirrors RunPodProvider."""
    from flash.providers.base import JobHandle, PollResult
    from flash.providers.hyperstack import HyperstackProvider
    from flash.providers.hyperstack.jobs import FIRST_LIVENESS_S, SETUP_GRACE_S

    captured = {}

    def fake_poll(handle, spec, seed, *, log=None, heartbeat_reader=None, deadline_s=None,
                  first_liveness_s=None, setup_grace_s=None):
        captured["first_liveness_s"] = first_liveness_s
        captured["setup_grace_s"] = setup_grace_s
        return PollResult(True)

    monkeypatch.setattr("flash.providers.hyperstack.jobs.poll_hs_job", fake_poll)
    monkeypatch.setattr("flash.providers.hyperstack.api.delete_vm", lambda vid: None)
    spec = _spec()
    handle = JobHandle.from_dict({**_handle().to_dict(), "provider": "hyperstack", "on_last_gpu": True})
    HyperstackProvider().poll(handle, spec, seed=0)
    assert captured["first_liveness_s"] == FIRST_LIVENESS_S * 1.5
    assert captured["setup_grace_s"] == SETUP_GRACE_S * 1.5
    captured.clear()
    handle2 = JobHandle.from_dict({**_handle().to_dict(), "provider": "hyperstack"})
    HyperstackProvider().poll(handle2, spec, seed=0)
    assert captured["first_liveness_s"] == FIRST_LIVENESS_S
    assert captured["setup_grace_s"] == SETUP_GRACE_S


def test_poll_surfaces_worker_progress(monkeypatch):
    done_seq = iter([None, "10500.0", "10500.0", "10500.0"])
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}], done=lambda: next(done_seq, "10500.0"),
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
    )
    log = io.StringIO()
    res = jobs.poll_hs_job(
        _handle(), _spec(), seed=0, interval_s=0, log=log,
        heartbeat_reader=lambda force=False: {"stage": "sft", "step": 5, "ts": 2.0, "loss": 1.5},
    )
    assert res.ok
    assert "stage=sft" in log.getvalue()


# ---------------------------------------------------------------------------
# cost-safety: every exit path deletes the VM
# ---------------------------------------------------------------------------
def _wire_runner(monkeypatch, poll_outcome):
    from flash.providers.base import PollResult
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    deleted = []
    monkeypatch.setattr(hs_api, "delete_vm", lambda vid: deleted.append(vid) or True)
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False, ignore_sick=False: [_inst()])
    monkeypatch.setattr(jobs, "launch_and_submit", lambda *a, **k: _handle())

    def fake_poll(*a, **k):
        if isinstance(poll_outcome, BaseException):
            raise poll_outcome
        return poll_outcome

    monkeypatch.setattr(jobs, "poll_hs_job", fake_poll)
    return jobs, deleted, PollResult


def test_runner_deletes_on_success(monkeypatch):
    from flash.providers.base import PollResult

    jobs, deleted, _ = _wire_runner(monkeypatch, PollResult(True, metrics={"a": 1}))
    handles = []
    res = jobs.submit_run_hyperstack(_spec(), seed=0, on_handle=handles.append)
    assert res.ok
    assert deleted == ["vm-9999"]
    assert handles
    assert handles[0]["provider"] == "hyperstack"
    assert handles[0]["vm_id"] == "vm-9999"


def test_runner_deletes_on_failure_and_exception(monkeypatch):
    from flash.providers.base import PollResult

    jobs, deleted, _ = _wire_runner(monkeypatch, PollResult(False, failure="stalled"))
    assert not jobs.submit_run_hyperstack(_spec(), seed=0).ok
    assert deleted == ["vm-9999"]

    jobs, deleted, _ = _wire_runner(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        jobs.submit_run_hyperstack(_spec(), seed=0)
    assert deleted == ["vm-9999"]


def test_runner_deletes_when_handle_persist_fails(monkeypatch):
    jobs, deleted, _ = _wire_runner(monkeypatch, None)

    def boom(_h):
        raise RuntimeError("status store unreachable")

    with pytest.raises(RuntimeError, match="status store unreachable"):
        jobs.submit_run_hyperstack(_spec(), seed=0, on_handle=boom)
    assert deleted == ["vm-9999"]


def test_submit_falls_back_to_quarantined_region_when_healthy_empty(monkeypatch):
    """allocate() can pick a class whose every fitting region is quarantined; submit must mirror that
    last-resort relaxation rather than re-fetching a healthy-only (empty) region list and hard-failing
    at launch -- which would turn the bounded-demotion quarantine into a submit-time kill switch. When
    the healthy usable_instances() is empty, submit re-fetches with ignore_sick=True and passes it
    through to launch_and_submit so the in-launch region walk stays consistent."""
    from flash.providers.base import PollResult
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "delete_vm", lambda vid: True)

    seen_ignore_sick = []

    def fake_usable(gpu, force=False, ignore_sick=False):
        seen_ignore_sick.append(ignore_sick)
        return [_inst()] if ignore_sick else []  # healthy view empty; relaxed recovers the region

    captured = {}

    def fake_launch(spec, seed, instances, **k):
        captured["instances"] = instances
        captured["ignore_sick"] = k.get("ignore_sick")
        return _handle()

    monkeypatch.setattr(jobs, "usable_instances", fake_usable)
    monkeypatch.setattr(jobs, "launch_and_submit", fake_launch)
    monkeypatch.setattr(jobs, "poll_hs_job", lambda *a, **k: PollResult(True, metrics={"a": 1}))

    res = jobs.submit_run_hyperstack(_spec(), seed=0)
    assert res.ok
    assert seen_ignore_sick == [False, True]  # healthy pass first (empty), then the relaxed fallback
    assert captured["instances"]  # launch got the recovered quarantined candidate, not an empty list
    assert captured["ignore_sick"] is True  # propagated so the in-launch walk relaxes too


def test_submit_rejects_policy_word_gpu():
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack.jobs import submit_run_hyperstack

    with pytest.raises(hs_api.HyperstackApiError, match="concrete gpu class"):
        submit_run_hyperstack(_spec(gpu_type="cheapest"), seed=0)


# ---------------------------------------------------------------------------
# labels, gc, orphan sweep, dispatch, capacity
# ---------------------------------------------------------------------------
def test_instance_label_always_sweepable():
    from flash.providers.hyperstack.jobs.builders import instance_label

    assert instance_label("flash-1700-abcd", 0, 1) == "flash-1700-abcd-s0-a1"
    assert instance_label("fail-fast", 0, 0) == "flash-fail-fast-s0-a0"


def test_handle_roundtrip():
    from flash.providers.hyperstack.jobs.builders import HyperstackJobHandle

    h = _handle()
    d = h.to_dict()
    assert d["provider"] == "hyperstack"
    assert HyperstackJobHandle.from_dict(d) == h


def test_terminate_run_instances_matches_forced_prefix(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    vms = [
        {"id": "vm-1", "name": "flash-fail-fast-s0-a0"},
        {"id": "vm-2", "name": "flash-other-run-s0-a0"},
    ]
    deleted = []
    monkeypatch.setattr(hs_api, "list_vms", lambda: vms)
    monkeypatch.setattr(hs_api, "delete_vms", lambda ids: deleted.extend(ids) or list(ids))
    assert jobs.terminate_run_instances("fail-fast") == ["vm-1"]
    assert deleted == ["vm-1"]


def test_sweep_orphans_label_safety(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    vms = [
        {"id": "vm-1", "name": "flash-1700-aaaa-s0-a0"},  # orphan
        {"id": "vm-2", "name": "flash-1700-bbbb-s0-a1"},  # active run -> keep
        {"id": "vm-3", "name": "someone-elses"},  # not ours
        {"id": "vm-4", "name": ""},  # unnamed
    ]
    deleted = []
    monkeypatch.setattr(hs_api, "list_vms", lambda: vms)
    monkeypatch.setattr(hs_api, "delete_vms", lambda ids: deleted.extend(ids) or list(ids))
    out = jobs.sweep_orphans(active_labels={"flash-1700-bbbb"})
    assert out == ["vm-1"]
    assert deleted == ["vm-1"]


def test_sweep_orphans_prefix_not_shielded_by_longer_run_id(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    vms = [
        {"id": "vm-1", "name": jobs.instance_label("flash-100", 0, 0)},  # live
        {"id": "vm-2", "name": jobs.instance_label("flash-1000", 0, 0)},  # orphan
    ]
    deleted = []
    monkeypatch.setattr(hs_api, "list_vms", lambda: vms)
    monkeypatch.setattr(hs_api, "delete_vms", lambda ids: deleted.extend(ids) or list(ids))
    assert jobs.sweep_orphans(active_labels={"flash-100"}) == ["vm-2"]


def test_sweep_orphans_exempts_warm_preload_boxes(monkeypatch):
    """Warm/preload boxes (``flash-preload-...``) are driver-owned: launched by
    preload.warm_instances, never persisted in the run DB (so never in the active set), and
    self-terminated by the warm driver. The periodic sweep must NOT reap an IN-DEADLINE preload box by
    the bare ``flash-`` prefix — a catalog warm can outlast the ~10-min sweep and would be killed
    mid-download. A box with no embedded deadline (legacy launch) is likewise exempt.
    """
    import time

    from flash.providers._instance import instance_label
    from flash.providers._poll import preload_instance_run_id
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    # Build the name the way a launch does (instance_label bounds it to the provider name budget) so the
    # reap parser is tested against the REAL, possibly-truncated VM name, not the raw run id.
    fresh = preload_instance_run_id("hyperstack", "canada-1", int(time.time()) + 1800, "abcdef")
    vms = [
        {"id": "vm-1", "name": instance_label(fresh, 0, 0)},  # in-deadline warm box -> KEEP
        {"id": "vm-legacy", "name": "flash-preload-hyperstack-canada-1-abcdef-s0-a0"},  # no deadline -> KEEP
        {"id": "vm-2", "name": "flash-1700-cccc-s0-a0"},  # genuine orphan -> delete
    ]
    deleted = []
    monkeypatch.setattr(hs_api, "list_vms", lambda: vms)
    monkeypatch.setattr(hs_api, "delete_vms", lambda ids: deleted.extend(ids) or list(ids))
    out = jobs.sweep_orphans(active_labels=set())  # none is a tracked active run
    assert out == ["vm-2"]
    assert deleted == ["vm-2"]


def test_sweep_orphans_reaps_stale_preload_box(monkeypatch):
    """A preload VM still alive past its embedded wall deadline + grace has lost its driver (the only
    thing that deletes a Hyperstack VM — nothing on the box self-terminates it). The sweep must reap it
    to bound the billing leak rather than exempt it forever."""
    import time

    from flash.providers._instance import instance_label
    from flash.providers._poll import PRELOAD_REAP_GRACE_S, preload_instance_run_id
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    # Name built via instance_label (longest provider, "hyperstack") so the front-loaded deadline token
    # must survive the provider name-budget truncation to be reaped.
    stale_deadline = int(time.time()) - int(PRELOAD_REAP_GRACE_S) - 600
    stale = preload_instance_run_id("hyperstack", "norway-1", stale_deadline, "deadbe")
    vms = [{"id": "vm-9", "name": instance_label(stale, 0, 0)}]
    deleted = []
    monkeypatch.setattr(hs_api, "list_vms", lambda: vms)
    monkeypatch.setattr(hs_api, "delete_vms", lambda ids: deleted.extend(ids) or list(ids))
    out = jobs.sweep_orphans(active_labels=set())
    assert out == ["vm-9"]
    assert deleted == ["vm-9"]


def test_provider_cancel_destroy_deletes_vm(monkeypatch):
    from flash.providers import get_provider
    from flash.providers.base import JobHandle
    from flash.providers.hyperstack import api as hs_api

    deleted = []
    monkeypatch.setattr(hs_api, "delete_vm", lambda vid: deleted.append(vid) or True)
    h = JobHandle("hyperstack", {"vm_id": "vm-9"})
    get_provider("hyperstack").cancel(h)
    get_provider("hyperstack").destroy(h)
    assert deleted == ["vm-9", "vm-9"]


def test_usable_instances_only_stock_regions(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack.jobs import usable_instances

    monkeypatch.setattr(hs_api, "regions_with_stock", lambda flavor, force=False: ["CANADA-1"])
    monkeypatch.setattr(hs_api, "environment_for_region", lambda r: f"default-{r}")
    monkeypatch.setattr("flash.providers.hyperstack.pricing.hourly_rate", lambda g: 1.00)
    out = usable_instances("L40")
    assert {i.region for i in out} == {"CANADA-1"}
    assert all(i.flavor == "n3-L40x1" for i in out)
    monkeypatch.setattr(hs_api, "regions_with_stock", lambda flavor, force=False: [])
    assert usable_instances("L40") == []


def test_allocator_capacity_aware(monkeypatch):
    from flash.providers import allocator
    from flash.providers.hyperstack.jobs.builders import HyperstackInstance

    monkeypatch.setenv("HYPERSTACK_API_KEY", "hk")

    def fake_usable(gpu, force=False, ignore_sick=False):
        if gpu == "RTX A6000":
            return [
                HyperstackInstance("RTX A6000", "n3-RTX-A6000x1", "NORWAY-1", "default-NORWAY-1", 48, 0.49)
            ]
        if gpu == "L40":  # in stock, but L40 is DROPPED from auto-allocation (broken-fleet only home)
            return [HyperstackInstance("L40", "n3-L40x1", "CANADA-1", "default-CANADA-1", 48, 1.00)]
        return []  # other hyperstack classes out of stock

    monkeypatch.setattr("flash.providers.hyperstack.jobs.usable_instances", fake_usable)
    a = allocator.allocate("Qwen/Qwen3.5-4B", "sft", train={"max_length": 4096, "lora_rank": 16})
    hs = {c.gpu for c in a.candidates if c.provider == "hyperstack"}
    # Only the in-stock, non-excluded class: capacity-aware filtering drops the out-of-stock classes,
    # and L40 is excluded even though it HAS stock (its sole home is the broken CANADA-1 fleet).
    assert hs == {"RTX A6000"}


# --- review-fix regressions ---
def test_poll_ok_marker_succeeds_with_stale_done(monkeypatch):
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}],
        done="9000.0",  # stale
        marker=json.dumps({"ok": True, "attempt": 0}),
        metrics=json.dumps({"wall_seconds": 50, "cost_usd": 0.0}),
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["notes"]["provider"] == "hyperstack"


def test_ambiguous_launch_reconciles_and_stops(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "k")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "img")
    reaped = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda rid: reaped.append(rid) or [])
    attempts = []

    def fake_launch(**k):
        attempts.append(k["environment_name"])
        raise hs_api.HyperstackApiError("POST /core/virtual-machines failed after 5 attempts: timeout")

    monkeypatch.setattr(hs_api, "launch_vm", fake_launch)
    insts = [_inst(region=r) for r in ("CANADA-1", "US-1")]
    with pytest.raises(hs_api.HyperstackApiError, match="ambiguous"):
        jobs.launch_and_submit(_spec(), seed=0, instances=insts, attempt=0)
    assert attempts == ["default-CANADA-1"]  # no second launch
    assert reaped == ["flash-1700000000-abcd1234"]


# ---------------------------------------------------------------------------
# API: VM listing must paginate (orphan sweep reads it; a missed page = a leaked, billing VM)
# ---------------------------------------------------------------------------
def test_list_vms_paginates_all_pages(monkeypatch):
    from flash.providers.hyperstack import api as hs_api

    page_size = hs_api._VM_PAGE_SIZE
    # Two full pages + a short third: list_vms must walk all three and concatenate.
    pages = {
        1: [{"id": i, "name": f"flash-r-s0-a0-{i}"} for i in range(page_size)],
        2: [{"id": page_size + i, "name": f"flash-r-s0-a0-{page_size + i}"} for i in range(page_size)],
        3: [{"id": 2 * page_size, "name": "flash-r-s0-a0-last"}],
    }
    seen_pages = []

    def fake_req(path, **k):
        # path like /core/virtual-machines?page=N&per_page=M
        page = int(path.split("page=")[1].split("&")[0])
        seen_pages.append(page)
        return {"instances": pages.get(page, [])}

    monkeypatch.setattr(hs_api, "request_with_retries", fake_req)
    vms = hs_api.list_vms()
    assert len(vms) == 2 * page_size + 1  # nothing dropped
    assert seen_pages == [1, 2, 3]  # walked every page, stopped at the short one


def test_list_vms_stops_when_server_ignores_pagination(monkeypatch):
    """An older API that echoes page 1 regardless of ?page must not loop forever / double-count:
    a full page that adds no NEW ids terminates the walk."""
    from flash.providers.hyperstack import api as hs_api

    page_size = hs_api._VM_PAGE_SIZE
    same_page = [{"id": i, "name": f"flash-r-s0-a0-{i}"} for i in range(page_size)]
    calls = {"n": 0}

    def fake_req(path, **k):
        calls["n"] += 1
        return {"instances": same_page}  # always page 1

    monkeypatch.setattr(hs_api, "request_with_retries", fake_req)
    vms = hs_api.list_vms()
    assert len(vms) == page_size  # de-duped, not page_size * many
    assert calls["n"] == 2  # one full page, then one more that added nothing -> stop


def test_list_vms_raises_on_malformed_response(monkeypatch):
    """An unexpected response schema (not a dict / no 'instances' list) must RAISE, not silently
    return a partial fleet — orphan sweeping a partial list could miss still-billing VMs."""
    from flash.providers.hyperstack import api as hs_api

    # First page malformed (not a dict).
    monkeypatch.setattr(hs_api, "request_with_retries", lambda path, **k: ["not", "a", "dict"])
    with pytest.raises(hs_api.HyperstackApiError, match="unexpected /core/virtual-machines"):
        hs_api.list_vms()

    # 'instances' present but not a list.
    monkeypatch.setattr(hs_api, "request_with_retries", lambda path, **k: {"instances": "oops"})
    with pytest.raises(hs_api.HyperstackApiError, match="no 'instances' list"):
        hs_api.list_vms()


def test_list_vms_malformed_mid_walk_raises_not_partial(monkeypatch):
    """A malformed LATER page (after valid pages) also raises rather than returning the partial list
    gathered so far — that partial list would look authoritative to the orphan sweep."""
    from flash.providers.hyperstack import api as hs_api

    page_size = hs_api._VM_PAGE_SIZE
    full = [{"id": i, "name": f"flash-r-s0-a0-{i}"} for i in range(page_size)]

    def fake_req(path, **k):
        page = int(path.split("page=")[1].split("&")[0])
        return {"instances": full} if page == 1 else {"unexpected": True}  # page 2 malformed

    monkeypatch.setattr(hs_api, "request_with_retries", fake_req)
    with pytest.raises(hs_api.HyperstackApiError, match="page 2"):
        hs_api.list_vms()


def test_list_vms_empty_page_one_is_valid_not_an_error(monkeypatch):
    """A valid empty fleet (page 1 returns an empty 'instances' list) is NOT an error -> []."""
    from flash.providers.hyperstack import api as hs_api

    monkeypatch.setattr(hs_api, "request_with_retries", lambda path, **k: {"instances": []})
    assert hs_api.list_vms() == []


def test_list_vms_raises_when_page_cap_hit_with_full_last_page(monkeypatch):
    """A GENUINELY-truncated fleet (every page full of NEW ids, AND the page past the cap is STILL
    full) must RAISE rather than return a truncated fleet — an orphan sweep keying off a partial
    list would miss still-billing VMs past the cap."""
    from flash.providers.hyperstack import api as hs_api

    page_size = hs_api._VM_PAGE_SIZE
    # Lower the cap so the test is cheap; every page (incl. the cap+1 probe) is full AND adds
    # brand-new ids, so none of the natural-termination conditions ever fire and the probe confirms
    # the fleet is genuinely over-cap.
    monkeypatch.setattr(hs_api, "_VM_MAX_PAGES", 3)

    def fake_req(path, **k):
        page = int(path.split("page=")[1].split("&")[0])
        base = (page - 1) * page_size
        return {"instances": [{"id": base + i, "name": f"flash-r-s0-a0-{base + i}"} for i in range(page_size)]}

    monkeypatch.setattr(hs_api, "request_with_retries", fake_req)
    with pytest.raises(hs_api.HyperstackApiError, match="did not terminate"):
        hs_api.list_vms()


def test_list_vms_exact_multiple_of_page_size_at_cap_does_not_raise(monkeypatch):
    """A COMPLETE fleet whose size is an exact multiple of _VM_PAGE_SIZE and exactly fills
    _VM_MAX_PAGES must NOT be mistaken for a truncated one: the last in-cap page is full, but the
    probe page (_VM_MAX_PAGES + 1) comes back EMPTY -> the fleet is complete, return it WITHOUT
    raising. (Regression for the exact-multiple edge case in the page-cap guard.)"""
    from flash.providers.hyperstack import api as hs_api

    page_size = hs_api._VM_PAGE_SIZE
    cap = 2
    monkeypatch.setattr(hs_api, "_VM_MAX_PAGES", cap)
    seen_pages = []
    total = cap * page_size  # exactly fills `cap` full pages, then page cap+1 is empty

    def fake_req(path, **k):
        page = int(path.split("page=")[1].split("&")[0])
        seen_pages.append(page)
        base = (page - 1) * page_size
        # Pages 1..cap are full of distinct ids; the cap+1 probe page is genuinely empty.
        items = [
            {"id": base + i, "name": f"flash-r-s0-a0-{base + i}"}
            for i in range(page_size)
            if base + i < total
        ]
        return {"instances": items}

    monkeypatch.setattr(hs_api, "request_with_retries", fake_req)
    vms = hs_api.list_vms()  # must NOT raise
    assert len(vms) == total  # whole fleet returned, nothing dropped
    # Walked the cap pages plus exactly ONE probe page (cap+1) that confirmed completeness.
    assert seen_pages == [1, 2, 3]


# ---------------------------------------------------------------------------
# API: managed-keypair create is race-tolerant (two concurrent launches into one env)
# ---------------------------------------------------------------------------
def test_resolve_key_name_tolerates_create_race(monkeypatch):
    """Two concurrent launches into the same env both see no managed key and both POST; the loser
    gets an 'already exists' rejection — that is SUCCESS (the env-scoped key now exists), so
    resolve_key_name returns the name instead of crashing the launch."""
    from flash.providers.hyperstack import api as hs_api

    monkeypatch.delenv("HYPERSTACK_KEYPAIR_NAME", raising=False)
    env = "default-CANADA-1"
    name = f"{hs_api._MANAGED_KEYPAIR}-{env}"
    monkeypatch.setattr(hs_api, "_generate_throwaway_public_key", lambda: "ssh-ed25519 AAAA")
    # First list: empty (race condition: both launches saw no key). Second list (post-conflict
    # re-check): the winner's key is now present.
    lists = iter([[], [{"name": name, "environment": {"name": env}}]])
    monkeypatch.setattr(hs_api, "list_keypairs", lambda: next(lists))

    def fake_req(path, method="GET", body=None, **k):
        raise hs_api.HyperstackApiError(
            "POST /core/keypairs -> HTTP 409: keypair name already exists"
        )

    monkeypatch.setattr(hs_api, "request_with_retries", fake_req)
    assert hs_api.resolve_key_name(env) == name  # race tolerated


def test_resolve_key_name_reraises_unrelated_create_error(monkeypatch):
    """An UNRELATED keypair-create failure (e.g. bad public key / perms) must still surface, not be
    swallowed as a benign duplicate."""
    from flash.providers.hyperstack import api as hs_api

    monkeypatch.delenv("HYPERSTACK_KEYPAIR_NAME", raising=False)
    monkeypatch.setattr(hs_api, "_generate_throwaway_public_key", lambda: "ssh-ed25519 AAAA")
    monkeypatch.setattr(hs_api, "list_keypairs", lambda: [])

    def fake_req(path, method="GET", body=None, **k):
        raise hs_api.HyperstackApiError("POST /core/keypairs -> HTTP 400: invalid public_key")

    monkeypatch.setattr(hs_api, "request_with_retries", fake_req)
    with pytest.raises(hs_api.HyperstackApiError, match="invalid public_key"):
        hs_api.resolve_key_name("default-CANADA-1")
