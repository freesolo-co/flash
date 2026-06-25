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


def _handle(started_ts=10_000.0, rate=1.00):
    from flash.providers.hyperstack.jobs.builders import HyperstackJobHandle

    return HyperstackJobHandle(
        vm_id="vm-9999", flavor="n3-L40x1", region="CANADA-1", name="flash-x-s0-a0",
        gpu="L40", hourly_usd=rate, attempt=0, started_ts=started_ts,
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


def test_launch_raises_when_no_stock(monkeypatch):
    from flash.providers.hyperstack import api as hs_api
    from flash.providers.hyperstack import jobs

    monkeypatch.setattr(hs_api, "resolve_key_name", lambda env: "k")
    monkeypatch.setattr(hs_api, "docker_image_for_region", lambda r, min_cuda="12.8": "img")
    monkeypatch.setattr(
        hs_api, "launch_vm", lambda **k: (_ for _ in ()).throw(hs_api.HyperstackApiError("POST /core/virtual-machines -> HTTP 400: no stock"))
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [])
    with pytest.raises(hs_api.HyperstackApiError, match="no stock"):
        jobs.launch_and_submit(_spec(), seed=0, instances=[_inst()], attempt=0)
    with pytest.raises(hs_api.HyperstackApiError, match="no Hyperstack stock"):
        jobs.launch_and_submit(_spec(), seed=0, instances=[], attempt=0)


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
        lambda gpu, force=False: refresh_calls.append(force) or [_inst(region="ELSEWHERE-9")],
    )

    with pytest.raises(hs_api.HyperstackApiError):
        jobs.launch_and_submit(
            _spec(network_volume="flash-weights"), seed=0, instances=[_inst(region="CANADA-1")],
            attempt=0, mode="preload", models=["a/b"],
        )
    assert launched == []  # never launched anywhere (not in the refreshed region)
    assert refresh_calls == []  # the stale-stock refresh was NOT consulted in preload mode


# ---------------------------------------------------------------------------
# poll_hs_job state machine
# ---------------------------------------------------------------------------
def _wire_poll(monkeypatch, vms, done=None, marker=None, metrics=None, boot=None, step=10.0):
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
            if "hyperstack_attempt" in path:
                return marker() if callable(marker) else marker
            if path.endswith("metrics.json"):
                return metrics() if callable(metrics) else metrics
            if path.endswith("hyperstack_boot.log"):
                return boot() if callable(boot) else boot
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
        _handle(), _spec(), seed=0, interval_s=0, heartbeat_reader=lambda force=False: {"retriable": True}
    )
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_dead_vm_without_marker_is_preempted(monkeypatch):
    jobs = _wire_poll(
        monkeypatch, vms=[{"status": "ACTIVE"}, {"status": "ERROR"}],
        boot="+ docker pull ...\nFLASH: gpu never became ready",
    )
    res = jobs.poll_hs_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"
    assert "gpu never became ready" in res.detail


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

    def fake_poll(handle, spec, seed, *, log=None, heartbeat_reader=None, deadline_s=None):
        captured["deadline_s"] = deadline_s
        return PollResult(True)

    monkeypatch.setattr("flash.providers.hyperstack.jobs.poll_hs_job", fake_poll)
    monkeypatch.setattr("flash.providers.hyperstack.api.delete_vm", lambda vid: None)
    spec = _spec()  # max_wall_seconds=3600
    handle = JobHandle.from_dict({"provider": "hyperstack", **_handle(started_ts=1.0).to_dict()})
    HyperstackProvider().poll(handle, spec, seed=0)
    assert captured["deadline_s"] == max(60.0, 3600 + PROVISION_GRACE_S)


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
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [_inst()])
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

    def fake_usable(gpu):
        if gpu == "L40":
            return [HyperstackInstance("L40", "n3-L40x1", "CANADA-1", "default-CANADA-1", 48, 1.00)]
        return []  # other hyperstack classes out of stock

    monkeypatch.setattr("flash.providers.hyperstack.jobs.usable_instances", fake_usable)
    a = allocator.allocate("Qwen/Qwen3.5-4B", "sft", train={"max_length": 4096, "lora_rank": 16})
    hs = {c.gpu for c in a.candidates if c.provider == "hyperstack"}
    assert hs == {"L40"}  # only the in-stock class


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
