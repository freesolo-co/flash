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
