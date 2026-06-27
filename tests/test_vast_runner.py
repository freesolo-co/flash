"""Vast.ai run lifecycle: container onstart/bootstrap, offer walk, poll state machine (incl. the
staged setup-vs-training stall grace that fixes the historical ~30-min death), guaranteed destroy,
and orphan sweep (CPU-only; vast API + HF readers mocked).

Vast is opt-in via VAST_API_KEY (the autouse offline fixture deletes it); these tests mock the vast
API entirely, so no key is needed.
"""

from __future__ import annotations

import base64
import itertools
import json

import pytest

from flash.spec import JobSpec


def _spec(gpu_type="RTX 4090", **gpu_kw) -> JobSpec:
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


def _offer(**kw):
    from tests._helpers.vast import make_vast_offer

    return make_vast_offer(**kw)


def _handle(started_ts=10_000.0, rate=0.47):
    from flash.providers.vast.jobs.builders import VastJobHandle

    return VastJobHandle(
        instance_id=9999,
        offer_id=1,
        machine_id=10,
        label="flash-x-s0-a0",
        gpu="RTX 4090",
        hourly_usd=rate,
        attempt=0,
        started_ts=started_ts,
    )


# ---------------------------------------------------------------------------
# container onstart + shared bootstrap
# ---------------------------------------------------------------------------
def test_onstart_ships_payload_and_runs_shared_bootstrap(monkeypatch):
    from flash.providers.vast.jobs import builders

    monkeypatch.setenv("VAST_API_KEY", "vk-supersecret")
    monkeypatch.setenv("HF_TOKEN", "hf-worker-token")
    payload = builders.build_payload(_spec(), seed=0, attempt=1)
    assert payload["phase"] == "sft"
    assert payload["attempt"] == 1
    assert payload["hf_prefix"] == "sft/flash-1700000000-abcd1234/seed0"
    assert payload["max_wall_s"] == 3600
    assert payload["hf_repo"] == "org/repo"
    assert payload["flash_arm"] == "vast"
    # The worker env's HF_REPO is sourced from the run's [train] hf_repo (not an operator default).
    assert payload["env"]["HF_REPO"] == "org/repo"

    script = builders.build_onstart(payload)
    # payload travels base64-encoded inside a quoted heredoc, byte-exact
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    assert json.loads(base64.b64decode(b64)) == payload
    # the SHARED instance bootstrap is embedded + run as the container command
    assert "FLASH_BOOTSTRAP_EOF" in script
    assert "/root/flash/bootstrap.py" in script
    # it is genuinely the shared module (a distinctive line only that file has)
    from pathlib import Path

    import flash.providers._instance_bootstrap as ib

    shared_src = Path(ib.__file__).read_text()
    assert "RetriableBootstrapError" in shared_src  # sanity: distinctive marker exists
    assert "RetriableBootstrapError" in script  # ...and it was shipped
    # the operator's Vast key NEVER ships to the box; the worker HF token rides inside the base64
    # payload's env (like RunPod), never interpolated raw into the shell.
    assert "vk-supersecret" not in script
    assert payload["env"]["HF_TOKEN"] == "hf-worker-token"
    # self-destroy backstop uses the instance-scoped CONTAINER_API_KEY, not the operator key
    assert "CONTAINER_API_KEY" in script
    assert "console.vast.ai/api/v0/instances/" in script
    # no base training-stack install (the worker image is baked); only the bootstrap's per-run extra_pip
    assert "torch==" not in script


def test_build_payload_sets_vast_arm():
    """build_payload stamps flash_arm='vast' so the metrics record attributes the substrate, and the
    shared bootstrap turns it into FLASH_ARM."""
    from flash.providers import _instance_bootstrap as ib
    from flash.providers.vast.jobs.builders import build_payload

    assert build_payload(_spec(), 0, 0)["flash_arm"] == "vast"
    env = ib.build_worker_env(
        {"job_spec_json": "{}", "phase": "sft", "seed": 0, "env": {}, "flash_arm": "vast"}
    )
    assert env["FLASH_ARM"] == "vast"


# ---------------------------------------------------------------------------
# deploy_and_submit: offer (market) walk
# ---------------------------------------------------------------------------
def test_deploy_walks_taken_offers(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    rented = []

    def fake_create(offer_id, **kw):
        if offer_id < 3:
            raise vast_api.VastApiError(f"offer {offer_id} taken")
        rented.append(offer_id)
        return 4242

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    offers = [_offer(offer_id=i, machine_id=i, dph_total=0.20 + i * 0.01) for i in (1, 2, 3)]
    h = vast.deploy_and_submit(_spec(), seed=0, offers=offers, attempt=2)
    assert rented == [3]
    assert h.instance_id == 4242
    assert h.offer_id == 3
    assert h.label == "flash-1700000000-abcd1234-s0-a2"


def test_deploy_refreshes_once_when_all_taken(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    def fake_create(offer_id, **kw):
        if offer_id != 99:
            raise vast_api.VastApiError("taken")
        return 7

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    # the refresh re-search returns a fresh in-pool offer (same class) the walk then rents
    monkeypatch.setattr(
        vast, "usable_offers", lambda *a, **k: [_offer(offer_id=99, machine_id=99, gpu="RTX 4090")]
    )
    h = vast.deploy_and_submit(_spec(), seed=0, offers=[_offer(offer_id=1)], attempt=0)
    assert h.instance_id == 7
    assert h.offer_id == 99


def test_deploy_raises_when_pool_exhausted(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    monkeypatch.setattr(
        vast_api,
        "create_instance",
        lambda *a, **k: (_ for _ in ()).throw(vast_api.VastApiError("taken")),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [])
    with pytest.raises(vast_api.VastApiError, match="rejected the job"):
        vast.deploy_and_submit(_spec(), seed=0, offers=[_offer()], attempt=0)
    with pytest.raises(vast_api.VastApiError, match="no usable vast offers"):
        vast.deploy_and_submit(_spec(), seed=0, offers=[], attempt=0)


# ---------------------------------------------------------------------------
# poll_vast_job state machine
# ---------------------------------------------------------------------------
def _wire_poll(
    monkeypatch, instances, done=None, marker=None, metrics=None, error=None, logs=None, step=10.0
):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    seq = iter(instances)
    last = {"inst": None}

    def fake_get(instance_id):
        last["inst"] = next(seq, last["inst"])
        return last["inst"]

    monkeypatch.setattr(vast_api, "get_instance", fake_get)
    monkeypatch.setattr(vast_api, "instance_logs", lambda iid: logs)
    monkeypatch.setattr(vast.time, "sleep", lambda s: None)
    clock = itertools.count(start=10_000, step=step)
    monkeypatch.setattr(vast.time, "time", lambda: float(next(clock)))

    def factory(hf_repo, path, min_interval_s=45.0):
        def read(force=False):
            if path.endswith("/DONE"):
                return done() if callable(done) else done
            if "vast_attempt" in path and path.endswith(".json"):
                return marker() if callable(marker) else marker
            if path.endswith("metrics.json"):
                return metrics() if callable(metrics) else metrics
            if "/error_" in path:
                return error() if callable(error) else error
            return None

        return read

    monkeypatch.setattr(vast, "_make_hf_file_reader", factory)
    return vast


def test_poll_success_stamps_real_cost(monkeypatch):
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        done="10500.0",
        metrics=json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0}),
    )
    res = vast.poll_vast_job(_handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["train_tokens"] == 4096
    # cost comes from the offer's real live $/hr x wall time, not a runpod table rate
    assert res.metrics["cost_usd"] > 0
    assert res.metrics["notes"]["provider"] == "vast"
    assert res.metrics["notes"]["vast_rate_usd_hr"] == 0.47
    assert res.metrics["notes"]["vast_offer_id"] == 1


def test_poll_caps_recovered_cost_at_done_timestamp(monkeypatch):
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        done="9100.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
    )
    res = vast.poll_vast_job(_handle(started_ts=9000.0), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["cost_usd"] == round((9100.0 - 9000.0) / 3600.0 * 0.47, 6)


def test_poll_stale_done_is_ignored(monkeypatch):
    """A DONE from a PRIOR attempt (ts < this launch - skew) is not this attempt's completion; the
    instance later dies as a host loss -> job_preempted, NOT a false success."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}, {"actual_status": "exited"}],
        done="500.0",  # long before launch (10_000)
    )
    res = vast.poll_vast_job(_handle(started_ts=10_000.0), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_marker_failure_is_job_failed(monkeypatch):
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RuntimeError: boom"}),
    )
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"  # real worker error fails fast
    assert "boom" in res.detail


def test_poll_retriable_marker_is_job_preempted(monkeypatch):
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "transient", "retriable": True}),
    )
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_dead_host_without_marker_is_preempted(monkeypatch):
    """An instance that vanished without writing DONE/marker is a host loss -> retryable, with the
    container log tail (Vast's console API) as the only window."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}, {"actual_status": "exited"}],
        logs="+ docker pull ...\nFLASH: gpu never became ready",
    )
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"
    assert "gpu never became ready" in res.detail


def test_poll_dead_host_with_error_file_is_job_failed(monkeypatch):
    """A worker that RAN and crashed early (left error_<phase>.txt) but died before the marker is a
    DETERMINISTIC worker error -> fail fast (job_failed), not burn fresh GPUs retrying a repeat crash."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}, {"actual_status": "exited"}],
        error="Traceback (most recent call last):\nFileNotFoundError: environment archive missing ...",
    )
    res = vast.poll_vast_job(
        _handle(), _spec(), seed=0, interval_s=0, heartbeat_reader=lambda force=False: {}
    )
    assert not res.ok
    assert res.failure == "job_failed"
    assert "environment archive" in res.detail


def test_poll_loading_timeout(monkeypatch):
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "loading"}], step=100.0)
    monkeypatch.setattr(vast, "LOAD_TIMEOUT_S", 300.0)
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "never started" in res.detail


# --- the fix: staged setup-vs-training stall grace ---------------------------
def test_poll_setup_grace_protects_long_cold_start(monkeypatch):
    """THE FIX. A container that is 'running' but has emitted only a SETUP heartbeat (model download /
    vLLM init, no per-step progress) must be governed by the LARGER setup grace, not the tight training
    window. With a SETUP-stage heartbeat frozen and a stall_after_s far below the elapsed gap, the run
    must NOT be killed until the (larger) setup_grace_s is exceeded — proving the box survives the
    cold-start window that used to kill it every ~30 min."""
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "running"}], step=100.0)
    # a fresh SETUP-stage heartbeat (boot/model-load), then frozen
    setup_hb = {"stage": "sft_model_load", "step": 0, "ts": 10_000.0}
    res = vast.poll_vast_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: setup_hb,
        setup_grace_s=4000.0,
        stall_after_s=200.0,  # tight training window — must NOT govern during setup
    )
    assert not res.ok
    assert res.failure == "stalled"
    # stalled on the SETUP grace (4000s), not the 200s training window
    assert "setup (pre-training)" in res.detail
    assert "limit 4000s" in res.detail


def test_poll_training_heartbeat_tightens_to_stall_window(monkeypatch):
    """Once a TRAINING heartbeat (a non-setup stage) arrives, the poll tightens to the smaller
    stall_after_s — a hung training loop is caught quickly (not given the full setup grace)."""
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "running"}], step=100.0)
    training_hb = {"stage": "rl_step", "step": 3, "ts": 10_000.0}  # training stage, then frozen
    res = vast.poll_vast_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: training_hb,
        setup_grace_s=9000.0,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "during training" in res.detail
    assert "limit 500s" in res.detail


def test_poll_running_no_heartbeat_first_liveness_fails_over(monkeypatch):
    """A container that reached 'running' but emitted NO heartbeat at all past first_liveness_s is a
    wedged worker -> fast retriable 'stalled' (the worker never came up), instead of burning the full
    setup grace. Vast has no host boot.log, so the heartbeat is the sole liveness signal."""
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "running"}], step=100.0)
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0, first_liveness_s=500.0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "no worker heartbeat" in res.detail
    assert "limit 500s" in res.detail


def test_poll_fresh_boot_heartbeat_satisfies_liveness(monkeypatch):
    """Any FRESH heartbeat (even the early 'boot' stage) proves the worker started, so the
    first-liveness deadline is satisfied; the box later dies as a host loss -> job_preempted."""
    vast = _wire_poll(
        monkeypatch,
        instances=[
            {"actual_status": "running"},
            {"actual_status": "running"},
            {"actual_status": "exited"},
        ],
        step=100.0,
    )
    res = vast.poll_vast_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        first_liveness_s=50.0,
        heartbeat_reader=lambda force=False: {"stage": "boot", "step": 0, "ts": 10_000.0},
    )
    assert res.failure == "job_preempted"
    assert "no worker heartbeat" not in (res.detail or "")


def test_poll_stale_heartbeat_does_not_arm_training_stall(monkeypatch):
    """A LEFTOVER training heartbeat from a PRIOR attempt (ts < this launch) must NOT be treated as
    current progress: heartbeat_progress_ts marks it not-fresh, so it neither satisfies first-liveness
    nor arms the tighter training stall window for THIS attempt. With no FRESH liveness, the correct
    Vast outcome is the first-liveness failover (a retriable 'stalled'), NOT a false 'during training'
    stall keyed off the stale heartbeat."""
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "running"}], step=10.0)
    stale = {"stage": "rl_step", "step": 2, "ts": 8000.0}  # training stage, predates launch 9000
    res = vast.poll_vast_job(
        _handle(started_ts=9_000.0),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: stale,
        setup_grace_s=3000.0,
        stall_after_s=500.0,
        first_liveness_s=50.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    # the stale training heartbeat did NOT arm the tight training window...
    assert "during training" not in res.detail
    assert "limit 500s" not in res.detail
    # ...and did NOT satisfy liveness -> fast first-liveness failover instead
    assert "no worker heartbeat" in res.detail


def test_poll_client_deadline(monkeypatch):
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "running"}], step=100.0)
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0, deadline_s=250.0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


def test_poll_recovered_deadline_persists_done_written_during_outage(monkeypatch):
    """A control-plane outage longer than the launch-anchored deadline must NOT discard a seed the
    worker actually finished during the downtime: before returning the deadline stall, the poller reads
    terminal artifacts once and persists a fresh DONE."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        done="10400.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
        step=10.0,
    )
    res = vast.poll_vast_job(
        _handle(started_ts=5_000.0), _spec(), seed=0, interval_s=0, deadline_s=250.0
    )
    assert res.ok, res
    assert res.metrics["cost_usd"] > 0


def test_poll_missing_started_ts_anchors_to_now_not_epoch(monkeypatch):
    """started_ts coerced to 0.0 (old/corrupt handle) means 'unknown launch' -> anchor to now, not the
    1970 epoch (else a running box is 'past' a ~57-yr load window and wall/cost bill from the epoch)."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        done="10500.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
        step=10.0,
    )
    res = vast.poll_vast_job(_handle(started_ts=0.0), _spec(), seed=0, interval_s=0)
    assert res.ok, res
    assert res.metrics["cost_usd"] < 1.0, res.metrics["cost_usd"]


# ---------------------------------------------------------------------------
# submit_run_vast: guaranteed teardown
# ---------------------------------------------------------------------------
def _wire_submit(monkeypatch, poll_result=None, poll_raises=None):
    from flash.providers.base import PollResult
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, log=None, runtime_secrets=None: _handle(),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])

    def fake_poll(handle, spec, seed, **kw):
        if poll_raises:
            raise poll_raises
        return poll_result or PollResult(True, metrics={})

    monkeypatch.setattr(vast, "poll_vast_job", fake_poll)
    return vast, destroyed


def test_runner_destroys_on_success(monkeypatch):
    vast, destroyed = _wire_submit(monkeypatch)
    res = vast.submit_run_vast(_spec(), seed=0)
    assert res.ok
    assert destroyed == [9999]  # the rented instance is torn down


def test_runner_destroys_on_exception(monkeypatch):
    vast, destroyed = _wire_submit(monkeypatch, poll_raises=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        vast.submit_run_vast(_spec(), seed=0)
    assert destroyed == [9999]  # destroyed even on interrupt


def test_runner_destroys_when_handle_persist_fails(monkeypatch):
    """on_handle (persisting the handle) raising must still tear down the already-billing instance."""
    vast, destroyed = _wire_submit(monkeypatch)

    def boom(_d):
        raise RuntimeError("db down")

    with pytest.raises(RuntimeError, match="db down"):
        vast.submit_run_vast(_spec(), seed=0, on_handle=boom)
    assert destroyed == [9999]


def test_submit_run_vast_rejects_policy_word_gpu(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    spec = _spec()
    object.__setattr__(
        spec.gpu, "type", "cheapest"
    )  # a policy word that never reached the allocator
    with pytest.raises(vast_api.VastApiError, match="concrete gpu class"):
        vast.submit_run_vast(spec, seed=0)


# ---------------------------------------------------------------------------
# labels, handle, sweep, gc
# ---------------------------------------------------------------------------
def test_instance_label_and_handle_roundtrip():
    from flash.providers.vast.jobs import instance_label, run_label_prefix
    from flash.providers.vast.jobs.builders import VastJobHandle

    label = instance_label("flash-run9", seed=0, attempt=2)
    assert label.startswith(run_label_prefix("flash-run9"))
    assert label.endswith("-s0-a2")
    h = _handle()
    back = VastJobHandle.from_dict(h.to_dict())
    assert back.to_dict()["provider"] == "vast"
    assert back.instance_id == h.instance_id
    assert back.offer_id == h.offer_id


def test_destroy_run_instances_matches_forced_prefix(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": 1, "label": "flash-run1-s0-a0"},  # ours
        {
            "id": 2,
            "label": "flash-run10-s0-a0",
        },  # a DIFFERENT run (prefix boundary) — must NOT match
        {"id": 3, "label": "someone-else"},  # not ours
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.destroy_run_instances("run1")  # raw id; run_label_prefix forces the flash- prefix
    assert out == [1]
    assert destroyed == [1]


def test_sweep_orphans_label_safety_and_active_protection(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": 1, "label": "flash-runA-s0-a0"},  # active -> protected
        {"id": 2, "label": "flash-runB-s0-a0"},  # orphan -> reaped
        {"id": 3, "label": "flash-runA10-s0-a0"},  # NOT runA (boundary) -> orphan, reaped
        {"id": 4, "label": "not-ours"},  # untouched
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.sweep_orphans(active_labels={"runA"})  # raw active id; prefix forced internally
    assert sorted(out) == [2, 3]
    assert 1 not in destroyed  # the active run's box survived
    assert 4 not in destroyed  # non-flash box untouched


def test_sweep_orphans_known_labels_multiplane_guard(monkeypatch):
    """With known_labels set, an instance is reaped only if its run id is one THIS plane knows — a box
    from ANOTHER control plane (run id absent from known) is left alone (multi-plane safety)."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": 1, "label": "flash-mine-s0-a0"},  # known + not active -> reaped
        {"id": 2, "label": "flash-other-s0-a0"},  # unknown to this plane -> left alone
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.sweep_orphans(active_labels=set(), known_labels={"mine"})
    assert out == [1]
    assert 2 not in destroyed


def test_sweep_orphans_callable_sets_resolved_after_listing(monkeypatch):
    """active_labels/known_labels may be CALLABLES resolved AFTER the instance list (closes the launch
    race). A callable that raises SKIPS the sweep (never falls through to reaping live boxes)."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    monkeypatch.setattr(vast_api, "list_instances", lambda: [{"id": 1, "label": "flash-x-s0-a0"}])
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: True)
    # protected by a callable-resolved active set
    assert vast.sweep_orphans(active_labels=lambda: {"x"}) == []

    # a raising callable -> sweep skipped (returns [], reaps nothing)
    def boom():
        raise RuntimeError("db down")

    assert vast.sweep_orphans(active_labels=boom) == []
