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


def _handle(started_ts=10_000.0, rate=0.47, attempt=0):
    from flash.providers.vast.jobs.builders import VastJobHandle

    return VastJobHandle(
        instance_id=9999,
        offer_id=1,
        machine_id=10,
        label=f"flash-x-s0-a{attempt}",
        gpu="RTX 4090",
        hourly_usd=rate,
        attempt=attempt,
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


def test_onstart_heredoc_terminators_on_own_line_and_python_fallback(monkeypatch):
    """Copilot MsGxp/MsGxy: the heredoc terminators must start on their own line (a bootstrap
    source without a trailing newline would otherwise swallow the rest of the script), and the
    python-interpreter resolution must fall back past python3 to python with a clear diagnostic."""
    from flash.providers.vast.jobs import builders

    monkeypatch.setenv("VAST_API_KEY", "vk")
    monkeypatch.setenv("HF_TOKEN", "hf")
    script = builders.build_onstart(builders.build_payload(_spec(), seed=0, attempt=1))
    # Each closing terminator is preceded by a newline (own line), regardless of payload/src content.
    for term in ("FLASH_PAYLOAD_EOF", "FLASH_BOOTSTRAP_EOF"):
        assert f"\n{term}\n" in script, f"{term} terminator must be on its own line"
    # PYBIN never silently empty: python fallback + a diagnostic when nothing resolves.
    assert "command -v python3 || command -v python" in script
    assert "no python interpreter" in script
    # Copilot Msbs6: an empty PYBIN must EXIT (after a log-retrieval hold), not fall through to the
    # doomed `"$PYBIN"` bootstrap + self-destroy invocations.
    assert 'if [ -z "$PYBIN" ]; then' in script
    assert "exit 1" in script


def test_onstart_spills_large_spec_to_hf(monkeypatch):
    """Codex MsMPw: a large inline job spec is spilled to HF (parity with Lambda's build_user_data)
    so it never inflates the base64 onstart past Vast's exec-arg / onstart length limit and fails the
    rent before a handle is persisted. A small spec rides inline unchanged."""
    import huggingface_hub

    from flash.providers.vast.jobs import builders

    uploaded = {}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type):
            uploaded.update(path=path_in_repo, repo=repo_id, type=repo_type)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)

    big = "x" * 20_000  # > _SPEC_SPILL_THRESHOLD (16k)
    payload = {
        "job_spec_json": big,
        "hf_prefix": "sft/run/seed0",
        "hf_repo": "org/repo",
        "env": {"HF_TOKEN": "t"},
        "flash_arm": "vast",
    }
    script = builders.build_onstart(payload)
    assert big not in script  # the giant spec is NOT embedded inline...
    assert uploaded["path"] == "sft/run/seed0/job_spec.json"  # ...it was spilled to the dataset repo
    assert uploaded["type"] == "dataset"
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    embedded = json.loads(base64.b64decode(b64))
    assert embedded["job_spec_in_hf"] is True
    assert embedded["job_spec_json"] == ""


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


def test_deploy_adopts_instance_after_ambiguous_create(monkeypatch):
    # Codex Mr72L: a 5xx/timeout on the NON-IDEMPOTENT create may have made a billed contract. The
    # walk must reconcile by our unique label and ADOPT it, not rent the next offer (double-billing).
    import io
    import urllib.error

    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    rented = []

    def fake_create(offer_id, **kw):
        rented.append(offer_id)
        e = vast_api.VastApiError("create failed: 503")
        e.__cause__ = urllib.error.HTTPError("u", 503, "boom", None, io.BytesIO(b""))
        raise e

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    # the contract DID materialize under our exact attempt label -> list_instances surfaces it, with
    # the box's real launch epoch in start_date
    label = "flash-1700000000-abcd1234-s0-a2"
    monkeypatch.setattr(
        vast_api, "list_instances", lambda: [{"id": 555, "label": label, "start_date": 1699999000.0}]
    )
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    h = vast.deploy_and_submit(_spec(), seed=0, offers=offers, attempt=2)
    assert h.instance_id == 555  # adopted the existing contract, not a fresh rent
    assert h.offer_id == 1
    assert h.started_ts == 1699999000.0  # Codex Mr72L / Cursor MsA6d: real launch time, not now
    assert rented == [1]  # did NOT walk on to offer 2 (no duplicate create)


def test_deploy_aborts_walk_when_ambiguous_create_left_nothing(monkeypatch):
    # Cursor MsA6X: an ambiguous failure with NO instance visible under our label must ABORT the walk
    # (the contract may exist but not be visible yet) rather than rent another offer and double-bill.
    # Codex MtbAD: the abort must raise the TERMINAL UnreconciledCreateError (not a plain VastApiError
    # that the orchestrator retries as poll_error) — a phantom contract that surfaces AFTER the
    # point-in-time destroy_run_instances sweep would otherwise bill under a retry's new instance.
    import io
    import urllib.error

    from flash.providers.base import UnreconciledCreateError
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    rented = []

    def fake_create(offer_id, **kw):
        rented.append(offer_id)
        e = vast_api.VastApiError("create failed: 503")
        e.__cause__ = urllib.error.HTTPError("u", 503, "boom", None, io.BytesIO(b""))
        raise e

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    monkeypatch.setattr(vast_api, "list_instances", lambda: [])  # nothing under our label
    # Cursor MsECQ: the abort must proactively destroy this run's instances (kill any phantom)
    destroyed_for = []
    monkeypatch.setattr(vast, "destroy_run_instances", lambda rid: destroyed_for.append(rid) or [])
    offers = [_offer(offer_id=1, machine_id=1), _offer(offer_id=2, machine_id=2)]
    with pytest.raises(UnreconciledCreateError, match="aborting the offer walk"):
        vast.deploy_and_submit(_spec(), seed=0, offers=offers, attempt=2)
    assert rented == [1]  # aborted after the FIRST offer — never rented offer 2
    assert destroyed_for  # destroy_run_instances was called to reap any phantom contract


def test_vast_image_honors_worker_image_override(monkeypatch):
    # Codex Mr72Q: Vast must honor FLASH_WORKER_IMAGE (and per-SM) via worker_image_for_gpu like
    # RunPod/Lambda, not always return the baked default.
    from flash.providers.vast.jobs.builders import vast_image

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/x/hotfix:test")
    assert vast_image("RTX 4090") == "ghcr.io/x/hotfix:test"
    monkeypatch.delenv("FLASH_WORKER_IMAGE", raising=False)
    assert vast_image("RTX 4090")  # default path still returns a real (baked) image


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


def test_poll_dead_host_stale_prior_attempt_error_is_preempted(monkeypatch):
    """Codex MtbAD: ``error_<phase>.txt`` is seed-scoped (shared across this seed's retries). When the
    latest heartbeat provably belongs to a PRIOR attempt (here attempt=0 while we poll attempt=1), the
    co-located error file is a LEFTOVER — a fresh host LOSS on attempt 1, not a deterministic crash.
    Without this guard, gating only the retriable flag (1a28224) would fail-fast a genuine retry."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}, {"actual_status": "exited"}],
        error="Traceback (most recent call last):\nRuntimeError: stale crash from a prior attempt ...",
    )
    # ts AFTER this attempt's launch (10_000) yet attempt=0 != the polled attempt=1 — the subtle
    # "fresh by timestamp but belongs to a different attempt" leftover.
    prior_hb = {"stage": "sft_train", "step": 5, "ts": 10_500.0, "attempt": 0}
    res = vast.poll_vast_job(
        _handle(started_ts=10_000.0, attempt=1),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: prior_hb,
    )
    assert not res.ok
    assert res.failure == "job_preempted"  # leftover crash artifact -> retry on a fresh host


def test_poll_dead_host_current_attempt_error_is_job_failed(monkeypatch):
    """The complement of the stale-leftover case: when the heartbeat belongs to THIS attempt (attempt
    matches) and the worker did not flag the failure retriable, the error file IS this attempt's
    deterministic crash -> fail fast even on a retry (attempt=1)."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}, {"actual_status": "exited"}],
        error="Traceback (most recent call last):\nValueError: bad config on this very attempt ...",
    )
    cur_hb = {"stage": "sft_train", "step": 5, "ts": 10_500.0, "attempt": 1}
    res = vast.poll_vast_job(
        _handle(started_ts=10_000.0, attempt=1),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: cur_hb,
    )
    assert not res.ok
    assert res.failure == "job_failed"
    assert "bad config" in res.detail


def test_poll_dead_host_unknown_launch_same_attempt_error_is_job_failed(monkeypatch):
    """Cursor MtgwT: with an unknown launch (started_ts=0.0) the dead-host path must NOT date the
    heartbeat against the now()-fallback. The attempt-attribution helpers get the TRUE launch (0.0),
    which disables ts-based staleness, so a same-attempt crash heartbeat (ts naturally < poll time) is
    correctly read as CURRENT evidence -> job_failed, not a false job_preempted. (Before the fix,
    launch_ts=now() made the normal heartbeat ts look pre-launch -> stale -> wrongly preempted.)"""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}, {"actual_status": "exited"}],
        error="Traceback (most recent call last):\nValueError: deterministic crash, unknown launch ...",
    )
    # attempt matches the handle (0); ts is a normal value below the poll clock (starts at 10_000).
    cur_hb = {"stage": "error_sft", "ts": 9_500.0, "attempt": 0}
    res = vast.poll_vast_job(
        _handle(started_ts=0.0, attempt=0),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: cur_hb,
    )
    assert not res.ok
    assert res.failure == "job_failed"
    assert "deterministic crash" in res.detail


def test_poll_running_then_unknown_is_dead_host_preempted(monkeypatch):
    """Codex MtrgK: a host that WAS running and then reports actual_status='unknown' (Vast's
    no-recent-heartbeat-won't-progress state) is a host loss -> take the dead-host path NOW (preempted)
    instead of waiting out the stall window while the box keeps billing."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}, {"actual_status": "unknown"}],
        logs="+ training ...\nFLASH: host went silent",
    )
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_frozen_is_dead_host_preempted(monkeypatch):
    """Codex: Vast's 'frozen' is a PAUSED container that keeps billing GPU charges yet emits no
    DONE/heartbeat, so a worker that freezes must take the dead-host path immediately (preempted)
    instead of waiting out the setup/training stall window while the box bills. Unlike 'unknown' it
    is never the poller's no-status fallback, so it needs no became_running gate (fails even if the
    box never reported running first)."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "frozen"}],
        logs="+ paused\nFLASH: container frozen",
    )
    assert "frozen" in vast._DEAD_STATES
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_unknown_before_running_is_not_dead(monkeypatch):
    """The became_running gate: 'unknown' is ALSO the fallback the poller substitutes for a present
    instance with no actual_status yet (normal provisioning), so a box that has NEVER run must NOT be
    failed as a dead host on 'unknown' — it stays governed by the load/stall window."""
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "unknown"}], step=100.0)
    monkeypatch.setattr(vast, "LOAD_TIMEOUT_S", 300.0)
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"  # never-started load timeout, NOT a dead-host preempt
    assert "never started" in res.detail


def test_poll_done_waits_for_eventually_consistent_metrics(monkeypatch):
    """Codex MtzrL: a fresh DONE can be visible before the separately-uploaded metrics.json is readable
    (HF read-after-write is eventually consistent). finish_ok must RE-READ metrics before failing — a
    successful run must not be classified job_failed on that transient gap. (time.sleep is mocked.)"""
    seq = {"n": 0}

    def metrics_seq():
        seq["n"] += 1
        # None on the first reads (metrics.json not visible yet), then it surfaces
        if seq["n"] <= 2:
            return None
        return json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0})

    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        done="10500.0",
        metrics=metrics_seq,
    )
    res = vast.poll_vast_job(_handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0)
    assert res.ok  # not a false job_failed
    assert res.metrics["train_tokens"] == 4096
    assert seq["n"] >= 3  # re-read past the initial misses


def test_poll_done_without_metrics_eventually_fails(monkeypatch):
    """The complement: if metrics.json NEVER surfaces (a genuine DONE-without-metrics), the retries are
    bounded and the poll still classifies job_failed rather than spinning forever."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        done="10500.0",
        metrics=None,  # never visible
    )
    res = vast.poll_vast_job(_handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"
    assert "DONE without metrics.json" in res.detail


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


def test_poll_container_log_output_protects_slow_bootstrap(monkeypatch):
    """Codex MsMPz: a 'running' container with NO worker heartbeat but ACTIVE container-log output
    (slow per-run pip install / code fetch) is a healthy cold start, not a wedged host — so the
    container-log signal latches and the run is governed by setup_grace_s, NOT fast-failed at
    first_liveness_s the way a genuinely silent box is. Mirrors Lambda's boot.log liveness."""
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        logs="Collecting torch...\nDownloading flash code...",  # bootstrap is producing output
        step=100.0,
    )
    res = vast.poll_vast_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        first_liveness_s=300.0,  # would fast-fail a SILENT box here
        setup_grace_s=4000.0,
        stall_after_s=200.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    # governed by the larger SETUP grace, not the first-liveness fast-fail
    assert "setup (pre-training)" in res.detail
    assert "limit 4000s" in res.detail
    assert "no worker heartbeat" not in res.detail


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


def test_submit_teardown_warns_on_unconfirmed_destroy_without_raising(monkeypatch, caplog):
    """Copilot Mtjga: the PRIMARY teardown (submit_run_vast ``finally``) must NOT silently ignore a
    success:false from destroy_instance — a raise there would mask the poll result, so instead it WARNS
    so operators see a possible leak immediately (not only at the next sweep). The run still returns."""
    import logging

    from flash.providers.base import PollResult
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: False)  # unconfirmed teardown
    monkeypatch.setattr(
        vast,
        "deploy_and_submit",
        lambda spec, seed, offers, attempt=0, log=None, runtime_secrets=None: _handle(),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [_offer()])
    monkeypatch.setattr(vast, "poll_vast_job", lambda *a, **k: PollResult(True, metrics={}))

    with caplog.at_level(logging.WARNING):
        res = vast.submit_run_vast(_spec(), seed=0)  # the finally must not raise on False
    assert res.ok
    assert any("teardown unconfirmed" in r.message for r in caplog.records), (
        "an unconfirmed teardown in the primary path must emit an operator-visible warning"
    )


def test_best_effort_destroy_returns_confirmation(monkeypatch):
    """The helper returns the destroy_instance bool and only warns on False (no warn on a clean True)."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: True)
    assert vast._best_effort_destroy(123, context="t") is True
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: False)
    assert vast._best_effort_destroy(123, context="t") is False


def test_best_effort_destroy_passes_raw_id_and_never_int_raises(monkeypatch):
    """Cursor MtlVb: the helper must NOT int()-convert the id itself — destroy_instance does that inside
    its own try/except (-> False on a bad id, "never raises"), so converting in the wrapper would
    re-introduce a ValueError in the very finally/suppress paths this helper exists to keep quiet.
    Assert the id reaches destroy_instance UNCONVERTED and a non-numeric id returns False, no raise."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    seen = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: seen.append(iid) or False)
    assert vast._best_effort_destroy("not-a-number", context="t") is False  # must not raise
    assert seen == ["not-a-number"]  # passed through raw — no int() in the wrapper


def test_submit_run_vast_rejects_policy_word_gpu(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    spec = _spec()
    object.__setattr__(
        spec.gpu, "type", "cheapest"
    )  # a policy word that never reached the allocator
    with pytest.raises(vast_api.VastApiError, match="concrete gpu class"):
        vast.submit_run_vast(spec, seed=0)


def test_provider_destroy_raises_on_unconfirmed_teardown(monkeypatch):
    """Codex MtbAK: ``destroy_instance`` returning False (success:false / breakdown) means the box is
    STILL billing. ``VastProvider.destroy`` must SURFACE that (raise) instead of returning normally —
    else the best-effort callers log "terminated" and clear the handle while it keeps billing."""
    from flash.providers.base import JobHandle
    from flash.providers.vast import PROVIDER
    from flash.providers.vast import api as vast_api

    handle = JobHandle.from_dict(_handle().to_dict())
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: False)  # unconfirmed
    with pytest.raises(vast_api.VastApiError, match="unconfirmed"):
        PROVIDER.destroy(handle)
    # confirmed teardown returns normally (no raise)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: True)
    PROVIDER.destroy(handle)
    # no instance_id -> nothing to destroy, no raise (idempotent)
    PROVIDER.destroy(JobHandle.from_dict({"provider": "vast"}))


def test_provider_destroy_passes_raw_id_no_int_raise(monkeypatch):
    """Copilot Mtugr: VastProvider.destroy must NOT int()-convert the id in the wrapper — a corrupt /
    non-numeric handle would raise ValueError/TypeError and break the retry teardown. destroy_instance
    does the int() inside its own try/except (-> False), so pass the id through raw and let the False ->
    raise VastApiError path handle a bad id (no uncaught ValueError)."""
    from flash.providers.base import JobHandle
    from flash.providers.vast import PROVIDER
    from flash.providers.vast import api as vast_api

    seen = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: seen.append(iid) or False)
    handle = JobHandle.from_dict({"provider": "vast", "instance_id": "not-a-number"})
    # surfaces as VastApiError (unconfirmed), NOT a raw ValueError from int() in the wrapper
    with pytest.raises(vast_api.VastApiError, match="unconfirmed"):
        PROVIDER.destroy(handle)
    assert seen == ["not-a-number"]  # passed through raw — destroy_instance owns the int()


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


def test_handle_from_dict_corrupt_instance_id_raises_clear_error():
    """Copilot MuX0a: a corrupt/partial PERSISTED handle (reattach/recovery) must fail with a CLEAR,
    actionable error naming the bad instance_id — not a bare KeyError/ValueError that crashes recovery
    with an opaque cause. instance_id has no safe default (it's the poll/destroy target)."""
    from flash.providers.vast.jobs.builders import VastJobHandle

    for bad in ({}, {"instance_id": None}, {"instance_id": "not-a-number"}):
        with pytest.raises(ValueError, match="corrupt vast handle"):
            VastJobHandle.from_dict(bad)


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


def test_run_instances_remaining_confirms_clear_and_raises_on_listing_failure(monkeypatch):
    # Codex: the handle-less recovery resubmit gates on this. [] == CONFIRMED no instance for the run
    # remains; a survivor (e.g. after an unconfirmed DELETE) is reported by id; it matches on the SAME
    # label boundary as destroy_run_instances (run1 must not match run10). A listing failure RAISES so
    # the caller can't mistake "couldn't list" for "clear".
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": 9, "label": "flash-run1-s0-a0"},  # ours -> remaining
        {"id": 10, "label": "flash-run10-s0-a0"},  # different run (boundary) -> NOT ours
        {"id": 11, "label": "someone-else"},  # not ours
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda strict=False: instances)
    assert vast.run_instances_remaining("run1") == [9]

    monkeypatch.setattr(vast_api, "list_instances", lambda strict=False: [])
    assert vast.run_instances_remaining("run1") == []  # confirmed clear

    def boom(strict=False):
        raise vast_api.VastApiError("list failed")

    monkeypatch.setattr(vast_api, "list_instances", boom)
    with pytest.raises(vast_api.VastApiError):
        vast.run_instances_remaining("run1")  # cannot confirm clear -> RAISE (caller defers)


def test_cleanup_loops_skip_non_intable_id_without_raising(monkeypatch):
    """Copilot Mtnjw/Mtnj2: destroy_run_instances and sweep_orphans are documented "never raises", but
    a bare int(iid) on a non-intable id (unexpected Vast API shape) would raise mid-loop and abort the
    cleanup, leaving the remaining reapable boxes billing. A bad id must be SKIPPED, the GOOD ones still
    destroyed."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": None, "label": "flash-run1-s0-a0"},  # missing id -> skip
        {"id": "not-an-int", "label": "flash-run1-s1-a0"},  # non-intable -> skip, must NOT raise
        {"id": 7, "label": "flash-run1-s2-a0"},  # good -> destroyed
    ]
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)

    assert vast.destroy_run_instances("run1") == [7]  # bad ids skipped, good one reaped, no raise
    assert destroyed == [7]

    # sweep_orphans walks the same list (no active/known protection here) and must behave the same.
    destroyed.clear()
    assert vast.sweep_orphans(active_labels=set()) == [7]
    assert destroyed == [7]
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
