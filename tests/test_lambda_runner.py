"""Lambda Cloud run lifecycle: cloud-init/bootstrap, region walk, poll state machine, guaranteed
terminate, orphan sweep, capacity-aware allocation (CPU-only; lambda API + HF readers mocked).

Lambda is opt-in via LAMBDA_API_KEY (the autouse offline fixture deletes it); these tests mock the
lambda API entirely, so no key is needed — except the allocator tests, which set it to make the
provider "available" and then mock the capacity lookup.
"""

from __future__ import annotations

import base64
import io
import itertools
import json
import time

import pytest

from flash.spec import JobSpec


def _spec(gpu_type="A10", **gpu_kw) -> JobSpec:
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


def _inst(gpu="A10", region="us-east-1", itype="gpu_1x_a10", price=1.29):
    from flash.providers.lambdalabs.jobs.builders import LambdaInstance

    return LambdaInstance(gpu=gpu, instance_type=itype, region=region, vram_gb=24, price_usd_hr=price)


def _handle(started_ts=10_000.0, rate=1.29):
    from flash.providers.lambdalabs.jobs.builders import LambdaJobHandle

    return LambdaJobHandle(
        instance_id="i-9999",
        instance_type="gpu_1x_a10",
        region="us-east-1",
        name="flash-x-s0-a0",
        gpu="A10",
        hourly_usd=rate,
        attempt=0,
        started_ts=started_ts,
    )


# ---------------------------------------------------------------------------
# cloud-init user_data + bootstrap
# ---------------------------------------------------------------------------
def test_user_data_ships_payload_and_runs_worker_image(monkeypatch):
    from flash.providers.lambdalabs.jobs import builders

    monkeypatch.setenv("LAMBDA_API_KEY", "lk-supersecret")
    monkeypatch.setenv("HF_TOKEN", "hf-worker-token")
    payload = builders.build_payload(_spec(), seed=0, attempt=1)
    assert payload["phase"] == "sft"
    assert payload["attempt"] == 1
    assert payload["hf_prefix"] == "sft/flash-1700000000-abcd1234/seed0"
    assert payload["max_wall_s"] == 3600
    assert payload["hf_repo"] == "org/repo"
    # The worker env's HF_REPO is sourced from the run's [train] hf_repo (not an operator default).
    assert payload["env"]["HF_REPO"] == "org/repo"

    script = builders.build_user_data(payload)
    # payload travels base64-encoded inside a quoted heredoc, byte-exact
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    assert json.loads(base64.b64decode(b64)) == payload
    # the self-contained bootstrap is embedded
    assert "FLASH_BOOTSTRAP_EOF" in script
    assert "metrics.json" in script
    # runs the prebuilt WORKER_IMAGE via Docker with the GPU + the bootstrap as the command
    from flash.providers.runpod.train import WORKER_IMAGE

    assert WORKER_IMAGE in script
    assert "docker run -d" in script
    assert "--gpus all" in script
    assert "/root/flash/bootstrap.py" in script
    # waits for docker + gpu before launching (cloud-init can beat them to ready)
    assert "waiting for docker+gpu" in script
    # the operator's Lambda key NEVER ships to the box (no instance-scoped key, teardown is
    # control-plane-side). The worker HF token IS carried — inside the base64 payload's env (like
    # RunPod's worker env), never interpolated raw into the shell.
    assert "lk-supersecret" not in script
    assert payload["env"]["HF_TOKEN"] == "hf-worker-token"


def test_user_data_skips_capacity_for_baked_image_default(monkeypatch):
    """build_user_data always uses the baked WORKER_IMAGE (no per-host stack install)."""
    from flash.providers.lambdalabs.jobs import builders

    payload = builders.build_payload(_spec(), seed=0, attempt=0)
    script = builders.build_user_data(payload)
    # No base training-stack pip install in the cloud-init (the image is baked); only the worker
    # container's own per-run extra_pip runs (inside _bootstrap, not the host script).
    assert "torch==2.10.0" not in script


def _bootstrap_env(monkeypatch, phase="sft", rc=0, metrics=True):
    from flash.providers import _instance_bootstrap as lb

    calls: list[str] = []
    markers: list[tuple[bool, str]] = []
    monkeypatch.setattr(
        lb,
        "load_payload",
        lambda path=lb.PAYLOAD_PATH: {
            "hf_repo": "org/repo",
            "job_spec_json": "{}",
            "phase": phase,
            "seed": 0,
            "flash_arm": "lambda",
            "env": {},
            "extra_pip": [],
            "hf_prefix": "sft/x/seed0",
            "max_wall_s": 60,
            "attempt": 0,
        },
    )
    monkeypatch.setattr(lb, "fetch_code", lambda p: None)
    monkeypatch.setattr(lb, "run_mode", lambda p, e, m, d: (calls.append(m), rc)[1])
    monkeypatch.setattr(lb, "write_attempt_marker", lambda p, ok, error="": markers.append((ok, error)))
    monkeypatch.setattr(lb.os.path, "exists", lambda p: metrics if "metrics.json" in p else False)
    return lb, calls, markers


def test_bootstrap_train_success(monkeypatch):
    lb, calls, markers = _bootstrap_env(monkeypatch)
    assert lb.main() == 0
    assert calls == ["sft"]  # one fresh worker process
    assert markers == [(True, "")]


def test_bootstrap_fails_without_metrics(monkeypatch):
    lb, _calls, markers = _bootstrap_env(monkeypatch, metrics=False)
    assert lb.main() == 1
    ok, error = markers[0]
    assert not ok
    assert "metrics.json" in error


def test_bootstrap_sets_lambda_arm():
    """The shared bootstrap stamps FLASH_ARM from payload['flash_arm'] so the metrics record
    attributes the substrate (Lambda's build_payload sets it to 'lambda')."""
    from flash.providers import _instance_bootstrap as lb

    env = lb.build_worker_env({"job_spec_json": "{}", "phase": "sft", "seed": 0, "env": {}, "flash_arm": "lambda"})
    assert env["FLASH_ARM"] == "lambda"
    # And Lambda's build_payload is what sets flash_arm='lambda'.
    from flash.providers.lambdalabs.jobs.builders import build_payload
    assert build_payload(_spec(), 0, 0)["flash_arm"] == "lambda"


# ---------------------------------------------------------------------------
# launch_and_submit: capacity (region) walk
# ---------------------------------------------------------------------------
def test_launch_walks_regions_on_capacity_rejection(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    attempts = []

    def fake_launch(*, region_name, instance_type_name, ssh_key_names, name, user_data):
        attempts.append(region_name)
        if len(attempts) < 3:
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: insufficient-capacity")
        return "i-4242"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    insts = [_inst(region=r) for r in ("us-east-1", "us-west-1", "us-west-2")]
    h = jobs.launch_and_submit(_spec(), seed=0, instances=insts, attempt=2)
    assert attempts == ["us-east-1", "us-west-1", "us-west-2"]
    assert h.instance_id == "i-4242"
    assert h.region == "us-west-2"
    assert h.gpu == "A10"
    assert h.name == "flash-1700000000-abcd1234-s0-a2"


def test_launch_refreshes_capacity_once_when_all_taken(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    created = []

    def fake_launch(*, region_name, **kw):
        if region_name != "us-fresh-1":
            raise lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: no capacity")
        created.append(region_name)
        return "i-7"

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [_inst(region="us-fresh-1")])
    h = jobs.launch_and_submit(_spec(), seed=0, instances=[_inst(region="us-east-1")], attempt=0)
    assert created == ["us-fresh-1"]
    assert h.instance_id == "i-7"


def test_launch_raises_when_no_capacity(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    monkeypatch.setattr(
        lambda_api,
        "launch_instance",
        lambda **k: (_ for _ in ()).throw(lambda_api.LambdaApiError("PUT /asks/1/ -> HTTP 400: no capacity")),
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [])
    with pytest.raises(lambda_api.LambdaApiError, match="no capacity"):
        jobs.launch_and_submit(_spec(), seed=0, instances=[_inst()], attempt=0)
    with pytest.raises(lambda_api.LambdaApiError, match="no Lambda capacity"):
        jobs.launch_and_submit(_spec(), seed=0, instances=[], attempt=0)


def test_resolve_ssh_key_names(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs.jobs import resolve_ssh_key_names

    monkeypatch.setenv("LAMBDA_SSH_KEY_NAME", "pinned-key")
    assert resolve_ssh_key_names() == ["pinned-key"]
    monkeypatch.delenv("LAMBDA_SSH_KEY_NAME", raising=False)
    monkeypatch.setattr(lambda_api, "list_ssh_keys", lambda: [{"name": "jk"}, {"name": "other"}])
    assert resolve_ssh_key_names() == ["jk"]  # first registered key
    monkeypatch.setattr(lambda_api, "list_ssh_keys", lambda: [])
    with pytest.raises(lambda_api.LambdaApiError, match="requires an SSH key"):
        resolve_ssh_key_names()


# ---------------------------------------------------------------------------
# poll_lambda_job state machine
# ---------------------------------------------------------------------------
def _wire_poll(monkeypatch, instances, done=None, marker=None, metrics=None, boot=None, step=10.0):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    seq = iter(instances)
    last = {"inst": None}

    def fake_get(instance_id):
        last["inst"] = next(seq, last["inst"])
        return last["inst"]

    monkeypatch.setattr(lambda_api, "get_instance", fake_get)
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=10_000, step=step)
    monkeypatch.setattr(jobs.time, "time", lambda: float(next(clock)))

    def factory(hf_repo, path, min_interval_s=45.0):
        def read(force=False):
            if path.endswith("/DONE"):
                return done() if callable(done) else done
            if "lambda_attempt" in path:
                return marker() if callable(marker) else marker
            if path.endswith("metrics.json"):
                return metrics() if callable(metrics) else metrics
            if path.endswith("lambda_boot.log"):
                return boot() if callable(boot) else boot
            return None

        return read

    monkeypatch.setattr(jobs, "_make_hf_file_reader", factory)
    return jobs


def test_poll_success_stamps_real_cost(monkeypatch):
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="10500.0",
        metrics=json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0}),
    )
    # started_ts precedes the mocked clock (starts 10_000) so wall is positive on the first tick.
    res = jobs.poll_lambda_job(_handle(started_ts=9_000.0), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["train_tokens"] == 4096
    # cost comes from the instance's real $/hr x wall time, not a runpod table rate
    assert res.metrics["cost_usd"] > 0
    assert res.metrics["notes"]["provider"] == "lambda"
    assert res.metrics["notes"]["lambda_rate_usd_hr"] == 1.29
    assert res.metrics["notes"]["lambda_region"] == "us-east-1"


def test_poll_caps_recovered_cost_at_done_timestamp(monkeypatch):
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="9100.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
    )
    res = jobs.poll_lambda_job(_handle(started_ts=9000.0), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["cost_usd"] == round((9100.0 - 9000.0) / 3600.0 * 1.29, 6)


def test_poll_marker_failure_is_job_failed(monkeypatch):
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RuntimeError: boom"}),
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"  # real worker error fails fast
    assert "boom" in res.detail


def test_poll_retriable_marker_is_job_preempted(monkeypatch):
    """A worker-flagged retriable failure retries on a fresh host (job_preempted), not job_failed."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "transient"}),
    )
    res = jobs.poll_lambda_job(
        _handle(), _spec(), seed=0, interval_s=0, heartbeat_reader=lambda force=False: {"retriable": True}
    )
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_dead_host_without_marker_is_preempted(monkeypatch):
    """A host that died without writing DONE/marker is a host loss -> retryable, with the boot log."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}, {"status": "terminated"}],
        boot="+ docker pull ...\nFLASH: gpu never became ready",
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_preempted"
    assert "gpu never became ready" in res.detail  # the host boot log is the only console window


def test_poll_loading_timeout(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "booting"}], step=100.0)
    monkeypatch.setattr(jobs, "LOAD_TIMEOUT_S", 300.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "never became active" in res.detail


def test_poll_heartbeat_stall(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    frozen = {"stage": "rl", "step": 3, "ts": 1.0}
    res = jobs.poll_lambda_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: frozen,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "no worker progress" in res.detail


def test_poll_recovery_seeds_load_clock_from_launch(monkeypatch):
    """Reattach after a control-plane restart: a still-booting box has been billing since LAUNCH
    (handle.started_ts), so LOAD_TIMEOUT_S is measured from launch, NOT from this poll's first
    tick. A box already past the load window fails over on the first reattach iteration instead of
    getting another full window. (The mocked clock starts at 10_000; launch was 5000s earlier.)"""
    import re

    jobs = _wire_poll(monkeypatch, instances=[{"status": "booting"}], step=10.0)
    res = jobs.poll_lambda_job(_handle(started_ts=5_000.0), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "never became active" in res.detail
    m = re.search(r"for (\d+)s", res.detail)
    assert m is not None, res.detail
    # launch-relative (~5000s); the old "reset to reattach tick" code would report ~LOAD_TIMEOUT_S.
    assert int(m.group(1)) >= 2000, res.detail


def test_poll_missing_started_ts_anchors_to_now_not_epoch(monkeypatch):
    """started_ts is a non-Optional float coerced to 0.0 when MISSING (old/corrupt handle), so 0.0
    means 'unknown launch' (a real launch is a large epoch ts). EVERYTHING (the timeout clocks AND
    done_is_fresh / finish_ok's wall+cost stamping) must anchor to now, NOT the epoch — otherwise a
    booting box would be 'past' a ~57-year-old load window and stall on the first tick, and wall/cost
    would be billed from 1970. DONE then completes the run normally with a sane (tiny) wall."""
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done="10500.0",
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
        step=10.0,
    )
    res = jobs.poll_lambda_job(_handle(started_ts=0.0), _spec(), seed=0, interval_s=0)
    assert res.ok, res  # not instantly stalled by an epoch-anchored deadline/load clock
    # wall/cost are NOT billed from the 1970 epoch: launch_ts fell back to now (~10_000 mocked),
    # so the stamped cost is a few seconds of wall, not ~57 years x $1.29/hr (= astronomically large).
    assert res.metrics["cost_usd"] < 1.0, res.metrics["cost_usd"]


def test_heartbeat_progress_ts_unknown_launch_treats_heartbeats_as_fresh():
    """When launch is UNKNOWN (launch_ts=0.0, from a recovered handle missing started_ts), the
    clamp floor must drop to 0.0 so a normal heartbeat — timestamped before it is read, i.e. < now —
    counts as FRESH and credits its own ts. Flooring to `now` would mark every such heartbeat stale
    and stall a healthy recovered worker after SETUP_GRACE_S despite continuous heartbeats."""
    from flash.providers._poll import heartbeat_progress_ts

    hb_ts = time.time() - 30.0  # a normal recent heartbeat, slightly in the past
    ts, fresh = heartbeat_progress_ts(("rl", 4, hb_ts), launch_ts=0.0)
    assert fresh is True  # unknown launch -> not discarded
    assert abs(ts - hb_ts) < 1.0  # credits the heartbeat's own ts (not clamped up to now)

    # A real (non-zero) launch still discriminates prior-attempt leftovers (ts < launch).
    launch = time.time() - 100.0
    _, fresh_old = heartbeat_progress_ts(("rl", 1, launch - 50.0), launch_ts=launch)
    assert fresh_old is False
    _, fresh_new = heartbeat_progress_ts(("rl", 9, launch + 10.0), launch_ts=launch)
    assert fresh_new is True


def test_poll_stale_heartbeat_does_not_buy_fresh_window(monkeypatch):
    """A heartbeat that was already stale before a restart must not reset the stall clock to the
    reattach time: its OWN ts is credited as last-progress, so an active worker frozen long ago
    stalls promptly instead of getting another full stall window. (Clock starts 10_000; the
    worker's last heartbeat was at 8500, launch at 8000, stall budget 500s.)"""
    import re

    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=10.0)
    hb = {"stage": "rl", "step": 7, "ts": 8500.0}
    res = jobs.poll_lambda_job(
        _handle(started_ts=8_000.0),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: hb,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "no worker progress" in res.detail
    m = re.search(r"for (\d+)s", res.detail)
    assert m is not None, res.detail
    # measured from the heartbeat ts (~1500s+), not the reattach tick (which the old code used,
    # yielding only ~stall_after_s).
    assert int(m.group(1)) >= 1000, res.detail


def test_poll_prior_attempt_heartbeat_does_not_arm_training_stall(monkeypatch):
    """A LEFTOVER heartbeat from a PRIOR attempt (ts < this attempt's launch; retries reuse the same
    seed heartbeat path) must not be treated as current progress. Clamping its ts up to launch made
    a stale training-stage heartbeat arm the tighter training stall window and fail a healthy new
    attempt mid-setup before it overwrote the file. With the freshness gate, a pre-launch heartbeat
    neither advances last_progress nor sets seen_training_hb, so the run gets the longer SETUP grace
    measured from launch. (Clock starts 10_000; launch 9000; old heartbeat ts 8000 < launch.)"""
    import re

    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=10.0)
    stale = {"stage": "rl", "step": 2, "ts": 8000.0}  # training stage, but predates this launch
    res = jobs.poll_lambda_job(
        _handle(started_ts=9_000.0),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda force=False: stale,
        setup_grace_s=3000.0,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    # Stalls on SETUP grace (3000s from launch), not the tighter 500s training window the stale
    # heartbeat would have armed -> the reported idle time exceeds the training budget.
    assert "setup (pre-training)" in res.detail
    m = re.search(r"for (\d+)s", res.detail)
    assert m is not None, res.detail
    assert int(m.group(1)) >= 3000, res.detail


def test_poll_client_deadline(monkeypatch):
    jobs = _wire_poll(monkeypatch, instances=[{"status": "active"}], step=100.0)
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0, deadline_s=250.0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


def test_provider_poll_passes_full_launch_relative_deadline(monkeypatch):
    """The reattach path must NOT pre-subtract elapsed-since-launch from the deadline: the poll loop
    already anchors its deadline check to handle.started_ts (= launch), so subtracting elapsed here
    too double-counts and tears down a still-valid instance once a recovered run is past half its
    window. LambdaProvider.poll must pass the FULL launch-relative budget regardless of how old
    started_ts is."""
    from flash.providers.base import JobHandle
    from flash.providers.lambdalabs import LambdaProvider
    from flash.providers.lambdalabs.jobs import PROVISION_GRACE_S

    captured = {}

    def fake_poll(handle, spec, seed, *, log=None, heartbeat_reader=None, deadline_s=None):
        captured["deadline_s"] = deadline_s
        from flash.providers.base import PollResult

        return PollResult(True)

    monkeypatch.setattr("flash.providers.lambdalabs.jobs.poll_lambda_job", fake_poll)
    monkeypatch.setattr("flash.providers.lambdalabs.api.terminate_instances", lambda ids: ids)
    spec = _spec()  # max_wall_seconds=3600
    # started_ts long in the past (recovered well past half its window).
    handle = JobHandle.from_dict({"provider": "lambda", **_handle(started_ts=1.0).to_dict()})
    LambdaProvider().poll(handle, spec, seed=0)
    assert captured["deadline_s"] == max(60.0, 3600 + PROVISION_GRACE_S)


def test_poll_surfaces_worker_progress_in_log(monkeypatch):
    # DONE appears only on the 2nd poll, so the loop reaches the heartbeat-surfacing block first.
    done_seq = iter([None, "10500.0", "10500.0", "10500.0"])
    jobs = _wire_poll(
        monkeypatch,
        instances=[{"status": "active"}],
        done=lambda: next(done_seq, "10500.0"),
        metrics=json.dumps({"wall_seconds": 100, "cost_usd": 0.0}),
    )
    log = io.StringIO()
    hb = {"stage": "sft", "step": 5, "ts": 2.0, "loss": 1.5}
    res = jobs.poll_lambda_job(
        _handle(), _spec(), seed=0, interval_s=0, log=log, heartbeat_reader=lambda force=False: hb
    )
    assert res.ok
    assert "stage=sft" in log.getvalue()


# ---------------------------------------------------------------------------
# the cost-safety invariant: every exit path terminates the instance
# ---------------------------------------------------------------------------
def _wire_runner(monkeypatch, poll_outcome):
    from flash.providers.base import PollResult
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.append(list(ids)) or True
    )
    monkeypatch.setattr(jobs, "usable_instances", lambda gpu, force=False: [_inst()])
    monkeypatch.setattr(jobs, "launch_and_submit", lambda *a, **k: _handle())

    def fake_poll(*a, **k):
        if isinstance(poll_outcome, BaseException):
            raise poll_outcome
        return poll_outcome

    monkeypatch.setattr(jobs, "poll_lambda_job", fake_poll)
    return jobs, terminated, PollResult


def test_runner_terminates_on_success(monkeypatch):
    from flash.providers.base import PollResult

    jobs, terminated, _ = _wire_runner(monkeypatch, PollResult(True, metrics={"a": 1}))
    handles = []
    res = jobs.submit_run_lambda(_spec(), seed=0, on_handle=handles.append)
    assert res.ok
    assert terminated == [["i-9999"]]
    assert handles
    assert handles[0]["provider"] == "lambda"
    assert handles[0]["instance_id"] == "i-9999"


def test_runner_terminates_on_failure_and_exception(monkeypatch):
    from flash.providers.base import PollResult

    jobs, terminated, _ = _wire_runner(monkeypatch, PollResult(False, failure="stalled"))
    res = jobs.submit_run_lambda(_spec(), seed=0)
    assert not res.ok
    assert terminated == [["i-9999"]]

    jobs, terminated, _ = _wire_runner(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        jobs.submit_run_lambda(_spec(), seed=0)
    assert terminated == [["i-9999"]]


def test_runner_terminates_when_handle_persist_fails(monkeypatch):
    """The launched instance is terminated even if on_handle raises — the teardown finally guards
    everything after the launch, not just the poll."""
    jobs, terminated, _ = _wire_runner(monkeypatch, None)

    def boom(_h):
        raise RuntimeError("status store unreachable")

    with pytest.raises(RuntimeError, match="status store unreachable"):
        jobs.submit_run_lambda(_spec(), seed=0, on_handle=boom)
    assert terminated == [["i-9999"]]


def test_submit_rejects_policy_word_gpu():
    """submit_run_lambda needs a concrete class; a policy word ("cheapest") — which the allocator
    resolves upstream — must fail with a clear error, not an opaque KeyError."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs.jobs import submit_run_lambda

    with pytest.raises(lambda_api.LambdaApiError, match="concrete gpu class"):
        submit_run_lambda(_spec(gpu_type="cheapest"), seed=0)


# ---------------------------------------------------------------------------
# labels, gc, orphan sweep
# ---------------------------------------------------------------------------
def test_instance_label_always_sweepable():
    from flash.providers.lambdalabs.jobs.builders import instance_label

    assert instance_label("flash-1700-abcd", 0, 1) == "flash-1700-abcd-s0-a1"
    assert instance_label("fail-fast", 0, 0) == "flash-fail-fast-s0-a0"  # prefix forced


def test_terminate_run_instances_matches_forced_prefix(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    instances = [
        {"id": "i-1", "name": "flash-fail-fast-s0-a0"},  # forced-prefix name
        {"id": "i-2", "name": "flash-other-run-s0-a0"},  # different run -> keep
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    assert jobs.terminate_run_instances("fail-fast") == ["i-1"]
    assert terminated == ["i-1"]


def test_handle_roundtrip():
    from flash.providers.lambdalabs.jobs.builders import LambdaJobHandle

    h = _handle()
    d = h.to_dict()
    assert d["provider"] == "lambda"
    assert LambdaJobHandle.from_dict(d) == h


def test_sweep_orphans_label_safety(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    instances = [
        {"id": "i-1", "name": "flash-1700-aaaa-s0-a0"},  # orphan -> terminate
        {"id": "i-2", "name": "flash-1700-bbbb-s0-a1"},  # active run -> keep
        {"id": "i-3", "name": "someone-elses-workload"},  # not ours -> NEVER touch
        {"id": "i-4", "name": ""},  # unnamed -> NEVER touch
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels={"flash-1700-bbbb"})
    assert out == ["i-1"]
    assert terminated == ["i-1"]


def test_sweep_orphans_prefix_not_shielded_by_longer_run_id(monkeypatch):
    """A live run id that is a STRING prefix of another must not shield the other's orphan."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    instances = [
        {"id": "i-1", "name": jobs.instance_label("flash-100", 0, 0)},  # live -> KEEP
        {"id": "i-2", "name": jobs.instance_label("flash-1000", 0, 0)},  # orphan -> terminate
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels={"flash-100"})
    assert out == ["i-2"]


def test_sweep_orphans_protects_unprefixed_active_run_id(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    instances = [
        {"id": "i-1", "name": jobs.instance_label("fail-fast", 0, 0)},  # live run -> KEEP
        {"id": "i-2", "name": jobs.instance_label("orphan-run", 0, 0)},  # no live run -> terminate
    ]
    terminated = []
    monkeypatch.setattr(lambda_api, "list_instances", lambda: instances)
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    out = jobs.sweep_orphans(active_labels={"fail-fast"})  # RAW run id (what the server tracks)
    assert out == ["i-2"]


# ---------------------------------------------------------------------------
# provider object dispatch + capacity-aware allocation
# ---------------------------------------------------------------------------
def test_provider_cancel_destroy_terminate(monkeypatch):
    from flash.providers import get_provider
    from flash.providers.base import JobHandle
    from flash.providers.lambdalabs import api as lambda_api

    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    h = JobHandle("lambda", {"instance_id": "i-9"})
    get_provider("lambda").cancel(h)
    get_provider("lambda").destroy(h)
    assert terminated == ["i-9", "i-9"]


def test_usable_instances_only_capacity_regions(monkeypatch):
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs.jobs import usable_instances

    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda itype, force=False: ["us-east-1", "us-west-1"])
    monkeypatch.setattr("flash.providers.lambdalabs.pricing.hourly_rate", lambda g: 1.29)
    out = usable_instances("A10")
    assert {i.region for i in out} == {"us-east-1", "us-west-1"}
    assert all(i.gpu == "A10" and i.instance_type == "gpu_1x_a10" for i in out)
    # no capacity -> empty (the allocator then skips the class)
    monkeypatch.setattr(lambda_api, "regions_with_capacity", lambda itype, force=False: [])
    assert usable_instances("A10") == []


def test_allocator_capacity_aware(monkeypatch):
    """Lambda joins the ranked candidate list only for classes with LIVE capacity; a class with no
    capacity is excluded so the runner never walks to a class that would immediately fail to launch."""
    from flash.providers import allocator
    from flash.providers.lambdalabs.jobs.builders import LambdaInstance

    monkeypatch.setenv("LAMBDA_API_KEY", "lk")  # make lambda "available"

    def fake_usable(gpu):
        # A10 has capacity; A100 SXM 40GB does not (excluded from candidates).
        if gpu == "A10":
            return [LambdaInstance("A10", "gpu_1x_a10", "us-east-1", 24, 1.29)]
        return []

    monkeypatch.setattr("flash.providers.lambdalabs.jobs.usable_instances", fake_usable)
    a = allocator.allocate("Qwen/Qwen3.5-0.8B", "sft")
    lam = {c.gpu for c in a.candidates if c.provider == "lambda"}
    assert lam == {"A10"}  # only the in-capacity class
    # RunPod still wins on price (cheaper static rates), so it's the chosen provider.
    assert a.provider == "runpod"


# --- review-fix regressions ---
def test_poll_ok_marker_succeeds_with_stale_done(monkeypatch):
    """A retry that hits the worker's already-complete path leaves DONE stale but writes ok marker +
    metrics; the poller must treat that as SUCCESS, not poll until it stalls."""
    jobs = _wire_poll(
        monkeypatch, instances=[{"status": "active"}],
        done="9000.0",  # STALE (before the handle's started_ts=10000)
        marker=json.dumps({"ok": True, "attempt": 0}),
        metrics=json.dumps({"wall_seconds": 50, "cost_usd": 0.0}),
    )
    res = jobs.poll_lambda_job(_handle(), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["notes"]["provider"] == "lambda"


def test_ambiguous_launch_reconciles_and_stops(monkeypatch):
    """An ambiguous launch failure (timeout/5xx, maybe created an instance) must NOT walk to another
    region — it reconciles by name and raises so the run retries cleanly (cost safety)."""
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs

    monkeypatch.setattr(jobs, "resolve_ssh_key_names", lambda: ["jk"])
    reaped = []
    monkeypatch.setattr(jobs, "terminate_run_instances", lambda rid: reaped.append(rid) or [])
    attempts = []

    def fake_launch(**k):
        attempts.append(k["region_name"])
        raise lambda_api.LambdaApiError("PUT /asks/1/ failed after 5 attempts: timed out")

    monkeypatch.setattr(lambda_api, "launch_instance", fake_launch)
    insts = [_inst(region=r) for r in ("us-east-1", "us-west-1")]
    with pytest.raises(lambda_api.LambdaApiError, match="ambiguous"):
        jobs.launch_and_submit(_spec(), seed=0, instances=insts, attempt=0)
    assert attempts == ["us-east-1"]  # stopped after the first ambiguous failure (no 2nd launch)
    assert reaped == ["flash-1700000000-abcd1234"]  # reconciled by run-name
