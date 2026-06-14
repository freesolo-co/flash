"""Vast run lifecycle: onstart/bootstrap, offer walk, poll state machine, guaranteed
destroy, orphan sweep (CPU-only; vast API + HF artifact readers mocked)."""

from __future__ import annotations

import base64
import itertools
import json

import pytest

from autoslm.worker_spec import JobSpec


def _spec(**gpu_kw) -> JobSpec:
    gpu = {"type": "RTX 3090", "provider": "vast", "max_wall_seconds": 3600, **gpu_kw}
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-0.6B",
            "algorithm": "sft",
            "run_id": "autoslm-1700000000-abcd1234",
            "train": {"epochs": 1, "seeds": [0]},
            "gpu": gpu,
        }
    )


def _offer_obj(offer_id=1, machine_id=10, gpu="RTX 3090", dph=0.25):
    from autoslm.providers.vast import VastOffer

    return VastOffer(
        offer_id=offer_id,
        machine_id=machine_id,
        gpu=gpu,
        vram_gb=24,
        dph_total=dph,
        cuda_max_good=12.8,
        disk_space=200.0,
        reliability=0.99,
        inet_down=5000.0,
        geolocation="CZ",
    )


# ---------------------------------------------------------------------------
# onstart + bootstrap
# ---------------------------------------------------------------------------
def test_onstart_ships_payload_and_bootstrap(monkeypatch):
    from autoslm.providers import vast

    monkeypatch.setenv("VAST_API_KEY", "vk-supersecret")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-supersecret")
    monkeypatch.setenv("HF_REPO", "org/repo")
    payload = vast.build_payload(_spec(), seed=0, attempt=1)
    assert payload["phase"] == "sft"
    assert payload["attempt"] == 1
    assert payload["hf_prefix"] == "sft/autoslm-1700000000-abcd1234/seed0"
    assert payload["max_wall_s"] == 3600

    script = vast.build_onstart(payload, install_deps=True)
    # payload travels base64-encoded inside a quoted heredoc, byte-exact
    b64 = script.split("AUTOSLM_PAYLOAD_EOF")[1].strip()
    assert json.loads(base64.b64decode(b64)) == payload
    # the self-contained bootstrap is embedded and keeps _train_body's semantics
    assert "AUTOSLM_BOOTSTRAP_EOF" in script
    assert "metrics.json" in script
    # worker deps install (skippable with a baked image)
    assert "pip install" in script
    assert "torch==2.10.0" in script
    # a base-deps install failure STOPS the script before bootstrap.py (no partial-env
    # crash), holding the box for log retrieval — mirrors the bootstrap-failure path.
    assert "base worker dependency install failed" in script
    pip_idx = script.index("pip install")
    boot_idx = script.index("/root/autoslm/bootstrap.py\n")  # the run, not the heredoc write
    guard_idx = script.index("base worker dependency install failed")
    assert pip_idx < guard_idx < boot_idx, "guard must sit between base pip and bootstrap run"
    assert "pip install" not in vast.build_onstart(payload, install_deps=False).replace(
        "extra_pip", ""
    )
    # verified live: args-mode wrapper resets PATH (OS python is PEP 668-managed)
    assert "PIP_BREAK_SYSTEM_PACKAGES=1" in script
    assert "/opt/conda/bin/python" in script  # prefer the image's torch-equipped env
    # self-destroy backstop uses the Vast-injected instance key only
    assert "CONTAINER_API_KEY" in script
    assert "CONTAINER_ID" in script
    # operator keys NEVER ship to the box
    assert "vk-supersecret" not in script
    assert "rp-supersecret" not in script


def _bootstrap_env(monkeypatch, phase="sft", rcs=(0,), metrics=True):
    from autoslm.providers import vast_bootstrap as vb

    calls: list[str] = []
    markers: list[tuple[bool, str]] = []
    rc_iter = iter(rcs)
    monkeypatch.setattr(
        vb,
        "load_payload",
        lambda path=vb.PAYLOAD_PATH: {
            "hf_repo": "org/repo",
            "job_spec_json": "{}",
            "phase": phase,
            "seed": 0,
            "env": {},
            "extra_pip": [],
            "hf_prefix": "sft/x/seed0",
            "max_wall_s": 60,
            "attempt": 0,
        },
    )
    monkeypatch.setattr(vb, "fetch_code", lambda p: None)
    monkeypatch.setattr(vb, "run_mode", lambda p, e, m, d: (calls.append(m), next(rc_iter))[1])
    monkeypatch.setattr(
        vb, "write_attempt_marker", lambda p, ok, error="": markers.append((ok, error))
    )
    # The train phase writes /tmp/metrics.json itself (or restores it from an earlier
    # attempt's DONE sentinel); its presence is the success signal.
    monkeypatch.setattr(vb.os.path, "exists", lambda p: metrics if "metrics.json" in p else False)
    return vb, calls, markers


def test_bootstrap_train(monkeypatch):
    vb, calls, markers = _bootstrap_env(monkeypatch)
    assert vb.main() == 0
    assert calls == ["sft"]  # one fresh process; no separate eval phase
    assert markers == [(True, "")]


def test_bootstrap_fails_without_metrics(monkeypatch):
    vb, calls, markers = _bootstrap_env(monkeypatch, metrics=False)
    assert vb.main() == 1
    assert calls == ["sft"]
    ok, error = markers[0]
    assert not ok
    assert "metrics.json" in error


# ---------------------------------------------------------------------------
# deploy_and_submit: live-market offer walk
# ---------------------------------------------------------------------------
def test_deploy_walks_taken_offers(monkeypatch):
    from autoslm.providers import vast, vast_api

    monkeypatch.setenv("HF_REPO", "org/repo")
    attempts = []

    def fake_create(offer_id, **kw):
        attempts.append((offer_id, kw["label"]))
        if len(attempts) < 3:
            raise vast_api.VastApiError("no such ask (taken)")
        return 9999

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    offers = [_offer_obj(offer_id=i, machine_id=i, dph=0.2 + i / 100) for i in (1, 2, 3)]
    h = vast.deploy_and_submit(_spec(), seed=0, offers=offers, attempt=2)
    assert [a[0] for a in attempts] == [1, 2, 3]
    assert h.instance_id == 9999
    assert h.offer_id == 3
    assert h.gpu == "RTX 3090"
    assert h.label == "autoslm-1700000000-abcd1234-s0-a2"


def test_deploy_refreshes_once_when_all_taken(monkeypatch):
    from autoslm.providers import vast, vast_api

    monkeypatch.setenv("HF_REPO", "org/repo")
    created = []

    def fake_create(offer_id, **kw):
        if offer_id < 100:
            raise vast_api.VastApiError("taken")
        created.append(offer_id)
        return 4242

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    monkeypatch.setattr(
        vast, "usable_offers", lambda *a, **k: [_offer_obj(offer_id=100, machine_id=50)]
    )
    h = vast.deploy_and_submit(_spec(), seed=0, offers=[_offer_obj(offer_id=1)], attempt=0)
    assert created == [100]
    assert h.instance_id == 4242


def test_deploy_raises_when_pool_exhausted(monkeypatch):
    from autoslm.providers import vast, vast_api

    monkeypatch.setenv("HF_REPO", "org/repo")
    monkeypatch.setattr(
        vast_api,
        "create_instance",
        lambda *a, **k: (_ for _ in ()).throw(vast_api.VastApiError("taken")),
    )
    monkeypatch.setattr(vast, "usable_offers", lambda *a, **k: [])
    with pytest.raises(vast_api.VastApiError, match="rejected"):
        vast.deploy_and_submit(_spec(), seed=0, offers=[_offer_obj()], attempt=0)
    with pytest.raises(vast_api.VastApiError, match="no usable vast offers"):
        vast.deploy_and_submit(_spec(), seed=0, offers=[], attempt=0)


# ---------------------------------------------------------------------------
# poll_vast_job state machine
# ---------------------------------------------------------------------------
def _wire_poll(monkeypatch, instances, done=None, marker=None, metrics=None, step=10.0):
    """Mock the instance status sequence + the HF artifact readers + the clock."""
    from autoslm.providers import vast, vast_api

    seq = iter(instances)
    last = {"inst": None}

    def fake_get(instance_id):
        last["inst"] = next(seq, last["inst"])
        return last["inst"]

    monkeypatch.setattr(vast_api, "get_instance", fake_get)
    monkeypatch.setattr(vast_api, "instance_logs", lambda iid, **kw: "+ pip install\nboom from pip")
    monkeypatch.setattr(vast.time, "sleep", lambda s: None)
    clock = itertools.count(start=10_000, step=step)
    monkeypatch.setattr(vast.time, "time", lambda: float(next(clock)))

    def factory(hf_repo, path, min_interval_s=45.0):
        def read(force=False):
            if path.endswith("/DONE"):
                return done() if callable(done) else done
            if "vast_attempt" in path:
                return marker() if callable(marker) else marker
            if path.endswith("metrics.json"):
                return metrics() if callable(metrics) else metrics
            return None

        return read

    monkeypatch.setattr(vast, "_make_hf_file_reader", factory)
    return vast


def _handle(started_ts=10_000.0, rate=0.25):
    from autoslm.providers.vast import VastJobHandle

    return VastJobHandle(
        instance_id=9999,
        offer_id=1,
        machine_id=10,
        label="autoslm-x-s0-a0",
        gpu="RTX 3090",
        hourly_usd=rate,
        attempt=0,
        started_ts=started_ts,
    )


def test_poll_success_stamps_real_cost(monkeypatch):
    monkeypatch.setenv("HF_REPO", "org/repo")
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        done="10500.0",  # fresh: after started_ts
        metrics=json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0}),
    )
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert res.ok
    assert res.metrics["train_tokens"] == 4096
    # cost comes from the OFFER's real rate x wall time, not a runpod table rate
    assert res.metrics["cost_usd"] > 0
    assert res.metrics["notes"]["provider"] == "vast"
    assert res.metrics["notes"]["vast_rate_usd_hr"] == 0.25


def test_poll_caps_recovered_cost_at_done_timestamp(monkeypatch):
    # Recovery case: the worker wrote DONE and stopped, but the control plane only
    # polls much later (the mocked clock is already past 10_000). Cost must be billed
    # to the DONE timestamp, NOT time.time() (which would add the downtime).
    monkeypatch.setenv("HF_REPO", "org/repo")
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        done="9100.0",  # 100s after the handle's started_ts, but BEFORE the poll clock
        metrics=json.dumps({"train_tokens": 4096, "wall_seconds": 100, "cost_usd": 0.0}),
    )
    res = vast.poll_vast_job(_handle(started_ts=9000.0), _spec(), seed=0, interval_s=0)
    assert res.ok
    # billed (9100 - 9000) = 100s at $0.25/hr, not (now - 9000) which would be hours
    assert res.metrics["cost_usd"] == round((9100.0 - 9000.0) / 3600.0 * 0.25, 6)


def test_poll_stale_done_is_ignored(monkeypatch):
    monkeypatch.setenv("HF_REPO", "org/repo")
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}, {"actual_status": "exited"}],
        done="9000.0",  # predates this attempt by > the clock-skew grace
        marker=json.dumps({"ok": False, "attempt": 0, "error": "RuntimeError: boom"}),
    )
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"
    assert "boom" in res.detail


def test_poll_attempt_marker_failure(monkeypatch):
    monkeypatch.setenv("HF_REPO", "org/repo")
    vast = _wire_poll(
        monkeypatch,
        instances=[{"actual_status": "running"}],
        marker=json.dumps({"ok": False, "attempt": 0, "error": "TimeoutError: wall cap"}),
    )
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"
    assert "wall cap" in res.detail


def test_poll_instance_vanished_without_done(monkeypatch):
    monkeypatch.setenv("HF_REPO", "org/repo")
    vast = _wire_poll(monkeypatch, instances=[None])
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "job_failed"
    # failure detail surfaces the container console (the only home of bootstrap errors)
    assert "boom from pip" in res.detail


def test_poll_loading_timeout(monkeypatch):
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "loading"}], step=100.0)
    monkeypatch.setattr(vast, "LOAD_TIMEOUT_S", 300.0)
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "loading" in res.detail


def test_poll_heartbeat_stall(monkeypatch):
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "running"}], step=100.0)
    frozen = {"stage": "rl", "step": 3, "ts": 1.0}
    res = vast.poll_vast_job(
        _handle(),
        _spec(),
        seed=0,
        interval_s=0,
        heartbeat_reader=lambda: frozen,
        stall_after_s=500.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "no worker progress" in res.detail


def test_poll_client_deadline(monkeypatch):
    vast = _wire_poll(monkeypatch, instances=[{"actual_status": "running"}], step=100.0)
    res = vast.poll_vast_job(_handle(), _spec(), seed=0, interval_s=0, deadline_s=250.0)
    assert not res.ok
    assert res.failure == "stalled"
    assert "deadline" in res.detail


# ---------------------------------------------------------------------------
# the cost-safety invariant: every exit path destroys the instance
# ---------------------------------------------------------------------------
def _wire_runner(monkeypatch, poll_outcome):
    from autoslm.flash.durable import PollResult
    from autoslm.providers import vast, vast_api

    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    monkeypatch.setattr(vast, "deploy_and_submit", lambda *a, **k: _handle())

    def fake_poll(*a, **k):
        if isinstance(poll_outcome, BaseException):
            raise poll_outcome
        return poll_outcome

    monkeypatch.setattr(vast, "poll_vast_job", fake_poll)
    return vast, destroyed, PollResult


def test_runner_destroys_on_success(monkeypatch):
    from autoslm.flash.durable import PollResult

    vast, destroyed, _ = _wire_runner(monkeypatch, PollResult(True, metrics={"a": 1}))
    handles = []
    res = vast.submit_train_durable_vast(
        _spec(), seed=0, on_handle=handles.append, offers=[_offer_obj()]
    )
    assert res.ok
    assert destroyed == [9999]
    assert handles
    assert handles[0]["provider"] == "vast"
    assert handles[0]["instance_id"] == 9999


def test_runner_destroys_on_failure_and_exception(monkeypatch):
    from autoslm.flash.durable import PollResult

    vast, destroyed, _ = _wire_runner(monkeypatch, PollResult(False, failure="stalled"))
    res = vast.submit_train_durable_vast(_spec(), seed=0, offers=[_offer_obj()])
    assert not res.ok
    assert destroyed == [9999]

    vast, destroyed, _ = _wire_runner(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        vast.submit_train_durable_vast(_spec(), seed=0, offers=[_offer_obj()])
    assert destroyed == [9999]


def test_instance_label_always_sweepable():
    from autoslm.providers.vast import instance_label

    # platform run ids pass through; anything else gets the prefix FORCED so the
    # orphan sweep can never miss an instance we rented (live incident: a unit
    # test's "fail-fast" run id produced unsweepable labels)
    assert instance_label("autoslm-1700-abcd", 0, 1) == "autoslm-1700-abcd-s0-a1"
    assert instance_label("fail-fast", 0, 0) == "autoslm-fail-fast-s0-a0"


def test_destroy_run_instances_matches_forced_prefix(monkeypatch):
    from autoslm.providers import vast, vast_api

    instances = [
        {"id": 1, "label": "autoslm-fail-fast-s0-a0"},  # forced-prefix label
        {"id": 2, "label": "autoslm-other-run-s0-a0"},  # different run -> keep
    ]
    destroyed = []
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    assert vast.destroy_run_instances("fail-fast") == [1]
    assert destroyed == [1]


def test_handle_roundtrip():
    from autoslm.providers.vast import VastJobHandle

    h = _handle()
    d = h.to_dict()
    assert d["provider"] == "vast"
    assert VastJobHandle.from_dict(d) == h


def test_sweep_orphans_label_safety(monkeypatch):
    from autoslm.providers import vast, vast_api

    instances = [
        {"id": 1, "label": "autoslm-1700-aaaa-s0-a0"},  # orphan -> destroy
        {"id": 2, "label": "autoslm-1700-bbbb-s0-a1"},  # active run -> keep
        {"id": 3, "label": "someone-elses-workload"},  # not ours -> NEVER touch
        {"id": 4, "label": ""},  # unlabeled -> NEVER touch
    ]
    destroyed = []
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.sweep_orphans(active_labels={"autoslm-1700-bbbb"})
    assert out == [1]
    assert destroyed == [1]
