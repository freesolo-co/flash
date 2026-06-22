"""Vast run lifecycle: onstart/bootstrap, offer walk, poll state machine, guaranteed
destroy, orphan sweep (CPU-only; vast API + HF artifact readers mocked)."""

from __future__ import annotations

import base64
import itertools
import json

import pytest

from flash.spec import JobSpec


def test_reliability_and_inet_floors_are_fixed_defaults(monkeypatch):
    """The reliability floor and inet-speed minimum are fixed correctness defaults (0.995 / 200
    Mbps) — NOT operator-tunable. Setting the old FLASH_VAST_MIN_* env vars has no effect."""
    import importlib

    from flash.providers.vast import jobs as vast

    monkeypatch.setenv("FLASH_VAST_MIN_RELIABILITY", "0.999")
    monkeypatch.setenv("FLASH_VAST_MIN_INET_MBPS", "1000")
    importlib.reload(vast)
    # The env vars are no longer read: the floors stay at their fixed defaults.
    assert vast.RELIABILITY_FLOOR == 0.995
    assert vast.MIN_INET_MBPS == 200.0
    monkeypatch.delenv("FLASH_VAST_MIN_RELIABILITY", raising=False)
    monkeypatch.delenv("FLASH_VAST_MIN_INET_MBPS", raising=False)
    importlib.reload(vast)  # restore module defaults for other tests


def _spec(**gpu_kw) -> JobSpec:
    gpu = {"type": "RTX 3090", "provider": "vast", "max_wall_seconds": 3600, **gpu_kw}
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": "flash-1700000000-abcd1234",
            "train": {"epochs": 1, "seeds": [0], "hf_repo": "org/repo"},
            "gpu": gpu,
        }
    )


def _offer_obj(offer_id=1, machine_id=10, gpu="RTX 3090", dph=0.25):
    from tests._helpers.vast import make_vast_offer

    return make_vast_offer(offer_id=offer_id, machine_id=machine_id, gpu=gpu, dph_total=dph)


# ---------------------------------------------------------------------------
# onstart + bootstrap
# ---------------------------------------------------------------------------
def test_onstart_ships_payload_and_bootstrap(monkeypatch):
    from flash.providers.vast import jobs as vast

    monkeypatch.setenv("VAST_API_KEY", "vk-supersecret")
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-supersecret")
    monkeypatch.setenv("HF_REPO", "org/repo")
    payload = vast.build_payload(_spec(), seed=0, attempt=1)
    assert payload["phase"] == "sft"
    assert payload["attempt"] == 1
    assert payload["hf_prefix"] == "sft/flash-1700000000-abcd1234/seed0"
    assert payload["max_wall_s"] == 3600

    script = vast.build_onstart(payload, install_deps=True)
    # payload travels base64-encoded inside a quoted heredoc, byte-exact
    b64 = script.split("FLASH_PAYLOAD_EOF")[1].strip()
    assert json.loads(base64.b64decode(b64)) == payload
    # the self-contained bootstrap is embedded and keeps _train_body's semantics
    assert "FLASH_BOOTSTRAP_EOF" in script
    assert "metrics.json" in script
    # worker deps install (skippable with a baked image)
    assert "pip install" in script
    assert "torch==2.10.0" in script
    # a base-deps install failure STOPS the script before bootstrap.py (no partial-env
    # crash), holding the box for log retrieval — mirrors the bootstrap-failure path.
    assert "base worker dependency install failed" in script
    pip_idx = script.index("pip install")
    boot_idx = script.index("/root/flash/bootstrap.py\n")  # the run, not the heredoc write
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


def test_build_payload_carries_per_run_hf_repo(monkeypatch):
    """The submit payload's hf_repo is the run's [train] hf_repo (there is no operator HF_REPO
    fallback). The worker fetches code + writes artifacts to this repo, its env's HF_REPO must
    match the payload's, and an operator HF_REPO in the env must NOT leak in."""
    from flash.providers.vast import jobs as vast

    # an operator HF_REPO in the control-plane env must be ignored
    monkeypatch.setenv("HF_REPO", "operator/default")
    per_run = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-0.6B",
            "algorithm": "sft",
            "run_id": "flash-1700000000-abcd1234",
            "train": {"epochs": 1, "seeds": [0], "hf_repo": "myorg/runs"},
            "gpu": {"type": "RTX 3090", "provider": "vast", "max_wall_seconds": 3600},
        }
    )
    payload = vast.build_payload(per_run, seed=0, attempt=0)
    assert payload["hf_repo"] == "myorg/runs"
    assert payload["env"]["HF_REPO"] == "myorg/runs"


def _bootstrap_env(monkeypatch, phase="sft", rcs=(0,), metrics=True):
    from flash.providers.vast import _bootstrap as vb

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


def _bootstrap_with_hub_env(monkeypatch, prime_present):
    """Drive _bootstrap.main() with one published env to install, recording subprocess cmds."""
    from flash.providers.vast import _bootstrap as vb

    cmds: list[list[str]] = []
    monkeypatch.setattr(
        vb,
        "load_payload",
        lambda path=vb.PAYLOAD_PATH: {
            "hf_repo": "org/repo",
            "job_spec_json": "{}",
            "phase": "sft",
            "seed": 0,
            "env": {"PRIME_API_KEY": "pit-test"},
            "extra_pip": [],
            "hub_env_ids": ["owner/env"],
            "hf_prefix": "sft/x/seed0",
            "max_wall_s": 60,
            "attempt": 0,
        },
    )
    monkeypatch.setattr(vb, "fetch_code", lambda p: None)
    monkeypatch.setattr(vb, "run_mode", lambda p, e, m, d: 0)
    monkeypatch.setattr(vb, "write_attempt_marker", lambda p, ok, error="": None)
    monkeypatch.setattr(vb.os.path, "exists", lambda p: "metrics.json" in p)

    class _Proc:
        returncode = 0

    monkeypatch.setattr(
        vb.subprocess, "run", lambda cmd, *a, **k: (cmds.append(list(cmd)), _Proc())[1]
    )
    monkeypatch.setattr(
        vb.shutil,
        "which",
        lambda name: "/usr/bin/prime" if (prime_present and name == "prime") else None,
    )
    return vb, cmds


def test_bootstrap_installs_prime_into_worker_python(monkeypatch):
    # When `prime` is already present (baked into the image), don't reinstall it; run the
    # LOCATED prime binary and install the env into THIS python via `--with pip` so the
    # trainer can import the env module at load_environment.
    vb, cmds = _bootstrap_with_hub_env(monkeypatch, prime_present=True)
    assert vb.main() == 0
    assert not any(c[-1] == "prime" and "install" in c for c in cmds)
    assert ["/usr/bin/prime", "env", "install", "owner/env", "--with", "pip"] in cmds


def test_bootstrap_installs_prime_when_missing(monkeypatch):
    # When `prime` isn't already on PATH, the bootstrap installs it and then proceeds to the env
    # install (rather than silently skipping it, which would crash later with ModuleNotFoundError).
    vb, cmds = _bootstrap_with_hub_env(monkeypatch, prime_present=False)
    assert vb.main() == 0
    assert any(c[-1] == "prime" and "install" in c for c in cmds)
    assert ["prime", "env", "install", "owner/env", "--with", "pip"] in cmds


# ---------------------------------------------------------------------------
# deploy_and_submit: live-market offer walk
# ---------------------------------------------------------------------------
def test_deploy_walks_taken_offers(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

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
    assert h.label == "flash-1700000000-abcd1234-s0-a2"


def test_deploy_refreshes_once_when_all_taken(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

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


def test_deploy_refresh_excludes_blacklisted_machines(monkeypatch):
    """Fix #3: the in-provider offer REFRESH must keep the run's blacklisted machines
    excluded — otherwise a sick machine the orchestrator just blacklisted can be
    re-selected from the fresh market."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    monkeypatch.setenv("HF_REPO", "org/repo")
    captured = {}

    def fake_create(offer_id, **kw):
        raise vast_api.VastApiError("taken")

    def fake_usable(min_vram_gb, disk_gb, exclude_machine_ids=frozenset()):
        captured["exclude"] = set(exclude_machine_ids)
        return []  # force the pool-exhausted raise after the (captured) refresh

    monkeypatch.setattr(vast_api, "create_instance", fake_create)
    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    # the initial offer is on machine 1; the run blacklist carries machine 42
    with pytest.raises(vast_api.VastApiError):
        vast.deploy_and_submit(
            _spec(),
            seed=0,
            offers=[_offer_obj(offer_id=1, machine_id=1)],
            attempt=0,
            exclude_machine_ids={42},
        )
    # the refresh excluded BOTH the just-tried machine (1) and the standing blacklist (42)
    assert captured["exclude"] == {1, 42}


def test_deploy_refresh_uses_disk_floor(monkeypatch):
    """Fix #2: both the initial and refresh offer searches use the same effective disk
    floor create_instance enforces (max(disk_gb, 60)); a spec asking <60 GB must NOT
    surface offers that then fail to rent."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    monkeypatch.setenv("HF_REPO", "org/repo")
    captured = {}

    monkeypatch.setattr(
        vast_api,
        "create_instance",
        lambda *a, **k: (_ for _ in ()).throw(vast_api.VastApiError("taken")),
    )

    def fake_usable(min_vram_gb, disk_gb, exclude_machine_ids=frozenset()):
        captured["disk_gb"] = disk_gb
        return []

    monkeypatch.setattr(vast, "usable_offers", fake_usable)
    with pytest.raises(vast_api.VastApiError):
        vast.deploy_and_submit(_spec(disk_gb=20), seed=0, offers=[_offer_obj()], attempt=0)
    assert captured["disk_gb"] == vast.MIN_DISK_GB  # floored to 60, not the spec's 20


def test_deploy_raises_when_pool_exhausted(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

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
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

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
    from flash.providers.vast.jobs import VastJobHandle

    return VastJobHandle(
        instance_id=9999,
        offer_id=1,
        machine_id=10,
        label="flash-x-s0-a0",
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
    from flash.providers.base import PollResult
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

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
    from flash.providers.base import PollResult

    vast, destroyed, _ = _wire_runner(monkeypatch, PollResult(True, metrics={"a": 1}))
    handles = []
    res = vast.submit_run_vast(_spec(), seed=0, on_handle=handles.append, offers=[_offer_obj()])
    assert res.ok
    assert destroyed == [9999]
    assert handles
    assert handles[0]["provider"] == "vast"
    assert handles[0]["instance_id"] == 9999


def test_runner_destroys_on_failure_and_exception(monkeypatch):
    from flash.providers.base import PollResult

    vast, destroyed, _ = _wire_runner(monkeypatch, PollResult(False, failure="stalled"))
    res = vast.submit_run_vast(_spec(), seed=0, offers=[_offer_obj()])
    assert not res.ok
    assert destroyed == [9999]

    vast, destroyed, _ = _wire_runner(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        vast.submit_run_vast(_spec(), seed=0, offers=[_offer_obj()])
    assert destroyed == [9999]


def test_runner_destroys_when_handle_persist_fails(monkeypatch):
    """Fix #1: the rented instance is destroyed even if on_handle (persisting the remote
    handle) raises — the teardown finally guards everything after the rent, not just the
    poll. Otherwise a persist failure leaks a billing instance."""
    vast, destroyed, _ = _wire_runner(monkeypatch, None)  # poll_outcome unused; on_handle raises

    def boom(_handle_dict):
        raise RuntimeError("status store unreachable")

    with pytest.raises(RuntimeError, match="status store unreachable"):
        vast.submit_run_vast(_spec(), seed=0, on_handle=boom, offers=[_offer_obj()])
    assert destroyed == [9999]  # the paid instance was torn down despite the persist crash


def test_instance_label_always_sweepable():
    from flash.providers.vast.jobs import instance_label

    # platform run ids pass through; anything else gets the prefix FORCED so the
    # orphan sweep can never miss an instance we rented (live incident: a unit
    # test's "fail-fast" run id produced unsweepable labels)
    assert instance_label("flash-1700-abcd", 0, 1) == "flash-1700-abcd-s0-a1"
    assert instance_label("fail-fast", 0, 0) == "flash-fail-fast-s0-a0"


def test_destroy_run_instances_matches_forced_prefix(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": 1, "label": "flash-fail-fast-s0-a0"},  # forced-prefix label
        {"id": 2, "label": "flash-other-run-s0-a0"},  # different run -> keep
    ]
    destroyed = []
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    assert vast.destroy_run_instances("fail-fast") == [1]
    assert destroyed == [1]


def test_handle_roundtrip():
    from flash.providers.vast.jobs import VastJobHandle

    h = _handle()
    d = h.to_dict()
    assert d["provider"] == "vast"
    assert VastJobHandle.from_dict(d) == h


def test_run_label_prefix_matches_instance_label():
    """Fix #8: run_label_prefix applies the SAME forced-`flash-` transform the labels
    carry, so the orphan-sweep allowlist (built from raw run ids) actually matches."""
    from flash.providers.vast.jobs import instance_label, run_label_prefix

    # a raw run id (no prefix) -> the prefix its labels start with IS prefixed
    assert run_label_prefix("fail-fast") == "flash-fail-fast"
    assert instance_label("fail-fast", 0, 0).startswith(run_label_prefix("fail-fast"))
    # an already-prefixed run id is idempotent
    assert run_label_prefix("flash-1700-abcd") == "flash-1700-abcd"
    assert instance_label("flash-1700-abcd", 1, 2).startswith(
        run_label_prefix("flash-1700-abcd")
    )


def test_sweep_orphans_protects_unprefixed_active_run_id(monkeypatch):
    """Fix #8: a live run whose RAW id lacks the `flash-` prefix is still protected —
    sweep_orphans transforms active ids through run_label_prefix before matching, so its
    forced-prefix instance label is not swept."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": 1, "label": vast.instance_label("fail-fast", 0, 0)},  # live run -> KEEP
        {"id": 2, "label": vast.instance_label("orphan-run", 0, 0)},  # no live run -> destroy
    ]
    destroyed = []
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    # active carries the RAW run id (what the server tracks), not the prefixed label
    out = vast.sweep_orphans(active_labels={"fail-fast"})
    assert out == [2]  # the live "fail-fast" run's instance (1) is protected
    assert destroyed == [2]


def test_sweep_orphans_label_safety(monkeypatch):
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": 1, "label": "flash-1700-aaaa-s0-a0"},  # orphan -> destroy
        {"id": 2, "label": "flash-1700-bbbb-s0-a1"},  # active run -> keep
        {"id": 3, "label": "someone-elses-workload"},  # not ours -> NEVER touch
        {"id": 4, "label": ""},  # unlabeled -> NEVER touch
    ]
    destroyed = []
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.sweep_orphans(active_labels={"flash-1700-bbbb"})
    assert out == [1]
    assert destroyed == [1]


def test_sweep_orphans_prefix_is_not_shielded_by_a_longer_run_id(monkeypatch):
    """A live run id that is a STRING prefix of another run id must not shield the other
    run's orphan. ``flash-100`` is a prefix of ``flash-1000-...`` but they are
    distinct runs; the match has to land on the ``-s`` seed boundary, not raw startswith."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    instances = [
        {"id": 1, "label": vast.instance_label("flash-100", 0, 0)},  # live run -> KEEP
        {"id": 2, "label": vast.instance_label("flash-1000", 0, 0)},  # orphan -> destroy
    ]
    # Sanity: the orphan's label really does start with the live run's prefix as a raw
    # string, so the old ``startswith`` would have wrongly shielded it.
    assert instances[1]["label"].startswith(vast.run_label_prefix("flash-100"))
    destroyed = []
    monkeypatch.setattr(vast_api, "list_instances", lambda: instances)
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    out = vast.sweep_orphans(active_labels={"flash-100"})
    assert out == [2]  # only the genuine orphan; the prefix-collision run is left alone


def test_submit_run_vast_rejects_policy_word_gpu(monkeypatch):
    """The offers=None fallback indexes GPU_INFO by concrete class; a policy word
    ("cheapest"/"auto") must fail with a clear error, not an opaque KeyError."""
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast

    spec = _spec(type="cheapest")
    with pytest.raises(vast_api.VastApiError, match="concrete gpu class"):
        vast.submit_run_vast(spec, seed=0)
