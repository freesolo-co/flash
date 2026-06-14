"""Orchestrator provider routing: vast vs runpod submit/cancel/attach, cross-provider
failover, handle persistence, cost flow, and config provider fields (all mocked)."""

from __future__ import annotations

import io

import pytest

from autoslm.worker_spec import JobSpec


def _spec(run_id="autoslm-1700000001-rt01", **gpu_kw) -> JobSpec:
    gpu = {"type": "RTX A5000", "provider": "auto", "requested": "auto", "max_retries": 2}
    gpu.update(gpu_kw)
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3-0.6B",
            "algorithm": "sft",
            "run_id": run_id,
            "train": {"epochs": 1, "seeds": [0]},
            "gpu": gpu,
        }
    )


def _offer_obj(offer_id=5, machine_id=2, gpu="RTX 3090", dph=0.25):
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


def _alloc(provider="vast", gpu="RTX 3090", rate=0.25, offer=None, vast_offers=()):
    from autoslm.providers.allocator import Allocation, Candidate

    cand = Candidate(provider, gpu, rate, 24, True, offer=offer)
    return Allocation(
        provider=provider,
        gpu=gpu,
        hourly_usd=rate,
        min_vram_gb=12,
        offer=offer,
        candidates=(cand,),
        vast_offers=tuple(vast_offers),
    )


def _vast_handle_dict(instance_id=1, machine_id=2):
    return {
        "provider": "vast",
        "instance_id": instance_id,
        "offer_id": 5,
        "machine_id": machine_id,
        "label": "autoslm-x-s0-a0",
        "gpu": "RTX 3090",
        "hourly_usd": 0.25,
        "attempt": 0,
        "started_ts": 0.0,
    }


@pytest.fixture
def orch(monkeypatch, tmp_path):
    from autoslm import orchestrator

    monkeypatch.delenv("AUTOSLM_SKIP_NET", raising=False)
    monkeypatch.setattr(orchestrator, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(orchestrator, "RESULTS_DIR", str(tmp_path / "results"))
    return orchestrator


def _seed_status(orch, spec):
    st = orch.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    orch._save_status(st)
    return st


def test_vast_allocation_routes_to_vast_runner(orch, monkeypatch):
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator
    from autoslm.providers import vast as vast_provider

    offer = _offer_obj()
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: _alloc(offer=offer, vast_offers=[offer])
    )
    captured = {}

    def fake_vast_submit(run_spec, seed, log=None, on_handle=None, attempt=0, offers=None):
        captured["gpu_type"] = run_spec.gpu.type
        captured["offers"] = offers
        on_handle(_vast_handle_dict())
        return PollResult(
            True,
            metrics={"train_tokens": 4096, "cost_usd": 0.123, "notes": {"provider": "vast"}},
        )

    monkeypatch.setattr(vast_provider, "submit_train_durable_vast", fake_vast_submit)
    spec = _spec()
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["cost_usd"] == 0.123
    # the attempt ran with the ALLOCATED class, not the parse-time provisional
    assert captured["gpu_type"] == "RTX 3090"
    assert [o.offer_id for o in captured["offers"]] == [5]
    # handle persisted for cross-process cancel/attach
    assert orch.get_status(spec.run_id).remote["provider"] == "vast"


def test_vast_cost_flows_into_run_status(orch, monkeypatch):
    spec = _spec()
    _seed_status(orch, spec)
    cost = orch._persist_metrics(
        spec, 0, {"train_tokens": 4096, "cost_usd": 0.2, "notes": {"provider": "vast"}}
    )
    assert cost == 0.2  # the stamped vast cost short-circuits the runpod projection


def test_failover_crosses_providers_and_blacklists_machine(orch, monkeypatch):
    import autoslm.flash.durable as durable
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator, vast_api
    from autoslm.providers import vast as vast_provider

    offer = _offer_obj(machine_id=42)
    allocate_calls = []

    def fake_allocate(model, algorithm, **kw):
        allocate_calls.append(kw["exclude_machine_ids"])
        if not kw["exclude_machine_ids"]:
            return _alloc(offer=offer, vast_offers=[offer])
        return _alloc(provider="runpod", gpu="RTX A5000", rate=0.27)

    monkeypatch.setattr(allocator, "allocate", fake_allocate)

    def fake_vast_submit(run_spec, seed, log=None, on_handle=None, attempt=0, offers=None):
        on_handle(_vast_handle_dict(instance_id=77, machine_id=42))
        return PollResult(False, failure="stalled", detail="no worker progress")

    monkeypatch.setattr(vast_provider, "submit_train_durable_vast", fake_vast_submit)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    monkeypatch.setattr(
        durable,
        "submit_train_durable",
        lambda spec, seed, log=None, on_handle=None, attempt=0: PollResult(
            True, metrics={"train_tokens": 4096}
        ),
    )
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, 0, log)
    assert metrics["train_tokens"] == 4096
    # the stalled attempt's instance was torn down and its machine blacklisted
    assert destroyed == [77]
    assert allocate_calls[0] == frozenset()
    assert allocate_calls[1] == frozenset({42})
    assert "blacklisted" in log.getvalue()


def test_genuine_worker_error_does_not_retry(orch, monkeypatch):
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator
    from autoslm.providers import vast as vast_provider

    offer = _offer_obj()
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: _alloc(offer=offer, vast_offers=[offer])
    )
    calls = []

    def fake_vast_submit(run_spec, seed, log=None, on_handle=None, attempt=0, offers=None):
        calls.append(attempt)
        return PollResult(False, failure="job_failed", detail="ValueError: bad reward fn")

    monkeypatch.setattr(vast_provider, "submit_train_durable_vast", fake_vast_submit)
    spec = _spec()
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError, match="bad reward fn"):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert calls == [0]  # code errors burn no retry budget


def test_concrete_requested_pins_class(orch, monkeypatch):
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator
    from autoslm.providers import vast as vast_provider

    pins = []

    def fake_allocate(model, algorithm, **kw):
        pins.append(kw["gpu"])
        offer = _offer_obj()
        return _alloc(offer=offer, vast_offers=[offer])

    monkeypatch.setattr(allocator, "allocate", fake_allocate)
    monkeypatch.setattr(
        vast_provider,
        "submit_train_durable_vast",
        lambda *a, **k: PollResult(True, metrics={"a": 1}),
    )
    spec = _spec(type="RTX 3090", requested="RTX 3090")
    _seed_status(orch, spec)
    orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert pins == ["RTX 3090"]  # concrete request -> class pinned through allocation
    pins.clear()
    spec = _spec(requested="cheapest")
    _seed_status(orch, spec)
    orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert pins == [None]  # policy request -> allocator free to re-pick


def test_empty_requested_pins_concrete_type(orch, monkeypatch):
    """An empty gpu.requested (a direct-API spec that skipped config parsing) is treated
    as a concrete pin of gpu.type: the allocator runs (so failover/pricing still apply)
    but only to pick the provider for that one class — it never re-picks the class."""
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator
    from autoslm.providers import vast as vast_provider

    pins = []

    def fake_allocate(model, algorithm, **kw):
        pins.append(kw["gpu"])
        offer = _offer_obj()
        return _alloc(offer=offer, vast_offers=[offer])

    monkeypatch.setattr(allocator, "allocate", fake_allocate)
    monkeypatch.setattr(
        vast_provider,
        "submit_train_durable_vast",
        lambda *a, **k: PollResult(True, metrics={"a": 1}),
    )
    spec = _spec(type="RTX A5000", requested="")  # no routing intent recorded
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["a"] == 1
    assert pins == ["RTX A5000"]  # empty requested -> concrete gpu.type pinned through


# ---------------------------------------------------------------------------
# cancel / attach routing
# ---------------------------------------------------------------------------
def test_cancel_routes_vast(orch, monkeypatch):
    import autoslm.flash.train as flash_train
    from autoslm.providers import vast as vast_provider

    cancelled, swept, runpod_calls, terminated = [], [], [], []
    monkeypatch.setattr(vast_provider, "cancel", lambda remote: cancelled.append(remote))
    monkeypatch.setattr(vast_provider, "destroy_run_instances", lambda rid: swept.append(rid))
    monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: terminated.append(a))
    from autoslm.flash import runpod_api

    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a: runpod_calls.append(a))
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = _vast_handle_dict(instance_id=7)
    orch._save_status(st)
    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert cancelled
    assert cancelled[0]["instance_id"] == 7
    assert swept == [spec.run_id]
    assert not runpod_calls
    assert not terminated


def test_cancel_legacy_handle_defaults_to_runpod(orch, monkeypatch):
    import autoslm.flash.train as flash_train
    from autoslm.flash import runpod_api

    cancelled_jobs, deleted_eps = [], []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: cancelled_jobs.append((e, j)))
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: deleted_eps.append(e))
    monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = {"endpoint_id": "ep1", "endpoint_name": "n", "job_id": "j1"}  # pre-provider era
    orch._save_status(st)
    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert cancelled_jobs == [("ep1", "j1")]
    assert deleted_eps == ["ep1"]


def test_attach_routes_vast_and_destroys(orch, monkeypatch):
    from autoslm.flash.durable import PollResult
    from autoslm.providers import vast as vast_provider
    from autoslm.providers import vast_api

    monkeypatch.setattr(
        vast_provider,
        "poll_vast_job",
        lambda handle, spec, seed, log=None, heartbeat_reader=None: PollResult(
            True, metrics={"train_tokens": 4096, "cost_usd": 0.3}
        ),
    )
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = _vast_handle_dict(instance_id=8)
    orch._save_status(st)
    out = orch.attach_run(spec.run_id, log_stream=io.StringIO())
    assert out.state == "done"
    assert out.cost_usd == 0.3
    assert destroyed == [8]


# ---------------------------------------------------------------------------
# config: provider fields
# ---------------------------------------------------------------------------
def test_config_provider_fields(monkeypatch):
    from autoslm.config_schema import ConfigError, spec_from_dict

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    monkeypatch.delenv("AUTOSLM_GPU_ALLOW_UNVALIDATED", raising=False)
    base = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "algorithm": "sft",
        "train": {"epochs": 1, "seeds": [0]},
        "environment": {"id": "owner/env"},
    }
    # omitted gpu.type -> smart-allocation default, original request preserved
    spec = spec_from_dict(dict(base), run_id="x")
    assert spec.gpu.requested == "auto"
    assert spec.gpu.provider == "auto"
    assert spec.gpu.type == "RTX A5000"  # deterministic offline provisional
    # round-trip keeps the routing fields
    again = JobSpec.from_dict(spec.to_dict())
    assert (again.gpu.provider, again.gpu.requested) == ("auto", "auto")

    with pytest.raises(ConfigError, match="provider"):
        spec_from_dict({**base, "gpu": {"provider": "lambda"}}, run_id="x")
    # class/provider mismatch fails at parse time
    with pytest.raises(ConfigError, match="not available on provider"):
        spec_from_dict(
            {**base, "gpu": {"type": "L40S", "provider": "runpod", "allow_unvalidated": True}},
            run_id="x",
        )
    # vast-only class pinned to vast parses (with the unvalidated opt-in)
    spec = spec_from_dict(
        {**base, "gpu": {"type": "L40S", "provider": "vast", "allow_unvalidated": True}},
        run_id="x",
    )
    assert spec.gpu.type == "L40S"
    assert spec.gpu.provider == "vast"
    assert spec.gpu.requested == "L40S"
    assert spec.gpu.allow_unvalidated is True
