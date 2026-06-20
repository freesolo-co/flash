"""Orchestrator provider routing: vast vs runpod submit/cancel/attach, cross-provider
failover, handle persistence, cost flow, and config provider fields (all mocked).

Everything dispatches through the ``base.Provider`` interface (the registry), so these
tests patch the provider's job functions / api modules — the same objects the
provider methods import lazily — rather than any hardcoded orchestrator branch.
"""

from __future__ import annotations

import io

import pytest

from flash.spec import JobSpec


def _spec(run_id="flash-1700000001-rt01", **gpu_kw) -> JobSpec:
    gpu = {"type": "RTX A5000", "provider": "auto", "requested": "auto", "max_retries": 2}
    gpu.update(gpu_kw)
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": run_id,
            "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
            "gpu": gpu,
        }
    )


def _offer_obj(offer_id=5, machine_id=2, gpu="RTX 3090", dph=0.25):
    from tests._helpers.vast import make_vast_offer

    return make_vast_offer(offer_id=offer_id, machine_id=machine_id, gpu=gpu, dph_total=dph)


def _alloc(provider="vast", gpu="RTX 3090", rate=0.25, offer=None, provider_offers=()):
    from flash.providers.base import Allocation, Candidate

    cand = Candidate(provider, gpu, rate, 24, offer=offer)
    return Allocation(
        provider=provider,
        gpu=gpu,
        hourly_usd=rate,
        min_vram_gb=12,
        offer=offer,
        candidates=(cand,),
        provider_offers=tuple(provider_offers),
    )


def _vast_handle_dict(instance_id=1, machine_id=2):
    return {
        "provider": "vast",
        "instance_id": instance_id,
        "offer_id": 5,
        "machine_id": machine_id,
        "label": "flash-x-s0-a0",
        "gpu": "RTX 3090",
        "hourly_usd": 0.25,
        "attempt": 0,
        "started_ts": 0.0,
    }


@pytest.fixture
def orch(monkeypatch, tmp_path):
    from flash import runner

    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    return runner


def _seed_status(orch, spec):
    st = orch.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    orch._save_status(st)
    return st


def test_vast_allocation_routes_to_vast_runner(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.vast import jobs as vast_jobs

    offer = _offer_obj()
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: _alloc(offer=offer, provider_offers=[offer])
    )
    captured = {}

    def fake_vast_submit(
        run_spec,
        seed,
        log=None,
        on_handle=None,
        attempt=0,
        offers=None,
        exclude_machine_ids=frozenset(),
    ):
        captured["gpu_type"] = run_spec.gpu.type
        captured["offers"] = offers
        on_handle(_vast_handle_dict())
        return PollResult(
            True,
            metrics={"train_tokens": 4096, "cost_usd": 0.123, "notes": {"provider": "vast"}},
        )

    monkeypatch.setattr(vast_jobs, "submit_run_vast", fake_vast_submit)
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
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast_jobs

    offer = _offer_obj(machine_id=42)
    allocate_calls = []

    def fake_allocate(model, algorithm, **kw):
        allocate_calls.append(kw["exclude_machine_ids"])
        if not kw["exclude_machine_ids"]:
            return _alloc(offer=offer, provider_offers=[offer])
        return _alloc(provider="runpod", gpu="RTX A5000", rate=0.27)

    monkeypatch.setattr(allocator, "allocate", fake_allocate)

    submit_excludes = []

    def fake_vast_submit(
        run_spec,
        seed,
        log=None,
        on_handle=None,
        attempt=0,
        offers=None,
        exclude_machine_ids=frozenset(),
    ):
        submit_excludes.append(exclude_machine_ids)
        on_handle(_vast_handle_dict(instance_id=77, machine_id=42))
        return PollResult(False, failure="stalled", detail="no worker progress")

    monkeypatch.setattr(vast_jobs, "submit_run_vast", fake_vast_submit)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    monkeypatch.setattr(
        rp_jobs,
        "submit_run",
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
    # the blacklist is threaded into the provider submit too, so an in-provider offer
    # refresh keeps the sick machine excluded (Fix #3)
    assert submit_excludes[0] == frozenset()


def test_genuine_worker_error_does_not_retry(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.vast import jobs as vast_jobs

    offer = _offer_obj()
    monkeypatch.setattr(
        allocator, "allocate", lambda *a, **k: _alloc(offer=offer, provider_offers=[offer])
    )
    calls = []

    def fake_vast_submit(
        run_spec,
        seed,
        log=None,
        on_handle=None,
        attempt=0,
        offers=None,
        exclude_machine_ids=frozenset(),
    ):
        calls.append(attempt)
        return PollResult(False, failure="job_failed", detail="ValueError: bad reward fn")

    monkeypatch.setattr(vast_jobs, "submit_run_vast", fake_vast_submit)
    spec = _spec()
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError, match="bad reward fn"):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert calls == [0]  # code errors burn no retry budget


def test_concrete_requested_pins_class(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.vast import jobs as vast_jobs

    pins = []

    def fake_allocate(model, algorithm, **kw):
        pins.append(kw["gpu"])
        offer = _offer_obj()
        return _alloc(offer=offer, provider_offers=[offer])

    monkeypatch.setattr(allocator, "allocate", fake_allocate)
    monkeypatch.setattr(
        vast_jobs,
        "submit_run_vast",
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
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.vast import jobs as vast_jobs

    pins = []

    def fake_allocate(model, algorithm, **kw):
        pins.append(kw["gpu"])
        offer = _offer_obj()
        return _alloc(offer=offer, provider_offers=[offer])

    monkeypatch.setattr(allocator, "allocate", fake_allocate)
    monkeypatch.setattr(
        vast_jobs,
        "submit_run_vast",
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
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train as rp_train
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast_jobs

    cancelled, destroyed, swept, runpod_calls, terminated = [], [], [], [], []
    monkeypatch.setattr(vast_jobs, "cancel", lambda remote: cancelled.append(remote))
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    monkeypatch.setattr(vast_jobs, "destroy_run_instances", lambda rid: swept.append(rid) or [])
    monkeypatch.setattr(rp_train, "terminate_endpoint", lambda *a, **k: terminated.append(a))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a: runpod_calls.append(a))
    # make the vast provider "available" so _gc_run_endpoints invokes its gc sweep
    monkeypatch.setenv("VAST_API_KEY", "x")
    monkeypatch.delenv("FLASH_SKIP_NET", raising=False)

    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = _vast_handle_dict(instance_id=7)
    orch._save_status(st)
    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    # cancel + belt-and-suspenders destroy both routed to vast via the handle's provider
    # (cancel_run destroys the handle; _gc_run_endpoints destroys it again — idempotent).
    assert cancelled
    assert cancelled[0]["instance_id"] == 7
    assert destroyed
    assert all(i == 7 for i in destroyed)
    assert swept == [spec.run_id]  # _gc_run_endpoints -> vast provider gc
    assert not runpod_calls  # never touched the runpod cancel path


def test_cancel_legacy_handle_defaults_to_runpod(orch, monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import train as rp_train

    cancelled_jobs, deleted_eps = [], []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: cancelled_jobs.append((e, j)))
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: deleted_eps.append(e) or True)
    monkeypatch.setattr(rp_train, "terminate_endpoint", lambda *a, **k: [])
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = {"endpoint_id": "ep1", "endpoint_name": "n", "job_id": "j1"}  # pre-provider era
    orch._save_status(st)
    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert cancelled_jobs == [("ep1", "j1")]  # no `provider` -> defaults to runpod
    assert "ep1" in deleted_eps


def test_attach_routes_vast_and_destroys(orch, monkeypatch):
    from flash.providers.base import PollResult
    from flash.providers.runpod import train as rp_train
    from flash.providers.vast import api as vast_api
    from flash.providers.vast import jobs as vast_jobs

    seen_kwargs: dict = {}

    def _fake_poll(handle, spec, seed, log=None, heartbeat_reader=None, **kwargs):
        # The reattach path must forward the stall tuning + wall-cap deadline (matching
        # submit_run_vast / RunPod's reattach), not silently drop them.
        seen_kwargs.update(kwargs)
        return PollResult(True, metrics={"train_tokens": 4096, "cost_usd": 0.3})

    monkeypatch.setattr(vast_jobs, "poll_vast_job", _fake_poll)
    destroyed = []
    monkeypatch.setattr(vast_api, "destroy_instance", lambda iid: destroyed.append(iid) or True)
    monkeypatch.setattr(rp_train, "terminate_endpoint", lambda *a, **k: [])
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = _vast_handle_dict(instance_id=8)
    orch._save_status(st)
    out = orch.attach_run(spec.run_id, log_stream=io.StringIO())
    assert out.state == "done"
    assert out.cost_usd == 0.3
    assert 8 in destroyed  # _gc_run_endpoints destroyed the instance via the handle
    # Reattach forwarded the stall window + wall-cap deadline (not dropped on recovery).
    assert "stall_after_s" in seen_kwargs
    assert seen_kwargs.get("deadline_s") == max(60, int(spec.gpu.max_wall_seconds)) + 1800


# ---------------------------------------------------------------------------
# config: provider fields
# ---------------------------------------------------------------------------
def test_config_gpu_fields(monkeypatch):
    from flash.schema import spec_from_dict

    monkeypatch.setenv("FLASH_SKIP_NET", "1")
    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "train": {"epochs": 1, "seeds": [0], "hf_repo": "owner/runs"},
        "environment": {"id": "owner/env"},
    }
    # omitted gpu.type -> smart-allocation default (cheapest fitting), original request preserved
    spec = spec_from_dict(dict(base), run_id="x")
    assert spec.gpu.requested == "auto"
    assert spec.gpu.type == "RTX 2000 Ada"  # deterministic offline cheapest-fitting provisional
    # round-trip keeps the request word
    again = JobSpec.from_dict(spec.to_dict())
    assert again.gpu.requested == "auto"
    # any known class is accepted with no provider pin or validation gate
    spec = spec_from_dict({**base, "gpu": {"type": "L40S"}}, run_id="x")
    assert spec.gpu.type == "L40S"
    assert spec.gpu.requested == "L40S"
