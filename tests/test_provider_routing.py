"""Orchestrator routing: RunPod submit/cancel/attach, retry/failover, handle
persistence, cost flow, and config provider fields (all mocked)."""

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


def _alloc(provider="runpod", gpu="RTX A5000", rate=0.27):
    from autoslm.providers.allocator import Allocation, Candidate

    cand = Candidate(provider, gpu, rate, 24, True)
    return Allocation(
        provider=provider,
        gpu=gpu,
        hourly_usd=rate,
        min_vram_gb=12,
        candidates=(cand,),
    )


def _handle_dict(endpoint_id="ep1", job_id="j1"):
    return {
        "endpoint_id": endpoint_id,
        "endpoint_name": "autoslm-a5000-x",
        "job_id": job_id,
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


def test_allocation_routes_to_runpod_runner(orch, monkeypatch):
    import autoslm.flash.durable as durable
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(gpu="RTX 3090", rate=0.46))
    captured = {}

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0):
        captured["gpu_type"] = run_spec.gpu.type
        on_handle(_handle_dict())
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(durable, "submit_train_durable", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["train_tokens"] == 4096
    # the attempt ran with the ALLOCATED class, not the parse-time provisional
    assert captured["gpu_type"] == "RTX 3090"
    # handle persisted for cross-process cancel/attach
    assert orch.get_status(spec.run_id).remote["endpoint_id"] == "ep1"


def test_runpod_cost_projection(orch, monkeypatch):
    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    spec = _spec()
    _seed_status(orch, spec)
    cost = orch._persist_metrics(spec, 0, {"train_tokens": 4096, "cost_usd": 0.2})
    assert cost == 0.2  # an explicit cost short-circuits the projection


def test_failover_retries_on_fresh_endpoint(orch, monkeypatch):
    import autoslm.flash.durable as durable
    from autoslm.flash import runpod_api
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    deleted = []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda *a: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: deleted.append(e))
    calls = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0):
        calls.append(attempt)
        on_handle(_handle_dict(endpoint_id=f"ep{attempt}"))
        if attempt == 0:
            return PollResult(False, failure="stalled", detail="no worker progress")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(durable, "submit_train_durable", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["train_tokens"] == 4096
    assert calls == [0, 1]  # the stalled attempt was retried on a fresh endpoint
    assert "ep0" in deleted  # the stalled endpoint was torn down before retry


def test_genuine_worker_error_does_not_retry(orch, monkeypatch):
    import autoslm.flash.durable as durable
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    calls = []

    def fake_submit(run_spec, seed, log=None, on_handle=None, attempt=0):
        calls.append(attempt)
        return PollResult(False, failure="job_failed", detail="ValueError: bad reward fn")

    monkeypatch.setattr(durable, "submit_train_durable", fake_submit)
    spec = _spec()
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError, match="bad reward fn"):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert calls == [0]  # code errors burn no retry budget


def test_concrete_requested_pins_class(orch, monkeypatch):
    import autoslm.flash.durable as durable
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator

    pins = []

    def fake_allocate(model, algorithm, **kw):
        pins.append(kw["gpu"])
        return _alloc()

    monkeypatch.setattr(allocator, "allocate", fake_allocate)
    monkeypatch.setattr(
        durable, "submit_train_durable", lambda *a, **k: PollResult(True, metrics={"a": 1})
    )
    spec = _spec(type="RTX 3090", requested="RTX 3090", allow_unvalidated=True)
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
    but only for that one class — it never re-picks the class."""
    import autoslm.flash.durable as durable
    from autoslm.flash.durable import PollResult
    from autoslm.providers import allocator

    pins = []

    def fake_allocate(model, algorithm, **kw):
        pins.append(kw["gpu"])
        return _alloc()

    monkeypatch.setattr(allocator, "allocate", fake_allocate)
    monkeypatch.setattr(
        durable, "submit_train_durable", lambda *a, **k: PollResult(True, metrics={"a": 1})
    )
    spec = _spec(type="RTX A5000", requested="")  # no routing intent recorded
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["a"] == 1
    assert pins == ["RTX A5000"]  # empty requested -> concrete gpu.type pinned through


# ---------------------------------------------------------------------------
# cancel / attach routing
# ---------------------------------------------------------------------------
def test_cancel_routes_runpod(orch, monkeypatch):
    import autoslm.flash.train as flash_train
    from autoslm.flash import runpod_api

    cancelled_jobs, deleted_eps, terminated = [], [], []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: cancelled_jobs.append((e, j)))
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: deleted_eps.append(e))
    monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: terminated.append(a))
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = _handle_dict()
    orch._save_status(st)
    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert cancelled_jobs == [("ep1", "j1")]
    assert deleted_eps == ["ep1"]
    assert terminated  # name-reconstructed endpoint GC also runs


def test_attach_routes_runpod(orch, monkeypatch):
    import autoslm.flash.durable as durable
    import autoslm.flash.train as flash_train
    from autoslm.flash.durable import PollResult

    monkeypatch.setattr(
        durable,
        "poll_job",
        lambda handle, log=None, heartbeat_reader=None: PollResult(
            True, metrics={"train_tokens": 4096, "cost_usd": 0.3}
        ),
    )
    monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
    from autoslm.flash import runpod_api

    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: None)
    spec = _spec()
    st = _seed_status(orch, spec)
    st.state = "running"
    st.remote = _handle_dict()
    orch._save_status(st)
    out = orch.attach_run(spec.run_id, log_stream=io.StringIO())
    assert out.state == "done"
    assert out.cost_usd == 0.3


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
        spec_from_dict({**base, "gpu": {"provider": "vast"}}, run_id="x")
    # a runpod pin parses cleanly
    spec = spec_from_dict({**base, "gpu": {"type": "RTX A5000", "provider": "runpod"}}, run_id="x")
    assert spec.gpu.provider == "runpod"
    assert spec.gpu.type == "RTX A5000"
