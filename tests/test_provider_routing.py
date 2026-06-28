"""Orchestrator RunPod routing: submit/cancel, retry, handle persistence, and cost flow."""

from __future__ import annotations

import io

import pytest

from flash.spec import JobSpec


def _spec(run_id="flash-1700000001-rt01", **gpu_kw) -> JobSpec:
    gpu = {"type": "RTX A6000", "max_retries": 2}
    gpu.update(gpu_kw)
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-0.8B",
            "algorithm": "sft",
            "run_id": run_id,
            "train": {"epochs": 1, "hf_repo": "owner/runs"},
            "gpu": gpu,
        }
    )


def _alloc(gpu="RTX A6000", rate=0.49, candidates=None):
    from flash.providers.base import Allocation, Candidate

    if candidates is None:
        candidates = (Candidate("runpod", gpu, rate, 48),)
    return Allocation(
        provider=candidates[0].provider,
        gpu=candidates[0].gpu,
        hourly_usd=candidates[0].hourly_usd,
        min_vram_gb=12,
        candidates=tuple(candidates),
    )


def _lambda_handle(instance_id="i1"):
    return {"provider": "lambda", "instance_id": instance_id, "region": "us-east-1", "name": "n"}


def _runpod_handle(endpoint_id="ep", job_id="j"):
    return {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"{endpoint_id}-name",
        "job_id": job_id,
    }


@pytest.fixture
def orch(monkeypatch, tmp_path):
    from flash import runner

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    return runner


def _seed_status(orch, spec):
    st = orch.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    orch._save_status(st)
    return st


def test_runpod_allocation_routes_to_runpod_submit(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    captured = {}

    def fake_runpod_submit(
        run_spec,
        seed,
        log=None,
        on_handle=None,
        attempt=0,
        runtime_secrets=None,
        **_,
    ):
        captured["gpu_type"] = run_spec.gpu.type
        captured["runtime_secrets"] = dict(runtime_secrets or {})
        if on_handle:
            on_handle(_runpod_handle())
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(
        spec,
        0,
        io.StringIO(),
        runtime_secrets={"WANDB_API_KEY": "user-wb"},
    )
    assert metrics["train_tokens"] == 4096
    assert captured["gpu_type"] == "RTX A6000"
    assert captured["runtime_secrets"] == {"WANDB_API_KEY": "user-wb"}
    remote = orch.get_status(spec.run_id).remote
    assert remote["provider"] == "runpod"
    assert remote["allocated_gpu"] == "RTX A6000"


def test_runpod_cost_projection_flows_into_run_status(orch, monkeypatch):
    spec = _spec()
    _seed_status(orch, spec)
    cost = orch._persist_metrics(
        spec,
        {"train_tokens": 4096, "wall_seconds": 1800, "allocated_gpu": "RTX A6000"},
    )
    assert cost == pytest.approx(0.245)  # 0.5 hr x $0.49/hr (RTX A6000)


def test_infra_retry_walks_to_next_runpod_class_and_deletes_endpoint(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("runpod", "L4", 0.39, 24),
        Candidate("runpod", "RTX A6000", 0.49, 48),
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    cancelled, deleted, submitted_gpus = [], [], []
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: cancelled.append((e, j)))
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: deleted.append(e) or True)

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted_gpus.append(run_spec.gpu.type)
        if attempt == 0:
            on_handle(_runpod_handle("ep1", "j1"))
            return PollResult(False, failure="stalled", detail="no worker progress")
        on_handle(_runpod_handle("ep2", "j2"))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, 0, log)
    assert metrics["train_tokens"] == 4096
    assert submitted_gpus == ["L4", "RTX A6000"]
    assert cancelled == [("ep1", "j1")]
    assert "ep1" in deleted
    assert "walking past the cheapest class" in log.getvalue()


def _oom_candidates():
    from flash.providers.base import Candidate

    # price- AND VRAM-ascending tiers so an OOM escalates to the next-larger card; >2 tiers exist so a
    # broken budget could (wrongly) walk through several pricier GPUs.
    return (
        Candidate("runpod", "A100", 1.0, 80),
        Candidate("runpod", "Pro6000", 2.0, 96),
        Candidate("runpod", "B200", 3.0, 180),
    )


def test_oom_escalates_exactly_once_at_max_retries_1(orch, monkeypatch):
    """OOM escalation is COST, so it respects the user's RAW max_retries: max_retries=1 grants ONE
    strictly-larger-GPU escalation, then fails terminally — NOT a walk through every pricier tier."""
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=_oom_candidates()))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    submitted = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        return PollResult(False, failure="oom", detail="CUDA out of memory")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1)
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    # smallest card + ONE strictly-larger escalation, then terminal (no third, pricier B200 attempt).
    assert submitted == ["A100", "Pro6000"]


def test_oom_after_infra_failure_still_escalates(orch, monkeypatch):
    """REGRESSION: an infra-shaped retry must NOT consume the OOM-escalation budget. A CUDA OOM that
    lands AFTER an earlier no_capacity still earns its larger-GPU escalation (the budgets are SEPARATE).
    Pre-fix the shared walk counter made the OOM read as budget-exhausted at max_retries=1, so the run
    stopped WITHOUT escalating (submitted would be ['A100', 'Pro6000'] — never reaching B200)."""
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=_oom_candidates()))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    submitted = []
    failures = iter(["no_capacity", "oom", "oom"])

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        return PollResult(False, failure=next(failures), detail="x")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1)
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    # infra(A100) -> infra-walk(Pro6000) OOMs -> ONE escalation to the strictly-larger B200 -> stop.
    # The load-bearing assertion: B200 (>96GB) was tried AFTER the OOM, i.e. the escalation happened.
    assert submitted == ["A100", "Pro6000", "B200"]


def test_oom_never_escalates_at_max_retries_0(orch, monkeypatch):
    """A deliberate single-shot run (max_retries=0) NEVER escalates a CUDA OOM onto a pricier card."""
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=_oom_candidates()))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    submitted = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        submitted.append(run_spec.gpu.type)
        on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        return PollResult(False, failure="oom", detail="CUDA out of memory")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=0)
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert submitted == ["A100"]  # one shot, no larger-GPU escalation


def test_select_candidate_escapes_failed_provider_then_walks_classes():
    """The retry picker prefers cheapest first, escapes a failed provider cross-provider on retry,
    and only walks classes within a provider once every provider has been burned."""
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate

    cands = (
        Candidate("runpod", "RTX A6000", 0.49, 48),
        Candidate("runpod", "RTX 6000 Ada", 0.50, 48),
        Candidate("lambda", "RTX A6000", 0.50, 48),
    )
    # Attempt 0 (nothing failed): cheapest overall.
    assert _select_candidate(cands, set(), set()) is cands[0]
    # RunPod burned an infra attempt -> escape to the OTHER provider, not the next RunPod class.
    chosen = _select_candidate(cands, {"runpod"}, {("runpod", "RTX A6000")})
    assert (chosen.provider, chosen.gpu) == ("lambda", "RTX A6000")
    # Both providers burned -> fall back to the cheapest class NOT yet tried (within-provider walk).
    chosen = _select_candidate(
        cands,
        {"runpod", "lambda"},
        {("runpod", "RTX A6000"), ("lambda", "RTX A6000")},
    )
    assert (chosen.provider, chosen.gpu) == ("runpod", "RTX 6000 Ada")


def test_select_candidate_single_provider_walks_classes():
    """With only one provider configured, the picker degrades to the cheapest untried class."""
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate

    cands = (Candidate("runpod", "L4", 0.39, 24), Candidate("runpod", "RTX A6000", 0.49, 48))
    assert _select_candidate(cands, {"runpod"}, {("runpod", "L4")}).gpu == "RTX A6000"


def test_select_candidate_single_fitting_gpu_never_breaks():
    """A large model with exactly ONE fitting class (e.g. only H200 fits a 35B run) must keep
    re-picking THAT class on every infra retry — the candidate list only ever holds *fitting*
    classes, so the walk can never escape to a card too small to hold the model, and the picker
    must still return it (not None / not raise) even after it's been tried and its provider burned."""
    from flash.providers.base import Candidate
    from flash.runner.lifecycle import _select_candidate

    only = Candidate("runpod", "H200", 4.0, 141)
    cands = (only,)
    # Attempt 0: the single class.
    assert _select_candidate(cands, set(), set()) is only
    # After it failed infra-shaped (provider burned, class tried), the next retry re-picks the SAME
    # class — there is nowhere else to walk, and the picker must not break.
    assert _select_candidate(cands, {"runpod"}, {("runpod", "H200")}) is only


def test_runpod_no_capacity_retry_escapes_to_other_provider(orch, monkeypatch):
    """Issue 7: a RunPod queue / no-capacity failure must retry on a DIFFERENT provider
    (Lambda) rather than walking to the next RunPod class while Lambda sits available."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("runpod", "RTX A6000", 0.49, 48),  # cheapest -> attempt 0
        Candidate("runpod", "RTX 6000 Ada", 0.50, 48),  # next RunPod class (the WRONG retry target)
        Candidate("lambda", "RTX A6000", 0.50, 48),  # the right cross-provider escape
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    rp_gpus, lam_gpus = [], []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep1", "j1"))
        return PollResult(
            False, failure="no_capacity", detail="job stuck IN_QUEUE (no RunPod capacity)"
        )

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle())
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, 0, log)
    assert metrics["train_tokens"] == 4096
    assert rp_gpus == ["RTX A6000"]  # RunPod tried exactly once...
    assert lam_gpus == ["RTX A6000"]  # ...then the retry escaped cross-provider to Lambda
    assert orch.get_status(spec.run_id).remote["provider"] == "lambda"
    assert "walking past the cheapest class" in log.getvalue()


def test_auto_cache_run_gets_cacheless_fallback_at_zero_retries(orch, monkeypatch):
    """The platform auto-attaches the SHARED weight cache, so its endpoint-pinning DC-set
    restriction must not cost the user a GPU-walk retry: a no_capacity on the cached spec earns ONE
    extra, cache-less cross-region attempt even at max_retries=0 (else the auto-cache could fail a
    run a cache-less launch would have won). Regression for the cache-fallback-vs-retry-budget gap.
    """
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: _alloc(candidates=(Candidate("runpod", "RTX A6000", 0.49, 48),)),
    )
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    volumes_seen = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        vol = getattr(run_spec.gpu, "network_volume", None)
        volumes_seen.append(vol)
        if on_handle:
            on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        # Cache-attached attempt -> no_capacity (the cache's DC set is starved); the cache-less
        # fallback attempt -> success.
        if vol == WEIGHT_CACHE_VOLUME_NAME:
            return PollResult(False, failure="no_capacity", detail="IN_QUEUE (cache DC set starved)")
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=0, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["train_tokens"] == 4096
    # Exactly two attempts: the cache-attached one (no_capacity) then the bonus cache-less retry.
    assert volumes_seen == [WEIGHT_CACHE_VOLUME_NAME, None]


def test_cache_fallback_does_not_consume_gpu_walk_retry(orch, monkeypatch):
    """The cache-drop fallback is a FREE attempt — it must not spend the user's GPU-walk retry budget.
    With max_retries=1 and both the cache attempt AND the cache-less same-class fallback hitting
    no_capacity, the run must still walk to a DIFFERENT class (its one real retry), not stop after the
    cache drop. Regression for the cache-fallback-steals-the-only-retry gap (the stop check counted the
    cache-drop attempt against max_retries)."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    candidates = (
        Candidate("runpod", "RTX A6000", 0.49, 48),  # cheapest -> cache attempt, then cache-less same class
        Candidate("runpod", "RTX 6000 Ada", 0.50, 48),  # the GPU-walk target the real retry must reach
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: None)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    seen = []

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        vol = getattr(run_spec.gpu, "network_volume", None)
        seen.append((run_spec.gpu.type, vol))
        if on_handle:
            on_handle(_runpod_handle(f"ep{attempt}", f"j{attempt}"))
        # Cache attempt AND its cache-less same-class fallback both starve; only the walk to the OTHER
        # class (the genuine retry that must survive the cache drop) succeeds.
        if run_spec.gpu.type == "RTX 6000 Ada":
            return PollResult(True, metrics={"train_tokens": 4096})
        return PollResult(False, failure="no_capacity", detail="IN_QUEUE (no capacity)")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec(max_retries=1, network_volume=WEIGHT_CACHE_VOLUME_NAME, network_volume_gb=100)
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["train_tokens"] == 4096
    # cache attempt -> cache-less SAME class (free fallback) -> GPU-walk to the OTHER class (real retry).
    assert seen == [
        ("RTX A6000", WEIGHT_CACHE_VOLUME_NAME),
        ("RTX A6000", None),
        ("RTX 6000 Ada", None),
    ]


def test_broken_gpu_preempt_retries_on_other_provider(orch, monkeypatch):
    """Issue 5: a Lambda instance whose CUDA never inits fails job_preempted; the retry must move to
    a DIFFERENT provider (RunPod) instead of re-rolling another broken Lambda instance, and the sick
    instance is torn down before the retry."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("lambda", "A10", 0.40, 24),  # cheapest -> attempt 0 (lands on broken instance)
        Candidate("lambda", "RTX A6000", 0.45, 48),  # next class on the SAME (sick) provider
        Candidate("runpod", "RTX A6000", 0.49, 48),  # the right cross-provider escape
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    terminated = []
    monkeypatch.setattr(
        lambda_api, "terminate_instances", lambda ids: terminated.extend(ids) or list(ids)
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    lam_gpus, rp_gpus = [], []

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle("i-broken"))
        return PollResult(
            False,
            failure="job_preempted",
            detail="GPU never became ready after 12 tries: cuda not available",
        )

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep2", "j2"))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    log = io.StringIO()
    metrics = orch._submit_seed_supervised(spec, 0, log)
    assert metrics["train_tokens"] == 4096
    assert lam_gpus == ["A10"]  # broken Lambda instance tried once...
    assert rp_gpus == ["RTX A6000"]  # ...then escaped cross-provider to RunPod
    assert "i-broken" in terminated  # sick instance torn down before the retry
    assert orch.get_status(spec.run_id).remote["provider"] == "runpod"


def test_no_liveness_stalled_escapes_to_other_provider(orch, monkeypatch):
    """The new fast first-liveness failover returns failure='stalled' (a sick region where the box
    reached 'active' but the worker never booted). It is infra-shaped, so the retry ESCAPES to a
    different provider rather than re-rolling the same sick substrate — the observed us-east-1 /
    CANADA-1 case, now caught in ~15 min instead of ~50."""
    from flash.providers import allocator
    from flash.providers.base import Candidate, PollResult
    from flash.providers.lambdalabs import api as lambda_api
    from flash.providers.lambdalabs import jobs as lambda_jobs
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs as rp_jobs

    candidates = (
        Candidate("lambda", "RTX A6000", 0.45, 48),  # cheapest -> attempt 0 (sick region)
        Candidate("runpod", "RTX A6000", 0.49, 48),  # the cross-provider escape
    )
    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc(candidates=candidates))
    monkeypatch.setattr(lambda_api, "terminate_instances", lambda ids: list(ids))
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: True)

    lam_gpus, rp_gpus = [], []

    def fake_lam(spec, seed, log=None, on_handle=None, attempt=0, **kw):
        lam_gpus.append(spec.gpu.type)
        if on_handle:
            on_handle(_lambda_handle())
        return PollResult(
            False,
            failure="stalled",
            detail="no worker liveness (boot.log/heartbeat) within 900s of instance active",
        )

    def fake_rp(run_spec, seed, log=None, on_handle=None, attempt=0, **kw):
        rp_gpus.append(run_spec.gpu.type)
        on_handle(_runpod_handle("ep3", "j3"))
        return PollResult(True, metrics={"train_tokens": 4096})

    monkeypatch.setattr(lambda_jobs, "submit_run_lambda", fake_lam)
    monkeypatch.setattr(rp_jobs, "submit_run", fake_rp)
    spec = _spec()
    _seed_status(orch, spec)
    metrics = orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert metrics["train_tokens"] == 4096
    assert lam_gpus == ["RTX A6000"]  # sick region tried once...
    assert rp_gpus == ["RTX A6000"]  # ...then escaped cross-provider to RunPod
    assert orch.get_status(spec.run_id).remote["provider"] == "runpod"


def test_genuine_worker_error_does_not_retry(orch, monkeypatch):
    from flash.providers import allocator
    from flash.providers.base import PollResult
    from flash.providers.runpod import jobs as rp_jobs

    monkeypatch.setattr(allocator, "allocate", lambda *a, **k: _alloc())
    calls = []

    def fake_runpod_submit(run_spec, seed, log=None, on_handle=None, attempt=0, **kwargs):
        calls.append(attempt)
        return PollResult(False, failure="job_failed", detail="ValueError: bad reward fn")

    monkeypatch.setattr(rp_jobs, "submit_run", fake_runpod_submit)
    spec = _spec()
    _seed_status(orch, spec)
    with pytest.raises(RuntimeError, match="bad reward fn"):
        orch._submit_seed_supervised(spec, 0, io.StringIO())
    assert calls == [0]


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
    st.remote = {"endpoint_id": "ep1", "endpoint_name": "n", "job_id": "j1"}
    orch._save_status(st)
    out = orch.cancel_run(spec.run_id)
    assert out.state == "cancelled"
    assert cancelled_jobs == [("ep1", "j1")]
    assert "ep1" in deleted_eps


def test_config_gpu_fields(monkeypatch):
    from flash.schema import spec_from_dict

    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "sft",
        "train": {"epochs": 1, "hf_repo": "owner/runs"},
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
    }
    spec = spec_from_dict(dict(base), run_id="x")
    assert spec.gpu.type == "RTX A6000"
    again = JobSpec.from_dict(spec.to_dict())
    assert again.gpu.type == "RTX A6000"
    spec = spec_from_dict({**base, "gpu": {"type": "A100 SXM"}}, run_id="x")
    assert spec.gpu.type == "RTX A6000"
