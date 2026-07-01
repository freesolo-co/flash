"""Run-execution machinery: the submit -> supervised training job -> GC flow.

Sibling helpers are imported function-locally to avoid the flash.runner.__init__ import cycle
and to keep monkeypatches reachable (``monkeypatch.setattr(runner, ...)`` vs a static copy).
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass

from flash.spec import JobSpec

# Floor so a streak of broken/busy GPUs doesn't kill a run that left retries enabled.
# max_retries==0 (single-shot) is always respected; floor only applies when retries are on.
INFRA_RETRY_FLOOR = 5
INFRA_RETRY_FAILURES = frozenset({"stalled", "no_capacity", "poll_error", "job_preempted"})
RETRY_FAILURES = INFRA_RETRY_FAILURES | {"oom"}


@dataclass
class _RetryBudget:
    infra_retries: int
    oom_retries: int
    cache_fallbacks: int
    infra_used: int = 0
    oom_used: int = 0

    @property
    def max_attempts(self) -> int:
        return 1 + self.infra_retries + self.oom_retries + self.cache_fallbacks

    def infra_exhausted(self, *, cache_fallback_available: bool) -> bool:
        return self.infra_used >= self.infra_retries and not cache_fallback_available

    def can_retry(self, failure: str | None, *, cache_drop: bool) -> bool:
        if failure not in RETRY_FAILURES:
            return False
        if cache_drop:
            return True
        if failure == "oom":
            return self.oom_used < self.oom_retries
        return self.infra_used < self.infra_retries

    def record_retry(self, failure: str | None, *, cache_drop: bool) -> None:
        if cache_drop:
            return
        if failure == "oom":
            self.oom_used += 1
        elif failure in INFRA_RETRY_FAILURES:
            self.infra_used += 1


def _run_job(spec: JobSpec, runtime_secrets: dict[str, str] | None = None) -> None:
    # Lazy import: dry-run / unit tests never construct a Flash endpoint.
    from flash.providers.runpod.train import upload_code
    from flash.runner import (
        RUNS_DIR,
        TERMINAL_STATES,
        _gc_run_endpoints,
        _run_job_inner,
        _update,
        get_status,
    )

    # Cancel can land before this thread starts; don't overwrite a terminal state with provisioning.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        return
    _update(spec.run_id, "provisioning")
    log_path = os.path.join(RUNS_DIR, f"{spec.run_id}.log")
    try:
        _run_job_inner(spec, log_path, upload_code, runtime_secrets=runtime_secrets)
    finally:
        # GC registered endpoints — undeleted endpoints count against the account-wide worker quota.
        _gc_run_endpoints(spec)


def _spec_with_gpu(spec: JobSpec, gpu_type: str) -> JobSpec:
    """The spec the workers/loggers see for THIS attempt's allocated class."""
    if spec.gpu.type == gpu_type:
        return spec
    d = spec.to_dict()
    d["gpu"] = {**d["gpu"], "type": gpu_type}
    return JobSpec.from_dict(d)


def _drop_weight_cache(spec: JobSpec) -> JobSpec:
    """Spec with the SHARED weight-cache volume removed for an unrestricted cross-region retry.

    Only drops the platform-managed shared cache (WEIGHT_CACHE_VOLUME_NAME); a custom per-org
    network_volume is the user's own choice and is preserved across retries.
    """
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    if getattr(spec.gpu, "network_volume", None) != WEIGHT_CACHE_VOLUME_NAME:
        return spec
    d = spec.to_dict()
    d["gpu"] = {**d["gpu"], "network_volume": None}
    return JobSpec.from_dict(d)


def _select_candidate(candidates, failed_providers: set[str], tried_classes: set[tuple[str, str]]):
    """Pick the next (provider, class) from the cross-provider ranked candidate list.

    Escapes a congested/sick provider cross-provider before walking classes within it.
    Falls back to cheapest untried class when every provider has already burned a retry.
    """
    return min(
        candidates,
        key=lambda c: (
            c.provider in failed_providers,  # 1) escape providers that already failed this run
            (c.provider, c.gpu) in tried_classes,  # 2) then prefer a class not yet tried
            c.hourly_usd,  # 3) then cheapest
            c.vram_gb,  # 4) then the smaller card (don't burn a big GPU on a small job)
        ),
    )


def _oom_escalated(candidates, oom_vram_floor: int):
    """Candidates strictly LARGER than the VRAM that just OOM'd. ``oom_vram_floor == 0`` (no prior OOM)
    leaves the list unchanged; otherwise an 80GB OOM leaves only the >80GB classes (a same-size retry
    would just OOM again). EMPTY means the run already OOM'd the largest available class."""
    if not oom_vram_floor:
        return list(candidates)
    return [c for c in candidates if c.vram_gb > oom_vram_floor]


def _submit_seed_supervised(
    spec: JobSpec,
    seed: int,
    log,
    runtime_secrets: dict[str, str] | None = None,
    code_prefix: str | None = None,
) -> dict:
    """Run one seed with bounded auto-retry on infra-shaped failures.

    Retries resume from the latest HF checkpoint on a fresh host. Genuine worker errors fail fast.
    """
    from flash.providers import get_provider
    from flash.providers.allocator import allocate, allocation_summary
    from flash.providers.base import PollResult
    from flash.runner import (
        TERMINAL_STATES,
        _RunCancelled,
        _spec_with_gpu,
        _update,
        flash_code_prefix,
        get_status,
    )

    code_prefix = code_prefix or flash_code_prefix()
    last_handle: dict = {}
    current_gpu: dict = {}
    # Persisted into the run handle so attach_run recovery polls with the same stall tuning.
    current_on_last_gpu: dict = {"value": False}
    current_attempt: dict = {"value": 0}
    # Tracks rN-suffixed retry endpoint ids that _gc_run_endpoints can't reconstruct by name.
    seen_endpoints: set[str] = set()

    def on_handle(handle: dict):
        last_handle.clear()
        last_handle.update(handle)
        if handle.get("endpoint_id"):
            seen_endpoints.add(handle["endpoint_id"])
        _update(
            spec.run_id,
            "running",
            remote={
                **handle,
                "seed": int(seed),
                "allocated_gpu": current_gpu.get("name"),
                "on_last_gpu": bool(current_on_last_gpu["value"]),
                "attempt": int(current_attempt["value"]),
                "code_prefix": code_prefix,
            },
        )

    def _gc_seen_endpoints() -> None:
        if not seen_endpoints:
            return
        from flash.providers.runpod import api as runpod_api

        for eid in seen_endpoints:
            with contextlib.suppress(Exception):
                runpod_api.delete_endpoint(eid)

    def _cancel() -> _RunCancelled:
        """Reap this seed's tracked endpoints before unwinding on cancel — a handle whose `running`
        write loses the terminal-stickiness race never lands in status.remote, so only seen_endpoints
        (rN walk endpoints _gc_run_endpoints can't name) can free it."""
        _gc_seen_endpoints()
        return _RunCancelled(f"run {spec.run_id} was cancelled")

    max_retries = int(spec.gpu.max_retries)
    infra_budget = max(max_retries, INFRA_RETRY_FLOOR) if max_retries else 0
    last_detail = None
    # Sticky: once dropped stays dropped so all remaining attempts run on the unrestricted all-DC pool.
    drop_weight_cache = False
    # One free cache-less fallback when the shared cache's DC-set restriction may have caused no_capacity.
    # A non-shared per-org volume earns no bonus — that's the user's own choice.
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    started_with_shared_cache = (
        getattr(spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
    )
    cache_fallback_attempts = 1 if started_with_shared_cache else 0
    retry_budget = _RetryBudget(infra_budget, max_retries, cache_fallback_attempts)
    # Grow only when an attempt actually provisioned a class and lost it to infra.
    failed_providers: set[str] = set()
    tried_classes: set[tuple[str, str]] = set()
    oom_vram_floor = 0
    for attempt in range(retry_budget.max_attempts):
        if attempt > 0 and last_handle:
            if last_handle.get("endpoint_id"):
                try:
                    from flash.providers.runpod import api as runpod_api

                    runpod_api.cancel_job(last_handle["endpoint_id"], last_handle["job_id"])
                    runpod_api.delete_endpoint(last_handle["endpoint_id"])
                    print(
                        f"retry {attempt}: deleted endpoint {last_handle['endpoint_id']} "
                        "(escaping throttled/sick host)",
                        file=log,
                        flush=True,
                    )
                except Exception:
                    pass
            elif last_handle.get("provider") == "lambda":
                # Instance-based providers bill until terminated; destroy before retry to stop paying.
                with contextlib.suppress(Exception):
                    from flash.providers import get_provider
                    from flash.providers.base import JobHandle

                    _prov = last_handle["provider"]
                    get_provider(_prov).destroy(JobHandle.from_dict(last_handle))
                    _iid = last_handle.get("instance_id")
                    print(
                        f"retry {attempt}: terminated {_prov} instance {_iid} (escaping sick host)",
                        file=log,
                        flush=True,
                    )
            # Clear the stale handle before the fresh deploy so a restart doesn't reattach to it.
            with contextlib.suppress(FileNotFoundError):
                st = get_status(spec.run_id)
                if st.state not in TERMINAL_STATES and st.remote is not None:
                    _update(spec.run_id, st.state, remote=None)
        res = None
        alloc = None
        chosen = None
        # A cancel can land after _run_training's pre-submit check but while
        # allocation/pricing runs, when no handle exists yet for cancel_run() to
        # delete. Re-read state right before paid provisioning so a cancelled run
        # never launches a worker (the later checks only stop the final-state
        # overwrite, after the GPU has already run and billed).
        with contextlib.suppress(FileNotFoundError):
            if get_status(spec.run_id).state == "cancelled":
                raise _cancel()
        try:
            alloc = allocate(
                spec.model,
                spec.algorithm,
                train=spec.train,
                thinking=spec.thinking,
            )
        except Exception as exc:
            from flash.providers.base import UnsupportedGpuError

            if isinstance(exc, UnsupportedGpuError):
                raise  # config-shaped: no GPU anywhere can run this job
            res = PollResult(False, failure="poll_error", detail=f"allocation: {exc}")
        if alloc is not None:
            with contextlib.suppress(FileNotFoundError):
                if get_status(spec.run_id).state == "cancelled":
                    raise _cancel()
            cands = _oom_escalated(alloc.candidates, oom_vram_floor)
            if not cands:
                last_detail = f"oom: exceeded the largest available GPU ({oom_vram_floor} GB)"
                print(
                    f"seed={seed} OOM on the largest GPU class ({oom_vram_floor} GB); not retrying",
                    file=log,
                    flush=True,
                )
                break
            chosen = _select_candidate(cands, failed_providers, tried_classes)
            untried = [c for c in cands if (c.provider, c.gpu) not in tried_classes]
            cache_fallback_available = (
                started_with_shared_cache
                and not drop_weight_cache
                and chosen is not None
                and chosen.provider == "runpod"
            )
            on_last_gpu = len(untried) <= 1 or (
                retry_budget.infra_exhausted(cache_fallback_available=cache_fallback_available)
            )
            current_on_last_gpu["value"] = on_last_gpu
            print(allocation_summary(alloc), file=log, flush=True)
            if (chosen.provider, chosen.gpu) != (alloc.provider, alloc.gpu):
                print(
                    f"retry {attempt}: walking past the cheapest class to {chosen.gpu} "
                    f"@ {chosen.provider} ${chosen.hourly_usd:.2f}/hr",
                    file=log,
                    flush=True,
                )
            run_spec = _spec_with_gpu(spec, chosen.gpu)
            if drop_weight_cache:
                run_spec = _drop_weight_cache(run_spec)
            current_gpu["name"] = chosen.gpu
            current_attempt["value"] = attempt
            provider = get_provider(chosen.provider)
            try:
                submit_kwargs = {
                    "log": log,
                    "on_handle": on_handle,
                    "attempt": attempt,
                    "on_last_gpu": on_last_gpu,
                    "code_prefix": code_prefix,
                }
                if runtime_secrets:
                    submit_kwargs["runtime_secrets"] = runtime_secrets
                res = provider.submit_run(run_spec, seed, **submit_kwargs)
            except Exception as exc:
                res = PollResult(False, failure="poll_error", detail=f"deploy/submit: {exc}")
                if attempt < infra_budget:
                    time.sleep(10 * (attempt + 1))  # let the transient clear
        if res.ok:
            # A late worker success must not resurrect a cancelled run.
            try:
                if get_status(spec.run_id).state == "cancelled":
                    raise _cancel()
            except FileNotFoundError:
                pass
            _gc_seen_endpoints()
            if chosen is not None and isinstance(res.metrics, dict):
                res.metrics.setdefault("allocated_gpu", chosen.gpu)
            return res.metrics
        last_detail = f"{res.failure}: {res.detail}"
        oom_shaped = res.failure == "oom"
        if oom_shaped and chosen is not None:
            oom_vram_floor = max(oom_vram_floor, chosen.vram_gb)
        # Cancel wins over any retry-shaped failure.
        try:
            if get_status(spec.run_id).state == "cancelled":
                raise _cancel()
        except FileNotFoundError:
            pass
        run_had_cache = bool(
            chosen is not None
            and chosen.provider == "runpod"
            and getattr(run_spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
        )
        first_cache_drop = (
            run_had_cache and not drop_weight_cache and res.failure in ("no_capacity", "poll_error")
        )
        oom_mode = oom_vram_floor > 0
        will_retry = retry_budget.can_retry(
            res.failure,
            cache_drop=first_cache_drop,
        )
        action = (
            f"retrying on a larger GPU (> {oom_vram_floor} GB)"
            if (will_retry and oom_mode)
            else "retrying (resume from last checkpoint)"
            if will_retry
            else "not retrying"
        )
        print(
            f"seed={seed} attempt={attempt} failed ({res.failure}); {action}"
            f"\n--- failure detail ---\n{(res.detail or '')[:2000]}\n---",
            file=log,
            flush=True,
        )
        if not will_retry:
            break
        if first_cache_drop:
            drop_weight_cache = True
            retry_budget.record_retry(res.failure, cache_drop=True)
        else:
            retry_budget.record_retry(res.failure, cache_drop=False)
            if chosen is not None:
                if not oom_shaped:
                    failed_providers.add(chosen.provider)
                tried_classes.add((chosen.provider, chosen.gpu))
    _gc_seen_endpoints()
    raise RuntimeError(f"seed {seed} failed after retries: {last_detail}")


def _run_job_inner(
    spec: JobSpec,
    log_path: str,
    upload_code,
    runtime_secrets: dict[str, str] | None = None,
) -> None:
    from flash.runner import _run_training, _RunCancelled, _update, flash_code_prefix, get_status

    try:
        code_prefix = flash_code_prefix()
        upload_code(spec.train.hf_repo, code_prefix=code_prefix)
        with open(log_path, "a") as log:
            _run_training(
                spec,
                log,
                prior_cost=0.0,
                runtime_secrets=runtime_secrets,
                code_prefix=code_prefix,
            )
    except _RunCancelled:
        return  # cancel_run already set the terminal state
    except Exception as exc:
        if get_status(spec.run_id).state != "cancelled":
            _update(spec.run_id, "failed", error=str(exc))
        raise


def _run_training(
    spec: JobSpec,
    log,
    *,
    prior_cost: float,
    runtime_secrets: dict[str, str] | None = None,
    code_prefix: str | None = None,
) -> None:
    """Train the run's single adapter under supervision; finalize the run.

    Shared by a fresh submit and post-restart recovery (the worker resumes from its last HF
    checkpoint on a fresh allocation). ``prior_cost`` carries spend already booked before a
    recovery so the total isn't under-reported."""
    from flash.runner import (
        FIXED_SEED,
        TERMINAL_STATES,
        _persist_metrics,
        _RunCancelled,
        _submit_seed_supervised,
        _update,
        artifacts_dir,
        charge_usd_for_spec,
        get_status,
    )

    # Defense in depth against the recovery TOCTOU (see attach_run): a run can be flipped into ANY
    # terminal state — not just `cancelled` — by a concurrent thread/process between the resume
    # decision and here. Bail before _update + the supervised submit so we never submit PAID GPU
    # work for an already-terminal run. _RunCancelled is the terminal signal; callers swallow it.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        raise _RunCancelled(f"run {spec.run_id} is already terminal; not submitting")
    # The pre-check above closes most of the window, but a concurrent flip can still land between
    # it and this transition. _update is a compare-and-set: it returns False when the run is already
    # terminal and leaves the state untouched. Gate the PAID supervised submit on that result so a
    # run cancelled in this last instant is never resumed onto a GPU.
    if not _update(spec.run_id, "running"):
        raise _RunCancelled(f"run {spec.run_id} went terminal before submit; not submitting")
    print(
        f"starting phase={spec.phase} model={spec.model} gpu={spec.gpu.type}",
        file=log,
        flush=True,
    )
    metrics = _submit_seed_supervised(
        spec, FIXED_SEED, log, runtime_secrets=runtime_secrets, code_prefix=code_prefix
    )
    # measured wall x $/hr is recorded in metrics.json for analytics, but is NOT what we charge.
    measured_cost = prior_cost + _persist_metrics(spec, metrics)
    # The customer is charged the QUOTE: the flash.cost estimate this run was priced at (planned
    # steps). Falls back to the measured cost only if the spec can't be re-priced.
    charge_usd = charge_usd_for_spec(spec, fallback=measured_cost)
    # A cancel can land while this thread writes metrics — after the supervised late-cancel check.
    # Re-read before the terminal "done" so a late worker success doesn't resurrect a cancelled run.
    with contextlib.suppress(FileNotFoundError):
        if get_status(spec.run_id).state == "cancelled":
            raise _RunCancelled(f"run {spec.run_id} was cancelled")
    # Gate side effects on the CAS succeeding — a concurrent cancel rejects the `done` write.
    applied = _update(
        spec.run_id,
        "done",
        cost_usd=charge_usd,
        artifacts_dir=artifacts_dir(spec),
    )
    print(
        f"done: train_wall={metrics.get('wall_seconds')} measured={measured_cost:.4f} "
        f"charge_usd={charge_usd:.4f}",
        file=log,
        flush=True,
    )
    if applied:
        _charge_completed_run_best_effort(spec, log)
        _register_checkpoints_best_effort(spec, log)


def _register_checkpoints_best_effort(spec: JobSpec, log) -> None:
    """Mirror a finished run's per-step checkpoints to the backend store (best-effort)."""
    from flash.runner import get_status

    try:
        from flash.server.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(get_status(spec.run_id), log=log)
    except Exception as exc:  # never let checkpoint bookkeeping disturb a run
        print(f"[ckpt] register warn ({spec.run_id}): {exc}", file=log, flush=True)


def _charge_completed_run_best_effort(spec: JobSpec, log) -> None:
    """Bill a successfully completed external run without changing its training result."""
    _charge_completed_run_by_id(spec.run_id, log)


def _charge_completed_run_by_id(run_id: str, log) -> None:
    """Bill a completed external run by run id, without changing its training result.

    The charge reads everything it needs from the persisted ``RunStatus`` (``billing_context`` +
    ``cost_usd`` + the raw ``spec`` dict), so a run id is the only input. The retry sweep calls this
    directly so a legacy/stale persisted spec that ``JobSpec.from_dict`` would reject does NOT block
    recovery of a real pending/failed charge."""
    from flash.server.billing import charge_completed_run

    _apply_charge_with_state(
        run_id,
        log,
        # "terminal" not "completed": the retry sweep bills cancelled-mid-training runs through here
        # too, so the not-billed log/error text must read accurately for both.
        noun="terminal",
        charge_call=lambda internal_key, status: charge_completed_run(
            internal_key=internal_key, status=status
        ),
    )


def _apply_charge_with_state(run_id: str, log, *, charge_call, noun: str) -> None:
    """Drive the billing state machine around one charge attempt (charging -> charged/failed).

    ``charge_call(internal_key, status)`` performs the actual backend charge and returns its response
    dict. Reading org/cost from the
    persisted ``RunStatus`` (never a reparsed spec) is what lets a legacy/stale spec still be charged.
    """
    from flash.runner import get_status, record_billing_state
    from flash.server.auth import INTERNAL_KEY_ENV
    from flash.server.billing import BillingError

    status = get_status(run_id)
    if not status.billing_context or status.billing_state == "charged":
        return

    internal_key = os.environ.get(INTERNAL_KEY_ENV, "").strip()
    if not internal_key:
        detail = f"{INTERNAL_KEY_ENV} is not configured; {noun} run was not billed"
        # Field-only billing write that re-reads state under the lock: never overwrite a `deployed`
        # that a concurrent /deploy may have written since we last read the run.
        record_billing_state(run_id, billing_state="failed", billing_error=detail)
        print(f"billing failed: {detail}", file=log, flush=True)
        return

    record_billing_state(run_id, billing_state="charging", billing_error=None)
    status = get_status(run_id)
    try:
        charge = charge_call(internal_key, status)
    except BillingError as exc:
        record_billing_state(run_id, billing_state="failed", billing_error=exc.detail)
        print(f"billing failed: {exc.detail}", file=log, flush=True)
        return

    record_billing_state(
        run_id,
        billing_state="charged",
        billing_error=None,
        billing_charge=charge,
    )
    print(
        f"billing charged: amount_cents={charge.get('amountCents')} "
        f"replay={bool(charge.get('replay'))}",
        file=log,
        flush=True,
    )


def _gc_run_endpoints(spec: JobSpec) -> None:
    """Best-effort teardown of every endpoint a run may have registered."""
    from flash.runner import get_status

    status = None
    with contextlib.suppress(Exception):
        status = get_status(spec.run_id)
    if status is not None and status.remote:
        try:
            from flash.providers import get_provider
            from flash.providers.base import JobHandle

            handle = JobHandle.from_dict(status.remote)
            get_provider(handle.provider).destroy(handle)
        except Exception:
            pass
    try:
        # RunPod gc reaps rN-suffixed endpoints the persisted handle can't name.
        from flash.providers import get_provider

        get_provider("runpod").gc(spec)
    except Exception:
        pass
    # Lambda bills until terminated; gc catches any instance left behind by a crashed supervisor.
    from flash.providers import available_providers, get_provider

    _avail = available_providers()
    for _prov in ("lambda",):
        if _prov in _avail:
            with contextlib.suppress(Exception):
                get_provider(_prov).gc(spec)
