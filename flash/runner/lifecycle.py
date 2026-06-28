"""Run-execution machinery: the submit -> seed-loop -> per-seed supervised job -> GC flow.

Sibling helpers are imported function-locally to avoid the flash.runner.__init__ import cycle
and to keep monkeypatches reachable (``monkeypatch.setattr(runner, ...)`` vs a static copy).
"""

from __future__ import annotations

import contextlib
import os
import time

from flash.spec import JobSpec

# Floor so a streak of broken/busy GPUs doesn't kill a run that left retries enabled.
# max_retries==0 (single-shot) is always respected; floor only applies when retries are on.
INFRA_RETRY_FLOOR = 5


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


def _submit_seed_supervised(
    spec: JobSpec,
    seed: int,
    log,
    runtime_secrets: dict[str, str] | None = None,
) -> dict:
    """Run one seed with bounded auto-retry on infra-shaped failures.

    Retries resume from the latest HF checkpoint on a fresh host. Genuine worker errors fail fast.
    """
    from flash.providers import get_provider
    from flash.providers.allocator import allocate, allocation_summary
    from flash.providers.base import PollResult
    from flash.runner import TERMINAL_STATES, _RunCancelled, _spec_with_gpu, _update, get_status

    last_handle: dict = {}
    current_gpu: dict = {}
    # Persisted into the run handle so attach_run recovery polls with the same stall tuning.
    current_on_last_gpu: dict = {"value": False}
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

    started_with_shared_cache = getattr(spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
    cache_fallback_attempts = 1 if started_with_shared_cache else 0
    # Grow only when an attempt that actually provisioned a class lost it to infra — never on failed alloc.
    failed_providers: set[str] = set()
    tried_classes: set[tuple[str, str]] = set()
    # walk_attempt excludes cache-drop attempts so the GPU walk gets its full max_retries budget.
    cache_drop_consumed = 0
    for attempt in range(infra_budget + 1 + cache_fallback_attempts):
        walk_attempt = attempt - cache_drop_consumed
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
            chosen = _select_candidate(alloc.candidates, failed_providers, tried_classes)
            untried = [c for c in alloc.candidates if (c.provider, c.gpu) not in tried_classes]
            # Don't let the budget clause mark last-GPU when a cache-drop fallback is still available;
            # class exhaustion (len(untried) <= 1) still marks it regardless.
            cache_fallback_available = (
                started_with_shared_cache
                and not drop_weight_cache
                and chosen is not None
                and chosen.provider == "runpod"
            )
            on_last_gpu = len(untried) <= 1 or (
                walk_attempt >= infra_budget and not cache_fallback_available
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
            provider = get_provider(chosen.provider)
            try:
                submit_kwargs = {
                    "log": log,
                    "on_handle": on_handle,
                    "attempt": attempt,
                    "on_last_gpu": on_last_gpu,
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
        infra_shaped = res.failure in ("stalled", "no_capacity", "poll_error", "job_preempted")
        # A cancel deletes the endpoint, which the poller sees as infra-shaped; cancel wins.
        try:
            if get_status(spec.run_id).state == "cancelled":
                raise _cancel()
        except FileNotFoundError:
            pass
        # Drop the shared weight-cache volume on no_capacity/poll_error so the retry runs unrestricted.
        # Non-volume flakes (stall/preempt) keep the cache. Lambda no_capacity isn't cache-caused.
        run_had_cache = bool(
            chosen is not None
            and chosen.provider == "runpod"
            and getattr(run_spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
        )
        first_cache_drop = (
            run_had_cache
            and not drop_weight_cache
            and res.failure in ("no_capacity", "poll_error")
        )
        print(
            f"seed={seed} attempt={attempt} failed ({res.failure}); "
            f"{'retrying (resume from last checkpoint)' if infra_shaped and (walk_attempt < infra_budget or first_cache_drop) else 'not retrying'}"
            f"\n--- failure detail ---\n{(res.detail or '')[:2000]}\n---",
            file=log,
            flush=True,
        )
        if not infra_shaped:
            break
        if walk_attempt >= infra_budget and not first_cache_drop:
            break
        if first_cache_drop:
            drop_weight_cache = True
            # Exclude from budget so max_retries GPU-walk retries remain after the cache-drop.
            cache_drop_consumed += 1
            # Retry the same GPU class without the volume first; only walk the class if that also fails.
        elif chosen is not None:
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
    from flash.runner import _run_training, _RunCancelled, _update, get_status

    try:
        upload_code(spec.train.hf_repo)
        with open(log_path, "a") as log:
            _run_training(spec, log, prior_cost=0.0, runtime_secrets=runtime_secrets)
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
    metrics = _submit_seed_supervised(spec, FIXED_SEED, log, runtime_secrets=runtime_secrets)
    total_cost = prior_cost + _persist_metrics(spec, metrics)
    # A cancel can land while this thread writes metrics — after the supervised late-cancel check.
    # Re-read before the terminal "done" so a late worker success doesn't resurrect a cancelled run.
    with contextlib.suppress(FileNotFoundError):
        if get_status(spec.run_id).state == "cancelled":
            raise _RunCancelled(f"run {spec.run_id} was cancelled")
    # Gate side effects on the CAS succeeding — a concurrent cancel rejects the `done` write.
    applied = _update(
        spec.run_id,
        "done",
        cost_usd=total_cost,
        artifacts_dir=artifacts_dir(spec),
    )
    print(
        f"done: train_wall={metrics.get('wall_seconds')} cost_usd={total_cost:.4f}",
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
    from flash.runner import get_status, record_billing_state
    from flash.server.auth import INTERNAL_KEY_ENV
    from flash.server.billing import BillingError, charge_completed_run

    status = get_status(run_id)
    if not status.billing_context or status.billing_state == "charged":
        return

    internal_key = os.environ.get(INTERNAL_KEY_ENV, "").strip()
    if not internal_key:
        detail = f"{INTERNAL_KEY_ENV} is not configured; completed run was not billed"
        # Field-only billing write that re-reads state under the lock: never overwrite a `deployed`
        # that a concurrent /deploy may have written since we last read the run.
        record_billing_state(run_id, billing_state="failed", billing_error=detail)
        print(f"billing failed: {detail}", file=log, flush=True)
        return

    record_billing_state(run_id, billing_state="charging", billing_error=None)
    status = get_status(run_id)
    try:
        charge = charge_completed_run(internal_key=internal_key, status=status)
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
