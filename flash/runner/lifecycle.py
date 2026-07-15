"""Run-execution machinery: the submit -> supervised training job -> GC flow.

Sibling helpers are imported function-locally to avoid the flash.runner.__init__ import cycle
and to keep monkeypatches reachable (``monkeypatch.setattr(runner, ...)`` vs a static copy).
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass

from flash.opd_retry_contract import OPD_RESUME_REVISION_ENV
from flash.providers._deadline import deadline_kwargs
from flash.spec import JobSpec, require_matching_seed

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
    cache_used: int = 0

    @property
    def max_attempts(self) -> int:
        return 1 + self.infra_retries + self.oom_retries + self.cache_fallbacks

    def infra_exhausted(self, *, cache_fallback_available: bool) -> bool:
        return self.infra_used >= self.infra_retries and not cache_fallback_available

    def can_retry(self, failure: str | None, *, cache_drop: bool) -> bool:
        if failure not in RETRY_FAILURES:
            return False
        if cache_drop:
            return self.cache_used < self.cache_fallbacks
        if failure == "oom":
            return self.oom_used < self.oom_retries
        return self.infra_used < self.infra_retries

    def record_retry(self, failure: str | None, *, cache_drop: bool) -> None:
        if cache_drop:
            self.cache_used += 1
            return
        if failure == "oom":
            self.oom_used += 1
        elif failure in INFRA_RETRY_FAILURES:
            self.infra_used += 1


def _run_job(spec: JobSpec, runtime_secrets: dict[str, str] | None = None) -> None:
    # Lazy import: dry-run / unit tests never construct a Flash endpoint.
    from flash.providers._worker import upload_code
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
        # Skip when the run is still non-terminal: that means another live supervisor already owns the
        # durable handle (see _submit_seed_supervised's "already has a durable provider handle" bail),
        # and reaping here would tear down its still-active provider resources.
        if get_status(spec.run_id).state in TERMINAL_STATES:
            _gc_run_endpoints(spec)


def _spec_with_gpu(spec: JobSpec, gpu_type: str) -> JobSpec:
    """The spec the workers/loggers see for THIS attempt's allocated class."""
    if spec.gpu.type == gpu_type:
        return spec
    d = spec.to_internal_dict()
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
    d = spec.to_internal_dict()
    d["gpu"] = {**d["gpu"], "network_volume": None}
    return JobSpec.from_dict(d)


def _canonical_provider_handle(handle):
    """Validate and canonicalize one complete provider-specific persisted handle."""
    from flash.providers.base import JobHandle

    data = handle.to_dict() if hasattr(handle, "to_dict") else dict(handle)
    provider = data.get("provider")
    if provider == "runpod":
        from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle

        return JobHandle.from_dict(RunpodJobHandle.from_dict(data).to_dict())
    if provider == "lambda":
        from flash.providers.lambdalabs.jobs.builders import LambdaJobHandle

        return JobHandle.from_dict(LambdaJobHandle.from_dict(data).to_dict())
    if provider == "vast":
        from flash.providers.vast.jobs.builders import VastJobHandle

        return JobHandle.from_dict(VastJobHandle.from_dict(data).to_dict())
    raise ValueError("persisted provider identity is missing or unsupported")


def _strict_teardown_handle(handle) -> None:
    """Synchronously confirm provider teardown before any replacement provisioning."""
    from flash.providers import INSTANCE_PROVIDERS, get_provider

    handle = _canonical_provider_handle(handle)
    provider = get_provider(handle.provider)
    data = handle.to_dict()
    if handle.provider == "runpod":
        if data.get("job_id"):
            with contextlib.suppress(Exception):
                provider.cancel(handle)
        try:
            provider.destroy(handle)
        except Exception as exc:
            raise RuntimeError("runpod endpoint deletion could not be confirmed") from exc
        return
    if handle.provider in INSTANCE_PROVIDERS:
        provider.destroy(handle)
        return
    provider.cancel(handle)
    provider.destroy(handle)


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
    attempt_start: int = 0,
) -> dict:
    """Run one seed with bounded auto-retry on infra-shaped failures.

    Retries resume from the latest HF checkpoint on a fresh host. Genuine worker errors fail fast.
    ``attempt_start`` offsets persisted identities without expanding this invocation's retry budget.
    """
    seed = require_matching_seed(spec, seed)
    from flash.providers import get_provider
    from flash.providers.allocator import allocate, allocation_summary
    from flash.providers.base import PollResult
    from flash.runner import (
        TERMINAL_STATES,
        _compare_and_clear_remote,
        _load_run_deadline_at,
        _persist_effective_worker_spec,
        _preserve_cleanup_remote,
        _reserve_attempt,
        _RunCancelled,
        _spec_with_gpu,
        _spec_with_remaining_wall,
        _TerminalHandleRace,
        _update,
        _verified_opd_retry_state,
        flash_code_prefix,
        get_status,
    )
    from flash.server._locks import _deploy_lock

    code_prefix = code_prefix or flash_code_prefix()
    last_handle: dict = {}
    current_gpu: dict = {}
    # Persisted into the run handle so attach_run recovery polls with the same stall tuning.
    current_on_last_gpu: dict = {"value": False}
    attempt_start = max(0, int(attempt_start))
    current_attempt: dict = {"value": attempt_start}
    # tracks complete rN-suffixed retry handles that registry-less gc cannot reconstruct by name.
    seen_endpoints: dict[str, dict] = {}
    submission_lock = None

    def on_handle(handle: dict):
        nonlocal submission_lock

        try:
            selected_provider = current_gpu.get("provider")
            if not isinstance(selected_provider, str) or not selected_provider:
                raise RuntimeError("selected provider identity is unavailable")
            canonical = _canonical_provider_handle(handle)
            canonical_handle = canonical.to_dict()
            if canonical.provider != selected_provider:
                raise RuntimeError("provider handle identity does not match the selected provider")
            expected_attempt = int(current_attempt["value"])
            if canonical_handle["attempt"] != expected_attempt:
                raise RuntimeError("provider handle attempt does not match the reserved attempt")
            last_handle.clear()
            last_handle.update(canonical_handle)
            if canonical_handle.get("endpoint_id"):
                seen_endpoints[canonical_handle["endpoint_id"]] = dict(canonical_handle)
            persisted_handle = {
                **canonical_handle,
                "seed": int(seed),
                "allocated_gpu": current_gpu.get("name"),
                "on_last_gpu": bool(current_on_last_gpu["value"]),
                "code_prefix": code_prefix,
            }
            if _update(spec.run_id, "running", remote=persisted_handle):
                return
            cleanup_confirmed = False
            try:
                _strict_teardown_handle(canonical_handle)
                cleanup_confirmed = True
            except Exception:
                pass
            if cleanup_confirmed:
                last_handle.clear()
            else:
                _preserve_cleanup_remote(spec.run_id, persisted_handle)
            raise _TerminalHandleRace(
                f"run {spec.run_id} became terminal while its provider handle was being persisted"
            )
        finally:
            lock = submission_lock
            submission_lock = None
            if lock is not None:
                lock.release()

    def _gc_seen_endpoints() -> None:
        if not seen_endpoints:
            return
        from flash.providers import get_provider
        from flash.providers.base import JobHandle

        rp = get_provider("runpod")
        for remote in seen_endpoints.values():
            with contextlib.suppress(Exception):
                rp.destroy(JobHandle.from_dict(remote))

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
    # one cache-less fallback is available only when the user enabled retries; max_retries=0 is
    # exactly one provider submission. a non-shared per-org volume earns no bonus.
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    started_with_shared_cache = (
        getattr(spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
    )
    cache_fallback_attempts = 1 if started_with_shared_cache and max_retries > 0 else 0
    retry_budget = _RetryBudget(infra_budget, max_retries, cache_fallback_attempts)
    # Grow only when an attempt actually provisioned a class and lost it to infra.
    failed_providers: set[str] = set()
    tried_classes: set[tuple[str, str]] = set()
    oom_vram_floor = 0
    for local_attempt in range(retry_budget.max_attempts):
        attempt = attempt_start + local_attempt
        if local_attempt > 0 and last_handle:
            from flash.providers import get_provider
            from flash.providers.base import JobHandle

            teardown_confirmed = True
            teardown_error: Exception | None = None
            try:
                _strict_teardown_handle(JobHandle.from_dict(last_handle))
            except Exception as exc:
                teardown_confirmed = False
                teardown_error = exc
            resource_kind = "endpoint" if last_handle.get("endpoint_id") else "instance"
            resource_id = last_handle.get("endpoint_id") or last_handle.get("instance_id")
            if teardown_confirmed:
                if not _compare_and_clear_remote(spec.run_id, last_handle):
                    raise RuntimeError(
                        f"seed {seed}: previous attempt's persisted remote changed before clear; "
                        "aborting replacement to avoid double-provisioning"
                    )
                print(
                    f"retry {attempt}: terminated {last_handle.get('provider')} {resource_kind} "
                    f"{resource_id} (escaping sick host)",
                    file=log,
                    flush=True,
                )
                last_handle.clear()
            else:
                with contextlib.suppress(Exception):
                    get_provider(last_handle["provider"]).gc(spec)
                _gc_seen_endpoints()
                print(
                    f"retry {attempt}: {last_handle.get('provider')} {resource_kind} {resource_id} "
                    f"teardown unconfirmed ({type(teardown_error).__name__}); "
                    "keeping the handle so the "
                    "possibly-billing resource stays reachable for cleanup",
                    file=log,
                    flush=True,
                )
                raise RuntimeError(
                    f"seed {seed}: previous attempt's {last_handle.get('provider')} {resource_kind} "
                    f"{resource_id} teardown could not be confirmed; failing to avoid "
                    "double-provisioning a second worker over a possibly-live resource"
                )
        try:
            attempt_spec = _spec_with_remaining_wall(spec, require_provider_minimum=True)
        except RuntimeError:
            _gc_seen_endpoints()
            raise
        if spec.algorithm == "opd":
            expected_next_attempt, opd_resume_revision = _verified_opd_retry_state(spec.run_id)
        else:
            expected_next_attempt, opd_resume_revision = None, None
        attempt = _reserve_attempt(
            spec.run_id,
            minimum_attempt=attempt_start if local_attempt == 0 else 0,
            expected_next_attempt=expected_next_attempt,
        )
        current_attempt["value"] = attempt
        attempt_runtime_secrets = dict(runtime_secrets or {})
        attempt_runtime_secrets.pop(OPD_RESUME_REVISION_ENV, None)
        if opd_resume_revision is not None:
            attempt_runtime_secrets[OPD_RESUME_REVISION_ENV] = opd_resume_revision
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
                attempt_spec.model,
                attempt_spec.algorithm,
                train=attempt_spec.train,
                thinking=attempt_spec.thinking,
                # The run's requested disk, so the Vast capacity check searches at the SAME effective
                # floor submit provisions with — else a high-disk run is advertised Vast capacity that
                # only exists at 60 GB and then can't rent.
                disk_gb=float(getattr(attempt_spec.gpu, "disk_gb", 0.0) or 0.0),
                # the remaining run-global wall cap, so retries cannot reset the duration budget.
                max_wall_seconds=float(getattr(attempt_spec.gpu, "max_wall_seconds", 0.0) or 0.0),
                provider=getattr(attempt_spec.gpu, "provider", ""),
                exact_type=getattr(attempt_spec.gpu, "exact_type", ""),
                model_revision=attempt_spec.model_revision,
            )
        except Exception as exc:
            from flash.providers.base import UnsupportedGpuError

            if isinstance(exc, UnsupportedGpuError):
                raise  # config-shaped: no GPU anywhere can run this job
            res = PollResult(
                False,
                failure="poll_error",
                detail=f"allocation failed ({type(exc).__name__})",
            )
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
                retry_budget.cache_used < retry_budget.cache_fallbacks
                and started_with_shared_cache
                and not drop_weight_cache
                and chosen is not None
                and getattr(get_provider(chosen.provider), "supports_weight_cache", False)
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
            effective_spec = _spec_with_gpu(spec, chosen.gpu)
            if drop_weight_cache:
                effective_spec = _drop_weight_cache(effective_spec)
            try:
                run_spec = _spec_with_remaining_wall(
                    effective_spec,
                    require_provider_minimum=True,
                )
            except RuntimeError:
                _gc_seen_endpoints()
                raise
            current_gpu["name"] = chosen.gpu
            current_gpu["provider"] = chosen.provider
            current_attempt["value"] = attempt
            retry_delay = 0
            submission_lock = _deploy_lock(spec.run_id)
            submission_lock.acquire()
            try:
                latest = get_status(spec.run_id)
                if latest.state in TERMINAL_STATES:
                    raise _cancel()
                if latest.remote:
                    raise _RunCancelled(
                        f"run {spec.run_id} already has a durable provider handle; not resubmitting"
                    )
                if not _persist_effective_worker_spec(effective_spec):
                    raise _cancel()
                if get_status(spec.run_id).state in TERMINAL_STATES:
                    raise _cancel()
                provider = get_provider(chosen.provider)
                try:
                    submit_kwargs = {
                        "log": log,
                        "on_handle": on_handle,
                        "attempt": attempt,
                        "on_last_gpu": on_last_gpu,
                        "code_prefix": code_prefix,
                        "_deadline_at": _load_run_deadline_at(spec.run_id),
                    }
                    if attempt_runtime_secrets:
                        submit_kwargs["runtime_secrets"] = attempt_runtime_secrets
                    res = provider.submit_run(run_spec, seed, **submit_kwargs)
                except _TerminalHandleRace:
                    raise
                except Exception as exc:
                    from flash.providers.base import UnreconciledCreateError

                    if isinstance(exc, UnreconciledCreateError):
                        res = PollResult(
                            False,
                            failure="job_failed",
                            detail=(
                                f"provider create could not be reconciled ({type(exc).__name__})"
                            ),
                        )
                    else:
                        res = PollResult(
                            False,
                            failure="poll_error",
                            detail=f"provider submit failed ({type(exc).__name__})",
                        )
                        if local_attempt < infra_budget:
                            remaining = _load_run_deadline_at(spec.run_id) - time.time()
                            if remaining > 0:
                                retry_delay = min(10 * (local_attempt + 1), remaining)
            finally:
                lock = submission_lock
                submission_lock = None
                if lock is not None:
                    lock.release()
            if retry_delay:
                time.sleep(retry_delay)  # let the transient clear
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
            and getattr(get_provider(chosen.provider), "supports_weight_cache", False)
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
    from flash.runner import (
        _load_run_deadline_at,
        _run_training,
        _RunCancelled,
        _update,
        flash_code_prefix,
        get_status,
    )

    try:
        code_prefix = flash_code_prefix()
        upload_code(
            spec.train.hf_repo,
            code_prefix=code_prefix,
            **deadline_kwargs(upload_code, _load_run_deadline_at(spec.run_id)),
        )
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
            _update(spec.run_id, "failed", error=f"{type(exc).__name__}: run failed")
        raise


def _run_training(
    spec: JobSpec,
    log,
    *,
    prior_cost: float,
    runtime_secrets: dict[str, str] | None = None,
    code_prefix: str | None = None,
    attempt_start: int = 0,
) -> None:
    """Train the run's single adapter under supervision; finalize the run.

    Shared by a fresh submit and post-restart recovery (the worker resumes from its last HF
    checkpoint on a fresh allocation). ``prior_cost`` carries spend already booked before a
    recovery so the total isn't under-reported. ``attempt_start`` preserves globally monotonic
    worker identities while each invocation keeps its own bounded retry budget."""
    from flash.runner import (
        TERMINAL_STATES,
        _persist_metrics,
        _RunCancelled,
        _status_estimated_charge,
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
    metrics = _submit_seed_supervised(
        spec,
        spec.seed,
        log,
        runtime_secrets=runtime_secrets,
        code_prefix=code_prefix,
        attempt_start=attempt_start,
    )
    # measured wall x $/hr is recorded in metrics.json for analytics, but is NOT what we charge.
    measured_cost = prior_cost + _persist_metrics(spec, metrics)
    # The customer is charged the submit-time QUOTE, not measured wall. Legacy runs without a
    # persisted quote are re-priced from the spec, falling back only for old/unpriceable records.
    charge_usd = _status_estimated_charge(get_status(spec.run_id), spec, fallback=measured_cost)
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
        print(
            f"[ckpt] register warn ({spec.run_id}): {type(exc).__name__}",
            file=log,
            flush=True,
        )


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
    from flash.runner import (
        _drain_cleanup_remotes,
        _remote_resource_identity,
        effective_spec_from_status,
        get_status,
    )

    attempted_cleanup = set()
    with contextlib.suppress(Exception):
        attempted_cleanup = _drain_cleanup_remotes(spec.run_id)
    status = None
    with contextlib.suppress(Exception):
        status = get_status(spec.run_id)
    if status is not None:
        with contextlib.suppress(Exception):
            spec = effective_spec_from_status(status)
    if (
        status is not None
        and status.remote
        and _remote_resource_identity(status.remote) not in attempted_cleanup
    ):
        with contextlib.suppress(Exception):
            _strict_teardown_handle(status.remote)
    try:
        # RunPod gc reaps rN-suffixed endpoints the persisted handle can't name.
        from flash.providers import get_provider

        get_provider("runpod").gc(spec)
    except Exception:
        pass
    from flash.providers import INSTANCE_PROVIDERS, available_providers, get_provider

    _avail = available_providers()
    for _prov in INSTANCE_PROVIDERS:
        if _prov in _avail:
            with contextlib.suppress(Exception):
                get_provider(_prov).gc(spec)
