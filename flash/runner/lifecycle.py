"""Run-execution machinery: the submit -> supervised training job -> GC flow.

Store helpers (get_status/_update/_save_status/artifacts_dir/_persist_metrics/RUNS_DIR/...)
and sibling lifecycle functions are pulled in via FUNCTION-LOCAL lazy
``from flash.runner import ...`` imports — never at module level. That avoids a
partially-initialized-package import cycle (``flash.runner.__init__`` imports this module
while still being defined) AND keeps the test monkeypatches reachable: a reader that resolves
``RUNS_DIR`` / ``_gc_run_endpoints`` / ``_run_job`` through the package global picks up
``monkeypatch.setattr(runner, ...)`` instead of a statically-bound copy.
"""

from __future__ import annotations

import contextlib
import os
import time

from flash.spec import JobSpec

# Floor on the GPU-walk budget for INFRA-shaped failures (broken/busy GPU, stall, preemption,
# no-capacity) when the user left retries enabled. A broken/busy rented GPU (NVML-init fail /
# cudaErrorDevicesUnavailable) is pure infra bad-luck and cheap to walk past, so a healthy host
# should be found rather than a streak of bad ones killing the run. Matches the default
# ``max_retries`` (5, see spec.GpuSpec); the floor still lifts an explicitly-LOWERED budget (e.g.
# ``max_retries=1``) so infra bad-luck never kills a run that left retries enabled.
# Genuine training errors are NON-infra and still fail fast (no retry). An explicit ``max_retries==0``
# (single-shot, no retries) is respected — the floor only applies when retries are enabled.
INFRA_RETRY_FLOOR = 5


def _run_job(spec: JobSpec, runtime_secrets: dict[str, str] | None = None) -> None:
    # Lazy import so dry-run / unit tests never construct a Flash endpoint.
    from flash.providers.runpod.train import upload_code
    from flash.runner import (
        RUNS_DIR,
        TERMINAL_STATES,
        _gc_run_endpoints,
        _run_job_inner,
        _update,
        get_status,
    )

    # A cancel can land between the queued status being returned to the client and
    # this background thread starting; don't overwrite a terminal state (cancelled)
    # with provisioning and then launch a paid seed as if the cancel never happened.
    if get_status(spec.run_id).state in TERMINAL_STATES:
        return
    _update(spec.run_id, "provisioning")
    log_path = os.path.join(RUNS_DIR, f"{spec.run_id}.log")
    try:
        _run_job_inner(spec, log_path, upload_code, runtime_secrets=runtime_secrets)
    finally:
        # Endpoint GC: every run leaves its uniquely-named endpoint registered, and the
        # account-wide *max workers quota* (5 by default) counts registered endpoints —
        # after a handful of runs, ALL new submissions fail with "Max workers across all
        # endpoints must not exceed your workers quota". Tear ours down on any terminal
        # state (best-effort; never raises).
        _gc_run_endpoints(spec)


def _spec_with_gpu(spec: JobSpec, gpu_type: str) -> JobSpec:
    """The spec the workers/loggers see for THIS attempt's allocated class."""
    if spec.gpu.type == gpu_type:
        return spec
    d = spec.to_dict()
    d["gpu"] = {**d["gpu"], "type": gpu_type}
    return JobSpec.from_dict(d)


def _drop_weight_cache(spec: JobSpec) -> JobSpec:
    """Spec with the SHARED weight-cache volume removed (run cold + fully cross-region).

    Used after a no-capacity attempt: attaching the cache restricts the endpoint to the cache's
    datacenter set, so if that whole set is momentarily starved the next attempt should fall back to
    the unrestricted all-DC pool. Dropping ``network_volume`` makes weight_cache_endpoint_kwargs
    return ``{}`` (no volume, no datacenter list) and turns off the worker's HF_HOME redirect — i.e.
    exactly today's cold cross-region behavior. Worst case for the cache is one capacity-grace wait,
    never a permanent IN_QUEUE block.

    ONLY the platform-managed SHARED cache (``WEIGHT_CACHE_VOLUME_NAME``) is dropped. A non-shared
    per-org/custom ``network_volume`` is a deliberate escape-hatch isolation (see
    runner._assign_weight_cache_volume) the user opted into — it is PRESERVED across retries rather
    than silently stripped.
    """
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    if getattr(spec.gpu, "network_volume", None) != WEIGHT_CACHE_VOLUME_NAME:
        return spec
    d = spec.to_dict()
    d["gpu"] = {**d["gpu"], "network_volume": None}
    return JobSpec.from_dict(d)


def _select_candidate(candidates, failed_providers: set[str], tried_classes: set[tuple[str, str]]):
    """Pick the next (provider, class) to try from the cross-provider ranked candidate list.

    ``candidates`` is already price-sorted (cheapest first). On the FIRST attempt — nothing failed
    yet — this returns the cheapest overall, unchanged. On an infra-shaped RETRY it ESCAPES the
    failed substrate *cross-provider* before walking classes within it:

      * a congested provider (RunPod queue timeout / no warm workers) is left for a DIFFERENT
        provider (Lambda) on retry instead of hopping to its next-cheapest class —
        which, when the whole provider is busy, is just as likely to time out (issue: A6000 queue
        timeout retried onto another RunPod class while a Lambda A6000 sat available); and
      * a provider handing out a broken GPU (an instance whose CUDA never comes up ->
        ``job_preempted``) is likewise escaped to another provider rather than re-rolling the same
        broken region.

    When every provider has already burned a retry (or only one provider is configured) it falls
    back to the cheapest class NOT yet tried, preserving the within-provider class walk.

    Keyed on (provider, gpu) IDENTITY, never a list index, so it stays correct even though each
    attempt re-allocates and the live-capacity ordering can shift between attempts.
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
) -> dict:
    """Run one seed with the job submit/poll path + bounded auto-retry.

    Each attempt first ALLOCATES the GPU: the cheapest fitting class across every active provider
    (RunPod's validated pool + any Lambda class with live capacity), price-ranked. There
    is no GPU pin — the cheapest fitting class wins the first attempt.

    Retries (fresh job on a fresh host; worker resumes from the latest HF checkpoint) when the
    failure looks infra-shaped: a stall (heartbeat frozen), no capacity, a client polling breakdown,
    or a platform TIMED_OUT/preemption/worker-loss. Each infra retry ESCAPES the provider that just
    failed cross-provider before walking classes within it (see ``_select_candidate``), so a
    congested provider (RunPod queue timeout) or one handing out a broken GPU (an instance whose
    CUDA never inits) is left for a healthy substrate rather than re-rolling the same failure.
    Genuine worker errors (the run's code crashed; traceback persisted to HF) fail
    immediately.
    """
    from flash.providers import get_provider
    from flash.providers.allocator import allocate, allocation_summary
    from flash.providers.base import PollResult
    from flash.runner import TERMINAL_STATES, _RunCancelled, _spec_with_gpu, _update, get_status

    last_handle: dict = {}
    # The friendly GPU class the CURRENT attempt provisioned (set right before each submit),
    # so on_handle persists it into the run handle and a recovery via attach_run costs the
    # class actually used rather than the parse-time provisional spec.gpu.type.
    current_gpu: dict = {}
    # Whether the CURRENT attempt's class is the last gpu-walk candidate (set right before each
    # submit). Persisted into the run handle so a recovery via attach_run polls with the SAME
    # no-capacity stall tuning the original submit used (see jobs.stall_kwargs / RunpodProvider.poll)
    # — otherwise a reattached last-candidate run would be judged on the shorter non-last grace.
    current_on_last_gpu: dict = {"value": False}
    # Every RunPod endpoint id this run registered across attempts. Retries run on
    # rN-suffixed endpoints whose names _gc_run_endpoints cannot reconstruct, and a
    # failed delete during the next attempt's teardown would otherwise lose the id;
    # GC the whole set at exit so no retry endpoint leaks against the worker quota.
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

    max_retries = int(spec.gpu.max_retries)
    # The effective GPU-walk budget for infra-shaped failures: floored to INFRA_RETRY_FLOOR so a streak
    # of broken/busy GPUs is walked past to a healthy host, but only when the user left retries enabled
    # (max_retries==0 stays 0 — a deliberate single-shot run is never forced to retry).
    infra_budget = max(max_retries, INFRA_RETRY_FLOOR) if max_retries else 0
    last_detail = None
    # Sticky: once a no-capacity failure shows the weight-cache datacenter set is starved, drop the
    # cache (volume) for every remaining attempt so they run on the unrestricted all-DC pool.
    drop_weight_cache = False
    # The platform auto-attaches the SHARED weight cache (runner._assign_weight_cache_volume), so its
    # endpoint-pinning DC-set restriction must not cost the USER a GPU-walk retry. Grant ONE extra,
    # cache-less fallback attempt — consumed ONLY by the cache-drop transition below (the stop check
    # gates the bonus on ``first_cache_drop``, never on a plain GPU walk) — so a no_capacity/poll_error
    # the cache's datacenter set could have caused always earns one unrestricted cross-region retry,
    # even at ``max_retries == 0`` (where the auto-cache would otherwise fail a run a cache-less launch
    # could have won). A non-shared per-org/custom volume is the user's own choice and earns no bonus.
    from flash.runner import WEIGHT_CACHE_VOLUME_NAME

    started_with_shared_cache = getattr(spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
    cache_fallback_attempts = 1 if started_with_shared_cache else 0
    # Cross-provider retry memory. ``failed_providers`` are the providers that consumed an
    # infra-shaped attempt; ``tried_classes`` the exact (provider, gpu) pairs already attempted.
    # Both grow only when an attempt that ACTUALLY provisioned a class lost it to an infra failure
    # (see the retry tail) — a failed allocation never tried a card, so it can't poison the next
    # pick. ``_select_candidate`` reads them to escape a sick/congested provider cross-provider on
    # retry before walking classes within it.
    failed_providers: set[str] = set()
    tried_classes: set[tuple[str, str]] = set()
    # Largest GPU VRAM (GB) that OOM'd this run: a CUDA-OOM retry must land on a STRICTLY larger card.
    oom_vram_floor = 0
    # Attempts spent on the cache-drop fallback, EXCLUDED from the GPU-walk budget. The bonus slot
    # ``cache_fallback_attempts`` widens the loop range, but the budget checks below use the raw attempt
    # counter; without this offset the cache-drop attempt would still tick the budget, so a run that
    # spends its bonus on the cache drop could never reach its real ``max_retries`` GPU-walk retries
    # (the fallback would silently steal the only user retry). ``walk_attempt`` = attempt index with the
    # cache-drop attempt(s) removed, so the GPU walk gets its full budget AFTER a cache drop.
    cache_drop_consumed = 0
    # Infra-shaped GPU-walk attempts (no_capacity/job_preempted/stalled/poll_error), EXCLUDED from the
    # OOM-escalation budget (NOT from the infra budget). The OOM budget counts ONLY OOM-shaped attempts:
    # without this offset an early infra retry would tick the shared ``walk_attempt`` and burn a CUDA
    # OOM's allowed larger-GPU escalation (notably max_retries=1: an infra retry on attempt 0 then a real
    # OOM on attempt 1 would read the OOM budget as already spent and refuse the one escalation the user
    # allowed). Mirrors ``cache_drop_consumed`` — both peel non-OOM attempts off the OOM walk.
    infra_walk_consumed = 0
    # The infra-shaped budget (floored ``infra_budget`` retries) and the OOM-escalation budget (raw
    # ``max_retries`` strictly-larger-GPU retries) are SEPARATE allowances, so the physical loop must
    # provide a slot for EACH independently: with only ``infra_budget + 1`` slots a warranted OOM
    # escalation that lands AFTER the infra budget is spent (e.g. five infra failures then a first CUDA
    # OOM) would have no iteration to run on and the run would FAIL instead of escalating. The per-
    # category budget BREAKS below still cap each allowance on its own (infra at the floored budget, OOM
    # at ``max_retries``), so over-sizing the range never lets infra-only failures loop past their
    # budget — it only guarantees the slot a separately-budgeted OOM escalation needs. ``max_retries == 0``
    # (single-shot) adds 0, so a no-retry run still never escalates.
    oom_escalation_slots = max_retries
    for attempt in range(infra_budget + 1 + cache_fallback_attempts + oom_escalation_slots):
        walk_attempt = attempt - cache_drop_consumed
        if attempt > 0 and last_handle:
            # A stalled/timed-out attempt often means the worker is pinned to a
            # throttled/sick host; tear it down so the fresh deploy lands elsewhere.
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
                    # Logging the host-escape note is cosmetic; never let it abort the retry.
                    pass
            elif last_handle.get("provider") == "lambda":
                # An instance-based provider bills until terminated: tear the previous attempt's
                # instance down so the retry lands on a fresh host (and we stop paying for the sick
                # one). Dispatched generically through the handle's provider (destroy() knows the
                # provider's own id field — instance_id for Lambda).
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
            # The previous endpoint is now deleted; clear the persisted handle so a cancel
            # or control-plane restart during the fresh deploy doesn't operate on (or get
            # shielded by) the dead handle. The next on_handle() records the new one.
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
                raise _RunCancelled(f"run {spec.run_id} was cancelled")
        try:
            alloc = allocate(
                spec.model,
                spec.algorithm,
                # Pass the run's train knobs + thinking so the VRAM estimate reflects THIS job's
                # max_length / group_size / batch_size / lora_rank (and the seq escalation) instead
                # of the generic defaults — else a long-context / big-group run is sized at seq=1024
                # and OOMs the card it picks.
                train=spec.train,
                thinking=spec.thinking,
            )
        except Exception as exc:
            from flash.providers.base import UnsupportedGpuError

            if isinstance(exc, UnsupportedGpuError):
                raise  # config-shaped: no GPU anywhere can run this job
            res = PollResult(False, failure="poll_error", detail=f"allocation: {exc}")
        if alloc is not None:
            # Re-check cancellation right before provisioning so a cancel during allocation
            # doesn't still launch a paid worker.
            with contextlib.suppress(FileNotFoundError):
                if get_status(spec.run_id).state == "cancelled":
                    raise _RunCancelled(f"run {spec.run_id} was cancelled")
            # After a CUDA OOM, restrict to cards STRICTLY larger than the one that OOM'd (a same-size
            # retry just OOMs again). Empty => already OOM'd the biggest class -> fail terminally.
            cands = _oom_escalated(alloc.candidates, oom_vram_floor)
            if not cands:
                last_detail = f"oom: exceeded the largest available GPU ({oom_vram_floor} GB)"
                print(f"seed={seed} OOM on the largest GPU class ({oom_vram_floor} GB); not retrying",
                      file=log, flush=True)
                break
            # Pick this attempt's (provider, class) from the cross-provider ranked list: the first
            # attempt takes the cheapest; each retry that provisioned a class and lost it to an infra
            # failure ESCAPES that provider before walking classes within it (see _select_candidate),
            # so a congested/sick provider can't burn the whole budget.
            chosen = _select_candidate(cands, failed_providers, tried_classes)
            # ``on_last_gpu`` == NO further GPU attempt will be made after this one — either the
            # candidate list is exhausted (``len(untried) <= 1``) OR the retry budget is exhausted
            # (``attempt >= max_retries``, including the single-attempt ``max_retries == 0`` case).
            # Any remaining alternates are only ever reached on a RETRY, so on the final iteration
            # there is no next-best GPU to fall back to regardless of how many candidates remain.
            # Tell the provider so its no-capacity backstops wait longer before giving up rather than
            # failing fast into a retry that will never happen. A pinned/single-candidate run is
            # "last" from attempt 0, which is what we want.
            untried = [c for c in cands if (c.provider, c.gpu) not in tried_classes]
            # The cache-drop fallback (cache_fallback_attempts) is a reserved attempt PAST the retry
            # budget, so when it's still available a cache-attached RunPod attempt is not "last" by
            # BUDGET — don't let ``attempt >= max_retries`` mark it last-GPU (long no-capacity grace),
            # so a no_capacity fails fast into that fallback (notably at max_retries == 0). This only
            # gates the BUDGET clause: genuine class exhaustion (``len(untried) <= 1``) still marks
            # last-GPU (the fallback re-uses the same class cache-less — there's no OTHER class to walk
            # to), preserving the walk semantics for non-cache-caused failures (e.g. a stalled walk).
            cache_fallback_available = (
                started_with_shared_cache
                and not drop_weight_cache
                and chosen is not None
                and chosen.provider == "runpod"
            )
            # Once a CUDA OOM has forced this run onto larger cards (``oom_vram_floor > 0``), the active
            # walk is the OOM-escalation budget (raw ``max_retries``), NOT the floored infra budget — so
            # compute "is this the last GPU attempt" against the OOM budget and the OOM walk index
            # (``walk_attempt - infra_walk_consumed`` = OOM escalations so far). Otherwise the last
            # OOM-eligible attempt would read non-final (``walk_attempt < infra_budget``), take the SHORT
            # no-capacity grace, and fail fast into extra infra-shaped retries past the OOM cost cap.
            escalating_oom = oom_vram_floor > 0
            walk_budget = max_retries if escalating_oom else infra_budget
            walk_spent = walk_attempt - infra_walk_consumed if escalating_oom else walk_attempt
            on_last_gpu = len(untried) <= 1 or (
                walk_spent >= walk_budget and not cache_fallback_available
            )
            # Mirror into the closure cell so on_handle persists THIS attempt's value (see
            # current_on_last_gpu) for a recovery to reproduce the same stall tuning.
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
            # After a no-capacity attempt, fall back to a cache-less cross-region run (see
            # drop_weight_cache below): the attached cache pins the endpoint to its DC set, so the
            # fallback must run on the unrestricted pool.
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
                # Deploy/submit themselves can fail transiently (observed: RunPod
                # GraphQL "Something went wrong" x3 during a retry deploy). That must
                # consume a retry, not kill the run — the budget exists precisely for flakes.
                res = PollResult(False, failure="poll_error", detail=f"deploy/submit: {exc}")
                if attempt < infra_budget:
                    time.sleep(10 * (attempt + 1))  # let the transient clear
        if res.ok:
            # A best-effort cancel may fail to stop the worker, which then completes
            # successfully after cancel_run() persisted `cancelled`. Don't let a late
            # worker success resurrect the run into running/done.
            try:
                if get_status(spec.run_id).state == "cancelled":
                    raise _RunCancelled(f"run {spec.run_id} was cancelled")
            except FileNotFoundError:
                # Status file not yet written (early race): treat as not-cancelled, proceed.
                pass
            # Worker is done (DONE sentinel seen); GC every endpoint this seed used,
            # including intermediate rN retries _gc_run_endpoints can't name.
            _gc_seen_endpoints()
            # Record the class actually allocated so _persist_metrics rates the right
            # RunPod card when a policy GPU was re-allocated away from the provisional.
            if chosen is not None and isinstance(res.metrics, dict):
                res.metrics.setdefault("allocated_gpu", chosen.gpu)
            return res.metrics
        last_detail = f"{res.failure}: {res.detail}"
        # Retry only on a structured failure category the provider already classified; a real job
        # failure fails fast. No detail-string parsing. (USER cancels are caught below, not here.)
        infra_shaped = res.failure in ("stalled", "no_capacity", "poll_error", "job_preempted")
        # A CUDA OOM retries too, but on a STRICTLY larger card: record the failed card's VRAM as the
        # escalation floor for the next attempt's candidate filter. It is NOT infra (the host was
        # fine), so it must NOT escape the provider (the record block below skips that for oom).
        oom_shaped = res.failure == "oom"
        if oom_shaped and chosen is not None:
            oom_vram_floor = max(oom_vram_floor, int(chosen.vram_gb))
        retry_shaped = infra_shaped or oom_shaped
        # A cancel deletes the endpoint, which the poller sees as an
        # infra-shaped failure; retrying would resurrect the run and keep
        # billing. The user's cancel wins over the retry budget.
        try:
            if get_status(spec.run_id).state == "cancelled":
                raise _RunCancelled(f"run {spec.run_id} was cancelled")
        except FileNotFoundError:
            # Status file not yet written (early race): treat as not-cancelled and proceed.
            pass
        # Best-effort cache-drop fallback — computed BEFORE the log + budget stop so both reflect it.
        # If a VOLUME-BACKED RunPod attempt failed in a way the cache could have caused — no_capacity
        # (the cache restricts the endpoint to its DC set) or a deploy/submit poll_error (e.g. the SDK
        # failing to create/attach a volume) — drop the cache so the run degrades to a cold, unrestricted
        # cross-region attempt instead of looping on the same volume-backed spec (the IN_QUEUE-forever /
        # persistent-volume-failure block). Sticky: once dropped it stays dropped. A non-volume flake
        # (stall/preempt) keeps the cache so the warm-weights benefit survives ordinary retries.
        # Gate to RunPod: instance providers (Lambda) already fall back to a cold run
        # per-region INSIDE the launch walk, so their no_capacity isn't cache-caused. Only the SHARED
        # platform cache triggers it (gate on the exact name); a non-shared per-org/custom volume is the
        # intended escape-hatch isolation (runner._assign_weight_cache_volume) and must NOT be stripped.
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
        # OOM escalation is COST (each retry is a strictly bigger, pricier GPU), so it respects the
        # user's RAW max_retries — NOT the INFRA_RETRY_FLOOR that lets infra bad-luck retries exceed it
        # (so max_retries=1 grants ONE larger-GPU attempt, not five). Infra keeps the floored budget.
        retry_budget = max_retries if oom_shaped else infra_budget
        # The OOM-escalation budget counts ONLY OOM-shaped attempts: subtract the infra-shaped walk
        # attempts already consumed (``infra_walk_consumed``, mirroring the ``cache_drop_consumed``
        # exclusion in ``walk_attempt``) so an early infra retry can't burn a CUDA OOM's allowed larger-
        # GPU escalation. Infra failures keep counting ALL walk attempts against the floored infra budget.
        budget_spent = walk_attempt - infra_walk_consumed if oom_shaped else walk_attempt
        # "retrying" is true when the budget remains OR a cache-drop fallback will retry this even past
        # it (first_cache_drop) — else the log would say "not retrying" while the loop actually
        # continues with the reserved cache-less fallback attempt.
        will_retry = retry_shaped and (budget_spent < retry_budget or first_cache_drop)
        action = (
            f"retrying on a larger GPU (> {oom_vram_floor} GB)"
            if (will_retry and oom_shaped)
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
        if not retry_shaped:
            break
        # Stop when the GPU-walk retry budget is exhausted — UNLESS a cache-drop fallback is still
        # available. The bonus attempt granted above is reserved for exactly this transition; once the
        # cache is dropped (sticky), ``first_cache_drop`` is False so the budget check applies normally
        # and the loop cannot spin past its one extra cache-less attempt.
        if budget_spent >= retry_budget and not first_cache_drop:
            break
        if first_cache_drop:
            drop_weight_cache = True
            # This attempt was the FREE cache-drop fallback, not a GPU-walk retry — exclude it from the
            # budget so the subsequent ``walk_attempt`` still counts ``max_retries`` real retries.
            cache_drop_consumed += 1
            # Do NOT advance the GPU walk on this transition: the next attempt should retry the SAME
            # cheapest GPU without the volume on the wider all-DC pool first — the miss may have been
            # the cache's datacenter set, not the GPU class globally. Only walk if THAT also fails.
        else:
            # A real GPU-walk retry (not the free cache-drop fallback). An INFRA-shaped one must NOT
            # consume the OOM-escalation budget: track it so ``budget_spent`` peels it off the OOM walk
            # (mirrors ``cache_drop_consumed``). Counted even when ``chosen is None`` (an allocation/
            # deploy poll_error that never provisioned a card) so every infra attempt is excluded from
            # the OOM count, not just provisioned ones.
            if not oom_shaped:
                infra_walk_consumed += 1
            if chosen is not None:
                # Record what THIS attempt burned so the next pick avoids it. An infra failure also
                # escapes the PROVIDER cross-provider; an OOM does NOT (the host was fine — just grow the
                # card, which the oom_vram_floor filter already enforces).
                if not oom_shaped:
                    failed_providers.add(chosen.provider)
                tried_classes.add((chosen.provider, chosen.gpu))
    # Retry budget exhausted: GC every endpoint this seed registered (the final
    # attempt's is in status.remote for _gc_run_endpoints, but intermediate rN ones
    # are only known here).
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
        # Ship the flash package to the run's HF repo (the per-run [train] hf_repo) so the GPU
        # worker — which fetches code/** from that same repo — can run it.
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
    _update(
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
    _charge_completed_run_best_effort(spec, log)
    _register_checkpoints_best_effort(spec, log)


def _register_checkpoints_best_effort(spec: JobSpec, log) -> None:
    """Mirror a finished run's deployable per-step checkpoints to the backend store.

    Best-effort and isolated from billing: the checkpoints live on HF regardless, so a
    persistence miss never changes the run's outcome."""
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
    """Best-effort teardown of every endpoint a run may have registered.

    Retried attempts run on rN-suffixed endpoints whose runpod_flash state is
    isolated per-suffix, so the name-based terminate_endpoint cannot see them;
    the persisted remote handle's endpoint id covers whichever attempt ran
    last via the plain REST API."""
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
            # Best-effort GC; the name-reconstructed RunPod gc below is the backstop.
            pass
    try:
        # RunPod's gc reaps rN-suffixed endpoints the persisted handle can't name.
        from flash.providers import get_provider

        get_provider("runpod").gc(spec)
    except Exception:
        # Best-effort GC; an undeleted endpoint only holds worker quota, never blocks the run.
        pass
    # Instance-based providers (Lambda) bill until terminated: the runner's per-attempt
    # `finally` already tears them down, but a crashed supervisor thread can leave one behind. Reap
    # any instance still named for this run via each configured provider's gc (best-effort).
    from flash.providers import available_providers, get_provider

    _avail = available_providers()
    for _prov in ("lambda",):
        if _prov in _avail:
            with contextlib.suppress(Exception):
                get_provider(_prov).gc(spec)
