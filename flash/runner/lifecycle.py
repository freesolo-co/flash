"""Run-execution machinery: the submit -> seed-loop -> per-seed supervised job -> GC flow.

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
        provider (Hyperstack / Lambda) on retry instead of hopping to its next-cheapest class —
        which, when the whole provider is busy, is just as likely to time out (issue: A6000 queue
        timeout retried onto another RunPod class while Hyperstack A6000 sat available); and
      * a provider handing out a broken GPU (a Hyperstack VM whose CUDA never comes up ->
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


def _submit_seed_supervised(
    spec: JobSpec,
    seed: int,
    log,
    runtime_secrets: dict[str, str] | None = None,
) -> dict:
    """Run one seed with the job submit/poll path + bounded auto-retry.

    Each attempt first ALLOCATES the GPU: the cheapest fitting class across every active provider
    (RunPod's validated pool + any Lambda/Hyperstack class with live capacity), price-ranked. There
    is no GPU pin — the cheapest fitting class wins the first attempt.

    Retries (fresh job on a fresh host; worker resumes from the latest HF checkpoint) when the
    failure looks infra-shaped: a stall (heartbeat frozen), no capacity, a client polling breakdown,
    or a platform TIMED_OUT/preemption/worker-loss. Each infra retry ESCAPES the provider that just
    failed cross-provider before walking classes within it (see ``_select_candidate``), so a
    congested provider (RunPod queue timeout) or one handing out a broken GPU (a Hyperstack VM whose
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
    last_detail = None
    # Sticky: once a no-capacity failure shows the weight-cache datacenter set is starved, drop the
    # cache (volume) for every remaining attempt so they run on the unrestricted all-DC pool.
    drop_weight_cache = False
    # Cross-provider retry memory. ``failed_providers`` are the providers that consumed an
    # infra-shaped attempt; ``tried_classes`` the exact (provider, gpu) pairs already attempted.
    # Both grow only when an attempt that ACTUALLY provisioned a class lost it to an infra failure
    # (see the retry tail) — a failed allocation never tried a card, so it can't poison the next
    # pick. ``_select_candidate`` reads them to escape a sick/congested provider cross-provider on
    # retry before walking classes within it.
    failed_providers: set[str] = set()
    tried_classes: set[tuple[str, str]] = set()
    for attempt in range(max_retries + 1):
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
            elif last_handle.get("provider") in ("lambda", "hyperstack"):
                # An instance-based provider bills until terminated: tear the previous attempt's
                # instance down so the retry lands on a fresh host (and we stop paying for the sick
                # one). Dispatched generically through the handle's provider (destroy() knows the
                # provider's own id field — instance_id for Lambda, vm_id for Hyperstack).
                with contextlib.suppress(Exception):
                    from flash.providers import get_provider
                    from flash.providers.base import JobHandle

                    _prov = last_handle["provider"]
                    get_provider(_prov).destroy(JobHandle.from_dict(last_handle))
                    _iid = last_handle.get("instance_id") or last_handle.get("vm_id")
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
        # A cancel can land after _run_seed_loop's pre-submit check but while
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
            # Pick this attempt's (provider, class) from the cross-provider ranked list: the first
            # attempt takes the cheapest; each retry that provisioned a class and lost it to an infra
            # failure ESCAPES that provider before walking classes within it (see _select_candidate),
            # so a congested/sick provider can't burn the whole budget.
            chosen = _select_candidate(alloc.candidates, failed_providers, tried_classes)
            # ``on_last_gpu`` == NO further GPU attempt will be made after this one — either the
            # candidate list is exhausted (``len(untried) <= 1``) OR the retry budget is exhausted
            # (``attempt >= max_retries``, including the single-attempt ``max_retries == 0`` case).
            # Any remaining alternates are only ever reached on a RETRY, so on the final iteration
            # there is no next-best GPU to fall back to regardless of how many candidates remain.
            # Tell the provider so its no-capacity backstops wait longer before giving up rather than
            # failing fast into a retry that will never happen. A pinned/single-candidate run is
            # "last" from attempt 0, which is what we want.
            untried = [c for c in alloc.candidates if (c.provider, c.gpu) not in tried_classes]
            on_last_gpu = len(untried) <= 1 or attempt >= max_retries
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
                if attempt < max_retries:
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
        # A cancel deletes the endpoint, which the poller sees as an
        # infra-shaped failure; retrying would resurrect the run and keep
        # billing. The user's cancel wins over the retry budget.
        try:
            if get_status(spec.run_id).state == "cancelled":
                raise _RunCancelled(f"run {spec.run_id} was cancelled")
        except FileNotFoundError:
            # Status file not yet written (early race): treat as not-cancelled and proceed.
            pass
        print(
            f"seed={seed} attempt={attempt} failed ({res.failure}); "
            f"{'retrying (resume from last checkpoint)' if infra_shaped and attempt < max_retries else 'not retrying'}"
            f"\n--- failure detail ---\n{(res.detail or '')[:2000]}\n---",
            file=log,
            flush=True,
        )
        if not infra_shaped or attempt >= max_retries:
            break
        # Best-effort cache: if a VOLUME-BACKED attempt failed in a way the cache could have caused
        # — no_capacity (the cache restricts the endpoint to its DC set) or a deploy/submit
        # poll_error (e.g. the SDK failing to create/attach a volume) — drop the cache so the run
        # degrades to a cold, unrestricted cross-region attempt instead of looping on the same
        # volume-backed spec (the IN_QUEUE-forever / persistent-volume-failure block). Sticky: once
        # dropped it stays dropped. A non-volume flake (stall/preempt) keeps the cache so the
        # warm-weights benefit survives ordinary retries.
        # Gate to RunPod: the cache-drop rationale (the attached volume pins the endpoint to its DC
        # set, so a no_capacity / volume poll_error wants a cache-less cross-region retry) is
        # RunPod-specific. Instance providers (Lambda/Hyperstack) already fall back to a cold run
        # per-region INSIDE the launch walk, so their no_capacity isn't cache-caused — dropping the
        # cache there would needlessly forfeit the warm-weights benefit on retry.
        # Only the SHARED platform cache triggers the cache-drop retry; a non-shared per-org/custom
        # volume is the intended escape-hatch isolation (runner._assign_weight_cache_volume) and must
        # NOT be stripped on no_capacity — gate on the exact shared name, not any truthy volume.
        from flash.runner import WEIGHT_CACHE_VOLUME_NAME

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
        if first_cache_drop:
            drop_weight_cache = True
            # Do NOT advance the GPU walk on this transition: the next attempt should retry the SAME
            # cheapest GPU without the volume on the wider all-DC pool first — the miss may have been
            # the cache's datacenter set, not the GPU class globally. Only walk if THAT also fails.
        elif chosen is not None:
            # Record what THIS attempt burned so the next pick escapes it cross-provider — only when
            # an attempt actually provisioned a class and lost it infra-shaped. An allocation/pricing
            # failure (chosen is None) never tried a card, so it must not poison the next pick.
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
    from flash.runner import _run_seed_loop, _RunCancelled, _update, get_status

    try:
        # Ship the flash package to the run's HF repo (the per-run [train] hf_repo) so the GPU
        # worker — which fetches code/** from that same repo — can run it.
        upload_code(spec.train.hf_repo)
        with open(log_path, "a") as log:
            _run_seed_loop(
                spec,
                log,
                start_index=0,
                prior_cost=0.0,
                runtime_secrets=runtime_secrets,
            )
    except _RunCancelled:
        return  # cancel_run already set the terminal state
    except Exception as exc:
        if get_status(spec.run_id).state != "cancelled":
            _update(spec.run_id, "failed", error=str(exc))
        raise


def _run_seed_loop(
    spec: JobSpec,
    log,
    *,
    start_index: int,
    prior_cost: float,
    runtime_secrets: dict[str, str] | None = None,
) -> None:
    """Run spec.train.seeds[start_index:] under supervision; finalize the run.

    Shared by a fresh submit (start_index=0) and post-restart recovery, which
    resumes the remaining seeds after the in-flight one completes."""
    from flash.runner import (
        TERMINAL_STATES,
        _persist_metrics,
        _RunCancelled,
        _submit_seed_supervised,
        _update,
        artifacts_dir,
        get_status,
    )

    total_cost = prior_cost
    seeds = spec.train.seeds
    for i in range(start_index, len(seeds)):
        seed = seeds[i]
        # Defense in depth against the recovery TOCTOU (see attach_run): a run can be flipped
        # into ANY terminal state — not just `cancelled` — by a concurrent thread/process
        # (e.g. another recovery marking it failed/done) between the resume decision and here.
        # Bail before _update + _submit_seed_supervised so we never submit PAID GPU work for an
        # already-terminal run. (The `running` _update below would be CAS-rejected anyway, but
        # the supervised submit would still have spent.) _RunCancelled is the loop's terminal
        # signal; its callers already swallow it / leave the existing terminal state intact.
        if get_status(spec.run_id).state in TERMINAL_STATES:
            raise _RunCancelled(f"run {spec.run_id} is already terminal; not submitting seed")
        _update(spec.run_id, "running")
        print(
            f"starting seed={seed} phase={spec.phase} model={spec.model} gpu={spec.gpu.type}",
            file=log,
            flush=True,
        )
        metrics = _submit_seed_supervised(spec, seed, log, runtime_secrets=runtime_secrets)
        total_cost += _persist_metrics(spec, seed, metrics)
        # A cancel can land while this thread writes metrics — after the supervised
        # late-cancel check. Re-read before the post-seed status writes so a late
        # worker success doesn't resurrect a user-cancelled run via this "running"
        # update (or the final "done" below).
        with contextlib.suppress(FileNotFoundError):
            if get_status(spec.run_id).state == "cancelled":
                raise _RunCancelled(f"run {spec.run_id} was cancelled")
        # If more seeds follow, this seed's endpoint/instance is already torn down, so
        # clear the now-stale remote handle: a restart in the gap before the next
        # seed's on_handle must not make recover_runs reattach to a deleted handle and
        # fail the run. Record the next seed index so a restart in that handle-less gap
        # RESUMES the remaining seeds (recover_runs) instead of discarding the completed
        # ones. The last seed keeps its handle for post-run observability (the run is
        # about to go terminal, which recover_runs never reattaches).
        more_seeds = (i + 1) < len(seeds)
        _update(
            spec.run_id,
            "running",
            cost_usd=total_cost,
            **({"remote": None, "resume_seed_index": i + 1} if more_seeds else {}),
        )
        print(
            f"seed={seed} done: train_wall={metrics.get('wall_seconds')} cost_usd={total_cost:.4f}",
            file=log,
            flush=True,
        )
    # Final guard: a cancel landing after the last seed's check must not be overwritten
    # by the terminal "done".
    with contextlib.suppress(FileNotFoundError):
        if get_status(spec.run_id).state == "cancelled":
            raise _RunCancelled(f"run {spec.run_id} was cancelled")
    _update(
        spec.run_id,
        "done",
        cost_usd=total_cost,
        artifacts_dir=artifacts_dir(spec),
        resume_seed_index=None,
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
    from flash.runner import _update, get_status
    from flash.server.auth import INTERNAL_KEY_ENV
    from flash.server.billing import BillingError, charge_completed_run

    status = get_status(spec.run_id)
    if not status.billing_context or status.billing_state == "charged":
        return

    internal_key = os.environ.get(INTERNAL_KEY_ENV, "").strip()
    if not internal_key:
        detail = f"{INTERNAL_KEY_ENV} is not configured; completed run was not billed"
        _update(
            spec.run_id,
            get_status(spec.run_id).state,
            billing_state="failed",
            billing_error=detail,
        )
        print(f"billing failed: {detail}", file=log, flush=True)
        return

    _update(
        spec.run_id,
        get_status(spec.run_id).state,
        billing_state="charging",
        billing_error=None,
    )
    status = get_status(spec.run_id)
    try:
        charge = charge_completed_run(internal_key=internal_key, status=status)
    except BillingError as exc:
        _update(
            spec.run_id,
            get_status(spec.run_id).state,
            billing_state="failed",
            billing_error=exc.detail,
        )
        print(f"billing failed: {exc.detail}", file=log, flush=True)
        return

    _update(
        spec.run_id,
        get_status(spec.run_id).state,
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
    # Instance-based providers (Lambda, Hyperstack) bill until terminated: the runner's per-attempt
    # `finally` already tears them down, but a crashed supervisor thread can leave one behind. Reap
    # any instance still named for this run via each configured provider's gc (best-effort).
    from flash.providers import available_providers, get_provider

    _avail = available_providers()
    for _prov in ("lambda", "hyperstack"):
        if _prov in _avail:
            with contextlib.suppress(Exception):
                get_provider(_prov).gc(spec)
