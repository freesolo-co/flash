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


def _run_job(spec: JobSpec) -> None:
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
        _run_job_inner(spec, log_path, upload_code)
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


def _submit_seed_supervised(
    spec: JobSpec,
    seed: int,
    log,
    bad_gpu_classes: set[str] | None = None,
    mig_seen_once: set[str] | None = None,
    *,
    initial_starved: frozenset = frozenset(),
) -> dict:
    """Run one seed with the job submit/poll path + bounded auto-retry.

    ``bad_gpu_classes`` is the RUN-level set of GPU classes excluded after one handed back a MIG
    slice / unusable GPU TWICE (exclude_class). ``mig_seen_once`` tracks classes that MIG'd ONCE so
    the SECOND occurrence is what excludes them. Both are passed in by the seed loop and SHARED
    across seeds, so a class found MIG-prone on one seed stays excluded for the rest of the run.

    ``initial_starved`` seeds the run-level CAPACITY-STARVED set with ``(provider, gpu)`` tuples the
    caller already knows are starved — used by the recovery path (``attach_run``) when a reattach
    finds the original (provider, class) stuck IN_QUEUE on a ``no_capacity`` failure, so the first
    re-allocation here can't re-pick the just-starved class and the allocator escalates immediately.
    Unlike ``bad_gpu_classes`` (bare strings, market-wide MIG bans) these are provider-SCOPED: a
    RunPod IN_QUEUE starvation says nothing about the same class on Vast.

    Each attempt first ALLOCATES the GPU: the cheapest LIVE-VALIDATED class across providers
    (RunPod live pricing + Vast verified-datacenter offers) that fits the model — re-resolved
    fresh per attempt because offers are a live market. There is no GPU pin and no provider pin —
    the cheapest fitting class in the validated pool always wins.

    Retries (fresh job on a fresh host; worker resumes from the latest HF
    checkpoint) when the failure looks infra-shaped: a stall (heartbeat frozen), a
    client polling breakdown, or a platform TIMED_OUT/worker-loss. Sick Vast machines
    are blacklisted for the run; failover naturally crosses providers.
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
            remote={**handle, "seed": int(seed), "allocated_gpu": current_gpu.get("name")},
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
    bad_machines: set[int] = set()
    # GPU CLASSES to walk OFF entirely after the worker flagged one as a class-level fault
    # (exclude_class — RunPod fulfilled a full-GPU request with a MIG slice, e.g. an "RTX A5000"
    # that was really a Blackwell MIG partition). Stepping the price walk by one index isn't enough
    # — the next-cheapest candidate can be the SAME class on another provider — so the whole class
    # is excluded from re-allocation, walking onto a different validated class (consumer cards like
    # RTX 4090/5090/3090 can't be MIG-sliced). SHARED across the run's seeds when the loop passes it
    # in, so a MIG-prone class stays excluded for every subsequent seed.
    if bad_gpu_classes is None:
        bad_gpu_classes = set()
    # Classes that returned a MIG slice ONCE: the second occurrence is what moves them into
    # bad_gpu_classes (retry the same class once before excluding — a one-off MIG isn't systemic).
    if mig_seen_once is None:
        mig_seen_once = set()
    # PROVIDER-SCOPED capacity-starved classes: ``(provider, gpu)`` tuples a submit found stuck
    # IN_QUEUE with no worker (``no_capacity`` — RunPod-only; Vast has no queue). Excluded from
    # re-allocation so the allocator ESCALATES to the next-fitting class (a pin is a PREFERENCE: once
    # the starved pin is dropped the allocator walks up to a larger validated class). Scoped, not
    # market-wide: the SAME class on another provider may have free capacity. Seeded from the
    # caller's ``initial_starved`` (recovery carries the reattached class's starvation in).
    starved_classes: set = set(initial_starved)
    # Operator escape hatch: seed the exclusion set from a control-plane env so a KNOWN-starved
    # (provider, class) — e.g. RunPod that silently accepts the job into an all-zero IN_QUEUE with
    # no worker and would otherwise burn the full ~50-min setup grace before escalating — is skipped
    # from the FIRST allocation. ``FLASH_EXCLUDE_CLASSES`` is comma-separated; an entry with a
    # ``provider:gpu`` form is provider-SCOPED (a (provider, gpu) tuple, same form the no_capacity
    # path uses), a bare ``gpu`` excludes the class market-wide. Inert on the worker (allocation is
    # control-plane-side only).
    for _raw in (os.environ.get("FLASH_EXCLUDE_CLASSES") or "").split(","):
        _raw = _raw.strip()
        if not _raw:
            continue
        if ":" in _raw:
            _prov, _gpu = _raw.split(":", 1)
            starved_classes.add((_prov.strip().lower(), _gpu.strip()))
        else:
            bad_gpu_classes.add(_raw)
    # Index into the ranked candidate list. It advances only after an attempt that
    # actually provisioned a class lost it to an infra failure (see the retry tail), so a
    # failed allocation — which never tried a card — can't skip past the cheapest class.
    gpu_walk_offset = 0
    for attempt in range(max_retries + 1):
        if attempt > 0 and last_handle:
            # A stalled/timed-out attempt often means the worker is pinned to a
            # throttled/sick host; tear it down so the fresh deploy lands elsewhere.
            # Dispatched generically via the handle's provider.
            if last_handle.get("provider") == "vast":
                with contextlib.suppress(Exception):
                    from flash.providers import get_provider
                    from flash.providers.base import JobHandle

                    get_provider("vast").destroy(JobHandle.from_dict(last_handle))
                if last_handle.get("machine_id"):
                    bad_machines.add(int(last_handle["machine_id"]))
                print(
                    f"retry {attempt}: destroyed vast instance "
                    f"{last_handle.get('instance_id')} (machine "
                    f"{last_handle.get('machine_id')} blacklisted for this run)",
                    file=log,
                    flush=True,
                )
            elif last_handle.get("endpoint_id"):
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
                disk_gb=spec.gpu.disk_gb,
                exclude_machine_ids=frozenset(bad_machines),
                # Walk OFF excluded classes: a MIG / unusable-GPU (RETRIABLE_INFRA_GPU) class is
                # banned market-wide (bare string), and a capacity-starved (provider, class) is
                # dropped provider-scoped (tuple) — the allocator then re-allocates to a DIFFERENT
                # validated class (or ESCALATES a starved pin) instead of re-picking and failing
                # identically.
                exclude_gpu_classes=frozenset(bad_gpu_classes) | frozenset(starved_classes),
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
            # allocate() above ran a live-market price walk; re-check cancellation
            # right before provisioning so a cancel during allocation doesn't still
            # launch a paid worker.
            with contextlib.suppress(FileNotFoundError):
                if get_status(spec.run_id).state == "cancelled":
                    raise _RunCancelled(f"run {spec.run_id} was cancelled")
            # Walk down the ranked candidates by the walk offset (clamped to the last): the
            # first attempt takes the cheapest; each retry that provisioned a class and lost
            # it to an infra failure steps to the next-cheapest, so a capacity-starved class
            # can't burn the whole budget. A concrete pin yields a single candidate, so the
            # clamp keeps a pinned run on its class.
            chosen = alloc.candidates[min(gpu_walk_offset, len(alloc.candidates) - 1)]
            print(allocation_summary(alloc), file=log, flush=True)
            if chosen.gpu != alloc.gpu:
                print(
                    f"retry {attempt}: walking past the cheapest class to {chosen.gpu} "
                    f"@ ${chosen.hourly_usd:.2f}/hr",
                    file=log,
                    flush=True,
                )
            run_spec = _spec_with_gpu(spec, chosen.gpu)
            current_gpu["name"] = chosen.gpu
            provider = get_provider(chosen.provider)
            # Vast needs the live-market offer book for the chosen class first, then the
            # other allocator-approved classes by price; RunPod ignores ``offers``.
            offers = None
            if chosen.provider == "vast":
                ok_classes = {c.gpu for c in alloc.candidates if c.provider == "vast"}
                offers = sorted(
                    (o for o in alloc.provider_offers if o.gpu in ok_classes),
                    key=lambda o: (o.gpu != chosen.gpu, o.dph_total),
                )
            try:
                res = provider.submit_run(
                    run_spec,
                    seed,
                    log=log,
                    on_handle=on_handle,
                    attempt=attempt,
                    offers=offers,
                    # The run's machine blacklist must reach the provider so an in-provider
                    # offer REFRESH (Vast) keeps stalled/sick machines excluded.
                    exclude_machine_ids=frozenset(bad_machines),
                )
            except Exception as exc:
                # Deploy/submit themselves can fail transiently (observed: RunPod
                # GraphQL "Something went wrong" x3 during a retry deploy; a vast offer
                # pool emptying between search and rent). That must consume a retry, not
                # kill the run — the budget exists precisely for flakes.
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
        # ``no_capacity`` (RunPod stuck IN_QUEUE, no worker assigned) is infra-shaped too: even a
        # CONCRETE pin (single-candidate list) must retry so the EXCLUSION below lets the allocator
        # escalate the starved pin to a larger fitting class — the old ncand>1 gate is gone.
        infra_shaped = res.failure in ("stalled", "poll_error", "job_preempted", "no_capacity")
        # A MIG / unusable-GPU failure (worker-flagged exclude_class) MIGHT be a property of the GPU
        # CLASS or just an unlucky host. Don't over-react on the FIRST one: a retry already lands on
        # a fresh host (usually a real full GPU), so retry the SAME class once; only when a class
        # hands back a MIG slice a SECOND time do we treat it as class-systemic and exclude it for
        # the rest of the run (shared across seeds) — re-allocating OFF it onto a different validated
        # class (consumer cards can't be MIG-sliced). Avoids a needless price jump on a one-off MIG.
        mig_failure = res.exclude_class
        if mig_failure and chosen is not None:
            if chosen.gpu in mig_seen_once:
                bad_gpu_classes.add(chosen.gpu)
                print(
                    f"retry: excluding GPU class {chosen.gpu!r} for the rest of this run "
                    "(handed back a MIG slice / unusable GPU TWICE); re-allocating to a different class",
                    file=log,
                    flush=True,
                )
            else:
                mig_seen_once.add(chosen.gpu)
                print(
                    f"retry: GPU class {chosen.gpu!r} returned a MIG slice / unusable GPU; "
                    "retrying the SAME class once on a fresh host before excluding it",
                    file=log,
                    flush=True,
                )
        # A capacity-starved class (no_capacity: stuck IN_QUEUE, no worker) is excluded
        # PROVIDER-SCOPED so re-allocation escalates off the starved (provider, class) — to a
        # different validated class, or up from a starved pin to a larger fitting one — while the
        # SAME class on another provider (which may have capacity) stays selectable. Unlike a MIG
        # ban this fires on the FIRST occurrence: an IN_QUEUE starvation won't clear on a retry of
        # the same class, so there's no "retry once" grace.
        starved_failure = res.failure == "no_capacity"
        if starved_failure and chosen is not None:
            starved_classes.add((chosen.provider, chosen.gpu))
            print(
                f"retry: excluding capacity-starved class {chosen.gpu!r} on {chosen.provider} "
                "(stuck IN_QUEUE, no worker); re-allocating to the next-best / larger class",
                file=log,
                flush=True,
            )
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
        # Step to the next-cheapest class only when THIS attempt actually provisioned one
        # and it failed infra-shaped. An allocation/pricing failure (chosen is None) never
        # tried a card, so the next attempt must retry from the cheapest, not walk past it.
        # A MIG or no_capacity failure already walks the run forward by EXCLUDING the class (the next
        # allocation drops it, so candidate[0] is the next class) — advancing the offset too would
        # over-walk and skip the new cheapest, so only the throttle/stall walk bumps the offset.
        if chosen is not None and not mig_failure and not starved_failure:
            gpu_walk_offset += 1
    # Retry budget exhausted: GC every endpoint this seed registered (the final
    # attempt's is in status.remote for _gc_run_endpoints, but intermediate rN ones
    # are only known here).
    _gc_seen_endpoints()
    raise RuntimeError(f"seed {seed} failed after retries: {last_detail}")


def _run_job_inner(spec: JobSpec, log_path: str, upload_code) -> None:
    from flash.runner import _run_seed_loop, _RunCancelled, _update, get_status

    try:
        # Ship the flash package to the run's HF repo (the per-run [train] hf_repo) so the GPU
        # worker — which fetches code/** from that same repo — can run it.
        upload_code(spec.train.hf_repo)
        with open(log_path, "a") as log:
            _run_seed_loop(spec, log, start_index=0, prior_cost=0.0)
    except _RunCancelled:
        return  # cancel_run already set the terminal state
    except Exception as exc:
        if get_status(spec.run_id).state != "cancelled":
            _update(spec.run_id, "failed", error=str(exc))
        raise


def _run_seed_loop(spec: JobSpec, log, *, start_index: int, prior_cost: float) -> None:
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
    # Shared across seeds: a GPU class confirmed MIG-prone (MIG'd twice) stays excluded for the
    # rest of the run, and a class that MIG'd once is remembered so its next MIG excludes it —
    # instead of re-discovering both seed by seed.
    bad_gpu_classes: set[str] = set()
    mig_seen_once: set[str] = set()
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
        metrics = _submit_seed_supervised(
            spec, seed, log, bad_gpu_classes=bad_gpu_classes, mig_seen_once=mig_seen_once
        )
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
    # Vast instances bill until destroyed: the runner's per-attempt `finally` already
    # destroys them, but a crashed supervisor thread can leave one behind. Reap any
    # instance still labeled for this run via the provider's gc (best-effort).
    from flash.providers import available_providers, get_provider

    if "vast" in available_providers():
        with contextlib.suppress(Exception):
            get_provider("vast").gc(spec)
