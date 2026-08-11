"""Turning a validated job spec into a run: `prepare_job` and `submit_job`.

`prepare_job` is the read-only half -- resolve the model, validate the effective spec, and settle
warm-start provenance -- so the caller can be told what a run would cost and consume before any
state exists. `submit_job` is the half that commits: it writes the first status, then either runs
the job inline or hands it to a background supervisor thread.

Split out of `flash.runner` to keep that module under the file-size limit.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from flash.core.spec import JobSpec
from flash.teacher.retry_contract import OPD_RETRY_CONTRACT_VERSION

if TYPE_CHECKING:
    # annotation-only: both are defined in `flash.runner` above the point where it imports this
    # module, so a runtime import would be circular. the constructors below go through `_runner()`.
    from flash.runner import PreparedJob, RunStatus


def _runner():
    """The runner package, imported lazily because it re-exports this module.

    Most of what these two functions call is patched as an attribute of `flash.runner` by the
    submit tests -- the status accessors, the persistence writes, the model resolver, and both
    job-launch entry points -- so every one of those is resolved through the package here. A direct
    call would bind this module's own copy and the patch would never be seen.
    """
    import flash.runner as runner

    return runner


def prepare_job(
    spec: JobSpec,
    *,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
    owner_key_id: int | None = None,
) -> PreparedJob:
    """Prepare all read-only submission inputs before persistence or allocation."""
    # before _resolve_model_revision, and before every sizing step below: a warm start inherits its
    # source's runner-assigned pin, and `resolve_model`/`_with_model_disk` size against whatever
    # revision the spec carries by then.
    spec = _runner()._inherit_warmstart_revision(
        spec,
        owner_org_id=_runner()._context_org_id(billing_context)
        or _runner()._context_org_id(platform_context),
        owner_key_id=owner_key_id,
    )
    spec = _runner()._resolve_model_revision(spec, required=spec.algorithm == "sft")
    _runner()._require_supported_adapter_continuation(spec)
    if spec.algorithm == "sft":
        spec = _runner()._require_pinned_profile_environment(spec)
        spec = _runner()._require_sft_workload_profile(spec)
    if spec.train.structured_outputs:
        from flash.serve.preflight import preflight_serving_path

        preflight_serving_path(spec)
    else:
        from flash.adapters.lora_rank import (
            ServingPreflightError,
            preflight_train_context_within_serving,
        )

        # mirror preflight_serving_path: surface the specific context error as a
        # ServingPreflightError so create_run re-raises it unchanged instead of the
        # warm-start path masking it with a generic preparation message
        try:
            preflight_train_context_within_serving(spec)
        except ValueError as exc:
            raise ServingPreflightError(str(exc)) from exc
    if spec.gpu.provider or spec.gpu.type:
        from flash.providers import PROVIDER_NAMES, available_providers
        from flash.providers.base import providers_for

        configured = available_providers()
        provider = spec.gpu.provider.strip().lower()
        if provider:
            if provider not in PROVIDER_NAMES:
                raise ValueError(f"unknown gpu.provider {spec.gpu.provider!r}")
            if provider not in configured:
                raise ValueError(f"requested gpu.provider {provider!r} is not configured")
        elif not any(name in configured for name in providers_for(spec.gpu.type)):
            raise ValueError(f"no configured provider can provision gpu.type {spec.gpu.type!r}")
    info = _runner().resolve_model(spec.model, spec.algorithm, model_revision=spec.model_revision)
    if spec.algorithm == "opd" and spec.train.structured_outputs:
        # the generic serving preflight above validates the schema's SHAPE, but the
        # constraint can still be one verl OPD deterministically refuses: a guidance-only
        # json feature (format, multipleOf, uniqueItems) or a vllm MistralTokenizer model.
        # that check lived only on the worker, so the user rented a gpu to receive a
        # permanent validation failure. only the allocation-independent half runs here.
        # the vocab-size comparison needs the ALLOCATED card count -- judging a shardable
        # run on one card would reject a shape the allocator would have placed -- so it
        # stays on the worker, which also keeps that check as defense in depth for any
        # path that reaches training without passing through here.
        from flash.engine.worker.train.opd.validation import preflight_opd_structured_outputs

        preflight_opd_structured_outputs(
            spec.train.structured_outputs,
            model_id=spec.model,
            model_revision=spec.model_revision,
        )
    run_id = spec.run_id if (spec.run_id and spec.run_id != "local") else _runner().new_run_id()
    spec = JobSpec.from_dict({**_runner()._with_model_disk(spec, info), "run_id": run_id})
    spec = _runner()._assign_managed_hf_repo(spec)
    spec = _runner()._assign_weight_cache_volume(spec, info)
    owner_org_id = _runner()._context_org_id(billing_context) or _runner()._context_org_id(
        platform_context
    )
    public_spec, worker_spec, adapter_identity = _runner()._prepare_init_from_adapter(
        spec,
        owner_org_id=owner_org_id,
        owner_key_id=owner_key_id,
        token=os.environ.get("HF_TOKEN"),
    )
    from flash.cost.spec import estimate_for_spec

    # profile jobs route to their bounded-wall charge inside estimate_for_spec; they cannot be priced
    # through the training estimator, which requires the profile this job produces.
    estimated_cost_usd = float(estimate_for_spec(worker_spec).total_usd)
    return _runner().PreparedJob(
        public_spec=public_spec,
        worker_spec=worker_spec,
        estimated_cost_usd=estimated_cost_usd,
        adapter_identity=adapter_identity,
    )


def _reject_managed_volume_removal(snapshot: object, worker_spec: JobSpec) -> None:
    """Fail closed if a re-prepared worker spec drops a non-shared weight-cache volume.

    network_volume is platform-managed and no longer travels in the public spec, so the committed
    volume lives only in the prior preparation snapshot. The SHARED platform cache
    (WEIGHT_CACHE_VOLUME_NAME) may be dropped on a capacity fallback, but a per-org escape-hatch
    volume a run opted into must never be silently removed or swapped.
    """
    if not isinstance(snapshot, dict):
        return
    committed = ((snapshot.get("worker_spec") or {}).get("gpu") or {}).get("network_volume")
    if not committed or committed == _runner().WEIGHT_CACHE_VOLUME_NAME:
        return
    if worker_spec.gpu.network_volume != committed:
        raise ValueError("persisted effective preparation drops a non-shared weight-cache volume")


def _persist_effective_worker_spec(
    worker_spec: JobSpec, *, estimated_cost_usd: float | None = None
) -> bool:
    """Persist the selected worker spec and exact quote before provider provisioning starts."""
    status = _runner().get_status(worker_spec.run_id)
    if status.state in _runner().TERMINAL_STATES:
        return False
    snapshot = status.effective_preparation
    public_spec = JobSpec.from_dict(status.spec)
    if public_spec.train.init_from_adapter:
        if not isinstance(snapshot, dict):
            raise ValueError("persisted effective preparation is malformed")
        _runner().effective_spec_from_status(status)
        adapter_identity = snapshot.get("adapter_identity")
    else:
        adapter_identity = None
    _reject_managed_volume_removal(snapshot, worker_spec)
    _runner()._validate_effective_spec(public_spec, worker_spec)
    # status.spec is never rewritten, so a pre-upgrade run keeps its dropped keys for life and every
    # later read rehashes with them restored. re-persisting (quote refresh, realloc) has to hash the
    # same way or the digest it writes now is one the next integrity check cannot reproduce.
    raw_public = status.spec if isinstance(status.spec, dict) else {}
    legacy_public_keys = {
        k: raw_public[k] for k in _runner()._DROPPED_TOP_LEVEL_KEYS if k in raw_public
    }
    effective_preparation = {
        "worker_spec": worker_spec.to_internal_dict(),
        "workload_profile": worker_spec.workload_profile or None,
        "adapter_identity": adapter_identity,
        "preparation_digest": _runner()._preparation_digest(
            public_spec,
            worker_spec,
            adapter_identity,
            legacy_public_keys=legacy_public_keys,
            legacy_public_alpha=_runner()._prepared_before_public_alpha(raw_public),
        ),
        "backend": _runner().TRAINER_BACKEND,
    }
    fields = {"effective_preparation": effective_preparation}
    if estimated_cost_usd is not None:
        fields["estimated_cost_usd"] = float(estimated_cost_usd)
    return _runner()._update(worker_spec.run_id, status.state, **fields)


def _persist_profile_submission(status: RunStatus, save_kwargs: dict) -> RunStatus | None:
    """Write a profile's submission record, returning a live run to join instead of restarting.

    A profile's run id is derived from the workload rather than the account, so this id is reused
    by design and the record it writes may not be the first under it.
    """
    with _runner()._status_guard(status.run_id):
        raw_existing = (
            _runner()._load_status_json(status.run_id)
            if os.path.exists(_runner().runs_file_path(status.run_id, ".json"))
            else None
        )
        existing = (
            _runner()._runstatus_from_json(raw_existing) if raw_existing is not None else None
        )
        # a live profile under this id is joined, never restarted: a concurrent submitter of the
        # same config lands here and must wait on the running one rather than launch a second
        # billed copy of identical work.
        if existing is not None and existing.state not in _runner()._UNDEPLOYABLE_STATES:
            return existing
        # a spent one is replaced. the caller only reaches this after winning the takeover on that
        # exact spent record, so overwriting it is the relaunch, not a lost update.
        if raw_existing is not None:
            # reused profile ids share artifact prefixes, so attempt ids must remain globally monotonic.
            # restarting at zero can bind the new worker to the spent lifecycle's attempt-scoped error
            # file and fail before using its hardware.
            carried_attempt = _runner()._infer_next_attempt(raw_existing)
            save_kwargs["_next_attempt"] = carried_attempt
            # carrying the counter keeps the ids monotonic, but it also means that
            # until THIS lifecycle reserves one, `next_attempt - 1` still names the
            # SPENT lifecycle's attempt. a prior worker that outlived its record
            # stamps exactly that id, and its heartbeats are genuinely recent, so the
            # provenance check would accept one and arm this run's work budget while
            # it is still queuing for a machine. record the carried counter as this
            # lifecycle's floor: every attempt below it belongs to the run that
            # already ended.
            save_kwargs["_profile_attempt_floor"] = carried_attempt
            # the wall, by contrast, must NOT carry: an arm records that a worker
            # spoke, and that worker was the previous lifecycle's. inheriting it dates
            # this run's budget to a heartbeat predating its own submission -- and
            # since _canonical_run_deadline rebuilds the deadline from that basis, the
            # stored pair stops matching and every read fails the tamper check,
            # wedging this workload's profile id for every submitter.
            save_kwargs["_profile_wall_armed_at"] = None
        _runner()._save_status_unlocked(status, **save_kwargs)
    return None


def submit_job(
    spec: JobSpec,
    dry_run: bool = False,
    background: bool = False,
    runtime_secrets: dict[str, str] | None = None,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
    owner_key_id: int | None = None,
    prepared_job: PreparedJob | None = None,
) -> RunStatus:
    """Submit a prepared job, allocating resources only outside dry-run mode.

    Launching a profile requires claiming its deterministic id FIRST (``db.claim_profile_run`` /
    ``db.reclaim_spent_profile_run``), because the id is derived from the workload rather than the
    account: without the claim two submitters of the same config both launch, the work is profiled
    and billed twice, and the takeover that unwedges a spent profile loses the ordering it compares
    against.
    """
    if prepared_job is not None:
        prepared = prepared_job
    else:
        prepared = prepare_job(
            spec,
            billing_context=billing_context,
            platform_context=platform_context,
            owner_key_id=owner_key_id,
        )
    public_spec = prepared.public_spec
    worker_spec = prepared.worker_spec
    estimated_cost_usd = prepared.estimated_cost_usd
    from flash.content.multimodal import preflight_validate_image_opd
    from flash.server.domain.teacher_broker import preflight_validate_managed_teacher

    preflight_validate_image_opd(worker_spec)
    preflight_validate_managed_teacher(worker_spec)
    from flash.providers import INSTANCE_PROVIDERS, available_providers

    if not dry_run:
        # Record the warm-start dependency on the SOURCE repo so the artifact GC spares it while this
        # child is around (best-effort; never blocks submission). A dry-run preview must not mutate
        # the source repo, so this HF write stays real-submit-only (unlike the read-only preflights
        # above, which now run in both modes).
        _runner()._mark_warmstart_source(worker_spec, public_spec.run_id)
    # env ref->sha pin is deferred (background) or after status save (sync) — never on creation path.
    status = _runner().RunStatus(
        run_id=public_spec.run_id,
        state="queued",
        spec=public_spec.to_dict(),
        estimated_cost_usd=estimated_cost_usd,
        billing_context=billing_context,
        billing_state="pending" if billing_context else None,
        platform_context=platform_context,
        workload_profile_kind=worker_spec.workload_profile_kind or None,
        workload_profile_input_digest=worker_spec.workload_profile_input_digest or None,
        workload_profile=worker_spec.workload_profile or None,
        effective_preparation={
            "worker_spec": worker_spec.to_internal_dict(),
            "workload_profile": worker_spec.workload_profile or None,
            "adapter_identity": prepared.adapter_identity,
            "preparation_digest": _runner()._preparation_digest(
                public_spec, worker_spec, prepared.adapter_identity
            ),
            "backend": _runner().TRAINER_BACKEND,
        },
        # Snapshot the instance providers available at submit so a later handle-less recovery can fail
        # closed for any phantom-capable one whose creds were since dropped (see _confirm_run_clear).
        # Creds-only check (available_providers -> is_configured), no network on the create path.
        submitted_instance_providers=[n for n in available_providers() if n in INSTANCE_PROVIDERS],
    )
    save_kwargs = {
        "_run_deadline_at": (
            status.created_at
            + float(public_spec.gpu.max_wall_seconds)
            + (
                _runner()._WORKLOAD_PROFILE_QUEUE_ALLOWANCE_SECONDS
                if worker_spec.workload_profile_kind
                else 0.0
            )
        ),
        "_next_attempt": 0,
        "_opd_retry_contract_version": (
            OPD_RETRY_CONTRACT_VERSION
            if public_spec.algorithm == "opd"
            else _runner()._PRIVATE_VALUE_UNSET
        ),
    }
    if worker_spec.workload_profile_kind:
        joined = _persist_profile_submission(status, save_kwargs)
        if joined is not None:
            return joined
    else:
        _runner()._save_status(status, **save_kwargs)
    _runner()._report_status(status)
    if dry_run:
        # A dry-run persists a state=dry_run record (retrievable, listable, and stageable for a
        # deploy dry-run) — same contract as a real submit minus GPU allocation, provisioning, and
        # billing. Everything above already validated the spec; just flip the state and return.
        status.state = "dry_run"
        _runner()._save_status(status)
        _runner()._report_status(status)
        return status
    if background:
        threading.Thread(
            target=_runner()._run_job_background,
            args=(worker_spec, runtime_secrets or {}),
            kwargs={"resolve_env_sha": True},
            daemon=True,
        ).start()
        return _runner().get_status(public_spec.run_id)
    worker_spec = _runner()._assign_resolved_env_sha(worker_spec)
    if runtime_secrets:
        _runner()._run_job(worker_spec, runtime_secrets=runtime_secrets)
    else:
        _runner()._run_job(worker_spec)
    return _runner().get_status(public_spec.run_id)
