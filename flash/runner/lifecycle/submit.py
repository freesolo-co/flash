"""Turning a validated job spec into a run: `prepare_job` and `submit_job`.

`prepare_job` is the read-only half -- resolve the model, validate the effective spec, and settle
warm-start provenance -- so the caller can be told what a run would cost and consume before any
state exists. `submit_job` is the half that commits: it writes the first status, then either runs
the job inline or hands it to a background supervisor thread.

Split out of `flash.runner` to keep that module under the file-size limit.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

from flash._internal.diagnostics import sanitize_diagnostic
from flash.core import catalog
from flash.core.spec import TRAINER_BACKEND, JobSpec
from flash.core.spec_persistence import PREPARATION_ENVELOPE_VERSION
from flash.providers._lifecycle.net import worker as provider_worker
from flash.runner.accounting import artifacts, weight_cache
from flash.runner.lifecycle import preparation, reporting, state
from flash.runner.lifecycle import status as status_ops
from flash.runner.supervise import lifecycle as supervision
from flash.teacher.retry_contract import OPD_RETRY_CONTRACT_VERSION

logger = logging.getLogger(__name__)


class SourceSnapshotPublicationError(RuntimeError):
    """Managed source publication failed before a provider could be created."""

    plane_fault = True


@dataclass(frozen=True)
class PreparedJob:
    public_spec: JobSpec
    worker_spec: JobSpec
    estimated_cost_usd: float
    adapter_identity: dict | None = None
    prompt_budget: object | None = None


def _with_model_disk(spec: JobSpec, info) -> dict:
    data = spec.to_internal_dict()
    required = int(getattr(info, "min_disk_gb", 0) or 0)
    if required > int(data["gpu"].get("disk_gb") or 0):
        data["gpu"] = {**data["gpu"], "disk_gb": required}
    return data


def prepare_job(
    spec: JobSpec,
    *,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
    owner_key_id: int | None = None,
) -> PreparedJob:
    """Prepare all read-only submission inputs before persistence or allocation."""
    # before _resolve_model_revision, and before every sizing step below: a warm start inherits its
    # source's pin and provenance, and `resolve_model`/`_with_model_disk` size against whatever
    # revision the spec carries by then.
    spec = preparation._inherit_warmstart_revision(
        spec,
        owner_org_id=preparation._context_org_id(billing_context)
        or preparation._context_org_id(platform_context),
        owner_key_id=owner_key_id,
    )
    spec = preparation._resolve_model_revision(spec, required=spec.algorithm == "sft")
    if spec.algorithm == "sft":
        spec = preparation._require_pinned_profile_environment(spec)
        spec = preparation._require_sft_workload_profile(spec)
    if spec.train.structured_outputs:
        from flash.serve.deployment.preflight import preflight_serving_path

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
    if spec.gpu.provider or spec.gpu.providers or spec.gpu.type:
        from flash.providers.core.base import providers_for
        from flash.providers.core.registry import (
            PROVIDER_NAMES,
            available_providers,
            validated_provider_preferences,
        )

        configured = available_providers()
        provider = spec.gpu.provider.strip().lower()
        providers = validated_provider_preferences(spec.gpu.providers, allow_empty=True)
        if provider and providers:
            raise ValueError("gpu.provider and gpu.providers cannot both be set")
        if provider:
            if provider not in PROVIDER_NAMES:
                raise ValueError(f"unknown gpu.provider {spec.gpu.provider!r}")
            if provider not in configured:
                raise ValueError(f"requested gpu.provider {provider!r} is not configured")
            for gpu_type in spec.gpu.acceptable_types:
                if provider not in providers_for(gpu_type):
                    raise ValueError(
                        f"gpu.provider {provider!r} cannot provision gpu.type {gpu_type!r}"
                    )
        else:
            for gpu_type in spec.gpu.acceptable_types:
                if not any(name in configured for name in providers_for(gpu_type)):
                    raise ValueError(f"no configured provider can provision gpu.type {gpu_type!r}")
    info = catalog.resolve_model(spec.model, spec.algorithm, model_revision=spec.model_revision)
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
        from flash.engine.worker.train.opd.orchestration.validation import (
            preflight_opd_structured_outputs,
        )

        preflight_opd_structured_outputs(
            spec.train.structured_outputs,
            model_id=spec.model,
            model_revision=spec.model_revision,
        )
    run_id = spec.run_id if (spec.run_id and spec.run_id != "local") else state.new_run_id()
    spec = JobSpec.from_dict({**_with_model_disk(spec, info), "run_id": run_id})
    spec = artifacts._assign_managed_hf_repo(spec)
    spec = weight_cache._assign_weight_cache_volume(spec, info)
    owner_org_id = preparation._context_org_id(billing_context) or preparation._context_org_id(
        platform_context
    )
    public_spec, worker_spec, adapter_identity, warm_start_context = (
        preparation._prepare_init_from_adapter(
            spec,
            owner_org_id=owner_org_id,
            owner_key_id=owner_key_id,
            token=os.environ.get("HF_TOKEN"),
        )
    )
    # these read-only gates belong to preparation: every submit path passes here exactly once, and
    # callers receive the pinned worker spec before quoting, affordability, persistence, or allocation.
    worker_spec, environment_ref_deferred = artifacts.preflight_validate_environment_ref(
        worker_spec
    )
    from flash.content.multimodal import preflight_validate_image_opd
    from flash.server.domain.teacher.broker import preflight_validate_managed_teacher

    preflight_validate_image_opd(
        worker_spec,
        scan_packaged_environment=not environment_ref_deferred,
    )
    preflight_validate_managed_teacher(worker_spec)
    from flash.cost.currency import usd_amount
    from flash.cost.spec import estimate_for_spec

    estimated_cost_usd = usd_amount(estimate_for_spec(worker_spec).total_usd)
    # derive the rl prompt budget from the same resolved spec the quote is built from, so the
    # reported budget describes the run that was actually priced and submitted.
    from flash.engine.plan.prompt_budget import rl_prompt_budget

    prompt_budget = rl_prompt_budget(
        worker_spec,
        warm_start_context=warm_start_context,
    )
    return PreparedJob(
        public_spec=public_spec,
        worker_spec=worker_spec,
        estimated_cost_usd=estimated_cost_usd,
        adapter_identity=adapter_identity,
        prompt_budget=prompt_budget,
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
    if not committed or committed == weight_cache.WEIGHT_CACHE_VOLUME_NAME:
        return
    if worker_spec.gpu.network_volume != committed:
        raise ValueError("persisted effective preparation drops a non-shared weight-cache volume")


def _effective_preparation_snapshot(
    public_spec: JobSpec,
    worker_spec: JobSpec,
    adapter_identity: dict | None,
    *,
    stored_public: object = None,
) -> dict:
    """Build the current versioned snapshot shape.

    `stored_public` is the public spec as it is ALREADY persisted, and it belongs on every path
    that REWRITES an existing record. The digest is recomputed here, and `_preparation_digest`
    reproduces a pre-1.1.35 run only by replaying the `lora_alpha` omission it reads from the
    stored spec -- but a rewrite reconstructs `public_spec` via `JobSpec.from_dict(status.spec)`,
    whose `to_dict()` re-MATERIALIZES the key. Omitting the argument therefore stamps a digest
    over bytes the stored spec still lacks, and since the rewrite never updates `status.spec`,
    the two halves disagree from that moment on. The create path passes nothing because it
    writes both halves from the same object, so it has no omission to replay.
    """
    return {
        "version": PREPARATION_ENVELOPE_VERSION,
        "worker_spec": worker_spec.to_internal_dict(),
        "workload_profile": worker_spec.workload_profile or None,
        "adapter_identity": adapter_identity,
        "preparation_digest": preparation._preparation_digest(
            public_spec, worker_spec, adapter_identity, stored_public=stored_public
        ),
        "backend": TRAINER_BACKEND,
    }


def _persist_effective_worker_spec(worker_spec: JobSpec) -> bool:
    """Persist the selected worker spec without changing the accepted customer quote."""
    status = status_ops.get_status(worker_spec.run_id)
    if status.state in state.TERMINAL_STATES:
        return False
    snapshot = status.effective_preparation
    public_spec = JobSpec.from_dict(status.spec)
    if public_spec.train.init_from_adapter:
        if not isinstance(snapshot, dict):
            raise ValueError("persisted effective preparation is malformed")
        status_ops.effective_spec_from_status(status)
        adapter_identity = snapshot.get("adapter_identity")
    else:
        adapter_identity = None
    _reject_managed_volume_removal(snapshot, worker_spec)
    preparation._validate_effective_spec(public_spec, worker_spec)
    effective_preparation = _effective_preparation_snapshot(
        public_spec, worker_spec, adapter_identity, stored_public=status.spec
    )
    return status_ops._update(
        worker_spec.run_id,
        status.state,
        effective_preparation=effective_preparation,
    )


def submit_job(
    spec: JobSpec,
    dry_run: bool = False,
    background: bool = False,
    runtime_secrets: dict[str, str] | None = None,
    billing_context: dict | None = None,
    platform_context: dict | None = None,
    owner_key_id: int | None = None,
    prepared_job: PreparedJob | None = None,
) -> state.RunStatus:
    """Submit a prepared job, allocating resources only outside dry-run mode."""
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
    from flash.providers.core.registry import INSTANCE_PROVIDERS, available_providers

    source_snapshot = None
    if not dry_run:
        # record the warm-start dependency on the source repo so artifact gc spares it while this
        # child is around. source publication is also real-submit-only and completes before status
        # persistence, so no provider can be created without a durable immutable descriptor.
        preparation._mark_warmstart_source(worker_spec, public_spec.run_id)
        try:
            source_snapshot = provider_worker.publish_source_snapshot(worker_spec.train.hf_repo)
        except Exception as exc:
            logger.warning(
                "managed source publication failed for run %s: %s",
                public_spec.run_id,
                sanitize_diagnostic(exc, limit=500),
            )
            raise SourceSnapshotPublicationError(
                "managed source publication failed; retry the submission later"
            ) from None
    # env ref->sha pin is deferred (background) or after status save (sync), never on creation path.
    # environment staging runs after status persistence and before provider allocation.
    status = state.RunStatus(
        run_id=public_spec.run_id,
        state="queued",
        spec=public_spec.to_dict(),
        estimated_cost_usd=estimated_cost_usd,
        billing_context=billing_context,
        billing_state="pending" if billing_context else None,
        platform_context=platform_context,
        workload_profile_input_digest=worker_spec.workload_profile_input_digest or None,
        workload_profile=worker_spec.workload_profile or None,
        prompt_budget=prepared.prompt_budget,
        effective_preparation=_effective_preparation_snapshot(
            public_spec, worker_spec, prepared.adapter_identity
        ),
        source_snapshot=source_snapshot,
        # snapshot the instance providers available at submit so a later handle-less recovery can fail
        # closed for any phantom-capable one whose creds were since dropped (see _confirm_run_clear).
        # Creds-only check (available_providers -> is_configured), no network on the create path.
        submitted_instance_providers=[n for n in available_providers() if n in INSTANCE_PROVIDERS],
    )
    save_kwargs = {
        "_run_deadline_at": status.created_at + float(public_spec.gpu.max_wall_seconds),
        "_next_attempt": 0,
        "_opd_retry_contract_version": (
            OPD_RETRY_CONTRACT_VERSION
            if public_spec.algorithm == "opd"
            else state._PRIVATE_VALUE_UNSET
        ),
    }
    state._save_status(status, **save_kwargs)
    reporting._report_status(status)
    if dry_run:
        # A dry-run persists a state=dry_run record (retrievable, listable, and stageable for a
        # deploy dry-run) — same contract as a real submit minus GPU allocation, provisioning, and
        # billing. Everything above already validated the spec; just flip the state and return.
        status.state = "dry_run"
        state._save_status(status)
        reporting._report_status(status)
        return status
    if background:
        threading.Thread(
            target=supervision._run_job_background,
            args=(worker_spec, runtime_secrets or {}),
            daemon=True,
        ).start()
        return status_ops.get_status(public_spec.run_id)
    if runtime_secrets:
        supervision._run_job(worker_spec, runtime_secrets=runtime_secrets)
    else:
        supervision._run_job(worker_spec)
    return status_ops.get_status(public_spec.run_id)
