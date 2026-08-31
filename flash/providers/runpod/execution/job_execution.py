"""RunPod endpoint deployment and job polling.

Split out of ``flash.providers.runpod.execution.jobs`` to keep that module under the file-size limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import flash.providers.runpod.serverless.endpoints as runpod_endpoints
import flash.providers.runpod.serverless.naming as runpod_naming
from flash._internal.logging import get_logger
from flash.core.spec import gpu_count_of
from flash.providers._lifecycle.instances.poll import (
    _attempt_int,
)
from flash.providers._lifecycle.net import worker as runpod_worker
from flash.providers._lifecycle.net.deadline import (
    CREATE_ALLOWANCE_S,
    deadline_kwargs,
    remaining_seconds,
    require_create_allowance,
    require_deadline_at,
)
from flash.providers.artifacts.hf import (
    make_hf_failure_detail_reader,
    make_hf_heartbeat_reader,
)
from flash.providers.core.base import PollResult, UnreconciledCreateError, canonical_gpu
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.client.gpus import flash_gpu
from flash.providers.runpod.execution import polling as runpod_polling
from flash.providers.runpod.execution import resources as runpod_resources
from flash.providers.runpod.execution.jobs import (
    JobHandle,
    stall_kwargs,
)
from flash.providers.runpod.execution.resources import WEIGHT_CACHE_GROW_BUDGET_S

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


def _is_workers_quota_error(exc: Exception) -> bool:
    """Return whether RunPod reports exhausted account worker quota."""
    return "max workers across all endpoints" in str(exc).lower()


def _is_balance_error(exc: Exception) -> bool:
    """Return whether RunPod refuses endpoint creation for account balance."""
    return "account balance" in str(exc).lower()


def submit_attempt(
    spec,
    log=None,
    on_handle=None,
    attempt: int = 0,
    runtime_secrets: dict[str, str] | None = None,
    on_last_gpu: bool = False,
    *,
    source_snapshot: dict,
    deadline_at: float | None = None,
) -> PollResult:
    """Deploy, submit, persist handle via ``on_handle``, and poll one attempt to completion."""
    from flash.envs.loading.base import worker_pip_with_extras
    from flash.snapshot.archive import parse_descriptor

    deadline_at = require_deadline_at(deadline_at)
    attempt_id = _attempt_int(attempt)
    if attempt_id is None:
        raise ValueError("RunPod attempt identity is invalid")
    source_descriptor = parse_descriptor(source_snapshot)
    timeout_s = int(require_create_allowance(deadline_at))
    # every attempt gets its own endpoint, so a later one lands on a fresh host rather than the
    # same throttled or sick one. attempt zero is suffixed too: it is an attempt like any other.
    suffix = runpod_naming.attempt_suffix(spec.run_id, attempt_id)
    # the author's [environment] pip is appended to the worker requirement, never substituted for it.
    extra_pip = worker_pip_with_extras(spec.environment.id, spec.environment.pip)
    worker_env = runpod_worker.build_worker_env(
        spec,
        runtime_secrets=runtime_secrets,
    )
    worker_env["ATTEMPT"] = str(attempt_id)
    endpoint_id, name, key_fingerprint = deploy_train_endpoint(
        spec.gpu.type,
        execution_timeout_ms=timeout_s * 1000,
        name_suffix=suffix,
        disk_gb=spec.gpu.disk_gb,
        spec=spec,
        **deadline_kwargs(deploy_train_endpoint, deadline_at),
    )
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "run_id": spec.run_id,
        "attempt": attempt_id,
        "seed": spec.seed,
        "env": worker_env,
        "extra_pip": extra_pip,
        "source_snapshot": source_descriptor.to_dict(),
        "deadline_at": deadline_at,
    }
    try:
        require_create_allowance(deadline_at)
        submitted_ts = time.time()
        job_id = runpod_api.submit_job(
            endpoint_id,
            payload,
            key_fingerprint=key_fingerprint,
            **deadline_kwargs(runpod_api.submit_job, deadline_at),
        )
    except Exception as exc:
        # the queue post is non-idempotent: only a positively confirmed endpoint deletion makes
        # the original failure safe to retry. otherwise persist exact cleanup identity and stop.
        deletion_confirmed = False
        with contextlib.suppress(Exception):
            deletion_confirmed = (
                runpod_api.delete_endpoint_for_fingerprint(endpoint_id, key_fingerprint) is True
            )
        if deletion_confirmed:
            raise
        if on_handle is not None:
            on_handle(
                JobHandle(
                    endpoint_id,
                    name,
                    key_fingerprint,
                    None,
                    attempt_id,
                    time.time(),
                ).to_dict()
            )
        raise UnreconciledCreateError(
            "RunPod queue submission could not be reconciled and endpoint deletion was unconfirmed"
        ) from exc
    handle = JobHandle(
        endpoint_id,
        name,
        key_fingerprint,
        job_id,
        attempt_id,
        submitted_ts,
    )
    if log is not None:
        print(
            f"submitted job: endpoint={name} ({endpoint_id}) job={job_id} "
            f"attempt={attempt} gpu={spec.gpu.type} phase={spec.phase}",
            file=log,
            flush=True,
        )
    if on_handle is not None:
        on_handle(handle.to_dict())
    hf_repo = spec.train.hf_repo
    prefix = f"{spec.phase}/{spec.run_id}"
    reader = (
        make_hf_heartbeat_reader(
            hf_repo,
            prefix,
            **deadline_kwargs(make_hf_heartbeat_reader, deadline_at),
        )
        if hf_repo
        else None
    )
    failure_reader = (
        make_hf_failure_detail_reader(
            hf_repo,
            prefix,
            spec.phase,
            attempt=attempt_id,
            **deadline_kwargs(make_hf_failure_detail_reader, deadline_at),
        )
        if hf_repo
        else None
    )
    return runpod_polling.poll_job(
        handle,
        log=log,
        heartbeat_reader=reader,
        failure_detail_reader=failure_reader,
        current_attempt=attempt_id,
        **deadline_kwargs(runpod_polling.poll_job, deadline_at),
        # the count actually rented for this attempt, which allocation may have resolved to fewer
        # cards than the spec's ceiling named -- so read the effective spec, not the run's request.
        **stall_kwargs(on_last_gpu=on_last_gpu, gpu_count=gpu_count_of(spec)),
    )


@dataclass
class _DeployContext:
    """Deadline and cache-reconciliation state shared by deploy attempts."""

    deadline_at: float | None
    spec: Any
    cache_volumes: dict[str, int] | None
    reconciled: set[str]

    def reconciles_a_managed_cache(self) -> bool:
        """Whether a grow on this call can actually spend budget.

        Mirrors ``grow_weight_cache_volumes``'s own early return: a run attaching no managed cache
        reconciles nothing, so reserving for it would shorten the deadline for a create that was
        never going to grow anything.
        """
        if self.cache_volumes is not None:
            return bool(self.cache_volumes)
        from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

        base = getattr(self.spec.gpu, "network_volume", None) if self.spec is not None else None
        return str(base or "") == WEIGHT_CACHE_VOLUME_NAME

    def create_deadline(self, key_count: int) -> float | None:
        """``deadline_at`` less the grow budget still owed to unreconciled accounts.

        Creates, sweeps, and backoffs cannot spend this reserve. Any attempt reaching create has
        reconciled; cache-free runs reserve nothing. See ``require_launchable`` for admission.
        """
        if self.deadline_at is None or not self.reconciles_a_managed_cache():
            return self.deadline_at
        owed = max(0, key_count - len(self.reconciled))
        return self.deadline_at - WEIGHT_CACHE_GROW_BUDGET_S * owed

    def attempt_deadline(self, key_count: int, active_key: str | None) -> float | None:
        """``deadline_at`` less the ONE grow slice this attempt's reconciliation can need.

        Admission funds only this attempt's grow. Once its account reconciles, do not charge the
        slice again; before selection, reserve conservatively.
        """
        if self.deadline_at is None or not self.reconciles_a_managed_cache():
            return self.deadline_at
        if active_key is not None and active_key in self.reconciled:
            return self.deadline_at
        if key_count <= len(self.reconciled):
            return self.deadline_at
        return self.deadline_at - WEIGHT_CACHE_GROW_BUDGET_S

    def require_launchable(self, key_count: int, active_key: str | None) -> None:
        """Fail closed unless this attempt can still reconcile itself and then create.

        Raw-deadline admission can yield zero growth and mount a stale volume. Reserving one slice
        prevents the later "Disk quota exceeded" failure.
        """
        if self.deadline_at is not None:
            require_create_allowance(self.attempt_deadline(key_count, active_key))


def _prepare_quota_retry(
    context: _DeployContext,
    name: str,
    quota_attempt: int,
    quota_max_retries: int,
    key_count: int,
    active_key: str | None,
) -> None:
    """Sweep safe idle endpoints and back off before a quota retry."""
    # under quota pressure, reap only scaled-to-zero orphans on this account.
    # `reap_warm=False` protects live runs' between-job workers; reserve failover growth
    # from this sweep and backoff.
    context.require_launchable(key_count, active_key)
    quota_deadline = context.create_deadline(key_count)
    swept = runpod_resources._sweep_idle_flash_endpoints(
        protected={runpod_resources.canonical_endpoint_name(name)},
        min_idle_s=0.0,
        reap_warm=False,
        **deadline_kwargs(runpod_resources._sweep_idle_flash_endpoints, quota_deadline),
    )
    wait_s = 30 * quota_attempt
    if context.deadline_at is not None:
        wait_s = min(wait_s, remaining_seconds(quota_deadline))
    logger.warning(
        "RunPod worker quota hit (attempt %d/%d): swept %d idle flash-* endpoint(s); "
        "retrying in %ds",
        quota_attempt + 1,
        quota_max_retries,
        swept,
        wait_s,
    )
    if wait_s > 0:
        time.sleep(wait_s)


def _deploy_with_failover(
    context: _DeployContext,
    name: str,
    deploy_once: Callable[[], tuple[object, str]],
    rp_keys: Any,
) -> tuple[object, str]:
    """Retry quota failures and walk configured accounts without losing cleanup guards."""
    quota_max_retries = 3
    resource = None
    owning_fingerprint = None
    # bound by count, not advance_key() return value: advance_key() always wraps so can't signal exhaustion.
    failovers_left = max(0, rp_keys.key_count() - 1)
    while resource is None:
        deploy_failover_exc: Exception | None = None
        for quota_attempt in range(quota_max_retries):
            if quota_attempt > 0:
                _prepare_quota_retry(
                    context,
                    name,
                    quota_attempt,
                    quota_max_retries,
                    rp_keys.key_count(),
                    rp_keys.active_key(),
                )
            try:
                resource, owning_fingerprint = deploy_once()
                break  # success
            except Exception as exc:
                if _is_balance_error(exc):
                    # a broke account can't be helped by sweeping idle endpoints: fail over now
                    deploy_failover_exc = exc
                    break
                if not _is_workers_quota_error(exc):
                    raise
                deploy_failover_exc = exc
        if resource is not None:
            break
        if failovers_left > 0:
            with runpod_endpoints.FLASH_SDK_LOCK:
                rp_keys.advance_key()
            failovers_left -= 1
            reason = (
                "has insufficient balance"
                if deploy_failover_exc is not None and _is_balance_error(deploy_failover_exc)
                else "worker quota exhausted after sweeping"
            )
            logger.warning(
                "RunPod account %s; failing over to the next RUNPOD_API_KEY account (%d configured)",
                reason,
                rp_keys.key_count(),
            )
            continue
        raise deploy_failover_exc or RuntimeError(
            "deploy_train_endpoint: deploy failover exhausted"
        )
    return resource, owning_fingerprint


def apply_disk_gb(config, disk_gb: int | None) -> None:
    """Raise the worker container disk while preserving the SDK default floor."""
    if not disk_gb:
        return
    template = getattr(config, "template", None)
    if template is None:
        logger.warning("disk_gb=%s requested but endpoint config has no template", disk_gb)
        return
    template.containerDiskInGb = max(int(disk_gb), int(template.containerDiskInGb or 0))


def deploy_train_endpoint(
    friendly_gpu: str,
    execution_timeout_ms: int | None = None,
    name_suffix: str | None = None,
    disk_gb: int | None = None,
    spec=None,
    endpoint_kwargs: dict | Callable[[], dict] | None = None,
    deadline_at: float | None = None,
    cache_volumes: dict[str, int] | None = None,
) -> tuple[str, str, str]:
    """Deploy a uniquely-named worker endpoint and return its id, name, and owning fingerprint.

    Rebuild callable `endpoint_kwargs` per account so failover cannot reuse another account's
    volume id. `cache_volumes` supplies managed sizes when `spec` is absent.
    """
    from runpod_flash import Endpoint
    from runpod_flash.core.resources.resource_manager import ResourceManager

    from flash.core.spec import gpu_count_of
    from flash.providers.runpod.client import auth as rp_keys
    from flash.providers.runpod.client.auth import ensure_auth

    runpod_endpoints._patch_runpod_backoff()
    friendly = canonical_gpu(friendly_gpu)
    name = runpod_naming.endpoint_name(friendly, name_suffix)
    # Scope the SDK registry to the run, not the attempt. The endpoint name identifies one attempt,
    # but teardown is run-scoped: it isolates on the run digest and reaps every attempt in one call.
    # Writing resources.pkl under the attempt gives terminate_endpoint a registry it never opens, so
    # its undeploy leg reads empty and cleanup silently rests on the REST sweep alone.
    registry_scope = runpod_naming.run_target_of(name_suffix) or name_suffix
    image = runpod_worker.worker_image_for_gpu(friendly)
    context = _DeployContext(deadline_at, spec, cache_volumes, set())

    def _deploy_once() -> tuple[object, str]:
        """Create under one serialized account selection and return its owning fingerprint."""
        context.require_launchable(rp_keys.key_count(), rp_keys.active_key())
        with runpod_endpoints.FLASH_SDK_LOCK:
            owning_key = ensure_auth()
            # re-check the selected key under the lock: another deploy may advance the global key
            # after admission. an unreconciled real account must retain its grow slice or fail
            # closed.
            context.require_launchable(rp_keys.key_count(), owning_key)
            owning_fingerprint = runpod_api.key_fingerprint(owning_key)
            runpod_endpoints.isolate_flash_state(registry_scope)
            kwargs = {
                "name": name,
                "gpu": flash_gpu(friendly),
                # one worker occupies gpu.count cards of this class; count == 1 is the historical path.
                "gpu_count": gpu_count_of(spec),
                "min_cuda_version": runpod_endpoints.min_cuda_for(friendly),
                "execution_timeout_ms": execution_timeout_ms
                or runpod_worker.DEFAULT_EXECUTION_TIMEOUT_MS,
                "workers": (0, 1),
            }
            kwargs["image"] = image
            # reconcile the selected account before attach because the SDK returns existing volumes
            # at their old size. do this once per account and preserve the later create allowance.
            if owning_key not in context.reconciled:
                context.reconciled.add(owning_key)
                # spends against the raw deadline: this account's slice of the reserve is released
                # by the line above, so the grow draws its own budget and still yields the create
                # allowance. the slices of the accounts not yet reconciled stay held back.
                runpod_resources.grow_weight_cache_volumes(
                    spec, owning_key, deadline_at, wanted=cache_volumes
                )
            # re-invoke factory per account (avoids reusing a volume id stamped for the prior account).
            override = endpoint_kwargs() if callable(endpoint_kwargs) else endpoint_kwargs
            kwargs.update(
                override
                if override is not None
                else runpod_resources.weight_cache_endpoint_kwargs(spec)
            )
            ep = Endpoint(**kwargs)
            config = ep._build_resource_config()
            apply_disk_gb(config, disk_gb)
            rm = ResourceManager()
            if deadline_at is None:
                resource = asyncio.run(rm.get_or_deploy_resource(config))
            else:
                # this attempt's own grow has run by now (the reconciled check above), so the
                # re-check is judged without its slice: charging it again would re-deduct a cost
                # already paid and reject a launchable create within one slice of the deadline.
                context.require_launchable(rp_keys.key_count(), owning_key)
                create_deadline = context.create_deadline(rp_keys.key_count())
                # create may spend only down to the failover grow reserve. yield the proven create
                # allowance so a large reserve cannot produce a zero timeout.
                remaining = max(CREATE_ALLOWANCE_S, remaining_seconds(create_deadline))
                resource = asyncio.run(
                    asyncio.wait_for(
                        rm.get_or_deploy_resource(config),
                        timeout=remaining,
                    )
                )
            return resource, owning_fingerprint

    resource, owning_fingerprint = _deploy_with_failover(context, name, _deploy_once, rp_keys)
    endpoint_id = getattr(resource, "id", None)
    if not endpoint_id:
        raise RuntimeError(f"deploy_train_endpoint: no endpoint id on resource {resource!r}")
    if owning_fingerprint is None:
        raise RuntimeError("deploy_train_endpoint: owning RunPod key is unavailable")
    return endpoint_id, name, owning_fingerprint
