"""Persisted remote context and pending RunPod creation recovery."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from flash.core.spec import JobSpec
from flash.providers._lifecycle.instances.poll import _attempt_int

if TYPE_CHECKING:
    from flash.providers.core.base import JobHandle


@dataclass(frozen=True)
class _AttachContext:
    worker_spec: JobSpec
    persisted_remote: dict
    handle: JobHandle
    seed: int
    recovered_attempt: int
    next_attempt: int
    source_snapshot: dict | None


class _PendingRunpodCreateAbsent(RuntimeError):
    def __init__(self, context: _AttachContext) -> None:
        super().__init__("runpod pending create was authoritatively absent")
        self.context = context


def _build_attach_context(
    worker_spec: JobSpec,
    persisted_remote: dict,
) -> _AttachContext:
    """Validate the persisted handle and collect the inputs needed to poll it."""
    from flash.providers.core.base import JobHandle
    from flash.runner.lifecycle.status import get_status, source_snapshot_from_status

    remote = dict(persisted_remote)
    seed = int(remote.pop("seed", worker_spec.seed))
    remote.pop("code_prefix", None)
    source_snapshot = source_snapshot_from_status(get_status(worker_spec.run_id))
    provider_name = remote.get("provider")
    if not isinstance(provider_name, str) or not provider_name:
        raise ValueError("persisted provider identity is missing or invalid")
    recovered_attempt = _attempt_int(remote.get("attempt"))
    if recovered_attempt is None:
        raise ValueError("persisted attempt identity is missing or invalid")
    # strip the allocation stamp off the handle copy only; the stamp belongs to
    # persisted_remote, which _carry_allocation_stamp reads when adopting metrics.
    remote.pop("allocated_gpu", None)
    remote.pop("allocated_gpu_count", None)
    return _AttachContext(
        worker_spec=worker_spec,
        persisted_remote=persisted_remote,
        handle=JobHandle.from_dict(remote),
        seed=seed,
        recovered_attempt=recovered_attempt,
        next_attempt=recovered_attempt + 1,
        source_snapshot=source_snapshot,
    )


def _resolve_pending_runpod_context(
    run_id: str,
    context: _AttachContext,
    *,
    deadline_at: float,
) -> _AttachContext:
    """Adopt and durably persist an exact RunPod Pod before any recovered poll."""
    if context.handle.provider != "runpod":
        return context
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.execution.identity import RunpodCreateAbsent, RunpodPodHandle
    from flash.providers.runpod.execution.pods import (
        resolve_pending_handle,
        terminate_handle,
    )
    from flash.runner.accounting.reconciliation import (
        _compare_and_replace_remote,
        _record_cleanup_remote,
    )
    from flash.runner.supervise.lifecycle import _CompletedAttemptPending

    strict = RunpodPodHandle.from_dict(context.handle.to_dict())
    while strict.pending:
        try:
            resolved = resolve_pending_handle(
                strict,
                context.worker_spec,
                context.seed,
                deadline_at=deadline_at,
            )
        except RunpodCreateAbsent:
            terminate_handle(strict, deadline_at=max(deadline_at, time.time() + 120.0))
            raise _PendingRunpodCreateAbsent(context) from None
        except Exception as exc:
            raise _CompletedAttemptPending(
                "runpod pending Pod identity is not yet complete enough to adopt"
            ) from exc
        replacement_remote = {**context.persisted_remote, **resolved.to_dict()}
        if not _compare_and_replace_remote(run_id, context.persisted_remote, replacement_remote):
            if not _record_cleanup_remote(run_id, replacement_remote):
                raise RuntimeError("exact RunPod cleanup target could not be persisted")
            raise _CompletedAttemptPending(
                "runpod pending Pod ownership changed before exact identity persistence"
            )
        context = replace(
            context,
            persisted_remote=replacement_remote,
            handle=JobHandle.from_dict(resolved.to_dict()),
        )
        strict = resolved
    return context
