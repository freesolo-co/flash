"""Durable GPU attempt handles and shared worker payload construction."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, ClassVar

from flash.providers._lifecycle.instances.poll import _attempt_int
from flash.providers._lifecycle.net.deadline import require_deadline_at

# Bounded so the name is never truncated at launch — truncation desyncs the sweep-matched prefix.
_MAX_NAME = 60
_SUFFIX_BUDGET = 12
_PREFIX_BUDGET = _MAX_NAME - _SUFFIX_BUDGET


def run_label_prefix(run_id: str) -> str:
    """The prefix EVERY instance label for ``run_id`` starts with, bounded to the name budget."""
    base = run_id if run_id.startswith("flash-") else f"flash-{run_id}"
    if len(base) <= _PREFIX_BUDGET:
        return base
    h = hashlib.sha1(base.encode()).hexdigest()[:8]
    return f"{base[: _PREFIX_BUDGET - 9]}-{h}"


def label_matches_run(label: str, prefix: str) -> bool:
    """True iff ``label`` belongs to the run whose prefix is ``prefix`` -- an EXACT match, or the
    prefix followed by the ``-a`` attempt boundary. Boundary-anchored so ``flash-100`` never matches
    ``flash-1000-...`` (or vice versa). The single label-ownership test every instance provider's
    sweep and run-scoped teardown shares."""
    return label == prefix or label.startswith(prefix + "-a")


def instance_label(run_id: str, attempt: int) -> str:
    """Instance name: run plus the attempt executing on this host.

    Run-derived so ``sweep_orphans`` can tell ours from anything else on the account, and bounded
    (via ``run_label_prefix``) so the provider never truncates it. The seed is not part of the name:
    a run has exactly one, so it distinguished nothing here while competing with the attempt for the
    digit budget, and a truncated attempt is one two attempts of a run can collide on.
    """
    attempt_i = _attempt_int(attempt)
    if attempt_i is None:
        raise ValueError("instance attempt identity is invalid")
    attempt_s = str(attempt_i)
    if len(attempt_s) > _SUFFIX_BUDGET - len("-a"):
        raise ValueError("instance attempt identity exceeds the provider name budget")
    return f"{run_label_prefix(run_id)}-a{attempt_s}"


@dataclass
class InstanceJobHandle:
    """Common resource identity and validation for persisted sandbox handles.

    Persisted in `RunStatus.remote` for reattach and cancellation. Subclasses add locator fields;
    `instance_id` has no safe default because it is the poll/destroy target.
    """

    instance_id: int | str
    gpu: str
    hourly_usd: float
    attempt: int
    started_ts: float

    provider: ClassVar[str] = "instance"

    @staticmethod
    def _coerce_instance_id(raw: Any) -> Any:
        """Validate the concrete resource identifier in the subclass."""
        return raw

    def _extra_to_dict(self) -> dict:
        """Provider-specific fields, serialized between ``instance_id`` and the shared tail."""
        return {}

    @staticmethod
    def _extra_from_dict(d: dict) -> dict:
        """Provider-specific fields, parsed from a persisted handle dict (kwargs for the constructor)."""
        return {}

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "instance_id": self.instance_id,
            **self._extra_to_dict(),
            "gpu": self.gpu,
            "hourly_usd": self.hourly_usd,
            "attempt": self.attempt,
            "started_ts": self.started_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InstanceJobHandle:
        if d.get("provider") != cls.provider:
            raise ValueError(f"persisted {cls.provider} provider identity is invalid")
        try:
            instance_id = cls._coerce_instance_id(d["instance_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"persisted {cls.provider} instance identity is invalid") from exc
        attempt = _attempt_int(d.get("attempt"))
        if attempt is None:
            raise ValueError(f"persisted {cls.provider} attempt identity is invalid")
        started_raw = d.get("started_ts")
        if isinstance(started_raw, bool) or not isinstance(started_raw, (int, float)):
            raise ValueError(f"persisted {cls.provider} launch timestamp is invalid")
        started_ts = float(started_raw)
        if not math.isfinite(started_ts) or started_ts <= 0:
            raise ValueError(f"persisted {cls.provider} launch timestamp is invalid")
        gpu = d.get("gpu")
        if not isinstance(gpu, str) or not gpu:
            raise ValueError(f"persisted {cls.provider} gpu identity is invalid")
        hourly_raw = d.get("hourly_usd")
        if isinstance(hourly_raw, bool) or not isinstance(hourly_raw, (int, float)):
            raise ValueError(f"persisted {cls.provider} hourly rate is invalid")
        hourly_usd = float(hourly_raw)
        if not math.isfinite(hourly_usd) or hourly_usd < 0:
            raise ValueError(f"persisted {cls.provider} hourly rate is invalid")
        return cls(
            instance_id=instance_id,
            gpu=gpu,
            hourly_usd=hourly_usd,
            attempt=attempt,
            started_ts=started_ts,
            **cls._extra_from_dict(d),
        )


def build_payload(
    spec,
    attempt: int,
    *,
    arm: str,
    runtime_secrets: dict | None = None,
    source_snapshot: dict | None = None,
    deadline_at: float | None = None,
) -> dict:
    """Build the immutable worker input with artifact location, deadline, and attempt identity."""
    from flash.envs.loading.base import worker_pip_with_extras
    from flash.providers._lifecycle.net.worker import (
        build_worker_env,
    )
    from flash.snapshot.archive import parse_descriptor

    env = build_worker_env(spec, runtime_secrets=runtime_secrets)
    absolute_deadline = require_deadline_at(deadline_at)
    attempt_id = _attempt_int(attempt)
    if attempt_id is None:
        raise ValueError("instance attempt identity is invalid")
    max_wall_seconds = float(spec.gpu.max_wall_seconds)
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "run_id": spec.run_id,
        "seed": spec.seed,
        "flash_arm": arm,
        "env": env,
        # per-run env wheel; the bootstrap pip-installs extra_pip for every job. the author's
        # [environment] pip is appended to the worker requirement, never substituted for it.
        "extra_pip": worker_pip_with_extras(spec.environment.id, spec.environment.pip),
        "hf_prefix": f"{spec.phase}/{spec.run_id}",
        "deadline_at": absolute_deadline,
        "run_created_at": absolute_deadline - max_wall_seconds,
        "run_max_wall_seconds": max_wall_seconds,
        "attempt": attempt_id,
    }
    payload["source_snapshot"] = parse_descriptor(source_snapshot).to_dict()
    return payload
