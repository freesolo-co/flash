"""Decoding rules for PERSISTED job-spec records, as distinct from authored configuration.

`JobSpec` serves four different contracts, and this module owns exactly one of them: reading back
bytes that were already written to `~/.flash/runs/<run_id>.json`. That is not the same job as
validating a config a user just wrote, and the two must not be confused:

- authored configuration is parsed by `flash.schema.spec_from_dict`, which is STRICT. an unknown key
  is a typo in a file the author can still fix, so it fails loudly.
- a persisted record is parsed by `JobSpec.from_dict`, which is TOLERANT by necessity. the record is
  immutable history: it was written by an older Flash, it is never rewritten, and a run that is
  still training depends on it decoding. rejecting it strands the run.

So the coercions here are deliberately looser than anything the authored parser would accept, and
the migrations exist to keep specific historical spellings readable. Neither belongs on the
submission path.

Serialized-byte warning: `flash/runner/preparation.py::_preparation_digest` sha256-hashes the
canonical JSON of `to_dict()` and `to_internal_dict()` output. Key ORDER is safe there
(`sort_keys=True`), but key PRESENCE is not -- adding, dropping, or renaming a key in a decode path
changes the digest and fails integrity validation for every warm-start and workload-profile run in
flight. Change decoding here only with the corpus gate green.

These functions take plain dicts and return plain dicts and scalars. That is the rule that keeps the
module importable by `spec.py` rather than the other way round, and it is why `validated_section`
takes its allowed-key set as an argument instead of reading `fields(TrainSpec)` itself.

What is deliberately NOT here: `_coerce_wandb` and `_parse_persisted_gpu_types` are also called only
from `JobSpec.from_dict`, so they belong to this contract by usage -- but they construct `WandbSpec`
and reach `_validated_gpu_type`, which `__post_init__` needs too. Moving them would make this module
import the dataclasses that import it. The boundary is therefore drawn at "persisted-decoding logic
that does not depend on the spec types themselves"; collapsing that seam properly is the job of the
later split that gives the persisted record its own type, not of this move.
"""

from __future__ import annotations

import logging
from typing import Any

from flash.core.catalog import samples_on_policy

# Removed public top-level fields that a PERSISTED record can still carry. Stored run records are
# never rewritten, and effective_preparation.worker_spec is written with to_internal_dict() (asdict),
# which emitted every field including the defaulted ones -- so every record written before a field
# was dropped still names it. This registry also drives historical public-digest replay for
# model_revision, which remains an internal field. Without it a still-running job can lose its
# recovery, deploy, and serving paths after the upgrade.
#
# Ignored on READ only. Most keys here no longer exist as JobSpec fields; model_revision remains an
# internal runner field, but the public schema rejects it and to_dict omits it. Keeping it in this set
# lets digest recovery identify the historical public key without weakening JobSpec.from_dict.
DROPPED_TOP_LEVEL_KEYS = frozenset(
    {"model_policy", "model_revision", "worker_env", "workload_profile_kind"}
)

# Tolerating a dropped key keeps a pre-upgrade run's recovery path, but the values behind it stop
# being applied -- so a run that authored them now trains on managed defaults instead of what was
# submitted. That is announced rather than silent. Only user-authorable keys qualify: model_policy
# was platform-managed (to_dict popped it), so its loss cannot change what a submitted config asked
# for. The values are deliberately NOT forwarded -- doing so would reinstate the override table this
# removal exists to close, and would deliver it without the validation the deleted parser performed.
_ANNOUNCED_DROPPED_KEYS = frozenset({"worker_env"})

# Removed [train] keys a persisted record can still carry. Deliberately narrow: a key is listed only
# once its behavior is gone, so an unlisted unknown key stays fatal rather than silently ignored.
REMOVED_PERSISTED_TRAIN_KEYS = frozenset({"advantage_clip"})


def announce_dropped_keys(data: dict[str, Any]) -> None:
    """Log the removed user-authored keys a persisted record still carries and no longer applies.

    Only announced when the payload names the run. A public spec is ``to_dict()`` output, which pops
    the server-assigned run_id, and the same stored run is read through both shapes -- so warning
    without an id would emit an unactionable "run <unknown>" line and duplicate the identified one
    the worker-spec read already produced. Every provisioned run records an internal worker spec
    (asdict, run_id included), and that is the read this fires on.
    """
    run_id = str(data.get("run_id") or "").strip()
    if not run_id:
        return
    for key in sorted(_ANNOUNCED_DROPPED_KEYS):
        value = data.get(key)
        if not value:
            continue
        names = ", ".join(sorted(str(k) for k in value)) if isinstance(value, dict) else str(value)
        logging.getLogger("flash.spec").warning(
            "run %s was submitted with [%s] (%s); that table was removed and its values are NOT "
            "applied -- this run uses managed defaults instead. resubmit without it if the run "
            "depends on those values.",
            run_id,
            key,
            names,
        )


def validated_section(
    data: dict[str, Any],
    name: str,
    allowed: set[str],
    *,
    removed: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Read one nested block (``[train]``, ``[gpu]``) of a persisted record, rejecting unknown keys.

    A missing block and an explicit null are the same state -- both mean "authored nothing" -- so both
    normalize to empty rather than one of them raising. ``removed`` names keys whose behavior is gone:
    they are dropped silently, because the record is immutable history and cannot be rewritten. An
    unlisted unknown key stays fatal, which is what keeps the tolerance narrow instead of turning
    every typo in a stored record into a silent default.
    """
    section = data.get(name)
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise TypeError(f"{name} must be an object")
    section = {key: value for key, value in section.items() if key not in removed}
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"{name} has unknown key(s): {', '.join(unknown)}")
    return section


def validated_persisted_providers(
    gpu: dict[str, Any], gpu_type: str, gpu_type_fallbacks: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """``(provider, providers)`` from a persisted ``[gpu]`` block, cross-checked against its classes.

    ``GpuSpec.__post_init__`` runs the same both-set check but cannot replace this one: ``allow_empty``
    here keys off whether the record CARRIED a ``providers`` key, and that presence is exactly what the
    dataclass has already lost by the time it validates. An explicitly empty preference and an absent
    one are different states in a stored record.
    """
    from flash.providers import PROVIDER_NAMES, validated_provider_preferences

    provider = gpu.get("provider", "")
    if not isinstance(provider, str):
        raise TypeError("gpu.provider must be a string")
    provider = provider.strip().lower()
    providers = validated_provider_preferences(
        gpu.get("providers", ()), allow_empty="providers" not in gpu
    )
    if provider and providers:
        raise ValueError("gpu.provider and gpu.providers cannot both be set")
    if provider or providers or gpu_type:
        from flash.providers.base import providers_for

        if provider and provider not in PROVIDER_NAMES:
            raise ValueError(f"unknown gpu.provider {provider!r}")
        # a hard provider pin must carry every acceptable class. provider preferences stay soft: an
        # ineligible named provider contributes no candidate, while eligible named or unnamed
        # configured providers remain available for failover.
        for candidate in (gpu_type, *gpu_type_fallbacks):
            if candidate and provider and provider not in providers_for(candidate):
                raise ValueError(
                    f"gpu.provider {provider!r} cannot provision gpu.type {candidate!r}"
                )
    return provider, providers


def str_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a string-or-list knob to a tuple of strings; a bare string is one element, not iterated."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(s for s in (str(x) for x in value) if s)


def volume_gb(value: Any, default: int = 100) -> int:
    """Parse volume size in GB; non-positive / non-numeric / missing values return default."""
    if isinstance(value, bool):
        # bool is an int subclass; reject to avoid int(True)==1 becoming a 1 GB volume.
        return default
    try:
        gb = int(value)
    except (TypeError, ValueError):
        return default
    return gb if gb > 0 else default


def opt_int(value: Any) -> int | None:
    """Parse optional int; rejects bools (bool is int subclass — int(True)==1 is a footgun)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return int(value)


def opt_float(value: Any) -> float | None:
    """Parse optional float; rejects bools (mirrors _opt_int)."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got bool {value!r}")
    return float(value)


def migrated_optimizer_batch(train: dict, algorithm: str) -> tuple[int | None, int | None]:
    """``(batch_size, prompts_per_step)`` with the pre-1.1.43 rollout spelling migrated.

    grpo/opd authored the optimizer batch as ``batch_size`` until it was split into
    ``prompts_per_step``. The schema rejects the old name on SUBMISSION, but `from_dict` also
    reparses PERSISTED specs, and the deployed release (1.1.40) has no ``prompts_per_step`` at all
    -- so every rollout run in flight across this upgrade carries only the old key. The workers read
    ``prompts_per_step`` alone, so without this a recovered run silently resumes on the recipe
    default: 64 instead of an authored 32 on grpo, which OOMs a card rented for 32, and 8 on opd.

    The old key is MOVED, not copied, and is dropped whenever the new one is present. Leaving it
    populated would re-emit both names from ``to_dict()``, and a spec carrying both is rejected by
    the schema -- breaking the resubmit that recovery and ``flash runs get`` perform, and leaving
    `vram.py::_optimizer_batch_value` (which takes the larger of the two) free to size a card off
    the stale value that ranking ignores. That applies to a payload written mid-upgrade carrying
    BOTH spellings too: ``prompts_per_step`` wins, so the superseded key has to go with it.

    A non-positive legacy value is discarded rather than migrated: ``minimum=1`` would have
    rejected it at submission, and carrying it forward only fails later on a rented GPU. It is
    discarded from BOTH names for the same round-trip reason -- retaining it under the old one
    would re-emit a key the schema rejects for this algorithm.
    """
    batch_size = opt_int(train.get("batch_size"))
    prompts_per_step = opt_int(train.get("prompts_per_step"))
    if not samples_on_policy(algorithm):
        return batch_size, prompts_per_step
    if prompts_per_step is not None:
        return None, prompts_per_step
    return None, batch_size if batch_size is not None and batch_size >= 1 else None
