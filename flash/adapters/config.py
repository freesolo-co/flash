"""one strict reader for the bytes of an ``adapter_config.json``.

Three boundaries admit an adapter by reading this one file, and all three are gated on the same
question: do these exact bytes describe a LoRA adapter that the engine about to load them can
actually serve? Answering it separately let them drift -- customer-owned resolution rejected a
config that hosted admission accepted, so the same artifact was refused before provisioning on one
path and refused inside the paid container on the other.

Every rule here is decidable from the config alone. Nothing in this module reads the hub, the
filesystem, or a provider, so a verdict costs a caller nothing beyond the bytes it already holds.
That is why it sits beside the rank and target rules rather than under a deployment path: the
GPU-side materializer revalidates a cache entry it already holds, and it needs the same verdict
without importing anything that resolves or deploys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from flash.adapters.lora_rank import rank_from_adapter_config
from flash.serve.contract.protocol import reject_non_finite_json_constant


class AdapterConfigError(ValueError):
    """these ``adapter_config.json`` bytes cannot describe a servable LoRA adapter."""


@dataclass(frozen=True)
class DeclaredAdapterConfig:
    """what an adapter stamps about itself, after every deterministic rule has passed."""

    config: dict[str, Any]
    lora_rank: int
    base_model: str
    base_revision: str | None


class _DuplicateConfigKey(AdapterConfigError):
    """raised through ``json.loads``, so it stays a ValueError for any caller expecting one."""


def parse_declared_adapter_config(raw: bytes, *, source: str) -> DeclaredAdapterConfig:
    """read one adapter's declared provenance, or refuse the bytes.

    ``source`` names the file in every message, so a caller that fetched the config from the hub and
    one that read it off disk both report a location the reader can go look at.
    """

    config = _strict_object(raw, source=source)
    try:
        rank = rank_from_adapter_config(config, source=source)
    except ValueError as exc:
        raise AdapterConfigError(f"{source} declares no usable lora rank: {exc}") from exc

    peft_type = config.get("peft_type")
    if peft_type != "LORA":
        raise AdapterConfigError(f"{source} peft_type must be LORA, not {peft_type!r}")
    task_type = config.get("task_type")
    if task_type not in {None, "CAUSAL_LM"}:
        raise AdapterConfigError(
            f"{source} task_type must be absent or CAUSAL_LM, not {task_type!r}"
        )
    modules_to_save = config.get("modules_to_save")
    if modules_to_save is not None and modules_to_save != []:
        raise AdapterConfigError(f"{source} modules_to_save adapters are not supported")

    # the engine compares `base_model_name_or_path` for equality, so an absent, empty, or
    # non-string one can never match and the deployment is already doomed. skipping the check
    # instead deferred a certain failure until after a provider had allocated and started billing.
    base_model = _text(config, "base_model_name_or_path", source=source)
    if base_model is None:
        raise AdapterConfigError(f"{source} declares no base_model_name_or_path")
    revision = config.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise AdapterConfigError(f"{source} revision must be a string when present")
    return DeclaredAdapterConfig(
        config=config,
        lora_rank=rank,
        base_model=base_model,
        base_revision=_text(config, "revision", source=source),
    )


def _strict_object(raw: bytes, *, source: str) -> dict[str, Any]:
    """decode exactly the way the gpu-side materializer does, from the same bytes.

    handing the bytes straight to ``json.load`` instead lets it auto-detect utf-16 and accept a bom
    (rfc 4627), so a config the container refuses outright parsed cleanly at admission -- and the
    deployment failed only after the provider resources had been allocated. duplicate keys are the
    same case: plain decoding takes the last value, so a config declaring ``r`` twice would be
    admitted against one rank and then rejected inside the container this check exists to avoid
    paying for.
    """

    try:
        config = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_non_finite_json_constant,
        )
    except _DuplicateConfigKey as exc:
        raise AdapterConfigError(f"{source} contains a duplicate key") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise AdapterConfigError(f"{source} is not readable json") from exc
    if not isinstance(config, dict):
        raise AdapterConfigError(f"{source} must be a json object")
    return config


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateConfigKey(f"{key!r} appears twice")
        result[key] = value
    return result


def _text(config: dict[str, Any], key: str, *, source: str) -> str | None:
    """read a config string exactly as the engine will compare it.

    the engine compares these raw bytes for equality, so a value that matches only after stripping
    is admitted here and then rejected inside the paid container -- the outcome this reader exists
    to prevent. normalizing the padding away instead would be worse for ``revision``, which the
    resolver *adopts* into the immutable manifest: that would launder a padded string into the record
    rather than surface it. rejecting keeps the admission-time and container-time verdicts identical.
    """

    value = config.get(key)
    if not isinstance(value, str):
        return None
    if value != value.strip():
        raise AdapterConfigError(f"{source} {key} has surrounding whitespace: {value!r}")
    return value or None
