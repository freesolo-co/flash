"""LoRA adapter rank parsing and control-plane preflight helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from flash.catalog import serving_lora_rank_cap
from flash.spec import JobSpec

_OWNER_REPO_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_RUN_ID_RE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
_ADAPTER_STORAGE_REF_RE = re.compile(
    rf"^(?P<repo>{_OWNER_REPO_RE}/{_OWNER_REPO_RE}):(?P<phase>sft|rl)/"
    rf"(?P<run_id>{_RUN_ID_RE})(?P<checkpoint>/checkpoints/step-\d+)?$"
)
_VL_WARMSTART_RECOMBINE_PREFIXES = ("Qwen/Qwen3.5-", "Qwen/Qwen3.6-")

AdapterConfigLoader = Callable[[str, str | None], Mapping[str, Any]]


def resolve_adapter_ref(adapter_ref: str) -> tuple[str, str] | None:
    """Resolve an internal storage ref into ``(repo, artifact_prefix)``."""
    match = _ADAPTER_STORAGE_REF_RE.fullmatch((adapter_ref or "").strip())
    if not match:
        return None
    repo = match.group("repo")
    prefix = f"{match.group('phase')}/{match.group('run_id')}{match.group('checkpoint') or ''}"
    return repo, prefix


def adapter_config_path_from_ref(adapter_ref: str) -> tuple[str, str]:
    resolved = resolve_adapter_ref(adapter_ref)
    if resolved is None:
        raise ValueError(
            "train.init_from_adapter could not be resolved to an internal adapter storage ref"
        )
    repo, prefix = resolved
    return repo, f"{prefix}/adapter/adapter_config.json"


def _positive_int(value: Any, *, source: str, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"could not verify adapter rank: {source} has invalid rank metadata ({field})"
        )
    try:
        rank = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"could not verify adapter rank: {source} has invalid rank metadata ({field})"
        ) from exc
    if rank <= 0:
        raise ValueError(f"could not verify adapter rank: {source} has non-positive rank {rank}")
    return rank


def rank_from_adapter_config(config: Mapping[str, Any], *, source: str) -> int:
    """Return the max LoRA rank advertised by PEFT adapter metadata."""
    if not isinstance(config, Mapping):
        raise ValueError(f"could not verify adapter rank: {source} is not a JSON object")
    ranks: list[int] = []
    if config.get("r") is not None:
        ranks.append(_positive_int(config["r"], source=source, field="r"))
    if "rank_pattern" in config and config["rank_pattern"] is not None:
        rank_pattern = config["rank_pattern"]
        if not isinstance(rank_pattern, Mapping):
            raise ValueError(
                f"could not verify adapter rank: {source} has invalid rank metadata "
                "(rank_pattern)"
            )
        ranks.extend(
            _positive_int(v, source=source, field="rank_pattern") for v in rank_pattern.values()
        )
    if not ranks:
        raise ValueError(f"could not verify adapter rank: {source} has no LoRA rank metadata")
    return max(ranks)


def uniform_rank_from_adapter_config(config: Mapping[str, Any], *, source: str) -> int:
    """Return a uniform adapter rank for the VL SFT+GRPO recombine path."""
    if not isinstance(config, Mapping):
        raise ValueError(f"adapter rank preflight: {source} is not a JSON object")
    if config.get("r") is None:
        raise ValueError(
            f"adapter rank preflight: {source} must contain a positive integer `r`"
        )
    rank = _positive_int(config["r"], source=source, field="r")
    for key in ("rank_pattern", "alpha_pattern"):
        if config.get(key):
            raise ValueError(
                f"adapter rank preflight: {source} has non-empty {key}={config[key]!r}; "
                "VL warm-start recombine requires a uniform LoRA rank/alpha"
            )
    return rank


def load_hf_adapter_config(adapter_ref: str, token: str | None = None) -> Mapping[str, Any]:
    """Read ``adapter_config.json`` for a Flash adapter ref from Hugging Face datasets."""
    repo, filename = adapter_config_path_from_ref(adapter_ref)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - package extra is present in supported installs
        raise ValueError(
            "could not verify train.init_from_adapter rank: huggingface_hub is not installed"
        ) from exc
    try:
        local = hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        raise ValueError(
            f"could not verify train.init_from_adapter rank: failed to read {repo}:{filename}"
        ) from exc
    try:
        with open(local, encoding="utf-8") as f:
            config = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"could not verify train.init_from_adapter rank: invalid JSON in {repo}:{filename}"
        ) from exc
    if not isinstance(config, Mapping):
        raise ValueError(
            f"could not verify train.init_from_adapter rank: {repo}:{filename} "
            "is not a JSON object"
        )
    return config


def _uses_vl_warmstart_recombine(model: str) -> bool:
    return model.startswith(_VL_WARMSTART_RECOMBINE_PREFIXES)


def preflight_init_adapter_lora_rank(
    spec: JobSpec,
    *,
    token: str | None = None,
    config_loader: AdapterConfigLoader | None = None,
) -> None:
    """Reject warm-start adapters that cannot fit the model's serving LoRA rank cap.

    The control plane calls this before creating/submitting a run. It intentionally uses only
    ``adapter_config.json`` so the preflight stays CPU-only and catches undeployable adapters before
    any training GPU is allocated.
    """
    adapter_storage_ref = (spec.train.init_from_adapter or "").strip()
    if not adapter_storage_ref:
        return
    max_lora_rank = serving_lora_rank_cap(spec.model)
    if max_lora_rank is None:
        return

    repo, filename = adapter_config_path_from_ref(adapter_storage_ref)
    source = f"{repo}:{filename}"
    loader = config_loader or load_hf_adapter_config
    config = loader(adapter_storage_ref, token)

    adapter_rank = rank_from_adapter_config(config, source=source)
    if adapter_rank > max_lora_rank:
        raise ValueError(
            f"train.init_from_adapter={adapter_storage_ref!r} has rank {adapter_rank}, exceeding "
            f"{spec.model}'s serving max_lora_rank={max_lora_rank}; use a lower-rank adapter "
            "or raise the serving cap after real-GPU validation"
        )

    if spec.algorithm != "grpo" or not _uses_vl_warmstart_recombine(spec.model):
        return

    sft_rank = uniform_rank_from_adapter_config(config, source=source)
    grpo_rank = int(spec.train.lora_rank)
    recombined_rank = sft_rank + grpo_rank
    if recombined_rank <= max_lora_rank:
        return

    allowed_grpo_rank = max_lora_rank - sft_rank
    if allowed_grpo_rank >= 1:
        guidance = f"set GRPO train.lora_rank <= {allowed_grpo_rank}"
    else:
        allowed_sft_rank = max_lora_rank - grpo_rank
        if allowed_sft_rank >= 1:
            guidance = f"retrain the SFT adapter at rank <= {allowed_sft_rank}"
        else:
            guidance = f"lower both SFT and GRPO ranks so their sum is <= {max_lora_rank}"
    raise ValueError(
        "train.init_from_adapter rank preflight failed: recombined SFT+GRPO adapter would be "
        f"rank {recombined_rank} (SFT rank {sft_rank} + GRPO rank {grpo_rank}), exceeding "
        f"{spec.model}'s serving max_lora_rank={max_lora_rank}; {guidance}"
    )
