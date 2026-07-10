"""LoRA adapter rank parsing and control-plane preflight helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from flash.catalog import serving_lora_rank_cap

if TYPE_CHECKING:
    from flash.spec import JobSpec

AdapterConfigLoader = Callable[[str, str | None], Mapping[str, Any]]


def resolve_adapter_ref(adapter_ref: str) -> tuple[str, str] | None:
    """Resolve an internal storage ref into ``(repo, artifact_prefix)``."""
    from flash.schema import parse_adapter_storage_ref

    return parse_adapter_storage_ref(adapter_ref)


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
                f"could not verify adapter rank: {source} has invalid rank metadata (rank_pattern)"
            )
        ranks.extend(
            _positive_int(v, source=source, field="rank_pattern") for v in rank_pattern.values()
        )
    if not ranks:
        raise ValueError(f"could not verify adapter rank: {source} has no LoRA rank metadata")
    return max(ranks)


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
            f"could not verify train.init_from_adapter rank: {repo}:{filename} is not a JSON object"
        )
    return config


def preflight_init_adapter_lora_rank(
    spec: JobSpec,
    *,
    token: str | None = None,
    config_loader: AdapterConfigLoader | None = None,
) -> None:
    """Reject a warm-start adapter the model can't serve, or that mismatches ``train.lora_rank``.

    Warm-start CONTINUES the prior adapter in place (SFT→GRPO/OPD keep the one LoRA, rank unchanged),
    so the run trains and serves at the SOURCE adapter's rank — ``train.lora_rank`` is ignored once
    ``init_from_adapter`` is set (the trainer keeps the loaded LoRA; see
    ``flash.engine.worker.adapter._init_adapter_model``). That rank must therefore (a) fit the serving
    cap — for cataloged serving models — and (b) equal ``spec.train.lora_rank`` for EVERY warm-start,
    because the cost model, allocator, and GRPO sleep-mode sizing all read ``spec.train.lora_rank`` — a
    mismatch (e.g. a rank-64 source with the default ``lora_rank=32``) is quoted/placed for the wrong
    rank and can then OOM or fail to load at the true rank, even on an uncataloged model that has no
    cap. The control plane calls this before creating/submitting a run; it intentionally uses only
    ``adapter_config.json`` so the preflight stays CPU-only (no GPU) and catches an undeployable or
    mis-sized adapter before any training GPU is allocated.
    """
    adapter_storage_ref = (spec.train.init_from_adapter or "").strip()
    if not adapter_storage_ref:
        return

    repo, filename = adapter_config_path_from_ref(adapter_storage_ref)
    source = f"{repo}:{filename}"
    loader = config_loader or load_hf_adapter_config
    config = loader(adapter_storage_ref, token)
    adapter_rank = rank_from_adapter_config(config, source=source)

    # The serving cap only bounds cataloged serving models; open-policy / uncataloged models have no
    # cap and skip THIS check (but not the sizing-mismatch check below, which is cap-independent).
    max_lora_rank = serving_lora_rank_cap(spec.model)
    if max_lora_rank is not None and adapter_rank > max_lora_rank:
        raise ValueError(
            f"train.init_from_adapter={adapter_storage_ref!r} has rank {adapter_rank}, exceeding "
            f"{spec.model}'s serving max_lora_rank={max_lora_rank}; use a lower-rank adapter "
            "or raise the serving cap after real-GPU validation"
        )

    # A continued warm-start keeps the SOURCE adapter, so it trains and serves at ``adapter_rank`` no
    # matter what ``train.lora_rank`` says. But cost/allocator/GRPO-sleep sizing all read
    # ``spec.train.lora_rank``; if it disagrees with the source rank the run is quoted and placed for
    # the wrong rank (a rank-64 source with the default lora_rank=32 is sized as 32) and then OOMs or
    # fails to load the rank-64 adapter. This mis-sizing is cap-independent — an uncataloged model is
    # quoted, billed, and placed at the wrong rank too — so the check runs for every warm-start, not
    # just capped ones. Reject the mismatch so every sizing consumer agrees on the one true rank once
    # the config is corrected.
    configured_rank = int(spec.train.lora_rank)
    if configured_rank != adapter_rank:
        raise ValueError(
            f"train.lora_rank={configured_rank} does not match the continued warm-start adapter's "
            f"rank {adapter_rank} (train.init_from_adapter={adapter_storage_ref!r}). A warm-start "
            f"CONTINUES the source adapter in place, so this run trains and serves at rank "
            f"{adapter_rank} regardless of train.lora_rank — but cost, GPU allocation, and GRPO "
            f"sleep-mode sizing all read train.lora_rank, so a mismatch is mis-quoted and mis-placed. "
            f"Set train.lora_rank={adapter_rank} to match the source adapter (or warm-start from a "
            f"rank-{configured_rank} adapter)."
        )


def preflight_train_context_within_serving(spec: JobSpec) -> None:
    """Reject a run whose training context exceeds the model's serving ``max_model_len``.

    A LoRA is served at the model's fixed serving context; training it at a LONGER sequence wastes
    compute and learns positions inference never uses. The control plane calls this before submitting
    a run — CPU-only (catalog lookup, no GPU, no network), like the rank preflight above.
    Open-policy / uncataloged models have no serving cap and are skipped.

    SFT training context is ``train.max_context_tokens``; GRPO is the rollout prompt+completion
    length (``grpo_rollout_seq_len``, which folds in ``train.max_completion_tokens`` and the recipe
    defaults). An unset SFT ``max_context_tokens`` uses the worker's small recipe default (always
    within the cap) and is skipped.
    """
    from flash.catalog import serving_context_cap
    from flash.engine.vram import grpo_rollout_seq_len

    cap = serving_context_cap(spec.model)
    if cap is None:
        return

    if spec.algorithm == "grpo":
        effective = grpo_rollout_seq_len(
            spec.train.max_context_tokens or 0, spec.train.max_completion_tokens, spec.thinking
        )
        knob = "train.max_context_tokens / train.max_completion_tokens (GRPO rollout prompt+completion)"
    else:
        effective = int(spec.train.max_context_tokens or 0)
        if effective <= 0:  # unset -> worker recipe default, always within the cap
            return
        knob = "train.max_context_tokens"

    if effective > cap:
        raise ValueError(
            f"{knob}={effective} exceeds {spec.model}'s serving max_model_len={cap}: a LoRA trained "
            f"at a longer context than it is served wastes compute and learns positions never used "
            f"at inference. Lower it to <= {cap}, or raise the serving context after real-GPU "
            f"validation."
        )
