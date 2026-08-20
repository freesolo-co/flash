"""Adapter-cache paths and engine-arg introspection for the LoRA engine.

Pure helpers split out of lora_engine.py: filesystem layout for the adapter cache, request-output
token accounting, and the vLLM AsyncEngineArgs compatibility probes. None of them touch engine
state, so they are testable without constructing an engine.
"""

import hashlib
import inspect
import re
from pathlib import Path
from typing import Any

_RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(
    {
        "tokenize",
        "add_generation_prompt",
        "return_dict",
        "return_tensors",
        "return_assistant_tokens_mask",
        "conversation",
        "documents",
        "chat_template",
        "padding",
        "truncation",
        "max_length",
    }
)


def _safe_chat_template_kwargs(raw: Any) -> dict[str, Any]:
    """Caller-supplied chat_template_kwargs, sanitized for apply_chat_template.

    Returns {} for a non-dict value (defends against malformed/untrusted requests) and drops the
    keys we pass explicitly or that would change the return shape. Using the *same* sanitized view
    to both render the prompt and key the prompt-token cache keeps genuine template controls distinct
    while never splitting identical rendered prompts into separate cache entries over reserved/
    ignored keys. ``enable_thinking`` is normalized later from the adapter record.
    """
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if k not in _RESERVED_CHAT_TEMPLATE_KWARGS}


def _load_adapters_for_base(settings: Any, base_model: str) -> list[Any]:
    from flash.serving.src.persistence import load_adapters

    try:
        return [
            adapter
            for adapter in load_adapters(settings)
            if adapter.base_model == base_model
            and adapter.status == "ready"
            and adapter.is_revision
        ]
    except Exception as exc:  # forwarded records still allow request-time serving
        print(
            f"adapter hydration skipped for base_model={base_model!r}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return []


def _adapter_source_ident(record: Any) -> tuple[str, str, str, str | None]:
    return (
        record.repo_id,
        getattr(record, "repo_type", "model") or "model",
        record.hf_revision or "",
        getattr(record, "subfolder", None),
    )


def _adapter_source_cache_dir(root: Path, record: Any) -> Path:
    repo_id, repo_type, hf_revision, subfolder = _adapter_source_ident(record)
    raw = "\0".join((repo_id, repo_type, hf_revision, subfolder or ""))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    slug = "".join(c if c.isalnum() or c in "-_." else "-" for c in repo_id)[:80].strip("-")
    return root / "sources" / f"{slug or 'adapter'}-{digest}"


def _assert_source_cache_containment(root: Path, path: Path) -> Path:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("adapter subfolder escapes its exact-SHA source cache") from exc
    return path


def _adapter_cache_path(root: Path, subfolder: str | None) -> Path:
    path = root / subfolder if subfolder else root
    return _assert_source_cache_containment(root, path)


def _is_adapter_tensor_file(path: Path) -> bool:
    name = path.name
    return name in {"adapter_model.safetensors", "adapter_model.bin"} or (
        name.startswith("adapter_model-") and name.endswith((".safetensors", ".bin"))
    )


def _adapter_cache_ready(path: Path) -> bool:
    if not (path / "adapter_config.json").is_file():
        return False
    try:
        return any(
            child.is_file() and _is_adapter_tensor_file(child) and child.stat().st_size > 0
            for child in path.iterdir()
        )
    except OSError:
        return False


def _stream_text_delta(
    text: str, previous_text: str, *, cumulative_output: bool | None
) -> tuple[str, str]:
    if not text:
        return "", previous_text
    if cumulative_output is not False and text.startswith(previous_text):
        return text[len(previous_text) :], text
    return text, previous_text + text


def _num_prompt_tokens(request_output: Any) -> int:
    prompt_token_ids = getattr(request_output, "prompt_token_ids", None)
    if prompt_token_ids:
        return len(prompt_token_ids)
    value = getattr(request_output, "num_prompt_tokens", None)
    if value is None:
        if prompt_token_ids is not None:
            return len(prompt_token_ids)
        raise RuntimeError("vLLM did not report the expanded prompt token count")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("vLLM reported an invalid expanded prompt token count") from exc
    if count < 0:
        raise RuntimeError("vLLM reported a negative expanded prompt token count")
    return count


def _num_cached_tokens(request_output: Any) -> int:
    """Prefix-cached prompt tokens for a finished request (vLLM ``RequestOutput.num_cached_tokens``).

    With prefix caching on, vLLM reports how many of the prompt's tokens were served from the KV
    cache (prefill skipped) — the count the backend bills at a discount. Returns 0 when the
    attribute is absent (an engine that doesn't expose it) or None (not computed), so it is always
    a safe non-negative int.
    """
    value = getattr(request_output, "num_cached_tokens", None)
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _cached_tokens_reported(request_output: Any) -> bool:
    """Whether the engine returned a valid cached-token measurement, including an explicit zero."""
    value = getattr(request_output, "num_cached_tokens", None)
    if value is None:
        return False
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def _async_engine_arg_names(async_engine_args_type: Any) -> set[str]:
    import dataclasses

    try:
        return {field.name for field in dataclasses.fields(async_engine_args_type)}
    except TypeError:
        try:
            return set(inspect.signature(async_engine_args_type).parameters)
        except (TypeError, ValueError):
            return set()


def _require_reasoning_api_compatibility(
    async_engine_args_type: Any, generate: Any, reasoning_parser: str | None
) -> None:
    """Fail closed when a configured parser cannot receive request-local reasoning state."""
    if reasoning_parser is None:
        return
    engine_args = _async_engine_arg_names(async_engine_args_type)
    try:
        generate_args = inspect.signature(generate).parameters
    except (TypeError, ValueError):
        generate_args = {}
    missing = [
        name
        for name, available in (
            ("reasoning_parser", "reasoning_parser" in engine_args),
            ("reasoning_ended", "reasoning_ended" in generate_args),
            ("reasoning_parser_kwargs", "reasoning_parser_kwargs" in generate_args),
        )
        if not available
    ]
    if missing:
        raise RuntimeError("vLLM reasoning API is incompatible; missing " + ", ".join(missing))


def _engine_is_dead(engine: Any) -> bool:
    """True once vLLM's EngineCore worker process has died. In V1 the engine is an ``AsyncLLM``
    (``AsyncLLMEngine`` aliases it) whose ``errored`` property is
    ``engine_core.resources.engine_dead or not is_running``; once set the engine is unrecoverable
    in-process, so the only fix is to replace the container. Accepts ``None`` (engine not built yet)
    so it is safe to call before ``_load`` completes."""
    return bool(engine is not None and getattr(engine, "errored", False))



def active_checkpoint_ref(record: Any) -> str:
    """Which checkpoint an adapter record is actually serving.

    Prefers the record's explicit checkpoint; otherwise recovers `step-N` from a
    `checkpoints/step-N` subfolder so a step-pinned deploy is identifiable from the path alone.
    """
    checkpoint = str(getattr(record, "checkpoint", "") or "").strip()
    if checkpoint:
        return checkpoint
    subfolder = str(getattr(record, "subfolder", "") or "").strip().strip("/")
    if not subfolder:
        return ""
    match = re.search(r"(?:^|/)checkpoints/(step-\d+)(?:/|$)", subfolder)
    if match:
        return f"{record.adapter_id}/{match.group(1)}"
    return str(record.adapter_id)


def enforce_expected_checkpoint(record: Any, expected_checkpoint: str | None) -> str:
    """Fail closed when the served checkpoint is not the one the caller pinned.

    A concurrent deploy can replace an adapter between resolution and generation; serving the
    wrong step silently would attribute one checkpoint's output to another.
    """
    active_checkpoint = active_checkpoint_ref(record)
    if expected_checkpoint is not None and expected_checkpoint.strip() != active_checkpoint:
        expected = expected_checkpoint.strip()
        raise ValueError(
            "checkpoint mismatch: "
            f"adapter {record.adapter_id} is serving checkpoint "
            f"{active_checkpoint or '<none>'}, not the expected "
            f"{expected or '<none>'}; a concurrent deploy likely replaced it. "
            "Re-deploy the intended step or drop the expectation."
        )
    return active_checkpoint
