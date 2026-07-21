"""Single-process frozen-base residency for sequential SFT jobs."""

from __future__ import annotations

import gc
import hashlib
from collections.abc import Callable, Hashable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResidentCompatibilityKey:
    """All state that can change the in-memory base model representation."""

    model_id: str
    revision: str
    loader_kind: str
    dtype: Hashable
    torch_dtype: Hashable
    attn_implementation: Hashable
    device_map: Hashable
    trust_remote_code: bool
    gpu_arch: Hashable
    context_length: int | None
    load_flags: tuple[tuple[str, Hashable], ...]


@dataclass
class ResidentJobState:
    """References retained until the resident loop has reset a completed job."""

    model: Any
    trainer: Any | None = None
    dataset: Any | None = None


def _stable_value(value: Any) -> Hashable:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, dict):
        return tuple(sorted((str(key), _stable_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_stable_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_stable_value(item) for item in value), key=repr))
    return repr(value)


def _gpu_arch() -> Hashable:
    try:
        import torch

        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    return None


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _base_parameter_items(model: Any) -> list[tuple[str, Any]]:
    items = []
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if any(
            marker in lowered
            for marker in ("lora_", "adapter_", "modules_to_save", "prompt_encoder")
        ):
            continue
        items.append((name, parameter))
    return sorted(items, key=lambda item: item[0])


def _sample_parameter_bytes(parameter: Any, sample_size: int = 64) -> bytes:
    custom = getattr(parameter, "_resident_fingerprint_bytes", None)
    if callable(custom):
        return bytes(custom())

    tensor = parameter.detach() if hasattr(parameter, "detach") else parameter
    if hasattr(tensor, "reshape"):
        tensor = tensor.reshape(-1)
    count = int(tensor.numel()) if hasattr(tensor, "numel") else len(tensor)
    if count > sample_size * 2:
        first = tensor[:sample_size]
        last = tensor[-sample_size:]
        try:
            import torch

            tensor = torch.cat((first, last))
        except Exception:
            return repr((first, last)).encode()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()
    if hasattr(tensor, "numpy"):
        return tensor.numpy().tobytes()
    if hasattr(tensor, "tobytes"):
        return tensor.tobytes()
    return repr(tensor).encode()


def base_fingerprint(model: Any, *, parameter_count: int = 16) -> str:
    """Hash deterministic samples of frozen, non-adapter parameters."""

    parameters = _base_parameter_items(model)
    if not parameters:
        raise ValueError("cannot fingerprint a model with no base parameters")
    if len(parameters) > parameter_count:
        indexes = {
            round(index * (len(parameters) - 1) / (parameter_count - 1))
            for index in range(parameter_count)
        }
        parameters = [parameters[index] for index in sorted(indexes)]

    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(str(getattr(parameter, "shape", "")).encode())
        digest.update(str(getattr(parameter, "dtype", "")).encode())
        digest.update(_sample_parameter_bytes(parameter))
    return digest.hexdigest()


def _adapter_names(model: Any) -> tuple[str, ...]:
    configs = getattr(model, "peft_config", None)
    if isinstance(configs, dict):
        return tuple(str(name) for name in configs)
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    if isinstance(active, str):
        return (active,)
    if isinstance(active, (list, tuple, set)):
        return tuple(str(name) for name in active)
    return ()


def _training_state_targets(model: Any) -> list[Any]:
    targets = []
    pending = [model]
    seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        targets.append(current)
        get_base_model = getattr(current, "get_base_model", None)
        if callable(get_base_model):
            with suppress(Exception):
                pending.append(get_base_model())
        for name in ("base_model", "model"):
            with suppress(Exception):
                pending.append(getattr(current, name, None))
    return targets


def _is_input_require_grads_hook(hook: Any) -> bool:
    callback = getattr(hook, "func", hook)
    return getattr(
        callback, "__name__", ""
    ) == "make_inputs_require_grads" and "enable_input_require_grads" in getattr(
        callback, "__qualname__", ""
    )


def _disable_training_state(model: Any) -> None:
    for target in _training_state_targets(model):
        disable_checkpointing = getattr(target, "gradient_checkpointing_disable", None)
        if callable(disable_checkpointing):
            with suppress(Exception):
                disable_checkpointing()

        disable_input_grads = getattr(target, "disable_input_require_grads", None)
        if callable(disable_input_grads):
            with suppress(Exception):
                disable_input_grads()

        handles = list(getattr(target, "_require_grads_hooks", None) or ())
        legacy_handle = getattr(target, "_require_grads_hook", None)
        if legacy_handle is not None:
            handles.append(legacy_handle)
        for handle in handles:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                with suppress(Exception):
                    remove()
        if hasattr(target, "_require_grads_hooks"):
            target._require_grads_hooks = []
        if hasattr(target, "_require_grads_hook"):
            with suppress(Exception):
                del target._require_grads_hook

        get_input_embeddings = getattr(target, "get_input_embeddings", None)
        if not callable(get_input_embeddings):
            continue
        with suppress(Exception):
            embeddings = get_input_embeddings()
            forward_hooks = getattr(embeddings, "_forward_hooks", None)
            if hasattr(forward_hooks, "items"):
                for hook_id, hook in list(forward_hooks.items()):
                    if _is_input_require_grads_hook(hook):
                        forward_hooks.pop(hook_id, None)


def _drop_job_references(trainer: Any | None, dataset: Any | None) -> None:
    if trainer is not None:
        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is not None:
            for name in ("_models", "_optimizers", "_schedulers", "_dataloaders"):
                references = getattr(accelerator, name, None)
                if hasattr(references, "clear"):
                    with suppress(Exception):
                        references.clear()
        for name in (
            "optimizer",
            "lr_scheduler",
            "train_dataset",
            "eval_dataset",
            "data_collator",
            "model",
            "model_wrapped",
            "callback_handler",
            "accelerator",
        ):
            if hasattr(trainer, name):
                with suppress(Exception):
                    setattr(trainer, name, None)
    if dataset is not None:
        cleanup = getattr(dataset, "cleanup_cache_files", None)
        if callable(cleanup):
            with suppress(Exception):
                cleanup()


def reset_after_job(
    model: Any,
    *,
    trainer: Any | None = None,
    dataset: Any | None = None,
) -> Any:
    """Unload all PEFT adapters and release completed-job state, returning the raw base."""

    _disable_training_state(model)
    cleaned = model
    unload = getattr(model, "unload", None)
    if callable(unload):
        unloaded = unload()
        if unloaded is not None:
            cleaned = unloaded
    else:
        delete_adapter = getattr(model, "delete_adapter", None)
        if callable(delete_adapter):
            for adapter_name in _adapter_names(model):
                delete_adapter(adapter_name)
        get_base_model = getattr(model, "get_base_model", None)
        if callable(get_base_model):
            cleaned = get_base_model()
    _disable_training_state(cleaned)

    for _name, parameter in cleaned.named_parameters():
        if hasattr(parameter, "requires_grad"):
            parameter.requires_grad = False
    eval_model = getattr(cleaned, "eval", None)
    if callable(eval_model):
        eval_model()

    _drop_job_references(trainer, dataset)
    gc.collect()
    _empty_cuda_cache()
    return cleaned


class ResidentBase:
    """Cache one frozen base model and tokenizer under a strict compatibility key."""

    def __init__(
        self,
        *,
        model_loader: Callable[[str, str, dict[str, Any]], Any] | None = None,
        tokenizer_loader: Callable[[str, str], Any] | None = None,
        gpu_arch: Callable[[], Hashable] = _gpu_arch,
        empty_cuda_cache: Callable[[], None] = _empty_cuda_cache,
    ) -> None:
        self._model_loader = model_loader or self._load_model
        self._tokenizer_loader = tokenizer_loader or self._load_tokenizer
        self._gpu_arch = gpu_arch
        self._empty_cuda_cache = empty_cuda_cache
        self._key: ResidentCompatibilityKey | None = None
        self._tokenizer_key: tuple[str, str] | None = None
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._fingerprint: str | None = None

    @staticmethod
    def _load_tokenizer(model_id: str, revision: str) -> Any:
        from flash.engine.worker.hf import load_tokenizer

        return load_tokenizer(model_id, revision=revision)

    @staticmethod
    def _load_model(model_id: str, loader_kind: str, kwargs: dict[str, Any]) -> Any:
        if loader_kind == "image-text":
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText.from_pretrained(
                model_id, trust_remote_code=True, **kwargs
            )

        # trl 1.6 uses this helper for the default sft string path, including
        # config.architectures[0] resolution, dtype conversion, and kwargs forwarding.
        from trl.trainer.utils import create_model_from_path

        return create_model_from_path(model_id, **kwargs)

    @staticmethod
    def _loader_kind(model_id: str, revision: str) -> str:
        from flash.engine.worker.lora import is_vl_checkpoint

        return "image-text" if is_vl_checkpoint(model_id, revision=revision) else "causal-lm"

    def get_tokenizer(self, model_id: str, revision: str = "") -> Any:
        tokenizer_key = (model_id, revision)
        if self._tokenizer is not None and self._tokenizer_key == tokenizer_key:
            return self._tokenizer
        if self._model is not None:
            self._evict()
        self._tokenizer = self._tokenizer_loader(model_id, revision)
        self._tokenizer_key = tokenizer_key
        return self._tokenizer

    def _compatibility_key(
        self,
        model_id: str,
        revision: str,
        model_init_kwargs: dict[str, Any],
        context_length: int | None,
        loader_kind: str,
    ) -> ResidentCompatibilityKey:
        normalized = {key: _stable_value(value) for key, value in sorted(model_init_kwargs.items())}
        recognized = {
            "dtype",
            "torch_dtype",
            "attn_implementation",
            "device_map",
            "trust_remote_code",
            "revision",
        }
        load_flags = tuple(
            (key, value) for key, value in normalized.items() if key not in recognized
        )
        return ResidentCompatibilityKey(
            model_id=model_id,
            revision=revision,
            loader_kind=loader_kind,
            dtype=normalized.get("dtype"),
            torch_dtype=normalized.get("torch_dtype"),
            attn_implementation=normalized.get("attn_implementation"),
            device_map=normalized.get("device_map"),
            trust_remote_code=bool(model_init_kwargs.get("trust_remote_code", False)),
            gpu_arch=_stable_value(self._gpu_arch()),
            context_length=context_length,
            load_flags=load_flags,
        )

    def get_base(
        self,
        model_id: str,
        revision: str = "",
        *,
        model_init_kwargs: dict[str, Any] | None = None,
        context_length: int | None = None,
        loader_kind: str | None = None,
    ) -> tuple[Any, Any]:
        """Return the compatible cached base and tokenizer, loading on a miss."""

        kwargs = dict(model_init_kwargs or {})
        kwargs.setdefault("dtype", "bfloat16")
        kwargs.setdefault("device_map", None)
        kw_revision = str(kwargs.get("revision") or "")
        if kw_revision and revision and kw_revision != revision:
            raise ValueError("resident model revision does not match model_init_kwargs")
        effective_revision = revision or kw_revision
        if effective_revision:
            kwargs["revision"] = effective_revision
        resolved_loader = loader_kind or self._loader_kind(model_id, effective_revision)
        key = self._compatibility_key(
            model_id,
            effective_revision,
            kwargs,
            context_length,
            resolved_loader,
        )
        if self._model is not None and self._key == key:
            return self._model, self.get_tokenizer(model_id, effective_revision)

        preserve_tokenizer = self._tokenizer_key == (model_id, effective_revision)
        self._evict(preserve_tokenizer=preserve_tokenizer)
        tokenizer = self.get_tokenizer(model_id, effective_revision)
        model = self._model_loader(model_id, resolved_loader, kwargs)
        for _name, parameter in model.named_parameters():
            if hasattr(parameter, "requires_grad"):
                parameter.requires_grad = False
        eval_model = getattr(model, "eval", None)
        if callable(eval_model):
            eval_model()
        self._model = model
        self._tokenizer = tokenizer
        self._tokenizer_key = (model_id, effective_revision)
        self._key = key
        self._fingerprint = base_fingerprint(model)
        return model, tokenizer

    def reset_after_job(
        self,
        model: Any,
        *,
        trainer: Any | None = None,
        dataset: Any | None = None,
    ) -> None:
        """Restore and verify the cached base after a successful resident SFT job."""

        if self._model is None or self._fingerprint is None:
            raise RuntimeError("resident base is not loaded")
        cleaned = reset_after_job(model, trainer=trainer, dataset=dataset)
        if cleaned is not self._model:
            self._model = cleaned
        actual = base_fingerprint(self._model)
        if actual != self._fingerprint:
            self._evict()
            raise RuntimeError("resident base parameters changed during SFT")

    def _evict(self, *, preserve_tokenizer: bool = False) -> None:
        self._model = None
        self._key = None
        self._fingerprint = None
        if not preserve_tokenizer:
            self._tokenizer = None
            self._tokenizer_key = None
        gc.collect()
        self._empty_cuda_cache()
