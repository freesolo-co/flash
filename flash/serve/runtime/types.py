"""import-light public types for the vllm serving runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import RuntimeConfigurationError
from .structured_outputs import normalize_structured_outputs

_RESERVED_MODEL_LOAD_KWARGS = frozenset({"token", "trust_remote_code"})
_RESERVED_ENGINE_ARGS = frozenset(
    {
        "model",
        "tokenizer",
        "trust_remote_code",
        "enable_lora",
        "max_loras",
        "max_lora_rank",
        "max_cpu_loras",
        "reasoning_parser",
        "limit_mm_per_prompt",
        "mm_processor_cache_gb",
        "enable_tower_connector_lora",
    }
)


def _nonempty(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeConfigurationError(f"{name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise RuntimeConfigurationError(f"{name} must not be empty")
    return cleaned


def _mapping_copy(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConfigurationError(f"{name} must be a mapping")
    return dict(value)


def _require_bool(value: Any, name: str, *, optional: bool = False) -> bool | None:
    if value is None and optional:
        return None
    if type(value) is not bool:
        suffix = " or none" if optional else ""
        raise RuntimeConfigurationError(f"{name} must be a boolean{suffix}")
    return value


def _require_int(value: Any, name: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise RuntimeConfigurationError(f"{name} must be a {qualifier} integer")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeConfigurationError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RuntimeConfigurationError(f"{name} must be a finite number")
    return normalized


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """all model-local mechanics needed to construct one vllm engine."""

    model: str
    served_model: str | None = None
    tokenizer_model: str | None = None
    hf_token: str | None = field(default=None, repr=False)
    trust_remote_code: bool = False
    max_loras: int = 16
    max_lora_rank: int = 64
    max_cpu_loras: int = 16
    pin_loras: bool | None = None
    image_limit: int | None = None
    prompt_cache_size: int = 128
    reasoning_parser: str | None = None
    liveness_interval_seconds: float = 5.0
    engine_args: Mapping[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _nonempty(self.model, "model"))
        for name in ("served_model", "tokenizer_model", "reasoning_parser", "hf_token"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        object.__setattr__(
            self,
            "trust_remote_code",
            _require_bool(self.trust_remote_code, "trust_remote_code"),
        )
        object.__setattr__(
            self,
            "pin_loras",
            _require_bool(self.pin_loras, "pin_loras", optional=True),
        )
        for name in ("max_loras", "max_lora_rank", "max_cpu_loras"):
            object.__setattr__(self, name, _require_int(getattr(self, name), name, minimum=1))
        if self.max_cpu_loras < self.max_loras:
            raise RuntimeConfigurationError("max_cpu_loras must be at least max_loras")
        if self.image_limit is not None:
            object.__setattr__(
                self,
                "image_limit",
                _require_int(self.image_limit, "image_limit", minimum=1),
            )
        object.__setattr__(
            self,
            "prompt_cache_size",
            _require_int(self.prompt_cache_size, "prompt_cache_size", minimum=0),
        )
        liveness_interval = _require_finite_number(
            self.liveness_interval_seconds,
            "liveness_interval_seconds",
        )
        if liveness_interval <= 0:
            raise RuntimeConfigurationError("liveness_interval_seconds must be positive")
        object.__setattr__(self, "liveness_interval_seconds", liveness_interval)
        engine_args = _mapping_copy(self.engine_args, "engine_args")
        reserved = sorted(_RESERVED_ENGINE_ARGS & engine_args.keys())
        if reserved:
            raise RuntimeConfigurationError(
                "engine_args cannot override runtime-owned keys: " + ", ".join(reserved)
            )
        object.__setattr__(self, "engine_args", engine_args)
        tokenizer_kwargs = _mapping_copy(self.tokenizer_kwargs, "tokenizer_kwargs")
        processor_kwargs = _mapping_copy(self.processor_kwargs, "processor_kwargs")
        for name, kwargs in (
            ("tokenizer_kwargs", tokenizer_kwargs),
            ("processor_kwargs", processor_kwargs),
        ):
            reserved_load = sorted(_RESERVED_MODEL_LOAD_KWARGS & kwargs.keys())
            if reserved_load:
                raise RuntimeConfigurationError(
                    f"{name} cannot override runtime-owned keys: " + ", ".join(reserved_load)
                )
        object.__setattr__(self, "tokenizer_kwargs", tokenizer_kwargs)
        object.__setattr__(self, "processor_kwargs", processor_kwargs)

    @property
    def effective_served_model(self) -> str:
        return self.served_model or self.model

    @property
    def effective_tokenizer_model(self) -> str:
        return self.tokenizer_model or self.model

    @property
    def effective_pin_loras(self) -> bool:
        if self.pin_loras is not None:
            return self.pin_loras
        return self.max_loras >= self.max_cpu_loras


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    """one exact local lora incarnation and its prompt defaults."""

    adapter_id: str
    path: str
    incarnation: str
    thinking: bool = False
    structured_outputs: Any = None
    pin: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "adapter_id", _nonempty(self.adapter_id, "adapter_id"))
        object.__setattr__(self, "path", _nonempty(self.path, "path"))
        object.__setattr__(self, "incarnation", _nonempty(self.incarnation, "incarnation"))
        object.__setattr__(
            self,
            "thinking",
            _require_bool(self.thinking, "adapter thinking"),
        )
        object.__setattr__(
            self,
            "pin",
            _require_bool(self.pin, "adapter pin", optional=True),
        )
        normalized = normalize_structured_outputs(self.structured_outputs) or None
        object.__setattr__(self, "structured_outputs", normalized)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """one generation against the base model or a registered adapter."""

    adapter_id: str | None = None
    expected_incarnation: str | None = None
    prompt: str | None = None
    messages: Sequence[Mapping[str, Any]] | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.95
    thinking: bool | None = None
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    structured_outputs: Any = None

    def __post_init__(self) -> None:
        if self.adapter_id is not None:
            object.__setattr__(self, "adapter_id", _nonempty(self.adapter_id, "adapter_id"))
        if self.expected_incarnation is not None:
            object.__setattr__(
                self,
                "expected_incarnation",
                _nonempty(self.expected_incarnation, "expected_incarnation"),
            )
        if self.adapter_id is None and self.expected_incarnation is not None:
            raise RuntimeConfigurationError("expected_incarnation requires a registered adapter_id")
        if self.adapter_id is not None and self.expected_incarnation is None:
            raise RuntimeConfigurationError(
                "registered adapter generation requires expected_incarnation"
            )
        has_prompt = self.prompt is not None
        has_messages = self.messages is not None
        if has_prompt == has_messages:
            raise RuntimeConfigurationError("exactly one of prompt or messages is required")
        if self.prompt is not None:
            object.__setattr__(self, "prompt", _nonempty(self.prompt, "prompt"))
        if self.messages is not None:
            if isinstance(self.messages, str) or not isinstance(self.messages, Sequence):
                raise RuntimeConfigurationError("messages must be a sequence of mappings")
            messages = tuple(dict(message) for message in self.messages)
            if not messages:
                raise RuntimeConfigurationError("messages must not be empty")
            object.__setattr__(self, "messages", messages)
        object.__setattr__(
            self,
            "max_tokens",
            _require_int(self.max_tokens, "max_tokens", minimum=1),
        )
        temperature = _require_finite_number(self.temperature, "temperature")
        if temperature < 0:
            raise RuntimeConfigurationError("temperature must be non-negative")
        object.__setattr__(self, "temperature", temperature)
        top_p = _require_finite_number(self.top_p, "top_p")
        if not 0 < top_p <= 1:
            raise RuntimeConfigurationError("top_p must be greater than zero and at most one")
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(
            self,
            "thinking",
            _require_bool(self.thinking, "thinking", optional=True),
        )
        object.__setattr__(
            self,
            "chat_template_kwargs",
            _mapping_copy(self.chat_template_kwargs, "chat_template_kwargs"),
        )
        object.__setattr__(
            self, "structured_outputs", normalize_structured_outputs(self.structured_outputs)
        )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """completed generation text, identity, accounting, and timing."""

    request_id: str
    runtime_id: str
    adapter_id: str | None
    incarnation: str | None
    text: str
    finish_reason: str | None
    token_ids: tuple[int, ...]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cached_tokens_reported: bool
    duration_seconds: float
    thinking: bool | None


@dataclass(frozen=True, slots=True)
class StreamReady:
    """the first engine output was obtained and streaming may begin."""

    request_id: str
    runtime_id: str
    adapter_id: str | None
    incarnation: str | None
    thinking: bool | None
    type: Literal["ready"] = field(default="ready", init=False)


@dataclass(frozen=True, slots=True)
class StreamDelta:
    """one normalized text delta."""

    text: str
    type: Literal["delta"] = field(default="delta", init=False)


@dataclass(frozen=True, slots=True)
class StreamFinished:
    """terminal stream accounting and normalized full text."""

    request_id: str
    runtime_id: str
    adapter_id: str | None
    incarnation: str | None
    text: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cached_tokens_reported: bool
    duration_seconds: float
    thinking: bool | None
    type: Literal["finished"] = field(default="finished", init=False)


StreamEvent = StreamReady | StreamDelta | StreamFinished


@dataclass(frozen=True, slots=True)
class RuntimeHealth:
    """process-local engine identity and liveness."""

    ok: bool
    started: bool
    engine_dead: bool
    runtime_id: str
    model: str
    served_model: str
    registered_adapters: int
    loaded_adapters: int
    prompt_cache_entries: int
