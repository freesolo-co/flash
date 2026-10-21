"""import-light public types for the vllm serving runtime."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from flash.serve.request.tool_calls import (
    FunctionTool,
    normalize_tools,
    tools_active,
    tools_wire,
    validate_tool_control_presence,
    validate_tool_history_replay,
    validate_tool_stop_sequences,
)
from flash.serve.request.validation import (
    MAX_SOURCE_CHARS,
    detached_messages,
    has_image_blocks,
    normalize_messages,
)

from .errors import RuntimeConfigurationError
from .sampling import (
    validate_choice_count,
    validate_logprobs,
    validate_penalty,
    validate_sampling_relationships,
    validate_seed,
    validate_top_logprobs,
)
from .structured_outputs import normalize_structured_outputs
from .tool_calls import ParsedToolCall

_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_RESERVED_MODEL_LOAD_KWARGS = frozenset({"revision", "token", "trust_remote_code"})
_RESERVED_ENGINE_ARGS = frozenset(
    {
        "model",
        "revision",
        "tokenizer",
        "tokenizer_revision",
        "trust_remote_code",
        "enable_lora",
        "max_loras",
        "max_lora_rank",
        "max_cpu_loras",
        "reasoning_parser",
        "tool_parser",
        "enable_auto_tool_choice",
        "limit_mm_per_prompt",
        "mm_processor_cache_gb",
        "enable_tower_connector_lora",
    }
)


class _FrozenList(tuple):
    __slots__ = ()


class _FrozenTuple(tuple):
    __slots__ = ()


class _FrozenSet(frozenset):
    __slots__ = ()


class _FrozenFrozenSet(frozenset):
    __slots__ = ()


def _freeze_value(value: Any, name: str, active: set[int]) -> Any:
    if isinstance(value, Mapping | list | tuple | set | frozenset):
        identity = id(value)
        if identity in active:
            raise RuntimeConfigurationError(f"{name} must not contain recursive containers")
        active.add(identity)
        try:
            if isinstance(value, Mapping):
                frozen: dict[str, Any] = {}
                for key, nested in value.items():
                    if type(key) is not str:
                        raise RuntimeConfigurationError(f"{name} keys must be strings")
                    frozen[key] = _freeze_value(nested, name, active)
                return MappingProxyType(frozen)
            if isinstance(value, list):
                return _FrozenList(_freeze_value(nested, name, active) for nested in value)
            if isinstance(value, tuple):
                return _FrozenTuple(_freeze_value(nested, name, active) for nested in value)
            if isinstance(value, set):
                return _FrozenSet(_freeze_value(nested, name, active) for nested in value)
            return _FrozenFrozenSet(_freeze_value(nested, name, active) for nested in value)
        finally:
            active.remove(identity)
    if type(value) in {bool, float, int, str, type(None)}:
        return value
    raise RuntimeConfigurationError(
        f"{name} values must use immutable scalar or supported built-in container types"
    )


def _thaw_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_value(nested, name) for key, nested in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw_value(nested, name) for nested in value]
    if isinstance(value, _FrozenTuple):
        return tuple(_thaw_value(nested, name) for nested in value)
    if isinstance(value, _FrozenSet):
        return {_thaw_value(nested, name) for nested in value}
    if isinstance(value, _FrozenFrozenSet):
        return frozenset(_thaw_value(nested, name) for nested in value)
    if type(value) in {bool, float, int, str, type(None)}:
        return value
    raise RuntimeConfigurationError(
        f"{name} values must use immutable scalar or supported built-in container types"
    )


def thaw_mapping(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    """return a detached mutable copy for one third-party call boundary."""

    return {key: _thaw_value(nested, name) for key, nested in value.items()}


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


def _frozen_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeConfigurationError(f"{name} must be a mapping")
    frozen = _freeze_value(value, name, set())
    assert isinstance(frozen, MappingProxyType)
    return frozen


def _revision(value: object, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _REVISION_RE.fullmatch(value) is None:
        raise RuntimeConfigurationError(
            f"{name} must be an exact 40-character lowercase hex revision"
        )
    return value


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


def _stop_sequence(value: Any, name: str) -> str:
    # deliberately not `_nonempty`: that trims, which is right for identifiers but destroys stop
    # sequences. "\n\n" is a common delimiter and would strip to empty and be rejected, and
    # " END" would silently become "END" -- a different sequence than the caller asked to stop on.
    if not isinstance(value, str):
        raise RuntimeConfigurationError(f"{name} must be a string")
    if not value:
        raise RuntimeConfigurationError(f"{name} must not be empty")
    return value


def _normalize_stop(value: Any, name: str) -> tuple[str, ...]:
    # a bare string is one stop sequence, not a sequence of single characters.
    if value is None:
        return ()
    if isinstance(value, str):
        return (_stop_sequence(value, name),)
    if not isinstance(value, Sequence):
        raise RuntimeConfigurationError(f"{name} must be a string or a sequence of strings")
    return tuple(_stop_sequence(entry, name) for entry in value)


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeConfigurationError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise RuntimeConfigurationError(f"{name} must be a finite number")
    return normalized


def validate_generation_max_tokens(value: Any) -> int:
    return _require_int(value, "max_tokens", minimum=1)


def validate_generation_temperature(value: Any) -> float:
    temperature = _require_finite_number(value, "temperature")
    if temperature < 0:
        raise RuntimeConfigurationError("temperature must be non-negative")
    return temperature


def validate_generation_top_p(value: Any) -> float:
    top_p = _require_finite_number(value, "top_p")
    if not 0 < top_p <= 1:
        raise RuntimeConfigurationError("top_p must be greater than zero and at most one")
    return top_p


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
    mm_processor_cache_gb: float = 0.0
    enable_tower_connector_lora: bool = False
    prompt_cache_size: int = 128
    reasoning_parser: str | None = None
    tool_parser: str | None = None
    liveness_interval_seconds: float = 5.0
    engine_args: Mapping[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    processor_kwargs: Mapping[str, Any] = field(default_factory=dict)
    model_revision: str | None = None
    tokenizer_revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _nonempty(self.model, "model"))
        for name in (
            "served_model",
            "tokenizer_model",
            "reasoning_parser",
            "tool_parser",
            "hf_token",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        if self.tool_parser not in {None, "qwen3_coder"}:
            raise RuntimeConfigurationError("tool_parser must be qwen3_coder or none")
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
        object.__setattr__(
            self,
            "enable_tower_connector_lora",
            _require_bool(
                self.enable_tower_connector_lora,
                "enable_tower_connector_lora",
            ),
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
        mm_cache = _require_finite_number(self.mm_processor_cache_gb, "mm_processor_cache_gb")
        if mm_cache < 0:
            raise RuntimeConfigurationError("mm_processor_cache_gb must be non-negative")
        object.__setattr__(self, "mm_processor_cache_gb", mm_cache)
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

        object.__setattr__(
            self,
            "model_revision",
            _revision(self.model_revision, "model_revision"),
        )
        object.__setattr__(
            self,
            "tokenizer_revision",
            _revision(self.tokenizer_revision, "tokenizer_revision"),
        )

        engine_args = _frozen_mapping(self.engine_args, "engine_args")
        tokenizer_kwargs = _frozen_mapping(self.tokenizer_kwargs, "tokenizer_kwargs")
        processor_kwargs = _frozen_mapping(self.processor_kwargs, "processor_kwargs")
        reserved = sorted(_RESERVED_ENGINE_ARGS & engine_args.keys())
        if reserved:
            raise RuntimeConfigurationError(
                "engine_args cannot override runtime-owned keys: " + ", ".join(reserved)
            )
        for name, kwargs in (
            ("tokenizer_kwargs", tokenizer_kwargs),
            ("processor_kwargs", processor_kwargs),
        ):
            reserved_load = sorted(_RESERVED_MODEL_LOAD_KWARGS & kwargs.keys())
            if reserved_load:
                raise RuntimeConfigurationError(
                    f"{name} cannot override runtime-owned keys: " + ", ".join(reserved_load)
                )
        object.__setattr__(self, "engine_args", engine_args)
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
    n: int = 1
    seed: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: bool = False
    top_logprobs: int = 0
    thinking: bool | None = None
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    structured_outputs: Any = None
    tools: Any = None
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None
    stop: Any = None

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
            messages = tuple(
                detached_messages(
                    self.messages,
                    sequence_types=Sequence,
                    sequence_error="messages must be a sequence of mappings",
                    error_type=RuntimeConfigurationError,
                )
            )
            if not messages:
                raise RuntimeConfigurationError("messages must not be empty")
            normalize_messages(
                messages,
                sequence_types=tuple,
                sequence_error="messages must be a sequence of mappings",
                error_type=RuntimeConfigurationError,
                max_source_chars=MAX_SOURCE_CHARS,
            )
            object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "max_tokens", validate_generation_max_tokens(self.max_tokens))
        object.__setattr__(
            self,
            "temperature",
            validate_generation_temperature(self.temperature),
        )
        object.__setattr__(self, "top_p", validate_generation_top_p(self.top_p))
        object.__setattr__(self, "n", validate_choice_count(self.n))
        object.__setattr__(self, "seed", validate_seed(self.seed))
        object.__setattr__(
            self,
            "frequency_penalty",
            validate_penalty(self.frequency_penalty, "frequency_penalty"),
        )
        object.__setattr__(
            self,
            "presence_penalty",
            validate_penalty(self.presence_penalty, "presence_penalty"),
        )
        object.__setattr__(self, "logprobs", validate_logprobs(self.logprobs))
        object.__setattr__(self, "top_logprobs", validate_top_logprobs(self.top_logprobs))
        validate_sampling_relationships(
            n=self.n,
            temperature=self.temperature,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )
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
        validate_tool_control_presence(
            self.tools,
            self.tool_choice,
            self.parallel_tool_calls,
            error_type=RuntimeConfigurationError,
        )
        if self.tools is not None:
            raw_tools = self.tools
            if (
                isinstance(raw_tools, Sequence)
                and not isinstance(raw_tools, str | bytes)
                and all(type(tool) is FunctionTool for tool in raw_tools)
            ):
                raw_tools = tools_wire(tuple(raw_tools))
            normalized_tools = normalize_tools(raw_tools, error_type=RuntimeConfigurationError)
            object.__setattr__(self, "tools", normalized_tools)
            replay_tools = (
                normalized_tools if tools_active(normalized_tools, self.tool_choice) else None
            )
            validate_tool_history_replay(
                self.messages or (), replay_tools, error_type=RuntimeConfigurationError
            )
            if self.tool_choice not in {"auto", "none"}:
                raise RuntimeConfigurationError("tool_choice must be auto or none")
            if self.parallel_tool_calls is not True:
                raise RuntimeConfigurationError("parallel_tool_calls must be true")
            if tools_active(normalized_tools, self.tool_choice):
                if self.prompt is not None:
                    raise RuntimeConfigurationError("tools require chat messages")
                if has_image_blocks(self.messages, sequence_types=tuple):
                    raise RuntimeConfigurationError("tools cannot be combined with image messages")
                if self.logprobs or self.structured_outputs:
                    raise RuntimeConfigurationError(
                        "tools cannot be combined with logprobs or structured outputs"
                    )
        else:
            validate_tool_history_replay(
                self.messages or (), None, error_type=RuntimeConfigurationError
            )
        # an empty sequence and none both mean "no stop sequences", so they normalize together.
        stop = _normalize_stop(self.stop, "stop")
        validate_tool_stop_sequences(
            stop,
            tools=self.tools,
            tool_choice=self.tool_choice,
            error_type=RuntimeConfigurationError,
        )
        object.__setattr__(self, "stop", stop)


@dataclass(frozen=True, slots=True)
class GenerationChoice:
    """one indexed completed output choice."""

    index: int
    text: str
    finish_reason: str | None
    token_ids: tuple[int, ...]
    logprobs: list[dict[str, Any]] | None = None
    tool_calls: tuple[ParsedToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """completed indexed choices, identity, and aggregate accounting."""

    request_id: str
    adapter_id: str | None
    incarnation: str | None
    choices: tuple[GenerationChoice, ...]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cached_tokens_reported: bool
    thinking: bool | None

    @property
    def text(self) -> str:
        return self.choices[0].text

    @property
    def finish_reason(self) -> str | None:
        return self.choices[0].finish_reason


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
    """one normalized indexed choice delta."""

    index: int
    text: str
    logprobs: list[dict[str, Any]] | None = None
    tool_calls: tuple[ParsedToolCall, ...] = ()
    type: Literal["delta"] = field(default="delta", init=False)


@dataclass(frozen=True, slots=True)
class StreamChoiceFinished:
    """one indexed choice terminal."""

    index: int
    text: str
    finish_reason: str
    token_ids: tuple[int, ...]
    type: Literal["choice_finished"] = field(default="choice_finished", init=False)


@dataclass(frozen=True, slots=True)
class StreamFinished:
    """request-level aggregate stream accounting."""

    request_id: str
    runtime_id: str
    adapter_id: str | None
    incarnation: str | None
    choices: tuple[GenerationChoice, ...]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cached_tokens_reported: bool
    thinking: bool | None
    type: Literal["finished"] = field(default="finished", init=False)

    @property
    def text(self) -> str:
        return self.choices[0].text

    @property
    def finish_reason(self) -> str | None:
        return self.choices[0].finish_reason


StreamEvent = StreamReady | StreamDelta | StreamChoiceFinished | StreamFinished


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
