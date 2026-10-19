"""Test-only stubs so the suite runs WITHOUT a real vLLM install (the serving image has it; CI and
dev boxes usually don't — vLLM is a multi-GB CUDA wheel). Activates ONLY when `vllm` can't be
imported, so a real-vLLM environment is untouched. The engine code imports vLLM lazily inside
methods, so a lightweight stub of just the symbols serving uses is enough to exercise the non-GPU
logic (engine-arg construction, LoRA request building, sampling params).
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import types
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from flash.serving.src.accounting.usage_outbox import (
    OfflineUsageStore,
    UsageEvent,
    UsageOutboxError,
)


@pytest.fixture(scope="module")
def load_modal_app_under_stub() -> Iterator[Callable[[Any], Any]]:
    module_name = "flash.serving.app.modal_app"
    missing = object()
    previous_modal = sys.modules.get("modal", missing)
    previous_app = sys.modules.get(module_name, missing)

    def load(modal_stub: Any) -> Any:
        sys.modules["modal"] = modal_stub
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

    try:
        yield load
    finally:
        sys.modules.pop(module_name, None)
        if previous_app is not missing:
            sys.modules[module_name] = previous_app
        if previous_modal is missing:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous_modal


def _install_vllm_stub() -> None:
    try:
        __import__("vllm")
    except Exception:
        # A discoverable vLLM package can still fail to import on CPU-only dev/CI hosts because its
        # CUDA/runtime libraries are unavailable. Clear any partial import before installing the stub.
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                sys.modules.pop(name, None)
    else:
        return

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []  # mark as package so importing vllm.* works under the stub

    # asyncengineargs is a dataclass so modal_app's `dataclasses.fields(...)` guard (_arg_supported)
    # sees these names and forwards optional args such as reasoning_parser and moe_backend. mirror the
    # fields serving actually passes; extras would raise, matching real-arg validation.
    @dataclasses.dataclass
    class AsyncEngineArgs:
        model: str | None = None
        revision: str | None = None
        tokenizer: str | None = None
        tokenizer_revision: str | None = None
        trust_remote_code: bool = False
        dtype: str = "auto"
        quantization: str | None = None
        kv_cache_dtype: str = "auto"
        tensor_parallel_size: int = 1
        gpu_memory_utilization: float = 0.9
        max_model_len: int | None = None
        enforce_eager: bool = False
        enable_lora: bool = False
        max_loras: int = 1
        max_lora_rank: int = 16
        max_cpu_loras: int | None = None
        max_num_seqs: int | None = None
        max_num_batched_tokens: int | None = None
        language_model_only: bool | None = None
        limit_mm_per_prompt: dict[str, int] | None = None
        mm_processor_cache_gb: float | None = None
        enable_tower_connector_lora: bool = False
        moe_backend: str | None = None
        reasoning_parser: str | None = None
        enable_prefix_caching: bool = False
        disable_log_stats: bool = False

    class AsyncLLMEngine:
        @staticmethod
        def from_engine_args(engine_args: Any) -> Any:
            # tests that need a live-ish engine monkeypatch this; default returns a do-nothing stub.
            return types.SimpleNamespace(engine_args=engine_args)

        async def generate(
            self,
            prompt: Any,
            sampling_params: Any,
            request_id: str,
            *,
            reasoning_ended: bool | None = None,
            reasoning_parser_kwargs: dict[str, Any] | None = None,
            **kwargs: Any,
        ):
            if False:
                yield None

    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class StructuredOutputsParams:
        """Mirrors the subset of vLLM 0.23's dataclass this server uses: an explicit signature (so an
        unknown kwarg raises TypeError like the real dataclass) that stores its kwargs and enforces
        the exactly-one-constraint rule the real __post_init__ raises ValueError for — so tests
        exercise the engine's spec validation realistically. grammar/structural_tag are unsupported
        and normalized out before construction, so they are omitted here."""

        _CONSTRAINTS = ("json", "regex", "choice", "json_object")

        def __init__(
            self,
            json: Any = None,
            regex: str | None = None,
            choice: list[str] | None = None,
            json_object: bool | None = None,
            disable_any_whitespace: bool = False,
            disable_additional_properties: bool = False,
            whitespace_pattern: str | None = None,
        ) -> None:
            self.json = json
            self.regex = regex
            self.choice = choice
            self.json_object = json_object
            self.disable_any_whitespace = disable_any_whitespace
            self.disable_additional_properties = disable_additional_properties
            self.whitespace_pattern = whitespace_pattern
            set_constraints = [c for c in self._CONSTRAINTS if getattr(self, c) is not None]
            if len(set_constraints) != 1:
                raise ValueError(
                    "Exactly one of json, regex, choice or json_object must be set, "
                    f"got {set_constraints or 'none'}"
                )

    class LoRARequest:
        def __init__(
            self, lora_name: str, lora_int_id: int, lora_path: str | None = None, *a: Any, **k: Any
        ) -> None:
            self.lora_name = lora_name
            self.lora_int_id = lora_int_id
            self.lora_path = lora_path

    sampling_params_mod = types.ModuleType("vllm.sampling_params")

    class RequestOutputKind:
        CUMULATIVE = 0
        DELTA = 1
        FINAL_ONLY = 2

    sampling_params_mod.RequestOutputKind = RequestOutputKind
    sampling_params_mod.StructuredOutputsParams = StructuredOutputsParams

    lora_mod = types.ModuleType("vllm.lora")
    lora_mod.__path__ = []  # mark as package so importing vllm.lora.* works under the stub
    lora_request_mod = types.ModuleType("vllm.lora.request")
    lora_request_mod.LoRARequest = LoRARequest

    outputs_mod = types.ModuleType("vllm.outputs")

    class RequestOutput:
        # minimal stand-in for vllm's RequestOutput as used by _num_prompt_tokens: it carries
        # prompt_token_ids and deliberately has no num_prompt_tokens attribute (matching vllm 0.23).
        def __init__(
            self,
            *,
            request_id=None,
            prompt=None,
            prompt_token_ids=None,
            prompt_logprobs=None,
            outputs=None,
            finished=False,
            num_cached_tokens=0,
            **_kwargs,
        ):
            self.request_id = request_id
            self.prompt = prompt
            self.prompt_token_ids = prompt_token_ids
            self.prompt_logprobs = prompt_logprobs
            self.outputs = outputs
            self.finished = finished
            self.num_cached_tokens = num_cached_tokens

    outputs_mod.RequestOutput = RequestOutput

    vllm.AsyncEngineArgs = AsyncEngineArgs
    vllm.AsyncLLMEngine = AsyncLLMEngine
    vllm.SamplingParams = SamplingParams
    vllm.sampling_params = sampling_params_mod
    vllm.lora = lora_mod
    vllm.LoRARequest = LoRARequest
    vllm.outputs = outputs_mod
    lora_mod.request = lora_request_mod

    sys.modules["vllm"] = vllm
    sys.modules["vllm.sampling_params"] = sampling_params_mod
    sys.modules["vllm.lora"] = lora_mod
    sys.modules["vllm.lora.request"] = lora_request_mod
    sys.modules["vllm.outputs"] = outputs_mod


_install_vllm_stub()


class RecordingUsageStore(OfflineUsageStore):
    enabled = True

    def __init__(self, *, fail: bool = False) -> None:
        self.fail_writes = fail
        self.captured: list[UsageEvent] = []
        self.finalized: list[UsageEvent] = []
        self.failed: list[tuple[UsageEvent, str]] = []
        self.closed = False

    async def capture(self, event: UsageEvent) -> None:
        if self.fail_writes:
            raise UsageOutboxError("usage store failure")
        self.captured.append(event)

    async def finalize(self, event: UsageEvent) -> None:
        if self.fail_writes:
            raise UsageOutboxError("usage store failure")
        self.finalized.append(event)

    async def fail(self, event: UsageEvent, code: str) -> None:
        if self.fail_writes:
            raise UsageOutboxError("usage store failure")
        self.failed.append((event, code))

    async def aclose(self) -> None:
        self.closed = True


def attest(record: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Add the engine-side adapter attestation a real engine would return.

    The router is a CPU front door and each engine is a separate GPU container, so the router
    cannot see which adapter the engine actually loaded - it only knows what it asked for. A real
    engine names the adapter it resolved, and the router refuses to bill a revision it cannot
    confirm. A fake pool that skips this is not modelling the engine contract, so every fake in
    the suite routes through here rather than hand-rolling the field.
    """
    if getattr(record, "is_checkpoint", False):
        result["lora_request_adapter"] = record.adapter_id
    return result
