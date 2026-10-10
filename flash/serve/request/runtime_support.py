"""shared safeguards that must not drift between hosted and customer-owned serving runtimes."""

import dataclasses
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any


def argument_names(argument_type: Any) -> set[str]:
    try:
        return {field.name for field in dataclasses.fields(argument_type)}
    except TypeError:
        try:
            return set(inspect.signature(argument_type).parameters)
        except (TypeError, ValueError):
            return set()


def reasoning_compatibility_guard(
    error_factory: type[Exception], message: str
) -> Callable[[Any, Any, str | None], None]:
    def require(async_engine_args_type: Any, generate: Any, reasoning_parser: str | None) -> None:
        if reasoning_parser is None:
            return
        engine_args = argument_names(async_engine_args_type)
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
            raise error_factory(message + ", ".join(missing))

    return require


def is_adapter_tensor_file(path: Path) -> bool:
    name = path.name
    return name in {"adapter_model.safetensors", "adapter_model.bin"} or (
        name.startswith("adapter_model-") and name.endswith((".safetensors", ".bin"))
    )
