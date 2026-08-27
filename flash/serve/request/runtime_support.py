"""shared safeguards that must not drift between hosted and customer-owned serving runtimes."""

import dataclasses
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

from flash.adapters.artifacts import has_loadable_adapter_weights


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


def adapter_dir_is_loadable(path: Path) -> bool:
    """True when ``path`` holds a config and weights the engine about to read them can load.

    Both serving runtimes ask this before handing a directory to vLLM, and both used to answer it
    by looking for any one file whose NAME looked like adapter weights. A name is not the question:
    peft binds one representation per suffix, and it discovers the sharded form only through
    ``adapter_model.<ext>.index.json``. A directory holding a lone
    ``adapter_model-00001-of-00002.safetensors`` and no index therefore has no loadable
    representation at all, but passed the name check -- which is exactly what a half-finished
    download looks like, so a cache entry could be declared ready mid-fetch and the engine then
    fails on weights it was told were there.

    Deferring to :func:`has_loadable_adapter_weights` keeps that verdict identical to the one
    deployment admission and the exporter already reach from a remote listing. It also drops
    ``.bin``, which the same authority has never accepted: deployment rejects a bin-only adapter
    before provisioning, so a runtime that loaded one would be serving something no supported
    path could have produced.
    """

    if not (path / "adapter_config.json").is_file():
        return False
    return has_loadable_adapter_weights(
        child.name for child in path.iterdir() if child.is_file() and child.stat().st_size > 0
    )
