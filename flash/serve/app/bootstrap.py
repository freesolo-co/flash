"""single-engine bootstrap from a fully materialized immutable manifest."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from flash.serve.request.tool_calls import qualified_tool_parser
from flash.serve.runtime import AdapterSpec, EngineConfig, VllmLoraRuntime

from .manifest import ManifestAdapter, ServingManifest
from .materialize import locked_manifest_cache
from .progress import emit_boot_progress, emit_filesystem_usage


@dataclass(frozen=True, slots=True)
class PublishedAdapter:
    """one public model id bound to an exact registered adapter incarnation."""

    requested_model: str
    adapter: ManifestAdapter


class ServingBootstrap:
    """own one runtime and atomically publish its immutable model maps."""

    def __init__(self, manifest: ServingManifest, runtime: VllmLoraRuntime) -> None:
        self.manifest = manifest
        self.runtime = runtime
        self._models: Mapping[str, PublishedAdapter] = MappingProxyType({})
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready and self.runtime.health().ok

    @property
    def models(self) -> Mapping[str, PublishedAdapter]:
        return self._models

    async def close(self) -> None:
        self._ready = False
        self._models = MappingProxyType({})
        await self.runtime.close()


# the factory is called with the engine config, and with `on_engine_death` too when the caller
# supplied one, so it cannot be pinned to a single-argument signature.
RuntimeFactory = Callable[..., VllmLoraRuntime]
EngineDeathHandler = Callable[[Any], Any]


def engine_config_from_manifest(manifest: ServingManifest) -> EngineConfig:
    """translate one exact engine identity into runtime-owned vllm arguments."""

    identity = manifest.engine
    named_args: dict[str, Any] = {
        "dtype": identity.dtype,
        "tensor_parallel_size": identity.tensor_parallel_size,
        "max_model_len": identity.max_model_len,
        "max_num_seqs": identity.max_num_seqs,
        "gpu_memory_utilization": identity.gpu_memory_utilization,
        "cpu_offload_gb": identity.cpu_offload_gb,
    }
    # an unset knob is omitted, never forwarded as None. vllm types these as literals and
    # validates them: `kv_cache_dtype=None` fails CacheConfig with "Input should be 'auto',
    # 'float16', ..." after the weights have already downloaded. omitting the key lets vllm apply
    # its own default ('auto' for both), which is exactly what "unset" is meant to mean, and
    # avoids restating a default that is vllm's to choose.
    named_args.update(
        {
            name: value
            for name, value in (
                ("quantization", identity.quantization),
                ("kv_cache_dtype", identity.kv_cache_dtype),
                ("max_num_batched_tokens", identity.max_num_batched_tokens),
            )
            if value is not None
        }
    )
    named_args.update(dict(manifest.engine_args))
    return EngineConfig(
        model=identity.served_model,
        served_model=identity.served_model,
        tokenizer_model=identity.tokenizer_model,
        model_revision=identity.model_revision,
        tokenizer_revision=identity.tokenizer_revision,
        trust_remote_code=False,
        max_loras=identity.max_loras,
        max_lora_rank=identity.max_lora_rank,
        max_cpu_loras=identity.max_cpu_loras,
        image_limit=identity.image_limit,
        mm_processor_cache_gb=identity.mm_processor_cache_gb,
        enable_tower_connector_lora=identity.enable_tower_connector_lora,
        reasoning_parser=identity.reasoning_parser,
        tool_parser=qualified_tool_parser(manifest.logical_base_model),
        engine_args=named_args,
        tokenizer_kwargs=dict(manifest.tokenizer_kwargs),
        processor_kwargs=dict(manifest.processor_kwargs),
    )


async def bootstrap_serving(
    manifest: ServingManifest,
    cache_root: str | Path,
    *,
    runtime_factory: RuntimeFactory = VllmLoraRuntime,
    on_engine_death: EngineDeathHandler | None = None,
) -> ServingBootstrap:
    """revalidate all data, register all adapters, then publish readiness once.

    `on_engine_death` is forwarded to the runtime so a dead vllm engine core can take the
    container down with it. without a handler the runtime only records the death and the http
    process stays bound, answering 503 forever, and the provider container is never replaced.
    """

    config = engine_config_from_manifest(manifest)
    runtime = (
        runtime_factory(config)
        if on_engine_death is None
        else runtime_factory(config, on_engine_death=on_engine_death)
    )
    owner = ServingBootstrap(manifest, runtime)
    try:
        with locked_manifest_cache(manifest, cache_root) as paths:
            emit_boot_progress(
                "engine-construction-starting",
                model=manifest.engine.served_model,
                revision=manifest.engine.model_revision,
            )
            await runtime.start()
            emit_filesystem_usage("engine-constructed", cache_root)
            emit_boot_progress(
                "engine-constructed",
                model=manifest.engine.served_model,
                revision=manifest.engine.model_revision,
            )
            checkpoints: dict[str, PublishedAdapter] = {}
            for adapter in manifest.adapters:
                path = paths[adapter.checkpoint_id]
                spec = AdapterSpec(
                    adapter_id=adapter.checkpoint_id,
                    path=str(path),
                    incarnation=adapter.aggregate_sha256,
                    thinking=adapter.thinking_default,
                    structured_outputs=(
                        None
                        if adapter.structured_outputs_default is None
                        else dict(adapter.structured_outputs_default)
                    ),
                )
                await runtime.register_adapter(spec)
                checkpoints[adapter.checkpoint_id] = PublishedAdapter(
                    requested_model=adapter.checkpoint_id,
                    adapter=adapter,
                )
            emit_boot_progress("adapters-registered", count=len(checkpoints))
        owner._models = MappingProxyType(dict(sorted(checkpoints.items())))
        owner._ready = True
        emit_filesystem_usage("serving-ready", cache_root)
        return owner
    except BaseException:
        await owner.close()
        raise
