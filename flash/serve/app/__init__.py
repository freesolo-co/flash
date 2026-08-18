"""provider-neutral packaged serving application."""

from .bootstrap import (
    BootstrapError,
    PublishedAdapter,
    ServingBootstrap,
    bootstrap_serving,
    engine_config_from_manifest,
)
from .manifest import (
    MANIFEST_SCHEMA,
    MANIFEST_VERSION,
    AdapterExecutionInput,
    ArtifactFile,
    ExecutionInputs,
    ManifestAdapter,
    ManifestError,
    ServingManifest,
    aggregate_file_digest,
    build_serving_manifest,
    load_serving_manifest,
)
from .materialize import (
    MaterializationError,
    adapter_cache_path,
    hydrate_manifest,
    locked_manifest_cache,
    validate_manifest_cache,
    validate_materialized_adapter,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "AdapterExecutionInput",
    "ArtifactFile",
    "BootstrapError",
    "ExecutionInputs",
    "ManifestAdapter",
    "ManifestError",
    "MaterializationError",
    "PublishedAdapter",
    "ServingBootstrap",
    "ServingManifest",
    "adapter_cache_path",
    "aggregate_file_digest",
    "bootstrap_serving",
    "build_serving_manifest",
    "create_app",
    "engine_config_from_manifest",
    "hydrate_manifest",
    "load_serving_manifest",
    "locked_manifest_cache",
    "validate_manifest_cache",
    "validate_materialized_adapter",
]


def __getattr__(name: str):
    if name == "create_app":
        from .http import create_app

        return create_app
    raise AttributeError(name)
