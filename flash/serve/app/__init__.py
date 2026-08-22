"""provider-neutral packaged serving application."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "MANIFEST_SCHEMA": (".manifest", "MANIFEST_SCHEMA"),
    "MANIFEST_VERSION": (".manifest", "MANIFEST_VERSION"),
    "AdapterExecutionInput": (".manifest", "AdapterExecutionInput"),
    "ArtifactFile": (".manifest", "ArtifactFile"),
    "ExecutionInputs": (".manifest", "ExecutionInputs"),
    "ManifestAdapter": (".manifest", "ManifestAdapter"),
    "ManifestError": (".manifest", "ManifestError"),
    "MaterializationError": (".materialize", "MaterializationError"),
    "PublishedAdapter": (".bootstrap", "PublishedAdapter"),
    "ServingBootstrap": (".bootstrap", "ServingBootstrap"),
    "ServingManifest": (".manifest", "ServingManifest"),
    "adapter_cache_path": (".materialize", "adapter_cache_path"),
    "aggregate_file_digest": (".manifest", "aggregate_file_digest"),
    "bootstrap_serving": (".bootstrap", "bootstrap_serving"),
    "build_serving_manifest": (".manifest", "build_serving_manifest"),
    "create_app": (".http", "create_app"),
    "engine_config_from_manifest": (".bootstrap", "engine_config_from_manifest"),
    "hydrate_manifest": (".materialize", "hydrate_manifest"),
    "load_serving_manifest": (".manifest", "load_serving_manifest"),
    "locked_manifest_cache": (".materialize", "locked_manifest_cache"),
    "validate_manifest_cache": (".materialize", "validate_manifest_cache"),
    "validate_materialized_adapter": (".materialize", "validate_materialized_adapter"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
