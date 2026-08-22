"""provider-neutral packaged serving application."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "AdapterExecutionInput": (".manifest", "AdapterExecutionInput"),
    "ArtifactFile": (".manifest", "ArtifactFile"),
    "ExecutionInputs": (".manifest", "ExecutionInputs"),
    "ManifestError": (".manifest", "ManifestError"),
    "aggregate_file_digest": (".manifest", "aggregate_file_digest"),
    "build_serving_manifest": (".manifest", "build_serving_manifest"),
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
