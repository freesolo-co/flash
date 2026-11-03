"""Build a previous-release artifact that retains the forward release's engine handles.

This file is copied outside the checkout before restoring the predecessor, then
shipped with that predecessor as ``flash.serving._rollback_compat``. Keep runtime
imports stdlib-only: the predecessor need not contain any other new serving code.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import shutil
from pathlib import Path
from typing import Any


def engine_manifest(source_sha: str) -> dict[str, Any]:
    from flash.serving.app.rollout import LEGACY_ENGINE_KEYS, model_engine_name
    from flash.serving.app.timeouts import ROUTER_TIMEOUT_SECONDS
    from flash.serving.src.engine.model_config import base_models

    classes: dict[str, list[str]] = {}
    for model, (gpu, concurrency) in LEGACY_ENGINE_KEYS.items():
        safe_gpu = "".join(ch if ch.isalnum() else "_" for ch in gpu)
        classes.setdefault(f"LoraEngine_{safe_gpu}_c{concurrency}", []).append(model)
    for model in base_models():
        classes[model_engine_name(model)] = [model]
    return validate_manifest(
        {
            "version": 1,
            "source_sha": source_sha,
            "startup_seconds": ROUTER_TIMEOUT_SECONDS + 60,
            "classes": classes,
        }
    )


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Unsupported rollback manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_sha", "")):
        raise ValueError("Rollback manifest requires the forward source SHA")
    budget = manifest.get("startup_seconds")
    if type(budget) is not int or budget <= 60:
        raise ValueError("Rollback startup budget must include engine readiness")
    classes = manifest.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("Rollback manifest has no engine identities")
    for name, models in classes.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Invalid rollback engine name")
        if (
            not isinstance(models, list)
            or not models
            or any(not isinstance(model, str) or not model for model in models)
        ):
            raise ValueError("Invalid rollback model list")
    return manifest


def _router_startup_budget(source: str, seconds: int) -> str:
    """Change only the router decorator, preserving the previous implementation."""
    tree = ast.parse(source)
    router = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "router"),
        None,
    )
    if router is None:
        raise ValueError("Previous release has no supported router")
    decorator = next(
        (
            dec
            for dec in router.decorator_list
            if isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Attribute)
            and isinstance(dec.func.value, ast.Name)
            and dec.func.value.id == "app"
            and dec.func.attr == "function"
        ),
        None,
    )
    if decorator is None:
        raise ValueError("Previous router has no supported Modal decorator")
    keyword = next((kw for kw in decorator.keywords if kw.arg == "startup_timeout"), None)
    if keyword is None:
        decorator.keywords.append(ast.keyword(arg="startup_timeout", value=ast.Constant(seconds)))
    else:
        keyword.value = ast.Call(
            func=ast.Name(id="max", ctx=ast.Load()),
            args=[keyword.value, ast.Constant(seconds)],
            keywords=[],
        )
    # AST offsets are UTF-8 byte offsets, not Python string indices.
    lines = source.encode().splitlines(keepends=True)
    start = sum(map(len, lines[: decorator.lineno - 1])) + decorator.col_offset
    end = sum(map(len, lines[: decorator.end_lineno - 1])) + decorator.end_col_offset
    return (
        source.encode()[:start] + ast.unparse(decorator).encode() + source.encode()[end:]
    ).decode()


def prepare_artifact(entrypoint: Path, manifest: dict[str, Any]) -> None:
    manifest = validate_manifest(manifest)
    source = _router_startup_budget(entrypoint.read_text(), manifest["startup_seconds"])
    # Both historical entrypoint layouts mount the flash package into the image.
    serving = next((parent for parent in entrypoint.parents if parent.name == "serving"), None)
    if serving is None or serving.parent.name != "flash":
        raise ValueError("Rollback entrypoint must be in flash/serving")
    source += (
        "\n# Rollback compatibility overlay; engine implementation remains the predecessor's.\n"
        "from flash.serving._rollback_compat import install_compatibility\n"
        f"install_compatibility(globals(), {manifest!r})\n"
    )
    ast.parse(source)
    shutil.copyfile(__file__, serving / "_rollback_compat.py")
    entrypoint.write_text(source)
    print(f"Prepared rollback with {len(manifest['classes'])} retained engine identities")


async def _warm_restored_model(model: str, target: Any, aliases: list[Any]) -> None:
    active = False
    for engine in [target, *aliases]:
        stats = await engine.health.get_current_stats.aio()
        active = active or any((stats.num_total_runners, stats.num_running_inputs, stats.backlog))
    if active:
        health = await target.health.remote.aio()
        if health.get("ok") is not True or health.get("base_model") != model:
            raise RuntimeError(f"Restored engine is not healthy for {model}")


def install_compatibility(namespace: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Restore old behavior under every name held by a forward router.

    Registering the same class name keeps its Modal function ID alive. Old/new
    replicas then participate in a rolling replacement instead of immediate
    function deletion. No failed inference or partial stream is retried.
    """
    manifest = validate_manifest(manifest)
    required = (
        "_build_engine",
        "_engine_cls_for",
        "_engine_concurrency",
        "gpu_for",
        "_ModalEnginePool",
    )
    if any(not callable(namespace.get(name)) for name in required):
        raise ValueError("Previous release cannot safely retain engine identities")
    bindings: dict[str, list[Any]] = {}
    for name, models in manifest["classes"].items():
        # Validate all models against the predecessor, even when the name exists.
        # Removing a newly introduced model cannot safely restore its requests.
        configs = {
            (namespace["gpu_for"](model), *namespace["_engine_concurrency"](model))
            for model in models
        }
        cls = namespace.get(name)
        if cls is None:
            if len(configs) != 1:
                raise ValueError(f"Previous release cannot retain shared engine {name}")
            gpu, concurrency, target = configs.pop()
            cls = namespace["_build_engine"](gpu, name, concurrency, target)
            namespace[name] = cls
        for model in models:
            engine = cls(base_model=model)
            for method in ("health", "generate", "stream_generate_call", "register", "unregister"):
                if not hasattr(engine, method):
                    raise ValueError(f"Previous engine {name} cannot retain method {method}")
            bindings.setdefault(model, []).append(engine)

    pool = namespace["_ModalEnginePool"]
    original_start = getattr(pool, "start", None)

    async def start(self):
        async with asyncio.timeout(manifest["startup_seconds"] - 60):
            if original_start is not None:
                await original_start(self)
            async with asyncio.TaskGroup() as tasks:
                for model, aliases in bindings.items():
                    target = namespace["_engine_cls_for"](model)(base_model=model)
                    tasks.create_task(_warm_restored_model(model, target, aliases))

    pool.start = start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--source-sha", required=True)
    snapshot.add_argument("--directory", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--entrypoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        manifest = engine_manifest(args.source_sha)
        args.directory.mkdir(parents=True, exist_ok=True)
        (args.directory / "manifest.json").write_text(json.dumps(manifest))
        shutil.copyfile(__file__, args.directory / "rollback.py")
    else:
        prepare_artifact(args.entrypoint, json.loads(args.manifest.read_text()))


if __name__ == "__main__":
    main()
