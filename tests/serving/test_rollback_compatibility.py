"""Rollback retains names while restoring behavior, readiness, and failure isolation."""

import ast
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from flash.serving.app.rollback import (
    engine_manifest,
    install_compatibility,
    prepare_artifact,
    validate_manifest,
)


def _manifest():
    return {
        "version": 1,
        "source_sha": "a" * 40,
        "startup_seconds": 120,
        "classes": {"LegacyEngine": ["base"], "StableEngine": ["base"]},
    }


def _predecessor(*, runners=0, original_start=None):
    built = []
    instances = {}

    class Pool:
        pass

    if original_start is not None:
        Pool.start = original_start

    def build(gpu, name, concurrency, target):
        built.append((gpu, name, concurrency, target))

        def bind(*, base_model):
            key = name, base_model
            if key not in instances:
                instances[key] = SimpleNamespace(
                    health=SimpleNamespace(
                        get_current_stats=SimpleNamespace(
                            aio=AsyncMock(
                                return_value=SimpleNamespace(
                                    num_total_runners=runners if name == "StableEngine" else 0,
                                    num_running_inputs=0,
                                    backlog=0,
                                )
                            )
                        ),
                        remote=SimpleNamespace(
                            aio=AsyncMock(return_value={"ok": True, "base_model": base_model})
                        ),
                    ),
                    generate=AsyncMock(return_value="previous implementation"),
                    stream_generate_call=AsyncMock(),
                    register=AsyncMock(),
                    unregister=AsyncMock(),
                )
            return instances[key]

        return bind

    legacy = build("L40S", "LegacyEngine", 64, 64)

    def gpu_for(model):
        if model != "base":
            raise ValueError("Unsupported base model")
        return "L40S"

    namespace = {
        "LegacyEngine": legacy,
        "_build_engine": build,
        "_engine_cls_for": lambda model: legacy,
        "_engine_concurrency": lambda model: (64, 64),
        "gpu_for": gpu_for,
        "_ModalEnginePool": Pool,
    }
    return namespace, built, instances


def test_retained_name_runs_predecessor_implementation_without_replacing_existing_name():
    namespace, built, instances = _predecessor()
    legacy = namespace["LegacyEngine"]
    install_compatibility(namespace, _manifest())
    assert namespace["LegacyEngine"] is legacy
    assert built == [("L40S", "LegacyEngine", 64, 64), ("L40S", "StableEngine", 64, 64)]
    assert asyncio.run(instances["StableEngine", "base"].generate()) == "previous implementation"


def test_replacement_router_waits_for_old_pool_while_forward_pool_is_active():
    namespace, _, instances = _predecessor(runners=2)
    install_compatibility(namespace, _manifest())

    async def scenario():
        entered, ready = asyncio.Event(), asyncio.Event()

        async def health():
            entered.set()
            await ready.wait()
            return {"ok": True, "base_model": "base"}

        instances["LegacyEngine", "base"].health.remote.aio.side_effect = health
        startup = asyncio.create_task(namespace["_ModalEnginePool"]().start())
        await entered.wait()
        assert not startup.done()
        ready.set()
        await startup

    asyncio.run(scenario())
    instances["StableEngine", "base"].health.remote.aio.assert_not_awaited()


def test_idle_pools_stay_at_zero_and_existing_startup_is_preserved():
    called = []

    async def original_start(self):
        called.append(self)

    namespace, _, instances = _predecessor(original_start=original_start)
    install_compatibility(namespace, _manifest())
    pool = namespace["_ModalEnginePool"]()
    asyncio.run(pool.start())
    assert called == [pool]
    for engine in instances.values():
        engine.health.remote.aio.assert_not_awaited()


@pytest.mark.parametrize(
    "health", [{"ok": False, "base_model": "base"}, {"ok": True, "base_model": "other"}]
)
def test_unhealthy_restored_pool_prevents_router_startup(health):
    namespace, _, instances = _predecessor(runners=1)
    install_compatibility(namespace, _manifest())
    instances["LegacyEngine", "base"].health.remote.aio.return_value = health
    with pytest.raises(ExceptionGroup) as error:
        asyncio.run(namespace["_ModalEnginePool"]().start())
    assert "Restored engine is not healthy" in str(error.value.exceptions[0])


def test_missing_previous_model_cannot_be_silently_removed():
    namespace, _, _ = _predecessor()
    manifest = _manifest()
    manifest["classes"]["NewModelEngine"] = ["new-model"]
    with pytest.raises(ValueError, match="Unsupported base model"):
        install_compatibility(namespace, manifest)


def test_unsupported_previous_app_is_rejected():
    with pytest.raises(ValueError, match="cannot safely retain"):
        install_compatibility({}, _manifest())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("source_sha", "bad"),
        ("startup_seconds", 60),
        ("classes", {}),
        ("classes", {"x();": ["base"]}),
    ],
)
def test_invalid_manifest_is_rejected(field, value):
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(ValueError, match=r"[Rr]ollback"):
        validate_manifest(manifest)


@pytest.mark.parametrize("layout", ["flash/serving/app/modal_app.py", "flash/serving/modal_app.py"])
@pytest.mark.parametrize("existing_timeout", [None, "700"])
def test_prepared_source_reconstructs_retained_classes_on_container_import(
    tmp_path, layout, existing_timeout
):
    entrypoint = tmp_path / layout
    entrypoint.parent.mkdir(parents=True)
    decorator = (
        "@app.function(timeout=900"
        + (f", startup_timeout={existing_timeout}" if existing_timeout else "")
        + ")"
    )
    source = f"""# previous implementation — preserve this Unicode text
{decorator}
def router():
    return "previous router"
"""
    entrypoint.write_text(source)
    prepare_artifact(entrypoint, _manifest())
    tree = ast.parse(entrypoint.read_text())
    router = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    keywords = {kw.arg: kw.value for kw in router.decorator_list[0].keywords}
    assert eval(compile(ast.Expression(keywords["startup_timeout"]), "timeout", "eval")) == (
        700 if existing_timeout else 120
    )
    assert ast.literal_eval(keywords["timeout"]) == 900
    assert "previous router" in entrypoint.read_text()
    copied = (tmp_path / "flash/serving/_rollback_compat.py").read_text()
    # The shipped module must remain usable by a predecessor without the new rollout module.
    copied_tree = ast.parse(copied)
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("flash")
        for node in copied_tree.body
    )
    # Execute the appended bootstrap twice, like local deployment and remote import.
    bootstrap = tree.body[-1]
    for _ in range(2):
        namespace, _, instances = _predecessor()
        namespace["install_compatibility"] = install_compatibility
        exec(compile(ast.Module(body=[bootstrap], type_ignores=[]), "bootstrap", "exec"), namespace)
        assert ("StableEngine", "base") in instances


def test_prepare_refuses_an_unknown_router_before_changing_source(tmp_path):
    app = tmp_path / "flash/serving/app/modal_app.py"
    app.parent.mkdir(parents=True)
    app.write_text("unknown = True\n")
    with pytest.raises(ValueError, match="no supported router"):
        prepare_artifact(app, _manifest())
    assert app.read_text() == "unknown = True\n"
    assert not (tmp_path / "flash/serving/_rollback_compat.py").exists()


def test_manifest_captures_only_current_model_identities():
    from flash.serving.app.rollout import model_engine_name
    from flash.serving.src.engine.model_config import base_models

    manifest = json.loads(json.dumps(engine_manifest("b" * 40)))
    assert {model for models in manifest["classes"].values() for model in models} == set(
        base_models()
    )
    for model in base_models():
        assert manifest["classes"][model_engine_name(model)] == [model]
    assert set(manifest["classes"]) == {model_engine_name(model) for model in base_models()}
