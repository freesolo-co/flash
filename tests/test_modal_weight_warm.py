"""Execute the generated warmer against a fake Hub to verify exact revision downloads."""

import asyncio
import json
import sys
from types import SimpleNamespace

from flash.core.catalog import MODELS
from flash.providers.artifacts import weight_cache
from flash.providers.modal import api
from tests.test_modal_provider import AsyncMethod


def test_warmer_downloads_catalog_pins_and_default_revisions(monkeypatch):
    models = ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-9B"]
    downloads = []

    def create(*args, **kwargs):
        env = kwargs["secrets"][0]
        assert json.loads(env["FLASH_CACHE_MODELS"]) == {
            model: MODELS[model].managed_revision or None for model in models
        }
        with monkeypatch.context() as patch:
            for key, value in env.items():
                patch.setenv(key, value)
            patch.setitem(
                sys.modules,
                "huggingface_hub",
                SimpleNamespace(
                    snapshot_download=lambda model, **kw: downloads.append((model, kw["revision"]))
                ),
            )
            exec(args[2], {})
        return SimpleNamespace(
            wait=AsyncMethod(lambda: None),
            poll=AsyncMethod(lambda: 0),
            terminate=AsyncMethod(lambda **kw: None),
        )

    modal = SimpleNamespace(
        App=SimpleNamespace(lookup=AsyncMethod(lambda *a, **kw: object())),
        Volume=SimpleNamespace(from_name=lambda *a, **kw: object()),
        Image=SimpleNamespace(
            debian_slim=lambda **kw: SimpleNamespace(pip_install=lambda *a: object())
        ),
        Secret=SimpleNamespace(from_dict=lambda values: values),
        Sandbox=SimpleNamespace(create=AsyncMethod(create)),
    )
    monkeypatch.setattr(
        api, "call", lambda operation, **kw: asyncio.run(operation(modal, object()))
    )
    weight_cache.warm_weight_cache(models=models)
    assert downloads == [(model, MODELS[model].managed_revision or None) for model in models]
