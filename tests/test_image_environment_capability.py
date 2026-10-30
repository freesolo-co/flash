from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import flash.engine.worker.runtime.state as worker_state
from flash.envs.loading.adapter import FreesoloEnvironment
from flash.envs.loading.base import BaseEnvironment


class _PlainEnvironment(BaseEnvironment):
    pass


def _spec(model: str, algorithm: str = "grpo"):
    return SimpleNamespace(
        model=model,
        algorithm=algorithm,
        train=SimpleNamespace(),
    )


def test_base_environment_defaults_image_observations_false():
    assert _PlainEnvironment("plain").image_observations is False


def test_freesolo_adapter_carries_only_an_explicit_class_capability():
    class DynamicImageEnvironment:
        image_observations = True

    class InstanceOnlyEnvironment:
        pass

    instance_only = InstanceOnlyEnvironment()
    instance_only.image_observations = True

    dynamic = FreesoloEnvironment(DynamicImageEnvironment(), "dynamic", source=None)
    instance = FreesoloEnvironment(instance_only, "instance", source=None)

    assert dynamic.image_observations is True
    assert instance.image_observations is False


def test_worker_environment_load_applies_the_capability_guard(worker_context, monkeypatch):

    spec = SimpleNamespace(
        environment=SimpleNamespace(id="local", params={}, resolved_sha=""),
        model="meta-llama/Llama-3.2-1B",
        algorithm="grpo",
        train=SimpleNamespace(hf_repo=""),
        thinking=False,
    )
    monkeypatch.setattr(worker_context, "config", replace(worker_context.config, spec=spec))
    monkeypatch.setattr(
        worker_state,
        "load_staged_freesolo_environment",
        lambda *_args, **_kwargs: (SimpleNamespace(image_observations=True), None),
    )

    with pytest.raises(ValueError, match="does not support image-bearing"):
        worker_context.require_active_env()
