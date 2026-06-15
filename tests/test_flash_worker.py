"""Regression tests for the Flash worker plumbing fixed in this PR.

Covers:
- build_worker_env forwards the documented RL/vLLM tuning knobs to the GPU worker
  (they were silently dropped) and sets a fragmentation-safe allocator default;
- the runpod_flash backoff OverflowError that aborted long runs is patched;
- the serve cold-start execution timeout is generous + env-overridable;
- per-phase error artifact names don't collide (train error survives a later eval error).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _spec():
    from autoslm.worker_spec import JobSpec, TrainSpec

    return JobSpec(
        model="Qwen/Qwen3-4B-Instruct-2507",
        algorithm="grpo",
        train=TrainSpec(steps=10, seeds=(0,)),
    )


def test_build_worker_env_forwards_tuning_knobs(monkeypatch):
    """DEFECT: RL_VLLM_GPU_UTIL / RL_PER_DEVICE_PROMPTS (and the others) were never added to
    the forward list, so the docs' OOM-fix advice couldn't reach the worker."""
    from autoslm.flash.train import build_worker_env

    knobs = {
        "RL_VLLM_GPU_UTIL": "0.40",
        "RL_VLLM_SLEEP": "1",
        "RL_PER_DEVICE_PROMPTS": "2",
        "RL_PROMPTS_PER_STEP": "16",
        "RL_GROUP_SIZE": "8",
        "RL_MAX_COMPLETION": "512",
        "VLLM_USE_V1": "0",
        "SFT_PER_DEVICE_BS": "4",
    }
    for k, v in knobs.items():
        monkeypatch.setenv(k, v)

    env = build_worker_env(_spec(), 0)
    for k, v in knobs.items():
        assert env.get(k) == v, f"{k} not forwarded to worker"
    # fragmentation-safe allocator default is always set
    assert "PYTORCH_CUDA_ALLOC_CONF" in env


def test_build_worker_env_respects_alloc_conf_override(monkeypatch):
    from autoslm.flash.train import build_worker_env

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    env = build_worker_env(_spec(), 0)
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:256"


def test_alloc_conf_default_avoids_expandable_under_grpo_sleep(monkeypatch):
    # vLLM sleep-mode CuMemAllocator is incompatible with expandable_segments; GRPO with sleep
    # ON (the default) must NOT default to expandable_segments or the run crashes at engine init.
    from autoslm.flash.train import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("RL_VLLM_SLEEP", raising=False)  # default = sleep on
    env = build_worker_env(_spec(), 0)
    assert "expandable_segments" not in env["PYTORCH_ALLOC_CONF"]
    assert env["PYTORCH_ALLOC_CONF"] == env["PYTORCH_CUDA_ALLOC_CONF"]


def test_alloc_conf_default_expandable_when_sleep_off(monkeypatch):
    from autoslm.flash.train import build_worker_env

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("RL_VLLM_SLEEP", "0")  # sleep off -> expandable is safe + preferred
    env = build_worker_env(_spec(), 0)
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_alloc_conf_default_expandable_for_sft(monkeypatch):
    from autoslm.flash.train import build_worker_env

    from autoslm.worker_spec import JobSpec, TrainSpec

    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    spec = JobSpec(model="Qwen/Qwen3-0.6B", algorithm="sft", train=TrainSpec(steps=2, seeds=(0,)))
    env = build_worker_env(spec, 0)
    assert env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"


def test_runpod_backoff_no_overflow_on_long_runs():
    """DEFECT: runpod_flash computed base*(2**attempt) then clamped, so a long poll loop
    overflowed (~80 min in) and killed a healthy job. The patch caps the exponent first."""
    pytest.importorskip("runpod_flash")
    from autoslm.flash.train import _patch_runpod_backoff

    _patch_runpod_backoff()
    from runpod_flash.core.utils import backoff

    # Pre-patch this raised OverflowError; now it must return a clamped, finite delay.
    delay = backoff.get_backoff_delay(5000, max_seconds=5)
    assert delay <= 5 * 1.2 + 1e-9
    # the serverless module's imported reference is patched too (that's the real call site)
    from runpod_flash.core.resources import serverless

    assert serverless.get_backoff_delay(100000, max_seconds=5) <= 5 * 1.2 + 1e-9


def test_serve_execution_timeout_default_and_override(monkeypatch):
    """DEFECT: serve execution cap (10 min) was shorter than a cold serving worker's
    startup, so the first slm chat/deploy failed with 'executionTimeout exceeded'."""
    from autoslm.serve import deploy

    monkeypatch.delenv("AUTOSLM_SERVE_TIMEOUT_MS", raising=False)
    assert deploy.serve_execution_timeout_ms() >= 20 * 60 * 1000  # generous default

    monkeypatch.setenv("AUTOSLM_SERVE_TIMEOUT_MS", str(30 * 60 * 1000))
    assert deploy.serve_execution_timeout_ms() == 30 * 60 * 1000


def test_error_artifact_name_is_per_phase():
    """Per-phase error files keep the root-cause train traceback under a stable name."""
    from autoslm.engine.worker import error_artifact_name

    names = {error_artifact_name(m) for m in ("sft", "rl")}
    assert len(names) == 2  # distinct per phase -> no clobber
    assert error_artifact_name("rl") == "error_rl.txt"


def test_train_body_imports_every_name_it_uses():
    """Flash ships only _train_body's source to the worker, where module-level
    imports are out of scope, so every stdlib/3p name it references must be
    imported inside the function body (else NameError before training)."""
    import ast
    import inspect

    from autoslm.flash import train

    tree = ast.parse(inspect.getsource(train._train_body))
    fn = tree.body[0]
    imported = {
        alias.asname or alias.name.split(".")[0]
        for node in ast.walk(fn)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    # Names that must be locally imported (regression: contextlib was missing).
    for name in ("contextlib", "json", "os", "subprocess", "sys"):
        assert name in imported, f"_train_body uses {name!r} without a local import"


def test_flash_dotted_and_from_imports_both_work():
    """Fix #10: the autoslm.flash compatibility shim must support BOTH import spellings
    for every aliased submodule:
      * ``from autoslm.flash import train``  (attribute lookup on the package)
      * ``import autoslm.flash.train``       (sys.modules resolution)
    and both must yield the SAME provider module object the new code uses (so a
    monkeypatch against the alias is effective)."""
    import importlib

    import autoslm.flash as flash
    from autoslm.providers.runpod import durable as rp_durable
    from autoslm.providers.runpod import pricing as rp_pricing
    from autoslm.providers.runpod import train as rp_train

    expected = {"train": rp_train, "durable": rp_durable, "pricing": rp_pricing}
    for name, real in expected.items():
        # `import autoslm.flash.<name>` (dotted form)
        dotted = importlib.import_module(f"autoslm.flash.{name}")
        assert dotted is real, f"import autoslm.flash.{name} resolved to the wrong object"
        # `from autoslm.flash import <name>` (attribute form)
        assert getattr(flash, name) is real, f"autoslm.flash.{name} attribute missing/wrong"
        # the dotted attribute is bound on the package after the dotted import
        assert getattr(flash, name) is dotted
