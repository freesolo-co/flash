"""Hermetic coverage for optional Weights and Biases logging boundaries."""

from __future__ import annotations

import importlib.util
import sys
import types

import flash.engine.worker.wandb_log as wandb_log


def test_wandb_report_to_skips_without_an_api_key(monkeypatch) -> None:
    """Training without a W&B key must not probe or import the optional package."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        wandb_log,
        "_wandb_importable",
        lambda: (_ for _ in ()).throw(AssertionError("package probe must be skipped")),
    )

    assert wandb_log.wandb_report_to() == []


def test_wandb_report_to_warns_when_the_package_is_unavailable(monkeypatch, capsys) -> None:
    """A configured key with no package must degrade to metrics-file logging with a warning."""
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setattr(wandb_log, "_wandb_importable", lambda: False)

    assert wandb_log.wandb_report_to() == []
    assert "package is missing" in capsys.readouterr().out


def test_wandb_run_info_returns_empty_without_an_active_run(monkeypatch) -> None:
    """Metadata collection must return an empty object when W&B has no current run."""
    fake = types.ModuleType("wandb")
    fake.run = None
    monkeypatch.setitem(sys.modules, "wandb", fake)

    assert wandb_log.wandb_run_info() == {}


def test_wandb_run_info_returns_populated_metadata(monkeypatch) -> None:
    """Active run metadata must be normalized to the fields persisted by the worker."""
    fake = types.ModuleType("wandb")
    fake.run = types.SimpleNamespace(url="https://wandb/run", id="abc", project="proj")
    monkeypatch.setitem(sys.modules, "wandb", fake)

    assert wandb_log.wandb_run_info() == {
        "wandb_url": "https://wandb/run",
        "wandb_id": "abc",
        "wandb_project": "proj",
    }


def test_wandb_run_info_swallows_import_or_attribute_failures(monkeypatch) -> None:
    """A partially broken W&B module must never interrupt worker status persistence."""
    class BrokenModule(types.ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("broken module")

    monkeypatch.setitem(sys.modules, "wandb", BrokenModule("wandb"))

    assert wandb_log.wandb_run_info() == {}


def test_wandb_finish_returns_when_no_run_is_active(monkeypatch) -> None:
    """Finalization must avoid starting a thread when W&B has no active run."""
    fake = types.ModuleType("wandb")
    fake.run = None
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setattr(wandb_log, "_wandb_importable", lambda: True)
    monkeypatch.setattr(
        wandb_log.threading,
        "Thread",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("thread must be skipped")),
    )

    wandb_log.wandb_finish()


def test_wandb_finish_warns_when_finish_raises(monkeypatch, capsys) -> None:
    """A synchronous W&B finish failure must be captured by the worker thread and only warn."""
    fake = types.ModuleType("wandb")
    fake.run = object()

    def finish(*, exit_code):
        raise RuntimeError(f"finish failed {exit_code}")

    fake.finish = finish
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    monkeypatch.setattr(wandb_log, "_wandb_importable", lambda: True)
    monkeypatch.setattr(wandb_log._w, "_WANDB_FINISH_WAIT_S", 0.1)

    wandb_log.wandb_finish(exit_code=0)

    assert "finish() warning: finish failed 0" in capsys.readouterr().out


def test_wandb_importability_treats_find_spec_failure_as_ambiguous_present(monkeypatch) -> None:
    """A broken importlib probe must still allow the guarded import path to make the final decision."""
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (_ for _ in ()).throw(ValueError("partial module")),
    )

    assert wandb_log._wandb_importable() is True
