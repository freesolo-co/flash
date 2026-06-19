"""flash <-> chalk wiring (CPU-safe).

Verifies the optional chalk-kernel hook: it no-ops when chalk isn't installed, calls the
right installers with/without the model, skips instance-only installers when model is None,
and never lets an installer error abort training.
"""

import sys
import types

from flash.engine.chalk_kernels import install_chalk_kernels


def _install_fake_chalk(monkeypatch, calls, *, raise_in=None):
    ck = types.ModuleType("chalk.transformers")

    def make(name, needs_model):
        def fn(model=None):
            calls.append((name, model))
            if raise_in == name:
                raise RuntimeError("boom")
            return True

        return fn

    for name in (
        "install_lora",
        "install_qwen35_mlp",
        "install_qwen35_qkv",
        "install_qwen35_rope",
        "install_qwen35_mlp_fp8",
        "install_fp8_base",
        "install_qwen35_embedding",
    ):
        setattr(ck, name, make(name, "fp8" in name or "embedding" in name))

    chalk_pkg = types.ModuleType("chalk")
    chalk_pkg.transformers = ck
    monkeypatch.setitem(sys.modules, "chalk", chalk_pkg)
    monkeypatch.setitem(sys.modules, "chalk.transformers", ck)


def test_noop_when_chalk_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "chalk", None)  # force ImportError
    assert install_chalk_kernels() == {}
    assert install_chalk_kernels(object()) == {}


def test_model_none_skips_instance_only_installers(monkeypatch):
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    res = install_chalk_kernels(None)
    names = {n for n, _ in calls}
    # class/function-level installers run with model=None
    assert {"install_lora", "install_qwen35_mlp", "install_qwen35_qkv", "install_qwen35_rope"} <= names
    # instance-only installers are skipped when there's no model
    assert "install_fp8_base" not in names
    assert "install_qwen35_embedding" not in names
    assert "install_qwen35_mlp_fp8" not in names
    assert res  # non-empty result map


def test_model_present_runs_instance_installers_with_model(monkeypatch):
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    sentinel = object()
    install_chalk_kernels(sentinel)
    by_name = dict(calls)
    assert by_name["install_fp8_base"] is sentinel
    assert by_name["install_qwen35_embedding"] is sentinel
    assert by_name["install_lora"] is None  # class-level: called without the model


def test_installer_error_is_swallowed(monkeypatch):
    calls = []
    _install_fake_chalk(monkeypatch, calls, raise_in="install_qwen35_mlp")
    res = install_chalk_kernels(object())  # must not raise
    assert str(res["install_qwen35_mlp"]).startswith("error:")
