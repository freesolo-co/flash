"""flash <-> chalk wiring (CPU-safe).

flash applies chalk standalone via ``apply_chalk_kernel_to_qwen35(model, liger=False, ...)`` on the
POST-build pass (chalk patches the live module). Kernel selection is FIXED: training kernels are ON,
eval-only / opt-in tradeoff kernels are OFF, with NO env override. These tests verify: the fixed
kwargs are passed; the selection ignores any FLASH_* env var; the pre-build pass (model=None) is a
no-op; the no-op when chalk is absent; and that a chalk apply error never aborts training.
"""

import sys
import types

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels

# The apply kwargs flash always passes (gap-fillers on, overlap/situational off) — fixed, no env.
_FIXED_KWARGS = {
    "rope": True,
    "rmsnorm": True,
    "swiglu": True,
    "fused_linear_cross_entropy": True,
    "fused_lora_delta": True,
    "trainable_attn_epilogue": True,
    "fused_embedding": True,
    "gdn": True,
    "fused_mlp": False,
    "attn_epilogue": False,
    "fp8_frozen_base": False,
}


def _install_fake_chalk(monkeypatch, calls, *, raise_in_apply=False, report=None):
    """Inject a fake ``chalk.transformers`` whose apply records (model, kwargs) and returns report."""
    ck = types.ModuleType("chalk.transformers")

    def apply_chalk_kernel_to_qwen35(model, **kwargs):
        calls.append((model, kwargs))
        if raise_in_apply:
            raise RuntimeError("boom")
        return report if report is not None else {k: v for k, v in kwargs.items() if v is True}

    ck.apply_chalk_kernel_to_qwen35 = apply_chalk_kernel_to_qwen35
    chalk_pkg = types.ModuleType("chalk")
    chalk_pkg.transformers = ck
    monkeypatch.setitem(sys.modules, "chalk", chalk_pkg)
    monkeypatch.setitem(sys.modules, "chalk.transformers", ck)


def test_applies_fixed_gap_fillers(monkeypatch):
    """Post-build -> apply is called once with liger=False and the fixed standalone kwargs."""
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    model = object()
    install_chalk_kernels(model)
    assert len(calls) == 1
    got_model, kwargs = calls[0]
    assert got_model is model
    assert kwargs.pop("liger") is False  # chalk standalone; TRL must not apply Liger first
    assert kwargs == _FIXED_KWARGS


def test_selection_ignores_env_flags(monkeypatch):
    """Selection is deterministic: a leftover FLASH_* env var does NOT change which kernels run."""
    # These would previously have toggled kernels; now they must be ignored entirely.
    monkeypatch.setenv("FLASH_ROPE_KERNEL", "0")
    monkeypatch.setenv("FLASH_MLP_KERNEL", "1")
    monkeypatch.setenv("FLASH_FP8_BASE", "1")
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object())
    _, kwargs = calls[0]
    kwargs.pop("liger")
    assert kwargs == _FIXED_KWARGS  # unchanged despite the env vars


def test_pre_build_pass_is_noop(monkeypatch):
    """The pre-build pass (model=None) does nothing — chalk's apply patches the live module."""
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    assert install_chalk_kernels() == {}
    assert install_chalk_kernels(None) == {}
    assert calls == []  # chalk not consulted before the model exists


def test_noop_when_chalk_absent(monkeypatch):
    """If freesolo-chalk isn't installed -> no-op (returns {})."""
    monkeypatch.setitem(sys.modules, "chalk", None)  # force ImportError on `from chalk.transformers ...`
    assert install_chalk_kernels(object()) == {}


def test_apply_error_is_swallowed(monkeypatch):
    """A chalk apply that raises must not abort training; install returns {} instead of propagating."""
    calls = []
    _install_fake_chalk(monkeypatch, calls, raise_in_apply=True)
    assert install_chalk_kernels(object()) == {}
    assert len(calls) == 1  # it was attempted, then swallowed


def test_returns_chalk_report(monkeypatch):
    """install_chalk_kernels returns chalk's per-kernel report verbatim."""
    rep = {"rope": True, "fused_lora_delta": 12, "fused_embedding": False, "liger": False}
    calls = []
    _install_fake_chalk(monkeypatch, calls, report=rep)
    assert install_chalk_kernels(object()) == rep


def test_active_kernels_filters_report():
    """active_kernels keeps only the kernels that ENGAGED (truthy, non-error) and drops liger."""
    rep = {
        "liger": False,  # excluded when present in chalk's report
        "rope": True,
        "fused_lora_delta": 12,  # a count is "engaged"
        "fused_embedding": False,  # fell back
        "fp8_frozen_base": {"error": "boom"},  # errored
    }
    assert active_kernels(rep) == ["fused_lora_delta", "rope"]
    assert active_kernels({}) == []
    assert active_kernels(None) == []
