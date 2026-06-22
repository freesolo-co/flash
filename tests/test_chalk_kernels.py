"""flash <-> chalk wiring (CPU-safe).

flash applies chalk via chalk's Liger-style ``apply_chalk_kernel_to_qwen35(model, liger=False, ...)``
on the POST-build pass (chalk patches the live module). These tests verify: production kernels are ON
by default and map to the right apply kwargs; FLASH_<K>=0 disables a default-on kernel and
FLASH_<K>=1 enables an opt-in one; the pre-build pass (model=None) is a no-op; the no-op when chalk
is absent or every kernel is disabled; and that a chalk apply error never aborts training.
"""

import sys
import types

from flash.engine.chalk_kernels import (
    _KERNELS,
    active_kernels,
    install_chalk_kernels,
    is_chalk_enabled,
)

# Every FLASH_* kernel flag (so a test can clear leftovers from the environment).
# Every FLASH_* kernel flag. Production kernels are DEFAULT-ON (chalk standalone: rms_norm, swiglu,
# FLCE, rope/lora-delta/embedding, GDN pieces, trainable attention epilogue); the rest are opt-in.
_DEFAULT_ON_FLAGS = tuple(flag for flag, _kw, on in _KERNELS if on)
# Derived from _KERNELS so adding a new opt-in flag (e.g. FLASH_FP8_DX) can't leave this stale.
_ALL_FLAGS = tuple(flag for flag, _kw, _on in _KERNELS)

# The apply kwargs flash should pass for a DEFAULT run: chalk runs STANDALONE (replaces Liger) so
# its own rms_norm/swiglu/FLCE are ON alongside the Qwen3.5-specific production kernels; only the
# situational kernels (eval-only attn epilogue / FP8) stay off. (The bf16 fused-MLP kernel was removed
# in freesolo-chalk 0.3.2 — verified net-negative + eval-only; swiglu already fuses its activation.)
# Derived from _KERNELS (kw -> default_on) so a new opt-in kwarg (e.g. fp8_dx) can't leave this
# stale: the default apply call must pass every _KERNELS kwarg at its default.
_DEFAULT_KWARGS = {kw: on for _flag, kw, on in _KERNELS}


def _clear_flags(monkeypatch):
    for k in _ALL_FLAGS:
        monkeypatch.delenv(k, raising=False)


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


def test_default_on_applies_gap_fillers(monkeypatch):
    """Default post-build apply runs chalk standalone with production kernels on."""
    _clear_flags(monkeypatch)
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    model = object()
    install_chalk_kernels(model)
    assert len(calls) == 1
    got_model, kwargs = calls[0]
    assert got_model is model
    assert kwargs.pop("liger") is False  # chalk STANDALONE (its own rms/swiglu/FLCE); flash uses no Liger
    assert kwargs == _DEFAULT_KWARGS


def test_pre_build_pass_is_noop(monkeypatch):
    """The pre-build pass (model=None) does nothing — chalk's apply patches the live module."""
    _clear_flags(monkeypatch)
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    assert install_chalk_kernels() == {}
    assert install_chalk_kernels(None) == {}
    assert calls == []  # chalk not consulted before the model exists


def test_noop_when_chalk_absent(monkeypatch):
    """Production kernels are default-on, but absent chalk is still a no-op."""
    _clear_flags(monkeypatch)
    monkeypatch.setitem(sys.modules, "chalk", None)  # force ImportError on `from chalk.transformers ...`
    assert install_chalk_kernels(object()) == {}


def test_flag_zero_disables_default_on_kernel(monkeypatch):
    """FLASH_<K>=0/false disables a default-on production kernel; the others stay on."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_ROPE_KERNEL", "0")
    monkeypatch.setenv("FLASH_EMBED_KERNEL", "false")
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object())
    _, kwargs = calls[0]
    assert kwargs["rope"] is False
    assert kwargs["fused_embedding"] is False
    assert kwargs["fused_lora_delta"] is True  # still default-on


def test_optin_flag_enables_extra_kernel(monkeypatch):
    """A default-off kernel turns on with FLASH_<K>=1 alongside the default-on kernels."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_QKV_KERNEL", "1")  # opt in to the attn epilogue
    monkeypatch.setenv("FLASH_FP8_BASE", "1")  # opt in to FP8 frozen base
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object())
    _, kwargs = calls[0]
    assert kwargs["attn_epilogue"] is True
    assert kwargs["fp8_frozen_base"] is True
    assert kwargs["rope"] is True  # default-on kernels still on


def test_all_kernels_disabled_is_noop(monkeypatch):
    """Disabling every default-on kernel (FLASH_<K>=0) -> apply is never called, returns {}."""
    _clear_flags(monkeypatch)
    for k in _DEFAULT_ON_FLAGS:
        monkeypatch.setenv(k, "0")
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    assert install_chalk_kernels(object()) == {}
    assert calls == []  # nothing enabled -> chalk not even imported


def test_apply_error_is_swallowed(monkeypatch):
    """A chalk apply that raises must not abort training; install returns {} instead of propagating."""
    _clear_flags(monkeypatch)
    calls = []
    _install_fake_chalk(monkeypatch, calls, raise_in_apply=True)
    assert install_chalk_kernels(object()) == {}
    assert len(calls) == 1  # it was attempted, then swallowed


def test_returns_chalk_report(monkeypatch):
    """install_chalk_kernels returns chalk's per-kernel report verbatim."""
    _clear_flags(monkeypatch)
    rep = {"rope": True, "fused_lora_delta": 12, "fused_embedding": False, "liger": False}
    calls = []
    _install_fake_chalk(monkeypatch, calls, report=rep)
    assert install_chalk_kernels(object()) == rep


def test_is_chalk_enabled(monkeypatch):
    """is_chalk_enabled is True by default and False only when all default-on kernels are disabled."""
    _clear_flags(monkeypatch)
    import os

    assert is_chalk_enabled(os.environ) is True
    disabled = dict.fromkeys(_DEFAULT_ON_FLAGS, "0")
    assert is_chalk_enabled(disabled) is False
    # a single opt-in flag is enough to need chalk installed
    assert is_chalk_enabled({**disabled, "FLASH_QKV_KERNEL": "1"}) is True


def test_active_kernels_filters_report():
    """active_kernels keeps only the kernels that ENGAGED (truthy, non-error) and drops liger."""
    rep = {
        "liger": False,  # chalk runs standalone — its report carries liger=False; excluded
        "rope": True,
        "fused_lora_delta": 12,  # a count is "engaged"
        "fused_embedding": False,  # fell back
        "attn_epilogue": {"error": "boom"},  # errored
    }
    assert active_kernels(rep) == ["fused_lora_delta", "rope"]
    assert active_kernels({}) == []
    assert active_kernels(None) == []
