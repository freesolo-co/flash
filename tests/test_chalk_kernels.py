"""flash <-> chalk wiring (CPU-safe).

flash applies chalk via chalk's Liger-style ``apply_chalk_kernel_to_qwen35(model, liger=False, ...)``
on the POST-build pass (chalk patches the live module). These tests verify: the gap-fillers are ON
by default and map to the right apply kwargs; FLASH_<K>=0 disables a default-on kernel and
FLASH_<K>=1 enables an opt-in one; the pre-build pass (model=None) is a no-op; the no-op when chalk
is absent or every kernel is disabled; and that a chalk apply error never aborts training.
"""

import importlib.util
import sys
import types

from flash.engine.chalk_kernels import (
    active_kernels,
    flce_flag_on,
    fused_ce_available,
    install_chalk_kernels,
    is_chalk_enabled,
)

# Every FLASH_* kernel flag (so a test can clear leftovers from the environment).
# Every FLASH_* kernel flag. The first six are DEFAULT-ON (chalk's standalone stack: rms_norm,
# swiglu, FLCE + the gap-fillers rope/lora-delta/embedding); the last three are opt-in.
_DEFAULT_ON_FLAGS = (
    "FLASH_RMSNORM_KERNEL",
    "FLASH_SWIGLU_KERNEL",
    "FLASH_FLCE_KERNEL",
    "FLASH_ROPE_KERNEL",
    "FLASH_TRITON_LORA",
    "FLASH_EMBED_KERNEL",
)
_ALL_FLAGS = (
    *_DEFAULT_ON_FLAGS,
    "FLASH_MLP_KERNEL",
    "FLASH_QKV_KERNEL",
    "FLASH_FP8_BASE",
)

# The apply kwargs flash should pass for a DEFAULT run: chalk runs STANDALONE (replaces Liger) so
# its own rms_norm/swiglu/FLCE are ON alongside the gap-fillers; only the overlap/situational
# kernels (fused MLP / eval-only attn epilogue / Hopper-only FP8 base) stay off.
_DEFAULT_KWARGS = {
    "rmsnorm": True,
    "swiglu": True,
    "fused_linear_cross_entropy": True,
    "rope": True,
    "fused_lora_delta": True,
    "fused_embedding": True,
    "fused_mlp": False,
    "attn_epilogue": False,
    "fp8_frozen_base": False,
}


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
    """Default (no FLASH_* flags), post-build -> apply is called once with liger=False and the
    gap-fillers ON (rope/fused_lora_delta/fused_embedding), the overlap/situational kernels OFF."""
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
    """Gap-fillers are default-on, but if freesolo-chalk isn't installed -> no-op (returns {})."""
    _clear_flags(monkeypatch)
    monkeypatch.setitem(sys.modules, "chalk", None)  # force ImportError on `from chalk.transformers ...`
    assert install_chalk_kernels(object()) == {}


def test_flag_zero_disables_default_on_kernel(monkeypatch):
    """FLASH_<K>=0/false disables a default-ON gap-filler; the others stay on."""
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
    """A default-OFF kernel turns on with FLASH_<K>=1, alongside the default-on gap-fillers."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "1")  # opt in to the fused MLP
    monkeypatch.setenv("FLASH_FP8_BASE", "1")  # opt in to FP8 frozen base
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object())
    _, kwargs = calls[0]
    assert kwargs["fused_mlp"] is True
    assert kwargs["fp8_frozen_base"] is True
    assert kwargs["rope"] is True  # gap-fillers still on


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
    """is_chalk_enabled is True by default (gap-fillers on) and False only when all are disabled."""
    _clear_flags(monkeypatch)
    import os

    assert is_chalk_enabled(os.environ) is True
    disabled = dict.fromkeys(_DEFAULT_ON_FLAGS, "0")
    assert is_chalk_enabled(disabled) is False
    # a single opt-in flag is enough to need chalk installed
    assert is_chalk_enabled({**disabled, "FLASH_MLP_KERNEL": "1"}) is True


def test_active_kernels_filters_report():
    """active_kernels keeps only the kernels that ENGAGED (truthy, non-error) and drops liger."""
    rep = {
        "liger": False,  # chalk runs standalone — its report carries liger=False; excluded
        "rope": True,
        "fused_lora_delta": 12,  # a count is "engaged"
        "fused_embedding": False,  # fell back
        "fused_mlp": {"error": "boom"},  # errored
    }
    assert active_kernels(rep) == ["fused_lora_delta", "rope"]
    assert active_kernels({}) == []
    assert active_kernels(None) == []


def _inject_fake_chalk(monkeypatch):
    """Make ``importlib.util.find_spec('chalk')`` succeed (CPU venv has no real chalk) by inserting a
    fake module WITH a __spec__ — find_spec returns the spec of an already-imported module."""
    m = types.ModuleType("chalk")
    m.__spec__ = importlib.util.spec_from_loader("chalk", loader=None)
    monkeypatch.setitem(sys.modules, "chalk", m)


def test_flce_flag_on_default_and_override(monkeypatch):
    """flce_flag_on() = the operator's FLASH_FLCE_KERNEL setting: default-ON, OFF when set falsey."""
    monkeypatch.delenv("FLASH_FLCE_KERNEL", raising=False)
    assert flce_flag_on() is True  # default-on
    monkeypatch.setenv("FLASH_FLCE_KERNEL", "0")
    assert flce_flag_on() is False
    monkeypatch.setenv("FLASH_FLCE_KERNEL", "1")
    assert flce_flag_on() is True


def test_fused_ce_available_requires_flag_and_chalk(monkeypatch):
    """fused_ce_available() (the WORKER gate) is True ONLY when the FLCE flag is on AND chalk is
    importable. The flag alone (chalk missing/failed-to-install) is NOT enough — gating on the flag
    only would skip run_sft's large-vocab logits cap and OOM ~248k-vocab runs."""
    # Flag on (default) but chalk NOT importable -> False (the bug this fixes).
    monkeypatch.delenv("FLASH_FLCE_KERNEL", raising=False)
    monkeypatch.delitem(sys.modules, "chalk", raising=False)
    assert fused_ce_available() is False
    # Flag on AND chalk importable -> True (the normal fused path).
    _inject_fake_chalk(monkeypatch)
    assert fused_ce_available() is True
    # Flag OFF, even with chalk importable -> False (operator disabled FLCE).
    monkeypatch.setenv("FLASH_FLCE_KERNEL", "0")
    assert fused_ce_available() is False


def test_chalk_importable_reflects_find_spec(monkeypatch):
    """_chalk_importable() is a real find_spec('chalk') probe (no import side effects)."""
    from flash.engine.chalk_kernels import _chalk_importable

    monkeypatch.delitem(sys.modules, "chalk", raising=False)
    assert _chalk_importable() is False  # CPU venv genuinely has no freesolo-chalk
    _inject_fake_chalk(monkeypatch)
    assert _chalk_importable() is True
