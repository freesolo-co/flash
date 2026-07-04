"""flash <-> chalk wiring (CPU-safe).

flash applies chalk standalone via ``apply_chalk_kernel_to_qwen35(model, liger=False, ...)`` on the
POST-build pass (chalk patches the live module). Kernel selection is FIXED: training kernels are ON,
eval-only / opt-in tradeoff kernels are OFF, with NO env override. These tests verify: the fixed
kwargs are passed; the selection ignores any FLASH_* env var; the pre-build pass (model=None) is a
no-op; the no-op when chalk is absent; and that a chalk apply error never aborts training.
"""

import sys
import types

from flash.engine.chalk_kernels import (
    active_kernels,
    chalk_fused_ce_available,
    install_chalk_kernels,
)

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
    monkeypatch.setitem(
        sys.modules, "chalk", None
    )  # force ImportError on `from chalk.transformers ...`
    assert install_chalk_kernels(object()) == {}


def test_chalk_fused_ce_available_requires_import_and_supported_model(monkeypatch):
    """SFT batch sizing can assume fused CE only for supported models with chalk importable."""
    monkeypatch.setitem(sys.modules, "chalk", None)
    assert chalk_fused_ce_available("Qwen/Qwen3.5-4B") is False

    calls = []
    _install_fake_chalk(monkeypatch, calls)
    assert chalk_fused_ce_available("Qwen/Qwen3.5-4B") is True
    assert chalk_fused_ce_available("Qwen/Qwen3.6-35B-A3B") is True
    assert chalk_fused_ce_available("openbmb/MiniCPM5-1B") is False


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


def test_kernels_match_real_chalk_signature():
    """Every _KERNELS key must be a real parameter of chalk's apply_chalk_kernel_to_qwen35.

    The other tests inject a fake chalk that accepts **kwargs, so they cannot catch a _KERNELS key
    the REAL chalk rejects (a stray key makes install_chalk_kernels TypeError -> swallow -> silently
    train on eager). This guards against that drift against the installed chalk; it is skipped when
    freesolo-chalk is not installed in the test env.
    """
    import inspect

    import pytest

    from flash.engine.chalk_kernels import _KERNELS

    try:
        from chalk.transformers import apply_chalk_kernel_to_qwen35
    except Exception:
        pytest.skip("freesolo-chalk not installed in this test env")

    accepted = set(inspect.signature(apply_chalk_kernel_to_qwen35).parameters)
    passed = {k for k, _ in _KERNELS}
    stray = passed - accepted
    assert not stray, (
        f"_KERNELS passes keys chalk rejects (would TypeError -> no-op -> eager): {sorted(stray)}"
    )


# --- FP8 frozen-base training (precision="fp8") ---------------------------------------------------


def _install_fake_chalk_fp8(monkeypatch, calls, *, report=None, no_wcache_param=True):
    """Fake chalk whose apply has an EXPLICIT fp8 signature so ``_fp8_kwargs`` feature-detection
    (via inspect.signature) can see ``fp8_no_wcache`` — the generic ``**kwargs`` fake hides it."""
    ck = types.ModuleType("chalk.transformers")

    if no_wcache_param:

        def apply_chalk_kernel_to_qwen35(
            model, *, liger=False, fp8_frozen_base=False, fp8_no_wcache=False, **kwargs
        ):
            kwargs.update(liger=liger, fp8_frozen_base=fp8_frozen_base, fp8_no_wcache=fp8_no_wcache)
            calls.append((model, kwargs))
            return report if report is not None else {}
    else:

        def apply_chalk_kernel_to_qwen35(model, *, liger=False, fp8_frozen_base=False, **kwargs):
            kwargs.update(liger=liger, fp8_frozen_base=fp8_frozen_base)
            calls.append((model, kwargs))
            return report if report is not None else {}

    ck.apply_chalk_kernel_to_qwen35 = apply_chalk_kernel_to_qwen35
    chalk_pkg = types.ModuleType("chalk")
    chalk_pkg.transformers = ck
    monkeypatch.setitem(sys.modules, "chalk", chalk_pkg)
    monkeypatch.setitem(sys.modules, "chalk.transformers", ck)


def test_fp8_false_keeps_frozen_base_off(monkeypatch):
    """Default precision (bf16): fp8_frozen_base stays OFF — unchanged from the fixed selection."""
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object(), fp8=False)
    _, kwargs = calls[0]
    assert kwargs["fp8_frozen_base"] is False
    assert "fp8_no_wcache" not in kwargs  # never added on the bf16 path


def test_fp8_true_enables_frozen_base_and_no_wcache(monkeypatch):
    """precision=fp8: fp8_frozen_base flips ON and (chalk supports it) fp8_no_wcache is added
    (baseline-memory mode), while every other fixed kernel is untouched."""
    calls = []
    _install_fake_chalk_fp8(monkeypatch, calls)
    install_chalk_kernels(object(), fp8=True)
    _, kwargs = calls[0]
    assert kwargs["fp8_frozen_base"] is True
    assert kwargs["fp8_no_wcache"] is True


def test_fp8_no_wcache_feature_detected(monkeypatch):
    """An older chalk without ``fp8_no_wcache`` must NOT get the kwarg (else TypeError -> whole
    kernel stack silently disabled); it still enables fp8_frozen_base (cached-weight mode)."""
    calls = []
    _install_fake_chalk_fp8(monkeypatch, calls, no_wcache_param=False)
    rep = install_chalk_kernels(object(), fp8=True)
    _, kwargs = calls[0]
    assert kwargs["fp8_frozen_base"] is True
    assert "fp8_no_wcache" not in kwargs  # gracefully omitted
    assert rep is not None  # apply succeeded, not swallowed


def test_fused_ce_false_disables_fused_cross_entropy(monkeypatch):
    """SFT passes fused_ce=False (chalk's None-logits fused-CE crashes TRL SFTTrainer sans liger);
    only fused_linear_cross_entropy flips off, every other kernel is untouched, fp8 still composes."""
    calls = []
    _install_fake_chalk_fp8(monkeypatch, calls)
    install_chalk_kernels(object(), fp8=True, fused_ce=False)
    _, kwargs = calls[0]
    assert kwargs["fused_linear_cross_entropy"] is False
    assert kwargs["fp8_frozen_base"] is True  # fp8 is independent of fused-CE
    assert kwargs["rope"] is True  # other kernels unaffected
    assert kwargs["swiglu"] is True


def test_fused_ce_default_on(monkeypatch):
    """GRPO uses the default (fused_ce=True): fused_linear_cross_entropy stays ON."""
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object())  # default fused_ce=True
    _, kwargs = calls[0]
    assert kwargs["fused_linear_cross_entropy"] is True


def test_fp8_env_flag_still_cannot_enable(monkeypatch):
    """Only the explicit fp8= param enables FP8; a leftover FLASH_FP8_BASE env stays inert."""
    monkeypatch.setenv("FLASH_FP8_BASE", "1")
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object())  # no fp8= -> default bf16
    _, kwargs = calls[0]
    assert kwargs["fp8_frozen_base"] is False


def test_fp8_base_engaged_reads_installed_count():
    """fp8_base_engaged reflects REAL engagement: installed>0 True; a no-op installed:0 report
    (non-FP8 GPU / unsupported model) or an error report is NOT engaged."""
    from flash.engine.chalk_kernels import fp8_base_engaged

    assert fp8_base_engaged({"fp8_frozen_base": {"installed": 42, "attn": 24, "mlp": 18}}) is True
    assert fp8_base_engaged({"fp8_frozen_base": {"installed": 0}}) is False
    assert fp8_base_engaged({"fp8_frozen_base": {"error": "boom"}}) is False
    assert fp8_base_engaged({"fp8_frozen_base": True}) is True
    assert fp8_base_engaged({"fp8_frozen_base": False}) is False
    assert fp8_base_engaged({}) is False
    assert fp8_base_engaged(None) is False


def test_fp8_kwargs_match_real_chalk_signature():
    """The fp8 kwargs flash sends (fp8_frozen_base + fp8_no_wcache) must be REAL chalk params, or
    the fp8 path TypeErrors -> swallowed -> silently trains eager bf16. Skipped without chalk."""
    import inspect

    import pytest

    try:
        from chalk.transformers import apply_chalk_kernel_to_qwen35
    except Exception:
        pytest.skip("freesolo-chalk not installed in this test env")

    accepted = set(inspect.signature(apply_chalk_kernel_to_qwen35).parameters)
    # fp8_frozen_base is the hard requirement; fp8_no_wcache is feature-detected so only warn-worthy.
    assert "fp8_frozen_base" in accepted, "chalk dropped fp8_frozen_base — flash fp8 path is dead"
