"""flash <-> chalk wiring (CPU-safe).

chalk is install-on-call (it reads NO env vars), so flash's install_chalk_kernels must decide
*which* chalk installers to call from per-kernel FLASH_* selection flags. These tests verify the
FLASH_-flag -> installer mapping (incl. kwargs), the default no-op (no flags), the no-op when chalk
is absent, the model=None skipping of instance-only installers, and that an installer error never
aborts training.
"""

import sys
import types

from flash.engine.chalk_kernels import install_chalk_kernels

# (chalk installer name, needs_model). Mirrors the real chalk.transformers signatures:
#   install_qwen35_mlp(model=None), install_qwen35_qkv(model=None), install_lora(),
#   install_qwen35_rope(), install_qwen35_mlp_fp8(model,*,down=True),
#   install_fp8_base(model,*,attn=True,mlp=True,min_k=256), install_qwen35_embedding(model)
_INSTALLERS = {
    "install_lora": False,
    "install_qwen35_mlp": False,
    "install_qwen35_qkv": False,
    "install_qwen35_rope": False,
    "install_qwen35_mlp_fp8": True,
    "install_fp8_base": True,
    "install_qwen35_embedding": True,
}

# All FLASH_* kernel-selection flags -> every installer on.
_ALL_FLAGS = {
    "FLASH_TRITON_LORA": "1",
    "FLASH_MLP_KERNEL": "1",
    "FLASH_QKV_KERNEL": "1",
    "FLASH_ROPE_KERNEL": "1",
    "FLASH_MLP_FP8": "1",
    "FLASH_FP8_BASE": "1",
    "FLASH_EMBED_KERNEL": "1",
}


def _install_fake_chalk(monkeypatch, calls, *, raise_in=None):
    ck = types.ModuleType("chalk.transformers")

    def make(name):
        # Accept model positionally + arbitrary kwargs so we can assert what flash forwards.
        def fn(model=None, **kwargs):
            calls.append((name, model, kwargs))
            if raise_in == name:
                raise RuntimeError("boom")
            return True

        return fn

    for name in _INSTALLERS:
        setattr(ck, name, make(name))

    chalk_pkg = types.ModuleType("chalk")
    chalk_pkg.transformers = ck
    monkeypatch.setitem(sys.modules, "chalk", chalk_pkg)
    monkeypatch.setitem(sys.modules, "chalk.transformers", ck)


def _clear_flags(monkeypatch):
    for k in (
        "FLASH_TRITON_LORA",
        "FLASH_MLP_KERNEL",
        "FLASH_QKV_KERNEL",
        "FLASH_ROPE_KERNEL",
        "FLASH_MLP_FP8",
        "FLASH_MLP_FP8_DOWN",
        "FLASH_FP8_BASE",
        "FLASH_FP8_BASE_ATTN",
        "FLASH_FP8_BASE_MLP",
        "FLASH_FP8_BASE_MIN_K",
        "FLASH_EMBED_KERNEL",
    ):
        monkeypatch.delenv(k, raising=False)


def test_no_flags_calls_nothing(monkeypatch):
    """Default (no FLASH_* flag set) -> install nothing, even if chalk is importable."""
    _clear_flags(monkeypatch)
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    assert install_chalk_kernels() == {}
    assert install_chalk_kernels(object()) == {}
    assert calls == []  # chalk wasn't even consulted


def test_noop_when_chalk_absent(monkeypatch):
    """A flag is set but chalk isn't installed -> warn + no-op (returns {})."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "1")
    monkeypatch.setitem(sys.modules, "chalk", None)  # force ImportError on `import chalk.transformers`
    assert install_chalk_kernels() == {}
    assert install_chalk_kernels(object()) == {}


def test_only_selected_installers_are_called(monkeypatch):
    """Only the chalk installers whose FLASH_* flag is set get called."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "1")
    monkeypatch.setenv("FLASH_TRITON_LORA", "1")
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels()  # model=None
    names = {n for n, _, _ in calls}
    assert names == {"install_qwen35_mlp", "install_lora"}
    # unselected installers (including instance-level FP8 base/embedding) were NOT called
    assert "install_fp8_base" not in names
    assert "install_qwen35_qkv" not in names


def test_falsey_flag_value_does_not_select(monkeypatch):
    """FLASH_*=0/false/empty must NOT enable the kernel (so a leftover 0 is inert)."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "0")
    monkeypatch.setenv("FLASH_QKV_KERNEL", "false")
    monkeypatch.setenv("FLASH_TRITON_LORA", "")
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    assert install_chalk_kernels() == {}
    assert calls == []


def test_model_none_skips_instance_only_installers(monkeypatch):
    """With every flag on but model=None, class/fn-level installers run; instance-only ones skip."""
    _clear_flags(monkeypatch)
    for k, v in _ALL_FLAGS.items():
        monkeypatch.setenv(k, v)
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    res = install_chalk_kernels(None)
    names = {n for n, _, _ in calls}
    assert {"install_lora", "install_qwen35_mlp", "install_qwen35_qkv", "install_qwen35_rope"} <= names
    # instance-only installers are skipped when there's no model
    assert "install_fp8_base" not in names
    assert "install_qwen35_embedding" not in names
    assert "install_qwen35_mlp_fp8" not in names
    assert res  # non-empty result map


def test_model_present_runs_instance_installers_with_model(monkeypatch):
    """With a model, instance-level installers receive it; class-level ones are called w/o it."""
    _clear_flags(monkeypatch)
    for k, v in _ALL_FLAGS.items():
        monkeypatch.setenv(k, v)
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    sentinel = object()
    install_chalk_kernels(sentinel)
    by_name = {n: model for n, model, _ in calls}
    assert by_name["install_fp8_base"] is sentinel
    assert by_name["install_qwen35_embedding"] is sentinel
    assert by_name["install_qwen35_mlp_fp8"] is sentinel
    assert by_name["install_lora"] is None  # class-level: called without the model
    assert by_name["install_qwen35_mlp"] is None


def test_per_kernel_kwargs_are_forwarded(monkeypatch):
    """Operator FP8 scope knobs (FLASH_FP8_BASE_*, FLASH_MLP_FP8_DOWN) reach the chalk kwargs."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_FP8_BASE", "1")
    monkeypatch.setenv("FLASH_FP8_BASE_ATTN", "0")
    monkeypatch.setenv("FLASH_FP8_BASE_MLP", "1")
    monkeypatch.setenv("FLASH_FP8_BASE_MIN_K", "512")
    monkeypatch.setenv("FLASH_MLP_FP8", "1")
    monkeypatch.setenv("FLASH_MLP_FP8_DOWN", "0")
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object())
    kw = {n: kwargs for n, _, kwargs in calls}
    assert kw["install_fp8_base"] == {"attn": False, "mlp": True, "min_k": 512}
    assert kw["install_qwen35_mlp_fp8"] == {"down": False}


def test_fp8_base_uses_chalk_defaults_when_no_knobs(monkeypatch):
    """FLASH_FP8_BASE alone -> no kwargs forwarded, so chalk's own defaults apply."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_FP8_BASE", "1")
    calls = []
    _install_fake_chalk(monkeypatch, calls)
    install_chalk_kernels(object())
    kw = {n: kwargs for n, _, kwargs in calls}
    assert kw["install_fp8_base"] == {}


def test_installer_error_is_swallowed(monkeypatch):
    """An installer raising must not abort training; the error is recorded, not propagated."""
    _clear_flags(monkeypatch)
    monkeypatch.setenv("FLASH_MLP_KERNEL", "1")
    calls = []
    _install_fake_chalk(monkeypatch, calls, raise_in="install_qwen35_mlp")
    res = install_chalk_kernels(object())  # must not raise
    assert str(res["install_qwen35_mlp"]).startswith("error:")
