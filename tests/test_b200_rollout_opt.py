"""B200 GRPO rollout optimizations (CPU-only, no GPU/vLLM):

  * ``patch_trl_colocate_llm_kwargs`` — the monkeypatch that injects ``kv_cache_dtype`` /
    ``max_num_batched_tokens`` / ``enforce_eager`` / ``attention_backend`` into TRL 1.6's colocate
    ``LLM(...)`` (GRPOConfig can't express them; TRL hardcodes max_num_batched_tokens=4096 and never
    sets kv_cache_dtype, and vLLM 0.19.1 dropped the attention/compile env vars from its registry).

These exercise the pure plumbing; the actual vLLM speedup is covered by the live B200 smoke.
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import pytest

from flash.engine.worker.gpu_setup import patch_trl_colocate_llm_kwargs


class _FakeLLM:
    """Records the kwargs it was constructed with (stands in for vLLM's LLM)."""

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs):
        _FakeLLM.last_kwargs = dict(kwargs)


@pytest.fixture
def fake_vg(monkeypatch):
    """A throwaway ``trl.generation.vllm_generation`` module carrying a fake ``LLM`` symbol, so the
    monkeypatch has something to wrap WITHOUT a real vLLM (which TRL only binds when vllm is
    importable). Restored automatically by monkeypatch."""
    mod = types.ModuleType("trl.generation.vllm_generation")
    mod.LLM = _FakeLLM
    # ensure the parent packages resolve for `import trl.generation.vllm_generation`
    for name in ("trl", "trl.generation"):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = []  # mark as package
            monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, "trl.generation.vllm_generation", mod)
    _FakeLLM.last_kwargs = {}
    return mod


def test_patch_injects_kv_cache_dtype_and_overrides_max_num_batched_tokens(fake_vg):
    assert patch_trl_colocate_llm_kwargs(kv_cache_dtype="fp8", max_num_batched_tokens=8192) is True
    # TRL builds the engine with its hardcoded 4096 — our wrapper must OVERRIDE it and ADD fp8 KV.
    fake_vg.LLM(model="m", max_num_batched_tokens=4096, gpu_memory_utilization=0.5)
    assert _FakeLLM.last_kwargs["kv_cache_dtype"] == "fp8"
    assert _FakeLLM.last_kwargs["max_num_batched_tokens"] == 8192
    # untouched kwargs pass straight through
    assert _FakeLLM.last_kwargs["gpu_memory_utilization"] == 0.5
    assert _FakeLLM.last_kwargs["model"] == "m"


def test_big_card_batched_tokens_default_covers_engine_length():
    """A big-card GRPO run whose [train].max_length exceeds 8192 must NOT pin
    max_num_batched_tokens to a fixed 8192: vLLM validates the scheduler token budget against the
    engine's max model length at colocate init, so 8192 < max_model_len (e.g. 16k) fails engine
    startup. The big-card default must therefore derive from vllm_max_len, not a fixed 8192. (The
    selection lives deep inside run_rl behind torch.cuda probing, so assert on its source — same
    approach as the other run_rl wiring tests.)"""
    import inspect
    import re

    from flash.engine.worker.rl import run_rl

    src = inspect.getsource(run_rl)
    m = re.search(r"_mnbt = (.+?) if _card_gb >= 140 else None", src)
    assert m, "big-card _mnbt default assignment not found in run_rl"
    expr = m.group(1)
    assert "vllm_max_len" in expr, f"big-card batched-token default must scale to the engine length, got: {expr}"
    assert "8192" in expr, f"default must keep the 8192 floor for short contexts, got: {expr}"
    assert "max(" in expr, f"default must take the larger of 8192 and the engine length, got: {expr}"


def test_patch_is_idempotent(fake_vg):
    assert patch_trl_colocate_llm_kwargs(kv_cache_dtype="fp8") is True
    first = fake_vg.LLM  # the wrapper
    # a second call must NOT double-wrap (re-entrant guard)
    assert patch_trl_colocate_llm_kwargs(kv_cache_dtype="fp8") is True
    assert fake_vg.LLM is first


def test_patch_no_overrides_is_a_noop(fake_vg):
    assert patch_trl_colocate_llm_kwargs() is False
    # the LLM symbol is left untouched (still the raw fake, not a wrapper)
    assert fake_vg.LLM is _FakeLLM


def test_patch_kv_only_does_not_set_max_num_batched_tokens(fake_vg):
    assert patch_trl_colocate_llm_kwargs(kv_cache_dtype="fp8") is True
    fake_vg.LLM(model="m", max_num_batched_tokens=4096)
    assert _FakeLLM.last_kwargs["kv_cache_dtype"] == "fp8"
    # we did NOT pass max_num_batched_tokens -> TRL's 4096 is preserved
    assert _FakeLLM.last_kwargs["max_num_batched_tokens"] == 4096


def test_patch_injects_enforce_eager(fake_vg):
    # Codex MsOqu: VLLM_TORCH_COMPILE_LEVEL was dropped from vLLM 0.19.1's env registry, so eager is
    # forced via the LLM(...) enforce_eager kwarg instead — assert it reaches the colocate engine.
    assert patch_trl_colocate_llm_kwargs(enforce_eager=True) is True
    fake_vg.LLM(model="m")
    assert _FakeLLM.last_kwargs["enforce_eager"] is True


def test_patch_injects_attention_backend(fake_vg):
    # Codex MsOqv: VLLM_ATTENTION_BACKEND was dropped from vLLM 0.19.1's env registry, so the sm120
    # backend is pinned via the LLM(...) attention_backend kwarg (EngineArgs coerces the bare name).
    assert patch_trl_colocate_llm_kwargs(attention_backend="TRITON_ATTN") is True
    fake_vg.LLM(model="m")
    assert _FakeLLM.last_kwargs["attention_backend"] == "TRITON_ATTN"


def test_patch_overrides_accumulate_across_calls(fake_vg):
    """run_rl injects the attention backend (sm120), the KV/prefill knobs, and eager mode at three
    SEPARATE points. Each call must MERGE into the one wrapper — the wrap-once guard must not drop the
    later calls' kwargs (the bug a naive idempotent-return would reintroduce)."""
    assert patch_trl_colocate_llm_kwargs(attention_backend="FLASHINFER") is True
    first = fake_vg.LLM  # wrapper installed by the first call
    # later calls extend the same wrapper (no re-wrap) and their kwargs still take effect
    assert patch_trl_colocate_llm_kwargs(kv_cache_dtype="fp8", max_num_batched_tokens=8192) is True
    assert patch_trl_colocate_llm_kwargs(enforce_eager=True) is True
    assert fake_vg.LLM is first
    fake_vg.LLM(model="m", max_num_batched_tokens=4096)
    assert _FakeLLM.last_kwargs["attention_backend"] == "FLASHINFER"
    assert _FakeLLM.last_kwargs["kv_cache_dtype"] == "fp8"
    assert _FakeLLM.last_kwargs["max_num_batched_tokens"] == 8192
    assert _FakeLLM.last_kwargs["enforce_eager"] is True


def test_patch_skipped_when_no_llm_symbol(monkeypatch):
    """If vLLM isn't importable, TRL never binds the module-global LLM -> safe no-op (not a crash)."""
    mod = types.ModuleType("trl.generation.vllm_generation")  # no LLM attribute
    for name in ("trl", "trl.generation"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, "trl.generation.vllm_generation", mod)
    assert patch_trl_colocate_llm_kwargs(kv_cache_dtype="fp8") is False
