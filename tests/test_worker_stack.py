"""Worker stack selection + TRL config compat + LoRA exclusion unit tests (CPU-only)."""

from __future__ import annotations

import sys
import types

from flash.providers.runpod.train import (
    WORKER_DEPS,
    resolve_worker_deps,
)


def test_resolve_worker_deps_default(monkeypatch):
    monkeypatch.delenv("AUTOSLM_WORKER_DEPS", raising=False)
    # The single pinned stack is the validated default (bench/results/phase1 matrix).
    assert resolve_worker_deps() == WORKER_DEPS


def test_resolve_worker_deps_explicit_list_wins(monkeypatch):
    # Whitespace-separated; a comma is part of a PEP 440 range, not a delimiter.
    monkeypatch.setenv("AUTOSLM_WORKER_DEPS", "torch==2.99  vllm==9.9.9   transformers>=5.6,<5.11")
    assert resolve_worker_deps() == ["torch==2.99", "vllm==9.9.9", "transformers>=5.6,<5.11"]


def test_resolve_worker_deps_json_list_supports_comma_specs(monkeypatch):
    monkeypatch.setenv(
        "AUTOSLM_WORKER_DEPS", '["torch==2.10.0", "transformers>=5.6,<5.11", "fla==0.5.0"]'
    )
    assert resolve_worker_deps() == ["torch==2.10.0", "transformers>=5.6,<5.11", "fla==0.5.0"]


def test_worker_stack_pins_qwen35_capable_versions():
    joined = " ".join(WORKER_DEPS)
    assert "vllm==0.19" in joined  # first transformers-5-compatible vllm line
    assert "transformers>=5" in joined  # qwen3_5 model types need transformers 5.x
    assert "trl>=1.6" in joined  # 1.6 adds the GRPO tools=/rollout_func multi-turn hooks
    assert "bitsandbytes" in joined  # QLoRA tier for the 35B-A3B MoE


# ---------------------------------------------------------------------------
# lora_exclude_modules: vision tower excluded for qwen3_5*, none for text models
# ---------------------------------------------------------------------------
def _import_worker(monkeypatch):
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as worker

    return worker


def _fake_transformers(monkeypatch, model_type: str):
    fake_cfg = types.SimpleNamespace(model_type=model_type)
    fake_auto = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: fake_cfg,
    )
    fake_mod = types.ModuleType("transformers")
    fake_mod.AutoConfig = fake_auto
    monkeypatch.setitem(sys.modules, "transformers", fake_mod)


def test_lora_exclude_modules_qwen35(monkeypatch):
    import re

    worker = _import_worker(monkeypatch)
    _fake_transformers(monkeypatch, "qwen3_5")
    excl = worker.lora_exclude_modules("Qwen/Qwen3.5-4B")
    assert excl is not None
    assert "visual" in excl
    # peft applies exclude_modules regex with fullmatch on the module path: leaf
    # modules under the vision tower MUST match (the earlier suffix-list form didn't,
    # which let LoRA onto visual.* and broke vLLM adapter loading).
    assert re.fullmatch(excl, "visual.blocks.0.attn.qkv")
    assert re.fullmatch(excl, "model.visual.blocks.3.mlp.linear_fc1")
    assert not re.fullmatch(excl, "model.layers.0.self_attn.q_proj")


def test_lora_exclude_modules_text_model(monkeypatch):
    worker = _import_worker(monkeypatch)
    _fake_transformers(monkeypatch, "llama")
    assert worker.lora_exclude_modules("openbmb/MiniCPM5-1B") is None


def test_heartbeat_commit_is_throttled(monkeypatch):
    """heartbeat() must rate-limit HF commits (per-step commits blow HF's 128/hour repo cap),
    while always committing milestone stages."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    calls = []
    monkeypatch.setattr(w, "hf_upload_file", lambda *a, **k: calls.append(a[1]))

    # Large interval -> only milestone + the first commit; per-step heartbeats throttled.
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 9999.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    w.heartbeat("rl_start")  # milestone -> commits
    w.heartbeat("rl_step", step=1)  # throttled
    w.heartbeat("rl_step", step=2)  # throttled
    assert calls.count("heartbeat.json") == 1

    # Zero interval -> every call commits.
    calls.clear()
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    w.heartbeat("rl_step", step=1)
    w.heartbeat("rl_step", step=2)
    assert len(calls) == 2


def test_heartbeat_hf_upload_runs_outside_lock(monkeypatch):
    """Perf regression guard: the synchronous hf_upload_file network call must run OUTSIDE
    _HB_LOCK. Holding the lock across the upload serializes the trainer's per-step reward
    callback behind the checkpoint daemon's HF commit during GRPO."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    # When hf_upload_file is invoked, the lock must be acquirable (i.e. not held).
    lock_free_during_upload = []

    def fake_upload(*a, **k):
        acquired = w._HB_LOCK.acquire(blocking=False)
        lock_free_during_upload.append(acquired)
        if acquired:
            w._HB_LOCK.release()

    monkeypatch.setattr(w, "hf_upload_file", fake_upload)
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)
    w.heartbeat("rl_start")
    assert lock_free_during_upload == [True], "hf_upload_file must run with _HB_LOCK released"


def test_heartbeat_terminal_only_mode(monkeypatch):
    """TERMINAL_ONLY mode throttles every non-terminal stage (not just rl_step) so a fan-out of
    runs sharing one HF_REPO stays under the 128-commits/hour cap; terminal done/error_* still
    always commit so the control plane never misses a transition."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    calls = []
    def _fake_upload(*a, **k):
        calls.append(a[1])
        return True  # simulate a successful commit so the throttle clock advances

    monkeypatch.setattr(w, "hf_upload_file", _fake_upload)
    monkeypatch.setattr(w, "_HB_TERMINAL_ONLY", True)
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 9999.0)
    monkeypatch.setattr(w, "_HB_LAST_UPLOAD", 0.0)

    # Short interval: terminal-only must STILL suppress non-terminal after the first, NOT
    # leak a commit once the window elapses (the bot-caught 128/hr re-breach).
    monkeypatch.setattr(w, "_HB_MIN_INTERVAL_S", 0.0)
    w.heartbeat("sft_start")  # first non-terminal -> commits (last_upload==0)
    w.heartbeat("sft_model_load")  # suppressed despite 0s interval
    w.heartbeat("sft_trained")  # suppressed
    assert len(calls) == 1
    w.heartbeat("error_sft", error="boom")  # terminal -> always commits
    w.heartbeat("done")  # terminal -> always commits
    assert calls.count("heartbeat.json") == 3


def test_optimal_attn_impl_no_cuda_is_none(monkeypatch):
    """optimal_attn_impl picks the arch-best backend for the live GPU; with no CUDA (CI) it
    leaves transformers' default (None). There is no env override."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    assert w.optimal_attn_impl() is None


def test_liger_on_requires_default_and_gpu(monkeypatch):
    """liger_on(False) is always off; liger_on(True) still needs a CUDA GPU + importable
    liger_kernel (both absent in CI), so it's off here too."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    assert w.liger_on(False) is False
    assert w.liger_on(True) is False  # no CUDA / liger_kernel in CI


def test_liger_default_model_size_gate(monkeypatch):
    """Liger default is OFF for small models (1B-class, measured net loss PR #174) and ON only
    for models ≥ ~3B where fused-CE's memory win pays off."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    # _estimate_params: ~1B vs ~4B configs
    small = types.SimpleNamespace(
        hidden_size=1536, vocab_size=130560, num_hidden_layers=24, tie_word_embeddings=False
    )
    big = types.SimpleNamespace(
        hidden_size=2560, vocab_size=151936, num_hidden_layers=36, tie_word_embeddings=False
    )
    assert w._estimate_params(small) < 3e9
    assert w._estimate_params(big) >= 3e9

    def fake_cfg(cfg):
        fake = types.SimpleNamespace(AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: cfg))
        monkeypatch.setitem(sys.modules, "transformers", fake)

    fake_cfg(small)
    assert w._liger_default_for_model("openbmb/MiniCPM5-1B") is False
    # grad checkpointing follows the same small=speed(off) / large=memory(on) principle
    assert w.grad_checkpointing_on("openbmb/MiniCPM5-1B") is False
    fake_cfg(big)
    assert w._liger_default_for_model("Qwen/Qwen3-4B") is True
    assert w.grad_checkpointing_on("Qwen/Qwen3-4B") is True
    # context-aware: a SMALL model at LONG context is memory-bound -> memory mode ON (PR #174:
    # 1B GRPO OOMs at 4096 in speed mode, fits in memory mode).
    fake_cfg(small)
    assert w._memory_mode("openbmb/MiniCPM5-1B", 512) is False
    assert w._memory_mode("openbmb/MiniCPM5-1B", 4096) is True
    assert w.grad_checkpointing_on("openbmb/MiniCPM5-1B", 4096) is True
