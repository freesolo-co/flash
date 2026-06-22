"""Worker stack selection + TRL config compat + LoRA exclusion unit tests (CPU-only)."""

from __future__ import annotations

import sys
import types

import pytest

from flash.providers.runpod.train import (
    WORKER_DEPS,
    resolve_worker_deps,
)


def test_resolve_worker_deps_default(monkeypatch):
    monkeypatch.delenv("FLASH_WORKER_DEPS", raising=False)
    # The single pinned stack is the validated default (bench/results/phase1 matrix).
    assert resolve_worker_deps() == WORKER_DEPS


def test_resolve_worker_deps_explicit_list_wins(monkeypatch):
    # Whitespace-separated; a comma is part of a PEP 440 range, not a delimiter.
    monkeypatch.setenv("FLASH_WORKER_DEPS", "torch==2.99  vllm==9.9.9   transformers>=5.6,<5.11")
    assert resolve_worker_deps() == ["torch==2.99", "vllm==9.9.9", "transformers>=5.6,<5.11"]


def test_resolve_worker_deps_json_list_supports_comma_specs(monkeypatch):
    monkeypatch.setenv(
        "FLASH_WORKER_DEPS", '["torch==2.10.0", "transformers>=5.6,<5.11", "fla==0.5.0"]'
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
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
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


def _fake_torch(monkeypatch):
    """Inject a stub ``torch`` (the CPU/server venv has no torch) exposing optim.AdamW."""

    class _AdamW:  # marker class; identity is all the tests check
        pass

    fake = types.ModuleType("torch")
    fake.optim = types.SimpleNamespace(AdamW=_AdamW)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return _AdamW


def _fake_bitsandbytes(monkeypatch):
    """Inject a stub ``bitsandbytes`` so loraplus_optimizer_cls can resolve the 8-bit class
    without a CUDA build of bnb installed."""

    class _PagedAdamW8bit:  # marker class; identity is all the test checks
        pass

    fake = types.ModuleType("bitsandbytes")
    fake.optim = types.SimpleNamespace(PagedAdamW8bit=_PagedAdamW8bit)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake)
    return _PagedAdamW8bit


def test_loraplus_optimizer_mirrors_8bit_optim(monkeypatch):
    """An `8bit` optim string -> bnb PagedAdamW8bit (LoRA+ and 8-bit state coexist), always-on."""
    worker = _import_worker(monkeypatch)
    _fake_torch(monkeypatch)
    paged = _fake_bitsandbytes(monkeypatch)
    cls, extra = worker.loraplus_optimizer_cls("paged_adamw_8bit")
    assert cls is paged
    assert extra == {}


def test_loraplus_optimizer_fp_optim_uses_adamw(monkeypatch):
    """A non-8-bit optim string keeps full-precision torch AdamW (mirrors the configured optim)."""
    worker = _import_worker(monkeypatch)
    adamw = _fake_torch(monkeypatch)
    cls, _extra = worker.loraplus_optimizer_cls("adamw_torch")
    assert cls is adamw


def test_loraplus_optimizer_bnb_missing_falls_back(monkeypatch):
    """If bitsandbytes can't be imported, fall back to fp32 AdamW (never block training)."""
    import builtins

    worker = _import_worker(monkeypatch)
    adamw = _fake_torch(monkeypatch)
    real_import = builtins.__import__

    def _no_bnb(name, *a, **k):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            raise ImportError("no bitsandbytes")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_bnb)
    assert worker.loraplus_optimizer_cls("paged_adamw_8bit")[0] is adamw


def test_grpo_no_op_failure_empty_reward_no_resume(monkeypatch):
    """Empty reward_history with no resume = the rollout scored nothing -> fail loudly (no-op run)."""
    worker = _import_worker(monkeypatch)
    assert worker._grpo_is_no_op_failure([], resume_ckpt=None, target_steps=10, steps_run=10) is True


def test_grpo_no_op_ok_when_rewards_present(monkeypatch):
    """A non-empty reward_history means the reward path ran -> never a no-op failure."""
    worker = _import_worker(monkeypatch)
    assert worker._grpo_is_no_op_failure([0.0], resume_ckpt=None, target_steps=10, steps_run=10) is False
    # An all-zero history (env returned all-zero rewards) still counts as real training.
    assert worker._grpo_is_no_op_failure([0.0, 0.0], resume_ckpt="ckpt", target_steps=10, steps_run=0) is False


def test_grpo_no_op_ok_when_resume_already_complete(monkeypatch):
    """A resume that already reached target steps has an empty fresh history but a trained policy."""
    worker = _import_worker(monkeypatch)
    assert worker._grpo_resume_already_complete("ckpt", target_steps=10, steps_run=10) is True
    # Empty history is tolerated -> NOT a no-op failure (finalize the completed policy).
    assert worker._grpo_is_no_op_failure([], resume_ckpt="ckpt", target_steps=10, steps_run=12) is False


def test_grpo_no_op_failure_resume_did_not_reach_target(monkeypatch):
    """A resume that did NOT reach the target steps with no reward is still a genuine no-op -> fail."""
    worker = _import_worker(monkeypatch)
    assert worker._grpo_resume_already_complete("ckpt", target_steps=10, steps_run=3) is False
    assert worker._grpo_is_no_op_failure([], resume_ckpt="ckpt", target_steps=10, steps_run=3) is True
    # No target steps configured can never count as a complete resume.
    assert worker._grpo_resume_already_complete("ckpt", target_steps=0, steps_run=0) is False


def test_heartbeat_commit_is_throttled(monkeypatch):
    """heartbeat() must rate-limit HF commits (per-step commits blow HF's 128/hour repo cap),
    while always committing milestone stages."""
    monkeypatch.setenv("RUN_MODE", "rl")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
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
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
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
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
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
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    assert w.optimal_attn_impl() is None


def test_liger_default_model_size_gate(monkeypatch):
    """The model-size gate (_liger_default_for_model) drives the memory-mode behaviors — sleep and
    grad checkpointing: OFF for small models (1B-class, speed mode — measured net loss PR #174) and
    ON for models ≥ ~3B where the memory headroom pays off. (chalk runs standalone and its FLCE is
    unconditionally on regardless of size; this ~3B cutoff is the old Liger threshold the memory
    decisions still share, hence the helper name.) Context-aware: a small model at long context is
    memory-bound, so memory mode flips ON."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
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


def test_make_lora_skips_pissa_on_4bit_qlora(monkeypatch):
    """PiSSA init raises on a 4-bit base (peft TypeError -> the whole run crashed), so make_lora
    must SKIP PiSSA on the QLoRA tier (catalog 9B) and keep it on the bf16/LoRA tier. rsLoRA stays
    on for both. Regression for the Qwen3.5-9B QLoRA training crash."""
    captured = {}
    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = lambda **kw: (captured.update(kw), kw)[1]
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    worker = _import_worker(monkeypatch)
    monkeypatch.setattr(worker, "lora_exclude_modules", lambda m: None)

    # 4-bit QLoRA tier -> NO PiSSA of ANY variant (any pissa_* init crashes on a 4-bit base);
    # rsLoRA still on.
    monkeypatch.setattr(worker, "model_quant", lambda m: "4bit-qlora")
    captured.clear()
    worker.make_lora("Qwen/Qwen3.5-9B")
    _init = str(captured.get("init_lora_weights", "")).lower()
    assert not _init.startswith("pissa"), f"PiSSA must be skipped on 4-bit, got {_init!r}"
    assert captured.get("use_rslora") is True

    # bf16/LoRA tier -> PiSSA on
    monkeypatch.setattr(worker, "model_quant", lambda m: "bf16")
    captured.clear()
    worker.make_lora("Qwen/Qwen3.5-0.8B")
    assert captured.get("init_lora_weights") == "pissa_niter_16"
    assert captured.get("use_rslora") is True


def test_force_vllm_backend_for_sm120(monkeypatch):
    """RTX 5090 / sm120 -> FLASHINFER pinned (PTX-independent rollout); an operator override and a
    non-sm120 GPU leave VLLM_ATTENTION_BACKEND untouched. Regression for the empty-5090-rollout."""
    import os
    import sys
    import types

    worker = _import_worker(monkeypatch)

    def _fake_torch(major):
        t = types.ModuleType("torch")
        t.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda *a: (major, 0),
        )
        return t

    # sm120, backend unset -> FLASHINFER is forced
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(12))
    assert worker.force_vllm_backend_for_sm120() == "FLASHINFER"
    assert os.environ["VLLM_ATTENTION_BACKEND"] == "FLASHINFER"

    # operator override wins (not clobbered)
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    assert worker.force_vllm_backend_for_sm120() is None
    assert os.environ["VLLM_ATTENTION_BACKEND"] == "TRITON_ATTN"

    # non-sm120 (sm90 Hopper) -> untouched
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(9))
    assert worker.force_vllm_backend_for_sm120() is None
    assert "VLLM_ATTENTION_BACKEND" not in os.environ


# ---------------------------------------------------------------------------
# MIG / NVML boot guard (assert_usable_gpu): retriable-infra detection of a partition /
# permission-restricted host before any real CUDA work (the NVML-assert crash class).
# ---------------------------------------------------------------------------
def _fake_torch_named(name, available=True):
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda *a: name,
    )
    return t


def test_assert_usable_gpu_full_card_ok(monkeypatch):
    """A normal full GPU (no MIG in the name, NVML bindings absent so no probe) passes."""
    from flash.engine.worker import perf

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_named("NVIDIA RTX A5000"))
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)
    # No pynvml installed in the offline test env -> the import inside fails (ImportError),
    # which the guard treats as "can't probe", not a failure. Name has no "MIG" -> usable.
    perf.assert_usable_gpu()  # must NOT raise


def test_assert_usable_gpu_mig_name_raises_retriable(monkeypatch):
    """A MIG slice (name contains 'MIG') raises the RETRIABLE infra error with the marker
    so the control plane resubmits on a fresh worker (the NVML-assert crash we hit live)."""
    from flash.engine.worker import perf

    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch_named("NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb"),
    )
    with pytest.raises(perf.RetriableInfraError, match=perf.RETRIABLE_INFRA_MARKER) as exc:
        perf.assert_usable_gpu()
    assert "MIG" in str(exc.value)


def test_assert_usable_gpu_no_cuda_raises_retriable(monkeypatch):
    """No CUDA device at boot -> retriable infra (resubmit on a fresh host), not a job_failed."""
    from flash.engine.worker import perf

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_named("", available=False))
    with pytest.raises(perf.RetriableInfraError, match=perf.RETRIABLE_INFRA_MARKER):
        perf.assert_usable_gpu()


def test_assert_usable_gpu_nvml_probe_failure_raises_retriable(monkeypatch):
    """An NVML probe that actively RAISES (permission-restricted MIG host whose name looks
    normal) trips the guard — this is the case nvidia-smi shows '[Insufficient Permissions]'."""
    from flash.engine.worker import perf

    monkeypatch.setitem(sys.modules, "torch", _fake_torch_named("NVIDIA RTX A5000"))

    fake_nvml = types.ModuleType("pynvml")
    fake_nvml.nvmlInit = lambda: None
    fake_nvml.nvmlShutdown = lambda: None
    fake_nvml.nvmlDeviceGetHandleByIndex = lambda i: object()

    def _boom(_h):
        raise RuntimeError("Insufficient Permissions")

    fake_nvml.nvmlDeviceGetMemoryInfo = _boom
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)
    with pytest.raises(perf.RetriableInfraError, match=perf.RETRIABLE_INFRA_MARKER):
        perf.assert_usable_gpu()


def test_retriable_infra_marker_is_an_infra_retry_marker():
    """The worker's marker phrase MUST be in the runner's infra-retry set, or a MIG host would
    be mis-classified as a non-retried job_failed (the live regression). Lock-step contract."""
    from flash.engine.worker.perf import RETRIABLE_INFRA_MARKER

    src = (
        __import__("pathlib")
        .Path(__import__("flash.runner.lifecycle", fromlist=["__file__"]).__file__)
        .read_text()
    )
    assert RETRIABLE_INFRA_MARKER in src
