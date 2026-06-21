"""Worker stack selection + TRL config compat + LoRA exclusion unit tests (CPU-only)."""

from __future__ import annotations

import sys
import types

from flash.providers.runpod.train import (
    WORKER_DEPS,
    resolve_worker_deps,
)


def test_resolve_worker_deps_default(monkeypatch):
    monkeypatch.delenv("FLASH_WORKER_DEPS", raising=False)
    # The single pinned stack is the validated default (bench/results/phase1 matrix).
    assert resolve_worker_deps() == WORKER_DEPS


def test_gdn_fastpath_deps_present_and_kept_on_hopper(monkeypatch):
    """The GDN fast-path stack (fla-from-git + tilelang + pinned apache-tvm-ffi) is baked in, and
    fla is KEPT on Hopper (sm90) — the #640 fix is fla's tilelang backend, not dropping fla."""
    monkeypatch.delenv("FLASH_WORKER_DEPS", raising=False)
    monkeypatch.delenv("FLASH_WORKER_EXTRA_DEPS", raising=False)
    joined = " ".join(WORKER_DEPS)
    assert (
        "git+https://github.com/fla-org/flash-linear-attention" in joined
    )  # complete fla, not the broken PyPI stub
    assert any(
        d.startswith("tilelang==") for d in WORKER_DEPS
    )  # correct GDN backend on Triton>=3.4, PINNED for reproducibility
    assert any(
        d.startswith("apache-tvm-ffi==0.1.11") for d in WORKER_DEPS
    )  # pin (0.1.12 aborts tilelang import)
    # fla must NOT be dropped on Hopper anymore (it was, pre-fix).
    deps_h100 = resolve_worker_deps("H100")
    assert any("flash-linear-attention" in d for d in deps_h100), (
        "fla must be kept on Hopper for the tilelang fast path"
    )


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


def test_liger_on_requires_default_and_gpu(monkeypatch):
    """liger_on(False) is always off; liger_on(True) still needs a CUDA GPU + importable
    liger_kernel (both absent in CI), so it's off here too."""
    monkeypatch.setenv("RUN_MODE", "sft")
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    sys.modules.pop("flash.engine.worker", None)
    import flash.engine.worker as w

    assert w.liger_on(False) is False
    assert w.liger_on(True) is False  # no CUDA / liger_kernel in CI


def test_liger_default_model_size_gate(monkeypatch):
    """Liger default is OFF for small models (1B-class, measured net loss PR #174) and ON only
    for models ≥ ~3B where fused-CE's memory win pays off."""
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
# Hopper fla GDN fast-path fallback: when the healthy fla+tilelang stack can't be
# assembled (probe `ok` false), fla must be DISABLED (physically removed) so transformers'
# is_fla_available() gate flips off and the model uses the correct pure-PyTorch delta rule
# instead of fla's broken Triton>=3.4 GDN chunk_bwd (fla #640). A print alone is not enough.
# ---------------------------------------------------------------------------
def _hopper_torch():
    """A stub ``torch`` that looks like Hopper (sm90) with CUDA available."""
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda *a: (9, 0),
    )
    return t


class _FakeCompleted:
    """Minimal stand-in for subprocess.CompletedProcess so _pip can read .returncode."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode


def _patch_hopper_stack(
    monkeypatch,
    *,
    pip_rc: int = 0,
    find_spec_ok: bool = True,
    tvm_ffi_version: str | None = "0.1.11",
    tilelang_version: str | None = "0.1.11",
    record_pip: list[str] | None = None,
):
    """Wire the perf helper's external touchpoints for the Hopper fast-path tests and return the
    list that records _remove_fla_from_disk (fla-disable) calls.

    * ``pip_rc`` -> the return code every mocked ``pip install`` reports (non-zero = failed install).
    * ``find_spec_ok`` -> whether the post-install import probe finds fla/fla.modules/tilelang.
    * ``tvm_ffi_version`` -> what importlib.metadata.version('apache-tvm-ffi') reports (None=absent).
    * ``tilelang_version`` -> what importlib.metadata.version('tilelang') reports (None=absent). The
      helper gates the tilelang (re)install AND the final ``ok`` on this exact version, so a value
      != the pin models a present-but-wrong-version stack.
    * ``record_pip`` -> if given, every mocked ``pip install`` appends its joined spec args here (so
      a test can assert WHICH packages were reinstalled).
    """
    import importlib.metadata
    import importlib.util
    import subprocess

    from flash.engine.worker import perf

    monkeypatch.setitem(sys.modules, "torch", _hopper_torch())

    # subprocess is imported locally inside the helper, so patch the real module. Return a fake
    # CompletedProcess so _pip can read .returncode (the install-success gate).
    def _fake_run(cmd, *a, **k):
        if record_pip is not None:
            # cmd == [sys.executable, "-m", "pip", "install", "-q", *specs]
            record_pip.append(" ".join(str(c) for c in cmd[5:]))
        return _FakeCompleted(pip_rc)

    monkeypatch.setattr(subprocess, "run", _fake_run, raising=True)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        (lambda name: object()) if find_spec_ok else (lambda name: None),
        raising=True,
    )

    def _fake_version(dist: str) -> str:
        ver = {"apache-tvm-ffi": tvm_ffi_version, "tilelang": tilelang_version}.get(dist)
        if ver is None:
            raise importlib.metadata.PackageNotFoundError(dist)
        return ver

    monkeypatch.setattr(importlib.metadata, "version", _fake_version, raising=True)

    removed: list[int] = []
    monkeypatch.setattr(
        perf, "_remove_fla_from_disk", lambda: (removed.append(1), (["/x/fla"], False))[1]
    )
    return perf, removed


def test_hopper_fla_fallback_disables_fla_when_stack_unavailable(monkeypatch):
    """Probe fails (fla absent after install attempt) -> _remove_fla_from_disk IS called so the
    fla gate flips off and the worker uses the pure-PyTorch delta path (not the broken fla kernel)."""
    perf, removed = _patch_hopper_stack(monkeypatch, find_spec_ok=False)

    perf._ensure_fla_fastpath_on_hopper()

    # The fallback gating actually fired: fla was disabled (the gate, not just a print).
    assert removed, "fallback must DISABLE fla (call _remove_fla_from_disk) when ok is false"


def test_hopper_fla_fallback_when_install_fails(monkeypatch):
    """A FAILED pip install (rc!=0) must flip the gate off + disable fla EVEN IF find_spec still
    succeeds on a stale/partial resident copy — a failed install is not silently treated as healthy
    (Copilot review on perf.py:~487). Without the rc check, find_spec alone would wrongly keep fla.
    A wrong tvm-ffi version makes the helper ATTEMPT the pinned reinstall (the conditional-install
    gate), so pip_rc=1 exercises the failed-install path."""
    perf, removed = _patch_hopper_stack(
        monkeypatch, pip_rc=1, find_spec_ok=True, tvm_ffi_version="0.1.12"
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert removed, "a failed install (rc!=0) must DISABLE fla even when find_spec passes"


def test_hopper_fla_fallback_when_tvm_ffi_pin_did_not_land(monkeypatch):
    """tilelang's apache-tvm-ffi~=0.1.0 range can keep the BROKEN 0.1.12 (find_spec-importable but
    aborts `import tilelang`). The gate verifies the RESOLVED tvm-ffi version, so a wrong version
    disables fla even though every install 'succeeded' and every module imports."""
    perf, removed = _patch_hopper_stack(
        monkeypatch, pip_rc=0, find_spec_ok=True, tvm_ffi_version="0.1.12"
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert removed, "wrong resolved apache-tvm-ffi version must DISABLE fla (pin didn't land)"


def test_hopper_fla_kept_when_stack_healthy(monkeypatch):
    """Success path: every install exits 0, fla + fla.modules + tilelang all import, AND the
    resolved apache-tvm-ffi is exactly the pin -> fla is KEPT (the fast path is engaged, not
    removed)."""
    perf, removed = _patch_hopper_stack(
        monkeypatch, pip_rc=0, find_spec_ok=True, tvm_ffi_version="0.1.11"
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert not removed, "healthy stack must KEEP fla (no disable on the success path)"


def test_hopper_tilelang_present_but_wrong_version_is_reinstalled(monkeypatch):
    """Regression (Copilot review on perf.py:~511): a DIFFERENT tilelang already resident (a job or
    the base image carries one) must NOT be treated as healthy. The helper gates on the installed
    version, so it (re)installs the exact pin; once the pin lands fla is KEPT."""
    pip_calls: list[str] = []
    # Resident wrong version first; after the (mocked) reinstall the metadata reports the pin.
    perf, removed = _patch_hopper_stack(
        monkeypatch,
        pip_rc=0,
        find_spec_ok=True,
        tvm_ffi_version="0.1.11",
        tilelang_version="0.1.11",  # post-reinstall resolved version
        record_pip=pip_calls,
    )
    # Make the FIRST _ver('tilelang') read (the install gate) see a stale wrong version, while the
    # final ok-gate read sees the pin — i.e. the reinstall corrected it.
    import importlib.metadata as _md

    gate_reads = iter(["0.1.9"])  # first read = stale; later reads -> pin (reinstall corrected it)
    orig_version = _md.version

    def _versioned(dist: str) -> str:
        if dist == "tilelang":
            return next(gate_reads, "0.1.11")
        return orig_version(dist)

    monkeypatch.setattr(_md, "version", _versioned, raising=True)

    perf._ensure_fla_fastpath_on_hopper()

    assert any(c.startswith("tilelang==0.1.11") for c in pip_calls), (
        f"present-but-wrong tilelang must trigger a pinned reinstall, got pip calls: {pip_calls}"
    )
    assert not removed, "after the pin reinstall lands, fla must be KEPT"


def test_hopper_tilelang_wrong_version_persists_disables_fla(monkeypatch):
    """If the resident tilelang is the WRONG version AND the reinstall doesn't land the pin (version
    still != pin), the final gate must DISABLE fla — the uncertain GDN backend is not trusted."""
    pip_calls: list[str] = []
    perf, removed = _patch_hopper_stack(
        monkeypatch,
        pip_rc=0,
        find_spec_ok=True,
        tvm_ffi_version="0.1.11",
        tilelang_version="0.1.9",  # wrong version throughout (reinstall didn't correct it)
        record_pip=pip_calls,
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert any(c.startswith("tilelang==0.1.11") for c in pip_calls), (
        "wrong resident tilelang must still attempt the pinned reinstall"
    )
    assert removed, "tilelang version != pin after install must DISABLE fla (pin didn't land)"


def test_hopper_tvm_ffi_pip_skipped_when_pin_already_present(monkeypatch):
    """Regression (Copilot review on perf.py:~521): when the EXACT apache-tvm-ffi pin is already
    resident AND tilelang was NOT (re)installed this invocation, the helper must SKIP the tvm-ffi
    pip — re-running it unconditionally adds avoidable cold-start latency and could spuriously
    disable fla on a transient network/resolver hiccup. The ok gate still re-verifies the version,
    so fla stays KEPT."""
    pip_calls: list[str] = []
    perf, removed = _patch_hopper_stack(
        monkeypatch,
        pip_rc=0,
        find_spec_ok=True,
        tvm_ffi_version="0.1.11",  # exact pin already resident
        tilelang_version="0.1.11",  # exact pin -> tilelang NOT reinstalled this invocation
        record_pip=pip_calls,
    )

    perf._ensure_fla_fastpath_on_hopper()

    assert not any("apache-tvm-ffi" in c for c in pip_calls), (
        "tvm-ffi pin already resident + tilelang not reinstalled must SKIP the tvm-ffi pip, "
        f"got pip calls: {pip_calls}"
    )
    assert not removed, "the pin being resident is the healthy path -> fla must be KEPT"


def test_hopper_outer_exception_disables_fla(monkeypatch):
    """Regression (Copilot review on perf.py:~580): an unexpected error mid-setup (AFTER the Hopper
    check passes) must FAIL-CLOSED — best-effort disable fla so transformers can't engage the broken
    Triton GDN path (#640) on a half-configured fla. The outer handler must call _remove_fla_from_disk
    and never re-raise."""
    import importlib

    perf, removed = _patch_hopper_stack(
        monkeypatch, pip_rc=0, find_spec_ok=True, tvm_ffi_version="0.1.11"
    )

    # Make a call inside the try (after the Hopper guard + installs) blow up. invalidate_caches runs
    # right before the import-probe gate, so this drives the outer `except`.
    def _boom() -> None:
        raise RuntimeError("invalidate_caches exploded")

    monkeypatch.setattr(importlib, "invalidate_caches", _boom, raising=True)

    # Must NOT propagate (the worker keeps running on the pure-PyTorch delta path).
    perf._ensure_fla_fastpath_on_hopper()

    assert removed, "outer exception path must FAIL-CLOSED: disable fla (call _remove_fla_from_disk)"


def test_non_hopper_fla_fastpath_is_noop(monkeypatch):
    """On a non-Hopper arch (fla's Triton kernel is correct there) the helper is a no-op — it must
    NOT touch fla at all (neither install nor disable)."""
    import importlib.util
    import subprocess

    from flash.engine.worker import perf

    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda *a: (12, 0),  # Blackwell consumer (sm120) — not Hopper
    )
    monkeypatch.setitem(sys.modules, "torch", t)
    touched: list[str] = []
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: touched.append("pip"), raising=True
    )
    monkeypatch.setattr(
        perf, "_remove_fla_from_disk", lambda: (touched.append("remove"), ([], False))[1]
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None, raising=True)

    perf._ensure_fla_fastpath_on_hopper()

    assert touched == [], "non-Hopper must be a no-op (don't install or disable fla)"


def test_fla_git_pin_is_consistent_and_pinned():
    """The fla git dependency is PINNED to an exact commit (not the moving default branch) and the
    SAME SHA is used in both WORKER_DEPS and Dockerfile.worker (worker venv == baked image)."""
    import pathlib
    import re

    spec = next(d for d in WORKER_DEPS if "flash-linear-attention" in d)
    m = re.search(r"flash-linear-attention\.git@([0-9a-f]{40})\b", spec)
    assert m, f"fla dep must be pinned to a 40-char commit SHA, got: {spec!r}"
    deps_sha = m.group(1)

    # repo root: tests/ -> repo root is the parent.
    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile.worker").read_text()
    dm = re.search(r"flash-linear-attention\.git@([0-9a-f]{40})\b", dockerfile)
    assert dm, "Dockerfile.worker fla install must be pinned to a 40-char commit SHA"
    assert dm.group(1) == deps_sha, (
        "Dockerfile.worker fla SHA must match WORKER_DEPS so the baked image and the worker "
        f"venv agree (deps={deps_sha}, dockerfile={dm.group(1)})"
    )

    # The worker's runtime fla reinstall (perf._ensure_fla_fastpath_on_hopper) must use the SAME pin —
    # an unpinned reinstall would pull the moving default branch and defeat reproducibility.
    perf_src = (root / "flash" / "engine" / "worker" / "perf.py").read_text()
    # The URL is built via implicit string concatenation across lines:
    #   "...flash-linear-attention.git"\n        "@<sha>" — so allow quotes/newline/space between.
    pm = re.search(r"flash-linear-attention\.git[\"'\s]*@([0-9a-f]{40})\b", perf_src)
    assert pm, "perf.py runtime fla reinstall must be pinned to a 40-char commit SHA"
    assert pm.group(1) == deps_sha, (
        f"perf.py fla SHA must match WORKER_DEPS (deps={deps_sha}, perf={pm.group(1)})"
    )


def test_tilelang_pin_is_consistent_and_pinned():
    """tilelang (the Hopper GDN correctness backend) is PINNED to an exact version (not unversioned)
    and the SAME pin is used in WORKER_DEPS, Dockerfile.worker, and perf.py's runtime reinstall, so
    cold-start installs / image rebuilds / runtime reinstalls all resolve the identical backend
    (Copilot review on deps.py:~70)."""
    import pathlib
    import re

    spec = next(d for d in WORKER_DEPS if d.split("==")[0].split(">")[0].strip() == "tilelang")
    m = re.match(r"tilelang==([0-9][0-9A-Za-z.\-]*)$", spec.strip())
    assert m, f"tilelang must be pinned to an exact version (tilelang==X.Y.Z), got: {spec!r}"
    pin = m.group(1)

    root = pathlib.Path(__file__).resolve().parent.parent
    dockerfile = (root / "Dockerfile.worker").read_text()
    dm = re.search(r'"tilelang==([0-9][0-9A-Za-z.\-]*)"', dockerfile)
    assert dm, "Dockerfile.worker must install a PINNED tilelang==X.Y.Z (not unversioned)"
    assert dm.group(1) == pin, (
        f"Dockerfile.worker tilelang pin must match WORKER_DEPS (deps={pin}, dockerfile={dm.group(1)})"
    )

    perf_src = (root / "flash" / "engine" / "worker" / "perf.py").read_text()
    # perf.py builds the spec via an f-string `f"tilelang=={TILELANG_PIN}"`, so assert the constant.
    pm = re.search(r'TILELANG_PIN\s*=\s*"([0-9][0-9A-Za-z.\-]*)"', perf_src)
    assert pm, "perf.py must define a pinned TILELANG_PIN constant for the runtime reinstall"
    assert pm.group(1) == pin, (
        f"perf.py TILELANG_PIN must match WORKER_DEPS (deps={pin}, perf={pm.group(1)})"
    )
