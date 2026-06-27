"""GPU/backend setup the worker runs at boot or just before TRL builds the rollout engine.

Run-scoped state (``PHASE``/``JOB_SPEC``) is read through the worker package at CALL time so a
test's ``monkeypatch.setattr(worker, ...)`` reaches these readers."""

from __future__ import annotations

import os

from flash.engine.worker._pkg import W as _w


def force_vllm_backend_for_sm120() -> str | None:
    """On RTX 5090 / consumer Blackwell (sm120), force a PTX-independent vLLM attention backend.

    vLLM's default rollout backend is flash-attn, whose PRE-BUILT PTX needs a newer driver JIT than
    many 5090 RunPod hosts have — when the JIT fails the colocated rollout silently produces NO
    completions (empty reward_history, ~1.4 s "done"; a whole 22-run sweep hit this on every 5090).
    FLASHINFER is vLLM's Blackwell-native backend (no flash-attn PTX dependency) and trains on a 5090
    (measured: FLASHINFER/TORCH_SDPA/TRITON_ATTN all train, ~116 s). This mirrors the trainer's
    cuDNN-SDPA forcing on sm120 (``_attn_impl_for_capability``). The GRPO no-op guard remains the
    backstop. Returns the backend set (None if not sm120). Fixed — no operator override."""
    try:
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 12:
            return None
    except Exception as e:
        print("[rl] sm120 vLLM backend probe skipped:", e)
        return None
    os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    print(
        "[rl] sm120 (RTX 5090): VLLM_ATTENTION_BACKEND=FLASHINFER (flash-attn PTX is unreliable "
        "on consumer Blackwell hosts -> empty-rollout failures)"
    )
    return "FLASHINFER"


def finalize_alloc_conf_for_sleep() -> None:
    """Sync the CUDA allocator conf with the worker's RESOLVED vLLM sleep default (RL runs only).

    The launcher (providers/*/train.py build_worker_env) picks the sleep-SAFE non-expandable
    PYTORCH_ALLOC_CONF for RL before this process starts, but it can't know the GRPO sleep decision:
    for a small model the worker resolves sleep OFF (the speed default), so the non-expandable conf
    is safe but fragments a long colocate run. Here (we have the model config + GPU) we resolve the
    SAME deterministic sleep default (``resolve_grpo_sleep_mode`` — the one gate run_rl uses) and, if
    sleep is OFF, switch to expandable_segments — which only crashes WITH sleep on, a case we've ruled
    out. PYTORCH_ALLOC_CONF is read lazily at the first CUDA allocation, so this must run before any
    allocation (it does — called at boot)."""
    if _w.PHASE != "rl":
        return
    try:
        from flash.engine.worker.grpo import resolve_grpo_sleep_mode

        sleep_on, _ctx, _card_gb = resolve_grpo_sleep_mode()
        if not sleep_on:  # sleep resolves OFF -> expandable is safe + better
            conf = "expandable_segments:True"
            os.environ["PYTORCH_ALLOC_CONF"] = conf
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
            print(f"[alloc] sleep resolves OFF -> {conf} (anti-fragmentation, matches worker gate)")
        else:
            print("[alloc] sleep resolves ON -> keeping launcher's non-expandable conf")
    except Exception as e:
        print("[alloc] auto-conf skipped:", e)
