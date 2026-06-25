"""GPU/backend setup the worker runs at boot or just before TRL builds the rollout engine.

Run-scoped state (``PHASE``/``JOB_SPEC``) is read through the worker package at CALL time so a
test's ``monkeypatch.setattr(worker, ...)`` reaches these readers."""

from __future__ import annotations

import os

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import grpo_sleep_mode


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
    SAME deterministic sleep default (``grpo_sleep_mode``, exactly run_rl's gate) and, if sleep is
    OFF, switch to expandable_segments — which only crashes WITH sleep on, a case we've just ruled
    out. PYTORCH_ALLOC_CONF is read lazily at the first CUDA allocation, so this must run before any
    allocation (it does — called at boot)."""
    if _w.PHASE != "rl":
        return
    try:
        model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else ""
        # Resolve the sleep decision EXACTLY as run_rl does (grpo_sleep_mode: the size/context gate
        # PLUS the resident-fit check against the live card), so the alloc conf matches the sleep
        # mode the trainer will actually use.
        _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None
        ctx = 0
        try:
            if _t and _t.max_length:
                ctx = int(_t.max_length)
        except Exception:
            ctx = 0
        card_gb = 0.0
        try:
            import torch as _torch_card

            if _torch_card.cuda.is_available():
                card_gb = _torch_card.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            card_gb = 0.0
        sleep_on = grpo_sleep_mode(
            model_id,
            max_length=ctx,
            group_size=int(_t.group_size) if _t and _t.group_size else 8,
            max_tokens=(_t.max_tokens if _t else None),
            lora_rank=int(_t.lora_rank) if _t and _t.lora_rank else 32,
            thinking=_w.THINKING,
            card_vram_gb=card_gb,
        )
        if not sleep_on:  # sleep resolves OFF -> expandable is safe + better
            conf = "expandable_segments:True"
            os.environ["PYTORCH_ALLOC_CONF"] = conf
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
            print(f"[alloc] sleep resolves OFF -> {conf} (anti-fragmentation, matches worker gate)")
        else:
            print("[alloc] sleep resolves ON -> keeping launcher's non-expandable conf")
    except Exception as e:
        print("[alloc] auto-conf skipped:", e)
