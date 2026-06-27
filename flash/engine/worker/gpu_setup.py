"""GPU/backend setup the worker runs at boot or just before TRL builds the rollout engine.

Run-scoped state is read through the worker package at CALL time so monkeypatch reaches these readers."""

from __future__ import annotations

import os

from flash.engine.worker._pkg import W as _w


def force_vllm_backend_for_sm120() -> str | None:
    """Force FLASHINFER on sm120 (RTX 5090): flash-attn PTX is unreliable on consumer Blackwell hosts,
    causing silent empty-rollout failures. Returns the backend set, or None if not sm120."""
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
    """Sync PYTORCH_ALLOC_CONF with the resolved GRPO sleep mode (RL only).

    PYTORCH_ALLOC_CONF is read at first CUDA allocation — must run before any allocation."""
    if _w.PHASE != "rl":
        return
    try:
        from flash.engine.worker.grpo import resolve_grpo_sleep_mode

        sleep_on, _ctx, _card_gb = resolve_grpo_sleep_mode()
        if not sleep_on:  # expandable_segments crashes only when sleep is ON
            conf = "expandable_segments:True"
            os.environ["PYTORCH_ALLOC_CONF"] = conf
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = conf
            print(f"[alloc] sleep resolves OFF -> {conf} (anti-fragmentation, matches worker gate)")
        else:
            print("[alloc] sleep resolves ON -> keeping launcher's non-expandable conf")
    except Exception as e:
        print("[alloc] auto-conf skipped:", e)
