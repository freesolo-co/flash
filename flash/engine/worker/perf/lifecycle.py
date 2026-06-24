"""GPU lifecycle helpers for the fine-tuning worker: readiness wait, cleanup, retriable-infra
signaling, and the metric-curve extractor. Torch is imported lazily so this is CPU-importable.
"""

from __future__ import annotations

# Human-readable sentinel embedded in the error message (debug tag only — the runner classifies
# structurally off the worker's heartbeat ``retriable`` flag, not by matching this phrase).
RETRIABLE_INFRA_MARKER = "RETRIABLE_INFRA_GPU"


class RetriableInfraError(RuntimeError):
    """An infrastructure failure the control plane should RETRY on a fresh worker.

    Raised for a host the run can never train on — e.g. a GPU that never comes up
    (``wait_for_gpu`` times out) or a required-upload failure. The worker's top-level handler
    stamps ``retriable=True`` into heartbeat.json so the runner retries on a fresh worker.
    """

    def __init__(self, reason: str):
        super().__init__(f"{RETRIABLE_INFRA_MARKER}: {reason}")


def wait_for_gpu():
    """Rented nodes sometimes report 'CUDA device not ready' transiently at startup.
    Poll a trivial CUDA op until it succeeds before doing real work; raise if never ready."""
    import time as _t

    last = None
    for i in range(12):
        try:
            import torch

            if torch.cuda.is_available():
                # Force an actual kernel launch (alloc + add) to confirm the GPU is live.
                _ = torch.zeros(8, device="cuda") + 1
                torch.cuda.synchronize()
                print(f"GPU ready after {i} retries: {torch.cuda.get_device_name(0)}")
                return True
            last = "cuda not available"
        except Exception as e:
            last = str(e)[:160]
        print(f"GPU not ready (try {i + 1}/12): {last}; sleeping 10s")
        _t.sleep(10)
    # Infra-shaped: a host whose GPU never comes up is dead, not a code bug -> retry on a fresh one.
    raise RetriableInfraError(f"GPU never became ready after 12 tries: {last}")


def free_gpu(trainer=None):
    try:
        import gc

        import torch

        try:
            if trainer is not None and hasattr(trainer, "model"):
                trainer.model = None
        except Exception:
            # Best-effort VRAM release before gc; any failure here is non-fatal cleanup.
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print("free_gpu warn:", e)


def _metric_curve(trainer, key: str) -> list:
    """The logged values of `key` (e.g. 'loss' or 'reward') from the trainer's log history,
    rounded + capped. Lets metrics.json carry the convergence/reward curve for an A/B without
    relying on a checkpoint's trainer_state.json (only written on save_steps) or the console
    (only uploaded on failure). Never raises."""
    try:
        vals = [round(float(h[key]), 4) for h in trainer.state.log_history if key in h]
        return vals[:400]
    except Exception:
        return []
