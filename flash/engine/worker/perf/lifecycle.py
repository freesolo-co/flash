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


def detect_mig_slice() -> str | None:
    """Return a reason string if this worker was handed a MIG slice (a partitioned GPU), else None.

    A MIG slice NVML-asserts PyTorch's CUDA allocator — observed when a provider substitutes a
    requested GPU type with a Blackwell MIG slice — which crashes the run with an opaque allocator
    assert partway through setup. Detect it up front (via nvidia-smi, before the first real CUDA op)
    so the worker can fail fast as RETRIABLE infra and the runner re-provisions a fresh FULL GPU,
    rather than letting the run die mid-setup. Best-effort: never raises (a missing/odd nvidia-smi
    just means "no MIG detected", which the subsequent CUDA readiness probe still guards)."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20
        ).stdout
        # A MIG slice appears as a nested device line, e.g.
        #   "  MIG 1g.10gb     Device  0: (UUID: MIG-xxxx)"  (or any "UUID: MIG-..." entry).
        for line in out.splitlines():
            s = line.strip()
            if s.startswith("MIG ") or "UUID: MIG-" in s:
                return f"MIG slice detected (nvidia-smi -L: {s[:120]!r})"
    except Exception:
        pass
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        # "Enabled" => the assigned GPU is partitioned into MIG instances (no full-GPU access).
        # "Disabled"/"N/A"/"[Not Supported]" (consumer + MIG-incapable cards) => fine.
        if q and "enabled" in q.lower():
            return f"MIG mode enabled on the assigned GPU (mig.mode.current={q!r})"
    except Exception:
        pass
    return None


def wait_for_gpu():
    """Rented nodes sometimes report 'CUDA device not ready' transiently at startup.
    Poll a trivial CUDA op until it succeeds before doing real work; raise if never ready.

    Also fails fast (retriable) if the assigned GPU is a MIG slice — a partitioned GPU crashes the
    CUDA allocator, so we re-provision on a fresh FULL GPU instead of dying mid-setup."""
    import time as _t

    mig = detect_mig_slice()
    if mig:
        # Infra-shaped: a MIG slice can never train this workload -> retry on a fresh full GPU.
        raise RetriableInfraError(f"{mig}; retrying on a fresh full (non-MIG) GPU")

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
