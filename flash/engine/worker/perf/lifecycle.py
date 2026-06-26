"""GPU lifecycle helpers for the fine-tuning worker: readiness wait, cleanup, retriable-infra
signaling, and the metric-curve extractor. Torch is imported lazily so this is CPU-importable.
"""

from __future__ import annotations

import contextlib

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


def _sm_major(sm: str | None) -> int | None:
    """Major compute-capability from a class ``sm`` token ('sm89'->8, 'sm120'->12), or None.
    (sm digits = compute capability with a single minor digit, so major = all-but-last-digit.)"""
    import re

    m = re.fullmatch(r"sm(\d+)", (sm or "").strip().lower())
    if not m:
        return None
    digits = m.group(1)
    return int(digits[:-1]) if len(digits) >= 2 else int(digits)


def _host_driver_cuda() -> float | None:
    """Host driver's max supported CUDA — the PTX-JIT ceiling, NOT ``torch.version.cuda`` (build CUDA).
    pynvml then ``nvidia-smi`` header; None if neither works (driver-floor check skipped, best-effort)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            try:
                v = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            except Exception:
                v = pynvml.nvmlSystemGetCudaDriverVersion()
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        # NVML encodes the version as 1000*major + 10*minor (CUDA 12.8 -> 12080 -> 12.8).
        return (v // 1000) + ((v % 1000) // 10) / 10.0
    except Exception:
        pass
    try:
        import re
        import subprocess

        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
        m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def _gpu_mismatch_reason(
    requested_gpu: str | None,
    live_cap: tuple[int, int] | None,
    live_vram_gb: float | None,
    driver_cuda: float | None,
) -> str | None:
    """Return a human reason the LIVE GPU can't satisfy the REQUESTED class, else None.

    PURE (no CUDA/NVML) so it is unit-testable; a ``None`` live input skips that dimension. Three
    independent failover triggers: driver CUDA below the class floor (``min_cuda_modern``: Blackwell
    sm120 needs 13.0 to JIT the wheels' PTX, else 12.8); VRAM < ~90% of spec (a smaller substituted
    card OOMs); compute-capability MAJOR below the class (generational downgrade — major only, since
    the minor digit isn't capability-ordered: A100 sm80 < A6000 sm86, so VRAM guards within a major).
    """
    try:
        from flash.providers.base import get_gpu_info, min_cuda_modern

        info = get_gpu_info(requested_gpu or "")
    except Exception:
        return None  # unknown / unset class -> nothing to verify against (best-effort)
    reasons: list[str] = []
    floor = float(min_cuda_modern(info.name))
    if driver_cuda is not None and driver_cuda + 1e-9 < floor:
        reasons.append(
            f"host driver CUDA {driver_cuda:g} < {floor:g} required for {info.name} ({info.sm})"
        )
    if live_vram_gb is not None and live_vram_gb < info.vram_gb * 0.9:
        reasons.append(
            f"only {live_vram_gb:.1f} GB VRAM but {info.name} needs ~{info.vram_gb} GB"
        )
    exp_major = _sm_major(info.sm)
    if live_cap is not None and exp_major is not None and live_cap[0] < exp_major:
        reasons.append(
            f"compute capability {live_cap[0]}.{live_cap[1]} below sm{exp_major}x for {info.name}"
        )
    return "; ".join(reasons) or None


def verify_gpu(requested_gpu: str | None) -> None:
    """Assert the LIVE GPU matches the REQUESTED class (model + CUDA floor), or raise retriable.
    No-op for a falsy/unknown class. Runs on every provider (standardizes the per-class CUDA floor that
    only Hyperstack/RunPod enforce pre-launch + a GPU-model check no provider does); a mismatch raises
    ``RetriableInfraError`` (fail over) instead of an opaque mid-setup crash. Best-effort reads."""
    if not requested_gpu:
        return
    import torch

    live_cap = None
    live_vram_gb = None
    with contextlib.suppress(Exception):
        live_cap = torch.cuda.get_device_capability(0)
    with contextlib.suppress(Exception):
        # Decimal GB (/1e9), NOT binary GiB: the catalog `vram_gb` and all of flash.engine.vram
        # (estimate_vram_gb, grpo_fits_resident) are decimal, so the `info.vram_gb * 0.9` comparison
        # in _gpu_mismatch_reason must read live VRAM the same way. (The classes — 40/48/80 GB — are
        # far enough apart that the unit doesn't flip any real check; this keeps the convention.)
        live_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    reason = _gpu_mismatch_reason(requested_gpu, live_cap, live_vram_gb, _host_driver_cuda())
    if reason:
        name = "?"
        with contextlib.suppress(Exception):
            name = torch.cuda.get_device_name(0)
        raise RetriableInfraError(
            f"assigned GPU does not match requested {requested_gpu!r}: {reason} (live: {name}); "
            "retrying on a fresh correctly-provisioned GPU"
        )


def wait_for_gpu(requested_gpu: str | None = None):
    """Rented nodes sometimes report 'CUDA device not ready' transiently at startup.
    Poll a trivial CUDA op until it succeeds before doing real work; raise if never ready.

    Also fails fast (retriable) if the assigned GPU is a MIG slice — a partitioned GPU crashes the
    CUDA allocator, so we re-provision on a fresh FULL GPU instead of dying mid-setup. Once CUDA is
    live, ``verify_gpu(requested_gpu)`` asserts the box is the RIGHT GPU (model + driver-CUDA floor)."""
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
                verify_gpu(requested_gpu)  # right GPU for the run? (model + CUDA floor) else fail over
                return True
            last = "cuda not available"
        except RetriableInfraError:
            raise  # a GPU/identity mismatch must propagate (fail over), not be masked as "never ready"
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
