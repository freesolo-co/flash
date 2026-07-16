"""GPU lifecycle helpers: readiness wait, cleanup, retriable-infra signaling, metric-curve extractor."""

from __future__ import annotations

import contextlib
import re

RETRIABLE_INFRA_MARKER = "RETRIABLE_INFRA_GPU"


class RetriableInfraError(RuntimeError):
    """Infrastructure failure the control plane should retry on a fresh worker."""

    def __init__(self, reason: str):
        super().__init__(f"{RETRIABLE_INFRA_MARKER}: {reason}")


def cuda_oom_count() -> int:
    """Cumulative torch allocator OOM count across visible CUDA devices."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return sum(
            int(torch.cuda.memory_stats(i).get("num_ooms", 0))
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return 0


def is_cuda_oom(exc: BaseException | None) -> bool:
    """Whether a failure was a CUDA OOM.

    Prefer structured torch allocator signals. Also classify vLLM's deterministic startup memory
    preflight errors: those can fail before torch records a CUDA OOM counter, but a larger GPU is the
    correct retry action.
    """
    if exc is None or isinstance(exc, MemoryError):
        return False
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = str(exc).lower()
    if (
        re.search(r"free memory on device\s+cuda:\d+.*less than desired gpu memory utilization", msg)
        or "no available memory for the cache blocks" in msg
    ):
        return True
    return cuda_oom_count() > 0


def _sm_major(sm: str | None) -> int | None:
    """Major compute capability from an sm token ('sm89'->8, 'sm120'->12), or None."""
    import re

    m = re.fullmatch(r"sm(\d+)", (sm or "").strip().lower())
    if not m:
        return None
    digits = m.group(1)
    return int(digits[:-1]) if len(digits) >= 2 else int(digits)


def _host_driver_cuda() -> float | None:
    """Host driver's max supported CUDA (PTX-JIT ceiling, not torch build CUDA). None if undetectable."""
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
    """Return a human reason the live GPU can't satisfy the requested class, else None."""
    try:
        from flash.providers.base import get_gpu_info, min_cuda_modern

        info = get_gpu_info(requested_gpu or "")
    except Exception:
        return None
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


def verify_gpu(requested_gpu: str | None, *, exact_type: str = "") -> None:
    """Assert the live GPU satisfies the requested class and optional exact identity."""
    if not requested_gpu and not exact_type:
        return
    import torch

    live_name = "?"
    with contextlib.suppress(Exception):
        live_name = torch.cuda.get_device_name(0)
    if exact_type:
        from flash.providers.base import canonical_gpu

        requested_canonical = canonical_gpu(exact_type)
        try:
            observed_canonical = canonical_gpu(live_name)
        except Exception:
            observed_canonical = "unrecognized"
        if observed_canonical != requested_canonical:
            raise RetriableInfraError(
                "assigned GPU exact-class mismatch: "
                f"requested={requested_canonical!r}, observed={observed_canonical!r}, "
                f"device_name={live_name!r}; retrying on a correctly-provisioned GPU"
            )
        # the pinned class is authoritative: validate the live card against IT below, not the softer
        # requested_gpu hint (which may name a larger class and false-reject this correct card). this
        # keeps the live vram/driver/capability safety net for the pinned card, so a same-named but
        # under-provisioned card (e.g. a mig slice, or a too-old host driver) is still caught.
        requested_gpu = exact_type

    live_cap = None
    live_vram_gb = None
    with contextlib.suppress(Exception):
        live_cap = torch.cuda.get_device_capability(0)
    with contextlib.suppress(Exception):
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


def _nvml_alive() -> bool:
    """True if NVML initializes. A busy GPU keeps NVML alive; a broken host (driver crash, GPU off PCIe bus) fails NVML and won't recover."""
    try:
        import pynvml

        pynvml.nvmlInit()
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()
        return True
    except Exception:
        pass
    try:
        import subprocess

        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def wait_for_gpu(requested_gpu: str | None = None, *, exact_type: str = ""):
    """Poll until CUDA is live; raise RetriableInfraError if the host NVML is dead or it never readies."""
    import time as _t

    last = None
    for i in range(12):
        try:
            import torch

            if torch.cuda.is_available():
                _ = torch.zeros(8, device="cuda") + 1
                torch.cuda.synchronize()
                print(f"GPU ready after {i} retries: {torch.cuda.get_device_name(0)}")
                verify_gpu(requested_gpu, exact_type=exact_type)
                return True
            last = "cuda not available"
        except RetriableInfraError:
            raise
        except Exception as e:
            last = str(e)[:160]
        # After a few retries, a dead NVML means the host won't recover — fail fast.
        if i >= 2 and not _nvml_alive():
            raise RetriableInfraError(
                f"GPU host NVML init failed (driver/host fault, won't recover): {last}; failing over"
            )
        print(f"GPU not ready (try {i + 1}/12): {last}; sleeping 10s")
        _t.sleep(10)
    raise RetriableInfraError(f"GPU never became ready after 12 tries: {last}")


def free_gpu(trainer=None):
    try:
        import gc

        import torch

        try:
            if trainer is not None and hasattr(trainer, "model"):
                trainer.model = None
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print("free_gpu warn:", e)


def _metric_curve(trainer, key: str) -> list:
    """Logged values of `key` from trainer log history, rounded and capped at 400. Never raises."""
    try:
        vals = [round(float(h[key]), 4) for h in trainer.state.log_history if key in h]
        return vals[:400]
    except Exception:
        return []
