"""GPU lifecycle helpers: readiness wait, cleanup, retriable-infra signaling, metric-curve extractor."""

from __future__ import annotations

import contextlib
import re

RETRIABLE_INFRA_MARKER = "RETRIABLE_INFRA_GPU"
_CUDA_OOM_PREFLIGHT_RE = re.compile(
    r"free memory on device\s+cuda:\d+.*less than desired gpu memory utilization"
)
_CUDA_OOM_CACHE_BLOCKS = "no available memory for the cache blocks"
# verl child processes lose the torch exception type and allocator counter. match torch's exact cuda
# OOM wording so training OOMs retry on larger GPUs without misclassifying host RAM or env errors.
#
# the `torch.` qualifier is REQUIRED, not optional. `OutOfMemoryError` is also RAY's class name, so
# a pattern whose prefixes were both optional degenerated to bare `outofmemoryerror` and matched
# `ray.exceptions.OutOfMemoryError` -- a HOST-RAM kill -- escalating to a larger card while the gpu
# sat idle. torch prints `torch.OutOfMemoryError` or `torch.cuda.OutOfMemoryError`, never unqualified.
_CUDA_OOM_TORCH_RE = re.compile(
    r"torch\.(?:cuda\.)?outofmemoryerror|cuda out of memory|cuda error: out of memory"
)
# the allocation detail that follows torch's OOM class -- how much was requested against how
# much the card had. carried SEPARATELY from the classification token above, because
# `is_cuda_oom` re-classifies the evidence string and a wider match there would let an
# unrelated line re-trigger VRAM escalation.
#
# torch prints the capacity and free figures in TWO spellings, and the caller returns the whole
# matched span -- so a spelling the pattern cannot reach is not merely unparsed, it is dropped from
# the operator's message entirely, which is the loss this pattern exists to prevent. the newer form
# puts the phrase first (a total capacity OF n GiB, of which n MiB IS FREE); the classic
# parenthetical form puts the unit first (n GiB TOTAL CAPACITY; n MiB FREE). each half therefore
# accepts either order. `test_allocation_detail_keeps_the_figures_in_torchs_classic_parenthetical_spelling`
# carries both literal strings.
_CUDA_OOM_ALLOCATION_RE = re.compile(
    r"tried to allocate\s+[\d.]+\s*[kmgt]?ib?"
    r"(?:.*?(?:(?:total capacity|capacity of)\s+[\d.]+\s*[kmgt]?ib?"
    r"|[\d.]+\s*[kmgt]?ib?\s+total capacity))?"
    r"(?:.*?([\d.]+\s*[kmgt]?ib?)\s+(?:is free|free))?"
)
# ray's host-memory monitor kills workers when NODE ram is exhausted. the gpu can be idle at the
# moment of death, so this is the opposite remedy from a cuda oom: the scarce resource is system ram.
_HOST_RAM_KILL_RE = re.compile(
    r"(?:worker\(s\) were killed due to the node running low on memory"
    r"|task was killed due to the node running low on memory)"
)


class RetriableInfraError(RuntimeError):
    """Infrastructure failure the control plane should retry on a fresh worker."""

    def __init__(self, reason: str):
        super().__init__(f"{RETRIABLE_INFRA_MARKER}: {reason}")


class DirtyGpuError(RetriableInfraError):
    """An allocated GPU arrived with substantial pre-existing occupancy.

    This is retriable infrastructure, deliberately not an OOM. An OOM means the run does not fit
    and retries on a larger card. Substantial boot-time occupancy instead calls for a fresh
    allocation, because a larger card can arrive just as occupied. This error does not claim that an
    accepted card has enough free VRAM for the run: occupancy at or below 5%, unreadable NVML, and
    invalid readings remain fail-open limits of the best-effort screen.
    """


def _nvml_memory_gb(device_index: int = 0) -> tuple[float, float] | None:
    """``(free, total)`` for one device in GB from NVML, or None if NVML will not answer.

    NVML, not ``torch.cuda.mem_get_info``: the torch call needs a CUDA context and creates one if the
    process has none, so the act of measuring adds our own memory to the number being measured. NVML
    queries the driver without initializing CUDA in this process. Device 0 remains the default for
    the diagnostic helpers that expose its free and total memory.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(info.free) / (1024**3), float(info.total) / (1024**3)
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
    except Exception:
        return None


def free_vram_gb() -> float | None:
    """Free VRAM on device 0 in GB, or None if it cannot be determined.

    Reads the DRIVER's number, not the torch allocator's. The allocator only knows about memory this
    process reserved, and the memory at issue here belongs to somebody else -- a co-tenant sharing
    the physical card from another container, which the allocator cannot see and would report as
    entirely free.
    """
    reading = _nvml_memory_gb()
    return None if reading is None else reading[0]


def cuda_is_initialized() -> bool:
    """True once THIS process has a live CUDA context, so a driver reading includes our own memory.

    ``torch.cuda.is_initialized()`` is false until the first real CUDA call, and importing torch does
    not make it true -- so this distinguishes "nothing of ours is on the card yet" from "some of what
    the driver reports is ours", which is the only thing that decides whether occupancy is readable.
    """
    try:
        import sys

        torch = sys.modules.get("torch")
        return bool(torch is not None and torch.cuda.is_initialized())
    except Exception:
        return True  # cannot prove the card is untouched -> treat the reading as unusable


def total_vram_gb() -> float | None:
    """Total VRAM on device 0 in GB as the DRIVER reports it, or None if unavailable.

    The driver's usable total, not the catalog's nominal tier: a "24 GB" RTX 4090 reports ~22.5 GB.
    Comparing against the nominal number is what makes an exact-fit run unschedulable.
    """
    reading = _nvml_memory_gb()
    return None if reading is None else reading[1]


def preflight_gpu_occupancy(gpu_count: int, *, max_occupied_fraction: float = 0.05) -> None:
    """Screen each allocated GPU for substantial occupancy before CUDA initialization.

    The worker reads exactly the allocator-resolved device indices ``range(gpu_count)`` through NVML.
    An unreadable device or a non-positive total is skipped without hiding later allocated devices.
    If any valid reading exceeds 5% occupancy, the allocation is rejected as retriable
    infrastructure after every allocated device has been inspected.

    This is a best-effort occupancy boundary, not an exact-fit guarantee. It deliberately does not
    reconstruct the allocator's run requirement. Occupancy at or below 5%, unreadable NVML, and
    invalid readings remain accepted. Once this process has initialized CUDA, the whole screen is
    fail-open because the driver reading includes memory owned by this process with no sound way to
    subtract it.
    """
    if cuda_is_initialized():
        # some of `used` would be ours and there is no sound way to tell how much
        return

    first_dirty: tuple[int, float, float] | None = None
    for device_index in range(gpu_count):
        reading = _nvml_memory_gb(device_index)
        if reading is None:
            # one unreadable device is not evidence about any later allocated device
            continue
        free, total = reading
        if total <= 0:
            continue
        used = max(0.0, total - free)
        if used > total * max_occupied_fraction and first_dirty is None:
            first_dirty = (device_index, used, total)

    if first_dirty is None:
        return
    device_index, used, total = first_dirty
    raise DirtyGpuError(
        f"allocated GPU device {device_index} has {used:.1f} GB of {total:.1f} GB "
        f"({used / total:.0%}) already in use before this run has touched it; the card is occupied "
        "by another tenant or a previous tenant's leak, so retrying on a freshly allocated instance"
    )


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


def cuda_oom_message_evidence(message: str) -> str | None:
    """the authoritative text evidence for a deterministic cuda oom, if present."""
    normalized = message.lower()
    preflight = _CUDA_OOM_PREFLIGHT_RE.search(normalized)
    if preflight is not None:
        return preflight.group(0)
    if _CUDA_OOM_CACHE_BLOCKS in normalized:
        return _CUDA_OOM_CACHE_BLOCKS
    torch_oom = _CUDA_OOM_TORCH_RE.search(normalized)
    if torch_oom is not None:
        return torch_oom.group(0)
    return None


def cuda_oom_allocation_detail(message: str) -> str | None:
    """How much a cuda OOM asked for, and what the card had, when the message says.

    ``cuda_oom_message_evidence`` deliberately returns only the classification token: its result is
    fed back through ``is_cuda_oom``, so widening it would let an unrelated line re-match and
    re-trigger VRAM escalation. That kept the retry logic honest but threw away the only numbers an
    operator needs -- a torch OOM line says it tried to allocate 2.00 GiB against a 179.06 GiB card
    with 1.31 GiB free, and all that survived was ``torch.outofmemoryerror``.

    This reads the same line for that detail WITHOUT feeding classification, so the numbers reach
    the diagnostics while the retry decision stays keyed on the token alone.
    """
    match = _CUDA_OOM_ALLOCATION_RE.search(message.lower())
    return match.group(0) if match is not None else None


def host_ram_kill_evidence(message: str) -> str | None:
    """The authoritative text evidence for a ray host-RAM kill, if present.

    Kept distinct from `cuda_oom_message_evidence` because the two failures need OPPOSITE remedies:
    a CUDA OOM retries on a larger-VRAM class, while a host-RAM kill needs more SYSTEM ram and would
    fail identically on a bigger card whose node ships the same memory.
    """
    return match.group(0) if (match := _HOST_RAM_KILL_RE.search(message.lower())) else None


def is_cuda_oom(exc: BaseException | None) -> bool:
    """Return whether a failure was a CUDA OOM.

    Prefer torch signals, but include deterministic vLLM startup preflights that occur before the
    allocator records an OOM.
    """
    if exc is None or isinstance(exc, MemoryError):
        return False
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    message = str(exc)
    if cuda_oom_message_evidence(message) is not None:
        return True
    # explicit host-RAM evidence outranks the allocator counter. `cuda_oom_count()` is CUMULATIVE
    # and process-wide: one recovered allocator OOM earlier in the run leaves it above zero for
    # every later failure, which would re-classify a known ray host-RAM kill as a cuda OOM and
    # escalate VRAM again. a failure that named its own cause is not diagnosed by a stale counter.
    if host_ram_kill_evidence(message) is not None:
        return False
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
        from flash.providers.core.base import get_gpu_info, min_cuda_modern

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
        reasons.append(f"only {live_vram_gb:.1f} GB VRAM but {info.name} needs ~{info.vram_gb} GB")
    exp_major = _sm_major(info.sm)
    if live_cap is not None and exp_major is not None and live_cap[0] < exp_major:
        reasons.append(
            f"compute capability {live_cap[0]}.{live_cap[1]} below sm{exp_major}x for {info.name}"
        )
    return "; ".join(reasons) or None


def verify_gpu(requested_gpu: str | None, *, gpu_type: str = "") -> None:
    """Assert the live GPU satisfies the requested class and optional exact identity."""
    if not requested_gpu and not gpu_type:
        return
    import torch

    live_name = "?"
    with contextlib.suppress(Exception):
        live_name = torch.cuda.get_device_name(0)
    if gpu_type:
        from flash.providers.core.base import canonical_gpu

        requested_canonical = canonical_gpu(gpu_type)
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
        requested_gpu = gpu_type

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


def wait_for_gpu(requested_gpu: str | None = None, *, gpu_type: str = ""):
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
                verify_gpu(requested_gpu, gpu_type=gpu_type)
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
