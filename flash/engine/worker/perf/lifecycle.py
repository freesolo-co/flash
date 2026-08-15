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
_CUDA_OOM_TORCH_RE = re.compile(r"torch\.(?:cuda\.)?outofmemoryerror|cuda out of memory")
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
    """The allocated card arrived with too little free VRAM to run on.

    A RetriableInfraError, deliberately not an OOM. The two recover differently and only one of them
    is right here: an OOM means the run does not fit the card and retries on a BIGGER one, spending
    from a small dedicated OOM budget. A dirty card means the run fits fine and the card was already
    occupied -- a co-tenant in another container, or a previous tenant's leak. Escalating size for
    that pays more for a card that can be just as dirty, and burns an OOM retry proving it.

    Infra retry reallocates instead: ``_handle_failure`` records the provider in ``failed_providers``
    and the shape in ``tried_classes``, so the next attempt prefers a DIFFERENT provider, then an
    untried shape, before clamping back. That is the right preference order for this failure --
    co-tenancy is a property of the host pool that handed the instance out, not of the run -- but it
    is a preference, not a guarantee, so nothing here promises which card comes back.
    """


def _nvml_memory_gb() -> tuple[float, float] | None:
    """``(free, total)`` on device 0 in GB straight from NVML, or None if NVML will not answer.

    NVML, not ``torch.cuda.mem_get_info``: the torch call needs a CUDA context and CREATES one if the
    process has none, so the act of measuring adds our own few hundred MB to the number being
    measured. NVML queries the driver without initializing CUDA in this process, which is what lets
    the boot reading be taken while the card is still provably untouched by us.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
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


def preflight_free_vram(*, max_occupied_fraction: float = 0.05) -> None:
    """Fail fast when the allocated card arrives already occupied by somebody else.

    Flash sizes the run, asks the provider for a card that fits, and then trains on whatever comes
    back without ever checking that the memory is FREE. A real RTX 4090 arrived with
    ``total=22.5 used=18.6 free=3.4``, only 0.486 GB of it owned by any process in this container:
    ~18 GB was a co-tenant in another container. The number was already in the first heartbeat,
    before any work -- recorded and not acted on -- so the run downloaded the model and built FSDP
    for ~80s of paid GPU before dying on an OOM that was knowable at boot.

    Measured as OCCUPANCY (``used/total`` as the driver reports it), not against a requirement.
    Re-deriving "what this run needs" here would be a second sizing model competing with the
    allocator's, and it loses both ways: the allocator sizes SFT from profile-measured knobs
    (``_overridden_train``) that reduce an authored batch 8 to the executed batch 1, so recomputing
    from the authored spec demands more than the card was rented for; and a run sized exactly at a
    catalog tier (24 GB) can exceed a real card's usable total (22.5 GB on a 4090), which no amount
    of free memory can ever satisfy. Both reject a perfectly clean card and retry until the infra
    budget is gone -- strictly worse than the OOM this exists to prevent.

    Occupancy has neither failure mode: it needs no requirement, no profile, and no catalog number.
    An empty card reads ~0 whatever is scheduled on it, so the threshold can sit near zero rather
    than being padded to cover a requirement it never consults. 5% is the driver's own reserve plus
    slack; the observed dirty card was at 83% with 0.486 GB attributable to us.

    The floor is low BECAUSE the check is occupancy: a loose threshold has to be justified against
    the biggest run that could be scheduled, and a 5 GB co-tenant that leaves a 20 GB run 17.5 GB on
    a 22.5 GB card is 22% occupied and fatal. Sizing the threshold to survive that means re-deriving
    the requirement, which is the failure above. Refusing every card with a stranger on it needs no
    requirement and covers the close-fitting case too.

    A threshold that low only works if none of the used memory is OURS, and that is established by
    WHEN the reading is taken, not by trying to attribute it afterwards. Attribution was tried and
    cannot work here: ``nvidia-smi --query-compute-apps`` reports HOST pids, while the worker
    container runs with a private pid namespace (``docker run`` in ``providers/_lifecycle/instance``
    passes no ``--pid=host``), so testing ``/proc/<pid>`` inside the container is not an ownership
    test in either direction -- our own rows fail it and get counted as a stranger's, and under
    ``--pid=host`` a real co-tenant passes it and gets credited to us, silently waving through the
    exact card this exists to refuse.

    So the check simply declines to run once this process has a CUDA context, which makes WHEN it
    runs the whole design. There is exactly one call site, ``_preflight_free_vram_for_spec`` at the
    top of ``_run_worker_mode``, and it is placed before ``_force_fla_triton_gdn_on_sm100`` because
    that reads ``get_device_capability`` and creates the process's first context. Calling from
    anywhere below that point is not a weaker check, it is no check at all: the guard above returns
    immediately and the card is never read. A single early call site is therefore the invariant, not
    an omission -- a later retry could only run after CUDA came up, when the reading is already
    unusable. That also means one NVML failure at boot is one lost reading, not a lost run: the
    guard below returns and training proceeds exactly as it did before this check existed.
    """
    if cuda_is_initialized():
        # some of `used` would be ours and there is no sound way to tell how much. a check that
        # cannot distinguish our context from a co-tenant would either false-reject clean cards or
        # need a threshold too loose to catch the tenant it exists for. staying silent is correct:
        # the boot call already read this card while it was provably untouched.
        return
    reading = _nvml_memory_gb()
    if reading is None:
        # NVML would not answer. somebody else's failure to report, not evidence of a dirty card --
        # training fails soon enough with a better message.
        return
    free, total = reading
    if total <= 0:
        return
    used = max(0.0, total - free)
    if used <= total * max_occupied_fraction:
        return
    raise DirtyGpuError(
        f"allocated GPU has {used:.1f} GB of {total:.1f} GB ({used / total:.0%}) already in use "
        "before this run has touched it; the card is occupied by another tenant or a previous "
        "tenant's leak, so retrying on a freshly allocated instance"
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
        from flash.providers.base import canonical_gpu

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
