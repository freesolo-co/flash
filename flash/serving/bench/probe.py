"""On-GPU provenance and kernel-path probes recorded with every capacity result.

A capacity number is only interpretable next to the hardware and kernel path that produced it. Two
probes matter enough to gate publication:

* **Card identity.** A result labelled B200 that ran on an H200 is worse than no result.
* **GDN prefill backend.** The served Qwen3.5/3.6 models are GatedDeltaNet hybrids. vLLM 0.23.0's
  ``_resolve_gdn_prefill_backend`` grants FlashInfer GDN prefill to SM90 (H100/H200) unconditionally,
  but on SM10.x (B200) only when ``linear_key_head_dim == 128`` AND CUDA runtime major >= 13 AND
  ``_is_libs_cu13_install_intact()``. That last condition fails whenever the two
  ``nvidia-cutlass-dsl-libs-*`` wheels race during install, and it fails SILENTLY: the resolver falls
  through to Triton after a single ``warning_once``.

  The trap is that the engine still boots, still serves, and still bills the B200 rate while running
  the slower prefill kernel. A boot smoke cannot see it. So the benchmark asks the resolver what it
  actually chose and labels the envelope with the answer, rather than assuming the fast path.

This module reports; it does not repair. Repairing the Cutlass install belongs to the runtime image,
not to a measurement tool that would then be measuring its own side effect.
"""

from __future__ import annotations

import contextlib
from typing import Any


def probe_gpu() -> dict[str, Any]:
    """Device identity, capability, and memory, read from the DRIVER via NVML.

    Deliberately not ``torch.cuda``. Under vLLM V1 the model runs in a separate EngineCore process,
    and this probe runs in the parent Modal class process. ``torch.cuda.current_device()`` and
    friends initialize a CUDA context *here*, and a prior campaign's extra parent-process context
    stole the post-init headroom EngineCore needs for FlashInfer's first-request decode workspace,
    OOM-killing the 35B engine on its first request (see the post-mortem comment in
    ``flash/serving/src/engine/lora_engine.py``). Since the canary runs this probe before every
    warmup and every sweep, a torch-based probe would either reproduce that outage or measure a
    memory shape production deliberately avoids.

    NVML queries the driver without creating a context, so identity costs nothing measurable.
    """
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - NVML ships with the serving image
        return {"available": False, "reason": f"pynvml unavailable: {exc}"}

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return {"available": False, "reason": "NVML reports no devices"}
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result: dict[str, Any] = {
            "available": True,
            "device_count": count,
            "name": str(name),
            "compute_capability": f"{major}.{minor}",
            "total_memory_bytes": int(memory.total),
            "source": "nvml",
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()

    # Version strings are module attributes; reading them creates no context.
    try:
        import torch

        result["torch_version"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
    except Exception:  # pragma: no cover - torch is present in the serving image
        pass
    return result


def probe_cutlass_integrity() -> dict[str, Any]:
    """Run vLLM's OWN ``_is_libs_cu13_install_intact`` check, uncached.

    The function is ``lru_cache``d upstream, so an earlier in-process call could return a stale
    answer; ``__wrapped__`` reaches the undecorated function when it is available.
    """
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    except Exception as exc:
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"}

    check = getattr(gdn, "_is_libs_cu13_install_intact", None)
    if check is None:
        return {"checked": False, "reason": "vllm build has no _is_libs_cu13_install_intact"}
    try:
        intact = bool(getattr(check, "__wrapped__", check)())
    except Exception as exc:
        return {"checked": False, "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "checked": True,
        "intact": intact,
        # The consequence, stated where a reader of the report will see it.
        "implication": (
            "FlashInfer GDN prefill is available on Blackwell"
            if intact
            else "GDN prefill silently falls back to Triton on Blackwell (slower path)"
        ),
    }


def probe_gdn_backend(base_model: str) -> dict[str, Any]:
    """What the GDN prefill resolver would choose for this model on this card.

    Best-effort by design: the resolver is internal to vLLM and its signature may move between
    builds. A failed probe is recorded as unknown rather than being allowed to assert the fast path
    by omission.
    """
    result: dict[str, Any] = {"base_model": base_model, "resolved": None}
    try:
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result

    resolver = getattr(gdn, "_resolve_gdn_prefill_backend", None)
    if resolver is None:
        result["reason"] = "vllm build has no _resolve_gdn_prefill_backend"
        return result
    result["resolver_present"] = True
    result["cutlass"] = probe_cutlass_integrity()
    # Presence of the resolver says nothing about WHICH backend it picks. The Blackwell failure mode
    # is `warning_once` then a silent Triton fallback -- the boot succeeds, serves, and bills the
    # fast-card rate while running the slow kernel. So call it and record what it actually returned;
    # an unresolved probe stays None and the report may not claim a backend.
    try:
        resolved = resolver()
    except Exception as exc:
        result["reason"] = f"resolver raised: {type(exc).__name__}: {exc}"
        return result
    if isinstance(resolved, tuple):
        result["resolved"] = str(resolved[-1]) if resolved else None
        result["resolved_raw"] = [str(part) for part in resolved]
    else:
        result["resolved"] = str(resolved) if resolved is not None else None
    return result


def probe_engine_kv_cache(engine: Any) -> dict[str, Any]:
    """KV-cache block counts from the running engine, when its build exposes them.

    Recorded because concurrency at a long context is bounded by the KV pool, so the pool size is
    what makes a near-32k concurrency number explicable rather than merely observed.
    """
    out: dict[str, Any] = {}
    for path in ("cache_config", "vllm_config"):
        config = getattr(getattr(engine, "engine", engine), path, None)
        if config is None:
            continue
        cache = getattr(config, "cache_config", config)
        for field in ("num_gpu_blocks", "num_cpu_blocks", "block_size", "cache_dtype"):
            value = getattr(cache, field, None)
            if value is not None:
                out[field] = value if isinstance(value, (int, str)) else str(value)
    return out


def probe_all(base_model: str, engine: Any | None = None) -> dict[str, Any]:
    """The full provenance block stored with each model's results."""
    payload: dict[str, Any] = {
        "gpu": probe_gpu(),
        "gdn_prefill": probe_gdn_backend(base_model),
    }
    if engine is not None:
        payload["kv_cache"] = probe_engine_kv_cache(engine)
    return payload


def gpu_matches(probe: dict[str, Any], expected: str) -> bool:
    """Whether the probed device is the expected card.

    Token match, NOT substring. Vendors decorate the reported name ("NVIDIA L40S",
    "NVIDIA H200 141GB HBM3e") so bare equality would reject the real card -- but a substring test
    makes ``"L4" in "NVIDIA L40S"`` true, and an L4-expected model would silently pass its identity
    gate while running on an L40S. Since the whole point of this check is to attribute a measured
    number to a specific card, it splits the name and compares whole tokens.

    Modal's tier spelling is normalized first: ``H100!`` pins a non-substitutable card and
    ``A100-80GB`` carries its memory in the tier name, neither of which appears in the device name.
    """
    gpu = probe.get("gpu") or {}
    name = str(gpu.get("name") or "")
    tier = expected.upper().rstrip("!").split("-")[0].strip()
    if not tier:
        return False
    return tier in {token.strip() for token in name.upper().replace("-", " ").split()}


__all__ = [
    "gpu_matches",
    "probe_all",
    "probe_cutlass_integrity",
    "probe_engine_kv_cache",
    "probe_gdn_backend",
    "probe_gpu",
]
