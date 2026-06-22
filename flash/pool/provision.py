"""Pre-rent the GPU fleet for the rollout pool and register it with the router.

The cost idea: rent the inference GPUs **once**, up front, and keep them warm behind the router so
every training run shares them. This module turns a :class:`~flash.pool.config.PoolPlan` (base
model -> GPU class + count) into running vLLM servers and registers each as a backend.

The vLLM launch contract (:func:`build_vllm_serve_command`) is a pure builder so it's unit-testable
on CPU; only :func:`provision_pool` (with ``dry_run=False``) touches providers and the network. We
launch **stock vLLM** OpenAI servers with dynamic multi-LoRA enabled — that is what lets one GPU
hold one base model + many run adapters at once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from flash.pool.config import PoolMember, PoolPlan

# vLLM must allow runtime LoRA load/unload for the router to hot-swap adapters.
VLLM_LORA_ENV = {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "1"}


def build_vllm_serve_command(
    base_model: str,
    *,
    port: int = 8000,
    max_loras: int = 8,
    max_lora_rank: int = 32,
    max_cpu_loras: int | None = None,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    enable_prefix_caching: bool = True,
    extra_args: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return ``(argv, env)`` to launch a stock vLLM OpenAI server that serves ``base_model`` with
    dynamic multi-LoRA — the inference backend the router talks to."""
    argv = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        base_model,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--enable-lora",
        "--max-loras",
        str(max_loras),
        "--max-lora-rank",
        str(max_lora_rank),
        "--max-cpu-loras",
        str(max_cpu_loras if max_cpu_loras is not None else max(max_loras * 2, max_loras)),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--trust-remote-code",
    ]
    if max_model_len is not None:
        argv += ["--max-model-len", str(max_model_len)]
    if enable_prefix_caching:
        argv += ["--enable-prefix-caching"]
    if extra_args:
        argv += list(extra_args)
    env = dict(VLLM_LORA_ENV)
    return argv, env


@dataclass
class ProvisionResult:
    base_model: str
    gpu: str
    backend_id: str
    url: str
    registered: bool
    cost_per_hour: float = 0.0
    note: str = ""


def register_backend(
    router_url: str,
    *,
    backend_id: str,
    url: str,
    base_model: str,
    gpu_label: str = "",
    max_loras: int = 8,
    cost_per_hour: float = 0.0,
    client: httpx.Client | None = None,
) -> dict:
    """POST a backend to the router's ``/pool/backends``."""
    body = {
        "id": backend_id,
        "url": url,
        "base_model": base_model,
        "gpu_label": gpu_label,
        "max_loras": max_loras,
        "cost_per_hour": cost_per_hour,
    }
    c = client or httpx.Client(timeout=30.0)
    try:
        r = c.post(f"{router_url.rstrip('/')}/pool/backends", json=body)
        r.raise_for_status()
        return r.json()
    finally:
        if client is None:
            c.close()


def plan_summary(plan: PoolPlan) -> dict:
    """A dry-run view: how many GPUs of each class, and the per-base-model capacity."""
    total = sum(m.count for m in plan.members)
    per_base: dict[str, int] = {}
    for m in plan.members:
        per_base[m.base_model] = per_base.get(m.base_model, 0) + m.count
    return {
        "total_gpus": total,
        "members": [
            {"base_model": m.base_model, "gpu": m.gpu, "count": m.count, "max_loras": m.max_loras}
            for m in plan.members
        ],
        "capacity_per_base_model": per_base,
        "max_concurrent_adapters": {b: _adapter_capacity_for(plan, b) for b in per_base},
    }


def _adapter_capacity_for(plan: PoolPlan, base_model: str) -> int:
    """Concurrent-adapter capacity the fleet can actually host for ``base_model``.

    A base model can appear in several :class:`PoolMember`s with DIFFERENT ``max_loras`` (e.g. a
    big-VRAM slice that holds more adapters plus a small slice that holds fewer). Each GPU only
    holds its OWN member's ``max_loras`` adapters, so the true capacity is the sum of
    ``count * max_loras`` over those members — NOT ``total_count * max(max_loras)``, which would
    credit the small GPUs with the big slice's slot count and overstate what the fleet can hold.
    """
    return sum(m.count * m.max_loras for m in plan.members if m.base_model == base_model)


def provision_pool(
    plan: PoolPlan,
    router_url: str,
    *,
    dry_run: bool = True,
    rent: RentFn | None = None,
    client: httpx.Client | None = None,
) -> list[ProvisionResult]:
    """Provision ``plan`` and register each backend with the router.

    ``dry_run`` (default) only computes + returns the plan without renting. With ``dry_run=False``
    you must pass ``rent`` — a callable ``rent(member, index) -> (backend_id, url, cost_per_hour)``
    that rents one GPU, launches vLLM on it (see :func:`build_vllm_serve_command`), waits for
    ``/health``, and returns its address. (The live renter lives in the operator CLI so this module
    stays provider-agnostic and unit-testable.)
    """
    # A live provision MUST actually rent. Failing here (rather than silently taking the dry-run
    # branch and returning unregistered "(dry-run)" backends) keeps a real provision from looking
    # like a success while renting nothing.
    if not dry_run and rent is None:
        raise ValueError(
            "provision_pool(dry_run=False) requires a `rent` callable to actually rent + launch "
            "GPUs; refusing to silently skip renting. Pass rent=... or use dry_run=True."
        )
    results: list[ProvisionResult] = []
    c = client or httpx.Client(timeout=30.0)
    try:
        for m in plan.members:
            for i in range(m.count):
                if dry_run:
                    results.append(
                        ProvisionResult(
                            base_model=m.base_model,
                            gpu=m.gpu,
                            backend_id=f"{m.gpu}-{m.base_model.split('/')[-1]}-{i}",
                            url="(dry-run)",
                            registered=False,
                            note="dry-run: not rented",
                        )
                    )
                    continue
                backend_id, url, cost = rent(m, i)
                reg = register_backend(
                    router_url,
                    backend_id=backend_id,
                    url=url,
                    base_model=m.base_model,
                    gpu_label=f"{m.gpu}/{backend_id}",
                    max_loras=m.max_loras,
                    cost_per_hour=cost,
                    client=c,
                )
                results.append(
                    ProvisionResult(
                        base_model=m.base_model,
                        gpu=m.gpu,
                        backend_id=backend_id,
                        url=url,
                        registered=bool(reg),
                        cost_per_hour=cost,
                    )
                )
    finally:
        if client is None:
            c.close()
    return results


# A renter rents+launches one GPU and returns (backend_id, url, cost_per_hour).
from collections.abc import Callable  # noqa: E402 - keep the type alias near its only use

RentFn = Callable[[PoolMember, int], tuple[str, str, float]]


def router_url_from_env() -> str:
    return os.environ.get("FLASH_ROLLOUT_POOL_URL", "http://127.0.0.1:8077")
