"""Shared rollout pool — a load-balancing router (an "nginx for GRPO rollouts") in front of a
fleet of pre-rented GPU inference workers, so many training runs SHARE the same generation +
reward capacity instead of each renting (and idling) its own.

Why this exists
---------------
A GRPO step is three phases on the trainer GPU: (1) **rollout** (vLLM decodes the group of
completions — HBM-bandwidth bound), (2) **reward** (often CPU / IO / LLM-judge bound), and
(3) the **optimizer** update (compute bound). During (1) and (2) the expensive trainer GPU's
matmul units sit idle — wasted wall-clock and wasted money.

The disaggregated path (`flash.engine.disaggregated`, PR #4) moves rollout onto dedicated
inference card(s), and verl one-step-off (`flash.engine.verl_runner`) overlaps gen(t+1) with
train(t). But a *single* run still can't keep a whole inference GPU busy, and reward still blocks.
The win at fleet scale is **sharing**: pre-rent N inference GPUs once, put a router in front, and
point *every* training run at it. While run A is in its optimizer phase (not generating), run B's
rollout requests keep the pool busy. Because vLLM serves **many LoRA adapters off one base model
on one GPU**, dozens of small runs co-reside on a card (one base weight, one adapter per run) —
"train multiple models on one GPU". The router is the nginx-equivalent: health-checked upstreams,
least-outstanding-request balancing, lazy + replicated adapter placement, per-step weight-sync
hot-swap, retry/failover, and reward fan-out to off-GPU reward workers.

Layout
------
* :mod:`flash.pool.state`     — ``Backend`` / ``Adapter`` / ``PoolState`` registry + balancer (pure)
* :mod:`flash.pool.protocol`  — vLLM OpenAI + dynamic-LoRA wire shapes (pure builders)
* :mod:`flash.pool.gateway`   — async httpx calls to one vLLM upstream (injectable for tests)
* :mod:`flash.pool.router`    — the FastAPI app (``create_pool_app``) — the nginx-equivalent
* :mod:`flash.pool.rewards`   — off-GPU reward-worker registry + dispatch
* :mod:`flash.pool.client`    — trainer-side ``RolloutPoolClient`` (generate / score / sync_weights)
* :mod:`flash.pool.config`    — ``RouterConfig`` (from env) + ``PoolPlan``/``PoolMember`` (from TOML)
* :mod:`flash.pool.provision` — rent a pool via the allocator and register it with the router
* :mod:`flash.pool.server`    — uvicorn entrypoint (``flash-pool serve``)

The control-plane side (state/router/client) is deliberately **torch-free** — it depends only on
``fastapi`` + ``httpx`` (the ``server`` extra), so the router runs on a cheap CPU box. Only the
GPU workers it points at run vLLM.
"""

from __future__ import annotations

from flash.pool.state import (
    Adapter,
    Backend,
    NoCapacityError,
    PlacementDecision,
    PoolState,
)

__all__ = [
    "Adapter",
    "Backend",
    "NoCapacityError",
    "PlacementDecision",
    "PoolState",
]
