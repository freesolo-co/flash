"""Off-GPU reward dispatch — so reward scoring never blocks the trainer GPU.

Reward functions (verifiers rubrics, LLM-judge calls, sandboxed code execution) are usually CPU /
IO / API bound. Running them inline on the trainer process stalls the GPU between rollout and the
optimizer step. Here reward computation runs on a pool of **reward workers** (cheap CPU boxes, or
the inference workers themselves) and the router fans scoring requests out to them, least-loaded
first. The trainer just POSTs prompts+completions and gets scores back — the work happens elsewhere.

Two pieces:
  * :class:`RewardRegistry` — the router's registry of reward-worker upstreams + least-load pick.
  * :func:`create_reward_app` — a standalone reward worker: wrap a ``scorer`` callable in a tiny
    ASGI app exposing ``POST /score`` and ``GET /health``. Run one per CPU box; register its URL.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

Scorer = Callable[[Sequence[str], Sequence[str], Sequence[dict]], Sequence[float]]


@dataclass
class RewardWorker:
    id: str
    url: str
    healthy: bool = True
    inflight: int = 0
    total_requests: int = 0
    total_failures: int = 0

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "healthy": self.healthy,
            "inflight": self.inflight,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
        }


class NoRewardCapacityError(RuntimeError):
    """No healthy reward worker is registered."""


class RewardRegistry:
    def __init__(self) -> None:
        self.workers: dict[str, RewardWorker] = {}

    def add(self, worker: RewardWorker) -> RewardWorker:
        self.workers[worker.id] = worker
        return worker

    def remove(self, worker_id: str) -> RewardWorker | None:
        return self.workers.pop(worker_id, None)

    def set_health(self, worker_id: str, healthy: bool) -> None:
        w = self.workers.get(worker_id)
        if w is not None:
            w.healthy = healthy

    def pick(self, *, exclude: set[str] | None = None) -> RewardWorker:
        exclude = exclude or set()
        cands = [w for w in self.workers.values() if w.healthy and w.id not in exclude]
        if not cands:
            raise NoRewardCapacityError("no healthy reward worker registered")
        return min(cands, key=lambda w: (w.inflight, w.total_requests))

    def acquire(self, worker_id: str) -> None:
        w = self.workers[worker_id]
        w.inflight += 1
        w.total_requests += 1

    def release(self, worker_id: str, *, failed: bool = False) -> None:
        w = self.workers.get(worker_id)
        if w is None:
            return
        w.inflight = max(0, w.inflight - 1)
        if failed:
            w.total_failures += 1

    def snapshot(self) -> dict:
        return {
            "workers": [w.snapshot() for w in self.workers.values()],
            "healthy": sum(1 for w in self.workers.values() if w.healthy),
        }


def score_request(
    prompts: Sequence[str],
    completions: Sequence[str],
    info: Sequence[dict] | None = None,
    *,
    reward_id: str = "default",
) -> dict[str, Any]:
    """Build the JSON body for ``POST /rewards/score`` (and the worker's ``/score``)."""
    n = len(completions)
    return {
        "reward_id": reward_id,
        "prompts": list(prompts),
        "completions": list(completions),
        "info": list(info) if info is not None else [{} for _ in range(n)],
    }


def create_reward_app(scorer: Scorer, *, reward_id: str = "default"):
    """A standalone reward worker as an ASGI app. ``scorer(prompts, completions, info) -> scores``.

    Run with ``uvicorn`` on a CPU box and register its URL with the router
    (``POST /rewards/workers {id,url}``). Importing FastAPI is deferred so the pure registry above
    has no hard ``server``-extra dependency.
    """
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title=f"flash reward worker [{reward_id}]")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "reward_id": reward_id}

    # A plain ``dict`` body (not a closure-local Pydantic model): with PEP 563 lazy annotations,
    # FastAPI can't resolve a model class defined inside this factory and would treat it as a query
    # param — so parse the body by hand (mirrors the router's endpoints).
    @app.post("/score")
    async def score(body: dict) -> dict:
        # Validate types/lengths up front: a non-list completions/prompts/info (e.g. a bare string)
        # would be iterated char-by-char and a mismatched length would mis-pair prompt->completion,
        # silently producing bad scores instead of a clear error. Reject with a 400.
        def _as_list(name: str, value):
            if value is None:
                return None
            if not isinstance(value, list):
                raise HTTPException(
                    status_code=400,
                    detail=f"{name} must be a list, got {type(value).__name__}",
                )
            return value

        completions = _as_list("completions", body.get("completions")) or []
        raw_info = _as_list("info", body.get("info"))
        raw_prompts = _as_list("prompts", body.get("prompts"))
        info = raw_info if raw_info is not None else [{} for _ in completions]
        prompts = raw_prompts if raw_prompts is not None else ["" for _ in completions]
        if raw_prompts is not None and len(prompts) != len(completions):
            raise HTTPException(
                status_code=400,
                detail=f"prompts ({len(prompts)}) and completions ({len(completions)}) length mismatch",
            )
        if raw_info is not None and len(info) != len(completions):
            raise HTTPException(
                status_code=400,
                detail=f"info ({len(info)}) and completions ({len(completions)}) length mismatch",
            )
        try:
            scores = list(scorer(prompts, completions, info))
        except Exception as e:  # surface scorer bugs as a 400 (don't 500 the worker)
            raise HTTPException(status_code=400, detail=f"scorer failed: {e}") from e
        if len(scores) != len(completions):
            raise HTTPException(
                status_code=400,
                detail=f"scorer returned {len(scores)} scores for {len(completions)} completions",
            )
        return {"scores": [float(s) for s in scores], "reward_id": reward_id}

    return app
