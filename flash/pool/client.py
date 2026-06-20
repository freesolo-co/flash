"""Trainer-side client for the rollout pool — the "training server" half of the split.

A GRPO trainer uses this to push all the GPU-idle work off its own card:

* :meth:`RolloutPoolClient.generate` — get a group of completions from the **rollout server**
  (a shared vLLM GPU), not the trainer's GPU.
* :meth:`RolloutPoolClient.score` — score them on the **reward workers** (CPU/IO boxes), not the
  trainer's GPU.
* :meth:`RolloutPoolClient.sync_weights` — after the optimizer step, push the new LoRA to the pool
  so the next rollout uses the fresh policy (the per-step weight transfer).

The cost win comes from :meth:`RolloutPoolClient.experience_stream`: a **pipelined producer** that
runs *ahead* of the trainer — while the trainer does the optimizer step for batch ``t`` on its GPU,
a background thread is already generating + scoring batch ``t+1`` on the rollout/reward servers. So
rollout and reward latency overlap the gradient step and **do not cost trainer-GPU wall-clock** —
which is the whole point: "the reward function latency doesn't matter for the cost." Per-prompt
generation requests are issued concurrently, so the router fans them across the whole GPU fleet.

This client is synchronous (it lives inside TRL/verl trainer loops); the concurrency is handled
with a thread pool and a prefetch queue.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx


@dataclass
class Experience:
    """One batch of rollouts + rewards, ready for a GRPO update. ``completions[i]`` is the group of
    ``n`` samples for ``prompts[i]``; ``rewards[i]`` aligns with it."""

    step: int
    prompts: list
    completions: list[list[str]]
    rewards: list[list[float]]
    gen_seconds: float = 0.0
    score_seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def num_prompts(self) -> int:
        return len(self.prompts)

    @property
    def num_completions(self) -> int:
        return sum(len(g) for g in self.completions)

    @property
    def mean_reward(self) -> float:
        flat = [r for g in self.rewards for r in g]
        return sum(flat) / len(flat) if flat else 0.0


class RolloutPoolClient:
    """Thin HTTP client to a running pool router (see :func:`flash.pool.router.create_pool_app`)."""

    def __init__(
        self,
        base_url: str,
        *,
        adapter: str | None = None,
        base_model: str | None = None,
        reward_id: str = "default",
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.adapter = adapter
        self.base_model = base_model
        self.reward_id = reward_id
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    @classmethod
    def from_env(cls, **kw) -> RolloutPoolClient:
        url = os.environ.get("FLASH_ROLLOUT_POOL_URL", "http://127.0.0.1:8077")
        return cls(url, **kw)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> RolloutPoolClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- registration / weight sync ----
    def register(self, *, uri: str, base_model: str | None = None, replicas: int = 1, place: bool = True) -> dict:
        """Register this run's adapter (``self.adapter``) with the router and (by default) eagerly
        place it on ``replicas`` backends so the first rollout is warm."""
        if not self.adapter:
            raise ValueError("RolloutPoolClient.adapter must be set to register an adapter")
        base = base_model or self.base_model
        if not base:
            raise ValueError("base_model required to register an adapter")
        body = {"name": self.adapter, "base_model": base, "uri": uri, "replicas": replicas, "place": place}
        r = self._client.post(f"{self.base_url}/adapters", json=body)
        r.raise_for_status()
        return r.json()

    def sync_weights(self, uri: str | None = None) -> dict:
        """Per-step weight transfer: tell the pool this run has new LoRA weights; the router
        hot-swaps them onto every backend hosting the adapter."""
        if not self.adapter:
            raise ValueError("RolloutPoolClient.adapter must be set to sync weights")
        r = self._client.post(f"{self.base_url}/adapters/{self.adapter}/sync", json={"uri": uri} if uri else {})
        r.raise_for_status()
        return r.json()

    # ---- generation / reward (single requests) ----
    def generate(
        self,
        messages: list | str,
        *,
        n: int = 1,
        max_tokens: int = 512,
        temperature: float = 1.0,
        model: str | None = None,
        extra: dict | None = None,
    ) -> list[str]:
        """One chat generation -> list of ``n`` completion strings (a GRPO group for one prompt).
        Routed to a shared GPU by the pool; this never touches the trainer's GPU."""
        msgs = [{"role": "user", "content": messages}] if isinstance(messages, str) else messages
        body = {
            "model": model or self.adapter or self.base_model,
            "messages": msgs,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra:
            body.update(extra)
        r = self._client.post(f"{self.base_url}/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        return [c["message"]["content"] for c in data.get("choices", [])]

    def score(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        info: Sequence[dict] | None = None,
    ) -> list[float]:
        """Score completions on the reward workers (off the trainer GPU)."""
        from flash.pool.rewards import score_request

        body = score_request(prompts, completions, info, reward_id=self.reward_id)
        r = self._client.post(f"{self.base_url}/rewards/score", json=body)
        r.raise_for_status()
        return [float(s) for s in r.json().get("scores", [])]

    # ---- the pipelined producer (overlap rollout+reward with training) ----
    def experience_stream(
        self,
        batches: Iterable[Sequence],
        *,
        n: int = 8,
        max_tokens: int = 512,
        temperature: float = 1.0,
        prefetch: int = 2,
        max_concurrency: int = 64,
        score_fn: Callable[[Sequence, list[list[str]]], list[list[float]]] | None = None,
        gen_extra: dict | None = None,
    ) -> Iterator[Experience]:
        """Yield :class:`Experience` per prompt batch, generating + scoring **ahead** of the
        consumer so the work overlaps the trainer's optimizer step.

        ``batches`` is an iterable of prompt batches (each a sequence of prompts — strings or chat
        message-lists). A background thread fetches up to ``prefetch`` batches ahead, issuing the
        per-prompt generation calls concurrently (``max_concurrency``) so the router spreads them
        across the GPU fleet. ``score_fn`` overrides the default pool reward call (e.g. to use a
        local rubric); it must return one reward per completion per prompt.
        """
        q: queue.Queue = queue.Queue(maxsize=max(1, prefetch))
        sentinel = object()
        err_box: list[BaseException] = []

        def _produce() -> None:
            try:
                with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
                    for step, batch in enumerate(batches):
                        prompts = list(batch)
                        t0 = time.monotonic()
                        completions = list(
                            pool.map(
                                lambda p: self.generate(
                                    p, n=n, max_tokens=max_tokens, temperature=temperature, extra=gen_extra
                                ),
                                prompts,
                            )
                        )
                        gen_s = time.monotonic() - t0
                        t1 = time.monotonic()
                        rewards = (
                            score_fn(prompts, completions)
                            if score_fn is not None
                            else self._score_groups(prompts, completions)
                        )
                        score_s = time.monotonic() - t1
                        q.put(
                            Experience(
                                step=step,
                                prompts=prompts,
                                completions=completions,
                                rewards=rewards,
                                gen_seconds=gen_s,
                                score_seconds=score_s,
                            )
                        )
            except BaseException as e:
                err_box.append(e)
            finally:
                q.put(sentinel)

        thread = threading.Thread(target=_produce, name="flash-rollout-producer", daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
        thread.join()
        if err_box:
            raise err_box[0]

    def _score_groups(self, prompts: Sequence, completions: list[list[str]]) -> list[list[float]]:
        """Default group scoring: flatten to the reward workers, then reshape back to groups."""
        flat_prompts: list[str] = []
        flat_completions: list[str] = []
        shapes: list[int] = []
        for p, group in zip(prompts, completions, strict=True):
            ptext = p if isinstance(p, str) else _last_user(p)
            shapes.append(len(group))
            for c in group:
                flat_prompts.append(ptext)
                flat_completions.append(c)
        flat_scores = self.score(flat_prompts, flat_completions) if flat_completions else []
        out: list[list[float]] = []
        i = 0
        for k in shapes:
            out.append(flat_scores[i : i + k])
            i += k
        return out


def _last_user(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""
