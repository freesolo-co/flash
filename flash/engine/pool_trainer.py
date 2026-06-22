"""Pool-mode GRPO training loop — the "training server" that pairs with the shared rollout server.

This is the orchestration half of the distributed setup. The expensive trainer GPU does ONLY the
policy update (LoRA forward/backward); everything that would otherwise idle that GPU is pushed off it:

  rollout  -> the shared vLLM pool (``RolloutPoolClient.generate`` via the router)
  reward   -> the off-GPU reward workers (``RolloutPoolClient.score``)
  prefetch -> ``experience_stream`` runs a step ahead, so rollout+reward for step t+1 overlap the
              optimizer of step t. Reward latency therefore never lands on the trainer's critical
              path — "the reward function latency doesn't matter for the cost".

Per step: pull a prefetched :class:`~flash.pool.client.Experience` (rollouts + rewards), compute
GRPO group-normalized advantages, run the injected ``policy_update`` (the GPU step — it writes the
updated LoRA and returns its uri), then ``sync_weights`` pushes the new adapter to the pool so the
next rollout uses the fresh policy. The loop is framework/model/env-agnostic (it "works for
everything"); the GPU-specific bit is isolated in ``policy_update`` so the orchestration is testable
on CPU with a stub.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from flash.pool.client import Experience, RolloutPoolClient

# policy_update(experience, advantages) -> uri of the newly-written LoRA adapter (or None to keep
# the current uri). This is the only GPU-touching step; on a real worker it does the PEFT
# forward/backward and saves the adapter.
PolicyUpdate = Callable[[Experience, list[list[float]]], str | None]


def _cycle_prompts_to(base: Sequence[str], need: int) -> list[str]:
    """Exactly ``need`` prompts: the base list, cycled (repeated) if it is shorter.

    Materializes ``need`` items in a SINGLE pass via ``itertools.cycle`` + ``islice``.
    The previous ``prompts += prompts`` doubling allocated O(need) extra entries and
    could blow memory for a large step count over a tiny prompt set. Same resulting
    order (the base list, repeated). An empty ``base`` (or ``need <= 0``) yields ``[]``
    rather than spinning ``cycle`` forever.
    """
    if not base or need <= 0:
        return []
    return list(itertools.islice(itertools.cycle(base), need))


def compute_group_advantages(rewards: Sequence[Sequence[float]], *, eps: float = 1e-6) -> list[list[float]]:
    """GRPO advantages: within each prompt's group, subtract the group mean and divide by the group
    std (the standard GRPO baseline — no value network). Returns the same nested shape."""
    out: list[list[float]] = []
    for group in rewards:
        g = list(group)
        if not g:
            out.append([])
            continue
        mean = sum(g) / len(g)
        var = sum((r - mean) ** 2 for r in g) / len(g)
        std = var**0.5
        out.append([(r - mean) / (std + eps) for r in g])
    return out


@dataclass
class StepResult:
    step: int
    mean_reward: float
    mean_abs_advantage: float
    num_prompts: int
    num_completions: int
    gen_seconds: float
    score_seconds: float
    adapter_version: int
    new_uri: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class GRPOPoolLoop:
    """Drives GRPO over a shared rollout pool. See module docstring."""

    client: RolloutPoolClient
    group_size: int = 8
    max_tokens: int = 512
    temperature: float = 1.0
    prefetch: int = 2
    max_concurrency: int = 64
    # Optional override for reward scoring (e.g. a local rubric instead of the pool reward workers).
    score_fn: Callable[[Sequence, list[list[str]]], list[list[float]]] | None = None

    def register(self, *, uri: str, replicas: int = 1) -> dict:
        return self.client.register(uri=uri, replicas=replicas, place=True)

    def run(
        self,
        prompt_batches: Iterable[Sequence],
        policy_update: PolicyUpdate,
        *,
        steps: int | None = None,
        on_step: Callable[[StepResult], None] | None = None,
    ) -> list[StepResult]:
        """Run the distributed GRPO loop. ``prompt_batches`` yields one prompt batch per step;
        ``policy_update`` is the GPU optimizer step. Stops after ``steps`` (or when the prompts run
        out). Returns one :class:`StepResult` per step."""
        results: list[StepResult] = []
        # ``contextlib.closing`` guarantees the generator is .close()'d on early break (the ``steps``
        # cap) or on an exception — NOT just on normal exhaustion. ``experience_stream`` only joins
        # its background producer thread (and tears down the rollout ThreadPoolExecutor) inside its
        # own ``finally``, which runs on close(); without this the producer would leak and keep
        # generating/scoring rollouts after the loop returns (live thread + wasted pool spend).
        with contextlib.closing(
            self.client.experience_stream(
                prompt_batches,
                n=self.group_size,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                prefetch=self.prefetch,
                max_concurrency=self.max_concurrency,
                score_fn=self.score_fn,
            )
        ) as stream:
            for exp in stream:
                advantages = compute_group_advantages(exp.rewards)
                new_uri = policy_update(exp, advantages)  # the GPU step (off-loaded everything else)
                sync = self.client.sync_weights(new_uri)  # push fresh policy to the pool
                flat_adv = [abs(a) for g in advantages for a in g]
                res = StepResult(
                    step=exp.step,
                    mean_reward=exp.mean_reward,
                    mean_abs_advantage=(sum(flat_adv) / len(flat_adv)) if flat_adv else 0.0,
                    num_prompts=exp.num_prompts,
                    num_completions=exp.num_completions,
                    gen_seconds=exp.gen_seconds,
                    score_seconds=exp.score_seconds,
                    adapter_version=int(sync.get("version", 0)),
                    new_uri=new_uri,
                )
                results.append(res)
                if on_step is not None:
                    on_step(res)
                if steps is not None and len(results) >= steps:
                    break
        return results


def run() -> dict:
    """Worker entry for ``FLASH_FRAMEWORK=pool``: run GRPO against the shared rollout pool.

    Rollout + reward run on the pool (off this GPU); only the LoRA optimizer step runs here. The
    pool router URL comes from ``FLASH_ROLLOUT_POOL_URL``. Finalizes like the other backends
    (uploads adapter + metrics.json + DONE so the control plane sees the run complete).
    """
    from flash.engine import pool_policy
    from flash.engine import worker as W

    spec = W.JOB_SPEC
    if spec is None:
        raise RuntimeError("FLASH_FRAMEWORK=pool requires a JobSpec (FLASH_JOB_SPEC_JSON/PATH)")
    W.require_active_env()
    tr = spec.train
    model_id = spec.model
    group_size = int(getattr(tr, "group_size", 8) or 8)
    max_len = int(getattr(tr, "max_length", 2048) or 2048)
    resp_len = int(getattr(tr, "max_tokens", 512) or 512)
    steps = int(getattr(tr, "steps", 0) or 0) or None
    lora_rank = int(getattr(tr, "lora_rank", 0) or 32)
    lora_alpha = int(getattr(tr, "lora_alpha", 0) or 0) or None
    batch_prompts = int(os.environ.get("FLASH_POOL_PROMPT_BATCH", "8"))
    adapter = os.environ.get("FLASH_POOL_ADAPTER") or f"{W.hf_prefix().replace('/', '-')}"
    pool_url = os.environ.get("FLASH_ROLLOUT_POOL_URL", "http://127.0.0.1:8077")
    replicas = int(os.environ.get("FLASH_POOL_REPLICAS", "1"))

    t0 = time.time()
    W.heartbeat("rl_start", framework="pool", adapter=adapter, pool=pool_url)
    W.prefetch_model(model_id)

    out_dir = "/tmp/flash_pool_adapter"
    policy_update, current_uri = pool_policy.build_lora_policy_update(
        model_id,
        out_dir=out_dir,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        max_len=max_len,
        render_prompt=W.render_prompt,
    )

    prompts = pool_policy.prompts_from_env(W.ACTIVE_ENV)
    if not prompts:
        raise RuntimeError("pool trainer: active env produced no prompts")
    if steps is not None:
        # Cap the prompt pool to EXACTLY what the step count needs, cycling if the env
        # yields fewer prompts than `need` (memory-efficient — see _cycle_prompts_to).
        prompts = _cycle_prompts_to(prompts, steps * batch_prompts)
    prompt_batches = list(pool_policy.batched(prompts, batch_prompts))

    client = RolloutPoolClient(pool_url, adapter=adapter, base_model=model_id, timeout=600.0)
    loop = GRPOPoolLoop(
        client,
        group_size=group_size,
        max_tokens=resp_len,
        prefetch=int(os.environ.get("FLASH_POOL_PREFETCH", "2")),
    )
    loop.register(uri=current_uri(), replicas=replicas)

    history: list[StepResult] = []

    def _on_step(r: StepResult) -> None:
        history.append(r)
        W.heartbeat("rl_step", step=r.step, mean_reward=r.mean_reward, version=r.adapter_version)

    # policy_update is the GPU step; the loop syncs its returned adapter uri to the pool each step.
    loop.run(prompt_batches, policy_update, steps=steps, on_step=_on_step)

    metrics = {
        "wall_seconds": time.time() - t0,
        "reward_history": [r.mean_reward for r in history],
        "notes": {
            "framework": "pool",
            "steps": len(history),
            "model": model_id,
            "group_size": group_size,
            "adapter": adapter,
            "pool_url": pool_url,
        },
        "pool_steps": [r.__dict__ for r in history],
    }
    _finalize(W, out_dir, current_uri(), metrics)
    client.close()
    return metrics


def _finalize(W, out_dir: str, final_uri: str, metrics: dict) -> None:
    """Upload the trained adapter + metrics.json + DONE (mirrors verl_runner's finalize so the
    control plane marks the run complete rather than orphan-restarting it)."""
    with open("/tmp/metrics.json", "w") as f:
        json.dump(metrics, f)
    try:
        W.hf_upload_folder(final_uri, "adapter", required=True)
        W.hf_upload_file("/tmp/metrics.json", "metrics.json", required=True)
        with open("/tmp/DONE", "w") as f:
            f.write(str(time.time()))
        W.hf_upload_file("/tmp/DONE", "DONE", required=True)
        W.heartbeat("done")
    except Exception as e:
        print(f"[pool] finalize upload failed ({e}) -> plane may orphan-restart", flush=True)
