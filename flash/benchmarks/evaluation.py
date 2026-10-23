"""Held-out conversation evaluation and concurrent, per-customer adapter requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from flash.benchmarks.schema import Request


def evaluate_episode(
    client, adapter: str, customer: str, episode: dict, recorder, *, max_tokens: int
):
    """Exact match on the final answer; retain full dialogue and failed-request latencies.

    A task with executable tools or a semantic grader supplies its own evaluator hook instead.
    These episodes are a deterministic default for SFT and single-turn GRPO smoke tasks.
    """
    messages = list(episode.get("prefix", []))
    for turn in episode["turns"]:
        messages.append({"role": "user", "content": turn["prompt"]})
        start = recorder.now()
        succeeded = False
        try:
            result = client.chat(adapter, messages, temperature=0.0, max_tokens=max_tokens)
            answer = result["choices"][0]["message"]["content"]
            if not isinstance(answer, str):
                raise ValueError("evaluation requires a text answer")
            succeeded = True
        finally:
            recorder.emit(
                Request(
                    start_s=start,
                    end_s=recorder.now(),
                    customer=customer,
                    adapter=adapter,
                    success=succeeded,
                )
            )
        messages.append({"role": "assistant", "content": answer})
    return answer.strip() == episode["expected"].strip()


def evaluate_targets(
    targets: list[tuple], episodes: list[dict], recorder, *, max_tokens: int = 128
):
    """Each target has its own credential-bearing client and sequential episode stream."""
    if not targets or not episodes or any(not episode.get("turns") for episode in episodes):
        raise ValueError("evaluation requires targets and nonempty held-out episodes")
    barrier = Barrier(len(targets))

    def evaluate_target(target):
        client, adapter, customer = target
        barrier.wait(timeout=30)
        return [
            evaluate_episode(client, adapter, customer, episode, recorder, max_tokens=max_tokens)
            for episode in episodes
        ]

    with ThreadPoolExecutor(max_workers=len(targets)) as pool:
        results = list(pool.map(evaluate_target, targets))
    scores = [score for group in results for score in group]
    return sum(scores) / len(scores), len(scores)
