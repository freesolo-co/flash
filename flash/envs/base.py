"""Small, serializable environment interface for SFT/RL jobs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Protocol, TypeVar

# the two units reward scoring overlaps, kept together so the difference between them is a decision
# rather than a coincidence of where each was written.
#
# SCALAR pools over individual rollouts, each one `reward()` call. GROUP pools over task groups,
# each a BATCHED scorer call that fans out again inside the env's own per-instance pool -- so a
# group is the heavier unit and gets the smaller cap. neither bounds the provider-facing rate: the
# env enforces `max_score_concurrency` across overlapping calls. these only keep a large rollout
# batch from creating one thread per row.
SCALAR_REWARD_CONCURRENCY = 16
REWARD_GROUP_CONCURRENCY = 8

_T = TypeVar("_T")
_R = TypeVar("_R")


def map_bounded(
    items: Sequence[_T], fn: Callable[[_T], _R], *, cap: int, serial: bool = False
) -> list[_R]:
    """Apply ``fn`` to every item, returning results in input order, at most ``cap`` at a time.

    Serial when ``serial`` is set (a scorer that cannot be raced) or when there is nothing to
    overlap -- a pool for a single call is pure overhead.

    Failures cost at most ``cap`` extra calls, because only ``cap`` items are ever submitted: the
    window refills as results are CONSUMED, never ahead of them. Submitting everything up front and
    relying on ``cancel_futures`` does not bound anything when ``fn`` is fast -- the workers drain
    the whole queue before the main thread gets back around to reading a result, so there is nothing
    left to cancel by the time the first failure is seen. Measured at width 8 failing on item 1:
    eager submission ran 36 of 40, and 10000 of 10000; this runs 9.

    Completions are consumed as they ARRIVE, not in input order, and results are placed back by
    index. Awaiting in input order (what ``pool.map`` yields) defers the raise behind a slow leading
    item until the batch drains: at width 8 with a slow head, 29 of 40 against 9 here.

    Both halves matter because they fail in opposite regimes -- fast scorers defeat an eager submit,
    a slow head defeats in-order consumption -- and reward scoring produces both.
    """
    if serial or len(items) <= 1:
        return [fn(item) for item in items]
    results: list[_R] = [None] * len(items)  # type: ignore[list-item]
    pool = ThreadPoolExecutor(max_workers=min(cap, len(items)))
    try:
        pending: dict[Future[_R], int] = {}
        cursor = 0
        while True:
            while cursor < len(items) and len(pending) < cap:
                pending[pool.submit(fn, items[cursor])] = cursor
                cursor += 1
            if not pending:
                return results
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                # raises on the first failed item, dropping the window with the pool teardown
                results[pending.pop(future)] = future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class RolloutReward:
    episode: float
    turns: tuple[float, ...] | None = None


class Environment(Protocol):
    id: str

    def dataset(self) -> list[dict]:
        """Return the training rows (the only split used; eval is on the serving side)."""

    def prompt_messages(self, example: dict) -> list[dict]:
        """Chat messages fed to the model for one example."""

    def sft_completion(self, example: dict) -> list[dict]:
        """Gold completion messages for one SFT example."""

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        """Scalar RL reward for a completion."""

    def rollout_rewards_many(self, items: list[tuple[dict, dict]]) -> list[RolloutReward]:
        """Return typed episode and optional per-turn rewards in input order."""

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        """Boolean correctness scorer the reward can build on."""


@dataclass
class BaseEnvironment:
    id: str

    def dataset(self) -> list[dict]:
        raise NotImplementedError

    def prompt_messages(self, example: dict) -> list[dict]:
        return [{"role": "user", "content": str(example.get("input") or "")}]

    def sft_completion(self, example: dict) -> list[dict]:
        return [{"role": "assistant", "content": str(example.get("output") or "")}]

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        return 1.0 if self.grade(completion, example, state) else 0.0

    def rollout_rewards_many(self, items: list[tuple[dict, dict]]) -> list[RolloutReward]:
        """Return typed rewards, using batched or bounded scalar scoring as available."""
        reward_many = getattr(self, "reward_many", None)
        if callable(reward_many):
            episode_rewards = reward_many(items)
        else:

            def score_one(item: tuple[dict, dict]) -> float:
                example, state = item
                return float(self.reward("", example, state))

            episode_rewards = map_bounded(
                items,
                score_one,
                cap=SCALAR_REWARD_CONCURRENCY,
                serial=not getattr(self, "reward_thread_safe", True),
            )

        return [
            RolloutReward(episode=float(episode_reward), turns=None)
            for episode_reward in episode_rewards
        ]

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        gold = str(example.get("output") or "").strip()
        # `"" in x` is always True — guard so missing output doesn't grade everything correct.
        return bool(gold) and gold in (completion or "")
