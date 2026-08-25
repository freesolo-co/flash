"""Small, serializable environment interface for SFT/RL jobs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar

from flash._internal.channel import CLI_NAME

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

    Failures cost at most ``cap`` extra calls. Three separate things are required for that bound,
    because three different shapes of ``fn`` each defeat a different one, and reward scoring
    produces all three:

    1. Only ``cap`` items are ever in flight, and the window refills as results are CONSUMED.
       Submitting everything up front and relying on ``cancel_futures`` bounds nothing when ``fn``
       is FAST: the workers drain the queue before the consumer reads its first result, so nothing
       is left to cancel. At width 8 failing on item 1, eager submission ran 36 of 40 and 10000 of
       10000.
    2. Completions are consumed as they ARRIVE, not in input order, and placed back by index.
       Consuming in input order (what ``pool.map`` yields) defers the raise behind a SLOW LEADING
       item until the batch drains: 29 of 40 at width 8.
    3. Refilling stops as soon as a failure is PENDING rather than observed. A call that fails
       SLOWLY -- an HTTP timeout or a late 5xx, which is the normal shape of a real failure -- sits
       in flight while fast successes are consumed around it, and each consumed success pulls
       another item in. Waste then tracks error-latency over success-latency instead of ``cap``:
       measured 2000 of 2000 at width 8 before this, against 8 after.

    The bound is on calls STARTED, not billed twice by accident: `rl_train.py` discards the batch on
    any raise and re-scores serially, so everything started here is charged again.
    """
    if serial or len(items) <= 1:
        return [fn(item) for item in items]
    results: list[_R] = [None] * len(items)  # type: ignore[list-item]
    pool = ThreadPoolExecutor(max_workers=min(cap, len(items)))
    try:
        pending: dict[Future[_R], int] = {}
        cursor = 0
        while True:
            # the window is anchored to the OLDEST unconsumed item, not to how many futures happen
            # to be settled. an in-flight call that will fail is indistinguishable from a slow one
            # that will succeed, so nothing can be inferred from its state -- the only safe rule is
            # to never run ahead of the item whose outcome is still unknown.
            while cursor < len(items) and cursor - min(pending.values(), default=cursor) < cap:
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
    image_observations: bool

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

    image_observations: ClassVar[bool] = False

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


def with_system_prompt(messages: list[dict], contract_text: str) -> list[dict]:
    """Ensure the training contract rides as the system prompt (fill blank / prepend).

    Shared, not duplicated: the control-plane quote and the worker must place the contract
    identically or they tokenize different prompts, and the frozen quote is the invoice.
    """
    system_text = str(contract_text or "").strip()
    out = [dict(message) for message in messages]
    if not system_text:
        return out
    first_blank_system_index: int | None = None
    for index, message in enumerate(out):
        if str(message.get("role") or "").strip().lower() != "system":
            continue
        content = message.get("content")
        has_content = bool(content.strip()) if isinstance(content, str) else bool(content)
        if has_content:
            return out
        if first_blank_system_index is None:
            first_blank_system_index = index
    if first_blank_system_index is not None:
        out[first_blank_system_index]["content"] = system_text
        return out
    return [{"role": "system", "content": system_text}, *out]


FREESOLO_WORKER_SPEC = "freesolo>=0.4.2"


def worker_pip_for_env(env_id: str) -> list[str]:
    """Pip deps the GPU worker needs to run a Freesolo environment."""
    return [FREESOLO_WORKER_SPEC]


def worker_pip_with_extras(env_id: str, extras: Iterable[str] | None) -> list[str]:
    """Worker requirements plus the author's ``[environment] pip``, in install order.

    The worker requirement stays first and is never displaced: a scorer's third-party dependency is
    additional to what Flash's own worker needs, not a replacement for it. Exact repeats are dropped
    so restating the worker spec cannot install it twice; a conflicting version pin is left for pip
    to resolve and report, which it does with a clearer message than a guard here could.
    """
    combined = list(worker_pip_for_env(env_id))
    for item in extras or ():
        requirement = str(item).strip()
        if requirement and requirement not in combined:
            combined.append(requirement)
    return combined


def load_environment(
    env_id: str, params: dict | None = None, resolved_sha: str | None = None
) -> Environment:
    """Load a Freesolo SDK environment and wrap it in Flash's protocol."""
    params = params or {}
    from flash.envs.loading.adapter import load_freesolo_environment

    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to the id returned by "
            f"`{CLI_NAME} env push --project <project-uuid> --name <name>` "
            "(for example 'your-org/your-project/your-env')"
        )
    # resolved_sha is positional-only so a user param named "resolved_sha" can't shadow it.
    return load_freesolo_environment(env_id, resolved_sha or None, **params)
