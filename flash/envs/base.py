"""Small, serializable environment interface for SFT/RL jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol


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

            if len(items) <= 1 or not getattr(self, "reward_thread_safe", True):
                episode_rewards = [score_one(item) for item in items]
            else:
                pool = ThreadPoolExecutor(max_workers=min(16, len(items)))
                try:
                    futures = {
                        pool.submit(score_one, item): index for index, item in enumerate(items)
                    }
                    episode_rewards = [0.0] * len(items)
                    for future in as_completed(futures):
                        episode_rewards[futures[future]] = future.result()
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)

        return [
            RolloutReward(episode=float(episode_reward), turns=None)
            for episode_reward in episode_rewards
        ]

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        gold = str(example.get("output") or "").strip()
        # `"" in x` is always True — guard so missing output doesn't grade everything correct.
        return bool(gold) and gold in (completion or "")


FREESOLO_WORKER_SPEC = "freesolo>=0.4.0"


def worker_pip_for_env(env_id: str) -> list[str]:
    """Pip deps the GPU worker needs to run a Freesolo environment."""
    return [FREESOLO_WORKER_SPEC]


def load_environment(
    env_id: str, params: dict | None = None, resolved_sha: str | None = None
) -> Environment:
    """Load a Freesolo SDK environment and wrap it in Flash's protocol."""
    params = params or {}
    from flash.envs.adapter import load_freesolo_environment

    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to the id returned by "
            "`flash env push --project <project-uuid> --name <name>` "
            "(for example 'your-name/your-env')"
        )
    # resolved_sha is positional-only so a user param named "resolved_sha" can't shadow it.
    return load_freesolo_environment(env_id, resolved_sha or None, **params)
