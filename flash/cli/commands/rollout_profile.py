"""gather rollout evidence for a grpo/opd submit, from the client.

this runs here rather than on the control plane because of where the two halves can be measured.
generation length needs a hosted inference endpoint, which either side could call; grading latency
needs the environment's own scorer, and the control plane never executes user environment code.
doing both here keeps one download and one place that can fail.

entirely best-effort. every failure returns None and the submit proceeds on the declared cap, which
is the pricing this path had before any of this existed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

_DEFAULT_ENVIRONMENT_PATH = "environment.py"

# how many dataset rows to draw prompts from. prompt difficulty dominates length variance, so a
# handful of distinct prompts beats many draws from one.
_PROMPT_ROWS = 8

# references handed to the grading timer. three real completions is what `profile_reward_latency`
# uses by default, plus its discarded warm-up call.
_REWARD_REFERENCES = 4


def collect_for_submit(client, spec, *, debug: bool = False) -> dict[str, Any] | None:
    """measured rollout aggregates for ``spec``, or None when nothing could be measured."""
    if spec.algorithm not in ("grpo", "opd"):
        return None
    from flash.engine.profiling.rollout_sampler import sampler_credentials

    _base_url, api_key = sampler_credentials()
    if not api_key:
        # no sampler configured. the common case, and not a problem: the quote falls back to the cap.
        return None

    try:
        return _collect(client, spec)
    except Exception:
        if debug:
            raise
        return None


def _collect(client, spec) -> dict[str, Any] | None:
    from flash.content.multimodal import assistant_completion_text
    from flash.engine.profiling.rollout_evidence import collect_rollout_evidence
    from flash.envs.loader import load_freesolo_environment
    from flash.envs.pull import pull_environment_package_from_archive

    with tempfile.TemporaryDirectory(prefix="flash-rollout-profile-") as workdir:
        package = client.download_env_package(spec.environment.id)
        root = pull_environment_package_from_archive(package, Path(workdir) / "package")
        entrypoint = (root / _DEFAULT_ENVIRONMENT_PATH).resolve()
        # absolute, so the loader takes its local-file branch. a relative dir matches the managed-slug
        # pattern and would resolve remotely, re-downloading what was just extracted.
        environment = load_freesolo_environment(str(entrypoint), **(spec.environment.params or {}))
        dataset = environment.dataset()
        if not dataset:
            return None

        prompts = _prompts_from(environment, dataset)
        if not prompts:
            return None

        score_one, references = _reward_inputs(environment, dataset, assistant_completion_text)
        return collect_rollout_evidence(
            model=spec.model,
            prompts=prompts,
            max_completion_tokens=_completion_cap(spec),
            temperature=_sampling_temperature(spec),
            score_one=score_one,
            reward_samples=references,
        )


def _prompts_from(environment, dataset) -> list[str]:
    """prompt text of the first distinct rows, rendered by the env's own prompt builder.

    uses ``prompt_messages`` rather than the raw ``input`` field so the sampled prompt carries the
    system turn and any templating the env applies. that is what sets prompt length, and prompt
    length is half of what this profile measures.
    """
    prompts: list[str] = []
    for example in dataset[:_PROMPT_ROWS]:
        try:
            messages = environment.prompt_messages(example)
        except Exception:
            continue
        text = _messages_to_text(messages)
        if text:
            prompts.append(text)
    return prompts


def _messages_to_text(messages) -> str:
    """flatten a chat prompt to the user text the sampler sends."""
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n\n".join(parts).strip()


def _reward_inputs(environment, dataset, assistant_completion_text):
    """(scorer, references) for grading-latency timing, or (None, None) when unavailable.

    grades the env's own gold completions: they are the closest available stand-in for what training
    will grade, and going through the env's accessor keeps block-structured outputs from being
    stringified into a python repr.
    """
    reward = getattr(environment, "reward", None)
    if not callable(reward) or not getattr(environment, "reward_thread_safe", True):
        # an env that declares itself thread-unsafe must not be graded off-thread here: the timer
        # runs the scorer on another thread, which could mutate scorer state before training does.
        return None, None
    if getattr(environment, "multi_turn", False):
        # a multi-turn episode is graded per episode, not per completion, so timing one completion
        # would measure a different operation than the one being priced.
        return None, None

    references: list[tuple[int, str]] = []
    for index, example in enumerate(dataset[:_REWARD_REFERENCES]):
        try:
            text = assistant_completion_text(environment.sft_completion(example))
        except Exception:
            continue
        if isinstance(text, str) and text.strip():
            references.append((index, text))
    if not references:
        return None, None
    return (lambda index, completion: float(reward(completion, dataset[index]))), references


def _completion_cap(spec) -> int:
    """the generation cap this run will actually use.

    mirrors ``RunConfig.normalized()``'s rollout branch exactly. sampling under a different cap than
    the run would truncate at a different point, and the truncation rate is what decides whether the
    sample is trustworthy at all.
    """
    cap = getattr(spec.train, "max_completion_tokens", None)
    if cap:
        return int(cap)
    from flash.engine.plan.recipe import RECIPE

    recipe = RECIPE.opd if spec.algorithm == "opd" else RECIPE.rl
    return int(recipe.max_completion_len_thinking if spec.thinking else recipe.max_completion_len)


def _sampling_temperature(spec) -> float:
    """the sampling temperature this run will use, recipe default when unset.

    temperature drives length variance, so sampling at a different one measures a different
    distribution than the one being priced.
    """
    temperature = getattr(spec.train, "temperature", None)
    if temperature is not None:
        return float(temperature)
    from flash.engine.plan.recipe import RECIPE

    recipe = RECIPE.opd if spec.algorithm == "opd" else RECIPE.rl
    return float(recipe.sampling_temperature)
