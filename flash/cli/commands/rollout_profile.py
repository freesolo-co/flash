"""gather rollout evidence for a grpo/opd submit, from the client.

this runs here rather than on the control plane because of where the two halves can be measured.
generation length needs a hosted inference endpoint, which either side could call; grading latency
needs the environment's own scorer, and the control plane never executes user environment code.
doing both here keeps one download and one place that can fail.

entirely best-effort. every failure returns None and the submit proceeds on the declared cap, which
is the pricing this path had before any of this existed.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_DEFAULT_ENVIRONMENT_PATH = "environment.py"

# how many dataset rows to draw prompts from. prompt difficulty dominates length variance, so a
# handful of distinct prompts beats many draws from one.
_PROMPT_ROWS = 8

# references handed to the grading timer. three real completions is what `profile_reward_latency`
# uses by default, plus its discarded warm-up call.
_REWARD_REFERENCES = 4


def collect_for_submit(
    client,
    spec,
    *,
    runtime_secrets: Mapping[str, str] | None = None,
    debug: bool = False,
) -> dict[str, Any] | None:
    """measured rollout aggregates for ``spec``, or None when nothing could be measured.

    ``runtime_secrets`` are the run's declared secrets as the worker will receive them. they are
    exported for the duration of the local measurement only (see ``_runtime_secret_env``), because
    the env code run here is the same code the worker runs: an env whose reward() reads a declared
    key would otherwise raise locally, and the broad except below would quietly return the quote to
    the cap. they are never persisted and never travel in the evidence, which is aggregates only.
    """
    if spec.algorithm not in ("grpo", "opd"):
        return None
    from flash.engine.profiling.rollout_sampler import sampler_credentials

    _base_url, api_key = sampler_credentials()
    if not api_key:
        # no sampler configured. the common case, and not a problem: the quote falls back to the cap.
        return None
    if unsamplable_reason(spec):
        # the hosted draw would not reproduce this run's generation, and a measurement that
        # describes different work must not set a price. declining returns the quote to the cap.
        return None

    try:
        with _runtime_secret_env(spec, runtime_secrets):
            return _collect(client, spec)
    except Exception:
        if debug:
            raise
        return None


@contextmanager
def _runtime_secret_env(spec, runtime_secrets: Mapping[str, str] | None) -> Iterator[None]:
    """export this run's DECLARED secrets for the measurement, then restore os.environ.

    a declared secret that lives only in a local .env file is collected for submission but never
    reaches ``os.environ``, so a reward() that reads one -- an external judge, typically -- raises
    here and the measurement is lost. the worker runs that same reward() with the secret present, so
    the local failure is an artifact of this process, not of the env.

    scoped three ways. only keys the spec DECLARES are exported, so an unrelated secret that
    happened to be collected (WANDB_API_KEY is always collected) is not handed to env code. a key
    already set in the process is left alone, since the user's own shell value is what the rest of
    this command already used. and every key set here is removed on exit, including on the exception
    path, so nothing outlives the measurement.
    """
    declared = {str(key) for key in (getattr(spec.environment, "secrets", ()) or ())}
    exported = {
        key: value
        for key, value in (runtime_secrets or {}).items()
        if key in declared and key not in os.environ and value
    }
    os.environ.update(exported)
    try:
        yield
    finally:
        for key in exported:
            os.environ.pop(key, None)


def unsamplable_reason(spec) -> str:
    """why a hosted draw cannot stand for this run's generation, or "" when it can.

    the sampler draws ONE unconstrained completion from the base model. a config that changes what
    generation is -- different weights, a decoding grammar, or many turns per episode -- produces a
    length distribution the draw does not describe, and a measurement of different work must never
    replace the cap. these are declined rather than approximated: fail-open costs nothing (the quote
    is the cap-based one this path always used), while a confident wrong number sets a price.
    """
    train = getattr(spec, "train", None)
    if getattr(train, "init_from_adapter", ""):
        # the worker loads the warm-start adapter before its first rollout. an adapter changes
        # stopping behaviour and length, and the hosted endpoint serves the base weights.
        return "warm-start adapter"
    if getattr(train, "structured_outputs", ""):
        # grammar-constrained decoding has its own length distribution, and the hosted request
        # carries no schema. forwarding one is a larger contract than this path has.
        return "structured outputs"
    if getattr(spec, "model_revision", ""):
        # the worker loads the pinned revision; the hosted endpoint serves whatever its model id
        # currently points at. keying the revision in the digest stops the measurement being reused
        # for a DIFFERENT run, but cannot make this draw come from the pinned weights.
        return "pinned model revision"
    if getattr(spec, "thinking", False):
        # the worker renders prompts with enable_thinking=spec.thinking. the chat-completions
        # request has no portable way to say that, and reasoning traces move completion length by
        # far more than the trust gate's tolerance, so a wrong default is not a small error.
        return "thinking mode"
    return ""


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
        if getattr(environment, "multi_turn", False):
            # the worker generates an assistant turn per step_episode call and trains on the whole
            # transcript. one hosted completion is the FIRST turn only, so its token mean
            # undercounts an episode -- and would still pass the trust gate. decline instead.
            return None
        dataset = environment.dataset()
        if not dataset:
            return None
        # the grpo and opd workers fence the training population to a max_examples prefix
        # (rl_train.py and opd_train.py both do `train = train[:max_examples]`). measuring outside
        # that fence would sample rows the run never consumes, and with a small max_examples those
        # rows could set the whole measured length and grading latency.
        dataset = _training_population(spec, dataset)
        if not dataset:
            return None

        prompts = _prompts_from(environment, dataset)
        if not prompts:
            return None

        # only grpo prices reward latency (analytical.py adds `completions x latency` in the grpo
        # branch alone); opd supervises from the teacher distribution and uses zero task reward. so
        # timing an opd env's reward() would call a possibly paid external judge locally to produce
        # a number nothing reads. completion-length sampling still runs for both.
        score_one, references = (
            _reward_inputs(environment, dataset, assistant_completion_text)
            if spec.algorithm == "grpo"
            else (None, None)
        )
        return collect_rollout_evidence(
            model=spec.model,
            prompts=prompts,
            max_completion_tokens=_completion_cap(spec),
            temperature=_sampling_temperature(spec),
            score_one=score_one,
            reward_samples=references,
            # a run configured with stop strings ends generation at the first one, so a sample
            # drawn without them measures a longer completion than training will ever produce.
            stop_sequences=tuple(getattr(spec.train, "stop_sequences", ()) or ()),
        )


def _training_population(spec, dataset):
    """the dataset prefix training will actually consume, mirroring the workers' own fence."""
    max_examples = int(getattr(spec.train, "max_examples", 0) or 0)
    if max_examples > 0:
        return dataset[:max_examples]
    return dataset


def _prompts_from(environment, dataset) -> list[list[dict]]:
    """chat prompts of the first distinct rows, as the env's own prompt builder returns them.

    uses ``prompt_messages`` rather than the raw ``input`` field so the sampled prompt carries the
    system turn and any templating the env applies. that is what sets prompt length, and prompt
    length is half of what this profile measures.

    the message list is passed through with its ROLES INTACT rather than flattened into one user
    turn: the worker sends what prompt_messages returns, and collapsing a system turn changes both
    chat-template tokenization and how the model answers, so a flattened sample would describe a
    different distribution than the run being quoted.
    """
    prompts: list[list[dict]] = []
    for example in dataset[:_PROMPT_ROWS]:
        try:
            messages = environment.prompt_messages(example)
        except Exception:
            continue
        usable = _chat_messages(messages)
        if usable:
            prompts.append(usable)
    return prompts


def _chat_messages(messages) -> list[dict]:
    """the message list as an openai-compatible payload, or [] when it is not one.

    only string content is kept: a multimodal part list is a different measurement problem, and a
    prompt this cannot represent faithfully is better dropped than sent in a shape the run will not
    use -- dropping returns the quote to the cap, sending would price a distribution that is not
    the run's.
    """
    if not isinstance(messages, list):
        return []
    out: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            return []
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            return []
        out.append({"role": role, "content": content})
    return out


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
