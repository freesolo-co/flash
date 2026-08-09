"""gather rollout evidence for a grpo/opd submit, from the client.

what is measured here is realized generation LENGTH, and nothing else. length is a property of
(model, prompt, sampling params), so a hosted endpoint measures the same quantity the rented gpu
would -- it transfers between hosts. seconds do not: they depend on the machine, its cpu and its
network path. that is why no grading latency is timed here even though the env's scorer is loadable
in this process; the worker times its own scorer on the machine that will run it.

the env is loaded locally because prompt length is half the measurement and only the env's own
prompt_messages() produces the real prompts.

entirely best-effort. every failure returns None and the submit proceeds on the declared cap, which
is the pricing this path had before any of this existed.
"""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from flash._internal.logging import get_logger

logger = get_logger("flash.cli")

_DEFAULT_ENVIRONMENT_PATH = "environment.py"

# how many dataset rows to draw prompts from. prompt difficulty dominates length variance, so a
# handful of distinct prompts beats many draws from one.
_PROMPT_ROWS = 8

# the only message keys this path can reproduce faithfully. the workers pass the original message
# dicts into the chat template, so anything else (`name`, tool-call fields) can change what the
# template renders -- and this sampler would not send it.
_SAMPLABLE_MESSAGE_KEYS = frozenset({"role", "content"})

# wall-clock ceiling for the env's own data hooks. generous: a dataset() that pulls a real corpus
# legitimately takes tens of seconds, and expiring early would silently return an otherwise
# measurable run to cap pricing. it exists to stop UNBOUNDED blocking, not to police slow hooks.
_LOCAL_HOOK_DEADLINE_S = 120.0


def collect_for_submit(
    client,
    spec,
    *,
    runtime_secrets: Mapping[str, str] | None = None,
    debug: bool = False,
    on_environment_loaded: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    """measured rollout aggregates for ``spec``, or None when nothing could be measured.

    ``runtime_secrets`` are the run's declared secrets as the worker will receive them. they are
    exported for the duration of the local measurement only (see ``_runtime_secret_env``), because
    the env code run here is the same code the worker runs: an env whose reward() reads a declared
    key would otherwise raise locally, and the broad except below would quietly return the quote to
    the cap. they are never persisted and never travel in the evidence, which is aggregates only.

    ``on_environment_loaded`` is called with the name of each env stage that actually ran
    ("import", "dataset", "prompt_messages"). returning None does NOT mean nothing ran: a multi-turn
    env, a dataset with no samplable prompts, or a dead sampler all execute some of the user's code
    first. the caller reports what ran from these calls rather than inferring it from the payload,
    and per-stage rather than one flag -- a multi-turn env declines after the import alone, so a
    single "it ran" would still overclaim which hooks were called.
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
        with _runtime_secret_env(spec, runtime_secrets), _restored_host_rng():
            return _collect(client, spec, on_environment_loaded=on_environment_loaded)
    except ImportError as exc:
        if debug:
            raise
        # the base `freesolo-flash` package declares no dependencies, and the env loader needs the
        # `freesolo` SDK -- so an isolated `uv tool`/`pipx` install cannot load an environment at
        # all. still fail open, but say so once: the user set a sampler key expecting a measured
        # quote, and silently pricing at the cap looks identical to the feature working.
        logger.warning(
            "rollout profiling is unavailable in this install, so this quote uses the declared "
            "completion cap: %s",
            exc,
        )
        return None
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
    return ""


def _thinking_for(spec) -> bool | None:
    """the reasoning setting a draw must reproduce, or None when the model has no reasoning mode.

    the worker renders every prompt with enable_thinking=spec.thinking, and BOTH values need saying
    out loud: an endpoint that reasons by default would answer a thinking=False run with traces the
    run never generates, which moves completion length by far more than the trust gate tolerates.
    so the request states the setting explicitly and the response is checked against it, rather than
    either value being left to the endpoint's default.
    """
    from flash.core.catalog import MODELS

    model = MODELS.get(getattr(spec, "model", ""))
    if model is not None and getattr(model, "thinking", "none") == "none":
        # nothing to diverge. keyed off the catalog rather than hardcoded, so this stays true if a
        # non-reasoning model is ever added.
        return None
    return bool(getattr(spec, "thinking", False))


def _collect(
    client, spec, *, on_environment_loaded: Callable[[str], None] | None = None
) -> dict[str, Any] | None:
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
        # the user's module has executed by now. every return below this point is a decline, and
        # each one still ran their code -- so the caller is told here, not from the return value.
        _ran("import", on_environment_loaded)
        if getattr(environment, "multi_turn", False):
            # the worker generates an assistant turn per step_episode call and trains on the whole
            # transcript. one hosted completion is the FIRST turn only, so its token mean
            # undercounts an episode -- and would still pass the trust gate. decline instead.
            return None
        # the workers seed before their first environment call, and env code may consume python or
        # numpy randomness while building its dataset. without this the rows measured here depend on
        # this process's incidental rng state, while the server binds the profile to spec.seed -- so
        # two runs with different seeds could share a measurement drawn from a different sample.
        # seed_host_rngs rather than seed_training_rngs: the same split the sft workload profile
        # uses, because it must not import torch into the cli.
        _seed_environment_rngs(spec)
        # bounded: dataset() is user code that may block on a download or a stalled mount, and a
        # submit must not hang because the optional sampler happened to be configured. no exception
        # is raised by a hang, so the outer `except` could never regain control.
        dataset = _within(_LOCAL_HOOK_DEADLINE_S, environment.dataset)
        _ran("dataset", on_environment_loaded)
        if not dataset:
            return None
        # the grpo and opd workers fence the training population to a max_examples prefix
        # (rl_train.py and opd_train.py both do `train = train[:max_examples]`). measuring outside
        # that fence would sample rows the run never consumes, and with a small max_examples those
        # rows could set the whole measured length and grading latency.
        dataset = _training_population(spec, dataset)
        if not dataset:
            return None

        # bounded for the same reason as dataset(): prompt_messages() is user code, called once per
        # row, and a stall in any one of them would hang the submit.
        prompts = _within(_LOCAL_HOOK_DEADLINE_S, _prompts_from, environment, dataset)
        _ran("prompt_messages", on_environment_loaded)
        if not prompts:
            return None

        # NO grading latency is measured here, for either algorithm. seconds measured on this
        # machine do not describe the worker: reward() is either cpu-bound (a different cpu) or an
        # external judge call (a different network path and geography), and the number would then be
        # multiplied by every completion of every step. this is the same rule the generation half
        # already follows -- token COUNTS transfer between hosts, SECONDS do not -- and the worker
        # already times its own scorer on the machine that will run it (rl_train.py
        # `_log_reward_profile`). timing it here would also call a possibly paid judge locally to
        # produce a number the run should not be priced from.
        return collect_rollout_evidence(
            model=spec.model,
            prompts=prompts,
            max_completion_tokens=_completion_cap(spec),
            temperature=_sampling_temperature(spec),
            # a run configured with stop strings ends generation at the first one, so a sample
            # drawn without them measures a longer completion than training will ever produce.
            stop_sequences=tuple(getattr(spec.train, "stop_sequences", ()) or ()),
            # stated explicitly and then verified against each response, because the worker renders
            # every prompt with this value and the endpoint would otherwise apply its own default.
            thinking=_thinking_for(spec),
        )


@contextmanager
def _restored_host_rng() -> Iterator[None]:
    """run with the caller's python/numpy rng state restored afterward.

    this is a cli process that keeps running after the measurement -- it goes on to submit the run.
    seeding for the env's benefit and leaving the seed in place would make every later random draw
    in this process deterministic as a side effect of asking for a quote.

    every generator _seed_environment_rngs seeds is restored here, torch included. restoring only
    some of them would leave the process half-seeded, which is the same defect in a subset.
    """
    python_state = random.getstate()
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except ImportError:
        pass
    torch = _optional_torch()
    torch_state = torch.get_rng_state() if torch is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            import numpy as np

            np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)


def _seed_environment_rngs(spec) -> None:
    """seed the generators the workers seed, so this samples the rows training will see.

    the workers call seed_training_rngs, which also seeds torch. env code that draws its dataset
    through torch would otherwise select rows from this process's incidental torch state: measured,
    two cli processes quoting the same spec.seed picked different rows ([44, 19, 93, 90, 71] vs
    [52, 74, 73, 81, 76]) where the worker was stable -- a profile describing work the run never does.

    cpu generators only. the worker also seeds cuda because model init consumes it, and this
    deliberately never reaches model init -- the client host generally has no gpu at all.
    """
    from flash.engine.worker.runtime.rng import seed_host_rngs

    seed = int(getattr(spec, "seed", 0) or 0)
    seed_host_rngs(seed)
    torch = _optional_torch()
    if torch is not None:
        torch.manual_seed(seed)


def _optional_torch():
    """the torch module, or None where it is not installed.

    torch is not a cli dependency, so this stays optional in the same shape ``seed_host_rngs`` uses
    for numpy. the import cost is irrelevant on this path: by the time it runs, the caller has
    already pulled the environment package over the network and is about to spend up to
    SAMPLING_DEADLINE_S on hosted draws.
    """
    try:
        import torch
    except ImportError:
        return None
    return torch


def _ran(stage: str, report: Callable[[str], None] | None) -> None:
    """record that one env stage actually executed."""
    if report is not None:
        report(stage)


def _within(deadline_s: float, call, *args):
    """``call(*args)``, or None when it does not finish within ``deadline_s``.

    a daemon thread rather than a signal: signals only work on the main thread and this is library
    code inside a cli, and rather than a subprocess because the env is already loaded in THIS
    process. the cost is that an expired call keeps running in the background -- it cannot be
    killed -- but it is a daemon, so it cannot keep the interpreter alive at exit.

    that trade is only acceptable because this whole path is advisory: an expired hook returns the
    quote to the declared cap, which is what an unmeasured run was priced at anyway.
    """
    import threading

    box: list = []

    def _run() -> None:
        try:
            box.append(call(*args))
        except Exception:
            # reported as "no measurement", like any other failure on this path. Exception rather
            # than BaseException so a KeyboardInterrupt still propagates out of the worker.
            box.append(None)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(deadline_s)
    return box[0] if box else None


def _training_population(spec, dataset):
    """the dataset prefix training will actually consume, mirroring the workers' own fence."""
    max_examples = int(getattr(spec.train, "max_examples", 0) or 0)
    if max_examples > 0:
        return dataset[:max_examples]
    return dataset


def _prompts_from(environment, dataset) -> list[list[dict]] | None:
    """chat prompts of the first distinct rows, or None when this dataset must not be measured.

    uses ``prompt_messages`` rather than the raw ``input`` field so the sampled prompt carries the
    system turn and any templating the env applies. that is what sets prompt length, and prompt
    length is half of what this profile measures.

    the message list is passed through with its ROLES INTACT rather than flattened into one user
    turn: the worker sends what prompt_messages returns, and collapsing a system turn changes both
    chat-template tokenization and how the model answers, so a flattened sample would describe a
    different distribution than the run being quoted.

    a multimodal row aborts the whole measurement rather than being skipped. the workers detect
    ``record_has_images`` and run ``normalize_prompt_images``, so the real request carries image
    blocks and the processor's expanded image tokens -- far more prompt tokens than the text alone,
    and a different completion distribution. skipping such rows is worse than declining: a mixed
    dataset would still yield a full set of text-only draws, pass the trust gate, and underprice
    every multimodal step.
    """
    from flash.content.multimodal import record_has_images

    prompts: list[list[dict]] = []
    for example in dataset[:_PROMPT_ROWS]:
        try:
            messages = environment.prompt_messages(example)
        except Exception:
            continue
        if record_has_images(example if isinstance(example, dict) else {}, messages):
            return None
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

    a message carrying keys beyond role/content is rejected for the same reason. the workers hand
    the ORIGINAL dicts to ``apply_chat_template``, so a template that reads ``name`` renders a
    different prompt than this reconstruction would -- different tokenization, and potentially a
    different completion length. reconstructing role/content only would silently drop that and
    report a confident measurement of a prompt the run never sends.
    """
    if not isinstance(messages, list):
        return []
    out: list[dict] = []
    for message in messages:
        if not isinstance(message, dict):
            return []
        if set(message) - _SAMPLABLE_MESSAGE_KEYS:
            return []
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
            return []
        out.append({"role": role, "content": content})
    return out


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
