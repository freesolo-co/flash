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
from typing import Any, NamedTuple

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

# the prefix flash/envs/loader.py puts on its OWN import failure. the only way to tell "this install
# cannot load environments" from "the user's environment.py imports something missing", which arrive
# here as the same exception type.
_MISSING_ENVIRONMENT_TOOLS = "could not import the Freesolo environment tools"

# marks a stage that STARTED but did not finish -- it raised, or was still running at the deadline.
# the user's code ran either way, which is the fact the dry-run disclosure turns on, but only a
# completed stage may be described as having run on a sample.
_ATTEMPTED_SUFFIX = "?"


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
        #
        # an import inside the user's own environment.py raises here too, and blaming the flash
        # install for that sends them to fix the wrong machine. the loader already distinguishes
        # them -- it re-raises its own failure with a fixed prefix -- so key off that rather than
        # guessing from the module name, which cannot tell a missing SDK dep from a missing env dep.
        if _MISSING_ENVIRONMENT_TOOLS in str(exc):
            logger.warning(
                "rollout profiling is unavailable in this install, so this quote uses the declared "
                "completion cap: %s",
                exc,
            )
        else:
            logger.warning(
                "your environment.py could not be imported, so this quote uses the declared "
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
        # bounded like the env's own hooks. ApiClient.download_env_package hardcodes a 1800s
        # timeout, which is right for an interactive `flash env pull` the user is waiting on and
        # wrong here: a stalled package route would hold `flash train` for half an hour before
        # falling back to the cap-based quote it would have used anyway. bounded at the call site
        # rather than by adding a timeout parameter to the shared client, which is a different
        # subsystem and used by callers that legitimately want the long wait.
        downloaded = _within(
            _LOCAL_HOOK_DEADLINE_S, client.download_env_package, spec.environment.id
        )
        package = downloaded.value
        if not downloaded.completed or package is None:
            # `is None` rather than falsy: only "did not finish" and "raised" are failures here,
            # and an empty body is left to the extractor, which is what validates archives.
            return None
        root = pull_environment_package_from_archive(package, Path(workdir) / "package")
        entrypoint = (root / _DEFAULT_ENVIRONMENT_PATH).resolve()
        # absolute, so the loader takes its local-file branch. a relative dir matches the managed-slug
        # pattern and would resolve remotely, re-downloading what was just extracted.
        # bounded like the hooks below it: module scope is user code too, and an environment.py
        # that blocks at import -- waiting on a mount, a client connect, a model download -- never
        # raises, so the outer `except` could not fail open and the submit would hang.
        # BEFORE the load, because module scope is user code and may draw. an environment.py that
        # picks data or prompt behaviour at import time would otherwise run off this process's
        # incidental rng state, and seeding afterwards cannot undo a value already stored in a
        # module global -- measured, the same spec.seed produced a different draw for every ambient
        # state, while the server binds the profile digest to spec.seed. same digest, different
        # sample. the workers have the same shape: they seed before their first environment call.
        #
        # dataset() and prompt_messages() below are covered by the same call: it seeds once, here,
        # and everything the env does afterwards inherits it. seed_host_rngs rather than
        # seed_training_rngs -- the same split the sft workload profile uses, because this must not
        # import torch into the cli.
        _seed_environment_rngs(spec)
        # reported BEFORE the call: this is the point after which the user's module has run, and a
        # load that raises or hangs never reaches the `_ran` below. without it the dry-run line
        # falls to its "did NOT import your environment.py" branch on exactly the runs where the
        # import is what failed -- and after a timeout the module is still executing in the daemon
        # thread while the user is told it never started.
        _attempted("import", on_environment_loaded)
        loaded = _within(
            _LOCAL_HOOK_DEADLINE_S,
            load_freesolo_environment,
            str(entrypoint),
            **(spec.environment.params or {}),
        )
        environment = loaded.value
        if loaded.error is not None:
            # re-raised so `collect_for_submit` can classify it. the loader distinguishes a missing
            # SDK from a missing env dependency by the message it re-raises with, and that
            # diagnostic is the only signal the user gets that profiling silently did not happen.
            raise loaded.error
        if not loaded.completed or environment is None:
            return None
        # the user's module has executed by now. every return below this point is a decline, and
        # each one still ran their code -- so the caller is told here, not from the return value.
        _ran("import", on_environment_loaded)
        if getattr(environment, "multi_turn", False):
            # the worker generates an assistant turn per step_episode call and trains on the whole
            # transcript. one hosted completion is the FIRST turn only, so its token mean
            # undercounts an episode -- and would still pass the trust gate. decline instead.
            return None
        # bounded: dataset() is user code that may block on a download or a stalled mount, and a
        # submit must not hang because the optional sampler happened to be configured. no exception
        # is raised by a hang, so the outer `except` could never regain control.
        _attempted("dataset", on_environment_loaded)
        built = _within(_LOCAL_HOOK_DEADLINE_S, environment.dataset)
        dataset = built.value
        if not built.completed:
            # still running at the deadline, so it did NOT run "on a small sample" -- saying it did
            # is the same overclaim the per-stage reporting exists to stop. the thread is a daemon
            # and keeps going, which is exactly why completion has to be reported, not attempted.
            return None
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
        _attempted("prompt_messages", on_environment_loaded)
        rendered = _within(_LOCAL_HOOK_DEADLINE_S, _prompts_from, environment, dataset)
        prompts = rendered.value
        if not rendered.completed:
            return None
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
            top_p=_sampling_top_p(spec),
            prompt_budget=_prompt_budget(spec),
            # a run configured with stop strings ends generation at the first one, so a sample
            # drawn without them measures a longer completion than training will ever produce.
            stop_sequences=tuple(getattr(spec.train, "stop_sequences", ()) or ()),
            # stated explicitly and then verified against each response, because the worker renders
            # every prompt with this value and the endpoint would otherwise apply its own default.
            thinking=_thinking_for(spec),
            # the whole training population, of which `prompts` is the leading slice. it is what
            # tells a sample that shrank under over-budget filtering (rows remain to replenish
            # from, so the mix is a fraction of the intended one) from one that is simply the whole
            # measurable workload.
            population_rows=len(dataset),
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


def _attempted(stage: str, report: Callable[[str], None] | None) -> None:
    """record that one env stage BEGAN, whatever it went on to do.

    reported before the call, and separately from ``_ran``, because the user's code has executed by
    then either way. a stage that raises or is still running at the deadline would otherwise leave no
    trace at all, and the dry-run line would tell its author their module was never imported while
    the import was in fact what failed -- or, after a timeout, is still running in the daemon thread.

    the caller distinguishes the two: ``import`` names a hook that COMPLETED, ``import?`` one that
    started and did not. collapsing them would swap one false claim for another, since "dataset() ran
    on a small sample" is exactly what a timed-out dataset() did not do.
    """
    if report is not None:
        report(f"{stage}{_ATTEMPTED_SUFFIX}")


class _Outcome(NamedTuple):
    """what a bounded call did: finished or not, what it returned, and what it raised.

    a NamedTuple rather than a bare tuple so each call site names the field it reads. every caller
    reads ``.completed``/``.value``; only the env load reads ``.error``, because it is the one
    failure the cli can say something useful about.
    """

    completed: bool
    value: Any
    error: BaseException | None


def _within(deadline_s: float, call, /, *args, **kwargs) -> _Outcome:
    """``(completed, value)`` for ``call(*args, **kwargs)``, bounded by ``deadline_s``.

    positional-only, because the kwargs are the user's ``[environment.params]``: a param named
    "call" or "deadline_s" would otherwise collide with this signature and crash the submit.

    a daemon thread rather than a signal: signals only work on the main thread and this is library
    code inside a cli, and rather than a subprocess because the env is already loaded in THIS
    process. the cost is that an expired call keeps running in the background -- it cannot be
    killed -- but it is a daemon, so it cannot keep the interpreter alive at exit.

    that trade is only acceptable because this whole path is advisory: an expired hook returns the
    quote to the declared cap, which is what an unmeasured run was priced at anyway.

    ``completed`` is separate from ``value`` because they answer different questions and a hook can
    return a falsy value legitimately. False means only one thing: still running at the deadline. a
    hook that RAISED completed -- the user's code ran and failed -- and the caller reports it as
    having run, because the notice describes what executed on their machine, not what succeeded.

    a raised exception is returned as ``error``, not flattened into a None ``value``. the two are
    different outcomes and one caller has to tell them apart: the loader raises ImportError to say
    the SDK or the env's own dependency is missing, and ``collect_for_submit`` exists to turn that
    into a one-line diagnostic. collapsing it here made that handler unreachable -- measured, a
    missing-SDK install returned None with no warning at all, which is exactly the silent fall back
    to cap pricing the handler was written to prevent. returning it does not change any behaviour on
    its own: a caller that ignores ``error`` still sees the same ``(completed, value)`` it did.
    """
    import threading

    box: list = []
    raised: list[BaseException] = []

    def _run() -> None:
        try:
            box.append(call(*args, **kwargs))
        except Exception as exc:
            # still "no measurement" for every caller that only reads value. Exception rather than
            # BaseException so a KeyboardInterrupt still propagates out of the worker.
            raised.append(exc)
            box.append(None)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(deadline_s)
    # an append is the only thing that ends _run, so a non-empty box IS "it finished".
    return _Outcome(bool(box), box[0] if box else None, raised[0] if raised else None)


def _training_population(spec, dataset):
    """the rows training will consume, in the order it will consume them.

    fence to ``max_examples`` FIRST, then shuffle the fence with the run's seed. order matters as
    much as the fence, because only a prefix of this is ever measured -- a dataset grouped by class
    or length would otherwise have its first rows priced for the whole run, and those rows are
    exactly the ones an ordered dataset makes unrepresentative.

    both algorithms select the SAME rows in the same order. they reach them differently -- grpo
    shuffles the examples and then renders (rl_train.py:1933-1939), opd renders every example in
    dataset order and shuffles the rendered rows after (opd_train.py:2146-2159) -- but one seed
    over one length is one permutation, so the fenced row at each position is identical either way.
    that is the property this function has to hold, and it is the one the measurement depends on.

    what does NOT carry across is the sequence prompt_messages is CALLED in, which is only
    observable for a stateful env. no subset render can reproduce that: both workers render the
    whole fence, so the worker's Nth call has seen N-1 predecessors this function only measures 8
    of. rendering the head in dataset order would match opd's call ORDER while still giving it the
    wrong call COUNT, so it is left alone rather than made to look faithful.

    a copy, not an in-place shuffle: this is the caller's dataset object, and reordering it would
    change what the rest of the submit sees as a side effect of asking for a quote.
    """
    max_examples = int(getattr(spec.train, "max_examples", 0) or 0)
    population = list(dataset[:max_examples] if max_examples > 0 else dataset)
    random.Random(int(getattr(spec, "seed", 0) or 0)).shuffle(population)
    return population


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
            # a raising row declines the whole measurement, exactly as an unrepresentable one does
            # below. dropping it instead shrinks `offered_prompts`, so draws over the survivors
            # report FULL coverage and pass the trust gate while the worker renders and trains on
            # the omitted row -- and a transient i/o error here says nothing about whether the
            # worker's own render will fail.
            return None
        if record_has_images(example if isinstance(example, dict) else {}, messages):
            return None
        usable = _chat_messages(messages)
        if not usable:
            # declined for the same reason a multimodal row is, and it has to be the same reason:
            # dropping the row instead would leave the plain rows as the WHOLE offered set, so 32
            # draws over them report full coverage and pass the trust gate while training still
            # renders the omitted originals. a partial population that looks complete is worse
            # than no measurement, which just returns the quote to the cap.
            return None
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


def _prompt_budget(spec) -> int | None:
    """the prompt length the workers will admit, or None when no budget can be derived.

    both workers compute `prompt_budget = context - max_completion` and DROP every row whose
    rendered prompt exceeds it (rl_train.py and opd_train.py), so a draw from such a row measures
    a prompt the run never trains on. filtered on the endpoint's reported ``prompt_tokens`` rather
    than a local tokenization, because the cli has no tokenizer and pulling one in for an advisory
    quote is the wrong trade.

    the context is the DECLARED one when the run sets it, and otherwise the same recipe default the
    workers fall back to -- grpo `max(1024, rl.max_prompt_len + cap)` (rl_train.py:1974-1976), opd
    `opd.max_prompt_len + cap` (opd_train.py:2194). an unset context is the normal case, so reading
    it as "no budget" left the common run measuring rows the worker is certain to drop.

    the architectural clamp both workers also apply is deliberately not reproduced: it needs an
    AutoConfig fetch this path must not make. that is safe in one direction only -- the clamp can
    only LOWER the budget, so the value here is the looser bound. it filters rows the workers are
    certain to drop and never invents a rejection they would not make.
    """
    context = int(getattr(spec.train, "max_context_tokens", 0) or 0)
    cap = _completion_cap(spec)
    if context <= 0:
        from flash.engine.plan.recipe import RECIPE

        if spec.algorithm == "opd":
            context = RECIPE.opd.max_prompt_len + cap
        else:
            context = max(1024, RECIPE.rl.max_prompt_len + cap)
    budget = context - cap
    return budget if budget > 0 else None


def _sampling_top_p(spec) -> float:
    """the nucleus-sampling setting this run will use.

    read from the same recipe the workers read (`actor_rollout_ref.rollout.top_p` is set from
    `sampling_top_p` in both rl_train.py and opd_train.py) rather than assumed, so this keeps
    matching if the recipe value changes. there is no per-run knob for it today, which is exactly
    why it must be SENT: an endpoint whose own default is not the recipe's would otherwise decode
    differently from every run this measures.
    """
    from flash.engine.plan.recipe import RECIPE

    recipe = RECIPE.opd if spec.algorithm == "opd" else RECIPE.rl
    return float(recipe.sampling_top_p)
