"""Adapter that runs Prime Intellect ``verifiers`` / Environments Hub envs on Flash.

Wraps a ``verifiers`` ``Environment`` (``SingleTurnEnv``, ``MultiTurnEnv``, ``ToolEnv`` and
its subclasses) in Flash's small ``Environment`` protocol so Hub environments run unchanged
on Flash's trainer.

GRPO supports all three shapes (the worker routes on ``multi_turn`` / ``is_tool_env``):
  * single-turn — TRL's single-shot generation + per-completion reward;
  * tool (``ToolEnv`` / ``StatefulToolEnv`` / ``SandboxEnv`` / ``PythonEnv``) — TRL drives the
    tool-call loop natively via ``GRPOTrainer(tools=...)`` (:meth:`tools`), masking tool tokens
    itself; the reward scores the full transcript (:meth:`reward_from_messages`);
  * pure multi-turn — ``flash.engine.multiturn_rollout`` supplies a ``rollout_func`` that
    drives this env's turn loop on the colocate engine via the adapter rollout helpers
    (:meth:`new_rollout_state` / :meth:`record_model_turn` / :meth:`env_reply` /
    :meth:`rollout_done`) and returns an ``env_mask`` so only model tokens are trained.

Caveats:
  * SFT on a multi-turn/tool env only fits the single assistant ``sft_target`` per row and
    ignores tool/env turns, so it should be avoided (see ``run_sft`` / ``sft_target``);
  * a ``StatefulToolEnv`` whose tools need verifiers' state-injection (``update_tool_args``)
    is only fully honored on the rollout path — under TRL's native tool loop the tools are
    called as plain functions.

verifiers contract (docs):
  * ``vf.load_environment(env_id, **kwargs) -> Environment``
  * rows have ``prompt`` (chat messages) + ``answer`` (+ optional ``info``)
  * ``env.dataset`` / ``env.get_dataset(n, seed)``, ``env.eval_dataset`` / ``get_eval_dataset``
  * ``env.system_prompt``, ``env.parser``, ``env.rubric`` (weighted reward funcs that take
    ``completion``/``prompt``/``answer``/``info``/``state``/``parser``/``judge`` by name; sync or async)
  * multi-turn: ``env.env_response(messages, state)`` -> env reply messages;
    ``env.is_completed(state)`` -> done flag (both async)

Hub conveniences handled here so the *documented* flow (``slm env install owner/name`` +
``[environment] id = "owner/name"``) works on real Prime Intellect envs:
  * the ``owner/name`` Hub slug is mapped to the bare ``verifiers`` load id;
  * a ``RubricGroup`` (rubrics-of-rubrics) is flattened so the real reward funcs are found;
    eval-metric monitor funcs still run (for shared-state side effects / logging) with their
    exceptions guarded, but contribute 0 — only weighted funcs count toward the reward;
  * a ``JudgeRubric``'s judge client/model/prompt is supplied to reward funcs that declare a
    ``judge``/``judge_client``/``judge_model``/``judge_prompt`` arg, so judge-based rewards run;
  * named per-scorer breakdowns (``scores_breakdown``) expose each reward func's weighted
    score so the frontend per-scorer view + W&B series survive;
  * an optional separate **eval** Hub env (``eval_env_id``) + a fixed eval subset
    (``eval_examples`` / ``eval_seed``) let you train on one env and evaluate on another.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import random

from .base import BaseEnvironment

# The judge-related kwarg names a reward func may declare, sourced from a JudgeRubric.
# Single source of truth for both ``_judge_kwargs`` and ``_AVAILABLE_REWARD_KWARGS``.
_JUDGE_KWARG_NAMES = ("judge", "judge_client", "judge_model", "judge_prompt")

# The kwargs this adapter can supply to a reward func. The non-judge keys are exactly the
# ones built into the ``available`` dict in VerifiersEnvironment._reward_available; the judge
# keys come from ``_judge_kwargs``. Deriving the frozenset from these shared names avoids the
# manual "keep in sync" coupling (adding a kwarg below without updating the set would
# re-trigger the false "requires unavailable arg" failure).
_BASE_REWARD_KWARG_NAMES = (
    "completion",
    "prompt",
    "answer",
    "info",
    "state",
    "parser",
    "task",
)
_AVAILABLE_REWARD_KWARGS = frozenset(_BASE_REWARD_KWARG_NAMES + _JUDGE_KWARG_NAMES)


def _reward_requires_unavailable_args(func) -> str | None:
    """Name of a required arg this adapter cannot supply, or None.

    Group/batch reward funcs declare plural required params (``completions``,
    ``prompts``, ``answers``, ...). The worker scores one completion at a time and has no
    batch, so such a func would be called without its required argument and silently score
    0.0 — train/eval on an all-zero signal. Detect it so the caller can fail fast."""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return None  # builtins/uninspectable: _invoke_reward passes everything
    for p in params:
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if p.default is inspect.Parameter.empty and p.name not in _AVAILABLE_REWARD_KWARGS:
            return p.name
    return None


def vf_load_id(env_ref: str) -> str:
    """Map a Hub slug (``owner/name``) to the bare ``verifiers`` load id (``name``)."""
    return env_ref.split("/", 1)[1] if "/" in env_ref else env_ref


# Flash-reserved keys that may historically have ridden in [environment.params] but are
# NOT verifiers ``load_environment`` kwargs. They are handled by the worker/adapter directly
# (eval_* via named params; GRPO recipe knobs now live in [train]/TrainSpec). A stray one must
# be dropped before forwarding to ``vf.load_environment`` — passing it through would raise a
# TypeError in the env's loader (or silently change its behavior). The eval_* keys are also
# listed here so the catch-all guard never forwards them even if they reach **kwargs.
_RESERVED_ENV_PARAM_KEYS = frozenset(
    {
        "eval_env_id",
        "eval_examples",
        "eval_seed",
        "grpo_config",
        "sft_config",
        "mode",
        "records",
        "eval_records",
        "reward_command",
    }
)


def _drop_reserved_kwargs(kwargs: dict) -> dict:
    """Strip Flash-reserved keys so only true verifiers-env kwargs are forwarded."""
    dropped = [k for k in kwargs if k in _RESERVED_ENV_PARAM_KEYS]
    if dropped:
        print(
            "[verifiers-adapter] dropping Flash-reserved [environment.params] keys not "
            f"accepted by vf.load_environment: {', '.join(sorted(dropped))}"
        )
    return {k: v for k, v in kwargs.items() if k not in _RESERVED_ENV_PARAM_KEYS}


def _run_async(coro):
    """Run an awaitable to completion from sync code, even inside a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a loop (rare for the worker): run in a fresh loop on a thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()


def _call_dataset_getter(obj, method_name: str, *, seed: int, n: int = -1):
    """Call a verifiers dataset getter, binding (n, seed) when it declares them.

    verifiers exposes get_dataset/get_eval_dataset as get_X(n=-1, seed=0); some Hub envs
    declare them WITHOUT defaults, so a no-arg call raised TypeError, swallowed into an empty
    dataset (a paid run over no data). Bind ``n`` (default -1 = all rows — the adapter does its
    own fixed subset selection; callers pass a positive cap to avoid materializing a huge split)
    and the seed when the signature declares them; a genuine failure propagates (fail loudly)
    instead of silently emptying the split."""
    fn = getattr(obj, method_name, None)
    if not callable(fn):
        return None
    try:
        param_names = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        param_names = set()
    kwargs = {}
    if "n" in param_names:
        kwargs["n"] = n
    if "seed" in param_names:
        kwargs["seed"] = seed
    return fn(**kwargs)


def _rows_to_list(ds) -> list[dict]:
    if ds is None:
        return []
    try:
        return [dict(r) for r in ds]
    except Exception:
        return list(ds)


def _flatten_rubric(rubric) -> list[tuple]:
    """Collect ``(func, weight)`` pairs from a rubric, recursing into ``RubricGroup``.

    verifiers composes rubrics (e.g. a ``RubricGroup`` wrapping a ``MathRubric`` plus a
    ``MultiTurnMonitorRubric``); the real reward funcs live on the *nested* rubrics while the
    group's own ``funcs`` is empty. Flattening finds them all.
    """
    funcs = list(getattr(rubric, "funcs", None) or getattr(rubric, "reward_funcs", None) or [])
    weights = list(
        getattr(rubric, "weights", None) or getattr(rubric, "reward_weights", None) or []
    )
    if len(weights) < len(funcs):
        weights += [1.0] * (len(funcs) - len(weights))
    pairs = list(zip(funcs, weights, strict=False))
    for sub in getattr(rubric, "rubrics", None) or []:
        pairs.extend(_flatten_rubric(sub))
    return pairs


def _find_judge_rubric(rubric):
    """Return the first ``JudgeRubric`` in a rubric tree (or None), for judge-arg injection."""
    if rubric is None:
        return None
    try:
        import verifiers as vf

        judge_cls = getattr(vf, "JudgeRubric", None)
    except ImportError:
        judge_cls = None
    if judge_cls is not None and isinstance(rubric, judge_cls):
        return rubric
    # Duck-type fallback: anything exposing a `judge` method + a judge_client attr.
    if callable(getattr(rubric, "judge", None)) and hasattr(rubric, "judge_client"):
        return rubric
    for sub in getattr(rubric, "rubrics", None) or []:
        found = _find_judge_rubric(sub)
        if found is not None:
            return found
    return None


def _judge_kwargs(judge_rubric) -> dict:
    """The judge-related kwargs a reward func may declare, sourced from a JudgeRubric."""
    if judge_rubric is None:
        return {}
    return {name: getattr(judge_rubric, name, None) for name in _JUDGE_KWARG_NAMES}


def _invoke_reward(func, available: dict) -> float:
    """Call a verifiers reward func passing only the kwargs it declares; await if async.

    Exceptions PROPAGATE. ``scores_breakdown`` invokes this for *weighted* reward funcs, so an
    exception here is a real (weighted) reward func genuinely failing (e.g. a JudgeRubric judge
    raising on an API/rate-limit error, or a parse error on row data). Swallowing it as 0.0
    would silently train/score on an all-zero signal and waste a paid run, so we fail loudly
    instead. Eval-metric (optional/monitor) funcs are run through ``_run_eval_metric``,
    which swallows their exceptions — they contribute 0 either way and may exist only for their
    side effects (mutating shared ``state`` / logging), so a thrown monitor must not fail a run.
    """
    try:
        params = inspect.signature(func).parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kwargs = dict(available)
        else:
            kwargs = {k: v for k, v in available.items() if k in params}
    except (TypeError, ValueError):
        kwargs = dict(available)
    result = func(**kwargs)
    if inspect.isawaitable(result):
        result = _run_async(result)
    return float(result or 0.0)


def _run_eval_metric(func, available: dict) -> None:
    """Run a eval-metric monitor/diagnostic reward func, swallowing any exception.

    Per verifiers semantics every reward func RUNS, even weight-0 ones: they may mutate the
    shared ``state`` (so a later weighted func sees their work) or simply be logged. They never
    contribute to the reward (weight is 0), so their result is discarded and a failure must NOT
    fail the run — guard the exception. Weighted funcs go through ``_invoke_reward`` instead,
    where exceptions propagate.
    """
    with contextlib.suppress(Exception):
        _invoke_reward(func, available)


def _substring_answer_score(completion: str, example: dict) -> float:
    """Fallback reward for an env with no rubric: 1.0 iff the example ``answer`` is a
    non-empty substring of the completion, else 0.0."""
    answer = str(example.get("answer") or "")
    return 1.0 if answer and answer in (completion or "") else 0.0


def _is_multi_turn(vf_env) -> bool:
    """True for a tool/multi-turn verifiers env (NOT a plain SingleTurnEnv)."""
    try:
        import verifiers as vf
    except ImportError:
        return False
    tool = getattr(vf, "ToolEnv", None)
    multi = getattr(vf, "MultiTurnEnv", None)
    single = getattr(vf, "SingleTurnEnv", None)
    if tool is not None and isinstance(vf_env, tool):
        return True
    if multi is not None and isinstance(vf_env, multi):
        # SingleTurnEnv subclasses MultiTurnEnv in verifiers; exempt it.
        return not (single is not None and isinstance(vf_env, single))
    return False


def _is_tool_env(vf_env) -> bool:
    """True for a verifiers ``ToolEnv`` or any subclass (Stateful/Sandbox/Python).

    Tool envs expose Python tool callables; the worker hands those to TRL's
    ``GRPOTrainer(tools=...)`` so TRL drives the tool-call loop natively (it owns generation,
    tool execution, and assistant-only token masking). A *pure* ``MultiTurnEnv`` (env turns are
    arbitrary content, e.g. a simulated user) is multi-turn but NOT a tool env, and takes the
    ``rollout_func`` path instead."""
    try:
        import verifiers as vf
    except ImportError:
        return False
    tool = getattr(vf, "ToolEnv", None)
    return tool is not None and isinstance(vf_env, tool)


class VerifiersEnvironment(BaseEnvironment):
    """Flash environment backed by a verifiers ``Environment`` instance.

    GRPO training supports three env shapes (the worker routes on these flags):
      * **single-turn** (``multi_turn`` False) — TRL's single-shot rollout (original path);
      * **tool** (``is_tool_env`` True) — TRL drives the tool-call loop natively via
        ``GRPOTrainer(tools=...)`` (:meth:`tools`); the reward scores the full transcript
        (:meth:`reward_from_messages`);
      * **pure multi-turn** (``multi_turn`` True, ``is_tool_env`` False) — TRL's
        ``rollout_func`` drives this env's turn loop (:meth:`new_rollout_state` /
        :meth:`record_model_turn` / :meth:`env_reply` / :meth:`rollout_done`).
    """

    def __init__(
        self,
        vf_env,
        env_id: str,
        eval_vf_env=None,
        eval_examples: int | None = None,
        eval_seed: int = 12345,
    ):
        super().__init__(id=env_id)
        self._env = vf_env
        self._eval_env = eval_vf_env  # optional separate eval Hub env
        self._eval_examples = int(eval_examples) if eval_examples else 0
        self._eval_seed = int(eval_seed)
        self.multi_turn = _is_multi_turn(vf_env)
        self.is_tool_env = _is_tool_env(vf_env)
        # Turn cap for the tool / multi-turn rollout loop (verifiers ToolEnv defaults to 10).
        self.max_turns = int(getattr(vf_env, "max_turns", 10) or 10)
        # The shared scorer is the TRAIN env's (flattened) rubric + parser, so the reward used
        # for RL and the grader used at eval are byte-for-byte identical.
        rubric = getattr(vf_env, "rubric", None)
        self._reward_pairs = _flatten_rubric(rubric) if rubric is not None else []
        self._judge_rubric = _find_judge_rubric(rubric)
        # Fail fast on a group/batch reward func: the worker scores one completion at a time
        # and cannot supply its plural batch args, so it would silently score 0.0 and train a
        # paid run on an all-zero signal. Only weighted funcs matter (eval-metric ones skip).
        for func, weight in self._reward_pairs:
            if not weight:
                continue
            missing = _reward_requires_unavailable_args(func)
            if missing:
                raise ValueError(
                    f"verifiers reward function {getattr(func, '__name__', func)!r} requires "
                    f"argument {missing!r}, which the Flash adapter cannot supply (it scores "
                    "one completion at a time, with no group/batch context such as "
                    "completions/prompts/answers). This environment uses a group-based reward "
                    "not supported on Flash; use a per-completion reward."
                )
        self._parser = getattr(vf_env, "parser", None)

    # -- data -------------------------------------------------------------
    def dataset(self, split: str, limit: int | None = None) -> list[dict]:
        is_eval = split in {"eval", "validation", "test"}
        if is_eval:
            src = self._eval_env or self._env
            # ``limit`` caps materialization at the source (mid-run eval only needs a small slice);
            # a getter that ignores ``n`` still returns everything, so _fixed_subset / the caller's
            # slice remain the backstop. -1 = all rows (the no-cap default).
            n = limit if (limit is not None and limit > 0) else -1
            # Resolve the eval source with explicit ``is None`` checks (NOT ``or``): an
            # empty-but-configured eval split (``[]``) is falsy, so ``or`` would wrongly
            # fall through to the next source and ultimately to the TRAIN split — evaluating
            # on training data. Only fall back when the eval source is genuinely *absent*
            # (None), not merely empty. ``get_eval_dataset``/``eval_dataset`` returning [] is
            # a deliberate empty eval set and must be honored as such.
            eval_ds = _call_dataset_getter(src, "get_eval_dataset", seed=self._eval_seed, n=n)
            if eval_ds is None:
                eval_ds = getattr(src, "eval_dataset", None)
            if eval_ds is None:  # no eval split configured at all: use the env's train split
                eval_ds = _call_dataset_getter(src, "get_dataset", seed=self._eval_seed, n=n)
                if eval_ds is None:
                    eval_ds = getattr(src, "dataset", None)
            rows = _rows_to_list(eval_ds)
            # An explicit positive ``limit`` means the caller (mid-run eval) wants a RAW pool of up
            # to ``limit`` rows and does its OWN seeded sampling on top — so don't also apply the
            # ``[environment.params] eval_examples`` subset here, which would silently shrink the
            # pool below ``limit`` and starve the caller's sample (it already capped the pool size
            # via ``limit``). ``_fixed_subset`` still governs the plain ``dataset("eval")`` path
            # (the "eval on a different env" feature, where the param IS the intended sample size).
            if limit is not None and limit > 0:
                return rows
            return self._fixed_subset(rows)
        ds = _call_dataset_getter(self._env, "get_dataset", seed=0)
        if ds is None:
            ds = getattr(self._env, "dataset", None)
        return _rows_to_list(ds)

    def has_eval_split(self) -> bool:
        """True when a DISTINCT held-out eval split exists (a separate eval env, or the env's
        ``get_eval_dataset``/``eval_dataset``). False means :meth:`dataset` would fall back to
        train rows — so a caller (mid-run eval) can warn instead of reporting train data as
        held-out. Best-effort: a getter that raises is treated as no eval split."""
        if self._eval_env is not None:
            return True
        try:
            if (
                _call_dataset_getter(self._env, "get_eval_dataset", seed=self._eval_seed)
                is not None
            ):
                return True
        except Exception:
            return False
        return getattr(self._env, "eval_dataset", None) is not None

    def _fixed_subset(self, rows: list[dict]) -> list[dict]:
        n = self._eval_examples
        if n <= 0 or n >= len(rows):
            return rows
        idx = sorted(random.Random(self._eval_seed).sample(range(len(rows)), n))
        return [rows[i] for i in idx]

    # -- task interface ---------------------------------------------------
    def prompt_messages(self, example: dict) -> list[dict]:
        prompt = example.get("prompt")
        if isinstance(prompt, list) and prompt:
            msgs = [dict(m) for m in prompt]
        else:
            question = example.get("question") or example.get("prompt") or ""
            msgs = [{"role": "user", "content": str(question)}]
        system_prompt = getattr(self._env, "system_prompt", None)
        if system_prompt and not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": system_prompt}, *msgs]
        return msgs

    def sft_target(self, example: dict) -> str:
        for key in ("answer", "completion", "target", "response"):
            value = example.get(key)
            if value:
                if isinstance(value, list):  # chat messages
                    return str(value[-1].get("content", ""))
                return str(value)
        return ""

    # -- reward / scoring -------------------------------------------------
    def _normalize_info(self, example: dict) -> dict:
        # Hub rows may store `info` as a JSON string (a supported Verifiers row shape);
        # parse it so reward funcs that do `info[...]` get a dict, not a str (which would
        # raise TypeError, be swallowed as 0.0, and poison the signal).
        info = example.get("info") or {}
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except (ValueError, TypeError):
                info = {}
        return info

    def _reward_available(self, completion: str, example: dict, state: dict | None) -> dict:
        # In multi-turn/tool mode the accumulated transcript lives on ``state`` (built by the
        # rollout helpers): ``state["completion"]`` is the full assistant + tool/env message
        # list and ``state["prompt"]`` is the initial prompt. Reward/tool funcs that inspect the
        # whole message list need that transcript, not the scalar ``completion`` string wrapped
        # as a lone synthesized assistant message. Single-turn falls back to wrapping the scalar.
        completion_msgs: list[dict] | None = None
        prompt_msgs = None
        if self.multi_turn and state:
            transcript = state.get("completion")
            if isinstance(transcript, list) and transcript:
                completion_msgs = [dict(m) for m in transcript]
            state_prompt = state.get("prompt")
            if isinstance(state_prompt, list) and state_prompt:
                prompt_msgs = [dict(m) for m in state_prompt]
        if completion_msgs is None:
            completion_msgs = [{"role": "assistant", "content": completion}]
        if prompt_msgs is None:
            prompt_msgs = example.get("prompt") or self.prompt_messages(example)
        available = {
            "completion": completion_msgs,
            "prompt": prompt_msgs,
            "answer": example.get("answer"),
            "info": self._normalize_info(example),
            "state": state if state is not None else {},
            "parser": self._parser,
            "task": example,
        }
        available.update(_judge_kwargs(self._judge_rubric))
        return available

    def scores_breakdown(
        self, completion: str, example: dict, state: dict | None = None
    ) -> dict[str, float]:
        """Per-scorer weighted scores: ``{func_name: weighted_score, ..., "total": sum}``.

        Every WEIGHTED rubric func contributes one entry (by ``func.__name__``); the
        ``"total"`` is their sum (== :meth:`reward`). Used to preserve the frontend per-scorer
        breakdown + W&B series instead of collapsing to a single binary ``correct``.

        Per verifiers semantics EVERY reward func runs, including eval-metric ones — they may
        mutate the shared ``state`` (so a subsequent weighted func sees their work) or exist
        only to be logged. Eval-metric funcs run with GUARDED exceptions (a thrown monitor must
        not fail the run) and contribute 0, so they are not added to the breakdown/total; the
        order is preserved so a eval-metric func can prepare state for a later weighted one.
        Weighted funcs propagate exceptions (a thrown weighted reward fails the run).
        """
        breakdown: dict[str, float] = {}
        if not self._reward_pairs:
            score = _substring_answer_score(completion, example)
            return {"answer_match": score, "total": score}
        available = self._reward_available(completion, example, state)
        total = 0.0
        for func, weight in self._reward_pairs:
            if not weight:
                # Eval-metric monitor/diagnostic func: RUN it (for its side effects on shared
                # state / logging) with guarded exceptions, but it contributes 0 and is not in
                # the named breakdown.
                _run_eval_metric(func, available)
                continue
            name = getattr(func, "__name__", str(func))
            score = float(weight) * _invoke_reward(func, available)
            # Collisions (two funcs share a name): keep them distinct so neither is lost.
            # Probe for an unused exact key — a prefix/length heuristic can recompute a
            # suffix that collides with an already-recorded key (e.g. ``score`` vs
            # ``score_detail``) and silently overwrite a scorer.
            if name in breakdown:
                base = name
                i = 1
                while name in breakdown:
                    name = f"{base}_{i}"
                    i += 1
            breakdown[name] = score
            total += score
        breakdown["total"] = total
        return breakdown

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        return float(self.scores_breakdown(completion, example, state)["total"])

    def evaluate(self, completion: str, example: dict, state: dict | None = None) -> dict:
        """One pass over the rubric returning the training ``reward`` AND the env's separate
        ``metrics`` — each EVAL-METRIC rubric func's raw (unweighted) score, by name.

        Eval-metric funcs are how a verifiers ``environment.py`` expresses an EVALUATION signal
        distinct from the GRPO reward: an ``exact_match`` / ``accuracy`` monitor added with
        ``rubric.add_metric(fn, weight=0.0)`` does not shape training (weight 0) but measures
        quality. :meth:`scores_breakdown`/:meth:`reward` deliberately omit these; mid-run eval
        wants them, so this method surfaces them WITHOUT a second rubric pass (a ``JudgeRubric``
        is sampled at most once). Weighted funcs still propagate exceptions (a broken weighted
        reward fails the run); eval-metric monitors stay guarded (a thrown monitor is skipped,
        never fails eval). Order is preserved so a monitor can prep shared ``state`` for a later
        weighted func, exactly as in :meth:`scores_breakdown`.

        Returns ``{"reward": <weighted total>, "metrics": {name: raw_score, ...}}``; an env with
        no rubric falls back to the substring ``answer_match`` as both reward and (empty) metrics.
        """
        if not self._reward_pairs:
            score = _substring_answer_score(completion, example)
            return {"reward": score, "metrics": {}}
        available = self._reward_available(completion, example, state)
        total = 0.0
        metrics: dict[str, float] = {}
        for func, weight in self._reward_pairs:
            name = getattr(func, "__name__", str(func))
            if not weight:
                # Eval-metric = an eval/monitor metric: run it (guarded — a thrown monitor must
                # not fail eval) and record its RAW score; it never touches the reward total.
                try:
                    raw = _invoke_reward(func, available)
                except Exception:
                    continue
                key = name
                i = 1
                while key in metrics:  # keep colliding metric names distinct
                    key = f"{name}_{i}"
                    i += 1
                metrics[key] = raw
                continue
            total += float(weight) * _invoke_reward(func, available)
        return {"reward": total, "metrics": metrics}

    def tools(self) -> list:
        """The underlying ToolEnv's Python tool callables (``[]`` for non-tool envs).

        Handed to ``GRPOTrainer(tools=...)`` so TRL runs the tool-call loop and does the
        assistant-only token masking itself. Each is a plain function with type hints + a
        Google-style docstring (verifiers and TRL share that requirement)."""
        return list(getattr(self._env, "tools", None) or [])

    def reward_from_messages(
        self, completion_msgs: list[dict], example: dict, prompt_msgs: list[dict] | None = None
    ) -> float:
        """Reward for a full transcript (assistant + tool/env messages) via the rubric.

        The tool / multi-turn training path produces a *message list* rollout rather than a
        single completion string; this routes it through the same weighted-rubric scoring as
        :meth:`reward` by handing the transcript to the env's reward funcs as ``state``."""
        state: dict = {"completion": [dict(m) for m in completion_msgs]}
        if prompt_msgs:
            state["prompt"] = [dict(m) for m in prompt_msgs]
        return self.reward("", example, state)

    def grade(self, completion: str, example: dict, state: dict | None = None) -> bool:
        threshold = getattr(self._env, "pass_threshold", 0.5)
        return self.reward(completion, example, state) >= threshold

    # -- multi-turn rollout (driven by the worker) ------------------------
    def new_rollout_state(self, example: dict) -> dict:
        """A fresh per-rollout ``state`` dict, threaded through env_reply/reward.

        Mirrors the verifiers rollout ``state``: holds the running ``prompt``, the
        accumulated ``completion`` (assistant + tool/env turns), the ``answer``/``info``, and
        a ``turn`` counter. Reward funcs that read ``state`` see this dict.
        """
        prompt = self.prompt_messages(example)
        state = {
            "prompt": [dict(m) for m in prompt],
            "completion": [],
            "answer": example.get("answer"),
            "info": self._normalize_info(example),
            "responses": [],
            "turn": 0,
        }
        setup = getattr(self._env, "setup_state", None)
        if callable(setup):
            with contextlib.suppress(Exception):
                state = _run_async(setup(state)) or state
        return state

    def env_reply(self, messages: list[dict], state: dict) -> list[dict]:
        """One environment turn: given the conversation so far (incl. the latest model
        message), return the env's reply messages (tool results / next user turn) and advance
        ``state``. Empty list when the env has nothing to add. Single-turn envs return []."""
        if not self.multi_turn:
            return []
        fn = getattr(self._env, "env_response", None)
        if not callable(fn):
            return []
        try:
            reply = _run_async(fn(messages, state))
        except NotImplementedError:
            # Legitimate "this env has no env turn" signal -> no env reply.
            return []
        except Exception as exc:
            # Mirror `_invoke_reward`: a genuine bug in the env's `env_response` must
            # NOT be swallowed. Silently returning [] would collapse every multi-turn
            # rollout to a single turn and train a paid GRPO run on degenerate
            # transcripts. The rollout loop (multiturn_rollout.py) calls this directly
            # with no surrounding swallow, so re-raising propagates and fails the run
            # fast (and the context is printed first so it never vanishes silently).
            print(f"[env_reply] env_response failed (turn={state.get('turn', 0)}): {exc!r}")
            raise
        if reply is None:
            return []
        if isinstance(reply, dict):
            reply = [reply]
        out = [dict(m) for m in reply]
        state["completion"].extend(out)
        state["turn"] = int(state.get("turn", 0)) + 1
        return out

    def rollout_done(self, state: dict, max_turns: int | None = None) -> bool:
        """Whether the multi-turn rollout should stop (env says completed, or turn cap hit)."""
        if not self.multi_turn:
            return True
        if max_turns is not None and int(state.get("turn", 0)) >= int(max_turns):
            return True
        fn = getattr(self._env, "is_completed", None)
        if not callable(fn):
            return True
        try:
            return bool(_run_async(fn(state)))
        except NotImplementedError:
            # Env doesn't implement a completion check -> rely on the turn cap only.
            return True
        except Exception as exc:
            # Mirror `_invoke_reward` / `env_reply`: a real bug in `is_completed` must
            # not be silently treated as "done" (which would truncate every rollout and
            # train on degenerate transcripts). Print context, then re-raise so the run
            # fails fast (the rollout loop calls this directly with no surrounding swallow).
            print(f"[rollout_done] is_completed failed (turn={state.get('turn', 0)}): {exc!r}")
            raise

    def record_model_turn(self, state: dict, content: str) -> dict:
        """Append a model (assistant) turn to ``state`` before calling ``env_reply``."""
        msg = {"role": "assistant", "content": content}
        state["completion"].append(msg)
        state.setdefault("responses", []).append(content)
        return msg


def _import_vf():
    try:
        import verifiers as vf

        return vf
    except ImportError as exc:
        raise ImportError(
            "the 'verifiers' package is required to run Prime Hub environments; "
            "install it (e.g. `uv pip install verifiers`) or run `slm env install <env>`"
        ) from exc


def load_verifiers_environment(
    env_id: str,
    eval_env_id: str | None = None,
    eval_examples: int | None = None,
    eval_seed: int = 12345,
    **kwargs,
) -> VerifiersEnvironment:
    """Load an installed / Hub verifiers environment by id and wrap it for Flash.

    ``env_id`` may be a Hub slug (``owner/name``); it is mapped to the bare verifiers load id.
    Pass ``eval_env_id`` to evaluate on a *different* Hub env, with ``eval_examples`` /
    ``eval_seed`` selecting a fixed eval subset. Remaining ``kwargs`` are forwarded to the train
    env's ``vf.load_environment``.
    """
    vf = _import_vf()
    vf_env = vf.load_environment(vf_load_id(env_id), **_drop_reserved_kwargs(kwargs))
    eval_vf_env = vf.load_environment(vf_load_id(eval_env_id)) if eval_env_id else None
    return VerifiersEnvironment(
        vf_env,
        env_id,
        eval_vf_env=eval_vf_env,
        eval_examples=eval_examples,
        eval_seed=eval_seed,
    )
