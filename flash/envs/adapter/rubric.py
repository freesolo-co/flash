"""Rubric/reward/introspection helpers for the verifiers adapter.

Leaf helpers split out of ``flash.envs.adapter``: rubric flattening, judge discovery,
reward-func invocation, eval-metric guarding, and env-shape introspection. None reference
the rest of the adapter package (no import cycle); the package ``__init__`` re-exports them.
"""

from __future__ import annotations

import asyncio
import inspect

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
    instead. (Unweighted monitor funcs are skipped entirely by ``scores_breakdown``.)
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


def _substring_answer_score(completion: str, example: dict) -> float:
    """Fallback reward for an env with no rubric: 1.0 iff the example ``answer`` is a
    non-empty substring of the completion, else 0.0."""
    answer = str(example.get("answer") or "")
    return 1.0 if answer and answer in (completion or "") else 0.0


def _unique_key(name: str, existing) -> str:
    """``name``, or the first ``name_1``/``name_2``/… not already a key of ``existing``.

    Probe for an unused exact key so two rubric funcs that share a name both survive — a
    prefix/length heuristic can recompute a suffix that collides with an already-recorded
    key (e.g. ``score`` vs ``score_detail``) and silently overwrite a scorer.
    """
    if name not in existing:
        return name
    i = 1
    while f"{name}_{i}" in existing:
        i += 1
    return f"{name}_{i}"


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
        return single is None or not isinstance(vf_env, single)
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
