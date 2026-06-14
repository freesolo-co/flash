"""Adapter that runs Prime Intellect ``verifiers`` / Environments Hub envs on AutoSLM.

Wraps a ``verifiers`` ``Environment`` (``SingleTurnEnv`` and friends) in AutoSLM's small
``Environment`` protocol so Hub environments run unchanged on AutoSLM's trainer. Single-turn
is fully supported; multi-turn/tool rollouts (``ToolEnv``/``MultiTurnEnv``) are a
follow-up — they require the rollout loop in ``autoslm.engine.worker`` (tracked for M4+).

verifiers contract (docs):
  * ``vf.load_environment(env_id, **kwargs) -> Environment``
  * rows have ``prompt`` (chat messages) + ``answer`` (+ optional ``info``)
  * ``env.dataset`` / ``env.get_dataset(n, seed)``
  * ``env.system_prompt``, ``env.parser``, ``env.rubric`` (weighted reward funcs that take
    ``completion``/``prompt``/``answer``/``info``/``state``/``parser`` by name; sync or async)

Hub conveniences handled here so the *documented* flow (``slm env install owner/name`` +
``[environment] id = "owner/name"``) actually works on real Prime Intellect envs:
  * the ``owner/name`` Hub slug is mapped to the bare ``verifiers`` load id;
  * a ``RubricGroup`` (rubrics-of-rubrics, e.g. ``MathRubric`` + a monitor rubric) is
    flattened so the real reward funcs are found and zero-weight/monitor funcs are skipped.
"""

from __future__ import annotations

import asyncio
import inspect
import json

from .base import BaseEnvironment

# The singular kwargs this single-turn adapter can supply to a reward func (kept in
# sync with the ``available`` dict built in VerifiersEnvironment.reward).
_AVAILABLE_REWARD_KWARGS = frozenset(
    {"completion", "prompt", "answer", "info", "state", "parser", "task"}
)


def _reward_requires_unavailable_args(func) -> str | None:
    """Name of a required arg this adapter cannot supply, or None.

    Group/batch reward funcs declare plural required params (``completions``,
    ``prompts``, ``answers``, ...). The single-turn worker scores one completion at
    a time and has no batch, so such a func would be called without its required
    argument and silently score 0.0 — train/eval on an all-zero signal. Detect it so
    the caller can fail fast instead."""
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


def _call_dataset_getter(obj, method_name, *, seed: int):
    """Call a verifiers dataset getter, binding (n, seed) when it declares them.

    verifiers exposes get_dataset as get_dataset(n=-1, seed=0); some Hub envs declare it
    WITHOUT defaults, so a no-arg call raised TypeError, which got swallowed into an empty
    dataset (a paid run over no data). Bind n=-1 (all rows — the adapter does its own
    fixed-subset selection) and the seed when the signature declares them; a genuine
    failure propagates (fail loudly) instead of silently emptying the split."""
    fn = getattr(obj, method_name, None)
    if not callable(fn):
        return None
    try:
        param_names = set(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        param_names = set()
    kwargs = {}
    if "n" in param_names:
        kwargs["n"] = -1
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
    ``MultiTurnMonitorRubric``); the real reward funcs live on the *nested* rubrics while
    the group's own ``funcs`` is empty. Flattening finds them all.
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


def _invoke_reward(func, available: dict) -> float:
    """Call a verifiers reward func passing only the kwargs it declares; await if async.

    Guarded: a monitor/diagnostic reward func may assume rollout ``state`` we don't have
    (e.g. ``num_turns`` reads ``state['trajectory']``); such funcs are zero-weighted and
    must not crash the reward, so any error scores 0.0.
    """
    try:
        params = inspect.signature(func).parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kwargs = dict(available)
        else:
            kwargs = {k: v for k, v in available.items() if k in params}
    except (TypeError, ValueError):
        kwargs = dict(available)
    try:
        result = func(**kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return float(result or 0.0)
    except Exception:
        return 0.0


class VerifiersEnvironment(BaseEnvironment):
    """AutoSLM environment backed by a verifiers ``Environment`` instance (single-turn)."""

    def __init__(self, vf_env, env_id: str):
        super().__init__(id=env_id)
        self._env = vf_env
        # The scorer is the env's (flattened) rubric + parser.
        self._reward_pairs = (
            _flatten_rubric(getattr(vf_env, "rubric", None))
            if getattr(vf_env, "rubric", None) is not None
            else []
        )
        # Fail fast on a group/batch reward func: the single-turn worker cannot supply
        # its plural batch args, so it would silently score 0.0 and train a paid run on
        # an all-zero signal. Only weighted funcs matter (zero-weight ones are skipped).
        for func, weight in self._reward_pairs:
            if not weight:
                continue
            missing = _reward_requires_unavailable_args(func)
            if missing:
                raise ValueError(
                    f"verifiers reward function {getattr(func, '__name__', func)!r} requires "
                    f"argument {missing!r}, which the single-turn AutoSLM adapter cannot supply "
                    "(it scores one completion at a time, with no group/batch context such as "
                    "completions/prompts/answers). This environment uses a group-based reward "
                    "not supported on AutoSLM; use a per-completion reward."
                )
        self._parser = getattr(vf_env, "parser", None)

    # -- data -------------------------------------------------------------
    def dataset(self, split: str) -> list[dict]:
        ds = _call_dataset_getter(self._env, "get_dataset", seed=0) or getattr(
            self._env, "dataset", None
        )
        return _rows_to_list(ds)

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

    def reward(self, completion: str, example: dict) -> float:
        if not self._reward_pairs:
            answer = str(example.get("answer") or "")
            return 1.0 if answer and answer in (completion or "") else 0.0
        completion_msgs = [{"role": "assistant", "content": completion}]
        # Hub rows may store `info` as a JSON string (a supported Verifiers row shape);
        # parse it so reward funcs that do `info[...]` get a dict, not a str (which
        # would raise TypeError, be swallowed as 0.0, and poison the signal).
        info = example.get("info") or {}
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except (ValueError, TypeError):
                info = {}
        available = {
            "completion": completion_msgs,
            "prompt": example.get("prompt") or self.prompt_messages(example),
            "answer": example.get("answer"),
            "info": info,
            "state": {},
            "parser": self._parser,
            "task": example,
        }
        total = 0.0
        for func, weight in self._reward_pairs:
            if not weight:  # skip zero-weight monitor/diagnostic funcs
                continue
            total += float(weight) * _invoke_reward(func, available)
        return float(total)

    def grade(self, completion: str, example: dict) -> bool:
        threshold = getattr(self._env, "pass_threshold", 0.5)
        return self.reward(completion, example) >= threshold

    def describe(self) -> dict:
        return {
            "environment": self.id,
            "reward_funcs": [getattr(f, "__name__", str(f)) for f, w in self._reward_pairs if w],
        }


def load_verifiers_environment(env_id: str, **kwargs) -> VerifiersEnvironment:
    """Load a verifiers/Hub environment by id and wrap it for AutoSLM.

    ``env_id`` may be a Hub slug (``owner/name``); it is mapped to the bare verifiers load
    id. Remaining ``kwargs`` are forwarded to ``vf.load_environment``.
    """
    try:
        import verifiers as vf
    except ImportError as exc:
        raise ImportError(
            "the 'verifiers' package is required to run Prime Hub environments; "
            "install it (e.g. `uv pip install verifiers`) or run `slm env install <env>`"
        ) from exc
    vf_env = vf.load_environment(vf_load_id(env_id), **kwargs)
    # The worker only runs single-turn rollouts. A ToolEnv / multi-turn env would be
    # called with no rollout state, its reward fns would error, and those errors are
    # swallowed as 0.0 — i.e. a paid run trains on an all-zero signal. Reject it here.
    # SingleTurnEnv subclasses MultiTurnEnv in verifiers, so exempt it explicitly.
    _ToolEnv = getattr(vf, "ToolEnv", None)
    _MultiTurnEnv = getattr(vf, "MultiTurnEnv", None)
    _SingleTurnEnv = getattr(vf, "SingleTurnEnv", None)
    if (_ToolEnv is not None and isinstance(vf_env, _ToolEnv)) or (
        _MultiTurnEnv is not None
        and isinstance(vf_env, _MultiTurnEnv)
        and not (_SingleTurnEnv is not None and isinstance(vf_env, _SingleTurnEnv))
    ):
        raise ValueError(
            f"verifiers environment {env_id!r} is a tool/multi-turn environment; the "
            "AutoSLM worker only runs single-turn rollouts (multi-turn reward functions "
            "would be scored 0.0). Use a single-turn environment."
        )
    return VerifiersEnvironment(vf_env, env_id)
