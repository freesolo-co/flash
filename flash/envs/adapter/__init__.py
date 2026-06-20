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
    unweighted (monitor) funcs are skipped — only weighted funcs count toward the reward;
  * a ``JudgeRubric``'s judge client/model/prompt is supplied to reward funcs that declare a
    ``judge``/``judge_client``/``judge_model``/``judge_prompt`` arg, so judge-based rewards run;
  * named per-scorer breakdowns (``scores_breakdown``) expose each reward func's weighted
    score so the frontend per-scorer view + W&B series survive.
"""

from __future__ import annotations

import contextlib
import json

from flash.envs.adapter.rubric import (  # noqa: F401
    _AVAILABLE_REWARD_KWARGS,
    _BASE_REWARD_KWARG_NAMES,
    _JUDGE_KWARG_NAMES,
    _call_dataset_getter,
    _find_judge_rubric,
    _flatten_rubric,
    _invoke_reward,
    _is_multi_turn,
    _is_tool_env,
    _judge_kwargs,
    _reward_requires_unavailable_args,
    _rows_to_list,
    _run_async,
    _substring_answer_score,
    _unique_key,
)
from flash.envs.base import BaseEnvironment


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

    def __init__(self, vf_env, env_id: str):
        super().__init__(id=env_id)
        self._env = vf_env
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
    def dataset(self, split: str = "train") -> list[dict]:
        ds = _call_dataset_getter(self._env, "get_dataset", seed=0)
        if ds is None:
            ds = getattr(self._env, "dataset", None)
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
        breakdown + W&B series instead of collapsing to a single binary ``correct``. Unweighted
        (monitor) funcs are skipped — only weighted funcs shape the reward. Weighted funcs
        propagate exceptions (a thrown weighted reward fails the run).
        """
        breakdown: dict[str, float] = {}
        if not self._reward_pairs:
            score = _substring_answer_score(completion, example)
            return {"answer_match": score, "total": score}
        available = self._reward_available(completion, example, state)
        total = 0.0
        for func, weight in self._reward_pairs:
            if not weight:
                continue  # unweighted monitor func: contributes nothing to the reward
            name = getattr(func, "__name__", str(func))
            score = float(weight) * _invoke_reward(func, available)
            breakdown[_unique_key(name, breakdown)] = score
            total += score
        breakdown["total"] = total
        return breakdown

    def reward(self, completion: str, example: dict, state: dict | None = None) -> float:
        return float(self.scores_breakdown(completion, example, state)["total"])

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


def load_verifiers_environment(env_id: str, **kwargs) -> VerifiersEnvironment:
    """Load an installed / Hub verifiers environment by id and wrap it for Flash.

    ``env_id`` may be a Hub slug (``owner/name``); it is mapped to the bare verifiers load id.
    Remaining ``kwargs`` are forwarded to the env's ``vf.load_environment``.
    """
    vf = _import_vf()
    vf_env = vf.load_environment(vf_load_id(env_id), **_drop_reserved_kwargs(kwargs))
    return VerifiersEnvironment(vf_env, env_id)
