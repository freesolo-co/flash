"""Periodic mid-run evaluation for GRPO (greedy eval on a held-out split, between steps).

GRPO's only live signal is the per-step *training* reward — noisy, on-policy, temperature>0,
and (by design) often a SHAPED reward rather than the metric you actually care about. This
module adds a deterministic eval signal: every ``N`` optimizer steps it samples the current
policy GREEDILY on a fixed held-out split and scores it, then streams the result through the
worker ``heartbeat`` so the orchestrating agent sees an eval curve mid-run (early-stop, plateau
detection, checkpoint selection) instead of only the reward stream.

Evaluation distinct from reward lives in the environment. A verifiers
``environment.py`` expresses an evaluation metric SEPARATE from the GRPO reward as a
**eval-metric rubric func** (``rubric.add_metric(fn, weight=0.0)``): it does not shape training
but is computed and reported. The adapter's :meth:`VerifiersEnvironment.evaluate` returns both
the weighted training ``reward`` and those eval-metric ``metrics`` in one rubric pass, and this
module surfaces both — so the agent watches the env's true eval metric, not just the reward.

It generates with the **trainer's own model** (``trainer.model.generate`` in eval/no-grad mode),
NOT the colocate vLLM engine. Calling ``engine.generate`` out-of-band from a ``TrainerCallback``
deadlocks GRPO (the engine is only safe to drive inside TRL's managed ``rollout_func`` window;
an external call from ``on_step_end`` hangs the training thread — verified on a live GPU run).
Generating on the already-loaded trainer model avoids the engine entirely, works on BOTH the
vLLM-colocate and transformers-generation backends, and keeps memory bounded: ONE prompt at a
time (a single sequence's KV cache), and an OOM there is a *catchable* error -> graceful
``eval_skipped``, never a training-freezing hang.

Design mirrors :mod:`flash.engine.multiturn_rollout`: the core (:class:`PeriodicEval`, the
scorers, :func:`evaluate_policy`, :func:`summarize`) is pure Python with ``render`` / ``generate``
/ engine access INJECTED, so cadence gating, engine binding, skip handling and heartbeat
emission are unit-tested without a GPU, tokenizer, or vLLM. Only :func:`build_hf_greedy_generate`
(wired in ``worker.run_rl``) touches torch / the live model.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from flash.engine.multiturn_rollout import rollout_one

# A failed single rollout (template violation, env exception) scores 0 rather than crashing the
# whole eval — and a crashing eval must never abort a paid training run, so the callback also
# swallows eval-wide failures and heartbeats a skip reason.
_EVAL_ERROR_REWARD = 0.0


@dataclass(frozen=True)
class EvalRecord:
    """One example's eval result: the training ``reward`` and the env's eval-metric ``metrics``."""

    reward: float
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSummary:
    """Aggregate of one mid-run eval pass over the held-out set."""

    n: int
    mean_reward: float
    pass_rate: float
    min_reward: float
    max_reward: float
    metric_means: dict[str, float] = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)
    step: int | None = None

    def as_heartbeat_fields(self) -> dict:
        """The flat fields streamed under the ``rl_eval`` heartbeat stage. Each env eval metric
        is reported as ``eval_metric_<name>`` so it sits alongside the reward, distinct from it."""
        fields: dict[str, float | int] = {
            "eval_n": self.n,
            "eval_reward": self.mean_reward,
            "eval_pass_rate": self.pass_rate,
            "eval_reward_min": self.min_reward,
            "eval_reward_max": self.max_reward,
        }
        for name, value in self.metric_means.items():
            fields[f"eval_metric_{name}"] = value
        return fields

    def as_record(self) -> dict:
        """A compact per-eval record for the run's ``metrics.json`` ``notes['eval_history']`` —
        what the orchestrating agent reads post-run to judge the model on the EVAL metric, not
        just the training reward. Drops the per-example list to keep metrics.json small."""
        return {
            "step": self.step,
            "eval_reward": self.mean_reward,
            "eval_pass_rate": self.pass_rate,
            "eval_n": self.n,
            "eval_metrics": dict(self.metric_means),
        }


def summarize(
    records: list[EvalRecord], *, step: int | None = None, pass_threshold: float = 0.5
) -> EvalSummary:
    """Reduce per-example records to an :class:`EvalSummary` (empty -> all-zero, n=0).

    ``pass_rate`` is the fraction with training ``reward >= pass_threshold``; each env metric is
    averaged independently across the examples that reported it.
    """
    if not records:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0, {}, [], step)
    rewards = [float(r.reward) for r in records]
    n = len(rewards)
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for rec in records:
        for name, value in rec.metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
            metric_counts[name] = metric_counts.get(name, 0) + 1
    metric_means = {name: metric_sums[name] / metric_counts[name] for name in metric_sums}
    return EvalSummary(
        n=n,
        mean_reward=sum(rewards) / n,
        pass_rate=sum(1 for r in rewards if r >= pass_threshold) / n,
        min_reward=min(rewards),
        max_reward=max(rewards),
        metric_means=metric_means,
        rewards=rewards,
        step=step,
    )


def evaluate_policy(
    examples: list[dict],
    score_one: Callable[[dict], EvalRecord],
    *,
    step: int | None = None,
    pass_threshold: float = 0.5,
    on_error: str = "zero",
    on_warn: Callable[[str], None] | None = None,
) -> EvalSummary:
    """Score every example with ``score_one`` and summarize.

    ``on_error="zero"`` (default) makes a per-example scoring failure a zero-reward record (so
    one bad rollout doesn't sink the pass); ``on_error="raise"`` re-raises (tests).

    A SYSTEMIC failure (every example errors — e.g. generation OOMs each call) raises instead of
    returning ``eval_reward=0``: a flat-zero curve reads as "the model got everything wrong",
    which is misleading when the eval simply couldn't run. The caller turns that into a skip.
    """
    records: list[EvalRecord] = []
    errors = 0
    last_error: Exception | None = None
    for ex in examples:
        try:
            records.append(score_one(ex))
        except Exception as exc:  # a bad rollout must not abort the eval/run
            if on_error == "raise":
                raise
            errors += 1
            last_error = exc
            if on_warn:
                on_warn(f"mid-run eval: example scored 0 after error: {exc}")
            records.append(EvalRecord(_EVAL_ERROR_REWARD, {}))
    if examples and errors == len(examples):
        raise RuntimeError(
            f"mid-run eval: all {errors} examples failed to generate "
            f"(last error: {last_error}) — reporting a skip, not eval_reward=0"
        )
    return summarize(records, step=step, pass_threshold=pass_threshold)


def single_turn_scorer(
    active_env,
    render_prompt_ids: Callable[[dict], list[int]],
    generate: Callable[[list[int], int], tuple[list[int], list[float], str]],
    max_new_tokens: int,
    graded_text: Callable[[str | None], str | None] = lambda c: c,
) -> Callable[[dict], EvalRecord]:
    """Per-example scorer for a single-turn env: render -> greedy generate -> env evaluate.

    ``generate(prefix_ids, max_tokens) -> (token_ids, logprobs, text)`` mirrors the colocate
    engine wrapper used by the multi-turn rollout; only ``text`` is used here. ``graded_text``
    strips ``<think>`` blocks before scoring on thinking runs (worker parity). Uses the env's
    :meth:`evaluate` so the env's eval-metric metrics come back alongside the reward."""

    def score(example: dict) -> EvalRecord:
        prefix_ids = render_prompt_ids(example)
        _ids, _lps, text = generate(prefix_ids, max_new_tokens)
        result = active_env.evaluate(graded_text(text), example)
        return EvalRecord(
            reward=float(result["reward"]),
            metrics={k: float(v) for k, v in (result.get("metrics") or {}).items()},
        )

    return score


def multi_turn_scorer(
    active_env,
    render_messages: Callable[[list, bool], list[int]],
    generate: Callable[[list[int], int], tuple[list[int], list[float], str]],
    *,
    max_turns: int,
    max_new_tokens: int,
    engine_max_len: int | None = None,
    on_warn: Callable[[str], None] | None = None,
) -> Callable[[dict], EvalRecord]:
    """Per-example scorer for a multi-turn env: drive the env turn loop GREEDILY via
    :func:`rollout_one` and read its rubric reward. (Per-metric breakdown for multi-turn is a
    follow-up — ``rollout_one`` returns only the scalar reward — so ``metrics`` is empty here.)"""

    def score(example: dict) -> EvalRecord:
        result = rollout_one(
            example=example,
            active_env=active_env,
            render=render_messages,
            generate=generate,
            max_turns=max_turns,
            per_turn_max_tokens=max_new_tokens,
            engine_max_len=engine_max_len,
            on_warn=on_warn,
        )
        return EvalRecord(reward=float(result["reward"]), metrics={})

    return score


class PeriodicEval:
    """Pure cadence + engine-binding + heartbeat logic for mid-run eval (no GPU/transformers).

    ``score_one_builder(engine) -> score_one`` defers all GPU-specific wiring (the vLLM
    SamplingParams, the tokenizer render) to call time, so this class is fully unit-testable
    with a fake engine getter, a fake builder, and a fake heartbeat sink.
    """

    def __init__(
        self,
        *,
        examples: list[dict],
        score_one_builder: Callable[[object], Callable[[dict], EvalRecord]],
        every_steps: int,
        heartbeat_fn: Callable[..., object],
        model_getter: Callable[[], object] | None = None,
        pass_threshold: float = 0.5,
        label: str = "rl_eval",
        on_warn: Callable[[str], None] = print,
    ) -> None:
        self.examples = list(examples)
        self.score_one_builder = score_one_builder
        self.every_steps = int(every_steps)
        self.heartbeat_fn = heartbeat_fn
        self.model_getter = model_getter
        self.pass_threshold = float(pass_threshold)
        self.label = label
        self.on_warn = on_warn
        self._disabled = False
        # Accumulated eval curve, persisted into metrics.json so the agent reads it post-run.
        self.history: list[EvalSummary] = []

    def history_records(self) -> list[dict]:
        """The eval curve as compact dicts for ``metrics.json`` ``notes['eval_history']``."""
        return [s.as_record() for s in self.history]

    def bind_model_getter(self, getter: Callable[[], object]) -> None:
        """Attach the (late-bound) accessor for the live generation model — the trainer that owns
        it does not exist when the callback is constructed."""
        self.model_getter = getter

    def should_run(self, step: int) -> bool:
        return (
            not self._disabled
            and self.every_steps > 0
            and step > 0
            and bool(self.examples)
            and step % self.every_steps == 0
        )

    def maybe_run(self, step: int) -> EvalSummary | None:
        if not self.should_run(step):
            return None
        return self.run_eval(step)

    def run_final(self, step: int) -> EvalSummary | None:
        """Evaluate the FINAL trained policy once, after training ends.

        The cadence (``maybe_run``) only fires on exact multiples of ``every_steps``; when the run
        length is not a multiple (the common case), the last cadence eval predates the saved
        adapter. This runs one more eval on the final model so ``eval_history`` ends on the policy
        that was actually saved — unless the last recorded eval is already this exact step (run
        length was an exact multiple, so the final step was already evaluated)."""
        if self._disabled or self.every_steps <= 0 or step <= 0 or not self.examples:
            return None
        if self.history and self.history[-1].step == step:
            return None  # the final step coincided with a cadence eval; don't double-run
        return self.run_eval(step)

    def run_eval(self, step: int) -> EvalSummary | None:
        """Evaluate the current policy once and heartbeat the result (or a skip reason).

        Never raises (even if the model getter itself throws): any failure heartbeats a skip
        reason for THIS step and lets training continue. A missing model only skips this cadence
        — it does NOT permanently disable eval, since the getter resolves the live model each
        time and a one-off ``None`` (e.g. a teardown race) can resolve by the next cadence.
        """
        try:
            model = self.model_getter() if self.model_getter is not None else None
        except Exception as exc:  # the getter (attribute access on the trainer) must not abort
            self.heartbeat_fn(
                self.label, step=step, eval_skipped=True, eval_reason=f"model getter failed: {exc}"
            )
            return None
        if model is None:
            self.heartbeat_fn(
                self.label, step=step, eval_skipped=True, eval_reason="no model available"
            )
            return None
        try:
            score_one = self.score_one_builder(model)
            summary = evaluate_policy(
                self.examples,
                score_one,
                step=step,
                pass_threshold=self.pass_threshold,
                on_warn=self.on_warn,
            )
        except Exception as exc:  # eval must never abort training
            self.heartbeat_fn(
                self.label,
                step=step,
                eval_skipped=True,
                eval_reason=f"{type(exc).__name__}: {exc}",
            )
            return None
        self.history.append(summary)  # accumulate the curve for metrics.json (agent reads it)
        self.heartbeat_fn(self.label, step=step, **summary.as_heartbeat_fields())
        return summary


def make_periodic_eval_callback(periodic: PeriodicEval):
    """Wrap a :class:`PeriodicEval` in a transformers ``TrainerCallback`` (lazy import).

    Mirrors ``worker.make_reward_heartbeat_callback``: the callback fires on every optimizer
    step end and delegates the cadence decision to ``periodic.maybe_run``."""
    from transformers import TrainerCallback

    class _PeriodicEvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0) or 0)
            periodic.maybe_run(step)

    return _PeriodicEvalCallback()


# ---------------------------------------------------------------------------
# GPU-side wiring (the only part that touches torch / the model). From worker.run_rl.
# ---------------------------------------------------------------------------
def eval_config(default_max_new: int, *, spec_every: int | None = None) -> dict:
    """Resolve the mid-run-eval knobs. The CADENCE is the only real knob: it comes from the run's
    ``[train] eval_every_steps`` TOML (``spec_every``), with ``AUTOSLM_EVAL_EVERY_STEPS`` as an
    operator override; 0/unset disables.

    Everything else comes from the ENVIRONMENT, not config: the eval queries are the env's
    held-out ``eval_dataset``, the grading is its rubric (reward + eval-metric metrics), the
    completion budget equals the run's normal ``max_tokens`` (``default_max_new``), and the pass
    threshold is the env's own. ``num_examples`` is ONLY a safety cap so a huge env eval split
    can't dominate training (``AUTOSLM_EVAL_NUM``, default 64) — the agent sizes the eval set via
    the environment, not here.
    """

    def _safe_int(value, fallback):
        # A malformed AUTOSLM_EVAL_* env var must not abort training at setup — fall back.
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    env_every = os.environ.get("AUTOSLM_EVAL_EVERY_STEPS")
    if env_every is not None and env_every.strip() != "":
        every = _safe_int(env_every, spec_every)  # bad env var -> fall back to the TOML value
    else:
        every = spec_every
    return {
        "every_steps": _safe_int(every, 0) if every is not None else 0,
        # safety cap only (negative would hit Python negative-slicing semantics on the split)
        "num_examples": max(0, _safe_int(os.environ.get("AUTOSLM_EVAL_NUM"), 64)),
        "max_new_tokens": max(1, int(default_max_new)),  # = the run's normal completion budget
    }


def build_hf_greedy_generate(
    model, tok, *, stop: list[str] | None
) -> Callable[[list[int], int], tuple[list[int], list[float], str]]:
    """A greedy (deterministic) generate over the TRAINER'S model via ``transformers.generate``,
    returning ``(token_ids, logprobs, text)`` like the multi-turn rollout's generate.

    Uses the model directly (NOT the colocate vLLM engine, whose out-of-band use from a callback
    hangs GRPO). Runs in eval + ``no_grad`` and restores train mode; ``use_cache=True`` is forced
    so generation works even when training has it off under gradient checkpointing (no backward
    happens here, so checkpointing is irrelevant). One prompt at a time keeps the KV footprint to
    a single sequence; an OOM raises (caught upstream -> ``eval_skipped``), never hangs.

    Stop sequences are honored via ``stop_strings`` so generation actually STOPS at the stop
    (rather than post-hoc truncating the text only) — this keeps the returned ``token_ids`` and
    ``text`` in lockstep (``text == decode(token_ids)``), which the multi-turn ``rollout_one``
    prefix invariant requires; truncating text alone would desync them."""
    import torch

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stop_kwargs = {"stop_strings": list(stop), "tokenizer": tok} if stop else {}

    def generate(prefix_ids: list[int], max_tokens: int):
        input_ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=model.device)
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=max(1, int(max_tokens)),
                    do_sample=False,  # greedy / deterministic eval
                    use_cache=True,
                    pad_token_id=pad_id,
                    **stop_kwargs,
                )
        finally:
            if was_training:
                model.train()
        # text == decode(new_ids): never post-hoc truncate one without the other (see docstring).
        new_ids = [int(t) for t in out[0][input_ids.shape[1] :].tolist()]
        text = tok.decode(new_ids, skip_special_tokens=True)
        return new_ids, [0.0] * len(new_ids), text

    return generate
