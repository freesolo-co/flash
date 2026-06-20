"""Tinker GRPO runner for the benchmark.

Must be invoked with a Python that has `verifiers` installed, e.g.:
    /path/to/venv-with-verifiers/bin/python benchmark/tinker_runner.py

Runs Qwen3.5-4B GRPO on the GSM8K verifiers env via tinker_cookbook,
matched to the flash side: 30 steps, groups_per_batch=4, group_size=4,
max_tokens=512.

Writes a JSON result to --output (default /tmp/tinker_bench_result.json)
so the bench.py orchestrator can read it after the subprocess exits.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import pathlib
import time

# Single shared Tinker cost proxy for the WHOLE benchmark (runner + assembler + results).
# Tinker is managed and does NOT expose per-run cost via API; this is an explicit, labelled
# proxy so the $ column is order-of-magnitude comparable to Flash's MEASURED cost. The
# basis is ACTIVE compute (rollout + train step), excluding managed-backend capacity pauses,
# matching assemble.py / results/comparison.json. assemble.py imports this constant.
TINKER_PROXY_USD_PER_HR = 2.00


_ROLLOUT_KEY = "time/do_group_rollout_and_filter_constant_reward:total"
_TRAIN_KEY = "time/train_step"


def active_compute_s(records: list[dict]) -> float | None:
    """Sum real per-step work (rollout + train step) from metrics.jsonl records.

    Tinker's wall-clock absorbs managed-backend capacity pauses (idle, unbilled) into a
    step's timing; summing rollout + train-step time is the fair active-compute basis used
    for both latency and the cost proxy.

    Returns ``None`` ONLY when no record carries either timing key (genuinely no timing
    data — fall back to wall). When timing rows ARE present but sum to ``0.0`` (e.g. a run
    that recorded zero work), return ``0.0`` — a real, pause-excluded zero that callers must
    NOT silently replace with wall. Callers therefore guard with ``is not None``, not truthiness.
    """
    roll = train = 0.0
    have_timing = False
    for r in records:
        if _ROLLOUT_KEY in r or _TRAIN_KEY in r:
            have_timing = True
        roll += r.get(_ROLLOUT_KEY, 0) or 0
        train += r.get(_TRAIN_KEY, 0) or 0
    if not have_timing:
        return None
    return roll + train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--groups-per-batch", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--env-id", default="gsm8k")
    p.add_argument("--log-path", default="/tmp/tinker-bench-gsm8k")
    p.add_argument("--output", default="/tmp/tinker_bench_result.json")
    return p.parse_args()


def _patch_dataset_builder_for_verifiers_skew() -> None:
    """Tolerate the tinker_cookbook 0.4.0 <-> verifiers 0.1.14 schema skew.

    The recipe's VerifiersRLDatasetBuilder hard-requires a ``task`` column, but
    verifiers 0.1.14 single-turn env datasets (gsm8k / hendrycks-math / reverse-text)
    expose ``[question/prompt/answer/example_id]`` with no ``task`` column. ``task`` is
    only used as a rollout-group label downstream (it does not feed the reward), so we
    default it to the env id when absent. Mirrors the original builder otherwise.
    """
    import verifiers as vf
    from tinker_cookbook.recipes.verifiers_rl import verifiers_env as _ve

    async def _call(self):
        env = _ve.get_vf_env()
        if env is None:
            env = vf.load_environment(self.vf_env_id, **self.vf_env_args)
            _ve.set_vf_env(env)
        ds = env.get_dataset(n=self.dataset_n, seed=self.dataset_seed)
        cols = ds.column_names
        rows = [
            {
                "prompt": ds["prompt"][i],
                "example_id": ds["example_id"][i] if "example_id" in cols else i,
                "task": ds["task"][i] if "task" in cols else self.vf_env_id,
                **({"answer": ds["answer"][i]} if "answer" in cols else {}),
                **({"info": ds["info"][i]} if "info" in cols else {}),
            }
            for i in range(len(ds))
        ]
        return _ve.VerifiersRLDataset(rows, env, self.groups_per_batch), None

    _ve.VerifiersRLDatasetBuilder.__call__ = _call


def _patch_rollout_for_verifiers(model_name: str, group_size: int) -> None:
    """Wire the verifiers rollout into the *actual* call site.

    The recipe sets ``train.do_group_rollout = custom`` for monkey-patching, but in
    tinker_cookbook 0.4.0 the sync training loop reaches rollouts via
    ``rollouts.do_group_rollout_and_filter_constant_reward`` -> the module-local
    ``rollouts.do_group_rollout`` — so the recipe's patch on ``train`` is dead code and
    the generic rollout path returns no trajectories (``zip(*[])`` -> ValueError). We
    patch ``rollouts.do_group_rollout`` directly with the recipe's verifiers logic
    (generate via the Tinker OpenAI shim, score with the verifiers env, convert to a
    TrajectoryGroup). State (client/renderer/tokenizer) is cached across groups.
    """
    from typing import cast

    import tinker_cookbook.rl.rollouts as rollouts
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.recipes.verifiers_rl.tinker_openai import TinkerAsyncOpenAIClient
    from tinker_cookbook.recipes.verifiers_rl.verifiers_env import (
        VerifiersEnvGroupBuilder,
        convert_states_to_trajectory_group,
    )
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from verifiers.utils.async_utils import maybe_semaphore

    cache: dict = {"client": None, "renderer": None, "tokenizer": None}

    async def _custom(env_group_builder, policy, strategy=None):
        if cache["tokenizer"] is None:
            cache["tokenizer"] = get_tokenizer(model_name)
            rname = model_info.get_recommended_renderer_name(model_name)
            cache["renderer"] = renderers.get_renderer(rname, cache["tokenizer"])
        sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
        if cache["client"] is None:
            cache["client"] = TinkerAsyncOpenAIClient(
                sampling_client, cache["renderer"], cache["tokenizer"]
            )
        else:
            cache["client"].set_sampling_client(sampling_client)

        vf_builder = cast(VerifiersEnvGroupBuilder, env_group_builder)
        rollout_inputs = vf_builder.get_rollout_inputs(group_size)
        gen_sem = await maybe_semaphore(-1)
        score_sem = await maybe_semaphore(-1)
        states = await vf_builder.vf_env.run_group(
            group_inputs=rollout_inputs,
            client=cache["client"],
            model="tinker",
            gen_sampling_args={
                "max_tokens": policy.max_tokens,
                "temperature": policy.temperature,
            },
            gen_sem=gen_sem,
            score_sem=score_sem,
        )
        return convert_states_to_trajectory_group(states)

    rollouts.do_group_rollout = _custom


async def _run(args: argparse.Namespace) -> dict:
    """Run Tinker GRPO training and return a metrics dict."""
    _patch_dataset_builder_for_verifiers_skew()
    _patch_rollout_for_verifiers(args.model, args.group_size)
    from tinker_cookbook.recipes.verifiers_rl.train import CLIConfig, cli_main

    cfg = CLIConfig(
        model_name=args.model,
        lora_rank=args.lora_rank,
        vf_env_id=args.env_id,
        groups_per_batch=args.groups_per_batch,
        group_size=args.group_size,
        max_tokens=args.max_tokens,
        max_steps=args.steps,
        # The verifiers_rl recipe does not wire evaluator_builders, so in-loop eval
        # would no-op. Held-out eval is done separately (eval_runner.py) for both
        # platforms on the same GSM8K test split — that is the apples-to-apples metric.
        eval_every=0,
        save_every=args.steps,
        log_path=args.log_path,
        behavior_if_log_dir_exists="delete",
        wandb_project=None,
    )

    t0 = time.monotonic()
    await cli_main(cfg, None)
    wall = time.monotonic() - t0
    return {"wall_s": wall}


def _parse_metrics_jsonl(path: str) -> list[dict]:
    """Parse one metrics.jsonl into a list of records (skip blank / non-JSON lines).

    Returns ``[]`` if the file is empty or has no parseable JSON lines. Single parser so
    the file selector and the public reader judge "has records" the same way.
    """
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    records.append(json.loads(line))
    return records


def select_metrics_file(log_path: str | None) -> str | None:
    """Deterministically pick the metrics.jsonl for a Tinker run's log dir.

    Single source of truth for BOTH the runner (``_read_metrics_jsonl``) and the
    assembler (``_tinker_active_compute_s``), so they always parse the SAME file and
    can never disagree on the active-compute total for a given log dir.

    Selection: prefer a direct ``log_path/metrics.jsonl`` **only when it actually has
    parseable records** — an empty/unparseable direct file must NOT shadow a nested
    ``**/metrics.jsonl`` that holds the real timing data. Otherwise fall through to the
    deterministically-sorted first nested file that has records. Returns ``None`` when no
    metrics file with records exists (caller degrades to [] / no active-compute, no crash).
    """
    if not log_path:
        return None
    import glob
    direct = os.path.join(log_path, "metrics.jsonl")
    if os.path.exists(direct) and _parse_metrics_jsonl(direct):
        return direct
    nested = sorted(glob.glob(os.path.join(log_path, "**", "metrics.jsonl"), recursive=True))
    for cand in nested:
        if cand != direct and _parse_metrics_jsonl(cand):
            return cand
    # Nothing with records anywhere: surface the direct file if it exists (so an empty run
    # still names its file) else None. Either way read_metrics_records returns [] gracefully.
    return direct if os.path.exists(direct) else None


def read_metrics_records(log_path: str | None) -> list[dict]:
    """Parse the canonically-selected metrics.jsonl (see :func:`select_metrics_file`).

    Shared parser so the runner and assembler read identical records. Returns [] if no
    file is found or it has no valid JSON lines.
    """
    path = select_metrics_file(log_path)
    if path is None:
        return []
    return _parse_metrics_jsonl(path)


def _read_metrics_jsonl(log_path: str) -> list[dict]:
    """Read Tinker's metrics.jsonl via the shared canonical selector/parser."""
    return read_metrics_records(log_path)


def main() -> None:
    args = parse_args()

    # Run the async training
    timing = asyncio.run(_run(args))

    # Extract reward trajectory from metrics.jsonl. The mean group reward per step is
    # logged under "env/all/reward/total" (tinker_cookbook 0.4.0 + verifiers 0.1.9).
    records = _read_metrics_jsonl(args.log_path)

    def _reward_of(rec: dict) -> float | None:
        for key in ("env/all/reward/total", "reward/total"):
            if key in rec:
                return rec[key]
        # fall back to any "<scope>/reward/total" key
        for k, v in rec.items():
            if k.endswith("reward/total"):
                return v
        return None

    reward_history = [r for r in (_reward_of(rec) for rec in records) if r is not None]

    # Cost on the SAME basis as assemble.py / comparison.json: active compute (rollout +
    # train step, capacity pauses excluded) x the shared $/hr proxy. Falls back to full wall
    # ONLY when active is None (no timing rows at all); a legitimate 0.0 active is kept
    # (use `is not None`, not truthiness). Emit active/paused so the JSON is self-consistent.
    wall = timing["wall_s"]
    active = active_compute_s(records)
    basis = active if active is not None else wall
    paused = (wall - active) if (active is not None and wall > active) else None

    result = {
        "platform": "tinker",
        "status": "done",
        "model": args.model,
        "env_id": args.env_id,
        "steps": args.steps,
        "wall_s": wall,
        "active_compute_s": active,
        "paused_s": paused,
        # Tinker does not expose billing via API; this is a labelled proxy, NOT a bill.
        # `basis` is active-compute (rollout+train) when present, else wall; a legitimate 0.0
        # basis (zero active compute) is a REAL value -> emit a 0.0 cost, never None. Guard on
        # `is not None`, not truthiness, so 0.0 isn't dropped as "missing" (matches the module
        # contract on active_compute_s; only a genuinely-None basis yields a None cost).
        "cost_usd_estimated": (
            round(basis / 3600 * TINKER_PROXY_USD_PER_HR, 4) if basis is not None else None
        ),
        "cost_note": (
            f"estimated: active-compute x ${TINKER_PROXY_USD_PER_HR:.2f}/hr proxy "
            "(pause excluded; Tinker cost not in API - check your dashboard)"
        ),
        "first_train_reward": reward_history[0] if reward_history else None,
        "final_train_reward": reward_history[-1] if reward_history else None,
        "reward_history": reward_history,
        "log_path": args.log_path,
    }

    out = pathlib.Path(args.output)
    out.write_text(json.dumps(result, indent=2))
    print(f"[tinker] done — result written to {out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
