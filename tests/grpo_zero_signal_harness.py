"""End-to-end reproduction harness for the 2026-11-18 zero-learning-signal GRPO runs.

Everything here is the production GRPO parent path, unmodified:

* the researcher's environment is loaded through Flash's own loader and adapter
  (``_load_resolved_freesolo_environment`` -> ``FreesoloEnvironment``);
* rollouts are scored by the real localhost reward bridge (``_start_reward_runtime`` ->
  ``start_reward_server`` -> ``score_single_turn_batch``), called from a child process through the
  real packaged verl reward callback (``flash.engine.worker.train.rl.child.reward.compute_score``)
  with real rollout identities registered on the real ledger;
* the child's stdout is streamed, parsed and acted on by ``_execute_rl_child`` /
  ``_ingest_step_metrics``, and the run is judged by ``_validate_rl_child`` (the publish-time
  ``_check_grpo_had_a_gradient`` verdict).

The only stand-in is the trainer inside the child. verl needs CUDA and a real model, so the child
runs the GRPO update itself on the smallest possible policy: a per-prompt softmax over a few fixed
candidate completions, sampled at temperature, rewarded through the bridge, with verl's GRPO
advantage (reward minus group mean; Flash sets ``norm_adv_by_std_in_grpo=False``), the
policy-gradient loss ``-mean(A * log pi)``, its exact gradient, and an SGD step. It prints verl's
LocalLogger step line, which is the only thing the parent reads from verl.

The two broken environments are the researcher's code verbatim (html-kimi package ``5f272542``
and html-astra package ``a0cf73c7``) with the judge URL replaced by a placeholder. The astra judge
is emulated in-process with the live evaluator's observed replies (HTTP 400 ``Invalid request`` for
``case_id: null``, 404 ``Unknown case`` for an unknown id).
"""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import textwrap
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

JUDGE_URL = "https://judge.invalid"
SOURCE_WORDS = ("alpha", "beta", "gamma", "delta")
CANDIDATES = (
    "<p>alpha beta gamma delta</p>",
    "<p>alpha beta</p>",
    "<p>alpha</p>",
    "alpha beta gamma delta",
)

# html-kimi package 5f272542: reward_url / reward_case_id read from example.metadata.
KIMI_ENVIRONMENT = '''
"""HTML-condense GRPO with the official case-scoped protocol-5 reward."""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from freesolo.datasets import TaskExample
from freesolo.datasets.records import load_task_examples
from freesolo.environments import EnvironmentSingleTurn, RewardResult

ROOT = Path(__file__).parent


def official_measure(metadata, response, finish_reason):
    url = metadata["reward_url"]
    if urlparse(url).scheme != "https":
        raise ValueError("Reward service requires HTTPS")
    case_id = metadata.get("reward_case_id", metadata["id"])
    request = urllib.request.Request(
        url.rstrip("/") + "/score",
        data=json.dumps({
            "case_id": case_id,
            "response": response,
            "finish_reason": finish_reason,
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(request, timeout=240) as stream:
                result = json.load(stream)
            if result.get("id") != case_id or result.get("protocol") != 5:
                raise RuntimeError("Invalid HTML judge service response")
            return result
        except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError) as error:
            last_error = error
            if attempt < 7:
                time.sleep(min(30, 2 ** attempt))
    # Infrastructure unavailability is never relabeled as a zero-quality response.
    raise RuntimeError("HTML judge service unavailable; no reward assigned") from last_error


class HtmlCondenseRewardEnv(EnvironmentSingleTurn):
    def __init__(self):
        self.dataset = load_task_examples(ROOT / "dataset" / "train.jsonl")

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [{"role": "user", "content": example.input}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        finish_reason = "stop"
        context = getattr(response_text, "rollout_context", None)
        if context is not None and getattr(context, "turns", None):
            finish_reason = context.turns[-1].finish_reason
        result = official_measure(example.metadata, str(response_text), finish_reason)
        return RewardResult(score=float(result["score"]), threshold=1.0)


def load_environment(**kwargs):
    return HtmlCondenseRewardEnv()
'''

# html-astra package a0cf73c7: case id taken from example.id.
ASTRA_CLIENT = '''
"""Case-scoped reward client. Provider credentials remain on the evaluator service."""

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse


def measure(case, response, *, finish_reason="stop"):
    url = case["reward_url"]
    if urlparse(url).scheme != "https":
        raise ValueError("Reward service requires HTTPS")
    request = urllib.request.Request(
        url.rstrip("/") + "/score",
        data=json.dumps(
            {
                "case_id": case.get("reward_case_id", case["id"]),
                "response": response,
                "finish_reason": finish_reason,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as stream:
            result = json.load(stream)
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise RuntimeError("HTML judge service unavailable; no reward assigned") from None
    if result.get("id") != case.get("reward_case_id", case["id"]) or result.get("protocol") != 5:
        raise RuntimeError("Invalid HTML judge service response")
    return result
'''

ASTRA_ENVIRONMENT = """
from pathlib import Path
import json,time
from concurrent.futures import ThreadPoolExecutor
from freesolo.environments import EnvironmentSingleTurn,RewardResult
from client import measure
ROOT=Path(__file__).parent

class Condense(EnvironmentSingleTurn):
    def build_prompt_messages(self,example,prompt_text):
        return [{'role':'system','content':(ROOT/'system.txt').read_text().strip()},{'role':'user','content':example.input}]
    def score_response(self,example,response_text):
        record={'id':example.id,'reward_url':example.metadata['reward_url']}
        for i in range(3):
            try:
                m=measure(record,response_text);return RewardResult(score=m['score'],threshold=.65)
            except RuntimeError:
                if i==2:raise
                time.sleep(2**i)
def load_environment(dataset_path=None,**kwargs):
    env=Condense();env.dataset=list(map(json.loads,open(dataset_path or ROOT/'dataset/train.jsonl')));return env
"""

# the one-line repair of the astra env: name the case by the record's own id. this is the
# working-reward control -- same judge, same data, same policy -- that must NOT trip the guard.
ASTRA_FIXED_ENVIRONMENT = ASTRA_ENVIRONMENT.replace(
    "record={'id':example.id,", "record={'id':example.record['id'],"
)


def _row(case_id: str, *, metadata_url: bool) -> dict:
    row = {
        "id": case_id,
        "input": f"Maximum output tokens (Qwen tokenizer): 8\n\nHTML:\n<p>{' '.join(SOURCE_WORDS)}</p>",
        "output": CANDIDATES[0],
    }
    if metadata_url:
        row["metadata"] = {"reward_url": JUDGE_URL}
    else:
        # the kimi dataset shape: judge fields are top-level, metadata carries provenance only.
        row["metadata"] = {"dataset": "hotpot", "token_budget": 8}
        row["reward_case_id"] = case_id
        row["reward_url"] = JUDGE_URL
    return row


SCENARIOS = {
    # name: (environment source, extra files, metadata-style rows?, prompts_per_step, group_size)
    "html-kimi": (KIMI_ENVIRONMENT, {}, False, 8, 4),
    "html-astra": (ASTRA_ENVIRONMENT, {"client.py": ASTRA_CLIENT}, True, 3, 4),
    "healthy": (ASTRA_FIXED_ENVIRONMENT, {"client.py": ASTRA_CLIENT}, True, 3, 4),
}


def write_environment(root: Path, scenario: str, *, rows: int = 8) -> Path:
    source, extra, metadata_url, _, _ = SCENARIOS[scenario]
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset").mkdir(exist_ok=True)
    (root / "environment.py").write_text(textwrap.dedent(source).lstrip())
    for name, text in extra.items():
        (root / name).write_text(textwrap.dedent(text).lstrip())
    (root / "system.txt").write_text("Condense the supplied HTML. Return only reduced HTML.\n")
    with (root / "dataset" / "train.jsonl").open("w") as handle:
        for index in range(rows):
            handle.write(json.dumps(_row(f"hotpot-{index:04d}", metadata_url=metadata_url)) + "\n")
    return root / "environment.py"


class _JudgeReply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@contextmanager
def emulated_judge(known_cases: set[str]):
    """the protocol-5 evaluator, answering exactly as the live one did for each case id."""
    real_urlopen = urllib.request.urlopen

    def urlopen(request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        if not url.startswith(JUDGE_URL):
            return real_urlopen(request, *args, **kwargs)
        payload = json.loads(request.data)
        case_id = payload.get("case_id")

        def refuse(code: int, detail: str):
            body = io.BytesIO(json.dumps({"detail": detail}).encode())
            raise urllib.error.HTTPError(url, code, detail, {}, body)

        if not isinstance(case_id, str):
            refuse(400, "Invalid request")
        if case_id not in known_cases:
            refuse(404, "Unknown case")
        response = str(payload.get("response", ""))
        valid = response.lstrip().startswith("<")
        kept = sum(word in response for word in SOURCE_WORDS) / len(SOURCE_WORDS)
        score = kept if valid else 0.0
        return _JudgeReply(json.dumps({"id": case_id, "protocol": 5, "score": score}).encode())

    urllib.request.urlopen = urlopen
    try:
        yield
    finally:
        urllib.request.urlopen = real_urlopen


# the stand-in trainer. it imports the real packaged reward callback from whichever flash tree is on
# PYTHONPATH, registers each step's rollout identities, scores every rollout through the parent's
# bridge, and does the GRPO update on a tabular softmax policy.
CHILD_TRAINER = r"""
import json, math, os, random, sys, time

from flash.engine.worker.train.rl.child.multiturn import post_json
from flash.engine.worker.train.rl.child.reward import compute_score

cfg = json.loads(os.environ["FLASH_REPRO_CFG"])
url = os.environ["FLASH_VERL_REWARD_URL"]
candidates = cfg["candidates"]
n_examples, ppstep, group = cfg["examples"], cfg["prompts_per_step"], cfg["group_size"]
temperature, lr = cfg["temperature"], cfg["learning_rate"]
rng = random.Random(cfg["seed"])
theta = [[0.0] * len(candidates) for _ in range(n_examples)]

def policy(p):
    logits = [value / temperature for value in theta[p]]
    top = max(logits)
    weights = [math.exp(value - top) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]

for step in range(1, cfg["steps"] + 1):
    prompts = [((step - 1) * ppstep + i) % n_examples for i in range(ppstep)]
    identities = [
        {"optimizer_step": step, "sample_index": p, "rollout_ordinal": k, "validate": False}
        for p in prompts for k in range(group)
    ]
    post_json(url, "/identity/register", {"identities": identities}, error_style="reward")
    grads = [[0.0] * len(candidates) for _ in range(n_examples)]
    advantages, rewards, loss = [], [], 0.0
    batch = len(prompts) * group
    for p in prompts:
        pi = policy(p)
        picks = rng.choices(range(len(candidates)), weights=pi, k=group)
        scores = [
            compute_score(
                "flash", candidates[m], None,
                extra_info={"index": p, "flash_rollout_identity": {
                    "optimizer_step": step, "sample_index": p,
                    "rollout_ordinal": k, "validate": False}},
            )
            for k, m in enumerate(picks)
        ]
        mean = sum(scores) / group
        for m, score in zip(picks, scores):
            advantage = score - mean  # verl grpo, norm_adv_by_std_in_grpo=False
            advantages.append(advantage)
            rewards.append(score)
            loss -= advantage * math.log(pi[m]) / batch
            for j in range(len(candidates)):
                # d(-A log pi_m)/d theta_j = -A * (1[j==m] - pi_j) / T
                grads[p][j] -= advantage * ((1.0 if j == m else 0.0) - pi[j]) / temperature / batch
    grad_norm = math.sqrt(sum(g * g for row in grads for g in row))
    for p in range(n_examples):
        for j in range(len(candidates)):
            theta[p][j] -= lr * grads[p][j]
    print(
        f"step:{step} - actor/pg_loss:{loss!r} - actor/grad_norm:{grad_norm!r} - "
        f"critic/rewards/mean:{sum(rewards) / batch!r} - "
        f"critic/advantages/min:{min(advantages)!r} - critic/advantages/max:{max(advantages)!r} - "
        f"response_length/mean:16.0 - timing_s/step:{cfg['step_seconds']!r}",
        flush=True,
    )
    time.sleep(cfg["step_seconds"])
"""


@dataclass
class ReproResult:
    scenario: str
    configured_steps: int
    steps_seen: int = 0
    outcome: str = ""
    error: str = ""
    grad_norms: dict[int, float] = field(default_factory=dict)
    rewards: list[float] = field(default_factory=list)
    reward_calls: int = 0
    child_alive_after: bool = False
    wall_s: float = 0.0

    def summary(self) -> str:
        positive = sum(value > 0.0 for value in self.grad_norms.values())
        lines = [
            (
                f"scenario={self.scenario} configured_steps={self.configured_steps} "
                f"steps_executed={self.steps_seen} outcome={self.outcome} wall={self.wall_s:.1f}s"
            ),
            (
                f"  reward per step: {sorted({round(r, 4) for r in self.rewards})} "
                f"(reward calls: {self.reward_calls})"
            ),
            (
                f"  grad_norm: positive on {positive}/{len(self.grad_norms)} steps; "
                f"child alive after return: {self.child_alive_after}"
            ),
        ]
        if self.error:
            lines.append(f"  error: {self.error}")
        return "\n".join(lines)


def run_scenario(
    scenario: str,
    workdir: Path,
    *,
    steps: int,
    abort_steps: int | None = None,
    step_seconds: float = 0.05,
    seed: int = 7,
) -> ReproResult:
    """drive one GRPO run through the production parent path and report how it ended."""
    import subprocess

    import flash.engine.worker.train.entry.rl_train_runner as runner
    from flash.engine.worker.runtime.state import WorkerContext
    from flash.envs.loading.loader import _load_resolved_freesolo_environment

    _, _, _, prompts_per_step, group_size = SCENARIOS[scenario]
    reference = write_environment(workdir / scenario / "env", scenario)
    env = _load_resolved_freesolo_environment(f"repro/{scenario}", str(reference), {})
    # the researcher's retry backoff (1 s, 2 s, up to 30 s) is wall time only; it never changes
    # what the reward returns, so it is skipped to keep the reproduction fast.
    sdk_module = sys.modules[type(env._env).__module__]
    if hasattr(sdk_module, "time"):
        sdk_module.time = type("_NoSleep", (), {"sleep": staticmethod(lambda _s: None)})
    records = env.dataset()
    examples = len(records)

    inp = {
        "prompts_per_step": prompts_per_step,
        "group_size": group_size,
        "prompt_opened_thinking": False,
        "think_penalty": 0.0,
        "multi_turn": False,
        "max_turns": 0,
        "package_root": str(reference.parent),
        "processor": None,
        "per_turn_credit": False,
        "tools": None,
        "max_completion": 64,
        "steps": steps,
        "zero_signal_abort_steps": abort_steps,
    }
    if abort_steps is None:
        from flash.engine.worker.train.rl.launch import signal_guard

        inp["zero_signal_abort_steps"] = signal_guard.DEFAULT_ZERO_SIGNAL_ABORT_STEPS
    files = {
        "rollout_examples": records,
        "message_prompts": [env.prompt_messages(record) for record in records],
        "resume_step": 0,
        "launch_config_path": str(workdir / "unused-launch-config.json"),
        "readiness_path": None,
    }
    context = WorkerContext.from_env()
    reward_runtime = runner._start_reward_runtime(inp, env, None, [], files, context=context)

    child = workdir / scenario / "child_trainer.py"
    child.write_text(CHILD_TRAINER)
    python_bin = workdir / scenario / "verl-python"
    # _execute_rl_child runs `<python_bin> -m flash.engine.worker.train.rl.child.entry ...`;
    # this interpreter shim runs the stand-in trainer in its place.
    python_bin.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{child}"\n')
    python_bin.chmod(python_bin.stat().st_mode | stat.S_IEXEC)
    flash_root = str(Path(runner.__file__).resolve().parents[5])
    env_for_verl = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([flash_root, os.environ.get("PYTHONPATH", "")]),
        "FLASH_VERL_REWARD_URL": reward_runtime.reward_url,
        "FLASH_REPRO_CFG": json.dumps(
            {
                "candidates": list(CANDIDATES),
                "examples": examples,
                "prompts_per_step": prompts_per_step,
                "group_size": group_size,
                "temperature": 1.0,
                "learning_rate": 4.0,
                "seed": seed,
                "steps": steps,
                "step_seconds": step_seconds,
            }
        ),
    }

    state = runner._StepMetricState(resume_step=0)
    state.set_prior_positive_step(None, checkpoint_step=0)
    arm = getattr(runner, "_arm_signal_guard", None)  # absent on unpatched origin/dev
    if arm is not None:
        arm(state, inp, files, reward_runtime)

    spawned: list[subprocess.Popen] = []
    real_popen = runner.subprocess.Popen

    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    real_heartbeat = runner._worker_heartbeat.heartbeat
    real_gpu = runner.gpu_diagnostics
    runner.subprocess.Popen = popen
    runner._worker_heartbeat.heartbeat = lambda *_a, **_k: True
    runner.gpu_diagnostics = lambda **_k: {}
    result = ReproResult(scenario, steps)
    started = time.monotonic()
    known = {record["id"] for record in records}
    try:
        with emulated_judge(known):
            try:
                rc = runner._execute_rl_child(
                    python_bin=str(python_bin),
                    overrides=[],
                    env_for_verl=env_for_verl,
                    inp=inp,
                    state=state,
                    reward_runtime=reward_runtime,
                    _reward_observability=dict,
                    files=files,
                    context=context,
                )
                runner._validate_rl_child(
                    rc, state, 0, steps, None, files=files, reward_runtime=reward_runtime
                )
                result.outcome = "published"
            except Exception as exc:  # the verdict under test
                result.outcome = "failed"
                result.error = f"{type(exc).__name__}: {exc}"
    finally:
        runner.subprocess.Popen = real_popen
        runner._worker_heartbeat.heartbeat = real_heartbeat
        runner.gpu_diagnostics = real_gpu
        reward_runtime.server.shutdown()
        result.wall_s = time.monotonic() - started
    result.steps_seen = state.progress["step"]
    result.grad_norms = dict(state.grad_norms)
    result.rewards = list(state.reward_history)
    result.reward_calls = (
        reward_runtime.faults.snapshot()[0]
        if hasattr(reward_runtime, "faults")
        else len(state.reward_history) * prompts_per_step * group_size
    )
    result.child_alive_after = any(proc.poll() is None for proc in spawned)
    return result


def main(argv: list[str]) -> None:
    """``python tests/grpo_zero_signal_harness.py STEPS SCENARIO...`` -- the PR's before/after.

    Runs against whichever flash tree is importable, so ``PYTHONPATH=<other checkout>`` replays
    the same scenarios on another revision.
    """
    import tempfile

    import flash.engine.worker.train.entry.rl_train_runner as runner

    print("flash tree:", Path(runner.__file__).resolve().parents[5], flush=True)
    steps = int(argv[0])
    for scenario in argv[1:]:
        with tempfile.TemporaryDirectory() as workdir:
            outcome = run_scenario(scenario, Path(workdir), steps=steps, abort_steps=8)
            print(outcome.summary(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
