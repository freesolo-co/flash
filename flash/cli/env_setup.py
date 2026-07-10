"""Starter environment scaffolds and the `flash env setup` command handler."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import render
from .training_doc import TRAINING_MD

_STARTER_ENV_PY = '''\
"""Starter Freesolo environment.

Edit dataset/train.jsonl and the reward code, then upload with
`flash env push --name my-env .`.

A managed run should use the returned [environment] id from
`flash env push --name my-env .`.

This starter keeps a tiny smoke-test dataset in dataset/train.jsonl. Replace it
with your real training rows before a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult


DEFAULT_DATASET_PATH = Path(__file__).parent / "dataset" / "train.jsonl"


def load_jsonl(path: str | Path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def exact_match_reward(example: TaskExample, response_text: str) -> RewardResult:
    expected = str(example.output or "").strip()
    score = 1.0 if expected and expected in response_text else 0.0
    return RewardResult(score=score, threshold=1.0)


class StarterEnv(EnvironmentSingleTurn):
    dataset = load_jsonl(DEFAULT_DATASET_PATH)

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [{"role": "user", "content": example.input}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        return exact_match_reward(example, response_text)


def load_environment(dataset_path: str | None = None, **kwargs) -> StarterEnv:
    env = StarterEnv()
    if dataset_path:
        env.dataset = load_jsonl(dataset_path)
    return env
'''

_STARTER_DATASET_JSONL = """\
{"input":"What is 2 + 2?","output":"4"}
{"input":"What is 3 + 5?","output":"8"}
"""


_STARTER_ENV_MULTITURN_PY = '''\
"""Starter Freesolo multi-turn environment.

A multi-turn environment runs a bounded episode: the model produces an assistant
action, `step_episode` advances the world (optionally appending an observation
message), and the loop repeats until `done` or `max_episode_turns`. The finished
transcript is graded by `score_episode`.

Edit dataset/train.jsonl and the episode logic, then upload with
`flash env push --name my-env .`.

A managed run should use the returned [environment] id from
`flash env push --name my-env .`.

This starter implements a tiny "guess the secret number" game so you can see the
episode hooks wired end-to-end. Replace it with your real task before a real run.

All three algorithms train off this file:
- GRPO (configs/rl.toml) rolls out full episodes and optimizes `score_episode`.
- SFT (configs/sft.toml) learns the gold trajectory. Provide it per row as
  `output = {"messages": [...]}` (a full assistant/tool trajectory) or a scalar
  `output` for a single gold assistant turn.
- OPD (configs/opd.toml) rolls out each episode and distils EVERY assistant turn against
  the managed Fireworks teacher (GLM 5.2 by default; pick another with [train] teacher_model),
  conditioned on the transcript so far — the multi-turn on-policy-distillation objective. The
  teacher key is platform-managed (nothing to set).
"""

from __future__ import annotations

import json
from pathlib import Path

from freesolo.datasets.types import TaskExample
from freesolo.environments import (
    EnvironmentEpisode,
    EnvironmentMultiTurn,
    EnvironmentStepResult,
    RewardResult,
)


DEFAULT_DATASET_PATH = Path(__file__).parent / "dataset" / "train.jsonl"

# How many assistant guesses the model gets before the episode is forced terminal.
MAX_TURNS = 5


def load_jsonl(path: str | Path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _secret(example: TaskExample) -> int:
    return int(str(example.output).strip())


class StarterMultiTurnEnv(EnvironmentMultiTurn):
    dataset = load_jsonl(DEFAULT_DATASET_PATH)

    def start_episode(self, example: TaskExample, prompt_text: str):
        # The opening prompt shown to the model. `example.input` describes the range.
        return [
            {
                "role": "user",
                "content": (
                    f"{example.input}\\n"
                    f"Reply with a single integer per turn. I will say 'higher', "
                    f"'lower', or 'correct'. You have {MAX_TURNS} guesses."
                ),
            }
        ]

    def max_episode_turns(self, example: TaskExample) -> int:
        return MAX_TURNS

    def step_episode(
        self,
        example: TaskExample,
        messages: list,
        assistant_response: str,
    ) -> EnvironmentStepResult:
        # Advance the world after one assistant action. Return done=True to end the
        # episode, or append an observation message and keep going.
        try:
            guess = int(assistant_response.strip().split()[0])
        except (ValueError, IndexError):
            return EnvironmentStepResult(
                done=False,
                messages=[{"role": "user", "content": "Please reply with a single integer."}],
            )
        secret = _secret(example)
        if guess == secret:
            return EnvironmentStepResult(done=True, final_response_text=str(guess))
        hint = "higher" if guess < secret else "lower"
        return EnvironmentStepResult(
            done=False,
            messages=[{"role": "user", "content": hint}],
        )

    def score_episode(
        self,
        example: TaskExample,
        episode: EnvironmentEpisode,
    ) -> RewardResult:
        # Grade the finished transcript. Reward a correct final guess; give partial
        # credit for getting close so GRPO has a usable gradient.
        secret = _secret(example)
        try:
            final = int(str(episode.response_text).strip())
        except ValueError:
            return RewardResult(score=0.0, threshold=1.0)
        if final == secret:
            return RewardResult(score=1.0, threshold=1.0, success=True)
        # Closeness in [0, 1): further guesses score lower.
        closeness = max(0.0, 1.0 - abs(final - secret) / 100.0)
        return RewardResult(score=closeness * 0.5, threshold=1.0, success=False)


def load_environment(dataset_path: str | None = None, **kwargs) -> StarterMultiTurnEnv:
    env = StarterMultiTurnEnv()
    if dataset_path:
        env.dataset = load_jsonl(dataset_path)
    return env
'''


# Multi-turn rows: `input` sets up the episode, `output` is the secret number.
# The scalar `output` also serves as the gold single-turn SFT target; swap in
# `{"messages": [...]}` to teach a full gold trajectory.
_STARTER_DATASET_MULTITURN_JSONL = """\
{"input":"I picked a secret whole number between 1 and 100.","output":"42"}
{"input":"I picked a secret whole number between 1 and 100.","output":"73"}
"""


_TURN_OPTIONS = [
    ("single", "single-turn", "one prompt -> one response"),
    ("multi", "multi-turn", "bounded episode: step_episode / score_episode"),
]
_REASONING_OPTIONS = [
    ("off", "no reasoning", "the model answers directly"),
    ("on", "reasoning", "thinking = true; spends more tokens per answer"),
]


def _setup_interactive(args) -> bool:
    """Whether to ask the setup questions interactively.

    Prompt only on a real terminal a human can answer from. Fall back to defaults (no prompt) when
    --yes is passed, under CI, or when stdin is closed/redirected, so automation never blocks on an
    unanswered prompt. A pseudo-TTY in CI can report isatty()=True, so CI is checked explicitly; a
    closed fd 0 can make sys.stdin None, so that is guarded before .isatty()."""
    if getattr(args, "yes", False):
        return False
    if os.environ.get("CI", "").strip().lower() not in ("", "0", "false", "no"):
        return False
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        return False
    return render.styled()


def cmd_env_setup(args) -> int:
    starter_env = Path("environment.py")
    dataset = Path("dataset/train.jsonl")
    # An existing environment.py is the authoritative signal for which turn mode this
    # scaffold already uses (the dataset is plain JSONL with no reliable mode marker).
    # Anchor to it so a re-run never leaves a single-turn env beside a multi-turn
    # dataset (or vice versa); the flag/answer only decides the mode when starting fresh.
    existing_multi: bool | None = None
    anchor = "environment.py"
    if starter_env.exists():
        existing_multi = "EnvironmentMultiTurn" in starter_env.read_text(encoding="utf-8")
    elif dataset.exists():
        # No env.py to anchor on, but the starter multi-turn dataset carries a
        # distinctive prompt; use it so we don't drop a single-turn env beside it.
        existing_multi = "secret whole number" in dataset.read_text(encoding="utf-8")
        anchor = "dataset/train.jsonl"

    # Resolve the turn mode. An existing scaffold wins (warn if a flag disagrees); otherwise an
    # explicit --single-turn/--multi-turn flag wins; otherwise ask on a terminal, else single-turn.
    flag_mode = getattr(args, "turn_mode", None)
    if existing_multi is not None:
        if flag_mode is not None and (flag_mode == "multi") != existing_multi:
            have = "multi-turn" if existing_multi else "single-turn"
            want = "multi-turn" if flag_mode == "multi" else "single-turn"
            msg = (
                f"existing {anchor} is {have}; keeping it and ignoring --{want}. "
                f"Delete environment.py and dataset/train.jsonl first to re-scaffold as {want}."
            )
            print(render.warn(msg) if render.styled() else f"warning: {msg}", file=sys.stderr)
        multi_turn = existing_multi
    elif flag_mode is not None:
        multi_turn = flag_mode == "multi"
    elif _setup_interactive(args):
        multi_turn = (
            render.select("How does the model interact with your task?", _TURN_OPTIONS) == "multi"
        )
    else:
        multi_turn = False

    # Resolve reasoning. Like the turn mode, an existing config is authoritative: configs are only
    # written when absent, so applying a reasoning flag to an already-scaffolded project would
    # silently no-op (or write one config with reasoning and leave the other without). Anchor to the
    # existing config's `thinking` state and warn if a flag disagrees; otherwise the flag wins;
    # otherwise ask on a terminal; else off.
    rl = Path("configs/rl.toml")
    sft = Path("configs/sft.toml")
    existing_reasoning: bool | None = None
    for cfg in (rl, sft):
        if cfg.exists():
            existing_reasoning = "thinking = true" in cfg.read_text(encoding="utf-8")
            break
    flag_reason = getattr(args, "reasoning", None)
    if existing_reasoning is not None:
        if flag_reason is not None and flag_reason != existing_reasoning:
            have = "reasoning" if existing_reasoning else "no reasoning"
            want = "reasoning" if flag_reason else "no-reasoning"
            msg = (
                f"existing configs are {have}; keeping them and ignoring --{want}. "
                f"Delete configs/rl.toml and configs/sft.toml first to re-scaffold with --{want}."
            )
            print(render.warn(msg) if render.styled() else f"warning: {msg}", file=sys.stderr)
        reasoning = existing_reasoning
    elif flag_reason is not None:
        reasoning = flag_reason
    elif _setup_interactive(args):
        reasoning = render.select("Train with reasoning (thinking)?", _REASONING_OPTIONS) == "on"
    else:
        reasoning = False

    env_py = _STARTER_ENV_MULTITURN_PY if multi_turn else _STARTER_ENV_PY
    dataset_jsonl = _STARTER_DATASET_MULTITURN_JSONL if multi_turn else _STARTER_DATASET_JSONL
    Path("configs").mkdir(exist_ok=True)
    Path("dataset").mkdir(exist_ok=True)
    if not dataset.exists():
        dataset.write_text(dataset_jsonl)
    if not starter_env.exists():
        starter_env.write_text(env_py)
    env_comment = (
        "# Environment: upload this project folder with\n"
        "# `flash env push --name my-env .`, then paste the returned id below.\n"
        "# If the environment reads secrets with os.environ, list only the env var names here.\n"
        "# Values are read from your shell or .env at submit time and are not stored in the spec.\n"
        "[environment]\n"
        'id = ""\n\n'
        '# secrets = ["SERPAPI_API_KEY"]\n\n'
    )
    # `thinking = true` opts the run into reasoning mode. Reasoning shares the generation budget with
    # the answer, so GRPO also gets a raised completion budget. These strings are empty when reasoning
    # is off, keeping the default scaffold byte-for-byte identical.
    thinking_line = "thinking = true\n" if reasoning else ""
    rl_reasoning_train = (
        "max_completion_tokens = 2048  # reasoning shares this budget with the answer; raised so it isn't truncated\n"
        if reasoning
        else ""
    )
    sft_reasoning_note = (
        "# reasoning is on (thinking = true): each gold `output` must contain a <think>...</think>\n"
        "# block; validate locally with freesolo.datasets.warn_missing_think_tags before a real run.\n"
        if reasoning
        else ""
    )
    if not rl.exists():
        rl.write_text(
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "grpo"\n'
            f"{thinking_line}"
            "\n"
            f"{env_comment}"
            "[train]\n"
            "epochs = 1\n"
            "max_examples = 2  # rows to train on; the starter dataset has 2 (raise as your dataset grows)\n"
            f"{rl_reasoning_train}"
            "lora_rank = 32\n"
            "# Constrain every rollout with guided decoding (a JSON schema is shown; regex and\n"
            "# choice forms also parse) — see `flash train --doc`:\n"
            '# structured_outputs = \'{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}\'\n'
            "# GPU and HF artifacts are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting managed class, and artifacts live in a private environment-scoped repo.\n"
        )
    if not sft.exists():
        sft.write_text(
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "sft"\n'
            f"{thinking_line}"
            "\n"
            f"{env_comment}"
            "[train]\n"
            "epochs = 1\n"
            "max_examples = 2  # rows to train on; the starter dataset has 2 (raise as your dataset grows)\n"
            "lora_rank = 32\n"
            f"{sft_reasoning_note}"
            "# GPU and HF artifacts are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting managed class, and artifacts live in a private environment-scoped repo.\n"
        )
    opd = Path("configs/opd.toml")
    if not opd.exists():
        # opd (on-policy distillation) works in BOTH modes: single-turn distils one sampled completion
        # per prompt; multi-turn rolls out each episode and distils every assistant turn against the
        # transcript so far. The scaffold differs only by a one-line note pointing at the mode.
        opd_multiturn_note = (
            "# NOTE: opd rolls out each episode and distils EVERY assistant turn (conditioned on the\n"
            "# transcript so far) against the managed Fireworks teacher — the multi-turn distillation path.\n\n"
            if multi_turn
            else ""
        )
        opd.write_text(
            f"{opd_multiturn_note}"
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "opd"   # on-policy distillation from a managed Fireworks teacher (default GLM 5.2)\n\n'
            "# Environment: upload this project folder with\n"
            "# `flash env push --name my-env .`, then paste the returned id below.\n"
            "# The teacher and its Fireworks key are platform-managed — nothing to set up or export.\n"
            "[environment]\n"
            'id = ""\n\n'
            "[train]\n"
            "epochs = 1\n"
            "max_examples = 2  # rows to train on; the starter dataset has 2 (raise as your dataset grows)\n"
            "lora_rank = 32\n"
            '# teacher_model = "glm-5.2"   # teacher to distil from: glm-5.2 (default) |\n'
            "#                             # deepseek-v4-pro | kimi-k2.6 (key stays managed)\n"
            "# GPU and HF artifacts are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting managed class, and artifacts live in a private environment-scoped repo.\n"
        )
    training = Path("TRAINING.md")
    if not training.exists():
        # Explicit UTF-8: TRAINING_MD has non-ASCII chars that raise UnicodeEncodeError under a non-UTF-8 locale.
        training.write_text(TRAINING_MD, encoding="utf-8")
    scaffolded = [
        "environment.py",
        "dataset/train.jsonl",
        "configs/sft.toml",
        "configs/rl.toml",
        "configs/opd.toml",
        "TRAINING.md",
    ]
    if render.styled():
        print(render.env_setup(scaffolded))
        return 0
    print(f"ensured {', '.join(scaffolded)}")
    return 0
