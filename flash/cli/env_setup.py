"""Starter environment scaffolds and the `flash env setup` command handler."""

from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

from flash.envs.evaluations import _DEFAULT_EVALUATIONS_PATH

from . import render, traces
from .training_doc import TRAINING_MD

_STARTER_ENV_PY = '''\
"""Starter Freesolo environment.

Edit dataset/train.jsonl and the reward code, then upload with
`flash env push --project PROJECT_UUID --name my-env .`.

A managed run should use the returned [environment] id from
`flash env push --project PROJECT_UUID --name my-env .`.

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

_STARTER_EVALUATIONS_PY = '''\
"""Held-out checks for this environment.

Run them against a deployed model with `flash env eval TARGET .`. This file is
published beside environment.py by `flash env push --project PROJECT_UUID --name my-env .`.
"""

from __future__ import annotations

from flash.envs.evaluations import BaseEvalSuite, EvalCase


class StarterEvaluationSuite(BaseEvalSuite):
    name = "starter"

    def __init__(self, environment=None):
        self.environment = environment

    def cases(self):
        return [EvalCase(id="addition", input="What is 7 + 5?", expected="12")]

    def score(self, case, response):
        if self.environment is None:
            return super().score(case, response)
        example = dict(case.metadata or {})
        example.update(input=case.input, output=case.expected)
        return self.environment.reward(response, example)


def load_evaluations(environment=None):
    return [StarterEvaluationSuite(environment)]
'''

_STARTER_EVALUATIONS_MULTITURN_PY = '''\
"""Held-out checks for this multi-turn environment.

Run them against a deployed model with `flash env eval TARGET .`. This file is
published beside environment.py by `flash env push --project PROJECT_UUID --name my-env .`.

`env eval` sends one prompt and grades one reply, so these cases check the FIRST
assistant action rather than a finished episode: given the opening prompt, does the
model emit the single integer `step_episode` needs to advance the game? That is a real
held-out check of the action format the episode depends on, and it is the honest scope
of a single-shot evaluation. Episode-level reward is what GRPO optimizes through
`score_episode`; validate it offline with `flash env test`.

Do NOT grade these by calling `environment.reward(response, example)`: with no episode
state that call scores an empty transcript, so an unrelated answer can score 1.0 and
publish a green check that measured nothing.
"""

from __future__ import annotations

from flash.envs.evaluations import BaseEvalSuite, EvalCase, EvalResult

LOW, HIGH = 1, 100


class StarterFirstActionSuite(BaseEvalSuite):
    name = "starter-first-action"

    def __init__(self, environment=None):
        self.environment = environment

    def cases(self):
        return [
            EvalCase(
                id="opening-guess",
                input=(
                    f"I picked a secret whole number between {LOW} and {HIGH}.\\n"
                    "Reply with a single integer per turn. I will say 'higher', "
                    "'lower', or 'correct'. You have 5 guesses."
                ),
            )
        ]

    def score(self, case, response):
        # exactly the parse `step_episode` runs on an assistant turn: a reply it cannot
        # read wastes a turn on a re-prompt, so parseability is the thing worth measuring.
        try:
            guess = int(response.strip().split()[0])
        except (ValueError, IndexError):
            return EvalResult(
                case_id=case.id,
                passed=False,
                score=0.0,
                response=response,
                reason="first turn is not a single integer, so step_episode cannot read it",
            )
        in_range = LOW <= guess <= HIGH
        return EvalResult(
            case_id=case.id,
            passed=in_range,
            score=1.0 if in_range else 0.0,
            response=response,
            reason=None if in_range else f"guess {guess} is outside {LOW}-{HIGH}",
        )


def load_evaluations(environment=None):
    return [StarterFirstActionSuite(environment)]
'''


_STARTER_ENV_MULTITURN_PY = '''\
"""Starter Freesolo multi-turn environment.

A multi-turn environment runs a bounded episode: the model produces an assistant
action, `step_episode` advances the world (optionally appending an observation
message), and the loop repeats until `done` or `max_episode_turns`. The finished
transcript is graded by `score_episode`.

Edit dataset/train.jsonl and the episode logic, then upload with
`flash env push --project PROJECT_UUID --name my-env .`.

A managed run should use the returned [environment] id from
`flash env push --project PROJECT_UUID --name my-env .`.

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


_SKIP_TRACES = ""  # sentinel option value: scaffold the starter rows instead of importing


def _require_setup_project(args) -> str:
    """Resolve and validate the one explicit project used by the whole scaffold."""
    from flash.client import ApiError, ClientError, get_project, list_projects
    from flash.client.config import load_credentials

    _api_url, api_key = load_credentials()
    if not api_key:
        raise ClientError(
            "not logged in. Run `flash login` before `flash env setup` so the project can be validated"
        )

    def resolve_project(project_id: str) -> str:
        try:
            return str(get_project(project_id, api_key)["id"])
        except ApiError as exc:
            if exc.status not in {403, 404}:
                raise
            raise ClientError(
                f"project {project_id!r} is not accessible; run `flash projects list` "
                "and pass a project UUID from the current organization"
            ) from exc

    supplied = str(getattr(args, "project", "") or "").strip()
    if supplied:
        return resolve_project(supplied)
    if not _setup_interactive(args):
        raise ClientError(
            "--project PROJECT_UUID is required in noninteractive mode, with redirected stdin, or with --yes"
        )

    projects = list_projects(api_key)
    options = traces.project_options(projects)
    if not options:
        raise ClientError(
            "no Freesolo projects are available for this organization; create one with `flash projects create NAME`"
        )
    selected = render.select_required("Choose the Freesolo project for this environment", options)
    return resolve_project(selected)


def _validate_existing_config_projects(project_id: str) -> None:
    """Reject projectless or conflicting generated configs without rewriting them."""
    from flash.client import ClientError
    from flash.spec import require_project_id

    for path in (Path("configs/sft.toml"), Path("configs/rl.toml"), Path("configs/opd.toml")):
        if not path.exists():
            continue
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ClientError(f"cannot read existing {path}: {exc}") from exc
        try:
            existing = require_project_id(raw.get("project"))
        except (TypeError, ValueError) as exc:
            raise ClientError(f"existing {path} has no valid top-level project UUID") from exc
        if existing != project_id:
            raise ClientError(
                f"existing {path} project {existing!r} does not match selected project {project_id!r}"
            )


def _traces_dataset(args, project_id: str) -> str | None:
    """Optionally export traces from the already-selected project."""
    if not _setup_interactive(args):
        return None
    use_traces = render.select(
        "Use this project's recorded traces as starter rows?",
        [
            ("yes", "use traces", project_id),
            (_SKIP_TRACES, "start from scratch", "a tiny starter dataset you edit"),
        ],
        default=1,
    )
    if use_traces == _SKIP_TRACES:
        return None
    try:
        exported = traces.fetch_records(project_id)
    except Exception as exc:  # trace import is optional after project validation succeeds
        _warn(f"could not export traces from {project_id} ({exc}); using the starter dataset")
        return None
    records = exported.get("records") or []
    if not records:
        _warn(
            f"no exportable traces in {project_id}. Traces need a recorded request and response "
            "to become training rows; using the starter dataset"
        )
        return None
    skipped = int(exported.get("skipped") or 0)
    detail = f" ({skipped} skipped)" if skipped else ""
    print(render.ok(f"exported {len(records)} rows from your traces{detail}"))
    return traces.records_to_jsonl(records)


def _warn(msg: str) -> None:
    print(render.warn(msg) if render.styled() else f"warning: {msg}", file=sys.stderr)


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
    project_id = _require_setup_project(args)
    _validate_existing_config_projects(project_id)
    starter_env = Path("environment.py")
    starter_evaluations = Path(_DEFAULT_EVALUATIONS_PATH)
    dataset = Path("dataset/train.jsonl")
    # trace import is optional and can only read the selected project. an existing dataset is never
    # overwritten, so importing into it would be a silently discarded download.
    traces_jsonl = None if dataset.exists() else _traces_dataset(args, project_id)
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

    env_py = (_STARTER_ENV_MULTITURN_PY if multi_turn else _STARTER_ENV_PY).replace(
        "PROJECT_UUID", project_id
    )
    dataset_jsonl = traces_jsonl or (
        _STARTER_DATASET_MULTITURN_JSONL if multi_turn else _STARTER_DATASET_JSONL
    )
    Path("configs").mkdir(exist_ok=True)
    Path("dataset").mkdir(exist_ok=True)
    if not dataset.exists():
        dataset.write_text(dataset_jsonl)
    # `max_examples` caps the rows a run trains on, so it tracks the dataset actually written:
    # the 2 starter rows, or every row exported from the user's traces.
    example_rows = dataset_jsonl.count("\n")
    max_examples_line = (
        f"max_examples = {example_rows}  # rows to train on; "
        f"{'exported from your traces' if traces_jsonl else 'the starter dataset has 2'}"
        f"{'' if traces_jsonl else ' (raise as your dataset grows)'}\n"
    )
    wrote_starter_env = not starter_env.exists()
    if wrote_starter_env:
        starter_env.write_text(env_py)
    # only alongside the starter environment it grades. rerun in a directory with a custom
    # environment.py, this used to drop in the arithmetic suite anyway, which then called that
    # env's reward with an unrelated `7 + 5` example -- a published check measuring nothing
    # (codex[bot]). the multi-turn scaffold gets its own suite: a single-shot eval cannot grade
    # a finished episode, so it checks the first action's format instead.
    if wrote_starter_env and not starter_evaluations.exists():
        evaluations_py = (
            _STARTER_EVALUATIONS_MULTITURN_PY if multi_turn else _STARTER_EVALUATIONS_PY
        )
        starter_evaluations.write_text(evaluations_py.replace("PROJECT_UUID", project_id))
    project_line = f"project = {json.dumps(project_id)}\n"
    env_comment = (
        "# Environment: upload this project folder with\n"
        f"# `flash env push --project {project_id} --name my-env .`, then paste the returned id below.\n"
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
            f"{project_line}"
            'algorithm = "grpo"\n'
            f"{thinking_line}"
            "\n"
            f"{env_comment}"
            "[train]\n"
            "epochs = 1\n"
            f"{max_examples_line}"
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
            f"{project_line}"
            'algorithm = "sft"\n'
            f"{thinking_line}"
            "\n"
            f"{env_comment}"
            "[train]\n"
            "epochs = 1\n"
            f"{max_examples_line}"
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
            f"{project_line}"
            'algorithm = "opd"   # on-policy distillation from a managed Fireworks teacher (default GLM 5.2)\n\n'
            "# Environment: upload this project folder with\n"
            f"# `flash env push --project {project_id} --name my-env .`, then paste the returned id below.\n"
            "# The teacher and its Fireworks key are platform-managed — nothing to set up or export.\n"
            "[environment]\n"
            'id = ""\n\n'
            "[train]\n"
            "epochs = 1\n"
            f"{max_examples_line}"
            "lora_rank = 32\n"
            '# teacher_model = "glm-5.2"   # teacher to distil from: glm-5.2 (default) |\n'
            "#                             # deepseek-v4-pro | kimi-k2.6 (key stays managed)\n"
            "# GPU and HF artifacts are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting managed class, and artifacts live in a private environment-scoped repo.\n"
        )
    training = Path("TRAINING.md")
    if not training.exists():
        # Explicit UTF-8: TRAINING_MD has non-ASCII chars that raise UnicodeEncodeError under a non-UTF-8 locale.
        training.write_text(
            TRAINING_MD.replace("PROJECT_UUID", project_id).replace("<project-uuid>", project_id),
            encoding="utf-8",
        )
    scaffolded = [
        "environment.py",
        _DEFAULT_EVALUATIONS_PATH,
        "dataset/train.jsonl",
        "configs/sft.toml",
        "configs/rl.toml",
        "configs/opd.toml",
        "TRAINING.md",
    ]
    if render.styled():
        print(render.env_setup(scaffolded, project_id))
        return 0
    print(f"ensured {', '.join(scaffolded)}")
    print(f"next: flash env push --project {project_id} --name my-env .")
    return 0
