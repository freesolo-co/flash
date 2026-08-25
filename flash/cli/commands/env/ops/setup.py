"""Starter environment scaffolds and the `flash env setup` command handler."""

from __future__ import annotations

import json
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path

from flash._internal.channel import CLI_NAME
from flash.cli.commands.env.ops.retained import (
    _warn_if_environment_form_disagrees,
    _warn_if_retained_starter_files_describe_another_plane,
)
from flash.cli.commands.env.ops.setup_wandb import _wandb_block, _wandb_project
from flash.cli.commands.ops import traces
from flash.cli.scaffold import TRAINING_MD
from flash.cli.ui import render
from flash.envs.meta.evaluations import _DEFAULT_EVALUATIONS_PATH

_STARTER_ENV_PY = '''\
"""Starter Freesolo environment.

Edit dataset/train.jsonl and the reward code, then ENVIRONMENT_GUIDANCE

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
    # Each dataset row's `output` does double duty: the scorer's expected answer, AND the gold
    # assistant turn SFT trains on. With no `sft_completion` hook defined, `flash env test` and
    # SFT both replay `output` verbatim as the model's response. So write it as the full text
    # the model should emit -- if `score_response` requires a wrapper (\\boxed{}, a JSON object,
    # a tag), the wrapper belongs in `output` too, or SFT trains the model to emit exactly what
    # the reward punishes. Per-row state the scorer needs but the model must not see goes in
    # `metadata`, never in `output`.
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

EVALUATIONS_GUIDANCE
"""

from __future__ import annotations

from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase, EvalResult


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
        # pass/fail comes from grade(), the environment's own success decision, not from the
        # reward being positive. a shaped reward pays partial credit for a wrong answer -- the
        # multi-turn starter scores a near miss as closeness * 0.5 with success=False -- and
        # returning the bare float would report every such case as passing.
        #
        # this scores the response twice. if your scorer calls a judge or any other paid
        # service, score once and build the EvalResult from that single result instead.
        return EvalResult(
            case_id=case.id or "case",
            passed=self.environment.grade(response, example),
            score=float(self.environment.reward(response, example)),
            response=response,
        )


def load_evaluations(environment=None):
    return [StarterEvaluationSuite(environment)]
'''

_STARTER_EVALUATIONS_MULTITURN_PY = '''\
"""Held-out checks for this multi-turn environment.

EVALUATIONS_GUIDANCE

By default `env eval` sends one prompt and grades one reply, which is what this suite
wants: these cases check the FIRST assistant action rather than a finished episode.
Given the opening prompt, does the model emit the single integer `step_episode` needs
to advance the game? That is a real held-out check of the action format the episode
depends on, and it is the honest scope of a single-shot evaluation.

To grade a finished transcript instead, set `grades_episodes = True` on the suite
and define `score(self, case, response, state)`. `env eval` then plays each case out
against the deployed model -- generating, stepping the environment, and repeating to
`max_episode_turns` -- and passes the resulting transcript in `state`; without that
argument the scorer receives only the episode's final response text. Such a suite
needs cases the environment can actually advance (the cases below carry no `output`,
so `step_episode` has no secret to compare against), and each case then costs one
generation per turn rather than one in total.

Do NOT grade these by calling `environment.reward(response, example)`: with no episode
state that call scores an empty transcript, so an unrelated answer can score 1.0 and
publish a green check that measured nothing.
"""

from __future__ import annotations

from flash.envs.meta.evaluations import BaseEvalSuite, EvalCase, EvalResult

LOW, HIGH = 1, 100


class StarterFirstActionSuite(BaseEvalSuite):
    name = "starter-first-action"

    def __init__(self, environment=None):
        self.environment = environment

    def cases(self):
        # `input` is a dataset row's input, not a finished prompt. `env eval` builds the
        # request through `environment.prompt_messages()`, which runs `start_episode` and
        # appends the reply instructions -- exactly as training does. Spelling that block
        # out here too would send it twice and evaluate a prompt training never used.
        return [
            EvalCase(
                id="opening-guess",
                input=f"I picked a secret whole number between {LOW} and {HIGH}.",
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

Edit dataset/train.jsonl and the episode logic, then ENVIRONMENT_GUIDANCE

This starter implements a tiny "guess the secret number" game so you can see the
episode hooks wired end-to-end. Replace it with your real task before a real run.

All three algorithms train off this file:
- GRPO (configs/rl.toml) rolls out full episodes and optimizes `score_episode`.
- SFT (configs/sft.toml) learns the gold trajectory. Provide it per row as
  `output = {"messages": [...]}` (a full assistant/tool trajectory) or a scalar
  `output` for a single gold assistant turn.
- OPD (configs/opd.toml) rolls out each episode and distils EVERY assistant turn against
  the managed Parasail teacher (GLM 5.2 by default; pick another with [train] teacher_model),
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
        # `messages` already ends with this action: messages[-1]["content"] is
        # `assistant_response`. if you rebuild state by replaying the transcript, replay
        # `messages[:-1]` and apply `assistant_response` once, or you apply it twice.
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


def _require_setup_project(args, *, api_url: str | None, api_key: str) -> Mapping[str, object]:
    """Resolve and validate the one explicit project used by the whole scaffold."""
    from flash.client import ClientError, list_projects, resolve_project
    from flash.client.http import has_freesolo_backend

    supplied = str(getattr(args, "project", "") or "").strip()
    if supplied:
        return resolve_project(supplied, api_key, api_url)
    if not _setup_interactive(args):
        raise ClientError(
            "--project PROJECT_UUID is required in noninteractive mode, with redirected stdin, or with --yes"
        )
    # a self-hosted plane pointed at an operator-run Freesolo-compatible backend DOES have a
    # directory to list, so classify on the backend rather than on the plane url alone -- the same
    # decision `flash projects list` makes. only a genuinely backend-less plane refuses here.
    if api_url is not None and not has_freesolo_backend(api_url):
        raise ClientError(
            "--project PROJECT_UUID is required for interactive setup against a plane with no "
            "Freesolo backend, which has no project directory"
        )

    projects = list_projects(api_key)
    options = traces.project_options(projects)
    if not options:
        raise ClientError(
            f"no Freesolo projects are available for this organization; create one with "
            f"`{CLI_NAME} projects create NAME`"
        )
    selected = render.select_required("Choose the Freesolo project for this environment", options)
    return resolve_project(selected, api_key, api_url)


# Guidance the starter .py docstrings carry, per plane kind. The templates hold placeholders
# rather than one plane's wording plus a rewrite pass: matching prose back out after rendering
# breaks silently the moment a template is reworded.
# `CLI_NAME`, not a literal `flash`: the executable is `flash-cli` (or `python -m flash.cli`) when
# `flash` on PATH belongs to RunPod, and the terminal-facing output below already renders it that
# way. A generated file that says `flash env push` sends the user to the shadowing binary, which
# exits 0 without publishing anything -- so the instructions look followed and nothing happened.
_HOSTED_GUIDANCE = {
    "ENVIRONMENT_GUIDANCE": (
        "upload with\n"
        f"`{CLI_NAME} env push --project PROJECT_UUID --name my-env .`.\n"
        "\n"
        "A managed run should use the returned [environment] id from\n"
        f"`{CLI_NAME} env push --project PROJECT_UUID --name my-env .`."
    ),
    "EVALUATIONS_GUIDANCE": (
        "Publish this file beside environment.py with\n"
        f"`{CLI_NAME} env push --project PROJECT_UUID --name my-env .`, then run the suites\n"
        f"against a model trained on it with `{CLI_NAME} env eval TARGET`. The run names the\n"
        "published environment, so `env eval` takes no local path."
    ),
}

_SELF_HOSTED_GUIDANCE = {
    "ENVIRONMENT_GUIDANCE": (
        "commit it to a git repo your plane can read.\n"
        "\n"
        "A managed run names that repo in [environment] id, as\n"
        "`github:OWNER/REPO@REF:environment.py` -- this plane is self-hosted, so publishing\n"
        "to Freesolo's managed environment hub does not apply.\n"
        "\n"
        "REF is your repo's actual default branch -- check it rather than assuming `main`, since\n"
        "it depends on how the repo was created. A ref that does not exist fails with GitHub's\n"
        '"No commit found for SHA: <ref>".'
    ),
    "EVALUATIONS_GUIDANCE": (
        "Keep this file beside environment.py in the git repo named by [environment] id."
    ),
}


def _render_starter(template: str, project_id: str, *, can_publish: bool) -> str:
    """Fill a starter .py template's placeholders for this plane, then its project uuid.

    Guidance first, because the hosted wording itself contains PROJECT_UUID -- substituting the
    uuid first would leave the placeholder text unrendered in the guidance that replaces it.
    """
    for placeholder, text in (_HOSTED_GUIDANCE if can_publish else _SELF_HOSTED_GUIDANCE).items():
        template = template.replace(placeholder, text)
    return template.replace("PROJECT_UUID", project_id)


def _plane_can_publish_environments(api_url: str | None) -> bool:
    """Whether `flash env push` can actually publish for the logged-in plane.

    Publishing uploads to the plane, which commits into the managed environment hub
    (`freesolo-co/environment-hub`, hardcoded in flash/server/domain/registry/envs.py). A self-hosted
    operator cannot write there whatever GITHUB_TOKEN they set, so scaffolding `flash env push`
    at them leaves an unusable empty `id` that fails config validation.

    Classified on the PLANE, not on `has_freesolo_backend`: a self-hosted plane pointed at an
    operator-run Freesolo-compatible backend has a project directory (which is what that helper
    answers) but still cannot publish to the hub.
    """
    from flash.serve.contract.urls import is_freesolo_hosted_url

    # Unset means the built-in default, which is the managed plane.
    return api_url is None or is_freesolo_hosted_url(api_url)


def _environment_comment(project_id: str, *, can_publish: bool, extra: str = "") -> str:
    """The `[environment]` block for a generated config."""
    if can_publish:
        head = (
            "# Environment: upload this project folder with\n"
            f"# `{CLI_NAME} env push --project {project_id} --name my-env .`, then paste the returned id below.\n"
        )
        placeholder = 'id = ""\n\n'
    else:
        head = (
            f"# Environment: this plane is self-hosted, so `{CLI_NAME} env push` does not apply -- it\n"
            "# publishes to Freesolo's managed environment hub. Name a git repo instead:\n"
            "#   github:OWNER/REPO@REF:PATH   (PATH is the file, or the directory holding environment.py)\n"
            "# REF is your repo's actual default branch -- check it rather than assuming `main`,\n"
            "# since it depends on how the repo was created. A ref that does not exist fails with\n"
            "# GitHub's 'No commit found for SHA: <ref>'.\n"
            "# Push this folder to a repo your plane can read, then fill in the id below.\n"
            "# A github: id needs a plane running with FLASH_STANDALONE=1; an identity-backed plane\n"
            "# accepts managed hub ids only. Setup classifies on the API URL and cannot see that\n"
            "# server-side setting, so if submit returns a 400 naming this id, that is the cause.\n"
        )
        # REF, not `main`: this line is a placeholder to be edited, and every other token in it
        # (OWNER, REPO) is obviously one. `main` looks like a value that is already correct, so it
        # survives the edit -- and on a repo whose default branch is anything else, it is silently
        # wrong, surfacing much later as a pinning error at submit.
        placeholder = 'id = "github:OWNER/REPO@REF:environment.py"\n\n'
    return f"{head}{extra}[environment]\n{placeholder}"


def _validate_existing_config_projects(project_id: str) -> None:
    """Reject projectless or conflicting generated configs without rewriting them."""
    from flash.client import ClientError
    from flash.core.spec import require_project_id

    for path in (Path("configs/sft.toml"), Path("configs/rl.toml"), Path("configs/opd.toml")):
        if not path.exists():
            continue
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ClientError(f"cannot read existing {path}: {exc}") from exc
        try:
            existing = require_project_id(raw.get("project"))
        except (TypeError, ValueError) as exc:
            raise ClientError(f"existing {path} has no valid top-level project UUID") from exc
        if existing != project_id:
            raise ClientError(
                f"existing {path} project {existing!r} does not match selected project {project_id!r}"
            )


def _existing_reasoning(configs: tuple[Path, ...]) -> bool | None:
    """The `thinking` state the generated configs already agree on, or None if none exist.

    Read every existing config so pre-#824 scaffolds with stale OPD settings are detected. Missing
    configs are not disagreements because they are about to be written.
    """
    from flash.client import ClientError

    found: dict[Path, bool] = {}
    for cfg in configs:
        if not cfg.exists():
            continue
        try:
            raw = tomllib.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ClientError(f"cannot read existing {cfg}: {exc}") from exc
        found[cfg] = raw.get("thinking") is True
    if len(set(found.values())) > 1:
        # no anchor exists: the configs disagree, so there is no "existing state" to keep. picking a
        # side would rewrite nothing (configs are only written when absent) and leave the project
        # training one algorithm with reasoning and another without. refusing is the only outcome
        # that cannot end in a silent cross-algorithm mismatch.
        on = ", ".join(str(cfg) for cfg, has in found.items() if has)
        off = ", ".join(str(cfg) for cfg, has in found.items() if not has)
        raise ClientError(
            f"existing configs disagree about reasoning: {on} set `thinking = true`, "
            f"{off} do not. Delete {', '.join(str(cfg) for cfg in found)} and re-run "
            f"`{CLI_NAME} env setup` with --reasoning or --no-reasoning to scaffold them together."
        )
    return next(iter(found.values()), None)


def _traces_dataset(args, project_id: str, api_key: str) -> str | None:
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
        exported = traces.fetch_records(project_id, api_key)
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

    Fall back to defaults (no prompt) when --yes is passed, and otherwise defer to the shared
    can-a-human-answer check. --yes lives here rather than in `render` because it is this command's
    own flag; the terminal/CI half is identical for every prompting command and is defined once."""
    if getattr(args, "yes", False):
        return False
    return render.can_prompt()


def _marker_present(path: Path, marker: str) -> bool | None:
    """Whether `path` carries `marker`, or None when the file cannot be read as UTF-8.

    Probing for a marker is a best-effort read of a file the operator owns: `# -*- coding:
    latin-1 -*-` is valid Python, and reading strictly here aborted `env setup` before it did
    anything. But "cannot read" is NOT "marker absent": collapsing the two would classify an
    undecodable multi-turn environment.py as single-turn and then override an explicit
    --multi-turn with that guess. Three states, so an unreadable anchor can defer to the flag.
    """
    try:
        return marker in path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _resolve_turn_mode(args, starter_env: Path, dataset: Path) -> tuple[bool, bool]:
    """resolve the turn-mode reconciliation phase."""
    # An existing environment.py is the authoritative signal for which turn mode this
    # scaffold already uses (the dataset is plain JSONL with no reliable mode marker).
    # Anchor to it so a re-run never leaves a single-turn env beside a multi-turn
    # dataset (or vice versa); the flag/answer only decides the mode when starting fresh.
    existing_multi: bool | None = None
    anchor = "environment.py"
    starter_env_exists = starter_env.exists()
    if starter_env_exists:
        # None (unreadable) stays None: an anchor we cannot read decides nothing, so the flag or
        # the prompt below resolves the mode instead of an unreadable file silently forcing one.
        existing_multi = _marker_present(starter_env, "EnvironmentMultiTurn")
    elif dataset.exists():
        # No env.py to anchor on, but the starter multi-turn dataset carries a
        # distinctive prompt; use it so we don't drop a single-turn env beside it.
        existing_multi = _marker_present(dataset, "secret whole number")
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
    return multi_turn, starter_env_exists


def _resolve_reasoning_mode(args, reasoning_configs: tuple[Path, ...]) -> bool:
    """resolve the reasoning-mode reconciliation phase."""
    # Resolve reasoning. Like the turn mode, an existing config is authoritative: configs are only
    # written when absent, so applying a reasoning flag to an already-scaffolded project would
    # silently no-op (or write one config with reasoning and leave the other without). Anchor to the
    # existing config's `thinking` state and warn if a flag disagrees; otherwise the flag wins;
    # otherwise ask on a terminal; else off.
    existing_reasoning = _existing_reasoning(reasoning_configs)
    flag_reason = getattr(args, "reasoning", None)
    if existing_reasoning is not None:
        if flag_reason is not None and flag_reason != existing_reasoning:
            have = "reasoning" if existing_reasoning else "no reasoning"
            want = "reasoning" if flag_reason else "no-reasoning"
            # name every config that exists, so deleting exactly what this lists cannot leave one behind.
            stale = ", ".join(str(c) for c in reasoning_configs if c.exists())
            msg = (
                f"existing configs are {have}; keeping them and ignoring --{want}. "
                f"Delete {stale} first to re-scaffold with --{want}."
            )
            print(render.warn(msg) if render.styled() else f"warning: {msg}", file=sys.stderr)
        reasoning = existing_reasoning
    elif flag_reason is not None:
        reasoning = flag_reason
    elif _setup_interactive(args):
        reasoning = render.select("Train with reasoning (thinking)?", _REASONING_OPTIONS) == "on"
    else:
        reasoning = False
    return reasoning


def _write_rl_config(
    rl: Path,
    project_line: str,
    thinking_line: str,
    wandb_block: str,
    env_comment: str,
    max_examples_line: str,
    rl_reasoning_train: str,
) -> None:
    """write the rl config-file phase."""
    if not rl.exists():
        rl.write_text(
            'model = "Qwen/Qwen3.5-9B"\n'
            f"{project_line}"
            'algorithm = "grpo"\n'
            f"{thinking_line}"
            "\n"
            f"{wandb_block}"
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


def _write_sft_config(
    sft: Path,
    project_line: str,
    thinking_line: str,
    wandb_block: str,
    env_comment: str,
    max_examples_line: str,
    sft_reasoning_note: str,
) -> None:
    """write the sft config-file phase."""
    if not sft.exists():
        sft.write_text(
            'model = "Qwen/Qwen3.5-9B"\n'
            f"{project_line}"
            'algorithm = "sft"\n'
            f"{thinking_line}"
            "\n"
            f"{wandb_block}"
            f"{env_comment}"
            "[train]\n"
            "epochs = 1\n"
            f"{max_examples_line}"
            "lora_rank = 32\n"
            f"{sft_reasoning_note}"
            "# GPU and HF artifacts are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting managed class, and artifacts live in a private environment-scoped repo.\n"
        )


_OPD_MANAGED_TEACHER_NOTE = (
    "# the teacher and its parasail key are platform-managed; nothing to set up or export.\n"
)

# Only the managed plane manages the key. `require_teacher_broker_configuration` reads
# PARASAIL_API_KEY and FLASH_PUBLIC_URL from the CONTROL PLANE's environment, so on a self-hosted
# plane the operator sets both -- and the failure lands at submit, after the scaffold has already
# claimed there was nothing to do.
_OPD_SELF_HOSTED_TEACHER_NOTE = (
    "# opd uses a managed parasail teacher, which this plane brokers itself: set PARASAIL_API_KEY\n"
    "# and FLASH_PUBLIC_URL in the CONTROL PLANE's environment (not your shell) before submitting.\n"
    "# FLASH_PUBLIC_URL must be an origin the rented worker can reach, not localhost or a tunnel.\n"
)


def _write_opd_config(
    opd: Path,
    multi_turn: bool,
    project_line: str,
    project_id: str,
    thinking_line: str,
    wandb_block: str,
    max_examples_line: str,
    *,
    can_publish: bool = True,
) -> None:
    """write the opd config-file phase."""
    if not opd.exists():
        # opd (on-policy distillation) works in BOTH modes: single-turn distils one sampled completion
        # per prompt; multi-turn rolls out each episode and distils every assistant turn against the
        # transcript so far. The scaffold differs only by a one-line note pointing at the mode.
        opd_multiturn_note = (
            "# note: opd rolls out each episode and distils every assistant turn (conditioned on the\n"
            "# transcript so far) against the managed parasail teacher - the multi-turn distillation path.\n\n"
            if multi_turn
            else ""
        )
        opd.write_text(
            f"{opd_multiturn_note}"
            'model = "Qwen/Qwen3.5-9B"\n'
            f"{project_line}"
            'algorithm = "opd"   # on-policy distillation from a managed parasail teacher (default glm 5.2)\n'
            f"{thinking_line}"
            "\n"
            f"{wandb_block}"
            + _environment_comment(
                project_id,
                can_publish=can_publish,
                extra=(_OPD_MANAGED_TEACHER_NOTE if can_publish else _OPD_SELF_HOSTED_TEACHER_NOTE),
            )
            + "[train]\n"
            "epochs = 1\n"
            f"{max_examples_line}"
            "lora_rank = 32\n"
            '# teacher_model = "glm-5.2"   # teacher: glm-5.2 (default) | kimi-k3 |\n'
            "#                             # qwen3.5-397b-a17b | deepseek-v4-pro |\n"
            "#                             # qwen3-vl-235b\n"
            "#                             # (key stays managed)\n"
            "# GPU and HF artifacts are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting managed class, and artifacts live in a private environment-scoped repo.\n"
        )


def _write_training_guide(training: Path, project_id: str) -> None:
    """write the training guide phase."""
    if not training.exists():
        # Explicit UTF-8: TRAINING_MD has non-ASCII chars that raise UnicodeEncodeError under a non-UTF-8 locale.
        training.write_text(
            TRAINING_MD.replace("PROJECT_UUID", project_id).replace("<project-uuid>", project_id),
            encoding="utf-8",
        )


def _resolve_setup_identity(args) -> tuple[str, str, str, str, str]:
    """Resolve the credential, project, and folder identity used by every setup phase."""
    from flash.client import ClientError
    from flash.client.config import load_credentials

    api_url, api_key = load_credentials()
    if not api_key:
        raise ClientError(
            f"not logged in. Run `{CLI_NAME} login` before `{CLI_NAME} env setup` "
            "so the project can be validated"
        )
    project = _require_setup_project(args, api_url=api_url, api_key=api_key)
    project_id = str(project["id"])
    folder_name = Path.cwd().name
    if not folder_name.strip():
        raise ClientError("the current environment folder name must be nonblank")
    return api_url, api_key, project_id, _wandb_project(project), folder_name


def cmd_env_setup(args) -> int:
    api_url, api_key, project_id, project_name, folder_name = _resolve_setup_identity(args)
    _validate_existing_config_projects(project_id)
    starter_env = Path("environment.py")
    starter_evaluations = Path(_DEFAULT_EVALUATIONS_PATH)
    dataset = Path("dataset/train.jsonl")
    # trace import is optional and can only read the selected project. an existing dataset is never
    # overwritten, so importing into it would be a silently discarded download.
    traces_jsonl = None if dataset.exists() else _traces_dataset(args, project_id, api_key)
    multi_turn, starter_env_exists = _resolve_turn_mode(args, starter_env, dataset)

    rl = Path("configs/rl.toml")
    sft = Path("configs/sft.toml")
    opd = Path("configs/opd.toml")
    # All THREE configs persist `thinking`, so all three must anchor it and all three must be named in
    # the deletion guidance. Anchoring on a subset lets a user follow this warning literally, delete
    # exactly what it names, and end up with the deleted configs rewritten one way and the unnamed one
    # still holding the old setting -- the same silent cross-algorithm mismatch, just relocated.
    reasoning_configs = (rl, opd, sft)
    reasoning = _resolve_reasoning_mode(args, reasoning_configs)

    can_publish = _plane_can_publish_environments(api_url)
    _warn_if_environment_form_disagrees(reasoning_configs, can_publish=can_publish, warn=_warn)
    env_py = _render_starter(
        _STARTER_ENV_MULTITURN_PY if multi_turn else _STARTER_ENV_PY,
        project_id,
        can_publish=can_publish,
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
    _warn_if_retained_starter_files_describe_another_plane(
        (starter_env, starter_evaluations),
        can_publish=can_publish,
        warn=_warn,
    )
    if not starter_env_exists:
        starter_env.write_text(env_py)
        # install starter suites only for the starter environment created in this run. multi-turn
        # uses a first-action format check because single-shot eval cannot grade a completed
        # episode.
        if not starter_evaluations.exists():
            starter_evaluations.write_text(
                _render_starter(
                    _STARTER_EVALUATIONS_MULTITURN_PY if multi_turn else _STARTER_EVALUATIONS_PY,
                    project_id,
                    can_publish=can_publish,
                )
            )
    project_line = f"project = {json.dumps(project_id)}\n"
    env_comment = (
        _environment_comment(
            project_id,
            can_publish=can_publish,
            extra=(
                "# If the environment reads secrets with os.environ, list only the env var names here.\n"
                "# Values are read from your shell or .env at submit time and are not stored in the spec.\n"
            ),
        )
        + '# secrets = ["SERPAPI_API_KEY"]\n\n'
    )
    # `thinking=true` is algorithm-agnostic. only GRPO needs a generated completion increase;
    # OPD raises its own via `opd_completion_len`, and SFT does not generate. empty strings preserve
    # the non-reasoning scaffold.
    thinking_line = "thinking = true\n" if reasoning else ""
    rl_reasoning_train = (
        "max_completion_tokens = 2048  # per model turn; reasoning shares it with the answer\n"
        if reasoning
        else ""
    )
    sft_reasoning_note = (
        "# reasoning is on (thinking = true): each gold `output` must contain a <think>...</think>\n"
        "# block; validate locally with freesolo.datasets.warn_missing_think_tags before a real run.\n"
        if reasoning
        else ""
    )
    _write_rl_config(
        rl,
        project_line,
        thinking_line,
        _wandb_block(project_name, f"{folder_name}-grpo"),
        env_comment,
        max_examples_line,
        rl_reasoning_train,
    )
    _write_sft_config(
        sft,
        project_line,
        thinking_line,
        _wandb_block(project_name, f"{folder_name}-sft"),
        env_comment,
        max_examples_line,
        sft_reasoning_note,
    )
    opd = Path("configs/opd.toml")
    _write_opd_config(
        opd,
        multi_turn,
        project_line,
        project_id,
        thinking_line,
        _wandb_block(project_name, f"{folder_name}-opd"),
        max_examples_line,
        can_publish=can_publish,
    )
    training = Path("TRAINING.md")
    _write_training_guide(training, project_id)
    scaffolded = [
        "environment.py",
        *([_DEFAULT_EVALUATIONS_PATH] if starter_evaluations.exists() else []),
        "dataset/train.jsonl",
        "configs/sft.toml",
        "configs/rl.toml",
        "configs/opd.toml",
        "TRAINING.md",
    ]
    if render.styled():
        print(render.env_setup(scaffolded, project_id, can_publish=can_publish))
        return 0
    print(f"ensured {', '.join(scaffolded)}")
    if can_publish:
        print(f"next: {CLI_NAME} env push --project {project_id} --name my-env .")
    else:
        # `env push` targets the managed hub, which a self-hosted plane cannot write to. Same
        # wording as the styled path in `render.env_setup`, which is what a TTY actually shows --
        # including the standalone caveat, so the two paths cannot drift apart again.
        print(
            "next: push this folder to a git repo, then set [environment] id = "
            "github:OWNER/REPO@REF:environment.py (REF is the repo's default branch, main or "
            "master) (needs FLASH_STANDALONE=1 on the plane)"
        )
    return 0
