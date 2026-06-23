"""CLI command handlers for the managed Flash service.

Every run-lifecycle command is a thin HTTP call to the Flash control plane —
users authenticate with their freesolo API key (`flash login` verifies it against
the freesolo backend), never with provider credentials. Config parsing/validation
and `--dry-run` stay fully local.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from flash import __version__
from flash._logging import get_logger
from flash.catalog import public_model_rows
from flash.client import (
    ApiClient,
    ClientError,
    client_from_config,
    save_credentials,
    verify_freesolo_key,
)
from flash.client.config import load_credentials
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.cost.spec import runconfig_from_spec
from flash.runner import TERMINAL_STATES, new_run_id
from flash.schema import ConfigError, spec_from_file

from . import render

logger = get_logger("flash.cli.main")


# Exceptions that represent expected user/config errors: report them as a clean one-line
# message instead of a Python traceback (use --debug to see the full trace).
_USER_ERRORS = (
    ConfigError,
    ClientError,
    FileNotFoundError,
    ValueError,
)

# Run states after which nothing more will happen (polling can stop).
_CLI_DONE_STATES = TERMINAL_STATES | {"deployed"}
_OK_STATES = {"done", "dry_run", "deployed"}
_SPINNER_FRAMES = "|/-\\"
_SPINNER_TICK_SECONDS = 0.1


class _LogFollowSpinner:
    def __init__(self, run_id: str):
        self._run_id = run_id
        self._frame = 0
        self._last_len = 0
        self._active = False
        self._enabled = sys.stderr.isatty()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def render(self, state: str) -> None:
        if not self._enabled:
            return
        frame = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
        self._frame += 1
        message = f"{frame} following logs for {self._run_id} ({state})"
        padding = " " * max(0, self._last_len - len(message))
        sys.stderr.write(f"\r{message}{padding}")
        sys.stderr.flush()
        self._last_len = len(message)
        self._active = True

    def clear(self) -> None:
        if not (self._enabled and self._active):
            return
        sys.stderr.write(f"\r{' ' * self._last_len}\r")
        sys.stderr.flush()
        self._active = False


def _sleep_with_spinner(interval: float, spinner: _LogFollowSpinner, state: str) -> None:
    if interval <= 0:
        return
    if not spinner.enabled:
        time.sleep(interval)
        return
    ticks = max(1, int(interval / _SPINNER_TICK_SECONDS))
    sleep_for = interval / ticks
    for _ in range(ticks):
        spinner.render(state)
        time.sleep(sleep_for)


def cmd_version(args) -> int:
    print(f"flash {__version__}")
    return 0


def cmd_login(args) -> int:
    # Login is handled by the freesolo backend (not the flash control plane): the user
    # supplies the freesolo API key they created at freesolo.co/sign-in, and we verify it against
    # freesolo before storing it. The same key authenticates flash's control plane.
    try:
        env_api_key = os.environ.get("FREESOLO_API_KEY")
        api_key = args.api_key or env_api_key
        if not api_key:
            raise ClientError(
                "no API key provided: pass `--api-key <key>` or set FREESOLO_API_KEY. "
                "Create or copy a key at https://freesolo.co/sign-in."
            )
        verify_freesolo_key(api_key, base_url=getattr(args, "freesolo_url", None))
    except ClientError as exc:
        # Login failed (no key, a rejected key, or an unreachable backend): say so plainly
        # and point the user back at `flash login` to try again. `--debug` still surfaces
        # the full traceback via the top-level handler.
        if getattr(args, "debug", False):
            raise
        print(render.login_failed(str(exc)), file=sys.stderr)
        return 1
    api_url = args.api_url or load_credentials()[0]
    # save_credentials clears the stored url when it's the default, so logging into the
    # default plane also drops a stale custom url from a previous custom-URL login.
    _ = save_credentials(api_key, api_url=api_url)
    if args.api_key and env_api_key and env_api_key != args.api_key:
        print(
            "warning: FREESOLO_API_KEY is set and will override this saved login for future "
            "commands; unset FREESOLO_API_KEY to use the saved key.",
            file=sys.stderr,
        )
    # Show who they are right away (the same identity `flash whoami` prints) so they don't
    # have to run a second command. Never echo the key itself. The identity lookup is
    # best-effort: the key is already verified and stored, so a momentary control-plane
    # hiccup must not turn a successful login into a failure.
    print(render.login_ok(_identity_or_none(api_key, api_url)))
    return 0


# A control-plane hiccup must not make a successful login appear to hang while we fetch a
# nonessential card, so the best-effort identity lookup uses a short timeout.
_IDENTITY_LOOKUP_TIMEOUT_S = 5.0


def _identity_or_none(api_key: str, api_url: str) -> dict | None:
    # Use the key/url we just verified and stored, not `client_from_config()`: an ambient
    # FREESOLO_API_KEY would otherwise win over the file and render the wrong identity.
    try:
        return ApiClient(api_url, api_key, timeout=_IDENTITY_LOOKUP_TIMEOUT_S).me()
    except (ClientError, OSError, ValueError):
        return None


def cmd_whoami(args) -> int:
    print(render.whoami(client_from_config().me()))
    return 0


_STARTER_ENV_PY = '''\
"""Starter Freesolo environment.

Edit the dataset and reward code, then upload with
`flash env push --name my-env .`.

A managed run should use the returned [environment] id from
`flash env push --name my-env .`.

Keep real SFT/RL datasets in Freesolo or Hugging Face dataset storage. This
inline dataset is only a smoke-test fixture.
"""

from __future__ import annotations

from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult


DATASET = [
    {"input": "What is 2 + 2?", "output": "4"},
    {"input": "What is 3 + 5?", "output": "8"},
]


def exact_match_reward(example: TaskExample, response_text: str) -> RewardResult:
    expected = str(example.expected_output or "").strip()
    score = 1.0 if expected and expected in response_text else 0.0
    return RewardResult(score=score, threshold=1.0)


class StarterEnv(EnvironmentSingleTurn):
    dataset = DATASET

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [{"role": "user", "content": example.task}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        return exact_match_reward(example, response_text)


def load_environment(**kwargs) -> StarterEnv:
    return StarterEnv()
'''


def cmd_env_setup(args) -> int:
    Path("configs").mkdir(exist_ok=True)
    starter_env = Path("environment.py")
    if not starter_env.exists():
        starter_env.write_text(_STARTER_ENV_PY)
    env_comment = (
        "# Environment: upload this project folder with\n"
        "# `flash env push --name my-env .`, then paste the returned id below.\n"
        "# If the environment reads secrets with os.environ, list only the env var names here.\n"
        "# Values are read from your shell or .env at submit time and are not stored in the spec.\n"
        "[environment]\n"
        'id = ""\n\n'
        '# secrets = ["SERPAPI_API_KEY"]\n\n'
    )
    grpo = Path("configs/grpo.toml")
    if not grpo.exists():
        grpo.write_text(
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "grpo"\n\n'
            f"{env_comment}"
            "[train]\n"
            "steps = 150\n"
            "lora_rank = 32\n"
            "seeds = [0]\n"
            "# GPU and the HF artifact repo are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting class across providers, and each run gets its own artifact repo.\n"
        )
    sft = Path("configs/sft.toml")
    if not sft.exists():
        sft.write_text(
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "sft"\n\n'
            f"{env_comment}"
            "[train]\n"
            "epochs = 1\n"
            "lora_rank = 32\n"
            "seeds = [0]\n"
            "# GPU and the HF artifact repo are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting class across providers, and each run gets its own artifact repo.\n"
        )
    print("ensured environment.py, configs/, configs/grpo.toml, configs/sft.toml")
    return 0


def _model_dimension(params: str) -> str:
    return params.split(" (", 1)[0]


def cmd_models(args) -> int:
    for row in public_model_rows():
        print(f"{row['id']}\t{_model_dimension(row['params'])}")
    return 0


def cmd_gpus(args) -> int:
    """List GPU classes, VRAM, and per-provider $/hr."""
    from flash.providers.base import GPU_INFO
    from flash.providers.runpod.pricing import static_rates as runpod_static_rates
    from flash.providers.vast.pricing import static_rates as vast_static_rates

    runpod_rates = runpod_static_rates()
    vast_rates = vast_static_rates()

    def fmt_rate(v: float | None) -> str:
        return f"{v:>10.2f}" if v else f"{'-':>10}"

    print(f"{'gpu':<16}{'vram':>6}{'runpod$/hr':>11}{'vast$/hr':>10}")
    for info in sorted(GPU_INFO.values(), key=lambda g: g.hourly_usd):
        runpod_rate = runpod_rates.get(info.name) if info.enum_member else None
        print(
            f"{info.name:<16}{info.vram_gb:>5}G{fmt_rate(runpod_rate):>11}"
            f"{fmt_rate(vast_rates.get(info.name))}"
        )
    print(
        "\nTip: GPU class selection is fully automatic — the submit-time allocator always picks the\n"
        "cheapest validated class that fits the model across all providers, so you don't pin a\n"
        "GPU type."
    )
    return 0


def cmd_env_list(args) -> int:
    from flash.envs.registry import list_installed_environments

    installed = list_installed_environments()
    if installed:
        print("installed environments:")
        for env_id in installed:
            print(f"  {env_id}")
    paths: list[str] = []
    if Path("environment.py").is_file():
        paths.append(".")
    local = Path("environments")
    if local.is_dir():
        # Prefer publishing folders. Single-file modules remain supported for small smoke tests.
        for p in local.iterdir():
            if p.name.startswith("__"):
                continue
            if p.is_dir():
                stem = p.name.replace("-", "_")
                module = p / f"{stem}.py"
                canonical = p / "environment.py"
                if canonical.is_file() or module.is_file():
                    paths.append(f"environments/{p.name}")
            elif p.suffix == ".py":
                paths.append(f"environments/{p.name}")
    if paths:
        print("local env sources (publish with `flash env push --name <name> <path>`):")
        for path in sorted(paths):
            print(f"  {path}")
    return 0


def _cmd_train_cost(args) -> int:
    """`flash train --cost`: print the pre-flight USD cost for the config and exit (no submit).

    Catalog-only and deterministic; an uncapped SFT run tries to count the env's train split, and
    falls back to a default example count (with a warning) when the environment isn't
    importable here."""
    from flash.cost import estimate_cost

    spec = spec_from_file(
        args.config,
        run_id=None,
        overrides=args.overrides,
        extra_configs=args.extra_configs,
    )
    print(estimate_cost(runconfig_from_spec(spec)).breakdown())
    return 0


def cmd_train(args) -> int:
    if getattr(args, "cost", False):
        return _cmd_train_cost(args)
    spec = spec_from_file(
        args.config,
        run_id=new_run_id() if args.dry_run else None,
        overrides=args.overrides,
        extra_configs=args.extra_configs,
    )
    if args.dry_run:
        # Fully local: validate the id-based config without credentials, a server, or a GPU.
        print(
            json.dumps(
                {"run_id": spec.run_id, "state": "dry_run", "spec": spec.to_dict()}, indent=2
            )
        )
        return 0
    client = client_from_config()
    status = client.create_run(
        spec_payload(spec),
        runtime_secrets=runtime_secrets_from_local_env(args.config, keys=spec.environment.secrets),
    )
    run_id = status["run_id"]
    logger.info(
        "submitted run %s: model=%s algorithm=%s gpu=%s seeds=%s",
        run_id,
        spec.model,
        spec.algorithm,
        spec.gpu.type,
        list(spec.train.seeds),
    )
    if args.background:
        print(json.dumps(status, indent=2))
        return 0
    print(
        f"run {run_id} submitted; following logs "
        f"(Ctrl-C detaches, `flash status {run_id} --follow` resumes)",
        file=sys.stderr,
    )
    return _follow_run(client, run_id)


def _poll_logs(client: ApiClient, run_id: str, interval: float) -> str:
    """Stream offset-paged logs until the run reaches a terminal state; return that state."""
    offset = 0
    spinner = _LogFollowSpinner(run_id)
    try:
        while True:
            page = client.get_logs(run_id, offset=offset)
            if page["logs"]:
                spinner.clear()
                print(page["logs"], end="", flush=True)
            offset = page["offset"]
            if page["state"] in _CLI_DONE_STATES:
                spinner.clear()
                return page["state"]
            _sleep_with_spinner(interval, spinner, page["state"])
    finally:
        spinner.clear()


def _follow_run(client: ApiClient, run_id: str) -> int:
    """Poll logs until the run reaches a terminal state, then print the final status."""
    state = _poll_logs(client, run_id, interval=2.0)
    print(json.dumps(client.get_run(run_id), indent=2))
    return 0 if state in _OK_STATES else 1


def cmd_status(args) -> int:
    client = client_from_config()
    if getattr(args, "follow", False):
        return _follow_run(client, args.run_id)
    if getattr(args, "logs", False):
        logs = client.get_logs(args.run_id)["logs"]
        if logs:
            print(logs, end="")
            if not logs.endswith("\n"):
                print()
    print(json.dumps(client.get_run(args.run_id), indent=2))
    return 0


def cmd_runs(args) -> int:
    runs = client_from_config().list_runs()
    if not runs:
        print("no runs yet")
        return 0
    print(f"{'RUN_ID':<32}  {'STATE':<11}  {'ALGO':<5}  {'COST($)':>8}  {'GPU':<22}  MODEL")
    for r in sorted(runs, key=lambda r: r.get("updated_at", 0), reverse=True):
        spec = r.get("spec") or {}
        model = spec.get("model", "")
        algorithm = str(spec.get("algorithm") or "-").upper()
        remote = r.get("remote") or {}
        # the remote handle knows what actually ran; the spec is the parse-time pick
        provider = remote.get("provider") or (
            "runpod" if remote else (spec.get("gpu") or {}).get("provider", "")
        )
        gpu = remote.get("gpu") or (spec.get("gpu") or {}).get("type", "")
        where = f"{gpu}@{provider}" if provider else gpu
        print(
            f"{r['run_id']:<32}  {r['state']:<11}  {algorithm:<5}  "
            f"{r.get('cost_usd', 0.0):>8.4f}  {where:<22}  {model}"
        )
    return 0


def cmd_cancel(args) -> int:
    status = client_from_config().cancel_run(args.run_id)
    print(json.dumps({"run_id": args.run_id, "state": status["state"]}, indent=2))
    return 0


def cmd_deploy(args) -> int:
    dep = client_from_config().deploy(
        args.run_id,
        dry_run=args.dry_run,
    )
    print(json.dumps(dep, indent=2))
    print(
        "note: serving is billed per token only; use "
        f"`flash undeploy {args.run_id}` to deregister the adapter.",
        file=sys.stderr,
    )
    return 0


def cmd_undeploy(args) -> int:
    print(json.dumps(client_from_config().undeploy(args.run_id), indent=2))
    return 0


def cmd_deployments(args) -> int:
    rows = client_from_config().deployments()
    if not rows:
        print("no active deployments")
        return 0
    print(f"{'RUN_ID':<32}  {'GPU':<9}  ENDPOINT")
    for r in rows:
        d = r.get("deployment") or {}
        print(f"{r['run_id']:<32}  {d.get('gpu', '?'):<9}  {d.get('endpoint_name', '')}")
    return 0


def cmd_chat(args) -> int:
    client = client_from_config()
    messages = [{"role": "user", "content": args.message}]
    stream = getattr(client, "chat_stream", None)
    if stream is not None:
        wrote = False
        for chunk in stream(
            args.run_id,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ):
            print(chunk, end="", flush=True)
            wrote = True
        if wrote:
            print()
        return 0

    resp = client.chat(
        args.run_id,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(resp["choices"][0]["message"]["content"])
    return 0
