"""CLI for the managed AutoSLM service.

Every run-lifecycle command is a thin HTTP call to the AutoSLM control plane —
users authenticate with their freesolo API key (`slm login` verifies it against
the freesolo backend), never with provider credentials. Config parsing/validation
and `--dry-run` stay fully local.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from autoslm import __version__
from autoslm._logging import configure_logging, get_logger
from autoslm.catalog import public_model_rows
from autoslm.client import (
    ApiClient,
    ClientError,
    client_from_config,
    save_credentials,
    verify_freesolo_key,
)
from autoslm.client.config import load_credentials
from autoslm.client.specs import spec_payload
from autoslm.runner import TERMINAL_STATES, new_run_id
from autoslm.schema import ConfigError, spec_from_file

logger = get_logger(__name__)

# Exceptions that represent expected user/config errors: report them as a clean one-line
# message instead of a Python traceback (use --debug / AUTOSLM_DEBUG=1 to see the full trace).
_USER_ERRORS = (
    ConfigError,
    ClientError,
    FileNotFoundError,
    ValueError,
)

# Run states after which nothing more will happen (polling can stop).
_CLI_DONE_STATES = TERMINAL_STATES | {"deployed"}
_OK_STATES = {"done", "dry_run", "deployed"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="slm", description="Managed LoRA post-training")
    parser.add_argument("-V", "--version", action="version", version=f"slm {__version__}")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show full tracebacks on error (or set AUTOSLM_DEBUG=1)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (-v for info, -vv for debug; or set AUTOSLM_LOG_LEVEL)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    version = sub.add_parser("version", help="print the AutoSLM version")
    version.set_defaults(func=cmd_version)

    login = sub.add_parser("login", help="log in with your freesolo API key (verified by freesolo)")
    login.add_argument(
        "--api-key",
        help="your freesolo API key (default: FREESOLO_API_KEY); created in the dashboard",
    )
    login.add_argument(
        "--freesolo-url",
        dest="freesolo_url",
        help="freesolo backend base URL (default: FREESOLO_BASE_URL or https://api.freesolo.co)",
    )
    login.add_argument(
        "--api-url", help="autoslm control-plane URL for training calls (default: AUTOSLM_API_URL)"
    )
    login.set_defaults(func=cmd_login)

    whoami = sub.add_parser("whoami", help="show the identity behind your stored key")
    whoami.set_defaults(func=cmd_whoami)

    lab = sub.add_parser("lab")
    lab_sub = lab.add_subparsers(dest="lab_cmd", required=True)
    setup = lab_sub.add_parser("setup")
    setup.set_defaults(func=cmd_lab_setup)

    models = sub.add_parser("models")
    models.set_defaults(func=cmd_models)

    gpus = sub.add_parser("gpus", help="list managed GPU classes with live $/hr")
    gpus.set_defaults(func=cmd_gpus)

    env = sub.add_parser("env")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    init = env_sub.add_parser("init")
    init.add_argument("name")
    init.set_defaults(func=cmd_env_init)

    env_list = env_sub.add_parser("list", help="list installed + local environments")
    env_list.set_defaults(func=cmd_env_list)

    env_install = env_sub.add_parser("install", help="install a published Prime Hub environment")
    env_install.add_argument("env_id", help='the env id to install (a Hub slug, "owner/name")')
    env_install.set_defaults(func=cmd_env_install)

    env_push = env_sub.add_parser(
        "push", help="publish a local verifiers env to the Prime Hub (private); prints its env id"
    )
    env_push.add_argument("path", nargs="?", default=".")
    env_push.set_defaults(func=cmd_env_push)

    train = sub.add_parser("train")
    train.add_argument("config")
    train.add_argument(
        "--config",
        dest="extra_configs",
        action="append",
        default=[],
        help="additional TOML to deep-merge (config composition); repeatable",
    )
    train.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="override a config value; repeatable",
    )
    train.add_argument("--dry-run", action="store_true")
    train.add_argument(
        "--background",
        action="store_true",
        help="submit and return immediately instead of following logs",
    )
    train.set_defaults(func=cmd_train)

    status = sub.add_parser("status")
    status.add_argument("run_id")
    status.set_defaults(func=cmd_status)

    attach = sub.add_parser(
        "attach", help="follow a running job's logs to completion (resumable any time)"
    )
    attach.add_argument("run_id")
    attach.set_defaults(func=cmd_attach)

    ps = sub.add_parser("ps", help="list runs and their state/cost")
    ps.set_defaults(func=cmd_ps)

    cost = sub.add_parser("cost", help="show a run's accrued cost (USD)")
    cost.add_argument("run_id")
    cost.set_defaults(func=cmd_cost)

    cancel = sub.add_parser("cancel", help="cancel a run (best-effort)")
    cancel.add_argument("run_id")
    cancel.set_defaults(func=cmd_cancel)

    logs = sub.add_parser("logs")
    logs.add_argument("run_id")
    logs.add_argument("-f", "--follow", action="store_true", help="stream new log lines")
    logs.set_defaults(func=cmd_logs)

    deploy = sub.add_parser("deploy")
    deploy.add_argument("run_id")
    deploy.add_argument(
        "--mode",
        choices=["dev", "always-on"],
        default="dev",
        help="dev: scale-to-zero, cold start after idle, $0 when unused (default). "
        "always-on: one warm worker 24/7, no cold starts, continuous billing.",
    )
    deploy.add_argument(
        "--idle-timeout",
        type=int,
        default=300,
        help="dev mode: seconds of inactivity before the worker scales to zero (default 300)",
    )
    deploy.add_argument("--dry-run", action="store_true")
    deploy.set_defaults(func=cmd_deploy)

    undeploy = sub.add_parser("undeploy", help="tear down a run's serving endpoint")
    undeploy.add_argument("run_id")
    undeploy.set_defaults(func=cmd_undeploy)

    deployments = sub.add_parser("deployments", help="list active serving deployments")
    deployments.set_defaults(func=cmd_deployments)

    chat = sub.add_parser("chat", help="chat with a deployed adapter")
    chat.add_argument("run_id")
    chat.add_argument("-m", "--message", required=True)
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.set_defaults(func=cmd_chat)

    # The control plane is operator-only and run as a separate one-off service via the
    # `autoslm-server` console script (autoslm.server.app:main), not a `slm` subcommand.

    args = parser.parse_args(argv)
    configure_logging(verbosity=getattr(args, "verbose", 0))
    debug = getattr(args, "debug", False) or os.environ.get("AUTOSLM_DEBUG") not in (None, "", "0")
    try:
        return args.func(args)
    except _USER_ERRORS as exc:
        if debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("aborted", file=sys.stderr)
        return 130


def cmd_version(args) -> int:
    print(f"slm {__version__}")
    return 0


def cmd_login(args) -> int:
    # Login is handled by the freesolo backend (not the autoslm control plane): the user
    # supplies the freesolo API key they created in the dashboard, and we verify it against
    # freesolo before storing it. The same key authenticates autoslm's control plane.
    api_key = args.api_key or os.environ.get("FREESOLO_API_KEY")
    if not api_key:
        raise ClientError(
            "no API key provided: pass `--api-key <key>` or set FREESOLO_API_KEY. "
            "Create a key in your freesolo dashboard."
        )
    verify_freesolo_key(api_key, base_url=getattr(args, "freesolo_url", None))
    api_url = args.api_url or load_credentials()[0]
    # save_credentials clears the stored url when it's the default, so logging into the
    # default plane also drops a stale custom url from a previous custom-URL login.
    path = save_credentials(api_key, api_url=api_url)
    # Never echo the key itself; the stored file is the single source of truth.
    print(f"logged in: freesolo verified your key (saved to {path})")
    print("you're ready to train — try `slm train <config.toml>`")
    return 0


def cmd_whoami(args) -> int:
    print(json.dumps(client_from_config().me(), indent=2))
    return 0


_STARTER_ENV_PY = '''\
"""Starter local verifiers environment.

Replace the dataset and rubric with your task, then publish it to the Prime Hub with
`slm env push environments/starter_env.py`. A managed run references the published env by
its Hub slug: set [environment] id = "owner/name" in the config.
See https://github.com/PrimeIntellect-ai/verifiers for the full API.
"""

import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {"prompt": [{"role": "user", "content": "What is 2 + 2?"}], "answer": "4"},
            {"prompt": [{"role": "user", "content": "What is 3 + 5?"}], "answer": "8"},
        ]
    )

    def correct_answer(completion, answer, **_):
        """Reward 1.0 when the gold answer appears in the model's final message."""
        text = completion[-1]["content"] if isinstance(completion, list) else str(completion)
        return 1.0 if str(answer) in text else 0.0

    rubric = vf.Rubric(funcs=[correct_answer], weights=[1.0])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
'''


def cmd_lab_setup(args) -> int:
    Path("environments").mkdir(exist_ok=True)
    Path("configs").mkdir(exist_ok=True)
    Path("configs/endpoints.toml").write_text(
        "# OpenAI-compatible endpoints returned by `slm deploy` can be stored here.\n"
    )
    starter_env = Path("environments/starter_env.py")
    if not starter_env.exists():
        starter_env.write_text(_STARTER_ENV_PY)
    sample = Path("configs/verifiers_grpo.toml")
    if not sample.exists():
        sample.write_text(
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "grpo"\n\n'
            "# Environment: a verifiers / Prime Hub env slug. Publish the scaffolded\n"
            "# environments/starter_env.py with `slm env push environments/starter_env.py`\n"
            "# (then `slm env install owner/name`) to get the slug, and set it below.\n"
            "[environment]\n"
            'id = "owner/name"   # a verifiers / Prime Hub env slug\n\n'
            "[train]\n"
            'hf_repo = "your-org/your-runs"   # HF dataset repo for adapters/checkpoints\n'
            "steps = 150\n"
            "lora_rank = 32\n"
            "seeds = [0]\n\n"
            "# Managed GPU (RTX 4090 or RTX 5090 only).\n"
            "[gpu]\n"
            'type = "RTX 5090"\n'
        )
    print(
        "created environments/, environments/starter_env.py, configs/, "
        "configs/verifiers_grpo.toml, configs/endpoints.toml"
    )
    return 0


def cmd_models(args) -> int:
    for row in public_model_rows():
        print(
            f"{row['id']}\t{row['params']}\talgos={','.join(row['algos'])}\t{row['quant']}"
            f"\tthinking={row.get('thinking', 'none')}"
        )
    return 0


def cmd_gpus(args) -> int:
    """List GPU classes, VRAM, per-provider $/hr and live validation."""
    import sys

    from autoslm.providers import available_providers
    from autoslm.providers.base import GPU_INFO
    from autoslm.providers.runpod.pricing import live_rates

    rates = live_rates()
    # Cheapest live verified-datacenter offer per class (vast key + network only).
    vast_rates: dict[str, float] = {}
    if "vast" in available_providers():
        try:
            from autoslm.providers.vast.jobs import usable_offers

            for offer in usable_offers(0, 0):
                vast_rates.setdefault(offer.gpu, offer.dph_total)  # offers are price-sorted
        except Exception as exc:
            print(f"warning: vast offers unavailable ({exc})", file=sys.stderr)

    def fmt_rate(v: float | None) -> str:
        return f"{v:>10.2f}" if v else f"{'-':>10}"

    print(f"{'gpu':<16}{'vram':>6}{'runpod$/hr':>11}{'vast$/hr':>10}  validated_on")
    for info in sorted(GPU_INFO.values(), key=lambda g: rates.get(g.name, g.hourly_usd)):
        runpod_rate = rates.get(info.name, info.hourly_usd) if info.enum_member else None
        validated = ",".join(info.validated_on) or "- (needs gpu.allow_unvalidated)"
        print(
            f"{info.name:<16}{info.vram_gb:>5}G{fmt_rate(runpod_rate):>11}"
            f"{fmt_rate(vast_rates.get(info.name))}  {validated}"
        )
    print(
        '\nTip: omit gpu.type (or set "cheapest") to allocate the cheapest validated class\n'
        "across providers that fits the model; gpu.provider pins runpod/vast."
    )
    return 0


def cmd_env_init(args) -> int:
    mod = args.name.replace("-", "_")
    root = Path("environments") / mod
    root.mkdir(parents=True, exist_ok=True)
    # Verifiers-only: scaffold a real verifiers env whose load_environment returns a
    # vf.Environment (here a SingleTurnEnv + Rubric over a datasets.Dataset). This is what
    # a Hub push expects, so a freshly scaffolded env actually loads.
    (root / f"{mod}.py").write_text(
        f'"""Custom verifiers environment ({args.name}).\n\n'
        "Replace the dataset and rubric with your task, then publish it to the Prime Hub\n"
        f"with `slm env push environments/{mod}/{mod}.py` and reference it by id\n"
        '([environment] id = "owner/name") in your config.\n'
        "See https://github.com/PrimeIntellect-ai/verifiers for the full API.\n"
        '"""\n\n'
        "import verifiers as vf\n"
        "from datasets import Dataset\n\n\n"
        "def load_environment(**kwargs) -> vf.Environment:\n"
        "    dataset = Dataset.from_list(\n"
        "        [\n"
        '            {"prompt": [{"role": "user", "content": "What is 2 + 2?"}], "answer": "4"},\n'
        '            {"prompt": [{"role": "user", "content": "What is 3 + 5?"}], "answer": "8"},\n'
        "        ]\n"
        "    )\n\n"
        "    def correct_answer(completion, answer, **_):\n"
        '        """Reward 1.0 when the gold answer appears in the model\'s final message."""\n'
        "        text = (\n"
        '            completion[-1]["content"] if isinstance(completion, list) else str(completion)\n'
        "        )\n"
        "        return 1.0 if str(answer) in text else 0.0\n\n"
        "    rubric = vf.Rubric(funcs=[correct_answer], weights=[1.0])\n"
        "    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)\n"
    )
    (root / "README.md").write_text(f"# {args.name}\n\nCustom verifiers environment for AutoSLM.\n")
    print(f"created {root}")
    print(
        f"publish it to the Prime Hub with `slm env push environments/{mod}/{mod}.py`, "
        'then reference it by id ([environment] id = "owner/name") in your config.'
    )
    return 0


def cmd_env_list(args) -> int:
    from autoslm.envs.registry import list_installed_verifiers_envs

    installed = list_installed_verifiers_envs()
    if installed:
        print("installed (verifiers / Prime Hub):")
        for env_id in installed:
            print(f"  {env_id}")
    local = Path("environments")
    if local.is_dir():
        # Both directory envs (environments/<name>/<name>.py) and top-level single-file
        # modules (environments/<name>.py, e.g. the `slm lab` starter env). These are local
        # env SOURCES — publish one with `slm env push <path>` to run it on the managed
        # service by its Hub id.
        paths: list[str] = []
        for p in local.iterdir():
            if p.name.startswith("__"):
                continue
            if p.is_dir():
                # `slm env init` maps a hyphenated dir to an underscored inner module file
                # (my-env/ -> my-env/my_env.py). List that exact path, and only when it
                # actually exists (an empty/incomplete folder isn't a publishable source).
                stem = p.name.replace("-", "_")
                module = p / f"{stem}.py"
                if module.is_file():
                    paths.append(f"environments/{p.name}/{stem}.py")
            elif p.suffix == ".py":
                paths.append(f"environments/{p.name}")
        if paths:
            print("local env sources (publish with `slm env push <path>`):")
            for path in sorted(paths):
                print(f"  {path}")
    return 0


# Prime Intellect Environments Hub pip index (used by default for owner/name Hub slugs).
PRIME_HUB_INDEX = "https://hub.primeintellect.ai/primeintellect/simple/"


def cmd_env_install(args) -> int:
    import shutil
    import subprocess

    from autoslm.envs.registry import _bare_wheel_name, record_installed_env

    env_id = args.env_id
    # Managed envs are Prime Hub slugs: exactly one `/` with non-empty owner and name. A bare
    # id (`gsm8k`) or a malformed slug can't be resolved on the Hub, so reject it up front
    # rather than letting `prime`/pip fail with an opaque error.
    parts = env_id.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        print(
            f'env id must be a Prime Hub slug "owner/name" (got {env_id!r})',
            file=sys.stderr,
        )
        return 1
    # `slm env install` is a LOCAL-client convenience: it installs the env into the client's
    # interpreter and records it in ~/.autoslm/envs.json for local authoring/dry-run. The
    # managed worker does NOT reinstall from this record — it installs Hub envs itself via an
    # authenticated `prime env install` on the GPU box. A Hub slug `owner/name` maps to the pip
    # wheel `name` on the Prime Intellect Hub index; we record that index alongside the env.
    extras = {"extra_index_url": PRIME_HUB_INDEX} if "/" in env_id else None
    is_hub_slug = "/" in env_id
    if shutil.which("prime"):
        # The `prime` CLI resolves the Hub + index itself (and is the only path that can fetch a
        # PRIVATE Hub env — autoslm publishes envs PRIVATE).
        cmd = ["prime", "env", "install", env_id]
    else:
        if is_hub_slug:
            # The pip fallback hits the PUBLIC Hub index only; it cannot fetch PRIVATE Hub envs
            # (the public index never serves private wheels). Be explicit instead of letting a
            # private install fail confusingly, but still attempt pip for the public case.
            print(
                f"note: `prime` CLI not found; attempting a pip install of {env_id} from the "
                "PUBLIC Hub index. PRIVATE Hub envs require the `prime` CLI — install it "
                "(https://docs.primeintellect.ai) to install a private env."
            )
        installer = (
            # `uv pip install` outside an active venv errors with "No virtual environment
            # found"; --python targets the CLI's own interpreter so a global/pipx `slm`
            # install still records the env.
            ["uv", "pip", "install", "--python", sys.executable]
            if shutil.which("uv")
            else [sys.executable, "-m", "pip", "install"]
        )
        cmd = [*installer, _bare_wheel_name(env_id)]
        if extras:
            cmd += ["--extra-index-url", PRIME_HUB_INDEX]
    print("running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print("install failed")
        return rc
    record_installed_env(env_id, package=_bare_wheel_name(env_id), extras=extras)
    print(f"installed {env_id}; recorded in ~/.autoslm/envs.json")
    print(f'use it via:  [environment]\\nid = "{env_id}"')
    return 0


# A verifiers env packaged for the Prime Hub is a pyproject + an importable module exposing
# load_environment(). When `slm env push` is pointed at a bare module (a single `.py`, as the
# freesolo training agent emits, or a dir without a pyproject), we wrap it in this layout so the
# push Just Works instead of erroring on "pyproject.toml not found".
_ENV_PUSH_PYPROJECT = """\
[project]
name = "{name}"
version = "{version}"
description = "AutoSLM verifiers environment ({name})."
requires-python = ">=3.10"
dependencies = ["verifiers"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["{module}"]
"""

_PUSH_INITIAL_VERSION = "0.1.0"
_PUSH_MAX_ATTEMPTS = 8
_PUSH_CONFLICT_MARKERS = ("already exists", "version already", "duplicate", "conflict", "409")


def _push_env_name(raw: str) -> str:
    import re

    name = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return name or "autoslm-env"


def _push_is_version_conflict(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _PUSH_CONFLICT_MARKERS)


def _push_slug_from(env_dir, output: str) -> str | None:
    import json
    import re
    from pathlib import Path

    meta = Path(env_dir) / ".prime" / ".env-metadata.json"
    try:
        data = json.loads(meta.read_text())
        owner, name = data.get("owner"), data.get("name")
        if owner and name:
            return f"{owner}/{name}"
    except (OSError, json.JSONDecodeError):
        pass
    match = re.search(r"[Ss]uccessfully pushed\s+([A-Za-z0-9][\w.-]*/[\w.-]+)", output)
    return match.group(1) if match else None


def _config_env_name(config_path) -> str | None:
    """The `name` part of a sibling autoslm.toml's `[environment] id = "owner/name"`, or None.

    Used so a bare `environment.py` re-publishes under its EXISTING Hub env (minting a new
    version) instead of deriving a fresh name from the file stem. Owner still comes from the
    authenticated Prime account/team, so only the name part is consumed here."""
    import tomllib
    from pathlib import Path

    path = Path(config_path)
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    env = data.get("environment")
    env_id = str(env.get("id") or "").strip() if isinstance(env, dict) else ""
    if "/" in env_id:
        name = env_id.split("/", 1)[1].strip()
        return name or None
    return None


def _run_prime_push(env_dir, *, is_new: bool, name: str | None = None) -> int:
    """Run `prime env push` on a packaged env dir (always PRIVATE), climbing past conflicts.

    When `name` is given it is passed as `--name` so the push targets that exact Hub env."""
    import subprocess

    # Published environments are always PRIVATE — they can hold proprietary task data.
    base = ["prime", "env", "push", "--plain", "--path", str(env_dir), "--visibility", "PRIVATE"]
    if name:
        base += ["--name", name]
    # Disable prime's interactive version check so a push isn't blocked in non-interactive
    # use (PRIME_API_KEY is inherited from the user's environment).
    env = {**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"}
    auto_bump = not is_new  # a re-publish must land on a fresh version
    for _ in range(_PUSH_MAX_ATTEMPTS):
        cmd = [*base, "--auto-bump"] if auto_bump else list(base)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        output = f"{proc.stdout or ''}{proc.stderr or ''}"
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="")
        if proc.returncode == 0:
            slug = _push_slug_from(env_dir, output)
            if slug:
                print(f"published {slug}")
            else:
                # Don't report a clean success we can't confirm: the push exited 0 but we
                # couldn't parse the owner/name id, so the env reference may be unrecorded.
                print(
                    "warning: `prime env push` exited 0 but no owner/name id could be parsed; "
                    "verify the environment on the Prime Hub before training against it",
                    file=sys.stderr,
                )
            return 0
        if _push_is_version_conflict(output):
            auto_bump = True
            continue
        return proc.returncode
    print(f"push failed after {_PUSH_MAX_ATTEMPTS} version-conflict retries", file=sys.stderr)
    return 1


def cmd_env_push(args) -> int:
    import shutil
    import tempfile
    from pathlib import Path

    if not shutil.which("prime"):
        print("the `prime` CLI is required to publish to the Environments Hub.")
        print("install it (https://docs.primeintellect.ai) then re-run `slm env push`.")
        return 1

    src = Path(args.path)
    if not src.exists():
        print(f"no such path: {src}", file=sys.stderr)
        return 1

    # A proper env directory (has a pyproject.toml) is pushed as-is; its name comes from the
    # pyproject. Otherwise the published env name is derived from the env's path.
    if src.is_dir() and (src / "pyproject.toml").is_file():
        # First attempt never forces --auto-bump; the version-conflict retry enables it only
        # when the version actually collides, so a genuine first publish keeps its version.
        return _run_prime_push(src, is_new=True)

    # Wrap a bare verifiers module (a single .py, or a one-module dir) into a Prime-compatible
    # env package and push that. `--auto-bump` retries handle re-publishes. `data_dir` is a
    # committed `datasets/` sibling of the module (if any); we ship it inside the package so an
    # env that reads a `__file__`-relative data file still resolves once installed.
    if src.is_file() and src.suffix == ".py":
        module_source = src.read_text()
        # Re-publish to the SAME Hub env when a sibling autoslm.toml names one: use its
        # `[environment] id` name part so an edited environment.py mints a new version of the
        # existing env instead of creating a fresh env from the file stem.
        sibling_name = _config_env_name(src.parent / "autoslm.toml")
        env_name = sibling_name or _push_env_name(src.stem)
        data_dir = src.parent / "datasets"
        # A sibling config id means we're re-publishing an EXISTING Hub env: auto-bump from the
        # first attempt so it doesn't restart at 0.1.0 and climb through version conflicts.
        is_new = sibling_name is None
    elif src.is_dir():
        modules = [p for p in sorted(src.glob("*.py")) if not p.name.startswith("__")]
        if len(modules) != 1:
            print(
                f"{src} has no pyproject.toml and {'no' if not modules else 'multiple'} "
                "top-level .py module(s); point `slm env push` at the env's .py file or add a "
                "pyproject.toml.",
                file=sys.stderr,
            )
            return 1
        module_source = modules[0].read_text()
        env_name = _push_env_name(src.name)
        data_dir = src / "datasets"
        is_new = True
    else:
        print(f"cannot publish {src}: expected a verifiers .py module or an env directory.")
        return 1

    module = env_name.replace("-", "_")
    # A Python package name can't start with a digit, so prefix one (e.g. "2026-task").
    if module[:1].isdigit():
        module = f"env_{module}"
    with tempfile.TemporaryDirectory(prefix="slm-env-push-") as tmp:
        pkg = Path(tmp)
        (pkg / module).mkdir()
        (pkg / module / "__init__.py").write_text(module_source)
        # Ship committed sibling data inside the package dir (it lands at <module>/datasets/, so a
        # `os.path.dirname(__file__)/datasets/...` read resolves on the worker); the whole package
        # dir ships via `[tool.hatch.build.targets.wheel] packages = ["<module>"]`.
        if data_dir.is_dir() and any(data_dir.iterdir()):
            import shutil

            shutil.copytree(data_dir, pkg / module / "datasets")
        (pkg / "pyproject.toml").write_text(
            _ENV_PUSH_PYPROJECT.format(name=env_name, module=module, version=_PUSH_INITIAL_VERSION)
        )
        (pkg / "README.md").write_text(f"# {env_name}\n\nAutoSLM verifiers environment.\n")
        return _run_prime_push(pkg, is_new=is_new, name=env_name)


def cmd_train(args) -> int:
    spec = spec_from_file(
        args.config,
        run_id=new_run_id() if args.dry_run else None,
        overrides=getattr(args, "overrides", None),
        extra_configs=getattr(args, "extra_configs", None),
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
    status = client.create_run(spec_payload(spec))
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
        f"run {run_id} submitted; following logs (Ctrl-C detaches, `slm attach {run_id}` resumes)",
        file=sys.stderr,
    )
    return _follow_run(client, run_id)


def _follow_run(client: ApiClient, run_id: str, final_status: bool = True) -> int:
    """Poll logs (offset-paged) until the run reaches a terminal state."""
    offset = 0
    while True:
        page = client.get_logs(run_id, offset=offset)
        if page["logs"]:
            print(page["logs"], end="", flush=True)
        offset = page["offset"]
        if page["state"] in _CLI_DONE_STATES:
            state = page["state"]
            break
        time.sleep(2.0)
    if final_status:
        print(json.dumps(client.get_run(run_id), indent=2))
    return 0 if state in _OK_STATES else 1


def cmd_status(args) -> int:
    print(json.dumps(client_from_config().get_run(args.run_id), indent=2))
    return 0


def cmd_attach(args) -> int:
    client = client_from_config()
    return _follow_run(client, args.run_id)


def cmd_ps(args) -> int:
    runs = client_from_config().list_runs()
    if not runs:
        print("no runs yet")
        return 0
    print(f"{'RUN_ID':<32}  {'STATE':<11}  {'COST($)':>8}  {'GPU':<22}  MODEL")
    for r in sorted(runs, key=lambda r: r.get("updated_at", 0), reverse=True):
        spec = r.get("spec") or {}
        model = spec.get("model", "")
        remote = r.get("remote") or {}
        # the remote handle knows what actually ran; the spec is the parse-time pick
        provider = remote.get("provider") or (
            "runpod" if remote else (spec.get("gpu") or {}).get("provider", "")
        )
        gpu = remote.get("gpu") or (spec.get("gpu") or {}).get("type", "")
        where = f"{gpu}@{provider}" if provider else gpu
        print(
            f"{r['run_id']:<32}  {r['state']:<11}  {r.get('cost_usd', 0.0):>8.4f}  "
            f"{where:<22}  {model}"
        )
    return 0


def cmd_cost(args) -> int:
    status = client_from_config().get_run(args.run_id)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "state": status["state"],
                "cost_usd": status.get("cost_usd", 0.0),
            },
            indent=2,
        )
    )
    return 0


def cmd_cancel(args) -> int:
    status = client_from_config().cancel_run(args.run_id)
    print(json.dumps({"run_id": args.run_id, "state": status["state"]}, indent=2))
    return 0


def cmd_logs(args) -> int:
    client = client_from_config()
    if not args.follow:
        print(client.get_logs(args.run_id)["logs"], end="")
        return 0
    offset = 0
    while True:
        page = client.get_logs(args.run_id, offset=offset)
        if page["logs"]:
            print(page["logs"], end="", flush=True)
        offset = page["offset"]
        if page["state"] in _CLI_DONE_STATES:
            return 0
        time.sleep(1.0)


def cmd_deploy(args) -> int:
    dep = client_from_config().deploy(
        args.run_id,
        mode=args.mode,
        idle_timeout_s=args.idle_timeout,
        dry_run=args.dry_run,
    )
    print(json.dumps(dep, indent=2))
    if dep.get("mode") == "always-on":
        print(
            f"note: always-on keeps a {dep.get('gpu')} warm 24/7 "
            f"(~${dep.get('est_idle_cost_usd_per_day')}/day). Use `slm undeploy {args.run_id}` "
            "to stop billing.",
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
    print(f"{'RUN_ID':<32}  {'MODE':<10}  {'GPU':<9}  {'$/DAY':>7}  ENDPOINT")
    for r in rows:
        d = r.get("deployment") or {}
        print(
            f"{r['run_id']:<32}  {d.get('mode', '?'):<10}  {d.get('gpu', '?'):<9}  "
            f"{d.get('est_idle_cost_usd_per_day', 0):>7}  {d.get('endpoint_name', '')}"
        )
    return 0


def cmd_chat(args) -> int:
    resp = client_from_config().chat(
        args.run_id,
        messages=[{"role": "user", "content": args.message}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(resp["choices"][0]["message"]["content"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
