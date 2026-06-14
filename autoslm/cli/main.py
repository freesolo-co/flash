"""CLI for the managed AutoSLM service.

Every run-lifecycle command is a thin HTTP call to the AutoSLM control plane —
users authenticate with a single claimed key (`slm login`), never with provider
credentials. Config parsing/validation and `--dry-run` stay fully local.
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
from autoslm.client import ApiClient, ClientError, client_from_config, save_credentials
from autoslm.client.config import load_credentials
from autoslm.client.specs import spec_payload
from autoslm.config_schema import ConfigError, spec_from_file
from autoslm.orchestrator import TERMINAL_STATES, new_run_id
from autoslm.worker_spec import JobSpec

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

    login = sub.add_parser("login", help="claim (or store) your AutoSLM API key")
    login.add_argument("--email", help="optionally tag the claimed key with your email")
    login.add_argument("--api-key", help="store an existing AutoSLM key instead of claiming one")
    login.add_argument("--api-url", help="control-plane URL (default: AUTOSLM_API_URL or hosted)")
    login.set_defaults(func=cmd_login)

    whoami = sub.add_parser("whoami", help="show the identity behind your stored key")
    whoami.set_defaults(func=cmd_whoami)

    lab = sub.add_parser("lab")
    lab_sub = lab.add_subparsers(dest="lab_cmd", required=True)
    setup = lab_sub.add_parser("setup")
    setup.set_defaults(func=cmd_lab_setup)

    models = sub.add_parser("models")
    models.add_argument("--experimental", action="store_true")
    models.set_defaults(func=cmd_models)

    gpus = sub.add_parser("gpus", help="list managed GPU classes with live $/hr")
    gpus.add_argument("--offline", action="store_true", help="static rates only (no API call)")
    gpus.set_defaults(func=cmd_gpus)

    env = sub.add_parser("env")
    env_sub = env.add_subparsers(dest="env_cmd", required=True)
    init = env_sub.add_parser("init")
    init.add_argument("name")
    init.set_defaults(func=cmd_env_init)

    env_list = env_sub.add_parser("list", help="list installed + local environments")
    env_list.set_defaults(func=cmd_env_list)

    env_install = env_sub.add_parser("install", help="install a verifiers / Prime Hub environment")
    env_install.add_argument("env_id", help="e.g. owner/name or a verifiers env id")
    env_install.add_argument("--package", help="pip wheel name to install (default: the env name)")
    env_install.add_argument(
        "--extra-index-url",
        dest="extra_index_url",
        help="extra pip index (Prime Hub slugs default to the Prime index)",
    )
    env_install.set_defaults(func=cmd_env_install)

    env_push = env_sub.add_parser("push", help="publish a local environment to the Prime Hub")
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

    proxy = sub.add_parser(
        "serve-proxy",
        help="local OpenAI-compatible HTTP shim (/v1/chat/completions) for a deployment",
    )
    proxy.add_argument("run_id")
    proxy.add_argument("--port", type=int, default=8000)
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.set_defaults(func=cmd_serve_proxy)

    chat = sub.add_parser("chat", help="chat with a deployed adapter")
    chat.add_argument("run_id")
    chat.add_argument("-m", "--message", required=True)
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.set_defaults(func=cmd_chat)

    server = sub.add_parser("server", help="run the AutoSLM control plane (operator-side)")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8080)
    server.set_defaults(func=cmd_server)

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
    api_url = args.api_url or load_credentials()[0]
    if args.api_key:
        api_key = args.api_key
    else:
        claimed = ApiClient(api_url).claim_key(email=args.email)
        api_key = claimed["api_key"]
    # Persist the plane we actually authenticated against (it may have come from
    # AUTOSLM_API_URL). save_credentials clears the stored url when it's the default, so
    # logging into default also drops a stale custom url from a previous self-hosted login.
    path = save_credentials(api_key, api_url=api_url)
    # Never echo the key itself; the stored file is the single source of truth.
    print(f"logged in: key saved to {path}")
    print("you're ready to train — try `slm train <config.toml>`")
    return 0


def cmd_whoami(args) -> int:
    print(json.dumps(client_from_config().me(), indent=2))
    return 0


_STARTER_ENV_PY = '''\
"""Starter LOCAL verifiers environment.

`slm train` loads this via [environment] path (no install needed). Replace the dataset and
rubric with your task. See https://github.com/PrimeIntellect-ai/verifiers for the full API.
To use a published Prime Hub env instead, run `slm env install owner/name` and set
[environment] id = "owner/name" in the config (drop the `path`).
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
            'model = "Qwen/Qwen3-4B-Instruct-2507"\n'
            'algorithm = "grpo"\n\n'
            "# Environment: either a Prime Hub slug (install it first with\n"
            '#   `slm env install owner/name`  then set  id = "owner/name")\n'
            "# or a LOCAL verifiers env module via `path` (scaffolded below).\n"
            "[environment]\n"
            'path = "environments/starter_env.py"\n'
            '# id = "owner/name"   # a verifiers / Prime Hub env slug\n\n'
            "[train]\n"
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
    for row in public_model_rows(include_experimental=args.experimental):
        suffix = " experimental" if row["experimental"] else ""
        print(
            f"{row['id']}\t{row['params']}\talgos={','.join(row['algos'])}\t{row['quant']}"
            f"\tthinking={row.get('thinking', 'none')}{suffix}"
        )
    return 0


def cmd_gpus(args) -> int:
    """List GPU classes, VRAM, per-provider $/hr and validation (live unless --offline)."""
    import os as _os

    from autoslm.flash.gpus import GPU_INFO
    from autoslm.flash.pricing import live_rates

    if args.offline:
        _os.environ["AUTOSLM_SKIP_NET"] = "1"
    rates = live_rates()

    def fmt_rate(v: float | None) -> str:
        return f"{v:>10.2f}" if v else f"{'-':>10}"

    print(f"{'gpu':<16}{'vram':>6}{'runpod$/hr':>11}  validated_on")
    for info in sorted(GPU_INFO.values(), key=lambda g: rates.get(g.name, g.hourly_usd)):
        runpod_rate = rates.get(info.name, info.hourly_usd) if info.enum_member else None
        validated = ",".join(info.validated_on) or "- (needs gpu.allow_unvalidated)"
        print(f"{info.name:<16}{info.vram_gb:>5}G{fmt_rate(runpod_rate):>11}  {validated}")
    print(
        '\nTip: omit gpu.type (or set "cheapest") to allocate the cheapest validated class\n'
        "that fits the model."
    )
    return 0


def cmd_env_init(args) -> int:
    mod = args.name.replace("-", "_")
    root = Path("environments") / mod
    root.mkdir(parents=True, exist_ok=True)
    # Verifiers-only: scaffold a real verifiers env whose load_environment returns a
    # vf.Environment (here a SingleTurnEnv + Rubric over a datasets.Dataset). This is what
    # the local-`path` loader (registry.load_environment -> VerifiersEnvironment) and a Hub
    # push both expect, so a freshly scaffolded env actually loads.
    (root / f"{mod}.py").write_text(
        f'"""Custom LOCAL verifiers environment ({args.name}).\n\n'
        'Run it locally via [environment] path = "environments/'
        f'{mod}/{mod}.py" in your config.\n'
        "Replace the dataset and rubric with your task, then publish to the Prime Hub with\n"
        "`slm env push` to reference it by id on the managed service.\n"
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
        f'run it locally:  [environment]\\npath = "environments/{mod}/{mod}.py"\n'
        "or publish it to the Prime Hub with `slm env push` and reference it by id "
        "(required on the managed service)."
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
        # modules (environments/<name>.py, e.g. the `slm lab` starter env). Print the
        # actual path so it can be pasted straight into [environment] path.
        paths: list[str] = []
        for p in local.iterdir():
            if p.name.startswith("__"):
                continue
            if p.is_dir():
                # The loader (and `slm env init`) maps a hyphenated dir to an underscored
                # inner module file (my-env/ -> my-env/my_env.py). List that exact path, and
                # only when it actually exists (an empty/incomplete folder isn't loadable).
                stem = p.name.replace("-", "_")
                module = p / f"{stem}.py"
                if module.is_file():
                    paths.append(f"environments/{p.name}/{stem}.py")
            elif p.suffix == ".py":
                paths.append(f"environments/{p.name}")
        if paths:
            print("local (set [environment] path to one of):")
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
    package = args.package
    extra_index = getattr(args, "extra_index_url", None)

    # Prefer the `prime` CLI when present (it resolves the Hub + index), unless the user
    # explicitly passed a pip --package/--extra-index-url (then use pip so the flag works).
    if not package and not extra_index and shutil.which("prime"):
        cmd = ["prime", "env", "install", env_id]
        print("running:", " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print("install failed")
            return rc
        # owner/name slugs live on the Prime Hub index; record it so the managed
        # worker can reinstall the wheel (worker_pip_for_env forwards it).
        extras = {"extra_index_url": PRIME_HUB_INDEX} if "/" in env_id else None
        record_installed_env(env_id, package=_bare_wheel_name(env_id), extras=extras)
    else:
        # A Hub slug `owner/name` maps to the pip wheel `name`; such slugs live on the Prime
        # Intellect Hub index by default (override with --extra-index-url).
        pkg = package or _bare_wheel_name(env_id)
        if extra_index is None and "/" in env_id:
            extra_index = PRIME_HUB_INDEX
        installer = (
            # `uv pip install` outside an active venv errors with "No virtual
            # environment found"; --python targets the CLI's own interpreter so a
            # global/pipx `slm` install still records the env.
            ["uv", "pip", "install", "--python", sys.executable]
            if shutil.which("uv")
            else [sys.executable, "-m", "pip", "install"]
        )
        cmd = [*installer, pkg]
        if extra_index:
            cmd += ["--extra-index-url", extra_index]
        print("running:", " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print("install failed")
            return rc
        extras = {"extra_index_url": extra_index} if extra_index else None
        record_installed_env(env_id, package=pkg, extras=extras)
    print(f"installed {env_id}; recorded in ~/.autoslm/envs.json")
    print(f'use it via:  [environment]\\nid = "{env_id}"')
    return 0


def cmd_env_push(args) -> int:
    import shutil
    import subprocess

    if shutil.which("prime"):
        return subprocess.run(["prime", "env", "push", args.path]).returncode
    print("the `prime` CLI is required to publish to the Environments Hub.")
    print("install it (https://docs.primeintellect.ai) then re-run `slm env push`.")
    return 1


def _check_managed_spec(spec: JobSpec) -> None:
    if spec.environment.path:
        raise ClientError(
            "local environment paths ([environment] path = ...) are not supported on the "
            "managed service; publish the environment with `slm env push` and reference it "
            'by id ([environment] id = "owner/name")'
        )


def cmd_train(args) -> int:
    spec = spec_from_file(
        args.config,
        run_id=new_run_id() if args.dry_run else None,
        overrides=getattr(args, "overrides", None),
        extra_configs=getattr(args, "extra_configs", None),
    )
    if args.dry_run:
        # Fully local: validate the config without credentials, a server, or a GPU. A local
        # [environment] path is legitimate here (the env runs client-side) — only a real
        # managed submit rejects it, so the managed check stays out of this branch.
        print(
            json.dumps(
                {"run_id": spec.run_id, "state": "dry_run", "spec": spec.to_dict()}, indent=2
            )
        )
        return 0
    _check_managed_spec(spec)
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


def cmd_serve_proxy(args) -> int:
    from autoslm.serve.proxy import run_proxy

    client = client_from_config()
    print(f"OpenAI-compatible proxy for {args.run_id} -> http://{args.host}:{args.port}/v1")
    run_proxy(client=client, run_id=args.run_id, host=args.host, port=args.port)
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


def cmd_server(args) -> int:
    from autoslm.server.app import run_server

    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
