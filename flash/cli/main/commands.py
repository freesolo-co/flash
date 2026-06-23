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


def cmd_version(args) -> int:
    print(f"flash {__version__}")
    return 0


def cmd_login(args) -> int:
    # Login is handled by the freesolo backend (not the flash control plane): the user
    # supplies the freesolo API key they created in the dashboard, and we verify it against
    # freesolo before storing it. The same key authenticates flash's control plane.
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
    _ = save_credentials(api_key, api_url=api_url)
    # Never echo the key itself; the stored file is the single source of truth.
    print("logged in with your freesolo key")
    return 0


def cmd_whoami(args) -> int:
    print(json.dumps(client_from_config().me(), indent=2))
    return 0


_STARTER_ENV_PY = '''\
"""Starter local Freesolo environment.

Replace the dataset and reward with your task, then upload it with
`flash env push environments/starter_env.py`. A managed run references the
returned environment id in [environment] id.
"""

from freesolo.datasets.types import TaskExample
from freesolo.environments import EnvironmentSingleTurn, RewardResult


DATASET = [
    {"task": "What is 2 + 2?", "expected_output": "4"},
    {"task": "What is 3 + 5?", "expected_output": "8"},
]


class StarterEnv(EnvironmentSingleTurn):
    dataset = DATASET

    def build_prompt_messages(self, example: TaskExample, prompt_text: str):
        return [{"role": "user", "content": example.task}]

    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:
        expected = str(example.expected_output or "").strip()
        score = 1.0 if expected and expected in response_text else 0.0
        return RewardResult(score=score, threshold=1.0)


def load_environment(**kwargs) -> StarterEnv:
    return StarterEnv()
'''


def cmd_lab_setup(args) -> int:
    Path("environments").mkdir(exist_ok=True)
    Path("configs").mkdir(exist_ok=True)
    Path("configs/endpoints.toml").write_text(
        "# OpenAI-compatible endpoints returned by `flash deploy` can be stored here.\n"
    )
    starter_env = Path("environments/starter_env.py")
    if not starter_env.exists():
        starter_env.write_text(_STARTER_ENV_PY)
    sample = Path("configs/freesolo_grpo.toml")
    if not sample.exists():
        sample.write_text(
            'model = "Qwen/Qwen3.5-4B"\n'
            'algorithm = "grpo"\n\n'
            "# Environment: a Freesolo environment ref. Publish the scaffolded\n"
            "# environments/starter_env.py with `flash env push environments/starter_env.py`\n"
            "# to get the id, and set it below.\n"
            "[environment]\n"
            'id = "github:owner/repo@main:path/to/freesolo/environment.py"\n\n'
            "[train]\n"
            "steps = 150\n"
            "lora_rank = 32\n"
            "seeds = [0]\n"
            "# GPU and the HF artifact repo are managed automatically by the platform: the GPU is\n"
            "# the cheapest fitting class across providers, and each run gets its own artifact repo.\n"
        )
    print(
        "created environments/, environments/starter_env.py, configs/, "
        "configs/freesolo_grpo.toml, configs/endpoints.toml"
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
    """List GPU classes, VRAM, and per-provider $/hr."""
    from flash.providers import available_providers
    from flash.providers.base import GPU_INFO
    from flash.providers.runpod.pricing import live_rates

    rates = live_rates()
    # Cheapest live verified-datacenter offer per class (vast key + network only).
    vast_rates: dict[str, float] = {}
    if "vast" in available_providers():
        try:
            from flash.providers.vast.jobs import usable_offers

            for offer in usable_offers(0, 0):
                vast_rates.setdefault(offer.gpu, offer.dph_total)  # offers are price-sorted
        except Exception as exc:
            print(f"warning: vast offers unavailable ({exc})", file=sys.stderr)

    def fmt_rate(v: float | None) -> str:
        return f"{v:>10.2f}" if v else f"{'-':>10}"

    print(f"{'gpu':<16}{'vram':>6}{'runpod$/hr':>11}{'vast$/hr':>10}")
    for info in sorted(GPU_INFO.values(), key=lambda g: rates.get(g.name, g.hourly_usd)):
        runpod_rate = rates.get(info.name, info.hourly_usd) if info.enum_member else None
        print(
            f"{info.name:<16}{info.vram_gb:>5}G{fmt_rate(runpod_rate):>11}"
            f"{fmt_rate(vast_rates.get(info.name))}"
        )
    print(
        "\nTip: GPU class selection is fully automatic — the submit-time allocator always picks the\n"
        "cheapest live-validated class that fits the model across all providers, so you don't pin a\n"
        "GPU type. You can still tune the run via the [gpu] config table (disk_gb, max_wall_seconds,\n"
        "max_retries, network_volume / network_volume_gb, datacenter)."
    )
    return 0


def cmd_env_init(args) -> int:
    mod = args.name.replace("-", "_")
    root = Path("environments") / mod
    root.mkdir(parents=True, exist_ok=True)
    # Scaffold a real Freesolo SDK environment whose load_environment returns an
    # EnvironmentSingleTurn.
    (root / f"{mod}.py").write_text(
        f'"""Custom Freesolo environment ({args.name}).\n\n'
        "Replace the dataset and reward with your task, then upload it\n"
        f"with `flash env push environments/{mod}/{mod}.py` and reference the returned id\n"
        "in your config.\n"
        '"""\n\n'
        "from freesolo.datasets.types import TaskExample\n"
        "from freesolo.environments import EnvironmentSingleTurn, RewardResult\n\n\n"
        "DATASET = [\n"
        '    {"task": "What is 2 + 2?", "expected_output": "4"},\n'
        '    {"task": "What is 3 + 5?", "expected_output": "8"},\n'
        "]\n\n\n"
        "class CustomEnv(EnvironmentSingleTurn):\n"
        "    dataset = DATASET\n\n"
        "    def build_prompt_messages(self, example: TaskExample, prompt_text: str):\n"
        '        return [{"role": "user", "content": example.task}]\n\n'
        "    def score_response(self, example: TaskExample, response_text: str) -> RewardResult:\n"
        '        expected = str(example.expected_output or "").strip()\n'
        "        score = 1.0 if expected and expected in response_text else 0.0\n"
        "        return RewardResult(score=score, threshold=1.0)\n\n\n"
        "def load_environment(**kwargs) -> CustomEnv:\n"
        "    return CustomEnv()\n"
    )
    (root / "README.md").write_text(f"# {args.name}\n\nCustom Freesolo environment for Flash.\n")
    print(f"created {root}")
    print(
        f"upload it with `flash env push environments/{mod}/{mod}.py`, then reference "
        "the returned id in your config."
    )
    return 0


def cmd_env_list(args) -> int:
    from flash.envs.registry import list_installed_environments

    installed = list_installed_environments()
    if installed:
        print("installed environments:")
        for env_id in installed:
            print(f"  {env_id}")
    local = Path("environments")
    if local.is_dir():
        # Both directory envs (environments/<name>/<name>.py) and top-level single-file
        # modules (environments/<name>.py, e.g. the `flash lab` starter env). These are local
        # env SOURCES — publish one with `flash env push <path>` to run it on the managed
        # service by its environment id.
        paths: list[str] = []
        for p in local.iterdir():
            if p.name.startswith("__"):
                continue
            if p.is_dir():
                # `flash env init` maps a hyphenated dir to an underscored inner module file
                # (my-env/ -> my-env/my_env.py). List that exact path, and only when it
                # actually exists (an empty/incomplete folder isn't a publishable source).
                stem = p.name.replace("-", "_")
                module = p / f"{stem}.py"
                if module.is_file():
                    paths.append(f"environments/{p.name}/{stem}.py")
            elif p.suffix == ".py":
                paths.append(f"environments/{p.name}")
        if paths:
            print("local env sources (publish with `flash env push <path>`):")
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
        runtime_secrets=runtime_secrets_from_local_env(args.config),
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
        f"run {run_id} submitted; following logs (Ctrl-C detaches, `flash attach {run_id}` resumes)",
        file=sys.stderr,
    )
    return _follow_run(client, run_id)


def _poll_logs(client: ApiClient, run_id: str, interval: float) -> str:
    """Stream offset-paged logs until the run reaches a terminal state; return that state."""
    offset = 0
    while True:
        page = client.get_logs(run_id, offset=offset)
        if page["logs"]:
            print(page["logs"], end="", flush=True)
        offset = page["offset"]
        if page["state"] in _CLI_DONE_STATES:
            return page["state"]
        time.sleep(interval)


def _follow_run(client: ApiClient, run_id: str) -> int:
    """Poll logs until the run reaches a terminal state, then print the final status."""
    state = _poll_logs(client, run_id, interval=2.0)
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
    _poll_logs(client, args.run_id, interval=1.0)
    return 0


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
            f"(~${dep.get('est_idle_cost_usd_per_day')}/day). Use `flash undeploy {args.run_id}` "
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
