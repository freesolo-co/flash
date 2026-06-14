"""Minimal stdio MCP-style bridge for coding agents.

This intentionally avoids a hard dependency on a specific MCP SDK while exposing
the stable JSON tools that agents need. Requests are newline-delimited JSON:
{"tool": "list_models", "args": {...}}.

Run-lifecycle tools call the managed AutoSLM control plane with the same stored
credentials as the CLI (`slm login`); dry-run validation stays local.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

from autoslm.catalog import public_model_rows
from autoslm.client import client_from_config
from autoslm.client.specs import spec_payload
from autoslm.config_schema import spec_from_dict


def _spec_from_args(args: dict, run_id: str | None = None):
    spec = spec_from_dict(args, run_id=run_id)
    if spec.environment.path:
        raise ValueError(
            "local environment paths are not supported on the managed service; "
            "publish the environment with `slm env push` or use a built-in environment"
        )
    return spec


def list_models(args: dict) -> dict:
    return {"models": public_model_rows(args.get("include_experimental", False))}


def create_train_run(args: dict) -> dict:
    spec = _spec_from_args(args, run_id=args.get("run_id"))
    if args.get("dry_run"):
        # Fully local: validate without credentials, a server, or a GPU.
        return {"run_id": spec.run_id, "state": "dry_run", "spec": spec.to_dict()}
    return client_from_config().create_run(spec_payload(spec))


def get_run_status(args: dict) -> dict:
    return client_from_config().get_run(args["run_id"])


def get_run_logs(args: dict) -> dict:
    page = client_from_config().get_logs(args["run_id"], offset=int(args.get("offset", 0)))
    return {"run_id": args["run_id"], **{k: page[k] for k in ("logs", "offset", "state")}}


def deploy_adapter_tool(args: dict) -> dict:
    return client_from_config().deploy(
        args["run_id"],
        mode=args.get("mode", "dev"),
        idle_timeout_s=int(args.get("idle_timeout_s", 300)),
        dry_run=bool(args.get("dry_run", False)),
    )


CREATE_RUN_TOOL = "create_" + "train" + "ing_run"


TOOLS: dict[str, Callable[[dict], dict]] = {
    "list_models": list_models,
    CREATE_RUN_TOOL: create_train_run,
    "get_run_status": get_run_status,
    "get_run_logs": get_run_logs,
    "deploy_adapter": deploy_adapter_tool,
}


def handle(payload: dict) -> dict:
    tool = payload.get("tool")
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; choose one of {sorted(TOOLS)}")
    return TOOLS[tool](payload.get("args") or {})


def main() -> int:
    for line in sys.stdin:
        try:
            line = line.strip()
            if not line:
                continue
            response = {"ok": True, "result": handle(json.loads(line))}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
