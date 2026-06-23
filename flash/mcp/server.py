"""Minimal stdio MCP-style bridge for coding agents.

This intentionally avoids a hard dependency on a specific MCP SDK while exposing
the stable JSON tools that agents need. Requests are newline-delimited JSON:
{"tool": "list_models", "args": {...}}.

Run-lifecycle tools call the managed Flash control plane with the same stored
credentials as the CLI (`flash login`); dry-run validation stays local.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

from flash.catalog import public_model_rows
from flash.client import client_from_config
from flash.client.runtime_secrets import runtime_secrets_from_local_env
from flash.client.specs import spec_payload
from flash.schema import spec_from_dict


def list_models(args: dict) -> dict:
    return {"models": public_model_rows()}


def create_train_run(args: dict) -> dict:
    spec = spec_from_dict(args, run_id=args.get("run_id"))
    if args.get("dry_run"):
        # Fully local: validate without credentials, a server, or a GPU.
        return {"run_id": spec.run_id, "state": "dry_run", "spec": spec.to_dict()}
    return client_from_config().create_run(
        spec_payload(spec),
        runtime_secrets=runtime_secrets_from_local_env(keys=spec.environment.secrets),
    )


def get_run_status(args: dict) -> dict:
    return client_from_config().get_run(args["run_id"])


def get_run_logs(args: dict) -> dict:
    page = client_from_config().get_logs(args["run_id"], offset=int(args.get("offset", 0)))
    return {"run_id": args["run_id"], **{k: page[k] for k in ("logs", "offset", "state")}}


def deploy_adapter_tool(args: dict) -> dict:
    return client_from_config().deploy(
        args["run_id"],
        dry_run=bool(args.get("dry_run", False)),
    )


TOOLS: dict[str, Callable[[dict], dict]] = {
    "list_models": list_models,
    "create_training_run": create_train_run,
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
