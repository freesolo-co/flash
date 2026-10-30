"""Run a user's local Python entry point against a normal managed Flash run."""

import asyncio
import inspect
import json
import runpy
import sys
import time
import uuid
from pathlib import Path

from flash.client import client_from_config
from flash.client.http import ApiError, ClientError
from flash.client.specs import spec_payload
from flash.custom.async_client import AsyncTrainingClient
from flash.custom.client import TrainingClient


def submit_custom(api, payload):
    """Reconcile a lost creation response without ever repeating GPU submission."""
    identity = uuid.uuid4().hex
    payload = {**payload, "custom": {**payload["custom"], "controller_id": identity}}
    # source publication and pinned-checkpoint validation can exceed the general HTTP timeout.
    api.timeout = max(api.timeout, 300)
    try:
        return api.create_run(payload)
    except ClientError as error:
        if isinstance(error, ApiError) and error.submission_outcome == "not-created":
            raise
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                runs = api._request("GET", "/v1/runs", timeout=10)["runs"]
                matches = [
                    run
                    for run in runs
                    if run.get("spec", {}).get("custom", {}).get("controller_id") == identity
                ]
                if len(matches) == 1:
                    return matches[0]
            except ClientError:
                pass
            time.sleep(2)
        raise ClientError(
            "Custom submission outcome is unknown; no second run was submitted. "
            f"Inspect `flash runs list` for controller_id {identity} before launching again."
        ) from error


def connect(api, run_id, *, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            connection = api._request("GET", f"/v1/runs/{run_id}/custom")
            try:
                client = TrainingClient(
                    connection["url"], api_key=connection["api_key"], timeout=10
                )
                client._http.timeout = 300
                client.run_id = run_id
                return client
            except ClientError:
                pass  # tunnel can exist before model loading finishes.
        except ApiError as exc:
            if exc.status != 425:
                raise
        time.sleep(2)
    raise TimeoutError(f"custom run {run_id} did not become ready")


def run_custom(args, spec, authored_train_keys):
    if getattr(args, "background", False):
        raise ValueError(
            "custom algorithms run locally; keep this process running (for example in tmux)"
        )
    path = (Path(args.config).resolve().parent / spec.custom["file"]).resolve()
    source = path.read_text()
    compile(source, str(path), "exec")
    api = client_from_config()
    preview = bool(args.dry_run or getattr(args, "cost", False))
    if preview:
        result = api.create_run(spec_payload(spec, authored_train_keys), dry_run=True)
        print(json.dumps(result, indent=2))
        return 0
    # import before renting a GPU, with sibling modules available to the user's file.
    sys.path.insert(0, str(path.parent))
    try:
        namespace = runpy.run_path(str(path), run_name="__flash_algorithm__")
        entry = namespace.get("main")
        if not callable(entry):
            raise ValueError("custom.file must define main(client) or async def main(client)")
        inspect.signature(entry).bind(None)
        status = submit_custom(api, spec_payload(spec, authored_train_keys))
        run_id = status["run_id"]
        print(f"Custom run: {run_id}", flush=True)
        try:
            client = connect(api, run_id, timeout=min(1200, spec.custom["max_wall_seconds"]))
            client.config = spec.to_dict()
            if inspect.iscoroutinefunction(entry):
                asyncio.run(_run_async(entry, client))
            else:
                entry(client)
                client.close()
        except BaseException:
            # a local exception is a cancelled run, never a falsely successful worker close.
            api.cancel_run(run_id)
            raise
        from flash.cli.commands.ops.runs import _follow_run

        return _follow_run(api, run_id)
    finally:
        sys.path.remove(str(path.parent))


async def _run_async(entry, client):
    async_client = AsyncTrainingClient(client)
    try:
        await entry(async_client)
        await async_client.close()
    finally:
        async_client._closing = True
        async_client._executor.shutdown(wait=False)
