"""Native Flash driver: train, evaluate checkpoint curves, or exercise concurrent adapters.

    python -m flash.benchmarks.managed /absolute/trial/request.json

The environment-specific hooks module prepares/attests cache state and collects worker spans
and provider receipts. Those facts cannot be recovered from a public polling response. A hook
may also supply evaluate(context, targets) for tool-using agents or another held-out grader.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

from flash.benchmarks.evaluation import evaluate_targets
from flash.benchmarks.evidence import Recorder, write_json
from flash.benchmarks.schema import Case, Complete, Quality
from flash.client import ApiClient


@dataclass
class Context:
    request: dict
    directory: Path
    recorder: Recorder
    case: Case
    run_id: str | None = None
    submission_pending: bool = False
    owned_deployments: list[str] = field(default_factory=list)

    @property
    def config(self):
        return self.case.configuration


def _client(config: dict) -> ApiClient:
    return ApiClient(config["api_url"], os.environ[config["credential_env"]], timeout=30.0)


def _evaluate(context: Context, hooks: ModuleType, targets: list[tuple], checkpoint: str) -> None:
    with context.recorder.span("evaluation", source="benchmark evaluator wall clock"):
        if hasattr(hooks, "evaluate"):
            value, samples = hooks.evaluate(context, targets)
        else:
            value, samples = evaluate_targets(
                targets,
                context.config["episodes"],
                context.recorder,
                max_tokens=context.config.get("max_tokens", 128),
            )
    context.recorder.emit(
        Quality(
            at_s=context.recorder.now(),
            metric=context.case.target.metric,
            value=float(value),
            samples=samples,
            checkpoint=checkpoint,
        )
    )


def _deploy_and_evaluate(context, hooks, client, checkpoint):
    # Only this trial's freshly submitted run is deployed by the driver; existing targets belong
    # to prepare/cleanup hooks and are never implicitly undeployed.
    context.owned_deployments.append(checkpoint)
    write_json(
        context.directory / "resources.json",
        {
            "run_id": context.run_id,
            "deployments": context.owned_deployments,
        },
    )
    deployment = client.deploy(checkpoint)
    while deployment.get("state") == "deploying":
        time.sleep(context.config.get("poll_s", 5.0))
        current = client.deployment_for(checkpoint)
        if current is None:
            raise ValueError("benchmark deployment disappeared")
        if current.get("generation") != deployment.get("generation"):
            raise ValueError("benchmark deployment generation changed")
        deployment = current
    if deployment.get("state") != "ready" or deployment.get("checkpoint_id") != checkpoint:
        raise ValueError("benchmark checkpoint did not become ready")
    _evaluate(context, hooks, [(client, checkpoint, "training-customer")], checkpoint)


def _train(context: Context, hooks: ModuleType, client: ApiClient) -> None:
    spec = {**context.config["training_spec"], "seed": context.request["seed"]}
    expected = "sft" if context.case.workload == "short_sft" else "grpo"
    if spec.get("algorithm") != expected:
        raise ValueError(f"{context.case.workload} requires algorithm={expected}")
    if spec.get("model") != context.case.revisions.model:
        raise ValueError("training model differs from benchmark revision lock")
    context.submission_pending = True
    write_json(context.directory / "submission.json", {"state": "pending", "spec": spec})
    status = client.create_run(spec)
    context.run_id = status["run_id"]
    context.submission_pending = False
    write_json(
        context.directory / "submission.json", {"state": "acknowledged", "run_id": context.run_id}
    )
    write_json(context.directory / "resources.json", {"run_id": context.run_id, "deployments": []})
    evaluated = set()
    with (context.directory / "status.jsonl").open("a") as stream:
        while True:
            stream.write(json.dumps({"at_s": context.recorder.now(), "status": status}) + "\n")
            stream.flush()
            if status["state"] in {"failed", "cancelled", "dry_run"}:
                raise ValueError(f"benchmark training ended in {status['state']}")
            for item in client.checkpoints(context.run_id):
                checkpoint = f"{context.run_id}/step-{item['step']}"
                if checkpoint not in evaluated:
                    _deploy_and_evaluate(context, hooks, client, checkpoint)
                    evaluated.add(checkpoint)
            if status["state"] == "done":
                _deploy_and_evaluate(context, hooks, client, f"{context.run_id}/final")
                return
            time.sleep(context.config.get("poll_s", 5.0))
            status = client.get_run(context.run_id)


def _cleanup(context, hooks, client) -> None:
    errors = ["unacknowledged_submission"] if context.submission_pending else []
    if context.run_id:
        try:
            status = client.get_run(context.run_id)
            if status["state"] not in {"done", "cancelled", "failed"}:
                client.cancel_run(context.run_id)
        except Exception as exc:
            errors.append(type(exc).__name__)
    for checkpoint in reversed(context.owned_deployments):
        try:
            client.undeploy(checkpoint)
        except Exception as exc:
            errors.append(type(exc).__name__)
    try:
        hooks.cleanup(context)
    except Exception as exc:
        errors.append(type(exc).__name__)
    write_json(context.directory / "cleanup.json", {"errors": errors})
    if errors:
        raise RuntimeError("benchmark cleanup incomplete; inspect resources.json and cleanup.json")


def run(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    if request["synthetic"]:
        raise ValueError("managed driver must not run under --smoke")
    case = Case.model_validate(request["case"])
    recorder = Recorder(request_path.parent / "events.jsonl", request["origin_monotonic_s"])
    context = Context(request, request_path.parent, recorder, case)
    hooks = importlib.import_module(context.config["hooks"])
    for name in ("prepare", "collect", "cleanup"):
        if not callable(getattr(hooks, name, None)):
            raise ValueError(f"benchmark hooks must implement {name}(context)")
    if case.workload == "long_multi_turn" and not callable(getattr(hooks, "evaluate", None)):
        raise ValueError("long multi-turn agents require an environment-aware evaluate hook")
    if not hasattr(hooks, "evaluate") and case.target.metric != "exact_match":
        raise ValueError("the default evaluator only measures exact_match")
    if hasattr(hooks, "validate"):
        hooks.validate(context)
    client = _client(context.config)
    succeeded = False
    ledgers = []
    try:
        hooks.prepare(context)
        if case.workload == "concurrent_adapters":
            targets = [
                (_client({**context.config, **target}), target["adapter"], target["customer"])
                for target in context.config["targets"]
            ]
            if (
                len({target[1] for target in targets}) < 2
                or len({target[2] for target in targets}) < 2
            ):
                raise ValueError("concurrent adapters require multiple customers and adapters")
            if {target[1] for target in targets} != set(case.revisions.adapters):
                raise ValueError("concurrent targets differ from the pinned adapter set")
            if len({target[0].api_key for target in targets}) < 2:
                raise ValueError("concurrent customers require distinct credentials")
            _evaluate(context, hooks, targets, "concurrent-adapter-set")
        else:
            _train(context, hooks, client)
        succeeded = True
    finally:
        try:
            try:
                _cleanup(context, hooks, client)
            finally:
                ledgers = hooks.collect(context)
        except BaseException:
            succeeded = False
            raise
        finally:
            recorder.emit(
                Complete(
                    at_s=recorder.now(),
                    outcome="succeeded" if succeeded else "failed",
                    complete_ledgers=ledgers,
                )
            )


def main() -> None:
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
