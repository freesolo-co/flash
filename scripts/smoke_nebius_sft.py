"""Paid Nebius Spot SFT and durable checkpoint resume qualification.

Set FLASH_NEBIUS_* and authenticate Nebius and Azure artifact storage, then run:
    python scripts/smoke_nebius_sft.py --run-paid-test --repo owner/private-test-dataset
The first VM is terminated after checkpoint 1 becomes durable. A replacement
resumes and finishes step 2. Each attempt is limited to 45 minutes; only this
run's VMs are deleted.
"""

import argparse
import json
import sys
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

from smoke_modal_sft import prepare

from flash.artifacts.api import ArtifactStore
from flash.providers._lifecycle.net.worker import publish_source_snapshot
from flash.providers.nebius import api, jobs
from flash.providers.nebius.config import configured
from flash.runner.lifecycle.state import RunStatus, _save_status
from flash.runner.lifecycle.status import _update

ATTEMPT_SECONDS = 2700


def checkpoint_is_durable(hub, repo, run_id, step):
    # Flash commits the complete folder atomically after all LFS uploads finish.
    prefix = f"sft/{run_id}/checkpoint/checkpoint-{step}/"
    required = {
        prefix + name
        for name in (
            "model_world_size_1_rank_0.pt",
            "optim_world_size_1_rank_0.pt",
            "data_0.pt",
            "lora_train_meta.json",
        )
    }
    return required.issubset(hub.list_repo_files(repo, repo_type="dataset"))


def resume_attempt(spec, snapshot, on_handle):
    result = jobs.submit_attempt(
        spec,
        log=sys.stdout,
        attempt=1,
        source_snapshot=snapshot,
        on_handle=on_handle,
        deadline_at=time.time() + ATTEMPT_SECONDS,
    )
    assert result.ok, result
    assert result.metrics.get("notes", {}).get("resumed") is True, result.metrics
    assert result.metrics["train_tokens"] > 0, result.metrics
    return result


def run(repo):
    if not configured():
        raise ValueError("Authenticate Nebius and configure FLASH_NEBIUS_* first")
    hub = ArtifactStore()
    if not hub.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Use a disposable private dataset repository")
    run_id = "flash-spot-sft-smoke-" + uuid.uuid4().hex[:12]
    print(json.dumps({"run_id": run_id}), flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="flash-spot-sft-") as directory:
            root = Path(directory) / "environment"
            root.mkdir()
            spec = prepare(hub, repo, run_id, "", root, "H100", provider="spot")
            spec = replace(spec, gpu=replace(spec.gpu, max_wall_seconds=ATTEMPT_SECONDS))
            _save_status(RunStatus(run_id=run_id, state="queued", spec=spec.to_dict()))
            snapshot = publish_source_snapshot(repo)

            def persisted(handle):
                _update(run_id, "running", remote=handle.to_dict())
                print(json.dumps({"instance": handle.to_dict()["instance_id"]}), flush=True)

            stop = Event()
            durable = Event()
            monitor_errors = []
            monitor = None

            def interrupt_after_checkpoint(handle):
                try:
                    while not stop.is_set():
                        if checkpoint_is_durable(hub, repo, run_id, 1):
                            durable.set()
                            print("Checkpoint 1 durable; simulating Spot interruption", flush=True)
                            api.terminate(handle.to_dict()["instance_id"])
                            return
                        stop.wait(5)
                except Exception as exc:
                    monitor_errors.append(exc)

            def first_handle(handle):
                nonlocal monitor
                persisted(handle)
                monitor = Thread(target=interrupt_after_checkpoint, args=(handle,), daemon=True)
                monitor.start()

            try:
                first = jobs.submit_attempt(
                    spec,
                    log=sys.stdout,
                    attempt=0,
                    source_snapshot=snapshot,
                    on_handle=first_handle,
                    deadline_at=time.time() + ATTEMPT_SECONDS,
                )
            finally:
                stop.set()
                if monitor is not None:
                    monitor.join(timeout=190)
                    if monitor.is_alive():
                        raise TimeoutError("Spot interruption cleanup did not finish")
            if monitor_errors:
                raise monitor_errors[0]
            assert durable.is_set(), "First checkpoint did not become durable"
            assert not first.ok, first
            assert first.failure == "job_preempted", first
            resume_attempt(spec, snapshot, persisted)
            assert checkpoint_is_durable(hub, repo, run_id, 2)
            _update(run_id, "done")
            print(
                json.dumps({"training": "passed", "resume": "passed", "run_id": run_id}), flush=True
            )
    except Exception as exc:
        _update(run_id, "failed", error=type(exc).__name__)
        raise
    finally:
        for instance in api.list_instances():
            if instance["metadata"].get("labels", {}).get("flash-run") == run_id:
                api.terminate(instance["metadata"]["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run-paid-test", action="store_true", required=True)
    run(parser.parse_args().repo)
