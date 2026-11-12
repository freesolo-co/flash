"""Paid Mithril Spot SFT and durable checkpoint resume qualification.

Set FLASH_MITHRIL_* and authenticate Mithril and Azure artifact storage, then run:
    python scripts/smoke_mithril_sft.py --run-paid-test --repo owner/private-test-dataset
The first bid is cancelled after checkpoint 1 becomes durable (it may finish step 2
before cancellation). A fresh VM restores the latest checkpoint and trains through step 4. Each attempt is limited to 45 minutes; only this
run's bids are cancelled.
"""

import argparse
import json
import sys
import tempfile
import time
import uuid
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from threading import Event, Thread

from smoke_modal_sft import prepare

from flash.artifacts.api import ArtifactStore
from flash.providers._lifecycle.net.worker import publish_source_snapshot
from flash.providers.mithril import api, jobs
from flash.providers.mithril.config import configured
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


def run(repo, gpu="H100", model="Qwen/Qwen3.5-9B", entry_price_cap=None):
    if not configured():
        raise ValueError("Authenticate Mithril and configure FLASH_MITHRIL_* first")
    from flash.providers.mithril.pricing import SPOT_GPUS

    if entry_price_cap is not None:
        cap = Decimal(str(entry_price_cap))
        if not cap.is_finite() or not 0 < cap <= SPOT_GPUS[gpu].keep_running_usd:
            raise ValueError("Test entry cap must be positive and no greater than the hold cap")
        # Explicit paid-test-only admission override; never alter the provider's hold cap.
        SPOT_GPUS[gpu] = replace(SPOT_GPUS[gpu], hourly_usd=cap)
    hub = ArtifactStore()
    hub.create_repo(repo, exist_ok=True, private=True)
    if not hub.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Use a disposable private dataset repository")
    run_id = "flash-spot-sft-smoke-" + uuid.uuid4().hex[:12]
    print(json.dumps({"run_id": run_id}), flush=True)
    _save_status(RunStatus(run_id=run_id, state="queued", spec={}))
    try:
        with tempfile.TemporaryDirectory(prefix="flash-spot-sft-") as directory:
            root = Path(directory) / "environment"
            root.mkdir()
            spec = prepare(hub, repo, run_id, "", root, gpu, model, provider="spot")
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
            if not first.ok:
                assert first.failure in {"job_preempted", "worker_retriable"}, first
            # Cancellation can race the tiny first run's final save. Extend the target and
            # reprofile, so a cold resume must execute new optimizer steps in either case.
            resume_root = Path(directory) / "resume-environment"
            resume_root.mkdir()
            spec = prepare(hub, repo, run_id, "", resume_root, gpu, model, provider="spot", steps=4)
            spec = replace(spec, gpu=replace(spec.gpu, max_wall_seconds=ATTEMPT_SECONDS))
            done = f"sft/{run_id}/DONE"
            if done in hub.list_repo_files(repo, repo_type="dataset"):
                hub.delete_file(repo_id=repo, repo_type="dataset", path_in_repo=done)
            _update(run_id, "queued", spec=spec.to_dict())
            resume_attempt(spec, snapshot, persisted)
            assert checkpoint_is_durable(hub, repo, run_id, spec.train.max_steps)
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
    parser.add_argument("--gpu", choices=("H100", "H200", "B200"), default="H100")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--entry-price-cap",
        type=Decimal,
        help="Paid-test admission override, bounded by the GPU hold cap",
    )
    args = parser.parse_args()
    run(args.repo, args.gpu, args.model, args.entry_price_cap)
