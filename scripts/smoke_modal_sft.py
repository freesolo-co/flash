"""Paid two-step SFT, artifact recovery, and checkpoint resume qualification on Modal.

Export MODAL_TOKEN_ID, MODAL_TOKEN_SECRET, and HF_TOKEN, then run:
    uv run python scripts/smoke_modal_sft.py --run-paid-test --repo owner/private-test-dataset

The dataset repo must already exist and be private. Test artifacts remain there for inspection.
Only this invocation's sandboxes and cache Volume are removed. Each attempt is capped at 30 minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from flash.core.spec import EnvironmentPackageSpec, EnvironmentSpec, GpuSpec, JobSpec, TrainSpec
from flash.engine.profiling.dataset_profile import _RawRecordEnvironment
from flash.engine.profiling.sft_workload import prepare_sft_workload
from flash.engine.profiling.tokenizer import load_control_plane_tokenizer
from flash.engine.profiling.workload_profile import sft_profile_input_digest
from flash.envs.loading.staged import (
    ResolvedEnvironmentSource,
    archive_path_for_digest,
    encode_manifest,
    manifest_path_for_digest,
    manifest_payload,
    write_environment_archive,
)
from flash.providers._lifecycle.net.worker import publish_source_snapshot
from flash.providers.modal import api, jobs
from flash.runner.supervise.recovery import _completed_worker_metrics

_ENVIRONMENT = """from freesolo.environments import EnvironmentSingleTurn, RewardResult
class SmokeEnvironment(EnvironmentSingleTurn):
    def build_prompt_messages(self, example, prompt_text):
        return [{"role": "user", "content": example.input}]
    def score_response(self, example, response_text):
        return RewardResult(score=1.0)
def load_environment(**kwargs):
    return SmokeEnvironment()
"""


def stage_environment(hub, repo, root):
    (root / "environment.py").write_text(_ENVIRONMENT)
    rows = [
        {
            "input": f"What is {i} plus one?",
            "output": [{"role": "assistant", "content": f"{i + 1}."}],
        }
        for i in range(4)
    ]
    (root / "dataset").mkdir()
    (root / "dataset/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    env_id = "smoke/modal/arithmetic"
    revision = hashlib.sha1(_ENVIRONMENT.encode()).hexdigest()
    source = ResolvedEnvironmentSource(env_id, revision, root, "environment.py")
    archive = root.parent / "environment.tar.gz"
    digest = write_environment_archive(source, archive)
    manifest = encode_manifest(manifest_payload(source, digest, archive_path_for_digest(digest)))
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    commit = hub.create_commit(
        repo_id=repo,
        repo_type="dataset",
        commit_message="Stage disposable Modal qualification environment",
        operations=[
            CommitOperationAdd(
                path_in_repo=archive_path_for_digest(digest), path_or_fileobj=str(archive)
            ),
            CommitOperationAdd(
                path_in_repo=manifest_path_for_digest(manifest_digest), path_or_fileobj=manifest
            ),
        ],
    )
    environment = EnvironmentSpec(
        id=env_id,
        resolved_sha=revision,
        package=EnvironmentPackageSpec(commit.oid, digest, manifest_digest),
    )
    return environment, rows


def prepare(hub, repo, run_id, volume, root, gpu):
    from importlib.metadata import version

    environment, rows = stage_environment(hub, repo, root)
    model = "Qwen/Qwen3.5-9B"
    revision = hub.model_info(model).sha
    producer = version("freesolo-flash")
    spec = JobSpec(
        run_id=run_id,
        model=model,
        model_revision=revision,
        model_revision_auto=True,
        algorithm="sft",
        thinking="off",
        environment=environment,
        train=TrainSpec(
            hf_repo=repo,
            max_steps=2,
            max_examples=4,
            batch_size=1,
            max_context_tokens=256,
            save_at_steps=(1, 2),
            lora_rank=8,
        ),
        gpu=GpuSpec(
            type=gpu, count=1, provider="modal", max_wall_seconds=1800, network_volume=volume
        ),
    )
    spec = replace(
        spec,
        workload_profile_input_digest=sft_profile_input_digest(
            spec, tokenizer_revision=revision, producer_version=producer
        ),
        workload_profile_producer_version=producer,
    )
    profile = prepare_sft_workload(
        spec,
        _RawRecordEnvironment(rows, root),
        tokenizer_loader=load_control_plane_tokenizer,
        producer_version=producer,
    )
    # Round-trip before renting compute: this is the same validation the worker performs.
    return JobSpec.from_dict(
        replace(spec, workload_profile=profile.profile.to_dict()).to_internal_dict()
    )


def run(repo, gpu):
    hub = HfApi()
    if not hub.repo_info(repo, repo_type="dataset").private:
        raise ValueError("qualification requires a disposable private dataset repository")
    run_id = "flash-modal-sft-smoke-" + uuid.uuid4().hex[:12]
    volume = run_id + "-cache"
    handles = []

    def persisted(handle):
        handles.append(handle.to_dict())
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "attempt": handles[-1]["attempt"],
                    "sandbox": handles[-1]["instance_id"],
                }
            ),
            flush=True,
        )

    try:
        with tempfile.TemporaryDirectory(prefix="flash-modal-sft-") as directory:
            root = Path(directory) / "environment"
            root.mkdir()
            spec = prepare(hub, repo, run_id, volume, root, gpu)
            snapshot = publish_source_snapshot(repo)
            first = jobs.submit_attempt(
                spec,
                attempt=0,
                source_snapshot=snapshot,
                on_handle=persisted,
                deadline_at=time.time() + 1800,
            )
            assert first.ok, first
            recovered = _completed_worker_metrics(handles[-1], deadline_at=time.time() + 60)
            assert recovered, "terminal metrics were not recovered"
            assert recovered["train_tokens"] > 0, recovered
            files = hub.list_repo_files(repo, repo_type="dataset")
            assert any(name.startswith(f"sft/{run_id}/") and "checkpoint" in name for name in files)
            # Simulate loss of the terminal commit after the checkpoint became durable.
            hub.delete_file(repo_id=repo, repo_type="dataset", path_in_repo=f"sft/{run_id}/DONE")
            resumed = jobs.submit_attempt(
                spec,
                attempt=1,
                source_snapshot=snapshot,
                on_handle=persisted,
                deadline_at=time.time() + 1800,
            )
            assert resumed.ok, resumed
            assert resumed.metrics.get("notes", {}).get("resumed") is True, resumed.metrics
            print(
                json.dumps(
                    {
                        "sft": "passed",
                        "artifact_recovery": "passed",
                        "checkpoint_resume": "passed",
                        "run_id": run_id,
                    }
                ),
                flush=True,
            )
    finally:
        for instance in api.list_instances():
            if instance["tags"].get("flash_run") == run_id:
                api.terminate(instance["id"])

        async def remove_volume(modal, client):
            await modal.Volume.objects.delete.aio(
                volume, allow_missing=True, environment_name=api.environment(), client=client
            )

        api.call(remove_volume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--gpu", choices=("L40S", "H100"), default="L40S")
    parser.add_argument("--run-paid-test", action="store_true", required=True)
    args = parser.parse_args()
    run(args.repo, args.gpu)
