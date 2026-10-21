"""Paid, opt-in Modal infrastructure smoke; always cleans its own sandboxes and Volume.

Run with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET set:
    uv run python scripts/smoke_modal_training.py --run-paid-test

Uses two L4 GPUs, checks CUDA/NCCL, duplicate adoption, persistent cache, and termination.
Full training/checkpoint tests additionally require an HF write token and a private dataset repo.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
import uuid
import zipfile

from flash.providers._lifecycle.net.worker import worker_image_for_gpu
from flash.providers.modal import api

_GPU_CHECK = '''import json,os,pathlib,subprocess,sys
payload=json.loads(os.environ.pop("FLASH_BOOTSTRAP_PAYLOAD"))
import torch
assert torch.cuda.device_count()==2
rank_code="""import sys,torch,torch.distributed as dist
rank=int(sys.argv[1])
torch.cuda.set_device(rank)
dist.init_process_group('nccl',init_method='tcp://127.0.0.1:29507',rank=rank,world_size=2)
x=torch.tensor([rank+1.0],device=f'cuda:{rank}')
dist.all_reduce(x)
assert x.item()==3.0
dist.destroy_process_group()
"""
workers=[subprocess.Popen([sys.executable,'-c',rank_code,str(rank)]) for rank in range(2)]
assert all(p.wait(timeout=90)==0 for p in workers)
pathlib.Path('/weight-cache/smoke-marker').write_text(payload['run_id'])
print('MODAL_SMOKE_OK: nccl and cache write',flush=True)
import time
time.sleep(120)
'''


def _capsule(code):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("__main__.py", code)
    return buffer.getvalue()


def run():
    run_id = "flash-live-smoke-" + uuid.uuid4().hex[:12]
    volume_name = run_id + "-cache"
    owned = []
    payload = {
        "run_id": run_id,
        "attempt": 0,
        "job_spec_json": "{}",
        "cache_volume": volume_name,
        "env": {"FLASH_WEIGHT_CACHE_DIR": "/weight-cache/hf-cache/hub"},
    }
    args = {
        "name": run_id + "-a0",
        "payload": payload,
        "image_reference": worker_image_for_gpu("L4"),
        "gpu": "L4",
        "gpu_count": 2,
        "capsule": _capsule(_GPU_CHECK),
        "deadline_at": time.time() + 900,
    }

    try:
        sandbox_id = api.create(**args)
        owned.append(sandbox_id)
        assert api.create(**args) == sandbox_id

        async def output(modal, client):
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id, client=client)
            async for line in sandbox.stdout:
                print(line, end="", flush=True)
                if "MODAL_SMOKE_OK" in line:
                    return True
            print(await sandbox.stderr.read.aio(), flush=True)
            return False

        assert api.call(output, timeout=180), "GPU/NCCL smoke failed"
        api.terminate(sandbox_id)
        assert api.get_instance(sandbox_id)["status"] == "terminated"

        async def verify_volume(modal, client):
            volume = modal.Volume.from_name(
                volume_name, environment_name=api.environment(), client=client
            )
            chunks = [chunk async for chunk in volume.read_file.aio("smoke-marker")]
            assert b"".join(chunks).decode() == run_id

        api.call(verify_volume)
        print(
            json.dumps(
                {
                    "cuda_nccl": "passed",
                    "persistent_cache": "passed",
                    "adoption": "passed",
                    "termination": "passed",
                }
            )
        )
    finally:
        # Include an acknowledged create whose response was lost before the ID was persisted.
        owned.extend(
            instance["id"]
            for instance in api.list_instances()
            if instance["tags"].get("flash_run") == run_id
        )
        for sandbox_id in set(owned):
            api.terminate(sandbox_id)

        async def delete_volume(modal, client):
            with contextlib.suppress(modal.exception.NotFoundError):
                await modal.Volume.objects.delete.aio(
                    volume_name, environment_name=api.environment(), client=client
                )

        api.call(delete_volume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-paid-test", action="store_true", required=True)
    parser.parse_args()
    run()
