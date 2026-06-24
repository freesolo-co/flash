"""FA3 de-risk probe: a SHORT Qwen3.5-0.8B SFT pinned to an H100 (Hopper sm90), on the
cu128-fa3 worker image, to confirm (1) FA3 builds/imports on the worker, (2) Qwen3.5's hybrid
arch composes with attn_implementation="flash_attention_3" without crashing on the GDN layers,
and (3) the worker logs '[attn] sm90 -> flash_attention_3'.

Run with PYTHONPATH pointing at THIS worktree so upload_code ships the FA3 perf.py to the worker,
and with FLASH_WORKER_IMAGE=...:cu128-fa3 so the H100 endpoint uses the FA3-baked image.
"""

import os
import sys

# The FA3-baked worker image (built by worker-image.yml with FLASH_ATTN_3_SPEC). The baked-image
# RunPod path reads FLASH_WORKER_IMAGE from os.environ at submit (providers/runpod/jobs.py).
os.environ["FLASH_WORKER_IMAGE"] = "ghcr.io/freesolo-co/flash-worker:cu128-fa3"
# Pin the rollout FA version too so the GRPO/serving path would also use FA3 (harmless for SFT).
os.environ.setdefault("VLLM_FLASH_ATTN_VERSION", "3")
# Upload the worker console on SUCCESS too (not just on crash) so we can read the
# '[attn] sm90 -> attn_implementation=flash_attention_3' line + HF Trainer timing.
os.environ.setdefault("FLASH_UPLOAD_CONSOLE", "1")

import os as _os

RUN_ID = _os.environ.get("FA3_RUN_ID", "fa3probeh100b")
MODEL = _os.environ.get("FA3_MODEL", "Qwen/Qwen3.5-0.8B")
MAX_EXAMPLES = int(_os.environ.get("FA3_MAX_EXAMPLES", "320"))
MAX_LENGTH = int(_os.environ.get("FA3_MAX_LENGTH", "4096"))  # long ctx -> attention is a real fraction
ENV_ID = _os.environ.get("FA3_ENV", "david-freesolo-co/input-compression")  # long-passage SFT data
CFG = {
    "model": MODEL,  # Qwen3.5 hybrid arch (has full-attention layers FA3 hits)
    "algorithm": "sft",
    "environment": {"id": ENV_ID},  # real env in freesolo-co/environment-hub
    "train": {
        "epochs": 1,
        "max_examples": MAX_EXAMPLES,
        "batch_size": 8,
        "max_length": MAX_LENGTH,
        "lora_rank": 16,
        "save_every": 1000,
        # hf_repo is platform-managed (control plane assigns Freesolo-Co/flashrun-<run_id>); a
        # user-set value here is ignored by spec_from_dict — we call _assign_managed_hf_repo below.
    },
    "wandb": {"project": "flash-fa3-probe", "run_name": RUN_ID},
}


def main() -> int:
    import flash

    print("flash pkg in use:", os.path.dirname(flash.__file__), flush=True)
    from flash.providers.runpod.jobs import submit_run
    from flash.providers.runpod.train import upload_code
    from flash.runner import _assign_managed_hf_repo
    from flash.runner.lifecycle import _spec_with_gpu
    from flash.schema import spec_from_dict

    spec = spec_from_dict(CFG, run_id=RUN_ID)
    # Assign the managed per-run artifact repo (Freesolo-Co/flashrun-<run_id>) exactly like the
    # runner does — spec_from_dict leaves train.hf_repo empty (control-plane-owned).
    spec = _assign_managed_hf_repo(spec)
    print("run_id:", spec.run_id, "| hf_repo:", spec.train.hf_repo, "| phase:", spec.phase, flush=True)
    # Ship THIS worktree's flash code (incl. engine/worker/perf.py FA3 selection) to the run repo.
    upload_code(spec.train.hf_repo)
    print("uploaded worktree code -> code/flash", flush=True)
    # Force Hopper (the allocator would otherwise route a 0.8B model to a cheap consumer card).
    spec = _spec_with_gpu(spec, "H100")
    print("pinned gpu:", spec.gpu.type, "| worker image:", os.environ["FLASH_WORKER_IMAGE"], flush=True)
    res = submit_run(spec, seed=0, log=sys.stderr)
    print("POLL RESULT:", res, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
