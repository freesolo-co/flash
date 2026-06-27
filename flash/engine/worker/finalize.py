"""Train-metadata + run-metrics finalize for the worker."""

from __future__ import annotations

import json
import os

from flash.engine.accounting import RunMetrics
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import gpu_diagnostics


def write_train_meta(
    phase, adapter_dir, model_id, train_wall, setup_seconds, train_tokens, generated_tokens, notes
):
    env = _w.require_active_env()
    meta = {
        "phase": phase,
        "adapter_dir": adapter_dir,
        "model_id": model_id,
        "train_wall": train_wall,
        "setup_seconds": setup_seconds,
        "train_tokens": train_tokens,
        "generated_tokens": generated_tokens,
        "notes": notes or {},
    }
    with open("/tmp/train_meta.json", "w") as f:
        json.dump(meta, f)
    _w.hf_upload_file("/tmp/train_meta.json", "train_meta.json")
    _w.heartbeat(
        f"{phase}_train_done",
        **{k: meta[k] for k in ("train_wall", "train_tokens", "generated_tokens")},
        gpu=gpu_diagnostics(),
    )
    m = RunMetrics(
        arm=os.environ.get("FLASH_ARM", "runpod"),
        phase=phase,
        seed=_w.SEED,
        model_id=model_id,
        wall_seconds=train_wall,
        setup_seconds=setup_seconds,
        train_throughput_toks_per_s=(
            (generated_tokens or train_tokens) / train_wall if train_wall else 0.0
        ),
        train_tokens=train_tokens,
        generated_tokens=generated_tokens,
        notes={
            **(notes or {}),
            "renderer": "flash_env",
            "thinking": _w.THINKING,
            "train_wall": train_wall,
            "model_id": model_id,
            "environment": env.id,
            "job_spec": _w.JOB_SPEC.to_dict() if _w.JOB_SPEC else None,
        },
    )
    _w._finalize(m)
