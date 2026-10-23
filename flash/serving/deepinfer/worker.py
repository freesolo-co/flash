"""One native scheduling loop per GPU; all four EP ranks stay in collectives."""

from __future__ import annotations

import os
import queue
import sys
import time
import traceback
from collections import defaultdict

from flash.serving.deepinfer.config import (
    GPU_COUNT,
    MAX_MODEL_LEN,
    RUNTIME_ROOT,
    SEQUENCES_PER_RANK,
)


def initialize(rank: int, model_path: str):
    # No torch/deepinfer imports before assigning the process's CUDA device.
    os.environ.update(
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "DEEPINFER_ACCELERATOR": "nvidia",
            "EP_DISABLE_GIN": "1",
            "DEEPINFER_ENGRAM_FORCE_GPU_STORAGE": "0",
            "DEEPINFER_USE_NEXTN_DECODE": "0",
            "DEEPINFER_ENABLE_MTP_VARLEN": "0",
            "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0",
            "DEEPINFER_BATCH_INVARIANT": "1",
            "DEEPINFER_PREFILL_STORE_EXTRA_STATE": "1",
            "DEEPINFER_PREFILL_STORE_EXTRA_SIZE": "0",
            "DEEPINFER_LOGGING_INTERVAL_STEP": "0",
        }
    )
    sys.path[:0] = [f"{RUNTIME_ROOT}/examples", RUNTIME_ROOT]
    from _config import create_llm, setup_common_env, setup_cuda_visible_devices

    setup_common_env(
        model_name=model_path, sampler_force_topk=0, set_override_experts=False, enable_eplb=False
    )
    setup_cuda_visible_devices(rank, 1, GPU_COUNT)
    from flash.serving.deepinfer.engram_transport import install

    install()
    return create_llm(
        model_path,
        rank,
        ep_size=GPU_COUNT,
        data_parallel_size=GPU_COUNT,
        max_num_batched_tokens=512,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=SEQUENCES_PER_RANK,
        gpu_memory_utilization=0.85,
        lookahead_tokens=0,
        disable_log_stats=True,
    )


def admit(engine, command, active, outgoing):
    import numpy as np
    from deepinfer.sampling_params import SamplingParams

    request_id = command["id"]
    if command["type"] == "cancel":
        engine.abort_request(request_id)
        active.discard(request_id)
        outgoing.put({"type": "cancelled", "id": request_id})
        return
    params = SamplingParams(**command["sampling"], eos_id=engine.eos_id)
    sequence = engine.add_request(
        user_id="flash",
        request_id=request_id,
        sampling_params=params,
        prompt_token_ids=np.asarray(command["tokens"], dtype=np.int32),
        multi_modal_data={"image_bytes": [], "image_sizes": []},
    )
    if sequence is None:
        raise RuntimeError("DeepInfer refused request admission")
    active.add(request_id)


def step(engine, active, outgoing):
    from deepinfer._C.engine import RequestOutput

    raw = engine.step() if engine.has_unfinished_requests() else engine.flush_inflight_reqs()
    grouped = defaultdict(list)
    finished = {}
    for pointer in raw:
        output = RequestOutput.clone(pointer)
        if output.request_id not in active:
            continue
        if output.token_id < 0 or (
            output.finished and str(output.finish_reason) not in {"stop", "length"}
        ):
            outgoing.put({"type": "error", "id": output.request_id})
            active.remove(output.request_id)
            grouped.pop(output.request_id, None)
            finished.pop(output.request_id, None)
            continue
        grouped[output.request_id].append(int(output.token_id))
        if output.finished:
            reason = str(output.finish_reason)
            finished[output.request_id] = reason
    for request_id, tokens in grouped.items():
        outgoing.put(
            {
                "type": "tokens",
                "id": request_id,
                "tokens": tokens,
                "finish_reason": finished.get(request_id),
            }
        )
        if request_id in finished:
            active.remove(request_id)


def run_rank(rank, model_path, incoming, outgoing, stopped):
    try:
        llm = initialize(rank, model_path)
        engine = llm.llm_engine
        outgoing.put({"type": "ready", "rank": rank})
        active = set()
        last_heartbeat = time.monotonic()
        while not stopped.is_set():
            # Bounded command draining leaves time for scheduling and other ranks.
            for _ in range(64):
                try:
                    command = incoming.get_nowait()
                except queue.Empty:
                    break
                admit(engine, command, active, outgoing)
            step(engine, active, outgoing)
            if time.monotonic() - last_heartbeat >= 1:
                outgoing.put({"type": "heartbeat", "rank": rank})
                last_heartbeat = time.monotonic()
            if not active:
                time.sleep(0.002)
    except BaseException:
        traceback.print_exc()
        outgoing.put({"type": "fatal", "rank": rank})
        stopped.set()
        raise
