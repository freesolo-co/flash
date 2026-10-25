"""One native scheduling loop per GPU; both EP ranks stay in collectives."""

from __future__ import annotations

import os
import queue
import sys
import time
import traceback
from collections import defaultdict

from flash.serve.runtime.token_batch import TokenBatch
from flash.serving.deepinfer.contract.config import (
    DELIVERY_SECONDS,
    DELIVERY_TOKENS,
    GPU_COUNT,
    MAX_MODEL_LEN,
    PREFILL_ANY_INTERVAL,
    PREFILL_TOKENS,
    RUNTIME_ROOT,
    SEQUENCES_PER_RANK,
)
from flash.serving.deepinfer.runtime.prefix_cache import PrefixCache


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
            "DEEPINFER_NUM_STATE_SLOTS_RATIO": "2",
            # Cache affinity/checkpoints leave unequal prefill queues across ranks.
            # The upstream 100-step fallback stalls those requests behind decoding.
            "DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL": str(PREFILL_ANY_INTERVAL),
        }
    )
    sys.path[:0] = [f"{RUNTIME_ROOT}/examples", RUNTIME_ROOT]
    from _config import create_llm, setup_common_env, setup_cuda_visible_devices

    setup_common_env(
        model_name=model_path, sampler_force_topk=0, set_override_experts=False, enable_eplb=False
    )
    setup_cuda_visible_devices(rank, 1, GPU_COUNT)
    from flash.serving.deepinfer.native.engram_transport import install

    install()
    return create_llm(
        model_path,
        rank,
        ep_size=GPU_COUNT,
        data_parallel_size=GPU_COUNT,
        max_num_batched_tokens=PREFILL_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=SEQUENCES_PER_RANK,
        gpu_memory_utilization=0.85,
        lookahead_tokens=0,
        disable_log_stats=True,
    )


def admit(engine, command, active, outgoing, cache, batches):
    from deepinfer.sampling_params import SamplingParams

    request_id = command["id"]
    if command["type"] == "cancel":
        engine.abort_request(request_id)
        active.discard(request_id)
        cache.forget(request_id)
        batches.pop(request_id, None)
        outgoing.put({"type": "cancelled", "id": request_id})
        return
    params = SamplingParams(**command["sampling"], eos_id=engine.eos_id)
    cache.admit(command, params)
    active.add(request_id)
    batches[request_id] = TokenBatch(DELIVERY_TOKENS, DELIVERY_SECONDS)


def step(engine, active, outgoing, cache, batches):
    from deepinfer._C.engine import RequestOutput

    raw = engine.step() if engine.has_unfinished_requests() else engine.flush_inflight_reqs()
    grouped = defaultdict(list)
    finished = {}
    outputs = [RequestOutput.clone(pointer) for pointer in raw]
    outputs.extend(cache.capture_ready())
    for output in outputs:
        if output.request_id not in active:
            continue
        if output.token_id < 0 or (
            output.finished and str(output.finish_reason) not in {"stop", "length"}
        ):
            outgoing.put({"type": "error", "id": output.request_id})
            active.remove(output.request_id)
            cache.forget(output.request_id)
            batches.pop(output.request_id, None)
            grouped.pop(output.request_id, None)
            finished.pop(output.request_id, None)
            continue
        grouped[output.request_id].append(int(output.token_id))
        if output.finished:
            reason = str(output.finish_reason)
            finished[output.request_id] = reason
    now = time.monotonic()
    for request_id, tokens in grouped.items():
        batches[request_id].add(tokens, now)
    # Check age even on steps without a new token for this request.
    for request_id, batch in list(batches.items()):
        tokens = batch.take(now, finished=request_id in finished)
        if not tokens:
            continue
        outgoing.put(
            {
                "type": "tokens",
                "id": request_id,
                "tokens": tokens,
                "finish_reason": finished.get(request_id),
                "cached_tokens": cache.hits.get(request_id, 0),
                "cached_tokens_reported": True,
            }
        )
        if request_id in finished:
            active.remove(request_id)
            cache.forget(request_id)
            del batches[request_id]


def run_rank(rank, model_path, incoming, outgoing, stopped):
    try:
        llm = initialize(rank, model_path)
        engine = llm.llm_engine
        from flash.serving.deepinfer.runtime.provenance import loaded_runtime

        proof = loaded_runtime(llm, rank)
        proof.update(
            expert_dtype="native_fp4",
            expert_source="native-fp4",
            converted_expert_tensors=0,
            prefill_any_attempt_interval=engine.scheduler_config.prefill_any_attempt_interval,
        )
        print(f"DEEPINFER_LOADED {proof}", flush=True)
        outgoing.put({"type": "ready", "rank": rank, "provenance": proof})
        active = set()
        cache = PrefixCache(engine)
        batches = {}
        last_heartbeat = time.monotonic()
        while not stopped.is_set():
            # Bounded command draining leaves time for scheduling and other ranks.
            for _ in range(64):
                try:
                    command = incoming.get_nowait()
                except queue.Empty:
                    break
                admit(engine, command, active, outgoing, cache, batches)
            step(engine, active, outgoing, cache, batches)
            if time.monotonic() - last_heartbeat >= 1:
                outgoing.put({"type": "heartbeat", "rank": rank})
                last_heartbeat = time.monotonic()
                cache.prune()
            if not active:
                time.sleep(0.002)
    except BaseException:
        traceback.print_exc()
        outgoing.put({"type": "fatal", "rank": rank})
        stopped.set()
        raise
