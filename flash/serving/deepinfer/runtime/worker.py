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
    CONTEXT_PARALLEL_SIZE,
    DELIVERY_SECONDS,
    DELIVERY_TOKENS,
    GPU_COUNT,
    GPU_PREFIX_CACHE_ENTRIES,
    GPU_PREFIX_CACHE_TOKENS,
    MAX_MODEL_LEN,
    PREFILL_ANY_INTERVAL,
    PREFILL_COALESCING_MIN_TOKENS,
    PREFILL_COALESCING_SECONDS,
    PREFILL_TOKENS,
    RANK_COUNT,
    RUNTIME_ROOT,
    SEQUENCES_PER_RANK,
)
from flash.serving.deepinfer.runtime.tiered_cache import TieredPrefixCache


def initialize(rank: int, model_path: str):
    # No torch/deepinfer imports before assigning the process's CUDA device.
    os.environ.update(
        {
            "PYTORCH_ALLOC_CONF": "expandable_segments:False",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE": "compact",
            "DEEPINFER_YOCO_PREFILL_KEEP_TOKENS": "140",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "DEEPINFER_ACCELERATOR": "nvidia",
            "EP_DISABLE_GIN": "1",
            "DEEPINFER_ENGRAM_FORCE_GPU_STORAGE": "1",
            "DEEPINFER_USE_NEXTN_DECODE": "5",
            "DEEPINFER_ENABLE_MTP_VARLEN": "1",
            "DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS": "0",
            "DEEPINFER_BATCH_INVARIANT": "0",
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
    setup_cuda_visible_devices(rank, CONTEXT_PARALLEL_SIZE, GPU_COUNT)
    from flash.serving.deepinfer.native.engram_transport import install

    install()
    return create_llm(
        model_path,
        rank,
        ep_size=GPU_COUNT,
        data_parallel_size=RANK_COUNT,
        cp_size=CONTEXT_PARALLEL_SIZE,
        max_num_batched_tokens=PREFILL_TOKENS,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=SEQUENCES_PER_RANK,
        gpu_memory_utilization=0.85,
        lookahead_tokens=1,
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


def step(engine, active, outgoing, cache, batches, prefill_scheduler=None):
    from deepinfer._C.engine import RequestOutput

    raw = engine.step() if engine.has_unfinished_requests() else engine.flush_inflight_reqs()
    grouped = defaultdict(list)
    finished = {}
    outputs = [RequestOutput.clone(pointer) for pointer in raw]
    outputs.extend(cache.capture_ready())
    for output in outputs:
        if prefill_scheduler is not None:
            prefill_scheduler.pending.pop(output.request_id, None)
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
        from deepinfer import envs

        proof.update(
            batch_invariant=envs.DEEPINFER_BATCH_INVARIANT,
            nextn=envs.DEEPINFER_USE_NEXTN_DECODE,
            zero_overhead_scheduling=envs.DEEPINFER_ZERO_OVERHEAD_SCHEDULING,
            prefill_tokens=engine.scheduler_config.max_num_batched_tokens,
            max_num_seqs=engine.scheduler_config.max_num_seqs,
            max_model_len=engine.model_config.max_model_len,
            expert_dtype="native_fp4",
            expert_source="native-fp4",
            converted_expert_tensors=0,
            prefill_coalescing_seconds=PREFILL_COALESCING_SECONDS,
            prefill_coalescing_small_prompt_threshold=PREFILL_COALESCING_MIN_TOKENS,
            gpu_prefix_cache_token_budget=GPU_PREFIX_CACHE_TOKENS,
            gpu_prefix_cache_entries=GPU_PREFIX_CACHE_ENTRIES,
            cpu_prefix_spill_enabled=False,
            prefill_any_attempt_interval=engine.scheduler_config.prefill_any_attempt_interval,
        )
        print(f"DEEPINFER_LOADED {proof}", flush=True)
        outgoing.put({"type": "ready", "rank": rank, "provenance": proof})
        active = set()
        from flash.serving.deepinfer.runtime.prefill_scheduler import PrefillScheduler

        prefill_scheduler = PrefillScheduler(engine)
        # CPU snapshots currently contain only the driver rank's state. They cannot
        # safely restore CP workers, so retain GPU checkpoints without host spill.
        cache = TieredPrefixCache(
            engine,
            cpu_bytes=0,
            max_tokens=GPU_PREFIX_CACHE_TOKENS,
            max_entries=GPU_PREFIX_CACHE_ENTRIES,
        )
        batches = {}
        last_directory = None
        last_heartbeat = time.monotonic()
        while not stopped.is_set():
            # Bounded command draining leaves time for scheduling and other ranks.
            for _ in range(64):
                try:
                    command = incoming.get_nowait()
                except queue.Empty:
                    break
                admit(engine, command, active, outgoing, cache, batches)
                prefill_scheduler.command(command, active, cache.hits.get(command["id"], 0))
            prefill_scheduler.before_step()
            step(engine, active, outgoing, cache, batches, prefill_scheduler)
            prefill_scheduler.after_step(active)
            directory = cache.directory()
            if directory != last_directory or time.monotonic() - last_heartbeat >= 1:
                outgoing.put({"type": "cache_directory", "rank": rank, "prefixes": directory})
                last_directory = directory
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
