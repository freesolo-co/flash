"""Prefill prompts except four final tokens, initializing MTP on the receiver."""

import time

import torch


def execute(native, command):
    started = time.perf_counter()
    pending = []
    for request in command["requests"]:
        seq = native.sequence(request["tokens"], request.get("output", 128))
        # One remaining token enters the native decode path, which requires
        # speculative draft tokens. Replaying four prompt tokens initializes
        # that state through prefill instead of inventing draft metadata.
        cut = len(request["tokens"]) - 4
        seq.chunk_checkpoints = [cut]
        assert native.engine.scheduler.try_allocate_seq(seq)
        pending.append((request["key"], seq, cut))
    detached = set()
    while True:
        for key, seq, cut in pending:
            assert seq.num_cached_tokens <= cut
            if seq.num_cached_tokens == cut and key not in detached:
                native.engine.scheduler.detach_waiting_for_prefix_cache(seq.request_id)
                native.sources[key] = seq
                detached.add(key)
        if not native.engine.has_unfinished_requests():
            break
        native.engine.step()
        native.engine.flush_inflight_reqs()
        assert time.perf_counter() - started < 300
    assert len(detached) == len(pending)
    torch.cuda.synchronize()
    return {
        "prefill_seconds": time.perf_counter() - started,
        "requests": len(pending),
        "cached_tokens": sum(cut for _, _, cut in pending),
    }
