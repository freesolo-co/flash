"""Bounded cache-aware admission; native vLLM still owns scheduling and KV state."""

from itertools import islice


def cached_candidate(waiting, cached_tokens, now):
    """Prefer cheaper cached work within sixteen arrivals and a 250 ms age bound."""
    head = waiting[0]
    if now - head.arrival_time >= 0.25 or head.num_computed_tokens:
        return head
    head_work = head.num_tokens - cached_tokens(head)
    best, best_work = head, head_work
    for request in islice(waiting, 1, 16):
        if request.num_computed_tokens:
            continue
        cached = cached_tokens(request)
        work = request.num_tokens - cached
        if cached and work < best_work:
            best, best_work = request, work
    return best


def qwen_scheduler_args(model, engine_args):
    if not model.startswith(("Qwen/", "Freesolo-Co/Qwen")):
        return {}
    name = (
        "CacheScheduler" if engine_args.get("async_scheduling") is False else "AsyncCacheScheduler"
    )
    return {"scheduler_cls": f"flash.serve.runtime.cache_scheduler.{name}"}
