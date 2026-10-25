from collections import deque
from types import SimpleNamespace

from flash.serve.runtime.cache_admission import cached_candidate, qwen_scheduler_args


def request(name, tokens, *, arrival=100, computed=0):
    return SimpleNamespace(
        request_id=name, num_tokens=tokens, arrival_time=arrival, num_computed_tokens=computed
    )


def test_cached_work_can_pass_a_cold_prompt_but_not_a_cheaper_short_prompt():
    cold, hit = request("cold", 5000), request("hit", 5000)

    def cached(r):
        return 4800 if r is hit else 0

    assert cached_candidate(deque([cold, hit]), cached, 100.1) is hit
    short = request("short", 30)
    assert cached_candidate(deque([short, hit]), cached, 100.1) is short


def test_old_and_preempted_head_requests_keep_their_place():
    head, hit = request("old", 5000), request("hit", 5000)

    def cached(r):
        return 4800 if r is hit else 0

    assert cached_candidate(deque([head, hit]), cached, 100.25) is head
    head.num_computed_tokens = 1
    assert cached_candidate(deque([head, hit]), cached, 100.01) is head


def test_candidate_search_is_bounded_and_only_qwen_uses_custom_admission():
    head, hit = request("head", 5000), request("hit", 5000)
    waiting = deque([head] * 16 + [hit])
    assert cached_candidate(waiting, lambda r: 4800 if r is hit else 0, 100.1) is head
    assert qwen_scheduler_args("other") == {}
    assert qwen_scheduler_args("Qwen/model")["scheduler_cls"].endswith(".QwenCacheScheduler")
