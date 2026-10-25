"""Scheduler extensions for the exact vLLM build used by Flash serving.

These run in EngineCore through vLLM's scheduler_cls hook, without touching the
GPU worker, model kernels, request priorities, or native cache reference counts.
"""

import time
from importlib.metadata import version

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler

from .cache_admission import cached_candidate
from .cache_retention import CacheRetention


class CacheAdmission:
    def __init__(self, *args, **kwargs):
        if version("vllm") != "0.23.0":
            raise RuntimeError("Flash cache scheduling requires the qualified vLLM 0.23.0 build")
        super().__init__(*args, **kwargs)
        self._cache_head = None
        self._cache_bypasses = 0

    def _cached_tokens(self, request):
        manager = self.kv_cache_manager
        if not manager.enable_caching or request.skip_reading_prefix_cache:
            return 0
        # This lookup does not touch reference counts or record another hit in
        # request/billing statistics. Native admission revalidates the result.
        _, count = manager.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return count

    def schedule(self):
        if (
            self.policy == SchedulingPolicy.FCFS
            and len(self.waiting) > 1
            and not self.skipped_waiting
            and len(self.running) < self.max_num_running_reqs
        ):
            head = self.waiting.peek_request()
            if head.request_id != self._cache_head:
                self._cache_head, self._cache_bypasses = head.request_id, 0
            if self._cache_bypasses < 2:
                candidate = cached_candidate(self.waiting, self._cached_tokens, time.time())
                if candidate is not head:
                    self.waiting.remove_request(candidate)
                    self.waiting.prepend_request(candidate)
                    self._cache_bypasses += 1
        return super().schedule()


class CacheScheduler(CacheRetention, CacheAdmission, Scheduler):
    pass


class AsyncCacheScheduler(CacheRetention, CacheAdmission, AsyncScheduler):
    pass


class QwenCacheScheduler:
    """Choose after vLLM resolves automatic async compatibility for the model."""

    def __new__(cls, *args, **kwargs):
        config = kwargs["vllm_config"] if "vllm_config" in kwargs else args[0]
        scheduler = (
            AsyncCacheScheduler if config.scheduler_config.async_scheduling else CacheScheduler
        )
        return scheduler(*args, **kwargs)
