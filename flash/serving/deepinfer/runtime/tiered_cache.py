"""Bounded CPU LRU for exact, tenant-scoped prefixes evicted from GPU memory."""

import time
from collections import OrderedDict

from flash.serving.deepinfer.runtime.prefix_cache import PrefixCache
from flash.serving.deepinfer.runtime.snapshots import Snapshots


class TieredPrefixCache(PrefixCache):
    def __init__(self, engine, *, cpu_bytes=1024**3, **kwargs):
        super().__init__(engine, **kwargs)
        self.cpu_limit = cpu_bytes
        if not 0 <= self.cpu_limit <= 8 * 1024**3:
            raise ValueError("CPU KV budget must be between zero and 8 GiB per rank")
        self.cpu = OrderedDict()
        self.cpu_bytes = 0
        self.snapshots = Snapshots(engine) if self.cpu_limit else None

    def _drop_cpu(self, key):
        prefix, snapshot = self.cpu.pop(key)
        self.cpu_bytes -= snapshot.nbytes
        return prefix, snapshot

    def _evict(self, key, *, spill=True):
        prefix = self.entries[key]
        if spill and self.snapshots is not None:
            size = self.snapshots.size(prefix.sequence)
            if size <= self.cpu_limit:
                while self.cpu and self.cpu_bytes + size > self.cpu_limit:
                    self._drop_cpu(next(iter(self.cpu)))
                snapshot = self.snapshots.capture(prefix.sequence)
                self.cpu[key] = (prefix, snapshot)
                self.cpu_bytes += snapshot.nbytes
        super()._evict(key, spill=spill)
        prefix.sequence = None

    def prune(self, prompt_tokens=0):
        now = time.monotonic()
        for key in list(self.cpu):
            if now - self.cpu[key][0].touched >= self.ttl:
                self._drop_cpu(key)
        super().prune(prompt_tokens)

    def _match(self, namespace, tokens, params):
        match = super()._match(namespace, tokens, params)
        if match is not None or not namespace:
            return match
        matches = [
            (key, prefix)
            for key, (prefix, _) in self.cpu.items()
            if prefix.namespace == namespace
            and len(prefix.tokens) < len(tokens) - 1
            and prefix.tokens == tuple(tokens[: len(prefix.tokens)])
        ]
        if not matches:
            return None
        key, prefix = max(matches, key=lambda item: len(item[1].tokens))
        prefix, snapshot = self._drop_cpu(key)
        # Keep the in-flight host snapshot within the budget while making room
        # for promotion. Dropped GPU victims can be recomputed on their next miss.
        self._make_room(prefix.storage_tokens, spill=False)
        donor = self.snapshots.restore(snapshot, params)
        if donor is None:
            return None  # Allocation pressure falls back to ordinary prefill.
        prefix.sequence = donor
        self.entries[key] = prefix
        return key, prefix

    def close(self):
        super().close()
        self.cpu.clear()
        self.cpu_bytes = 0
