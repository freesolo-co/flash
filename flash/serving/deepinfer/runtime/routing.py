"""Rank-local prefix advertisements and bounded work-aware request placement."""

import hashlib
import json
import time

from flash.serving.deepinfer.contract.config import SEQUENCES_PER_RANK


def fingerprint(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


class CacheRouter:
    # A directory is a hint: exact token and namespace checks remain in the worker.
    def __init__(self, ranks, *, max_age=2.0):
        self.directories = [([], 0.0) for _ in range(ranks)]
        self.max_age = max_age

    def update(self, rank, prefixes):
        self.directories[rank] = (prefixes, time.monotonic())

    def choose(self, namespace, tokens, loads, pending):
        available = [rank for rank, load in enumerate(loads) if load < SEQUENCES_PER_RANK]
        if not available:
            raise OverflowError("DeepInfer replica is at capacity")
        digests = {}
        now = time.monotonic()
        work = [0] * len(loads)
        for request in pending.values():
            work[request.rank] += request.work
        scores = {}
        for rank in available:
            cached = 0
            directory, updated = self.directories[rank]
            if namespace and now - updated <= self.max_age:
                for entry in directory:
                    length = entry["length"]
                    if entry["namespace"] != namespace or not 0 < length < len(tokens) - 1:
                        continue
                    if length not in digests:
                        digests[length] = fingerprint(tokens[:length])
                    if entry["digest"] == digests[length]:
                        cached = max(cached, length)
            scores[rank] = (work[rank] + len(tokens) - cached, loads[rank], rank)
        return min(available, key=scores.__getitem__)
