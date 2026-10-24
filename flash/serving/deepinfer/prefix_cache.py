"""Bounded, tenant-scoped native prefix snapshots with copy-on-write continuations."""

import math
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property

from flash.serving.deepinfer.routing import fingerprint


@dataclass
class Prefix:
    namespace: str
    tokens: tuple[int, ...]
    sequence: object
    touched: float
    storage_tokens: int = 0

    @cached_property
    def digest(self):
        return fingerprint(self.tokens)


class PrefixCache:
    def __init__(self, engine, *, max_entries=4, max_tokens=32768, ttl=600):
        self.engine = engine
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self.ttl = ttl
        self.entries = OrderedDict()
        self.pending = {}
        self.hits = {}

    def directory(self):
        return [
            {"namespace": prefix.namespace, "length": len(prefix.tokens), "digest": prefix.digest}
            for prefix in self.entries.values()
        ]

    def _evict(self, key):
        prefix = self.entries.pop(key)
        self.engine.scheduler.free_seq(prefix.sequence)

    def prune(self, prompt_tokens=0):
        now = time.monotonic()
        for key, value in list(self.entries.items()):
            if now - value.touched >= self.ttl:
                self._evict(key)
        reserve = math.ceil(prompt_tokens / self.engine.cache_config.block_size)
        while self.entries and self.engine.scheduler.get_num_free_gpu_blocks() < reserve:
            self._evict(next(iter(self.entries)))

    def _fork(self, donor, request_id, params, tokens=None):
        seq = self.engine.scheduler.fork_seq(
            donor,
            None,
            "_flash-" + request_id,
            params,
            donor.so_matcher,
            donor.data.get_last_token_id(),
            -1,
            next(self.engine.seq_counter),
            time.monotonic(),
        )
        seq.request_id = request_id
        if tokens is not None:
            import numpy as np

            # Use the native request constructor for compressed Engram token metadata too.
            metadata = self.engine.add_request(
                user_id="flash",
                request_id="metadata-" + request_id,
                sampling_params=params,
                prompt_token_ids=np.asarray(tokens, dtype=np.int32),
                multi_modal_data={"image_bytes": [], "image_sizes": []},
                enqueue=False,
            )
            seq.data, seq.compressed_data = metadata.data, metadata.compressed_data
        return seq

    def admit(self, command, params):
        import numpy as np

        request_id, tokens = command["id"], command["tokens"]
        namespace = command.get("cache_namespace")
        self.prune(len(tokens) + command.get("sampling", {}).get("max_tokens", 0))
        matches = [
            (key, value)
            for key, value in self.entries.items()
            if namespace
            and value.namespace == namespace
            and len(value.tokens) < len(tokens) - 1
            and value.tokens == tuple(tokens[: len(value.tokens)])
        ]
        self.hits[request_id] = 0
        if matches:
            key, prefix = max(matches, key=lambda item: len(item[1].tokens))
            prefix.touched = time.monotonic()
            self.entries.move_to_end(key)
            self._fork(prefix.sequence, request_id, params, tokens)
            self.hits[request_id] = len(prefix.tokens)
            return
        length = command.get("prefix_tokens", 0)
        block_size = self.engine.cache_config.block_size
        storage_tokens = math.ceil(len(tokens) / block_size) * block_size
        retain = bool(
            namespace
            and self.max_entries
            and 128 <= length <= self.max_tokens
            and storage_tokens <= self.max_tokens
            and length < len(tokens) - 1
        )
        seq = self.engine.add_request(
            user_id="flash",
            request_id=request_id,
            sampling_params=params,
            prompt_token_ids=np.asarray(tokens, dtype=np.int32),
            multi_modal_data={"image_bytes": [], "image_sizes": []},
            chunk_checkpoints=[length] if retain else None,
        )
        if seq is None:
            raise RuntimeError("DeepInfer refused request admission")
        if retain:
            self.pending[request_id] = (
                seq,
                namespace,
                tuple(tokens[:length]),
                params,
                storage_tokens,
            )

    def capture_ready(self):
        ready = [
            (rid, values)
            for rid, values in self.pending.items()
            if values[0].num_cached_tokens == len(values[2])
        ]
        if not ready:
            return []
        from deepinfer._C.engine import RequestOutput

        # Complete asynchronous state writes before sharing the prefix with a continuation.
        outputs = [RequestOutput.clone(pointer) for pointer in self.engine.flush_inflight_reqs()]
        for request_id, (seq, namespace, tokens, params, storage_tokens) in ready:
            del self.pending[request_id]
            if seq.num_cached_tokens != len(tokens):
                raise RuntimeError("Native prefix checkpoint advanced before snapshot")
            while self.entries and (
                len(self.entries) >= self.max_entries
                or sum(value.storage_tokens for value in self.entries.values()) + storage_tokens
                > self.max_tokens
            ):
                self._evict(next(iter(self.entries)))
            donor = self.engine.scheduler.detach_waiting_for_prefix_cache(request_id)
            donor.request_id = "prefix-" + uuid.uuid4().hex
            self.entries[donor.request_id] = Prefix(
                namespace, tokens, donor, time.monotonic(), storage_tokens
            )
            self._fork(donor, request_id, params)
        return outputs

    def forget(self, request_id):
        self.pending.pop(request_id, None)
        self.hits.pop(request_id, None)

    def close(self):
        for key in list(self.entries):
            self._evict(key)
