"""Opt-in approximate reuse after small, position-preserving prompt edits.

This is boundary recomputation, not layerwise CacheBlend. Later cached KV and
recurrent state still encode the old prefix; answers can change. Operators must
explicitly allow each tenant namespace. Approximate state is never cached again
or billed as an exact prefix hit.
"""

import math
import os
import re

from flash.serving.deepinfer.snapshots import Snapshots
from flash.serving.deepinfer.tiered_cache import TieredPrefixCache


def recompute_boundary(previous, tokens, length, block_size):
    if len(previous) != length or length < 1024:
        return None
    changed = [
        i for i, (old, new) in enumerate(zip(previous, tokens[:length], strict=True)) if old != new
    ]
    if not 1 <= len(changed) <= 32:
        return None
    boundary = math.ceil((changed[-1] + 1 + 128) / block_size) * block_size
    # Require a substantial unchanged tail; arbitrary insertions and large edits
    # fall back to full prefill instead of shifting positional cache entries.
    return boundary if boundary <= length // 2 else None


class ApproximatePrefixCache(TieredPrefixCache):
    def __init__(self, engine, **kwargs):
        super().__init__(engine, **kwargs)
        self.namespaces = set(
            filter(None, os.environ.get("FLASH_DEEPINFER_APPROXIMATE_NAMESPACES", "").split(","))
        )
        if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in self.namespaces):
            raise ValueError("Approximate reuse requires explicit SHA256 tenant namespaces")
        self.blend_snapshots = (self.snapshots or Snapshots(engine)) if self.namespaces else None
        self.blends = {}
        self.approximate = set()

    def admit(self, command, params):
        eligible = command.get("cache_namespace") in self.namespaces
        if eligible and command.get("prefix_tokens", 0) >= 1152:
            # Always replay the final 128 prefix tokens with the new query.
            command = dict(command, prefix_tokens=command["prefix_tokens"] - 128)
        super().admit(command, params)
        request_id = command["id"]
        if not eligible or self.blends or request_id not in self.pending:
            return
        seq, namespace, tokens, _, _ = self.pending[request_id]
        for prefix in self.entries.values():
            if prefix.namespace != namespace or len(prefix.sequence.data.get_token_ids()) != len(
                command["tokens"]
            ):
                continue
            cut = recompute_boundary(
                prefix.tokens, tokens, len(tokens), self.engine.cache_config.block_size
            )
            if cut is None or self.blend_snapshots.size(prefix.sequence) > 256 * 1024**2:
                continue
            snapshot = self.blend_snapshots.capture(prefix.sequence)
            seq.chunk_checkpoints = [cut]
            self.blends[request_id] = (seq, snapshot, params, cut, tuple(command["tokens"]))
            self.approximate.add(request_id)
            del self.pending[request_id]
            return

    def capture_ready(self):
        outputs = super().capture_ready()
        ready = [
            (rid, plan) for rid, plan in self.blends.items() if plan[0].num_cached_tokens == plan[3]
        ]
        if not ready:
            return outputs
        import numpy as np
        from deepinfer._C.engine import RequestOutput

        outputs.extend(
            RequestOutput.clone(pointer) for pointer in self.engine.flush_inflight_reqs()
        )
        for request_id, (seq, snapshot, params, cut, tokens) in ready:
            del self.blends[request_id]
            if seq.num_cached_tokens != cut:
                raise RuntimeError("Approximate prefill checkpoint advanced")
            fresh = self.engine.scheduler.detach_waiting_for_prefix_cache(request_id)
            try:
                front = self.blend_snapshots.capture(fresh)
                count = cut // self.engine.cache_config.block_size
                snapshot.kv[:count].copy_(front.kv[:count])
                snapshot.tokens = tokens
            finally:
                self.engine.scheduler.free_seq(fresh)
            donor = self.blend_snapshots.restore(snapshot, params)
            if donor is None:
                # A busy native state pool can refuse restoration. Recompute the
                # full request and preserve the original cancellation identity.
                seq = self.engine.add_request(
                    user_id="flash",
                    request_id=request_id,
                    sampling_params=params,
                    prompt_token_ids=np.asarray(tokens, dtype=np.int32),
                    multi_modal_data={"image_bytes": [], "image_sizes": []},
                )
                if seq is None:
                    raise RuntimeError("DeepInfer refused approximate-reuse fallback")
                self.approximate.discard(request_id)
            else:
                try:
                    self._fork(donor, request_id, params, tokens)
                finally:
                    self.engine.scheduler.free_seq(donor)
        return outputs

    def forget(self, request_id):
        super().forget(request_id)
        self.blends.pop(request_id, None)
        self.approximate.discard(request_id)

    def close(self):
        super().close()
        self.blends.clear()
        self.approximate.clear()
