"""Process-local raw KV snapshots, including DeepInfer's recurrent/SWA state."""

import uuid
from dataclasses import dataclass


@dataclass
class Snapshot:
    owner: str
    tokens: tuple[int, ...]
    cached_tokens: int
    swa_valid_start: int
    kv: object
    state: object

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in (self.kv, self.state))


class Snapshots:
    def __init__(self, engine):
        self.engine = engine
        self.cache = engine.driver_worker.cache_engine
        # Snapshots never leave this engine instance: model, native build, rank,
        # precision, and cache layout must all remain identical.
        self.owner = uuid.uuid4().hex

    def size(self, sequence):
        block = self.cache.gpu_blocks[0]
        state = self.cache.gpu_states[sequence.state_id]
        return len(sequence.block_table) * block.numel() * block.element_size() + (
            state.numel() * state.element_size()
        )

    def capture(self, sequence):
        import torch

        # Donors were detached at a flushed prefill checkpoint. Synchronize other
        # streams before reading blocks which may also have active COW readers.
        torch.cuda.synchronize()
        blocks = torch.tensor(sequence.block_table, dtype=torch.long, device="cuda")
        return Snapshot(
            self.owner,
            tuple(sequence.data.get_token_ids()),
            sequence.num_cached_tokens,
            sequence.swa_valid_start,
            self.cache.gpu_blocks.index_select(0, blocks).cpu(),
            self.cache.gpu_states[sequence.state_id].cpu(),
        )

    def restore(self, snapshot, params):
        import numpy as np
        import torch

        if snapshot.owner != self.owner:
            raise ValueError("KV snapshot belongs to another engine")
        sequence = self.engine.add_request(
            user_id="flash",
            request_id="prefix-" + uuid.uuid4().hex,
            sampling_params=params,
            prompt_token_ids=np.asarray(snapshot.tokens, dtype=np.int32),
            multi_modal_data={"image_bytes": [], "image_sizes": []},
            enqueue=False,
        )
        if sequence is None:
            raise RuntimeError("DeepInfer refused snapshot metadata")
        # enqueue=False bypasses the asynchronous metadata processor. Run its
        # native mapper before a restored donor can enter a mixed prefill batch.
        # Raw image_bytes/segment_input dictionaries are not model-ready metadata.
        if self.engine.multi_modal_processor is not None:
            sequence.multi_modal_data = self.engine.multi_modal_processor.multi_modal_mapper(
                sequence.multi_modal_data
            )
        if not self.engine.scheduler.try_allocate_seq(sequence):
            return None
        try:
            if (
                len(sequence.block_table) != snapshot.kv.shape[0]
                or self.cache.gpu_blocks.shape[1:] != snapshot.kv.shape[1:]
                or self.cache.gpu_states[sequence.state_id].shape != snapshot.state.shape
                or self.cache.gpu_blocks.dtype != snapshot.kv.dtype
                or self.cache.gpu_states.dtype != snapshot.state.dtype
            ):
                raise ValueError("KV snapshot layout changed")
            torch.cuda.synchronize()
            blocks = torch.tensor(sequence.block_table, dtype=torch.long, device="cuda")
            self.cache.gpu_blocks.index_copy_(0, blocks, snapshot.kv.to("cuda"))
            self.cache.gpu_states[sequence.state_id].copy_(snapshot.state)
            sequence.num_cached_tokens = snapshot.cached_tokens
            sequence.swa_valid_start = snapshot.swa_valid_start
            torch.cuda.synchronize()
            return sequence
        except BaseException:
            self.engine.scheduler.free_seq(sequence)
            raise
