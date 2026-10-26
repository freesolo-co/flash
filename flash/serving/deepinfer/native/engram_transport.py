"""Single-node Engram transport for the pinned DeepInfer runtime: CUDA IPC or mapped shared host RAM.

Keeps native FP8 embedding values and packed UE8M0 scales unchanged. No RDMA,
no duplicate GPU embedding tables, and no CPU involvement in the decode loop.
Adapted from the local four-B200 benchmark. GPU qualification remains required for each runtime revision.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["TOKENS", "SF_TOKEN_STRIDE", "SF_HIDDEN_STRIDE"])
def _fetch(
    indices,
    weight_ptrs,
    sf_ptrs,
    output,
    out_sf,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ENTRIES: tl.constexpr,
    TOKENS,
    SF_TOKEN_STRIDE,
    SF_HIDDEN_STRIDE,
    RANKS: tl.constexpr,
    BLOCK_ENTRIES: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    entry = tl.program_id(0) * BLOCK_ENTRIES + tl.arange(0, BLOCK_ENTRIES)
    active = entry < TOKENS * ENTRIES
    global_row = tl.load(indices + entry, mask=active, other=-1)
    valid = active & (global_row >= 0) & (global_row < ROWS * RANKS)
    owner = tl.minimum(tl.maximum(global_row // ROWS, 0), RANKS - 1)
    row = tl.maximum(global_row % ROWS, 0)
    wptr = tl.load(weight_ptrs + owner).to(tl.pointer_type(tl.uint8))
    cols = tl.arange(0, BLOCK_HIDDEN)
    value = tl.load(
        wptr[:, None] + row[:, None].to(tl.int64) * HIDDEN + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < HIDDEN),
        other=0,
    )
    tl.store(
        output + entry[:, None].to(tl.int64) * HIDDEN + cols[None, :],
        value,
        mask=active[:, None] & (cols[None, :] < HIDDEN),
    )
    packs: tl.constexpr = HIDDEN // 128
    sptr = tl.load(sf_ptrs + owner).to(tl.pointer_type(tl.int32))
    sf = tl.load(
        sptr[:, None] + row[:, None].to(tl.int64) * packs + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < packs),
        other=2139062143,
    )
    token = entry // ENTRIES
    sfcol = (entry[:, None] % ENTRIES) * packs + cols[None, :]
    tl.store(
        out_sf + token[:, None] * SF_TOKEN_STRIDE + sfcol * SF_HIDDEN_STRIDE,
        sf,
        mask=active[:, None] & (cols[None, :] < packs),
    )


class LocalEngramBuffer:
    is_local_transport = True

    def __init__(
        self, group, num_entries_per_layer, hidden_size, max_num_tokens, num_entries_per_token
    ):
        self.group = group
        self.rank = group.rank()
        self.world = group.size()
        self.rows = num_entries_per_layer
        self.hidden = hidden_size
        self.max_tokens = max_num_tokens
        self.entries = num_entries_per_token
        if not self.rows or any(row <= 0 for row in self.rows):
            raise ValueError("Engram requires nonempty, positive shard sizes")
        if hidden_size <= 0 or hidden_size % 128:
            raise ValueError("FP8 Engram width must be a positive multiple of 128")
        if max_num_tokens <= 0 or num_entries_per_token <= 0:
            raise ValueError("Engram token capacity and entries per token must be positive")
        self.owned = []
        self.peers = []
        self.registered = []
        self.tables = []

    def _exchange(self, tensor):
        from torch.multiprocessing.reductions import reduce_storage, reduce_tensor

        if not tensor.is_cuda:
            # Conversion to shared storage happens one layer at a time.
            tensor.share_memory_()

        def export():
            if tensor.is_cuda:
                rebuild, args = reduce_tensor(tensor.detach())
                return True, rebuild, args, None
            rebuild, args = reduce_storage(tensor.untyped_storage())
            return (
                False,
                rebuild,
                args,
                (tensor.dtype, tensor.shape, tensor.stride(), tensor.storage_offset()),
            )

        # CUDA IPC reference counters and resource-sharer FDs are single-consumer
        # handles. Export an independent handle for each receiving process.
        metadata = [None if r == self.rank else export() for r in range(self.world)]
        self.owned.append(tensor)
        gathered = [None] * self.world
        dist.all_gather_object(gathered, metadata, group=self.group)
        peers = []
        pointers = []
        for rank, payload in enumerate(gathered):
            cuda = tensor.is_cuda
            if rank == self.rank:
                peer = tensor
            else:
                cuda, rebuild, args, layout = payload[self.rank]
            if rank != self.rank and cuda:
                # Each engine exposes its own device as cuda:0. Import the IPC
                # allocation into this rank's current CUDA context.
                import inspect

                arguments = inspect.signature(rebuild).bind(*args)
                arguments.arguments["storage_device"] = torch.cuda.current_device()
                peer = rebuild(*arguments.args, **arguments.kwargs)
            elif rank != self.rank:
                import time

                until = time.monotonic() + 120
                while True:
                    try:
                        storage = rebuild(*args)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= until:
                            raise
                        time.sleep(0.1)
                dtype, shape, stride, offset = layout
                peer = torch.empty(0, dtype=dtype, device="cpu").set_(
                    storage, offset, shape, stride
                )
            peers.append(peer)
        if not tensor.is_cuda:
            # Finish FD exchange before pinning multi-GB tables. CUDA host
            # registration can hold the GIL and starve the FD-sharing thread.
            dist.barrier(group=self.group)
        for peer in peers:
            if tensor.is_cuda:
                pointers.append(peer.data_ptr())
            else:
                runtime = torch.cuda.cudart()
                result = runtime.cudaHostRegister(peer.data_ptr(), peer.nbytes, 3)
                if int(result) != 0:
                    raise RuntimeError(f"cudaHostRegister failed: {result}")
                self.registered.append(peer.data_ptr())
                # Registered host memory on this UVA platform shares the virtual
                # address used by CUDA. The dedicated probe verifies GPU reads.
                pointers.append(peer.data_ptr())
        self.peers.append(peers)
        return torch.tensor(pointers, dtype=torch.int64, device="cuda")

    def write(self, storages, sfs):
        if self.tables:
            raise RuntimeError("Engram storage is already initialized")
        if sfs is None or len(storages) != len(self.rows) or len(sfs) != len(self.rows):
            raise ValueError("One FP8 weight and scale shard is required per Engram layer")
        # Validate every local shard before starting collective handle exchange.
        for index, (weight, scale) in enumerate(zip(storages, sfs, strict=True)):
            if weight.dtype != torch.float8_e4m3fn or weight.shape != (
                self.rows[index],
                self.hidden,
            ):
                raise ValueError("Engram weights must be FP8 with the configured shard shape")
            if not weight.is_contiguous():
                raise ValueError("Engram weights must be contiguous")
            if scale.dtype != torch.float32 or scale.shape != (self.rows[index], self.hidden // 32):
                raise ValueError("Engram scales must be FP32 with one scale per 32 values")
            if not scale.is_contiguous():
                raise ValueError("Engram scales must be contiguous")
        for weight, scale in zip(storages, sfs, strict=True):
            # Pack on the original device, preserving every power-of-two scale.
            bits = scale.view(torch.int32)
            if bool(torch.any((bits < 0) | ((bits & 0x7FFFFF) != 0))):
                raise ValueError("Engram scales must be nonnegative powers of two")
            packed = (bits >> 23).to(torch.uint8).view(torch.int32).contiguous()
            # Native loading can keep FP32 scales on CUDA while weights live in
            # host RAM. Exchange the smaller packed scales beside their weights.
            packed = packed.to(device=weight.device)
            self.tables.append((self._exchange(weight), self._exchange(packed)))
        torch.cuda.synchronize()
        dist.barrier(group=self.group)

    def fetch(self, indices, use_tma_aligned_col_major_sf=True):
        if len(self.tables) != len(self.rows):
            raise RuntimeError("Engram storage is not initialized")
        if indices.ndim != 3 or indices.dtype != torch.int32 or not indices.is_cuda:
            raise ValueError("Engram indices must be a three-dimensional CUDA int32 tensor")
        if not indices.is_contiguous():
            raise ValueError("Engram indices must be contiguous")
        layers, tokens, entries = indices.shape
        if layers != len(self.rows) or entries != self.entries or tokens > self.max_tokens:
            raise ValueError(
                "Engram indices do not match the configured layer, entry, or token limits"
            )
        results = []
        for layer in range(layers):
            output = torch.empty(
                (tokens, entries * self.hidden), dtype=torch.float8_e4m3fn, device="cuda"
            )
            sf_columns = entries * self.hidden // 128
            if use_tma_aligned_col_major_sf:
                stride = triton.cdiv(tokens, 4) * 4
                scales = torch.empty((sf_columns, stride), dtype=torch.int32, device="cuda").T[
                    :tokens
                ]
            else:
                scales = torch.empty((tokens, sf_columns), dtype=torch.int32, device="cuda")
            if tokens:
                wp, sp = self.tables[layer]
                # Four-B200 ABBA tests favor grouping at larger batches. Keep one
                # row per program for small batches to retain GPU parallelism.
                block_entries = 4 if tokens >= 32 else 1
                _fetch[(triton.cdiv(tokens * entries, block_entries),)](
                    indices[layer],
                    wp,
                    sp,
                    output.view(torch.uint8),
                    scales,
                    self.rows[layer],
                    self.hidden,
                    entries,
                    tokens,
                    *scales.stride(),
                    self.world,
                    block_entries,
                    triton.next_power_of_2(self.hidden),
                    num_warps=4,
                )
            results.append(lambda output=output, scales=scales: (output, scales))
        return results

    def destroy(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        for pointer in self.registered:
            torch.cuda.cudart().cudaHostUnregister(pointer)
        self.registered.clear()
        self.peers.clear()
        self.tables.clear()
        # CUDA IPC producers must outlive every importing process's tensor views.
        dist.barrier(group=self.group)
        self.owned.clear()


def install():
    """Patch only Engram transport; retain DeepInfer's engine and model math."""
    import types

    from deepinfer.model_executor.models.v4_modules.engram import EngramRuntime
    from deepinfer.model_executor.parallel_utils.userbuffer import userbuffer

    def allocate(
        manager,
        engram_group,
        num_entries_per_layer,
        hidden_size,
        max_num_tokens,
        num_entries_per_token,
        dtype,
        num_sf_packs=0,
    ):
        if dtype != torch.float8_e4m3fn or num_sf_packs != hidden_size // 128:
            raise ValueError("Local Engram transport requires FP8 weights and packed UE8M0 scales")
        assert manager.engram is None
        manager.engram = LocalEngramBuffer(
            engram_group, num_entries_per_layer, hidden_size, max_num_tokens, num_entries_per_token
        )
        return manager.engram

    def initialize(runtime):
        embeddings = [layer.ngram_embedding for layer in runtime._layers]
        if runtime._buffer is None:
            runtime._buffer = runtime.allocate(embeddings)
        runtime._buffer.write([x.weight for x in embeddings], [x.sf for x in embeddings])
        for embedding in embeddings:
            embedding.release_storage()

    userbuffer.allocate_engram = types.MethodType(allocate, userbuffer)
    EngramRuntime.initialize = initialize
