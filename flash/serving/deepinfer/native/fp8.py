"""Losslessly widen native E2M1 expert values to E4M3, retaining UE8M0 scales.

FP8 execution preserves the released FP4 values and their original scales.
This does not reconstruct a separate pre-quantization checkpoint.
"""

import torch

CONVERTED_EXPERT_TENSORS = 0


def _interleave_cuda(src):
    """One allocation instead of stack + reshape + copied output."""
    import triton
    import triton.language as tl

    @triton.jit
    def interleave(
        a, b, COUNT: tl.constexpr, ROWS: tl.constexpr, ROW_BYTES: tl.constexpr, BLOCK: tl.constexpr
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row = (i // ROW_BYTES) % ROWS
        source_row = (row // 16) * 8 + row % 8 + ((row % 16) // 8) * (ROWS // 2)
        source = (
            (i // (ROWS * ROW_BYTES)) * (ROWS * ROW_BYTES) + source_row * ROW_BYTES + i % ROW_BYTES
        )
        value = tl.load(a + source, mask=i < COUNT, other=0)
        tl.store(b + i, value, mask=i < COUNT)

    assert src.is_contiguous()
    assert src.shape[-2] % 16 == 0
    result = torch.empty_like(src)
    count = src.numel() * src.element_size()
    interleave[(triton.cdiv(count, 4096),)](
        src.view(torch.uint8),
        result.view(torch.uint8),
        count,
        src.shape[-2],
        src.shape[-1] * src.element_size(),
        4096,
    )
    return result


def convert_mega_moe_weight(src_l1, src_l2):
    """Keep transformed storage directly; free only inputs that do not alias it."""
    from deep_gemm.mega import _transpose_sf_for_utccp

    torch.cuda.empty_cache()
    has_sf = src_l1.dtype != torch.bfloat16
    w1 = src_l1 if getattr(src_l1, "_flash_interleaved", False) else _interleave_cuda(src_l1)
    w2 = src_l2
    if has_sf:
        sf1 = _transpose_sf_for_utccp(_interleave_cuda(src_l1.sf.contiguous()))
        sf2 = _transpose_sf_for_utccp(src_l2.sf)

        # DeepGEMM's grouped SF contract is column-major within each expert.
        def column_major(sf):
            n, k = sf.shape[-2:]
            strides = (n * k, 1, n) if sf.ndim == 3 else (1, n)
            return torch.empty_strided(sf.shape, strides, device=sf.device, dtype=sf.dtype).copy_(
                sf
            )

        sf1, sf2 = column_major(sf1), column_major(sf2)
    l1 = torch.nn.Parameter(w1, requires_grad=False)
    l2 = torch.nn.Parameter(w2, requires_grad=False)
    if has_sf:
        l1.sf, l2.sf = sf1, sf2
    retained = [l1, l2] + ([l1.sf, l2.sf] if has_sf else [])
    keep = {x.untyped_storage().data_ptr() for x in retained}
    sources = [src_l1, src_l2] + ([src_l1.sf, src_l2.sf] if has_sf else [])
    for x in sources:
        if x.untyped_storage().data_ptr() not in keep:
            x.untyped_storage().resize_(0)
    return l1, l2


def _widen_cuda(packed, widened, interleave=False):
    import triton
    import triton.language as tl

    @triton.jit
    def convert(
        src,
        dst,
        count: tl.constexpr,
        ROWS: tl.constexpr,
        PACKED_K: tl.constexpr,
        INTERLEAVE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        source = i
        if INTERLEAVE:
            row = (i // PACKED_K) % ROWS
            source_row = (row // 16) * 8 + row % 8 + ((row % 16) // 8) * (ROWS // 2)
            source = (
                (i // (ROWS * PACKED_K)) * (ROWS * PACKED_K) + source_row * PACKED_K + i % PACKED_K
            )
        byte = tl.load(src + source, mask=i < count, other=0).to(tl.uint32)
        low, high = byte & 15, byte >> 4
        lm, hm = low & 7, high & 7
        lb = tl.where(lm == 0, 0, tl.where(lm == 1, 48, 48 + 4 * lm)) | ((low & 8) << 4)
        hb = tl.where(hm == 0, 0, tl.where(hm == 1, 48, 48 + 4 * hm)) | ((high & 8) << 4)
        tl.store(dst + 2 * i, lb.to(tl.uint8), mask=i < count)
        tl.store(dst + 2 * i + 1, hb.to(tl.uint8), mask=i < count)

    convert[(triton.cdiv(packed.numel(), 1024),)](
        packed.view(torch.uint8),
        widened.view(torch.uint8),
        packed.numel(),
        packed.shape[-2],
        packed.shape[-1],
        interleave,
        1024,
    )


def widen_expert(weight, interleave=False):
    global CONVERTED_EXPERT_TENSORS
    assert weight.ndim == 3
    assert weight.dtype == torch.int8
    n_experts, rows, packed_k = weight.shape
    if weight.is_cuda:
        # Release unused storage from earlier layer conversions before the
        # temporary widened allocation, retaining the strict allocator ceiling.
        torch.cuda.empty_cache()
    widened = torch.empty(
        (n_experts, rows, packed_k * 2), dtype=torch.float8_e4m3fn, device=weight.device
    )
    if weight.is_cuda:
        _widen_cuda(weight, widened, interleave)
    else:
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32
        )
        for e in range(n_experts):
            packed = weight[e].view(torch.uint8)
            widened[e, :, 0::2] = lut[(packed & 15).long()].to(torch.float8_e4m3fn)
            widened[e, :, 1::2] = lut[(packed >> 4).long()].to(torch.float8_e4m3fn)
        if interleave:
            half = rows // 2
            widened = torch.stack(
                [
                    widened[:, :half].reshape(n_experts, half // 8, 8, packed_k * 2),
                    widened[:, half:].reshape(n_experts, half // 8, 8, packed_k * 2),
                ],
                dim=2,
            ).reshape_as(widened)
    result = torch.nn.Parameter(widened, requires_grad=False)
    for key, value in weight.__dict__.items():
        setattr(result, key, value)
    CONVERTED_EXPERT_TENSORS += 1
    result.mxfp4 = False
    result._flash_interleaved = interleave
    return result
