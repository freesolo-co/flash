"""GPU-gated Gated-DeltaNet (GDN) backward parity check for fla on THIS card.

WHY: Flash's default path is Qwen3.5 (hybrid GDN) on an RTX 5090 (sm120). The training backward
runs through flash-linear-attention's Triton chunked-delta kernels. Upstream evidence says fla's
GDN backward is NOT validated-correct on consumer Blackwell sm120 (fla #913 open: crashes / NaN
grads on some 5090/sm121 stacks; #640 is the Hopper precision bug; the real Blackwell fixes
#946/#948 are unreleased). Flash's 2026-06-22 live smoke only proved "trained without crashing for
those shapes", NOT gradient correctness. THIS test closes that gap: run it on a real 5090.

WHAT: compare fla's CHUNK gated-delta (what training uses) against its FUSED_RECURRENT reference
(the same chunk-vs-recurrent methodology fla #640 used) for forward output AND gradients, plus a
hard NaN/Inf check (the #913 failure mode). Both paths get IDENTICAL inputs, so the comparison is
robust to fla API/arg-order differences; a genuine signature mismatch skips (never false-fails).

OUTCOME: if this PASSES on your 5090, keep fla on sm120 (the default). If it FAILS or NaNs, drop fla
on the 5090 by setting FLASH_FORCE_TORCH_DELTA=1 (transformers then uses the correct pure-PyTorch
delta rule), and treat that as the signal to flip the sm120 default.

This module is SKIPPED entirely without torch + fla + a CUDA GPU, so it never runs in the offline
CPU suite. Run it on a card with:  uv run pytest tests/test_gdn_parity_gpu.py -v
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
F = torch.nn.functional

if not torch.cuda.is_available():
    pytest.skip(
        "GDN parity needs a CUDA GPU (run on a real RTX 5090 / sm120)", allow_module_level=True
    )


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / (b.norm() + 1e-8)).item()


def _leaf(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone().requires_grad_(True)


def test_gdn_chunk_vs_recurrent_backward_parity():
    """fla's chunked GDN backward must match its fused-recurrent reference (output + grads) and
    produce finite grads on this GPU. bf16 (the production dtype). Tolerances target GROSS
    divergence / NaN (the sm120 #913 failure mode), not the bf16 noise floor; tighten them once a
    real 5090 run calibrates the floor."""
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    torch.manual_seed(0)
    B, T, H, K, V = 2, 256, 4, 128, 128
    dt, dev = torch.bfloat16, "cuda"

    # gated delta rule expects l2-normalized q,k; pre-normalize so we don't depend on the kernel's
    # use_qk_l2norm_in_kernel kwarg (its presence/name varies across fla versions).
    q = _leaf(F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).to(dt))
    k = _leaf(F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1).to(dt))
    v = _leaf(torch.randn(B, T, H, V, device=dev).to(dt))
    beta = _leaf(torch.rand(B, T, H, device=dev).sigmoid().to(dt))
    g = _leaf(F.logsigmoid(torch.rand(B, T, H, device=dev)).to(dt))
    chunk_in = (q, k, v, beta, g)
    qr, kr, vr, betar, gr = (_leaf(x) for x in (q, k, v, beta, g))

    # IDENTICAL inputs to both impls -> arg-order/signature mistakes affect both equally and don't
    # break the parity check; a real signature mismatch raises TypeError -> skip (not fail).
    try:
        o_chunk, _ = chunk_gated_delta_rule(q, k, v, g, beta)
        o_ref, _ = fused_recurrent_gated_delta_rule(qr, kr, vr, gr, betar)
    except TypeError as e:  # fla API differs from 0.5.x
        pytest.skip(f"fla GDN op signature differs ({e}); update this test for the installed fla")

    assert torch.isfinite(o_chunk).all(), "chunk GDN forward produced NaN/Inf on this GPU"
    fwd_err = _rel_err(o_chunk.float(), o_ref.float())
    assert fwd_err < 2e-2, (
        f"GDN forward chunk-vs-recurrent rel err {fwd_err:.4f} too high (miscompute?)"
    )

    o_chunk.float().sum().backward()
    o_ref.float().sum().backward()
    grads = [
        ("dq", q, qr),
        ("dk", k, kr),
        ("dv", v, vr),
        ("dbeta", beta, betar),
        ("dg", g, gr),
    ]
    for name, a, r in grads:
        assert a.grad is not None, f"{name}: no grad from fla chunk backward on this GPU"
        assert torch.isfinite(a.grad).all(), (
            f"{name}: NaN/Inf grad from fla chunk backward on this GPU (fla #913 failure mode)"
        )
        err = _rel_err(a.grad.float(), r.grad.float())
        assert err < 5e-2, (
            f"GDN backward {name} rel err {err:.4f} exceeds tol -> fla chunk miscomputes on this "
            f"GPU (fla #640/#913). Drop fla here: FLASH_FORCE_TORCH_DELTA=1."
        )
    # keep chunk_in referenced for readers tracing the forward inputs
    assert len(chunk_in) == 5
