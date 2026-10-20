# Resident training workers: reset experiment

`experiments/warm_lora.py` is an executable CUDA/FSDP2 proof for the prerequisite
of reusing a loaded LoRA model. It is deliberately outside the managed training
path and does not enable a production worker pool.

A resident instance accepts one exact owner/runtime profile and one lease at a
time. Before each lease it restores all model tensors from an immutable CPU
snapshot, clears gradients, restores the training mode, reseeds RNGs, and creates
new optimizer and scheduler instances. Frozen weights and mutable buffers are
included: freezing gradients alone does not prevent in-place mutations. A failed
or cancelled lease evicts the instance. The experiment compares every loss,
model tensor, optimizer value and scheduler value with both a second warm lease
and a freshly initialized model; it also checks owner rejection and eviction.

Run inside the pinned worker image with:

```sh
/opt/verl-venv/bin/torchrun --standalone --nproc_per_node=2 experiments/warm_lora.py
```

The fixture is a tiny Llama with two GPUs. Its timings cannot establish savings
for Qwen 9B/35B/122B, GRPO rollouts, or real hosted jobs. Keeping a complete CPU
reset snapshot also consumes host RAM and transfers base tensors at reset; it
trades those costs for avoiding model reconstruction and kernel initialization.

## Production integration still required

Flash currently starts isolated trainer subprocesses in one-attempt Modal
Sandboxes. Reusing a container alone does not retain the trainer's loaded model.
Modal Sandbox memory snapshots do not support GPU Sandboxes; the GPU snapshot
feature applies to Functions/Classes, so adding a Sandbox decorator is not a
working migration.

A production pool needs a resident training service or a Function/Class backend,
lease-aware attempt handles, per-organization authorization, bounded idle GPU
billing, cancellation and eviction, checkpoint/recovery integration, and reset
proofs for every supported trainer. GRPO additionally owns Ray, vLLM KV caches,
reward servers and rollout RNG state. This experiment does not claim to reset
those systems, and the existing isolated backend remains the production path.

This PR should remain a draft until that integration is implemented and tested.

The live two-H100 fixture passed: both the warm second lease and a fresh model
produced exactly equal losses, model state, optimizer state and scheduler state.
Owner mismatch was rejected and a failed lease could not be reused. The tiny
fixture measured 2.10 seconds for construction plus three updates and 0.032
seconds for reset plus three updates; these are fixture-only observations.

Modal references: [Sandbox snapshots](https://modal.com/docs/guide/sandbox-snapshots)
and [GPU memory snapshots](https://modal.com/docs/examples/gpu_snapshot).
