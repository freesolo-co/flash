# SFT chunked NLL matched GPU A/B plan

## Objective

Verify that TRL 1.6 `chunked_nll` preserves SFT learning behavior while removing the dense `[micro_batch, sequence, vocabulary]` logits allocation, increasing realized microbatch, lowering peak GPU memory, and improving throughput.

No paid GPU run is part of this branch task.

## Frozen comparison

- Control: `origin/dev` at `8fdfc8580b9058d8c40289d4fa1a2e6e3b7bced2`
- Treatment: commit from `perf/sft-fused-chunked-nll`
- Provider: RunPod
- GPU: one RTX 4090 24 GB per arm, same validated SKU and worker image digest
- Launch: both arms concurrently after capacity is confirmed
- Dependencies: identical worker dependency lock, including torch 2.10.0, transformers 5.10.2, TRL 1.6.0, and PEFT 0.19.1
- Model: `Qwen/Qwen3.5-0.8B` at one immutable Hugging Face revision
- Environment: one real hub SFT environment pinned to an immutable commit, for example the production GSM8K environment if its SFT completion path is present
- Seed: 42 for both arms
- Data: identical retained rows, order, tokenization, completion masks, and truncation
- Training: 30 optimizer updates, `max_context_tokens = 1024`, effective batch size 8, identical LoRA rank, learning rate, optimizer, warmup, checkpoint schedule, and thinking setting
- Evaluation boundary: compare the same update numbers and the final immutable adapter checkpoint

The effective batch and update count must remain identical. Expected realized batching is control `per_device = 1, gradient_accumulation = 8` and treatment `per_device = 4, gradient_accumulation = 2`.

## Metrics to capture

1. Per-update train loss, using the same logging cadence and update indices.
2. Total training tokens and valid completion-token count.
3. Tokens per second and median steady-state step wall time, excluding model load, compilation, and the first three warmup steps.
4. Realized per-device microbatch and gradient accumulation.
5. Peak allocated GPU memory from the worker metric and `torch.cuda.max_memory_allocated()`.
6. Completion-mask count, optimizer update count, NaN or inf events, OOMs, and worker health.

## Expected direction

- Train loss: no meaningful change. The treatment curve should match the control within deterministic numerical noise.
- Tokens per second: higher for treatment.
- Steady-state step wall time: lower for treatment.
- Realized microbatch: treatment rises from 1 to 4 at the frozen shape.
- Peak GPU memory: lower for treatment despite the larger microbatch.
- Completion masking, packed boundaries, update count, and token accounting: identical.

## Pass and fail thresholds

The treatment passes only if every correctness gate and every optimization gate passes.

### Correctness gates

- Both arms finish all 30 requested optimizer updates with no OOM, NaN, inf, or worker-health failure.
- Retained examples, completion-mask counts, total train tokens, and checkpoint update numbers are identical.
- Mean absolute paired loss difference after the first three updates is at most 2% of the control mean loss.
- Final three-update mean loss differs by at most 2% from control and is not worse by more than 0.02 absolute loss.
- The final immutable treatment checkpoint completes the same smoke evaluation as control with no new formatting, EOS, or empty-output failure.

### Optimization gates

- Treatment realizes `per_device_train_batch_size = 4`; control realizes 1 for the frozen configuration.
- Treatment median steady-state tokens per second is at least 10% higher than control, equivalently median step wall time is at least 9% lower.
- Treatment peak allocated GPU memory is at least 2 GB lower than control and does not exceed 90% of the control peak.

Fail the optimization if correctness passes but either speed or memory misses its threshold. Report the result without changing thresholds or selecting a different update window after observing the run.

## Run order

1. Resolve and record immutable model, environment, image, and source revisions.
2. Confirm both RTX 4090 workers expose the same GPU model and memory.
3. Launch control and treatment concurrently with the frozen configuration.
4. Collect raw per-step metrics and immutable final checkpoints without tuning either arm.
5. Compare paired update losses, steady-state throughput, realized batching, and peak memory against the preregistered thresholds.
6. Tear down both workers and report pass or fail for correctness, speed, and memory separately.
