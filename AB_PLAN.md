# GRPO decode-only CUDA graph A/B plan

## Objective

Measure whether decode-only CUDA graphs speed up the colocated GRPO rollout engine without changing training behavior. This plan does not authorize a paid run.

## Frozen comparison

- Control: `origin/dev` at `8fdfc8580b9058d8c40289d4fa1a2e6e3b7bced2`. On RTX 5090/sm120 with vLLM 0.19.1, this selects `enforce_eager=True`.
- Treatment: the committed head of `perf/grpo-cudagraphs`. On the same GPU and vLLM version, this selects `enforce_eager=False` with `mode=0` and `cudagraph_mode=FULL_DECODE_ONLY`.
- Hardware: two separate, matched RunPod RTX 5090 workers from the same GPU offer, image, region, CPU/RAM class, and storage configuration. Do not substitute unmatched cards between arms.
- Model: `Qwen/Qwen3.5-0.8B`, which fits within an RTX 4090-class memory budget.
- Environment: the real `entropy-R1` hub environment, pinned to one immutable `freesolo-co/environment-hub` commit before launch.
- Seed: 42 for both arms.
- Budget: 30 optimizer updates after a two-update exact-path smoke. Use the same prompts per step, group size, maximum context, maximum completion length, LoRA settings, optimizer settings, reward implementation, dataset order, and checkpoint schedule.
- Concurrency: launch both paid arms together only after both two-update smokes pass. Do not reuse a worker across arms.

## Preflight and token-parity gate

1. Resolve and record the exact Flash commit, environment-hub commit, model revision, container digest, vLLM version, CUDA/driver versions, and RunPod machine identifiers for each arm.
2. Run the same fixed prompt batch through both immutable rollout engines with greedy decoding and the same maximum token count.
3. Require identical token IDs, finish reasons, and decoded text for every fixed prompt. Any difference is a correctness failure and stops the A/B.
4. Run two GRPO updates per arm. Require successful graph capture or vLLM's eager fallback, finite loss/reward values, and no OOM before scaling to 30 updates.

## Metrics

Measure rollout work separately from training work where telemetry permits.

- Primary performance: generated rollout tokens per rollout-engine second, excluding engine startup, graph capture, checkpointing, reward evaluation, and optimizer time.
- Secondary performance: median rollout wall time per update and end-to-end median step wall time. Report graph-capture startup time separately rather than amortizing it away.
- Correctness: fixed-prompt rollout token parity, reward curve, training loss curve, completion length, finish-reason distribution, and invalid/empty completion rate.
- Safety: peak allocated and reserved GPU memory, OOM count, graph-capture failures, eager-fallback events, worker restarts, and failed updates.

## Decision thresholds

The expected direction is faster treatment decode with unchanged learning behavior.

- Performance works: treatment improves steady-state rollout tokens/second by at least 10% over control. A 10% to 40% improvement is the expected range. Confirm the direction with at least a 9% reduction in median rollout wall time over updates 3 through 30.
- End-to-end value: treatment reduces median total step wall time by at least 5%. Report a decode-only win below this threshold as technically real but operationally marginal.
- Reward parity: treatment reward AUC over the 30 matched updates must remain within 5% relative or 0.02 absolute of control, whichever tolerance is larger. There must be no sustained three-update reward regression larger than 0.05 absolute.
- Loss parity: treatment loss AUC must remain within 5% relative of control, with finite values at every update and no new divergence pattern.
- Rollout parity: the fixed greedy prompt batch must have exact token-ID parity. During stochastic training, completion length, finish-reason, and empty/invalid-rate differences must each remain within 5% relative or one sample, whichever is larger.
- Memory and reliability: no OOM, no worker restart, no failed update, and treatment peak reserved memory no more than 5% above control after graph capture.

The treatment passes only if all correctness and reliability gates pass and the primary performance threshold is met. If vLLM falls back to eager, report the fallback and classify the performance result as inconclusive rather than a decode-graph success.
