# OPD rollout-memory utilization A/B plan

## Question

Does sizing the colocated vLLM executor from measured post-load free VRAM, while reserving the modeled OPD training peak and allocator margin, eliminate the Qwen3.5 4B negative-KV startup failure and improve rollout throughput without changing training behavior or causing an OOM?

No paid GPU run is part of this change.

## Frozen comparison

- control: `origin/dev` at `8fdfc858`
- treatment: branch `perf/opd-rollout-util`
- hardware: one H100 80 GB for each arm, from the same RunPod GPU type and image
- launch both arms concurrently on separate matched GPUs
- use the same base checkpoint, dataset revision, environment revision, seed, teacher, token budget, batch order, and worker image inputs
- use immutable commits for both arms

An H100 80 GB is the smallest card for this A/B. The reported failure is a large-card utilization defect, with roughly 60 GB left idle and a negative-KV startup failure for Qwen3.5 4B. An RTX 4090 24 GB does not have enough residual memory to expose the intended utilization increase and therefore cannot demonstrate the optimization.

## OPD configuration

Use the existing reproducing environment and dataset with this training shape held constant:

```toml
model = "Qwen/Qwen3.5-4B"
algorithm = "opd"
seed = 42

[environment]
id = "<frozen-reproduction-environment>"

[train]
epochs = 1
batch_size = 8
group_size = 1
max_context_tokens = 1536
max_completion_tokens = 512
lora_rank = 32
teacher_model = "glm-5.2"
```

Use the same bounded number of optimizer steps in both arms. Do not tune either arm after launch.

## Measurements

Record at vLLM initialization and for every rollout step:

1. measured free and total VRAM immediately before vLLM construction
2. modeled OPD training peak reserve, allocator margin, rollout budget, and final `gpu_memory_utilization`
3. vLLM executor budget in GiB and available KV-cache budget or cache blocks
4. rollout batch size and generated tokens per second
5. peak allocated and peak reserved VRAM across the OPD loss forward/backward
6. initialization outcome, negative-KV/cache-block errors, CUDA OOMs, and worker health
7. per-step OPD loss, reward or task metric already emitted by the environment, and completed optimizer steps

## Gates

### Feasibility and memory safety

- treatment must initialize vLLM and complete every planned optimizer step with zero negative-KV/cache-block failures and zero CUDA OOMs
- treatment executor budget must never exceed measured free VRAM minus the logged training reserve and allocator margin
- treatment loss forward/backward peak must retain at least the allocator margin after accounting for the executor budget
- if the control reproduces the startup crash, treatment must complete the same configuration; otherwise the A/B is inconclusive on crash recovery but can still evaluate utilization and throughput

### Utilization and throughput

- direction: treatment `gpu_memory_utilization` and KV budget must be higher than control on the H100
- threshold: treatment executor budget must increase by at least 16 GiB or utilization by at least 0.20 absolute
- direction: treatment rollout generated tokens per second must be no worse than control
- threshold: when both arms complete, treatment median steady-state rollout throughput must improve by at least 10 percent; if control crashes, report treatment throughput without claiming a relative speedup

### Training parity

- both arms must consume the same prompts and completion-token budget and complete the same optimizer-step count
- per-checkpoint OPD loss curves must remain within normal deterministic/runtime noise, with no sustained treatment regression above 2 percent relative to control
- final task metric must not regress by more than 1 percentage point absolute
- any change in gradients, optimizer state, sampled inputs, or training math invalidates the A/B

## Decision

Accept the optimization only if the treatment passes all memory-safety and training-parity gates and either:

1. recovers the exact control startup failure, or
2. meets both the executor-budget and throughput thresholds when both arms complete.

Reject or revise if treatment OOMs, violates the protected-memory budget, changes the training curve beyond the parity threshold, or fails to increase the H100 rollout budget materially.
