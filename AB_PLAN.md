# OPD adapter sync A/B plan

## Objective

Measure whether publishing each versioned PEFT adapter through `/dev/shm` reduces the optimizer-step-to-next-rollout synchronization cost without changing training behavior.

This plan is documentation only. Do not launch a paid GPU run as part of this change.

## Controlled arms

Run the two arms concurrently on separate, otherwise identical RunPod RTX 4090 workers.

- Control: `origin/dev` at `8fdfc8580b9058d8c40289d4fa1a2e6e3b7bced2`
- Treatment: `perf/opd-adapter-sync` at the commit containing this change

Hold constant across arms:

- model: `Qwen/Qwen3.5-0.8B`
- algorithm: OPD
- provider and exact GPU: RunPod, RTX 4090
- seed: `42`
- environment revision and dataset order
- immutable model revision
- teacher model and endpoint
- LoRA initialization, rank, alpha, and learning rate
- prompt, rollout, optimizer, and checkpoint settings
- dependency image, including vLLM `0.19.1`

Before launch, verify `/dev/shm` is mounted as `tmpfs` in the treatment worker. The treatment is invalid if it logs the filesystem fallback warning.

## Short configuration

Use one frozen environment revision with at least 16 valid single-turn rows and apply this same configuration to both arms:

```toml
model = "Qwen/Qwen3.5-0.8B"
algorithm = "opd"
seed = 42

[environment]
id = "<frozen-representative-environment>"

[gpu]
provider = "runpod"
exact_type = "RTX 4090"

[train]
epochs = 1
max_examples = 16
max_steps = 8
batch_size = 1
group_size = 4
max_context_tokens = 1024
max_completion_tokens = 128
lora_rank = 32
lora_alpha = 64
learning_rate = 0.00005
teacher_model = "glm-5.2"
```

Use the same immutable environment and model revisions in both submitted specs. Do not tune either arm after observing results.

## Metrics and decision thresholds

Compute sync time per measured update as:

`opd_phase_vllm_sync_seconds / opd_phase_vllm_syncs`

Primary performance gate:

- Direction: treatment must be lower.
- Pass: treatment mean sync time per measured update is at least 25% lower than control.
- No-regression floor: treatment must not be more than 5% higher than control even if the 25% improvement target is missed.

End-to-end step gate:

- Compute measured training-step wall time as total OPD training wall divided by completed optimizer steps, excluding model download and vLLM initialization.
- Direction: treatment must be lower.
- Pass: treatment mean per-step wall time is at least 3% lower than control.
- Fail: treatment mean per-step wall time is more than 2% higher than control.

Training parity gates:

- Both arms must complete exactly 8 optimizer steps with the same rollout seed sequence and no additional skipped or truncated rollouts.
- Final loss relative difference must be at most 1%.
- Loss-curve normalized RMSE, using the control curve mean magnitude as the denominator, must be at most 1%.
- Mean coverage and mean alignment granularity must each differ by at most 1 percentage point relative to control.
- Any NaN, adapter load error, stale-version generation, or difference in completed optimizer steps fails parity.

## Required evidence

Record for each arm:

- git commit and immutable model/environment revisions
- RunPod worker and exact GPU identity
- `/dev/shm` filesystem type for treatment
- `opd_phase_vllm_sync_seconds`
- `opd_phase_vllm_syncs`
- total OPD training wall and completed optimizer steps
- loss curve, mean coverage, and mean alignment granularity
- worker logs covering the measured sync interval and any filesystem fallback warning

The optimization is accepted only if both performance gates pass and all training parity gates pass. Report a negative result if the sync metric improves but per-step wall or training parity fails.
