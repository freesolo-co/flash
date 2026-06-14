# Algorithms

AutoSLM trains a **LoRA adapter** on a single GPU with one of two algorithms,
selected by `algorithm = "..."` in the run TOML. Both share the same machinery
around them — the active [environment](environments.md) supplies the data and the
grader, and both produce a LoRA adapter uploaded to the artifact repo. They differ
only in the training signal.

| algorithm | engine | training signal | driven by | needs |
|---|---|---|---|---|
| [`sft`](#sft) | TRL `SFTTrainer` | imitate reference completions | `train.epochs` | env `sft_target` |
| [`grpo`](#grpo) | TRL `GRPOTrainer` + colocated vLLM | verifiable reward on the model's own samples | `train.steps` | env `reward` (a grader) |

> **Step- vs epoch-driven.** GRPO consumes `train.steps` (optimizer steps); SFT
> consumes `train.epochs`. The config validator rejects a non-positive count for
> whichever the algorithm uses, so a wrong-axis config fails before a worker is
> provisioned.

## Which one should I use?

Pick by **what you already have**:

- **Reference completions** (gold answers, demonstrations) → **SFT**. Always the
  cheapest, and the right first run to validate the pipeline end-to-end.
- **A programmatic grader / verifiable reward** → **GRPO**. The model bootstraps
  from its own correct samples.

A common sequence is **SFT then GRPO**: SFT to teach the format and basic
competence, then RL to push capability. Chain them by pointing the GRPO run at the
SFT adapter via `train.init_from_adapter`.

Both are illustrated on one task in [`examples/gsm8k/`](../examples/gsm8k/), which
ships a ready-to-run config for each.

---

## SFT

Supervised fine-tuning: the model is trained to reproduce the environment's
`sft_target(example)` for each training example (for GSM8K, the gold
chain-of-thought reformatted to end in `\boxed{ANSWER}`). No reward, no generation
during training — a plain LoRA fit, so it is the cheapest arm and fits the smallest
GPUs.

```toml
model     = "Qwen/Qwen3-0.6B"
algorithm = "sft"

[environment]
id = "gsm8k"

[train]
epochs     = 3
lora_rank  = 32
lora_alpha = 64
```

- **Targets must match what you want at inference.** With `thinking = true`, the
  targets should contain `<think>` traces; the worker warns loudly when none do
  (training thinking-rendered prompts on non-reasoning targets teaches the model to
  skip thinking).
- **Knobs** (env-var passthrough): `SFT_EPOCHS`, `SFT_PER_DEVICE_BS` (default 4;
  grad-accum fills the recipe's effective batch), `SFT_MAX_STEPS` /
  `SFT_MAX_EXAMPLES` (caps, used by the pre-flight smoke), `SFT_PACKING=1` (pack
  short examples into full sequences — useful for GSM8K, whose targets are far
  shorter than the sequence length, so unpacked batches are mostly padding),
  `SFT_LIGER=1` (Liger fused kernels; needs `AUTOSLM_WORKER_EXTRA_DEPS=liger-kernel`).

## GRPO

Group-Relative Policy Optimization — verifiable-reward RL. Per step the model
samples a group of completions per prompt through a **colocated vLLM engine**, the
environment's `reward(completion, example)` scores each, and GRPO shifts
probability mass toward the higher-reward completions in each group. Needs a grader.

```toml
model     = "Qwen/Qwen3-4B-Instruct-2507"
algorithm = "grpo"

[environment]
id = "gsm8k"

[train]
steps     = 150
lora_rank = 32
```

- **GPU is managed by default** — omit `gpu.type` (or set `"cheapest"`) and the
  allocator picks the cheapest class with enough VRAM for GRPO (≥24 GB for a 4B
  run). Pin a class only to force one.
- **Single-GPU memory is the hard part.** The trainer and the rollout engine share
  one card. The defaults and known-good recipes (resident vs. sleep-mode vLLM,
  `RL_VLLM_GPU_UTIL`, `RL_PER_DEVICE_PROMPTS`, allocator tuning) are in
  [config-reference.md → Colocated GRPO on one consumer GPU](config-reference.md#colocated-grpo-on-one-consumer-gpu).
- **Reward shaping lives in the environment** (`reward`), so scoring is defined in
  one place alongside the task.
- **Knobs**: `RL_PROMPTS_PER_STEP` (unique prompts per step, default 64),
  `RL_GROUP_SIZE` (completions per prompt, default 8), `RL_PER_DEVICE_PROMPTS`
  (completion micro-batch; 8 fastest for 4B on a 5090, 2 under `thinking`),
  `RL_MAX_COMPLETION` (320 non-thinking / 1536 thinking), `RL_VLLM_SLEEP`,
  `RL_VLLM_GPU_UTIL`, `RL_VLLM_MAX_LEN`.
- **Cost note:** rollout cost scales linearly with completion length — `thinking`
  runs generate roughly 5× the tokens per step.

---

## Common to both arms

- **Adapter shape.** `train.lora_rank` (default 32) and `train.lora_alpha`
  (default 64) set the LoRA rank and scale (`alpha/rank`); both must be positive
  (a non-positive alpha trains a no-op adapter). For natively-multimodal
  checkpoints the vision tower is excluded and the model is trained/served
  text-only.
- **Seeds.** `train.seeds = [0, 1, ...]` runs one **dedicated GPU per seed** in
  parallel; each is an independent durable job.
- **Metrics.** Each run streams the trainer's loss curve (and, for GRPO, the
  reward-per-step history captured in the metrics notes) to the artifact repo;
  there is no separate evaluation phase.
- **Thinking mode** (`thinking = true`) applies uniformly — SFT targets, RL
  rollouts, and serving all render with the same reasoning flag. See
  [config-reference.md → Thinking mode](config-reference.md#thinking-mode-thinking--true).
- **Reliability.** Each seed is a durable job: checkpoints stream to the artifact
  repo, and a dead worker is resumed from the latest checkpoint on a fresh endpoint
  (bounded by `gpu.max_retries`). `slm attach <run_id>` re-attaches after a client
  crash.

See [config-reference.md](config-reference.md) for the full TOML schema and the
complete environment-variable list, and [environments.md](environments.md) for
authoring the environment that feeds these algorithms.
