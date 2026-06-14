# GSM8K example environment

The simplest end-to-end AutoSLM task: grade-school math (GSM8K) with a verifiable
numeric answer. It is the reference example for **both algorithms** — SFT and GRPO
train against this one environment, and both are scored by the same grader.

The registry path-loads this folder as the built-in environment id `gsm8k`
(`autoslm/envs/registry.py`), so any config can use it with:

```toml
[environment]
id = "gsm8k"
```

## Layout

Each concern is a small, independently readable module:

| file | responsibility |
|---|---|
| `env.py` | the `GSM8KEnvironment` the registry loads as `gsm8k`; wires data + grading into the `Environment` interface (`dataset`, `prompt_messages`, `sft_target`, `reward`, `grade`) |
| `data.py` | dataset loading (`openai/gsm8k`), the shared system prompt, and SFT-target construction |
| `grading.py` | the dependency-free grader — answer extraction (`\boxed{}` › `####` › last number) + numeric equivalence. The **single source of truth** for scoring, shared by the SFT target check and the GRPO reward/grading |
| `__init__.py` | re-exports `load_environment` / `GSM8KEnvironment` for the path loader |
| `gsm8k_*.toml` | one ready-to-run config per algorithm (below) |

The split mirrors the design rule in the package README: **task-specific grading
lives with its example, not in the engine**, so every training arm scores a model
identically.

## Configs

| config | algorithm | model(s) | needs | GPU |
|---|---|---|---|---|
| `gsm8k_sft.toml` | SFT | Qwen3-0.6B | gold completions (env `sft_target`) | managed |
| `gsm8k_grpo.toml` | GRPO | Qwen3-4B-Instruct | a grader (env `reward`) | managed (≥24 GB) |

> **GPU and disk are fully managed.** Neither config pins `gpu.type` — it defaults
> to `"cheapest"`, so the allocator picks the cheapest fitting GPU across providers
> at submit time, and disk auto-sizes (64 GB default, raised for big-checkpoint
> models). Set `gpu.type = "RTX 4090"` (etc.) only to force a class.

Run either of them:

```bash
slm train examples/gsm8k/gsm8k_sft.toml      # cheapest; validates the pipeline first
slm train examples/gsm8k/gsm8k_grpo.toml
```

Override any value without editing the file:

```bash
slm train examples/gsm8k/gsm8k_grpo.toml --set train.steps=50 --set gpu.type="RTX 4090"
slm train examples/gsm8k/gsm8k_sft.toml  --set train.seeds=[0,1,2]   # one GPU per seed
```

A new run uploads its LoRA adapter to the operator's HF artifact repo and reports
training metrics (loss, reward history, token throughput, wall time). Then serve it:

```bash
slm deploy <run_id> --mode dev
slm chat   <run_id> -m "Natalia sold clips to 48 friends..."
```

See `docs/algorithms.md` for when to pick each algorithm and `docs/config-reference.md`
for the full TOML schema (and the colocated-GRPO memory recipes for single-GPU runs).

## Use it as a template

To author your own task, copy this folder, keep the same three-file split, and
implement the `Environment` interface (`docs/environments.md`). Keep one grader as
the single source of truth so the SFT target check and GRPO reward/grading agree.
Publish a custom
environment with `slm env push` (the managed service does not load local
`[environment] path` dirs).
