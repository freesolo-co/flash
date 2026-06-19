# Reproducing "Self-CTRL" (arXiv:2606.18327) with Flash

This directory reproduces **"Self-CTRL: Self-Consistency Training with Reinforcement
Learning"** (Pres, Ruis, Ghebreselassie, Li, Andreas) on the Flash LoRA service.

Self-CTRL uses RL to optimize **consistency between a model's self-explanation (what it
says it will do) and its actual behavior**, after an SFT warm-start. Headline results:

| Task | Metric | Paper |
|---|---|---|
| Biased-sampler reasoning | R² (self-reported vs. measured bias) | 0.24 → 0.64 |
| Constitutional-AI refusal | refusal-prediction accuracy | 36% → 92% |
| Constitutional-AI refusal | HarmBench failure rate | 15.0% → 0.5% |

## What's here

```
environments/selfctrl_sampler/selfctrl_sampler.py   # biased-sampler verifiers env
environments/selfctrl_refusal/selfctrl_refusal.py   # refusal verifiers env
configs/sampler_sft.toml   configs/sampler_grpo.toml
configs/refusal_sft.toml   configs/refusal_grpo.toml
tests/test_selfctrl_envs.py                          # CPU-only reward-logic tests
```

Each env serves **both phases**: SFT reads the gold `answer` (a self-consistent
demonstration) via the adapter's `sft_target`; GRPO uses the rubric.

## How the paper maps onto Flash

- **REINFORCE → GRPO.** Flash's `algorithm = "grpo"` is group-normalized REINFORCE, the
  in-package analog of the paper's policy gradient.
- **Full fine-tune of Llama 3.1 8B → LoRA on a catalog Qwen model.** Flash is LoRA-only;
  these configs use `Qwen/Qwen3.5-4B` (RTX 5090). Swap to `Qwen/Qwen3.5-9B` (QLoRA) for a
  larger run via `--set model=Qwen/Qwen3.5-9B`.
- **Consistency reward → an intra-completion rubric.** Flash scores one completion at a
  time and rejects group/batch reward funcs (`flash/envs/adapter.py`), so the model emits
  *both* its self-explanation and its behavior in one completion and the reward measures
  their agreement:
  - **sampler:** reward = `1 - TV(stated distribution, empirical distribution of the
    completion's own draws)`. The paper's R² is a weight-0 eval metric
    (`r2_self_vs_empirical`).
  - **refusal:** reward = consistency (predicted refuse/comply == actual behavior) +
    constitutional term (refuse harmful, comply benign). `refusal_prediction_accuracy`
    and `harmbench_failure` are weight-0 eval metrics.

## Run it

Prereqs: `uv sync --extra server`, `slm login --api-key <key>`, the `prime` CLI for
`slm env push`, and an HF repo for artifacts (set `[train] hf_repo`).

```bash
# 1. Publish the environments (note the owner/name slug each prints)
slm env push environments/selfctrl_sampler/selfctrl_sampler.py
slm env push environments/selfctrl_refusal/selfctrl_refusal.py

# 2. Point the configs at your slugs + HF repo
#    edit [environment] id and [train] hf_repo in configs/*.toml
#    (or override per-run with --set environment.id=... --set train.hf_repo=...)

# 3. Validate locally (no GPU, no credentials)
slm train configs/sampler_grpo.toml --dry-run

# 4. Phase 1 — SFT warm-start
slm train configs/sampler_sft.toml
slm train configs/refusal_sft.toml

# 5. Phase 2 — GRPO, warm-started from the SFT adapter
#    take the SFT run id from step 4 and set it as init_from_adapter:
slm train configs/sampler_grpo.toml --set train.init_from_adapter=sft/<sft-run-id>/seed0
slm train configs/refusal_grpo.toml --set train.init_from_adapter=sft/<sft-run-id>/seed0

# 6. Inspect a trained adapter
slm deploy <grpo-run-id>
slm chat <grpo-run-id> -m "<a sampler or refusal prompt>"
```

Watch the mid-run eval series (`eval_every_steps`): `r2_self_vs_empirical` should climb
for the sampler; `refusal_prediction_accuracy` should climb and `harmbench_failure`
should fall for refusal.

### Cheap smoke before committing compute

```bash
slm train configs/sampler_grpo.toml --set model=Qwen/Qwen3.5-0.8B \
  --set gpu.type="RTX 4090" --set train.steps=10
```

### Local reward-logic tests (CPU, no GPU)

```bash
FLASH_SKIP_NET=1 uv run --extra server pytest tests/test_selfctrl_envs.py -q
```

## Fidelity caveats

- **LoRA, not full FT; Qwen, not Llama 3.1 8B** — absolute numbers will differ from the
  paper; the goal is reproducing the *direction and magnitude* of the consistency gains.
- **GRPO ≈ REINFORCE** (group-normalized) — faithful but not identical.
- **Intra-completion consistency reward** is a per-completion reformulation of the paper's
  distribution-level metric, required by Flash's per-completion scoring; the weight-0 eval
  metrics report the paper-style aggregates.
- **Reconstructed data.** The paper releases no code/data, so the datasets are rebuilt from
  its descriptions: the biased-sampler data is *synthetic* (generated in
  `selfctrl_sampler.build_rows`); the refusal env ships a small illustrative harmful/benign
  seed and a deterministic refusal classifier (`is_refusal`). For a serious run, load the
  public **HarmBench** behaviors + a benign mix and/or swap in an LLM judge
  (`vf.JudgeRubric`, which Flash's adapter wires through automatically).
- **Hyperparameters** (`learning_rate`, `kl_penalty_coef`, `steps`) are set to sane
  GRPO-recipe values; confirm against the PDF before a full reproduction run.
