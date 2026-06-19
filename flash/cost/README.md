# `flash.cost` — training-cost estimator

Pre-flight **USD cost estimates** for a Flash SFT/GRPO run, plus an experiment that
measures how well an LLM can reproduce those estimates as it is fed more of the Flash
framework. Built on top of the GPU cost/sizing matrix from PR #185 — it consumes the
same pricing (`providers/base.py`), VRAM matrix (`engine/vram.py`), recipe
(`engine/recipe.py`), and cheapest-fit allocation (`providers/allocator.py`) the runner
already uses, so a quote matches what a real run would be billed.

## Two estimators over one `RunConfig`

| | What it is |
|---|---|
| **`estimate_cost`** (analytical) | Deterministic, first-principles model — the **ground truth**. `cost = wall-clock hours x GPU $/hr`; wall clock = cold-start setup + `steps x seconds_per_step`; per-step time is a FLOPs estimate (`multiplier x active_params x tokens / (peak_bf16 x MFU)`). GRPO splits each step into a vLLM rollout + the policy/reference update. Offline, no creds. |
| **`LLMCostEstimator(version)`** | Claude (`claude-opus-4-8`) prices the same run under a prompt loaded with `version` layers of framework knowledge (`prompts.py`). |

```python
from flash.cost import RunConfig, estimate_cost

e = estimate_cost(RunConfig("Qwen/Qwen3.5-4B", "grpo", steps=150))
print(e.describe())   # ... -> $4.62 on A100 PCIe@auto ($1.39/hr x 3.32h)
print(e.breakdown())  # itemized: GPU pick, setup, per-step, wall clock, total, notes
```

## CLI

```bash
python -m flash.cost estimate --model Qwen/Qwen3.5-9B --method grpo --steps 300
python -m flash.cost experiment --effort low --out runs/cost      # live (needs ANTHROPIC_API_KEY)
python -m flash.cost experiment --offline --out runs/cost         # deterministic stub, no API
```

The experiment writes `report.md`, `results.json`, and `mape.png`.

## The prompt-convergence experiment

The independent variable is the system prompt. v1 asks Claude to price a run with no
Flash-specific knowledge; each later version appends one more layer of framework fact
(every layer is rendered from the live framework data, so the prompts can't drift):

```
v1 base        v2 +pricing      v3 +GPU pick
v4 +timing     v5 +GRPO/MoE     v6 +formula
```

For each version we price a grid of runs and measure MAPE (mean absolute % error)
against the analytical reference. Two grids ship (`--grid default|diverse`):

- **default** — 24 runs: 4 models x {SFT, GRPO} x {100, 500, 1000} steps.
- **diverse** — 8 runs that vary *every* axis at once: GPU class (RTX 4090 / A5000 /
  3090 / 5090 / Pro 6000 WK / A100 PCIe / H100 NVL), algorithm (SFT + GRPO), settings
  (seq_len, LoRA rank, group size, completion budget, thinking, batch), and verifiers
  environment. Each pinned GPU is validated and fits, so the estimate prices on exactly
  that card.

### Captured run (`claude-opus-4-8`, 24-run grid)

Full artifacts in [`../../cost_estimator_results/`](../../cost_estimator_results/)
(`report.md`, `results.json`, `mape.png`).

| Version | Adds | MAPE | SFT | GRPO | GPU-pick acc |
|---|---|---:|---:|---:|---:|
| v1 base | base task only | 257% | 350% | 164% | 0% |
| v2 +pricing | GPU price/VRAM catalog | 171% | 260% | 82% | 0% |
| v3 +GPU pick | cheapest-fit selection | 59% | 73% | 45% | 71% |
| v4 +timing | per-step compute model | 169% | **12%** | **326%** | 75% |
| v5 +GRPO/MoE | rollout split, MoE, QLoRA | 17% | 13% | 22% | 71% |
| v6 +formula | cold-start + exact formula | **5.7%** | 11% | **0.5%** | 67% |

**257% → 5.7%: a 98% error reduction.** Two findings worth calling out:

- **GPU-pick accuracy jumps 0% → 71%** the moment v3 supplies the cheapest-validated-fit
  rule. General priors send runs to big datacenter GPUs; Flash runs land on
  consumer/prosumer cards.
- **Partial knowledge can hurt.** v4 adds the per-step *SFT* timing model but not the
  GRPO rollout split (that arrives in v5). SFT error collapses (73% → 12%), but GRPO
  error *triples* (45% → 326%) because the estimator applies SFT-shaped arithmetic to
  GRPO's 512-completion rollouts and massively over-counts (e.g. 9B GRPO x1000 → $232
  vs. $24). v5's rollout model fixes it (→ 22%) and v6's exact formula + wall cap nails
  it (→ 0.5%). The residual at v6 is all SFT: Claude assumes the 9B *QLoRA* SFT fits a
  cheap 24 GB card and under-prices it.

### Diverse run (`--grid diverse`, 8 runs across GPUs/settings/envs/algorithms)

Artifacts in [`../../cost_estimator_results/diverse/`](../../cost_estimator_results/diverse/).

| Version | MAPE | Version | MAPE |
|---|---:|---|---:|
| v1 base | 48% | v4 +timing | 119% |
| v2 +pricing | 49% | v5 +GRPO/MoE | 33% |
| v3 +GPU pick | 34% | **v6 +formula** | **0.2%** |

At v6, every one of the 8 runs lands within 0.6% of ground truth — correctly pricing
on each pinned card (e.g. `9B grpo / A100 / thinking g16` → \$33.36 exact;
`9B sft / H100NVL / seq4096` → \$6.31; `2B sft / RTX3090 / thinking` → \$4.11) — vs.
v1's 16–90% per-run errors. The same v4 GRPO-timing dip appears here too.

## Real-run validation (measured ground truth)

Grading the LLM against the analytical equation is partly circular (v6's prompt
*is* the equation). The honest test is real GPU runs. `measured.py` maps a
control-plane run status → the `RunConfig` it priced + its actual billed `cost_usd`;
`cost_estimator_results/real_runs/analyze.py` then grades **both** estimators against
**13 completed real runs** (SFT + GRPO; MiniCPM-1B / Qwen3.5-0.8B/2B/4B/9B; RTX
3090/4090/5090 + A100 PCIe; real reward envs gsm8k / linkd-search / support-ticket /
flash-bench). Full results in [`../../cost_estimator_results/real_runs/`](../../cost_estimator_results/real_runs/).

Real costs are tiny ($0.03–$0.77), so mean MAPE is dominated by a single bad estimate
on a $0.03 run — **median APE is the robust metric**.

**Analytical equation vs measured:** 115% MAPE (static $/hr) → **77%** (actual billed
rate) → **54%** (actual rate + cold-start calibrated to these runs). The biggest gaps
are real-world, not formula errors: the runs hit vast/runpod **spot** rates well below
the static fallback, and short runs cold-start faster than the default overhead assumes.

**LLM vs measured (median APE):** v1 ~410% → v4–v6 **~42–77%**. Framework knowledge
takes the estimate from ~10× off real cost down to within ~2×.

The lesson: **converging to the equation (5.7%) is not converging to reality (~50%).**
Prompting gets the LLM to reproduce the model; closing the last gap to *measured* cost
needs live spot pricing + cold-start calibration (and a real reward-latency measurement
per env), not a richer prompt. That's the difference between a plausible quote and a
billed one.

## Ground-truth scope

The analytical model is a transparent, calibrated reference, not a per-run measurement
harness: the compute constants (MFU, FLOPs multipliers, cold-start overhead) are tuned
so end-to-end figures land in a realistic band (cents for a small SFT smoke; a few-to-
tens of dollars for a large GRPO run), and GPU price/VRAM/selection come verbatim from
the live registry. Swap in measured per-step throughput and it becomes exact.

## Tests

Six suites under `tests/` (`test_cost_{hardware,models,analytical,estimate,prompts,experiment}.py`),
all CPU-only — the experiment tests drive the harness with a deterministic offline stub.

```bash
FLASH_SKIP_NET=1 uv run pytest tests/test_cost_*.py
```

Deps: the analytical model and tests need nothing extra; the LLM estimator and PNG are
the optional `cost` extra (`uv sync --extra cost` → `anthropic`, `matplotlib`), imported
lazily.
