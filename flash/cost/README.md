# `flash.cost` — training-cost estimator

Pre-flight **USD cost estimates** for a Flash SFT/GRPO run, **calibrated to break even
against measured cost**. Built on the GPU cost/sizing matrix from PR #185 — it consumes
the same pricing (`providers/base.py`), VRAM matrix (`engine/vram.py`), recipe
(`engine/recipe.py`), and cheapest-fit allocation (`providers/allocator.py`) the runner
uses, so a quote tracks what a real run is billed.

## Two figures over one `RunConfig`

| | What it is |
|---|---|
| **`estimate_cost`** (analytical) | Deterministic, first-principles model. `cost = wall-clock hours x GPU $/hr`; wall clock = cold-start setup + `steps x seconds_per_step`; per-step time is a FLOPs estimate (`multiplier x active_params x tokens / (peak_bf16 x MFU)`). GRPO splits each step into a vLLM rollout + reward grading + the policy/reference update. Offline, no creds. This is the un-calibrated *reference*. |
| **`breakeven_estimate`** (calibrated) | `estimate_cost` scaled by a per-method break-even factor so the summed quote equals measured cost. **This is the figure to quote.** |

```python
from flash.cost import RunConfig, breakeven_estimate

e = breakeven_estimate(RunConfig("Qwen/Qwen3.5-9B", "grpo", steps=300, environment="gsm8k"))
print(e.breakdown())   # itemized: GPU pick, setup, per-step, wall clock, raw vs break-even, total
```

## CLI

```bash
python -m flash.cost estimate --model Qwen/Qwen3.5-9B --method grpo --steps 300   # break-even quote
python -m flash.cost estimate --model Qwen/Qwen3.5-9B --method grpo --steps 300 --raw  # reference
python -m flash.cost breakeven                                                    # centering + env sweep
```

## Break-even calibration

The first-principles model is transparent but runs **~1.42x high** versus MEASURED
control-plane cost over real runs (SFT 1.29x, GRPO 1.90x). Two real-world gaps, neither
a formula bug: the static fallback `$/hr` sits above the spot/queue rate runs are billed
at, and fixed cold-start overhead over-weights short cheap runs.

For *pricing* we want quotes that, summed over a workload, equal what runs actually cost
— the platform **breaks even**. The unbiased estimator for that is the ratio of totals,
applied per method (`calibration.py`):

> break-even factor = Σ measured cost / Σ analytical (static) cost

| Method | factor | effect |
|---|---:|---|
| SFT | **0.776** | centers the 9 SFT runs (Σ quote = Σ measured) |
| GRPO | **0.528** | GRPO's rollout + reward + vLLM-init is over-estimated ~2x harder |
| global | 0.703 | whole-portfolio fallback |

Per-method calibration makes each method break even on its own, so the aggregate does
too. `estimate_cost` is **left untouched** (it stays the transparent reference);
`breakeven_estimate` wraps it and records the multiplier on `CostEstimate.calibration_factor`.

The factors are **regenerated from the committed measured data** (`breakeven_factor_from_real_runs`)
and pinned by tests, so a data refresh that moves them fails loudly. As new RunPod/Vast
runs are measured (`measured.py` parses run status → `RunConfig` + billed `cost_usd`),
re-derive and update the constants.

```bash
FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/breakeven.py
```

writes [`cost_estimator_results/real_runs/breakeven_report.md`](../../cost_estimator_results/real_runs/breakeven_report.md):
the centering proof (Σ quote ≈ Σ measured, ratio ~1.00) and the per-environment cost
sweep (a GRPO run priced across reward-latency tiers, with the average).

### Known over-estimate (next calibration target)

Reward latency is modeled as **fully serial** across all completions
(`reward_s_per_completion x prompts x group_size`), so heavy-reward envs (LLM-judge,
code-exec at ~3 s/completion x 512) blow past the 24h wall cap. Real verifiers envs
grade concurrently — adding a reward-concurrency factor, calibrated against real
heavy-env runs, is the #1 open item.

## Tests

CPU-only suites under `tests/` (`test_cost_{hardware,models,analytical,estimate,rewards,measured,calibration}.py`).

```bash
FLASH_SKIP_NET=1 uv run pytest tests/test_cost_*.py
```

The estimator is pure-Python — no optional extras required.
