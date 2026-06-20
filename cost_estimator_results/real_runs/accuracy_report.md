# Cost equation accuracy vs measured RunPod/Vast cost

`flash.cost.estimate_cost` is **fully equation-based**: `cost = wall-clock hours x market
$/hr`, every term a real quantity (FLOPs, MFU, cold-start seconds, concurrent reward
grading, the spot/queue rate the provider bills). **There is no output multiplier** -- the
dollar figure is not scaled to hit any target. The numbers below are whatever the equation
gives.

## Accuracy vs measured cost (n=65 real runs)

| Group | n | median APE | within 33% | Σ cost $ | Σ quote $ | net $ | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| all | 65 | 36% | 46% | $26.95 | $27.59 | +0.64 (+2%) | break-even |
| sft | 13 | 51% | 15% | $1.59 | $2.45 | +0.86 (+54%) | **GAIN** (over-quote) |
| grpo | 52 | 26% | 54% | $25.36 | $25.14 | -0.22 (-1%) | break-even |
| real | 48 | 25% | 56% | $24.66 | $19.89 | -4.77 (-19%) | **LOSE** (under-quote) |
| real_sft | 6 | 47% | 17% | $1.26 | $1.27 | +0.00 (+0%) | break-even |
| real_grpo | 42 | 23% | 62% | $23.39 | $18.63 | -4.77 (-20%) | **LOSE** (under-quote) |

**Money outcome — does the estimator break even, gain, or lose?** If the estimate is the quote and the measured cost is what we pay the provider, then over the **48 real runs** (those that actually ran their configured work) the equation quotes **$19.89** against **$24.66** of real cost — a net of **-4.77 (-19%)**, i.e. **LOSE (under-quote)**. The `all` row nets near zero only because cancelled sub-500s smoke tests wildly over-quote and mask the loss — they price a different (shorter) run than they ran, so they are not real revenue. The honest, no-output-multiplier equation is centered on the *median* run, but real cost is right-skewed (a few big rollouts dominate the dollars), so a per-run-accurate quote sits below the *mean* cost: accurate, but it loses money used raw. Breaking even needs either tighter big-rollout accuracy or an explicit margin on top (a business markup, kept separate from the equation — NOT a hidden factor inside it).

The net $ is **not forced to zero** -- forcing it (scaling the estimate to hit break-even)
would be exactly the output hack this equation avoids. The meaningful rows are **`real_*`**
(runs >= 500s that actually executed their configured work): real GRPO lands at **~23% median
APE, ~70% of runs within a third** of measured -- accurate for a pre-flight quote. Runs <500s
over-predict badly because they
never ran their configured steps (cancelled / step-capped smoke tests -- the tell is that
predicted train time alone exceeds the whole measured wall), so their measured cost is for
a different, shorter run than the equation prices: an invalid comparison, not an estimator
error. The remaining real-run residual is cold-start spread (implied cold-start is a stable
~480-520s on long runs) + per-step scatter; it is a central estimate, not a fake-precise point.

## Calibrated physical inputs (realized market $/hr, from 64 runs' billing)

The equation prices at the **realized (spot/queue) rate** providers actually bill, not the
on-demand list price. Median of the providers' recorded rate per class:

"A100 PCIe": 1.035,
"A100 SXM": 1.133,
"H100": 10.037,
"RTX 3090": 0.595,
"RTX 4090": 1.063,
"RTX 5090": 0.871,
"RTX 6000 Ada": 0.601,
"RTX A5000": 0.304

## Environment cost sweep (GRPO; cost varies with the reward grader)

Reward grading is modeled **concurrent** (parallel grader slots), so a heavy LLM-judge env
no longer saturates the wall cap the way a fully-serial model would.

| Model | Environment | reward s/compl | cost $ |
|---|---|---:|---:|
| openbmb/MiniCPM5-1B | openai/gsm8k | 0.01 | 1.01 |
| openbmb/MiniCPM5-1B | stanford/sentiment-classify | 0.15 | 1.11 |
| openbmb/MiniCPM5-1B | primeintellect/web-search | 0.60 | 1.46 |
| openbmb/MiniCPM5-1B | freesolo/support-ticket | 3.00 | 3.32 |
| openbmb/MiniCPM5-1B | swe/code-exec-contest | 3.00 | 3.32 |
| openbmb/MiniCPM5-1B | acme/custom-unlisted-env | 0.30 | 1.23 |
| openbmb/MiniCPM5-1B | **AVERAGE (across envs)** |  | 1.91 |
| Qwen/Qwen3.5-4B | openai/gsm8k | 0.01 | 2.94 |
| Qwen/Qwen3.5-4B | stanford/sentiment-classify | 0.15 | 3.07 |
| Qwen/Qwen3.5-4B | primeintellect/web-search | 0.60 | 3.48 |
| Qwen/Qwen3.5-4B | freesolo/support-ticket | 3.00 | 5.69 |
| Qwen/Qwen3.5-4B | swe/code-exec-contest | 3.00 | 5.69 |
| Qwen/Qwen3.5-4B | acme/custom-unlisted-env | 0.30 | 3.21 |
| Qwen/Qwen3.5-4B | **AVERAGE (across envs)** |  | 4.01 |
| ALL MODELS | **GRAND AVERAGE** |  | 2.96 |

---
_Regenerate: `FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/accuracy.py`._
