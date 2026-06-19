# Cost equation accuracy vs measured RunPod/Vast cost

`flash.cost.estimate_cost` is **fully equation-based**: `cost = wall-clock hours x market
$/hr`, every term a real quantity (FLOPs, MFU, cold-start seconds, concurrent reward
grading, the spot/queue rate the provider bills). **There is no output multiplier** -- the
dollar figure is not scaled to hit any target. The numbers below are whatever the equation
gives.

## Accuracy vs measured cost (n=56 real runs)

| Group | n | mean MAPE | median APE | agg bias (Σest/Σmeas) | within 33% | within 50% |
|---|---:|---:|---:|---:|---:|---:|
| all | 56 | 111% | 34% | 1.180 | 50% | 64% |
| sft | 13 | 186% | 51% | 1.535 | 15% | 46% |
| grpo | 43 | 89% | 23% | 1.150 | 60% | 70% |
| real | 39 | 38% | 22% | 0.902 | 64% | 79% |
| real_sft | 6 | 50% | 47% | 0.999 | 17% | 50% |
| real_grpo | 33 | 35% | 22% | 0.894 | 73% | 85% |

`agg bias` is **not forced to 1.0** -- forcing it would be exactly the output hack this
equation avoids. The meaningful rows are **`real_*`** (runs >= 500s that actually executed
their configured work): real GRPO lands at **~23% median APE, ~70% of runs within a third**
of measured -- accurate for a pre-flight quote. Runs <500s over-predict badly because they
never ran their configured steps (cancelled / step-capped smoke tests -- the tell is that
predicted train time alone exceeds the whole measured wall), so their measured cost is for
a different, shorter run than the equation prices: an invalid comparison, not an estimator
error. The remaining real-run residual is cold-start spread (implied cold-start is a stable
~480-520s on long runs) + per-step scatter; it is a central estimate, not a fake-precise point.

## Calibrated physical inputs (realized market $/hr, from 55 runs' billing)

The equation prices at the **realized (spot/queue) rate** providers actually bill, not the
on-demand list price. Median of the providers' recorded rate per class:

"A100 PCIe": 1.035,
"A100 SXM": 1.133,
"RTX 3090": 0.239,
"RTX 4090": 0.412,
"RTX 5090": 0.863,
"RTX 6000 Ada": 0.601,
"RTX A5000": 0.304

## Environment cost sweep (GRPO; cost varies with the reward grader)

Reward grading is modeled **concurrent** (parallel grader slots), so a heavy LLM-judge env
no longer saturates the wall cap the way a fully-serial model would.

| Model | Environment | reward s/compl | cost $ |
|---|---|---:|---:|
| openbmb/MiniCPM5-1B | openai/gsm8k | 0.01 | 1.00 |
| openbmb/MiniCPM5-1B | stanford/sentiment-classify | 0.15 | 1.10 |
| openbmb/MiniCPM5-1B | primeintellect/web-search | 0.60 | 1.45 |
| openbmb/MiniCPM5-1B | freesolo/support-ticket | 3.00 | 3.29 |
| openbmb/MiniCPM5-1B | swe/code-exec-contest | 3.00 | 3.29 |
| openbmb/MiniCPM5-1B | acme/custom-unlisted-env | 0.30 | 1.22 |
| openbmb/MiniCPM5-1B | **AVERAGE (across envs)** |  | 1.89 |
| Qwen/Qwen3.5-4B | openai/gsm8k | 0.01 | 2.94 |
| Qwen/Qwen3.5-4B | stanford/sentiment-classify | 0.15 | 3.07 |
| Qwen/Qwen3.5-4B | primeintellect/web-search | 0.60 | 3.48 |
| Qwen/Qwen3.5-4B | freesolo/support-ticket | 3.00 | 5.69 |
| Qwen/Qwen3.5-4B | swe/code-exec-contest | 3.00 | 5.69 |
| Qwen/Qwen3.5-4B | acme/custom-unlisted-env | 0.30 | 3.21 |
| Qwen/Qwen3.5-4B | **AVERAGE (across envs)** |  | 4.01 |
| ALL MODELS | **GRAND AVERAGE** |  | 2.95 |

---
_Regenerate: `FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/accuracy.py`._
