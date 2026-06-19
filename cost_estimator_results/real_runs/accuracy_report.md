# Cost equation accuracy vs measured RunPod/Vast cost

`flash.cost.estimate_cost` is **fully equation-based**: `cost = wall-clock hours x market
$/hr`, every term a real quantity (FLOPs, MFU, cold-start seconds, concurrent reward
grading, the spot/queue rate the provider bills). **There is no output multiplier** -- the
dollar figure is not scaled to hit any target. The numbers below are whatever the equation
gives.

## Accuracy vs measured cost (n=56 real runs)

| Group | n | mean MAPE | median APE | agg bias (Σest/Σmeas) | within 33% | within 50% |
|---|---:|---:|---:|---:|---:|---:|
| sft | 13 | 116% | 51% | 1.020 | 31% | 46% |
| grpo | 43 | 69% | 40% | 0.852 | 49% | 65% |
| all | 56 | 80% | 41% | 0.865 | 45% | 61% |

`agg bias` is **not forced to 1.0** -- forcing it would be exactly the kind of output hack
this equation avoids. The residual per-run error is largely irreducible: measured wall =
provider queue + container boot + model download + train, and that scheduling/cold-start
overhead swings by minutes in ways no config predicts. So the equation targets an unbiased
central estimate with a typical run within ~30-50%, not a fake-precise point.

## Calibrated physical inputs (realized market $/hr, from 56 runs' billing)

The equation prices at the **realized (spot/queue) rate** providers actually bill, not the
on-demand list price. Median of the providers' recorded rate per class:

"A100 PCIe": 1.035,
"A100 SXM": 1.133,
"RTX 3090": 0.238,
"RTX 4090": 0.412,
"RTX 5090": 0.863,
"RTX 6000 Ada": 0.601,
"RTX A5000": 0.304

## Environment cost sweep (GRPO; cost varies with the reward grader)

Reward grading is modeled **concurrent** (parallel grader slots), so a heavy LLM-judge env
no longer saturates the wall cap the way a fully-serial model would.

| Model | Environment | reward s/compl | cost $ |
|---|---|---:|---:|
| openbmb/MiniCPM5-1B | openai/gsm8k | 0.01 | 0.72 |
| openbmb/MiniCPM5-1B | stanford/sentiment-classify | 0.15 | 0.83 |
| openbmb/MiniCPM5-1B | primeintellect/web-search | 0.60 | 1.17 |
| openbmb/MiniCPM5-1B | freesolo/support-ticket | 3.00 | 3.01 |
| openbmb/MiniCPM5-1B | swe/code-exec-contest | 3.00 | 3.01 |
| openbmb/MiniCPM5-1B | acme/custom-unlisted-env | 0.30 | 0.94 |
| openbmb/MiniCPM5-1B | **AVERAGE (across envs)** |  | 1.61 |
| Qwen/Qwen3.5-4B | openai/gsm8k | 0.01 | 2.06 |
| Qwen/Qwen3.5-4B | stanford/sentiment-classify | 0.15 | 2.19 |
| Qwen/Qwen3.5-4B | primeintellect/web-search | 0.60 | 2.60 |
| Qwen/Qwen3.5-4B | freesolo/support-ticket | 3.00 | 4.81 |
| Qwen/Qwen3.5-4B | swe/code-exec-contest | 3.00 | 4.81 |
| Qwen/Qwen3.5-4B | acme/custom-unlisted-env | 0.30 | 2.33 |
| Qwen/Qwen3.5-4B | **AVERAGE (across envs)** |  | 3.13 |
| ALL MODELS | **GRAND AVERAGE** |  | 2.37 |

---
_Regenerate: `FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/accuracy.py`._
