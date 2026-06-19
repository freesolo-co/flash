# Break-even calibration: center the equation on measured cost

The first-principles analytical model (`flash.cost.estimate_cost`) is the experiment's
*reference*, but graded against MEASURED control-plane cost it runs **~1.42x high**
(SFT 1.29x, GRPO 1.90x). For pricing we want quotes that, summed over a workload,
equal what the runs actually cost -- the platform **breaks even**. The unbiased
estimator for that is the ratio of totals, applied per method:

> break-even factor = Σ measured cost / Σ analytical (static) cost

Published constants (`flash.cost.calibration.BREAKEVEN_FACTORS`), regenerated from the
committed `summary_real.json`:

| Method | factor (published) | factor (recomputed) |
|---|---:|---:|
| SFT | 0.776 | 0.7762 |
| GRPO | 0.528 | 0.5277 |
| global | 0.703 | 0.7027 |

## 1. Centering — does the calibrated equation break even?

Apply the per-method factor to every measured run's analytical quote; the calibrated
total should equal the measured total (ratio 1.0).

| Group | n | Σ measured $ | Σ break-even quote $ | ratio |
|---|---:|---:|---:|---:|
| sft | 9 | 1.4223 | 1.4219 | 1.000 |
| grpo | 4 | 0.4066 | 0.4068 | 1.000 |
| all | 13 | 1.8289 | 1.8287 | 1.000 |

**Σ calibrated quote = Σ measured cost (ratio ≈ 1.00): the equation now sits right
down the middle.** Per-run spread remains (this centers the *total*, not each run);
tightening the per-run error is the job of the environment-testing loop below.

## 2. Environment cost sweep — average GRPO cost across reward tiers

Environment enters cost only through the GRPO reward grader (`rewards.py`): a free
exact-match check vs. a seconds-per-completion LLM-judge. Same run otherwise.

| Model | Environment | reward s/compl | raw $ | break-even $ |
|---|---|---:|---:|---:|
| openbmb/MiniCPM5-1B | openai/gsm8k | 0.01 | 1.06 | 0.56 |
| openbmb/MiniCPM5-1B | stanford/sentiment-classify | 0.15 | 3.03 | 1.60 |
| openbmb/MiniCPM5-1B | primeintellect/web-search | 0.60 | 9.37 | 4.95 |
| openbmb/MiniCPM5-1B | freesolo/support-ticket ⚠cap | 3.00 | 23.76 | 12.55 |
| openbmb/MiniCPM5-1B | swe/code-exec-contest ⚠cap | 3.00 | 23.76 | 12.55 |
| openbmb/MiniCPM5-1B | acme/custom-unlisted-env | 0.30 | 5.14 | 2.71 |
| openbmb/MiniCPM5-1B | **AVERAGE (across envs)** |  |  | 5.82 |
| Qwen/Qwen3.5-4B | openai/gsm8k | 0.01 | 3.33 | 1.76 |
| Qwen/Qwen3.5-4B | stanford/sentiment-classify | 0.15 | 6.10 | 3.22 |
| Qwen/Qwen3.5-4B | primeintellect/web-search | 0.60 | 14.99 | 7.92 |
| Qwen/Qwen3.5-4B | freesolo/support-ticket ⚠cap | 3.00 | 33.36 | 17.61 |
| Qwen/Qwen3.5-4B | swe/code-exec-contest ⚠cap | 3.00 | 33.36 | 17.61 |
| Qwen/Qwen3.5-4B | acme/custom-unlisted-env | 0.30 | 9.06 | 4.79 |
| Qwen/Qwen3.5-4B | **AVERAGE (across envs)** |  |  | 8.82 |
| ALL MODELS | **GRAND AVERAGE** |  |  | 7.32 |

**Grand average GRPO break-even cost ≈ $7.32.** Rows marked ⚠cap saturate the
24h wall cap because reward latency is modeled as **fully serial** across all 512
completions (3.0 s/compl x 512 ~ 25 min/step). Real verifiers envs grade concurrently,
so this over-estimates heavy-reward GRPO — the **#1 fix** for the env-testing loop
(add a reward-concurrency factor, calibrated against real heavy-env runs).

---
_Regenerate: `FLASH_SKIP_NET=1 uv run python cost_estimator_results/real_runs/breakeven.py`._
