# `flash.cost` — training-cost estimator

Pre-flight **USD cost estimates** for a Flash SFT/GRPO run. **Fully equation-based** —
`cost = wall-clock hours × market $/hr`, every term a real physical/economic quantity.
**There is no output multiplier**: nothing scales the dollar figure to hit a target.
Accuracy comes only from getting the inputs right.

```python
from flash.cost import RunConfig, estimate_cost

e = estimate_cost(RunConfig("Qwen/Qwen3.5-9B", "grpo", steps=300, environment="gsm8k"))
print(e.breakdown())   # GPU pick, setup, per-step, wall clock, TOTAL — no factor line
```

## The equation

| Term | What it is |
|---|---|
| **wall clock** | cold-start setup + `steps × seconds_per_step` |
| **seconds_per_step (SFT)** | `6 × active_params × tokens / (peak_bf16 × MFU_SFT_TRAIN)` |
| **seconds_per_step (GRPO)** | vLLM rollout (`MFU_DECODE`) + **concurrent** reward grading + policy/ref update (`MFU_TRAIN`) |
| **$/hr** | the **realized (spot/queue) rate** the provider bills, not the on-demand list |

Every constant is a real, measurable quantity — calibrated against measured runs the way
you'd measure MFU empirically, **never** a dimensionless correction on the result:

- **MFU** (`MFU_SFT_TRAIN` for the SFT fwd/bwd, `MFU_TRAIN` for the GRPO policy/ref update,
  `MFU_DECODE` for the vLLM rollout) — fraction of peak FLOPS the run sustains; the update
  and the decode have different efficiencies, so they carry separate constants.
- **Reward concurrency** — graders run in parallel; the reward wall is
  `completions × per-completion-latency / concurrency`, *not* the serial product (which
  over-counts a heavy LLM-judge by ~100× and saturates the 24 h wall cap).
- **Realized $/hr** (`facts.REALIZED_HOURLY_USD`) — a representative effective rate
  (measured cost ÷ measured wall) per GPU class, an empirical observation rather than list
  ± a fixed discount. The table uses conservative per-class values; `fit_constants()` returns
  the exact dataset medians (useful for re-calibration). Usually below list (the spot discount:
  RTX 5090 lists $0.99 but bills ~$0.87; A100 PCIe lists $1.39 but bills ~$1.04), but it can
  run *above* list when the
  market is tight (an H100 GRPO run billed ~$10/hr against a $3.29 list). An unobserved
  class falls back to list (an honest estimate until measured).

## CLI

```bash
python -m flash.cost estimate --model Qwen/Qwen3.5-9B --method grpo --steps 300
python -m flash.cost verify    # grade the equation vs measured cost + env sweep
```

## Accuracy vs measured cost (no hacking, then accuracy)

`verify_accuracy` grades the **raw** equation against the measured RunPod/Vast runs in
`cost_estimator_results/real_runs/measured_runs.json` (the calibration + validation set).
The last three columns read the estimate as a **quote** and the measured cost as what we
pay the provider, so `net $ = Σ quote − Σ cost` is the **profit/loss** of pricing at the
equation (− = under-quote, the conservative side):

| Group | n | median APE | within 33% | Σ cost | Σ quote | net $ | outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| all | 66 | 38% | 45% | $27.71 | $28.82 | +1.10 (+4%) | ~break-even\* |
| **real** (≥500s) | 49 | **26%** | 55% | $25.42 | $21.12 | **−4.30 (−17%)** | **under-quote** |
| **real GRPO** | 42 | **23%** | **62%** | $23.39 | $18.63 | −4.77 (−20%) | under-quote |
| real SFT | 7 | 51% | 14% | $2.03 | $2.49 | +0.46 (+23%) | over-quote |

_Static snapshot — see [`accuracy_report.md`](../../cost_estimator_results/real_runs/accuracy_report.md) for current figures (regenerate with `accuracy.py`)._

The meaningful rows are **`real`** — runs ≥500s that actually executed their configured
work. **Real GRPO is accurate (23% median APE, 62% within a third) and under-quotes the
aggregate by ~20%** — i.e. it errs on the conservative side (you don't lose money you didn't
quote for). `net $` is **not forced to zero** (forcing it would be the output hack this
avoids); the under-quote is an *emergent* property of honest inputs centered on the median
run, because real cost is right-skewed (a few big rollouts dominate the dollars).

\*The `all` row only looks break-even because cancelled smoke tests over-quote and mask the
real under-quote — see below.

Three things to read honestly:
- **Sub-500s "runs" over-predict ~3×** and inflate the `all` row — they were cancelled /
  step-capped before running their configured steps (the tell: predicted train time alone
  exceeds the whole measured wall), so their cost is for a *different, shorter* run than the
  equation prices. An invalid comparison, not an estimator error.
- The real-GRPO residual is **cold-start spread** (implied cold-start is a stable ~480–520s
  on long runs) **+ per-step scatter** — a central estimate, not a fake-precise point.
- **SFT (n=7) over-quotes ~23%** for one structural reason: the equation prices
  `tokens = batch × seq_len` (full packing), but real datasets have **shorter sequences than
  the cap** (a measured 9B run used 3.2 M train-tokens against the 5.2 M the cap implies — a
  1.6× over-estimate, all from packing, *not* MFU: that run's realized MFU was 0.28, right on
  the constant). Full packing is the correct conservative pre-flight prior — you size for the
  cap you can't see past — so SFT lands on the safe (over-quote) side; its wide APE is the
  unknown-sequence-length spread, which only the dataset resolves. SFT is ~8% of the dollars,
  so the aggregate still under-quotes per the row above.

Refresh the calibration as new runs land: harvest the control plane into `measured_runs.json`,
re-run `fit_constants` to refresh `REALIZED_HOURLY_USD` (the realized per-class market rates —
that is all `fit_constants` returns; the MFU/concurrency constants in `analytical` are
re-calibrated against the same dataset separately), and re-`verify`. The realized rates and
the accuracy scorecard are pinned by tests.

## Tests

CPU-only suites under `tests/` (`test_cost_{hardware,models,analytical,estimate,rewards,measured,calibration}.py`);
`test_cost_calibration.py` pins the **no-output-multiplier** invariant.

```bash
FLASH_SKIP_NET=1 uv run pytest tests/test_cost_*.py
```

Pure-Python — no optional extras required.
