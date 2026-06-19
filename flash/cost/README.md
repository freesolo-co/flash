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
| **seconds_per_step (SFT)** | `6 × active_params × tokens / (peak_bf16 × MFU_train)` |
| **seconds_per_step (GRPO)** | vLLM rollout (`MFU_decode`) + **concurrent** reward grading + policy/ref update |
| **$/hr** | the **realized (spot/queue) rate** the provider bills, not the on-demand list |

Every constant is a real, measurable quantity — calibrated against measured runs the way
you'd measure MFU empirically, **never** a dimensionless correction on the result:

- **MFU** (`MFU_train`, `MFU_decode`) — fraction of peak FLOPS the run sustains.
- **Reward concurrency** — graders run in parallel; the reward wall is
  `completions × per-completion-latency / concurrency`, *not* the serial product (which
  over-counts a heavy LLM-judge by ~100× and saturates the 24 h wall cap).
- **Realized $/hr** (`hardware.REALIZED_HOURLY_USD`) — the median effective rate
  (measured cost ÷ measured wall) per GPU class. RTX 5090 lists $0.99 but bills ~$0.86;
  A100 PCIe lists $1.39 but bills ~$1.04. An unobserved class falls back to list (an
  honest over-estimate until measured).

## CLI

```bash
python -m flash.cost estimate --model Qwen/Qwen3.5-9B --method grpo --steps 300
python -m flash.cost verify    # grade the equation vs measured cost + env sweep
```

## Accuracy vs measured cost (no hacking, then accuracy)

`verify_accuracy` grades the **raw** equation against the measured RunPod/Vast runs in
`cost_estimator_results/real_runs/measured_runs.json` (the calibration + validation set):

| Group | n | mean MAPE | median APE | agg bias | within 33% |
|---|---:|---:|---:|---:|---:|
| sft | 13 | 116% | 51% | 1.02 | 31% |
| grpo | 43 | 69% | 40% | 0.85 | 49% |
| all | 56 | 80% | 41% | 0.87 | 45% |

**`agg bias` is not forced to 1.0** — forcing it would be exactly the output hack this
equation avoids. The per-run residual is largely **irreducible**: measured wall = provider
queue + container boot + model download + train, and that scheduling/cold-start overhead
swings by minutes in ways no config predicts. So the equation targets an *unbiased central
estimate, typical run within ~30–50%* — not a fake-precise point.

Refresh the calibration as new runs land: harvest the control plane into `measured_runs.json`,
re-run `fit_constants` to refresh `REALIZED_HOURLY_USD` + the MFU/concurrency constants, and
re-`verify`. The realized rates and the accuracy scorecard are pinned by tests.

## Tests

CPU-only suites under `tests/` (`test_cost_{hardware,models,analytical,estimate,rewards,measured,calibration}.py`);
`test_cost_calibration.py` pins the **no-output-multiplier** invariant.

```bash
FLASH_SKIP_NET=1 uv run pytest tests/test_cost_*.py
```

Pure-Python — no optional extras required.
