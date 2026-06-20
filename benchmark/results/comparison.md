# Flash vs Tinker — GRPO benchmark (Qwen3.5-4B, 30 steps)

Same base model, same verifiers environment, same GRPO hyper-parameters (group_size=4, batch_size=4, max_tokens=512, 30 steps) on each side.

- **Flash** trains on a rented RunPod **A100 PCIe** (4B GRPO needs ≥35 GB; the allocator escalates from the requested RTX 5090). Cost is **measured** (RunPod billed).
- **Tinker** trains on Thinking Machines' **managed** backend. Per-run cost is **not exposed via API**, so its $ column is a wall-clock proxy (labelled).
- **Performance** = mean group reward over the 30-step GRPO run (step 1 → final). Flash also reports a native held-out eval (50 examples).

## gsm8k

| metric | Flash | Tinker | Δ (Flash − Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | A100 PCIe | managed (Tinker) | |
| **reward** step 1 | 1 | 0.250 | |
| **reward** final (last 5 avg) | 0.900 | 0.125 | +0.775 |
| reward best step | 1 | 0.312 | +0.688 |
| reward final (raw step 30) | 1 | 0.000 | |
| held-out eval | 0.880 (n=50) | — | |
| **latency** wall total | 45m52s | 39m40s | |
| latency setup/queue | 0m11s | none (managed) | |
| latency active compute | 45m41s | 29m32s | |
| latency capacity-pause | — | 10m08s | |
| latency per-step | 182.8s | 59.1s | |
| **cost** | 1.0585 USD | 0.9848 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## reverse-text

| metric | Flash | Tinker | Δ (Flash − Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | A100 PCIe | managed (Tinker) | |
| **reward** step 1 | 0.437 | 0.000 | |
| **reward** final (last 5 avg) | 0.362 | 0.000 | +0.362 |
| reward best step | 0.512 | 0.000 | +0.512 |
| reward final (raw step 30) | 0.408 | 0.000 | |
| held-out eval | 0.259 (n=50) | — | |
| **latency** wall total | 8m37s | 39m14s | |
| latency setup/queue | 0m10s | none (managed) | |
| latency active compute | 8m26s | 28m47s | |
| latency capacity-pause | — | 10m27s | |
| latency per-step | 33.8s | 57.6s | |
| **cost** | 0.1956 USD | 0.9596 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## hendrycks-math

| metric | Flash | Tinker | Δ (Flash − Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | RTX Pro 6000 WK | managed (Tinker) | |
| **reward** step 1 | 0.062 | 0.000 | |
| **reward** final (last 5 avg) | 0.237 | 0.087 | +0.150 |
| reward best step | 0.500 | 0.312 | +0.188 |
| reward final (raw step 30) | 0.312 | 0.062 | |
| held-out eval | 0.320 (n=50) | — | |
| **latency** wall total | 38m07s | 39m14s | |
| latency setup/queue | 0m10s | none (managed) | |
| latency active compute | 37m57s | 27m36s | |
| latency capacity-pause | — | 11m38s | |
| latency per-step | 151.8s | 55.2s | |
| **cost** | 1.1957 USD | 0.9203 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## Held-out eval — gsm8k (clean cross-version scorer)

In-training GRPO reward is NOT comparable across stacks (different verifiers versions + task presentation; at matched 512 max_tokens the model's verbose answer is cut off before `\boxed{}`). This eval removes those confounds: greedy decode, max_tokens=1024, one exact-match scorer on 50 held-out examples.

| model | gsm8k accuracy | answer-truncated frac |
|---|---|---|
| base Qwen3.5-4B | 0.360 | 0.90 |
| Tinker-trained (30 GRPO steps) | 0.440 | 0.68 |
| **Δ (trained − base)** | **+0.080** | |

Tinker GRPO improved held-out accuracy AND cut answer truncation — the model learned to reach the answer sooner. (Flash-trained not re-served under the same scorer; Flash's native on-GPU eval is in the per-task tables above.)

## Summary

| task | winner (reward) | flash cost | tinker cost (est) | flash wall | tinker wall |
|---|---|---|---|---|---|
| gsm8k | Flash | 1.0585 USD | 0.9848 USD | 45m52s | 39m40s |
| reverse-text | Flash | 0.1956 USD | 0.9596 USD | 8m37s | 39m14s |
| hendrycks-math | Flash | 1.1957 USD | 0.9203 USD | 38m07s | 39m14s |

## Reliability & operability (observed this run)

- **Flash** rents a GPU per run. 4B GRPO needs ≥35 GB → the allocator escalates RTX 5090 → A100 PCIe. On the long-generation math tasks the colocated-vLLM rollout **hung ~13-15 min at eval boundaries** then self-recovered; a true >25-min freeze trips the **stall watchdog**, which **kills the sick host, escalates the GPU class** (A100 → RTX Pro 6000), and **resumes from the last checkpoint**. reverse-text (short generations) ran clean. Net: dedicated + auto-healing, but rented-GPU flakiness adds real tail latency.
- **Tinker** is managed: **no setup/queue**, but the backend **paused all jobs ~10 min** mid-run ("running short on capacity, please wait") — out of the user's control, and slower per active step.
- A **shared Flash control plane** dropped a run's watcher on restart → the record stuck at `running` forever (orphaned); re-run on a dedicated plane fixed it.
- **HF caps repo creation at 300/day/user**; once hit, every new Flash run 429s at submit. Worked around by reusing pre-existing artifact repos (`run_flash_plane.py`).

## Methodology caveats

- **In-training reward is not cross-stack comparable.** Flash's worker installs verifiers ~0.1.14 (continuous/partial-credit reward); Tinker pins 0.1.9 (the recipe's requirement). At matched max_tokens the two stacks present the task differently and truncate differently. Use the **held-out eval** (one scorer, generous tokens) for performance, and **cost/latency** (measured) for the clean cross-stack comparison.
- **Tinker cost is an estimate.** Tinker does not expose per-run cost via API; the $ column is active-compute-time × a GPU-rate proxy (pause excluded), not a bill. Flash cost is the measured RunPod charge.
- **Scale is deliberately small** (30 steps, 16 rollouts/step) to keep spend low, so per-step reward is noisy and 30-step gains are modest by design.
