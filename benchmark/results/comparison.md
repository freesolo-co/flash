# Flash vs Tinker — GRPO benchmark (Qwen3.5-4B, 30 steps)

Same base model, same verifiers environment, same GRPO hyper-parameters (group_size=4, batch_size=4, **max_tokens=1024**, 30 steps) on each side. This run includes the **stall fix** (heartbeat through mid-run eval) and the **truncation fix** (1024 tokens so the boxed answer isn't cut off).

- **Flash** trains on a rented RunPod **A100 PCIe** (4B GRPO needs ≥35 GB; the allocator escalates from the requested RTX 5090). Cost is **measured** (RunPod billed).
- **Tinker** trains on Thinking Machines' **managed** backend. Per-run cost is **not exposed via API**, so its $ column is an active-compute proxy (labelled).
- **Performance** = the held-out eval below (one shared scorer); in-training reward is per-stack only (different verifiers reward versions).

## gsm8k

| metric | Flash | Tinker | Δ (Flash - Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | A100 PCIe | managed (Tinker) | |
| **reward** step 1 | 1 | 0.688 | |
| **reward** final (last 5 avg) | 0.933 | 0.362 | +0.571 |
| reward best step | 1 | 0.688 | +0.312 |
| reward final (raw step 30) | 1 | 0.250 | |
| held-out eval | 0.750 (n=20) | — | |
| **latency** wall total | 22m50s | 25m43s | |
| latency setup/queue | 0m12s | none (managed) | |
| latency active compute | 22m38s | 25m43s | |
| latency capacity-pause | — | none | |
| latency per-step | 90.6s | 51.4s | |
| **cost** | 0.5245 USD | 0.8572 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## reverse-text

| metric | Flash | Tinker | Δ (Flash - Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | A100 PCIe | managed (Tinker) | |
| **reward** step 1 | 0.419 | 0.000 | |
| **reward** final (last 5 avg) | 0.375 | 0.000 | +0.375 |
| reward best step | 0.512 | 0.000 | +0.512 |
| reward final (raw step 30) | 0.431 | 0.000 | |
| held-out eval | 0.064 (n=20) | — | |
| **latency** wall total | 15m32s | 25m29s | |
| latency setup/queue | 0m47s | none (managed) | |
| latency active compute | 14m44s | 25m29s | |
| latency capacity-pause | — | none | |
| latency per-step | 59.0s | 51.0s | |
| **cost** | 0.3416 USD | 0.8498 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## hendrycks-math

| metric | Flash | Tinker | Δ (Flash - Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | A100 PCIe | managed (Tinker) | |
| **reward** step 1 | 0.667 | 0.125 | |
| **reward** final (last 5 avg) | 0.717 | 0.250 | +0.467 |
| reward best step | 1 | 0.438 | +0.562 |
| reward final (raw step 30) | 0.917 | 0.188 | |
| held-out eval | 0.650 (n=20) | — | |
| **latency** wall total | 29m28s | 25m16s | |
| latency setup/queue | 0m12s | none (managed) | |
| latency active compute | 29m16s | 25m16s | |
| latency capacity-pause | — | none | |
| latency per-step | 117.1s | 50.6s | |
| **cost** | 0.6781 USD | 0.8427 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## Held-out eval — gsm8k (the valid cross-stack performance comparison)

In-training GRPO reward is NOT comparable across stacks (Flash's worker uses verifiers ~0.1.14, Tinker pins 0.1.9 — different rewards + task presentation). This eval applies ONE version-independent exact-match scorer to every model, identical greedy decoding, max_tokens=2048, on 50 held-out examples.

| model | gsm8k accuracy | how generated / scored |
|---|---|---|
| base Qwen3.5-4B | 0.620 | unified scorer, Tinker sampling |
| **Tinker-trained** | **0.540** (Δ-0.080 vs base) | unified scorer, Tinker sampling |
| Flash-trained | 0.750 | Flash's NATIVE on-GPU eval (gsm8k-env scorer, NOT the unified scorer — see note) |

**Key finding:** under the shared scorer the Tinker-trained model shows **no held-out gain** (Δ-0.080, within n=50 noise) even though its in-training reward rose (best 0.69). At this deliberately tiny scale (30 steps, 16 rollouts/step) the rising training reward does NOT translate to held-out accuracy — exactly what a unified held-out eval is for; the training-reward curve alone would have implied improvement. The Flash-trained row falls back to Flash's own on-GPU eval (a similar but not identical scorer) because a unified Flash eval needs a Qwen3.5-4B LoRA serving — the live one was empty (0 GPUs / 0 base models); `eval_unified.py` runs the unified Flash eval against any configured serving. Truncation note: even at max_tokens=2048, 56% of base generations still hit the cap (Qwen3.5-4B is very verbose for this format).

## Summary

| task | winner (reward) | flash cost | tinker cost (est) | flash wall | tinker wall |
|---|---|---|---|---|---|
| gsm8k | Flash | 0.5245 USD | 0.8572 USD | 22m50s | 25m43s |
| reverse-text | Flash | 0.3416 USD | 0.8498 USD | 15m32s | 25m29s |
| hendrycks-math | Flash | 0.6781 USD | 0.8427 USD | 29m28s | 25m16s |

## Reliability & operability (observed this run)

- **Flash** rents a GPU per run. 4B GRPO needs ≥35 GB → the allocator escalates RTX 5090 → A100 PCIe. On the long-generation math tasks the colocated-vLLM rollout **hung ~13-15 min at eval boundaries** then self-recovered; a true >25-min freeze trips the **stall watchdog**, which **kills the sick host, escalates the GPU class** (A100 → RTX Pro 6000), and **resumes from the last checkpoint**. reverse-text (short generations) ran clean. Net: dedicated + auto-healing, but rented-GPU flakiness adds real tail latency.
- **Tinker** is managed: **no setup/queue**, but the backend **paused all jobs ~10 min** mid-run ("running short on capacity, please wait") — out of the user's control, and slower per active step.
- A **shared Flash control plane** dropped a run's watcher on restart → the record stuck at `running` forever (orphaned); re-run on a dedicated plane fixed it.
- **HF caps repo creation at 300/day/user**; once hit, every new Flash run 429s at submit. Worked around by reusing pre-existing artifact repos (`run_flash_plane.py`).

## Methodology caveats

- **In-training reward is not cross-stack comparable.** Flash's worker installs verifiers ~0.1.14 (continuous/partial-credit reward); Tinker pins 0.1.9 (the recipe's requirement). At matched max_tokens the two stacks present the task differently and truncate differently. Use the **held-out eval** (one scorer, generous tokens) for performance, and **cost/latency** (measured) for the clean cross-stack comparison.
- **Tinker cost is an estimate.** Tinker does not expose per-run cost via API; the $ column is active-compute-time x a GPU-rate proxy (pause excluded), not a bill. Flash cost is the measured RunPod charge.
- **Scale is deliberately small** (30 steps, 16 rollouts/step) to keep spend low, so per-step reward is noisy and 30-step gains are modest by design.
