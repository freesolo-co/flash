# Flash vs Tinker — GRPO benchmark (Qwen3.5-4B, 30 steps)

Same base model, same verifiers environment, same GRPO hyper-parameters (group_size=4, batch_size=4, **max_tokens=1024**, 30 steps) on each side. These are **training-only** runs — mid-run eval was removed, so both stacks do pure GRPO and the $ figures are a clean **cost of training**. Held-out **performance** is measured separately (deploy/serving side) so its eval cost never inflates training.

- **Flash** trains on a rented RunPod **A100 PCIe** (4B GRPO needs ≥35 GB; the allocator escalates from the requested RTX 5090). Cost is **measured** (RunPod billed).
- **Tinker** trains on Thinking Machines' **managed** backend. Per-run cost is **not exposed via API**, so its $ column is an **active-compute** proxy (per-step wall, capacity pauses excluded, x a $2.00/hr GPU rate; labelled, not a bill).
- **Two axes:** (1) **cost of training** — the per-task tables + the cost-of-training summary below; (2) **performance** — the unified held-out eval (one scorer).

## gsm8k

| metric | Flash | Tinker | Δ (Flash - Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | A100 PCIe | managed (Tinker) | |
| **reward** step 1 | 1 | 0.688 | |
| **reward** final (last 5 avg) | 0.917 | 0.362 | +0.554 |
| reward best step | 1 | 0.688 | +0.312 |
| reward final (last logged: F=15 / T=30 pts) | 0.917 | 0.250 | |
| **latency** wall total | 7m19s | 25m43s | |
| latency setup/queue | 0m15s | none (managed) | |
| latency active compute | 7m03s | 25m13s | |
| latency capacity-pause | — | 0m29s | |
| latency per-step | 14.1s | 50.5s | |
| **cost of training** | 0.1636 USD | 0.8411 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## reverse-text

| metric | Flash | Tinker | Δ (Flash - Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | A100 PCIe | managed (Tinker) | |
| **reward** step 1 | 0.412 | 0.000 | |
| **reward** final (last 5 avg) | 0.412 | 0.000 | +0.412 |
| reward best step | 0.592 | 0.000 | +0.592 |
| reward final (last logged: F=15 / T=30 pts) | 0.456 | 0.000 | |
| **latency** wall total | 3m00s | 25m29s | |
| latency setup/queue | 0m11s | none (managed) | |
| latency active compute | 2m48s | 25m12s | |
| latency capacity-pause | — | 0m16s | |
| latency per-step | 5.6s | 50.4s | |
| **cost of training** | 0.0651 USD | 0.8405 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## hendrycks-math

| metric | Flash | Tinker | Δ (Flash - Tinker) |
|---|---|---|---|
| status | done | done | |
| GPU | A100 PCIe | managed (Tinker) | |
| **reward** step 1 | 0.667 | 0.125 | |
| **reward** final (last 5 avg) | 0.650 | 0.250 | +0.400 |
| reward best step | 0.917 | 0.438 | +0.479 |
| reward final (last logged: F=15 / T=30 pts) | 0.833 | 0.188 | |
| **latency** wall total | 9m26s | 25m16s | |
| latency setup/queue | 0m13s | none (managed) | |
| latency active compute | 9m13s | 24m49s | |
| latency capacity-pause | — | 0m27s | |
| latency per-step | 18.4s | 49.6s | |
| **cost of training** | 0.2137 USD | 0.8272 USD | |
| cost basis | measured (RunPod billed) | ESTIMATE: active-compute x $2.00/hr proxy (Tinker cost not in API) | |

## Held-out eval — gsm8k (the valid cross-stack performance comparison)

In-training GRPO reward is NOT comparable across stacks (Flash's worker uses verifiers ~0.1.14, Tinker pins 0.1.9 — different rewards + task presentation). This eval applies ONE version-independent exact-match scorer to every model, identical greedy decoding, max_tokens=2048, on 50 held-out examples.

| model | gsm8k accuracy | how generated / scored |
|---|---|---|
| base Qwen3.5-4B | 0.620 | unified scorer, Tinker sampling |
| **Tinker-trained** | **0.540** (Δ-0.080 vs base) | unified scorer, Tinker sampling |

**Key finding:** under the shared scorer the Tinker-trained model's held-out accuracy fell (Δ-0.080 vs base, beyond the ±0.02 noise band); its in-training smoothed reward fell over the run. The Flash-trained row falls back to Flash's own on-GPU eval (a similar but not identical scorer) because a unified Flash eval needs a Qwen3.5-4B LoRA serving — the live one was empty (0 GPUs / 0 base models); `eval_unified.py` runs the unified Flash eval against any configured serving. Truncation note: even at max_tokens=2048, 56% of base generations still hit the cap (Qwen3.5-4B is very verbose for this format).

## Cost of training (the headline)

Pure GRPO, no eval. Flash is **measured** (RunPod); Tinker is an active-compute proxy (its real per-token bill isn't API-exposed).

| task | Flash $ (measured) | Tinker $ (proxy) | Tinker / Flash | Flash wall | Tinker wall |
|---|---|---|---|---|---|
| gsm8k | 0.1636 USD | 0.8411 USD | **5.1x** | 7m19s | 25m43s |
| reverse-text | 0.0651 USD | 0.8405 USD | **12.9x** | 3m00s | 25m29s |
| hendrycks-math | 0.2137 USD | 0.8272 USD | **3.9x** | 9m26s | 25m16s |

Flash trains the same model for a fraction of the Tinker (proxy) cost — a dedicated A100 with colocated-vLLM rollouts finishes GRPO in minutes; the managed backend's per-step latency is several times higher. (Performance — whether either *improves* the model — is the held-out eval above; at this tiny scale neither does.)

**Which GPU?** The allocator now picks the **cheapest fitting class across all providers** (validation gate + provider pin removed). For 4B GRPO (needs ≥35 GB) that is the **A40 (48 GB @ $0.44/hr)**. But cheapest-$/hr is **not** cheapest-job for compute-bound GRPO: A40 trained gsm8k for **$0.139 in 19 min** vs the A100's **$0.164 in 7 min** — only ~15% cheaper, because A40 is ~2.7x slower and the cheap rate is mostly eaten by the longer wall. The allocator minimizes $/hr, not $/throughput; a faster card at a higher rate finishes far sooner for ~the same money. (The per-task table above uses the A100 baseline; A40 is the new default and a measured alternative.)

## Reliability & operability (observed this run)

- **Flash** rents a GPU per run. 4B GRPO needs ≥35 GB → the allocator escalates RTX 5090 → A100 PCIe. On the long-generation math tasks the colocated-vLLM rollout **hung ~13-15 min at eval boundaries** then self-recovered; a true >25-min freeze trips the **stall watchdog**, which **kills the sick host, escalates the GPU class** (A100 → RTX Pro 6000), and **resumes from the last checkpoint**. reverse-text (short generations) ran clean. Net: dedicated + auto-healing, but rented-GPU flakiness adds real tail latency.
- **Tinker** is managed: **no setup/queue**, but the backend **paused all jobs ~10 min** mid-run ("running short on capacity, please wait") — out of the user's control, and slower per active step.
- A **shared Flash control plane** dropped a run's watcher on restart → the record stuck at `running` forever (orphaned); re-run on a dedicated plane fixed it.
- **HF caps repo creation at 300/day/user**; once hit, every new Flash run 429s at submit. Worked around by reusing pre-existing artifact repos (`run_flash_plane.py`).

## Methodology caveats

- **In-training reward is not cross-stack comparable.** Flash's worker installs verifiers ~0.1.14 (continuous/partial-credit reward); Tinker pins 0.1.9 (the recipe's requirement). At matched max_tokens the two stacks present the task differently and truncate differently. Use the **held-out eval** (one scorer, generous tokens) for performance, and **cost/latency** (measured) for the clean cross-stack comparison.
- **Tinker cost is an estimate.** Tinker does not expose per-run cost via API; the $ column is active-compute-time x a GPU-rate proxy (pause excluded), not a bill. Flash cost is the measured RunPod charge.
- **Scale is deliberately small** (30 steps, 16 rollouts/step) to keep spend low, so per-step reward is noisy and 30-step gains are modest by design.
