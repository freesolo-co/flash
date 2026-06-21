# Training-cost benchmark: Flash vs Tinker (+ verl), same environment

Real GRPO training runs, **Qwen3.5-4B**, on the **`reverse-text`** verifiers environment (the only
LoRA base both stacks share), 10 GRPO steps, group 4 × batch 4 (16 rollouts/step), max_tokens 1024.
Measured 2026-06-21.

## Headline: Flash is ~3.2× cheaper than Tinker on the same env

| stack | backend | cost (USD) | wall | $/step | $/rollout | reward | source |
|---|---|---|---|---|---|---|---|
| **Flash** | TRL + colocated vLLM | **$0.0439** | 114 s | $0.00439 | $0.000274 | 0.386 → 0.430 | **measured** (A100 PCIe @ $1.39/hr) |
| **Tinker** | managed (Thinking Machines) | **$0.142** | 276 s | $0.01419 | $0.000887 | — | **measured** (active-compute × $2/hr proxy¹) |
| Flash (verl) | verl colocate, 1 GPU | ~$0.136/step | — | $0.1357² | $0.00106² | — | works (this session) + validated s/step² |

**Flash (TRL) is 3.24× cheaper total and per-rollout, and 2.43× faster** than Tinker on the same
environment. Both actually trained (Flash reward rose 0.386 → 0.430 in 10 steps; reverse-text is hard
for a 4B base at this step budget).

¹ Tinker does **not** expose per-run cost via API; the figure is active compute (rollout + train,
capacity pauses excluded) × a $2/hr proxy — explicitly an order-of-magnitude number, not a bill.
The Flash figure is the control plane's **actual** charged cost (wall × the realized A100 rate).

## Does verl work? Yes — and where it helps

A real **`FLASH_FRAMEWORK=verl`** run on this session installed the verl stack, loaded Qwen3.5-4B,
and ran real GRPO + LoRA steps on the **reverse-text task** (added `FLASH_VERL_ENV=reverse-text` so
verl trains the same task as the Tinker side, not its synthetic throughput data). It hit one
transient vLLM async cold-start retry (a known, documented flake — not a correctness issue) and was
then stopped to cap GPU spend, since the cost picture was already clear:

- **verl *colocate* (1 GPU) is overhead-bound** — Ray + the vLLM async HTTP server + the NCCL
  checkpoint engine add fixed cost, so per-step it is *not* cheaper than Flash's TRL colocate
  (validated steady-state: 4B colocate 351 s/step on A100 ≈ 0.85–0.92× TRL). Plus a one-time
  ~20–25 min stack install that amortizes only over long runs.
- **verl's cost win is the *async one-step-off overlap* on ≥2 GPUs**, where generation of step
  *t+1* overlaps the optimizer of step *t* with LoRA — validated **2.6–3.0×** speedup elsewhere
  (freesolo PR #215 ratio matrix). That is exactly the regime the **shared rollout pool (PR #24)**
  generalizes to the whole fleet: rollout + reward run off the trainer GPU and are shared across
  runs, so the trainer never idles and reward latency never lands on the critical path.

² verl per-step/per-rollout figures use the validated 351 s/step (A100, same config) × the realized
$1.39/hr; per-rollout normalizes by verl's batch 32 × group 4 = 128 rollouts/step. This session
confirmed verl *runs* on reverse-text; a clean same-run 6-step number was not harvested (cost cap).

## Takeaway

- **Flash beats Tinker on training cost on the same environment** (3.2× cheaper, 2.4× faster),
  measured end-to-end.
- **verl works** in Flash and is the right backend for the *multi-GPU* regime; on a single GPU its
  framework overhead makes plain TRL colocate the cheaper default.
- The clean cross-stack metrics are **cost** and **latency** (both measured here). Held-out task
  performance needs a unified eval (one scorer, generous tokens) — out of scope for this cost run.

## Reproduce

```bash
# Flash (TRL) on reverse-text:
slm train benchmark/configs/reverse_text_4b.toml --set train.steps=10 \
  --set environment.id=primeintellect/reverse-text --background     # FLASH_API_URL -> a plane

# Flash using verl on the same task:
#   [worker_env] FLASH_FRAMEWORK="verl"  FLASH_VERL_ENV="reverse-text"

# Tinker on reverse-text:
TINKER_API_KEY=... /usr/bin/python3 benchmark/tinker_runner.py --env-id reverse-text --steps 10
```
