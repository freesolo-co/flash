# Cost of training — Flash vs Tinker vs **Prime Lab**

Extends the [Flash-vs-Tinker GRPO benchmark](../primelab/README.md) with a third
stack: **Prime Lab** (Prime Intellect Hosted Training, `prime train`). Same
`Qwen/Qwen3.5-4B`, same {gsm8k, reverse-text, hendrycks-math} Prime Intellect
`verifiers` envs, same GRPO knobs (30 steps, batch 4 × group 4, max_tokens 1024).

## The three billing models

| stack | architecture | meter | what you pay for |
|---|---|---|---|
| **Flash** | dedicated rented GPU, **colocated** vLLM rollouts + trainer | GPU-**hours** | wall-clock × $/hr (incl. setup/idle) |
| **Tinker** | managed backend | **tokens** (prefill / sample / train) | tokens processed |
| **Prime Lab** | **disaggregated** async RL — shared Trainer + Inference, dedicated Orchestrator | **tokens** (input / output / train) | tokens that actually move the model |

Tinker and Prime Lab share a billing model (per token) → compare them directly.
Flash is the odd one out (GPU-hours), so **time vs tokens** is the axis between
Flash and the two per-token stacks: Flash's cost scales with how long a GPU is
held; the per-token cost scales with how many tokens the run touches. They cross.

> The original Flash-vs-Tinker run used a Tinker **GPU-time proxy** (~$2/hr ×
> active compute) because we couldn't read Tinker's bill via API. That proxy
> badly overstates Tinker's real **per-token** cost on small runs; this doc uses
> Tinker's published per-token card instead, so Tinker and Prime Lab are
> apples-to-apples.

## Per-token rate cards (Qwen3.5-4B, $ / 1M tokens)

**Both Tinker and Prime Lab bill per token** — so the fair comparison is
per-token vs per-token, not against a GPU-time proxy.

| meter | Tinker | Prime Lab | Tinker / Prime Lab |
|---|---|---|---|
| input (prefill) | $0.22 | $0.10 | 2.2× |
| output (sample) | $0.67 | $0.30 | 2.23× |
| training | $0.67 | $0.30 | 2.23× |

Prime Lab rates: confirmed live from `prime train models` (June 2026; identical
to the banner `prime train` prints at launch). Tinker rates: published card at
<https://thinkingmachines.ai/tinker>.

## Cost of training (headline — apples-to-apples per token)

Tinker and Prime Lab are both **per-token, same token volume**, so their $ ride
on the rate cards above (Tinker is a flat ~2.2× Prime Lab across all three
meters). Flash is a **measured GPU-rental** bill (different billing model).
Prime Lab's real bill is API-exposed via `prime train usage <run_id>`; our live
run was billing-gated (wallet $0), so its column (and Tinker's, also not run
here) is a published-rate estimate on the token model below.

| task | Flash $ (measured) | **Tinker $ (per-token)** | **Prime Lab $ (per-token)** | Tinker / Prime Lab | Tinker / Flash |
|---|---|---|---|---|---|
| gsm8k | **$0.164** | **~$0.52** | **~$0.23** | **2.2×** | 3.2× |
| reverse-text | **$0.065** | **~$0.072** | **~$0.032** | **2.2×** | 1.1× |
| hendrycks-math | **$0.214** | **~$0.53** | **~$0.24** | **2.2×** | 2.5× |

### What this says

1. **Prime Lab is ~2.2× cheaper than Tinker — flat, on every task.** Because both
   bill per token over the identical token volume, the ratio collapses to the
   rate-card ratio (~2.2×) and the token-length assumptions cancel out of it
   entirely. (They still set the absolute $, not the ratio.)
2. **Mind the proxy.** The earlier Flash-vs-Tinker run priced Tinker with a
   GPU-**time** proxy (~$0.84 = ~25 min × $2/hr). That **overstated Tinker's real
   per-token cost** badly on cheap runs — e.g. reverse-text's true per-token cost
   is ~$0.07, not $0.84. Don't compare a per-token stack against a per-hour proxy.
3. **Flash (measured GPU) is cheapest on the math tasks** (~$0.16–0.21 vs Prime
   Lab ~$0.23–0.24 vs Tinker ~$0.52–0.53). On short reverse-text Flash ≈ Tinker
   per-token, and Prime Lab is the cheapest (~$0.03).
4. **Time vs tokens — the crossover (Flash vs the per-token stacks).** Per-token
   wins short / low-token runs (no setup, no idle, no GPU held). Flat GPU-rental
   (Flash) pulls ahead as token volume climbs — long context, large groups, many
   steps — because $/hr is fixed while the per-token meter keeps running. This
   tiny 480-rollout matrix is where per-token is most favorable.

Sensitivity: math tasks span ~$0.23→0.28 (Prime Lab) / ~$0.52→0.63 (Tinker)
between the completion-only and full-sequence training meters (whether Prime
Lab's training line counts only the completion tokens or the full prompt+
completion sequence); the **2.2× Tinker/Prime Lab ratio holds regardless**.
Reproduce: `cost.py` (completion-only) vs `cost.py --train-full` (full-sequence).

## Token model (how the Prime Lab estimate is built)

Per run, GRPO with the matrix knobs:

```
rollouts R   = max_steps · batch_size · group_size  = 30 · 4 · 4 = 480
input_tok    = prompt_len     · R     @ $0.10/Mtok   (inference input)
output_tok   = completion_len · R     @ $0.30/Mtok   (inference output)
train_tok    = completion_len · R     @ $0.30/Mtok   (training; loss is on
                                                       completion tokens)
```

Per-task length assumptions:

| task | prompt_len | completion_len | rationale |
|---|---|---|---|
| gsm8k | 256 | 768 | CoT near the 1024 cap; 0.75·cap matches Flash's own trainer-throughput basis (368,640 trained tok = 480·768) |
| hendrycks-math | 320 | 768 | same verbose boxed-CoT regime |
| reverse-text | 96 | 96 | output ≈ input string length; Flash ran it ~2.5× faster → far fewer tokens |

`completion_len` is the dominant lever (it drives both the output and training
meters). The committed configs + `prime train usage` settle it exactly on a
funded account; until then the table is an estimate with the range above.

## Prime Intellect GPU marketplace (self-hosted prime-rl) — why hosted wins here

Prime Intellect also rents raw GPUs; you could run open-source `prime-rl`
yourself. But `prime-rl` is **disaggregated** — separate inference + trainer →
**≥2 GPUs** for a 4B run. At ~$1.2–2/hr per A100-class card × 2 × the ~0.1–0.16 h
wall ⇒ **≳ $0.3–0.6** plus setup — i.e. self-hosting on the marketplace is **not**
cheaper than the hosted per-token product for these small runs. The hosted
per-token path is the cost-optimal Prime route at this scale.

## Caveats

- **Prime Lab $ is an estimate, not a bill.** Rates are confirmed live; the run
  was billing-gated. `prime train usage <run_id>` gives the exact charge on a
  funded account (see `../primelab/README.md`).
- **Reward parity (performance) not re-measured here** — this is a *cost*
  comparison. In-training reward is not cross-stack comparable (verifiers
  version/scoring differences, per the Flash-vs-Tinker writeup); use a unified
  held-out eval for performance, not the reward curve.
- **Tinker remains a GPU-time proxy** (its per-run $ is not API-exposed); Flash
  remains a measured RunPod charge. Only Prime Lab's meter is per-token, so the
  three columns are "best available basis," labelled per cell.
- **Scale is deliberately tiny** to keep spend low — which structurally favors
  the per-token meter. The crossover above is the load-bearing caveat for
  generalizing these numbers.
