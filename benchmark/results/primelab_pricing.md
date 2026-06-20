# Cost of training — Flash vs Tinker vs **Prime Lab**

Extends the [Flash-vs-Tinker GRPO benchmark](../primelab/README.md) with a third
stack: **Prime Lab** (Prime Intellect Hosted Training, `prime train`). Same
`Qwen/Qwen3.5-4B`, same {gsm8k, reverse-text, hendrycks-math} Prime Intellect
`verifiers` envs, same GRPO knobs (30 steps, batch 4 × group 4, max_tokens 1024).

## The three billing models

| stack | architecture | meter | what you pay for |
|---|---|---|---|
| **Flash** | dedicated rented GPU, **colocated** vLLM rollouts + trainer | GPU-**hours** | wall-clock × $/hr (incl. setup/idle) |
| **Tinker** | managed backend | not exposed | (proxy: active-compute × $2/hr) |
| **Prime Lab** | **disaggregated** async RL — shared Trainer + Inference, dedicated Orchestrator | **tokens** | tokens that actually move the model |

The interesting axis is **time vs tokens**: Flash's cost scales with how long a
GPU is held; Prime Lab's scales with how many tokens the run touches. They cross.

## Prime Lab rates (confirmed live)

From `prime train models` for `Qwen/Qwen3.5-4B` (June 2026) — identical to the
banner `prime train` prints at launch:

| meter | $ / 1M tokens |
|---|---|
| inference input | $0.10 |
| inference output | $0.30 |
| training | $0.30 |

## Cost of training (headline)

Flash = **measured** (RunPod billed). Tinker = active-compute **proxy**
(per-step wall × $2/hr; its real per-run $ is not API-exposed). Prime Lab =
**published-rate estimate** — the rates above × the GRPO token volume (central
token model below). Prime Lab's real bill **is** API-exposed via
`prime train usage <run_id>`; our live run was billing-gated (wallet $0), so the
column is an estimate, not a charge.

| task | Flash $ (measured) | Tinker $ (proxy) | **Prime Lab $ (est.)** | Tinker / Flash | Tinker / Prime Lab | Prime Lab / Flash |
|---|---|---|---|---|---|---|
| gsm8k | **$0.164** | ~$0.84 | **~$0.23** | 5.1× | **3.6×** | 1.4× |
| reverse-text | **$0.065** | ~$0.84 | **~$0.03** | 12.9× | **26×** | **0.5×** |
| hendrycks-math | **$0.214** | ~$0.83 | **~$0.24** | 3.9× | **3.5×** | 1.1× |

Range (sensitivity): the math tasks span **~$0.23 (central) → ~$0.34 (training
metered on the full prompt+completion sequence at the cap)**; reverse-text stays
**~$0.03–0.05** because its completions are short. See `cost.py --train-full`.

### What this says

1. **Both "pay-for-compute" stacks crush the GPU-time proxy.** Prime Lab's
   per-token bill is **3.5–26× under Tinker's** estimate and on the same order as
   Flash. Tinker's number is the outlier because the proxy charges ~25 min of
   managed GPU time per run regardless of how few tokens a 30-step run touches.
2. **Prime Lab ≈ Flash for tiny runs, and *cheaper* when generations are short.**
   reverse-text (short outputs → few tokens) costs Prime Lab **~half** of Flash,
   which still pays for ~3 min of held A100 plus setup. On the verbose math tasks
   Prime Lab is ~1.1–1.4× Flash.
3. **Time vs tokens — the crossover.** Per-token (Prime Lab) wins when a run is
   short / low-token (no setup, no idle, no GPU held during slow eval). Flat
   GPU-rental (Flash) wins as token volume climbs — long context, large groups,
   many steps — because $/hr is fixed while the per-token meter keeps running.
   At this deliberately small scale (480 rollouts, 30 steps) per-token is at its
   most favorable; scale up and Flash's flat rate pulls ahead.

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
