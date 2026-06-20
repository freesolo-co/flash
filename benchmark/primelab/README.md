# Prime Lab (Prime Intellect Hosted Training) — third cost column

Adds **Prime Lab** to the Flash-vs-Tinker cost-of-training benchmark. Same model
(`Qwen/Qwen3.5-4B`), same tasks, same GRPO hyper-parameters, same Prime Intellect
`verifiers` environments — now priced on a third stack.

| stack | what it is | how it bills | our $ basis |
|---|---|---|---|
| **Flash** | managed LoRA on rented GPUs | RunPod GPU-time | **measured** (billed) |
| **Tinker** | managed fine-tuning API | not API-exposed | proxy (active-compute × $2/hr) |
| **Prime Lab** | Prime Intellect Hosted Training | **per token** (in/out/train) | **published-rate estimate** (rates confirmed live; run is billing-gated) |

Prime Lab is the disaggregated-async-RL hosted product (`prime train`): a shared
Trainer + Inference + a dedicated Orchestrator, billed **per token** on three
meters. Confirmed live from `prime train models` for `Qwen/Qwen3.5-4B` (Jun 2026):

```
inference input   $0.10 / 1M tokens
inference output  $0.30 / 1M tokens
training          $0.30 / 1M tokens
```

## Reproduce (the real bill)

Per-token billing **is** API-exposed (unlike Tinker), so a funded run yields the
exact cost — no proxy needed:

```bash
export PATH="$HOME/.local/bin:$PATH"
prime config remove-team-id            # team-scoped token 403s on Lab/billing; use personal (then restore)

prime train models | grep Qwen3.5-4B   # confirm Available + rates
prime train benchmark/primelab/gsm8k.toml -y
#   ...repeat for reverse-text.toml, hendrycks-math.toml

prime train list                       # get the run ids
prime train usage <run_id>             # EXACT token counts + $ (the real bill)
prime train metrics <run_id>           # reward curve, parity check vs Flash/Tinker
```

> Status: each config was accepted by `prime train` end-to-end — config valid,
> `primeintellect/<env>` action check passed, pricing banner printed — and then
> stopped at **"Payment required"** (personal wallet $0.00; the team token 403s
> on the wallet/Lab endpoints). The live run is one funded command away.

## Estimate (until the run is funded)

`cost.py` applies the confirmed rates to the GRPO token volume:

```bash
python3 benchmark/primelab/cost.py            # central estimate + at-cap ceiling
python3 benchmark/primelab/cost.py --train-full --json
```

Token model and per-task length assumptions are documented at the top of
`cost.py` and in `../results/primelab_pricing.md`.
