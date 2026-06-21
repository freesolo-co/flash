# Rollout pool — real multi-GPU end-to-end run

This closes the "not proven yet" gap: the pool was run against a **real multi-GPU vLLM fleet**, with
**real LoRA trainers pushing real adapters through the router** — not the fake-vLLM/CPU-stub tests.

## Setup (one 4× RTX 3090 box, Vast.ai, ~$0.55/hr)

- **GPU 0, 1** — two **real `vllm serve --enable-lora`** OpenAI servers (vLLM 0.23.0, torch 2.11,
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1`), serving `Qwen/Qwen2.5-3B/1.5B-Instruct`. *These are the pool.*
- **router + reward worker** — `flash-pool serve` (the nginx router) + a reverse-text reward worker,
  both registered with the two vLLM backends.
- **GPU 2, 3** — two **real GRPO LoRA trainers** (`GRPOPoolLoop` + `pool_policy`, real
  transformers+peft forward/backward), one adapter each (`runA`, `runB`), trained **concurrently**,
  pulling rollouts from the router and rewards from the worker, then syncing weights back.

Everything on one box (shared filesystem, so the trainer writes the adapter dir and the vLLM servers
load it by path). Public stack (vLLM image + open model) for reproducibility.

## What actually happened (from the run report)

```
runA: ok=True, 3 steps   runB: ok=True, 3 steps     (both real LoRA trainers finished)
per-step (runA): reward 0.041/0.027/0.037,  loss -0.47 -> 0.26 -> 0.32,  version 1 -> 2 -> 3,
                 gen_s 0.9/0.4/0.4  score_s 0.01      (gen on the pool GPUs; reward off-GPU)
distribution:  gpu0  total_requests=6  adapters=[runA, runB]
               gpu1  total_requests=6  adapters=[runA, runB]
adapters:      runA  placements=[gpu0, gpu1]  version=3
               runB  placements=[gpu0, gpu1]  version=3
```

Every claim the pool makes, observed on real hardware:

- **Real multi-GPU vLLM fleet** — two real `vllm --enable-lora` servers booted, health-checked, and
  registered as backends (`healthy: true`).
- **Real trainers, real adapters through the router** — two real LoRA trainers generated via the
  router (→ vLLM), scored on the off-GPU reward worker, ran real optimizer steps (non-trivial
  losses), and produced real adapters.
- **Distributed across GPUs** — generation load-balanced **6 / 6** across gpu0 / gpu1.
- **Multiple models on one GPU** — both `runA` and `runB` LoRAs co-resident on **both** GPUs.
- **Per-step weight-sync hot-swap on real vLLM** — each adapter reached `version=3` (synced + reloaded
  on the live vLLM servers every step), replicated on both backends.
- **Concurrent runs share the pool** — runA and runB trained at the same time over the shared fleet.

## Notes / honesty

- Model is `Qwen2.5-1.5B/3B-Instruct` (public, so public vLLM serves it + LoRA out of the box), not
  the internal `Qwen3.5-4B` — to keep the run on a fully public, reproducible stack. The pool /
  router / trainer code is model-agnostic; only the served weights differ.
- Getting here took several Vast attempts: the `ghcr.io/.../flash-worker:cu128` image (which *does*
  serve Qwen3.5) pulls unreliably on Vast, and `vllm/vllm-openai` needs `--entrypoint bash` (its
  default entrypoint is `vllm serve`) and the `--disable-log-requests` flag removed in vLLM 0.23.
  The successful run used the public vLLM image + entrypoint override + a base64-bundled pool package.
- Raw artifacts (report + vLLM/router/trainer logs) were captured to HF
  (`DavidBShan/bench-verl-q4b-rt-v1/pool_e2e/`). The instance was torn down after the run.
