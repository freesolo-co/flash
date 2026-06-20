# Flash vs Tinker — head-to-head GRPO benchmark

This directory trains the **same model** on the **same task** with the **same GRPO
hyper-parameters** on two post-training stacks, and compares them on **cost**,
**performance**, and **latency**:

- **[Flash](https://github.com/freesolo-co/flash)** — this repo. Managed LoRA
  post-training on rented GPUs (RunPod / Vast), one dedicated GPU per run.
- **[Tinker](https://thinkingmachines.ai)** — Thinking Machines' managed fine-tuning
  API (`tinker` + `tinker_cookbook`), training server-side.

## What is held constant

| knob | value |
|---|---|
| base model | `Qwen/Qwen3.5-4B` (the only LoRA-trainable model both stacks share) |
| algorithm | GRPO |
| environment | a published **Prime Intellect `verifiers` environment** — consumed *identically* by both stacks |
| tasks | `gsm8k`, `reverse-text`, `hendrycks-math` (3 tasks × 2 stacks = **6 runs**) |
| steps | 30 |
| group size / batch | 4 / 4 (16 rollouts per step) |
| max tokens | 1024 |
| LoRA rank | 32 |

The shared **verifiers environment** is the linchpin: Flash is verifiers-only by design
(`[environment] id = "owner/name"`), and `tinker_cookbook` ships a `verifiers_rl` recipe,
so both train against the byte-identical task definition + reward function.

## The three comparison axes

- **Performance** — mean group reward across the GRPO run (step 1 → final). **In-training
  reward is NOT directly comparable across stacks**: the two sides install different
  `verifiers` versions (Flash ~0.1.14 continuous/partial-credit vs Tinker 0.1.9, the
  recipe's pin) and present/truncate the task differently at matched `max_tokens`, so the
  reward scales and definitions differ. For an apples-to-apples performance number, use the
  **held-out eval** (one version-independent scorer, generous tokens — see
  `results/comparison.md`); the in-training curves show learning *direction* per stack, not
  a cross-stack ranking. Mid-run (in-worker) eval was removed: these are **training-only**
  runs, and held-out performance is measured separately on the deploy/serving side
  (`eval_unified.py`), so its eval cost never inflates the cost-of-training figures.
- **Latency** — total wall-clock per run, the setup/queue component (Flash rents a GPU:
  queue + boot + model download), and per-step time.
- **Cost** — Flash is the **measured** RunPod bill (`metrics.json:cost_usd`). Tinker is
  managed and **does not expose per-run cost via API**, so its $ column is an explicit
  **active-compute** proxy — rollout+train time with managed-backend capacity pauses
  excluded, times a $/hr GPU rate (clearly labelled in the output, never presented as
  measured).

## Layout

```
benchmark/
  configs/                 flash GRPO TOMLs (one per task)
  bench.py                 orchestrator: run both stacks for one task, print a table
  flash_runner.py          submit to a flash control plane, poll, read metrics.json
  tinker_runner.py         run tinker_cookbook verifiers_rl GRPO (needs `verifiers`)
  run_flash_plane.py       a local flash control plane (see "HF repo cap" below)
  assemble.py              pull all 6 runs -> results/comparison.{json,md}
  results/                 runs_manifest.json + generated comparison + per-run JSON
```

## Running it

```bash
# Flash side needs a reachable control plane (FLASH_API_URL + a freesolo key).
# Tinker side needs TINKER_API_KEY and an interpreter with `tinker` + `verifiers`.
# Tinker trains server-side, so the local env needs NO torch/vLLM — just:
#   pip install tinker tinker-cookbook 'verifiers==0.1.9'   # 0.1.9 matches tinker_cookbook 0.4.0
#   pip install gsm8k reverse-text hendrycks-math --extra-index-url \
#       https://hub.primeintellect.ai/primeintellect/simple/

# One task, both stacks, side by side:
uv run python benchmark/bench.py --env-id gsm8k

# All 6 -> comparison table:
uv run python benchmark/assemble.py
```

## Operational notes (real-world frictions hit while building this)

These are documented because they bit during the run and will bite again:

1. **`tinker_cookbook` 0.4.0 ↔ `verifiers` version skew.** The recipe is pinned to
   `verifiers>=0.1.9,<0.1.10`; newer verifiers (0.1.10+) change `Environment.run_group`'s
   signature and the dataset schema. Use **`verifiers==0.1.9`**. `tinker_runner.py` also
   patches two residual skews: it injects the `task` dataset column the builder expects,
   and it wires the verifiers rollout into the *actual* call site
   (`rollouts.do_group_rollout`) — the recipe monkey-patches `train.do_group_rollout`,
   which the sync training path never calls, so without the fix every rollout returns
   empty (`zip(*[])` → ValueError).

2. **HuggingFace repo-creation cap (300/day/user).** Flash creates a per-run HF dataset
   repo for code upload + checkpoint streaming. Once the daily cap is hit, *every* new
   run fails at submit with a 429 from `repos/create` — even for repos that already exist.
   `run_flash_plane.py` is a drop-in local control plane whose only change is a shim that
   tolerates the create-429 by **reusing a pre-existing artifact repo** (commits via
   `upload_folder` are not daily-capped). Flash core is untouched. For the recorded run,
   two tasks reused disposable smoke repos for exactly this reason (see
   `results/runs_manifest.json`).

3. **4B GRPO escalates GPU class.** `Qwen3.5-4B` GRPO needs ≥35 GB VRAM (model + the
   hybrid-arch engine state-cache), so Flash's allocator escalates the requested RTX 5090
   (32 GB) to **A100 PCIe**. Both auto-escalated, so the hardware is held constant across
   the Flash runs.

See `results/comparison.md` for the numbers.
