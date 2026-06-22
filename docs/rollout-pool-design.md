# Shared rollout pool — distributed GRPO that doesn't pay for idle GPUs

## The problem (where the money goes)

A GRPO step on a single trainer GPU is three phases:

1. **rollout** — vLLM decodes the group of completions (HBM-bandwidth bound)
2. **reward** — score the completions (often CPU / IO / LLM-judge / sandbox bound)
3. **optimizer** — the LoRA forward/backward + step (compute bound)

During (1) and (2) the expensive trainer GPU's matmul units sit **idle**. On a real run that is
30–70% of wall-clock — you are renting an H100 to wait on a vLLM decode and a reward HTTP call. That
is wasted time and wasted money.

The earlier work narrowed this:

- **`flash.engine.disaggregated`** (PR #4) moves rollout onto a dedicated inference card (vLLM
  server mode) — but the trainer still blocks on it (server mode is synchronous), and a single run
  can't keep a whole inference GPU busy.
- **`flash.engine.verl_runner`** (the `verl-async` base of this branch) overlaps gen(t+1) with
  train(t) via verl one-step-off — real overlap, but still **one run per rented box**, and reward is
  still inline.

## The idea (this PR)

Rent the inference GPUs **once**, put a router in front, and point **every** training run at it:

```
   training servers                 ROLLOUT ROUTER  (the "nginx")               rollout servers
   (one per run)            /v1/chat/completions  → least-load upstream         (pre-rented GPUs)
  ┌───────────┐            ┌──────────────────────────────────────┐           ┌──────────────┐
  │ run-alpha │──generate─▶│ resolve adapter → pick GPU            │──────────▶│ gpuA0  baseA │
  │ (LoRA A)  │──sync─────▶│ lazy-load / hot-swap LoRA, balance    │           │ LoRA: α, β   │
  └───────────┘            │ retry / failover                      │──────────▶│ gpuA1  baseA │
  ┌───────────┐            │                                       │           │ LoRA: α, β   │
  │ run-beta  │──generate─▶│ /rewards/score → off-GPU reward pool  │──────────▶│ gpuB0  baseB │
  │ (LoRA B)  │──score────▶│                                       │           │ LoRA: γ      │
  └───────────┘            └───────────────────┬──────────────────┘           └──────────────┘
                                               │ fan-out
                                       ┌───────┴────────┐  reward workers (CPU)
                                       │ rw0   rw1 ...   │  rubric / judge / sandbox
                                       └────────────────┘
```

Three things fall out of this, and they are exactly the asks:

1. **Train multiple models on one GPU.** vLLM serves many LoRA adapters off one base model on one
   GPU (`--enable-lora --max-loras`). Each run = one adapter. Dozens of small runs co-reside on a
   card — one base weight, N adapters. The router places adapters and routes each request to a GPU
   that holds the right one.
2. **Distribute across many GPUs (the nginx).** With several GPUs per base model, the router
   load-balances generation by least-outstanding-requests and replicates hot adapters — so a burst
   of rollouts from any run spreads across the fleet.
3. **Rollout + reward never cost trainer-GPU time.** The trainer offloads rollout to the pool and
   reward to off-GPU workers, and the **pipelined producer** runs a step ahead: while the trainer
   does the optimizer step for batch `t`, the pool is already generating + scoring batch `t+1`. The
   reward function's latency therefore overlaps the gradient step instead of blocking the GPU — *the
   reward latency doesn't matter for the cost*. And because the pool is shared, when run A is in its
   optimizer phase (not generating) run B's rollouts keep the GPUs busy: the pool, not any single
   trainer, is what stays saturated.

## Components (`flash/pool/`)

| module | role |
|---|---|
| `state.py` | `Backend` / `Adapter` / `PoolState` registry + balancing policy (pure, no IO) |
| `protocol.py` | vLLM OpenAI + dynamic-LoRA wire shapes (pure builders) |
| `gateway.py` | async httpx calls to one vLLM upstream (injectable for tests) |
| `router.py` | the FastAPI router — `create_pool_app()` (least-load, lazy/replicated LoRA, hot-swap, failover, reward fan-out) |
| `rewards.py` | off-GPU reward-worker registry + dispatch + `create_reward_app()` |
| `client.py` | trainer-side `RolloutPoolClient` (generate / score / sync_weights / **`experience_stream`** pipelined producer) |
| `provision.py` | pre-rent the fleet (`build_vllm_serve_command`) + register with the router |
| `config.py` | `RouterConfig` / `PoolPlan` from env / TOML |
| `server.py` + `__main__.py` | the `flash-pool` operator CLI |

The training-server half lives in `flash/engine/`:

| module | role |
|---|---|
| `pool_trainer.py` | `GRPOPoolLoop` orchestration + `compute_group_advantages` + the `FLASH_FRAMEWORK=pool` worker entry |
| `pool_policy.py` | the GPU step — advantage-weighted LoRA update (the only part that touches the trainer GPU) |

The control-plane side (router/state/client) is **torch-free** — only `fastapi` + `httpx` (the
`server` extra). Only the rollout backends run vLLM.

## Running it

```bash
# 1) operator: start the router (CPU box)
pip install 'freesolo-flash[server]'   # dist name is freesolo-flash; CLI/import stay `flash`
flash-pool serve --port 8077

# 2) operator: pre-rent the fleet and register it (one base model per GPU, multi-LoRA on each).
#    Each GPU runs a stock vLLM server (see build_vllm_serve_command):
#    python -m vllm.entrypoints.openai.api_server --model <base> --enable-lora --max-loras 8 ...
#    with VLLM_ALLOW_RUNTIME_LORA_UPDATING=1, then:
curl -X POST $ROUTER/pool/backends -d '{"id":"gpuA0","url":"http://10.0.0.5:8000","base_model":"<base>","max_loras":8}'

# 3) operator: start reward workers (CPU) and register them
flash-pool reward my_rewards:score --port 8078
curl -X POST $ROUTER/rewards/workers -d '{"id":"rw0","url":"http://10.0.0.9:8078"}'

# 4) each training run points at the pool
export FLASH_ROLLOUT_POOL_URL=http://router:8077
export FLASH_FRAMEWORK=pool          # the worker runs flash.engine.pool_trainer
```

`flash-pool plan pool.toml` dry-runs a fleet (`[[pool]] base_model=... gpu=... count=...`) and
prints capacity + how many concurrent adapters it holds.

### Using the client directly

```python
from flash.pool.client import RolloutPoolClient

c = RolloutPoolClient("http://router:8077", adapter="run-alpha", base_model="org/Qwen-A")
c.register(uri="/lora/run-alpha/v0", replicas=2)        # warm it on 2 GPUs
for exp in c.experience_stream(prompt_batches, n=8, prefetch=2):
    advantages = compute_group_advantages(exp.rewards)  # rollout+reward already done, off-GPU
    new_uri = policy_update(exp, advantages)            # the only GPU work
    c.sync_weights(new_uri)                             # hot-swap the fresh policy onto the pool
```

## What's verified (CPU) vs. what's a live step

- **CPU-tested end-to-end** (`tests/test_pool_*.py`, 35 tests): balancing/eviction/failover, lazy +
  replicated multi-LoRA placement, per-step weight-sync hot-swap, reward fan-out, the
  `RolloutPoolClient`, the pipelined overlap (a *slow* reward barely changes the trainer's
  wall-clock — proving it's overlapped), and the full `GRPOPoolLoop`. The fake backend implements
  vLLM's real dynamic-LoRA + OpenAI surface, so the router/client exercise the production code path.
  See `examples/rollout_pool_demo.py` for a runnable demo.
- **Live multi-GPU run (done)**: validated end-to-end on a real **4× RTX 3090** box — two real
  `vllm serve --enable-lora` servers behind the router + two real concurrent LoRA trainers pushing
  real adapters through it (generation load-balanced 6/6 across both GPUs, both LoRAs co-resident on
  both GPUs, per-step weight-sync hot-swap to vLLM, version 1→3). See
  [`rollout-pool-live-run.md`](rollout-pool-live-run.md).

## Relationship to the other rollout paths

`disaggregated` (sync, one box) → `verl_runner` (overlap, one box) → **rollout pool (overlap +
shared across runs and GPUs)**. The pool is the fleet-level generalization: it keeps the GPUs busy
across *all* runs, which is where the cost actually leaks.
