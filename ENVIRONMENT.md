# Flash environment variables

A reference for every environment variable the `flash` package reads, what it actually does, and
who sets it. Scope tags:

- **operator** — set on the control plane (`.env` / process env); see `.env.example`.
- **worker** — read on the GPU worker at train time.
- **fwd** — an operator/process env var that `build_worker_env` forwards to the worker via an
  allowlist (and that a run can override per-run through the `[worker_env]` TOML table).
- **internal** — set by the code itself (control plane → worker, or build/bootstrap markers).

> Training **environments** themselves are not configured here — they're external Prime Hub
> verifiers slugs referenced by `[environment] id = "owner/name"` in the run TOML.

---

## Credentials & auth (operator, `.env`)

| Var | What it does |
|-----|--------------|
| `RUNPOD_API_KEY` | RunPod Flash API auth — provision/poll/destroy GPU endpoints. RunPod is the default substrate. |
| `VAST_API_KEY` | Vast.ai API auth. Opt-in: the Vast provider is only active when this is set. |
| `HF_TOKEN` | HuggingFace **write** token for each run's `[train] hf_repo` (code upload + streamed checkpoints/adapters land in that per-run dataset repo). Canonical name; the legacy `HUGGINGFACE_TOKEN` is mirrored into `HF_TOKEN` at startup for back-compat. |
| `PRIME_API_KEY` | Prime Intellect key — the worker needs it to `prime env install` the run's Prime Hub verifiers environment. |
| `FREESOLO_INTERNAL_KEY` | Shared internal key (same value the platform/SDK hold) → resolves to a single service identity with no network call; also the bearer the control-plane REST API accepts. |
| `FREESOLO_BASE_URL` | Where unknown user bearer tokens are verified (`{BASE}/api/auth/verify`). Default `https://api.freesolo.co` (in compose, `http://backend:8000`). |
| `FREESOLO_API_KEY` | Client-side SDK auth (the `slm` client's key). |
| `WANDB_API_KEY` | When present, the worker reports training to Weights & Biases; absent ⇒ `metrics.json` is the only record. |

## Control-plane behavior (operator)

| Var | What it does | Default |
|-----|--------------|---------|
| `AUTOSLM_API_URL` | Control-plane URL the `slm` client targets (env → config file → built-in default). | built-in |
| `AUTOSLM_HF_REPO_PRIVATE` | `1`/unset ⇒ create per-run artifact repos **private**; `0`/`false` ⇒ **public** (workaround for the private-storage-quota 403 when a worker reads its code repo). | `1` (private) |
| `AUTOSLM_GPU_ALLOW_UNVALIDATED` | Truthy ⇒ allocator may pick GPU classes outside the validated matrix (per-run `gpu.allow_unvalidated` does the same). | off |
| `AUTOSLM_VAST_ALLOW_COMMUNITY` | `1`/`true` ⇒ allow Vast **community/marketplace** hosts (not just verified datacenters). Off by default because run secrets ship to the box. | off |
| `AUTOSLM_PRICE_TTL_S` | TTL (seconds) for the RunPod price cache. | `21600` (6h) |
| `AUTOSLM_MIN_CUDA` | Override the per-GPU minimum host-driver CUDA version the allocator requires. | per-GPU `min_cuda_modern` |
| `AUTOSLM_ENVS_MANIFEST` | Path to the local env-manifest JSON. | `~/.autoslm/envs.json` |
| `AUTOSLM_SKIP_NET` | Skip all network in CPU test runs. | off |
| `AUTOSLM_LOG_LEVEL` / `AUTOSLM_DEBUG` | Logging verbosity / debug output. | — |

## Worker image & dependency overrides (operator)

| Var | What it does |
|-----|--------------|
| `AUTOSLM_WORKER_IMAGE` | Override the baked worker image (`ghcr.io/freesolo-co/autoslm-worker:cu128`). Blank it to fall back to live dep-install. |
| `AUTOSLM_WORKER_DEPS` | Replace the entire pinned worker dep stack (`"pkgA==1 pkgB>=2"`, or a JSON list for specs with commas). |
| `AUTOSLM_WORKER_EXTRA_DEPS` | *Append* extra pip deps on top of the pinned stack. |
| `AUTOSLM_SERVE_DEPS` | Override the serving (vLLM) deployment's dep set. |

## Worker job inputs (internal: control plane → worker)

| Var | What it does |
|-----|--------------|
| `RUN_MODE` | `sft` \| `rl` — selects the worker handler. |
| `RUN_ID` | Run identity (artifact paths key off it). **Reserved** from `[worker_env]` override. |
| `SEED` | Training seed. |
| `HF_REPO` | The run's artifact repo the worker reads/writes. **Reserved** from `[worker_env]` override. |
| `AUTOSLM_JOB_SPEC_JSON` / `AUTOSLM_JOB_SPEC_PATH` | The full `JobSpec` — inline JSON, or via a file path for large specs. |
| `AUTOSLM_ARM` | Which substrate the worker runs under (`runpod` default \| `vast`; Vast rewrites it in its bootstrap). **Reserved** from `[worker_env]`. |
| `AUTOSLM_ALLOC_AUTO` | `1` lets the worker upgrade the CUDA allocator to `expandable_segments` (anti-fragmentation on long colocate runs). |
| `HF_HOME` | HF cache dir (e.g. `/runpod-volume/hf-cache` on RunPod network volumes). |
| `HF_HUB_ENABLE_HF_TRANSFER` | Enable the fast HF transfer backend. |
| `BENCH_HF_MODEL` | Model id override for the built-in bench path. |

## SFT knobs (fwd)

| Var | What it does |
|-----|--------------|
| `SFT_PER_DEVICE_BS` | Per-device micro-batch size. |
| `SFT_PACKING` | `0` disables example packing (which otherwise concatenates short examples to fill `max_length`). |
| `SFT_EPOCHS` | Epoch-count override. |

## GRPO / vLLM knobs (fwd)

| Var | What it does |
|-----|--------------|
| `RL_STEPS` | GRPO optimizer-step count. |
| `RL_VLLM_GPU_UTIL` | vLLM `gpu_memory_utilization` — lower it to fix colocate OOM / KV-cache errors. |
| `RL_VLLM_SLEEP` | vLLM sleep-mode toggle (offload the engine between steps). May be pinned per-run via `[worker_env]`. |
| `RL_PER_DEVICE_PROMPTS` | Per-device prompt count (colocate memory). |
| `RL_LOGITS_BUDGET_GB` | Caps fp32-logits memory (the GRPO OOM driver); the worker memory-caps `per_device` to stay under it. Default `6` GB. |
| `VLLM_USE_V1` | Select the vLLM v1 engine. |
| `VLLM_ATTENTION_BACKEND` | Attention-backend escape hatch (`TRITON_ATTN`/`FLASHINFER`) when vLLM's bundled flash-attn PTX is newer than the host driver's JIT (sm_120 + 12.8 drivers). |

## Mid-run GRPO eval (fwd)

| Var | What it does | Default |
|-----|--------------|---------|
| `AUTOSLM_EVAL_EVERY_STEPS` | Operator override for the periodic-eval cadence (`>0` enables). Normally comes from the run's `[train] eval_every_steps`. Everything else (eval set, grading, completion budget, threshold) comes from the environment. | from TOML |
| `AUTOSLM_EVAL_NUM` | Safety cap on held-out rows scored per eval, so eval can't dominate training. | `64` |

## LoRA / quant / judge (fwd)

| Var | What it does |
|-----|--------------|
| `LORA_TARGETS` | Override which modules receive LoRA adapters. |
| `AUTOSLM_QUANT` | Base-model quantization tier (e.g. `4bit-qlora` for NF4 QLoRA). |
| `AUTOSLM_JUDGE_MODEL` | Judge model id the optimizer-authored verifiers environment uses to run its reward/judge. |
| `WANDB_ENTITY` / `WANDB_PROJECT` | W&B routing (project defaults to `flash`/`autoslm`). |

## Heartbeat throttle (fwd)

| Var | What it does | Default |
|-----|--------------|---------|
| `AUTOSLM_HEARTBEAT_MIN_S` | Min interval between `rl_step` heartbeat **uploads**. Raise it to stay under HuggingFace's 128-commits/hour-per-repo limit when several concurrent GRPO runs share one `HF_REPO`. A non-positive/unparseable value resets to 60s. | `60` |

## Perf / CUDA (operator/worker)

| Var | What it does |
|-----|--------------|
| `PYTORCH_CUDA_ALLOC_CONF` / `PYTORCH_ALLOC_CONF` | CUDA caching-allocator config (fragmentation control on long runs). Resolvable per-run via `[worker_env]`. |
| `TORCHDYNAMO_DISABLE` | Disable TorchDynamo compile. |
| `PRIME_DISABLE_VERSION_CHECK` | Set to `1` around Prime CLI calls to skip its version check. |

## Vast bootstrap / self-destroy (internal)

| Var | What it does |
|-----|--------------|
| `CONTAINER_ID` / `CONTAINER_API_KEY` | Vast-injected; used by the worker's self-destroy backstop to tear its own instance down. |
| `FLASH_IS_LIVE_PROVISIONING` | Set `"true"` so the worker knows it's a live-provisioned (ad-hoc) Flash run. |
| `AUTOSLM_BOOTSTRAP_EOF` / `AUTOSLM_DESTROY_EOF` / `AUTOSLM_PAYLOAD_EOF` | Unique heredoc delimiter markers in the Vast bootstrap script — **not** runtime config knobs. |

---

## The `[worker_env]` table and `_worker_env()` (schema.py)

A run's TOML may carry a `[worker_env]` table — a string→string map of **per-run** worker env
overrides. `build_worker_env` applies them **after** the global allowlist, so a per-run value wins
over the operator default (this is what lets one concurrent run differ — e.g. a per-run
optimizer/LoRA-init A/B — while every other run keeps the global default).

Two guardrails:

1. **Secret rejection (`_worker_env`).** `[worker_env]` is serialized into `job_spec_json`, which is
   persisted and logged — so it must never carry secrets (they'd leak into run artifacts).
   `_worker_env` rejects secret-looking keys by **`_`-delimited word components** (not substring):
   a key is refused if any word is a secret word (`TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL`, …) or
   if it contains the word `KEY` qualified by `API`/`SECRET`/`PRIVATE`/`ACCESS`/`INTERNAL`/`AUTH`/…
   That catches `HF_TOKEN`, `*_API_KEY`, `SECRET_KEY`, `INTERNAL_KEY`, `AWS_SECRET_ACCESS_KEY` while
   allowing legit knobs that merely contain a marker substring (`RL_VLLM_MAX_BATCHED_TOKENS` → word
   `TOKENS`, not `TOKEN`; a bare `SORT_KEY` → `KEY` with no secret qualifier). Operators set real
   secrets as process env vars instead, which reach the worker out-of-band (never via the spec).
2. **Reserved identity keys.** `RUN_ID`, `HF_REPO`, and `AUTOSLM_ARM` are control-plane-owned and
   excluded from `[worker_env]`: the poller, deploy, and artifact paths all key off
   `spec.run_id` / `spec.train.hf_repo`, so letting a run override them would make the worker upload
   under a different repo/prefix and orphan its artifacts.
</content>
