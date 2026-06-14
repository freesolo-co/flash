# Config reference

AutoSLM runs are described by a single TOML, parsed into a `JobSpec`.
Compose multiple files with `--config` (deep-merged in order) and override any value with
`--set key=value` (dotted path; highest precedence).

```toml
model = "Qwen/Qwen3.5-4B"     # catalog model (`slm models`), or any HF id with model_policy="allow"
model_policy = "catalog"      # "catalog" (default) | "allow" (any HF model that fits the GPU)
thinking = false              # opt-in reasoning mode for thinking-capable models (see below)
algorithm = "grpo"            # "sft" | "grpo" | "opd" | "dpo"

[environment]
id = "gsm8k"                  # built-in (gsm8k|math|tests_pass) or a verifiers/Prime Hub env id
# pip = ["verifiers", "my-env-wheel"]  # optional: explicit worker pip requirements for the
#                                      # environment (default: derived from `slm env install`'s
#                                      # local manifest and shipped with the run)
# path = "environments/my_env"         # local custom env dir — NOT supported on the managed
#                                      # service (publish with `slm env push` instead)
[environment.params]          # optional kwargs forwarded to load_environment(**params)
# preference_dataset = "trl-lib/ultrafeedback_binarized"   # required for algorithm="dpo"

[distill]                     # required for algorithm="opd"
teacher = "Qwen/Qwen3-4B-Instruct-2507"  # teacher loads in-process on the same GPU (bf16)
lmbda = 1.0                   # fraction of on-policy (student-generated) batches
max_completion = 320

[train]
steps = 150                   # GRPO/OPD optimizer steps
epochs = 2                    # SFT/DPO epochs
lora_rank = 32
lora_alpha = 64
seeds = [0, 1]                # one dedicated GPU per seed
eval_examples = 300

[gpu]
# type = "RTX 5090"          # pin a GPU class; OMIT (or "cheapest") for smart allocation
# provider = "auto"          # "auto" | "runpod" | "vast" (vast = verified datacenters only)
disk_gb = 60                 # worker container disk GB (raised automatically to the
                             # model's min_disk_gb; raise-only above the 64 GB default)
max_wall_seconds = 86400     # execution cap (per seed; enforced on every substrate)
max_retries = 2              # auto-resubmit budget for infra failures (stall/worker loss/timeouts)
# allow_unvalidated = true   # opt in to GPU classes that haven't passed AutoSLM's live smoke
# network_volume = "mycache" # OPT-IN persistent volume: cross-run HF model cache (see below)
# network_volume_gb = 100
# datacenter = "EU-RO-1"     # volume datacenter (pins the run's GPU pool!)
```

## GPU selection (smart allocation)

When `gpu.type` is omitted (or set to **`"cheapest"`**/`"auto"`), AutoSLM allocates the
**cheapest GPU across providers** with enough VRAM to comfortably run the full job
(sized for GRPO — the heavier phase of the usual SFT→GRPO pipeline — with headroom for
open models; curated catalog entries carry measured minimums). Allocation happens live
at submit time, per attempt:

- **RunPod**: every Flash class, ranked by live RunPod pricing (cached 6 h; static
  snapshot offline).
- **Vast.ai**: live offers from **verified datacenter hosts only** — never consumer
  machines — additionally filtered by reliability (≥ 0.95), download bandwidth
  (≥ 200 Mbps), disk, and the class's minimum CUDA driver (Blackwell needs CUDA 13).
  Enabled when the operator sets `VAST_API_KEY`.

`gpu.type` accepts any managed class (`RTX 4090`, `RTX 5090`, `RTX A4000`, `RTX A4500`,
`RTX 4000 Ada`, `RTX 2000 Ada`, `RTX A5000`, `RTX 3090`, `L4`, `RTX Pro 4000`, `A40`,
`RTX A6000`, `RTX 6000 Ada`, `L40S`, `A100 SXM 40GB`, `A100 PCIe`, `A100 SXM`, `H100`,
`H200`, `RTX Pro 6000`, `B200`); a concrete name pins the class and the allocator only
picks the cheaper provider for it. `gpu.provider` pins the substrate (`L40S`,
`RTX Pro 4000`, `A100 SXM 40GB` are vast-only; `RTX Pro 6000`/`B200` runpod-only).
`slm gpus` shows the live per-provider price book.

Two guard rails:

- Validation is **per provider**: only classes that passed AutoSLM's live train+eval
  smoke on a substrate are selectable there by default; others need
  `gpu.allow_unvalidated = true` (or `AUTOSLM_GPU_ALLOW_UNVALIDATED=1`).
  `"cheapest"` honors the same gate.
- "Cheapest" means cheapest **per hour**, not per run: older Ampere cards
  (A4000/A5000/3090/A40) are 2–4x cheaper than a 4090/5090 but also slower per step —
  a good trade for queue-pressure escape and small models; measure before committing
  long runs.

Serving (`slm deploy` / `slm chat`) runs on RunPod Flash only: a run trained on a
vast-only class is served from its model's default RunPod class automatically.

### Big checkpoints (`disk_gb` / `min_disk_gb`)

The platform's default worker disk is 64 GB. Catalog models that need more declare
`min_disk_gb` (e.g. Qwen3.6-35B-A3B's ~72 GB bf16 checkpoint -> 160 GB) and the
orchestrator raises `gpu.disk_gb` automatically; for unlisted big models set
`gpu.disk_gb` yourself. Verified live: RunPod honors the larger container disk on
serverless GPU workers.

### Network volume (opt-in cross-run model cache)

`gpu.network_volume = "<name>"` mounts a persistent RunPod volume at `/runpod-volume`
and points the worker's `HF_HOME` at it, so repeat runs skip the model download
entirely. Trade-offs (why it's off by default): the volume pins runs to **one
datacenter** (`gpu.datacenter`), which usually hurts more via a smaller GPU pool than
the download costs (Flash workers pull at ~10 Gbit/s — an 8 GB model lands in ~6 s),
and the volume bills monthly while it exists. Most useful for repeated 35B-class runs
(72 GB pulls) inside one datacenter.

## Algorithms

| algorithm | engine | data | typical use |
|---|---|---|---|
| `sft` | TRL `SFTTrainer` | env `dataset("train")` + `sft_target` | imitate reference completions |
| `grpo` | TRL `GRPOTrainer` + colocated vLLM | env prompts + `reward` | verifiable-reward RL |
| `opd` | TRL `DistillationTrainer` | env prompts + `[distill] teacher` | on-policy distillation: dense token-level teacher supervision on the student's own rollouts — often RL-level lift at a fraction of the cost |
| `dpo` | TRL `DPOTrainer` | `[environment.params] preference_dataset` (prompt/chosen/rejected) | offline preference tuning |

## Thinking mode (`thinking = true`)

Off by default: every run renders with `enable_thinking=false` so out-of-the-box
behavior (token budgets, prompts, eval) is unchanged. Setting `thinking = true` turns
on the model's reasoning mode **for the whole run** — SFT targets, RL rollouts, eval,
and serving all render with the same flag (decoding parity is per-run).

- **Capability-gated**: `slm models` shows each model's `thinking` capability —
  `hybrid` (template honors the flag) or `none` (e.g. the default
  Qwen3-4B-Instruct-2507, a non-thinking variant; `thinking = true` is rejected).
  Always-thinking models (R1-style distills) work via `model_policy = "allow"` and
  *require* `thinking = true`.
- **Grading**: `<think>...</think>` blocks are stripped before the environment grades
  / rewards (in `worker.graded_text`, so every environment benefits). A completion
  whose reasoning never closes (budget exhausted) scores **0** — deliberate reward
  pressure to think within budget. Eval generations record per-completion `n_tokens`
  and `truncated` so cap saturation is visible.
- **Budgets**: thinking-aware defaults replace the non-thinking ones — RL completion
  cap 320 -> 1536, eval `max_new_tokens` 512 -> 2048, SFT `max_seq_len` 1024 -> 2048,
  GRPO per-device completion micro-batch 8 -> 2 (the fp32 logits pass scales with
  sequence length; grad-accum compensates, so the effective batch is unchanged).
  `RL_MAX_COMPLETION` / `EVAL_MAX_NEW_TOKENS` / `RL_PER_DEVICE_PROMPTS` still
  override. Bigger models think longer — raise `EVAL_MAX_NEW_TOKENS` to 3072–4096 for
  4B-class thinking evals. **Cost warning**: RL rollout cost scales linearly with
  completion length — expect roughly 5x the generated tokens per step vs non-thinking.
- **SFT**: targets should contain `<think>` traces; the worker warns loudly when none
  do (training thinking-rendered prompts on non-reasoning targets teaches the model to
  skip thinking).
- **Serving**: a run trained with thinking serves with thinking; responses carry the
  raw `<think>...</think>` block in `message.content` (no `reasoning_content`
  separation yet). Raise `slm chat --max-tokens` accordingly.

## Reliability model

Every seed runs as a durable job: the `{endpoint_id, job_id}` handle is persisted in the
run status, so `slm attach <run_id>` can re-attach from any process after a client
crash. Trainer checkpoints stream to the HF repo on every save; if the GPU worker dies,
the auto-retry (bounded by `gpu.max_retries`) resubmits on a FRESH endpoint and the
replacement worker resumes from the latest streamed checkpoint. Endpoints are torn down
on every terminal state (lingering endpoints exhaust the account's max-workers quota).

## Serving

```bash
slm deploy <run_id> --mode dev          # scale-to-zero: cold start after idle, $0 when unused
slm deploy <run_id> --mode always-on    # one warm worker 24/7: no cold starts, continuous billing
slm deploy <run_id> --idle-timeout 300  # dev mode: seconds before scale-to-zero
slm chat <run_id> -m "..."              # OpenAI-shaped chat through the managed GPU
slm serve-proxy <run_id> --port 8000    # local OpenAI-compatible /v1 shim (any OpenAI SDK client)
slm deployments                          # list active deployments + projected $/day
slm undeploy <run_id>                    # tear the serving endpoint down
```

Measured on an RTX 4090 dev-mode deployment (Qwen3-0.6B + adapter): cold start ~4 min
(deps + model + engine boot on a fresh host), warm requests ~10 s end-to-end through the
queue. `always-on` eliminates cold starts at ~$11–19/day depending on GPU.

## Overrides & composition

```bash
slm train base.toml --config prod.toml --set train.steps=300 --set gpu.type="RTX 4090"
slm train base.toml --set train.seeds=[0,1,2]
slm train base.toml --set model=openbmb/MiniCPM5-1B-Base --set model_policy=allow
```

Values are coerced: `true/false` -> bool, ints/floats parsed, `[a,b]` -> list.

## Environment variables

Client-side:

| Var | Purpose |
|---|---|
| `AUTOSLM_API_KEY` | your AutoSLM key (normally stored by `slm login`) |
| `AUTOSLM_API_URL` | control-plane URL (e.g. a self-hosted server; see [self-hosting](self-hosting.md)) |

Server-side (operator; the remaining variables below also apply on the control-plane
host — see [self-hosting](self-hosting.md)):

| Var | Purpose |
|---|---|
| `RUNPOD_API_KEY` | RunPod auth (operator credential) |
| `VAST_API_KEY` | Vast.ai auth (operator credential, optional) — enables verified-datacenter offers in cross-provider allocation (see [self-hosting](self-hosting.md) for the `AUTOSLM_VAST_*` tuning knobs) |
| `HF_REPO` | HF **dataset** repo for adapters/checkpoints + code delivery |
| `HUGGINGFACE_TOKEN` | write access to `HF_REPO` |
| `AUTOSLM_DB_PATH` | control-plane SQLite (keys + run ownership) |
| `AUTOSLM_RUNS_DIR` | run status/log files |
| `AUTOSLM_WORKER_STACK` | `modern` (default; trl 1.x / vllm 0.19 / transformers 5.x) or `legacy` |
| `AUTOSLM_WORKER_DEPS` | fully custom worker dependency list (whitespace-separated, or a JSON list for specs containing commas like `transformers>=5.6,<5.11`) |
| `AUTOSLM_WORKER_IMAGE` | optional prebuilt worker image (deps and base model baked in); any image exposing the pinned worker stack works |
| `AUTOSLM_MIN_CUDA` | minimum host driver CUDA version. Auto: 13.0 for Blackwell classes (RTX 5090 / RTX Pro 6000 / B200 — their wheels ship no SASS; older drivers cannot JIT the PTX) on the modern stack, 12.8 otherwise |
| `AUTOSLM_GPU_ALLOW_UNVALIDATED` | set `1` to allow GPU classes that haven't passed AutoSLM's live smoke (same as `gpu.allow_unvalidated`) |
| `AUTOSLM_PRICE_TTL_S` | live GPU pricing cache TTL (default 6 h) |
| `AUTOSLM_QUANT_REPO` | override the pre-quantized weights repo for the QLoRA tier |
| `AUTOSLM_WORKER_EXTRA_DEPS` | additive worker pip deps (whitespace-separated), e.g. `liger-kernel` for `SFT_LIGER=1` |
| `SFT_PACKING` | set `1` to pack short SFT examples into full sequences (A/B-gated) |
| `SFT_LIGER` | set `1` for Liger fused kernels in SFT (needs `AUTOSLM_WORKER_EXTRA_DEPS=liger-kernel`) |
| `AUTOSLM_LEGACY_SUBMIT` | set `1` to use the legacy blocking submit path (no durable handles) |
| `AUTOSLM_STALL_AFTER_S` | supervisor stall watchdog (default 1500 s of no worker progress) |
| `AUTOSLM_EXECUTION_TIMEOUT_MS` | default server-side execution cap |
| `AUTOSLM_LOG_LEVEL` | CLI log level (`DEBUG`/`INFO`/`WARNING`/...); same effect as `-v`/`-vv` |
| `AUTOSLM_DEBUG` | set `1` to show full tracebacks on error (same as `--debug`) |
| `SFT_MAX_STEPS`, `SFT_MAX_EXAMPLES`, `SFT_PER_DEVICE_BS`, `SFT_SAVE_STEPS`, `RL_MAX_COMPLETION`, `RL_SAVE_STEPS`, `EVAL_MAX_NEW_TOKENS` | smoke/tuning passthrough (`RL_MAX_COMPLETION`/`EVAL_MAX_NEW_TOKENS` defaults are thinking-aware: 320/512 normally, 1536/2048 with `thinking = true`) |
| `EVAL_MAX_MODEL_LEN` | eval engine context bound (default `max(2048, prompt + completion + 128)`) |
| `AUTOSLM_THINKING` | worker-side thinking override for the no-JobSpec bench path (runs use the TOML `thinking` flag) |
| `OPD_PER_DEVICE_BS`, `OPD_BATCH`, `OPD_LR`, `DPO_LR`, `DPO_BETA` | opd/dpo tuning |
| `RL_PROMPTS_PER_STEP` | unique prompts optimized per GRPO step (default 64). Lower it to reduce per-step memory + wall-time (smaller, noisier batch). |
| `RL_GROUP_SIZE` | GRPO completions sampled per prompt (default 8). |
| `RL_PER_DEVICE_PROMPTS` | GRPO per-device *completion* micro-batch (VRAM/throughput knob; 8 measured fastest for 4B on a 5090). Thinking runs default to 2 — the fp32 logits pass scales with sequence length and OOMs a 24 GB card at 8 with 2048-token thinking sequences; grad-accum compensates, so the effective batch is unchanged. |
| `RL_VLLM_GPU_UTIL` | fraction of GPU memory for the colocated vLLM rollout engine. |
| `RL_VLLM_SLEEP` | offload vLLM weights between steps. Default on; **disable (`0`) when both fit resident — measured ~2x faster per step** (4B on a 32 GB 5090 fits at `RL_VLLM_GPU_UTIL=0.35` with bf16 loading). |
| `RL_VLLM_MAX_LEN` | colocated engine context bound (default prompt+completion; prevents full-context KV sizing). |
| `EVAL_VLLM_GPU_UTIL`, `EVAL_ENFORCE_EAGER` | eval engine knobs (`EVAL_ENFORCE_EAGER=0` required for Qwen3.5-4B+, whose kernels need the compile path) |
| `VLLM_ATTENTION_BACKEND` | escape hatch (e.g. `TRITON_ATTN`) when a host driver cannot JIT vllm's bundled flash-attn PTX |
| `PYTORCH_ALLOC_CONF` | allocator tuning (torch >= 2.10 name; both names are set on the worker). Default `expandable_segments:True`; with `RL_VLLM_SLEEP=1` use `garbage_collection_threshold:0.8,max_split_size_mb:256` instead. |

### Colocated GRPO on one consumer GPU (memory recipes)

Measured known-good recipes on a 32 GB RTX 5090 (modern stack, bf16 loading):

```bash
# Qwen3-4B / Qwen3.5-4B — FAST recipe (vLLM resident; ~2x faster per step than sleep mode)
export RL_VLLM_SLEEP=0 RL_VLLM_GPU_UTIL=0.35 RL_PER_DEVICE_PROMPTS=8 \
       PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256

# Qwen3.5-4B — conservative recipe (sleep mode; use if the fast recipe OOMs on your run)
export RL_VLLM_SLEEP=1 RL_VLLM_GPU_UTIL=0.48 RL_PER_DEVICE_PROMPTS=2 \
       EVAL_ENFORCE_EAGER=0 \
       PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
```
