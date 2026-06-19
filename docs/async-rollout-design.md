# Multi-GPU async (disaggregated) GRPO rollouts — design + benchmark plan

Status: **Feature BUILT and MEASURED end-to-end.** Multi-GPU CONTAINER PROVISIONING is SOLVED on
Vast by filtering to **whole-machine** offers (`num_gpus={"eq":k}` + `gpu_frac >= 0.99`), which hands
one worker all `k` usable GPUs (earlier fractional-offer containers exposed only 1 GPU — see "How
provisioning was solved" below). The colocate / 1:1 / 1:2 / 2:2 ratios are MEASURED on real 5090
nodes (table below). The only configs still open are higher train-heavy ratios (3:1) that need a
larger bench dataset, and a RunPod **Pods** path for >32 GB multi-GPU classes. Numbers tables are
filled ONLY from measured runs — never estimated. (Real GPU-money spend authorized by the user; many
real runs executed on Vast + RunPod to drive this down.)

What landed + is verified working on real hardware:
- **Multi-GPU provisioning plumbing**: `[gpu] count` ([spec.py](../flash/spec.py)/[schema.py](../flash/schema.py)
  with cross-field validation); Vast `search_offers(num_gpus=count)` (exact-match) threaded through
  `usable_offers`/`deploy_and_submit`/`submit_run_vast`; RunPod `gpu_count=spec.gpu.count`.
- **Disaggregated worker mode** ([engine/disaggregated.py](../flash/engine/disaggregated.py) +
  `run_rl`): launches `trl vllm-serve` on the inference GPU(s), health-checks it, runs GRPO with
  `vllm_mode="server"`, TRL's automatic per-step `sync_weights()` (LoRA-merge → NCCL broadcast),
  server torn down in a `finally`. **On a node with 2 real GPUs this reaches `rl_train_start` and
  the weight-sync handshake** (verified on real runs).
- **Capacity-aware allocation** ([providers/runpod/jobs.py](../flash/providers/runpod/jobs.py)
  poll `no_capacity` fast-fail at `queue_grace_s`=720s + [runner](../flash/runner.py)
  class-walk): a throttled/capacity-starved class is detected in ~12 min (not the 50-min stall
  grace) and the run walks to the next-cheapest AVAILABLE class — **validated live** (a RunPod
  A100×2 throttle fast-failed at 724s and advanced). No blacklisting (a class may free up).

Six worker-side bugs found + fixed on real GPUs (each via server-log instrumentation):
log-capture → vLLM `spawn` (NVML fork) → server pinned to device 0 (NVML index) → standard LoRA
init on the pinned trainer (PiSSA SVD crash) → 1200s server-health timeout (spawn boot ~450s) →
**early trainer device-pin in `main()` before any CUDA context** (was binding device 0, colliding
with the server) → `detect_total_gpus` uses the REAL `nvidia-smi -L` count, not the requested hint.

### How provisioning was solved (and the failure mode the filter avoids)

The benchmark NUMBERS require a worker container with ≥2 CUDA-usable GPUs. The fix is to filter to
**whole-machine** Vast offers; the historical failure mode below is exactly what that filter avoids.

- **Vast — SOLVED via whole-machine offers.** The original failure: a **fractional** `num_gpus==2`
  offer (e.g. 2 of an 8-GPU box, `gpu_frac` 0.25/0.5) rents 2 GPUs to the **instance** (verified via
  a rent→`get_instance`→destroy probe: `INSTANCE num_gpus=2, running`) but the **container's** device
  cgroup is limited to its fraction, so the worker saw only **1** GPU (clean `nvidia-smi -L` = 1).
  Setting `NVIDIA_VISIBLE_DEVICES=all` alone could not widen a cgroup that only contained 1 device.
  **The fix is twofold and is in the code now:** (1) `usable_offers` requires an EXPLICIT
  `gpu_frac >= 0.99` (whole-machine; a missing/fractional `gpu_frac` is rejected — `flash/providers/
  vast/jobs.py`), so the rented machine isn't a sliver of a bigger box; and (2) `NVIDIA_VISIBLE_DEVICES
  =all` is set both in the create-time `env` (`flash/providers/vast/jobs.py`) and baked into the image
  (`ENV NVIDIA_VISIBLE_DEVICES=all` in `Dockerfile.worker`). On a whole-machine offer the cgroup
  already contains all `k` GPUs and `=all` surfaces them — `nvidia-smi -L` then shows all `k` and the
  disaggregated split works (the 1:1 / 1:2 / 2:2 rows below are measured on these nodes). The earlier
  "device collision"/NVML/UUID symptoms were two roles contending for the single container-visible GPU
  of a fractional offer — gone once whole-machine offers are enforced.
- **RunPod Flash (serverless) — still capacity-limited.** A100×2 (and the classes the fallback walked
  to) **throttle** — no free multi-GPU serverless worker; the run fast-fails `no_capacity`. RunPod's
  true multi-GPU is **Pods**, a different substrate flash does not use yet.

So the disaggregated CODE is complete + correct and trains end-to-end on the whole-machine Vast nodes;
the measured ratios are below. **Still open:** (a) a RunPod **Pods** provisioner for >32 GB multi-GPU
classes (5090 whole-machine offers are plentiful but only 32 GB, too small for a 4B replica); (b)
larger-dataset 3:1 train-heavy bench; (c) retry RunPod Flash A100 multi-GPU when capacity frees.

Run matrix (the ratios):
- **colocate (1 GPU)** — MEASURED baseline (107.4 s/step, MiniCPM5-1B heavy).
- **1:1 / 1:2 / 1:3** (train_gpus==1) — provisioning SOLVED (whole-machine multi-GPU 5090s rented);
  1:1 (1.02×) + 1:2 (**1.33×**) MEASURED, 1:3 in flight. See the benchmark table below.
- **2:1 / 3:1 / 2:2** (train_gpus>1) — REMAINING: need the accelerate-FSDP multi-trainer launch
  (guarded with a clear error today).

## Why

Today GRPO is **single-GPU, colocated**: TRL runs vLLM in the same process as the trainer and
time-shares the one GPU via `sleep` mode (`trainer.vllm_generation.llm`). Generation (rollout) and
the optimizer step serialize on one device, so the GPU is idle for inference during the backward
and idle for training during generation. For larger models / long completions this is the dominant
cost.

**Disaggregated rollout** puts the inference engine (vLLM server) on its own GPU(s) and the
DDP/accelerate trainer on the rest. Note the realized win here is **NOT** one-step-off async overlap:
TRL `vllm_mode="server"` is **synchronous** (generate → train → sync, no overlap flag), so step _N+1_
generation does NOT overlap the optimizer of step _N_ (measured: 1:1 ≈ colocate at 1.02×). The
speedup instead comes from (a) **tensor-parallel inference throughput** on the dedicated card(s) and
(b) the trainer no longer time-sharing its GPU with vLLM. This is a partial slice of verl's model
(3D-HybridEngine / flexible device mapping: FSDP|Megatron trainer × vLLM|SGLang rollout, with NCCL
weight resharding); the true one-step-off overlap verl provides would need a custom trainer loop (see
the verl-lever table and the "what the numbers say" corrections below).
Ref: https://github.com/verl-project/verl

## Current state (blockers this design removed — now DONE)

- ~~Provisioning is hardwired single-GPU~~ → `[gpu] count` flows to Vast `num_gpus={"eq":count}` and
  RunPod `gpu_count`. (Vast is the multi-GPU substrate; RunPod Flash serverless cold-start proved
  unreliable in testing — a 1-GPU baseline stalled IN_QUEUE for >50 min — so benchmarks run on Vast.)
- ~~The worker only knows the TRL colocate vLLM path~~ → `run_rl` branches to the disaggregated
  server path on `inference_gpus>0`.
- ~~No trainer:inference GPU split / weight-sync transport~~ → `engine.rollout_bench.select_rollout_split`
  + `engine.disaggregated` do the split + server launch; TRL's `sync_weights()` is the weight transport.

## Design

### 1. Provisioning multi-GPU
- **Vast** (primary; has live multi-GPU supply — verified: 2–8 GPU verified offers exist for
  4090/5090/A100/H100 classes): `search_offers(..., num_gpus=k)` → `num_gpus={"eq":k}` (EXACT-match,
  not `>=`) so we rent exactly `k` GPUs and never pay for a larger machine than requested, plus a
  whole-machine `gpu_frac >= 0.99` filter so the container actually sees all `k` GPUs (a fractional
  multi-GPU offer rents `k` GPUs to the instance but exposes only its fraction to the container).
  New `[gpu] count` spec field (default 1).
- **RunPod**: Flash is 1-GPU serverless; multi-GPU needs **Pods**. Out of scope for v1 (Vast covers
  the multi-GPU benchmark); document the Pods path as follow-up.

### 2. Worker disaggregated mode (`run_rl`)
- New `[train] inference_gpus = j` (0 = colocate, the default; j>0 = disaggregated). With a node of
  `count = train_gpus + j`:
  - launch `trl vllm-serve` pinned to `CUDA_VISIBLE_DEVICES = <first j devices>` (tensor-parallel = j;
    inference takes device 0 so vLLM's NVML model-inspection probe queries a valid index — see
    `select_rollout_split`);
  - run the GRPO trainer (FSDP/accelerate) on the first `train_gpus`;
  - TRL `vllm_mode="server"` + `vllm_server_base_url` connects the trainer to the server.
- Weight sync each step via TRL's server weight-update path (NCCL/collective), as TRL 1.6 supports.
- `_select_rollout_mode(count, inference_gpus)` is a **pure function** (unit-tested) returning
  `(mode, train_devices, infer_devices)`; only the subprocess launch touches hardware.

### 3. Re-add Qwen3.6-35B-A3B (disaggregated-only)
- Re-add to the catalog **gated** to the disaggregated path (`requires_disaggregated=True`): the
  reason it was pulled was colocate OOM/instability on one GPU; with a dedicated inference GPU (the
  35B MoE served in 4-bit on its own card) + a sharded trainer, it fits. Colocate requests for it
  are rejected with an actionable message.

### 4. Best GPU per model (starting map; refined by the benchmark)
| Model | colocate GPU (today) | disaggregated node (planned) |
|---|---|---|
| Qwen3.5-0.8B / 2B | RTX 4090 | 2× 4090 |
| Qwen3.5-4B | RTX 5090 | 2× 5090 |
| Qwen3.5-9B | RTX 5090 (QLoRA) | 2–3× 5090 / A100 |
| Qwen3.6-35B-A3B | (n/a — was pulled) | A100/H100 ×N, infer GPU 4-bit |

### 5. Ratio benchmark methodology
For a fixed short GRPO workload (same model, env, steps, group_size, max_tokens, seed), run each
configuration and record **tokens/s**, **s/step**, **wall**, and **$/1k-steps**:

`s/step` = `wall_seconds / steps`; speedup = colocate `s/step` ÷ row `s/step`. **Numbers are filled
ONLY from real runs — none are estimated.**

**MEASURED — MiniCPM5-1B, HEAVY generation-bound workload** (thinking=true, max_tokens 1024,
group_size 8, max_length 2048, steps 12; 6.29M generated tokens/run so decode dominates the step).
Whole-machine RTX 5090 nodes on Vast:

| config (train:infer) | GPUs | s/step | speedup vs colocate | run id |
|---|---|---|---|---|
| colocate | 1 | 107.438 | 1.00× (baseline) | `flash-1781818137-2a1d96c6` |
| 1:1 | 2 | 105.223 | 1.02× | `flash-1781819532-78ed5560` |
| 1:2 (TP=2 infer) | 3 | 80.972 | **1.33×** | `flash-1781820903-6c57a95e` |
| 1:3 (TP=3 infer) | 4 | **invalid** | — | TP=3 rejected: MiniCPM5-1B has 16 heads, not divisible by 3 (`flash-1781822128-7d59f3bd`) |
| 1:4 (TP=4 infer) | 5 | — | — | valid TP but no 5-GPU whole-machine offer available |
| **2:2** | 4 | **54.545** | **1.97×** | DDP 2-trainer + TP-2 infer — BEST; `flash-1781832449-a03db4ba` |
| 2:1 | 3 | — | — | supply-contended (3-GPU scarcity); between 1:2 and 2:2 |
| 3:1 | 4 | n/a | — | 64-prompt bench too thin across 3 DDP ranks (needs larger dataset) |

**Updated conclusion — more GPUs keep helping (correction).** colocate 1.00× < 1:1 1.02× < 1:2 1.33×
< **2:2 1.97×**. The DDP multi-trainer SCALES: each of the 2 trainer ranks issues its half of the
step's generation concurrently to the TP-2 server (better saturation) and runs the optimizer in
parallel, so the balanced 4-GPU 2:2 beats the 3-GPU 1:2 on this generation-bound workload. (Uses DDP,
not FSDP — FSDP param-sharding breaks TRL's per-step LoRA-merge weight-sync.) An earlier
"trainer-heavy doesn't help" note came from prematurely-cancelled runs; the completed 2:2 disproves it.

**The win generalizes across models** (same heavy generation-bound workload, whole-machine 5090):

| model | colocate s/step | 1:2 (TP=2) s/step | speedup | run ids (colocate / 1:2) |
|---|---|---|---|---|
| MiniCPM5-1B | 107.438 | 80.972 | **1.33×** | `…2a1d96c6` / `…6c57a95e` |
| Qwen3.5-0.8B (VL) | 169.173 | 133.653 | **1.27×** | `…7e28f5d5` / `…20ff770b` |
| Qwen3.5-2B (VL) | 119.668 | 140.515 | **0.85× (slower)** | `…71582718` / `…8133d764` |
| Qwen3.5-9B (VL, qlora) | _in flight_ | _in flight_ | — | `…0115f700` / resubmit |
| Qwen3.5-4B (VL) | A100 (≥35 GB) _baseline running_ | **215.49 (3× 5090)** | cross-GPU† | `…2e412dae` / `…942dc159` |

**The 1:2 win is model-size-dependent — a key result.** Net speedup = (TP-2 generation throughput)
− (TRL's per-step weight-sync: merge the LoRA into the policy and NCCL-broadcast it to the server,
every generation batch + server round-trip; and for the VL Qwen models the standalone `trl
vllm-serve` loads the **full** model incl. the vision tower, since it has no language-model-only
patch). For small models generation dominates → **win** (MiniCPM-1B 1.33×, Qwen0.8B 1.27×); by
Qwen3.5-2B the sync + vision-tower overhead **exceeds** the generation gain → **0.85× (slower than
colocate)**. This is exactly the regime verl's **3D-HybridEngine** (zero-redundancy weight
resharding) targets — TRL's naive full-broadcast sync is the ceiling here and caps the win to small
models. The optimal config is therefore per-model: disaggregate small models; colocate ≥2B unless an
optimized resharding path is added.

\* 4B colocate needs ≥35 GB (>32 GB 5090) so it runs on A100; the 4B **1:2** is not provisionable
right now — there are no 3-GPU whole-machine offers of a ≥40 GB card on Vast (5090 multi-GPU is
plentiful but too small for a 4B replica). The small/qlora models (≤2B and the 9B-qlora) run the
full colocate-vs-1:2 comparison on the 5090.

† 4B is the case where disaggregation *enables* a run that colocate can't fit: the 4B 1:2 fits 3× 5090 (24 GB/GPU via the disaggregated server/trainer split) at 215.49 s/step, while 4B colocate needs an A100 (≥35 GB). The s/step are on different GPU classes, so it's an accessibility win (cheap 5090s) rather than a same-card speedup.

The Qwen3.5 `*ForConditionalGeneration` (VL) models load on the standalone `trl vllm-serve` without
the colocate-only language-model-only patch, so disaggregation works across the catalog. The
**train_gpus>1 ratios (2:1/3:1/2:2)** are now wired via an accelerate-FSDP trainer group (`run_rl`
launcher re-execs the worker under `accelerate launch`); a 2:1 MiniCPM validation run is in flight.

**Ratio validity is model-dependent (TP must divide the attention-head count).** vLLM requires
`num_attention_heads % tensor_parallel_size == 0`, and TP = `inference_gpus`. MiniCPM5-1B has 16
heads → valid inference-GPU counts are {1, 2, 4, 8}; **1:3 is mathematically invalid** (caught now by
a fast-fail guard in `run_rl` that reads the model config before renting/booting, instead of
crashing after a ~20-min server boot). So for this model the practical optimum is **1:2 (1.33×)**;
the next valid step 1:4 (TP=4) needs a 5-GPU node. Per-model valid ratios differ with head count.

**What the numbers say (and two corrections to the original design's assumptions):**

1. **1:1 ≈ colocate (1.02×), NOT a win.** TRL's GRPO server mode is **synchronous** — it generates a
   batch, *then* trains, *then* syncs; there is no GRPOConfig flag for one-step-off overlap. So at
   1:1 the two GPUs run sequentially (generate on the server card while the trainer idles, then
   train while the server idles) and you pay server HTTP + NCCL-sync overhead for no throughput
   gain. The 2% edge is only the trainer no longer time-sharing its card with vLLM. **The speedup
   does NOT come from async overlap** (the original "async rollout / one-step-off" premise below was
   wrong for TRL).

2. **The win is inference THROUGHPUT scaling via tensor-parallel.** 1:2 (TP=2) hits **1.33×**: two
   dedicated inference cards give the rollout server ~2× aggregate HBM bandwidth, and decode on a
   generation-bound step is bandwidth-bound, so generation gets ~2× faster while the trainer runs
   undisturbed on its own card. This is the verl "TP rollout" lever, and it is what makes higher
   inference ratios pay off. **Data-parallel replicas do NOT work here**: vLLM's offline engine
   rejects data-parallel for *dense* models ("Offline data parallel mode is not supported/useful
   for dense models"), so `parallel="tp"` is the default and `parallel="dp"` is reserved for the MoE
   35B-A3B. (Confirmed by a failed DP=2 run before the TP switch.)

Driven by `engine/rollout_bench.py` (`select_rollout_split`) + `engine/disaggregated.py`
(`build_vllm_serve_cmd`, default TP); the worker writes `rollout_split` + `wall_seconds` into
`metrics.json` for each row.

### 6. verl-derived efficiency levers (heavily referenced)

verl (HybridFlow; https://github.com/verl-project/verl) is the reference implementation for
efficient multi-GPU RL. Each lever below maps a verl mechanism to a concrete flash change:

| verl mechanism | What it does in verl | flash adoption |
|---|---|---|
| **3D-HybridEngine** (actor resharding) | reshapes actor weights between the **train** layout (FSDP/Megatron sharding) and the **rollout** layout (vLLM TP) with *zero redundancy* and minimal comm during the train→generate transition | the disaggregated weight-sync step pushes only the resharded delta to the vLLM server each step, not a full state-dict copy |
| **Flexible device mapping / placement** | a `ResourcePool` assigns actor vs rollout to arbitrary GPU sets (colocated **or** split) | `select_rollout_split` is the same idea: `inference_gpus` carves the node into trainer vs vLLM-server device sets |
| **Async rollout (one-step-off)** | generation for the next batch overlaps the current optimizer step; the rollout engine runs continuously | **NOT available in TRL** — TRL `vllm_mode="server"` is synchronous (generate→train→sync, no overlap flag), so this verl lever is unrealized here and 1:1 ≈ colocate (measured 1.02×). The speedup instead comes from **inference TP throughput** (next row), not overlap. A true one-step-off would need a custom trainer loop (future) |
| **Continuous batching in the rollout engine** | vLLM/SGLang server batches across requests, keeping the inference GPU saturated | the dedicated vLLM server (vs colocate's per-step burst then idle) keeps the infer GPU busy |
| **TP rollout + FSDP/Megatron train** | inference is tensor-parallel across its GPUs; training is sharded across the rest | vLLM `tensor_parallel_size = infer_gpus` (**the measured 1.33× win at 1:2**); trainer FSDP across `train_gpus` is the REMAINING piece for train_gpus>1 ratios |
| **Single-controller orchestration** | one process drives the dataflow; workers are stateless | the worker stays the single controller; only the vLLM-serve subprocess is added |

The benchmark's job is to find, per model, the `inference_gpus` split where the inference-TP
throughput win outweighs the extra GPU cost — i.e. the cheapest config at the target tokens/s. (For
a model that fits one card the win is purely TP decode-bandwidth scaling, since TRL gives no
generate/train overlap; a bigger or memory-tight model would add colocate's sleep-reload penalty on
top, widening the gap further.)
