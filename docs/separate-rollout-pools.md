# Separate rollout GPU pilot

This opt-in GRPO prototype assigns vLLM to dedicated GPUs and lets its replica
count and tensor parallelism differ from the trainer. It preserves synchronous
generate → reward → update → publish execution. The default remains colocated.

Status: the full offline CPU suite and pinned upstream Verl contracts pass.
The optional CuPy image layer builds. Live GPU validation is recorded below;
no production-scale performance or learning-quality improvement is claimed.

The first stage partitions **one existing, single-node allocation**. It does
not provision separate Modal sandboxes, resize a running pool, share replicas
across customers, or overlap generation with training. Those are subsequent
steps toward an elastic service, not capabilities of this pilot.

## Capacity and launch

Use a dense LoRA model already supported by Flash, such as Qwen3.5-9B, on text tasks,
and explicitly choose the total GPU count. Ensure the model and the training
batch fit the smaller training partition; the existing provider planner still
sizes the original colocated job and does not credit the new memory layout.
Multimodal tasks and fused-expert adapters are excluded from the pilot. Existing
GDN capability checks still apply to hybrid-attention models; rollout replicas
remain resident, as required by Flash's current Qwen configuration.

Build the regular worker from this checkout, then build the optional layer:

```sh
docker build -f deploy/rollout-pilot/Dockerfile \
  --build-arg WORKER_IMAGE=flash-worker:rollout-source \
  -t flash-worker:rollout-pilot .
```

This adds `cupy-cuda12x==13.6.0` to the **Verl interpreter**, required by the
remote NCCL backend. [CuPy publishes this wheel for CUDA 12.x and Python 3.12](https://pypi.org/project/cupy-cuda12x/13.6.0/).
The pilot image also sets `NCCL_CUMEM_HOST_ENABLE=0` consistently for all
processes. The Modal A10G smoke run failed during NCCL shared-memory attachment
without it and passed initial publication with it. This selects NCCL's legacy
host-allocation path; see [NVIDIA's NCCL troubleshooting guidance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html).
The normal worker image and dependency resolver are unchanged. A missing NCCL
backend fails before Ray launches training actors; the worker does not install
dependencies at runtime.

Supply this environment variable to the **GPU worker process**, alongside its
normal prepared job spec, credentials, environment and artifact configuration:

```sh
export FLASH_GRPO_ROLLOUT_POOL='{"gpus":2,"tensor_parallel":1,"gpu_memory_utilization":0.8}'
python -m flash.engine.support.worker_entrypoint
```

With `gpu.count=4`, this gives two training GPUs and two one-GPU vLLM replicas.
With `gpus=2,tensor_parallel=2`, it gives one two-GPU replica. At least one GPU
must remain for training. Flash still reduces trainer ranks if the number of
sequences in an update does not divide across all available training ranks.
Resume checkpoint validation uses that actual training width, not the total
rented count. The same pool setting must be supplied on retries.

This is an operator worker setting, not a new public TOML or CLI option.
Exporting it in a local client shell does **not** automatically forward it to a
remote worker. Configure the pilot sandbox's environment explicitly. Direct
child launches carry the equivalent `rollout_pool` object in `--flash-config`
and must use the matching overrides from `build_verl_overrides`.

Run cost continues to include the entire provider allocation, including idle
GPUs. No pricing or admission policy is changed. This synchronous first stage
can improve generation time without improving dollars per training step.

## Execution and correctness

The trainer uses the upstream pure Actor role; it creates no vLLM replicas on
training GPUs. Standalone replicas reserve separate Ray placement groups and
load the same local base-model snapshot. Generation GPUs retain their base
weights between updates. Training checkpoint loading completes before the
first adapter publication.

Flash sends the training model's actual PEFT configuration to every receiving
worker, including warm-start adapter settings. Its sender requests only LoRA
tensors, with layered collection. The receiver forwards the PEFT metadata and
`base_sync_done=True` to vLLM's adapter loader. This is required because the
[pinned upstream remote checkpoint worker](https://github.com/verl-project/verl/blob/ddf700725c34bb8966685c8ecf5ca36a2cc54a35/verl/checkpoint_engine/base.py)
does not carry PEFT metadata through its generic weight-transfer path.

The checkpoint manager gates requests, broadcasts the adapter with NCCL,
clears KV state and waits for all replicas. Only then can the next synchronous
batch begin. The initial publication must succeed before worker readiness.
A publication error aborts the attempt; no subset of replicas is silently
accepted and no policy-lag allowance is introduced. Existing adapter-presence
checks, rollout log probabilities, identity bookkeeping, reward handling,
multi-turn policy and GRPO loss remain in place.

## Evidence to collect on GPUs

For an infrastructure smoke test inside a disposable three-GPU pilot worker:

```sh
/opt/verl-venv/bin/python scripts/check_rollout_pool_gpu.py \
  --output /tmp/rollout-pool-check
```

The script creates a tiny random Qwen3 model and synthetic tokenizer locally,
runs two updates with one training GPU and two rollout replicas, then resumes
for one more update. It checks nonzero gradients and completed publication
versions. It downloads no model and writes no artifacts to shared storage.
The fixture uses the standard loss path: pinned Verl's fused Qwen3 forward
directly reads the FSDP2-sharded output weight and fails on this fixture.
This does not qualify fused-loss kernels, production model sizes,
tensor-parallel replicas, failure injection, or learning quality.
Live smoke result (2026-10-22): **passed on three Modal A10G GPUs**, with
one training rank and two TP=1 rollout replicas. Fresh steps 1 and 2 and resumed
step 3 had nonzero gradients and matching completed publication versions.
Resume loaded adapter weights, optimizer, RNG and scheduler state before
initial publication. The sandbox was terminated after evidence collection.

Local validation also passed the full offline suite (10,764 tests, 193 skips),
the separate wall-clock suite (25 tests, 2 skips), and 96 pinned-Verl contracts.
The normal CPU environment skips optional GPU/upstream dependencies; the
pinned-Verl contracts and live run exercise the new pool integration separately. Required CI
runs the offline suite on Python 3.11 and 3.12.

Start with two GPUs: one trainer, one replica, several updates and a recovery
checkpoint. Then repeat with more one-GPU replicas at the same training width.
Keep the model revision, seed, prompts, generation budget and optimizer recipe
fixed. Compare against the colocated baseline as well as against the smaller
separate pool.

Check that training and rollout processes occupy disjoint GPU UUIDs; every
replica generates using the freshly published adapter after initial load,
warm-start and resume; and a replica killed during publication stops the run
without emitting a successful next update. Verify adapter tensors reach vLLM
unchanged, then compare rollout log probabilities and reward/learning curves.
Multiple replicas need not produce bit-identical stochastic trajectories.

The metric journal and stdout include:

| Metric | Meaning |
| --- | --- |
| `timing_s/gen` | Time waiting for the rollout batch, including agent/reward work in that path |
| `timing_s/update_actor` | Actor update duration |
| `timing_s/update_weights` | Publication duration |
| `timing_s/step` | Full step duration |
| `rollout_pool/gpus` | Reserved rollout GPUs |
| `rollout_pool/replicas` | Rollout replica count |
| `rollout_pool/training_gpus` | Actual trainer world size |
| `rollout_pool/published_step` | Completed publication version |

Run notes retain the pool configuration and training GPU count. Throughput
normalization includes both active pools. Measure GPU utilization, peak VRAM,
tokens/sec, end-to-end step latency, startup time and total billed GPU-seconds
per update. Publish no savings claim until the latter improves at comparable
learning quality.

## Local validation and subsequent stages

```sh
python -m pytest tests/test_rollout_pool.py tests/test_rl_train.py tests/test_verl_child_plugins.py
# In a CPU environment with Flash's exact upstream Verl pin installed:
python -m pytest tests/test_rollout_pool_upstream.py tests/test_upstream_verl_contract.py
```

The tests cover admission, independent GPU/TP sizing, native Hydra validation,
Actor-only placement, publication after checkpoint restoration, adapter-only
export, metadata/version forwarding, transfer failures, and total GPU counting.
They substitute transport and GPU endpoints; they do not establish CUDA safety.

After GPU validation, independent cloud allocations require authenticated
trainer-to-pool transport, distribution of the pinned base snapshot, remotely
reachable reward/environment services, provisioning/reconciliation, deadlines
and cancellation across both allocations, and combined resource accounting.
Then add scaling at publication barriers: drain retiring replicas, publish the
current version to newcomers before routing, and bound pending trajectories.
Asynchronous learning remains a separate change with its own quality tests.
