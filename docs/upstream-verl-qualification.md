# Upstream Verl GPU qualification

PR #1420 was exercised on Modal on October 20–21, 2026. These are short integration runs with real training and generation, rather than a throughput benchmark or qualification of every supported workload.

## Runtime and image

The [worker build](https://github.com/freesolo-co/flash/actions/runs/39538783019) published `ghcr.io/freesolo-co/flash-worker:pr1420-upstream-verl-final` at:

```
sha256:d349042a78bd2fdd8cea5dda72640b59e19c3197a7ac2eee5d9526c8183fdb61
```

The image's source revision is `874bf30f0c8845ef148cfe04c227619041225279`. Flash is deployed separately as a wheel. The initial final-image matrix used `ac00d23d5b1043d4f87ae6acf5c495c345201c95` (wheel SHA-256 `2d418acefa6c50347e3d9a5b90db2a36bdc538c29005dce1f9875705a9c29a49`). The GDN GRPO recovery check used the follow-up review fixes at `25aa56620e87b46cb18b23b7f64557fa75f36f06` (wheel SHA-256 `92be81c927df3a9065989e6b8dbfe28029c209318d5dc55b61e9f256748a223e`). The final GPU-tested wheel includes `dev` through `5e6730071bb5ffab037141ce6fd94df9a6e4816d`: Flash revision `09568d90ff66ec05a425172c6152fda355f9b6de`, wheel SHA-256 `c713ca84ddb3a5da894cc847a897228dd6c34a24a2d0ac473fb0bb4b5c463b67`.

Subsequent queue-timeout cleanup and optional SFT tracking lookup fixes are covered by CPU regressions; they do not alter the successful GPU training path.

Inside the running image, provenance checks confirmed official, unmodified `verl-project/verl` at `ddf700725c34bb8966685c8ecf5ca36a2cc54a35`, Torch 2.10.0, Transformers 5.12.1, vLLM 0.19.1, and TransferQueue 0.1.11. No Flash-generated startup module or dependency source rewrite is used. Ubuntu's stock `sitecustomize.py` only installs its optional Apport exception handler.

The initial sandbox had two H100 80GB GPUs. Modal assigned two H200 GPUs to the final-image sandbox; physical UUID checks verified distinct devices. Model revisions were Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca` and Qwen3.5-9B `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

## Completed checks

Each fresh run performed two optimizer steps and saved both checkpoints. Resume runs restored step 1 and continued to step 2 with the same training horizon. Validation required finite, nonzero gradient norms, nonzero LoRA B tensors, and changed adapter tensors between steps. Exports use Flash's production adapter exporter and upstream's native shard merger and LoRA serializer.

| Workload | Hardware | Evidence |
| --- | --- | --- |
| Qwen3-0.6B SFT | 2 × H100 | Distributed fresh training and resume; resumed loss and every final adapter tensor exactly match uninterrupted training. |
| Qwen3.5-9B SFT, text only | 1 × H200 | Reentrant GDN forward/backward, compact checkpoint save/resume/export, and exact resumed adapter equality. All 496 adapter tensors belong to the language model; the compact model shard is 43,854,395 bytes. |
| Qwen3-0.6B GRPO | 1 × H200 | Fresh training and compact checkpoint resume, real vLLM generation and LoRA transfer, whole-batch identity registration, completion-dependent rewards, and adapter export. |
| Qwen3.5-9B GRPO, text only | 1 × H200 | Fresh training and compact checkpoint resume, native GDN generation/backward, language-only adapters, and AOT cache activation. Recovery also exercises the per-episode observation fix. |
| Qwen3.5-9B OPD, text only | 2 × H200 | Fresh training and compact checkpoint resume, real GPU teacher scoring, native GDN rollout/backward, language-only adapter export, and teacher-cycle acknowledgments. Resume produces finite, nonzero gradients and updated adapters; sampled outputs are not bit-identical. |
| Qwen3-0.6B OPD | 2 × H200 | Fresh training and compact checkpoint resume, real teacher scoring, teacher-cycle acknowledgment before optimizer mutation, and adapter export. Fresh and resumed adapter SHA-256 hashes match exactly. |

GRPO and OPD use the production override builders and packaged entrypoints, native Ray/TransferQueue scheduling, and the minimum configured seven-CPU Ray budget for one student GPU. The harness provides local HTTP endpoints for reward/teacher coordination. The OPD teacher is a real GPU model using the same pinned base with temperature-adjusted logits and the production shifted-group encoder. This tests the child protocol and training objective; it does not exercise the hosted teacher service or billing.

FlashInfer GDN warmup compiled and staged a valid AOT library in the sandbox cache. A separate interpreter selected and loaded it, and a subsequent training worker selected it through the explicit engine hook. Cache selection no longer requires interpreter startup or import-finder hooks.

## Fixes discovered during qualification

- Reentrant SFT checkpointing must follow the runtime option even when the native model flag is deliberately disabled. OPD now receives the same GDN/MoE policy.
- GRPO's storage-unit count and Ray CPU reservation must describe the same queue topology.
- TransferQueue 0.1.11 is required by the selected upstream API. Flash installs bare Verl alongside its explicit CUDA stack; upstream's optional vLLM extra selects a different stack.
- Text-only parquet nulls can arrive as pandas null scalars. The dataset handles them after Arrow conversion.
- Tracking finalization must accept upstream's destructor call without overwriting an already recorded failure.
- The custom rollout server must decode `FlashModelConfig`, and its replica actor names must match the native adapter's lookup prefix.
- Queue cleanup starts only after initialization succeeds, so an initialization timeout cannot race `close()` against the blocked initialization thread.
- Rollout version observations are scoped per episode with async context isolation. Multimodal OPD carries its explicit vision-gradient flag, and W&B link extraction remains non-fatal for all three algorithms.
- Compact checkpoints from current `dev` use scoped model hooks around explicit engine save/load methods. Their adapter export must not require a full-model export; the Flash merger subclass delegates shard reconstruction and adapter serialization to upstream.

The replay-buffer shutdown review suggestion does not apply to this pin: upstream `ReplayBuffer.sample()` polls synchronously, and the class has neither a background synchronization thread nor a `close()` API. The task runner drains the actual dump executor before closing TransferQueue.

The actual pinned-upstream CPU contract suite passed 25 tests, including queue initialization timeout cleanup. The offline runtime policy suite passed 21 tests, including optional tracking lookups for all algorithms. [CI](https://github.com/freesolo-co/flash/actions/runs/39544135510) passed the offline suite, lint, formatting, and size/layout checks on Python 3.11 and 3.12 at the GPU-tested revision. The final failure-path fixes are also covered by the CPU suites above and the PR’s latest checks.

## Coverage limits

These runs do not establish long-duration stability, performance parity with the former fork, multi-node recovery, multimodal image/video training, multi-turn bridge behavior, or warm-start/KL-reference behavior on GPUs. The offline tests cover several of those policies, but do not replace their live qualification. Production API admission, billing, artifact publication, and hosted-teacher failure handling are outside this child-runtime harness.
