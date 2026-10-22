# Upstream Verl GPU qualification

PR #1420 was exercised on Modal on October 20–21, 2026 (Pacific time). The qualification uses real optimizer updates, generation, GPU teacher scoring, checkpoint recovery, and adapter export. It is an integration/stability exercise, not a throughput or model-quality benchmark.

## Runtime and image

The [worker build](https://github.com/freesolo-co/flash/actions/runs/39538783019) published `ghcr.io/freesolo-co/flash-worker:pr1420-upstream-verl-final` at:

```
sha256:d349042a78bd2fdd8cea5dda72640b59e19c3197a7ac2eee5d9526c8183fdb61
```

The image's source revision is `874bf30f0c8845ef148cfe04c227619041225279`. Flash is deployed separately as a wheel. The initial final-image matrix used `ac00d23d5b1043d4f87ae6acf5c495c345201c95` (wheel SHA-256 `2d418acefa6c50347e3d9a5b90db2a36bdc538c29005dce1f9875705a9c29a49`). The GDN GRPO recovery check used the follow-up review fixes at `25aa56620e87b46cb18b23b7f64557fa75f36f06` (wheel SHA-256 `92be81c927df3a9065989e6b8dbfe28029c209318d5dc55b61e9f256748a223e`). The October 20 GPU-tested wheel includes `dev` through `5e6730071bb5ffab037141ce6fd94df9a6e4816d`: Flash revision `09568d90ff66ec05a425172c6152fda355f9b6de`, wheel SHA-256 `c713ca84ddb3a5da894cc847a897228dd6c34a24a2d0ac473fb0bb4b5c463b67`.

Subsequent queue-timeout cleanup and optional SFT tracking lookup fixes are covered by CPU regressions; they do not alter the successful GPU training path.

Inside the running image, provenance checks confirmed official, unmodified `verl-project/verl` at `ddf700725c34bb8966685c8ecf5ca36a2cc54a35`, Torch 2.10.0, Transformers 5.12.1, vLLM 0.19.1, and TransferQueue 0.1.11. No Flash-generated startup module or dependency source rewrite is used. Ubuntu's stock `sitecustomize.py` only installs its optional Apport exception handler.

The initial sandbox had two H100 80GB GPUs. Modal assigned two H200 GPUs to the final-image sandbox; physical UUID checks verified distinct devices. Model revisions were Qwen3-0.6B `c1899de289a04d12100db370d81485cdf75e47ca` and Qwen3.5-9B `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

## Initial text-only matrix

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

## Extended image and multi-turn matrix (October 21)

The extended matrix used actual NVIDIA B200 GPUs (178.34 GiB each), two per sandbox, and the following immutable model revisions. Large-model names follow Flash's current catalog.

| Model | Revision |
| --- | --- |
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Qwen/Qwen3.6-35B-A3B | `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` |

| Workload | Completed evidence |
| --- | --- |
| 9B image SFT, GRPO, OPD | Two fresh optimizer steps per algorithm; finite nonzero gradients, changed adapter tensors, and nonzero vision-adapter B tensors. OPD uses real GPU teacher logits with image pixels and multimodal token types. |
| 27B image SFT, GRPO, OPD | Two fresh optimizer steps per algorithm, native GDN forward/backward and rollout, complete adapter export and production provenance validation. GRPO uses a creative image prompt so sampled completions produce a measurable advantage. |
| 35B MoE text and image SFT | Two steps on two GPU ranks; complete fused-expert and image-adapter export. All 80 fused-expert B tensors in the text adapter have nonzero norms; this does not assert that every individual expert was activated. |
| 0.6B multi-turn GRPO, warm start and KL | Three steps, 12 two-turn sessions, all 12 closed. KL starts at zero and becomes positive after optimizer updates against the frozen warm-start reference. |
| 35B MoE image/multi-turn GRPO, warm start and KL | Three steps, 12 two-turn sessions, all 12 closed. Gradient norms 0.138151, 0.324438, 0.657400; KL 0, 0.097543, 0.226996. Complete adapter export passes production validation. |

The 35B MoE uses exact fused-expert targets (`mlp.experts.gate_up_proj` and `mlp.experts.down_proj`) and resident rollout because this model does not support the sleep path. B200 execution selects FlashInfer decode attention and Torch SDPA vision attention through the production capability resolver.

### Additional fixes exposed by these runs

- Multi-turn prompt construction now calls the pinned upstream Continuous Token API (`ct_build_initial_tokens`) instead of the removed chat-template entrypoint.
- Flash's worker boundary rebuilds multimodal position tensors with the public PyTorch jagged-tensor constructor. TransferQueue's equal-length packing can select the mRoPE axis as ragged; merely relabeling that axis leaves incompatible offsets. Contract tests cover equal/unequal sequence lengths, native layouts, and repeated normalization. Upstream source remains untouched.
- OPD multi-turn outputs explicitly carry a zero scalar reward because their teacher scores are already attached. This prevents redundant native reward computation and its incompatible padding of variable-length multimodal turns.
- Optional tracking finalization cannot mask a training result or prevent dump-executor cleanup.
- Adapter-only export no longer rejects a valid export using the size of full-model input shards. Actual disk/quota errors still retain their specific diagnosis and cleanup behavior.

Harness corrections are recorded separately from product fixes: image teacher inputs needed multimodal token types; export verification must ignore pruned checkpoint directories; the small OPD fixture needs multiple epochs; and skipped/truncated OPD turns need masked teacher rows, as the production bridge already provides. Failed attempts remain in the evidence and do not count as passed checks.

### Two-hour concurrent checks

The duration starts at the first observed optimizer update, excluding model download, initialization, and kernel warmup. Each run must continue for at least 7,200 seconds, reach a newly durable checkpoint, stop its training process, restore that checkpoint, complete two additional optimizer steps, and export a finite, nonzero adapter. The three algorithms run concurrently in independent sandboxes.

| Algorithm | Workload | Active duration | Durable checkpoint | Recovered steps | Result |
| --- | --- | --- | --- | --- | --- |
| SFT | 27B images, warm start, two GPU ranks | 7,205.64 seconds | 6,650 | 6,651–6,652 | Passed; 1,212 exported adapter tensors, with changes in all vision and language tensors. |
| GRPO | 35B MoE images, two-turn episodes, warm start, frozen KL reference | 7,207.63 seconds | 850 | 851–852 | Passed; 3,412 scored two-turn episodes and 1,080 exported adapter tensors. |
| OPD | 35B MoE images, two-turn episodes, warm start, real GPU teacher | 7,502.98 seconds | 900 | 901–902 | Passed; 3,610 scored two-turn episodes and 1,080 exported adapter tensors. |

All three recovered exports contain finite tensors and changed vision/language adapters compared with their warm-start sources; both MoE runs also changed fused-expert adapters. The GRPO reference stayed frozen: KL began at zero, remained finite, and was positive after updates and recovery. OPD ran past two hours to finish its next scheduled checkpoint.

The fixture contains 16 synthetic 112×112 images/prompts, rank-8 LoRA, and a 1,024-token context. SFT and GRPO use learning rate 1e-4; OPD's sustained check uses 1e-6 to avoid rapidly overfitting the small fixture. Recovery is a process restart within the same sandbox, not a full-machine preemption test.

### Updated source and image provenance

The branch includes `dev` through `f6cf4611`; package metadata is 1.3.7. The final `dev` merge changes deployment recovery and cache-bake validation, with no changes to the GPU-tested trainer or adapter-export code. The GPU wheels below were built as 1.3.6 (or 1.3.3 for the earlier SFT run). The merged GRPO/OPD wheel is `6776eb505afee956c4020d6d569852374666ee87` (SHA-256 `5d64ff04a79dd785122c4f15d3f1ddbddb54f88d22ad18b2cca84697ecafc082`). The SFT soak started on `fef822339bff93aca205483c2fe667c1f67e9d0a` (SHA-256 `7f650dfce93db793bd77807a91caf43ccfbcf2cfd5c5de6aa0493b7540f0284a`). Later changes to the shared RL worker and multi-turn path do not alter that SFT child execution path. Final-image checks use the merged wheel plus the final review fixes.

The [extended worker build](https://github.com/freesolo-co/flash/actions/runs/39680907453) passed at source `6776eb50`, publishing test tag `pr1420-upstream-verl-extended`:

```
sha256:34c84092f87b9ae43457620a8e5c0dc2e3dc30fdd5de5b4300c1eeb384e9a066
```

The long runs use the preceding `d349042a…` image with separately installed Flash wheels. On the rebuilt image, 27B image SFT passed two warm-start optimizer steps; 35B MoE image/multi-turn GRPO and OPD each passed three. Both RL runs completed and closed 12 two-turn sessions. GRPO KL was 0, 0.127626, and 0.041550. All runs exported valid adapters. Comparisons against the warm-start adapters confirmed changes in all 220 vision tensors and all language tensors; both 35B runs also changed all 160 fused-expert adapter tensors.

Modal discarded the final-image OPD command's exit-status record after its retention window. Its native training had exited successfully and completed every harness assertion before writing the persisted result. The result, bridge log, exports, and weight comparison were verified separately; this transport error is retained in the evidence. No production image tag was promoted. The trainer interpreter has PEFT 0.21.2; the base interpreter used by the local teacher/export harness has PEFT 0.20.0. Both have Torch 2.10.0, Transformers 5.12.1 and vLLM 0.19.1; the trainer uses TransferQueue 0.1.11.

All 357 installed Verl Python files were hashed and compared with the pristine official checkout at `ddf700725c34bb8966685c8ecf5ca36a2cc54a35`: zero differences. The rebuilt image passed all **44 pinned-upstream CPU contracts**, including tracking failures on successful and failed training paths. Its 357 installed Verl Python files also match the official checkout exactly. [Python 3.11/3.12 CI](https://github.com/freesolo-co/flash/actions/runs/39692836499) passed after integrating the latest `dev` changes and aligning their deployment-state tests, at revision `406ea3b5`. The final wheel SHA-256 is `8ada64fb917ed716b68c13b14c77af8786e1a1976d98373b129d53533cdc1309`.

## Evidence and cleanup

The local `flash-pr1420-validation-2026-10-21` artifact set contains run configurations, immutable model pins, native logs and metrics, wheel/source hashes, adapter comparisons, image-build logs, and cleanup receipts. Failed attempts are retained separately from passing results. All eight owned Modal sandboxes were confirmed stopped, the qualification app was stopped, and its temporary model/checkpoint volume was deleted after collection. Published GHCR test images remain available; no production tag was promoted.

## Coverage limits

Image inputs are tested; video and audio are not. These runs do not establish performance parity with the former fork, model-quality convergence, long-context limits, multi-node or machine-preemption recovery, production API admission, billing, artifact publication, or hosted-teacher failure handling. The local bridge uses real GPU teacher computation and the production scoring representation, but is not the hosted service.
