# Packaged trainer runtime

SFT, GRPO, and OPD start through ordinary Python modules in the verified Flash package:

- `flash.engine.worker.train.sft.child.entry` (launched by `torchrun`)
- `flash.engine.worker.train.rl.child.entry` (owns the Ray driver and metric journal reader)
- `flash.engine.worker.train.opd.child.entry` (owns the Ray driver and transfer-queue trainer)

Each receives `--flash-config <launch.json>` followed by Hydra overrides. The JSON file contains nonsecret runtime options, including checkpoint schedules and model policy. Credentials remain in the existing filtered environment/bridge configuration. Scratch directories contain data and readiness records; they contain no generated Python. Children use the verified package root on `PYTHONPATH` and do not inherit external-module plugin selection.

`core.child.launch` validates the environment and fork interface version before constructing a trainer. `FlashFSDPEngine` owns model initialization, exact LoRA targets, checkpointing, the SFT scheduler, the frozen KL reference, and the OPD optimizer pre-step acknowledgment. Each trainer owns tracking and checkpoint policy; GRPO owns per-turn credit and rollout identities; OPD owns teacher cycles and bounded replay. A readiness record is published only after trainer/worker initialization succeeds. The parent rejects progress or successful exit without that record.

Model-specific behavior uses PyTorch forward hooks on constructed model instances. Optimizer notification uses `register_step_pre_hook`. Backend selection uses the pinned FLA dispatcher's hardware/shape checks, including FlashQLA selection, rather than replacing a kernel function. The fork's native vLLM server rejects generation if a required LoRA adapter is absent.

The small opt-in construction contract lives in [the company Verl fork](https://github.com/freesolo-co/verl/pull/6), with `verl.runtime_interfaces.API_VERSION == 2`. Flash pins the reachable fork commit in both worker installation paths. Flash selects the fork's explicit Qwen dense/MoE model factory, bypassing its legacy model/attention patch dispatcher. This removes Flash's import interception, copied-source plugins, dependency-method replacements, and `sitecustomize` startup path.

CPU tests cover launch data, package imports, exact LoRA targeting and isolation, SFT data order/partial batches/boundary resume, checkpoint publication backpressure, GDN packed-boundary hooks, tracking ownership, GRPO identity and credit calculations, and OPD failure/replay/mutation ordering. Fork tests also exercise real tiny Transformers 5.12.1 dense/MoE models, compare text and multimodal outputs and gradients against the previous implementation, and round-trip native checkpoints. Rollout tests cover instance-local vocabulary hooks, tensor-adapter ownership, complete bucket assembly, and native cache replacement/validation ordering.

The model factory preserves fused torch/triton loss selection, native attention, the vision offload behavior, and exact module names. It explicitly rejects sequence parallelism and unsupported model/kernel choices. The fork declares its supported Transformers 5.12 line and vLLM 0.19.1 rather than carrying older API fallbacks.

The fork's `TensorLoRAWorkerManager` owns tensor decoding and shares the native per-worker adapter cache; it does not replace vLLM class methods. Vocabulary filtering is a PyTorch output hook before logprob normalization. Remaining compatibility code is explicit: legacy recipes for other architectures, optional backend/quantization paths, the conditional FSDP/ABC context, and the MoE Parameter weight-loader repair. The last repair is still needed when vLLM post-load transforms discard loader attributes; it has not been removed solely because its comment names an old version.

Before marking the PR ready, qualify all three algorithms on the pinned worker image: a fresh run and checkpoint resume, authored checkpoint publication, warm-start LoRA and KL reference, GRPO single-/multi-turn identities and final metrics, OPD scoring failure/no-signal replacement/first-mutation acknowledgment, and multi-rank initialization. Include text and supported multimodal/GDN workloads. CPU tests and module import smoke checks do not establish CUDA kernel, FSDP, Ray, or vLLM compatibility.
