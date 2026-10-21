# Packaged trainer runtime

SFT, GRPO, and OPD launch ordinary Python modules from the verified Flash package:

- `flash.engine.worker.train.sft.child.entry` (torchrun)
- `flash.engine.worker.train.rl.child.entry` (Ray and synchronous V1 PPO)
- `flash.engine.worker.train.opd.child.entry` (Ray, synchronous V1 PPO, remote teacher bridge)

Each receives `--flash-config <launch.json>` and Hydra overrides. The JSON contains runtime options, checkpoint schedules, and model policy. Credentials remain in the filtered environment and bridge configuration. Scratch directories contain data and readiness records, with no generated Python or `sitecustomize` startup hook.

## Exact upstream dependency

Both worker installation paths pin unmodified [`verl-project/verl` at `ddf700725c34bb8966685c8ecf5ca36a2cc54a35`](https://github.com/verl-project/verl/commit/ddf700725c34bb8966685c8ecf5ca36a2cc54a35). Its distribution version is `0.10.0.dev0`; the immutable commit, rather than that development version string, identifies the source. This revision supports Flash's Transformers 5.12.1 dependency. Flash also pins vLLM 0.19.1 and TransferQueue 0.1.11 in its worker environments.

The installer uses the bare upstream package plus Flash's explicit CUDA 12.8 / Torch 2.10 stack. Upstream's `[vllm]` extra selects its complete CUDA 13 / Torch 2.13 environment and is deliberately not combined with Flash's worker image.

Before constructing a trainer, `core.child.upstream` checks installed VCS metadata for the official repository and exact commit. An installation from a fork, another revision, or a wheel without VCS provenance fails with a rebuild instruction. `scripts/check_verl_pin_exists.py` checks the installation pins and fetches the commit from upstream.

No change or PR to Verl is required. Flash no longer depends on the company fork's runtime interface, model factory, trainer callbacks, or worker subclasses. Upstream's own compatibility implementations remain upstream-owned and unmodified; this does not claim that upstream itself contains no patches.

## Flash-owned extensions

`FlashModelConfig`, `FlashActorConfig`, and `FlashEngineConfig` are selected through Hydra. The engine registers distinct `flash_fsdp` / `flash_fsdp2` names and delegates model construction to upstream. Flash retains exact language LoRA targeting, frozen KL references, its SFT scheduler, instance-local GDN/vision hooks, and optimizer pre-step acknowledgments. These use subclasses, PyTorch hooks, and owned configuration rather than replacing dependency methods.

The SFT trainer owns its dataset order, partial batches, boundary resume, checkpoint publication backpressure, and a small training loop that calls upstream's training client. Compact checkpoint hooks are scoped to the model during explicit engine save/load calls. Adapter export subclasses the upstream merger and uses its LoRA serializer without constructing a full base-model export. The loop explicitly closes tracking with the actual success or failure outcome.

GRPO and OPD share the upstream V1 synchronous trainer lifecycle and TransferQueue replay implementation. Flash bounds replay waiting and rejects failed prompt groups before consuming them. The common task runner initializes TransferQueue, trainers, and agent managers, publishes readiness, then runs training and closes tracking and queue resources.

The GRPO manager registers the complete expected identity set before dispatch. Agents carry those identities into scoring and collect model-version observations across turns; the trainer applies per-turn credit. Truncation masking is a Flash runtime option, not an extra upstream dataclass field. Trainability is checked before advantages and again before actor updates because rollout correction can reject additional tokens.

OPD agents score through the existing remote teacher bridge. The trainer does not allocate a local teacher GPU pool. Teacher facts travel through upstream's native agent output serialization; no-signal batches retain bounded replacement and teacher-cycle acknowledgment before mutation. Native upstream policy forwards remain enabled; the former fork's dead-compute optimization is not carried over.

`flash_vllm` is a separate rollout registration using upstream's registry and replica/server extension points. Its server preserves the exact engine finish reason and refuses generation when the training LoRA adapter is missing. Adapter transfer, model construction, and vLLM compatibility remain upstream-owned.

Image-baked FlashInfer libraries are selected explicitly after GPU device initialization by the configured `FlashGPUWorker` subclass and the Flash training engine. Cache identity checks cover the package, Torch/CUDA/Python versions, and GPU architecture; a missing or incompatible bake retains native JIT behavior. No interpreter startup or import-finder hook is installed. Kernel baking calls the same native FLA dispatcher as training.

The four agent classes are declared in packaged `core/child/agents.yaml`. Upstream's unmodified TransferQueue worker instantiates them. A readiness record is published only after trainer and rollout initialization; the parent rejects progress or successful exit without it.

## Validation

The offline suite covers launch policy, source verification, tracking outcome, SFT data order/resume/checkpointing, exact LoRA targeting, GDN boundaries, GRPO identities and credit, and OPD failure/replay/mutation ordering. `tests/test_upstream_verl_contract.py` additionally runs against the actual pinned package: all three production configurations, trainer and agent imports, OPD pool ownership, native replay, ragged GRPO credit serialization, whole-batch identity registration, exact truncation masking, OPD teacher-fact serialization, and task-runner cleanup. It skips when Verl is not installed.

Run that contract suite in an environment with the exact VCS requirement from `backend_common.py`, Transformers 5.12.1, and the worker dependencies installed. The default development environment intentionally does not install the GPU trainer stack.

The rebuilt image and live Modal results are recorded in [Upstream Verl GPU qualification](upstream-verl-qualification.md). Extend that smoke coverage before treating the entire workload matrix as qualified: fresh training and checkpoint resume, authored checkpoint publication, warm-start LoRA and KL reference, GRPO single-/multi-turn identities and metrics, OPD scoring failures/no-signal replacement/first-mutation acknowledgment, and multi-rank startup. Include text and supported multimodal/GDN workloads. CPU tests do not establish CUDA kernel, FSDP, distributed Ray, vLLM adapter transfer, or end-to-end training compatibility.
