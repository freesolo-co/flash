"""Trainer:inference GPU-split selection + the ratio-benchmark grid for disaggregated GRPO.

Pure logic (no torch / no provisioning): given a node's GPU ``count`` and the number of
``inference_gpus`` to dedicate to the vLLM rollout server, decide the rollout MODE and the
``CUDA_VISIBLE_DEVICES`` split for trainer vs inference. ``ratio_grid`` enumerates the *reasonable*
trainer:inference splits within a node for the benchmark sweep (colocate, 1:1, 1:2, 2:1, ...) —
deliberately excluding absurd splits (e.g. 7 train : 1 infer).

The live worker (``engine.worker.run_rl``) consumes :func:`select_rollout_split` to launch
``trl vllm-serve`` on the inference devices and the FSDP trainer on the rest; this module stays
GPU-free so the split math + grid are unit-testable on CPU.

verl ref (3D-HybridEngine / flexible device mapping): https://github.com/verl-project/verl
"""

from __future__ import annotations

from dataclasses import dataclass

# Don't dedicate more than this fraction of a node to one side — keeps the sweep to "reasonable"
# splits. A 3:1 (or 1:3) imbalance is the practical ceiling; beyond that the small side starves.
_MAX_RATIO = 3


@dataclass(frozen=True)
class RolloutSplit:
    """How a node's GPUs are partitioned for one GRPO config."""

    mode: str  # "colocate" (vLLM shares the trainer GPU) | "disaggregated" (separate infer GPUs)
    total_gpus: int
    train_gpus: int
    infer_gpus: int
    train_devices: tuple[int, ...]
    infer_devices: tuple[int, ...]

    @property
    def label(self) -> str:
        if self.mode == "colocate":
            return "colocate"
        return f"{self.train_gpus}:{self.infer_gpus}"


def select_rollout_split(total_gpus: int, inference_gpus: int) -> RolloutSplit:
    """Partition ``total_gpus`` into a trainer set + an inference (vLLM-server) set.

    ``inference_gpus == 0`` → colocate (the current single-process TRL path; vLLM shares device 0).
    ``inference_gpus > 0``  → disaggregated: the FIRST ``inference_gpus`` devices serve vLLM, the
    rest train. The vLLM server is pinned to device 0 deliberately: vLLM's model-inspection probe
    queries NVML by the *physical* device id of its first visible card, and NVML (which respects
    CUDA_VISIBLE_DEVICES) only exposes the restricted set — so a server pinned to a non-zero device
    (e.g. CVD="1") makes vLLM query NVML index 1 in a 1-device view → NVMLError_InvalidArgument
    ("architectures failed to be inspected"). Putting inference on device 0 keeps that query at the
    always-valid index 0. The trainer (in-process torch, no vLLM inspection) handles a non-zero CVD
    fine. Raises on an impossible split so a bad config fails fast at setup rather than mid-run.
    """
    if total_gpus < 1:
        raise ValueError(f"total_gpus must be >= 1, got {total_gpus}")
    if inference_gpus < 0:
        raise ValueError(f"inference_gpus must be >= 0, got {inference_gpus}")
    if inference_gpus == 0:
        return RolloutSplit(
            mode="colocate",
            total_gpus=total_gpus,
            train_gpus=total_gpus,
            infer_gpus=0,
            train_devices=tuple(range(total_gpus)),
            infer_devices=(),
        )
    if inference_gpus >= total_gpus:
        raise ValueError(
            f"inference_gpus ({inference_gpus}) must be < total_gpus ({total_gpus}); "
            "at least one GPU must train"
        )
    train_gpus = total_gpus - inference_gpus
    return RolloutSplit(
        mode="disaggregated",
        total_gpus=total_gpus,
        train_gpus=train_gpus,
        infer_gpus=inference_gpus,
        # Inference on the FIRST devices (server gets device 0 → NVML index 0, always valid);
        # trainer on the rest.
        train_devices=tuple(range(inference_gpus, total_gpus)),
        infer_devices=tuple(range(inference_gpus)),
    )


def validate_disaggregated_requirement(
    *,
    requires_disaggregated: bool,
    algorithm: str,
    inference_gpus: int,
    single_trainer_only: bool = False,
    gpu_count: int | None = None,
) -> None:
    """Reject an unfittable disaggregated GRPO topology at SUBMIT time (before renting).

    A ``requires_disaggregated`` model (e.g. Qwen3.6-35B-A3B) OOMs when the trainer and the vLLM
    rollout share one GPU, so its GRPO runs must dedicate inference GPUs (``inference_gpus>0`` on a
    multi-GPU node). SFT has no rollout engine and is unaffected.

    A ``single_trainer_only`` model additionally cannot be REPLICATED across multiple trainer cards:
    the disaggregated trainer is plain DDP (TRL's per-step LoRA merge breaks under FSDP sharding), so
    >1 trainer card would load the whole policy once per rank — host-RAM/OOM for a model this large.
    Reject any multi-trainer ratio (train_gpus = gpu_count - inference_gpus > 1) so a 2:2 / 2:1 split
    fails here instead of crashing the DDP launch on a paid multi-GPU node. Only 1:N is allowed.

    Raising here fails a bad config at submit instead of mid-run on a paid GPU.
    """
    algo = (algorithm or "").lower()
    if requires_disaggregated and algo == "grpo" and inference_gpus <= 0:
        raise ValueError(
            "this model requires the disaggregated GRPO path: set [train].inference_gpus>0 on a "
            "multi-GPU node ([gpu] count = train_gpus + inference_gpus). Colocated GRPO OOMs for it."
        )
    if (
        single_trainer_only
        and algo == "grpo"
        and inference_gpus > 0
        and gpu_count is not None
        and (gpu_count - inference_gpus) > 1
    ):
        train_gpus = gpu_count - inference_gpus
        raise ValueError(
            f"this model only supports a SINGLE-trainer disaggregated split (1:N), but [gpu] count "
            f"({gpu_count}) - [train].inference_gpus ({inference_gpus}) = {train_gpus} trainer GPUs. "
            f"The disaggregated trainer replicates the whole policy on every trainer card (plain DDP), "
            f"which OOMs for a model this large. Set [gpu] count = inference_gpus + 1 (e.g. a 1:1 or "
            f"1:2 split) — one trainer card, the rest for the rollout server."
        )


def ratio_grid(max_gpus: int = 4) -> list[RolloutSplit]:
    """The reasonable (train:infer) configs to benchmark, in a sensible order.

    Always starts with ``colocate`` (1 GPU, the baseline), then the 2..``max_gpus`` node splits with
    both sides >= 1 and the imbalance bounded by ``_MAX_RATIO`` (so no 7:1). Ordered by total GPUs,
    then by infer count, so the table reads colocate → 1:1 → 1:2 → 2:1 → 3:1 → ...
    """
    grid = [select_rollout_split(1, 0)]  # colocate baseline
    seen = {grid[0].label}
    for total in range(2, max_gpus + 1):
        for infer in range(1, total):
            train = total - infer
            if max(train, infer) / min(train, infer) > _MAX_RATIO:
                continue
            split = select_rollout_split(total, infer)
            if split.label not in seen:
                seen.add(split.label)
                grid.append(split)
    # colocate first, then by (total gpus, infer gpus) for a readable sweep
    head, tail = grid[0], grid[1:]
    tail.sort(key=lambda s: (s.total_gpus, s.infer_gpus))
    return [head, *tail]
