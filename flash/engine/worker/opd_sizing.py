"""OPD rollout/teacher/loss batch sizing helpers (cost/estimate math for the opd worker loop).

Pure integer sizing math extracted from ``opd`` so the main training path stays readable. The step
pipeline chunk count, teacher batch size / worker count, and the loss microbatch size are all
derived here from the per-step sample count; ``opd`` re-imports these so they stay importable.
"""

from __future__ import annotations

OPD_ROLLOUT_PIPELINE_TARGET_CHUNK_SIZE = 16
OPD_ROLLOUT_PIPELINE_MAX_CHUNKS = 8
OPD_TEACHER_BATCH_SIZE = 8
OPD_LOSS_MICROBATCH_SIZE = 4


def _opd_rollout_pipeline_target_chunk_size(total_prompts: int) -> int:
    total = max(1, int(total_prompts))
    return max(1, min(total, OPD_ROLLOUT_PIPELINE_TARGET_CHUNK_SIZE))


def _opd_rollout_pipeline_max_chunks(total_prompts: int) -> int:
    total = max(1, int(total_prompts))
    return max(1, min(total, OPD_ROLLOUT_PIPELINE_MAX_CHUNKS))


def _opd_rollout_pipeline_chunks(total_prompts: int) -> int:
    """Number of single-turn rollout chunks in one OPD step.

    Small steps still split once so teacher scoring can overlap later vLLM generation. Larger steps
    target moderately sized vLLM batches, which starts remote teacher scoring earlier without reducing
    rollout generation to one request per prompt.
    """
    total = max(1, int(total_prompts))
    if total < 8:
        return 1
    target_chunk = _opd_rollout_pipeline_target_chunk_size(total)
    max_chunks = _opd_rollout_pipeline_max_chunks(total)
    chunks = (total + target_chunk - 1) // target_chunk
    if max_chunks == 1:
        return 1
    return max(2, min(max_chunks, chunks))


def _opd_rollout_chunk_size(total_prompts: int) -> int:
    """Split a single OPD step into a small number of rollout chunks so remote teacher scoring for
    earlier chunks overlaps with vLLM generation for later chunks without collapsing vLLM batching."""
    total = max(1, int(total_prompts))
    chunks = _opd_rollout_pipeline_chunks(total)
    return max(1, (total + chunks - 1) // chunks)


def _opd_teacher_batch_size(total_samples: int) -> int:
    total = max(1, int(total_samples))
    return max(1, min(total, OPD_TEACHER_BATCH_SIZE))


def _opd_teacher_workers(total_samples: int, batch_size: int) -> int:
    total = max(1, int(total_samples))
    return max(1, (total + max(1, int(batch_size)) - 1) // max(1, int(batch_size)))


def _opd_loss_microbatch_size(model_id: str, total_samples: int) -> int:
    total = max(1, int(total_samples))
    try:
        from flash.engine.vram import resolve_params_b

        params_b = float(resolve_params_b(model_id) or 0.0)
    except Exception:
        params_b = 0.0
    # The dense vocab logits from GKD are the peak. Batch small/medium dense models, but keep 35B-class
    # OPD serial by default so the B200 path does not trade speed for OOM risk.
    default = OPD_LOSS_MICROBATCH_SIZE if params_b and params_b <= 10.0 else 1
    return max(1, min(total, default))
