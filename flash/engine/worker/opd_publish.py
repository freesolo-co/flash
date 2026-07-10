"""OPD adapter persistence + deployable publish helpers, extracted from ``opd`` (which re-imports
them so they stay importable from it)."""

from __future__ import annotations

from flash.engine.worker._pkg import W as _w


def _save_adapter(model, tok, adapter_dir: str) -> None:
    """Persist the LoRA adapter + tokenizer for deploy (identical layout to SFT)."""
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)


def _publish_opd_deployable(
    adapter_dir: str, step: int, *, as_default: bool, best_effort: bool = False
) -> None:
    """Publish the step-``step`` deployable adapter (and, when ``as_default``, the ``<prefix>/adapter``
    served default). The opd stage CONTINUES the one warm-started adapter in place, so ``adapter_dir``
    already carries SFT+opd on the catalog base and deploys as-is (same for fresh-LoRA runs) — no
    recombine. Mirrors GRPO finalize (rl.py).

    ``best_effort`` (mid-run per-step publish): swallow a publish failure and KEEP training — a
    transient upload error during a save_every publish must not terminate run_opd after real optimizer
    steps (GRPO's per-step checkpoint callback is likewise best-effort). At finalize
    (``best_effort=False``) a publish failure is FATAL."""
    try:
        if as_default:
            _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        _w.publish_deployable_checkpoint(adapter_dir, step)
    except Exception as e:
        if not best_effort:
            raise
        print(f"[opd] deployable publish failed at step {step}; skipping, training continues: {e}")
