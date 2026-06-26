"""HF artifact channel: code-delivery + adapter/metrics/checkpoint upload (works without inbound net).

Artifacts (adapter, metrics.json, heartbeat.json, checkpoints) are streamed to a Hugging Face
dataset repo. Run-scoped state (``HF_REPO``/``PHASE``/``RUN_ID``/``SEED``) and the patchable
``hf_api``/``heartbeat``/``hf_upload_file`` are read THROUGH the worker package (``_w.<name>``) at
CALL time, so tests that ``monkeypatch.setattr(worker, "<name>", ...)`` then call these functions
(e.g. ``worker.publish_deployable_checkpoint``) take effect.
"""

from __future__ import annotations

import json
import os
import threading
import time

from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import RetriableInfraError, gpu_diagnostics


def error_artifact_name(mode: str) -> str:
    """Per-mode error filename (e.g. error_sft.txt) so a run's traceback is uploaded
    under a stable name even though heartbeat.json is single-file/overwritten."""
    return f"error_{mode}.txt"


def hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def hf_prefix() -> str:
    return f"{_w.PHASE}/{_w.RUN_ID}/seed{_w.SEED}"


def _hf_upload(do_upload, repo_subpath: str, required: bool, label: str) -> None:
    """Shared HF upload loop for files/folders: HF_REPO guard + retry/raise-or-warn.

    ``required=True`` (completion artifacts DONE/metrics.json, the trained adapter) retries
    and finally raises: a swallowed upload failure would make the control plane mark a
    finished run failed/retried, or mark the run done while deployment can never download
    the missing adapter. Optional artifacts (generations, logs) only warn.
    """
    if not _w.HF_REPO:
        return
    attempts = 3 if required else 1
    for attempt in range(attempts):
        try:
            do_upload()
            return
        except Exception as e:
            if required and attempt + 1 < attempts:
                print(f"{label} retry {attempt + 1}/{attempts}: {e}")
                time.sleep(5 * (attempt + 1))
                continue
            if required:
                # Already retried 3x -> the host/network is bad, not the run. Infra-shaped.
                raise RetriableInfraError(f"required upload of {repo_subpath!r} failed: {e}") from e
            print(f"{label} warn:", e)
            return


def hf_upload_file(local_path: str, repo_subpath: str, required: bool = False):
    """Upload one file to the run's HF prefix."""
    _hf_upload(
        lambda: _w.hf_api().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{hf_prefix()}/{repo_subpath}",
            repo_id=_w.HF_REPO,
            repo_type="dataset",
        ),
        repo_subpath,
        required,
        "hf_upload_file",
    )


_DEBUG_UPLOAD_LOCK = threading.Lock()


def upload_debug_jsonl(name: str, rows: list[dict], *, keep_last: int = 200) -> None:
    """Append bounded JSONL debug rows and upload them as an optional artifact.

    This is intentionally best-effort: debug visibility must not fail a paid run.
    """
    if not rows or not _w.HF_REPO:
        return
    repo_name = os.path.basename(name if name.endswith(".jsonl") else f"{name}.jsonl")
    path = os.path.join("/tmp", repo_name)
    try:
        with _DEBUG_UPLOAD_LOCK:
            existing: list[str] = []
            # try/except (not `suppress(...), open(...)`): open() is evaluated BEFORE the suppress()
            # context is entered, so a missing file on the first call would raise and skip all
            # writing/uploading. Swallow only the not-found case.
            try:
                with open(path) as f:
                    existing = f.readlines()[-keep_last:]
            except FileNotFoundError:
                pass
            with open(path, "w") as f:
                f.writelines(existing)
                for row in rows:
                    f.write(json.dumps(row, default=str, ensure_ascii=True, sort_keys=True) + "\n")
            _w.hf_upload_file(path, repo_name)
    except Exception as e:
        print(f"debug upload warn ({repo_name}): {e}")


def hf_upload_folder(local_dir: str, repo_subpath: str, required: bool = False):
    """Upload a folder to the run's HF prefix."""
    _hf_upload(
        lambda: _w.hf_api().upload_folder(
            folder_path=local_dir,
            path_in_repo=f"{hf_prefix()}/{repo_subpath}",
            repo_id=_w.HF_REPO,
            repo_type="dataset",
        ),
        repo_subpath,
        required,
        "hf_upload_folder",
    )


def hf_resume_checkpoint() -> str | None:
    """Latest streamed trainer checkpoint for this run (or None).

    Checkpoints are uploaded DURING the run by ``make_checkpoint_upload_callback`` as
    ``<prefix>/checkpoint/checkpoint-<step>/``; a replacement worker downloads the
    newest one so a mid-run preemption costs at most one save interval.
    """
    if not _w.HF_REPO:
        return None
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=_w.HF_REPO,
            repo_type="dataset",
            allow_patterns=[f"{hf_prefix()}/checkpoint/**"],
            local_dir="/tmp/resume",
            token=os.environ.get("HF_TOKEN"),
        )
        base = os.path.join("/tmp/resume", hf_prefix(), "checkpoint")
        if not os.path.isdir(base):
            return None
        cands = [d for d in os.listdir(base) if d.startswith("checkpoint-")]
        if not cands:
            return None
        latest = max(cands, key=lambda d: int(d.split("-")[-1]))
        path = os.path.join(base, latest)
        print(f"[resume] found streamed checkpoint: {path}")
        return path
    except Exception as e:
        print("hf_resume_checkpoint warn:", e)
        return None


def prefetch_model(model_id: str) -> float:
    """Pull the model weights into the local HF cache up front; return seconds spent.

    The trainer/vLLM would download lazily anyway — doing it explicitly (a) makes the
    download a first-class, timed stage in the heartbeat stream (the cold-start metric
    the speed work optimizes), and (b) fails fast with a clear disk/network error
    instead of dying inside trainer construction. Idempotent: a warm cache costs ~0 s.
    """
    from huggingface_hub import snapshot_download

    t0 = time.time()
    try:
        snapshot_download(
            repo_id=model_id,
            # weights + tokenizer/config only (same exclusions as the image bake)
            ignore_patterns=["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
        )
    except Exception as e:
        # Surface but don't fail here: gated/local-only models still load fine through
        # the normal from_pretrained path the trainer uses next.
        print("prefetch_model warn:", e)
    secs = round(time.time() - t0, 1)
    _w.heartbeat(
        "model_prefetched",
        model=model_id,
        download_seconds=secs,
        hf_transfer=os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
        gpu=gpu_diagnostics(),
    )
    return secs


# Trainer-state files a serving engine never needs: optimizer/scheduler/rng/loss-curve
# state. Excluded when publishing the deployable per-step adapter so each step's snapshot is
# just the LoRA weights + config (a few MB), small enough to KEEP every step (no pruning).
_CHECKPOINT_TRAINER_STATE = (
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
    "trainer_state.json",
    "training_args.bin",
    "*.distcp",
    "global_step*/**",
    "latest",
    "zero_to_fp32.py",
)

# The PEFT adapter weights file a checkpoint must carry to be loadable/servable (safetensors is
# the default; .bin is the legacy fallback). A step with adapter_config.json but no weights is
# NOT deployable, so it's never published/listed.
_ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def _has_deployable_adapter(ckpt_dir: str) -> bool:
    """Whether ``ckpt_dir`` carries a loadable LoRA adapter (config AND weights) — i.e. is publishable
    as a deployable step. A step missing either is never advertised."""
    return os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json")) and any(
        os.path.isfile(os.path.join(ckpt_dir, w)) for w in _ADAPTER_WEIGHT_FILES
    )


def publish_deployable_checkpoint(ckpt_dir: str, step: int) -> str | None:
    """Mirror a trainer checkpoint's LoRA adapter to a stable, NON-pruned per-step path so a
    run cancelled mid-RL is still one-command-deployable from its last good step.

    The trainer's checkpoint folder already contains the PEFT adapter (``adapter_config.json``
    + ``adapter_model.safetensors``) that ``deploy_adapter`` serves; we re-upload just those
    (dropping optimizer/scheduler/rng state) to ``<prefix>/checkpoints/step-<step>/adapter``.
    Unlike the resume checkpoint (``checkpoint/**``, kept latest-only), these accumulate, so
    EVERY step stays deployable. Returns the deployable adapter subfolder, or ``None`` when
    there's no adapter to publish. Best-effort: a failure here never fails a paid run.
    """
    if not _w.HF_REPO:
        return None
    # Only publish a checkpoint that actually carries a loadable adapter — never advertise a
    # non-deployable step.
    if not _has_deployable_adapter(ckpt_dir):
        return None
    subfolder = f"{hf_prefix()}/checkpoints/step-{step}/adapter"
    try:
        _w.hf_api().upload_folder(
            folder_path=ckpt_dir,
            path_in_repo=subfolder,
            repo_id=_w.HF_REPO,
            repo_type="dataset",
            ignore_patterns=list(_CHECKPOINT_TRAINER_STATE),
        )
        _w.heartbeat("checkpoint_deployable", step=step, subfolder=subfolder)
        return subfolder
    except Exception as e:
        print(f"[ckpt] deployable publish warn (step {step}):", e)
        return None


# How long ``on_train_end`` waits to flush the final deployable checkpoint before the worker
# moves on to publish the run's final adapter and exit. The per-save uploads run in DAEMON
# threads, which the interpreter kills the instant the worker process exits — so a fast RL run
# can finish (and tear down) before its last save's deployable snapshot finishes uploading,
# leaving ``flash checkpoints`` empty even though training succeeded. The flush blocks (inside
# ``trainer.train()``) up to this long so the last good step is reliably deployable.
_CKPT_FLUSH_TIMEOUT_S = 300.0
# When the flush can't take the lock in time (a daemon upload outran the window), it publishes the
# final deployable WITHOUT the lock; these bound a tiny retry that rides out a transient
# concurrent-commit 409 from the still-running daemon. Linear backoff, and ONLY on failure — a clean
# publish (the common case) adds zero latency to worker exit.
_CKPT_FLUSH_RETRIES = 3
_CKPT_FLUSH_BACKOFF_S = 1.0


def _publish_deployable_with_retry(ckpt_dir: str, step: int) -> None:
    """Best-effort publish of the final deployable when the flush couldn't hold the upload lock.

    Gated on the adapter actually being present so the retry is provably non-futile:
    ``publish_deployable_checkpoint`` returns ``None`` for BOTH "no adapter" and "upload failed", so
    after this precheck a ``None`` can only mean a real (likely transient-409) failure worth a retry —
    an empty step returns immediately with no wasted sleeps."""
    if not _has_deployable_adapter(ckpt_dir):
        return
    for attempt in range(_CKPT_FLUSH_RETRIES):
        if publish_deployable_checkpoint(ckpt_dir, step) is not None:
            return
        if attempt + 1 < _CKPT_FLUSH_RETRIES:
            time.sleep(_CKPT_FLUSH_BACKOFF_S * (attempt + 1))
    print(f"[ckpt] could not publish final deployable for step {step} before exit")


def _latest_checkpoint_dir(output_dir: str) -> tuple[int, str] | None:
    """The highest-step ``checkpoint-<n>`` directory the trainer has written under
    ``output_dir`` (its most recent save), as ``(step, path)`` — or ``None`` if none exist.

    The trainer writes the checkpoint folder to local disk synchronously at each save, so this
    sees the latest save even when its async HF upload was still in flight (or got dropped by
    the busy-lock skip). ``on_train_end`` uses it to guarantee a deployable final snapshot.
    """
    best: tuple[int, str] | None = None
    try:
        entries = os.listdir(output_dir)
    except OSError:
        return None
    for name in entries:
        if not name.startswith("checkpoint-"):
            continue
        suffix = name[len("checkpoint-") :]
        path = os.path.join(output_dir, name)
        if not suffix.isdigit() or not os.path.isdir(path):
            continue
        step = int(suffix)
        if best is None or step > best[0]:
            best = (step, path)
    return best


def make_checkpoint_upload_callback():
    """Stream each trainer save to HF so preemption loses <= one save interval.

    Uploads run in a background thread (the train loop never blocks on the network);
    older checkpoints are deleted in the same commit. If an upload is still in flight
    when the next save fires, the new save is skipped (the following one catches up).

    Each save also publishes a deployable per-step adapter snapshot (``publish_deployable_
    checkpoint``) so a run cancelled mid-RL can still be deployed from its latest step. The
    deployable snapshot is published FIRST (it's a few-MB adapter, durable and accumulating)
    and the larger latest-only resume checkpoint second, so the artifact that survives a
    teardown lands soonest. ``on_train_end`` then flushes the final snapshot synchronously,
    because the per-save uploads run in daemon threads that the worker would otherwise kill on
    exit before a fast run's last deployable checkpoint finishes uploading.
    """
    from transformers import TrainerCallback

    lock = threading.Lock()

    class _CheckpointUpload(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if not _w.HF_REPO:
                return
            step = int(state.global_step)
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            if not os.path.isdir(ckpt_dir):
                return
            if not lock.acquire(blocking=False):
                print(f"[ckpt] upload busy; skipping step {step}")
                return

            def _upload():
                try:
                    # Deployable per-step adapter FIRST: it's small, kept-forever, and the only
                    # artifact that makes a cancelled/preempted run deployable from this step, so
                    # it must land before the larger resume checkpoint (best-effort, latest-only).
                    publish_deployable_checkpoint(ckpt_dir, step)
                    _w.hf_api().upload_folder(
                        folder_path=ckpt_dir,
                        path_in_repo=f"{hf_prefix()}/checkpoint/checkpoint-{step}",
                        repo_id=_w.HF_REPO,
                        repo_type="dataset",
                        delete_patterns=[f"{hf_prefix()}/checkpoint/**"],
                    )
                    _w.heartbeat("checkpoint_uploaded", step=step)
                except Exception as e:
                    print("ckpt upload warn:", e)
                finally:
                    lock.release()

            threading.Thread(target=_upload, daemon=True).start()

        def on_train_end(self, args, state, control, **kwargs):
            # The per-save uploads run in DAEMON threads, killed the instant the worker exits at run
            # end. A fast RL run can reach "done" (and tear down) before its last save's deployable
            # checkpoint finishes uploading — leaving `flash checkpoints` empty and `flash deploy
            # --step` impossible even though training succeeded. Flush here, INSIDE trainer.train()
            # (before the worker publishes the final adapter and exits), so the latest on-disk
            # checkpoint is reliably published as a deployable snapshot. publish_deployable_checkpoint
            # is idempotent (content-addressed path), so re-publishing a step the async upload already
            # handled is a cheap no-op.
            if not _w.HF_REPO:
                return
            latest = _latest_checkpoint_dir(args.output_dir)
            if latest is None:
                return
            step, ckpt_dir = latest
            # ALWAYS publish the final deployable before returning (the worker exits right after).
            # Under the lock when we can get it (serialized against on_save -> no commit conflict);
            # on timeout the lock is held by an over-budget on_save upload that the worker exit will
            # kill before its OWN deployable publish runs, and this step's on_save may itself have
            # been skipped on the busy lock -- so this flush is the only publisher of the final
            # deployable. Publishing here (synchronously, in the train-end thread, so exit can't kill
            # it) without the lock is strictly better than the silent skip this callback exists to
            # prevent; the bounded retry rides out a transient concurrent-commit 409.
            if lock.acquire(timeout=_CKPT_FLUSH_TIMEOUT_S):
                try:
                    publish_deployable_checkpoint(ckpt_dir, step)
                finally:
                    lock.release()
            else:
                print(
                    f"[ckpt] flush lock busy after {_CKPT_FLUSH_TIMEOUT_S:.0f}s; publishing final "
                    f"deployable (step {step}) without it"
                )
                _publish_deployable_with_retry(ckpt_dir, step)

    return _CheckpointUpload()
