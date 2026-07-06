"""HF artifact channel: code-delivery + adapter/metrics/checkpoint upload (works without inbound net).

State and callables (hf_api, heartbeat, hf_upload_file) are read through _w at call time so
monkeypatch.setattr(worker, ...) takes effect in tests.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import threading
import time

from flash.adapter_artifacts import ADAPTER_WEIGHT_FILES
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import RetriableInfraError, gpu_diagnostics


def error_artifact_name(mode: str, attempt=0) -> str:
    """Per-mode, per-attempt error filename (e.g. error_sft_attempt0.txt). Attempt-scoped so a prior
    attempt's stale traceback can't be mistaken for the current attempt's crash on a retry host-loss."""
    return f"error_{mode}_attempt{int(attempt or 0)}.txt"


def hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def hf_prefix() -> str:
    return f"{_w.PHASE}/{_w.RUN_ID}"


def _hf_upload(do_upload, repo_subpath: str, required: bool, label: str) -> bool:
    """HF upload loop: retries + raises on required artifacts; warn-only on optional.

    Returns True when a commit landed (or HF_REPO is unset), False on best-effort failure.
    """
    if not _w.HF_REPO:
        return True
    attempts = 3 if required else 1
    for attempt in range(attempts):
        try:
            do_upload()
            return True
        except Exception as e:
            if required and attempt + 1 < attempts:
                print(f"{label} retry {attempt + 1}/{attempts}: {e}")
                time.sleep(5 * (attempt + 1))
                continue
            if required:
                raise RetriableInfraError(f"required upload of {repo_subpath!r} failed: {e}") from e
            print(f"{label} warn:", e)
            return False
    return False


def hf_upload_file(local_path: str, repo_subpath: str, required: bool = False) -> bool:
    """Upload one file to the run's HF prefix. Returns True on success (see ``_hf_upload``)."""
    return _hf_upload(
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
    """Append bounded JSONL debug rows and upload as an optional artifact (best-effort)."""
    if not rows or not _w.HF_REPO:
        return
    repo_name = os.path.basename(name if name.endswith(".jsonl") else f"{name}.jsonl")
    path = os.path.join("/tmp", repo_name)
    try:
        with _DEBUG_UPLOAD_LOCK:
            existing: list[str] = []
            # open() is evaluated before suppress() context enters — use try/except, not suppress.
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


def hf_upload_folder(local_dir: str, repo_subpath: str, required: bool = False) -> bool:
    """Upload a folder to the run's HF prefix. Returns True on success (see ``_hf_upload``)."""
    return _hf_upload(
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


# Stamped into every streamed resume checkpoint; read back before resuming (see write_gpu_stamp).
GPU_STAMP_FILE = "flash_gpu.json"


def _live_gpu_sm() -> str | None:
    """The live device's sm arch string (e.g. ``sm90``), or None off-GPU."""
    try:
        import torch

        cap = torch.cuda.get_device_capability(0)
        return f"sm{cap[0]}{cap[1]}"
    except Exception:
        return None


def write_gpu_stamp(ckpt_dir: str) -> None:
    """Record which GPU arch produced this checkpoint (``flash_gpu.json`` inside the checkpoint dir).

    ``hf_resume_checkpoint`` reads it back so a resume that lands on a different arch is flagged
    loudly instead of happening silently (the 2026-07-05 incident: a false-stall retry moved a
    healthy H200 SFT onto a B200 mid-run, resumed its checkpoint, and sm100's then-broken GDN
    backward corrupted every step after the resume while loss looked normal — with zero trace of
    the switch on the run). Best-effort: a stamp failure never blocks the save."""
    try:
        sm = _live_gpu_sm()
        if not sm:
            return
        name = None
        with contextlib.suppress(Exception):
            import torch

            name = torch.cuda.get_device_name(0)
        stamp = {
            "sm": sm,
            "device_name": name,
            "gpu_class": (_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None),
            "attempt": _w.ATTEMPT,
        }
        with open(os.path.join(ckpt_dir, GPU_STAMP_FILE), "w") as f:
            json.dump(stamp, f)
    except Exception as e:
        print("[ckpt] gpu-arch stamp warn:", e)


# "smA->smB" when the current attempt resumed a checkpoint produced on a different GPU arch, else
# None. Read into the run's DONE metrics notes so a cross-arch resume is auditable on the artifact.
_RESUME_CROSS_ARCH: str | None = None


def resume_cross_arch_note() -> str | None:
    """The current attempt's cross-arch resume marker ("sm90->sm100"), or None."""
    return _RESUME_CROSS_ARCH


def _note_resume_arch(ckpt_path: str) -> None:
    """Loudly record when this worker's GPU arch differs from the resumed checkpoint's stamp.

    Cross-arch resume is ALLOWED: every attempt re-selects its whole kernel set from the LIVE
    device at worker boot, before any model import (sm100 fla-tilelang opt-out, Blackwell autotune
    restriction, Hopper fast path, chalk/attn dispatch) — nothing numerical is carried over from
    the producing arch. What must never happen again is a SILENT switch: the 2026-07-05 incident
    resumed an H200 checkpoint on a B200 whose GDN backward was broken and trained 271 corrupt
    steps invisibly. The marker goes to the console AND the run's metrics notes
    (``resume_cross_arch``) so the artifact stays auditable. Unstamped (pre-stamp) checkpoints and
    unreadable stamps pass silently — fail-open."""
    global _RESUME_CROSS_ARCH
    try:
        with open(os.path.join(ckpt_path, GPU_STAMP_FILE)) as f:
            stamp = json.load(f)
        ckpt_sm = str(stamp.get("sm") or "")
    except FileNotFoundError:
        return
    except Exception as e:
        print("[resume] gpu-arch stamp unreadable:", e)
        return
    live = _live_gpu_sm()
    if not ckpt_sm or live is None or live == ckpt_sm:
        return
    _RESUME_CROSS_ARCH = f"{ckpt_sm}->{live}"
    print(
        f"[resume] CROSS-ARCH RESUME: checkpoint {os.path.basename(ckpt_path)} was trained on "
        f"{ckpt_sm} ({stamp.get('device_name')}, class {stamp.get('gpu_class')}); this worker is "
        f"{live}. Kernel backends were re-selected for {live} at boot; flagging in metrics notes "
        "(resume_cross_arch) for auditability."
    )


def hf_resume_checkpoint() -> str | None:
    """Download the latest streamed trainer checkpoint for this run, or return None.

    A checkpoint stamped with a different GPU arch than the live device still resumes — the
    kernel set is re-selected per-arch at worker boot — but the switch is flagged loudly (see
    ``_note_resume_arch``)."""
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
    except Exception as e:
        print("hf_resume_checkpoint warn:", e)
        return None
    _note_resume_arch(path)
    print(f"[resume] found streamed checkpoint: {path}")
    return path


def _shared_weight_cache_dir() -> str | None:
    """Return the shared weight-cache hub dir (FLASH_WEIGHT_CACHE_DIR), or None if absent/unmounted.

    Base-model downloads land here; all other HF fetches stay in the ephemeral per-worker cache (#252).
    """
    cache_dir = os.environ.get("FLASH_WEIGHT_CACHE_DIR")
    if not cache_dir:
        return None
    mount = os.path.dirname(os.path.dirname(cache_dir.rstrip("/")))
    if not mount or not os.path.isdir(mount):
        return None
    return cache_dir


def _repo_folder_name(model_id: str) -> str:
    """HF cache folder for a model repo (``models--org--name``), preferring the library helper."""
    try:
        from huggingface_hub.file_download import repo_folder_name

        return repo_folder_name(repo_id=model_id, repo_type="model")
    except Exception:  # older/newer hub layout — the format is stable, so fall back to it
        return "models--" + model_id.replace("/", "--")


def _link_base_model_into_ephemeral_cache(model_id: str, shared_hub: str) -> None:
    """Symlink base-model repo dir from the shared mount into the ephemeral HF hub cache.

    Trainer/vLLM load via model_id (not cache_dir) so without this they'd re-download the weights.
    Best-effort: on error the loaders fall back to a cold download.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = _repo_folder_name(model_id)
    src = os.path.join(shared_hub, folder)
    if not os.path.isdir(src):
        return  # download didn't land on the mount (gated/local-only) — nothing to link
    dst = os.path.join(HF_HUB_CACHE, folder)
    if os.path.realpath(HF_HUB_CACHE) == os.path.realpath(shared_hub):
        return  # ephemeral cache IS the mount (shouldn't happen) — already a hit, don't self-link
    if os.path.lexists(dst):
        return  # already linked (warm worker) or a real dir an env created first — leave it
    try:
        os.makedirs(HF_HUB_CACHE, exist_ok=True)
        os.symlink(src, dst, target_is_directory=True)
        print(f"[weight-cache] linked base model {model_id} from shared mount into {HF_HUB_CACHE}")
    except OSError as e:
        print("prefetch_model link warn:", e)


def _hf_cache_bytes(model_id: str, cache_dir: str | None = None) -> int | None:
    """Bytes downloaded for model_id under cache_dir (or default HF cache), scanning blobs/ only.

    Returns 0 if repo dir exists but no blobs yet; None if repo dir missing or on error.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = os.path.join(cache_dir or HF_HUB_CACHE, _repo_folder_name(model_id))
        if not os.path.isdir(repo):
            return None
        blobs = os.path.join(repo, "blobs")
        if not os.path.isdir(blobs):
            return 0
        total = 0
        for fn in os.listdir(blobs):
            fp = os.path.join(blobs, fn)
            with contextlib.suppress(OSError):
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        return total
    except Exception:
        return None


def prefetch_model(model_id: str) -> float:
    """Pull base-model weights into the HF cache up front; return seconds spent.

    When the shared weight-cache volume is attached, downloads onto the mount and symlinks into
    the ephemeral cache so trainer/vLLM get a cache hit without re-downloading (#252).
    """
    from huggingface_hub import snapshot_download

    shared_hub = _shared_weight_cache_dir()
    t0 = time.time()
    # Cold downloads can take tens of GB; liveness_heartbeat keeps the worker alive during the fetch.
    from flash.engine.worker.heartbeat import liveness_heartbeat

    with liveness_heartbeat(
        "model_prefetching", progress=lambda: _hf_cache_bytes(model_id, shared_hub)
    ):
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=shared_hub,
                ignore_patterns=["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
            )
            if shared_hub:
                _link_base_model_into_ephemeral_cache(model_id, shared_hub)
        except Exception as e:
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


# Trainer-state files the serving engine never needs; excluded to keep per-step adapter snapshots small.
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
    GPU_STAMP_FILE,  # resume-only arch stamp; meaningless to the serving engine
)


def _has_deployable_adapter(ckpt_dir: str) -> bool:
    """Return True if ckpt_dir has a loadable LoRA adapter (config + weights)."""
    return os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json")) and any(
        os.path.isfile(os.path.join(ckpt_dir, w)) for w in ADAPTER_WEIGHT_FILES
    )


def publish_deployable_checkpoint(
    ckpt_dir: str, step: int, *, retries: int = 1, backoff_s: float = 0.0
) -> str | None:
    """Mirror a trainer checkpoint's LoRA adapter to a stable per-step path for mid-RL deployability.

    Uploads adapter only (no optimizer/scheduler state) to <prefix>/checkpoints/step-<step>/adapter.
    Returns the subfolder path, or None if no adapter or upload failed. Best-effort: never fails a run.
    retries > 1 loops only the upload (linear backoff_s); the adapter gate runs once up front.
    """
    if not _w.HF_REPO:
        return None
    if not _has_deployable_adapter(ckpt_dir):
        return None
    subfolder = f"{hf_prefix()}/checkpoints/step-{step}/adapter"
    for attempt in range(retries):
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
            if attempt + 1 < retries:
                time.sleep(backoff_s * (attempt + 1))
    return None


# Retry/backoff for the deployable-adapter publish (rides out a transient concurrent-commit 409).
_CKPT_FLUSH_RETRIES = 3
_CKPT_FLUSH_BACKOFF_S = 1.0
# Retry/backoff for each synchronous checkpoint upload. `on_save` BLOCKS the training loop on the
# upload, so a transient HF error is retried until the step lands rather than costing the step.
_CKPT_UPLOAD_RETRIES = 3
_CKPT_UPLOAD_BACKOFF_S = 5.0


def _latest_checkpoint_dir(output_dir: str) -> tuple[int, str] | None:
    """Return (step, path) for the highest checkpoint-<n> dir under output_dir, or None."""
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


def _prune_stale_resume_checkpoints(keep_step: int) -> None:
    """Delete every ``{prefix}/checkpoint/checkpoint-N`` except ``keep_step``.

    The streamed resume checkpoint is meant to be latest-only, but ``upload_folder``'s delete_patterns
    are matched RELATIVE to path_in_repo (the per-step dir), so they can never reach sibling step dirs —
    without this they accumulate unbounded on HF and ``hf_resume_checkpoint`` re-downloads them all.
    Runs AFTER the new checkpoint lands, so the latest is always present. The deployable-adapter tree
    (``{prefix}/checkpoints/...``, plural) has a different prefix and is untouched.
    """
    if not _w.HF_REPO:
        return
    api = _w.hf_api()
    base = f"{hf_prefix()}/checkpoint/"
    try:
        files = api.list_repo_files(repo_id=_w.HF_REPO, repo_type="dataset")
    except Exception as e:
        print("ckpt prune warn (list):", e)
        return
    stale: set[str] = set()
    for f in files:
        if not f.startswith(base):
            continue
        seg = f[len(base) :].split("/", 1)[0]
        n = seg[len("checkpoint-") :]
        if seg.startswith("checkpoint-") and n.isdigit() and int(n) != keep_step:
            stale.add(f"{base}{seg}")
    for folder in sorted(stale):
        with contextlib.suppress(Exception):
            api.delete_folder(path_in_repo=folder, repo_id=_w.HF_REPO, repo_type="dataset")


def make_checkpoint_upload_callback():
    """Return a TrainerCallback that streams each save to HF and publishes deployable per-step adapters.

    Uploads are SYNCHRONOUS: on_save blocks the training loop until the checkpoint is durably on
    HF. A save therefore never returns while its upload is still running, so there is never a
    second upload in flight when the next save fires — the "upload busy" contention (and the
    coalescing/dropping it used to cause) cannot happen by construction. Every `save_every` step is
    guaranteed to upload, retrying transient HF errors instead of skipping.
    """
    from transformers import TrainerCallback

    def _publish_deployable_recombined(
        ckpt_dir: str, step: int, *, with_retry: bool = False
    ) -> None:
        """Publish a step's deployable adapter, stacking the SFT back in for a VL warm-start.

        For a VL merge-into-base warm-start (#296) the trainer checkpoint's adapter is GRPO-ONLY
        (trained on the SFT-merged base) — on the catalog base it drops the SFT and collapses to
        ~base. ``recombined_warmstart_adapter_dir`` stacks the original SFT LoRA back in (into a
        SEPARATE temp dir, so the resume checkpoint keeps the raw GRPO LoRA that reattaches to the
        re-merged base on resume). ``recombined_warmstart_adapter_dir`` returns None for the
        continued-adapter / fresh-LoRA paths (raw IS the deployable), but RAISES for a VL warm-start
        that required a recombine and couldn't (e.g. the recorded SFT dir was evicted). On that raise
        we must NOT fall back to the raw checkpoint: it's GRPO-only / SFT-less and collapses to ~base
        on the catalog base, so publishing it would advertise a known-broken deployable. Skip this
        step's deployable publish and surface the failure instead (the resume checkpoint is still
        uploaded by the caller, so the run can resume and re-merge).
        """
        recombined: str | None = None
        try:
            try:
                recombined = _w.recombined_warmstart_adapter_dir(ckpt_dir)
            except Exception as e:
                print(
                    f"[ckpt] warm-start recombine FAILED (step {step}); skipping deployable publish "
                    f"to avoid registering an SFT-less adapter: {e}"
                )
                return
            deploy_src = recombined or ckpt_dir
            if with_retry:
                # #295 folded _publish_deployable_with_retry into publish_deployable_checkpoint(retries=).
                publish_deployable_checkpoint(
                    deploy_src, step, retries=_CKPT_FLUSH_RETRIES, backoff_s=_CKPT_FLUSH_BACKOFF_S
                )
            else:
                publish_deployable_checkpoint(deploy_src, step)
        finally:
            if recombined:
                shutil.rmtree(recombined, ignore_errors=True)

    def _upload_once(step: int, ckpt_dir: str) -> None:
        # Deployable per-step adapter FIRST: it's small, kept-forever, and the only
        # artifact that makes a cancelled/preempted run deployable from this step, so
        # it must land before the larger resume checkpoint (best-effort, latest-only).
        _publish_deployable_recombined(ckpt_dir, step)
        _w.hf_api().upload_folder(
            folder_path=ckpt_dir,
            path_in_repo=f"{hf_prefix()}/checkpoint/checkpoint-{step}",
            repo_id=_w.HF_REPO,
            repo_type="dataset",
        )
        # Prune older step dirs only after the new one is safely up (latest-only resume).
        _prune_stale_resume_checkpoints(step)
        _w.heartbeat("checkpoint_uploaded", step=step)

    uploaded_steps: set[int] = set()

    def _upload_with_retries(step: int, ckpt_dir: str) -> bool:
        """Run the full upload for `step`, retrying transient failures; True once it lands."""
        for attempt in range(_CKPT_UPLOAD_RETRIES):
            try:
                _upload_once(step, ckpt_dir)
                uploaded_steps.add(step)
                return True
            except Exception as e:
                if attempt + 1 < _CKPT_UPLOAD_RETRIES:
                    print(
                        f"[ckpt] step {step} upload retry {attempt + 1}/{_CKPT_UPLOAD_RETRIES}: {e}"
                    )
                    time.sleep(_CKPT_UPLOAD_BACKOFF_S * (attempt + 1))
                else:
                    print(
                        f"[ckpt] step {step} upload FAILED after {_CKPT_UPLOAD_RETRIES} attempts: {e}"
                    )
                    with contextlib.suppress(Exception):
                        _w.heartbeat("checkpoint_upload_failed", step=step)
        return False

    class _CheckpointUpload(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # SYNCHRONOUS on the training thread: the trainer blocks here until this checkpoint is
            # uploaded. `save_total_limit=1` rotation (which runs earlier, inside this same save's
            # `_save_checkpoint`) deletes the PREVIOUS checkpoint, never the current one — and the
            # next save can't begin until we return — so the dir is stable for the whole upload and
            # no snapshot/queue is needed. Because the upload finishes before the save returns,
            # there is never a concurrent upload for a later save to be "busy" against.
            if not _w.HF_REPO:
                return
            step = int(state.global_step)
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            if not os.path.isdir(ckpt_dir):
                return
            # Stamp the producing GPU arch into the dir BEFORE the (now synchronous, #445) upload,
            # so the streamed resume checkpoint carries flash_gpu.json for cross-arch-resume detection.
            write_gpu_stamp(ckpt_dir)
            _upload_with_retries(step, ckpt_dir)

        def on_train_end(self, args, state, control, **kwargs):
            # Safety net for a final checkpoint the trainer wrote without an on_save (e.g. a
            # load_best_model_at_end / end-of-training save). Synchronous on_save already uploaded
            # every step it saw, so this only publishes an as-yet-unuploaded latest step.
            if not _w.HF_REPO:
                return
            latest = _latest_checkpoint_dir(args.output_dir)
            if latest is None:
                return
            step, ckpt_dir = latest
            if step in uploaded_steps:
                return
            _upload_with_retries(step, ckpt_dir)

    return _CheckpointUpload()
