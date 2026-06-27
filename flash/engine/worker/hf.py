"""HF artifact channel: code-delivery + adapter/metrics/checkpoint upload (works without inbound net).

Artifacts (adapter, metrics.json, heartbeat.json, checkpoints) are streamed to a Hugging Face
dataset repo. Run-scoped state (``HF_REPO``/``PHASE``/``RUN_ID``/``SEED``) and the patchable
``hf_api``/``heartbeat``/``hf_upload_file`` are read THROUGH the worker package (``_w.<name>``) at
CALL time, so tests that ``monkeypatch.setattr(worker, "<name>", ...)`` then call these functions
(e.g. ``worker.publish_deployable_checkpoint``) take effect.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
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


def _hf_upload(do_upload, repo_subpath: str, required: bool, label: str) -> bool:
    """Shared HF upload loop for files/folders: HF_REPO guard + retry/raise-or-warn.

    ``required=True`` (completion artifacts DONE/metrics.json, the trained adapter) retries
    and finally raises: a swallowed upload failure would make the control plane mark a
    finished run failed/retried, or mark the run done while deployment can never download
    the missing adapter. Optional artifacts (generations, logs) only warn.

    Returns ``True`` when a commit actually landed (or there is no HF_REPO, so there is nothing
    to retry) and ``False`` when a best-effort upload failed. Callers that claimed throttle/quiet
    state on the strength of a commit (heartbeat()) read this to roll back when nothing landed; the
    return is advisory and the many callers that ignore it keep their warn-only behavior.
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
                # Already retried 3x -> the host/network is bad, not the run. Infra-shaped.
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


def _shared_weight_cache_dir() -> str | None:
    """The shared weight-cache hub dir for the BASE-MODEL prefetch, or None for the ephemeral default.

    Both substrates set ``FLASH_WEIGHT_CACHE_DIR`` to ``<mount>/hf-cache/hub`` when the run carries the
    shared, multi-tenant weight cache — RunPod's network volume (``deps.weight_cache_env``, mount
    ``/runpod-volume``) or an instance provider's per-region bind mount (``_instance.build_payload``,
    mount ``/weight-cache``). We download ONLY the trusted public base model there and symlink it into
    the per-worker ephemeral cache (see ``prefetch_model``), so the base-model download is shared across
    runs while every OTHER HF download the run makes (env/reward datasets/models, pulled with the
    platform HF_TOKEN) stays in the ephemeral cache and never reaches the shared mount (issue #252).

    Returns None — the default ephemeral cache, a correct cold run — when the var is unset OR the mount
    is absent. The cache dir is ``<mount>/hf-cache/hub``, so the mount is two levels up; require it to
    actually exist. (The RunPod ``_train_body`` guard already strips the var when ``/runpod-volume``
    isn't mounted; this is the substrate-agnostic re-check on the worker itself, which also covers the
    instance path whose bootstrap does not strip.)
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
    """Symlink the base-model repo dir from the shared mount into the worker's EPHEMERAL hub cache.

    The base model was just downloaded to the shared mount (``shared_hub``) with an explicit
    ``cache_dir``. But the trainer (TRL) and the colocated vLLM engine load the model from a bare
    ``model_id`` string — they resolve the HF cache from the environment, NOT a cache_dir we control —
    so without this they'd re-download the multi-GB weights to the ephemeral default cache. A symlink
    at repo-folder granularity makes the mount's base model resolve there as a cache hit, while every
    OTHER repo the run fetches (env/reward assets) is written by HF into the real ephemeral cache and
    never touches the shared multi-tenant mount. Best-effort: on any error the loaders simply
    re-download to the ephemeral cache (slower, still correct AND still isolated).
    """
    from huggingface_hub.constants import HF_HUB_CACHE  # the worker's default (ephemeral) hub cache

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
    """Downloaded-byte total for ``model_id`` under ``cache_dir`` (or the default HF hub cache): the
    sum of the repo's ``blobs/`` files (the data, incl. the ``.incomplete`` partials an in-flight
    download grows). Scans ONLY ``blobs/`` (snapshots/ are symlinks; refs/metadata are tiny), so it
    matches "bytes downloaded" and stays cheap. ``cache_dir`` must be the dir snapshot_download writes
    to (the shared weight-cache mount when set, else the ephemeral default) or growth is invisible.

    Returns the blob byte total (``0`` if the repo dir exists but no blob is written yet — a real
    "0 bytes" measurement that still lets the silence timer trip), or ``None`` when the repo cache dir
    does not exist yet / on any error — the unmeasurable window ``liveness_heartbeat`` treats as
    "no advancement".
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo = os.path.join(cache_dir or HF_HUB_CACHE, _repo_folder_name(model_id))
        if not os.path.isdir(repo):
            return None  # cache structure not created yet -> can't measure
        blobs = os.path.join(repo, "blobs")
        if not os.path.isdir(blobs):
            return 0  # repo dir exists but no blobs written yet -> 0 bytes downloaded (measurable)
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
    """Pull the base-model weights into the HF cache up front; return seconds spent.

    The trainer/vLLM would download lazily anyway — doing it explicitly (a) makes the
    download a first-class, timed stage in the heartbeat stream (the cold-start metric
    the speed work optimizes), and (b) fails fast with a clear disk/network error
    instead of dying inside trainer construction. Idempotent: a warm cache costs ~0 s.

    When the shared weight-cache volume is attached (``FLASH_WEIGHT_CACHE_DIR``), the base model is
    downloaded ONTO the mount and symlinked into the ephemeral cache so the trainer/vLLM hit it
    without re-downloading — while the run's env/reward HF downloads stay off the shared mount (#252).
    """
    from huggingface_hub import snapshot_download

    shared_hub = _shared_weight_cache_dir()
    t0 = time.time()
    # snapshot_download blocks with NO heartbeat until it returns, but a cold cache can pull tens of GB
    # over many minutes — longer than the provider setup grace — so a silent
    # download would look like a hang and self-kill a HEALTHY cold start. Keep a model_prefetching
    # heartbeat alive, gated on downloaded-byte GROWTH (in the dir the download actually writes to, so a
    # genuinely WEDGED transfer still yields to the stall path). See heartbeat.liveness_heartbeat.
    from flash.engine.worker.heartbeat import liveness_heartbeat

    with liveness_heartbeat(
        "model_prefetching", progress=lambda: _hf_cache_bytes(model_id, shared_hub)
    ):
        try:
            snapshot_download(
                repo_id=model_id,
                # Base model ONLY onto the shared mount; None => the per-worker ephemeral default cache.
                cache_dir=shared_hub,
                # weights + tokenizer/config only (same exclusions as the image bake)
                ignore_patterns=["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
            )
            if shared_hub:
                _link_base_model_into_ephemeral_cache(model_id, shared_hub)
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
                _publish_deployable_with_retry(deploy_src, step)
            else:
                publish_deployable_checkpoint(deploy_src, step)
        finally:
            if recombined:
                shutil.rmtree(recombined, ignore_errors=True)

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
                    _publish_deployable_recombined(ckpt_dir, step)
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
                    _publish_deployable_recombined(ckpt_dir, step)
                finally:
                    lock.release()
            else:
                print(
                    f"[ckpt] flush lock busy after {_CKPT_FLUSH_TIMEOUT_S:.0f}s; publishing final "
                    f"deployable (step {step}) without it"
                )
                _publish_deployable_recombined(ckpt_dir, step, with_retry=True)

    return _CheckpointUpload()
