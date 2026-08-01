"""HF artifact channel: code-delivery + adapter/metrics/checkpoint upload (works without inbound net).

State and callables (hf_api, heartbeat, hf_upload_file) are read through _w at call time so
monkeypatch.setattr(worker, ...) takes effect in tests.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

try:
    import fcntl
except ImportError:
    fcntl = None

from flash.adapter_artifacts import ADAPTER_WEIGHT_FILES
from flash.diagnostics import sanitize_diagnostic
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.perf import RetriableInfraError, gpu_diagnostics
from flash.opd_retry_contract import (
    canonical_opd_optimizer_start_json,
    opd_optimizer_start_marker_path,
)

_MAX_ATTEMPT_ID = (1 << 63) - 1


class RequiredSaveError(RuntimeError):
    """permanent exact-save contract failure."""


def _attempt_scoped_name(kind: str, mode: str, attempt: int) -> str:
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise ValueError("attempt must be a nonnegative integer")
    if attempt < 0 or attempt > _MAX_ATTEMPT_ID:
        raise ValueError("attempt must be a bounded nonnegative integer")
    return f"{kind}_{mode}_attempt{attempt}.txt"


def error_artifact_name(mode: str, attempt: int = 0) -> str:
    """Per-mode, per-attempt error filename (e.g. error_sft_attempt0.txt). Attempt-scoped so a prior
    attempt's stale traceback can't be mistaken for the current attempt's crash on a retry host-loss."""
    return _attempt_scoped_name("error", mode, attempt)


def ray_log_artifact_name(mode: str, attempt: int = 0) -> str:
    """Per-mode, per-attempt name for ray's own failure logs (e.g. raylogs_rl_attempt0.txt).

    Attempt-scoped for the same reason the traceback beside it is: ``hf_prefix()`` is per-RUN, not
    per-attempt, so an unscoped name would let a retry's ray logs overwrite the attempt that actually
    reproduced the raylet failure -- the one case this artifact exists to explain."""
    return _attempt_scoped_name("raylogs", mode, attempt)


def hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def model_revision_kwargs(revision: str = "") -> dict[str, str]:
    """return the hf revision keyword only for a nonempty pinned revision."""
    return {"revision": revision} if revision else {}


def load_tokenizer(model_id: str, revision: str = ""):
    """load the student tokenizer under the run's optional model revision."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        **model_revision_kwargs(revision),
    )


def hf_prefix() -> str:
    return f"{_w.PHASE}/{_w.RUN_ID}"


def publish_opd_optimizer_start_marker() -> None:
    """Synchronously publish the first-update mutation marker before optimizer.step()."""
    if not isinstance(_w.HF_REPO, str) or not _w.HF_REPO.strip():
        raise RuntimeError("opd optimizer-start marker requires a private HF repository")
    marker_path = opd_optimizer_start_marker_path(_w.RUN_ID, _w.ATTEMPT)
    local_path = f"/tmp/opd-optimizer-start-attempt-{_w.ATTEMPT}.json"
    payload = canonical_opd_optimizer_start_json(
        run_id=_w.RUN_ID,
        attempt=_w.ATTEMPT,
        seed=_w.SEED,
    )
    with open(local_path, "wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    try:
        _require_hf_deadline_allowance()
        _w.hf_api().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=marker_path,
            repo_id=_w.HF_REPO,
            repo_type="dataset",
        )
    except Exception as error:
        detail = sanitize_diagnostic(error, limit=500)
        raise RetriableInfraError(
            f"required upload of OPD optimizer-start marker failed: {detail}"
        ) from error


def _require_hf_deadline_allowance() -> float | None:
    remaining = _w._remaining_worker_wall_seconds()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("run wall deadline exceeded")
    return remaining


def _sleep_with_hf_deadline(delay: float) -> bool:
    remaining = _require_hf_deadline_allowance()
    sleep_for = delay if remaining is None else min(delay, remaining)
    if sleep_for > 0:
        time.sleep(sleep_for)
    remaining = _w._remaining_worker_wall_seconds()
    return remaining is None or remaining > 0


def _hf_upload(do_upload, repo_subpath: str, required: bool, label: str) -> bool:
    """HF upload loop: retries + raises on required artifacts; warn-only on optional.

    Returns True when a commit landed (or HF_REPO is unset), False on best-effort failure.
    """
    if not _w.HF_REPO:
        return True
    attempts = 3 if required else 1
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            _require_hf_deadline_allowance()
            do_upload()
            return True
        except Exception as e:
            last_err = e
            detail = sanitize_diagnostic(e, limit=500)
            if required and attempt + 1 < attempts:
                print(f"{label} retry {attempt + 1}/{attempts}: {detail}")
                try:
                    if not _sleep_with_hf_deadline(5 * (attempt + 1)):
                        break
                except Exception:
                    break
                continue
            if not required:
                print(f"{label} warn: {detail}")
                return False
            break
    if required:
        detail = sanitize_diagnostic(last_err, limit=500)
        raise RetriableInfraError(
            f"required upload of {repo_subpath!r} failed: {detail}"
        ) from last_err
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


_OPTIONAL_UPLOAD_FLUSH_TIMEOUT_S = 300.0
_REQUIRED_FINAL_UPLOAD_RESERVE_S = 60.0
_OPTIONAL_UPLOAD_STAGE_ROOT = "/tmp/flash-optional-uploads"
_RESUME_CHECKPOINT_UPLOAD_LOCK = threading.Lock()
_FICLONE = 0x40049409


@contextlib.contextmanager
def _resume_checkpoint_upload_slot(timeout_s: float | None = None):
    if timeout_s is None:
        acquired = _RESUME_CHECKPOINT_UPLOAD_LOCK.acquire()
    else:
        acquired = _RESUME_CHECKPOINT_UPLOAD_LOCK.acquire(timeout=max(0.0, timeout_s))
    try:
        yield acquired
    finally:
        if acquired:
            _RESUME_CHECKPOINT_UPLOAD_LOCK.release()


@dataclass(frozen=True)
class _OptionalUpload:
    sequence: int
    label: str
    staged_dir: str
    run: Callable[[], None]
    on_coalesce: Callable[[_OptionalUpload], None] | None = None


class _SingleSlotUploader:
    """run one optional upload at a time with one newest-wins pending slot."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._in_flight: _OptionalUpload | None = None
        self._pending: _OptionalUpload | None = None
        self._next_sequence = 0
        self._thread: threading.Thread | None = None

    def enqueue(
        self,
        label: str,
        staged_dir: str,
        run: Callable[[], None],
        on_coalesce: Callable[[_OptionalUpload], None] | None = None,
    ) -> None:
        replaced: _OptionalUpload | None = None
        thread: threading.Thread | None = None
        with self._condition:
            self._next_sequence += 1
            task = _OptionalUpload(self._next_sequence, label, staged_dir, run, on_coalesce)
            replaced, self._pending = self._pending, task
            if self._thread is None:
                thread = threading.Thread(
                    target=self._drain,
                    name="flash-optional-uploader",
                    daemon=True,
                )
                self._thread = thread
            self._condition.notify_all()
        if replaced is not None:
            print(
                f"[upload] coalesced optional {replaced.label} into newer {label} "
                f"(sequence {task.sequence})"
            )
            if replaced.on_coalesce is not None:
                # the coalesce hook takes over cleanup of the replaced staged tree.
                replaced.on_coalesce(replaced)
            else:
                shutil.rmtree(replaced.staged_dir, ignore_errors=True)
        if thread is not None:
            thread.start()

    def _drain(self) -> None:
        while True:
            with self._condition:
                if self._pending is None:
                    self._thread = None
                    self._condition.notify_all()
                    return
                task, self._pending = self._pending, None
                self._in_flight = task
            try:
                task.run()
            except Exception as e:
                print(f"optional upload warn ({task.label}): {sanitize_diagnostic(e, limit=500)}")
            finally:
                shutil.rmtree(task.staged_dir, ignore_errors=True)
                with self._condition:
                    self._in_flight = None
                    self._condition.notify_all()

    def flush(self, timeout_s: float) -> bool:
        """wait for the in-flight and newest pending optional uploads to finish."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while self._in_flight is not None or self._pending is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


class _FifoUploader:
    """run uploads one at a time in FIFO order, never dropping a pending task.

    unlike _SingleSlotUploader this keeps every enqueued upload, so per-step artifacts that must
    each land (deployable adapters) are not coalesced away by a newer save.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._in_flight: _OptionalUpload | None = None
        self._pending: list[_OptionalUpload] = []
        self._next_sequence = 0
        self._thread: threading.Thread | None = None

    def enqueue(self, label: str, staged_dir: str, run: Callable[[], None]) -> None:
        thread: threading.Thread | None = None
        with self._condition:
            self._next_sequence += 1
            self._pending.append(_OptionalUpload(self._next_sequence, label, staged_dir, run))
            if self._thread is None:
                thread = threading.Thread(
                    target=self._drain, name="flash-deployable-uploader", daemon=True
                )
                self._thread = thread
            self._condition.notify_all()
        if thread is not None:
            thread.start()

    def _drain(self) -> None:
        while True:
            with self._condition:
                if not self._pending:
                    self._thread = None
                    self._condition.notify_all()
                    return
                task = self._pending.pop(0)
                self._in_flight = task
            try:
                task.run()
            except Exception as e:
                print(f"deployable upload warn ({task.label}): {sanitize_diagnostic(e, limit=500)}")
            finally:
                shutil.rmtree(task.staged_dir, ignore_errors=True)
                with self._condition:
                    self._in_flight = None
                    self._condition.notify_all()

    def flush(self, timeout_s: float) -> bool:
        """wait for the in-flight and all pending FIFO uploads to finish."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._condition:
            while self._in_flight is not None or self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True


_OPTIONAL_CHECKPOINT_UPLOADER = _SingleSlotUploader()
_OPTIONAL_AUX_UPLOADER = _SingleSlotUploader()
_OPTIONAL_DEPLOYABLE_UPLOADER = _FifoUploader()
_DEBUG_UPLOAD_LOCK = threading.Lock()


def _copy_snapshot_file(source: str, destination: str) -> str:
    """reflink a file when supported, otherwise make an independent copy."""
    if fcntl is None:
        shutil.copy2(source, destination, follow_symlinks=False)
        return destination
    try:
        with open(source, "rb") as src, open(destination, "wb") as dst:
            fcntl.ioctl(dst.fileno(), _FICLONE, src.fileno())
        shutil.copystat(source, destination, follow_symlinks=False)
    except OSError:
        with contextlib.suppress(FileNotFoundError):
            os.remove(destination)
        shutil.copy2(source, destination, follow_symlinks=False)
    return destination


def _stage_optional_file(source: str, label: str) -> tuple[str, str]:
    os.makedirs(_OPTIONAL_UPLOAD_STAGE_ROOT, exist_ok=True)
    staged_dir = tempfile.mkdtemp(prefix=f"{label}-", dir=_OPTIONAL_UPLOAD_STAGE_ROOT)
    staged_path = os.path.join(staged_dir, os.path.basename(source))
    try:
        _copy_snapshot_file(source, staged_path)
    except Exception:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    return staged_dir, staged_path


def _stage_optional_directory(source: str, label: str) -> tuple[str, str]:
    os.makedirs(_OPTIONAL_UPLOAD_STAGE_ROOT, exist_ok=True)
    staged_dir = tempfile.mkdtemp(prefix=f"{label}-", dir=_OPTIONAL_UPLOAD_STAGE_ROOT)
    staged_path = os.path.join(staged_dir, "artifact")
    try:
        shutil.copytree(
            source,
            staged_path,
            copy_function=_copy_snapshot_file,
        )
    except Exception:
        shutil.rmtree(staged_dir, ignore_errors=True)
        raise
    return staged_dir, staged_path


def _bounded_optional_flush_timeout(timeout_s: float) -> float:
    timeout_s = max(0.0, timeout_s)
    remaining = _w._remaining_worker_wall_seconds()
    if remaining is None:
        return timeout_s
    return min(timeout_s, max(0.0, remaining - _REQUIRED_FINAL_UPLOAD_RESERVE_S))


def _checkpoint_upload_lock_timeout() -> float:
    return _bounded_optional_flush_timeout(_OPTIONAL_UPLOAD_FLUSH_TIMEOUT_S)


def flush_optional_uploads(timeout_s: float = _OPTIONAL_UPLOAD_FLUSH_TIMEOUT_S) -> bool:
    """bounded best-effort flush that preserves time for required terminal artifacts."""
    timeout_s = _bounded_optional_flush_timeout(timeout_s)
    started = time.monotonic()
    checkpoints_flushed = _OPTIONAL_CHECKPOINT_UPLOADER.flush(timeout_s)
    remaining = max(0.0, timeout_s - (time.monotonic() - started))
    deployables_flushed = _OPTIONAL_DEPLOYABLE_UPLOADER.flush(remaining)
    remaining = max(0.0, timeout_s - (time.monotonic() - started))
    aux_flushed = _OPTIONAL_AUX_UPLOADER.flush(remaining)
    return checkpoints_flushed and deployables_flushed and aux_flushed


def upload_debug_jsonl(name: str, rows: list[dict], *, keep_last: int = 200) -> None:
    """append bounded debug rows and enqueue an immutable optional upload snapshot."""
    if not rows or not _w.HF_REPO:
        return
    repo_name = os.path.basename(name if name.endswith(".jsonl") else f"{name}.jsonl")
    path = os.path.join("/tmp", repo_name)
    try:
        with _DEBUG_UPLOAD_LOCK:
            existing: list[str] = []
            # open() is evaluated before suppress() enters, so handle absence explicitly.
            try:
                with open(path) as f:
                    existing = f.readlines()[-keep_last:]
            except FileNotFoundError:
                pass
            with open(path, "w") as f:
                f.writelines(existing)
                for row in rows:
                    f.write(json.dumps(row, default=str, ensure_ascii=True, sort_keys=True) + "\n")
            staged_dir, staged_path = _stage_optional_file(path, "debug-jsonl")
            _OPTIONAL_AUX_UPLOADER.enqueue(
                f"debug {repo_name}",
                staged_dir,
                lambda: _w.hf_upload_file(staged_path, repo_name),
            )
    except Exception as e:
        print(f"debug upload warn ({repo_name}): {sanitize_diagnostic(e, limit=500)}")


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


def hf_resume_checkpoint(fail_closed: bool = False, revision: str | None = None) -> str | None:
    """Download the latest streamed trainer checkpoint for this run, or return none."""
    required = bool(revision)
    strict = bool(fail_closed or required)
    if not _w.HF_REPO:
        if required:
            raise RetriableInfraError("required resume checkpoint has no artifact repository")
        return None
    base = os.path.join("/tmp/resume", hf_prefix(), "checkpoint")
    try:
        from huggingface_hub import snapshot_download

        from flash.engine.worker.heartbeat import liveness_heartbeat

        # remove prior local materialization so pinned absence cannot reuse a stale checkpoint.
        shutil.rmtree(base, ignore_errors=True)
        # resume checkpoints carry the full optimizer state (multi-gb); keep the heartbeat fresh.
        with liveness_heartbeat("checkpoint_prefetching"):
            _require_hf_deadline_allowance()
            snapshot_download(
                repo_id=_w.HF_REPO,
                repo_type="dataset",
                allow_patterns=[f"{hf_prefix()}/checkpoint/**"],
                local_dir="/tmp/resume",
                token=os.environ.get("HF_TOKEN"),
                revision=revision,
            )
        if not os.path.isdir(base):
            if required:
                detail = f" at revision {revision}" if revision else ""
                raise RetriableInfraError(f"required resume checkpoint is missing{detail}")
            return None
        candidates: list[tuple[int, str]] = []
        for name in os.listdir(base):
            if not name.startswith("checkpoint-"):
                continue
            suffix = name[len("checkpoint-") :]
            if not suffix.isdigit():
                continue
            step = int(suffix)
            if step > 0 and suffix == str(step):
                candidates.append((step, name))
        if not candidates:
            if required:
                detail = f" at revision {revision}" if revision else ""
                raise RetriableInfraError(f"required resume checkpoint is missing{detail}")
            return None
        _, latest = max(candidates)
        path = os.path.join(base, latest)
        print(f"[resume] found streamed checkpoint: {path}")
        return path
    except Exception as e:
        print("hf_resume_checkpoint warn:", e)
        if strict:
            if isinstance(e, RetriableInfraError):
                raise
            raise RetriableInfraError(f"required resume checkpoint fetch failed: {e}") from e
        return None


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


def _prefetch_error_is_retriable(exc: BaseException) -> bool:
    import httpx
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import Timeout as RequestsTimeout

    if isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError)):
        return False
    if isinstance(exc, LocalEntryNotFoundError):
        return True
    if isinstance(exc, EntryNotFoundError):
        return False
    if isinstance(
        exc,
        (
            RequestsConnectionError,
            RequestsTimeout,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, HfHubHTTPError):
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
        if status in {401, 403, 404}:
            return False
        return status is None or status == 429 or (isinstance(status, int) and 500 <= status <= 599)
    return False


def _is_commit_sha(value: str) -> bool:
    """True when value is a full 40-hex-char git commit id (an immutable HF revision)."""
    return len(value) == 40 and all(c in "0123456789abcdef" for c in value.lower())


def resolve_cached_model_commit(model_id: str, revision: str = "") -> str:
    """Best-effort, offline lookup of the immutable base-model commit already in the HF cache.

    Returns the 40-hex commit that (model_id, revision) resolved to when training loaded the
    weights, or "" when it cannot be determined. Lets a run record its provable base-weight
    identity even when a mutable ref (branch/tag) was pinned. Never raises, never hits the network.
    """
    from huggingface_hub import snapshot_download

    tried: set[str | None] = set()
    for cache_dir in (None, _shared_weight_cache_dir()):
        if cache_dir in tried:
            continue
        tried.add(cache_dir)
        try:
            snapshot_dir = snapshot_download(
                repo_id=model_id,
                revision=revision or None,
                local_files_only=True,
                cache_dir=cache_dir,
            )
        except Exception:
            continue
        commit = os.path.basename(os.path.normpath(snapshot_dir))
        if _is_commit_sha(commit):
            return commit
    return ""


def write_base_model_provenance(adapter_dir: str, model_id: str, model_revision: str = "") -> None:
    """Record the resolved immutable base-model commit next to the saved adapter.

    Writes base_model_provenance.json into adapter_dir so the base weights a run trained on are
    provable from the uploaded adapter, even when a mutable ref was pinned. Best-effort on the
    resolved commit (null when the cache cannot be read); the record itself always lands.
    """
    payload = {
        "model_id": model_id,
        "requested_revision": model_revision or None,
        "resolved_commit": resolve_cached_model_commit(model_id, model_revision) or None,
    }
    os.makedirs(adapter_dir, exist_ok=True)
    with open(os.path.join(adapter_dir, "base_model_provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


class _SnapshotWeightsMissing(RuntimeError):
    """A forced re-download still produced a snapshot without model weights."""


def _snapshot_has_weights(snapshot_dir: str) -> bool:
    """True when a downloaded snapshot contains resolvable model weights (all indexed shards)."""
    weight_names = ("model", "pytorch_model", "tf_model", "flax_model")

    def _resolves(name: str) -> bool:
        path = os.path.join(snapshot_dir, name)
        return os.path.isfile(os.path.realpath(path))

    try:
        entries = os.listdir(snapshot_dir)
        # sharded checkpoints: the index enumerates every required shard; a partial download with
        # SOME shards present must not pass. validate all indexed shards resolve.
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            if index_name in entries:
                try:
                    with open(os.path.join(snapshot_dir, index_name)) as f:
                        index = json.load(f)
                    shards = set((index.get("weight_map") or {}).values())
                except (OSError, ValueError):
                    return False
                return bool(shards) and all(_resolves(shard) for shard in shards)
        for name in entries:
            if not name.endswith((".safetensors", ".bin")):
                continue
            # exclude non-weight .bin artifacts (tokenizer.bin, training_args.bin, adapters)
            if not name.startswith(weight_names):
                continue
            # HF cache entries are symlinks into blobs/: a dangling link (partial download) must
            # not count as weights present.
            if _resolves(name):
                return True
    except OSError:
        return False
    return False


def prefetch_model(model_id: str, revision: str = "") -> float:
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

        def _download() -> None:
            _require_hf_deadline_allowance()
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=shared_hub,
                ignore_patterns=["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
                **model_revision_kwargs(revision),
            )
            # a shared-volume snapshot can be stale/partial (e.g. a serving preload that only
            # warmed configs): snapshot_download returns it as a cache hit without weights, and
            # the trainer then fails offline with "no pytorch_model.bin or model.safetensors".
            # validate weights exist before trusting the hit; one forced re-download repairs it.
            if (
                isinstance(local_path, str)
                and os.path.isdir(local_path)
                and not _snapshot_has_weights(local_path)
            ):
                print(
                    f"prefetch_model: cached snapshot for {model_id} has no weight files; re-downloading"
                )
                local_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=shared_hub,
                    ignore_patterns=[
                        "*.pth",
                        "*.gguf",
                        "original/*",
                        "*.onnx",
                        "*.msgpack",
                        "*.h5",
                    ],
                    force_download=True,
                    **model_revision_kwargs(revision),
                )
                if (
                    isinstance(local_path, str)
                    and os.path.isdir(local_path)
                    and not _snapshot_has_weights(local_path)
                ):
                    raise _SnapshotWeightsMissing(
                        f"model snapshot for {model_id} has no weight files even after a forced "
                        "re-download; the repo layout is unsupported or the cache volume is corrupt"
                    )
            if shared_hub:
                _link_base_model_into_ephemeral_cache(model_id, shared_hub)

        if revision:
            try:
                _download()
            except Exception as e:
                if _prefetch_error_is_retriable(e):
                    detail = sanitize_diagnostic(e, limit=500)
                    raise RetriableInfraError(f"pinned model prefetch failed: {detail}") from e
                raise
        else:
            try:
                _download()
            except _SnapshotWeightsMissing:
                # a FORCED re-download still had no weights — swallowing it would let the trainer
                # fail later with a far more confusing offline error. propagate.
                raise
            except Exception as e:
                # transient fetch errors stay non-fatal on the default revision: the trainer's own
                # cache lookup may still succeed (warm cache), and prefetch is best-effort there.
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
)


def _has_deployable_adapter(ckpt_dir: str) -> bool:
    """Return True if ckpt_dir has a loadable LoRA adapter (config + weights)."""
    return os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json")) and any(
        os.path.isfile(os.path.join(ckpt_dir, w)) for w in ADAPTER_WEIGHT_FILES
    )


def _write_deployable_provenance(ckpt_dir: str) -> None:
    spec = getattr(_w, "JOB_SPEC", None)
    if spec is not None and spec.model:
        write_base_model_provenance(ckpt_dir, spec.model, getattr(spec, "model_revision", "") or "")


def publish_deployable_checkpoint(
    ckpt_dir: str,
    step: int,
    *,
    retries: int = 1,
    backoff_s: float = 0.0,
    required: bool = False,
    _provenance_ready: bool = False,
    _emit_heartbeat: bool = True,
) -> str | None:
    """Mirror a trainer checkpoint's LoRA adapter to a stable per-step path.

    Periodic saves remain best-effort. ``required=True`` fails loudly when an exact required save
    cannot be published.
    """
    if not _w.HF_REPO:
        if required:
            raise RequiredSaveError(f"required save step {step} has no artifact repository")
        return None
    if not _has_deployable_adapter(ckpt_dir):
        if required:
            raise RequiredSaveError(
                f"required save step {step} has no deployable adapter in {ckpt_dir}"
            )
        return None
    # staged optional checkpoints already carry provenance and must remain immutable during upload.
    if not _provenance_ready:
        _write_deployable_provenance(ckpt_dir)
    subfolder = f"{hf_prefix()}/checkpoints/step-{step}/adapter"
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            _require_hf_deadline_allowance()
            _w.hf_api().upload_folder(
                folder_path=ckpt_dir,
                path_in_repo=subfolder,
                repo_id=_w.HF_REPO,
                repo_type="dataset",
                ignore_patterns=list(_CHECKPOINT_TRAINER_STATE),
            )
            if _emit_heartbeat:
                _w.heartbeat("checkpoint_deployable", step=step, subfolder=subfolder)
            return subfolder
        except Exception as e:
            last_error = e
            print(f"[ckpt] deployable publish warn (step {step}):", e)
            if attempt + 1 < attempts:
                try:
                    if not _sleep_with_hf_deadline(backoff_s * (attempt + 1)):
                        break
                except Exception:
                    break
    if required:
        raise RetriableInfraError(
            f"required save step {step} deployable upload failed"
        ) from last_error
    return None


# Retry/backoff for each synchronous checkpoint upload. `on_save` BLOCKS the training loop on the
# upload, so a transient HF error is retried until the step lands rather than costing the step.
_CKPT_UPLOAD_RETRIES = 3
_CKPT_UPLOAD_BACKOFF_S = 5.0


def _deployable_adapter_on_hf(step: int) -> bool:
    """True when a required step's deployable adapter is durably present on hf.

    Resume credits a required save only after confirming its published adapter exists, so
    on_train_end verifies the durability guarantee against hf instead of assuming it from the
    restored step counter (a pre-resume worker could have advanced past the step without ever
    landing its deployable). publish_deployable_checkpoint uploads the adapter folder in a single
    atomic upload_folder commit, so the config marker's presence implies the whole folder landed.

    Raises RetriableInfraError when hf cannot be reached: a transient lookup outage must retry the
    resume, not be misread as a permanently-missing required save. file_exists returns False cleanly
    for a genuinely absent file (that stays uncredited and fails completeness in on_train_end).
    """
    if not _w.HF_REPO:
        return False
    marker = f"{hf_prefix()}/checkpoints/step-{step}/adapter/adapter_config.json"
    try:
        return bool(
            _w.hf_api().file_exists(repo_id=_w.HF_REPO, filename=marker, repo_type="dataset")
        )
    except Exception as e:
        raise RetriableInfraError(f"could not verify required save step {step} on hf") from e


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
    """Delete older ``{prefix}/checkpoint/checkpoint-N`` directories.

    The streamed resume checkpoint is meant to be latest-only, but ``upload_folder``'s delete_patterns
    are matched relative to path_in_repo (the per-step dir), so they can never reach sibling step dirs.
    Only lower steps are stale: an older upload can finish after a newer one, and must never delete the
    newer checkpoint. A later upload removes any lower directory left by that race. The deployable tree
    (``{prefix}/checkpoints/...``, plural) has a different prefix and is untouched.
    """
    if not _w.HF_REPO:
        return
    api = _w.hf_api()
    base = f"{hf_prefix()}/checkpoint/"
    try:
        _require_hf_deadline_allowance()
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
        if seg.startswith("checkpoint-") and n.isdigit() and int(n) < keep_step:
            stale.add(f"{base}{seg}")
    for folder in sorted(stale):
        try:
            _require_hf_deadline_allowance()
            api.delete_folder(path_in_repo=folder, repo_id=_w.HF_REPO, repo_type="dataset")
        except Exception as e:
            print(f"ckpt prune warn ({folder}):", e)
            break


def upload_resume_checkpoint(
    step: int,
    ckpt_dir: str,
    *,
    before_upload=None,
    after_upload=None,
    skip_upload=None,
    emit_heartbeat: bool = True,
    lock_timeout_s: float | None = None,
) -> bool:
    """synchronously stream one full-state resume checkpoint and ordered companion artifacts."""
    if not _w.HF_REPO:
        return True

    from flash.engine.worker.heartbeat import liveness_heartbeat

    before_completed = before_upload is None
    resume_completed = False
    after_completed = after_upload is None
    with _resume_checkpoint_upload_slot(lock_timeout_s) as acquired:
        if not acquired:
            print(f"[ckpt] step {step} upload skipped; another checkpoint upload is still active")
            return False
        if skip_upload is not None and skip_upload():
            return True
        heartbeat_context = (
            liveness_heartbeat(
                "checkpoint_uploading", progress=lambda: step, progress_step=True, keepalive=True
            )
            if emit_heartbeat
            else contextlib.nullcontext()
        )
        with heartbeat_context:
            for attempt in range(_CKPT_UPLOAD_RETRIES):
                failure_stage = "resume"
                try:
                    if not before_completed:
                        failure_stage = "before"
                        before_upload()
                        before_completed = True
                    if not resume_completed:
                        failure_stage = "resume"
                        _require_hf_deadline_allowance()
                        _w.hf_api().upload_folder(
                            folder_path=ckpt_dir,
                            path_in_repo=f"{hf_prefix()}/checkpoint/checkpoint-{step}",
                            repo_id=_w.HF_REPO,
                            repo_type="dataset",
                        )
                        # prune only after the atomic folder commit lands, so the prior complete save
                        # survives every failed replacement upload.
                        _prune_stale_resume_checkpoints(step)
                        resume_completed = True
                    if not after_completed:
                        failure_stage = "after"
                        after_upload()
                        after_completed = True
                    if emit_heartbeat:
                        failure_stage = "heartbeat"
                        _w.heartbeat("checkpoint_uploaded", step=step)
                    return True
                except RequiredSaveError:
                    raise
                except Exception as e:
                    if attempt + 1 < _CKPT_UPLOAD_RETRIES:
                        print(
                            f"[ckpt] step {step} upload retry "
                            f"{attempt + 1}/{_CKPT_UPLOAD_RETRIES}: {e}"
                        )
                        try:
                            if not _sleep_with_hf_deadline(_CKPT_UPLOAD_BACKOFF_S * (attempt + 1)):
                                break
                        except Exception:
                            break
                    else:
                        print(
                            f"[ckpt] step {step} upload FAILED after "
                            f"{_CKPT_UPLOAD_RETRIES} attempts: {e}"
                        )
                        if emit_heartbeat:
                            with contextlib.suppress(Exception):
                                _w.heartbeat("checkpoint_upload_failed", step=step)
                        if failure_stage in {"before", "after"}:
                            raise
    return False


def make_checkpoint_upload_callback(save_at_steps=()):
    """stream optional saves in the background while keeping required saves synchronous."""
    from transformers import TrainerCallback

    required_steps = frozenset(int(step) for step in save_at_steps)
    deployable_steps: set[int] = set()
    uploaded_steps: set[int] = set()

    def _publish_deployable(
        ckpt_dir: str,
        step: int,
        *,
        provenance_ready: bool = False,
        emit_heartbeat: bool = True,
    ) -> None:
        """publish the trainer checkpoint's adapter without changing its contents."""
        publish_deployable_checkpoint(
            ckpt_dir,
            step,
            required=step in required_steps,
            _provenance_ready=provenance_ready,
            _emit_heartbeat=emit_heartbeat,
        )
        if step in required_steps:
            deployable_steps.add(step)

    def _upload(
        step: int,
        ckpt_dir: str,
        *,
        provenance_ready: bool = False,
        emit_heartbeat: bool = True,
        lock_timeout_s: float | None = None,
    ) -> bool:
        # publish the small durable deployable before the latest-only resume checkpoint.
        def _prepare() -> None:
            if step not in deployable_steps:
                _publish_deployable(
                    ckpt_dir,
                    step,
                    provenance_ready=provenance_ready,
                    emit_heartbeat=emit_heartbeat,
                )

        return upload_resume_checkpoint(
            step,
            ckpt_dir,
            before_upload=_prepare,
            after_upload=lambda: uploaded_steps.add(step),
            skip_upload=lambda: step in uploaded_steps,
            emit_heartbeat=emit_heartbeat,
            lock_timeout_s=lock_timeout_s,
        )

    def _enqueue_optional(step: int, ckpt_dir: str) -> None:
        try:
            _write_deployable_provenance(ckpt_dir)
            staged_dir, staged_checkpoint = _stage_optional_directory(
                ckpt_dir, f"checkpoint-{step}"
            )
        except Exception as e:
            # surface the miss explicitly rather than logging a soft warning and continuing as if
            # the periodic save reached hf.
            print(
                f"[ckpt] step {step} snapshot failed; step not published: "
                f"{sanitize_diagnostic(e, limit=500)}"
            )
            return

        def _publish_coalesced_deployable(
            replaced: _OptionalUpload,
            step: int = step,
            staged_checkpoint: str = staged_checkpoint,
        ) -> None:
            # a newer optional save coalesced this resume checkpoint away; still publish this step's
            # small durable deployable through the non-coalescing fifo path (which owns the staged
            # tree cleanup) so per-step deployables are never dropped.
            _OPTIONAL_DEPLOYABLE_UPLOADER.enqueue(
                f"coalesced deployable step {step}",
                replaced.staged_dir,
                lambda: publish_deployable_checkpoint(
                    staged_checkpoint, step, _provenance_ready=True, _emit_heartbeat=False
                ),
            )

        _OPTIONAL_CHECKPOINT_UPLOADER.enqueue(
            f"checkpoint step {step}",
            staged_dir,
            lambda: _upload(step, staged_checkpoint, provenance_ready=True, emit_heartbeat=False),
            on_coalesce=_publish_coalesced_deployable,
        )

    class _CheckpointUpload(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if int(getattr(state, "global_step", 0) or 0) in required_steps:
                control.should_save = True
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            # resume credits a required save only when its deployable adapter is verified on hf.
            # crediting the restored step alone could accept a save that never reached hf.
            resumed_step = int(getattr(state, "global_step", 0) or 0)
            for step in required_steps:
                if step <= resumed_step and _deployable_adapter_on_hf(step):
                    deployable_steps.add(step)
                    uploaded_steps.add(step)
            return control

        def on_save(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if not _w.HF_REPO:
                if step in required_steps:
                    raise RuntimeError(f"required save step {step} has no artifact repository")
                return
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            if not os.path.isdir(ckpt_dir):
                if step in required_steps:
                    raise RuntimeError(
                        f"required save step {step} has no trainer checkpoint directory"
                    )
                return
            if step not in required_steps:
                _enqueue_optional(step, ckpt_dir)
                return
            if not _upload(
                step,
                ckpt_dir,
                lock_timeout_s=_checkpoint_upload_lock_timeout(),
            ):
                raise RetriableInfraError(
                    f"required save step {step} full-state checkpoint was not durably published"
                )

        def on_train_end(self, args, state, control, **kwargs):
            if not _w.HF_REPO:
                if required_steps:
                    raise RuntimeError("required saves have no artifact repository")
                return
            latest = _latest_checkpoint_dir(args.output_dir)
            if latest is not None:
                step, ckpt_dir = latest
                should_flush = not required_steps or step in required_steps
                if (
                    should_flush
                    and step not in uploaded_steps
                    and not _upload(
                        step,
                        ckpt_dir,
                        lock_timeout_s=_checkpoint_upload_lock_timeout(),
                    )
                ):
                    if step in required_steps:
                        raise RetriableInfraError(
                            f"required save step {step} full-state checkpoint was not durably published"
                        )
                    print(
                        f"[ckpt] final resume checkpoint step {step} not durable on HF after retries; "
                        "the deployable adapter save is preserved."
                    )
            missing_required = sorted(required_steps - deployable_steps)
            if missing_required:
                raise RuntimeError(f"required saves were not durably published: {missing_required}")

    return _CheckpointUpload()
