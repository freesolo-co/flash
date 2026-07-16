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


def error_artifact_name(mode: str, attempt: int = 0) -> str:
    """Per-mode, per-attempt error filename (e.g. error_sft_attempt0.txt). Attempt-scoped so a prior
    attempt's stale traceback can't be mistaken for the current attempt's crash on a retry host-loss."""
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise ValueError("attempt must be a nonnegative integer")
    if attempt < 0 or attempt > _MAX_ATTEMPT_ID:
        raise ValueError("attempt must be a bounded nonnegative integer")
    return f"error_{mode}_attempt{attempt}.txt"


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


def hf_resume_checkpoint(
    fail_closed: bool = False, revision: str | None = None
) -> str | None:
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
            raise RetriableInfraError(
                f"required resume checkpoint fetch failed: {e}"
            ) from e
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
        return status is None or status == 429 or (
            isinstance(status, int) and 500 <= status <= 599
        )
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
            snapshot_download(
                repo_id=model_id,
                cache_dir=shared_hub,
                ignore_patterns=["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"],
                **model_revision_kwargs(revision),
            )
            if shared_hub:
                _link_base_model_into_ephemeral_cache(model_id, shared_hub)

        if revision:
            try:
                _download()
            except Exception as e:
                if _prefetch_error_is_retriable(e):
                    detail = sanitize_diagnostic(e, limit=500)
                    raise RetriableInfraError(
                        f"pinned model prefetch failed: {detail}"
                    ) from e
                raise
        else:
            try:
                _download()
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
)


def _has_deployable_adapter(ckpt_dir: str) -> bool:
    """Return True if ckpt_dir has a loadable LoRA adapter (config + weights)."""
    return os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json")) and any(
        os.path.isfile(os.path.join(ckpt_dir, w)) for w in ADAPTER_WEIGHT_FILES
    )


def publish_deployable_checkpoint(
    ckpt_dir: str,
    step: int,
    *,
    retries: int = 1,
    backoff_s: float = 0.0,
    required: bool = False,
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
    # a published deployable is a servable adapter, so it carries the same base-weight provenance
    # sidecar as the final adapter and opd saves. companions published straight from a trainer dir
    # (per-step saves, opd resume reconcile) never passed through the final _save_adapter path, so
    # write the sidecar here from the job spec's base model. written into ckpt_dir before upload so
    # it lands inside the same atomic upload_folder commit as the adapter it describes. guard on a
    # non-empty base model so an unspecified base never stamps a misleading empty-model_id sidecar.
    _spec = getattr(_w, "JOB_SPEC", None)
    if _spec is not None and _spec.model:
        write_base_model_provenance(
            ckpt_dir, _spec.model, getattr(_spec, "model_revision", "") or ""
        )
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
        if seg.startswith("checkpoint-") and n.isdigit() and int(n) != keep_step:
            stale.add(f"{base}{seg}")
    for folder in sorted(stale):
        try:
            _require_hf_deadline_allowance()
            api.delete_folder(path_in_repo=folder, repo_id=_w.HF_REPO, repo_type="dataset")
        except Exception as e:
            print(f"ckpt prune warn ({folder}):", e)
            break


def upload_resume_checkpoint(
    step: int, ckpt_dir: str, *, before_upload=None, after_upload=None
) -> bool:
    """synchronously stream one full-state resume checkpoint and ordered companion artifacts."""
    if not _w.HF_REPO:
        return True

    from flash.engine.worker.heartbeat import liveness_heartbeat

    before_completed = before_upload is None
    resume_completed = False
    after_completed = after_upload is None
    with liveness_heartbeat(
        "checkpoint_uploading", progress=lambda: step, progress_step=True, keepalive=True
    ):
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
                    # prune only after the atomic folder commit lands, so the prior complete save survives
                    # every failed replacement upload.
                    _prune_stale_resume_checkpoints(step)
                    resume_completed = True
                if not after_completed:
                    failure_stage = "after"
                    after_upload()
                    after_completed = True
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
                    with contextlib.suppress(Exception):
                        _w.heartbeat("checkpoint_upload_failed", step=step)
                    if failure_stage in {"before", "after"}:
                        raise
    return False


def make_checkpoint_upload_callback(save_at_steps=()):
    """Return a TrainerCallback that streams each save to HF and publishes deployable per-step adapters.

    Uploads are SYNCHRONOUS: on_save blocks the training loop until the checkpoint is durably on
    HF. A save therefore never returns while its upload is still running, so there is never a
    second upload in flight when the next save fires — the "upload busy" contention (and the
    coalescing/dropping it used to cause) cannot happen by construction. Every `save_every` step is
    guaranteed to upload, retrying transient HF errors instead of skipping.
    """
    from transformers import TrainerCallback

    required_steps = frozenset(int(step) for step in save_at_steps)
    deployable_steps: set[int] = set()

    def _publish_deployable(ckpt_dir: str, step: int) -> None:
        """Publish a step's deployable adapter directly from the trainer checkpoint.

        Warm-start CONTINUES the one adapter in place (VL and non-VL) and fresh runs train a single
        adapter, so the trainer checkpoint's adapter IS the deployable — it carries the full policy on
        the catalog base and serves as-is (no merge, no SFT rank-stack recombine).
        """
        publish_deployable_checkpoint(ckpt_dir, step, required=step in required_steps)
        if step in required_steps:
            deployable_steps.add(step)

    uploaded_steps: set[int] = set()

    def _upload(step: int, ckpt_dir: str) -> bool:
        # publish the small durable deployable first, inside the shared upload keepalive.
        def _prepare() -> None:
            if step not in deployable_steps:
                _publish_deployable(ckpt_dir, step)

        if not upload_resume_checkpoint(step, ckpt_dir, before_upload=_prepare):
            return False
        uploaded_steps.add(step)
        return True

    class _CheckpointUpload(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if int(getattr(state, "global_step", 0) or 0) in required_steps:
                control.should_save = True
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            # resume credits a required save only when its deployable adapter is verified on hf.
            # crediting straight from the restored step counter would let on_train_end report a
            # required save as satisfied even if the pre-resume worker never durably uploaded it.
            resumed_step = int(getattr(state, "global_step", 0) or 0)
            for step in required_steps:
                if step <= resumed_step and _deployable_adapter_on_hf(step):
                    deployable_steps.add(step)
                    uploaded_steps.add(step)
            return control

        def on_save(self, args, state, control, **kwargs):
            # SYNCHRONOUS on the training thread: the trainer blocks here until this checkpoint is
            # uploaded. `save_total_limit=1` rotation (which runs earlier, inside this same save's
            # `_save_checkpoint`) deletes the PREVIOUS checkpoint, never the current one — and the
            # next save can't begin until we return — so the dir is stable for the whole upload and
            # no snapshot/queue is needed. Because the upload finishes before the save returns,
            # there is never a concurrent upload for a later save to be "busy" against.
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
            if not _upload(step, ckpt_dir):
                if step in required_steps:
                    raise RetriableInfraError(
                        f"required save step {step} full-state checkpoint was not durably published"
                    )
                print(
                    f"[ckpt] step {step} resume checkpoint not durable on HF after retries; "
                    "the deployable adapter save is preserved and on_train_end will re-flush the latest "
                    "checkpoint."
                )

        def on_train_end(self, args, state, control, **kwargs):
            # Safety net for a final checkpoint the trainer wrote without an on_save (e.g. a
            # load_best_model_at_end / end-of-training save). Synchronous on_save already uploaded
            # every step it saw, so this only publishes an as-yet-unuploaded latest step.
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
                    and not _upload(step, ckpt_dir)
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
