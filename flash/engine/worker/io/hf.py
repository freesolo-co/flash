"""hf artifact channel for code delivery, adapters, metrics, and checkpoints."""

from __future__ import annotations

import contextlib
import os
import shutil
import threading
import time
from collections.abc import Callable

import flash.engine.worker.io.heartbeat as _worker_heartbeat
import flash.engine.worker.runtime.state as _worker_state
from flash._internal.diagnostics import sanitize_diagnostic
from flash.adapters.artifacts import attempt_scoped_artifact_name, has_loadable_adapter_weights
from flash.engine.profiling.tokenizer import (  # noqa: F401
    load_tokenizer,
    model_revision_kwargs,
)

# `gpu_diagnostics` has no call site here since prefetch moved to `.prefetch`, but it is kept
# imported on purpose: the prefetch tests patch `hf.gpu_diagnostics` and the prefetcher reads it
# back through this module. removing it as unused would silently break them.
from flash.engine.worker.perf import (  # noqa: F401
    RetriableInfraError,
    gpu_diagnostics,
)
from flash.teacher.retry_contract import (
    canonical_opd_optimizer_start_json,
    opd_optimizer_start_marker_path,
)


class RequiredSaveError(RuntimeError):
    """permanent exact-save contract failure."""


def error_artifact_name(mode: str, attempt: int = 0) -> str:
    """Per-mode, per-attempt error filename (e.g. error_sft_attempt0.txt). Attempt-scoped so a prior
    attempt's stale traceback can't be mistaken for the current attempt's crash on a retry host-loss."""
    return attempt_scoped_artifact_name("error", mode, attempt)


def ray_log_artifact_name(mode: str, attempt: int = 0) -> str:
    """Per-mode, per-attempt name for ray's own failure logs (e.g. raylogs_rl_attempt0.txt).

    Attempt-scoped for the same reason the traceback beside it is: ``hf_prefix()`` is per-RUN, not
    per-attempt, so an unscoped name would let a retry's ray logs overwrite the attempt that actually
    reproduced the raylet failure -- the one case this artifact exists to explain."""
    return attempt_scoped_artifact_name("raylogs", mode, attempt)


def _disable_xet_upload_staging() -> None:
    """Route this process's hf uploads through the streaming lfs path instead of xet.

    `hf_xet` is an unconditional dependency of `huggingface-hub` on every arch flash runs on, and
    `is_xet_available()` turns it on merely because it imports (utils/_runtime.py) -- so the xet
    path is the default for `upload_folder`. Xet chunks and dedups through a local cache rooted at
    `HF_XET_CACHE` (default `$HF_HOME/xet`), i.e. the SAME container disk holding the checkpoint
    being uploaded. The legacy lfs path streams from the source file handle instead
    (`CommitOperationAdd.as_file` -> `http_backoff(data=fileobj)`), so it adds no second on-disk
    copy of a ~60 GB fsdp save.

    The verl child, the rl child, and the model-merger subprocess already pin this. The parent is
    the process that actually uploads checkpoints, and it was the one left reading the default --
    which is where a real 35b run hit `No space left on device` under
    `.../huggingface/xet/.../staging/`.

    How much that staging holds at peak is not knowable from source (it lives in the compiled
    `hf_xet` extension), which is the point: the streaming path needs no such assumption. The
    checkpoint is already on this disk, so uploading it should not require room for a second copy
    of unknown size.

    `HF_HUB_DISABLE_XET` is captured into a module constant when `huggingface_hub.constants` is
    imported, so this must run before the first import. Every `huggingface_hub` import in the
    worker is function-local, so calling this during worker startup is early enough.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def hf_prefix() -> str:
    return f"{_worker_state.PHASE}/{_worker_state.RUN_ID}"


def publish_opd_optimizer_start_marker() -> None:
    """Synchronously publish the first-update mutation marker before optimizer.step()."""
    if not isinstance(_worker_state.HF_REPO, str) or not _worker_state.HF_REPO.strip():
        raise RuntimeError("opd optimizer-start marker requires a private HF repository")
    marker_path = opd_optimizer_start_marker_path(_worker_state.RUN_ID, _worker_state.ATTEMPT)
    local_path = f"/tmp/opd-optimizer-start-attempt-{_worker_state.ATTEMPT}.json"
    payload = canonical_opd_optimizer_start_json(
        run_id=_worker_state.RUN_ID,
        attempt=_worker_state.ATTEMPT,
        seed=_worker_state.SEED,
    )
    with open(local_path, "wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    try:
        _require_hf_deadline_allowance()
        hf_api().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=marker_path,
            repo_id=_worker_state.HF_REPO,
            repo_type="dataset",
        )
    except Exception as error:
        detail = sanitize_diagnostic(error, limit=500)
        raise RetriableInfraError(
            f"required upload of OPD optimizer-start marker failed: {detail}"
        ) from error


def _require_hf_deadline_allowance() -> float | None:
    remaining = _worker_state._remaining_worker_wall_seconds()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("run wall deadline exceeded")
    return remaining


def _sleep_with_hf_deadline(delay: float) -> bool:
    remaining = _require_hf_deadline_allowance()
    sleep_for = delay if remaining is None else min(delay, remaining)
    if sleep_for > 0:
        time.sleep(sleep_for)
    remaining = _worker_state._remaining_worker_wall_seconds()
    return remaining is None or remaining > 0


def _hf_upload(do_upload, repo_subpath: str, required: bool, label: str) -> bool:
    """HF upload loop: retries + raises on required artifacts; warn-only on optional.

    Returns True when a commit landed (or HF_REPO is unset), False on best-effort failure.
    """
    if not _worker_state.HF_REPO:
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
        lambda: hf_api().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{hf_prefix()}/{repo_subpath}",
            repo_id=_worker_state.HF_REPO,
            repo_type="dataset",
        ),
        repo_subpath,
        required,
        "hf_upload_file",
    )


_RESUME_CHECKPOINT_UPLOAD_LOCK = threading.Lock()


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


def hf_upload_folder(local_dir: str, repo_subpath: str, required: bool = False) -> bool:
    """Upload a folder to the run's HF prefix. Returns True on success (see ``_hf_upload``)."""
    return _hf_upload(
        lambda: hf_api().upload_folder(
            folder_path=local_dir,
            path_in_repo=f"{hf_prefix()}/{repo_subpath}",
            repo_id=_worker_state.HF_REPO,
            repo_type="dataset",
        ),
        repo_subpath,
        required,
        "hf_upload_folder",
    )


def _highest_resume_candidate(
    base: str, candidates: list[tuple[int, str]], prefer: Callable[[str], bool] | None
) -> str:
    """the candidate name hf_resume_checkpoint stages, given the caller's optional ``prefer``.

    every candidate is already staged on local disk by the snapshot_download above, so evaluating
    ``prefer`` per candidate costs no extra fetch. picks the highest step ``prefer`` accepts and
    falls back to the highest step overall when none do -- the same answer this returned before
    ``prefer`` existed, so a caller with nothing to prefer sees no behaviour change, and the caller's
    own discard log (not this function) is left to explain a restart from zero.
    """
    if prefer is not None:
        for _step, name in sorted(candidates, reverse=True):
            if prefer(os.path.join(base, name)):
                return name
    return max(candidates)[1]


def hf_resume_checkpoint(
    fail_closed: bool = False,
    revision: str | None = None,
    *,
    prefer: Callable[[str], bool] | None = None,
) -> str | None:
    """Download the latest streamed verl checkpoint for this run, or return none.

    ``prefer`` selects the highest downloaded candidate it accepts instead of the highest overall;
    see ``_highest_resume_candidate``. Left unset, behaviour is unchanged from before it existed.
    """
    required = bool(revision)
    strict = bool(fail_closed or required)
    if not _worker_state.HF_REPO:
        if required:
            raise RetriableInfraError("required resume checkpoint has no artifact repository")
        return None
    base = os.path.join("/tmp/resume", hf_prefix(), "checkpoint")
    try:
        from huggingface_hub import snapshot_download

        from flash.engine.worker.io.heartbeat import liveness_heartbeat

        # remove prior local materialization so pinned absence cannot reuse a stale checkpoint.
        shutil.rmtree(base, ignore_errors=True)
        # resume checkpoints carry the full optimizer state (multi-gb); keep the heartbeat fresh.
        with liveness_heartbeat("checkpoint_prefetching"):
            _require_hf_deadline_allowance()
            snapshot_download(
                repo_id=_worker_state.HF_REPO,
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
        latest = _highest_resume_candidate(base, candidates, prefer)
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
    """Return True if ckpt_dir has a loadable LoRA adapter (config + weights).

    Weights via the shared rule rather than the two single-file names, because a save past peft's
    shard size writes ``adapter_model-0000N-of-0000M.<ext>`` plus an index instead. Spelling only the
    single-file names here made this the strict side of a disagreement: serving and export accept the
    sharded save, so a sharded per-step adapter was silently skipped -- or failed a required save --
    for an artifact the rest of the pipeline would have deployed.
    """
    if not os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json")):
        return False
    try:
        return has_loadable_adapter_weights(os.listdir(ckpt_dir))
    except OSError:
        return False


def _write_deployable_provenance(ckpt_dir: str) -> None:
    spec = _worker_state.JOB_SPEC
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
    """Mirror a verl checkpoint's LoRA adapter to a stable per-step path.

    Periodic saves remain best-effort. ``required=True`` fails loudly when an exact required save
    cannot be published.
    """
    if not _worker_state.HF_REPO:
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
            hf_api().upload_folder(
                folder_path=ckpt_dir,
                path_in_repo=subfolder,
                repo_id=_worker_state.HF_REPO,
                repo_type="dataset",
                ignore_patterns=list(_CHECKPOINT_TRAINER_STATE),
            )
            if _emit_heartbeat:
                _worker_heartbeat.heartbeat("checkpoint_deployable", step=step, subfolder=subfolder)
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


# Retry/backoff for each synchronous checkpoint upload. the watcher BLOCKS on the upload, so a
# transient HF error is retried until the step lands rather than costing the step.
_CKPT_UPLOAD_RETRIES = 3
_CKPT_UPLOAD_BACKOFF_S = 5.0


def _deployable_adapter_on_hf(step: int) -> bool:
    """True when a required step's deployable adapter is durably present on hf.

    Resume credits a required save only after confirming its published adapter exists, so the
    final completeness check verifies the durability guarantee against hf instead of assuming it from the
    restored step counter (a pre-resume worker could have advanced past the step without ever
    landing its deployable). publish_deployable_checkpoint uploads the adapter folder in a single
    atomic upload_folder commit, so the config marker's presence implies the whole folder landed.

    Raises RetriableInfraError when hf cannot be reached: a transient lookup outage must retry the
    resume, not be misread as a permanently-missing required save. file_exists returns False cleanly
    for a genuinely absent file (that stays uncredited and fails the final completeness check).
    """
    if not _worker_state.HF_REPO:
        return False
    marker = f"{hf_prefix()}/checkpoints/step-{step}/adapter/adapter_config.json"
    try:
        return bool(
            hf_api().file_exists(
                repo_id=_worker_state.HF_REPO, filename=marker, repo_type="dataset"
            )
        )
    except Exception as e:
        raise RetriableInfraError(f"could not verify required save step {step} on hf") from e


def _prune_stale_resume_checkpoints(keep_step: int) -> None:
    """Delete older ``{prefix}/checkpoint/checkpoint-N`` directories.

    The streamed resume checkpoint is meant to be latest-only, but ``upload_folder``'s delete_patterns
    are matched relative to path_in_repo (the per-step dir), so they can never reach sibling step dirs.
    Only lower steps are stale: an older upload can finish after a newer one, and must never delete the
    newer checkpoint. A later upload removes any lower directory left by that race. The deployable tree
    (``{prefix}/checkpoints/...``, plural) has a different prefix and is untouched.
    """
    if not _worker_state.HF_REPO:
        return
    api = hf_api()
    base = f"{hf_prefix()}/checkpoint/"
    try:
        _require_hf_deadline_allowance()
        files = api.list_repo_files(repo_id=_worker_state.HF_REPO, repo_type="dataset")
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
            api.delete_folder(
                path_in_repo=folder, repo_id=_worker_state.HF_REPO, repo_type="dataset"
            )
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
    if not _worker_state.HF_REPO:
        return True

    from flash.engine.worker.io.heartbeat import liveness_heartbeat

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
                        hf_api().upload_folder(
                            folder_path=ckpt_dir,
                            path_in_repo=f"{hf_prefix()}/checkpoint/checkpoint-{step}",
                            repo_id=_worker_state.HF_REPO,
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
                        _worker_heartbeat.heartbeat("checkpoint_uploaded", step=step)
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
                                # worker stdout is not part of control-plane run logs, so the stage
                                # name alone would otherwise be the entire failure report.
                                _worker_heartbeat.heartbeat(
                                    "checkpoint_upload_failed",
                                    step=step,
                                    checkpoint_failure={
                                        "step": step,
                                        "operation": failure_stage,
                                        "error": sanitize_diagnostic(e, limit=300),
                                    },
                                )
                        if failure_stage in {"before", "after"}:
                            raise
    return False


# re-exported so the prefetch surface stays reachable as `hf.<name>`: the tests patch
# `_shared_weight_cache_dir`, `resolve_cached_model_commit` and `_hf_cache_bytes` here, and the
# worker entry points import `prefetch_model` from here.
from flash.engine.worker.io.prefetch import (  # noqa: E402,F401
    _hf_cache_bytes,
    _prefetch_error_is_retriable,
    _shared_weight_cache_dir,
    _snapshot_has_weights,
    _SnapshotWeightsMissing,
    prefetch_model,
    resolve_cached_model_commit,
    write_base_model_provenance,
)

# the sha predicate lives in the client-safe `envs.loading.loader`; re-exported here because the
# provenance tests patch and assert it as `hf.is_commit_sha`.
from flash.envs.loading.loader import is_commit_sha  # noqa: E402,F401
