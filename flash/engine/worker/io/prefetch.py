"""Get a base model onto this worker's disk, and record which revision landed.

`prefetch_model` is the whole point: pull the weights before the trainer needs them, preferring a
shared cache volume over a per-pod download, retrying only the failures worth retrying, and
refusing a snapshot that arrived without weights. The rest resolves the commit that was actually
fetched and writes it down, so a later deployment can name the exact base it trained against.

Split out of `flash.engine.worker.io.hf` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import json
import os
import time

from flash._internal.diagnostics import sanitize_diagnostic
from flash.engine.huggingface import hub_error_transience, model_revision_kwargs
from flash.engine.worker.perf import RetriableInfraError
from flash.envs.loader import is_commit_sha


def _hf():
    """The io module, imported lazily because it imports this one.

    Five names are patched as attributes of `io.hf` by the prefetch and provenance tests -- the
    worker proxy `_w`, `gpu_diagnostics`, `_require_hf_deadline_allowance`,
    `_shared_weight_cache_dir` and `resolve_cached_model_commit` -- and every caller of theirs is
    in this file. Resolving them through the parent is what keeps those patches effective; a
    direct call would bind this module's own function and run the real one under the patch.
    """
    from flash.engine.worker.io import hf

    return hf


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
    """HF cache folder for a model repo (``models--org--name``), from the library helper."""
    from huggingface_hub.file_download import repo_folder_name

    return repo_folder_name(repo_id=model_id, repo_type="model")


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
    # generic local os errors remain permanent here; only hub/cache and transport failures retry.
    return hub_error_transience(exc) is True


def resolve_cached_model_commit(model_id: str, revision: str = "") -> str:
    """Best-effort, offline lookup of the immutable base-model commit already in the HF cache.

    Returns the 40-hex commit that (model_id, revision) resolved to when training loaded the
    weights, or "" when it cannot be determined. Lets a run record its provable base-weight
    identity even when a mutable ref (branch/tag) was pinned. Never raises, never hits the network.
    """
    from huggingface_hub import snapshot_download

    tried: set[str | None] = set()
    for cache_dir in (None, _hf()._shared_weight_cache_dir()):
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
        if is_commit_sha(commit):
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
        "resolved_commit": _hf().resolve_cached_model_commit(model_id, model_revision) or None,
    }
    os.makedirs(adapter_dir, exist_ok=True)
    with open(os.path.join(adapter_dir, "base_model_provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


class _SnapshotWeightsMissing(RuntimeError):
    """A forced re-download still produced a snapshot without model weights."""


class _SnapshotOutsideSharedCache(RuntimeError):
    """Hugging Face resolved a shared-cache download outside the mounted cache root."""


def _snapshot_under_cache(snapshot_dir: str, cache_dir: str) -> bool:
    try:
        snapshot = os.path.realpath(snapshot_dir)
        root = os.path.realpath(cache_dir)
        return os.path.commonpath((snapshot, root)) == root
    except (OSError, ValueError):
        return False


def _snapshot_cache_evidence(snapshot_dir: str | None, cache_dir: str | None) -> dict:
    evidence = {
        "shared_mount": bool(cache_dir),
        "snapshot_preexisting": False,
        "snapshot_under_mount": False,
        "weights_present": False,
        "snapshot_relative": None,
    }
    if not snapshot_dir or not cache_dir:
        return evidence
    under_mount = _snapshot_under_cache(snapshot_dir, cache_dir)
    evidence["snapshot_under_mount"] = under_mount
    if under_mount:
        evidence["weights_present"] = _snapshot_has_weights(snapshot_dir)
        evidence["snapshot_relative"] = os.path.relpath(
            os.path.realpath(snapshot_dir), os.path.realpath(cache_dir)
        )
    return evidence


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
    """Pull base-model weights up front and emit verified shared-cache evidence."""
    from huggingface_hub import snapshot_download

    shared_hub = _hf()._shared_weight_cache_dir()
    ignore = ["*.pth", "*.gguf", "original/*", "*.onnx", "*.msgpack", "*.h5"]
    evidence = _snapshot_cache_evidence(None, shared_hub)
    t0 = time.time()
    from flash.engine.worker.io.heartbeat import liveness_heartbeat

    def local_probe() -> None:
        if not shared_hub:
            return
        try:
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=shared_hub,
                ignore_patterns=ignore,
                local_files_only=True,
                **model_revision_kwargs(revision),
            )
        except Exception:
            return
        probed = _snapshot_cache_evidence(path, shared_hub)
        if probed["snapshot_under_mount"] and probed["weights_present"]:
            evidence.update(probed)
            evidence["snapshot_preexisting"] = True

    def download() -> None:
        _hf()._require_hf_deadline_allowance()
        local_path = snapshot_download(
            repo_id=model_id,
            cache_dir=shared_hub,
            ignore_patterns=ignore,
            **model_revision_kwargs(revision),
        )
        if not isinstance(local_path, str) or not os.path.isdir(local_path):
            if shared_hub:
                raise _SnapshotWeightsMissing(
                    f"model snapshot for {model_id} did not resolve to a shared-cache directory"
                )
            return
        if shared_hub and not _snapshot_under_cache(local_path, shared_hub):
            raise _SnapshotOutsideSharedCache(
                f"model snapshot for {model_id} resolved outside the mounted shared cache"
            )
        if not _snapshot_has_weights(local_path):
            print(f"prefetch_model: cached snapshot for {model_id} has no weights; re-downloading")
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=shared_hub,
                ignore_patterns=ignore,
                force_download=True,
                **model_revision_kwargs(revision),
            )
            if not isinstance(local_path, str) or not os.path.isdir(local_path):
                raise _SnapshotWeightsMissing(
                    f"model snapshot for {model_id} did not resolve after forced download"
                )
            if shared_hub and not _snapshot_under_cache(local_path, shared_hub):
                raise _SnapshotOutsideSharedCache(
                    f"model snapshot for {model_id} resolved outside the mounted shared cache"
                )
            if not _snapshot_has_weights(local_path):
                raise _SnapshotWeightsMissing(
                    f"model snapshot for {model_id} has no weights after forced download"
                )
        preexisting = evidence["snapshot_preexisting"]
        evidence.update(_snapshot_cache_evidence(local_path, shared_hub))
        evidence["snapshot_preexisting"] = preexisting
        if shared_hub and not (evidence["snapshot_under_mount"] and evidence["weights_present"]):
            raise _SnapshotOutsideSharedCache(
                f"model snapshot for {model_id} lacks verified shared-cache weights"
            )
        if shared_hub:
            _link_base_model_into_ephemeral_cache(model_id, shared_hub)

    with liveness_heartbeat(
        "model_prefetching", progress=lambda: _hf_cache_bytes(model_id, shared_hub)
    ):
        local_probe()
        if revision:
            try:
                download()
            except Exception as exc:
                if _prefetch_error_is_retriable(exc):
                    detail = sanitize_diagnostic(exc, limit=500)
                    raise RetriableInfraError(f"pinned model prefetch failed: {detail}") from exc
                raise
        else:
            try:
                download()
            except (_SnapshotWeightsMissing, _SnapshotOutsideSharedCache):
                raise
            except Exception as exc:
                print("prefetch_model warn:", exc)
    secs = round(time.time() - t0, 1)
    _hf()._w.heartbeat(
        "model_prefetched",
        model=model_id,
        download_seconds=secs,
        cache=evidence,
        hf_transfer=os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", ""),
        gpu=_hf().gpu_diagnostics(),
    )
    return secs
