"""Turn a verl checkpoint directory into a deployable peft adapter.

verl writes a full training checkpoint per save step, in a layout that differs between its sft and
ppo trainers. These helpers find the right directory inside it, work out which steps are already
complete, and export the one flash actually deploys.

Split out of `flash.engine.worker.backend_common` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

from flash.adapters.fused_experts import (
    fused_expert_lora_tensor_pairs,
    is_non_language_lora_key,
    lora_target_parameters,
    normalize_verl_fused_expert_export,
    validate_fused_expert_adapter_config,
)
from flash.adapters.lora_rank import (
    _rank_for_module,
    lora_tensor_rank_disagrees,
    strict_declared_lora_ranks,
)
from flash.engine.worker.model.lora import (
    _open_safetensors_numpy,
    _read_adapter_tensor_metadata,
)


class MergeDiskHeadroomError(RuntimeError):
    """the merged model this export must write does not fit beside the checkpoint it reads."""


class MergeDiskExhaustedError(RuntimeError):
    """the merge started with room and then ran the disk out partway through writing its output."""


# a genuine short write leaves almost no free space; keep the fallback absolute so a normal partial
# merge is not misclassified merely because its output is large.
_MERGE_DISK_EXHAUSTED_FREE_BYTES = 64 * 1024 * 1024

# quota exhaustion can leave the underlying filesystem reporting free space, so preserve both errno
# and text forms emitted by torch and safetensors.
_DISK_EXHAUSTED_ERRNOS = (errno.ENOSPC, errno.EDQUOT)
_DISK_EXHAUSTED_MARKERS = (
    "no space left on device",
    "errno 28",
    "os error 28",
    "disk quota exceeded",
    "errno 122",
    "os error 122",
)
_SHORT_WRITE_MARKERS = ("unexpected pos",)

# disk size is platform-managed, so checkpoint frequency is the reachable remedy.
_FEWER_CHECKPOINTS_ADVICE = (
    "publish fewer checkpoints: raise train.save_every (save_every == the run's total steps "
    "publishes only the final adapter), or drop entries from train.save_at_steps if the run sets "
    "them -- exact steps override save_every and are always published."
)


def _model_shard_bytes(path: str) -> int:
    """total size of the top-level fsdp model shards the merger reads."""
    try:
        names = os.listdir(path)
    except OSError:
        return 0
    total = 0
    for name in names:
        if _FSDP_SHARD_RE.fullmatch(name) is None:
            continue
        try:
            total += os.stat(os.path.join(path, name)).st_size
        except OSError:
            continue
    return total


def require_merge_headroom(ckpt_actor_dir: str, merge_out: str) -> None:
    """fail before the merger when its full-bf16 output cannot fit beside its input shards."""
    need = _model_shard_bytes(ckpt_actor_dir)
    if need <= 0:
        return
    free = _free_bytes(merge_out)
    if free is None:
        # an unreadable mount is not evidence of exhaustion; let the merger run and report for real.
        return
    if free >= need:
        return
    raise MergeDiskHeadroomError(
        f"cannot export the adapter: merging {ckpt_actor_dir} needs about "
        f"{need / 1e9:.1f} GB beside it but only {free / 1e9:.1f} GB is free. "
        + _FEWER_CHECKPOINTS_ADVICE
    )


def _run_merger(cmd: list[str], env: dict[str, str]) -> None:
    """stream merger output and prefer direct disk evidence over a short-write marker."""
    from flash.engine.worker import backend_common

    disk_line = ""
    short_write_line = ""

    def handle_line(line: str) -> None:
        nonlocal disk_line, short_write_line
        print(line, end="", flush=True)
        lowered = line.lower()
        if not disk_line and any(marker in lowered for marker in _DISK_EXHAUSTED_MARKERS):
            disk_line = line.strip()
        elif not short_write_line and any(marker in lowered for marker in _SHORT_WRITE_MARKERS):
            short_write_line = line.strip()

    merger_env = {**env, "PYTHONUNBUFFERED": "1"}
    return_code = backend_common._run_streaming_verl_subprocess(
        cmd, env=merger_env, on_line=handle_line, errors="replace"
    )
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code, cmd, output=disk_line or short_write_line or None
        )


def _free_bytes(path: str) -> int | None:
    """free bytes on the filesystem holding ``path``, or none when it cannot be read."""
    try:
        return shutil.disk_usage(os.path.dirname(path.rstrip("/")) or ".").free
    except OSError:
        return None


def _error_chain_matches(
    error: BaseException, *, errnos: tuple[int, ...] = (), markers: tuple[str, ...] = ()
) -> bool:
    """whether an error chain contains any requested errno or text marker."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in errnos:
            return True
        for evidence in (
            current,
            getattr(current, "output", None),
            getattr(current, "stderr", None),
        ):
            if isinstance(evidence, bytes):
                evidence = evidence.decode("utf-8", "replace")
            if evidence is not None and any(marker in str(evidence).lower() for marker in markers):
                return True
        current = current.__cause__ or current.__context__
    return False


def _disk_exhausted_error(error: BaseException) -> bool:
    """whether an error chain contains direct volume or quota exhaustion evidence."""
    return _error_chain_matches(
        error, errnos=_DISK_EXHAUSTED_ERRNOS, markers=_DISK_EXHAUSTED_MARKERS
    )


def raise_for_merge_disk_exhaustion(
    error: BaseException, ckpt_actor_dir: str, merge_out: str, *, merger_succeeded: bool = False
) -> None:
    """classify direct disk evidence or a low-space short write before cleanup."""
    direct_evidence = _disk_exhausted_error(error)
    if not direct_evidence and isinstance(error, OSError):
        return
    if merger_succeeded and not direct_evidence:
        return
    short_write_evidence = _error_chain_matches(error, markers=_SHORT_WRITE_MARKERS)
    if not direct_evidence and not short_write_evidence:
        return
    free = _free_bytes(merge_out)
    if not direct_evidence and (free is None or free > _MERGE_DISK_EXHAUSTED_FREE_BYTES):
        return
    free_text = "unknown" if free is None else f"{free / 1e9:.2f} GB"
    raise MergeDiskExhaustedError(
        f"ran out of disk while merging {ckpt_actor_dir} into {merge_out}: "
        f"{free_text} free on that filesystem. publishing a checkpoint materializes the FULL base "
        "model beside the checkpoint it reads, so a small lora adapter still needs room for a whole "
        "model copy. the underlying error is a short write, not a corrupt checkpoint. "
        + _FEWER_CHECKPOINTS_ADVICE
    ) from error


def resolve_checkpoint_actor_dir(step_dir: str) -> str:
    """return the directory inside ``global_step_N`` that holds the saved model + ``huggingface/``.

    verl's two trainers do not agree on this layout, and the difference is not configurable:

    * RL nests per-role, ``global_step_N/actor/`` (``trainer/ppo/ray_trainer.py`` builds
      ``os.path.join(local_global_step_folder, "actor")``).
    * SFT writes the shards straight into ``global_step_N/``
      (``utils/checkpoint/checkpoint_handler.py`` passes ``local_path`` unmodified).
    """
    nested = os.path.join(step_dir, "actor")
    if os.path.isdir(os.path.join(nested, "huggingface")):
        return nested
    if os.path.isdir(os.path.join(step_dir, "huggingface")):
        return step_dir
    # neither marker is present (an interrupted save, or a layout this code has not seen). prefer the
    # nested dir when it exists at all so the error names the RL path the caller most likely wanted.
    return nested if os.path.isdir(nested) else step_dir


def latest_global_step_dir(local_dir: str) -> tuple[str, int]:
    """return (actor_dir, step) for the highest global_step_N checkpoint verl wrote."""
    best_step, best = -1, ""
    if os.path.isdir(local_dir):
        for name in os.listdir(local_dir):
            m = re.fullmatch(r"global_step_(\d+)", name)
            if m and int(m.group(1)) > best_step:
                best_step = int(m.group(1))
                best = resolve_checkpoint_actor_dir(os.path.join(local_dir, name))
    if best_step < 0:
        raise RuntimeError(f"no global_step_N checkpoint found under {local_dir}")
    return best, best_step


def completed_checkpoint_step(local_dir: str) -> int:
    """read verl's completion marker; 0 when it is absent or unreadable.

    verl writes ``global_step_N`` in full and only THEN advances
    ``latest_checkpointed_iteration.txt`` (the file ``stage_verl_resume`` writes above), so gating a
    publish on this marker never reads a half-written checkpoint dir. an unreadable marker reads as
    "nothing completed yet" rather than raising: the watchers poll this on a loop, and a torn read
    mid-write must retry on the next tick, not fail the run.
    """
    tracker = os.path.join(local_dir, "latest_checkpointed_iteration.txt")
    try:
        with open(tracker) as file:
            return int(file.read().strip())
    except (OSError, ValueError):
        return 0


def undiscovered_checkpoint_dirs(
    local_dir: str, completed_step: int, discovered_steps: set[int]
) -> list[tuple[int, str]]:
    """``(step, dir)`` for every completed ``global_step_N`` not yet in ``discovered_steps``, ascending.

    Bounded by ``completed_step`` so a directory verl is still writing is never handed to a
    publisher, and by ``discovered_steps`` so each checkpoint is handed over exactly once.

    Discovery only. A returned step is one nobody has claimed yet; a filtered step was claimed, which
    covers publishing it, intentionally skipping it, coalescing it away and crediting it from a
    previous attempt. Callers must read durability from the lifecycle ledger, never from this filter.
    """
    found: list[tuple[int, str]] = []
    try:
        names = os.listdir(local_dir)
    except OSError:
        return found
    for name in names:
        match = re.fullmatch(r"global_step_(\d+)", name)
        if match is None:
            continue
        step = int(match.group(1))
        path = os.path.join(local_dir, name)
        if step <= completed_step and step not in discovered_steps and os.path.isdir(path):
            found.append((step, path))
    return sorted(found)


# verl stamps the writing topology next to the shards on every save
# (`utils/checkpoint/fsdp_checkpoint_manager.py` writes `FSDPConfig(world_size=...)`, and its own
# model_merger reads the file back). flash reads verl's stamp rather than writing a second one of
# its own, because every checkpoint already on the hub carries it.
VERL_FSDP_CONFIG_FILE = "fsdp_config.json"

# the shards themselves carry the same number: verl saves and loads
# `model_world_size_{W}_rank_{R}.pt`, which is exactly why a cross-count resume cannot open a file.
_FSDP_SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_\d+\.pt")


def checkpoint_world_size(step_dir: str) -> int | None:
    """the number of ranks that wrote ``step_dir``'s fsdp shards, or none when it is unreadable.

    prefers verl's ``fsdp_config.json`` stamp and falls back to the shard filenames, which encode
    the same width. a checkpoint carrying neither is unclassifiable rather than single-rank, so it
    reports none and lets the caller decide.
    """
    actor_dir = resolve_checkpoint_actor_dir(step_dir)
    try:
        with open(os.path.join(actor_dir, VERL_FSDP_CONFIG_FILE), encoding="utf-8") as file:
            decoded = json.load(file)
    except (OSError, ValueError):
        decoded = None
    # decoded json that is null, a list, or a bare scalar is not verl's stamp (a mapping with a
    # "world_size" key); treat it as unreadable rather than let .get() raise, so the shard-filename
    # fallback below still runs instead of the exception escaping to the caller.
    stamped = decoded.get("world_size") if isinstance(decoded, dict) else None
    if isinstance(stamped, int) and not isinstance(stamped, bool) and stamped >= 1:
        return stamped
    try:
        names = os.listdir(actor_dir)
    except OSError:
        return None
    widths = {int(m.group(1)) for m in map(_FSDP_SHARD_RE.fullmatch, names) if m is not None}
    # more than one width means a torn or merged directory: not a topology this code can vouch for.
    return widths.pop() if len(widths) == 1 else None


def resume_checkpoint_is_loadable(resume_dir: str, *, world_size: int) -> bool:
    """whether an attempt running ``world_size`` ranks can load ``resume_dir``'s fsdp shards.

    quiet by design (no logging): ``resume_topology_matches`` below is the loud form that explains a
    discard, but a caller ranking several downloaded candidates (``hf_resume_checkpoint``'s
    ``prefer``) needs to probe many of them before any discard decision, and thus any discard log
    line, has been made.
    """
    return checkpoint_world_size(resume_dir) == world_size


def resume_topology_matches(resume_dir: str, *, world_size: int, job_label: str) -> bool:
    """whether an attempt running ``world_size`` ranks can load ``resume_dir``'s fsdp shards.

    infra recovery deliberately re-allocates a retry across card counts (2x and 4x of the same
    class are distinct rentable shapes), and the worker then launches verl at the new count. verl
    loads exactly ``model_world_size_{world_size}_rank_{rank}.pt``, so a retry that changed shape
    cannot open a single shard: the load dies with a missing-file error that is not oom-shaped, so
    the supervisor burns another retry on it. discarding instead restarts from step 0, which is
    what a failed resume fetch already does, and costs only the steps since the last save.

    an unreadable topology is always a mismatch, with no single-gpu exemption: ``rentable_gpu_counts``
    in ``flash/providers/base.py`` walks every power-of-two card count up to a spec's cap, largest
    first, and the allocator rewrites the retried attempt to whichever count it lands on, so a
    single-gpu retry can follow a multi-gpu attempt that wrote the checkpoint -- the card count this
    attempt runs at is no evidence of the count the checkpoint was written at. a directory with no
    readable topology (no verl stamp, no single consistent shard width) is not one this code can
    vouch for at any world size, and discarding it costs only the same steps-since-last-save the
    mismatch path already pays.
    """
    if resume_checkpoint_is_loadable(resume_dir, world_size=world_size):
        return True
    written = checkpoint_world_size(resume_dir)
    print(
        f"[{job_label}] discarding resume checkpoint {os.path.basename(resume_dir)}: its fsdp "
        f"shards were written at world size {written if written is not None else 'unknown'} but "
        f"this attempt runs at {world_size}; restarting from step 0",
        flush=True,
    )
    return False


def stage_verl_resume(resume_dir: str, local_dir: str, *, job_label: str, world_size: int) -> int:
    """stage a downloaded ``checkpoint-N`` into local_dir where verl looks; return its step.

    the resume artifact is keyed on the run prefix, not the job type, so the control plane hands
    every trainer the same ``checkpoint-N`` layout. verl finds it via
    latest_checkpointed_iteration.txt under trainer.default_local_dir once resume_mode=auto.
    ``job_label`` names the job in the error raised for an unparseable path and in the discard log,
    which is the sole thing the trainers ever varied here.

    returns 0, staging nothing, when the checkpoint's world size does not match ``world_size``: see
    ``resume_topology_matches``. 0 is the fresh-run answer every caller already handles.
    """
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume_dir))
    if match is None:
        raise RuntimeError(f"invalid {job_label} resume checkpoint path {resume_dir!r}")
    step = int(match.group(1))
    if not resume_topology_matches(resume_dir, world_size=world_size, job_label=job_label):
        return 0
    shutil.copytree(resume_dir, os.path.join(local_dir, f"global_step_{step}"), dirs_exist_ok=True)
    with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w") as file:
        file.write(str(step))
    return step


def _largest_file_bytes(checkpoint_dir: str) -> int:
    """the biggest single file under ``checkpoint_dir``, or 0 when it cannot be walked.

    per-file, not total: the artifact store rejects an oversized MEMBER, and a checkpoint whose
    bytes are spread across many shards uploads fine at any total size.
    """
    largest = 0
    for root, _dirs, names in os.walk(checkpoint_dir):
        for name in names:
            try:
                size = os.path.getsize(os.path.join(root, name))
            except OSError:
                # a file verl retention pruned mid-walk cannot be the reason an upload failed.
                continue
            largest = max(largest, size)
    return largest


def resume_upload_unavailable(step: int, checkpoint_dir: str, *, job_label: str) -> None:
    """report that ``step``'s resume state could not be uploaded, without failing the run.

    the resume checkpoint is internal restart convenience: it exists so a preempted attempt can
    continue instead of replaying from step 0. the artifact a required save is OWED is the
    deployable adapter, and that is published from this upload's ``before_upload`` callback -- it
    raises ``RequiredSaveError`` on its own when a required adapter cannot be published, and
    ``stop(require_complete=True)`` independently re-checks ``missing_deployables`` at the end of
    the run. so the deployable guarantee is enforced twice and neither path runs through here.

    raising here instead destroyed finished work: a lora run of a large model writes one
    ``model_world_size_*_rank_*.pt`` holding every base parameter (verl saves the whole state dict,
    with no trainable-only filtering), which for a 27.59B model is ~55 GB in ONE file and exceeds
    the artifact store's 50 GB per-file ceiling. that upload can never succeed at any retry count,
    so a run whose steps had all converged and whose adapter was already durably published was
    failed for state nothing was going to read. the frozen base weights in it are recoverable from
    the public base checkpoint anyway -- the next attempt re-downloads them regardless.

    grpo already treats this same upload as non-fatal (``rl/checkpoints.py``, "resume uploads are
    ungated internal retry state"); this makes sft and opd agree with it rather than inventing a
    new policy. the step stays marked failed on the ledger, so the loss of restart state is
    recorded rather than hidden.
    """
    largest = _largest_file_bytes(checkpoint_dir)
    detail = f" largest member {largest / 1e9:.1f} GB." if largest else ""
    print(
        f"[{job_label}] step {step} resume checkpoint was not uploaded; continuing without "
        f"restart state for it.{detail} the deployable adapter is unaffected and is enforced "
        "separately -- only a crash or preemption after this point would have to replay from an "
        "earlier step.",
        flush=True,
    )


def export_peft_adapter(
    ckpt_actor_dir: str,
    out_adapter_dir: str,
    *,
    base_model_id: str,
    python_bin: str,
) -> None:
    """turn verl's saved lora checkpoint into a flash-servable peft adapter dir.

    verl saves fsdp-sharded checkpoints (model/optim shards + a ``huggingface/`` config+tokenizer
    subfolder) under ``<local_dir>/global_step_N/actor`` for RL and ``<local_dir>/global_step_N``
    for SFT -- pass the dir that ``resolve_checkpoint_actor_dir`` picked, not a hardcoded one.
    """
    os.makedirs(out_adapter_dir, exist_ok=True)
    merge_out = out_adapter_dir.rstrip("/") + "_merge"
    shutil.rmtree(merge_out, ignore_errors=True)
    require_merge_headroom(ckpt_actor_dir, merge_out)
    merge_env = dict(os.environ)
    # the merger reads only local checkpoint files: strip credentials/tokens from its env (least
    # privilege), and keep it fully offline so it never touches hf's rate-limited api.
    for _k in list(merge_env):
        if any(t in _k.upper() for t in ("TOKEN", "SECRET", "API_KEY", "PASSWORD", "CREDENTIAL")):
            merge_env.pop(_k)
    merge_env["HF_HUB_OFFLINE"] = "1"
    merge_env["TRANSFORMERS_OFFLINE"] = "1"
    merge_env["HF_HUB_DISABLE_XET"] = "1"
    # after a successful merge, low free space is expected and cannot classify placement failures.
    merger_succeeded = False
    try:
        _run_merger(
            [
                python_bin,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                ckpt_actor_dir,
                "--target_dir",
                merge_out,
            ],
            merge_env,
        )
        merger_succeeded = True
        lora_dir = os.path.join(merge_out, "lora_adapter")
        if not os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
            raise RuntimeError(
                f"verl model_merger did not produce a peft adapter at {lora_dir} (no adapter_config.json); "
                "the merger output layout must be adjusted for this verl version."
            )
        # move within the shared filesystem so placement does not create another adapter copy.
        for name in os.listdir(lora_dir):
            os.replace(os.path.join(lora_dir, name), os.path.join(out_adapter_dir, name))
    except Exception as error:
        # classify before cleanup frees the merge tree and its low-space evidence.
        # cancellation remains a base exception and is never relabelled as disk exhaustion.
        raise_for_merge_disk_exhaustion(
            error, ckpt_actor_dir, merge_out, merger_succeeded=merger_succeeded
        )
        raise
    finally:
        shutil.rmtree(merge_out, ignore_errors=True)


# bare `lora_A.weight` ONLY, deliberately. this validates the verl merger's output at the export
# boundary, and that producer always strips the adapter name, so a namespaced key here means the
# directory holds something the merger did not write. the fused validator accepts both because it
# also runs at the warm-start boundary over previously published adapters.
_TEXT_LORA_KEY_RE = re.compile(
    r"^(?P<module>base_model\.model\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+)"
    r"(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+))*)\.lora_(?P<factor>[AB])\.weight$"
)
_NON_LANGUAGE_ADAPTER_SEGMENTS = frozenset(
    {
        "mtp",
        "multi_modal_projector",
        "patch_embed",
        "visual",
        "vision",
        "vision_encoder",
        "vision_model",
        "vision_tower",
    }
)


def _inert_non_language_keys(adapter_dir: str, keys: set[str]) -> set[str]:
    """The subset of ``keys`` whose LoRA-B factors are provably all zero, i.e. contribute nothing.

    A text-only run never TARGETS a non-language module, so such a tensor is normally proof the
    merger wrote something the run never trained. But warm start admits a pre-upgrade all-linear
    text adapter carrying inert visual/projector pairs -- `_legacy_adapter_is_multimodal` classifies
    it text-only exactly because every non-language LoRA-B is zero -- and PEFT then rebuilds the
    adapter from that SOURCE config, so those keys reappear in every checkpoint the run exports.
    Rejecting them on presence alone would let such a run train to completion and then fail every
    periodic and final publish, which is strictly worse than never admitting it.

    So apply the same rule the admission used: tolerate a non-language key only when it is provably
    inert. A live one is still contamination and still rejected. Unreadable weights decide nothing,
    so they are reported as not-inert and the caller rejects -- the safe direction.

    Exemption requires a COMPLETE A/B pair whose B is provably zero. `_non_lm_liveness_from_key`
    settles a `lora_A` key not-live on its NAME alone, so exempting per-key would admit an orphan
    A whose B the merger dropped: both callers skip exempted keys before pair-topology validation
    runs, so that orphan would never be rejected and a structurally incomplete adapter could
    publish. Pairing here keeps the skip narrow to the grandfathered artifact -- which always
    carries both factors -- and lets a genuinely broken export still fail.
    """
    from flash.adapters.artifacts import loadable_adapter_weight_files
    from flash.serve.export import (
        _load_bin_state,
        _non_lm_liveness_from_key,
        _non_lm_tensor_is_live,
        _read_safetensors_header,
    )

    inert: set[str] = set()
    try:
        selected = loadable_adapter_weight_files(os.listdir(adapter_dir))
        if not selected:
            return set()
        unread = set(keys)
        for name in selected:
            path = os.path.join(adapter_dir, name)
            if name.endswith(".bin"):
                state = _load_bin_state(Path(path))
                for key in unread & state.keys():
                    decided = _non_lm_liveness_from_key(key)
                    if decided is False or (decided is None and not bool(state[key].any())):
                        inert.add(key)
                continue
            header, data_start, file_size = _read_safetensors_header(Path(path))
            with open(path, "rb") as source:
                for key in unread & header.keys():
                    if not _non_lm_tensor_is_live(
                        source,
                        key,
                        header[key],
                        data_start=data_start,
                        file_size=file_size,
                    ):
                        inert.add(key)
    except (OSError, ValueError, ImportError):
        return set()
    return _paired_inert_keys(inert)


def _paired_inert_keys(inert: set[str]) -> set[str]:
    """Drop exemptions that do not form a complete A/B pair within the inert set."""
    factors: dict[str, dict[str, str]] = {}
    for key in inert:
        for factor in ("A", "B"):
            token = f".lora_{factor}."
            if token in key:
                factors.setdefault(key.replace(token, ".lora_*."), {})[factor] = key
                break
    return {key for pair in factors.values() if set(pair) == {"A", "B"} for key in pair.values()}


def _pair_has_nonzero_delta(factor_a, factor_b, *, require_finite_scan: bool) -> bool:
    """return whether B @ A is nonzero, checking every block when overflow is possible."""
    import numpy as np

    active = np.flatnonzero(np.any(factor_b != 0, axis=0) & np.any(factor_a != 0, axis=1))
    if active.size == 0:
        return False
    factor_a = factor_a[active].astype(np.float32, copy=False)
    factor_b = factor_b[:, active].astype(np.float32, copy=False)
    found_nonzero = False
    for row_start in range(0, factor_b.shape[0], 64):
        row_block = factor_b[row_start : row_start + 64]
        for column_start in range(0, factor_a.shape[1], 256):
            delta = row_block @ factor_a[:, column_start : column_start + 256]
            if not np.isfinite(delta).all():
                raise RuntimeError("composed LoRA delta contains non-finite values")
            if np.count_nonzero(delta):
                found_nonzero = True
                if not require_finite_scan:
                    return True
    return found_nonzero


def _validate_adapter_tensor_values(
    adapter_dir: str,
    metadata: dict[str, tuple[int, ...]],
    pairs: Mapping[object, tuple[str, str]],
    *,
    label: str,
) -> None:
    """validate finite payload values and require one nonzero composed LoRA delta."""
    import numpy as np

    from flash.adapters.artifacts import loadable_adapter_weight_files

    selected = loadable_adapter_weight_files(os.listdir(adapter_dir))
    with contextlib.ExitStack() as stack:
        sources = {}
        for name in selected:
            path = os.path.join(adapter_dir, name)
            if name.endswith(".safetensors"):
                handle = stack.enter_context(_open_safetensors_numpy(path))
                tensor_keys = handle.keys()
                sources.update({key: (handle, key) for key in tensor_keys})
            else:
                import torch

                sources.update(torch.load(path, map_location="cpu", weights_only=True))
        if sources.keys() != metadata.keys():
            raise RuntimeError(f"{label} tensor sources disagree with their metadata")

        def tensor(key: str):
            source = sources[key]
            if isinstance(source, tuple):
                return source[0].get_tensor(source[1])
            return source.detach().cpu().numpy()

        for key in metadata:
            if not np.isfinite(tensor(key)).all():
                raise RuntimeError(f"{label} tensor {key!r} contains non-finite values")

        any_nonzero_delta = False
        ordered_pairs = sorted(pairs.values(), key=lambda pair: metadata[pair[0]][0])
        for a_key, b_key in ordered_pairs:
            factor_a = tensor(a_key)
            factor_b = tensor(b_key)
            if factor_b.shape[1] != factor_a.shape[0]:
                raise RuntimeError(f"{label} LoRA pair {a_key!r}, {b_key!r} cannot compose B @ A")
            max_a = float(np.max(np.abs(factor_a.astype(np.float32, copy=False))))
            max_b = float(np.max(np.abs(factor_b.astype(np.float32, copy=False))))
            finite_bound = max_a * max_b * factor_a.shape[0]
            require_finite_scan = finite_bound > np.finfo(np.float32).max
            if any_nonzero_delta and not require_finite_scan:
                continue
            try:
                pair_nonzero = _pair_has_nonzero_delta(
                    factor_a, factor_b, require_finite_scan=require_finite_scan
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{label} LoRA pair {a_key!r}, {b_key!r} has a non-finite composed delta"
                ) from exc
            any_nonzero_delta = any_nonzero_delta or pair_nonzero
        if not any_nonzero_delta:
            raise RuntimeError(f"{label} has no nonzero composed LoRA delta")


def _validate_lora_adapter_tensors(adapter_dir: str, config: dict, *, multimodal: bool) -> None:
    """validate the actual non-moe LoRA payload before canonicalizing its config.

    a text-only run carries `resolve_lora_targeting`'s exclude regex, so a vision tensor in the
    artifact means the merger wrote something the run never targeted and the export is rejected. a
    multimodal run targets `all-linear` with no exclusion on purpose, so its vision linears are
    trained weights: they are validated like any other tensor rather than treated as contamination.
    """
    label = "exported multimodal adapter" if multimodal else "exported text adapter"
    try:
        metadata = _read_adapter_tensor_metadata(adapter_dir) or {}
    except ValueError as exc:
        raise RuntimeError(f"{label} tensor artifact is invalid") from exc

    targets = config.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        raise RuntimeError(
            "exported text adapter must declare the concrete non-empty target_modules list emitted "
            "by the Verl merger"
        )
    try:
        declared = strict_declared_lora_ranks(config)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    pairs: dict[str, dict[str, str]] = {}
    language_pairs: dict[str, dict[str, str]] = {}
    target_evidence: set[str] = set()
    inert_non_language: set[str] = set()
    if not multimodal:
        carried = {
            key for key in metadata if set(key.lower().split(".")) & _NON_LANGUAGE_ADAPTER_SEGMENTS
        }
        if carried:
            inert_non_language = _inert_non_language_keys(adapter_dir, carried)
    for key, shape in metadata.items():
        segments = set(key.lower().split("."))
        non_language = bool(segments & _NON_LANGUAGE_ADAPTER_SEGMENTS)
        if non_language and not multimodal:
            # inert pairs are the grandfathered warm-start case, not contamination: they compose to
            # a zero delta, so publishing them changes no output. a LIVE one still means the run
            # trained something it never targeted, and is still rejected.
            if key not in inert_non_language:
                raise RuntimeError(f"{label} contains non-language tensor {key!r}")
            continue
        match = _TEXT_LORA_KEY_RE.fullmatch(key)
        if match is None:
            raise RuntimeError(f"{label} contains a non-canonical tensor key {key!r}")
        module = match.group("module")
        matched_targets = {
            target for target in targets if module == target or module.endswith(f".{target}")
        }
        if not matched_targets:
            raise RuntimeError(
                f"{label} tensor module {module!r} is not declared in target_modules"
            )
        target_evidence.update(matched_targets)
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 2
            or any(
                not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0
                for dimension in shape
            )
        ):
            raise RuntimeError(f"{label} tensor {key!r} is not positive and 2-D")
        if _rank_for_module(module, declared) is None:
            raise RuntimeError(f"{label} tensor module {module!r} has no configured LoRA rank")
        if lora_tensor_rank_disagrees(key, shape, declared):
            raise RuntimeError(f"{label} tensor {key!r} disagrees with its configured LoRA rank")
        pairs.setdefault(module, {})[match.group("factor")] = key
        if not non_language:
            language_pairs.setdefault(module, {})[match.group("factor")] = key

    incomplete = sorted(module for module, factors in pairs.items() if set(factors) != {"A", "B"})
    if not pairs or incomplete:
        raise RuntimeError(
            f"{label} must contain at least one complete LoRA A/B pair and no orphan "
            f"factors; incomplete_modules={incomplete[:4]}"
        )
    # even a multimodal adapter must have trained the language stack: a run whose whole payload is
    # vision would serve as a text model with no learned text delta at all.
    if not language_pairs:
        raise RuntimeError(f"{label} contains no language-stack LoRA pair")

    pair_keys = {module: (factors["A"], factors["B"]) for module, factors in pairs.items()}
    _validate_adapter_tensor_values(adapter_dir, metadata, pair_keys, label=label)
    # the concrete list is peft's attached surface, including vision modules for multimodal runs;
    # fused-expert targets use their separate topology validator instead of this path.
    #
    # a target reachable ONLY through skipped inert non-language keys is excused: the legacy config
    # declares it (`proj` on a grandfathered all-linear adapter), but every tensor proving it was
    # skipped above, so demanding evidence would reject exactly the artifact the skip exists to
    # allow. only those targets are excused -- a language target with no tensors still fails.
    excused = {
        target
        for key in inert_non_language
        if (match := _TEXT_LORA_KEY_RE.fullmatch(key)) is not None
        for target in targets
        if (module := match.group("module")) == target or module.endswith(f".{target}")
    }
    missing_targets = sorted(set(targets) - target_evidence - excused)
    if missing_targets:
        raise RuntimeError(
            f"{label} has no tensors for declared target_modules {missing_targets[:4]}"
        )


def stamp_adapter_dir_provenance(
    adapter_dir: str,
    model_id: str,
    model_revision: str = "",
    *,
    exclude_modules: str | None,
) -> None:
    """stamp the saved adapter's immutable base identity into adapter_config.json.

    dir-based analogue of the in-memory peft-model provenance stamp: same validation + fields,
    applied to the json verl produced. raises if the adapter already names a different base.

    also normalizes fused-expert targeting at the exporter boundary. this is the one call every
    export path (sft and rl, final publish and per-step staging) already funnels through, so every
    published adapter carries the current loadable config shape.

    ``exclude_modules`` also names the run's modality. `resolve_lora_targeting` emits the language
    prefix regex exactly when the run is text-only and leaves it None for a multimodal one, which
    targets `all-linear` and trains vision linears on purpose. the tensor validation below reads it
    that way: a vision tensor is contamination in a text-only export and a trained weight in a
    multimodal one. it is required rather than defaulted so a new export path must state which.
    """
    multimodal = exclude_modules is None
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    current_base = str(cfg.get("base_model_name_or_path", "") or "").strip()
    if current_base and current_base != model_id:
        raise RuntimeError(
            f"adapter base model {current_base!r} does not match validated target {model_id!r}"
        )
    current_rev = str(cfg.get("revision", "") or "").strip()
    if current_rev and model_revision and current_rev != model_revision:
        raise RuntimeError("adapter base revision does not match the validated target commit")
    cfg["base_model_name_or_path"] = model_id
    cfg["revision"] = model_revision or None
    # write the key in BOTH directions, because its presence is the modality marker that
    # `validate_warmstart_adapter` reads: a present `None` means multimodal, a present list means
    # text-only, and only a genuinely absent key means "legacy artifact, guess from the tensors".
    # popping it for the multimodal case would stamp a fresh multimodal adapter as unmarked, so
    # warm start would fall back to the legacy tensor classifier -- and when that classifier
    # cannot decide it returns None, which skips the mismatch gate and lets a text-only run
    # continue an image-trained adapter.
    cfg["exclude_modules"] = exclude_modules or None
    normalize_verl_fused_expert_export(cfg, model_id)
    validate_fused_expert_adapter_config(cfg, model_id)
    if lora_target_parameters(model_id):
        try:
            tensors = _read_adapter_tensor_metadata(adapter_dir) or {}
        except ValueError as exc:
            raise RuntimeError("exported fused-expert adapter tensor artifact is invalid") from exc
        # a text-only run never targets non-language modules, so a LIVE one is an invalid export;
        # a multimodal run trains them under `all-linear` on purpose. this rejection is what
        # enforces that, and it must run BEFORE `fused_expert_lora_tensor_pairs`, which returns
        # non-language pairs as ordinary evidence rather than skipping them. inert pairs are
        # exempt for the same reason as the non-moe path: warm start admits a legacy adapter
        # carrying zero-delta visual factors, peft rebuilds them from that source config, and
        # rejecting them on presence would make the run unpublishable after it already trained.
        pair_tensors = tensors
        if not multimodal:
            carried = {key for key in tensors if is_non_language_lora_key(key)}
            live = carried - _inert_non_language_keys(adapter_dir, carried)
            if live:
                raise RuntimeError(
                    f"exported text adapter contains non-language tensor {sorted(live)[0]!r}"
                )
            # prune ONLY the pair input. `_validate_adapter_tensor_values` re-reads every tensor
            # on disk and requires its key set to equal `metadata` exactly, so handing it the
            # pruned map would fail as "tensor sources disagree with their metadata" on precisely
            # the grandfathered artifact the skip above exists to admit. the inert keys still get
            # their finite-value check that way; they are excluded only from pair evidence, which
            # is what `fused_expert_lora_tensor_pairs` would otherwise count as real.
            pair_tensors = {key: shape for key, shape in tensors.items() if key not in carried}
        pairs = fused_expert_lora_tensor_pairs(pair_tensors, cfg, model_id)
        if pairs is None:
            raise RuntimeError(
                f"exported adapter for {model_id} does not contain complete fused expert LoRA "
                "weights; refusing to stamp it as warm-start compatible"
            )
        _validate_adapter_tensor_values(
            adapter_dir, tensors, pairs, label="exported fused-expert adapter"
        )
    else:
        _validate_lora_adapter_tensors(adapter_dir, cfg, multimodal=multimodal)
    cfg["target_modules"] = "all-linear"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
