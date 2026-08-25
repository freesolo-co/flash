"""Turn a verl checkpoint directory into a deployable peft adapter.

verl writes a full training checkpoint per save step, in a layout that differs between its sft and
ppo trainers. These helpers find the right directory inside it, work out which steps are already
complete, and export the one flash actually deploys.

Split out of `flash.engine.worker.train.entry.backend_common` to keep that module under the file-size limit.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from typing import Any

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
from flash.engine.support.verl_checkpoint import (
    VERL_FSDP_CONFIG_FILE,
    FsdpCheckpointInspection,
    FsdpManifestFile,
    inspect_fsdp_checkpoint_manifest,
)
from flash.engine.support.verl_policy import FsdpGeneration
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
_FSDP_MODEL_SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_\d+\.pt")

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
        if _FSDP_MODEL_SHARD_RE.fullmatch(name) is None:
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
    from flash.engine.worker.train.entry import backend_common

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


def inspect_resume_checkpoint(
    resume_dir: str,
    *,
    world_size: int,
    expected_fsdp_generation: FsdpGeneration,
) -> FsdpCheckpointInspection:
    """Parse and validate one local native checkpoint exactly once."""
    actor_dir = resolve_checkpoint_actor_dir(resume_dir)
    try:
        with open(os.path.join(actor_dir, VERL_FSDP_CONFIG_FILE), "rb") as file:
            stamp_raw = file.read()
    except OSError:
        stamp_raw = None
    try:
        names = os.listdir(actor_dir)
    except OSError:
        names = []
    files = []
    for name in names:
        path = os.path.join(actor_dir, name)
        size = None
        if os.path.isfile(path):
            try:
                with open(path, "rb") as file:
                    if file.read(1):
                        size = os.path.getsize(path)
            except OSError:
                pass
        files.append(FsdpManifestFile(name, size if size is not None else 0))
    return inspect_fsdp_checkpoint_manifest(
        stamp_raw,
        files,
        expected_generation=expected_fsdp_generation,
        expected_world_size=world_size,
    )


def select_verl_resume_checkpoint(
    *,
    world_size: int,
    expected_fsdp_generation: FsdpGeneration,
    accept: Callable[[str, FsdpCheckpointInspection], bool] | None = None,
) -> tuple[str | None, FsdpCheckpointInspection | None]:
    """select the newest caller-accepted loadable checkpoint and return its cached inspection."""
    import flash.engine.worker.io.hf as worker_hf

    inspections: dict[str, FsdpCheckpointInspection] = {}

    def prefer(path: str) -> bool:
        inspection = inspections.get(path)
        if inspection is None:
            inspection = inspect_resume_checkpoint(
                path,
                world_size=world_size,
                expected_fsdp_generation=expected_fsdp_generation,
            )
            inspections[path] = inspection
        return inspection.loadable and (accept is None or accept(path, inspection))

    resume = worker_hf.hf_resume_checkpoint(prefer=prefer)
    if not resume:
        return None, None
    inspection = inspections.get(resume)
    if inspection is None:
        prefer(resume)
        inspection = inspections[resume]
    return resume, inspection


def restore_verl_resume(
    local_dir: str,
    *,
    world_size: int,
    expected_fsdp_generation: FsdpGeneration,
    job_label: str,
) -> int:
    """select, inspect once, and stage the newest loadable streamed checkpoint."""
    resume, inspection = select_verl_resume_checkpoint(
        world_size=world_size,
        expected_fsdp_generation=expected_fsdp_generation,
    )
    if not resume:
        return 0
    if inspection is None:
        raise RuntimeError("selected verl resume checkpoint is missing its inspection")
    return stage_verl_resume(
        resume,
        local_dir,
        job_label=job_label,
        world_size=world_size,
        expected_fsdp_generation=expected_fsdp_generation,
        inspection=inspection,
    )


def resume_topology_matches(
    resume_dir: str,
    *,
    world_size: int,
    expected_fsdp_generation: FsdpGeneration,
    job_label: str,
    inspection: FsdpCheckpointInspection | None = None,
) -> bool:
    """inspect one native checkpoint and report why it is discarded."""
    if inspection is None:
        inspection = inspect_resume_checkpoint(
            resume_dir,
            world_size=world_size,
            expected_fsdp_generation=expected_fsdp_generation,
        )
    if inspection.loadable:
        return True
    print(
        f"[{job_label}] discarding resume checkpoint {os.path.basename(resume_dir)}: "
        f"{inspection.diagnostic()}; restarting from step 0",
        flush=True,
    )
    return False


def stage_verl_resume(
    resume_dir: str,
    local_dir: str,
    *,
    job_label: str,
    world_size: int,
    expected_fsdp_generation: FsdpGeneration,
    inspection: FsdpCheckpointInspection | None = None,
) -> int:
    """stage a downloaded ``checkpoint-N`` into local_dir where verl looks; return its step.

    the resume artifact is keyed on the run prefix, not the job type, so the control plane hands
    every trainer the same ``checkpoint-N`` layout. verl finds it via
    latest_checkpointed_iteration.txt under trainer.default_local_dir once resume_mode=auto.
    ``job_label`` names the job in the error raised for an unparseable path and in the discard log.

    returns 0, staging nothing, when the checkpoint is not complete native state for the expected
    generation and ``world_size``. 0 is the fresh-run answer every caller handles.
    """
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume_dir))
    if match is None:
        raise RuntimeError(f"invalid {job_label} resume checkpoint path {resume_dir!r}")
    step = int(match.group(1))
    if not resume_topology_matches(
        resume_dir,
        world_size=world_size,
        expected_fsdp_generation=expected_fsdp_generation,
        job_label=job_label,
        inspection=inspection,
    ):
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
    # "not confirmed", not "not uploaded": the False this reports on is also returned when the
    # folder commit landed and only the closing heartbeat exhausted its retries, so the restart
    # state may well be present. the run is continued either way, and the next attempt re-checks
    # what is actually in the repo rather than trusting this line.
    print(
        f"[{job_label}] step {step} resume checkpoint was not confirmed uploaded; continuing "
        f"without relying on restart state for it.{detail} the deployable adapter is unaffected "
        "and is enforced separately -- only a crash or preemption after this point would have to "
        "replay from an earlier step.",
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


# bare `lora_A.weight` only, deliberately. this validates the verl merger's output at the export
# boundary, and that producer always strips the adapter name, so a namespaced key here means the
# directory holds something the merger did not write. the fused validator accepts both because it
# also runs at the warm-start boundary over previously published adapters.
# collapse layer indexes so every layer of one stack shares a width bucket, while the vision and
# language stacks stay distinct: `...layers.0.mlp.down_proj` and `...layers.31.mlp.down_proj` are
# the same base module shape, `...visual.blocks.0.mlp.down_proj` is not.
#
# the collapse is not exhaustive, deliberately. `re.sub` is non-overlapping, so consecutive indexes
# only partly collapse (`a.b.0.1.2.proj` -> `a.b.1.proj`), and a trailing index is left alone
# (`a.proj.7`). both OVER-split. a partial bucket is always contained in a single fully-collapsed
# bucket -- substitution only ever removes text, so two paths that survive to the same partial form
# cannot separate under further passes -- so an unhandled shape can only invent EXTRA buckets. that
# costs a missed catch and never a false reject of a healthy export, which is why leaving these
# shapes unhandled is safe rather than merely untested. fused experts do collapse fully:
# `layers.0.mlp.experts.3.down_proj` -> `layers.mlp.experts.down_proj`.
_LAYER_INDEX_RE = re.compile(r"\.\d+\.")
_TEXT_LORA_KEY_RE = re.compile(
    r"^(?P<module>base_model\.model\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+)"
    r"(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+))*)\.lora_(?P<factor>[AB])\.weight$"
)


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
    pairs: Mapping[Any, tuple[str, str]],
    *,
    label: str,
    must_train: Mapping[Any, tuple[str, str]] | None = None,
    must_train_subject: str = "",
) -> None:
    """validate finite payload values and require one nonzero composed LoRA delta.

    every pair is shape- and finite-checked. `must_train` narrows which pairs may SATISFY the
    nonzero requirement: a multimodal export passes its language subset, because "some pair moved"
    is satisfied by a vision pair alone and would publish a run whose whole text stack is zero.
    omitting it keeps the whole set eligible, which is what a text-only export wants.
    """
    import numpy as np

    from flash.adapters.artifacts import loadable_adapter_weight_files

    selected = loadable_adapter_weight_files(os.listdir(adapter_dir))
    with contextlib.ExitStack() as stack:
        sources = {}
        for name in selected:
            path = os.path.join(adapter_dir, name)
            handle = stack.enter_context(_open_safetensors_numpy(path))
            tensor_keys = handle.keys()
            sources.update({key: (handle, key) for key in tensor_keys})
        if sources.keys() != metadata.keys():
            raise RuntimeError(f"{label} tensor sources disagree with their metadata")

        def tensor(key: str):
            source = sources[key]
            return source[0].get_tensor(source[1])

        for key in metadata:
            if not np.isfinite(tensor(key)).all():
                raise RuntimeError(f"{label} tensor {key!r} contains non-finite values")

        eligible = {tuple(pair) for pair in (pairs if must_train is None else must_train).values()}
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
            # an ineligible pair is still composed above, because that is where a non-finite delta
            # surfaces; it just cannot be the pair that discharges the requirement.
            if pair_nonzero and (a_key, b_key) in eligible:
                any_nonzero_delta = True
        if not any_nonzero_delta:
            subject = f" in its {must_train_subject}" if must_train_subject else ""
            raise RuntimeError(f"{label} has no nonzero composed LoRA delta{subject}")


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
            f"{label} must declare the concrete non-empty target_modules list emitted "
            "by the Verl merger"
        )
    try:
        declared = strict_declared_lora_ranks(config)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    pairs: dict[str, dict[str, str]] = {}
    language_pairs: dict[str, dict[str, str]] = {}
    target_evidence: set[str] = set()
    module_widths: dict[str, dict[str, int]] = {}
    for key, shape in metadata.items():
        non_language = is_non_language_lora_key(key)
        if non_language and not multimodal:
            raise RuntimeError(f"{label} contains non-language tensor {key!r}")
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
        # the outer dimension is the base module's own width, so tensors on the same base module
        # must agree on it. rank and mutual composability are checked above and neither constrains
        # it: a correctly-ranked pair whose outer dim names a different module's width publishes
        # here and only fails later, at peft or vllm load, with no provenance back to the export.
        #
        # keyed by the LAYER-STRIPPED module path, NOT the bare target suffix. a VL model's vision
        # tower and language model legitimately share leaf names at different widths -- a
        # multimodal run has no exclude regex, so `all-linear` covers both stacks and the merger
        # writes `...language_model...mlp.down_proj` (text intermediate) alongside
        # `...visual.blocks.N.mlp.down_proj` (vision intermediate). those are two different base
        # modules, and requiring them to agree fails a HEALTHY image export at publish time, after
        # the paid run has already finished.
        stack = _LAYER_INDEX_RE.sub(".", module)
        width = shape[1] if match.group("factor") == "A" else shape[0]
        seen = module_widths.setdefault(stack, {}).setdefault(match.group("factor"), width)
        if seen != width:
            raise RuntimeError(
                f"{label} tensor {key!r} has outer dimension {width} where module "
                f"{stack!r} was already {seen}"
            )

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
    language_keys = {
        module: (factors["A"], factors["B"]) for module, factors in language_pairs.items()
    }
    # the nonzero-delta requirement is discharged by the language subset. presence alone (checked
    # above) does not mean the text stack trained: a vision-only delta would otherwise satisfy it
    # and publish a paid run that serves as a text model with no learned text delta.
    _validate_adapter_tensor_values(
        adapter_dir,
        metadata,
        pair_keys,
        label=label,
        must_train=language_keys,
        must_train_subject="language stack",
    )
    missing_targets = sorted(set(targets) - target_evidence)
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
    if exclude_modules is not None and not exclude_modules.strip():
        # modality is read from this argument as `is None` and written below as `or None`. an empty
        # string would validate as text-only and then persist as multimodal, so warm start would
        # read back the opposite modality from the one the tensors were checked against. reject it
        # rather than pick a side: `resolve_lora_targeting` never emits it, so it can only reach
        # here from a new caller that has not stated a modality.
        raise RuntimeError("exclude_modules must be a non-empty regex or None")
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
    # write the key in both directions because every warm-start artifact must state its modality.
    # a present none means multimodal and a present regex means text-only. the empty string was
    # rejected above, so this writes back exactly the value the validation below was keyed on.
    cfg["exclude_modules"] = exclude_modules
    normalize_verl_fused_expert_export(cfg, model_id)
    validate_fused_expert_adapter_config(cfg, model_id)
    if lora_target_parameters(model_id):
        try:
            tensors = _read_adapter_tensor_metadata(adapter_dir) or {}
        except ValueError as exc:
            raise RuntimeError("exported fused-expert adapter tensor artifact is invalid") from exc
        # a text-only run never targets non-language modules, while a multimodal run trains them
        # under `all-linear` on purpose.
        if not multimodal:
            carried = {key for key in tensors if is_non_language_lora_key(key)}
            if carried:
                raise RuntimeError(
                    f"exported text adapter contains non-language tensor {sorted(carried)[0]!r}"
                )
        pairs = fused_expert_lora_tensor_pairs(tensors, cfg, model_id)
        if pairs is None:
            raise RuntimeError(
                f"exported adapter for {model_id} does not contain complete fused expert LoRA "
                "weights; refusing to stamp it as warm-start compatible"
            )
        # a key the pair builder did not claim is neither paired nor value-checked below -- it is
        # simply published. peft loads the saved state dict with `strict=False`, so a declared
        # `modules_to_save` entry or a bare bias key whose name matches the base model is restored
        # OVER the base weights at warm start and serve. the non-moe path rejects every key it does
        # not recognize; require the same here rather than letting the fused topology be the one
        # that carries unvalidated tensors.
        claimed = {key for pair in pairs.values() for key in pair}
        unclaimed = sorted(set(tensors) - claimed)
        if unclaimed:
            raise RuntimeError(
                f"exported fused-expert adapter contains an unpaired tensor {unclaimed[0]!r}"
            )
        # same language-subset requirement as the non-moe path: this model is image-capable, so a
        # multimodal export must not discharge "something trained" with a vision pair alone.
        language_pairs = {
            module: pair for module, pair in pairs.items() if not is_non_language_lora_key(module)
        }
        if not language_pairs:
            raise RuntimeError("exported fused-expert adapter contains no language-stack LoRA pair")
        _validate_adapter_tensor_values(
            adapter_dir,
            tensors,
            pairs,
            label="exported fused-expert adapter",
            must_train=language_pairs,
            must_train_subject="language stack",
        )
    else:
        _validate_lora_adapter_tensors(adapter_dir, cfg, multimodal=multimodal)
    cfg["target_modules"] = "all-linear"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
