"""Turn a verl checkpoint directory into a deployable peft adapter.

verl writes a full training checkpoint per save step, in a layout that differs between its sft and
ppo trainers. These helpers find the right directory inside it, work out which steps are already
complete, and export the one flash actually deploys.

Split out of `flash.engine.worker.backend_common` to keep that module under the file-size limit.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import subprocess

from flash.adapters.fused_experts import (
    has_complete_fused_expert_tensors,
    lora_target_parameters,
    normalize_verl_fused_expert_export,
    validate_fused_expert_adapter_config,
)
from flash.engine.worker.model.lora import _read_adapter_tensor_metadata


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


def unprocessed_checkpoint_dirs(
    local_dir: str, completed_step: int, processed_steps: set[int]
) -> list[tuple[int, str]]:
    """``(step, dir)`` for every completed ``global_step_N`` not yet in ``processed_steps``, ascending.

    Bounded by ``completed_step`` so a directory verl is still writing is never handed to a
    publisher, and by ``processed_steps`` so each checkpoint is published exactly once.
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
        if step <= completed_step and step not in processed_steps and os.path.isdir(path):
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


def stamp_adapter_dir_provenance(adapter_dir: str, model_id: str, model_revision: str = "") -> None:
    """stamp the saved adapter's immutable base identity into adapter_config.json.

    dir-based analogue of the in-memory peft-model provenance stamp: same validation + fields,
    applied to the json verl produced. raises if the adapter already names a different base.

    also normalizes fused-expert targeting at the exporter boundary. this is the one call every
    export path (sft and rl, final publish and per-step staging) already funnels through, so every
    published adapter carries the current loadable config shape.
    """
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
    normalize_verl_fused_expert_export(cfg, model_id)
    validate_fused_expert_adapter_config(cfg, model_id)
    if lora_target_parameters(model_id):
        tensors = _read_adapter_tensor_metadata(adapter_dir) or {}
        if not has_complete_fused_expert_tensors(tensors, cfg, model_id):
            raise RuntimeError(
                f"exported adapter for {model_id} does not contain complete fused expert LoRA "
                "weights; refusing to stamp it as warm-start compatible"
            )
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
