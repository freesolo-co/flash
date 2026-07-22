"""shared out-of-process OpenRLHF harness for flash training workers.

flash's worker image uses cuda 12.8, while the pinned OpenRLHF runtime needs the separate cuda 13
image ``ghcr.io/freesolo-co/flash-openrlhf:cu13-vllm0221``. flash therefore never imports
OpenRLHF in-process. the cuda 13 worker sets ``FLASH_OPENRLHF_PYTHON`` and launches an explicit
OpenRLHF trainer entrypoint as a subprocess. flash-specific logic stays in the parent process and is
reached through authenticated localhost bridges, starting with rewards and later the OPD teacher
path, matching the verl boundary.

this module contains only framework-neutral process, export, and provenance helpers. worker-specific
GRPO, OPD, and SFT entrypoints and argument construction belong in later migration PRs.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable

_DEFAULT_OPENRLHF_ENTRYPOINT = "openrlhf.cli.train_ppo_ray"
_PROCESS_GROUP_TERM_TIMEOUT_S = 10.0
_PROCESS_GROUP_POLL_INTERVAL_S = 0.05
_OUTPUT_EOF = object()
_KERNEL_CACHE_ENV_VARS = (
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "FLASHINFER_CUBIN_DIR",
    "FLASHINFER_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
)
_BIN_TO_SAFETENSORS = """
import sys

import torch
from safetensors.torch import save_file

source, target = sys.argv[1:]
state = torch.load(source, map_location="cpu", weights_only=True)
if not isinstance(state, dict) or not all(isinstance(name, str) for name in state):
    raise TypeError("OpenRLHF adapter_model.bin is not a tensor state dict")
if not all(isinstance(tensor, torch.Tensor) for tensor in state.values()):
    raise TypeError("OpenRLHF adapter_model.bin contains non-tensor values")
state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in state.items()}
save_file(state, target)
""".strip()


def resolve_openrlhf_python(workdir: str) -> str:
    """return the interpreter for the prebuilt OpenRLHF worker image.

    ``workdir`` is retained for API symmetry with the verl harness. no environment is installed at
    runtime because OpenRLHF's cuda 13, torch 2.11, vllm, and flash-attn stack is baked into the
    dedicated image and cannot coexist with flash's cuda 12.8 worker environment.
    """
    del workdir
    preset = os.environ.get("FLASH_OPENRLHF_PYTHON", "").strip()
    return preset or sys.executable


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    """terminate and reap the subprocess session, escalating after a bounded wait."""
    try:
        process_group_id = os.getpgid(proc.pid)
    except ProcessLookupError:
        process_group_id = proc.pid

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return

    deadline = time.monotonic() + _PROCESS_GROUP_TERM_TIMEOUT_S
    while time.monotonic() < deadline:
        proc.poll()
        if not _process_group_exists(process_group_id):
            break
        time.sleep(_PROCESS_GROUP_POLL_INTERVAL_S)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
    proc.wait()


def run_openrlhf_training(
    python_bin: str,
    entry_args: list[str],
    *,
    env: dict[str, str],
    entrypoint: str = _DEFAULT_OPENRLHF_ENTRYPOINT,
    cwd: str | None = None,
    on_step: Callable[[int], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
    step_pattern: str = r"(?<![\w])(?:global[_ ]step|step)[:=\s]+(\d+)\b",
    heartbeat_interval_s: float = 20.0,
) -> int:
    """run an OpenRLHF trainer module and stream its merged output.

    returns the child exit code. heartbeats run on wall-clock cadence independently of output.
    callback failures terminate the entire subprocess session before being re-raised so failed
    checkpoint or heartbeat callbacks cannot leave paid Ray workers running unattended.
    """
    if not entrypoint.strip():
        raise ValueError("OpenRLHF entrypoint must not be empty")
    if heartbeat is not None and heartbeat_interval_s <= 0:
        raise ValueError("OpenRLHF heartbeat interval must be positive")

    step_re = re.compile(step_pattern)
    cmd = [python_bin, "-u", "-m", entrypoint, *entry_args]
    child_env = dict(env)
    child_env["PYTHONUNBUFFERED"] = "1"
    for name in _KERNEL_CACHE_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            child_env.setdefault(name, value)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output: queue.Queue[object] = queue.Queue()

    def read_output() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                output.put(line)
        except BaseException as exc:
            output.put(exc)
        finally:
            output.put(_OUTPUT_EOF)

    reader = threading.Thread(target=read_output, name="openrlhf-output", daemon=True)
    reader.start()
    next_heartbeat = time.monotonic() + heartbeat_interval_s
    last_step: int | None = None
    try:
        while True:
            timeout = None
            if heartbeat is not None:
                now = time.monotonic()
                if now >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = now + heartbeat_interval_s
                timeout = max(0.0, next_heartbeat - time.monotonic())
            try:
                item = output.get(timeout=timeout)
            except queue.Empty:
                continue
            if item is _OUTPUT_EOF:
                break
            if isinstance(item, BaseException):
                raise item
            line = str(item)
            print(line, end="", flush=True)
            if on_line is not None:
                on_line(line)
            match = step_re.search(line)
            if match and on_step is not None:
                step = int(match.group(1))
                if step != last_step:
                    on_step(step)
                    last_step = step
    except BaseException:
        _terminate_process_group(proc)
        raise
    finally:
        if proc.poll() is None:
            proc.wait()
        reader.join(timeout=1)
    return int(proc.returncode)


def _is_full_model_export(path: str) -> bool:
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    try:
        names = os.listdir(path)
    except OSError:
        return False
    return any(
        name in {"pytorch_model.bin", "model.safetensors"}
        or (name.startswith("pytorch_model-") and name.endswith(".bin"))
        or (name.startswith("model-") and name.endswith(".safetensors"))
        for name in names
    )


def _openrlhf_adapter_source(ckpt_dir: str) -> str:
    """resolve a PEFT export from a final, HF, or matching DeepSpeed checkpoint path."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    direct_config = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.isfile(direct_config):
        return ckpt_dir
    if _is_full_model_export(ckpt_dir):
        raise RuntimeError(
            f"OpenRLHF checkpoint {ckpt_dir} is a full-model HF export, not a PEFT adapter"
        )

    candidates: list[str] = []
    parent = os.path.dirname(ckpt_dir)
    tag = os.path.basename(ckpt_dir)
    checkpoint_root = os.path.dirname(parent) if os.path.basename(parent) == "_actor" else parent
    candidates.append(os.path.join(checkpoint_root, f"{tag}_hf"))

    for latest_path in (
        os.path.join(ckpt_dir, "_actor", "latest"),
        os.path.join(ckpt_dir, "latest"),
    ):
        if os.path.isfile(latest_path):
            with open(latest_path) as latest_file:
                latest_tag = latest_file.read().strip()
            if latest_tag:
                candidates.append(os.path.join(ckpt_dir, f"{latest_tag}_hf"))

    candidates = list(dict.fromkeys(candidates))
    resolved = [
        path for path in candidates if os.path.isfile(os.path.join(path, "adapter_config.json"))
    ]
    if len(resolved) == 1:
        return resolved[0]
    if len(resolved) > 1:
        raise RuntimeError(f"multiple OpenRLHF PEFT exports match checkpoint {ckpt_dir}")
    full_model_exports = [path for path in candidates if _is_full_model_export(path)]
    if full_model_exports:
        raise RuntimeError(
            f"OpenRLHF checkpoint {full_model_exports[0]} is a full-model HF export, not a PEFT adapter"
        )

    raise RuntimeError(
        f"no PEFT adapter export found for OpenRLHF checkpoint {ckpt_dir}; pass ckpt.output_dir "
        "or enable --ckpt.save_hf and pass the DeepSpeed tag or checkpoint root. raw DeepSpeed "
        "recovery shards and full-model HF weights are not flash-servable LoRA adapters."
    )


def export_openrlhf_adapter(
    ckpt_dir: str,
    out_adapter_dir: str,
    base_model_id: str,
    base_model_revision: str,
    python_bin: str,
) -> None:
    """export an OpenRLHF PEFT checkpoint as a flash-servable safetensors adapter.

    ``base_model_revision`` is the externally validated immutable revision from the job spec when
    the run pins one; unpinned runs leave it empty and stamp ``revision`` as null.
    OpenRLHF's final ``save_model`` consolidates ZeRO-3 state before PEFT export. with ZeRO-3 it
    writes authoritative ``adapter_model.bin`` after PEFT's safetensors output and then removes the
    safetensors file, so a crash state containing both is converted from the bin. conversion runs in
    the isolated OpenRLHF interpreter where torch and safetensors are already installed. a raw
    DeepSpeed recovery checkpoint is accepted only when its matching ``<tag>_hf`` export exists.
    """
    expected_revision = base_model_revision.strip()
    source_dir = _openrlhf_adapter_source(ckpt_dir)
    source_realpath = os.path.realpath(source_dir)
    output_realpath = os.path.realpath(os.path.abspath(out_adapter_dir))
    common_path = os.path.commonpath((source_realpath, output_realpath))
    if common_path in {source_realpath, output_realpath}:
        raise ValueError("OpenRLHF checkpoint and adapter output directories must not overlap")

    source_config = os.path.join(source_dir, "adapter_config.json")
    with open(source_config) as config_file:
        adapter_config = json.load(config_file)
    if adapter_config.get("peft_type") != "LORA":
        raise RuntimeError(f"OpenRLHF PEFT export is not a LoRA adapter: {source_dir}")

    source_safetensors = os.path.join(source_dir, "adapter_model.safetensors")
    source_bin = os.path.join(source_dir, "adapter_model.bin")
    if not os.path.isfile(source_safetensors) and not os.path.isfile(source_bin):
        raise RuntimeError(f"OpenRLHF PEFT export has no adapter weights: {source_dir}")

    shutil.rmtree(out_adapter_dir, ignore_errors=True)
    os.makedirs(out_adapter_dir, exist_ok=True)
    shutil.copy2(source_config, os.path.join(out_adapter_dir, "adapter_config.json"))
    output_weights = os.path.join(out_adapter_dir, "adapter_model.safetensors")
    if os.path.isfile(source_bin):
        subprocess.run(
            [python_bin, "-c", _BIN_TO_SAFETENSORS, source_bin, output_weights],
            check=True,
        )
        if not os.path.isfile(output_weights):
            raise RuntimeError(
                "OpenRLHF adapter conversion did not produce adapter_model.safetensors"
            )
    else:
        shutil.copy2(source_safetensors, output_weights)

    stamp_adapter_dir_provenance(out_adapter_dir, base_model_id, expected_revision)


def _hf_snapshot_identity(path: str) -> tuple[str, str] | None:
    parts = os.path.normpath(path).split(os.sep)
    try:
        snapshots_index = len(parts) - 1 - parts[::-1].index("snapshots")
    except ValueError:
        return None
    if snapshots_index == 0 or snapshots_index + 1 >= len(parts):
        return None
    repo_folder = parts[snapshots_index - 1]
    if not repo_folder.startswith("models--"):
        return None
    encoded_model_id = repo_folder.removeprefix("models--")
    if not encoded_model_id:
        return None
    model_id = encoded_model_id.replace("--", "/")
    revision = parts[snapshots_index + 1]
    if not revision:
        return None
    return model_id, revision


def stamp_adapter_dir_provenance(
    adapter_dir: str,
    model_id: str,
    model_revision: str = "",
) -> None:
    """validate and stamp a saved adapter's immutable base identity."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    current_base = str(config.get("base_model_name_or_path", "") or "").strip()
    snapshot_identity = _hf_snapshot_identity(current_base) if current_base else None
    if snapshot_identity is not None:
        snapshot_model_id, snapshot_revision = snapshot_identity
        if snapshot_model_id != model_id:
            raise RuntimeError(
                f"adapter base model {snapshot_model_id!r} does not match validated target {model_id!r}"
            )
        if snapshot_revision != model_revision:
            raise RuntimeError("adapter base revision does not match the validated target commit")
    elif current_base and current_base != model_id:
        raise RuntimeError(
            f"adapter base model {current_base!r} does not match validated target {model_id!r}"
        )
    current_revision = str(config.get("revision", "") or "").strip()
    if current_revision and current_revision != model_revision:
        raise RuntimeError("adapter base revision does not match the validated target commit")
    config["base_model_name_or_path"] = model_id
    config["revision"] = model_revision or None
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)
