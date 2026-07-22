"""shared out-of-process OpenRLHF harness for flash training workers.

flash's worker image uses cuda 12.8, while the pinned OpenRLHF runtime needs the separate cuda 13
image ``ghcr.io/freesolo-co/flash-openrlhf:cu13-vllm0221``. flash therefore never imports
OpenRLHF in-process. the cuda 13 worker sets ``FLASH_OPENRLHF_PYTHON`` and launches OpenRLHF as a
subprocess. flash-specific logic stays in the parent process and is reached through authenticated
localhost bridges, starting with rewards and later the OPD teacher path, matching the verl boundary.

this module contains only framework-neutral process, export, and provenance helpers. worker-specific
GRPO, OPD, and SFT argument construction belongs in later migration PRs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable

_OPENRLHF_ENTRYPOINT = "openrlhf.cli.train_ppo_ray"
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


def run_openrlhf_training(
    python_bin: str,
    entry_args: list[str],
    *,
    env: dict[str, str],
    cwd: str | None = None,
    on_step: Callable[[int], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
    step_pattern: str = r"(?:global[_ ]step|step)[:=\s]+(\d+)",
    heartbeat_interval_s: float = 20.0,
) -> int:
    """run ``openrlhf.cli.train_ppo_ray`` and stream its merged output.

    returns the child exit code. callback failures terminate the child before being re-raised so a
    failed required checkpoint or heartbeat callback cannot leave paid training running unattended.
    """
    step_re = re.compile(step_pattern)
    cmd = [python_bin, "-m", _OPENRLHF_ENTRYPOINT, *entry_args]
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_heartbeat = 0.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if on_line is not None:
                on_line(line)
            match = step_re.search(line)
            if match and on_step is not None:
                on_step(int(match.group(1)))
            if heartbeat is not None:
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval_s:
                    heartbeat()
                    last_heartbeat = now
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    finally:
        if proc.poll() is None:
            proc.wait()
    return int(proc.returncode)


def _openrlhf_adapter_source(ckpt_dir: str) -> str:
    """resolve a PEFT export from a final, HF, or matching DeepSpeed checkpoint path."""
    ckpt_dir = os.path.abspath(ckpt_dir)
    direct_config = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.isfile(direct_config):
        return ckpt_dir

    candidates: list[str] = []
    parent = os.path.dirname(ckpt_dir)
    tag = os.path.basename(ckpt_dir)
    if os.path.basename(parent) == "_actor":
        candidates.append(os.path.join(os.path.dirname(parent), f"{tag}_hf"))

    actor_dir = os.path.join(ckpt_dir, "_actor")
    latest_path = os.path.join(actor_dir, "latest")
    if os.path.isfile(latest_path):
        with open(latest_path) as latest_file:
            latest_tag = latest_file.read().strip()
        if latest_tag:
            candidates.append(os.path.join(ckpt_dir, f"{latest_tag}_hf"))

    resolved = [path for path in candidates if os.path.isfile(os.path.join(path, "adapter_config.json"))]
    if len(resolved) == 1:
        return resolved[0]
    if len(resolved) > 1:
        raise RuntimeError(f"multiple OpenRLHF PEFT exports match checkpoint {ckpt_dir}")

    raise RuntimeError(
        f"no PEFT adapter export found for OpenRLHF checkpoint {ckpt_dir}; pass ckpt.output_dir "
        "or enable --ckpt.save_hf and pass the DeepSpeed tag or checkpoint root. raw DeepSpeed "
        "recovery shards and full-model HF weights are not flash-servable LoRA adapters."
    )


def export_openrlhf_adapter(
    ckpt_dir: str,
    out_adapter_dir: str,
    base_model_id: str,
    python_bin: str,
) -> None:
    """export an OpenRLHF PEFT checkpoint as a flash-servable safetensors adapter.

    OpenRLHF's final ``save_model`` consolidates ZeRO-3 state before PEFT export. with ZeRO-3 it
    writes ``adapter_model.bin`` after removing PEFT's safetensors output, so conversion runs in the
    isolated OpenRLHF interpreter where torch and safetensors are already installed. a raw DeepSpeed
    recovery checkpoint is accepted only when its matching ``<tag>_hf`` export exists.
    """
    source_dir = _openrlhf_adapter_source(ckpt_dir)
    if os.path.realpath(source_dir) == os.path.realpath(out_adapter_dir):
        raise ValueError("OpenRLHF checkpoint and adapter output directories must differ")

    source_config = os.path.join(source_dir, "adapter_config.json")
    source_safetensors = os.path.join(source_dir, "adapter_model.safetensors")
    source_bin = os.path.join(source_dir, "adapter_model.bin")
    if not os.path.isfile(source_safetensors) and not os.path.isfile(source_bin):
        raise RuntimeError(f"OpenRLHF PEFT export has no adapter weights: {source_dir}")
    with open(source_config) as config_file:
        source_revision = str(json.load(config_file).get("revision", "") or "").strip()

    shutil.rmtree(out_adapter_dir, ignore_errors=True)
    os.makedirs(out_adapter_dir, exist_ok=True)
    shutil.copy2(source_config, os.path.join(out_adapter_dir, "adapter_config.json"))
    output_weights = os.path.join(out_adapter_dir, "adapter_model.safetensors")
    if os.path.isfile(source_safetensors):
        shutil.copy2(source_safetensors, output_weights)
    else:
        subprocess.run(
            [python_bin, "-c", _BIN_TO_SAFETENSORS, source_bin, output_weights],
            check=True,
        )
        if not os.path.isfile(output_weights):
            raise RuntimeError("OpenRLHF adapter conversion did not produce adapter_model.safetensors")

    stamp_adapter_dir_provenance(out_adapter_dir, base_model_id, source_revision)


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
    if current_base and current_base != model_id:
        raise RuntimeError(
            f"adapter base model {current_base!r} does not match validated target {model_id!r}"
        )
    current_revision = str(config.get("revision", "") or "").strip()
    if current_revision and model_revision and current_revision != model_revision:
        raise RuntimeError("adapter base revision does not match the validated target commit")
    config["base_model_name_or_path"] = model_id
    config["revision"] = model_revision or None
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)
