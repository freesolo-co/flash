"""shared out-of-process verl harness reused by the sft and opd verl trainers.

verl pins its own torch/vllm, incompatible with flash's, so flash never imports verl in-process: it
runs a verl trainer entrypoint as a subprocess against a separate interpreter and merges the
fsdp-sharded lora checkpoint back into a flash-servable peft adapter. these are the framework- and
algorithm-neutral pieces (interpreter resolution, checkpoint export, provenance, progress
streaming); the sft/opd modules layer their own dataset rows, hydra overrides, and orchestration on
top. the grpo verl path uses the same pattern and will consolidate onto this module once it lands.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable


def resolve_verl_python(workdir: str) -> str:
    """return an interpreter that can import verl.

    prefers FLASH_VERL_PYTHON (set by the caller when verl is preinstalled, e.g. on a verl image).
    otherwise provisions an isolated venv on the pod so verl's torch/vllm never touch flash's env.
    """
    preset = os.environ.get("FLASH_VERL_PYTHON", "").strip()
    if preset:
        return preset
    venv = os.path.join(workdir, "verl-venv")
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        # dev-only fallback (production uses FLASH_VERL_PYTHON on a prebuilt verl image): verl brings
        # its own torch/vllm; a full install (not --no-deps) so runtime deps are present. unpinned so
        # it tracks the current verl (vllm 0.23/0.24, Qwen3.5-capable) rather than a stale release.
        subprocess.run(["uv", "venv", venv], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                py,
                "verl",
                "liger-kernel",
                "bitsandbytes>=0.49",
                "qwen-vl-utils",
                "torchvision",
            ],
            check=True,
        )
    return py


def latest_global_step_dir(local_dir: str) -> tuple[str, int]:
    """return (actor_dir, step) for the highest global_step_N checkpoint verl wrote."""
    best_step, best = -1, ""
    if os.path.isdir(local_dir):
        for name in os.listdir(local_dir):
            m = re.fullmatch(r"global_step_(\d+)", name)
            if m and int(m.group(1)) > best_step:
                best_step = int(m.group(1))
                best = os.path.join(local_dir, name, "actor")
    if best_step < 0:
        raise RuntimeError(f"no global_step_N checkpoint found under {local_dir}")
    return best, best_step


def export_peft_adapter(
    ckpt_actor_dir: str,
    out_adapter_dir: str,
    *,
    base_model_id: str,
    python_bin: str,
) -> None:
    """turn verl's saved lora checkpoint into a flash-servable peft adapter dir.

    verl saves fsdp-sharded checkpoints under ``<local_dir>/global_step_N/actor`` (model/optim
    shards + a ``huggingface/`` config+tokenizer subfolder). ``verl.model_merger merge`` writes a
    standard peft adapter (adapter_config.json + adapter_model.safetensors) to a ``lora_adapter/``
    subfolder of its target; we copy just that adapter into flash's adapter dir (the co-produced
    merged full model is discarded -- flash serves the lora on the immutable base).

    verified against verl 0.8 on an h100: merger emits ``<target>/lora_adapter/{adapter_config
    .json,adapter_model.safetensors}`` with ``base_model_name_or_path: null``.
    """
    os.makedirs(out_adapter_dir, exist_ok=True)
    merge_out = out_adapter_dir.rstrip("/") + "_merge"
    shutil.rmtree(merge_out, ignore_errors=True)
    merge_env = dict(os.environ)
    # the merger reads only local checkpoint files: strip credentials/tokens from its env (least
    # privilege), and keep it fully offline so it never touches hf's rate-limited api.
    for _k in list(merge_env):
        if any(t in _k.upper() for t in ("TOKEN", "SECRET", "API_KEY", "PASSWORD", "CREDENTIAL")):
            merge_env.pop(_k)
    merge_env["HF_HUB_OFFLINE"] = "1"
    merge_env["TRANSFORMERS_OFFLINE"] = "1"
    merge_env["HF_HUB_DISABLE_XET"] = "1"
    subprocess.run(
        [python_bin, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
         "--local_dir", ckpt_actor_dir, "--target_dir", merge_out],
        check=True, env=merge_env,
    )
    lora_dir = os.path.join(merge_out, "lora_adapter")
    if not os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        raise RuntimeError(
            f"verl model_merger did not produce a peft adapter at {lora_dir} (no adapter_config.json); "
            "the merger output layout must be adjusted for this verl version."
        )
    for name in os.listdir(lora_dir):
        shutil.copy2(os.path.join(lora_dir, name), os.path.join(out_adapter_dir, name))
    shutil.rmtree(merge_out, ignore_errors=True)


def stamp_adapter_dir_provenance(adapter_dir: str, model_id: str, model_revision: str = "") -> None:
    """stamp the saved adapter's immutable base identity into adapter_config.json.

    dir-based analogue of the in-memory peft-model provenance stamp: same validation + fields,
    applied to the json verl produced. raises if the adapter already names a different base.
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
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


def run_verl_training(
    cmd: list[str],
    *,
    env: dict[str, str],
    on_step: Callable[[int], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
    step_pattern: str = r"step:\s*(\d+)",
    heartbeat_interval_s: float = 20.0,
) -> int:
    """run a verl trainer subprocess, streaming stdout and surfacing step progress.

    returns the process exit code. stdout+stderr are merged and scanned line by line: ``on_line``
    receives every line, ``on_step`` receives each parsed training step, and ``heartbeat`` is called
    at most once per ``heartbeat_interval_s``. callback failures terminate the child before they are
    re-raised so a failed required checkpoint upload cannot leave paid training running unattended.
    """
    step_re = re.compile(step_pattern)
    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    last_hb = 0.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if on_line is not None:
                on_line(line)
            m = step_re.search(line)
            if m and on_step is not None:
                on_step(int(m.group(1)))
            if heartbeat is not None:
                now = time.monotonic()
                if now - last_hb >= heartbeat_interval_s:
                    heartbeat()
                    last_hb = now
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
