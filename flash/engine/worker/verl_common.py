"""shared out-of-process verl harness reused by the sft and opd verl trainers.

verl pins its own torch/vllm, incompatible with flash's, so flash never imports verl in-process: it
runs a verl trainer entrypoint as a subprocess against a separate interpreter and merges the
fsdp-sharded lora checkpoint back into a flash-servable peft adapter. these are the framework- and
algorithm-neutral pieces (interpreter resolution, checkpoint export, provenance, progress
streaming); the sft/opd/grpo modules layer their own dataset rows, hydra overrides, and orchestration
on top.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable

# verl 0.8.0 exactly, plus the truncation-mask and 3d position id commits. it must stay on the 0.8.0
# base: the opd plugin patches 0.8.0 internals and imports verl.trainer.main_ppo_sync, which verl
# deleted after 0.8.0, and opd's exact-version gate reads the version file this branch pins to the
# release value.
VERL_REQUIREMENT = (
    "verl @ git+https://github.com/freesolo-co/verl@b7492fa3b7ab843294d06dbf754e887950f559c7"
)


def agent_loop_workers(rollout_batch: int, *, cap: int = 8) -> int:
    """largest divisor of ``rollout_batch`` that is <= ``cap`` (verl's default worker count).

    verl 0.8.0 hardcodes async rollout (ray_trainer.py:914), so every rollout goes through
    AgentLoopManager, which splits the batch across agent.num_workers with DataProto.chunk and
    asserts the split is exact (agent_loop.py:1111 -> protocol.py:874). the worker count must
    therefore divide the batch. returning a divisor keeps the run alive for any batch size while
    preserving parallelism whenever the batch permits it.
    """
    if rollout_batch <= 0:
        raise ValueError("rollout_batch must be positive")
    return next(n for n in range(min(cap, rollout_batch), 0, -1) if rollout_batch % n == 0)


def verl_supports_rollout_field(python_bin: str, field: str) -> bool:
    """report whether python_bin's verl declares `field` on RolloutConfig.

    the fork adds rollout fields stock verl does not have. hydra composes an unknown key happily and
    only fails later in omega_conf_to_dataclass, so callers ask here before emitting a fork-only
    override rather than letting the run abort at dataclass conversion.
    """
    probe = (
        "from verl.workers.config.rollout import RolloutConfig;"
        f"print('1' if {field!r} in RolloutConfig.__dataclass_fields__ else '0')"
    )
    try:
        done = subprocess.run(
            [python_bin, "-c", probe], capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and done.stdout.strip().endswith("1")


def resolve_verl_python(workdir: str, *, install_wandb: bool = False) -> str:
    """return an interpreter that can import verl.

    prefers FLASH_VERL_PYTHON (set by the caller when verl is preinstalled, e.g. on a verl image).
    otherwise provisions an isolated venv on the pod so verl's torch/vllm never touch flash's env.
    optionally installs wandb best-effort for callers that enable verl's wandb logger.

    the preset is returned as-is: flash does not own that interpreter and must not mutate it. it can
    hold a verl without the fork's rollout fields, so callers emitting a fork-only override gate it
    on verl_supports_rollout_field.

    a self-provisioned venv is flash's own, and is rebuilt whenever it does not record the current
    VERL_REQUIREMENT, so unsetting FLASH_VERL_PYTHON always yields the pinned verl.
    """
    preset = os.environ.get("FLASH_VERL_PYTHON", "").strip()
    if preset:
        return preset
    venv = os.path.join(workdir, "verl-venv")
    py = os.path.join(venv, "bin", "python")
    stamp = os.path.join(venv, "flash-verl-requirement")
    installed = ""
    if os.path.exists(stamp):
        with open(stamp) as f:
            installed = f.read().strip()
    if installed != VERL_REQUIREMENT or not os.path.exists(py):
        # a retry reuses the pod workdir, so this venv can be from an earlier attempt, from an earlier
        # flash release pinning a different verl, or from an install that died partway. remove it and
        # start clean: reusing it would train on the wrong verl, and `uv venv` refuses to write into a
        # directory that already holds a pyvenv.cfg, so a half-built venv wedges every later retry.
        shutil.rmtree(venv, ignore_errors=True)
        # dev-only fallback (production uses FLASH_VERL_PYTHON on a prebuilt verl image): verl brings
        # its own torch/vllm, so use a full install rather than --no-deps to include runtime deps.
        subprocess.run(["uv", "venv", venv], check=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                py,
                VERL_REQUIREMENT,
                "liger-kernel",
                "bitsandbytes>=0.49",
                "qwen-vl-utils",
                "torchvision",
                "xgrammar==0.1.25",
                "tqdm",
                "pyarrow",
            ],
            check=True,
        )
        with open(stamp, "w") as f:
            f.write(VERL_REQUIREMENT)
        if install_wandb:
            # verl does not pull wandb; install it best-effort so logger setup can fall back to console.
            subprocess.run(["uv", "pip", "install", "--python", py, "wandb"], check=False)
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
