"""shared out-of-process verl harness reused by the sft and opd verl trainers.

verl pins its own torch/vllm, incompatible with flash's, so flash never imports verl in-process: it
runs a verl trainer entrypoint as a subprocess against a separate interpreter and merges the
fsdp-sharded lora checkpoint back into a flash-servable peft adapter. these are the framework- and
algorithm-neutral pieces (interpreter resolution, checkpoint export, provenance, progress
streaming); the sft/opd/grpo modules layer their own dataset rows, hydra overrides, and orchestration
on top.
"""

from __future__ import annotations

import collections
import contextlib
import json
import math
import os
import re
import shutil
import signal
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


def clamp_engine_len(engine_len: int, max_position_embeddings: int | None) -> int:
    """the engine length verl will accept: the job's context, capped at the model's own limit.

    verl raises ValueError when rollout.max_model_len exceeds the model's max_position_embeddings,
    so a job whose max_context_tokens overshoots the architecture would die at rollout startup
    rather than train on a shorter context. clamp instead. an unknown limit (probe failed) passes
    through untouched, leaving verl's own resolution in charge.
    """
    if max_position_embeddings is None or max_position_embeddings <= 0:
        return int(engine_len)
    return min(int(engine_len), int(max_position_embeddings))


def model_max_position_embeddings(model_id: str, revision: str = "") -> int | None:
    """the model's architectural context limit, or None when it cannot be determined.

    mirrors verl's own lookup (workers/rollout/utils.py:get_max_position_embeddings): the top-level
    attribute, falling back to the nested text_config that multimodal architectures use. every model
    in the flash catalog nests it there. None on any probe failure, so a config that cannot be read
    never blocks a run that works today.
    """
    try:
        from transformers import AutoConfig

        from flash.engine.worker.hf import model_revision_kwargs

        cfg = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, **model_revision_kwargs(revision)
        )
        limit = getattr(cfg, "max_position_embeddings", None)
        if limit is None:
            text_cfg = getattr(cfg, "text_config", None)
            limit = getattr(text_cfg, "max_position_embeddings", None) if text_cfg else None
        return int(limit) if limit else None
    except Exception as e:  # an unreadable config must not fail the run
        print(f"[verl] max_position_embeddings probe failed for {model_id!r}: {e}", flush=True)
        return None


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
    VERL_REQUIREMENT, so an empty/absent FLASH_VERL_PYTHON always yields the pinned verl.

    an EMPTY value is deliberately equivalent to absent, and it is the only route a run has: worker
    images can export FLASH_VERL_PYTHON themselves, and [worker_env] can set a key but never delete
    one, so omitting it from the spec leaves the image's value in place. `""` is what lets a spec
    say "ignore the image's interpreter and provision the pinned one".
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


def resolve_checkpoint_actor_dir(step_dir: str) -> str:
    """return the directory inside ``global_step_N`` that holds the saved model + ``huggingface/``.

    verl's two trainers do not agree on this layout, and the difference is not configurable:

    - RL nests per-role, ``global_step_N/actor/`` (``trainer/ppo/ray_trainer.py`` builds
      ``os.path.join(local_global_step_folder, "actor")``).
    - SFT writes the shards straight into ``global_step_N/`` (``utils/checkpoint/
      checkpoint_handler.py`` passes ``local_path=local_global_step_folder`` unmodified).

    So detect the layout from what verl actually wrote rather than hardcoding either convention.
    ``huggingface/`` is the right marker because it is the subfolder ``verl.model_merger`` resolves
    ``hf_model_config_path`` against, which is precisely what a wrong answer here breaks.

    Getting this wrong does not surface as a missing path: ``AutoConfig.from_pretrained`` treats a
    nonexistent local directory as a *hub repo id*, so the failure arrives as
    ``HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`` and reads
    like a credentials or network problem instead of a path bug.
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
    ``verl.model_merger merge`` writes a standard peft adapter (adapter_config.json +
    adapter_model.safetensors) to a ``lora_adapter/`` subfolder of its target; we copy just that
    adapter into flash's adapter dir (the co-produced merged full model is discarded -- flash
    serves the lora on the immutable base).

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


# how many of the child's most recent output lines to retain for stall reporting. the child's last
# words before it wedges are the whole diagnostic, and a stall is usually preceded by a short burst
# (a ray warning, a placement-group notice, a partial traceback), so a small window suffices.
CHILD_TAIL_LINES = 60
# per-line cap when rendering the retained tail. verl prints resolved-config blocks thousands of
# characters wide; an unbounded tail would blow the heartbeat payload it has to travel inside.
_CHILD_TAIL_LINE_CHARS = 300
# how many retained lines ride along on a pre-first-step heartbeat. narrower than what is retained:
# this payload is uploaded every tick, so it stays small enough not to bloat the snapshot.
STALL_TAIL_LINES = 15


class ChildOutputTail:
    """bounded ring buffer of a subprocess's most recent output lines.

    exists because the verl child's stdout reaches **no collected log stream**. ``run_verl_training``
    re-prints every child line to the parent worker's stdout, and the parent's stdout is not on the
    path the plane scrapes -- only the ``HEARTBEAT <json>`` marker line is (``heartbeat.py``). so a
    child that wedges produces zero retrievable bytes: an OPD arm stalled 3/3 attempts on 3 separate
    endpoints and every attempt cost ~50 minutes of paid H100 time to learn nothing (ISSUES VERL-061).

    retaining the tail in memory lets a stall report the child's own last words through the marker
    channel that provably survives, instead of inferring the cause from the outside.
    """

    def __init__(self, limit: int = CHILD_TAIL_LINES) -> None:
        self._lines: collections.deque[str] = collections.deque(maxlen=limit)

    def record(self, line: str) -> None:
        text = line.rstrip("\n")
        if text:
            self._lines.append(text[:_CHILD_TAIL_LINE_CHARS])

    def tail(self, limit: int | None = None) -> list[str]:
        """the retained lines, oldest first, optionally narrowed to the most recent ``limit``."""
        lines = list(self._lines)
        if limit is not None and limit >= 0:
            lines = lines[len(lines) - limit :] if limit < len(lines) else lines
        return lines


def stall_tail_fields(
    step: int, tail: ChildOutputTail, limit: int = STALL_TAIL_LINES
) -> dict[str, object]:
    """heartbeat fields carrying the child's last words, but only while it has made no progress.

    a run that is training is diagnosable from its step/loss stream, so attaching the tail then would
    add an uploaded payload every tick for no information. before the first step there is no such
    stream, and that is exactly the window a setup stall lands in -- the child prints its complaint,
    nobody collects the parent's stdout, and the run dies to a watchdog with zero evidence.

    returns an empty dict once ``step`` advances, or when the child has said nothing yet.
    """
    if step > 0:
        return {}
    recent = tail.tail(limit=limit)
    return {"child_tail": recent} if recent else {}


def run_verl_training(
    cmd: list[str],
    *,
    env: dict[str, str],
    on_step: Callable[[int], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    heartbeat: Callable[[], None] | None = None,
    step_pattern: str = r"step:\s*(\d+)",
    heartbeat_interval_s: float = 20.0,
    tail: ChildOutputTail | None = None,
) -> int:
    """run a verl trainer subprocess, streaming stdout and surfacing step progress.

    returns the process exit code. stdout+stderr are merged and scanned line by line: ``on_line``
    receives every line, ``on_step`` receives each parsed training step, and ``heartbeat`` is called
    at most once per ``heartbeat_interval_s``. callback failures terminate the child before they are
    re-raised so a failed required checkpoint upload cannot leave paid training running unattended.

    ``tail``, when supplied, retains the child's most recent lines so a caller that observes a stall
    can report what the child last said. the parent's own stdout reaches no collected stream, so this
    buffer is the only way the child's words escape the container (see ``ChildOutputTail``).

    the child gets its own session so teardown can signal the whole process group. verl spawns vllm's
    EngineCore as a grandchild, and terminating only the direct child reparents that grandchild to
    init with its cuda context intact, so the gpu stays allocated and the next run waits forever on
    device memory that nothing will release.
    """
    step_re = re.compile(step_pattern)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    last_hb = 0.0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if tail is not None:
                tail.record(line)
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
        _kill_process_group(proc)
        raise
    finally:
        if proc.poll() is None:
            proc.wait()
    return int(proc.returncode)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """signal the child's whole process group, escalating to SIGKILL if it does not exit.

    signalling the group rather than the pid is what reaches vllm's EngineCore grandchild; a survivor
    holds its cuda context and strands the gpu for every later run.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


# --------------------------- w&b run link (all three verl backends) ---------------------------
# trl spreads wandb_run_info() into its notes, giving the sdk's link_wandb a clickable
# notes["wandb_url"]. verl calls wandb.init INSIDE the training subprocess, so the flash parent's
# wandb.run is None and that spread would be a silent no-op. the url is not derivable up here
# either: it needs entity/project/runs/<wandb-generated-id>, and flash only knows the project and
# its own run NAME. so the child reports it back over the marker channel the parent already scans.
FLASH_WANDB_LINK_MARKER = "FLASH_WANDB_LINK"

_WANDB_LINK_RE = re.compile(rf"{FLASH_WANDB_LINK_MARKER}\s+(\{{.*\}})\s*$")


def render_wandb_link_shim() -> str:
    """child-side sitecustomize fragment that reports the live w&b run's url and id.

    wraps wandb.init rather than reading wandb.run at import time: sitecustomize runs long before
    verl's tracking module initializes the run, so anything read here would always be None.

    never raises. wandb is optional and a logging link must not be able to abort paid training, so
    every failure path leaves the run unlinked instead of dead.
    """
    return f'''
# --- flash: report the w&b run link to the parent (see verl_common.render_wandb_link_shim) ---
try:
    import json as _flash_wandb_json

    import wandb as _flash_wandb

    _flash_wandb_init = _flash_wandb.init

    def _flash_wandb_init_reporting(*args, **kwargs):
        _run = _flash_wandb_init(*args, **kwargs)
        try:
            _url = getattr(_run, "url", None)
            _id = getattr(_run, "id", None)
            if _url:
                print(
                    "{FLASH_WANDB_LINK_MARKER} "
                    + _flash_wandb_json.dumps({{"wandb_url": _url, "wandb_id": _id}}),
                    flush=True,
                )
        except Exception:
            pass
        return _run

    _flash_wandb.init = _flash_wandb_init_reporting
except Exception:
    pass
'''


def parse_wandb_link(line: str) -> dict | None:
    """the {{wandb_url, wandb_id}} a marker line carries, or None for every other line.

    returns None rather than raising on a malformed payload: this parses child stdout, and a
    truncated or interleaved line under multi-rank logging must not take down the run.
    """
    match = _WANDB_LINK_RE.search(line)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    url = payload.get("wandb_url")
    # https only: the sdk renders this as a clickable dashboard link, and child stdout also carries
    # rollout text, so the field must not be able to become an arbitrary scheme.
    if not isinstance(url, str) or not url.startswith("https://"):
        return None
    wandb_id = payload.get("wandb_id")
    return {"wandb_url": url, "wandb_id": wandb_id if isinstance(wandb_id, str) else None}


# --------------------------- per-step grpo metrics (verl -> `flash runs log -f`) ---------------------------
# trl feeds `metrics_last` from a TrainerCallback (heartbeat.make_reward_heartbeat_callback), which
# verl cannot use: its trainer runs out of process. verl's LocalLogger prints exactly one line per
# optimizer update -- "step:N - key:value - key:value" over every scalar metric -- so the parent
# reconstructs the same backlog from that line. the payload schema is the CLI's, not verl's: keys
# below are what flash/cli/commands.py:_FOLLOW_METRIC_FIELDS renders.
#
# not anchored at line start: ray tags worker stdout with a "(TaskRunner pid=123) " prefix, so an
# anchored match would parse nothing at all in production. the existing progress regex in rl_verl
# is unanchored for the same reason.
_VERL_STEP_RE = re.compile(r"(?:^|\s)step:(\d+) - ")

# verl metric key -> flash `metrics_last` field. verl has no counterpart for trl's
# frac_reward_zero_std (an advantage-collapse fraction trl computes per group), so that column is
# simply absent for verl runs rather than faked; the renderer skips fields it does not find.
_VERL_METRIC_FIELDS = (
    ("critic/rewards/mean", "reward"),
    ("actor/grad_norm", "grad_norm"),
    ("actor/kl_loss", "kl"),
    ("actor/entropy", "entropy"),
    ("response_length/mean", "mean_completion_tokens"),
    ("response_length/clip_ratio", "truncation_rate"),
)

# verl reduces most metrics with np.mean and formats them through pprint, so under numpy>=2 a value
# prints as "np.float32(1.25)" rather than "1.25". verl's own requirements pin numpy<2, but the
# production interpreter comes from FLASH_VERL_PYTHON (a prebuilt image flash does not own), so
# accept both spellings -- the numpy-2 form would otherwise drop these columns silently.
_NUMPY_SCALAR_RE = re.compile(r"^np\.\w+\((.*)\)$")


def _metric_value(raw: str) -> float | None:
    """one verl-printed scalar as a finite float, or None when it is not usable."""
    wrapper = _NUMPY_SCALAR_RE.match(raw)
    if wrapper is not None:
        raw = wrapper.group(1)
    try:
        value = float(raw)
    except ValueError:
        return None
    # nan/inf survive float() ("nan", "inf") but would render as a meaningless column and can poison
    # a json payload downstream, so drop them the way the trl callback does.
    return value if math.isfinite(value) else None


def parse_verl_metric(line: str, verl_key: str) -> float | None:
    """one verl metric's finite float value from a step line, or None when absent.

    anchors the key on a separator so a metric whose name merely ends with another's
    (response_length_non_aborted/mean vs response_length/mean) cannot cross-match, and accepts the
    numpy>=2 ``np.float64(...)`` spelling (see _NUMPY_SCALAR_RE).
    """
    hit = re.search(rf"(?:^|[\s-]){re.escape(verl_key)}:(\S+)", line)
    return None if hit is None else _metric_value(hit.group(1))


def parse_verl_step_metrics(line: str) -> dict | None:
    """the flash `metrics_last` entry a verl step line carries, or None when it carries none.

    returns None rather than raising: this parses child stdout under multi-rank logging, where a
    truncated or interleaved line must not take down a paid run.

    a step line with no renderable metric yields None rather than a bare ``{"step": n}``: verl logs
    its pre-training validation pass as its own record at the *current* step counter
    (ray_trainer.py `logger.log(data=val_metrics, step=self.global_steps)`), and every key on that
    line is namespaced val-core/ or val-aux/. keeping it would render a row with a step number and
    no metrics, and -- because the backlog is deduplicated by step -- a resumed run's validation
    pass would land on the resume step and displace a real training row.
    """
    match = _VERL_STEP_RE.search(line)
    if match is None:
        return None
    metrics: dict[str, float | int] = {}
    for verl_key, flash_key in _VERL_METRIC_FIELDS:
        value = parse_verl_metric(line, verl_key)
        if value is not None:
            metrics[flash_key] = value
    if not metrics:
        return None
    return {"step": int(match.group(1)), **metrics}


def append_step_metrics(backlog: list[dict], metrics: dict, *, limit: int) -> None:
    """record one step in a bounded, de-duplicated backlog, mirroring the trl callback.

    verl reprints a step on a validation pass, and a resumed run replays its resume step, so a
    repeat must replace rather than append -- otherwise the CLI renders the same step twice.

    the heartbeat thread reads ``backlog`` while the stdout loop writes it, so the new contents are
    published in ONE slice assignment: a filter/append/truncate sequence would let that reader
    observe a torn intermediate state with the step momentarily missing.
    """
    step = metrics.get("step")
    kept = [item for item in backlog if item.get("step") != step]
    kept.append(metrics)
    backlog[:] = kept[-limit:]
