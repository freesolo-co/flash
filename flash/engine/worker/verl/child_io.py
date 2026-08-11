"""Child-side shims the verl subprocess installs, and the stdout lines it reports back.

Two halves of one conversation with the verl child. The `render_*_shim` functions emit
sitecustomize source that the child executes -- tf32 matmul, the w&b link report, the GDN varlen
patch -- and the `parse_*` functions read what those shims print back out of the child's stdout,
turning it into the per-step metrics `flash runs log -f` renders.

Split out of `flash.engine.worker.backend_common` to keep that module under the file-size limit.
"""

from __future__ import annotations

import json
import math
import re

# --------------------------- tf32 matmul (all three verl backends) ---------------------------
# torch's TF32 flags are per-process, so setting them in the flash parent does not affect the verl
# child. opt in inside each child; bf16 parameters are unaffected.
FLASH_TF32_MARKER = "[flash-verl] tf32 matmul/cuDNN enabled"


def render_tf32_shim() -> str:
    """child-side sitecustomize fragment that enables TF32 matmul/cuDNN in the verl interpreter.

    never raises. tf32 is a throughput optimization, so a torch that cannot take these flags must
    leave training running on the default path rather than abort a paid run.
    """
    return f"""
# --- flash: enable tf32 in the trainer process (see backend_common.render_tf32_shim) ---
try:
    import torch as _flash_tf32_torch

    _flash_tf32_torch.set_float32_matmul_precision("high")
    _flash_tf32_torch.backends.cuda.matmul.allow_tf32 = True
    _flash_tf32_torch.backends.cudnn.allow_tf32 = True
    print({FLASH_TF32_MARKER!r}, flush=True)
except Exception:
    pass
"""


# ----------------------- tilelang libcudart stub (all three verl backends) -----------------------
# tilelang's bundled libcudart_stub.so does not export cudaDeviceReset. vLLM's CuMemAllocator
# resolves libcudart out of the loaded-library set and can bind that stub instead of the real
# runtime, so engine init dies with
#   AttributeError: .../tilelang/lib/libcudart_stub.so: undefined symbol: cudaDeviceReset
#   RuntimeError: Engine core initialization failed
#
# `perf._neutralize_tilelang_cudart_stub` fixes exactly this, but it runs in the flash PARENT and
# locates tilelang with `importlib.util.find_spec` THERE. Training runs in a separate interpreter
# (`FLASH_VERL_PYTHON`, `/opt/verl-venv`) that is built without --system-site-packages and where
# flash is not installed at all -- and Dockerfile.worker installs tilelang into BOTH (the main
# interpreter at the fla/Hopper layer, the child venv at its own layer), as does the run-time
# rebuild in `verl.capabilities.resolve_verl_python`. So the parent fix never reaches the copy the
# trainer actually loads, and the stub survives into the process that builds the vLLM engine.
#
# The allocator is only constructed when rollout sleep mode is on, which is why the one catalog
# model flagged `sleep_unsupported` (35B-A3B, pinned resident by `rollout_resident_overrides`) got
# past engine init while every other GRPO model crashed here.
#
# This is the same repoint the parent performs, re-expressed as a child fragment so it runs in the
# interpreter that owns the loaded stub. It must run BEFORE any vLLM/model import, so it is emitted
# at the top of the generated sitecustomize next to the tf32 fragment.
FLASH_CUDART_STUB_MARKER = "[flash-verl] tilelang libcudart stub repointed"


def render_tilelang_cudart_shim() -> str:
    """child-side sitecustomize fragment that repoints tilelang's libcudart stub at the real runtime.

    mirrors ``perf._neutralize_tilelang_cudart_stub`` inside the verl interpreter, and keeps its two
    load-bearing rules: never ``dlopen`` the stub to test it (that maps the stub into this process,
    which is the crash being avoided), and leave the stub alone when no real libcudart is found.

    never raises. an unrepointed stub only crashes the runs that build a sleeping vLLM engine, so a
    fragment that cannot find a runtime must leave the child to start rather than abort it here.
    """
    return f'''
# --- flash: repoint tilelang's libcudart stub (see child_io.render_tilelang_cudart_shim) ---
try:
    import ctypes as _flash_cudart_ctypes
    import ctypes.util as _flash_cudart_ctypes_util
    import glob as _flash_cudart_glob
    import importlib.util as _flash_cudart_importlib_util
    import os as _flash_cudart_os

    def _flash_cudart_verify(cand):
        """resolved path if cand loads AND exports cudaDeviceReset, else None."""
        try:
            _lib = _flash_cudart_ctypes.CDLL(cand)
        except OSError:
            return None
        if not hasattr(_lib, "cudaDeviceReset"):
            return None
        if _flash_cudart_os.path.isabs(cand) and _flash_cudart_os.path.exists(cand):
            return _flash_cudart_os.path.realpath(cand)
        _base = _flash_cudart_os.path.basename(cand)
        try:
            with open("/proc/self/maps") as _maps:
                for _line in _maps:
                    if _base in _line and "/" in _line:
                        _path = _line[_line.index("/"):].rstrip()
                        if _flash_cudart_os.path.basename(_path).startswith(
                            _base
                        ) and _flash_cudart_os.path.exists(_path):
                            return _flash_cudart_os.path.realpath(_path)
        except OSError:
            pass
        return None

    def _flash_cudart_find_real():
        """a libcudart that exports cudaDeviceReset, or None. same probe order as the parent."""
        _candidates = []
        try:
            import nvidia as _flash_cudart_nvidia  # namespace pkg; subpackage import may fail

            for _base_dir in sorted(map(str, getattr(_flash_cudart_nvidia, "__path__", []) or [])):
                _candidates += sorted(
                    _flash_cudart_glob.glob(
                        _flash_cudart_os.path.join(_base_dir, "*", "lib", "libcudart.so.*")
                    )
                )
        except Exception:
            pass
        for _pattern in (
            "/usr/local/cuda*/lib64/libcudart.so.*",
            "/usr/local/cuda*/targets/*/lib/libcudart.so.*",
            "/usr/lib/x86_64-linux-gnu/libcudart.so.*",
        ):
            _candidates += sorted(_flash_cudart_glob.glob(_pattern))
        _found = _flash_cudart_ctypes_util.find_library("cudart")
        if _found:
            _candidates.append(_found)
        for _cand in _candidates:
            _real = _flash_cudart_verify(_cand)
            if _real is not None:
                return _real
        return None

    try:
        _flash_cudart_spec = _flash_cudart_importlib_util.find_spec("tilelang")
    except Exception:
        _flash_cudart_spec = None
    _flash_cudart_locs = (
        list(getattr(_flash_cudart_spec, "submodule_search_locations", None) or [])
        if _flash_cudart_spec
        else []
    )
    if _flash_cudart_locs:
        _flash_cudart_stub = _flash_cudart_os.path.join(
            _flash_cudart_locs[0], "lib", "libcudart_stub.so"
        )
        # lexists: a dangling symlink still shadows, and exists() would follow it away.
        # do NOT probe the stub with CDLL -- that maps it into this process, the exact crash.
        if _flash_cudart_os.path.lexists(_flash_cudart_stub) and not (
            _flash_cudart_os.path.islink(_flash_cudart_stub)
            and _flash_cudart_os.path.exists(_flash_cudart_stub)
        ):
            _flash_cudart_real = _flash_cudart_find_real()
            if _flash_cudart_real is None:
                print(
                    "[flash-verl] tilelang libcudart stub: no real libcudart found; left as-is",
                    flush=True,
                )
            else:
                # ray starts several child interpreters against ONE venv, so every step below has
                # to be safe under concurrency. no check-then-act: link() and symlink() both fail
                # with FileExistsError rather than clobbering, which is the serialization.
                _flash_cudart_backup = _flash_cudart_stub + ".orig"
                try:
                    # hard link, NOT replace(): the backup is created without unlinking the stub,
                    # so a worker that loses this race still finds the stub in place. replace()
                    # would move it and hand the loser FileNotFoundError -- and once .orig exists,
                    # a second replace() would overwrite the preserved original with the symlink.
                    _flash_cudart_os.link(_flash_cudart_stub, _flash_cudart_backup)
                except FileExistsError:
                    pass  # another worker already preserved it
                except OSError:
                    pass  # cross-device or a filesystem without hard links: proceed unbacked
                # atomic swap through a private temp name: symlink() cannot overwrite, and an
                # unlink-then-symlink would leave a window where the path does not resolve at all.
                _flash_cudart_tmp = _flash_cudart_stub + ".flash-" + str(_flash_cudart_os.getpid())
                try:
                    _flash_cudart_os.symlink(_flash_cudart_real, _flash_cudart_tmp)
                    _flash_cudart_os.replace(_flash_cudart_tmp, _flash_cudart_stub)
                finally:
                    try:
                        _flash_cudart_os.remove(_flash_cudart_tmp)
                    except OSError:
                        pass
                print(
                    {FLASH_CUDART_STUB_MARKER!r} + " -> " + _flash_cudart_real,
                    flush=True,
                )
except Exception as _flash_cudart_exc:
    print("[flash-verl] tilelang libcudart stub repoint failed: " + repr(_flash_cudart_exc), flush=True)
'''


# --------------------------- w&b run link (all three verl backends) ---------------------------
# verl calls wandb.init INSIDE the training subprocess, so the flash parent's
# wandb.run is always None and anything read up here would be a silent no-op. the url is not
# derivable up here
# either: it needs entity/project/runs/<wandb-generated-id>, and flash only knows the project and
# its own run NAME. so the child reports it back over the marker channel the parent already scans.
FLASH_WANDB_LINK_MARKER = "FLASH_WANDB_LINK"

_WANDB_LINK_RE = re.compile(rf"{FLASH_WANDB_LINK_MARKER}\s+(\{{.*\}})\s*$")


def render_wandb_link_shim() -> str:
    """child-side sitecustomize fragment that reports the live w&b run's url and id.

    wraps wandb.init rather than reading wandb.run at import time: sitecustomize runs long before
    verl's tracking module initializes the run, so anything read here would always be None. never
    raises. wandb is optional and a logging link must not be able to abort paid training, so every
    failure path leaves the run unlinked instead of dead.
    """
    return f"""
# --- flash: report the w&b run link to the parent (see backend_common.render_wandb_link_shim) ---
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
"""


FLASH_GDN_VARLEN_MARKER = "[flash-verl] gdn packed-boundary resets active"


def render_gdn_varlen_shim(model_type: str) -> str:
    """child-side sitecustomize fragment that resets GDN state at packed example boundaries.

    patch the checkpoint's exact ``model_type`` because ``qwen3_5`` and ``qwen3_5_moe`` use different
    modules; model names are not authoritative. patch the text-model forward so one derivation reaches
    every GDN layer, and use transformers' own boundary helper to match attention segmentation.

    keep padded input inert, preserve explicit kwargs, and fail closed because silent boundary
    contamination is worse than refusing to start.
    """
    return f'''
# --- flash: reset gdn state at packed example boundaries (backend_common.render_gdn_varlen_shim) ---
import importlib as _flash_gdn_importlib

import torch as _flash_gdn_torch
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence as _flash_gdn_is_packed,
    prepare_fa_kwargs_from_position_ids as _flash_gdn_prepare_fa_kwargs,
)

_flash_gdn_modeling = _flash_gdn_importlib.import_module(
    "transformers.models.{model_type}.modeling_{model_type}"
)
# raises if absent, deliberately: this shim is only rendered once the gate says resets are honored,
# so a missing TextModel means the gate and the model disagree -- refuse rather than train packed
# with an unpatched forward.
_flash_gdn_text_model = next(
    c
    for n, c in vars(_flash_gdn_modeling).items()
    if isinstance(c, type) and n.endswith("TextModel")
)


def _flash_gdn_seq_idx(position_ids, cu_seq_lens):
    """per-token example ordinal, int32 (1, total_nnz) -- what causal_conv1d_fn wants."""
    lengths = cu_seq_lens.diff()
    return (
        _flash_gdn_torch.repeat_interleave(
            _flash_gdn_torch.arange(
                lengths.numel(), device=position_ids.device, dtype=_flash_gdn_torch.int32
            ),
            lengths.to(_flash_gdn_torch.int64),
        )
        .unsqueeze(0)
        .contiguous()
    )


def _flash_patch_gdn_varlen():
    original = _flash_gdn_text_model.forward

    def forward(self, *args, **kwargs):
        # only derive what the caller did not supply, and only for a genuinely packed batch.
        if kwargs.get("cu_seq_lens_q") is None and kwargs.get("seq_idx") is None:
            position_ids = kwargs.get("position_ids")
            # the model reshapes position_ids to 4-way mrope internally; the text row is what the
            # boundaries live on. a 3d (4, batch, seq) tensor indexes to it, 2d is already it.
            text_position_ids = position_ids
            if text_position_ids is not None and text_position_ids.ndim == 3:
                text_position_ids = text_position_ids[0]
            if text_position_ids is not None and text_position_ids.ndim == 2:
                if _flash_gdn_is_packed(text_position_ids, text_position_ids.shape[0]):
                    (cu_seq_lens_q, cu_seq_lens_k), (max_q, max_k) = (
                        _flash_gdn_prepare_fa_kwargs(text_position_ids)
                    )
                    kwargs["cu_seq_lens_q"] = cu_seq_lens_q
                    kwargs["cu_seq_lens_k"] = cu_seq_lens_k
                    kwargs["max_length_q"] = max_q
                    kwargs["max_length_k"] = max_k
                    kwargs["seq_idx"] = _flash_gdn_seq_idx(text_position_ids, cu_seq_lens_q)
        return original(self, *args, **kwargs)

    _flash_gdn_text_model.forward = forward


if not getattr(_flash_gdn_text_model.forward, "_flash_gdn_varlen_patched", False):
    _flash_patch_gdn_varlen()
    _flash_gdn_text_model.forward._flash_gdn_varlen_patched = True
    print({FLASH_GDN_VARLEN_MARKER!r}, "{model_type}", flush=True)
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
# an in-process trainer would feed `metrics_last` from a TrainerCallback, which verl cannot host:
# its trainer runs out of process. verl's LocalLogger prints exactly one line per optimizer update
# -- "step:N - key:value - key:value" over every scalar metric -- so the parent reconstructs the
# same backlog from that line. the payload schema is the CLI's, not verl's: keys below are what
# flash/cli/commands/__init__.py:_FOLLOW_METRIC_FIELDS renders.

# the left edge every verl step pattern shares. "not part of a longer word" rather than start-of-
# line, because ray tags worker stdout with a "(TaskRunner pid=123) " prefix -- an anchored match
# would parse nothing at all in production. and not "preceded by whitespace" either (VERL-134):
# verl's LocalLogger shares its stream with tqdm, which ends a bar with "]" and no trailing
# newline, so the metric line arrives glued to it -- "...81.49s/it]step:2 - train/loss:...". a \s
# edge matched step 1 and missed every step after it, which froze the sft heartbeat on step 1's
# metrics and kept the zero-grad guard from ever arming. excluding / as well as word characters
# still refuses "global_step:" and a path's ".../step:".
_VERL_STEP_LEFT_EDGE = r"(?:^|[^\w/])"

# the gate the trainers share, via verl_step_number below: any step-tagged line.
_VERL_STEP_GATE_RE = re.compile(_VERL_STEP_LEFT_EDGE + r"step:(\d+)(?:\s|$)")

# the `metrics_last` row pattern additionally requires the " - " metric separator, so a step-tagged
# line carrying no metrics at all cannot open a row.
_VERL_STEP_RE = re.compile(_VERL_STEP_LEFT_EDGE + r"step:(\d+) - ")

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


def verl_step_number(line: str) -> int | None:
    """the optimizer step a verl log line is tagged with, or None when it is not a step line.

    the trainers' stdout callbacks gate every metric read on this: a line with no step carries no
    step-scoped metric worth recording. it lives here rather than in each trainer because all three
    verl paths read one stdout format -- VERL-134 was one format change that had to be fixed at
    three copies at once, and a path whose copy was missed drops steps silently rather than failing.
    """
    match = _VERL_STEP_GATE_RE.search(line)
    return None if match is None else int(match.group(1))


def parse_verl_step_metrics(line: str) -> dict | None:
    """the flash `metrics_last` entry a verl step line carries, or None when it carries none.

    malformed multi-rank output must not fail a paid run. return None for validation-only lines with no
    renderable metric, or they can occupy a deduplicated step and displace its training record.
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
    repeat must replace rather than append -- otherwise the CLI renders the same step twice. the
    heartbeat thread reads ``backlog`` while the stdout loop writes it, so the new contents are
    published in ONE slice assignment: a filter/append/truncate sequence would let that reader
    observe a torn intermediate state with the step momentarily missing.
    """
    step = metrics.get("step")
    kept = [item for item in backlog if item.get("step") != step]
    kept.append(metrics)
    backlog[:] = kept[-limit:]
