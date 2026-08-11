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


def _gdn_varlen_patch_source(model_type: str) -> str:
    """the shim's import-time-free half: the target, the seq_idx helper, and the patch itself."""
    return f'''
# --- flash: reset gdn state at packed example boundaries (backend_common.render_gdn_varlen_shim) ---
import sys as _flash_gdn_sys

_FLASH_GDN_TARGET = "transformers.models.{model_type}.modeling_{model_type}"


def _flash_gdn_seq_idx(position_ids, cu_seq_lens):
    """per-token example ordinal, int32 (1, total_nnz) -- what causal_conv1d_fn wants."""
    import torch as _flash_gdn_torch

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


def _flash_patch_gdn_varlen(modeling):
    # raises if absent, deliberately: this shim is only rendered once the gate says resets are
    # honored, so a missing TextModel means the gate and the model disagree -- refuse rather than
    # train packed with an unpatched forward.
    text_model = next(
        c
        for n, c in vars(modeling).items()
        if isinstance(c, type) and n.endswith("TextModel")
    )
    if getattr(text_model.forward, "_flash_gdn_varlen_patched", False):
        return
    # imported here, not at module scope: these pull in torch and transformers' cuda probes.
    from transformers.modeling_flash_attention_utils import (
        _is_packed_sequence as _flash_gdn_is_packed,
        prepare_fa_kwargs_from_position_ids as _flash_gdn_prepare_fa_kwargs,
    )

    original = text_model.forward

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

    forward._flash_gdn_varlen_patched = True
    text_model.forward = forward
    print({FLASH_GDN_VARLEN_MARKER!r}, "{model_type}", flush=True)
'''


# the arming half. kept separate only so neither piece runs past the repo's function-length limit;
# the two are concatenated verbatim and must stay in this order.
_GDN_VARLEN_ARM_SOURCE = '''

class _FlashGdnLoader:
    """wrap the real loader so the patch lands once the module is fully executed.

    ``exec_module`` returning means the module object the importer will hand the caller is
    finished, so patching here is the first safe moment. patching from ``find_spec`` instead
    would run against a half-built module that the real import then replaces, leaving the
    caller's class unpatched while the marker still printed.
    """

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _flash_gdn_uninstall()
        _flash_patch_gdn_varlen(module)

    def __getattr__(self, name):  # keep the rest of the loader protocol intact
        return getattr(self._inner, name)


class _FlashGdnFinder:
    """intercept the first import of the modeling module, then get out of the way.

    delegating to the finders AFTER this one resolves the real spec without importing anything,
    so nothing here touches torch or cuda; only the loader is wrapped.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _FLASH_GDN_TARGET:
            return None
        import importlib.util

        rest = [f for f in _flash_gdn_sys.meta_path if not isinstance(f, _FlashGdnFinder)]
        for finder in rest:
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _FlashGdnLoader(spec.loader)
                return spec
        return None


def _flash_gdn_uninstall():
    _flash_gdn_sys.meta_path[:] = [
        f for f in _flash_gdn_sys.meta_path if not isinstance(f, _FlashGdnFinder)
    ]


# already imported (a parent process may have loaded it): patch now, there is no import left to
# intercept. otherwise arm the finder and let the child's own import trigger it.
_flash_gdn_loaded = _flash_gdn_sys.modules.get(_FLASH_GDN_TARGET)
if _flash_gdn_loaded is not None:
    _flash_patch_gdn_varlen(_flash_gdn_loaded)
else:
    _flash_gdn_sys.meta_path.insert(0, _FlashGdnFinder())
'''


def render_gdn_varlen_shim(model_type: str) -> str:
    """child-side sitecustomize fragment that resets GDN state at packed example boundaries.

    patch the checkpoint's exact ``model_type`` because ``qwen3_5`` and ``qwen3_5_moe`` use different
    modules; model names are not authoritative. patch the text-model forward so one derivation reaches
    every GDN layer, and use transformers' own boundary helper to match attention segmentation.

    keep padded input inert, preserve explicit kwargs, and fail closed because silent boundary
    contamination is worse than refusing to start.

    defer every import to the moment the child imports the modeling module itself. sitecustomize
    runs at INTERPRETER STARTUP, and ray starts each actor's interpreter before it narrows that
    actor's ``CUDA_VISIBLE_DEVICES`` to its own card -- so an import here runs while the process
    still sees every gpu. importing the modeling module transitively calls transformers'
    ``is_flash_linear_attention_available``/``is_causal_conv1d_available``, both of which call
    ``torch.cuda.is_available()``, and fla queries device capability at import. that initializes
    cuda against the FULL device list, and a later env-only change cannot rebuild the device map,
    so every rank keeps device 0 and nccl aborts with ``Duplicate GPU detected``. a meta_path
    finder carries no import cost and fires after ray has pinned the actor.
    """
    return _gdn_varlen_patch_source(model_type) + _GDN_VARLEN_ARM_SOURCE


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
