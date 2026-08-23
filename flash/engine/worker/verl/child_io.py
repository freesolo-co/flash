"""startup shims and stdout parsing for isolated verl children.

Only TF32 and TileLang libcudart repair run from sitecustomize. Verl-resident patches are installed
by copied algorithm plugins through VERL_USE_EXTERNAL_MODULES. Parent-side marker and metric parsing
remain here because they consume child files and stdout without importing verl.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import textwrap

# imported BY VALUE on purpose. perf's docstring notes that tests monkeypatch
# `perf._find_real_libcudart` and its callers resolve it through the patched module globals -- this
# caller must NOT. it needs the function's SOURCE to ship into the child, so resolving through
# `perf.<attr>` would render a test double's body into the fragment and emit a broken shim. the
# from-import keeps the real object bound here regardless of what is patched over there.
from flash.engine.worker.perf import _find_real_libcudart

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
# This repoint used to live in the flash PARENT, which located tilelang with
# `importlib.util.find_spec` THERE. Training runs in a separate interpreter
# (`FLASH_VERL_PYTHON`, `/opt/verl-venv`) that is built without --system-site-packages and where
# flash is not installed at all -- and Dockerfile.worker installs tilelang into BOTH (the main
# interpreter at the fla/Hopper layer, the child venv at its own layer), as does the run-time
# rebuild in `verl.capabilities.resolve_verl_python`. So a parent-side repoint never reached the copy
# the trainer loads, and the stub survived into the process that builds the vLLM engine.
#
# The allocator is only constructed when rollout sleep mode is on, which is why the one catalog
# model flagged `sleep_unsupported` (35B-A3B, pinned resident by `rollout_resident_overrides`) got
# past engine init while every other GRPO model crashed here.
#
# So the repoint lives HERE and only here: a child fragment running in the interpreter that owns the
# loaded stub. It must run BEFORE any vLLM/model import, so it is emitted at the top of the generated
# sitecustomize next to the tf32 fragment. The parent no longer repoints anything -- it never builds a
# vLLM engine, so its own copy of the stub is inert (see the note in `perf/__init__.py`).
FLASH_CUDART_STUB_MARKER = "[flash-verl] tilelang libcudart stub repointed"


def render_tilelang_cudart_shim() -> str:
    """child-side sitecustomize fragment that repoints tilelang's libcudart stub at the real runtime.

    this is the only place the repoint happens: the stub only matters to the interpreter that builds
    a sleeping vLLM engine, and that is the verl child. two load-bearing rules: never ``dlopen`` the
    stub to test it (that maps the stub into this process, which is the crash being avoided), and
    leave the stub alone when no real libcudart is found.

    never raises. an unrepointed stub only crashes the runs that build a sleeping vLLM engine, so a
    fragment that cannot find a runtime must leave the child to start rather than abort it here.

    SHIPS THE PARENT'S OWN PROBE rather than restating it. ``perf._find_real_libcudart`` is the
    canonical one -- nvidia wheel dirs across cuda majors, the -devel toolkit, the system resolver,
    plus proc-filesystem resolution for a bare soname -- and a hand-copy here would silently keep the
    old probe the next time a cuda release moves those paths, which is exactly the parent/child skew
    this whole fragment exists to fix. ``inspect.getsource`` keeps one definition.

    the probe only closes over ``os`` from its module scope (everything else is a builtin or bound
    inside it), so the fragment imports ``os`` under both the real name -- which the shipped source
    refers to -- and a private alias the swap below uses, so it cannot be shadowed by child code.
    """
    probe_src = textwrap.indent(textwrap.dedent(inspect.getsource(_find_real_libcudart)), "    ")
    # the parent's docstrings carry `\"\"\"`, but they arrive through {probe_src} at runtime, so they
    # never touch this literal and the delimiter below stays the repo-standard double quote.
    return f"""
# --- flash: repoint tilelang's libcudart stub (see child_io.render_tilelang_cudart_shim) ---
# the probe below is perf._find_real_libcudart's own source, shipped verbatim. edit it THERE.
try:
    import importlib.util as _flash_cudart_importlib_util
    import os
    import os as _flash_cudart_os

{probe_src}

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
            _flash_cudart_real = _find_real_libcudart()
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
                # the name carries os.urandom entropy, not just the pid: a worker killed between
                # symlink() and replace() leaves its temp link behind, and a pid-only name would
                # collide once the container reuses that pid -- symlink() would raise, the swap
                # would be skipped, and the stub would silently stay in place.
                _flash_cudart_tmp = ""
                try:
                    for _flash_cudart_attempt in range(8):
                        _flash_cudart_tmp = (
                            _flash_cudart_stub
                            + ".flash-"
                            + str(_flash_cudart_os.getpid())
                            + "-"
                            + _flash_cudart_os.urandom(8).hex()
                        )
                        try:
                            _flash_cudart_os.symlink(_flash_cudart_real, _flash_cudart_tmp)
                            break
                        except FileExistsError:
                            # astronomically unlikely; clear it and draw a fresh name.
                            try:
                                _flash_cudart_os.remove(_flash_cudart_tmp)
                            except OSError:
                                pass
                    else:
                        raise RuntimeError("could not create a temp swap link")
                    _flash_cudart_os.replace(_flash_cudart_tmp, _flash_cudart_stub)
                finally:
                    if _flash_cudart_tmp:
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
"""


def render_sitecustomize_bootstrap() -> str:
    """render the minimal process-start bootstrap shared by all verl children."""
    return render_tf32_shim() + render_tilelang_cudart_shim()


from flash.engine.worker.train.core.child.runtime import (  # noqa: E402,F401
    FLASH_GDN_VARLEN_MARKER,
    FLASH_LORA_ROLLOUT_MARKER,
    FLASH_WANDB_LINK_MARKER,
    LORA_ROLLOUT_GUARD_SHIM,
    SHIM_FRAGMENT_FAILED_EXIT_CODE,
)

_WANDB_LINK_RE = re.compile(rf"{FLASH_WANDB_LINK_MARKER}\s+(\{{.*\}})\s*$")
SHIM_MARKER_FILENAME = "applied_shims.txt"


def shim_marker_file(shim_dir: str) -> str:
    """return the copied plugins' shared applied-patch marker file."""
    return os.path.join(shim_dir, SHIM_MARKER_FILENAME)


def read_applied_shim_markers(marker_file: str) -> set[str]:
    """return the patch names the child recorded as applied."""
    try:
        with open(marker_file, encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    except OSError:
        return set()


def verify_applied_shim_markers(marker_file: str, expected) -> None:
    """raise unless every expected copied-plugin patch proved it applied."""
    missing = sorted(set(expected) - read_applied_shim_markers(marker_file))
    if missing:
        raise RuntimeError(
            f"the verl child never proved these required runtime patches applied: {missing}. "
            "its configured external module did not load or a deferred target was never patched; "
            "refusing to train unpatched. this is permanent for this interpreter, not a retriable "
            "infra fault."
        )


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
_ADVANTAGE_MIN_KEY = "critic/advantages/min"
_ADVANTAGE_MAX_KEY = "critic/advantages/max"

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
    advantage_min = parse_verl_metric(line, _ADVANTAGE_MIN_KEY)
    advantage_max = parse_verl_metric(line, _ADVANTAGE_MAX_KEY)
    if advantage_min is not None and advantage_max is not None:
        spread = advantage_max - advantage_min
        if math.isfinite(spread) and spread >= 0.0:
            metrics["advantage_min"] = advantage_min
            metrics["advantage_max"] = advantage_max
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
