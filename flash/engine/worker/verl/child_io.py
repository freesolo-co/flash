"""Child-side shims the verl subprocess installs, and the stdout lines it reports back.

Two halves of one conversation with the verl child. The `render_*_shim` functions emit
sitecustomize source that the child executes -- tf32 matmul, the w&b link report, the GDN varlen
patch -- and the `parse_*` functions read what those shims print back out of the child's stdout,
turning it into the per-step metrics `flash runs log -f` renders.

Split out of `flash.engine.worker.backend_common` to keep that module under the file-size limit.
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


FLASH_FLASH_QLA_MARKER = "[flash-verl] flashqla gdn backend active"


# Why the fragment below binds the kernel directly, on sm90 only, and never fails the run.
#
# MEASURED, on a real H200 (Qwen3.5-4B, seq 4096, 24 GDN layers), through THIS fragment as rendered
# and wrapped by the shipped code path: 1.065x end-to-end training step, null-corrected against an
# A/A null of 0.9914, 22/24 paired wins. both kernels timed in ONE process, ABBA-interleaved per
# step and both pre-warmed, because two identical control PROCESSES differ by more than the effect.
# a standalone kernel harness on the same shape independently measured 1.055x, with forward and all
# five gradients matching fla's own kernel to 8e-3 at identical peak memory.
#
# sm90 ONLY, and that is a correctness floor rather than a preference. flashqla is tilelang-based
# and on sm100 it computes WRONG gradients at the production call shapes: measured dq 1.006,
# dk 0.942, dg 0.997 relative error on a B200, with the training loss diverging to 14.91 against
# the control's 13.96 from identical weights -- while REPORTING a 1.047x speedup with a clean null.
# that is the same failure class as the tilelang backward this worker already opts out of
# (`perf._force_fla_triton_gdn_on_sm100`), and it caused the B200 35B-A3B SFT incident. note that
# `FLA_TILELANG=0` does NOT cover this: flashqla is a separate registry backend keyed on
# `FLA_FLASH_QLA`, so the existing sm100 guard would leave it enabled.
#
# BINDING, rather than setting the env var, because the env var provably does nothing: at import
# transformers captures `chunk_gated_delta_rule` into a module global and each GDN layer copies that
# object onto `self` in `__init__`, so the layer holds a bare `@torch.compiler.disable` function and
# fla's `@dispatch` wrapper -- the only thing that reads `FLA_FLASH_QLA` -- is never on the call
# path. setting the env var alone passes every availability probe and changes nothing, which is
# exactly how the FA3 wheel stayed dead in this image for its whole life.
#
# DEFERRED through a meta_path finder for the same reason as the varlen shim: an import at
# sitecustomize time initializes cuda against the full device list before ray narrows the actor to
# its own card.
#
# FAIL-OPEN, unlike every other required fragment: the boundary shims fail closed because skipping
# them corrupts training, while skipping this one only costs ~6% and leaves the child on the kernel
# it used before. that matters for the deploy window -- `worker_image_for_gpu` returns a MUTABLE tag
# whose image presets FLASH_VERL_PYTHON, so `resolve_verl_python` returns early and never installs
# the wheel; until the image is rebuilt, every GDN child takes the unavailable path.
def render_flash_qla_shim(model_type: str) -> str:
    """child-side sitecustomize fragment routing GDN through fla's flashqla backend on sm90.

    takes the arch the gate already verified, exactly like ``render_gdn_varlen_shim``: `qwen3_5`
    and `qwen3_5_moe` are different modules, and a fragment that watched BOTH would have to decide
    when to stop watching. one target makes "patch it, then step off meta_path" unambiguous.
    """
    return f'''
# --- flash: route gdn through the flashqla backend on sm90 (child_io.render_flash_qla_shim) ---
import sys as _flash_qla_sys

_FLASH_QLA_TARGET = "transformers.models.{model_type}.modeling_{model_type}"


def _flash_qla_supported():
    """sm90 only. anything else keeps the kernel fla would have chosen on its own."""
    import torch as _flash_qla_torch

    if not _flash_qla_torch.cuda.is_available():
        return False
    return _flash_qla_torch.cuda.get_device_capability()[0] == 9


def _flash_qla_impl():
    """the flashqla backend's own chunk_gated_delta_rule, or None when unavailable.

    resolved through fla's registry rather than imported by path so a pin bump that moves the
    module cannot silently bind something else.

    every failure is a None, never an exception. this is called from an import LOADER, so anything
    raised here propagates out of the child's `import transformers...` and kills the run. an fla
    older than 0.5.2 has no ``backends`` subpackage at all and raises ModuleNotFoundError on the
    line below -- which is precisely the state of a worker whose image has not been rebuilt yet.
    """
    try:
        from fla.ops.gated_delta_rule.backends import gdr_registry as _flash_qla_registry

        for backend in _flash_qla_registry._get_sorted_backends():
            if getattr(backend, "backend_type", "") != "flash_qla":
                continue
            if not backend.is_available():
                return None
            return getattr(backend, "chunk_gated_delta_rule", None)
    except Exception:
        return None
    return None


def _flash_qla_patch(module):
    if not _flash_qla_supported():
        return
    impl = _flash_qla_impl()
    if impl is None:
        # NOT fatal, and deliberately so. this is a pure speed swap: without it the child trains on
        # fla's own kernel, which is exactly what every run before this fragment did. the boundary
        # shims fail closed because skipping them CORRUPTS training; skipping this one only costs
        # ~6%.
        #
        # and the failure is expected in production for a window: the worker image is a mutable tag
        # (worker_image_for_gpu -> flash-worker:cu128-smXX) that presets FLASH_VERL_PYTHON, so
        # resolve_verl_python returns that interpreter and never runs the venv provisioning that
        # installs FLASH_QLA_REQUIREMENT. until the image carrying the wheel is built and pushed,
        # every GDN child would hit this line -- hard-exiting here would take down all GDN SFT.
        print(
            "[flash-verl] flashqla gdn backend unavailable (flash-qla wheel missing, or "
            "flash-linear-attention older than the 0.5.2 that carries the dispatch); "
            "continuing on fla's own kernel",
            file=_flash_qla_sys.stderr,
            flush=True,
        )
        return

    def _flash_qla_call(*args, **kwargs):
        # fla's dispatcher passes cu_seqlens_cpu to backends that accept it; flashqla does not.
        kwargs.pop("cu_seqlens_cpu", None)
        return impl(*args, **kwargs)

    _flash_qla_call._flash_qla_patched = True
    # the module global is the only binding that needs rewriting HERE: each GDN layer copies it onto
    # self in __init__, and this runs from the loader the moment the modeling module finishes
    # executing -- before any layer exists. patching after a layer was built would miss it.
    module.chunk_gated_delta_rule = _flash_qla_call
    print({FLASH_FLASH_QLA_MARKER!r}, module.__name__, flush=True)


class _FlashQlaLoader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _flash_qla_uninstall()
        _flash_qla_patch(module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FlashQlaFinder:
    # delegate only to finders positioned AFTER this one, rather than to "everything that is not
    # my own class". two fragments can target the SAME modeling module -- the varlen shim is
    # rendered under the same gate as this one -- and each is rendered independently, so neither
    # can name the other's class.
    #
    # skipping only `isinstance(self)` sends each into the other's find_spec forever
    # (RecursionError before the model is built, killing every GDN SFT child). skipping ALL flash
    # finders is worse: the first one resolves the spec, python never consults the second, and that
    # fragment's patch silently never applies while its marker still records success. slicing past
    # our own position keeps both wrappers stacked, so both patches land.
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _FLASH_QLA_TARGET:
            return None
        here = [i for i, f in enumerate(_flash_qla_sys.meta_path) if f is self]
        rest = _flash_qla_sys.meta_path[here[0] + 1 :] if here else _flash_qla_sys.meta_path
        for finder in rest:
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _FlashQlaLoader(spec.loader)
                return spec
        return None


def _flash_qla_uninstall():
    _flash_qla_sys.meta_path[:] = [
        f for f in _flash_qla_sys.meta_path if not isinstance(f, _FlashQlaFinder)
    ]


# already imported (a parent process may have loaded it): patch now, there is no import left to
# intercept. otherwise arm the finder and let the child's own import trigger it.
_flash_qla_loaded = _flash_qla_sys.modules.get(_FLASH_QLA_TARGET)
if _flash_qla_loaded is not None:
    _flash_qla_patch(_flash_qla_loaded)
else:
    _flash_qla_sys.meta_path.insert(0, _FlashQlaFinder())
'''


FLASH_GDN_VARLEN_MARKER = "[flash-verl] gdn packed-boundary resets active"

FLASH_LORA_ROLLOUT_MARKER = "[flash-verl] lora rollout guard active"

_LORA_ROLLOUT_TARGET = "verl.workers.rollout.vllm_rollout.vllm_async_server"


# why this fragment exists at all.
#
# verl's `vLLMHttpServer.generate` builds a LoRARequest only when `lora_as_adapter` is set and
# `VLLM_LORA_INT_ID` is already in `await self.engine.list_loras()`; it then passes whatever it has
# to `self.engine.generate(..., lora_request=lora_request, ...)`.
#
# when the adapter is missing, `lora_request` stays none and generation proceeds from the base
# model. there is no raise, no warning and no counter, so the run completes and the loss descends
# while every rollout came from a different policy than the one being trained. for opd that silently
# turns on-policy distillation into off-policy distillation from the stock base model; for grpo it
# scores trajectories the policy never produced. the symptom is "the method did not beat sft", which
# reads as a failed experiment rather than a broken run, and no loss curve can distinguish the two.
#
# the guard wraps `self.engine.generate` rather than `vLLMHttpServer.generate`, because that is
# where the decision is actually consumed. checking `list_loras` ourselves and then calling verl's
# original would leave verl free to repeat its own lookup and reach the opposite conclusion, so an
# adapter that disappeared between the two calls would still generate from the base model -- the
# exact outcome this exists to prevent. reading the `lora_request` verl computed is also one fewer
# engine round trip per rollout: `list_loras` is an awaited zeromq utility rpc, not a local read.
#
# every flash rollout is a lora rollout (train.lora_rank parses with minimum=1), and there is no
# legitimate window where the adapter is missing:
#   - ray_trainer.fit loads the checkpoint and calls update_weights before any generation,
#     including the optional pre-train validation pass, so the first rollout is already post-sync.
#   - the adapter is re-added on every step's weight sync (utils.py _update_weights -> add_lora),
#     and the one remove_lora is inside that same awaited sync, with no generation scheduled in it.
#   - lora rollouts sleep at level 1 rather than 2 specifically to keep the adapter across steps.
#   - vllm is configured max_loras=1 against a single fixed id, so nothing evicts it under pressure.
# `lora_as_adapter` is false for a merged-lora rollout, which serves the adapter through the base
# weights and legitimately carries no LoRARequest, so mirroring that flag keeps the guard off the
# one path where generating without a LoRARequest is correct.
def render_lora_rollout_guard_shim() -> str:
    """child-side sitecustomize fragment that fails a rollout whose lora never reached the engine.

    deferred through a meta_path finder for the same reason as the gdn shim: importing the target at
    interpreter startup pulls in vllm and torch, initializing cuda against every visible gpu before
    ray narrows the actor to its own card.
    """
    return f'''
import sys as _flash_lora_sys

_FLASH_LORA_TARGET = {_LORA_ROLLOUT_TARGET!r}


def _flash_guard_lora_engine(engine, adapter_id):
    """refuse an engine.generate that carries no LoRARequest on a lora rollout."""
    inner = engine.generate

    def guarded(*args, **kwargs):
        if kwargs.get("lora_request") is None:
            raise RuntimeError(
                "[flash-verl] refusing to roll out from the base model: this run trains a lora "
                f"adapter (id {{adapter_id}}), but it was not loaded in the vllm engine when this "
                "request was built, so verl passed no LoRARequest. generating here would train "
                "the adapter on tokens a different policy produced, which no loss curve can "
                "distinguish from a healthy run."
            )
        return inner(*args, **kwargs)

    engine.generate = guarded
    engine._flash_lora_guarded = True


def _flash_patch_lora_rollout(module):
    server = module.vLLMHttpServer
    if getattr(server.generate, "_flash_lora_guarded", False):
        return
    original = server.generate

    async def generate(self, *args, **kwargs):
        # arm the engine once, on the first lora request it serves. `lora_as_adapter` is derived
        # from config and never changes for the life of the process, so from here on any request
        # reaching the engine without a LoRARequest is the defect. installing once (rather than
        # around each call) keeps this correct while many rollouts share the one engine object.
        engine = self.engine
        if self.lora_as_adapter and not getattr(engine, "_flash_lora_guarded", False):
            _flash_guard_lora_engine(engine, module.VLLM_LORA_INT_ID)
        return await original(self, *args, **kwargs)

    generate._flash_lora_guarded = True
    server.generate = generate
    print({FLASH_LORA_ROLLOUT_MARKER!r}, flush=True)


class _FlashLoraLoader:
    """patch after the real loader finishes, so the class the caller receives is the patched one."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _flash_lora_uninstall()
        _flash_patch_lora_rollout(module)

    def __getattr__(self, name):  # keep the rest of the loader protocol intact
        return getattr(self._inner, name)


class _FlashLoraFinder:
    """intercept the first import of the rollout server module, then get out of the way."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _FLASH_LORA_TARGET:
            return None
        # delegate only to finders positioned after this one: skipping every flash finder would let
        # a sibling shim's patch be dropped while its marker still recorded success, and skipping
        # only our own class recurses when two flash finders sit on the same module.
        here = [i for i, f in enumerate(_flash_lora_sys.meta_path) if f is self]
        rest = _flash_lora_sys.meta_path[here[0] + 1 :] if here else _flash_lora_sys.meta_path
        for finder in rest:
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _FlashLoraLoader(spec.loader)
                return spec
        return None


def _flash_lora_uninstall():
    _flash_lora_sys.meta_path[:] = [
        f for f in _flash_lora_sys.meta_path if not isinstance(f, _FlashLoraFinder)
    ]


_flash_lora_loaded = _flash_lora_sys.modules.get(_FLASH_LORA_TARGET)
if _flash_lora_loaded is not None:
    _flash_patch_lora_rollout(_flash_lora_loaded)
else:
    _flash_lora_sys.meta_path.insert(0, _FlashLoraFinder())
'''


# Why the shim below defers every import to the moment the child imports the modeling module.
#
# sitecustomize runs at INTERPRETER STARTUP, and ray starts each actor's interpreter before it
# narrows that actor's CUDA_VISIBLE_DEVICES to its own card -- so an import here runs while the
# process still sees every gpu. importing the modeling module transitively calls transformers'
# is_flash_linear_attention_available / is_causal_conv1d_available, both of which call
# torch.cuda.is_available(), and fla queries device capability at import. that initializes cuda
# against the FULL device list, and a later env-only change cannot rebuild the device map, so every
# rank keeps device 0 and nccl aborts with "Duplicate GPU detected". a meta_path finder carries no
# import cost and fires after ray has pinned the actor.
#
# the finder patches from exec_module, never find_spec: find_spec runs against a half-built module
# that the real import then replaces, which leaves the caller's class unpatched while the marker
# still prints.
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

    delegate to the finders positioned AFTER this one, not to "everything that is not this class":
    the flashqla fragment arms a finder on the SAME module under the same gate, and each fragment
    is rendered independently so neither can name the other's class. delegating into it by class
    recurses until RecursionError; skipping every flash finder instead would let whichever runs
    first resolve the spec alone, silently dropping the other patch while its marker still records
    success. slicing past our own position keeps both loaders stacked.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _FLASH_GDN_TARGET:
            return None
        here = [i for i, f in enumerate(_flash_gdn_sys.meta_path) if f is self]
        rest = _flash_gdn_sys.meta_path[here[0] + 1 :] if here else _flash_gdn_sys.meta_path
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


# --------------------------- fail-closed fragment wrapping (sft + grpo sitecustomize) ---------
# cpython's site.execsitecustomize catches every Exception raised while sitecustomize imports,
# prints a two-line note, and starts the interpreter anyway, so a failing fragment silently
# disables itself AND every fragment concatenated after it, and the child trains unpatched.
# wrapping gives every required fragment two guarantees: an exception hard-exits the child with
# this code (os._exit cannot be swallowed by execsitecustomize), and successful application
# appends the fragment's name to a marker file the parent verifies before trusting the run.
SHIM_FRAGMENT_FAILED_EXIT_CODE = 97
SHIM_MARKER_FILENAME = "applied_shims.txt"


def shim_marker_file(shim_dir: str) -> str:
    """the marker file the wrapped fragments in ``shim_dir``'s sitecustomize append to."""
    return os.path.join(shim_dir, SHIM_MARKER_FILENAME)


def render_shim_marker_prologue(marker_file: str) -> str:
    """sitecustomize prologue defining the recorder every wrapped fragment reports through."""
    return f"""
# --- flash: record each applied runtime patch (see child_io.wrap_shim_fragment) ---
_FLASH_SHIM_MARKER_FILE = {marker_file!r}


def _flash_record_applied_shim(name):
    # append-mode: torchrun ranks and ray actors import this same sitecustomize concurrently,
    # and short O_APPEND writes keep each line intact. the parent reads the file as a set.
    with open(_FLASH_SHIM_MARKER_FILE, "a") as _flash_shim_handle:
        _flash_shim_handle.write(name + "\\n")
"""


def wrap_shim_fragment(name: str, source: str) -> str:
    """wrap one required sitecustomize fragment so it fails closed and proves it applied.

    "" stays "": a feature that is off has nothing to prove. requires the prologue above earlier
    in the same sitecustomize. optional fragments (tf32, the wandb link) stay unwrapped, since they
    swallow their own failures by design and must never be able to kill a paid run.
    """
    if not source:
        return ""
    return f"""
# --- flash required fragment: {name} (fails closed; see child_io.wrap_shim_fragment) ---
try:
{textwrap.indent(source, "    ")}
    _flash_record_applied_shim({name!r})
except BaseException:
    import os as _flash_shim_os
    import sys as _flash_shim_sys
    import traceback as _flash_shim_traceback

    _flash_shim_traceback.print_exc()
    print(
        "[flash-verl] required shim fragment {name} failed to apply; "
        "exiting {SHIM_FRAGMENT_FAILED_EXIT_CODE} rather than training unpatched",
        file=_flash_shim_sys.stderr,
        flush=True,
    )
    _flash_shim_sys.stderr.flush()
    _flash_shim_os._exit({SHIM_FRAGMENT_FAILED_EXIT_CODE})
"""


def read_applied_shim_markers(marker_file: str) -> set[str]:
    """the fragment names the child recorded as applied; empty when it recorded none."""
    try:
        with open(marker_file, encoding="utf-8") as handle:
            return {line.strip() for line in handle if line.strip()}
    except OSError:
        return set()


def verify_applied_shim_markers(marker_file: str, expected) -> None:
    """raise unless every expected fragment proved it applied in the child.

    the wrapped fragments hard-exit the child on failure, so the only way a marker goes missing
    is the sitecustomize never running at all (a shadowing sitecustomize on a foreign
    FLASH_VERL_PYTHON, or a lost PYTHONPATH entry). that child is training with NO flash patch
    (seeding, kl anchoring, save gating, boundary resets), so the attempt must fail. permanent
    rather than retriable by design: the same interpreter reproduces the same skip on retry.
    """
    missing = sorted(set(expected) - read_applied_shim_markers(marker_file))
    if missing:
        raise RuntimeError(
            f"the verl child never proved these required runtime patches applied: {missing}. "
            "its sitecustomize did not run (a shadowing sitecustomize or a dropped PYTHONPATH "
            "entry on a foreign FLASH_VERL_PYTHON can cause this); refusing to train unpatched. "
            "this is permanent for this interpreter, not a retriable infra fault."
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
