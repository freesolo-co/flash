"""Pure GPU/perf/optimizer probes for the fine-tuning worker.

These helpers take the model id / max length / capability as ARGUMENTS and read NONE of
the worker's run-scoped module globals (HF_REPO/RUN_ID/SEED/RUN_MODE/PHASE/JOB_SPEC/
ACTIVE_ENV/THINKING or the _HB_* heartbeat family), so they live here as a leaf package.
``flash.engine.worker`` re-exports them; this package must NOT import that package (no cycle).
Torch and other heavy deps are imported lazily inside the functions (CPU-importable).

The probes are split into cohesive submodules (``attn``/``liger``/``memory``/``diagnostics``/
``lifecycle``/``loraplus``) and re-exported here, so both ``from flash.engine.worker.perf import X``
and ``from flash.engine.worker import perf; perf.X`` keep working for every existing name.

The perf-backend + fla/tilelang/cudart fixup group stays IN THIS MODULE on purpose: tests do
``monkeypatch.setattr(perf, "_find_real_libcudart", ...)`` / ``perf._remove_fla_from_disk`` and then
call the functions that use them (``_neutralize_tilelang_cudart_stub`` /
``_ensure_fla_fastpath_on_hopper``). Keeping both the patched names and their callers in this one
namespace is what makes that monkeypatch take effect (the callers resolve the names through the
patched module globals). Do NOT move this group into a submodule.

NOTE: ``tests/test_worker_stack.py`` reads THIS file as TEXT to assert the worker's runtime
fla SHA / TILELANG_PIN stay in lockstep with WORKER_DEPS / Dockerfile.worker — keep the pins in
``_ensure_fla_fastpath_on_hopper`` below consistent with those.
"""

from __future__ import annotations

import contextlib
import os
import sys

# canonical fused-CE / long-context thresholds (single source of truth in flash.engine.vram); kept
# importable from this package for back-compat with anything that read them off ``perf``.
from flash.engine.vram import _LIGER_LONG_CTX_TOKENS, _LIGER_MIN_PARAMS_B
from flash.engine.worker.perf.attn import (
    _attn_impl_for_capability,
    _flash_attn_3_available,
    _flash_attn_available,
    _sdpa_cudnn_ctx,
    optimal_attn_impl,
)
from flash.engine.worker.perf.diagnostics import (
    _clean_diag,
    _float_or_none,
    _GpuPeakSampler,
    _int_or_none,
    _peak_gpu_gb,
    _query_nvidia_gpu,
    _query_nvidia_processes,
    _reset_peak_gpu,
    _round_gb_from_mib,
    gpu_diagnostics,
)
from flash.engine.worker.perf.lifecycle import (
    RETRIABLE_INFRA_MARKER,
    RetriableInfraError,
    _metric_curve,
    detect_mig_slice,
    free_gpu,
    wait_for_gpu,
)
from flash.engine.worker.perf.liger import (
    _LIGER_MIN_PARAMS,
    _estimate_params,
    _liger_default_for_model,
    liger_on,
)
from flash.engine.worker.perf.loraplus import loraplus_optimizer_cls
from flash.engine.worker.perf.memory import (
    _LONG_CONTEXT_TOKENS,
    _memory_mode,
    fused_optim_name,
    grad_checkpointing_on,
)


def setup_perf_backends() -> None:
    """Universal, arch-agnostic throughput knobs — safe on every CUDA arch, no JIT/compile cost.

    - TF32 for fp32 matmuls/cuDNN (Ampere+): the residual fp32 ops in a bf16 LoRA run (some
      norms, the optimizer's fp32 master step, any fp32 GEMM) run on the TF32 tensor cores at
      ~no accuracy cost. No-op on pre-Ampere.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.set_float32_matmul_precision("high")  # TF32 for fp32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[perf] TF32 matmul/cuDNN enabled")
    except Exception as e:
        print("setup_perf_backends skipped:", e)


def _remove_fla_from_disk() -> tuple[list[str], bool]:
    """Physically delete every importable ``fla`` package dir from the worker's REAL sys.path.

    Loops until ``find_spec('fla')`` is clean (removing one copy can expose another further down
    the path) and invalidates import caches so transformers' is_fla_available() probe sees it
    gone. ``pip uninstall`` alone is unreliable here — it targets one site-packages but the base
    image bakes ``fla`` into another dir on the path (and can report success while leaving the
    package dir). Returns ``(removed_dirs, still_importable)``. Used by the Hopper auto-drop.
    """
    import importlib
    import importlib.util
    import shutil

    removed: list[str] = []
    for _ in range(6):  # a few passes: removing one copy can reveal another earlier on the path
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("fla")
        if spec is None:
            break
        # Resolve the package directory (submodule_search_locations for a package, else the file dir).
        locs = list(getattr(spec, "submodule_search_locations", None) or [])
        if not locs and spec.origin:
            locs = [os.path.dirname(spec.origin)]
        progressed = False
        for loc in locs:
            if loc and os.path.isdir(loc) and os.path.basename(loc.rstrip("/")) == "fla":
                try:
                    shutil.rmtree(loc)
                    removed.append(loc)
                    progressed = True
                except Exception as e:
                    print(f"[fla] could not remove {loc}: {e}", flush=True)
        if not progressed:
            break
    importlib.invalidate_caches()
    return removed, importlib.util.find_spec("fla") is not None


def _find_real_libcudart() -> str | None:
    """Path to a REAL ``libcudart.so.12`` that exports ``cudaDeviceReset`` (the symbol tilelang's
    stub lacks), or None if none can be found. Prefers the nvidia-cuda-runtime wheel, then the CUDA
    toolkit baked into the worker's -devel base image, then the system loader's own resolver — and
    VERIFIES the symbol is actually present (a path is only returned if ``CDLL(path)`` exposes
    ``cudaDeviceReset``). Never raises."""
    import ctypes
    import ctypes.util
    import glob

    def _verify(cand: str) -> str | None:
        """Absolute path to ``cand`` if it loads AND exports cudaDeviceReset, else None. Handles both
        absolute paths (glob results) and bare sonames like ``libcudart.so.12`` (find_library, which
        the loader resolves but ``os.path.exists`` would reject)."""
        try:
            lib = ctypes.CDLL(cand)  # an abs path opens directly; a bare soname is loader-resolved
        except OSError:
            return None
        if not hasattr(lib, "cudaDeviceReset"):
            return None
        if os.path.isabs(cand) and os.path.exists(cand):
            return os.path.realpath(cand)
        # Bare soname: resolve to the file the loader actually opened, via /proc/self/maps.
        base = os.path.basename(cand)
        try:
            with open("/proc/self/maps") as f:
                for line in f:
                    if base in line and "/" in line:
                        p = line[line.index("/"):].rstrip()
                        if os.path.basename(p).startswith(base) and os.path.exists(p):
                            return os.path.realpath(p)
        except OSError:
            pass
        return None

    candidates: list[str] = []
    # 1. nvidia-cuda-runtime PyPI wheel (a torch/vLLM dep on many images).
    try:
        import nvidia.cuda_runtime as _cr  # type: ignore

        candidates += glob.glob(
            os.path.join(os.path.dirname(_cr.__file__), "lib", "libcudart.so.12*")
        )
    except Exception:
        pass
    # 2. CUDA toolkit in the -devel base image (Dockerfile.worker: cuda12.8-cudnn9-devel).
    for pat in (
        "/usr/local/cuda*/lib64/libcudart.so.12*",
        "/usr/local/cuda*/targets/*/lib/libcudart.so.12*",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.12*",
    ):
        candidates += glob.glob(pat)
    # 3. The loader's own resolver (LD_LIBRARY_PATH / ldconfig) — returns a bare soname, handled above.
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    for cand in candidates:
        real = _verify(cand)
        if real is not None:
            return real
    return None


def _neutralize_tilelang_cudart_stub() -> None:
    """Stop tilelang's bundled ``libcudart_stub.so`` from shadowing the real CUDA runtime in vLLM.

    tilelang ships a minimal ``libcudart_stub.so`` (soname ``libcudart_stub.so``) that
    ``libtilelang.so`` / ``libtvm.so`` link against; it exports only a SUBSET of the CUDA runtime —
    notably it is MISSING ``cudaDeviceReset``. vLLM's ``vllm/device_allocator/cumem.py`` builds a
    ``CudaRTLibrary`` at MODULE TOP LEVEL (``libcudart = CudaRTLibrary()``), and that module is
    imported on EVERY vLLM init via ``gpu_worker.load_model`` ->
    ``_maybe_get_memory_pool_context(tag="weights")`` — so the crash is NOT gated on sleep mode or
    model size (a 0.8B GRPO run hit it too); any GRPO vLLM init is exposed. ``CudaRTLibrary`` finds
    libcudart by a SUBSTRING scan of ``/proc/self/maps`` and returns the FIRST matching line
    (address-ordered, so host-dependent ~coin-flip). Once tilelang is loaded — the Hopper fla fast
    path, or fla's backend probe on any arch — the stub is mapped into the process and can win that
    scan, so ``CudaRTLibrary()`` dlopens the stub and aborts the import with ``undefined symbol:
    cudaDeviceReset`` before step 0. See flash #184.

    Fix: BEFORE anything imports tilelang/fla/vllm, repoint the stub path at the REAL
    ``libcudart.so.12`` via a symlink. Then whichever copy the loader (or vLLM's scan) resolves has
    the full symbol set, and the real lib's soname (``libcudart.so.12``) dedupes against the copy
    torch already loaded — so no second CUDA-runtime instance is created and the stub-named mapping
    drops out of ``/proc/self/maps`` entirely. tilelang keeps working: the real runtime is a strict
    superset of the stub it linked against. Applies on EVERY arch and model size (the crash spans
    0.8B/4B and A100/cheaper classes) and to every provisioning path (baked image or runtime pip),
    since it runs in the worker before the first tilelang load. Must run AFTER
    ``_ensure_fla_fastpath_on_hopper`` (a tilelang (re)install there would otherwise rewrite the
    stub) and BEFORE the model/vLLM import.

    Idempotent and best-effort: a missing tilelang, a missing stub, an already-healthy stub, or no
    discoverable real runtime is a clean no-op; any error is swallowed (the worker must never crash
    on this hygiene step). No GPU required.
    """
    import importlib.util

    try:
        spec = importlib.util.find_spec("tilelang")
    except Exception:
        spec = None
    locs = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    if not locs:
        return  # tilelang not installed -> nothing can shadow libcudart
    stub = os.path.join(locs[0], "lib", "libcudart_stub.so")
    if not os.path.lexists(stub):  # lexists: a dangling symlink still counts as present
        return
    # Idempotency WITHOUT loading the stub: we only ever turn the stub into a symlink, and a pristine
    # tilelang always ships it as a regular file, so a RESOLVING symlink here means a prior pass
    # already neutralized it. Crucially, do NOT probe the stub with ctypes.CDLL — that dlopens it (it
    # loads fine under lazy binding despite the missing cudaDeviceReset) and maps it into THIS
    # process's /proc/self/maps, which is exactly the libcudart line vLLM's CudaRTLibrary scan would
    # then pick up -> the very crash we're preventing. The stub must never be loaded; only the file
    # is touched. A DANGLING symlink (our target moved/was removed) is NOT done — it leaves tilelang
    # with a broken libcudart_stub.so, so fall through and re-point it (os.path.exists follows links).
    if os.path.islink(stub) and os.path.exists(stub):
        return
    real = _find_real_libcudart()
    if real is None:
        print(
            "[worker] libcudart stub shadow: no real libcudart found; left as-is (flash #184)",
            flush=True,
        )
        return
    try:
        # Preserve the original stub ONCE (reversible / debuggable), then point the stub path at the
        # real runtime. os.replace is atomic; symlink keeps soname-dedup (no duplicate libcudart).
        backup = stub + ".orig"
        if not os.path.exists(backup):
            os.replace(stub, backup)
        else:
            with contextlib.suppress(FileNotFoundError):
                os.remove(stub)
        os.symlink(real, stub)
        print(f"[worker] redirected tilelang libcudart_stub.so -> {real} (flash #184)", flush=True)
    except Exception as e:
        print(f"[worker] libcudart stub neutralize failed: {e}", flush=True)


def _ensure_fla_fastpath_on_hopper() -> None:
    """Make flash-linear-attention's GatedDeltaNet fast path CORRECT + fast on Hopper (sm90)
    instead of dropping it.

    fla's gated chunk_bwd Triton kernel is miscomputed on Hopper with Triton>=3.4 and HARD-RAISES
    (fla #640). The worker historically DROPPED fla here and fell back to the pure-PyTorch delta
    rule — correct but slow + memory-heavy. The real fix is fla's **tilelang** backend, which is
    correct on Triton>=3.4. So on Hopper we ensure the working stack is present rather than
    removing fla:
      * the pinned ``tilelang==0.1.11`` (the correct GDN chunk_bwd backend) + the pinned
        ``apache-tvm-ffi==0.1.11`` (0.1.12 double-registers the TVM-FFI runtime -> ``import
        tilelang`` aborts; and tilelang's own ``apache-tvm-ffi~=0.1.0`` range would let 0.1.12
        back in, so the pin is force-installed last and its resolved version is verified), and
      * a COMPLETE ``fla`` (the PyPI ``flash-linear-attention`` wheel is a broken stub missing
        ``fla.modules``; reinstall from git if the resident copy is incomplete).
    Validated A/B (H100 SXM, Qwen3.5 hidden-2560 LoRA, controlled fla on/off): seq4096 435->105
    ms/step & 9.9->6.1 GB (4.2x / 1.6x); seq8192 7.1x; seq16384 3106->247 ms & 32->17 GB (12.6x /
    1.9x). Forward loss matches the torch delta to 1.8e-4 (correct). Runs in the worker process,
    after all installs and BEFORE any model import. Non-Hopper:
    no-op (fla's Triton kernel is correct there). Best-effort + FAIL-CLOSED: a failed install
    (pip rc!=0), a missing module, or the wrong resolved ``apache-tvm-ffi`` version all flip the
    gate off and DISABLE fla, leaving the (correct) pure-PyTorch delta rule in place — a worker
    never crashes on a dep hiccup, and it never silently runs fla's broken Hopper GDN kernel.
    """
    import importlib
    import importlib.util
    import subprocess

    try:
        import torch

        if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9):
            return  # not Hopper: fla's Triton kernel is correct here.
    except Exception:
        return

    def _have(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    def _ver(dist: str) -> str | None:
        """Installed version of a distribution (by metadata), or None if absent/unreadable.

        Distinct from _have (a find_spec import probe): the install can silently leave the WRONG
        version resolved (e.g. tilelang's ``apache-tvm-ffi~=0.1.0`` range happily keeps 0.1.12,
        which still find_spec-imports but aborts ``import tilelang``), so the gate must check the
        actual installed version, not just importability.
        """
        try:
            import importlib.metadata as _md

            return _md.version(dist)
        except Exception:
            return None

    def _pip(*args: str) -> bool:
        """Run pip install; return True only if pip exited 0. A failed install (network/build/
        resolver) must NOT be silently treated as success — the caller gates ``ok`` on this."""
        try:
            rc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", *args], check=False
            ).returncode
        except Exception:
            return False
        return rc == 0

    # The exact apache-tvm-ffi pin the tilelang backend needs (0.1.12 double-registers the TVM-FFI
    # runtime -> `import tilelang` aborts). Kept as a constant so the install spec and the post-
    # install version gate below can't drift apart. Keep in lockstep with WORKER_DEPS / Dockerfile.
    TVM_FFI_PIN = "0.1.11"
    TILELANG_PIN = "0.1.11"  # pin the GDN backend too (same rationale as the fla SHA pin)

    try:
        # 1. tilelang backend (correct GDN chunk_bwd on Triton>=3.4) + the pinned tvm-ffi.
        #    Track whether each install actually succeeded — a failed pip (rc!=0) must flip the
        #    gate to the pure-PyTorch fallback rather than be ignored. (_have-only would also pass
        #    on a stale/partial copy from a previous boot.) tilelang pulls apache-tvm-ffi via a
        #    range that allows the broken 0.1.12, so force-reinstall the exact pin AFTER tilelang
        #    and verify the resolved version below.
        # Enforce the EXACT pin: (re)install when tilelang is absent OR a different version is
        # resident (a job or the base image may carry another tilelang; _have-only would treat that
        # as healthy and skip the install, leaving the wrong/uncertain GDN backend in place). Mirror
        # the apache-tvm-ffi handling: check the installed version via _ver and reinstall on mismatch.
        tilelang_ok = True
        tilelang_reinstalled = False
        if _ver("tilelang") != TILELANG_PIN:
            tilelang_ok = _pip(f"tilelang=={TILELANG_PIN}")
            tilelang_reinstalled = True
        # Only force the tvm-ffi pin when it's actually wrong OR tilelang was just (re)installed
        # (tilelang's apache-tvm-ffi~=0.1.0 range can have pulled the broken 0.1.12). Skipping the pip
        # when the exact pin is already resident avoids avoidable cold-start latency and a spurious
        # disable on a transient network/resolver failure — the ok gate still re-verifies the version.
        # If this install runs and fails we DON'T trust the resident copy — tvm_ffi_ok gates `ok` below.
        if _ver("apache-tvm-ffi") != TVM_FFI_PIN or tilelang_reinstalled:
            tvm_ffi_ok = _pip(f"apache-tvm-ffi=={TVM_FFI_PIN}")
        else:
            tvm_ffi_ok = True
        # 2. a COMPLETE fla — the PyPI wheel ships a stub without `fla.modules`. Reinstall from git
        #    when the resident copy is missing the real package (or absent entirely).
        fla_ok = True
        if not (_have("fla") and _have("fla.modules")):
            _remove_fla_from_disk()  # clear any broken stub before the git install
            # Pinned to the same commit as WORKER_DEPS / Dockerfile.worker so a runtime reinstall is
            # reproducible (the moving default branch could pull a broken/incompatible fla).
            fla_ok = _pip(
                "--no-deps",
                "git+https://github.com/fla-org/flash-linear-attention.git"
                "@f0e213dbd8b5fb90c3c7eca869ac1706d5377139",
            )
        importlib.invalidate_caches()
        # Gate on BOTH (a) every install we ran exiting 0 — a failed pip (network/build/resolver)
        # must NOT be treated as healthy just because a stale/partial copy still find_spec-imports —
        # AND (b) the modules importing AND (c) the resolved apache-tvm-ffi being exactly the pin.
        # (c) matters because tilelang depends on `apache-tvm-ffi~=0.1.0`, so the resolver can keep
        # the broken 0.1.12 (which find_spec-imports fine but aborts `import tilelang`); checking the
        # version is the only reliable signal the pin actually landed.
        tvm_ffi_ver = _ver("apache-tvm-ffi")
        tilelang_ver = _ver("tilelang")
        installs_ok = tilelang_ok and tvm_ffi_ok and fla_ok
        ok = (
            installs_ok
            and _have("fla")
            and _have("fla.modules")
            and _have("tilelang")
            and tilelang_ver == TILELANG_PIN
            and tvm_ffi_ver == TVM_FFI_PIN
        )
        if not ok:
            # The healthy fla+tilelang stack could not be assembled, so fla's GDN chunk_bwd would
            # still hit the broken Triton>=3.4 path on Hopper (fla #640) and HARD-RAISE. A print
            # alone does NOT prevent that: transformers gates GDN on is_fla_available() (a
            # find_spec('fla') probe), so as long as fla stays importable it gets engaged. PHYSICALLY
            # remove fla so the probe sees it gone and transformers uses the correct pure-PyTorch
            # delta rule instead of crashing. _remove_fla_from_disk loops over the real sys.path +
            # invalidates caches, so find_spec('fla') is None afterwards (the gate flips off).
            _removed, _still = _remove_fla_from_disk()
            print(
                "[hopper] fla GDN fast path unavailable -> DISABLING fla "
                f"(installs_ok={installs_ok} [tilelang={tilelang_ok} tvm_ffi={tvm_ffi_ok} "
                f"fla={fla_ok}], tilelang_ver={tilelang_ver!r} (want {TILELANG_PIN}), "
                f"tvm_ffi_ver={tvm_ffi_ver!r} (want {TVM_FFI_PIN}); "
                f"removed {len(_removed)} copy(ies); still_importable={_still}); "
                "pure-PyTorch delta fallback",
                flush=True,
            )
        else:
            print(
                "[hopper] fla GDN fast path ENABLED (fla+tilelang "
                f"{tilelang_ver}/tvm-ffi {tvm_ffi_ver}, fla #640 fixed)",
                flush=True,
            )
    except Exception as e:  # never let a dep hiccup crash the worker — torch delta still runs
        # Fail-closed: an unexpected error mid-setup must still leave Hopper on the correct
        # pure-PyTorch delta path, not a half-configured fla that transformers would engage and
        # crash on (#640). Best-effort disable fla; never re-raise.
        with contextlib.suppress(Exception):
            _remove_fla_from_disk()
        print(
            f"[hopper] fla fast-path setup errored ({type(e).__name__}: {e}); "
            "disabled fla -> pure-PyTorch delta",
            flush=True,
        )


__all__ = [
    "RETRIABLE_INFRA_MARKER",
    "_LIGER_LONG_CTX_TOKENS",
    "_LIGER_MIN_PARAMS",
    "_LIGER_MIN_PARAMS_B",
    "_LONG_CONTEXT_TOKENS",
    "RetriableInfraError",
    "_GpuPeakSampler",
    "_attn_impl_for_capability",
    "_clean_diag",
    "_ensure_fla_fastpath_on_hopper",
    "_estimate_params",
    "_find_real_libcudart",
    "_flash_attn_3_available",
    "_flash_attn_available",
    "_float_or_none",
    "_int_or_none",
    "_liger_default_for_model",
    "_memory_mode",
    "_metric_curve",
    "_neutralize_tilelang_cudart_stub",
    "_peak_gpu_gb",
    "_query_nvidia_gpu",
    "_query_nvidia_processes",
    "_remove_fla_from_disk",
    "_reset_peak_gpu",
    "_round_gb_from_mib",
    "_sdpa_cudnn_ctx",
    "detect_mig_slice",
    "free_gpu",
    "fused_optim_name",
    "gpu_diagnostics",
    "grad_checkpointing_on",
    "liger_on",
    "loraplus_optimizer_cls",
    "optimal_attn_impl",
    "setup_perf_backends",
    "wait_for_gpu",
]
