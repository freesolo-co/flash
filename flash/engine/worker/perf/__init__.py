"""Pure GPU/perf/optimizer probes for the fine-tuning worker.

The perf-backend + fla/tilelang/cudart fixup group stays IN THIS MODULE on purpose: tests
monkeypatch ``perf._find_real_libcudart`` / ``perf._remove_fla_from_disk`` and the callers
resolve them through the patched module globals. Do NOT move this group into a submodule.

NOTE: ``tests/test_worker_stack.py`` reads THIS file as TEXT to assert the fla SHA /
TILELANG_PIN stay in lockstep with WORKER_DEPS / Dockerfile.worker.
"""

from __future__ import annotations

import contextlib
import os
import sys

# back-compat re-export from flash.engine.vram
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
    grpo_sleep_mode,
)


def setup_perf_backends() -> None:
    """Enable TF32 matmul/cuDNN (no-op pre-Ampere)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[perf] TF32 matmul/cuDNN enabled")
    except Exception as e:
        print("setup_perf_backends skipped:", e)


def _remove_fla_from_disk() -> tuple[list[str], bool]:
    """Delete every importable ``fla`` dir from sys.path; returns (removed_dirs, still_importable).

    pip uninstall is unreliable when the base image bakes fla into a separate dir on the path.
    """
    import importlib
    import importlib.util
    import shutil

    removed: list[str] = []
    for _ in range(6):  # removing one copy can reveal another earlier on the path
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("fla")
        if spec is None:
            break
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
    """Find a real libcudart.so that exports cudaDeviceReset (tilelang's stub lacks it).

    Probes nvidia wheel dirs (any CUDA major), the -devel image toolkit, and the system resolver.
    """
    import ctypes
    import ctypes.util
    import glob

    def _verify(cand: str) -> str | None:
        """Return resolved path if cand loads and exports cudaDeviceReset, else None."""
        try:
            lib = ctypes.CDLL(cand)
        except OSError:
            return None
        if not hasattr(lib, "cudaDeviceReset"):
            return None
        if os.path.isabs(cand) and os.path.exists(cand):
            return os.path.realpath(cand)
        # bare soname: resolve via /proc/self/maps
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
    # 1. nvidia cuda-runtime PyPI wheel (any CUDA major — cu12/cu13 layouts differ)
    try:
        import nvidia  # type: ignore  # namespace package; subpkg import may fail, this won't

        for base in sorted(map(str, getattr(nvidia, "__path__", []) or [])):
            candidates += sorted(glob.glob(os.path.join(base, "*", "lib", "libcudart.so.*")))
    except Exception:
        pass
    # 2. CUDA toolkit in a -devel base image
    for pat in (
        "/usr/local/cuda*/lib64/libcudart.so.*",
        "/usr/local/cuda*/targets/*/lib/libcudart.so.*",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.*",
    ):
        candidates += sorted(glob.glob(pat))
    # 3. system loader resolver
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    for cand in candidates:
        real = _verify(cand)
        if real is not None:
            return real
    return None


def _neutralize_tilelang_cudart_stub() -> None:
    """Repoint tilelang's libcudart_stub.so at the real libcudart to prevent a vLLM crash (flash #184).

    tilelang's stub is missing cudaDeviceReset; vLLM scans /proc/self/maps for libcudart and can
    pick up the stub, aborting import. Must run AFTER _ensure_fla_fastpath_on_hopper (tilelang
    reinstall would overwrite the stub) and BEFORE any model/vLLM import.
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
    if not os.path.lexists(stub):  # lexists: dangling symlink still counts as present
        return
    # Do NOT probe with ctypes.CDLL — that dlopens the stub into /proc/self/maps, the exact crash.
    # A dangling symlink is NOT done (os.path.exists follows links), fall through to re-point.
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
    """Enable fla's tilelang GDN backend on Hopper (sm90); no-op elsewhere.

    fla's Triton chunk_bwd kernel is broken on Hopper+Triton>=3.4 (fla #640); the tilelang backend
    is correct. Fail-closed: any install failure physically removes fla so transformers falls back
    to the pure-PyTorch delta rule rather than crashing.
    """
    import importlib
    import importlib.util
    import subprocess

    try:
        import torch

        if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9):
            return
    except Exception:
        return

    def _have(mod: str) -> bool:
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    def _ver(dist: str) -> str | None:
        """Installed version by metadata (not importability — wrong version can still find_spec)."""
        try:
            import importlib.metadata as _md

            return _md.version(dist)
        except Exception:
            return None

    def _pip(*args: str) -> bool:
        """Run pip install; return True iff exit 0. Timeout prevents a hung resolver burning the GPU."""
        try:
            rc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", *args],
                check=False,
                timeout=600,
            ).returncode
        except Exception:
            return False
        return rc == 0

    # 0.1.12 double-registers the TVM-FFI runtime -> `import tilelang` aborts; keep in lockstep with WORKER_DEPS / Dockerfile.
    TVM_FFI_PIN = "0.1.11"
    TILELANG_PIN = "0.1.11"

    try:
        tilelang_ok = True
        tilelang_reinstalled = False
        if _ver("tilelang") != TILELANG_PIN:
            tilelang_ok = _pip(f"tilelang=={TILELANG_PIN}")
            tilelang_reinstalled = True
        # Force the tvm-ffi pin when wrong OR tilelang was just installed (its range allows 0.1.12).
        if _ver("apache-tvm-ffi") != TVM_FFI_PIN or tilelang_reinstalled:
            tvm_ffi_ok = _pip(f"apache-tvm-ffi=={TVM_FFI_PIN}")
        else:
            tvm_ffi_ok = True
        # PyPI flash-linear-attention wheel is a stub missing fla.modules; reinstall from git.
        fla_ok = True
        if not (_have("fla") and _have("fla.modules")):
            _remove_fla_from_disk()
            # Pinned commit — keep in lockstep with WORKER_DEPS / Dockerfile.worker.
            fla_ok = _pip(
                "--no-deps",
                "git+https://github.com/fla-org/flash-linear-attention.git"
                "@f0e213dbd8b5fb90c3c7eca869ac1706d5377139",
            )
        importlib.invalidate_caches()
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
            # Must physically remove fla — transformers gates GDN on find_spec('fla'), so a print
            # alone won't prevent the broken Triton>=3.4 Hopper kernel from being engaged (fla #640).
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
    except Exception as e:  # fail-closed: remove fla so the pure-PyTorch delta rule runs
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
    "free_gpu",
    "fused_optim_name",
    "gpu_diagnostics",
    "grad_checkpointing_on",
    "grpo_sleep_mode",
    "liger_on",
    "loraplus_optimizer_cls",
    "optimal_attn_impl",
    "setup_perf_backends",
    "wait_for_gpu",
]
