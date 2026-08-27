"""Pinned VERL child-interpreter requirements and provisioning primitives."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

# verl 0.8.0 exactly, plus the truncation-mask, 3d position-id, and ulysses fused-label fixes, the
# whole-adapter lora bucket sync, and the fused-linear grad-flag preservation.
# it must stay on the 0.8.0 base: the opd plugin patches 0.8.0 internals and imports
# verl.trainer.main_ppo_sync, which verl deleted after 0.8.0, and opd's exact-version gate reads the
# version file this branch pins to the release value.
VERL_REQUIREMENT_NAME = "verl"
VERL_REQUIREMENT_URL = (
    "git+https://github.com/freesolo-co/verl@f71a02ddb32a9c6a6915f7519bda6dede92e9dd0"
)
# the pin, as the venv stamp records it. the provisioning install asks for the [vllm] extra of this
# same commit; the stamp stays extra-free so it identifies the verl a venv holds, not how it was
# installed.
VERL_REQUIREMENT = f"{VERL_REQUIREMENT_NAME} @ {VERL_REQUIREMENT_URL}"

# the fa2 wheel the verl interpreter needs, kept byte-identical to dockerfile.worker's
# arg flash_attn_spec default so the fallback venv and /opt/verl-venv resolve the same build.
# prebuilt for cu128/torch2.10/cp312; installed --no-build-isolation, never source-built here.
FLASH_ATTN_SPEC = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
    "v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl"
)
# sha256 of the wheel above, kept in lockstep with dockerfile.worker's arg flash_attn_sha256.
# the wheel is a community rebuild published by an individual, not dao-ailab, and a github release
# asset can be deleted and re-uploaded at the same url by whoever holds the account, so the url
# alone pins nothing, and this interpreter is the one that trains. bump it together with the url.
FLASH_ATTN_SHA256 = "a58c95a080363606e691c342d47dd173b510f4c013a8c5fcb8744def12e36a0f"
# what the install actually asks for. uv hashes the downloaded artifact and refuses to install it
# when the fragment does not match (verified against uv 0.11), so the check happens after the
# download and before anything is unpacked into the venv, the same order as the sha256sum -c on
# the infisical .deb in dockerfile. the spec above stays bare so it keeps matching dockerfile.worker's
# arg default and worker-image.yml's fa2_spec, which fetch and verify in two separate steps.
FLASH_ATTN_INSTALL_SPEC = f"{FLASH_ATTN_SPEC}#sha256={FLASH_ATTN_SHA256}"
# the wheel above is cp312-only, and flash itself supports 3.11 (pyproject requires-python >=3.11),
# so a bare `uv venv` on a 3.11 host builds an interpreter the wheel cannot install into -- and that
# install is required, so the run dies during provisioning instead of training. name the interpreter
# rather than inheriting the host's.
VERL_VENV_PYTHON = "3.12"

# gated-deltanet kernels for the verl interpreter, kept byte-identical to dockerfile.worker's
# verl-venv layer. fla resets the deltanet recurrence at packed example boundaries (cu_seqlens)
# and causal_conv1d resets the short causal conv (seq_idx); without them transformers falls back
# to implementations that accept both arguments and silently discard them, so packed gdn training
# is contaminated across example boundaries while appearing patched. both are required in the
# image; here causal_conv1d stays best-effort to install because this path has no build environment
# to guarantee, but a venv that misses it is left unstamped rather than reused (see below).
FLA_REQUIREMENT = (
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git"
    "@9c8e42e762fce087c27b673af4922795d9edb85e"
)
# the flashqla gdn backend lives in a separate wheel that fla 0.5.2 dispatches to; the child's
# shim binds it on sm90 only (child_io.render_flash_qla_shim).
FLASH_QLA_REQUIREMENT = "flash-qla==0.1.2"
CAUSAL_CONV1D_REQUIREMENT = "causal-conv1d==1.6.2.post1"

# the same transformers range the main interpreter and dockerfile.worker's verl-venv layer use.
# this venv is the interpreter that trains, and transformers owns the gdn modelling code the
# boundary-reset shim patches (chunk_gated_delta_rule, the causal conv, and the kwargs they read),
# so an unbounded resolve here means the training path silently rides a transformers line nothing
# validated. bounding it to the range flash tests against is the whole reason. verl and vllm both
# depend on transformers, so this has to be in the override file as well as the direct list -- a
# direct pin alone loses to their transitive declarations.
#
# not the reason: this pin does not affect is_flash_linear_attention_available() or
# is_causal_conv1d_available(). those are byte-identical in 5.12.1 and 5.14.1 and both open with
# is_torch_cuda_available(), so they answer false on any machine without a gpu regardless of the
# installed version. an earlier revision of this comment claimed a 5.13 import-path move made them
# answer false; that mechanism does not exist. see dockerfile.worker's sanity block.
TRANSFORMERS_REQUIREMENT = "transformers>=5.8,<5.13"
# exact deployed install and override pin; the range above remains public package metadata.
TRANSFORMERS_INSTALL_REQUIREMENT = "transformers==5.12.1"

# the stamp must identify every separately installed package the venv holds. omitting flash-attn,
# fla, causal_conv1d, or the exact transformers pin lets an older partial venv match forever; conv1d
# leaves grpo/opd failing ``require_gdn_boundary_resets`` with no rebuild path, and a stale venv
# resolved before the transformers pin keeps training on a different transformers indefinitely.
# bump when a repair applied at build time changes what a fully provisioned venv looks like. the
# libcudart repair runs only on the rebuild path, so a venv stamped by a release that predates it
# matches, is reused, and keeps tilelang's original stub -- vllm then aborts its import in the child
# after the gpu is already rented. carrying it in the stamp identity forces exactly one rebuild of
# those venvs instead of re-running the repair on every reuse.
VERL_VENV_BUILD_REPAIRS = "libcudart-stub-neutralized-v1"

# carries flash_attn_install_spec (the sha256-fragmented spec), not the bare flash_attn_spec: the
# digest is part of the venv's identity, not just how it got installed, so bumping or rotating the
# checksum while the mutable release url stays constant must invalidate a venv stamped under the old
# digest instead of letting it match forever and reuse an install that was never re-verified.
VERL_VENV_STAMP = (
    f"{VERL_REQUIREMENT}\n{FLASH_ATTN_INSTALL_SPEC}\n{FLA_REQUIREMENT}\n{FLASH_QLA_REQUIREMENT}\n"
    f"{CAUSAL_CONV1D_REQUIREMENT}\n{TRANSFORMERS_INSTALL_REQUIREMENT}\n"
    f"{VERL_VENV_BUILD_REPAIRS}"
)

# how many times the prebuilt-wheel install is attempted before the arm is handed back to the plane.
# uv already retries the download 3x internally, so each attempt here is a fresh uv invocation, not a
# fresh request: the point is to outlast an outage that spans one uv lifetime, not to hammer github.
FLASH_ATTN_INSTALL_ATTEMPTS = 3
FLASH_ATTN_INSTALL_BACKOFF_S = 15.0


def _install_flash_attn(py: str) -> None:
    """Install the required prebuilt FA2 wheel with bounded retries."""
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    command = [
        "uv",
        "pip",
        "install",
        "--python",
        py,
        "--no-build-isolation",
        FLASH_ATTN_INSTALL_SPEC,
    ]
    for attempt in range(FLASH_ATTN_INSTALL_ATTEMPTS):
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as error:
            if attempt == FLASH_ATTN_INSTALL_ATTEMPTS - 1:
                raise RetriableInfraError(
                    f"prebuilt flash-attn wheel install failed {FLASH_ATTN_INSTALL_ATTEMPTS}x "
                    f"(exit {error.returncode}): {FLASH_ATTN_SPEC}"
                ) from error
            time.sleep(FLASH_ATTN_INSTALL_BACKOFF_S * (attempt + 1))


def _install_causal_conv1d(py: str) -> bool:
    """Install and import-check the child interpreter's causal-conv1d extension."""
    completed = subprocess.run(
        ["uv", "pip", "install", "--python", py, "--no-build-isolation", CAUSAL_CONV1D_REQUIREMENT],
        check=False,
        env={**os.environ, "CAUSAL_CONV1D_FORCE_BUILD": "TRUE"},
    )
    if getattr(completed, "returncode", 0) != 0:
        return False
    imported = subprocess.run(
        [py, "-c", "import causal_conv1d"],
        check=False,
        capture_output=True,
    )
    if getattr(imported, "returncode", 0) != 0:
        subprocess.run(
            ["uv", "pip", "uninstall", "--python", py, "causal-conv1d"],
            check=False,
            capture_output=True,
        )
        return False
    return True


_CHILD_CUDART_FIX = """
import ctypes, ctypes.util, glob, importlib.util, os

def find_real():
    def verify(cand):
        try:
            lib = ctypes.CDLL(cand)
        except OSError:
            return None
        if not hasattr(lib, "cudaDeviceReset"):
            return None
        if os.path.isabs(cand) and os.path.exists(cand):
            return os.path.realpath(cand)
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

    candidates = []
    try:
        import nvidia
        for base in sorted(map(str, getattr(nvidia, "__path__", []) or [])):
            candidates += sorted(glob.glob(os.path.join(base, "*", "lib", "libcudart.so.*")))
    except Exception:
        pass
    for pat in (
        "/usr/local/cuda*/lib64/libcudart.so.*",
        "/usr/local/cuda*/targets/*/lib/libcudart.so.*",
        "/usr/lib/x86_64-linux-gnu/libcudart.so.*",
    ):
        candidates += sorted(glob.glob(pat))
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    for cand in candidates:
        real = verify(cand)
        if real is not None:
            return real
    return None

try:
    spec = importlib.util.find_spec("tilelang")
except Exception:
    spec = None
locs = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
if not locs:
    print("no tilelang in child venv; nothing to neutralize")
else:
    stub = os.path.join(locs[0], "lib", "libcudart_stub.so")
    if not os.path.lexists(stub):
        print("no libcudart_stub.so in child tilelang")
    elif os.path.islink(stub) and os.path.exists(stub):
        print("child stub already repointed")
    else:
        real = find_real()
        if real is None:
            print("no real libcudart found in child; left as-is")
            raise SystemExit(1)
        else:
            backup = stub + ".orig"
            if not os.path.exists(backup):
                os.replace(stub, backup)
            else:
                try:
                    os.remove(stub)
                except FileNotFoundError:
                    pass
            os.symlink(real, stub)
            print("redirected child tilelang libcudart_stub.so -> " + real)
"""


def _neutralize_child_tilelang_cudart_stub(py: str) -> bool:
    """Repoint the child venv's tilelang libcudart stub at a real libcudart."""
    try:
        result = subprocess.run(
            [py, "-c", _CHILD_CUDART_FIX],
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"[verl] child libcudart stub neutralize did not run: {error}", flush=True)
        return False
    for stream in (getattr(result, "stdout", None), getattr(result, "stderr", None)):
        if not stream:
            continue
        if isinstance(stream, bytes):
            stream = stream.decode("utf-8", "replace")
        for line in stream.splitlines():
            if line.strip():
                print(f"[verl] child cudart: {line}", flush=True)
    return getattr(result, "returncode", 0) == 0


def resolve_verl_python(workdir: str, *, install_wandb: bool = False) -> str:
    """Return or provision the pinned interpreter used by VERL children."""
    preset = os.environ.get("FLASH_VERL_PYTHON", "").strip()
    if preset:
        return preset
    venv = os.path.join(workdir, "verl-venv")
    py = os.path.join(venv, "bin", "python")
    stamp = os.path.join(venv, "flash-verl-requirement")
    installed = ""
    if os.path.exists(stamp):
        with open(stamp) as file:
            installed = file.read().strip()
    if installed != VERL_VENV_STAMP or not os.path.exists(py):
        shutil.rmtree(venv, ignore_errors=True)
        subprocess.run(["uv", "venv", "--python", VERL_VENV_PYTHON, venv], check=True)
        overrides = os.path.join(workdir, "verl-overrides.txt")
        with open(overrides, "w") as file:
            file.write(
                f"numpy==2.2.6\nxgrammar==0.1.25\nvllm==0.19.1\n"
                f"{TRANSFORMERS_INSTALL_REQUIREMENT}\n"
            )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                py,
                "--override",
                overrides,
                f"{VERL_REQUIREMENT_NAME}[vllm] @ {VERL_REQUIREMENT_URL}",
                "vllm==0.19.1",
                "numpy==2.2.6",
                TRANSFORMERS_INSTALL_REQUIREMENT,
                "cachetools==7.1.7",
                "uvicorn==0.52.4",
                "fastapi==0.141.1",
                "TransferQueue==0.1.7",
                "datasets==5.0.1",
                "bitsandbytes==0.50.1",
                "qwen-vl-utils==0.0.14",
                "torchvision==0.25.0",
                "xgrammar==0.1.25",
                "tqdm==4.70.0",
                "pyarrow==25.0.1",
                FLA_REQUIREMENT,
                FLASH_QLA_REQUIREMENT,
                "tilelang==0.1.9",
                "apache-tvm-ffi==0.1.9",
            ],
            check=True,
        )
        _install_flash_attn(py)
        conv_installed = _install_causal_conv1d(py)
        cudart_safe = _neutralize_child_tilelang_cudart_stub(py)
        if conv_installed and cudart_safe:
            with open(stamp, "w") as file:
                file.write(VERL_VENV_STAMP)
        elif not conv_installed:
            print(
                f"[verl] {CAUSAL_CONV1D_REQUIREMENT} did not install; leaving the venv unstamped so "
                "the next attempt rebuilds it rather than reusing an interpreter that cannot honor "
                "gdn boundary resets",
                flush=True,
            )
        else:
            print(
                "[verl] the child tilelang libcudart stub was not neutralized; leaving the venv "
                "unstamped so the next attempt rebuilds it rather than reusing an interpreter "
                "whose child aborts on vLLM import",
                flush=True,
            )
    if install_wandb:
        subprocess.run(["uv", "pip", "install", "--python", py, "wandb==0.28.2"], check=False)
    return py
