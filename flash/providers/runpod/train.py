"""RunPod Flash fine-tuning endpoints (queue-based, one dedicated GPU per run).

Flash provisions a dedicated RunPod GPU (RTX 4090 / 5090, no Docker), installs
``WORKER_DEPS``, runs the handler, returns the metrics dict, and scales to zero.

Flash's live ("ad-hoc") provisioning does not bundle local project code, so the
handler fetches the ``flash`` package from the HF dataset repo (uploaded by
``upload_code`` before submit), adds it to ``PYTHONPATH``, and runs
``flash.engine.worker`` to train. The worker streams the adapter + checkpoints to
the same HF repo for serving and preemption-resilient resume.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import threading
from typing import Any

from flash._logging import get_logger
from flash.providers.base import canonical_gpu, gpu_short
from flash.providers.runpod.gpus import flash_gpu
from flash.spec import JobSpec

logger = get_logger(__name__)

# The control plane runs each training run in its own thread. All runpod_flash deploy/
# undeploy work goes through a shared asyncio singleton whose Lock binds to the first event
# loop that touches it; two threads each calling asyncio.run() get distinct loops and the
# second fails with "Lock ... is bound to a different event loop". Serialize every Flash SDK
# async section (deploy AND undeploy) behind this one process-wide lock. Deploys/teardowns
# are infrequent vs training, so the serialization cost is negligible.
FLASH_SDK_LOCK = threading.Lock()


# Worker stack: trl 1.6 (colocate default; adds the GRPO `tools=` / `rollout_func`
# multi-turn hooks used for verifiers ToolEnv / MultiTurnEnv training), vllm 0.19.1
# (Qwen3.5/3.6 archs, native RL APIs, transformers-5
# compatible metadata), transformers 5.x (qwen3_5/qwen3_5_moe model types),
# bitsandbytes (4-bit NF4 QLoRA tier). trl 1.6 requires transformers>=4.56,
# satisfied by the 5.6+ pin; GRPOConfig is field-compatible with the 1.5 usage here.
# Resolver/driver notes: vllm 0.17/0.18 hard-pin transformers<5 (uv refuses the
# combo), so the first transformers-5-compatible vllm line is 0.19.1. vllm >=0.20
# pins torch 2.11 whose default pypi wheels are CUDA-13 builds — RunPod 4090/5090
# hosts filtered at min_cuda 12.8 often run 12.8/12.9 drivers where cu13 torch sees
# NO GPU (observed: "cuda not available" + vLLM "cumem allocator not supported").
# vllm 0.19.1 pins torch 2.10 (cu128 default) which matches those drivers.
# trl's *optional* [vllm] extra caps at 0.18, but we install plain trl, so the only
# constraint that matters is runtime API compat — validated per-model on real
# RTX 4090/5090 workers before promotion to default (see bench/results/phase1).
WORKER_DEPS = [
    "torch==2.10.0",
    "transformers>=5.6,<5.13",
    "trl>=1.6,<1.7",
    "peft>=0.19",
    "vllm==0.19.1",
    "bitsandbytes>=0.49",
    "datasets>=4.7,<6",
    "huggingface_hub>=0.25",
    "accelerate>=1.4",
    # NB: the HF `kernels` Hub package is intentionally NOT pinned here — the versions
    # compatible with torch2.10 break transformers 5.6-5.10's hub_kernels integration at IMPORT
    # (LayerRepository now requires a version; transformers passes none -> ValueError on every
    # `import transformers`). FlashAttention via the Hub is therefore disabled; attention uses
    # SDPA (already a flash/efficient backend on Ampere/Ada) + the Liger fused kernels below,
    # which are the dominant LoRA speedup anyway. (FA via a pinned flash-attn wheel is a future
    # per-arch experiment, kept out of the default deps to avoid a fragile cold-start install.)
    # Liger fused Triton kernels (pure Triton -> JITs on every arch incl. Blackwell): fused
    # linear cross-entropy for SFT (use_liger_kernel) and the chunked GRPO loss
    # (use_liger_loss) — the big large-vocab (Qwen ~152k) memory/throughput win.
    "wandb>=0.17",
    "liger-kernel>=0.5",
    # Fused Triton kernels for Gated-DeltaNet (Qwen3.5/3.6 family): without this,
    # transformers falls back to a pure-PyTorch delta rule and GRPO trainer steps are
    # 2-3x slower (measured A/B on Qwen3.5-2B: ~65 s/step -> ~20 s/step steady).
    "flash-linear-attention==0.5.0",
    # NB: fla's gated chunk_bwd is broken on HOPPER (H100) with Triton >= 3.4 (fla #640), so
    # resolve_worker_deps DROPS fla on sm90 (the correct pure-PyTorch delta rule runs instead). The
    # dense Qwen3.5 GDN models route to consumer cards by default, where fla works.
    # NB: freesolo-chalk (custom Triton/CUDA kernels that complement Liger) is NOT in this base dep
    # list, but its gap-fillers are default-on (flash/engine/chalk_kernels.py), so the submit path
    # (chalk_extra_pip) appends ``freesolo-chalk`` from PyPI to the worker's `extra_pip` for EVERY
    # job by default — installed + applied automatically, like Liger. Override the source with
    # FLASH_CHALK_SPEC (a pinned version / git URL / wheel), or disable every kernel (FLASH_<K>=0) to
    # skip the install. The worker pip-installs extra_pip for EVERY job (baked-image RunPod
    # _train_body + Vast bootstrap). Do NOT rely on
    # FLASH_WORKER_EXTRA_DEPS / FLASH_WORKER_DEPS for this: the durable baked-image submit path
    # (jobs.build_function_input) returns the raw payload and never consults resolve_worker_deps, so
    # those vars don't reach a default run; and FLASH_WORKER_DEPS would also REPLACE the whole stack.
]
# NOTE on download speed: Flash's runtime already ships hf_transfer and exports
# HF_HUB_ENABLE_HF_TRANSFER=1 on workers (measured: Qwen3-4B's ~8 GB pulled in 6.3 s,
# NIC-saturated — bench/results/phase6). Adding hf_transfer here is redundant; don't.
# Override the whole pinned stack per-run with FLASH_WORKER_DEPS="pkgA==1 pkgB>=2"
# (whitespace-separated, or a JSON list for specs containing commas).
WORKER_SYSTEM_DEPS = ["build-essential"]  # Triton/Inductor need a C compiler

# The prebuilt worker image (full training stack baked in; built by Dockerfile.worker /
# .github/workflows/worker-image.yml). PUBLIC under the org namespace, so no registry login is
# ever needed. Always used on both Vast and RunPod — there is no operator override and no generic
# fallback image. Must be published to GHCR + made public before runs can pull it.
WORKER_IMAGE = "ghcr.io/freesolo-co/flash-worker:cu128"


def resolve_worker_deps(friendly_gpu: str | None = None) -> list[str]:
    """The dependency list Flash installs on the GPU worker for this run.

    Precedence: FLASH_WORKER_DEPS (explicit list) > the pinned ``WORKER_DEPS``.

    GPU-specific: on HOPPER (sm90, H100), DROP flash-linear-attention — its gated
    chunk_bwd Triton kernel is miscomputed there (Triton>=3.4, fla #640). Without fla,
    transformers uses the correct pure-PyTorch delta rule (slower but correct).
    Ampere/Ada/Blackwell keep fla for the speedup.
    """
    explicit = os.environ.get("FLASH_WORKER_DEPS")
    if explicit:
        # JSON list (use this for specs containing commas, e.g.
        # "transformers>=5.6,<5.13") or a whitespace-separated string.
        if explicit.strip().startswith("["):
            import json as _json

            deps = [str(d).strip() for d in _json.loads(explicit) if str(d).strip()]
        else:
            # shlex (whitespace) splitting, NOT comma: a comma is part of a PEP 440
            # range like `transformers>=5.6,<5.11` and must not be split.
            import shlex

            deps = [d for d in shlex.split(explicit) if d.strip()]
        if deps:
            return deps
    deps = list(WORKER_DEPS)
    # Hopper (sm90) fla strategy: DROP flash-linear-attention -> the correct pure-PyTorch delta
    # rule. fla's gated chunk_bwd Triton kernel is miscomputed on Hopper (Triton>=3.4, fla #640),
    # so on H100 we run without it (the dense Qwen3.5 GDN models route to consumer cards by
    # default, where fla stays). Ampere/Ada/Blackwell keep fla for the speedup.
    if friendly_gpu:
        try:
            from flash.providers.base import get_gpu_info

            if get_gpu_info(friendly_gpu).sm == "sm90":
                deps = [d for d in deps if not d.startswith("flash-linear-attention")]
        except Exception:
            pass
    # Additive per-run extras (e.g. an extra pinned wheel for an A/B) without
    # restating the whole pinned stack the way FLASH_WORKER_DEPS requires.
    extra = os.environ.get("FLASH_WORKER_EXTRA_DEPS")
    if extra:
        import shlex

        deps = deps + [d for d in shlex.split(extra) if d.strip()]
    return deps


def _effective_worker_env(spec=None) -> dict[str, str]:
    """The env the WORKER process will actually see, for chalk-selection decisions.

    chalk install-on-call is selected by ``FLASH_*`` flags read on the worker from its own process
    env, which ``build_worker_env`` builds as the control-plane ``os.environ`` allowlist with the
    run's ``[worker_env]`` overrides merged ON TOP (per-run ``spec.worker_env`` wins). A run that
    opts into chalk via its ``[worker_env]`` block therefore sets the flag the worker reads — so the
    SAME merge must decide whether chalk is selected and whether its spec is added to ``extra_pip``;
    reading bare ``os.environ`` here would miss a per-run ``[worker_env]`` opt-in and the kernels
    would never install for that run.

    Returns ``os.environ`` overlaid with ``spec.worker_env`` (string-coerced). ``spec=None`` (no
    per-run env) collapses to plain ``os.environ``.
    """
    eff: dict[str, str] = dict(os.environ)
    for k, v in (getattr(spec, "worker_env", None) or {}).items():
        eff[str(k)] = str(v)
    return eff


def _chalk_selected(spec=None) -> bool:
    """True if ANY chalk kernel would run on the worker -> chalk must be installed.

    chalk's gap-fillers (RoPE/QKV/LoRA/embedding) are ON BY DEFAULT (engine.chalk_kernels), so this
    is True for a normal run. It is False only when every kernel that would otherwise be enabled is
    explicitly set to 0 — i.e. all four default-on gap-fillers disabled via ``FLASH_<K>=0`` (the
    default-off opt-in kernels need no flag to stay off). Resolved against the EFFECTIVE worker env
    — the run's ``[worker_env]`` merged over ``os.environ`` so a per-run override is honored (see
    ``_effective_worker_env``). Delegates to ``chalk_kernels.is_chalk_enabled`` (the single source
    of truth for the flag/default logic) rather than re-parsing the kernel table here.
    """
    from flash.engine.chalk_kernels import is_chalk_enabled

    return is_chalk_enabled(_effective_worker_env(spec))


def chalk_extra_pip(spec=None) -> list[str]:
    """Chalk pip spec(s) to ADD to the worker's ``extra_pip`` when a chalk kernel is selected.

    This is the install hook that runs for DEFAULT remote jobs: the baked-image RunPod path
    (``_train_body`` -> ``pip install *extra_pip``) and the Vast bootstrap both consume the
    payload's ``extra_pip`` regardless of ``WORKER_IMAGE`` — unlike ``FLASH_WORKER_EXTRA_DEPS``
    / ``resolve_worker_deps``, which the durable ``build_function_input`` baked-image path skips.

    Selection (and the ``FLASH_CHALK_SPEC`` lookup) is resolved against the EFFECTIVE worker env —
    the run's ``[worker_env]`` merged over ``os.environ`` — so it matches exactly what the worker
    process will see (``build_worker_env``) and a per-run ``[worker_env]`` opt-in installs chalk.

    Chalk's gap-fillers are default-on, so chalk is selected for a normal run even with no FLASH_*
    flags set. freesolo-chalk is published on PyPI, so it auto-installs by DEFAULT (just like Liger):
    when chalk is selected and ``FLASH_CHALK_SPEC`` is unset we add ``freesolo-chalk``. Set
    ``FLASH_CHALK_SPEC`` to override the source (a pinned version, a git URL, or a wheel/path), or
    disable every kernel (``FLASH_<K>=0``) to skip the install.
    """
    if not _chalk_selected(spec):
        return []
    # PyPI default — chalk is published, so a normal run installs + applies it automatically.
    spec_str = _effective_worker_env(spec).get("FLASH_CHALK_SPEC", "").strip() or "freesolo-chalk"
    import shlex

    return [d for d in shlex.split(spec_str) if d.strip()]


DEFAULT_EXECUTION_TIMEOUT_MS = 6 * 3600 * 1000  # 6h RunPod worker execution cap

_ENDPOINT_CACHE: dict[str, Any] = {}


def upload_code(repo: str | None = None) -> str:
    """Upload the ``flash`` package to the run's HF artifact repo.

    ``repo`` is the per-run artifact repo (``spec.train.hf_repo``); the worker fetches
    ``code/**`` from the same repo it is given in the submit payload, so the code must land in
    that per-run repo.

    The worker downloads ``code/**`` to ``/runcode``. Verifiers-only: there are no built-in
    example environments to ship — Hub/installed envs are pip-installed on the worker (see
    ``registry.worker_pip_for_env``).

    Only the ``flash`` package is uploaded, NOT the client's project tree. Managed runs must
    reference a published Hub env by ``id`` (``slm env push`` to publish a local env first); the
    worker pip-installs the env wheel.
    """
    from huggingface_hub import HfApi

    import flash

    if not repo:
        raise RuntimeError(
            "hf_repo must be set (the run's [train] hf_repo: HF dataset repo for code + artifacts)"
        )
    token = os.environ.get("HF_TOKEN")
    pkg_dir = os.path.dirname(os.path.abspath(flash.__file__))
    api = HfApi(token=token)
    # Run artifact repos are always private (they carry run code, adapters, and metrics).
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=True)
    # create_repo(exist_ok=True) is a no-op on an EXISTING repo, so `private=True` above does NOT
    # change the visibility of a repo that was created earlier as public. Force private explicitly
    # so a reused/public artifact repo can't leak run code/adapters/metrics under the always-private
    # invariant. (Idempotent: a no-op on a repo that is already private.)
    api.update_repo_settings(repo_id=repo, repo_type="dataset", private=True)
    api.upload_folder(
        folder_path=pkg_dir,
        path_in_repo="code/flash",
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=["__pycache__/*", "*.pyc"],
    )
    return repo


def _train_body(input_data: dict) -> dict:
    """Runs ON the RunPod GPU worker: fetch code, train (phase), return metrics.

    NOTE: Flash serializes this handler and runs it standalone, so every name it uses
    must be imported INSIDE the function body (module-level imports are not in scope).
    """
    import contextlib
    import json
    import os
    import shutil
    import subprocess
    import sys

    from huggingface_hub import snapshot_download

    # NB: the Hopper fla guard lives in engine.worker._drop_fla_on_hopper (runs in the worker
    # process AFTER all installs, before any model import) — doing it here would be undone by a
    # later extra_pip / `prime env install` that pulls fla back, and depends on a handler redeploy.

    # Extra pip deps for verifiers / Prime Hub environments (installed per-run).
    extra_pip = input_data.get("extra_pip") or []
    if extra_pip:
        # check=True: a deterministic dependency failure should fail fast here,
        # not after model download + worker startup with a less actionable error.
        subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip], check=True)

    # NB: fla is dropped on Hopper (sm90) automatically — resolve_worker_deps omits it from the
    # install list, and engine.worker._drop_fla_on_hopper removes any baked-in copy at worker
    # startup (fla's GDN backward is miscomputed on sm90, #640). No env toggle: fla only ever runs
    # on the consumer archs where its Triton kernel is correct.

    # Install the run's verifiers environment(s) from the Prime Hub via the authenticated
    # `prime` CLI. The public pip index does not serve PRIVATE env wheels, so a plain pip
    # install can't fetch them; `prime env install` pulls/builds/installs public + private
    # alike, authenticated by PRIME_API_KEY forwarded from the control plane.
    hub_env_ids = input_data.get("hub_env_ids") or []
    if hub_env_ids:
        worker_env = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
        prime_key = worker_env.get("PRIME_API_KEY") or os.environ.get("PRIME_API_KEY")
        if not prime_key:
            raise RuntimeError(
                "PRIME_API_KEY is required to install the Prime Hub environment on the worker"
            )
        # Only install `prime` when it isn't already on the worker (it's often baked into
        # the worker image) — an unconditional install adds latency and a per-run PyPI
        # failure point every run.
        if shutil.which("prime") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "prime"], check=True)
        # --with pip: install the env into THIS (the trainer's) python via pip. The default
        # (`--with uv`) installs into prime's own isolated uv env, so the trainer then can't
        # import the env module (ModuleNotFoundError at load_environment). PIP_BREAK_SYSTEM_PACKAGES
        # lets pip write to a PEP-668 "externally-managed" base python (the worker image's).
        install_env = {
            **os.environ,
            "PRIME_API_KEY": prime_key,
            "PRIME_DISABLE_VERSION_CHECK": "1",
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
        }
        for env_id in hub_env_ids:
            subprocess.run(
                ["prime", "env", "install", env_id, "--with", "pip"], check=True, env=install_env
            )

    overrides = {k: str(v) for k, v in (input_data.get("env") or {}).items()}
    snapshot_download(
        repo_id=input_data["hf_repo"],
        repo_type="dataset",
        allow_patterns=["code/**"],
        local_dir="/runcode",
        token=overrides.get("HF_TOKEN"),
    )
    code_dir = "/runcode/code"

    env = dict(os.environ)
    env.update(overrides)
    # Always pass the spec via a file (FLASH_JOB_SPEC_PATH): a large inline spec can blow past the
    # ~128 KiB per-env-string exec limit ("Argument list too long"), and a file is ONE code path for
    # every size (cheap write). load_job_spec_from_env reads it.
    spec_path = "/tmp/job_spec.json"
    with open(spec_path, "w") as sf:
        sf.write(input_data["job_spec_json"])
    env["FLASH_JOB_SPEC_PATH"] = spec_path
    env.pop("FLASH_JOB_SPEC_JSON", None)
    env["PHASE"] = input_data["phase"]
    env["SEED"] = str(input_data["seed"])
    env["PYTHONPATH"] = code_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    def run_mode(mode: str, check: bool) -> int:
        """Run one worker process, tee its console to a file, and upload the tail to HF as
        console_<mode>.txt on failure — the engine-core root cause of crashes like vLLM
        EngineDeadError only ever appears on the subprocess console, never in the Python
        traceback. With FLASH_UPLOAD_CONSOLE=1 (forwarded via build_worker_env) the console
        is also uploaded on SUCCESS, so an operator can verify which optimizations engaged."""
        console = f"/tmp/console_{mode}.txt"
        with open(console, "w") as cf:
            proc = subprocess.Popen(
                [sys.executable, "-m", "flash.engine.worker"],
                cwd=code_dir,
                env={**env, "RUN_MODE": mode},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                print(line, end="")  # keep streaming to the platform console
                cf.write(line)
            proc.wait()
        # Console is uploaded on FAILURE (crash root-cause). FLASH_UPLOAD_CONSOLE=1 also uploads it
        # on SUCCESS so an operator can verify which optimizations engaged — LoRA+/8-bit-AdamW/
        # Liger/PiSSA/rsLoRA/fla/chalk all log their engagement (or fallback) to the console.
        _force_console = env.get("FLASH_UPLOAD_CONSOLE", "").strip().lower() not in (
            "", "0", "false", "no", "off",
        )
        if proc.returncode != 0 or _force_console:
            try:
                from huggingface_hub import HfApi

                spec = json.loads(input_data["job_spec_json"])
                phase_ns = "rl" if spec.get("algorithm") == "grpo" else spec["algorithm"]
                prefix = f"{phase_ns}/{spec['run_id']}/seed{input_data['seed']}"
                with open(console) as f:
                    tail = f.read()[-64_000:]
                with open(console + ".tail", "w") as f:
                    f.write(tail)
                HfApi(token=env.get("HF_TOKEN")).upload_file(
                    path_or_fileobj=console + ".tail",
                    path_in_repo=f"{prefix}/console_{mode}.txt",
                    repo_id=input_data["hf_repo"],
                    repo_type="dataset",
                )
            except Exception as up_err:
                print("console upload warn:", up_err)
        if proc.returncode != 0 and check:
            raise RuntimeError(
                f"worker mode '{mode}' exited {proc.returncode}; see console_{mode}.txt "
                f"and error_{mode}.txt in the HF dataset repo"
            )
        return proc.returncode

    # A warm worker can carry a previous seed's metrics files; a stale metrics.json
    # would let a crashed train phase report the previous run's numbers. Clear before
    # training.
    for stale in ("/tmp/train_meta.json", "/tmp/metrics.json"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(stale)
    # Train. check=False — RL's colocated vLLM can segfault at interpreter exit AFTER
    # the adapter + metrics.json + DONE are saved; don't treat that as a failure.
    run_mode(input_data["phase"], check=False)
    # The train phase writes metrics.json + the DONE sentinel itself (RunPod can also
    # redeliver a completed job, whose worker restores metrics.json from DONE). If it
    # is missing, the train phase crashed before finishing — fail fast with the real
    # cause (full traceback in error_<phase>.txt / console_<phase>.txt in the HF repo).
    if not os.path.exists("/tmp/metrics.json"):
        phase = input_data["phase"]
        raise RuntimeError(
            f"train phase '{phase}' produced no /tmp/metrics.json (it crashed before "
            f"finishing); see error_{phase}.txt and console_{phase}.txt in the HF "
            f"dataset repo for the full traceback"
        )
    with open("/tmp/metrics.json") as f:
        return json.load(f)


def isolate_flash_state(scope: str | None = None) -> None:
    """Point the Flash SDK's resource registry at a per-process/private directory.

    The SDK persists its registry to ``./.flash/resources.pkl`` — shared, whole-dict,
    last-writer-wins across every process in the CWD. Observed failure modes: stale
    entries resurrecting long-dead endpoints on later syncs, and concurrent processes
    clobbering each other's bookkeeping. Each Flash process gets its own registry
    under ``~/.flash/flash-state/<scope>``; remote cleanup never relies on the
    registry anyway (REST by id/name — see api.py).
    """
    try:
        from pathlib import Path

        import runpod_flash.core.resources.resource_manager as rm

        scope = scope or f"pid{os.getpid()}"
        state_dir = Path.home() / ".flash" / "flash-state" / scope
        state_dir.mkdir(parents=True, exist_ok=True)
        rm.FLASH_STATE_DIR = state_dir
        rm.RESOURCE_STATE_FILE = state_dir / "resources.pkl"
    except Exception as exc:  # never block a run on this
        logger.warning("flash state isolation skipped: %s", exc)


def _patch_runpod_backoff() -> None:
    """Work around a runpod_flash bug that aborts long-running jobs.

    The SDK polls a synchronous job with exponential backoff computed as
    ``base * (2 ** attempt)`` and only clamps to ``max_seconds`` afterwards. On a long
    run the poll ``attempt`` grows without bound, so ``2 ** attempt`` becomes a huge int
    and the float multiply raises ``OverflowError: int too large to convert to float``
    (observed ~80 min in), killing an otherwise-healthy job mid-run. We patch the symbol
    to cap the exponent before the power so the delay still saturates at ``max_seconds``.
    """
    try:
        import math
        import random

        from runpod_flash.core.utils import backoff as _bo

        if getattr(_bo, "_flash_backoff_patched", False):
            return

        def _safe_get_backoff_delay(
            attempt,
            base=0.1,
            max_seconds=10.0,
            jitter=0.2,
            strategy=_bo.BackoffStrategy.EXPONENTIAL,
        ):
            a = min(int(attempt), 30)  # cap exponent: 2**30 is plenty; delay saturates anyway
            if strategy == _bo.BackoffStrategy.EXPONENTIAL:
                delay = base * (2**a)
            elif strategy == _bo.BackoffStrategy.LINEAR:
                delay = base + (attempt * base)
            elif strategy == _bo.BackoffStrategy.LOGARITHMIC:
                delay = base * math.log2(attempt + 2)
            else:
                raise ValueError(f"Unsupported backoff strategy: {strategy}")
            delay = min(delay, max_seconds)
            return delay * random.uniform(1 - jitter, 1 + jitter)

        _bo.get_backoff_delay = _safe_get_backoff_delay
        _bo._flash_backoff_patched = True
        # serverless.py did `from ..utils.backoff import get_backoff_delay`, so patch its ref too.
        try:
            from runpod_flash.core.resources import serverless as _sl

            _sl.get_backoff_delay = _safe_get_backoff_delay
        except Exception:
            # serverless.py may not import the symbol in this SDK version; the primary
            # patch above still applies, so a missing alias is fine to ignore.
            pass
    except Exception as exc:  # never let the patch break submission
        logger.warning("runpod backoff patch skipped: %s", exc)


def min_cuda_for(friendly_gpu: str) -> str:
    """Minimum host CUDA (driver) version for this GPU class on the active stack.

    Blackwell classes (sm_120 — RTX 5090, RTX Pro 6000): pypi wheels for
    the modern stack (vllm 0.19) ship no Blackwell SASS, so every custom CUDA kernel
    is PTX-JIT'd by the driver — and their PTX is built with a newer toolchain than
    CUDA-12.8-era drivers can JIT (observed: "the provided PTX was compiled with an
    unsupported toolchain" on driver 570.x). CUDA-13 drivers JIT it fine, so those
    classes are pinned to >=13.0 on the modern stack (per-GPU ``min_cuda_modern`` in
    providers.base.GPU_INFO). Ampere/Ada/Hopper have SASS in the wheels and run on 12.8.
    Fully managed per-GPU (no override).
    """
    from flash.providers.base import min_cuda_modern

    return min_cuda_modern(friendly_gpu)


def endpoint_name(friendly_gpu: str, suffix: str | None = None) -> str:
    """Flash endpoint/template name for a GPU class, optionally made unique per run.

    A fixed name (``flash-5090``) collides across back-to-back runs: runpod_flash's
    ``get_or_deploy_resource`` finds the prior run's still-registered resource and tries to
    *update* it, which fails with ``GraphQL errors: Template name must be unique`` (there is
    no endpoint GC/reuse). A per-run ``suffix`` (the run id tail) gives each run its own
    endpoint so it deploys fresh instead of colliding. RunPod scales each to zero when idle.
    """
    base = f"flash-{gpu_short(friendly_gpu)}"
    if not suffix:
        return base
    safe = "".join(c for c in str(suffix) if c.isalnum() or c == "-").strip("-")[:24]
    return f"{base}-{safe}" if safe else base


def get_train_endpoint(
    friendly_gpu: str,
    execution_timeout_ms: int | None = None,
    name_suffix: str | None = None,
    disk_gb: int | None = None,
    spec=None,
):
    """Build (and cache) the live Flash endpoint handler for a GPU class."""
    # Live ("ad-hoc") provisioning: provision on call, no separate `flash deploy`.
    os.environ["FLASH_IS_LIVE_PROVISIONING"] = "true"
    from runpod_flash import Endpoint

    from flash.providers.runpod.auth import ensure_auth
    from flash.providers.runpod.jobs import volume_endpoint_kwargs

    ensure_auth()
    _patch_runpod_backoff()
    isolate_flash_state(name_suffix)

    friendly = canonical_gpu(friendly_gpu)
    name = endpoint_name(friendly, name_suffix)
    if name in _ENDPOINT_CACHE:
        return _ENDPOINT_CACHE[name]
    kwargs = dict(
        name=name,
        gpu=flash_gpu(friendly),
        gpu_count=1,
        min_cuda_version=min_cuda_for(friendly),
        execution_timeout_ms=execution_timeout_ms or DEFAULT_EXECUTION_TIMEOUT_MS,
        workers=(0, 1),  # one dedicated worker per run; scale to zero when idle
        **volume_endpoint_kwargs(spec),
    )
    # RunPod Flash needs its serverless runtime baked into the worker image; the prebuilt
    # WORKER_IMAGE is Vast's cold-start image (no Flash runtime) and leaves the worker unhealthy if
    # forced. RunPod boot-installs WORKER_DEPS on Flash's default template instead (cached as a
    # Flash artifact). Optional FLASH_WORKER_IMAGE override for a RunPod-serverless-compatible image.
    image = os.environ.get("FLASH_WORKER_IMAGE")
    if image:
        kwargs["image"] = image
    else:
        kwargs["dependencies"] = resolve_worker_deps(friendly)
        kwargs["system_dependencies"] = WORKER_SYSTEM_DEPS
    ep = Endpoint(**kwargs)
    handler = ep(_train_body)  # register the queue-based handler; returns the callable
    # The resource config is cached on the Endpoint, so raising the disk on it here
    # carries through to the deploy that the first handler call triggers.
    from flash.providers.runpod.jobs import apply_disk_gb

    cfg = ep._build_resource_config()
    apply_disk_gb(cfg, disk_gb)
    _ENDPOINT_CACHE[name] = handler
    return handler


def _run_suffix(run_id: str | None) -> str | None:
    """Short, COLLISION-FREE per-run endpoint suffix.

    Must be unique per run_id: the endpoint name is ``endpoint_name(friendly, suffix)`` and
    RunPod reuses an endpoint by name -- two runs with the same suffix share one endpoint (and
    its cached image/deps/registry-auth/template), so a later run silently reuses the earlier
    one's config. The old ``run_id.split("-")[-1]`` only worked for hash-tailed default ids; a
    descriptive run_id ending in e.g. the card name (``...-a100``) collided across every run.
    Use a stable short hash of the WHOLE run_id, with a sanitized prefix for readability."""
    if not run_id:
        return None
    import hashlib
    import re

    h = hashlib.sha1(run_id.encode()).hexdigest()[:8]
    prefix = re.sub(r"[^a-z0-9]", "", run_id.lower())[-12:]
    return f"{prefix}{h}" if prefix else h


def stop_endpoint(friendly_gpu: str, name: str | None = None) -> None:
    """Best-effort: scale cached endpoint(s) to zero / drop them.

    With ``name`` only that run's cached endpoint is dropped; without it, every
    cached endpoint of the GPU class is — so a per-run teardown passes ``name``
    to avoid evicting a concurrent run's handler in the same process.

    NOTE: this only touches THIS process's in-memory cache, so it does nothing in a fresh
    ``slm cancel`` process. Use ``terminate_endpoint`` to actually delete the remote endpoint.
    """
    friendly = canonical_gpu(friendly_gpu)
    prefix = f"flash-{gpu_short(friendly)}"
    if name:
        match = [k for k in _ENDPOINT_CACHE if k == name]
    else:
        match = [k for k in _ENDPOINT_CACHE if k.startswith(prefix)]
    for key in match:
        handler = _ENDPOINT_CACHE.pop(key, None)
        ep = getattr(handler, "__self__", None) or getattr(handler, "endpoint", None)
        for meth in ("scale_to_zero", "stop", "delete"):
            fn = getattr(ep, meth, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    continue


def _select_endpoint_resources(resources: dict, target: str) -> list[str]:
    """Resource ids whose resource ``.name`` contains ``target``.

    The live-provisioned resource is named ``live-<endpoint_name>``, so we match by substring
    to catch the prefix. ``target`` is the endpoint name (``flash-<gpu>[-<run>]``).
    """
    if not target:
        return []
    out = []
    for uid, res in (resources or {}).items():
        name = str(getattr(res, "name", "") or "")
        if target in name:
            out.append(uid)
    return out


def terminate_endpoint(friendly_gpu: str, run_id: str | None = None) -> list[dict]:
    """Reliably tear down the remote Flash endpoint(s) for a run — cross-process.

    Unlike ``stop_endpoint`` (which only touches this process's in-memory cache), this looks
    the endpoint up by name in runpod_flash's *persisted* resource registry and deletes it via
    the RunPod API (``ResourceManager.undeploy_resource`` -> ``delete_endpoint``), which stops
    any running worker. Best-effort: never raises. Returns the per-resource undeploy results.

    With ``run_id`` it targets exactly that run's uniquely-named endpoint; without it, the
    bare ``flash-<gpu>`` prefix matches every endpoint of that GPU class.
    """
    friendly = canonical_gpu(friendly_gpu)
    target = endpoint_name(friendly, _run_suffix(run_id))
    # Hold FLASH_SDK_LOCK across the ENTIRE Flash critical section, not just the undeploy.
    # isolate_flash_state() swaps runpod_flash's process-wide registry globals and
    # ResourceManager shares the SDK's asyncio singleton, so a concurrent deploy/undeploy on
    # another thread could swap the registry scope between our lookup and our undeploy and tear
    # down the wrong run's resources. Serialize isolation + lookup + undeploy together.
    with FLASH_SDK_LOCK:
        try:
            from flash.providers.runpod.auth import ensure_auth

            ensure_auth()
            isolate_flash_state(_run_suffix(run_id))
            from runpod_flash.core.resources.resource_manager import ResourceManager
        except Exception as exc:  # SDK/auth unavailable
            return [{"success": False, "name": target, "message": f"flash unavailable: {exc}"}]

        try:
            rm = ResourceManager()
            resources = rm.list_all_resources()
            uids = _select_endpoint_resources(resources, target)
        except Exception as exc:
            return [{"success": False, "name": target, "message": f"resource lookup failed: {exc}"}]

        async def _undeploy_all() -> list:
            out = []
            for uid in uids:
                res = resources.get(uid)
                name = getattr(res, "name", None)
                try:
                    out.append(
                        await rm.undeploy_resource(uid, resource_name=name, force_remove=True)
                    )
                except Exception as exc:
                    out.append({"success": False, "name": name, "message": str(exc)})
            return out

        try:
            results = asyncio.run(_undeploy_all())
        except Exception as exc:
            results = [{"success": False, "name": target, "message": str(exc)}]

    # Registry-less fallback: isolate_flash_state() keeps the Flash SDK's resource
    # registry per-process under ~/.flash, so a recreated container (or a crash before
    # on_handle() persisted the endpoint id) leaves the live endpoint invisible to the
    # lookup above. Delete it via the RunPod REST API by its reconstructed name so it
    # can't keep a paid worker alive.
    if not uids:
        with contextlib.suppress(Exception):
            from flash.providers.runpod import api as runpod_api

            for ep in runpod_api.find_endpoints_by_name(target):
                if ep.get("name") == target and runpod_api.delete_endpoint(ep["id"]):
                    results.append(
                        {"success": True, "name": target, "message": "deleted via REST API"}
                    )

    # also drop the in-process cached handler for THIS run only (a class-wide
    # drop would evict a concurrent run's endpoint on the same GPU class).
    with contextlib.suppress(Exception):
        stop_endpoint(friendly, name=target)
    return results


def build_worker_env(spec: JobSpec, seed: int) -> dict:
    """Per-run env passed to the worker (secrets + recipe overrides)."""
    # CUDA allocator conf. Colocate (TRL trainer + vLLM on one GPU) fragments over a long run,
    # so expandable_segments (which reclaims fragmentation) is the right default — EXCEPT under
    # GRPO vLLM sleep mode, whose CuMemAllocator memory pool is incompatible with
    # expandable_segments (vLLM asserts and the run crashes at engine init). So for RL with
    # sleep mode ON (the default), default to a non-expandable conf instead; SFT and
    # sleep-off RL keep expandable_segments. An explicit operator override always wins.
    _is_rl = str(getattr(spec, "algorithm", "")).lower() not in ("sft",)
    # RL_VLLM_SLEEP may be pinned per-run via [worker_env] (highest precedence, merged into the
    # worker env later) OR via the control-plane process env. Resolve it from BOTH here — with
    # worker_env winning — so a per-run explicit pin counts as explicit: otherwise _sleep_set stays
    # false, FLASH_ALLOC_AUTO=1 is sent, and the worker can upgrade to expandable_segments while
    # run_rl still enables vLLM sleep, hitting the CuMemAllocator incompatibility after provisioning.
    _sleep_raw = (spec.worker_env or {}).get("RL_VLLM_SLEEP", os.environ.get("RL_VLLM_SLEEP"))
    _sleep_set = _sleep_raw is not None
    _sleep_on = (_sleep_raw if _sleep_raw is not None else "1") not in ("0", "false", "False")
    _alloc_default = (
        "garbage_collection_threshold:0.8,max_split_size_mb:256"
        if (_is_rl and _sleep_on)
        else "expandable_segments:True"
    )
    # torch >= 2.10 renamed the env to PYTORCH_ALLOC_CONF — set BOTH names for either stack.
    # Resolve the override from [worker_env] AND the control-plane process env (worker_env wins,
    # mirroring RL_VLLM_SLEEP above): a per-run [worker_env] PYTORCH_ALLOC_CONF is merged into the
    # worker env later (and would win), but if it isn't counted as an explicit override HERE,
    # _alloc_override stays falsy, FLASH_ALLOC_AUTO=1 is sent, and finalize_alloc_conf_for_sleep
    # overwrites the operator's per-run pin.
    _we = spec.worker_env or {}
    _alloc_override = (
        _we.get("PYTORCH_ALLOC_CONF")
        or _we.get("PYTORCH_CUDA_ALLOC_CONF")
        or os.environ.get("PYTORCH_ALLOC_CONF")
        or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    )
    _alloc_conf = _alloc_override or _alloc_default
    env: dict[str, str] = {
        "RUN_ID": spec.run_id,
        # Compute substrate, read back by engine.worker for the RunMetrics record. Vast's
        # on-instance bootstrap overrides this to "vast" (it reuses this same env builder).
        "FLASH_ARM": "runpod",
        "BENCH_HF_MODEL": spec.model,
        "PYTORCH_CUDA_ALLOC_CONF": _alloc_conf,
        "PYTORCH_ALLOC_CONF": _alloc_conf,
        # We picked a DEFAULT alloc conf above without knowing the worker's resolved vLLM sleep
        # decision (RL + RL_VLLM_SLEEP unset + no operator override). Cede the final choice to the
        # worker, which resolves sleep from the model config and upgrades to expandable_segments
        # when sleep is OFF (engine.worker.finalize_alloc_conf_for_sleep). Never set when the
        # operator pinned an alloc conf or RL_VLLM_SLEEP explicitly — their choice is authoritative.
        **(
            {"FLASH_ALLOC_AUTO": "1"}
            if (_is_rl and not _sleep_set and not _alloc_override)
            else {}
        ),
        # Escape hatch for torch.compile/inductor spikes (Qwen3.5 DeltaNet kernels
        # compile at first forward and can OOM a tight colocate budget).
        **(
            {"TORCHDYNAMO_DISABLE": os.environ["TORCHDYNAMO_DISABLE"]}
            if os.environ.get("TORCHDYNAMO_DISABLE")
            else {}
        ),
    }
    # HF artifact creds + PRIME_API_KEY (the worker `prime env install`s the run's Hub
    # env(s), public + private) + optional reward-judge creds: a verifiers env whose rubric
    # calls an LLM judge (e.g. OpenRouter gpt-oss-120b) needs the API key ON THE WORKER,
    # where the reward runs. FLASH_JUDGE_MODEL is the judge model id the optimizer-authored env
    # reads (agents/common/prompt.py) to pick the JudgeRubric client model; forward the operator's
    # control-plane override so SFT-eval/GRPO-reward/rejection-sampling judges don't silently fall
    # back to the env's generated default. Forward any that the operator has set; absent ones are
    # simply not passed (the env then uses its own default model).
    for key in (
        "HF_TOKEN",
        "PRIME_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "FLASH_JUDGE_MODEL",
    ):
        if os.environ.get(key):
            env[key] = os.environ[key]
    # Seed the worker's own HF_REPO env from the run's [train] hf_repo (adapter/checkpoint/
    # code storage + heartbeats). The worker reads HF_REPO from its own process env; that env
    # is now sourced from the spec, not the operator's HF_REPO.
    env["HF_REPO"] = spec.train.hf_repo
    # Opt-in network volume: point the whole HF cache at the persistent mount so
    # model weights survive across runs (the download becomes a one-time cost per
    # volume instead of per run).
    if getattr(spec.gpu, "network_volume", None):
        env["HF_HOME"] = "/runpod-volume/hf-cache"
    if spec.train.steps is not None:
        env["RL_STEPS"] = str(spec.train.steps)
    if spec.train.epochs is not None:
        env["SFT_EPOCHS"] = str(spec.train.epochs)
    # Forward the worker-side knobs the worker / vLLM / mid-run eval actually read. flash is fully
    # managed: there are no per-run env tuning knobs — the only per-run config is the spec's
    # structured fields, and the worker hardcodes the vLLM-util / quant / heartbeat defaults.
    for k in (
        "SFT_PER_DEVICE_BS",
        # RL_VLLM_SLEEP drives the alloc-conf choice above (CuMemAllocator vs expandable_segments).
        "RL_VLLM_SLEEP",
        "VLLM_USE_V1",
        # Attention-backend escape hatch: vllm's bundled flash-attn PTX can be newer
        # than the host driver's JIT (sm_120 + 12.8 drivers); TRITON_ATTN/FLASHINFER
        # sidestep it without restricting the host pool to CUDA-13 drivers.
        "VLLM_ATTENTION_BACKEND",
        # W&B credential that enables logging. The project + run name come from the spec's typed
        # [wandb] config, NOT env vars (see engine.worker.wandb_report_to / wandb_run_name).
        "WANDB_API_KEY",
        # Periodic mid-run GRPO eval cadence override (engine.worker reads it; >0 enables). The
        # held-out sample size comes from the run's [train] eval_examples, not an env var.
        "FLASH_EVAL_EVERY_STEPS",
        # Upload the worker console (which optimizations engaged) on SUCCESS too, not just on crash.
        # run_mode() in _train_body reads this from the `env` dict it builds (os.environ updated with
        # this forwarded input_data["env"] allowlist), NOT from its own process os.environ — so a
        # control-plane `FLASH_UPLOAD_CONSOLE=1` only reaches run_mode if it's forwarded here.
        # Without this it silently no-ops on success.
        "FLASH_UPLOAD_CONSOLE",
        # FLASH_* chalk kernel-selection flags: chalk is install-on-call (reads NO env vars), so
        # the WORKER decides which installers to run from these flags. install_chalk_kernels runs
        # INSIDE the worker subprocess and reads them from its own process env, so a control-plane
        # FLASH_* selection must be forwarded here or every chalk kernel silently no-ops on every
        # remote run. FLASH_CHALK_SPEC is the install spec install_chalk_kernels points operators
        # at (and is also consumed at submit time to add chalk to the worker's extra_pip).
        "FLASH_MLP_KERNEL",
        "FLASH_MLP_FP8",
        "FLASH_MLP_FP8_DOWN",
        "FLASH_FP8_BASE",
        "FLASH_FP8_BASE_ATTN",
        "FLASH_FP8_BASE_MLP",
        "FLASH_FP8_BASE_MIN_K",
        "FLASH_TRITON_LORA",
        "FLASH_EMBED_KERNEL",
        "FLASH_QKV_KERNEL",
        "FLASH_ROPE_KERNEL",
        # The chalk install spec itself — install_chalk_kernels warns pointing at it when a
        # FLASH_* flag is set but chalk is absent.
        "FLASH_CHALK_SPEC",
    ):
        # Forward when SET, even if empty: an explicit "" is a meaningful override.
        if os.environ.get(k) is not None:
            env[k] = os.environ[k]
    # Per-run worker_env overrides win over the global os.environ allowlist: this is what lets
    # ONE run differ (e.g. a per-run optimizer or LoRA-init A/B) while every other concurrent run
    # keeps the global default. Run-IDENTITY keys are control-plane-owned and excluded: the poller,
    # deploy, and artifact paths all key off spec.run_id / spec.train.hf_repo, so letting a
    # [worker_env] override RUN_ID/HF_REPO would make the worker upload under a different repo/prefix
    # and orphan the artifacts (the poller would never find DONE/metrics, deploy can't locate the
    # adapter). FLASH_ARM identifies the substrate (Vast rewrites it in its own bootstrap).
    _RESERVED_WORKER_ENV = {"RUN_ID", "HF_REPO", "FLASH_ARM"}
    for k, v in (getattr(spec, "worker_env", None) or {}).items():
        if str(k).upper() in _RESERVED_WORKER_ENV:
            continue  # control plane owns run identity; a per-run override would orphan artifacts
        env[str(k)] = str(v)
    return env


def submit_train(spec: JobSpec, seed: int, log=None) -> dict:
    """Provision a dedicated GPU via Flash, run training, return the metrics dict."""
    timeout_s = max(60, int(spec.gpu.max_wall_seconds))
    from flash.envs.registry import worker_hub_env_ids, worker_pip_for_env

    handler = get_train_endpoint(
        spec.gpu.type,
        execution_timeout_ms=timeout_s * 1000,
        name_suffix=_run_suffix(spec.run_id),
        disk_gb=spec.gpu.disk_gb,
        spec=spec,
    )
    payload = {
        "hf_repo": spec.train.hf_repo,
        "job_spec_json": spec.to_json(),
        "phase": spec.phase,
        "seed": int(seed),
        "env": build_worker_env(spec, seed),
        # extra_pip is installed by the worker for EVERY job (baked-image RunPod _train_body and
        # Vast bootstrap both pip-install it), so it's where the chalk spec must go to reach a
        # default run — see chalk_extra_pip().
        "extra_pip": (list(spec.environment.pip) or worker_pip_for_env(spec.environment.id))
        + chalk_extra_pip(spec),
        "hub_env_ids": worker_hub_env_ids(spec.environment.id, spec.environment.params),
    }
    if log is not None:
        print(
            f"submitting Flash job: gpu={spec.gpu.type} phase={spec.phase} "
            f"seed={seed} model={spec.model}",
            file=log,
            flush=True,
        )

    async def _call():
        res = handler(payload)
        if inspect.isawaitable(res):
            res = await res
        return res

    out = asyncio.run(_call())
    if not isinstance(out, dict):
        raise RuntimeError(f"flash job returned no metrics: {out!r}")
    return out
