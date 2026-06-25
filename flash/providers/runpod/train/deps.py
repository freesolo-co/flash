"""Worker dependency stack + per-run env / chalk-kernel selection (leaf module).

The substrate-neutral training dependency list (``WORKER_DEPS``), the prebuilt worker
image, the per-run worker env builder, and the chalk-kernel install-selection helpers.
This is the leaf of the ``train`` package: it imports nothing else from the package and
defines the shared ``logger`` the other submodules import.
"""

from __future__ import annotations

import os

from flash._logging import get_logger
from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
from flash.providers.base import get_gpu_info
from flash.spec import JobSpec

# Literal name (NOT __name__) so the logger stays "flash.providers.runpod.train" after the
# module split into a package — callers/tests assert that exact name.
logger = get_logger("flash.providers.runpod.train")


# Worker stack: trl 1.6 (colocate default; adds the GRPO `tools=` / `rollout_func`
# multi-turn hooks used for Freesolo EnvironmentMultiTurn training), vllm 0.19.1
# (Qwen3.5/3.6 archs, native RL APIs, transformers-5
# compatible metadata), transformers 5.x (qwen3_5/qwen3_5_moe model types),
# bitsandbytes (the 8-bit paged AdamW optimizer state — LoRA+ coexists with it).
# trl 1.6 requires transformers>=4.56,
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
    # >=0.2.49: first version exposing Environment.sft_completion + datasets.target_messages,
    # which the worker's SFT/multi-turn path now calls (flash #162 multi-turn SFT).
    "freesolo>=0.2.49",
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
    # (use_liger_loss) — the big large-vocab (Qwen3.5 ~248k) memory/throughput win.
    "wandb>=0.17",
    "liger-kernel>=0.5",
    # Fused Triton kernels for Gated-DeltaNet (Qwen3.5/3.6 family): without this, transformers
    # falls back to a pure-PyTorch delta rule that is dramatically slower + memory-heavier (measured
    # A/B, H100 SXM, Qwen3.5 hidden-2560 LoRA: seq4096 435->105 ms/step & 9.9->6.1 GB = 4.2x/1.6x;
    # seq16384 3106->247 ms & 32->17 GB = 12.6x/1.9x; forward loss matches to 1.8e-4). Installed
    # from git: the PyPI ``flash-linear-attention`` wheel is a broken stub missing ``fla.modules``.
    # PINNED to a specific commit (not the moving default branch) so cold-start installs are
    # reproducible and a breaking upstream change can't silently land on a run; bump intentionally
    # after validating. Keep this SHA in lockstep with Dockerfile.worker's fla install.
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@f0e213dbd8b5fb90c3c7eca869ac1706d5377139",
    # fla's gated chunk_bwd is INCORRECT on Hopper (H100) with Triton>=3.4 (fla #640); its
    # ``tilelang`` backend is the correct path there, so we KEEP fla on every arch (the worker's
    # _ensure_fla_fastpath_on_hopper ensures tilelang is live on sm90 before any model import).
    # PINNED (like the fla SHA) so cold-start installs / image rebuilds are reproducible and a
    # breaking upstream tilelang can't silently land on the Hopper correctness path. Keep this
    # version in lockstep with Dockerfile.worker + perf.py's runtime reinstall.
    "tilelang==0.1.11",
    "apache-tvm-ffi==0.1.11",  # pin: 0.1.12 double-registers TVM-FFI -> `import tilelang` aborts
    # NB: ``causal_conv1d`` (the conv kernel that, with fla's cu_seqlens, lets GatedDeltaNet hybrids
    # PACK boundary-correctly — engine.worker.packing) is NOT pip-listed here: it's a CUDA-extension
    # build, so Dockerfile.worker compiles it best-effort with TORCH_CUDA_ARCH_LIST set (a plain pip
    # entry would try to compile at the no-GPU image-build step for the wrong/native arch). If it's
    # absent the GDN packing path stays off (gdn_packing_available) and Qwen3.5/3.6 train unpacked.
    # NB: freesolo-chalk (custom Triton/CUDA kernels that complement Liger) is NOT in this base dep
    # list, but its gap-fillers are default-on (flash/engine/chalk_kernels.py), so the submit path
    # (chalk_extra_pip) appends the version-pinned ``freesolo-chalk`` (DEFAULT_CHALK_SPEC) from PyPI
    # to the worker's `extra_pip` for EVERY job by default — installed + applied automatically, like
    # Liger. Override the install SOURCE with FLASH_CHALK_SPEC (an exact version / git URL / wheel).
    # The worker pip-installs extra_pip for EVERY job through the baked-image RunPod _train_body.
]
# NOTE on download speed: Flash's runtime already ships hf_transfer and exports
# HF_HUB_ENABLE_HF_TRANSFER=1 on workers (measured: Qwen3-4B's ~8 GB pulled in 6.3 s,
# NIC-saturated — bench/results/phase6). Adding hf_transfer here is redundant; don't.
WORKER_SYSTEM_DEPS = ["build-essential"]  # Triton/Inductor need a C compiler

# The prebuilt worker image (full training stack baked in; built by Dockerfile.worker /
# .github/workflows/worker-image.yml). PUBLIC under the org namespace, so no registry login is
# ever needed. Must be published to GHCR + made public before the paths that use it can pull it.
#   * RunPod baked-image submit (jobs.deploy_train_endpoint / build_function_input): the default —
#     a self-contained serverless worker whose rp_handler runs _train_body. FLASH_WORKER_IMAGE
#     overrides the tag (e.g. a hotfix); the boot-install fallback is only reachable if BOTH are
#     cleared (not a normal configuration).
#   * RunPod Flash live-endpoint (train.endpoints.get_train_endpoint): does NOT use this baked image
#     by default — RunPod Flash needs its serverless runtime baked in, which WORKER_IMAGE lacks, so
#     it boot-installs resolve_worker_deps() on Flash's default template instead. It uses an image
#     only when the operator sets FLASH_WORKER_IMAGE to a RunPod-serverless-compatible one.
# So FLASH_WORKER_IMAGE IS an operator override (consulted by every RunPod path).
WORKER_IMAGE = "ghcr.io/freesolo-co/flash-worker:cu128"
WORKER_IMAGE_TEMPLATE_ENV = "FLASH_WORKER_IMAGE_TEMPLATE"
WORKER_IMAGE_PER_SM_ENV = "FLASH_WORKER_IMAGE_PER_SM"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _append_tag_suffix(image: str, suffix: str) -> str:
    slash = image.rfind("/")
    colon = image.rfind(":")
    if colon > slash:
        return f"{image[:colon]}:{image[colon + 1:]}-{suffix}"
    return f"{image}-{suffix}"


def worker_image_for_gpu(friendly_gpu: str | None, *, allow_default: bool = True) -> str | None:
    """Return the RunPod worker image for a GPU class.

    ``FLASH_WORKER_IMAGE`` remains the absolute override. Per-SM warmed images are opt-in through
    either ``FLASH_WORKER_IMAGE_TEMPLATE`` (for custom tag layouts) or ``FLASH_WORKER_IMAGE_PER_SM``
    (which appends ``-smXX`` to ``WORKER_IMAGE``'s tag). Without either opt-in, the current base
    image is returned for durable jobs and ``None`` is returned for live endpoints that request no
    default image.
    """
    override = os.environ.get("FLASH_WORKER_IMAGE", "").strip()
    if override:
        return override
    # per-sm / template images are durable RunPod queue-worker images (CMD runs rp_handler.py), NOT
    # RunPod Flash live-function images, so they must never leak into the live path. only the
    # absolute FLASH_WORKER_IMAGE override (handled above) is treated as live-compatible.
    if friendly_gpu and allow_default:
        info = get_gpu_info(friendly_gpu)
        template = os.environ.get(WORKER_IMAGE_TEMPLATE_ENV, "").strip()
        if template:
            return template.format(
                base_image=WORKER_IMAGE,
                gpu=info.name,
                gpu_short=info.short,
                sm=info.sm,
                sm_num=info.sm.removeprefix("sm"),
            )
        if _truthy(os.environ.get(WORKER_IMAGE_PER_SM_ENV)):
            return _append_tag_suffix(WORKER_IMAGE, info.sm)
    return WORKER_IMAGE if allow_default else None


def resolve_worker_deps() -> list[str]:
    """The dependency list Flash installs on the GPU worker for this run.

    The pinned ``WORKER_DEPS`` is authoritative — flash is fully managed, no per-run override.

    fla is kept on ALL arches (including Hopper): the worker's _ensure_fla_fastpath_on_hopper
    ensures fla's correct ``tilelang`` backend is live on sm90 before any model import (fla #640's
    Triton>=3.4 miscompute is a tilelang-backend fix, not a reason to drop fla). This makes Hopper
    GDN training ~4-13x faster + ~2x less memory than the pure-PyTorch delta fallback.
    """
    return list(WORKER_DEPS)


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


# Default chalk install spec when FLASH_CHALK_SPEC is unset. VERSION-PINNED (bounded range, like the
# rest of WORKER_DEPS) so a default run is reproducible and a breaking freesolo-chalk release can't
# silently land on production jobs — 0.1.x patches are allowed, 0.2 is not. Bump intentionally after
# validating a new line; an operator can pin exactly via FLASH_CHALK_SPEC=freesolo-chalk==X.Y.Z.
DEFAULT_CHALK_SPEC = "freesolo-chalk>=0.1.0,<0.2.0"


def chalk_extra_pip(spec=None) -> list[str]:
    """Chalk pip spec(s) to ADD to the worker's ``extra_pip`` when a chalk kernel is selected.

    This is the install hook that runs for DEFAULT remote jobs: the baked-image RunPod path
    (``_train_body`` -> ``pip install *extra_pip``) consumes the payload's ``extra_pip``
    regardless of ``WORKER_IMAGE`` — unlike ``resolve_worker_deps``, which the durable
    ``build_function_input`` baked-image path skips.

    Selection (and the ``FLASH_CHALK_SPEC`` lookup) is resolved against the EFFECTIVE worker env —
    the run's ``[worker_env]`` merged over ``os.environ`` — so it matches exactly what the worker
    process will see (``build_worker_env``) and a per-run ``[worker_env]`` opt-in installs chalk.

    Chalk's gap-fillers are default-on, so chalk is selected for a normal run even with no FLASH_*
    flags set. freesolo-chalk is published on PyPI, so it auto-installs by DEFAULT (just like Liger):
    when chalk is selected and ``FLASH_CHALK_SPEC`` is unset we add the version-pinned
    :data:`DEFAULT_CHALK_SPEC`. Set ``FLASH_CHALK_SPEC`` to override the source (an exact version, a
    git URL, or a wheel/path), or disable every kernel (``FLASH_<K>=0``) to skip the install.
    """
    # PyPI default (version-pinned for reproducibility) — chalk is published, so a normal run
    # installs + applies it automatically. An explicit FLASH_CHALK_SPEC overrides the source.
    spec_str = _effective_worker_env(spec).get("FLASH_CHALK_SPEC", "").strip() or DEFAULT_CHALK_SPEC
    import shlex

    return [d for d in shlex.split(spec_str) if d.strip()]


DEFAULT_EXECUTION_TIMEOUT_MS = 6 * 3600 * 1000  # 6h RunPod worker execution cap


_RUNTIME_SECRET_KEYS = DEFAULT_RUNTIME_SECRET_KEYS

# Optimization-toggle env knobs REMOVED in PR #175 (deterministic behavior). flash is fully
# managed: these used to let a run override alloc conf / attention backend / torch.compile / the
# FlashAttention selection / chalk-kernel selection / the whole worker dep stack. They were dropped
# from forwarding to make every run behave identically — but the per-run ``spec.worker_env`` block
# is still merged into the worker env below, so without filtering, a recipe could RE-INJECT any of
# them and silently re-enable the non-deterministic (and sometimes crash-prone) behavior:
#   * PYTORCH_(CUDA_)ALLOC_CONF — overrides the sleep-SAFE alloc conf build_worker_env computes for
#     RL; expandable_segments:True crashes GRPO vLLM sleep mode (CuMemAllocator assert at init).
#   * RL_VLLM_SLEEP / FLASH_ALLOC_AUTO — the worker resolves sleep + alloc-conf deterministically.
#   * VLLM_ATTENTION_BACKEND / VLLM_FLASH_ATTN_VERSION / FLASH_DISABLE_FA2 / FLASH_DISABLE_FA3 /
#     TORCHDYNAMO_DISABLE — attention / compile escape hatches; FA is used whenever importable.
#   * FLASH_*_KERNEL / FLASH_FP8_BASE / FLASH_TRITON_LORA — chalk kernel SELECTION is now fixed in
#     engine.chalk_kernels._KERNELS (reads no env); a per-run flag is dead but still stripped.
#   * FLASH_WORKER_DEPS / FLASH_WORKER_EXTRA_DEPS — the pinned WORKER_DEPS stack is authoritative.
# Matched case-INSENSITIVELY (env keys are upper-cased on merge); FLASH_CHALK_SPEC is deliberately
# NOT here — it is a still-supported install-SOURCE override (see build_worker_env's forward list).
_REMOVED_OPTIMIZATION_ENV = frozenset(
    {
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "RL_VLLM_SLEEP",
        "FLASH_ALLOC_AUTO",
        "TORCHDYNAMO_DISABLE",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_FLASH_ATTN_VERSION",
        "FLASH_DISABLE_FA2",
        "FLASH_DISABLE_FA3",
        "FLASH_ROPE_KERNEL",
        "FLASH_QKV_KERNEL",
        "FLASH_MLP_KERNEL",
        "FLASH_EMBED_KERNEL",
        "FLASH_FP8_BASE",
        "FLASH_TRITON_LORA",
        "FLASH_WORKER_DEPS",
        "FLASH_WORKER_EXTRA_DEPS",
    }
)


def build_worker_env(
    spec: JobSpec,
    seed: int,
    runtime_secrets: dict[str, str] | None = None,
) -> dict:
    """Per-run env passed to the worker (platform creds + recipe overrides).

    Provider and artifact credentials still come from the control-plane process environment.
    User runtime secrets (W&B plus [environment].secrets) are injected from ``runtime_secrets``
    below so the control plane never stores user-owned secret values in the spec.
    """
    # CUDA allocator conf. Colocate (TRL trainer + vLLM on one GPU) fragments over a long run, so
    # expandable_segments (which reclaims fragmentation) is the right default — EXCEPT under GRPO
    # vLLM sleep mode, whose CuMemAllocator memory pool is incompatible with expandable_segments
    # (vLLM asserts and the run crashes at engine init). RL is fully managed — no sleep/alloc knob:
    # we set the sleep-SAFE non-expandable conf for RL here (the launcher can't yet know the model's
    # resolved sleep decision), and the worker upgrades RL runs to expandable_segments when it
    # resolves sleep OFF for the model/context (engine.worker.finalize_alloc_conf_for_sleep, same
    # deterministic _memory_mode gate run_rl uses). SFT has no vLLM engine, so expandable is safe.
    # torch >= 2.10 renamed the env to PYTORCH_ALLOC_CONF — set BOTH names for either stack.
    _is_rl = str(getattr(spec, "algorithm", "")).lower() not in ("sft",)
    _alloc_conf = (
        "garbage_collection_threshold:0.8,max_split_size_mb:256"
        if _is_rl
        else "expandable_segments:True"
    )
    env: dict[str, str] = {
        "RUN_ID": spec.run_id,
        # Compute substrate, read back by engine.worker for the RunMetrics record.
        "FLASH_ARM": "runpod",
        "BENCH_HF_MODEL": spec.model,
        "PYTORCH_CUDA_ALLOC_CONF": _alloc_conf,
        "PYTORCH_ALLOC_CONF": _alloc_conf,
    }
    # HF artifact creds + managed environment hub creds + optional reward-judge creds: a Freesolo
    # environment whose reward calls an LLM judge (e.g. OpenRouter gpt-oss-120b) needs the API key ON THE WORKER,
    # where the reward runs. FLASH_JUDGE_MODEL is the judge model id the optimizer-authored env
    # reads (agents/common/prompt.py) to pick the JudgeRubric client model; forward the operator's
    # control-plane override so SFT-eval/GRPO-reward/rejection-sampling judges don't silently fall
    # back to the env's generated default. Forward any that the operator has set; absent ones are
    # simply not passed (the env then uses its own default model).
    for key in (
        "HF_TOKEN",
        "GITHUB_TOKEN",
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
    if spec.train.steps is not None:
        env["RL_STEPS"] = str(spec.train.steps)
    if spec.train.epochs is not None:
        env["SFT_EPOCHS"] = str(spec.train.epochs)
    # Forward the worker-side knobs the worker / vLLM actually read. flash is fully
    # managed: there are no per-run env tuning knobs — the only per-run config is the spec's
    # structured fields, and the worker hardcodes the vLLM-util / quant / heartbeat defaults.
    for k in (
        "SFT_PER_DEVICE_BS",
        "VLLM_USE_V1",
        # Upload the worker console (which optimizations engaged) on SUCCESS too, not just on crash.
        # run_mode() in _train_body reads this from the `env` dict it builds (os.environ updated with
        # this forwarded input_data["env"] allowlist), NOT from its own process os.environ — so a
        # control-plane `FLASH_UPLOAD_CONSOLE=1` only reaches run_mode if it's forwarded here.
        "FLASH_UPLOAD_CONSOLE",
        # The chalk install SOURCE (an exact version / git URL / wheel). Kernel SELECTION is fixed
        # in engine.chalk_kernels (no env flags); this only points install_chalk_kernels at a
        # specific chalk build, and is also consumed at submit time to add chalk to extra_pip.
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
    # adapter). FLASH_ARM identifies the substrate.
    _RESERVED_WORKER_ENV = {"RUN_ID", "HF_REPO", "FLASH_ARM"}
    for k, v in (getattr(spec, "worker_env", None) or {}).items():
        ku = str(k).upper()
        if ku in _RESERVED_WORKER_ENV:
            continue  # control plane owns run identity; a per-run override would orphan artifacts
        if ku in _REMOVED_OPTIMIZATION_ENV:
            # A removed optimization toggle (PR #175). flash is deterministic + fully managed: drop
            # it so a recipe can't re-inject e.g. an unsafe PYTORCH_ALLOC_CONF that crashes GRPO
            # sleep mode, or a stale chalk/FA selection flag. See _REMOVED_OPTIMIZATION_ENV.
            logger.warning(
                "ignoring removed optimization toggle %s in [worker_env] (flash is fully "
                "managed; behavior is deterministic)",
                k,
            )
            continue
        env[str(k)] = str(v)
    allowed_runtime_secrets = set(_RUNTIME_SECRET_KEYS) | set(spec.environment.secrets)
    for k, v in (runtime_secrets or {}).items():
        if k in allowed_runtime_secrets and v:
            env[k] = str(v)
    return env
