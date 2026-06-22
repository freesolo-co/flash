"""Worker dependency stack + per-run env / chalk-kernel selection (leaf module).

The substrate-neutral training dependency list (``WORKER_DEPS``), the prebuilt worker
image, the per-run worker env builder, and the chalk-kernel install-selection helpers.
This is the leaf of the ``train`` package: it imports nothing else from the package and
defines the shared ``logger`` the other submodules import.
"""

from __future__ import annotations

import os

from flash._logging import get_logger
from flash.spec import JobSpec, gpus_per_node

# Literal name (NOT __name__) so the logger stays "flash.providers.runpod.train" after the
# module split into a package — callers/tests assert that exact name.
logger = get_logger("flash.providers.runpod.train")


# Worker stack: trl 1.6 (colocate default; adds the GRPO `tools=` / `rollout_func`
# multi-turn hooks used for verifiers ToolEnv / MultiTurnEnv training), vllm 0.19.1
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
    "huggingface_hub>=0.25",
    "accelerate>=1.4",
    # NB: the HF `kernels` Hub package is intentionally NOT pinned here — the versions
    # compatible with torch2.10 break transformers 5.6-5.10's hub_kernels integration at IMPORT
    # (LayerRepository now requires a version; transformers passes none -> ValueError on every
    # `import transformers`). FlashAttention via the Hub is therefore disabled; attention uses
    # SDPA (already a flash/efficient backend on Ampere/Ada) + the chalk fused kernels below,
    # which are the dominant LoRA speedup anyway. (FA via a pinned flash-attn wheel is a future
    # per-arch experiment, kept out of the default deps to avoid a fragile cold-start install.)
    "wandb>=0.17",
    # freesolo-chalk: flash's STANDALONE fused-kernel stack — it REPLACES Liger entirely. chalk
    # supplies its OWN RMSNorm + SwiGLU + fused-linear-CE (the FLCE is the big large-vocab
    # Qwen3.5 ~248k memory/throughput win that keeps the logits from materializing), PLUS the
    # kernels Liger never had (RoPE / LoRA-delta / embedding). Pure Triton/CUDA, JITs on every arch
    # incl. Blackwell. Baked in (as Liger was) so cold starts don't pip-install it. Keep this spec
    # in lockstep with DEFAULT_CHALK_SPEC below + the freesolo-chalk install in Dockerfile.worker.
    "freesolo-chalk>=0.3.0,<0.5.0",
    # Fused Triton kernels for Gated-DeltaNet (Qwen3.5/3.6 family): without this, transformers
    # falls back to a pure-PyTorch delta rule that is dramatically slower + memory-heavier (measured
    # A/B, Vast H100 SXM, Qwen3.5-0.8B LoRA: seq4096 4413->371 ms/step = 11.9x; seq8192 13.7x;
    # seq16384 14498->945 ms = 15.3x; forward loss matches to ~6e-4). Installed from git: the PyPI
    # ``flash-linear-attention`` wheel is a broken stub missing ``fla.modules``. PINNED to a specific
    # commit (not the moving default branch) so cold-start installs are reproducible and a breaking
    # upstream change can't silently land on a run; bump intentionally. Keep this SHA in lockstep
    # with Dockerfile.worker + perf.py's runtime reinstall.
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@f0e213dbd8b5fb90c3c7eca869ac1706d5377139",
    # fla's gated chunk_bwd is INCORRECT on Hopper (H100) with Triton>=3.4 (fla #640); its
    # ``tilelang`` backend is the correct path there, so we KEEP fla on every arch (the worker's
    # _ensure_fla_fastpath_on_hopper ensures tilelang is live on sm90 before any model import) rather
    # than dropping it to the slow pure-PyTorch delta. PINNED (like the fla SHA) so cold-start
    # installs / image rebuilds are reproducible. Keep in lockstep with Dockerfile.worker + perf.py.
    "tilelang==0.1.11",
    "apache-tvm-ffi==0.1.11",  # pin: 0.1.12 double-registers TVM-FFI -> `import tilelang` aborts
    # NB on causal_conv1d (#14, INVESTIGATED — deliberately NOT added here): the Qwen3.5/3.6 GDN
    # layer runs a short depthwise causal conv before the delta rule, and the `causal_conv1d` package
    # would fuse it (HF auto-detects via is_causal_conv1d_available). BUT: (a) it only gates the small
    # conv step — the DOMINANT GDN cost (chunk_gated_delta_rule) is gated INDEPENDENTLY on fla and
    # ALREADY runs on the fla kernel without it, so the win is bounded to the conv (modest); and
    # (b) causal-conv1d ships SDIST-ONLY on PyPI (no torch2.10/cu128 wheels) -> pip BUILDS FROM SOURCE
    # at cold-start, which FAILED REPRODUCIBLY on the Vast worker across two GPU classes (RTX 5090 +
    # A100 SXM) with check=True killing the run. Putting it in WORKER_DEPS would break every cold-start
    # boot-install run. If revisited, it must be BAKED into Dockerfile.worker (build once at image-build
    # time with the -devel toolkit) AND the published image rebuild must be VALIDATED to actually
    # compile it — it is not a safe per-run pip dep. See /tmp/flash_combine_results.md NEW-OPT TESTS.
    # NB: freesolo-chalk is baked into the base deps above (it's REQUIRED now — chalk runs every
    # fused kernel, standalone). The submit path (chalk_extra_pip) ALSO appends the resolved chalk
    # spec to the worker's `extra_pip` so an operator can OVERRIDE the version/source per-run via
    # FLASH_CHALK_SPEC (an exact version / git URL / wheel) — e.g. pin to the chalk standalone commit
    # before 0.2.0 lands on PyPI; a redundant install matching the baked version is harmless. The
    # worker pip-installs extra_pip for EVERY job (baked-image RunPod
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
# ever needed. Must be published to GHCR + made public before the paths that use it can pull it.
#   * Vast: ALWAYS used (jobs.builders pins the container to WORKER_IMAGE).
#   * RunPod baked-image submit (jobs.deploy_train_endpoint / build_function_input): the default —
#     a self-contained serverless worker whose rp_handler runs _train_body. FLASH_WORKER_IMAGE
#     overrides the tag (e.g. a hotfix); the boot-install fallback is only reachable if BOTH are
#     cleared (not a normal configuration).
#   * RunPod Flash live-endpoint (train.endpoints.get_train_endpoint): does NOT use this baked image
#     by default — RunPod Flash needs its serverless runtime baked in, which WORKER_IMAGE lacks, so
#     it boot-installs resolve_worker_deps() on Flash's default template instead. It uses an image
#     only when the operator sets FLASH_WORKER_IMAGE to a RunPod-serverless-compatible one.
# So FLASH_WORKER_IMAGE IS an operator override (consulted by every RunPod path).
# FLASH_TRAIN_WORKER_IMAGE overrides the baked-image TAG (e.g. to validate a candidate image like
# :cu128-mgpu without overwriting the production :cu128) — read at import, so set it in the
# control-plane env. .strip() so a whitespace-only FLASH_TRAIN_WORKER_IMAGE falls back to the
# default tag instead of becoming an invalid Docker ref that RunPod Flash / Vast would try (and
# fail) to provision with.
WORKER_IMAGE = (
    os.environ.get("FLASH_TRAIN_WORKER_IMAGE") or ""
).strip() or "ghcr.io/freesolo-co/flash-worker:cu128"


def resolve_worker_deps(friendly_gpu: str | None = None) -> list[str]:
    """The dependency list Flash installs on the GPU worker for this run.

    Precedence: FLASH_WORKER_DEPS (explicit list) > the pinned ``WORKER_DEPS``.

    fla is kept on ALL arches now (including Hopper): the worker's
    _ensure_fla_fastpath_on_hopper ensures fla's correct ``tilelang`` backend is live on sm90
    before any model import (fla #640's Triton>=3.4 miscompute is a tilelang-backend fix, not a
    reason to drop fla). This makes Hopper GDN training ~4-13x faster + ~2x less memory than the
    pure-PyTorch delta fallback.
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
    # fla is kept on ALL arches (incl. Hopper sm90). On Hopper the correctness fix is fla's
    # tilelang backend (baked into WORKER_DEPS + ensured by _ensure_fla_fastpath_on_hopper), NOT
    # dropping fla — keeping it gives the ~4-13x faster / ~2x lighter GDN training the pure-PyTorch
    # delta fallback can't. (friendly_gpu retained for signature/back-compat; no longer drops fla.)
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

    chalk's gap-fillers (RoPE/LoRA/embedding) are ON BY DEFAULT (engine.chalk_kernels), so this
    is True for a normal run. It is False only when every kernel that would otherwise be enabled is
    explicitly set to 0 — i.e. the three default-on gap-fillers (ROPE/TRITON_LORA/EMBED) disabled
    via ``FLASH_<K>=0`` (the default-off opt-in kernels — QKV/MLP/FP8 base — need no flag to stay
    off, so deselecting chalk does NOT require setting them to 0). Resolved against the EFFECTIVE worker env
    — the run's ``[worker_env]`` merged over ``os.environ`` so a per-run override is honored (see
    ``_effective_worker_env``). Delegates to ``chalk_kernels.is_chalk_enabled`` (the single source
    of truth for the flag/default logic) rather than re-parsing the kernel table here.
    """
    from flash.engine.chalk_kernels import is_chalk_enabled

    return is_chalk_enabled(_effective_worker_env(spec))


# Default chalk install spec when FLASH_CHALK_SPEC is unset. VERSION-PINNED (bounded range, like the
# rest of WORKER_DEPS) so a default run is reproducible and a breaking freesolo-chalk release can't
# silently land on production jobs — 0.3.x patches are allowed, 0.4 is not. 0.3.0 is the PER-ARCH
# line (chalk's own RMSNorm/SwiGLU/FLCE/RoPE/LoRA/embedding, replacing Liger, with arch-tuned
# autotune configs + FLCE chunk_mult dispatched on cuda capability — sm_80+ -> 8, sm_86 -> 4 — so
# each kernel WINS on speed/memory on every GPU arch). Keep in lockstep with the freesolo-chalk pin
# in WORKER_DEPS + Dockerfile.worker. Bump intentionally after validating a new line; an operator can
# pin exactly via FLASH_CHALK_SPEC=freesolo-chalk==X.Y.Z.
DEFAULT_CHALK_SPEC = "freesolo-chalk>=0.3.0,<0.5.0"


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
    when chalk is selected and ``FLASH_CHALK_SPEC`` is unset we add the version-pinned
    :data:`DEFAULT_CHALK_SPEC`. Set ``FLASH_CHALK_SPEC`` to override the source (an exact version, a
    git URL, or a wheel/path), or disable every kernel (``FLASH_<K>=0``) to skip the install.
    """
    if not _chalk_selected(spec):
        return []
    # PyPI default (version-pinned for reproducibility) — chalk is published, so a normal run
    # installs + applies it automatically. An explicit FLASH_CHALK_SPEC overrides the source.
    spec_str = _effective_worker_env(spec).get("FLASH_CHALK_SPEC", "").strip() or DEFAULT_CHALK_SPEC
    import shlex

    return [d for d in shlex.split(spec_str) if d.strip()]


DEFAULT_EXECUTION_TIMEOUT_MS = 6 * 3600 * 1000  # 6h RunPod worker execution cap


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
        # GPUs provisioned on the node (= one trainer card + train.inference_gpus). The worker reads
        # this (engine.disaggregated.detect_total_gpus) to compute the disaggregated rollout split
        # WITHOUT initializing a torch CUDA context first. 1 = single-GPU (colocate). Derived from
        # the rollout topology (gpus_per_node) — there is no [gpu].count field, so the old
        # getattr(spec.gpu, "count", 1) silently pinned every node to 1 GPU.
        "FLASH_GPU_COUNT": str(gpus_per_node(spec)),
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
    # Forward the worker-side knobs the worker / vLLM actually read. flash is fully
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
        # Disaggregated (multi-GPU async) rollout knobs (engine/worker.py + disaggregated.py):
        # SERVER_TIMEOUT raises the vllm-serve health-wait for a slow big-model boot (the 35B can
        # take >20 min); SERVER_UTIL sizes the dedicated rollout card's vLLM pool; DISAGG_PARALLEL
        # picks tp (default) vs dp (MoE-only); ENFORCE_EAGER skips CUDA-graph capture so a huge
        # model boots inside the timeout. Operators set these via the control-plane env (the
        # disaggregated topology is control-plane-owned, not per-run), so forward them or the worker
        # silently falls back to defaults.
        "RL_VLLM_SERVER_TIMEOUT",
        "RL_VLLM_SERVER_UTIL",
        "FLASH_DISAGG_PARALLEL",
        "FLASH_RL_VLLM_ENFORCE_EAGER",
        # W&B credential that enables logging. Project + run name come from the spec's typed
        # [wandb] config (NOT env vars); the run's entity is the API key's default account/team
        # (wandb_report_to does not pass entity=).
        "WANDB_API_KEY",
        # Upload the worker console (which optimizations engaged) on SUCCESS too, not just on crash.
        # run_mode() in _train_body reads this from the `env` dict it builds (os.environ updated with
        # this forwarded input_data["env"] allowlist), NOT from its own process os.environ — so a
        # control-plane `FLASH_UPLOAD_CONSOLE=1` only reaches run_mode if it's forwarded here.
        # Without this it silently no-ops on success.
        "FLASH_UPLOAD_CONSOLE",
        # FLASH_* chalk kernel-selection flags: chalk is install-on-call (reads NO env vars), so
        # the WORKER decides which kernels to enable from these flags. install_chalk_kernels runs
        # INSIDE the worker subprocess and reads them from its own process env, so a control-plane
        # FLASH_* selection must be forwarded here or every chalk kernel silently no-ops on every
        # remote run. These are exactly the per-kernel boolean flags in chalk_kernels._KERNELS
        # (one per apply_chalk_kernel_to_qwen35 keyword); FLASH_<K>=0/1 overrides the kernel's
        # default. FLASH_CHALK_SPEC is the install spec install_chalk_kernels points operators at
        # (and is also consumed at submit time to add chalk to the worker's extra_pip).
        # Default-on chalk standalone kernels (replace Liger's rms_norm/swiglu/FLCE):
        # forwarding these is what lets an operator/per-run FLASH_<K>=0 actually DISABLE one on
        # the worker — without them a control-plane override silently no-ops and the kernel
        # stays on regardless of the flag.
        "FLASH_RMSNORM_KERNEL",
        "FLASH_SWIGLU_KERNEL",
        "FLASH_FLCE_KERNEL",
        "FLASH_FP8_BASE",
        "FLASH_TRITON_LORA",
        "FLASH_EMBED_KERNEL",
        "FLASH_QKV_KERNEL",
        "FLASH_ROPE_KERNEL",
        # The chalk install spec itself — install_chalk_kernels warns pointing at it when a
        # FLASH_* flag is set but chalk is absent.
        "FLASH_CHALK_SPEC",
        # LoRA-init + rollout-knob A/B levers (read on the worker: make_lora / run_rl /
        # patch_trl_colocate_vllm_args). Forward a control-plane default; a per-run [worker_env]
        # still wins (merged below). FLASH_USE_DORA (weight-decomposed LoRA, serve/warm-start-safe),
        # FLASH_ROLLOUT_TUNE (raise TRL's hardcoded max_num_batched_tokens), FLASH_FP8_KV (fp8 KV
        # cache on sm89+, ~halves the rollout KV pool).
        "FLASH_USE_DORA",
        "FLASH_ROLLOUT_TUNE",
        "FLASH_FP8_KV",
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
    # FLASH_GPU_COUNT is the control-plane topology hint (= the sized/billed [gpu].count); the worker
    # falls back to it when nvidia-smi is unavailable and honors it over an over-exposed node, so a
    # [worker_env] override would let the worker compute a different disaggregated split than was
    # provisioned/billed (or fail as under-provisioned if it exceeds the visible GPUs).
    # The disaggregated ROLE / TOPOLOGY / RANK keys are control-plane- and launcher-owned too and
    # must NOT be forwardable per run (worker.py asserts FLASH_INFERENCE_GPUS is "reserved in
    # [worker_env]"):
    #   FLASH_INFERENCE_GPUS  — _env_inference_gpus() reads it from os.environ and lets it OVERRIDE
    #     the spec value, so a per-run override changes the effective rollout split away from the
    #     sized/billed topology (e.g. =0 forces colocate on a model provisioned for a split).
    #   FLASH_RL_TRAINER_ONLY — the role flag the launcher sets ONLY on its accelerate trainer
    #     children; forwarded on the parent it makes the paid launcher skip starting vllm-serve and
    #     try to connect as a trainer child to a server that never comes up (hang/fail).
    #   RANK / LOCAL_RANK     — accelerate-assigned distributed ranks; is_artifact_writer() keys off
    #     RANK=="0", so a forwarded RANK can suppress the ONLY process that writes the adapter/DONE.
    #   FLASH_DISAGG_PARALLEL — the rollout split (tp default vs dp MoE-only). The allocator's
    #     required_vram_gb() sizes the inference card from the SUBMITTER os.environ: tp divides the
    #     bf16 server footprint by inference_gpus, dp does NOT (each card is a full replica). A
    #     per-run override applied here (after sizing) would let a TP-sized provision start DP
    #     replicas that each need the whole server footprint -> the paid worker OOMs. Topology-owned
    #     like FLASH_INFERENCE_GPUS, so it must be set via the control-plane env (it IS forwarded from
    #     os.environ above), never per-run.
    _RESERVED_WORKER_ENV = {
        "RUN_ID",
        "HF_REPO",
        "FLASH_ARM",
        "FLASH_GPU_COUNT",
        "FLASH_INFERENCE_GPUS",
        "FLASH_RL_TRAINER_ONLY",
        "FLASH_DISAGG_PARALLEL",
        "RANK",
        "LOCAL_RANK",
    }
    for k, v in (getattr(spec, "worker_env", None) or {}).items():
        if str(k).upper() in _RESERVED_WORKER_ENV:
            continue  # control plane owns run identity; a per-run override would orphan artifacts
        env[str(k)] = str(v)
    return env
