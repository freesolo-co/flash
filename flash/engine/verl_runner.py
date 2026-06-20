"""verl GRPO + LoRA runner (FLASH_FRAMEWORK=verl).

Runs verl in a SIDECAR venv on the provisioned box, isolated from the baked TRL/vLLM
stack (baked = torch 2.10 + vllm 0.19.1 + transformers 5; verl needs vllm<=0.12 +
transformers<5 — hard conflict, so a clean venv). Benchmarks verl's one-step-off async
overlap WITH LoRA (the path TRL's AsyncGRPOTrainer can't do — it's full-FT-only).

Single GPU (inference_gpus=0): verl.trainer.main_ppo colocate (hybrid_engine, no overlap)
  -> apples-to-apples vs our TRL colocate s/step.
>=2 GPU (inference_gpus>0): verl.experimental.one_step_off_policy.main_ppo, hybrid_engine=False
  -> the real generation<->training OVERLAP (gen(t+1) overlaps train(t)).

Writes /tmp/metrics.json in the shape the flash pipeline reads
({wall_seconds, notes:{steps,...}, reward_history}).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time

VENV = "/opt/verl-venv"
VPY = f"{VENV}/bin/python"
VPIP = f"{VENV}/bin/pip"
VERL_DIR = "/opt/verl"
WORKDIR = "/tmp/verl"
# The python verl actually runs with. OLD stack -> the sidecar venv (VPY). BAKED stack -> the SYSTEM
# python (sys.executable), reusing the baked torch/vllm/transformers/torchvision (all matching), since
# a --system-site-packages venv + uv reinstalls torch and breaks the baked torchvision. Set by _install.
RUN_PY = VPY


def _hb(stage: str, **kw):
    try:
        from flash.engine.worker import heartbeat

        heartbeat(stage, **kw)
    except Exception:
        print(f"[verl][hb] {stage} {kw}", flush=True)


def _spec():
    from flash.engine import worker as W

    return W.JOB_SPEC


def _run(cmd, cwd=None, env=None, check=True, capture=False):
    print(f"[verl] $ {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=isinstance(cmd, str),
        text=True,
        capture_output=capture,
    )
    if capture:
        if r.stdout:
            print(r.stdout[-4000:], flush=True)
        if r.stderr:
            print(r.stderr[-4000:], flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={r.returncode}): {cmd if isinstance(cmd, str) else ' '.join(map(str, cmd))}"
        )
    return r


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _write(path: str, data: str) -> None:
    with open(path, "w") as f:
        f.write(data)


def _ensure_uv() -> str:
    """Return a path to `uv` (creates venvs without python3-venv/ensurepip, installs fast).
    Prefer one already on the image; else bootstrap via system pip, else the standalone installer."""
    import shutil

    def _find():
        cands = [
            "uv",
            os.path.expanduser("~/.local/bin/uv"),
            "/root/.local/bin/uv",
            "/usr/local/bin/uv",
            "/opt/uv/uv",
        ]
        for c in cands:
            p = shutil.which(c) if "/" not in c else (c if os.path.exists(c) else None)
            if p:
                return p
        return None

    p = _find()
    if p:
        return p
    for pipcmd in (
        [sys.executable, "-m", "pip", "install", "-q", "uv"],
        ["pip", "install", "-q", "uv"],
    ):
        if subprocess.run(pipcmd, capture_output=True, text=True).returncode == 0:
            break
    p = _find()
    if p:
        return p
    subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)
    p = _find()
    if p:
        return p
    raise RuntimeError("could not obtain uv for the verl sidecar venv")


def _install():
    """Install the verl stack and return the python executable verl should run with.

    OLD stack (default): a fresh uv venv with PUBLIC-PyPI vllm 0.11 + transformers 4.57 + a MATCHING
      torchvision -> MiniCPM/Qwen2.5/Qwen3-dense. Cannot load the qwen3_5 arch.
    BAKED stack (FLASH_VERL_USE_BAKED_STACK=1): for Qwen3.5/3.6. The baked image's vllm 0.19.1 +
      transformers 5 + torch + torchvision are INTERNAL builds (not on PyPI) and MUTUALLY MATCHED.
      Install verl + its non-baked deps straight into the SYSTEM python (no venv) so all of them are
      reused as-is. (A --system-site-packages venv + uv reinstalls torch and breaks the baked
      torchvision -> `operator torchvision::nms does not exist`.)
    """
    global RUN_PY
    # BAKED is the DEFAULT: the host-matched vllm0.19.1/tf5 boots reliably and loads every model we
    # benchmark (MiniCPM/Qwen2.5/Qwen3-dense AND Qwen3.5/3.6). The OLD uv-venv stack (set
    # FLASH_VERL_USE_BAKED_STACK=0) installs fresh vllm0.11 wheels -> fragile torch/CUDA resolution
    # (vllm==0.11.0 install now fails outright) -> retired as the default.
    use_baked = os.environ.get("FLASH_VERL_USE_BAKED_STACK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    py = sys.executable if use_baked else VPY
    RUN_PY = py
    # Idempotency: if verl already imports under `py`, skip. Guard on the interpreter existing — the
    # OLD-stack venv python won't exist on a fresh box (FileNotFoundError otherwise).
    if os.path.exists(py):
        chk = subprocess.run(
            [
                py,
                "-c",
                "import verl, vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__)",
            ],
            text=True,
            capture_output=True,
        )
        if chk.returncode == 0:
            print(
                f"[verl] stack ready ({py}): vllm/torch/transformers = {chk.stdout.strip()}",
                flush=True,
            )
            return py

    verl_ref = os.environ.get("FLASH_VERL_REF", "").strip()
    vllm_pin = os.environ.get("FLASH_VERL_VLLM", "vllm==0.11.0").strip()
    tf_pin = os.environ.get("FLASH_VERL_TRANSFORMERS", "transformers==4.57.0").strip()
    fa_pin = os.environ.get("FLASH_VERL_FLASHATTN", "flash-attn==2.8.1").strip()
    tv_pin = os.environ.get(
        "FLASH_VERL_TORCHVISION", "torchvision==0.23.0"
    ).strip()  # matches torch 2.8 (vllm 0.11)
    uv = _ensure_uv()

    # The baked system python is PEP-668 "externally managed" -> uv/pip refuse it without this flag.
    bsp = ["--break-system-packages"] if use_baked else []

    def upip(*pkgs, check=True):
        return _run([uv, "pip", "install", "-p", py, *bsp, *pkgs], check=check, capture=True)

    _hb("verl_install", step="venv", baked=use_baked)
    if not use_baked:
        # uv venv: no python3-venv/ensurepip needed (baked image lacks it).
        _run([uv, "venv", VENV, "--python", sys.executable])
        upip("pip", "setuptools", "wheel")
    if not os.path.exists(os.path.join(VERL_DIR, "setup.py")) and not os.path.exists(
        os.path.join(VERL_DIR, "pyproject.toml")
    ):
        _hb("verl_install", step="clone")
        clone = ["git", "clone", "--depth", "1"]
        if verl_ref:
            clone += ["--branch", verl_ref]
        clone += ["https://github.com/verl-project/verl", VERL_DIR]
        _run(clone, check=not verl_ref)
        if not os.path.exists(os.path.join(VERL_DIR, "pyproject.toml")):
            _run(["rm", "-rf", VERL_DIR], check=False)
            _run(["git", "clone", "https://github.com/verl-project/verl", VERL_DIR])
            if verl_ref:
                _run(["git", "checkout", verl_ref], cwd=VERL_DIR)
    if not use_baked:
        _hb("verl_install", step="vllm_transformers", vllm=vllm_pin, transformers=tf_pin)
        upip(vllm_pin, tf_pin)  # vllm pins torch; resolve it first
        # torchvision UNPINNED in a second pass so uv matches it to whatever torch vllm resolved
        # (a hard pin like ==0.23.0 conflicts/flakes host-to-host); tv must match torch or nms op missing.
        _tvr = upip("torchvision", check=False)
        if _tvr.returncode != 0:
            upip(tv_pin, check=False)
        _hb("verl_install", step="flash_attn")
        upip(fa_pin, "--no-build-isolation", check=False)  # wheel; tolerate fail
    else:
        _hb(
            "verl_install",
            step="baked_stack_reused",
            note="vllm/transformers/torch/torchvision from baked system python",
        )
    _hb("verl_install", step="deps")
    # --no-deps on the ones likely to drag torch so the BAKED torch/torchvision are never disturbed;
    # install their needed extras explicitly. On the OLD venv this is harmless (torch already pinned).
    upip(
        "ray[default]>=2.41.0",
        "tensordict>=0.8.0,<=0.10.0",
        "hydra-core",
        "omegaconf",
        "datasets",
        "pyarrow",
        "pandas",
        "codetiming",
        "dill",
        "pylatexenc",
        "wandb",
        "math_verify",
        "accelerate",
        "peft>=0.19",
        "torchdata",
        "torchmetrics",
    )
    # one_step_off async overlap transfers weights trainer->rollout via the NCCL checkpoint engine,
    # which only REGISTERS if cupy + pyzmq import (else verl dies "Checkpoint engine nccl not
    # registered"). Install them for the overlap path (heavy cupy wheel -> skip for plain colocate).
    if os.environ.get("FLASH_VERL_OVERLAP", "").strip().lower() in ("1", "true", "yes"):
        _hb("verl_install", step="nccl_ckpt_engine_deps")
        upip("cupy-cuda12x", "pyzmq", check=False)
    _hb("verl_install", step="verl")
    _run([uv, "pip", "install", "-p", py, *bsp, "--no-deps", "-e", "."], cwd=VERL_DIR)
    _install_chalk(uv, py)
    chk = _run(
        [
            py,
            "-c",
            "import verl, vllm, torch, torchvision, transformers; print('verl-ok', vllm.__version__, torch.__version__, torchvision.__version__, transformers.__version__)",
        ],
        capture=True,
        check=False,
    )
    print(f"[verl] install verified ({py}): {chk.stdout.strip()}", flush=True)
    return py


def _install_chalk(uv, py):
    """pip-install our custom chalk kernels (freesolo-chalk: hand-written Triton/CUDA Qwen3.5/3.6
    kernels — fused MLP, RoPE, fused embedding, fused LoRA-delta, FP8 base) into ``py``. The kernels
    are APPLIED to the verl actor model by _patch_verl_chalk() (a verl-file patch that calls
    apply_chalk_kernel_to_qwen35 in _build_module) — a sitecustomize get_peft_model wrap was tried
    first but never fired in verl's Ray trainer actor. Gated by FLASH_VERL_CHALK (default on);
    FLASH_CHALK_SPEC overrides the pip spec (e.g. a local wheel or git ref)."""
    if os.environ.get("FLASH_VERL_CHALK", "1").strip().lower() not in ("1", "true", "yes"):
        return
    spec = os.environ.get("FLASH_CHALK_SPEC", "").strip() or "freesolo-chalk"
    _hb("verl_install", step="chalk", spec=spec)
    _run([uv, "pip", "install", "-p", py, spec], check=False, capture=True)
    print(f"[verl][chalk] installed {spec} (applied via _patch_verl_chalk)", flush=True)


def _build_dataset(model_path: str, n_rows: int, prompt_tokens: int):
    """Synthetic GRPO dataset: ~prompt_tokens-long chat prompts + a simple rule reward.
    Throughput (gen n*resp + train) dominates s/step, so a simple reward is fine for the bench."""
    os.makedirs(WORKDIR, exist_ok=True)
    import pandas as pd  # baked image has pandas; if not, the sidecar does

    # Build prompts COMFORTABLY UNDER max_prompt_length (~prompt_tokens). English ≈ 1.3 tok/word, so
    # target ~0.55*prompt_tokens/1.3 words to stay well within the limit (avoids all-rows-filtered ->
    # num_samples=0). truncation=right + filter_overlong_prompts=False in the config also protect this.
    filler = "Reason step by step about the following problem and give the final answer. " * 60
    words = filler.split()
    approx = max(8, int(prompt_tokens * 0.55 / 1.3))
    content = " ".join((words * (approx // len(words) + 1))[:approx])
    rows = [
        {
            "data_source": "flash_bench",
            "prompt": [{"role": "user", "content": f"[{i}] {content}"}],
            "ability": "bench",
            "reward_model": {"style": "rule", "ground_truth": "x"},
            "extra_info": {"index": i, "split": "train"},
        }
        for i in range(n_rows)
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(f"{WORKDIR}/train.parquet")
    df.head(max(8, n_rows // 8)).to_parquet(f"{WORKDIR}/val.parquet")
    print(
        f"[verl] dataset: {n_rows} rows, ~{prompt_tokens} prompt tok -> {WORKDIR}/train.parquet",
        flush=True,
    )


def _write_reward():
    """A trivial length-aware reward (verl custom_reward_function contract). Throughput bench
    doesn't depend on the reward signal; this just exercises the scoring path."""
    src = (
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    n = len(solution_str or '')\n"
        "    return 1.0 if n > 0 else 0.0\n"
    )
    with open(f"{WORKDIR}/verl_reward.py", "w") as f:
        f.write(src)


def _build_cmd(
    model_path, split, *, group_size, prompt_len, resp_len, lora_rank, lora_alpha, steps, train_bs
):
    overlap = split.infer_gpus > 0
    if overlap:
        entry = [
            "-m",
            "verl.experimental.one_step_off_policy.main_ppo",
            "--config-path=config",
            "--config-name=one_step_off_ppo_trainer",
        ]
    else:
        entry = ["-m", "verl.trainer.main_ppo"]
    # Trainer strategy. fsdp2 (default) uses DTensor — incompatible with bnb 8-bit optim / 4-bit
    # QLoRA. fsdp (FSDP1) uses FlatParameter (no DTensor) and on 1 trainer GPU is NO_SHARD (= a
    # single un-sharded model on one GPU, what we want), which UNBLOCKS 8-bit + QLoRA.
    _strategy = os.environ.get("FLASH_VERL_STRATEGY", "fsdp2").strip() or "fsdp2"
    ov = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={WORKDIR}/train.parquet",
        f"data.val_files={WORKDIR}/val.parquet",
        f"data.train_batch_size={train_bs}",
        f"data.max_prompt_length={prompt_len}",
        f"data.max_response_length={resp_len}",
        "data.filter_overlong_prompts=False",  # never drop rows (would zero the dataset -> num_samples=0)
        "data.truncation=right",  # truncate instead of erroring on length
        "data.dataloader_num_workers=0",  # no dataloader worker threads (Vast pids cap is tight)
        f"actor_rollout_ref.model.path={model_path}",
        # MiniCPM / custom-arch models load via remote code -> without this the vLLM EngineCore dies
        # at startup ("Failed core proc(s): {}"). Harmless for natively-supported archs.
        "actor_rollout_ref.model.trust_remote_code=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.model.lora_rank={lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={lora_alpha}",
        # all-linear LoRA-wraps EVERY nn.Linear incl. Qwen3.5's VISION tower -> the one_step_off
        # bucketed weight-transfer then sends vision 'qkv.base_layer.weight' which vLLM's qwen3_vl
        # load_weights rejects (KeyError). Set FLASH_VERL_TARGET_MODULES to a language-only list
        # (e.g. "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]") for Qwen3.5 async.
        f"actor_rollout_ref.model.target_modules={os.environ.get('FLASH_VERL_TARGET_MODULES', 'all-linear').strip()}",
        "actor_rollout_ref.actor.optim.lr=1e-5",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={max(8, train_bs // 2)}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        # Trainer strategy (fsdp2=DTensor default; fsdp=FSDP1/FlatParameter for 8-bit+QLoRA, no DTensor).
        f"actor_rollout_ref.actor.strategy={_strategy}",
        f"actor_rollout_ref.ref.strategy={_strategy}",
        # verl requires explicit micro-batch sizes (or use_dynamic_bsz). Set small per-GPU micro-batches.
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={group_size}",
        "actor_rollout_ref.rollout.temperature=1.0",
        f"actor_rollout_ref.rollout.response_length={resp_len}",
        # CAP vLLM's max_model_len to the ACTUAL prompt+response budget. Without this the rollout
        # vLLM defaults to the model's full context (Qwen3.5 = 262144) and sizes its KV cache for a
        # 256K-token request -> "KV cache needed (3.04 GiB) > available" -> EngineCore dies before
        # step 0. This was THE async-rollout blocker (host-independent; verified across 4 GPU classes).
        f"actor_rollout_ref.rollout.max_model_len={prompt_len + resp_len}",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.enforce_eager=True",  # skip CUDA-graph capture (lighter/faster vLLM init)
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=True",
        # Rollout parallelism: DEFAULT TP=1 so verl auto-runs dp=infer_gpus INDEPENDENT vLLM replicas
        # (data-parallel), NOT a tensor-sharded engine. TP=infer_gpus all-reduces every layer/token —
        # pure comm waste on small models (TP=2 on 0.8B made 4-GPU SLOWER than 2-GPU). DP scales
        # rollout throughput with zero per-token cross-GPU comm. Big models that don't fit one card
        # set FLASH_VERL_ROLLOUT_TP=2+ to shard. (dense DP is fully supported by verl/modern vLLM.)
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={int(os.environ.get('FLASH_VERL_ROLLOUT_TP', '1'))}",
        # Two reward paths exist. The STANDARD reward manager reads top-level custom_reward_function.*;
        # the async-server AgentLoop/RewardLoop (what colocate uses, since rollout boots a vLLMHttpServer)
        # reads reward.custom_reward_function.* and otherwise routes by data_source -> "Reward function
        # is not implemented for data_source='flash_bench'". Set BOTH so whichever path runs is wired.
        f"custom_reward_function.path={WORKDIR}/verl_reward.py",
        "custom_reward_function.name=compute_score",
        f"reward.custom_reward_function.path={WORKDIR}/verl_reward.py",
        "reward.custom_reward_function.name=compute_score",
        "trainer.logger=[console]",
        "trainer.val_before_train=False",
        "trainer.nnodes=1",
        f"trainer.total_training_steps={steps}",
        "trainer.save_freq=-1",
        "trainer.test_freq=-1",
        "trainer.project_name=flash_verl",
    ]
    if overlap:
        ov += [
            "actor_rollout_ref.hybrid_engine=False",
            f"critic.strategy={_strategy}",
            # verl FSDP2 loads the base model fp32 by default (FSDPEngineConfig.model_dtype="fp32") ->
            # LoRA trainable params stay fp32 (grad_dtype=fp32) but the bf16 autocast backward emits
            # bf16 grads -> "assign a gradient with dtype BFloat16 to a tensor with grad_dtype Float"
            # (verl #3470/#2969). Load base+LoRA in bf16 so grad_dtype==grad dtype==bf16. (colocate's
            # known-good LoRA path sets this; one_step_off's config never exercised LoRA -> inherits fp32.)
            "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
            f"trainer.n_gpus_per_node={split.train_gpus}",
            "rollout.nnodes=1",
            f"rollout.n_gpus_per_node={split.infer_gpus}",
            # Rollout vLLM executor. DEFAULT = mp (MultiprocExecutor) — the SAME path colocate uses,
            # which WORKS on Vast hosts that allow the pidfd_getfd syscall (CUDA IPC); pidfd is
            # HOST-dependent seccomp, so retry async across hosts to land on a permissive one. The
            # Ray executor (FLASH_VERL_ROLLOUT_EXECUTOR=ray) dodges pidfd but CONFLICTS with verl's
            # one_step_off placement groups ("Current node has no GPU available"), so it's opt-in only.
        ]
        if os.environ.get("FLASH_VERL_ROLLOUT_EXECUTOR", "mp").strip().lower() == "ray":
            ov += [
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.distributed_executor_backend=ray",
            ]
    else:
        ov += [f"trainer.n_gpus_per_node={split.train_gpus or 1}"]
    # ---- Parity with the colocate (TRL) path: chalk kernels (+ Liger) + token-based dynamic batch.
    # chalk is installed by _install_chalk() and applied by _patch_verl_chalk(); Liger rides along via
    # chalk's apply(liger="auto"). (8-bit optimizer and QLoRA were removed — both are incompatible
    # with verl's FSDP2/DTensor and the 8-bit is useless for LoRA.) Default ON; override via env.
    _chalk_on = os.environ.get("FLASH_VERL_CHALK", "1").strip().lower() in ("1", "true", "yes")
    if (
        os.environ.get("FLASH_VERL_USE_LIGER", "1").strip().lower() in ("1", "true", "yes")
        and not _chalk_on
    ):
        # verl applies Liger via _apply_liger_kernel_to_instance in the FSDP engine (fused RMSNorm/
        # SwiGLU; fused_linear_cross_entropy auto-disabled — conflicts with verl's forward patching).
        # When chalk is ON we DON'T set this: chalk's apply_chalk_kernel_to_qwen35(liger="auto") owns
        # Liger (it forces Liger's RoPE off so chalk's RoPE installs) — setting both double-applies.
        ov.append("actor_rollout_ref.model.use_liger=True")
    # 8-bit AdamW (bitsandbytes). Crashes under fsdp2 (DTensor: "optimizer_update_8bit_blockwise got
    # mixed torch.Tensor and DTensor"), so it's gated and only usable with strategy=fsdp (FSDP1,
    # FlatParameter — no DTensor). Opt-in via FLASH_VERL_OPTIM_8BIT; pair with FLASH_VERL_STRATEGY=fsdp.
    if os.environ.get("FLASH_VERL_OPTIM_8BIT", "").strip().lower() in ("1", "true", "yes"):
        ov += [
            "actor_rollout_ref.actor.optim.optimizer_impl=bitsandbytes.optim",
            "actor_rollout_ref.actor.optim.optimizer=AdamW8bit",
        ]
    if os.environ.get("FLASH_VERL_QLORA", "").strip().lower() in ("1", "true", "yes"):
        # QLoRA: serve the rollout's base in 4-bit too (vLLM bitsandbytes) to match the 4-bit trainer
        # base — so a big model's rollout also fits the smaller/cheaper GPU. verl _apply_quantization
        # passes this straight to vLLM.
        ov += [
            "actor_rollout_ref.rollout.load_format=bitsandbytes",
            "actor_rollout_ref.rollout.quantization=bitsandbytes",
        ]
    # NOTE: token-based dynamic batching (use_dynamic_bsz) was REMOVED — measured SLOWER, not faster:
    # 4B 1:1 went 383.4 (off) -> 540.3 (on, +41%). With the fixed-size benchmark prompts the larger
    # token-budget micro-batches overlap worse with generation, so plain fixed micro-batches win here.
    # Escape hatch: space-separated extra hydra overrides via env (test verl config fixes without a
    # code change), e.g. FLASH_VERL_EXTRA_OVERRIDES="actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16".
    _extra = os.environ.get("FLASH_VERL_EXTRA_OVERRIDES", "").strip()
    if _extra:
        ov += _extra.split()
    return [RUN_PY, *entry, *ov]


def _patch_vllm_qwen35_moe_sync():
    """Make vLLM's qwen3_5 FUSED-EXPERT runtime weight load tolerant during verl one_step_off sync.

    Qwen3.6-35B-A3B is MoE + `*ForConditionalGeneration` (the LM nests under
    ``language_model.model.``). verl's one_step_off checkpoint engine streams the updated policy to
    the rollout vLLM via ``load_weights``; for the fused experts vLLM's
    ``Qwen3_5Model.load_fused_expert_weights`` does a bare ``params_dict[name]`` — and when the name
    doesn't resolve in a given (TP/EP) rank's ``params_dict`` it raises ``KeyError
    'layers.0.mlp.experts.w13_weight'``, which kills the rollout EngineCore at the FIRST weight
    transfer (step 0) and hangs/fails the run. The non-fused branch right below it already SKIPS a
    name that isn't in ``params_dict`` (it's tolerant); the fused branch isn't. This patches the
    installed qwen3_5.py to make the fused branch equally tolerant: try common prefix variants, and
    if still absent SKIP (return False) instead of a fatal KeyError. Idempotent + anchor-guarded
    (no-op on vllm version drift). Set FLASH_VERL_MOE_DEBUG=1 to log the actual param keys."""
    try:
        out = subprocess.run(
            [RUN_PY, "-c", "import vllm.model_executor.models.qwen3_5 as m; print(m.__file__)"],
            text=True,
            capture_output=True,
            timeout=120,
        )
        path = (out.stdout.strip().splitlines() or [""])[-1].strip()
        if not path or not os.path.exists(path):
            print(f"[verl] MoE-sync patch: qwen3_5.py not found ({out.stderr[-200:]})", flush=True)
            return
        src = _read(path)
        if "FLASH-MOE-PATCH" in src:
            print("[verl] MoE-sync patch: already applied", flush=True)
            return
        anchor = "        num_experts: int,\n    ) -> bool:\n        param = params_dict[name]\n"
        if anchor not in src:
            print("[verl] MoE-sync patch: anchor not found (vllm drift) — skipping", flush=True)
            return
        repl = (
            "        num_experts: int,\n    ) -> bool:\n"
            "        # FLASH-MOE-PATCH: tolerate prefix mismatch / absent fused-expert param during\n"
            "        # verl one_step_off runtime weight sync (Qwen *ForConditionalGeneration nests the\n"
            "        # LM under language_model.model.; a TP/EP rank may not own a given fused param).\n"
            "        # Resolve prefix variants; if still absent SKIP like the non-fused branch instead\n"
            "        # of a fatal KeyError that kills the rollout EngineCore at the first weight sync.\n"
            "        if name not in params_dict:\n"
            "            # Candidate names: the fused-expert param often lives under .base_layer.\n"
            "            # (PEFT LoRA wraps the FusedMoE -> vLLM registers experts.base_layer.w13_weight),\n"
            "            # and Qwen *ForConditionalGeneration nests the LM under language_model.model.\n"
            "            _cands = [name]\n"
            "            for _suf in ('w13_weight', 'w2_weight'):\n"
            "                if 'experts.' + _suf in name:\n"
            "                    _cands.append(name.replace('experts.' + _suf, 'experts.base_layer.' + _suf))\n"
            "            _alt = None\n"
            "            for _c in _cands:\n"
            "                for _p in ('', 'language_model.model.', 'language_model.', 'model.'):\n"
            "                    if _p + _c in params_dict:\n"
            "                        _alt = _p + _c; break\n"
            "                    if _p and _c.startswith(_p) and _c[len(_p):] in params_dict:\n"
            "                        _alt = _c[len(_p):]; break\n"
            "                if _alt is not None:\n"
            "                    break\n"
            "            if _alt is None:\n"
            "                import os as _os\n"
            "                if _os.environ.get('FLASH_VERL_MOE_DEBUG'):\n"
            "                    _ek = [k for k in params_dict if 'expert' in k.lower()][:6]\n"
            "                    _mk = [k for k in params_dict if '.mlp.' in k][:6]\n"
            "                    print('[flash-moe-patch] skip ' + repr(name) + '; expert keys=' + repr(_ek) + '; mlp keys=' + repr(_mk), flush=True)\n"
            "                return False\n"
            "            name = _alt\n"
            "        param = params_dict[name]\n"
        )
        _write(path, src.replace(anchor, repl, 1))
        # compile-check the patched file; revert on syntax error
        chk = subprocess.run(
            [RUN_PY, "-c", f"import ast; ast.parse(open({path!r}).read())"],
            text=True,
            capture_output=True,
            timeout=60,
        )
        if chk.returncode != 0:
            _write(path, src)
            print(f"[verl] MoE-sync patch: reverted (syntax) {chk.stderr[-200:]}", flush=True)
            return
        print(f"[verl] MoE-sync patch: applied to {path}", flush=True)
    except Exception as e:
        print(f"[verl] MoE-sync patch skipped: {e}", flush=True)


def _patch_verl_chalk():
    """Apply our chalk kernels (+ Liger) on the verl actor by patching verl's FSDP _build_module to
    call ``apply_chalk_kernel_to_qwen35(module, liger="auto")`` right after the model loads (where
    verl applies Liger). This is RELIABLE across Ray trainer actors — the earlier sitecustomize/
    get_peft_model wrap never fired in the actor (console showed no [chalk] lines, use_liger:False).

    chalk's kernels are CLASS-LEVEL installers (fused MLP/RoPE/embedding patch the transformers
    Qwen3.5 classes; fused LoRA-delta patches the PEFT LoRA Linear class), so applying here — BEFORE
    verl's later get_peft_model in _build_lora_module — still installs fused_lora_delta for the LoRA
    layers built afterward. liger="auto" applies Liger too (RoPE forced off so chalk's RoPE installs);
    fused_linear_cross_entropy=False matches verl's requirement. Gated by FLASH_VERL_CHALK (default
    on), idempotent, compile-checked + reverted on syntax error. No-op off-GPU / non-Qwen3.5."""
    if os.environ.get("FLASH_VERL_CHALK", "1").strip().lower() not in ("1", "true", "yes"):
        return
    try:
        out = subprocess.run(
            [
                RUN_PY,
                "-c",
                "import verl.workers.engine.fsdp.transformer_impl as m; print(m.__file__)",
            ],
            text=True,
            capture_output=True,
            timeout=120,
        )
        path = (out.stdout.strip().splitlines() or [""])[-1].strip()
        if not path or not os.path.exists(path):
            print(
                f"[verl] chalk patch: transformer_impl not found ({out.stderr[-200:]})", flush=True
            )
            return
        src = _read(path)
        if "FLASH-CHALK" in src:
            print("[verl] chalk patch: already applied", flush=True)
            return
        # Insert right after verl's Liger block, before fused_kernel_options handling.
        anchor = "            fused_kernel_options = self.model_config.fused_kernel_options\n"
        if anchor not in src:
            print("[verl] chalk patch: anchor not found (verl drift) — skipping", flush=True)
            return
        inject = (
            "            try:  # FLASH-CHALK\n"
            "                import os as _os\n"
            "                if _os.environ.get('FLASH_VERL_CHALK', '1').strip().lower() in ('1', 'true', 'yes'):\n"
            "                    from chalk.transformers import apply_chalk_kernel_to_qwen35\n"
            "                    _rep = apply_chalk_kernel_to_qwen35(module, liger='auto', fused_linear_cross_entropy=False)\n"
            "                    print('[chalk] applied to verl actor: ' + repr(_rep), flush=True)\n"
            "            except Exception as _e:\n"
            "                print('[chalk] apply skipped: ' + repr(_e), flush=True)\n"
        )
        _write(path, src.replace(anchor, inject + anchor, 1))
        chk = subprocess.run(
            [RUN_PY, "-c", f"import ast; ast.parse(open({path!r}).read())"],
            text=True,
            capture_output=True,
            timeout=60,
        )
        if chk.returncode != 0:
            _write(path, src)
            print(f"[verl] chalk patch: reverted (syntax) {chk.stderr[-200:]}", flush=True)
            return
        print(f"[verl] chalk patch: applied apply_chalk_kernel_to_qwen35 to {path}", flush=True)
    except Exception as e:
        print(f"[verl] chalk patch skipped: {e}", flush=True)


def _patch_verl_qlora():
    """Load the frozen base in 4-bit NF4 (QLoRA) on the verl trainer — the COST lever: a 4-bit base
    (~1/4 the VRAM) lets a big model train + serve on a SMALLER/cheaper GPU (e.g. 35B ~70GB bf16 ->
    ~18GB). verl has no native 4-bit support, so patch its FSDP from_pretrained to pass a
    BitsAndBytesConfig and skip verl's `module.to(torch_dtype)` (bitsandbytes forbids casting a 4-bit
    model). REQUIRES strategy=fsdp (FSDP1/FlatParameter): fsdp2's fully_shard can't wrap Params4bit
    ("only Tensors of floating point dtype can require gradients"). Gated by FLASH_VERL_QLORA,
    idempotent, compile-checked + reverted on syntax error."""
    try:
        out = subprocess.run(
            [
                RUN_PY,
                "-c",
                "import verl.workers.engine.fsdp.transformer_impl as m; print(m.__file__)",
            ],
            text=True,
            capture_output=True,
            timeout=120,
        )
        path = (out.stdout.strip().splitlines() or [""])[-1].strip()
        if not path or not os.path.exists(path):
            print(
                f"[verl] QLoRA patch: transformer_impl not found ({out.stderr[-200:]})", flush=True
            )
            return
        src = _read(path)
        if "FLASH-QLORA" in src:
            print("[verl] QLoRA patch: already applied", flush=True)
            return
        anchor = (
            "                    config=self.model_config.hf_config,\n"
            "                    trust_remote_code=self.model_config.trust_remote_code,\n"
            "                )\n"
        )
        if anchor not in src:
            print("[verl] QLoRA patch: anchor not found (verl drift) — skipping", flush=True)
            return
        repl = (
            "                    config=self.model_config.hf_config,\n"
            "                    quantization_config=__import__('transformers').BitsAndBytesConfig(  # FLASH-QLORA\n"
            "                        load_in_4bit=True, bnb_4bit_quant_type='nf4',\n"
            "                        bnb_4bit_compute_dtype=__import__('torch').bfloat16,\n"
            "                        bnb_4bit_use_double_quant=True),\n"
            "                    trust_remote_code=self.model_config.trust_remote_code,\n"
            "                )\n"
        )
        src = src.replace(anchor, repl, 1)
        cast_anchor = "            # some parameters may not in torch_dtype\n            module.to(torch_dtype)\n"
        if cast_anchor in src:
            src = src.replace(
                cast_anchor,
                "            # some parameters may not in torch_dtype\n"
                "            if not (getattr(module, 'is_loaded_in_4bit', False) or getattr(module, 'is_quantized', False)):  # FLASH-QLORA\n"
                "                module.to(torch_dtype)\n",
                1,
            )
        else:
            print(
                "[verl] QLoRA patch: .to(dtype) cast anchor not found (may still crash)", flush=True
            )
        _write(path, src)
        chk = subprocess.run(
            [RUN_PY, "-c", f"import ast; ast.parse(open({path!r}).read())"],
            text=True,
            capture_output=True,
            timeout=60,
        )
        if chk.returncode != 0:
            _write(path, src)
            print(f"[verl] QLoRA patch: reverted (syntax) {chk.stderr[-200:]}", flush=True)
            return
        print(f"[verl] QLoRA patch: applied 4-bit NF4 base load to {path}", flush=True)
    except Exception as e:
        print(f"[verl] QLoRA patch skipped: {e}", flush=True)


def run():
    from flash.engine import worker as W
    from flash.engine.disaggregated import detect_total_gpus
    from flash.engine.rollout_bench import select_rollout_split

    t0 = time.time()
    _hb("rl_start")
    spec = _spec()
    tr = spec.train
    model_id = spec.model
    group_size = int(getattr(tr, "group_size", 8) or 8)
    max_len = int(getattr(tr, "max_length", 2048) or 2048)
    resp_len = int(getattr(tr, "max_tokens", 1024) or 1024)
    prompt_len = max(256, max_len - resp_len)
    steps = int(getattr(tr, "steps", 12) or 12)
    lora_rank = int(getattr(tr, "lora_rank", 0) or 32)
    lora_alpha = int(getattr(tr, "lora_alpha", 0) or (2 * lora_rank))
    inf = int(getattr(tr, "inference_gpus", 0) or 0)
    inf = int(os.environ.get("FLASH_INFERENCE_GPUS", inf))
    train_bs = 32

    # Tell _install whether to pull the NCCL-checkpoint-engine deps (cupy/pyzmq) for the overlap path.
    os.environ["FLASH_VERL_OVERLAP"] = "1" if inf > 0 else "0"
    _hb("verl_install", note="sidecar venv + verl stack (slow cold start)")
    _install()
    # MoE rollout weight-sync tolerance (Qwen3.6-35B-A3B one_step_off) — patch the installed
    # qwen3_5.py so a fused-expert name that doesn't resolve in a rank's params_dict is skipped,
    # not a fatal KeyError. Only relevant to the overlap path, harmless elsewhere; idempotent.
    if inf > 0:
        _patch_vllm_qwen35_moe_sync()
    # Apply our chalk kernels (+ Liger) on the verl actor via a verl-file patch — reliable across
    # Ray workers (the sitecustomize approach didn't fire in the trainer actor).
    _patch_verl_chalk()
    # QLoRA (4-bit NF4 base) — the COST lever (fit a big model on a smaller GPU). Needs FSDP1, so the
    # submit pairs FLASH_VERL_QLORA=1 with FLASH_VERL_STRATEGY=fsdp.
    if os.environ.get("FLASH_VERL_QLORA", "").strip().lower() in ("1", "true", "yes"):
        _patch_verl_qlora()

    _hb("verl_prefetch")
    W.prefetch_model(model_id)
    # local HF snapshot dir
    model_path = model_id
    try:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_id)
    except Exception as e:
        print(f"[verl] snapshot_download fallback to id ({e})", flush=True)

    total = detect_total_gpus()
    split = (
        select_rollout_split(total, inf)
        if inf > 0
        else type("S", (), {"train_gpus": total or 1, "infer_gpus": 0})()
    )
    print(
        f"[verl] total_gpus={total} inference_gpus={inf} -> train={getattr(split, 'train_gpus', '?')} infer={getattr(split, 'infer_gpus', '?')} "
        f"({'one-step-off OVERLAP' if inf > 0 else 'colocate (no overlap)'})",
        flush=True,
    )

    _build_dataset(model_path, n_rows=max(64, train_bs * 4), prompt_tokens=prompt_len)
    _write_reward()

    cmd = _build_cmd(
        model_path,
        split,
        group_size=group_size,
        prompt_len=prompt_len,
        resp_len=resp_len,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        steps=steps,
        train_bs=train_bs,
    )
    env = dict(os.environ)
    env["VLLM_USE_V1"] = env.get("VLLM_USE_V1", "1")
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"  # full trace incl. the vLLM EngineCore root cause
    env["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = env.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "1800")
    # verl/ray must SEE ALL GPUs (one_step_off splits them into trainer+rollout pools itself). The
    # worker may have pre-pinned CUDA_VISIBLE_DEVICES to a subset (the TRL disaggregated split) ->
    # ray then reports "Total available GPUs 0". Expose every GPU to verl.
    if total and total > 0:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(total))
    # vLLM's CuMemAllocator (sleep-mode memory pool — verl colocate uses it to offload base weights
    # while the actor steps) HARD-ASSERTS expandable_segments is OFF. The baked image ships
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -> the vLLM rollout EngineCore dies at startup
    # with "Expandable segments are not compatible with memory pool" (Failed core proc(s): {}).
    # Strip ONLY the expandable_segments token (keep any other alloc knobs) for the verl subprocess.
    _alloc = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    _kept = [p for p in _alloc.split(",") if p.strip() and "expandable_segments" not in p]
    if _kept:
        env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(_kept)
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTHONPATH", None)  # keep the sidecar clean of the baked stack
    # Cheap/shared Vast containers ship a LOW soft process+fd limit. Ray's CoreWorker spawns a
    # thread pool at init; when pthread_create hits RLIMIT_NPROC it THROWS -> C++ std::terminate ->
    # SIGABRT ("Fatal Python error: Aborted", init_once.cold) that retries can't clear. Raise the
    # soft limits to the hard cap (inherited by the verl child) so Ray/vLLM can spawn their threads.
    try:
        import resource

        for _lim, _name in ((resource.RLIMIT_NPROC, "NPROC"), (resource.RLIMIT_NOFILE, "NOFILE")):
            _soft, _hard = resource.getrlimit(_lim)
            if _hard == resource.RLIM_INFINITY or _soft < _hard:
                resource.setrlimit(_lim, (_hard, _hard))
                print(f"[verl] raised RLIMIT_{_name} {_soft}->{_hard}", flush=True)
    except Exception as _e:
        print(f"[verl] rlimit bump skipped: {_e}", flush=True)
    # Cap math-lib thread pools so the process doesn't fan out hundreds of threads on a Vast
    # container with a low pids cap ("RuntimeError: can't start new thread" hit even on an A100 at
    # the heavier matched config); verl/vLLM still parallelize on the GPU.
    for _tv in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(_tv, "4")
    # HuggingFace tokenizers (Rust/Rayon) + any other Rayon crate: when Ray forks worker processes
    # Rayon tries to spin up a global thread pool in EACH Ray actor. On RunPod/Vast the per-process
    # thread cap is low and the attempt panics with EAGAIN (ThreadPoolBuildError / "Resource
    # temporarily unavailable"). Limit Rayon to 1 thread (the calling thread — no new threads
    # spawned). TOKENIZERS_PARALLELISM=false prevents HF tokenizers from even trying to init Rayon.
    env.setdefault("RAYON_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # glibc spawns one heap arena PER thread by default (each reserves virtual memory) -> compounds
    # the thread/memory pressure on a constrained container. Cap arenas to shrink the footprint.
    env.setdefault("MALLOC_ARENA_MAX", "2")
    # Multi-GPU (one_step_off async) NCCL: many Vast nodes are plain-PCIe with NO CUDA P2P between
    # cards -> "peer access is not supported between these two devices" / NCCL_ERROR_UNHANDLED.
    # Force NCCL onto the SHM/host path so cross-GPU weight transfer works without P2P.
    if total and total > 1:
        # Vast containers block CUDA peer access (cudaDeviceEnablePeerAccess -> "peer access is not
        # supported between these two devices"), even on NVLink cards — it's the container sandbox,
        # not the hardware. The cupy/ray-collective nccl checkpoint engine (trainer<->rollout weight
        # transfer) hits this in NCCL's P2P AND SHM transports. Force NCCL onto the NET (socket)
        # transport, which needs no peer access. (Set here AND best to also pass via [worker_env] so
        # it reaches every Ray actor's env.)
        # Container blocks CUDA peer access, BUT disabling SHM forces the slow TCP NET/socket transport
        # -> the trainer DDP all-reduce crawls (2:2 update_actor 32s->138s = 4.3x, made 4-GPU SLOWER
        # than 2-GPU). Keep SHM but force the LEGACY host-staged mmap path (NCCL_CUMEM_HOST_ENABLE=0):
        # host-bounce SHM needs NO peer access yet is far faster than sockets. Only P2P stays disabled.
        env.setdefault("NCCL_P2P_DISABLE", "1")
        env.setdefault("NCCL_SHM_DISABLE", "0")
        env.setdefault("NCCL_CUMEM_ENABLE", "0")
        env.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")
        # vLLM's Ray executor (rollout) places a worker bundle that requests CPU:10 each; verl's own
        # Ray actors already hold most of the box's CPUs -> "No available node types can fulfill
        # resource request {'CPU': 10.0}". Tell Ray the node has plenty of CPUs (scheduling bookkeeping
        # only; the box isn't actually CPU-bound) so the placement group is satisfiable.
        env.setdefault("RAY_OVERRIDE_RESOURCES", json.dumps({"CPU": 64}))
        # one_step_off async init (Ray placement groups + Ray-executor vLLM workers + weight transfer)
        # is much slower than colocate -> give the first step a longer watchdog budget.
        os.environ.setdefault("FLASH_VERL_FIRST_STEP_TIMEOUT", "2400")

    _hb("rl_train_start", setup_seconds=time.time() - t0)
    log_path = "/tmp/verl_console.txt"
    print(f"[verl] launching: {' '.join(map(str, cmd))}", flush=True)
    # verl's console logger (trainer.logger=[console]) prints ONE line per training step via
    # concat_dict_to_str: "step:N - perf/time_per_step:12.3 - actor/.. - .." (numeric keys only).
    step_line_re = re.compile(r"\bstep:(\d+)\b")
    tps_re = re.compile(
        r"(?:perf/time_per_step|timing_s/step|time_per_step)\s*[:=]\s*([0-9.eE+-]+)"
    )
    step_times: list[float] = []  # per-step wall times (s) when verl logs the perf key
    seen_steps: set[int] = (
        set()
    )  # step numbers seen — robust counter even if the timing key differs
    train_t0 = [time.time()]  # reset at each launch attempt; for the wall/steps fallback
    stop = threading.Event()

    def _beat():
        while not stop.is_set():
            _hb("verl_running", steps_seen=len(seen_steps), timed=len(step_times))
            stop.wait(60)

    th = threading.Thread(target=_beat, daemon=True)
    th.start()

    def _launch_once():
        """Run verl once, streaming combined output to log_path. Returns (rc, aborted); appends
        any timed steps to step_times. A watchdog kills a STALLED run (no first step within
        FIRST_STEP_TIMEOUT, or no new step within STALL_TIMEOUT) so a hang (seen on the async
        one_step_off path: stuck at step1 for 60-90 min) is terminated and its console is still
        uploaded for diagnosis instead of burning GPU forever."""
        aborted = False
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=VERL_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            first_to = float(
                os.environ.get("FLASH_VERL_FIRST_STEP_TIMEOUT", "1500")
            )  # 25 min cold budget
            stall_to = float(
                os.environ.get("FLASH_VERL_STALL_TIMEOUT", "900")
            )  # 15 min between steps
            progress = [time.time(), 0]  # [last-progress wallclock, step count at that time]
            wd_stop = threading.Event()

            def _watchdog():
                while not wd_stop.wait(30):
                    n = len(seen_steps)
                    now = time.time()
                    if n > progress[1]:
                        progress[0], progress[1] = now, n
                    budget = first_to if n == 0 else stall_to
                    if now - progress[0] > budget:
                        print(
                            f"[verl] WATCHDOG: no step progress in {int(budget)}s (steps_seen={n}) "
                            f"-> killing verl (hang)",
                            flush=True,
                        )
                        _hb("verl_watchdog_kill", steps_seen=n)
                        with contextlib.suppress(Exception):
                            proc.kill()
                        return

            wt = threading.Thread(target=_watchdog, daemon=True)
            wt.start()
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                print(line, end="", flush=True)
                if "SIGABRT" in line or "Fatal Python error: Aborted" in line:
                    aborted = True
                sm = step_line_re.search(line)
                if sm:
                    snum = int(sm.group(1))
                    if snum not in seen_steps:
                        seen_steps.add(snum)
                        tm = tps_re.search(line)
                        if tm:
                            with contextlib.suppress(ValueError):
                                step_times.append(float(tm.group(1)))
                        _hb("rl_step", step=snum, timed=len(step_times))
            rc = proc.wait()
            wd_stop.set()
            return rc, aborted

    # The vLLM async server intermittently SIGABRTs at startup (init_once.cold, "Fatal Python error:
    # Aborted") BEFORE any step — a transient native-init race. Retry the whole verl launch a couple
    # times when that happens; never re-run once we've already timed real steps.
    max_attempts = 3
    rc = 1
    for attempt in range(max_attempts):
        step_times.clear()
        seen_steps.clear()
        train_t0[0] = time.time()
        rc, aborted = _launch_once()
        if rc == 0 or seen_steps:
            break
        if aborted and attempt < max_attempts - 1:
            print(
                f"[verl] vLLM SIGABRT at startup (attempt {attempt + 1}/{max_attempts}) -> retrying",
                flush=True,
            )
            _hb("verl_retry_abort", attempt=attempt + 1)
            continue
        break
    stop.set()
    train_wall = time.time() - train_t0[0]

    # s/step preference: (1) verl's logged per-step times (drop step 0 warmup, average the rest);
    # (2) fallback to train-phase wall / #steps when the perf key wasn't logged but steps DID run
    # (excludes install/model-load/vLLM-boot since train_t0 is set right before the launch).
    useful = step_times[1:] if len(step_times) > 1 else step_times
    s_per_step = (sum(useful) / len(useful)) if useful else None
    s_per_step_source = "verl_perf_key" if s_per_step is not None else None
    if s_per_step is None and len(seen_steps) >= 2:
        # avg over steps after the first (warmup) — approximate but real (timed steps actually ran)
        s_per_step = train_wall / len(seen_steps)
        s_per_step_source = "train_wall_div_steps"
    wall = time.time() - t0
    print(
        f"[verl] DONE rc={rc} steps_seen={len(seen_steps)} steps_timed={len(step_times)} "
        f"s/step={s_per_step} ({s_per_step_source})",
        flush=True,
    )
    if rc != 0 and not seen_steps:
        tail = ""
        root = ""
        try:
            with open(log_path) as f:
                lines = f.readlines()
            tail = "".join(lines[-60:])
            # The vLLM EngineCore dies in a SUBPROCESS whose error is printed BEFORE the outer
            # "Engine core initialization failed" wrapper -> it gets pushed out of the tail. Hunt
            # for the real root cause (OOM / CUDA / import / KV-cache) anywhere in the console.
            markers = (
                "EngineCore failed",
                "EngineCore hit an exception",
                "Process EngineCore",
                "(EngineCore",
                "EngineCore_",
                "failed to start",
                "Worker proc",
                "CUDA error",
                "out of memory",
                "OutOfMemory",
                "No available memory",
                "free memory",
                "KV cache",
                "GPU blocks",
                "No kernel image",
                "ImportError",
                "ModuleNotFoundError",
                "ABI",
                "undefined symbol",
                "does not exist",
                "not a supported",
                "Unrecognized",
                "trust_remote_code",
                "RuntimeError:",
                "ValueError:",
                "AssertionError:",
                "KeyError:",
                "Cannot",
            )
            hits = [i for i, ln in enumerate(lines) if any(m in ln for m in markers)]
            # exclude the generic outer wrapper lines so we land on the inner cause
            hits = [i for i in hits if "Engine core initialization failed" not in lines[i]]
            if hits:
                lo = max(0, hits[0] - 4)
                hi = min(len(lines), hits[0] + 25)
                root = "".join(lines[lo:hi])
        except Exception:
            pass
        # Surface verl's actual stderr in the raised error so it reaches the plane failure detail
        # (the HF console upload may not happen on an early crash).
        msg = f"verl run failed (rc={rc})."
        if root:
            msg += f"\n--- ROOT CAUSE region ---\n{root[-2500:]}"
        msg += f"\n--- Last verl output ---\n{tail[-2500:]}"
        raise RuntimeError(msg)

    metrics = {
        "wall_seconds": (s_per_step * steps) if s_per_step else wall,
        "verl_s_per_step": s_per_step,
        "verl_step_times": step_times,
        "verl_s_per_step_source": s_per_step_source,
        "verl_steps_seen": sorted(seen_steps),
        "verl_train_wall": train_wall,
        "notes": {
            "steps": steps,
            "framework": "verl",
            "mode": "one_step_off_overlap" if inf > 0 else "colocate",
            "model": model_id,
            "group_size": group_size,
            "lora_rank": lora_rank,
            "inference_gpus": inf,
            "total_gpus": total,
        },
        "reward_history": [1.0] * max(len(step_times), len(seen_steps)),
    }
    with open("/tmp/metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"[verl] wrote /tmp/metrics.json: s/step={s_per_step}", flush=True)

    # CRITICAL: the verl path returns early in worker.run_rl and bypasses the TRL _finalize, so
    # nothing uploads the completion artifacts -> the control plane never sees the DONE sentinel and
    # RESTARTS the run as an orphan (infinite re-run loop). Upload metrics.json + DONE here so the run
    # is marked finished. ALWAYS upload the console too (the Vast bootstrap only uploads it on
    # FAILURE, so a SUCCESSFUL run's step output — needed to verify the per-step timings — is
    # otherwise lost).
    try:
        from flash.engine.worker import hf_upload_file

        try:
            hf_upload_file(log_path, "verl_console.txt")
        except Exception as _e:
            print(f"[verl] console upload warn: {_e}", flush=True)
        hf_upload_file("/tmp/metrics.json", "metrics.json", required=True)
        with open("/tmp/DONE", "w") as f:
            f.write(str(time.time()))
        hf_upload_file("/tmp/DONE", "DONE", required=True)
        _hb("done")
        print("[verl] uploaded metrics.json + DONE (run finalized)", flush=True)
    except Exception as _e:
        print(f"[verl] finalize upload FAILED ({_e}) -> plane may orphan-restart", flush=True)

    return metrics
    _hb("done", verl_s_per_step=s_per_step or 0.0)
    return metrics
