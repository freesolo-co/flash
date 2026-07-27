"""verl-backed grpo training path for the fine-tuning worker.

selected by the private worker env FLASH_RL_BACKEND=verl. this path exists so grpo can run on
verl (volcengine hybridflow) instead of trl, as the first step toward multi-node rl. verl's
torch/vllm pins are incompatible with flash's, so flash NEVER imports verl in-process: verl runs
as a subprocess against a separate interpreter (FLASH_VERL_PYTHON, or a venv provisioned on the
pod). reward parity with the trl path is provided by a localhost rpc bridge that scores each
completion against flash's live env, so verl and trl compute identical rewards.

scope (bounded first pr): single-turn, non-multimodal, non-tool grpo only. sft, opd, and
multi-turn / tool / rollout_func rewards stay on the trl path (run_rl). anything outside this
scope raises rather than silently training on a different contract.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flash.engine.recipe import RECIPE
from flash.engine.steps import on_policy_steps, resolve_update_horizon
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.perf import gpu_diagnostics, setup_perf_backends, wait_for_gpu
from flash.engine.worker.rng import backend_seed, seed_training_rngs
from flash.engine.worker.rollout_samples import sanitize_rollout_text

DATA_SOURCE = "flash_env"


# --------------------------------------------------------------------------------------------
# pure helpers (no verl, no gpu, no network) -- unit tested directly.
# --------------------------------------------------------------------------------------------
def build_verl_dataset_rows(
    message_prompts: list[list[dict]],
    example_indices: list[int],
    ground_truths: list[str],
) -> list[dict]:
    """convert flash chat-message prompts into verl parquet rows.

    verl passes ``extra_info`` through to the reward fn; we carry the flash ``example_idx`` in
    ``extra_info.index`` so the reward bridge can map a completion back to its rollout example.
    """
    if not (len(message_prompts) == len(example_indices) == len(ground_truths)):
        raise ValueError("message_prompts / example_indices / ground_truths length mismatch")
    rows = []
    for messages, idx, gt in zip(message_prompts, example_indices, ground_truths, strict=True):
        rows.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": messages,
                "ability": "flash",
                "reward_model": {"style": "rule", "ground_truth": str(gt)},
                # verl's example index is the flash rollout_examples index; the reward bridge keys on it.
                "extra_info": {"split": "train", "index": int(idx)},
            }
        )
    return rows


def build_verl_overrides(cfg: dict) -> list[str]:
    """build the hydra override list for `python -m verl.trainer.main_ppo` (grpo + lora + vllm).

    carries the flash grpo recipe: dr-grpo advantages (no std norm, constant-length loss), the
    job's kl coefficient (flash default 0 = no kl term), constant lr, seed, sampling top_p,
    num_iterations (ppo_epochs), the max-steps horizon, and the save schedule.
    """
    kl_on = float(cfg["kl_coef"]) > 0
    o = [
        "algorithm.adv_estimator=grpo",
        # dr-grpo recipe: group-mean-centered advantages with NO std normalization, and a
        # constant-length loss aggregation (no per-response length bias). matches the trl path's
        # scale_rewards=none + loss_type=dr_grpo.
        "algorithm.norm_adv_by_std_in_grpo=False",
        f"actor_rollout_ref.actor.loss_agg_mode={cfg['loss_agg_mode']}",
        "algorithm.use_kl_in_reward=False",
        # truncated importance sampling (token-level, cap 2.0): corrects the vllm-rollout vs
        # fsdp-train policy mismatch. matches the trl path's tis recipe (token_truncate, c_max=2.0);
        # verl otherwise defaults to sequence-level tis, so pin token to match flash.
        "algorithm.rollout_correction.rollout_is=token",
        "algorithm.rollout_correction.rollout_is_threshold=2.0",
        f"data.train_files={cfg['train_files']}",
        f"data.val_files={cfg['val_files']}",
        f"data.train_batch_size={cfg['prompts_per_step']}",
        f"data.max_prompt_length={cfg['max_prompt_len']}",
        f"data.max_response_length={cfg['max_completion']}",
        "data.prompt_key=prompt",
        f"data.seed={cfg['seed']}",
        f"actor_rollout_ref.model.path={cfg['model_id']}",
        f"actor_rollout_ref.model.lora_rank={cfg['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={cfg['lora_alpha']}",
        f"actor_rollout_ref.model.target_modules={cfg['target_modules']}",
        # memory: match the trl path's gradient checkpointing.
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        *(
            [f"actor_rollout_ref.model.lora_adapter_path={cfg['warmstart_adapter']}"]
            if cfg.get("warmstart_adapter")
            else []
        ),
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        # 0 warmup -> verl's warmup+constant scheduler holds lr flat, matching the trl constant recipe.
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['prompts_per_step']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg['micro_batch']}",
        # num_iterations: amortize each generation batch over N optimizer passes.
        f"actor_rollout_ref.actor.ppo_epochs={cfg['num_iterations']}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={cfg['group_size']}",
        # safetensors load format is required for lora rollout on vllm.
        "actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['gpu_mem_util']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg['tp_size']}",
        f"actor_rollout_ref.rollout.temperature={cfg['temperature']}",
        f"actor_rollout_ref.rollout.top_p={cfg['top_p']}",
        # verl recomputes rollout log-probs for the importance ratio regardless of the kl term.
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg['micro_batch']}",
        f"custom_reward_function.path={cfg['reward_path']}",
        f"custom_reward_function.name={cfg['reward_name']}",
        # verl trainer-v1 reward loop reads reward.custom_reward_function (the legacy top-level key
        # is migrated in the main process but not visible to the RewardLoopWorker actor); emit both.
        f"reward.custom_reward_function.path={cfg['reward_path']}",
        f"reward.custom_reward_function.name={cfg['reward_name']}",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.total_epochs={cfg['total_epochs']}",
        # honor [train].max_steps: total_training_steps caps the optimizer-update horizon.
        f"trainer.total_training_steps={cfg['steps']}",
        f"trainer.save_freq={cfg['save_freq']}",
        "trainer.max_actor_ckpt_to_keep=1",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        f"trainer.logger=[{cfg['loggers']}]",
        "trainer.project_name=flash_verl",
        "trainer.experiment_name=grpo",
        f"trainer.default_local_dir={cfg['local_dir']}",
    ]
    if kl_on:
        # a reference policy is active -> add the token-level kl loss + the ref log-prob micro batch.
        o += [
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={cfg['kl_coef']}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={cfg['micro_batch']}",
        ]
    else:
        # flash default: dr-grpo with no kl term, so no reference policy.
        o.append("actor_rollout_ref.actor.use_kl_loss=False")
    if cfg.get("fp8_kv"):
        # fp8 kv cache on ada/hopper+ (cc>=8.9), matching the trl colocate path; tis (above) covers
        # the extra rollout-vs-train mismatch fp8 introduces. '+' appends the key under the existing
        # engine_kwargs.vllm struct (it is not a default field).
        o.append("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8")
    return o


def render_reward_module(url_env: str = "FLASH_VERL_REWARD_URL") -> str:
    """source for the verl custom reward module.

    runs INSIDE the verl interpreter, so it must be self-contained (stdlib only, no flash import).
    it forwards (index, solution_str) to the flash reward bridge and returns the float score.
    """
    return (
        '"""flash reward bridge shim (generated). posts each completion to the flash worker."""\n'
        "import json\n"
        "import os\n"
        "import urllib.request\n"
        "\n"
        f"_URL = os.environ.get({url_env!r}, '')\n"
        "\n"
        "\n"
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    idx = (extra_info or {}).get('index')\n"
        "    if idx is None or not _URL:\n"
        "        return 0.0\n"
        "    body = json.dumps({'index': int(idx), 'solution_str': solution_str or ''}).encode()\n"
        "    req = urllib.request.Request(_URL, data=body, headers={'Content-Type': 'application/json'})\n"
        "    try:\n"
        "        with urllib.request.urlopen(req, timeout=120) as r:\n"
        "            return float(json.loads(r.read().decode()).get('score', 0.0))\n"
        "    except Exception as exc:  # noqa: BLE001 -- a scoring error must not kill training\n"
        "        print('[flash-reward-bridge] scoring failed:', exc, flush=True)\n"
        "        return 0.0\n"
    )


def score_single_turn(
    env,
    solution_str: str,
    ex,
    *,
    tok,
    thinking: bool,
    prompt_opened_thinking: bool,
    think_penalty: float,
) -> float:
    """score one single-turn text completion exactly as the trl reward path does.

    mirrors run_rl.reward_fn's single-turn text branch: graded text -> env.scores_breakdown
    (preferred) or env.reward, minus the optional thinking-length penalty. env scoring errors
    are swallowed to 0.0 so a bad completion never kills the run.
    """
    try:
        graded = _w.graded_text(solution_str, prompt_opened_thinking=prompt_opened_thinking)
        state = (
            {
                "raw": solution_str,
                "completion": graded,
                "thinking": _w.thinking_text(solution_str, prompt_opened_thinking=prompt_opened_thinking),
            }
            if thinking
            else None
        )
        if hasattr(env, "scores_breakdown"):
            r = float(env.scores_breakdown(graded, ex, state).get("total", 0.0))
        else:
            r = float(env.reward(graded, ex, state))
    except Exception as exc:  # env scoring must not kill the run
        print(f"[rl-verl] env scoring raised ({type(exc).__name__}: {exc}); scoring 0.0", flush=True)
        return 0.0
    if think_penalty > 0 and thinking:
        r -= think_penalty * _w.think_token_count(
            solution_str, tok, prompt_opened_thinking=prompt_opened_thinking
        )
    return float(r)


# --------------------------------------------------------------------------------------------
# reward rpc bridge: verl subprocess -> flash live env.
# --------------------------------------------------------------------------------------------
def start_reward_server(score_by_index):
    """start a localhost http reward server. score_by_index(index, solution_str) -> float.

    returns (server, url). the server runs in a daemon thread; call server.shutdown() when done.
    """
    # serialize scoring so the flash env sees sequential calls, matching the trl reward path's
    # contract; verl's reward manager may otherwise call the reward with several workers at once.
    score_lock = threading.Lock()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path != "/score":
                self.send_response(404)
                self.end_headers()
                return
            n = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                with score_lock:
                    score = float(score_by_index(int(payload["index"]), payload.get("solution_str", "")))
            except Exception as exc:  # never 500 the trainer; score 0
                print(f"[rl-verl] reward server error: {exc}", flush=True)
                score = 0.0
            body = json.dumps({"score": score}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}/score"


# --------------------------------------------------------------------------------------------
# verl interpreter + checkpoint export.
# --------------------------------------------------------------------------------------------
def _resolve_verl_python(workdir: str) -> str:
    """return an interpreter that can import verl.

    prefers FLASH_VERL_PYTHON (set by the caller when verl is preinstalled, e.g. on a verl image).
    otherwise provisions an isolated venv on the pod so verl's torch/vllm never touch flash's env.
    """
    preset = os.environ.get("FLASH_VERL_PYTHON", "").strip()
    if preset:
        return preset
    venv = os.path.join(workdir, "verl-venv")
    py = os.path.join(venv, "bin", "python")
    if not os.path.exists(py):
        # isolated env: verl brings its own torch/vllm; --no-deps would miss runtime deps, so a
        # full install is used. exact pins are validated on the gpu pod before paid launch.
        subprocess.run(["uv", "venv", venv], check=True)
        subprocess.run(["uv", "pip", "install", "--python", py, "verl"], check=True)
        if os.environ.get("WANDB_API_KEY"):
            # verl does not pull wandb; add it (best-effort) so verl's wandb logger can import it in
            # this isolated interpreter instead of aborting. a failure here -> console logger below.
            subprocess.run(["uv", "pip", "install", "--python", py, "wandb"], check=False)
    return py


def _resolve_verl_loggers(python_bin: str) -> str:
    """verl's ``trainer.logger`` list. verl logs from its own interpreter, so enable the wandb
    logger only when WANDB_API_KEY is set AND wandb is importable in that interpreter; otherwise
    console-only. this never inits a flash-side run (flash does not train in-process on this path,
    so a flash-side run would stay empty) and never aborts verl when its env lacks wandb."""
    if not os.environ.get("WANDB_API_KEY"):
        return "console"
    has_wandb = (
        subprocess.run([python_bin, "-c", "import wandb"], capture_output=True).returncode == 0
    )
    if not has_wandb:
        print("[verl] WANDB_API_KEY set but wandb is unavailable in the verl interpreter; using console logger only")
    return "console,wandb" if has_wandb else "console"


def _export_peft_adapter(
    ckpt_actor_dir: str,
    out_adapter_dir: str,
    *,
    base_model_id: str,
    python_bin: str,
) -> None:
    """turn verl's saved lora checkpoint into a flash-servable peft adapter dir.

    verl saves fsdp-sharded checkpoints under ``<local_dir>/global_step_N/actor`` (model/optim
    shards + a ``huggingface/`` config+tokenizer subfolder). ``verl.model_merger merge`` writes a
    standard peft adapter (adapter_config.json + adapter_model.safetensors) to a ``lora_adapter/``
    subfolder of its target; we copy just that adapter into flash's adapter dir (the co-produced
    merged full model is discarded -- flash serves the lora on the immutable base).

    verified against verl 0.8 on an h100 (r2): merger emits ``<target>/lora_adapter/{adapter_config
    .json,adapter_model.safetensors}`` with ``base_model_name_or_path: null``.
    """
    os.makedirs(out_adapter_dir, exist_ok=True)
    merge_out = out_adapter_dir.rstrip("/") + "_merge"
    shutil.rmtree(merge_out, ignore_errors=True)
    merge_env = dict(os.environ)
    # the base config/tokenizer are already cached; keep the merger off hf's rate-limited api.
    merge_env["HF_HUB_OFFLINE"] = "1"
    merge_env["TRANSFORMERS_OFFLINE"] = "1"
    merge_env["HF_HUB_DISABLE_XET"] = "1"
    subprocess.run(
        [python_bin, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
         "--local_dir", ckpt_actor_dir, "--target_dir", merge_out],
        check=True, env=merge_env,
    )
    lora_dir = os.path.join(merge_out, "lora_adapter")
    if not os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
        raise RuntimeError(
            f"verl model_merger did not produce a peft adapter at {lora_dir} (no adapter_config.json); "
            "the merger output layout must be adjusted for this verl version."
        )
    for name in os.listdir(lora_dir):
        shutil.copy2(os.path.join(lora_dir, name), os.path.join(out_adapter_dir, name))
    shutil.rmtree(merge_out, ignore_errors=True)


def _latest_global_step_dir(local_dir: str) -> tuple[str, int]:
    """return (actor_dir, step) for the highest global_step_N checkpoint verl wrote."""
    best_step, best = -1, ""
    if os.path.isdir(local_dir):
        for name in os.listdir(local_dir):
            m = re.fullmatch(r"global_step_(\d+)", name)
            if m and int(m.group(1)) > best_step:
                best_step = int(m.group(1))
                best = os.path.join(local_dir, name, "actor")
    if best_step < 0:
        raise RuntimeError(f"no global_step_N checkpoint found under {local_dir}")
    return best, best_step


def _stamp_adapter_dir_provenance(adapter_dir: str, model_id: str, model_revision: str = "") -> None:
    """stamp the saved adapter's immutable base identity into adapter_config.json.

    dir-based analogue of _w.stamp_adapter_provenance (which needs an in-memory peft model). same
    validation + fields, applied to the json verl produced.
    """
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    current_base = str(cfg.get("base_model_name_or_path", "") or "").strip()
    if current_base and current_base != model_id:
        raise RuntimeError(
            f"adapter base model {current_base!r} does not match validated target {model_id!r}"
        )
    current_rev = str(cfg.get("revision", "") or "").strip()
    if current_rev and model_revision and current_rev != model_revision:
        raise RuntimeError("adapter base revision does not match the validated target commit")
    cfg["base_model_name_or_path"] = model_id
    cfg["revision"] = model_revision or None
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)


# --------------------------------------------------------------------------------------------
# orchestration.
# --------------------------------------------------------------------------------------------
def _resolve_single_turn_inputs():
    """reproduce run_rl's front-half config + dataset prep, fenced to single-turn text grpo."""
    env = _w.require_active_env()
    if getattr(env, "multi_turn", False) or getattr(env, "is_tool_env", False):
        raise RuntimeError(
            "FLASH_RL_BACKEND=verl supports single-turn, non-tool grpo only; this env is "
            "multi-turn/tool. use the trl backend (unset FLASH_RL_BACKEND) for it."
        )
    seed_training_rngs(_w.SEED)
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""

    rl = RECIPE.rl
    _t = _w.JOB_SPEC.train if _w.JOB_SPEC else None
    # fail loud on grpo features the verl backend does not yet honor, so a migration never silently
    # trains without a requested behavior.
    if _t and getattr(_t, "structured_outputs", ""):
        raise RuntimeError(
            "train.structured_outputs is not yet supported on the verl backend; use the trl backend "
            "(unset FLASH_RL_BACKEND) until guided-decoding parity lands."
        )
    if _t and getattr(_t, "stop_sequences", ()):
        raise RuntimeError(
            "train.stop_sequences is not yet supported on the verl backend (verl's vllm rollout has "
            "no stop-string field here); use the trl backend (unset FLASH_RL_BACKEND) for it."
        )
    if _t and getattr(_t, "save_at_steps", ()):
        raise RuntimeError(
            "train.save_at_steps is not yet supported on the verl backend (verl saves on a fixed "
            "save_freq interval, not arbitrary steps); use the trl backend for it."
        )
    if model_revision:
        raise RuntimeError(
            "model_revision pinning is not yet supported on the verl backend (verl loads the model "
            "at its default revision); use the trl backend for revision-pinned runs."
        )
    # entropy_quantile drives trl GRPOTrainer's internal top-entropy token masking (top_entropy_quantile);
    # verl has no equivalent, so honoring the flash default (1.0 = no masking) is fine but any customer
    # value < 1.0 must fail loud rather than train without the requested masking (silent drift).
    _eq = getattr(_t, "entropy_quantile", None) if _t else None
    if _eq is not None and float(_eq) < 1.0:
        raise RuntimeError(
            f"train.entropy_quantile={_eq} is not yet supported on the verl backend (trl's top-entropy "
            "token masking has no verl equivalent); use the trl backend (unset FLASH_RL_BACKEND) for it."
        )
    # flash's lora dropout is fixed at 0.0; verl matches (peft default). guard defensively so a future
    # non-zero recipe value can never be silently ignored.
    if float(RECIPE.lora.dropout) != 0.0:
        raise RuntimeError(
            f"RECIPE.lora.dropout={RECIPE.lora.dropout} is not yet wired on the verl backend; "
            "use the trl backend or add actor_rollout_ref.model.lora_dropout support."
        )
    gcfg = _w.grpo_overrides()
    prompts_per_step = int(_t.batch_size if _t and _t.batch_size is not None else rl.prompts_per_step)
    group_size = int(gcfg.get("group_size") or rl.group_size)
    _gcfg_temp = gcfg.get("temperature")
    temperature = float(_gcfg_temp if _gcfg_temp is not None else rl.sampling_temperature)
    think_penalty = float(gcfg.get("thinking_length_penalty_coef") or 0.0)
    # flash defaults kl_penalty_coef to 0 (dr-grpo, no kl term); honor it rather than forcing a kl.
    kl_coef = float(gcfg.get("kl_penalty_coef") or 0.0)
    # advantage_clip is recorded but not applied (the trl path centers advantages without a value
    # clip; verl's grpo advantage is likewise group-centered). log it for parity, do not apply.
    if float(gcfg.get("advantage_clip") or 0.0) > 0:
        print(
            f"[rl-verl] advantage_clip={gcfg['advantage_clip']} recorded; verl centers grpo "
            "advantages (no value clip), matching the trl path",
            flush=True,
        )
    # trl drops truncated (non-eos) completions from the grpo loss (mask_truncated_completions,
    # default True); verl's vllm rollout here has no per-completion truncation-mask knob, so verl
    # keeps them. stop_sequences (the only case that turns masking off) already fails loudly above,
    # so this is always the default-True case. record the divergence for parity observability
    # rather than silently ignoring it; adding verl-side masking is a tracked follow-up.
    if _w.grpo_mask_truncated_completions(_t):
        print(
            "[rl-verl] mask_truncated_completions=True; verl keeps truncated completions in the "
            "grpo loss (no verl truncation-mask knob) -- recorded parity caveat, not applied",
            flush=True,
        )
    learning_rate = float(_t.learning_rate if _t and _t.learning_rate is not None else rl.learning_rate)
    # warm-start forbids lora_rank, so a set init_from_adapter already raised above; read rank/alpha
    # from the job spec (falling back to the recipe) exactly like the trl path's lora config.
    lora_rank = int(_t.lora_rank) if (_t and _t.lora_rank) else int(RECIPE.lora.rank)
    lora_alpha = int(_t.lora_alpha) if (_t and _t.lora_alpha) else int(RECIPE.lora.alpha)
    # warm-start: continue the sft adapter in place (verl lora_adapter_path). uses the SOURCE
    # adapter's rank/alpha (flash forbids a child lora_rank on warm-start).
    warmstart_adapter = ""
    if _t and getattr(_t, "init_from_adapter", ""):
        if kl_coef > 0:
            raise RuntimeError(
                "warm-start (init_from_adapter) with kl_penalty_coef>0 anchors the kl reference to "
                "the sft adapter; the verl backend's reference is the base, so this is not yet "
                "supported. set kl_penalty_coef=0, or use the trl backend for kl-anchored warm-start."
            )
        from flash.engine.worker.adapter import _download_adapter

        warmstart_adapter = _download_adapter(_t.init_from_adapter)
        if not warmstart_adapter:
            raise RuntimeError(
                "warm-start source adapter could not be downloaded; refusing to start from the base."
            )
        with open(os.path.join(warmstart_adapter, "adapter_config.json")) as f:
            _src_cfg = json.load(f)
        lora_rank = int(_src_cfg.get("r", lora_rank))
        lora_alpha = int(_src_cfg.get("lora_alpha", lora_alpha))
        print(
            f"[rl-verl] warm-start: continuing source adapter (r={lora_rank}, alpha={lora_alpha}) "
            f"from {_t.init_from_adapter}",
            flush=True,
        )

    train = env.dataset()
    _max_examples = getattr(_t, "max_examples", None) if _t else None
    if _max_examples:
        train = train[: int(_max_examples)]
    rng = random.Random(_w.SEED)
    rng.shuffle(train)
    message_prompts = [env.prompt_messages(ex) for ex in train]

    from flash.multimodal import record_has_images

    if any(record_has_images(ex, m) for ex, m in zip(train, message_prompts, strict=True)):
        raise RuntimeError(
            "FLASH_RL_BACKEND=verl supports non-multimodal grpo only; this env has image prompts. "
            "use the trl backend for it."
        )

    tok = _w.load_tokenizer(model_id, revision=model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    max_completion = int(
        gcfg.get("max_tokens")
        or (rl.max_completion_len_thinking if _w.THINKING else rl.max_completion_len)
    )
    train_ctx = _t.max_context_tokens if (_t and _t.max_context_tokens) else 0
    vllm_max_len = int(train_ctx or max(1024, rl.max_prompt_len + max_completion))
    prompt_budget = vllm_max_len - max_completion
    if prompt_budget <= 0:
        raise ValueError("engine length leaves no room for the completion; raise max_context_tokens")

    prompts = []
    for ex, messages in zip(train, message_prompts, strict=True):
        rendered = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=_w.THINKING
        )
        if 0 < len(tok(rendered, add_special_tokens=False).input_ids) <= prompt_budget:
            prompts.append({"prompt": messages, "rendered": rendered, "example": ex})
    if not prompts:
        raise ValueError(f"every training prompt exceeds the {prompt_budget}-token prompt budget")

    prompts_per_step = _w.resolve_grpo_prompts_per_step(prompts_per_step, len(prompts))
    prompt_opened_thinking = bool(_w.THINKING) and _w.prompt_opens_thinking(prompts[0]["rendered"])

    # optimizer-update horizon, honoring [train].max_steps exactly like the trl path.
    epochs = int(_t.epochs) if (_t and _t.epochs is not None) else int(rl.num_epochs)
    derived_steps = on_policy_steps(
        epochs=epochs, prompt_count=len(prompts), prompts_per_step=prompts_per_step
    )
    steps = resolve_update_horizon(derived_steps, getattr(_t, "max_steps", None) if _t else None)
    save_every = int(_t.save_every) if (_t and _t.save_every) else 20

    return {
        "env": env,
        "tok": tok,
        "model_id": model_id,
        "model_revision": model_revision,
        "prompts": prompts,
        "prompts_per_step": prompts_per_step,
        "group_size": group_size,
        "temperature": temperature,
        "top_p": float(rl.sampling_top_p),
        "think_penalty": think_penalty,
        "kl_coef": kl_coef,
        "max_completion": max_completion,
        "max_prompt_len": prompt_budget,
        "prompt_opened_thinking": prompt_opened_thinking,
        "lr": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "warmstart_adapter": warmstart_adapter,
        "epochs": epochs,
        "steps": int(steps),
        "save_every": save_every,
        # flash amortizes each generation batch over 2 optimizer passes (num_iterations=2).
        "num_iterations": 2,
        "seed": int(backend_seed(_w.SEED)),
    }


def run_rl_verl():
    """grpo training on verl, output-compatible with run_rl. see module docstring for scope."""
    import pandas as pd

    t_start = time.time()
    _w.heartbeat("rl_start", gpu=gpu_diagnostics())
    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        gpu_type=_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else "",
    )
    setup_perf_backends()

    inp = _resolve_single_turn_inputs()
    env, tok = inp["env"], inp["tok"]
    prompts = inp["prompts"]

    # cache the base model before launching verl, then run verl fully offline so its vllm /
    # transformers never hit hf's (rate-limited) api. flash already owns model prefetch; the verl
    # subprocess simply reuses that cache.
    if inp["model_revision"]:
        _w.prefetch_model(inp["model_id"], revision=inp["model_revision"])
    else:
        _w.prefetch_model(inp["model_id"])

    # stable int index -> rollout example, exactly as the trl path (reward maps back via this).
    ds_rows, rollout_examples = _w.build_grpo_prompt_dataset(prompts)
    message_prompts = [p["prompt"] for p in prompts]
    indices = [int(r["example_idx"]) for r in ds_rows]
    # ground_truth is a verl-schema placeholder only; the reward bridge scores by example_idx
    # against the live env and never reads it.
    ground_truths = [str(ex.get("answer", "") or "") if isinstance(ex, dict) else "" for ex in rollout_examples]

    workdir = f"/tmp/rl_verl_seed{_w.SEED}"
    os.makedirs(workdir, exist_ok=True)
    local_dir = os.path.join(workdir, "ckpt")
    train_pq = os.path.join(workdir, "train.parquet")
    val_pq = os.path.join(workdir, "val.parquet")
    reward_py = os.path.join(workdir, "reward.py")

    rows = build_verl_dataset_rows(message_prompts, indices, ground_truths)
    pd.DataFrame(rows).to_parquet(train_pq)
    pd.DataFrame(rows[: max(1, min(4, len(rows)))]).to_parquet(val_pq)
    with open(reward_py, "w") as f:
        f.write(render_reward_module())

    # reward bridge: verl (out of process) -> flash live env, identical to trl scoring. also keep a
    # rolling buffer of recent (completion, score) so the training loop can dump one sample per step
    # to the flash log, matching the trl path's #607 per-step completion dump.
    recent_samples: list[tuple[str, float]] = []
    _samples_lock = threading.Lock()

    def _score(index: int, solution_str: str) -> float:
        ex = rollout_examples[int(index)]
        score = score_single_turn(
            env, solution_str, ex,
            tok=tok, thinking=bool(_w.THINKING),
            prompt_opened_thinking=inp["prompt_opened_thinking"],
            think_penalty=inp["think_penalty"],
        )
        with _samples_lock:
            recent_samples.append((solution_str, score))
            del recent_samples[:-64]
        return score

    server, reward_url = start_reward_server(_score)
    try:
        proc = None
        python_bin = _resolve_verl_python(workdir)
        micro_batch = 1
        expected_steps = int(inp["steps"])
        # verl logs from its own interpreter; gate wandb on that env (see _resolve_verl_loggers).
        loggers = _resolve_verl_loggers(python_bin)
        # fp8 kv cache on ada/hopper+ (cc>=8.9), exactly like the trl colocate path — but NOT for
        # hybrid linear-attention (GDN) models: vllm's fp8-kv wake path (init_fp8_kv_scales) assumes a
        # plain kv tensor and crashes on the hybrid cache ('list' has no zero_) under verl sleep/wake.
        try:
            import torch as _torch_cc

            from flash.engine.worker.packing import model_is_gdn_hybrid

            _cc_ok = bool(
                _torch_cc.cuda.is_available() and _torch_cc.cuda.get_device_capability() >= (8, 9)
            )
            fp8_kv = _cc_ok and not model_is_gdn_hybrid(inp["model_id"])
        except Exception:  # no cuda / probe failure -> conservative bf16 kv
            fp8_kv = False
        cfg = {
            "train_files": train_pq, "val_files": val_pq,
            "model_id": inp["model_id"], "lora_rank": inp["lora_rank"],
            "lora_alpha": inp["lora_alpha"], "target_modules": "all-linear",
            "lr": inp["lr"], "group_size": inp["group_size"],
            "prompts_per_step": inp["prompts_per_step"], "micro_batch": micro_batch,
            "max_prompt_len": inp["max_prompt_len"], "max_completion": inp["max_completion"],
            "temperature": inp["temperature"], "top_p": inp["top_p"], "kl_coef": inp["kl_coef"],
            "loss_agg_mode": "seq-mean-token-sum-norm", "seed": inp["seed"],
            "num_iterations": inp["num_iterations"], "steps": expected_steps,
            "warmstart_adapter": inp["warmstart_adapter"],
            "gpu_mem_util": 0.5, "tp_size": 1, "loggers": loggers, "fp8_kv": fp8_kv,
            "reward_path": reward_py, "reward_name": "compute_score",
            "total_epochs": inp["epochs"], "save_freq": inp["save_every"], "local_dir": local_dir,
        }
        overrides = build_verl_overrides(cfg)

        # clear any checkpoints left by a prior run on this fixed per-seed worker dir, so finalize's
        # highest-global_step_N pick can only ever see THIS run's checkpoints (a retry or partial
        # run otherwise leaves stale global_step_* folders that could export a prior run's adapter).
        shutil.rmtree(local_dir, ignore_errors=True)

        setup_seconds = time.time() - t_start
        _w.heartbeat("rl_train_start", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
        step_box = [0]

        def _progress():
            return step_box[0]

        env_for_verl = dict(os.environ)
        env_for_verl["FLASH_VERL_REWARD_URL"] = reward_url
        # the model is prefetched above; keep the subprocess off hf's rate-limited api.
        env_for_verl["HF_HUB_OFFLINE"] = "1"
        env_for_verl["TRANSFORMERS_OFFLINE"] = "1"
        env_for_verl["HF_HUB_DISABLE_XET"] = "1"
        step_re = re.compile(r"step:\s*(\d+)")
        reward_re = re.compile(r"critic/rewards/mean:([-\d.eE+]+)")
        loss_re = re.compile(r"actor/pg_loss:([-\d.eE+]+)")
        reward_history: list[float] = []
        loss_curve: list[float] = []
        last_dump_step = [-1]
        with liveness_heartbeat("rl_verl_training", progress=_progress, progress_step=True):
            proc = subprocess.Popen(
                [python_bin, "-m", "verl.trainer.main_ppo", *overrides],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env_for_verl,
            )
            for line in proc.stdout:
                print(f"[verl] {line}", end="", flush=True)
                m = step_re.search(line)
                if m:
                    step_box[0] = int(m.group(1))
                    # dump one sample completion per new step to the flash log (matches trl #607).
                    if step_box[0] != last_dump_step[0]:
                        with _samples_lock:
                            samp = recent_samples[-1] if recent_samples else None
                        if samp:
                            last_dump_step[0] = step_box[0]
                            preview = " ".join(sanitize_rollout_text(samp[0])[:300].split())
                            print(
                                f"[rl-verl] step {step_box[0]} sample (reward={samp[1]:.3f}): {preview}",
                                flush=True,
                            )
                # capture verl's per-step reward + policy loss for train_meta observability parity.
                for pat, sink in ((reward_re, reward_history), (loss_re, loss_curve)):
                    hit = pat.search(line)
                    if hit:
                        with contextlib.suppress(ValueError):
                            sink.append(float(hit.group(1)))
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"verl.trainer.main_ppo exited {rc}; see the flash log for the traceback")
    finally:
        # ensure the verl child can't outlive us holding gpu memory if stdout reading raised before
        # proc.wait() (shutting down the reward server alone would leave main_ppo running).
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)
        server.shutdown()

    # collect verl's lora checkpoint -> flash-servable peft adapter, then reuse flash finalize.
    out_dir = f"/tmp/rl_seed{_w.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    shutil.rmtree(adapter_dir, ignore_errors=True)
    os.makedirs(adapter_dir, exist_ok=True)
    actor_dir, steps_run = _latest_global_step_dir(local_dir)
    if steps_run < expected_steps:
        raise RuntimeError(f"grpo completed {steps_run}/{expected_steps} requested optimizer updates")
    # an empty reward_history means the reward bridge never scored a completion (rollout produced
    # nothing, or scoring never ran): a no-op run, not a success. mirror the trl path and fail
    # loudly instead of exporting/publishing an untrained adapter. verl never resumes -> ckpt None.
    if _w._grpo_is_no_op_failure(reward_history, None, expected_steps, steps_run):
        raise RuntimeError(
            f"verl grpo scored no reward over {steps_run} step(s) — the rollout produced no "
            "completions, so the policy was never actually trained. failing loudly instead of "
            "publishing a no-op run as done."
        )

    with liveness_heartbeat("rl_verl_finalizing", progress=lambda: steps_run, progress_step=True, keepalive=True):
        _export_peft_adapter(actor_dir, adapter_dir, base_model_id=inp["model_id"], python_bin=python_bin)
        tok.save_pretrained(adapter_dir)
        _stamp_adapter_dir_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
        _w.write_base_model_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        _w.publish_deployable_checkpoint(adapter_dir, steps_run)

    train_wall = time.time() - t_start
    _w.heartbeat("rl_trained", train_wall=train_wall, step=steps_run, gpu=gpu_diagnostics())
    _w.write_train_meta(
        phase="rl",
        adapter_dir=adapter_dir,
        model_id=inp["model_id"],
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=0,
        generated_tokens=0,
        notes={
            "backend": "verl",
            "steps": steps_run,
            "epochs": inp["epochs"],
            "retained_prompts": len(prompts),
            "group_size": inp["group_size"],
            "reward_history": reward_history,
            "loss_curve": loss_curve,
            "grpo_recipe": {
                "kl_coef": inp["kl_coef"],
                "temperature": inp["temperature"],
                "top_p": inp["top_p"],
                "num_iterations": inp["num_iterations"],
                "seed": inp["seed"],
                "loss_agg_mode": "seq-mean-token-sum-norm",
                "norm_adv_by_std_in_grpo": False,
            },
        },
    )
