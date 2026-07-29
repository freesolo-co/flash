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
import math
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
from flash.engine.worker.verl_common import (
    VERL_REQUIREMENT,
    agent_loop_workers,
    clamp_engine_len,
    model_max_position_embeddings,
    resolve_verl_python,
    verl_supports_rollout_field,
)
from flash.spec import gpu_count_of

DATA_SOURCE = "flash_env"


# --------------------------------------------------------------------------------------------
# pure helpers (no verl, no gpu, no network) -- unit tested directly.
# --------------------------------------------------------------------------------------------
def _verl_epochs_for_horizon(
    *, epochs: int, prompt_count: int, prompts_per_step: int, steps: int
) -> int:
    if prompt_count <= 0:
        raise ValueError("prompt_count must be positive")
    if prompts_per_step <= 0:
        raise ValueError("prompts_per_step must be positive")
    if prompt_count < prompts_per_step:
        raise ValueError("prompt_count must be at least prompts_per_step")
    # verl hardcodes drop_last=True, so each epoch serves floor(prompt_count / batch_size) steps.
    # the flag cannot be configured off, and ceil here would still under-serve the requested horizon.
    steps_per_epoch = prompt_count // prompts_per_step
    return max(epochs, math.ceil(steps / steps_per_epoch))


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
    job's kl coefficient (flash default 0 = no kl term), constant lr, seed, sampling top_p, and one
    ppo epoch so total_training_steps counts optimizer updates. unlike trl's generation-batch reuse,
    verl samples a fresh rollout per update, so rollout volume and policy staleness still differ.
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
        # rollout prompt parity: verl renders raw messages with the tokenizer's chat template;
        # thread flash's thinking mode so the rollout sees the same prompt as the trl path.
        f"+data.apply_chat_template_kwargs.enable_thinking={str(bool(cfg.get('thinking', False))).lower()}",
        f"data.seed={cfg['seed']}",
        # rollout sampling seed. NOT `rollout.seed`: verl 0.8.0's RolloutConfig declares no such
        # field, so a bare key fails hydra composition and a `+`/`++` prefix composes but then dies
        # in omega_conf_to_dataclass with an unexpected-kwarg TypeError. engine_kwargs is a declared
        # dict on both 0.8.0 and 0.9.x and is spread into the vllm engine args *after* verl's own
        # "seed" entry, so it wins. `++` because the sub-key is absent from the composed node.
        # single replica here (tp == n_gpus, nnodes=1), so verl's replica_rank offset is always 0.
        f"++actor_rollout_ref.rollout.engine_kwargs.vllm.seed={cfg['seed']}",
        f"actor_rollout_ref.model.path={cfg['model_id']}",
        f"actor_rollout_ref.model.lora_rank={cfg['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={cfg['lora_alpha']}",
        f"actor_rollout_ref.model.target_modules={cfg['target_modules']}",
        # memory: match the trl path's gradient checkpointing.
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        # 32k contexts: fused linear-CE computes logprobs/entropy from hidden states + lm_head in
        # chunks (FusedLinearForPPO), never materializing the [tokens, vocab] logits tensor
        # (~130 GB at 32k on a 248k vocab). torch backend = numerically exact, no extra deps.
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.use_fused_kernels=True",
        "actor_rollout_ref.model.fused_kernel_options.impl_backend=torch",
        *(
            [f"actor_rollout_ref.model.lora_adapter_path={cfg['warmstart_adapter']}"]
            if cfg.get("warmstart_adapter")
            else []
        ),
        f"actor_rollout_ref.actor.optim.lr={cfg['lr']}",
        # only the lora adapter is trainable; decaying its weights toward zero is not part of grpo.
        "actor_rollout_ref.actor.optim.weight_decay=0.0",
        # 0 warmup -> verl's warmup+constant scheduler holds lr flat, matching the trl constant recipe.
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg['prompts_per_step']}",
        # 32k contexts: bound the backward pass by TOKENS, not by sequence count. a fixed
        # micro_batch_size_per_gpu of N sequences is N*engine_len tokens in the worst case, so the
        # only count that is safe at 32k is 1 -- which then wastes most of the budget on the short
        # sequences that make up a typical batch. dynamic bsz packs each micro-batch up to
        # max_token_len instead, so long sequences get their own pass and short ones share.
        # verl's engine multiplies this per-gpu budget by sp_size itself, so it must NOT be
        # pre-divided by the ulysses width.
        "actor_rollout_ref.actor.use_dynamic_bsz=true",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={cfg['max_token_len_per_gpu']}",
        # ppo_epochs multiplies verl's update loop, so its default of 1 preserves the requested update
        # count and on-policy baseline. unlike trl reuse, verl samples a fresh rollout for every update.
        f"actor_rollout_ref.actor.ppo_epochs={cfg['ppo_epochs']}",
        # multi-gpu shape: shard every card along the sequence (ulysses) rather than the batch, and
        # give vllm the same width for tensor parallelism. verl builds mesh_shape=(dp, sp) from this,
        # so sp == n_gpus keeps dp == 1: the optimizer still sees ONE global batch of exactly
        # prompts_per_step * group_size, identical to the single-gpu recipe. splitting the batch
        # instead would silently change the gradient (dp shards ppo_mini_batch_size). sequence
        # sharding is also what makes long contexts fit, since activations divide by n_gpus.
        # ref (when kl is on) inherits this through dp_ref.yaml's oc.select on the actor key.
        # requires use_remove_padding, which this recipe already sets above.
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={cfg['n_gpus']}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={cfg['group_size']}",
        # verl 0.8.0 hardcodes async rollout (ray_trainer.py:914), and AgentLoopManager splits the
        # rollout batch across agent.num_workers with DataProto.chunk, which asserts EXACT
        # divisibility (agent_loop.py:1111 -> protocol.py:874). the batch it chunks is
        # prompts_per_step * group_size, so verl's default of 8 workers aborts the run before the
        # first step whenever that product is not a multiple of 8 - e.g. the last short batch of an
        # epoch, or any small job. flash's own defaults (64 x 8) happen to divide, but nothing
        # constrains a user's [train].batch_size or group_size to keep that true. size the worker
        # pool to the batch instead: the largest divisor of the rollout batch that is <= 8, which is
        # always valid and still parallelizes whenever the batch allows it.
        f"actor_rollout_ref.rollout.agent.num_workers={agent_loop_workers(int(cfg['prompts_per_step']) * int(cfg['group_size']))}",
        # fork-only rollout field, so `++` (append-or-override): it exists in the fork's rollout.yaml
        # but not in stock verl 0.8.0's, where a bare key or `+` would break. omitted entirely when
        # false, since not masking is already stock behavior and stock verl rejects the unknown key.
        *(
            [
                "++actor_rollout_ref.rollout.mask_truncated_completions="
                f"{str(bool(cfg['mask_truncated_completions'])).lower()}"
            ]
            if cfg["mask_truncated_completions"]
            else []
        ),
        # safetensors load format is required for lora rollout on vllm.
        "actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={cfg['gpu_mem_util']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={cfg['n_gpus']}",
        f"actor_rollout_ref.rollout.temperature={cfg['temperature']}",
        f"actor_rollout_ref.rollout.top_p={cfg['top_p']}",
        # size vllm's kv cache for the job's context, not the architecture's. left unset, verl
        # substitutes the model's full max_position_embeddings and hands that straight to the
        # engine, so a 4k job on a 40k-capable model reserves ten times the kv cache it can ever
        # use -- and on a long-context model that reservation alone can exhaust the
        # gpu_memory_utilization budget before a single rollout runs.
        f"actor_rollout_ref.rollout.max_model_len={cfg['max_model_len']}",
        # verl recomputes rollout log-probs for the importance ratio regardless of the kl term.
        # the engine asserts actor.use_dynamic_bsz == rollout.log_prob_use_dynamic_bsz, so the
        # log-prob pass switches to token budgeting with the actor, never independently.
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true",
        f"actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu={cfg['max_token_len_per_gpu']}",
        f"custom_reward_function.path={cfg['reward_path']}",
        f"custom_reward_function.name={cfg['reward_name']}",
        # verl trainer-v1 reward loop reads reward.custom_reward_function (the legacy top-level key
        # is migrated in the main process but not visible to the RewardLoopWorker actor); emit both.
        f"reward.custom_reward_function.path={cfg['reward_path']}",
        f"reward.custom_reward_function.name={cfg['reward_name']}",
        f"trainer.n_gpus_per_node={cfg['n_gpus']}",
        "trainer.nnodes=1",
        f"trainer.total_epochs={cfg['total_epochs']}",
        # honor [train].max_steps: total_training_steps caps the optimizer-update horizon.
        f"trainer.total_training_steps={cfg['steps']}",
        f"trainer.save_freq={cfg['save_freq']}",
        "trainer.max_actor_ckpt_to_keep=1",
        # resume from whatever _restore_verl_resume staged into default_local_dir. auto is a no-op
        # when nothing was staged, so a fresh run is unaffected; without it a preempted run silently
        # restarts at step 0 and re-bills the whole budget.
        "trainer.resume_mode=auto",
        "trainer.test_freq=-1",
        "trainer.val_before_train=False",
        f"trainer.logger=[{cfg['loggers']}]",
        "trainer.project_name=flash_verl",
        "trainer.experiment_name=grpo",
        f"trainer.default_local_dir={cfg['local_dir']}",
    ]
    if kl_on:
        # a reference policy is active -> add the token-level kl loss. the ref worker needs no
        # batching keys of its own: ref.yaml defaults log_prob_use_dynamic_bsz and
        # log_prob_max_token_len_per_gpu to oc.select on the matching actor keys, which the actor
        # block above sets, and the engine pops them onto the ref's own config.
        o += [
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={cfg['kl_coef']}",
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


def _build_verl_training_cfg(
    inp: dict,
    *,
    train_files: str,
    val_files: str,
    model_id: str,
    thinking: bool,
    loggers: str,
    fp8_kv: bool,
    reward_path: str,
    local_dir: str,
    n_gpus: int = 1,
) -> dict:
    engine_len = int(inp["engine_len"])
    return {
        "train_files": train_files,
        "val_files": val_files,
        "model_id": model_id,
        "lora_rank": inp["lora_rank"],
        "lora_alpha": inp["lora_alpha"],
        "target_modules": "all-linear",
        "lr": inp["lr"],
        "group_size": inp["group_size"],
        "prompts_per_step": inp["prompts_per_step"],
        "mask_truncated_completions": inp["mask_truncated_completions"],
        # already clamped to the model's limit by the resolver, which derives max_prompt_len from
        # the same value -- so the engine, the prompt filter, and the token budget cannot disagree.
        "max_model_len": engine_len,
        # one full-length sequence is the floor: a budget below engine_len cannot schedule the
        # longest sequence the engine will ever produce, and verl would fail to place it in any
        # micro-batch. this is the same sequence_length-derived budget sft and opd use.
        "max_token_len_per_gpu": engine_len,
        "max_prompt_len": inp["max_prompt_len"],
        "max_completion": inp["max_completion"],
        "temperature": inp["temperature"],
        "top_p": inp["top_p"],
        "kl_coef": inp["kl_coef"],
        "thinking": thinking,
        "loss_agg_mode": "seq-mean-token-sum-norm",
        "seed": inp["seed"],
        "ppo_epochs": inp["ppo_epochs"],
        "steps": int(inp["steps"]),
        "warmstart_adapter": inp["warmstart_adapter"],
        "gpu_mem_util": 0.5,
        "n_gpus": n_gpus,
        "loggers": loggers,
        "fp8_kv": fp8_kv,
        "reward_path": reward_path,
        "reward_name": "compute_score",
        "total_epochs": inp["verl_total_epochs"],
        "save_freq": inp["save_every"],
        "local_dir": local_dir,
    }


def _build_verl_train_notes(
    inp: dict,
    *,
    steps_run: int,
    retained_prompts: int,
    reward_history: list[float],
    loss_curve: list[float],
    resumed: bool = False,
) -> dict:
    return {
        "backend": "verl",
        "steps": steps_run,
        "epochs": inp["epochs"],
        "retained_prompts": retained_prompts,
        # matches the trl path: without it a resumed run is indistinguishable from a fresh one.
        "resumed": resumed,
        "group_size": inp["group_size"],
        "reward_history": reward_history,
        "loss_curve": loss_curve,
        "grpo_recipe": {
            "kl_coef": inp["kl_coef"],
            "temperature": inp["temperature"],
            "top_p": inp["top_p"],
            "ppo_epochs": inp["ppo_epochs"],
            "verl_total_epochs": inp["verl_total_epochs"],
            "seed": inp["seed"],
            "loss_agg_mode": "seq-mean-token-sum-norm",
            "norm_adv_by_std_in_grpo": False,
        },
    }


def render_reward_module(url_env: str = "FLASH_VERL_REWARD_URL") -> str:
    """source for the verl custom reward module.

    runs INSIDE the verl interpreter, so it must be self-contained (stdlib only, no flash import).
    it forwards (index, solution_str) to the flash reward bridge and returns the float score.
    """
    return (
        '"""flash reward bridge shim (generated). posts each completion to the flash worker."""\n'
        "import json\n"
        "import os\n"
        "import urllib.error\n"
        "import urllib.request\n"
        "\n"
        f"_URL = os.environ.get({url_env!r}, '')\n"
        "\n"
        "\n"
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    idx = (extra_info or {}).get('index')\n"
        "    if idx is None:\n"
        "        raise RuntimeError('flash reward bridge received no example index')\n"
        "    if not _URL:\n"
        "        raise RuntimeError('flash reward bridge url is not configured')\n"
        "    if isinstance(idx, bool) or getattr(getattr(idx, 'dtype', None), 'kind', None) == 'b':\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx)\n"
        "    try:\n"
        "        exact_idx = int(idx)\n"
        "    except (TypeError, ValueError, OverflowError) as exc:\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx) from exc\n"
        "    if exact_idx != idx:\n"
        "        raise RuntimeError('flash reward bridge received an invalid example index: %r' % idx)\n"
        "    idx = exact_idx\n"
        "    body = json.dumps({'index': idx, 'solution_str': solution_str or ''}).encode()\n"
        "    req = urllib.request.Request(_URL, data=body, headers={'Content-Type': 'application/json'})\n"
        "    try:\n"
        "        with urllib.request.urlopen(req, timeout=120) as r:\n"
        "            payload = json.loads(r.read().decode())\n"
        "            return float(payload['score'])\n"
        "    except urllib.error.URLError as exc:\n"
        "        raise RuntimeError('flash reward bridge request failed: %s' % exc) from exc\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError('flash reward bridge returned an invalid response: %s' % exc) from exc\n"
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
def start_reward_server(score_by_index, *, example_count: int):
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
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n).decode("utf-8"))
                index = int(payload["index"])
                if index < 0 or index >= example_count:
                    raise IndexError(f"reward example index {index} is outside [0, {example_count})")
                with score_lock:
                    score = float(score_by_index(index, payload.get("solution_str", "")))
            except Exception as exc:
                print(f"[rl-verl] reward server request failed: {exc}", flush=True)
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            else:
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


class _VerlResumeUploader:
    """stream each completed verl checkpoint to hf so a preempted grpo run can resume from it.

    grpo publishes only a final deployable adapter (both backends), so unlike sft this uploads the
    resume state alone. verl writes global_step_N under default_local_dir and only then advances
    latest_checkpointed_iteration.txt, so gating on that marker never uploads a half-written dir.
    """

    def __init__(self, local_dir: str, *, resume_step: int) -> None:
        self.local_dir = local_dir
        # whatever this run resumed from is already durable; re-uploading it would waste the
        # upload slot on state hf already holds.
        self.processed_steps: set[int] = {resume_step} if resume_step else set()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=600)
        if self._thread.is_alive():
            raise RuntimeError("verl resume uploader did not stop")

    def _completed_step(self) -> int:
        tracker = os.path.join(self.local_dir, "latest_checkpointed_iteration.txt")
        try:
            with open(tracker) as file:
                return int(file.read().strip())
        except (FileNotFoundError, OSError, ValueError):
            return 0

    def _pending(self, completed_step: int) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        try:
            names = os.listdir(self.local_dir)
        except OSError:
            return found
        for name in names:
            match = re.fullmatch(r"global_step_(\d+)", name)
            if match is None:
                continue
            step = int(match.group(1))
            path = os.path.join(self.local_dir, name)
            if step <= completed_step and step not in self.processed_steps and os.path.isdir(path):
                found.append((step, path))
        return sorted(found)

    def _run(self) -> None:
        # a failed resume upload must not fail the run: the policy is still trained and published at
        # the end, and the only loss is having to restart from an earlier step after a preemption.
        while True:
            for step, path in self._pending(self._completed_step()):
                try:
                    _w.upload_resume_checkpoint(step, path)
                except Exception as error:
                    print(f"[rl-verl] resume checkpoint upload failed at step {step}: {error}", flush=True)
                self.processed_steps.add(step)
            if self._stop.is_set() and not self._pending(self._completed_step()):
                return
            time.sleep(0.5)


def _restore_verl_resume(local_dir: str) -> int:
    """stage this run's streamed resume checkpoint into local_dir; return the step it resumes at.

    the resume artifact is keyed on the run prefix, not the job type, so the control plane hands
    grpo the same ``checkpoint-N`` layout it hands sft. verl finds it via
    latest_checkpointed_iteration.txt under trainer.default_local_dir once resume_mode=auto.
    returns 0 when there is nothing to resume, which is the ordinary fresh-run path.
    """
    resume = _w.hf_resume_checkpoint()
    if not resume:
        return 0
    match = re.fullmatch(r"checkpoint-(\d+)", os.path.basename(resume))
    if match is None:
        raise RuntimeError(f"invalid GRPO resume checkpoint path {resume!r}")
    step = int(match.group(1))
    target = os.path.join(local_dir, f"global_step_{step}")
    shutil.copytree(resume, target, dirs_exist_ok=True)
    with open(os.path.join(local_dir, "latest_checkpointed_iteration.txt"), "w") as file:
        file.write(str(step))
    return step


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
    mask_truncated_completions = _w.grpo_mask_truncated_completions(_t)
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
        # a patterned adapter trains some modules at higher rank than the base `r`; verl allocates
        # one uniform rank, so it must cover the MAXIMUM prepared rank or the load truncates.
        _ranks = [int(_src_cfg.get("r", lora_rank))]
        _ranks += [int(v) for v in (_src_cfg.get("rank_pattern") or {}).values()]
        lora_rank = max(_ranks)
        _alphas = [int(_src_cfg.get("lora_alpha", lora_alpha))]
        _alphas += [int(v) for v in (_src_cfg.get("alpha_pattern") or {}).values()]
        lora_alpha = max(_alphas)
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
    requested_len = int(train_ctx or max(1024, rl.max_prompt_len + max_completion))
    # clamp to the architecture BEFORE deriving the prompt budget, so every downstream length agrees.
    # clamping only the engine would admit prompts up to the unclamped budget and then fail them at
    # rollout, and would let the token budget pack more than the one-sequence memory floor intends.
    vllm_max_len = clamp_engine_len(
        requested_len, model_max_position_embeddings(model_id, model_revision)
    )
    if vllm_max_len < requested_len:
        print(
            f"[rl-verl] max_context_tokens {requested_len} exceeds the {model_id} context limit; "
            f"training at {vllm_max_len}",
            flush=True,
        )
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
    # verl stops at both total_epochs and total_training_steps. give its hardcoded drop_last
    # dataloader enough epoch capacity to serve the horizon, but preserve the configured epoch count
    # for user-facing metadata. total_training_steps prevents the capacity headroom from over-training.
    verl_total_epochs = _verl_epochs_for_horizon(
        epochs=epochs,
        prompt_count=len(prompts),
        prompts_per_step=prompts_per_step,
        steps=int(steps),
    )
    save_every = int(_t.save_every) if (_t and _t.save_every) else 20

    return {
        "env": env,
        "tok": tok,
        "model_id": model_id,
        "model_revision": model_revision,
        "prompts": prompts,
        "prompts_per_step": prompts_per_step,
        "group_size": group_size,
        "mask_truncated_completions": mask_truncated_completions,
        "temperature": temperature,
        "top_p": float(rl.sampling_top_p),
        "think_penalty": think_penalty,
        "kl_coef": kl_coef,
        "max_completion": max_completion,
        "max_prompt_len": prompt_budget,
        # the engine's full sequence length (prompt + completion), already clamped to the model's
        # own limit. sizes vllm's kv cache and the training token budget, and prompt_budget above is
        # derived from this same value so all three lengths agree.
        "engine_len": vllm_max_len,
        "prompt_opened_thinking": prompt_opened_thinking,
        "lr": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "warmstart_adapter": warmstart_adapter,
        "epochs": epochs,
        "verl_total_epochs": verl_total_epochs,
        "steps": int(steps),
        "save_every": save_every,
        # verl's default preserves the update horizon and on-policy baseline; unlike trl reuse, each
        # update gets a fresh rollout batch.
        "ppo_epochs": 1,
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
    model_path_for_verl = inp["model_id"]
    if inp["model_revision"]:
        _w.prefetch_model(inp["model_id"], revision=inp["model_revision"])
        # verl resolves model.path offline against the HF cache; a bare repo id would pick the
        # cached "main" ref, not the pin. hand verl the pinned revision's snapshot dir instead.
        from huggingface_hub import snapshot_download as _snap

        from flash.engine.worker.hf import _shared_weight_cache_dir

        # prefetch_model lands pinned weights on the shared volume when attached; resolve from the
        # same cache_dir it used, else fall back to the ephemeral default cache.
        _shared = _shared_weight_cache_dir()
        try:
            model_path_for_verl = _snap(
                inp["model_id"], revision=inp["model_revision"],
                cache_dir=_shared, local_files_only=True,
            )
        except Exception:
            model_path_for_verl = _snap(
                inp["model_id"], revision=inp["model_revision"], local_files_only=True
            )
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
    # a retry reuses the pod workdir; stale global_step_N dirs from a prior attempt would satisfy
    # _latest_global_step_dir and publish an old policy as if this attempt trained it.
    shutil.rmtree(local_dir, ignore_errors=True)
    # restore after the wipe, never before: the wipe is what makes a stale local dir safe, and the
    # resume checkpoint is the one global_step_N this attempt is entitled to start from.
    os.makedirs(local_dir, exist_ok=True)
    resume_step = _restore_verl_resume(local_dir)
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

    server, reward_url = start_reward_server(_score, example_count=len(rollout_examples))
    # bound before the try so the finally can always ask whether it was started.
    resume_uploader: _VerlResumeUploader | None = None
    try:
        python_bin = resolve_verl_python(
            workdir, install_wandb=bool(os.environ.get("WANDB_API_KEY"))
        )
        # masking truncated completions is a fork-only rollout field. FLASH_VERL_PYTHON can point at a
        # stock verl, and hydra would compose the unknown key only to abort in dataclass conversion.
        # fail here with the cause instead, and never silently train on truncated completions.
        if inp["mask_truncated_completions"] and not verl_supports_rollout_field(
            python_bin, "mask_truncated_completions"
        ):
            raise RuntimeError(
                f"grpo requested mask_truncated_completions but the verl at {python_bin} does not "
                "support it. that verl predates the freesolo fork; point FLASH_VERL_PYTHON at an "
                f"interpreter with '{VERL_REQUIREMENT}' installed, or unset it to provision one."
            )
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
            fp8_kv = _cc_ok and not model_is_gdn_hybrid(
                inp["model_id"], revision=inp["model_revision"]
            )
        except Exception:  # no cuda / probe failure -> conservative bf16 kv
            fp8_kv = False
        cfg = _build_verl_training_cfg(
            inp,
            train_files=train_pq,
            val_files=val_pq,
            model_id=model_path_for_verl,
            thinking=bool(_w.THINKING),
            loggers=loggers,
            fp8_kv=fp8_kv,
            reward_path=reward_py,
            local_dir=local_dir,
            n_gpus=gpu_count_of(_w.JOB_SPEC),
        )
        overrides = build_verl_overrides(cfg)

        setup_seconds = time.time() - t_start
        _w.heartbeat("rl_train_start", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
        _w.heartbeat("rl_step", step=0, initial=True)
        t_train = time.time()
        resume_uploader = _VerlResumeUploader(local_dir, resume_step=resume_step)
        resume_uploader.start()
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
        resp_len_re = re.compile(r"response_length/mean:([\d.eE+-]+)")
        resp_len_history: list[float] = []
        loss_curve: list[float] = []
        last_dump_step = [-1]
        with liveness_heartbeat("rl_step", progress=_progress, progress_step=True):
            proc = subprocess.Popen(
                [python_bin, "-m", "verl.trainer.main_ppo", *overrides],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env_for_verl,
                start_new_session=True,
            )
            try:
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
                    for pat, sink in (
                        (reward_re, reward_history),
                        (loss_re, loss_curve),
                        (resp_len_re, resp_len_history),
                    ):
                        hit = pat.search(line)
                        if hit:
                            with contextlib.suppress(ValueError):
                                sink.append(float(hit.group(1)))
                rc = proc.wait()
            except BaseException:
                # the stream loop died (upload error, cancel, oom in the parent): a still-running
                # verl child would keep burning the gpu unattended — kill its whole process group.
                with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                    os.killpg(os.getpgid(proc.pid), 15)
                with contextlib.suppress(Exception):
                    proc.wait(timeout=10)
                raise
        if rc != 0:
            raise RuntimeError(f"verl.trainer.main_ppo exited {rc}; see the flash log for the traceback")
    finally:
        # drain before the reward server goes down: on a cancel or crash the last completed
        # checkpoint is exactly the one a retry needs, so it is worth uploading on the way out.
        if resume_uploader is not None:
            with contextlib.suppress(Exception):
                resume_uploader.stop()
        server.shutdown()

    # collect verl's lora checkpoint -> flash-servable peft adapter, then reuse flash finalize.
    out_dir = f"/tmp/rl_seed{_w.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    shutil.rmtree(adapter_dir, ignore_errors=True)
    os.makedirs(adapter_dir, exist_ok=True)
    train_wall = time.time() - t_train
    if not reward_history:
        raise RuntimeError(
            "verl reported no reward metrics for the whole run — the flash reward bridge was "
            "never consulted (wiring regression); refusing to publish a policy trained on "
            "default rewards"
        )
    actor_dir, steps_run = _latest_global_step_dir(local_dir)
    if steps_run < expected_steps:
        raise RuntimeError(f"grpo completed {steps_run}/{expected_steps} requested optimizer updates")

    with liveness_heartbeat(
        "rl_finalizing",
        progress=lambda: steps_run,
        progress_step=True,
        keepalive=True,
    ):
        _export_peft_adapter(actor_dir, adapter_dir, base_model_id=inp["model_id"], python_bin=python_bin)
        tok.save_pretrained(adapter_dir)
        _stamp_adapter_dir_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
        _w.write_base_model_provenance(adapter_dir, inp["model_id"], inp["model_revision"])
        _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        _w.publish_deployable_checkpoint(adapter_dir, steps_run)

    _w.heartbeat("rl_trained", train_wall=train_wall, step=steps_run, gpu=gpu_diagnostics())
    _w.write_train_meta(
        phase="rl",
        adapter_dir=adapter_dir,
        model_id=inp["model_id"],
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=0,
        generated_tokens=int(
            sum(resp_len_history) / max(1, len(resp_len_history))
            * steps_run * inp["prompts_per_step"] * inp["group_size"]
        )
        if resp_len_history
        else 0,
        notes=_build_verl_train_notes(
            inp,
            steps_run=steps_run,
            retained_prompts=len(prompts),
            reward_history=reward_history,
            loss_curve=loss_curve,
            resumed=bool(resume_step),
        ),
    )
