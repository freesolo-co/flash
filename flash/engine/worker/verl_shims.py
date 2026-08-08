"""Source for sitecustomize shims rendered into the verl child interpreter.

Renderers return source text because the pinned child cannot import flash. Active shims print a
marker so the parent can prove each patch executed.
"""

from __future__ import annotations

_ENTROPY_QUANTILE_MARKER = "[flash-verl] top-entropy token masking active"
_STOP_SEQUENCES_MARKER = "[flash-verl] rollout stop strings active"
_STRUCTURED_OUTPUTS_MARKER = "[flash-verl] rollout structured outputs active"
_EXACT_SAVE_STEPS_MARKER = "[flash-verl] exact save steps active"
_IMAGE_PAD_BAN_MARKER = "[flash-verl] image-pad token banned from rollouts"
_KL_REF_ADAPTER_MARKER = "[flash-verl] kl reference anchored to the warm-start adapter"


def render_kl_ref_adapter_shim(warmstart: bool) -> str:
    """return source that anchors verl's kl reference to the warm-start adapter.

    ``no_lora_adapter=True`` otherwise evaluates the bare base and pulls a warm start backward.
    Store the frozen reference as non-persistent buffers so FSDP, optimizers, state dicts, and
    ``base_model_merger.save_lora_adapter`` never treat it as a second trainable adapter. Rebuild it
    from ``lora_adapter_path`` on resume. Swap ``BaseTunerLayer._active_adapter`` directly because
    peft's ``set_adapter`` changes ``requires_grad`` and breaks FSDP flat-parameter uniformity.
    """
    if not warmstart:
        return ""
    return f'''
from contextlib import contextmanager as _flash_ref_contextmanager

import torch.nn as _flash_ref_nn
from peft.tuners.tuners_utils import BaseTunerLayer as _flash_ref_tuner_layer
from verl.workers.engine.fsdp import transformer_impl as _flash_ref_impl

_FLASH_REF_ADAPTER = "flash_kl_ref"


def _flash_ref_snapshot(module):
    """freeze a copy of the warm-start adapter under a second, non-trainable adapter name."""
    if _FLASH_REF_ADAPTER in getattr(module, "peft_config", {{}}):
        return module
    module.add_adapter(_FLASH_REF_ADAPTER, module.peft_config[module.active_adapter])
    for name, param in module.named_parameters():
        if ".default." in name:
            twin = module.get_parameter(name.replace(".default.", f".{{_FLASH_REF_ADAPTER}}."))
            twin.data.copy_(param.data)
    # demote the snapshot from parameters to non-persistent buffers. lora_A/lora_B are ModuleDicts
    # keyed by adapter name, so the snapshot's leaves are exactly the ones under our key.
    demoted = 0
    for container in module.modules():
        if isinstance(container, _flash_ref_nn.ModuleDict) and _FLASH_REF_ADAPTER in container:
            leaf = container[_FLASH_REF_ADAPTER]
            for attr, value in list(leaf.named_parameters(recurse=False)):
                frozen = value.detach().clone()
                delattr(leaf, attr)
                leaf.register_buffer(attr, frozen, persistent=False)
                demoted += 1
    if not demoted:
        raise RuntimeError("flash kl reference snapshot found no adapter weights to freeze")
    print({_KL_REF_ADAPTER_MARKER!r} + " " + repr(demoted), flush=True)
    return module


_flash_ref_original_build_lora = _flash_ref_impl.FSDPEngine._build_lora_module


def _flash_ref_build_lora_module(self, module):
    # after the warm-start adapter is loaded, before _build_fsdp_module wraps it.
    return _flash_ref_snapshot(_flash_ref_original_build_lora(self, module))


@_flash_ref_contextmanager
def _flash_ref_use_ref_adapter(module):
    """activate the frozen snapshot without touching any requires_grad flag."""
    layers = [m for m in module.modules() if isinstance(m, _flash_ref_tuner_layer)]
    if not layers:
        raise RuntimeError("flash kl reference: no lora layers on the actor module")
    saved = [layer._active_adapter for layer in layers]
    for layer in layers:
        layer._active_adapter = [_FLASH_REF_ADAPTER]
    try:
        yield
    finally:
        for layer, previous in zip(layers, saved, strict=True):
            layer._active_adapter = previous


def _flash_ref_disable_adapter(self):
    # this shim is only written for a warm start, so the snapshot must exist. falling back to the
    # stock disable_adapter() here would silently anchor the kl term to the base -- the exact
    # defect this patch removes -- and the run would look healthy while training the wrong thing.
    module = self.module
    inner = getattr(module, "_fsdp_wrapped_module", module)
    if _FLASH_REF_ADAPTER not in getattr(inner, "peft_config", {{}}):
        raise RuntimeError(
            "flash kl reference adapter missing: expected " + _FLASH_REF_ADAPTER + " on the actor"
        )
    return _flash_ref_use_ref_adapter(module)


_flash_ref_impl.FSDPEngine._build_lora_module = _flash_ref_build_lora_module
_flash_ref_impl.FSDPEngine.disable_adapter = _flash_ref_disable_adapter
'''


def render_structured_outputs_shim(structured_outputs: dict | None) -> str:
    """return source that constrains verl rollout sampling to a guided grammar.

    Wrap the value in ``StructuredOutputsParams``: vllm silently accepts a raw dict but applies no
    constraint. The object survives ``server.generate.remote(...)`` through Ray cloudpickle. The
    reasoning-parser engine override remains in ``_build_verl_overrides``.
    """
    if not structured_outputs:
        return ""
    return f'''
from verl.experimental.agent_loop import agent_loop as _flash_so_agent_loop
from vllm.sampling_params import StructuredOutputsParams as _FlashStructuredOutputsParams

_flash_structured_outputs = {structured_outputs!r}


def _flash_patch_structured_outputs():
    """add ``structured_outputs`` to the per-sample sampling params on their way into the loop.

    patched on ``_run_agent_loop`` for the same reason as the stop strings: it receives the
    per-sample dict after verl's validate/greedy overrides, so the constraint also applies to
    validation rollouts -- as on the retired trl path, where it lived in generation_kwargs and was not swapped out
    for eval.
    """
    original = _flash_so_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        # build a fresh params object per request: vllm mutates sampling params in place (it
        # resolves the structured-outputs backend on first use and caches it on the instance), so a
        # shared one would leak that resolution across requests.
        params["structured_outputs"] = _FlashStructuredOutputsParams(**_flash_structured_outputs)
        return await original(self, params, *args, **kwargs)

    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


if not getattr(
    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_so_patched", False
):
    _flash_patch_structured_outputs()
    _flash_so_agent_loop.AgentLoopWorker._run_agent_loop._flash_so_patched = True
    print({_STRUCTURED_OUTPUTS_MARKER!r} + " " + repr(_flash_structured_outputs), flush=True)
'''


def render_exact_save_steps_shim(save_at_steps: tuple[int, ...], total_steps: int) -> str:
    """return source that suppresses verl's extra checkpoint writes.

    verl's gcd ``save_freq`` writes a superset of arbitrary ``save_at_steps``. Keep only requested
    and final steps. Read ``self.global_steps`` because ``RayPPOTrainer._save_checkpoint`` has no
    step argument; skipped steps must not advance ``latest_checkpointed_iteration.txt``.
    """
    if not save_at_steps:
        return ""
    return f'''
from verl.trainer.ppo import ray_trainer as _flash_save_ray_trainer

_flash_required_save_steps = frozenset({tuple(sorted(save_at_steps))!r})
_flash_total_steps = {int(total_steps)}

def _flash_patch_exact_save_steps():
    """save only the steps flash asked for, plus the final step.

    the gcd interval makes verl save a superset; every extra write is a full-state dump that is
    never published and is pruned again a few steps later. the last step stays because the run's
    final publish reads the checkpoint verl writes there.
    """
    original = _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint

    def _save_checkpoint(self):
        step = int(self.global_steps)
        if step not in _flash_required_save_steps and step != _flash_total_steps:
            return None
        return original(self)

    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint = _save_checkpoint

if not getattr(
    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint, "_flash_save_patched", False
):
    _flash_patch_exact_save_steps()
    _flash_save_ray_trainer.RayPPOTrainer._save_checkpoint._flash_save_patched = True
    print(
        {_EXACT_SAVE_STEPS_MARKER!r} + " " + repr(sorted(_flash_required_save_steps))
        + " final=" + repr(_flash_total_steps),
        flush=True,
    )
'''


def render_stop_sequences_shim(stop_sequences: tuple[str, ...]) -> str:
    """return source that adds flash stop strings to verl rollouts.

    Patch ``_run_agent_loop`` where the per-sample dict is owned and passed into vllm
    ``SamplingParams``. vllm truncates text but preserves ``token_ids``, so trained tokens retain
    trailing delimiter tokens.
    """
    if not stop_sequences:
        return ""
    return f'''
from verl.experimental.agent_loop import agent_loop as _flash_agent_loop

_flash_stop_sequences = {list(stop_sequences)!r}


def _flash_patch_run_agent_loop():
    """add ``stop`` to the per-sample sampling params on their way into the agent loop.

    ``_run_agent_loop`` receives the fully-built dict for one sample, after verl has applied its
    validation/greedy overrides. patching here rather than at dict construction means the stop
    strings survive those overrides and apply to validation rollouts too, as on the retired trl path, where the
    stop list lives in generation_kwargs and is not swapped out for eval.
    """
    original = _flash_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        params["stop"] = list(_flash_stop_sequences)
        return await original(self, params, *args, **kwargs)

    _flash_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


# patch once. wrapping twice would be harmless here (the key is overwritten, not appended) but the
# guard keeps the behavior obvious and matches the entropy shim.
if not getattr(_flash_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_stop_patched", False):
    _flash_patch_run_agent_loop()
    _flash_agent_loop.AgentLoopWorker._run_agent_loop._flash_stop_patched = True
    print({_STOP_SEQUENCES_MARKER!r} + " " + repr(_flash_stop_sequences), flush=True)
'''


def render_image_pad_ban_shim(image_pad_token_id: int | None) -> str:
    """return source that prevents multimodal rollouts from emitting the image-pad token.

    A sampled placeholder has no image and breaks processor alignment on the next forward pass.
    Apply ``-100.0`` logit bias to every row in a multimodal job, including text-only rows.
    """
    if image_pad_token_id is None:
        return ""
    return f"""
from verl.experimental.agent_loop import agent_loop as _flash_image_agent_loop

_flash_image_pad_token_id = {int(image_pad_token_id)!r}


def _flash_patch_image_pad_ban():
    original = _flash_image_agent_loop.AgentLoopWorker._run_agent_loop

    async def _run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        logit_bias = dict(params.get("logit_bias") or {{}})
        logit_bias[_flash_image_pad_token_id] = -100.0
        params["logit_bias"] = logit_bias
        return await original(self, params, *args, **kwargs)

    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop = _run_agent_loop


# patch once, and independently of the stop-strings patch: both wrap the same method, so each
# needs its own marker attribute or the second would be skipped by the first one's flag.
if not getattr(
    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop, "_flash_image_pad_patched", False
):
    _flash_patch_image_pad_ban()
    _flash_image_agent_loop.AgentLoopWorker._run_agent_loop._flash_image_pad_patched = True
    print({_IMAGE_PAD_BAN_MARKER!r} + " " + repr(_flash_image_pad_token_id), flush=True)
"""


def render_per_turn_credit_shim(per_turn_credit: bool) -> str:
    """return source that adds per-turn group-relative credit to verl.

    Wrap ``compute_advantage`` because custom estimators do not receive ``non_tensor_batch`` spans.
    Keep stock GRPO as the baseline and overwrite token advantages. Fallback must cover the whole
    group so episode and per-turn scales are never mixed.
    """
    if not per_turn_credit:
        return ""
    return '''
import torch as _flash_pt_torch
from verl.trainer.ppo import ray_trainer as _flash_pt_ray_trainer

_flash_pt_original_compute_advantage = _flash_pt_ray_trainer.compute_advantage
_flash_pt_logged = False


def _flash_pt_rows(non_tensor_batch, batch_size):
    """per-row (spans, turns) or None when this batch carries no usable per-turn metadata."""
    spans_column = non_tensor_batch.get("flash_turn_spans")
    rewards_column = non_tensor_batch.get("flash_turn_rewards")
    if spans_column is None or rewards_column is None:
        return None
    if len(spans_column) != batch_size or len(rewards_column) != batch_size:
        return None
    rows = []
    for spans, turns in zip(spans_column, rewards_column):
        if spans is None or turns is None or len(spans) != len(turns):
            # a row the loop could not align. keep the row so its group can be identified and
            # dropped whole, rather than silently centring the rest against a smaller sample.
            rows.append(None)
            continue
        rows.append(
            (
                tuple((int(start), int(end)) for start, end in spans),
                tuple(float(value) for value in turns),
            )
        )
    return rows


def _flash_pt_per_turn_advantages(rows, index, episode_advantages):
    """centre each turn against the same turn of its group; returns [B, width].

    starts from stock grpo's own output and overwrites only the rows of groups that earned
    per-turn credit. a group that falls back therefore keeps the exact tensor grpo produced rather
    than a reconstruction of it -- there is no scalar to recover, so nothing can drift.
    """
    advantages = episode_advantages.clone()
    groups = {}
    for row_index, uid in enumerate(index):
        groups.setdefault(uid, []).append(row_index)
    for member_indexes in groups.values():
        if any(rows[row_index] is None for row_index in member_indexes):
            continue
        for row_index in member_indexes:
            advantages[row_index] = 0.0
        turn_total = max(len(rows[row_index][1]) for row_index in member_indexes)
        for turn_index in range(turn_total):
            scoring = [
                row_index
                for row_index in member_indexes
                if turn_index < len(rows[row_index][1])
                and rows[row_index][0][turn_index][1] > rows[row_index][0][turn_index][0]
            ]
            if not scoring:
                # every member emitted nothing for this turn; an empty turn carries no signal and
                # must not skew the baseline for the members that did emit one.
                continue
            baseline = sum(rows[row_index][1][turn_index] for row_index in scoring) / len(scoring)
            for row_index in scoring:
                start, end = rows[row_index][0][turn_index]
                advantages[row_index, start:end] = rows[row_index][1][turn_index] - baseline
    return advantages


def _flash_pt_compute_advantage(data, *args, **kwargs):
    global _flash_pt_logged
    data = _flash_pt_original_compute_advantage(data, *args, **kwargs)
    episode = data.batch.get("advantages")
    if episode is None or episode.dim() != 2:
        return data
    batch_size, width = episode.shape
    rows = _flash_pt_rows(data.non_tensor_batch, batch_size)
    if rows is None or all(row is None for row in rows):
        return data
    index = data.non_tensor_batch.get("uid")
    if index is None or len(index) != batch_size:
        return data
    for row in rows:
        if row is None:
            continue
        for start, end in row[0]:
            if not 0 <= start <= end <= width:
                raise ValueError(
                    f"turn span [{start}, {end}) exceeds the response width {width}"
                )
    advantages = _flash_pt_per_turn_advantages(rows, index, episode)
    if not bool(_flash_pt_torch.isfinite(advantages).all()):
        raise ValueError("per-turn advantages must be finite")
    # keep the response mask authoritative: glue tokens sit inside a turn span only when the
    # environment reply was appended mid-turn, and they must never carry gradient.
    response_mask = data.batch.get("response_mask")
    if response_mask is not None:
        advantages = advantages * response_mask.to(dtype=advantages.dtype)
    data.batch["advantages"] = advantages
    # returns feeds the critic, which grpo does not use; stock grpo sets it to the same tensor.
    data.batch["returns"] = advantages
    if not _flash_pt_logged:
        print("[rl-verl] multi-turn per-turn group-relative credit is active", flush=True)
        _flash_pt_logged = True
    return data


_flash_pt_ray_trainer.compute_advantage = _flash_pt_compute_advantage
'''


def render_reentrant_checkpointing_shim(reentrant: bool, *, multimodal: bool = False) -> str:
    """return source that makes verl gradient checkpointing reentrant.

    verl hardcodes ``use_reentrant=False`` at ``workers/engine/fsdp/transformer_impl.py:304``;
    MoE and GDN metadata changes make that fail on the first backward. Patch only the kwarg because
    ``transformer_impl.py:433-434`` also uses the enable flag for activation offloading.

    Reentrant LoRA needs ``enable_input_require_grads()`` or frozen embeddings drop all adapter
    gradients (GRAD-001). Multimodal runs also need the vision patch-embed hook in
    ``tests/test_multimodal_input_grads.py``.
    """
    if not reentrant:
        return ""
    # 8 spaces, not 4: this is interpolated INSIDE the `if enable_gradient_checkpointing` body
    # below. at 4 it dedents out of the block and the rendered sitecustomize is a SyntaxError, so
    # every multimodal reentrant run dies before the shim can do anything.
    vision_hook = (
        """
        _flash_install_vision_input_grads(module)"""
        if multimodal
        else ""
    )
    vision_helper = (
        '''

def _flash_install_vision_input_grads(module):
    """mark the vision patch-embed output as requiring grad; see the docstring above."""
    import torch as _flash_vision_torch

    _get_base = getattr(module, "get_base_model", None)
    if callable(_get_base):
        _base = _get_base()
    else:
        _peft_base = getattr(module, "base_model", None)
        _base = getattr(_peft_base, "model", module)

    def _flash_require_output_grad(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) and output else output
        if isinstance(tensor, _flash_vision_torch.Tensor) and tensor.is_floating_point():
            tensor.requires_grad_(True)

    for _path, _submodule in _base.named_modules():
        if _path.endswith("visual.patch_embed"):
            _submodule.register_forward_hook(_flash_require_output_grad)
            print(f"[rl-verl] vision input gradients enabled at {_path}", flush=True)
            return
    print("[rl-verl] no visual.patch_embed found; vision input gradients not installed", flush=True)
'''
        if multimodal
        else ""
    )
    return f"""
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine as _FlashReentrantEngine

_flash_reentrant_original_build_module = _FlashReentrantEngine._build_module
{vision_helper}

def _flash_reentrant_build_module(self):
    module = _flash_reentrant_original_build_module(self)
    # only when verl actually enabled checkpointing: calling gradient_checkpointing_enable on a
    # model verl deliberately left uncheckpointed would turn it ON and change the memory profile.
    if getattr(self.model_config, "enable_gradient_checkpointing", False):
        # the LANGUAGE-side counterpart of the vision hook below, and required for the same
        # reason (GRAD-001): lora freezes the embeddings, so nothing entering the first
        # checkpointed decoder layer requires grad and reentrant recompute drops the backward
        # for the whole segment -- where every lora parameter lives. the vision hook only ever
        # covered the patch embeddings on multimodal runs; text-only runs had no hook at all and
        # trained nothing while reporting success.
        module.enable_input_require_grads()
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={{"use_reentrant": True}}){vision_hook}
        print("[rl-verl] reentrant gradient checkpointing is active", flush=True)
    return module


_FlashReentrantEngine._build_module = _flash_reentrant_build_module
"""


def render_entropy_quantile_shim(entropy_quantile: float | None) -> str:
    """return source that adds top-entropy token masking to verl.

    Patch ``ppo_loss`` because registered policy losses cannot see per-token entropy. Mask only the
    policy-gradient term; KL and entropy bonus keep the full response mask. Flash pins
    ``seq-mean-token-sum-norm`` so removing numerator tokens does not rescale the rest.
    """
    if entropy_quantile is None or float(entropy_quantile) >= 1.0:
        return ""
    threshold = 1.0 - float(entropy_quantile)
    return f'''
import threading as _flash_threading

import torch as _flash_torch
import torch.distributed as _flash_dist
from verl.workers.utils import losses as _flash_losses
from verl.workers.utils.padding import no_padding_2_padding as _flash_no_padding_2_padding

_flash_entropy_threshold_q = {threshold!r}
_flash_original_ppo_loss = _flash_losses.ppo_loss
_flash_original_get_policy_loss_fn = _flash_losses.get_policy_loss_fn
# carries this micro-batch's padded entropy from ppo_loss down to the policy loss, which verl
# never passes it. thread-local rather than a module global so the handoff cannot leak across
# concurrent loss calls, and always cleared in a finally.
_flash_entropy_state = _flash_threading.local()


def _flash_high_entropy_mask(entropy, response_mask):
    """keep tokens at or above the global entropy quantile."""
    local = entropy[response_mask.bool()].float().reshape(-1)
    # the quantile is over the whole global batch, not the local shard, so every rank thresholds
    # identically. verl's actor shards over the default process group, so the quantile must be global.
    if _flash_dist.is_available() and _flash_dist.is_initialized() and _flash_dist.get_world_size() > 1:
        sizes = [_flash_torch.zeros(1, dtype=_flash_torch.long, device=local.device)
                 for _ in range(_flash_dist.get_world_size())]
        _flash_dist.all_gather(sizes, _flash_torch.tensor([local.numel()], dtype=_flash_torch.long, device=local.device))
        largest = int(max(int(s.item()) for s in sizes))
        if largest == 0:
            return _flash_torch.zeros_like(entropy, dtype=_flash_torch.bool)
        # pad with a negative sentinel: entropy is non-negative, so it can never collide with a
        # real value and is dropped after the gather.
        padded = _flash_torch.full((largest,), -1e9, dtype=_flash_torch.float32, device=local.device)
        padded[: local.numel()] = local
        buckets = [_flash_torch.empty_like(padded) for _ in range(_flash_dist.get_world_size())]
        _flash_dist.all_gather(buckets, padded)
        gathered = _flash_torch.cat(buckets)
        gathered = gathered[gathered != -1e9]
    else:
        gathered = local
    if gathered.numel() == 0:
        return _flash_torch.zeros_like(entropy, dtype=_flash_torch.bool)
    cutoff = _flash_torch.quantile(gathered, _flash_entropy_threshold_q)
    return ((entropy * response_mask.float()) >= cutoff) & response_mask.bool()


def _flash_masked_policy_loss_fn(loss_mode):
    """wrap the resolved policy loss so it sees the entropy-masked response mask.

    masking here rather than in ppo_loss keeps the mask on the policy-gradient term only: ppo_loss
    aggregates the kl and entropy-bonus terms against its own unmasked response_mask, exactly as
    the kl term is added after multiplying per_token_loss by the entropy mask.
    """
    inner = _flash_original_get_policy_loss_fn(loss_mode)

    def _masked(*args, **kwargs):
        entropy = getattr(_flash_entropy_state, "entropy", None)
        response_mask = kwargs.get("response_mask", None)
        if entropy is not None and response_mask is not None:
            kwargs["response_mask"] = _flash_high_entropy_mask(entropy, response_mask)
        return inner(*args, **kwargs)

    return _masked


def _flash_entropy_masked_ppo_loss(config, model_output, data, dp_group=None):
    entropy = model_output.get("entropy", None)
    if entropy is None:
        raise RuntimeError(
            "train.entropy_quantile needs per-token entropy, but verl produced none; "
            "actor.calculate_entropy must be true."
        )
    # ppo_loss re-derives this internally; convert here too so the mask is padded to the same
    # (bsz, response_len) shape the policy loss receives its response_mask in.
    _flash_entropy_state.entropy = _flash_no_padding_2_padding(entropy, data)
    try:
        return _flash_original_ppo_loss(config, model_output, data, dp_group)
    finally:
        _flash_entropy_state.entropy = None


# patch once. python imports sitecustomize a single time per interpreter, so this should not come
# up -- but wrapping an already-wrapped loss would mask the top quantile OF the top quantile and
# train on a fraction of the requested tokens, with nothing in the logs to show for it.
if not getattr(_flash_original_ppo_loss, "_flash_entropy_masked", False):
    _flash_entropy_masked_ppo_loss._flash_entropy_masked = True
    _flash_losses.get_policy_loss_fn = _flash_masked_policy_loss_fn
    _flash_losses.ppo_loss = _flash_entropy_masked_ppo_loss
    print({_ENTROPY_QUANTILE_MARKER!r} + " quantile={entropy_quantile:g}", flush=True)
'''


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
        "    req = urllib.request.Request(\n"
        "        _URL.rstrip('/') + '/score', data=body, headers={'Content-Type': 'application/json'}\n"
        "    )\n"
        "    try:\n"
        # NO client deadline. verl fans this call out hard: RewardLoopManager builds
        # reward.num_workers (8) ray workers unconditionally on the grpo path
        # (ray_trainer.py:901-910), and each one asyncio.gathers every row in its chunk
        # (reward_loop.py:138-143). start_reward_server coalesces those requests behind one scoring
        # thread, so a per-request timeout would still bound QUEUE WAIT as well as the env call -- a
        # caller can fail for arriving behind a slow-but-healthy judge batch.
        # a wedged env is caught by the training stall watchdog instead (STALL_AFTER_S=1500s in
        # providers/_poll.py), which measures training progress rather than one request.
        "        with urllib.request.urlopen(req) as r:\n"
        "            payload = json.loads(r.read().decode())\n"
        "            return float(payload['score'])\n"
        "    except urllib.error.URLError as exc:\n"
        "        raise RuntimeError('flash reward bridge request failed: %s' % exc) from exc\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError('flash reward bridge returned an invalid response: %s' % exc) from exc\n"
    )
