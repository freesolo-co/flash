"""Source for the sitecustomize shims flash renders into the verl child interpreter.

Each renderer returns PYTHON SOURCE TEXT, not behaviour: the verl child runs on incompatible
torch/vllm pins and cannot import flash, so every patch it needs has to arrive as a string that
python's automatic sitecustomize import executes at child startup. Keeping them together
separates "text flash generates" from "code flash runs" -- an important distinction, since nothing
in this module is exercised by importing it.

A shim is rendered only when its feature is active; each prints its marker so the parent can prove
from the child log that the patch actually took effect.
"""

from __future__ import annotations

_ENTROPY_QUANTILE_MARKER = "[flash-verl] top-entropy token masking active"
_STOP_SEQUENCES_MARKER = "[flash-verl] rollout stop strings active"
_STRUCTURED_OUTPUTS_MARKER = "[flash-verl] rollout structured outputs active"
_EXACT_SAVE_STEPS_MARKER = "[flash-verl] exact save steps active"
_IMAGE_PAD_BAN_MARKER = "[flash-verl] image-pad token banned from rollouts"
_KL_REF_ADAPTER_MARKER = "[flash-verl] kl reference anchored to the warm-start adapter"


def render_kl_ref_adapter_shim(warmstart: bool) -> str:
    """return the sitecustomize source that anchors verl's kl reference to the warm-start adapter.

    verl computes reference logprobs on the actor whenever lora is active (``ref_in_actor`` in
    ray_trainer.py is ``lora_rank > 0 or lora_adapter_path is not None``, always true on flash) and
    marks that call ``no_lora_adapter=True``, which engine_workers.py turns into
    ``engine.disable_adapter()`` -- the BARE BASE. for a fresh-start run that is correct. for a
    warm-started run it is not: the kl term would pull the policy back toward the base and undo the
    sft adapter the run was told to continue, which is why the warm-start + kl combination was
    refused until now. the retired trl driver instead snapshotted a frozen reference adapter and
    evaluated the reference under it; this ports that behavior.

    the snapshot is registered as NON-PERSISTENT BUFFERS rather than as a second peft adapter's
    parameters, which is what keeps it out of every downstream consumer:

    - ``named_parameters()`` never sees it, so fsdp does not flatten it and the optimizer cannot
      train it (a trainable reference would drift with the policy and anchor nothing).
    - ``state_dict()`` never sees it, so it stays out of the saved shards. that matters more than
      it looks: verl's merger does not call ``save_pretrained`` for the adapter, it hand-builds one
      from every state-dict key containing ``lora_`` and derives ``target_modules`` from
      ``key.split(".")[-3]`` (base_model_merger.save_lora_adapter). a second adapter's keys do not
      match its ``.default.weight`` rewrite, so they would resolve to ``lora_A``/``lora_B`` and ship
      a deliverable adapter with bogus target modules.
    - a resumed run reloads the same shards with ``strict=True``; absent keys would fail that load.
      the snapshot is rebuilt from ``lora_adapter_path``, which flash passes on every warm-start
      run including a resume, so the anchor re-forms identically rather than being restored.

    the swap is ``BaseTunerLayer._active_adapter``, not peft's ``set_adapter``: ``set_adapter``
    flips ``requires_grad`` on both adapters, and verl wraps fsdp1 with ``use_orig_params=false``
    where that breaks flat-param uniformity. writing ``_active_adapter`` changes zero flags and
    restores the policy forward bit-exactly.
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
    """return the sitecustomize source that constrains verl's rollout to a guided grammar.

    the sampling half of ``train.structured_outputs``. it rides the same per-sample dict as the stop
    strings, so the mechanism is identical to render_stop_sequences_shim; only the value differs.
    the engine half (``reasoning_parser``, applied when thinking is also on) is a plain hydra
    override and needs no shim -- see _build_verl_overrides.

    the value MUST be wrapped in ``StructuredOutputsParams``. vllm accepts a raw dict here, passes
    ``_verify_args()``, and then stores a plain dict with no ``.json`` attribute -- constraining
    nothing, with no error and no log line. the retired trl path got the wrapping for free from its
    colocate generation layer and so passed the spec as a plain dict; on verl nothing wraps it, so
    the shim must.

    the object survives the worker -> server hop: that hop is ``server.generate.remote(...)``, a ray
    actor rpc (cloudpickle), not http/json, so it arrives as the same dataclass it left as.
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
    """return the sitecustomize source that suppresses verl's superset checkpoint writes.

    verl only saves when ``global_steps % save_freq == 0``, so it cannot hit an arbitrary set of
    steps. the resolver picks the gcd of the required steps, which makes verl save a SUPERSET and
    the uploader publish deployables at exactly the required ones. correct, but the gcd can be
    tiny -- save_at_steps=(7, 13) gives gcd 1, a full checkpoint written every single step -- and
    each write is a full-state dump of a multi-billion-parameter policy.

    this drops the writes flash never asked for, so only the required steps (and the last step,
    which the run's final publish needs) reach disk. the sft verl backend already does exactly this
    (sft_train.py); this is the same suppression on the ppo driver.

    ``RayPPOTrainer._save_checkpoint`` takes no step argument -- it reads ``self.global_steps`` --
    so the filter reads it off the instance rather than a parameter. returning early is safe
    because the method's only other effect is advancing latest_checkpointed_iteration.txt, and a
    step with no checkpoint on disk must not be advertised as resumable: the uploader gates on that
    marker precisely so it never uploads a half-written or absent directory.
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
    """return the sitecustomize source that gives verl's rollout flash's stop-string behavior.

    on the retired trl backend flash puts ``stop`` into ``generation_kwargs``, which reaches vllm's
    ``SamplingParams`` unchanged. verl builds its sampling params as a literal dict in
    ``AgentLoopWorker.generate_sequences`` (agent_loop.py) with no stop field and no passthrough, so
    the key has to be inserted there. the value then rides the existing dict all the way into
    ``SamplingParams(max_tokens=..., **sampling_params)`` in the vllm server, which accepts it.

    the patch lands on ``_run_agent_loop`` rather than the vllm server because the worker owns the
    dict: patching further down would have to reconstruct which request the params belong to, and
    the tool/multi-turn loops pass the same dict through untouched.

    token-level semantics match the retired trl path exactly. vllm truncates ``output_text`` at a stop-string match
    but leaves ``token_ids`` intact, and both backends read ``output.token_ids`` -- so the trained
    tokens are the same on either backend, including the trailing delimiter tokens.
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
    """return the sitecustomize source that stops a multimodal rollout emitting the image-pad token.

    the vision placeholder token is a real vocabulary entry, so an unconstrained sampler can emit
    it inside a *completion*. nothing there expands it back into pixels, so the trained sequence
    then contains a token the model can only have produced by hallucinating an image -- and on the
    next forward pass the processor's image/text alignment counts a placeholder with no image
    behind it. the retired trl driver banned it through ``generation_kwargs["logit_bias"]``; verl
    builds its sampling params as a literal dict, so the key is inserted the same way the
    stop-strings shim inserts ``stop``.

    unconditional rather than gated on the row: this shim is only written for a multimodal job, and
    a text-only row in such a job still must not invent a placeholder. -100.0 is a large enough
    negative bias to make the token unreachable at any temperature this trainer allows.
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
    """return the sitecustomize source that gives verl per-turn group-relative credit.

    verl credits a whole episode: ``compute_grpo_outcome_advantage`` centres one scalar per rollout
    against its group and broadcasts it across every response token. per-turn mode instead centres
    each TURN against the same turn of its group siblings, so a good turn inside a bad episode
    still gets positive advantage.

    this wraps ``compute_advantage`` rather than registering a custom estimator. a registered
    estimator would be the tidier hook, but ``compute_advantage`` forwards ``non_tensor_batch`` to
    exactly one estimator by name (``if adv_estimator in (AdvantageEstimator.GDPO, "gdpo")``), so a
    custom one could never see the spans it needs. wrapping keeps stock grpo as the baseline and
    overwrites only the token axis, so the episode-level centring stays exactly as stock grpo
    computed it and only the per-turn refinement is layered on top.

    the fallback is per GROUP, not per row: grpo centres each rollout against its group, so a group
    holding a mix of per-turn and episode credit would compare quantities of different scales. one
    unusable row therefore drops its whole group to episode credit.
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
    """return the sitecustomize source that makes verl's gradient checkpointing REENTRANT.

    verl hardcodes ``use_reentrant=False`` at its single checkpointing site
    (``workers/engine/fsdp/transformer_impl.py:304``) and exposes no knob for it. non-reentrant
    recompute asserts that every recomputed activation's metadata matches the forward pass, which
    the MoE router and the GDN chunk-scan both violate: they save shape-/data-dependent tensors the
    recompute lays out differently, so the run dies on the FIRST backward, before a single optimizer
    step. ``grpo_use_reentrant`` documents both live-confirmed cases.

    patches ``_build_module`` and re-enables checkpointing with ``use_reentrant=True`` on the way
    out, so verl's own call still runs and only the flag differs. the same hook the SFT verl path
    uses (``sft_train.py``), against the same class: GRPO's actor is ``FSDPEngineWithLMHead``, which
    inherits ``_build_module`` from ``FSDPEngine``.

    deliberately NOT unified with SFT's version, which instead tells verl
    ``enable_gradient_checkpointing=False`` and enables checkpointing itself. GRPO cannot do that:
    verl reads the same flag a SECOND time, to decide activation offloading
    (``transformer_impl.py:433-434``), so clearing it would silently change the memory profile as
    well as the recompute flag. leaving it True and correcting only the kwarg keeps this a
    flag-level change. the guard below exists for the same reason -- ``_build_module`` also builds
    engines whose config may legitimately have checkpointing off, and enabling it there would turn
    on a feature verl chose to leave off.

    reentrant recompute drops the backward for a checkpointed block when none of that block's
    inputs require grad. that hits the LANGUAGE side on every lora run: lora freezes the
    embeddings, so the hidden states entering the first checkpointed decoder layer have
    ``requires_grad=False`` and the whole segment -- containing every lora parameter -- receives no
    gradient, while the run reports success and bills (GRAD-001). ``enable_input_require_grads()``
    is what prevents it, and it is unconditional here because every flash rl run is lora.

    on a MULTIMODAL run the same hook additionally restores VISION input gradients. the vision
    tower's patch embeddings are the same failure in a place the language-side call does not
    reach: the pixels are frozen inputs, so without a forward hook marking the patch-embed output
    as requiring grad the visual modules silently receive nothing while the language side trains
    normally (``tests/test_multimodal_input_grads.py``). the retired trl path installed that hook
    via a trainer callback; verl has no callback surface, so it rides this shim instead -- and only
    here, because non-reentrant recompute does not have the behaviour that makes it necessary
    (codex[bot]).
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
    """return the sitecustomize source that adds top-entropy token masking to verl.

    the objective keeps only the top ``entropy_quantile`` fraction of response tokens in the
    policy-gradient term; verl has no such knob. the mask is expressible
    as an extra factor on ``response_mask``, so this patches ``ppo_loss`` rather than registering a
    custom policy loss: ``ppo_loss`` computes per-token entropy but never forwards it to
    ``policy_loss_fn``, so a registered loss could not see the entropy it needs to threshold on.

    the mask applies to the policy-gradient term ONLY: it multiplies
    ``per_token_loss`` by the mask and then adds the kl term, so kl and the entropy bonus stay on the
    full response mask. equivalence also needs a mask-independent denominator: flash pins
    ``seq-mean-token-sum-norm``, which divides by ``global_batch_size * loss_scale_factor``, so
    dropping tokens from the numerator does not rescale the remaining ones.
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
