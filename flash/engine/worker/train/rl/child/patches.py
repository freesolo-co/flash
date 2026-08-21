"""callable GRPO patch installers for the isolated verl child."""

from __future__ import annotations

if __name__ == "flash_grpo_patches":
    import flash_verl_runtime as runtime
else:
    from flash.engine.worker.train.core.child import runtime

_ENTROPY_QUANTILE_MARKER = "[flash-verl] top-entropy token masking active"
_STOP_SEQUENCES_MARKER = "[flash-verl] rollout stop strings active"
_STRUCTURED_OUTPUTS_MARKER = "[flash-verl] rollout structured outputs active"
_EXACT_SAVE_STEPS_MARKER = "[flash-verl] exact save steps active"
_IMAGE_PAD_BAN_MARKER = "[flash-verl] image-pad token banned from rollouts"
_KL_REF_ADAPTER_MARKER = "[flash-verl] kl reference anchored to the warm-start adapter"
_EXACT_ROLLOUT_IDENTITY_MARKER = "[flash-verl] exact rollout identity active"
_PINNED_RUN_AGENT_LOOP_SHA256 = "46f1af635d13854a8955fc44b8595da127997c68e818aaa6d7bf28acc33902cf"
_PINNED_WORKER_GENERATE_SHA256 = "3b359e9034375860464368b4990e7d2178dde9bbd35d7513652d295820b53f65"
_PINNED_MANAGER_GENERATE_SHA256 = "3ea1e08ea29b6162d20ea8dd765abcccc233088f1c4b7f7716096c43d0a82ebb"


def _guard_exact_identity_boundary(function, expected_hash, label, snippets) -> None:
    import hashlib
    import inspect

    source = inspect.getsource(function)
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if source_hash != expected_hash or any(source.count(snippet) != 1 for snippet in snippets):
        raise RuntimeError(
            f"flash exact rollout identity {label} boundary drifted from pinned Verl "
            f"(got sha256 {source_hash})"
        )


def _trajectory_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"flash rollout trajectory {name} must be an integer")
    return value


def _trajectory_bool(value, name):
    if not isinstance(value, bool):
        raise RuntimeError(f"flash rollout trajectory {name} must be a boolean")
    return value


def _identity_from_trajectory(trajectory):
    return {
        "optimizer_step": _trajectory_int(trajectory.get("step"), "step"),
        "sample_index": _trajectory_int(trajectory.get("sample_index"), "sample_index"),
        "rollout_ordinal": _trajectory_int(trajectory.get("rollout_n"), "rollout_n"),
        "validate": _trajectory_bool(trajectory.get("validate"), "validate"),
    }


def _trajectory_info_from_identities(step, index, validate, identities):
    if len(identities) != len(index):
        raise RuntimeError("flash rollout identity sidecar length does not match worker batch")
    trajectories = []
    for position, (identity, sample_index) in enumerate(zip(identities, index, strict=True)):
        if identity["optimizer_step"] != _trajectory_int(step, "step"):
            raise RuntimeError("flash rollout identity sidecar step does not match worker batch")
        if identity["sample_index"] != _trajectory_int(sample_index, "sample_index"):
            raise RuntimeError(
                f"flash rollout identity sidecar index mismatch at position {position}"
            )
        if identity["validate"] is not _trajectory_bool(validate, "validate"):
            raise RuntimeError("flash rollout identity sidecar validate flag does not match batch")
        trajectories.append(
            {
                "step": identity["optimizer_step"],
                "sample_index": identity["sample_index"],
                "rollout_n": identity["rollout_ordinal"],
                "validate": identity["validate"],
            }
        )
    return trajectories


def _build_worker_identity_wrapper(original_worker_generate):
    async def generate_sequences_with_identities(self, batch, identities):
        import types

        identity_sidecar = tuple(dict(identity) for identity in identities)

        async def exact_trajectory_info(step, index, validate):
            return _trajectory_info_from_identities(step, index, validate, identity_sidecar)

        worker_globals = dict(original_worker_generate.__globals__)
        worker_globals["get_trajectory_info"] = exact_trajectory_info
        worker_generate = types.FunctionType(
            original_worker_generate.__code__,
            worker_globals,
            original_worker_generate.__name__,
            original_worker_generate.__defaults__,
            original_worker_generate.__closure__,
        )
        worker_generate.__kwdefaults__ = original_worker_generate.__kwdefaults__
        return await worker_generate(self, batch)

    return generate_sequences_with_identities


def _build_manager_identity_wrapper(agent_loop, original_trajectory_info, post_json):
    import os

    async def manager_generate_sequences(self, prompts):
        index = prompts.non_tensor_batch.get("index")
        indexes = list(range(len(prompts))) if index is None else index.tolist()
        trajectories = await original_trajectory_info(
            prompts.meta_info.get("global_steps", -1),
            indexes,
            prompts.meta_info.get("validate", False),
        )
        identities = [_identity_from_trajectory(trajectory) for trajectory in trajectories]
        url = os.environ.get("FLASH_VERL_REWARD_URL", "")
        if not url:
            raise RuntimeError(
                "flash reward bridge url is not configured for identity registration"
            )
        post_json(
            url,
            "/identity/register",
            {"identities": [dict(identity) for identity in identities]},
            error_style="reward",
        )
        chunkes = prompts.chunk(len(self.agent_loop_workers))
        identity_chunks = []
        offset = 0
        for chunk in chunkes:
            identity_chunks.append(identities[offset : offset + len(chunk)])
            offset += len(chunk)
        if offset != len(identities):
            raise RuntimeError("flash rollout identity sidecar did not cover the manager batch")
        outputs = await agent_loop.asyncio.gather(
            *[
                worker.generate_sequences_with_flash_identities.remote(chunk, identity_chunk)
                for worker, chunk, identity_chunk in zip(
                    self.agent_loop_workers, chunkes, identity_chunks, strict=True
                )
            ]
        )
        output = agent_loop.DataProto.concat(outputs)
        metrics = [item.meta_info.pop("metrics") for item in outputs]
        timing = self._performance_metrics(metrics, output)
        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    return manager_generate_sequences


def _build_run_identity_wrapper(original_run):
    async def run_agent_loop(
        self,
        sampling_params,
        trajectory,
        *,
        agent_name,
        trace=True,
        **kwargs,
    ):
        identity = _identity_from_trajectory(trajectory)
        extra_info = kwargs.get("extra_info")
        if not isinstance(extra_info, dict):
            raise RuntimeError("flash rollout trajectory is missing dictionary extra_info")
        if "flash_rollout_identity" in extra_info or "flash_rollout_identity" in kwargs:
            raise RuntimeError("flash rollout identity was already present before attachment")
        forwarded = dict(kwargs)
        forwarded_extra = dict(extra_info)
        forwarded_extra["flash_rollout_identity"] = dict(identity)
        forwarded["extra_info"] = forwarded_extra
        forwarded["flash_rollout_identity"] = dict(identity)
        return await original_run(
            self,
            sampling_params,
            trajectory,
            agent_name=agent_name,
            trace=trace,
            **forwarded,
        )

    return run_agent_loop


def install_exact_rollout_identity() -> None:
    """Register and attach exact identities at pinned Verl occurrence boundaries."""
    from flash_grpo_multiturn import post_json
    from verl.experimental.agent_loop import agent_loop

    original_run = agent_loop.AgentLoopWorker._run_agent_loop
    if getattr(original_run, "_flash_exact_rollout_identity", False):
        return
    original_worker_generate = agent_loop.AgentLoopWorker.generate_sequences
    original_manager_generate = agent_loop.AgentLoopManager.generate_sequences
    original_trajectory_info = agent_loop.get_trajectory_info
    _guard_exact_identity_boundary(
        original_run,
        _PINNED_RUN_AGENT_LOOP_SHA256,
        "occurrence",
        ('trajectory["rollout_n"]', "self._agent_loop_postprocess"),
    )
    _guard_exact_identity_boundary(
        original_worker_generate,
        _PINNED_WORKER_GENERATE_SHA256,
        "worker generation",
        ("trajectory_info = await get_trajectory_info", "self._run_agent_loop("),
    )
    _guard_exact_identity_boundary(
        original_manager_generate,
        _PINNED_MANAGER_GENERATE_SHA256,
        "manager generation",
        ("chunkes = prompts.chunk", "worker.generate_sequences.remote(chunk)"),
    )
    worker_generate = _build_worker_identity_wrapper(original_worker_generate)
    manager_generate = _build_manager_identity_wrapper(
        agent_loop, original_trajectory_info, post_json
    )
    run_agent_loop = _build_run_identity_wrapper(original_run)
    run_agent_loop._flash_exact_rollout_identity = True
    manager_generate = agent_loop.auto_await(manager_generate)
    manager_generate._flash_exact_rollout_identity = True
    manager_generate._flash_pinned_source_sha256 = _PINNED_MANAGER_GENERATE_SHA256
    agent_loop.AgentLoopWorker.generate_sequences_with_flash_identities = worker_generate
    agent_loop.AgentLoopWorker._run_agent_loop = run_agent_loop
    agent_loop.AgentLoopManager.generate_sequences = manager_generate
    print(_EXACT_ROLLOUT_IDENTITY_MARKER, flush=True)


def install_rank_device_assert(n_gpus: int) -> None:
    """refuse a multi-rank launch when two ranks opened the same physical gpu."""
    if int(n_gpus) < 2:
        return

    import os

    import torch
    from verl.single_controller.base import worker

    original_init = worker.Worker.__init__
    if getattr(original_init, "_flash_rank_checked", False):
        return
    expected_ranks = int(n_gpus)

    def device_identity():
        if not torch.cuda.is_available():
            return None
        ordinal = torch.cuda.current_device()
        try:
            uuid = str(getattr(torch.cuda.get_device_properties(ordinal), "uuid", "") or "")
        except Exception:
            uuid = ""
        return ordinal, uuid

    def check(worker_instance):
        identity = device_identity()
        if identity is None:
            return
        ordinal, uuid = identity
        rank = getattr(worker_instance, "_rank", None)
        if rank is None:
            rank = getattr(worker_instance, "rank", None)
        rank = int(rank) if rank is not None else int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        print(
            "[rl-verl] rank-device binding rank="
            + repr(rank)
            + " local_rank="
            + repr(local_rank)
            + " cuda_ordinal="
            + repr(ordinal)
            + " visible_devices="
            + repr(visible)
            + " uuid="
            + repr(uuid or "<unavailable>"),
            flush=True,
        )
        if not uuid:
            return
        claims_path = os.environ.get("FLASH_RANK_DEVICE_CLAIMS", "")
        if not claims_path:
            return
        with open(claims_path, "a", encoding="utf-8") as handle:
            handle.write(repr(rank) + " " + uuid + "\n")
        claims = {}
        with open(claims_path, encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) == 2:
                    claims.setdefault(parts[1], set()).add(int(parts[0]))
        collided = {device: sorted(ranks) for device, ranks in claims.items() if len(ranks) > 1}
        if collided:
            raise RuntimeError(
                "flash: ranks were bound to the SAME physical gpu before training started -- "
                + repr(collided)
                + " (uuid -> ranks). this rank: rank="
                + repr(rank)
                + " local_rank="
                + repr(local_rank)
                + " cuda_ordinal="
                + repr(ordinal)
                + " CUDA_VISIBLE_DEVICES="
                + repr(visible)
                + ". expected "
                + repr(expected_ranks)
                + " ranks on distinct cards; nccl would abort a minute later with "
                "'Duplicate GPU detected' and no rank mapping."
            )

    def init(worker_instance, *args, **kwargs):
        result = original_init(worker_instance, *args, **kwargs)
        check(worker_instance)
        return result

    init._flash_rank_checked = True
    worker.Worker.__init__ = init


def install_nonempty_response_mask() -> None:
    import os

    from verl.trainer.ppo import rollout_corr_helper

    original = rollout_corr_helper.compute_rollout_correction_and_add_to_batch
    if getattr(original, "_flash_nonempty_response_mask", False):
        return

    def compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config):
        response_mask = batch.batch.get("response_mask")
        if response_mask is not None and not bool(response_mask.any()):
            message = (
                "flash: no trainable response tokens remain in this batch: every rollout was "
                "truncated or unusable, so no optimizer update will run. increase "
                "train.max_completion_tokens or disable thinking."
            )
            context = [
                f"{name}={os.environ[name]}"
                for name in (
                    "FLASH_VERL_THINKING",
                    "FLASH_VERL_MAX_COMPLETION_TOKENS",
                )
                if name in os.environ
            ]
            if context:
                message += " child context: " + ", ".join(context) + "."
            raise RuntimeError(message)
        return original(batch, rollout_corr_config)

    compute_rollout_correction_and_add_to_batch._flash_nonempty_response_mask = True
    rollout_corr_helper.compute_rollout_correction_and_add_to_batch = (
        compute_rollout_correction_and_add_to_batch
    )


def install_kl_ref_adapter() -> None:
    from contextlib import contextmanager

    import torch.nn as nn
    from peft.tuners.tuners_utils import BaseTunerLayer
    from verl.workers.engine.fsdp import transformer_impl

    adapter_name = "flash_kl_ref"

    def snapshot(module):
        if adapter_name in getattr(module, "peft_config", {}):
            return module
        module.add_adapter(adapter_name, module.peft_config[module.active_adapter])
        for name, param in module.named_parameters():
            if ".default." in name:
                twin = module.get_parameter(name.replace(".default.", f".{adapter_name}."))
                twin.data.copy_(param.data)
        demoted = 0
        for container in module.modules():
            if isinstance(container, nn.ModuleDict) and adapter_name in container:
                leaf = container[adapter_name]
                for attr, value in list(leaf.named_parameters(recurse=False)):
                    frozen = value.detach().clone()
                    delattr(leaf, attr)
                    leaf.register_buffer(attr, frozen, persistent=False)
                    demoted += 1
        if not demoted:
            raise RuntimeError("flash kl reference snapshot found no adapter weights to freeze")
        print(_KL_REF_ADAPTER_MARKER + " " + repr(demoted), flush=True)
        return module

    original_build_lora = transformer_impl.FSDPEngine._build_lora_module

    def build_lora_module(self, module):
        return snapshot(original_build_lora(self, module))

    @contextmanager
    def use_ref_adapter(module):
        layers = [value for value in module.modules() if isinstance(value, BaseTunerLayer)]
        if not layers:
            raise RuntimeError("flash kl reference: no lora layers on the actor module")
        saved = [layer._active_adapter for layer in layers]
        for layer in layers:
            layer._active_adapter = [adapter_name]
        try:
            yield
        finally:
            for layer, previous in zip(layers, saved, strict=True):
                layer._active_adapter = previous

    def disable_adapter(self):
        module = self.module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        if adapter_name not in getattr(inner, "peft_config", {}):
            raise RuntimeError(
                "flash kl reference adapter missing: expected " + adapter_name + " on the actor"
            )
        return use_ref_adapter(module)

    transformer_impl.FSDPEngine._build_lora_module = build_lora_module
    transformer_impl.FSDPEngine.disable_adapter = disable_adapter


def install_structured_outputs(structured_outputs: dict) -> None:
    from verl.experimental.agent_loop import agent_loop
    from vllm.sampling_params import StructuredOutputsParams

    original = agent_loop.AgentLoopWorker._run_agent_loop
    if getattr(original, "_flash_so_patched", False):
        return

    async def run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        params["structured_outputs"] = StructuredOutputsParams(**structured_outputs)
        return await original(self, params, *args, **kwargs)

    run_agent_loop._flash_so_patched = True
    agent_loop.AgentLoopWorker._run_agent_loop = run_agent_loop
    print(_STRUCTURED_OUTPUTS_MARKER + " " + repr(structured_outputs), flush=True)


def install_exact_save_steps(save_at_steps, total_steps: int) -> None:
    from verl.trainer.ppo import ray_trainer

    required_steps = frozenset(int(step) for step in save_at_steps)
    original = ray_trainer.RayPPOTrainer._save_checkpoint
    if getattr(original, "_flash_save_patched", False):
        return

    def save_checkpoint(self):
        step = int(self.global_steps)
        if step not in required_steps and step != int(total_steps):
            return None
        return original(self)

    save_checkpoint._flash_save_patched = True
    ray_trainer.RayPPOTrainer._save_checkpoint = save_checkpoint
    print(
        _EXACT_SAVE_STEPS_MARKER
        + " "
        + repr(sorted(required_steps))
        + " final="
        + repr(int(total_steps)),
        flush=True,
    )


def install_stop_sequences(stop_sequences) -> None:
    from verl.experimental.agent_loop import agent_loop

    stops = list(stop_sequences)
    original = agent_loop.AgentLoopWorker._run_agent_loop
    if getattr(original, "_flash_stop_patched", False):
        return

    async def run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        params["stop"] = list(stops)
        return await original(self, params, *args, **kwargs)

    run_agent_loop._flash_stop_patched = True
    agent_loop.AgentLoopWorker._run_agent_loop = run_agent_loop
    print(_STOP_SEQUENCES_MARKER + " " + repr(stops), flush=True)


def install_image_pad_ban(image_pad_token_id: int) -> None:
    from verl.experimental.agent_loop import agent_loop

    token_id = int(image_pad_token_id)
    original = agent_loop.AgentLoopWorker._run_agent_loop
    if getattr(original, "_flash_image_pad_patched", False):
        return

    async def run_agent_loop(self, sampling_params, *args, **kwargs):
        params = dict(sampling_params)
        logit_bias = dict(params.get("logit_bias") or {})
        logit_bias[token_id] = -100.0
        params["logit_bias"] = logit_bias
        return await original(self, params, *args, **kwargs)

    run_agent_loop._flash_image_pad_patched = True
    agent_loop.AgentLoopWorker._run_agent_loop = run_agent_loop
    print(_IMAGE_PAD_BAN_MARKER + " " + repr(token_id), flush=True)


def install_per_turn_credit() -> None:
    import torch
    from verl.trainer.ppo import ray_trainer

    original_compute_advantage = ray_trainer.compute_advantage
    logged = False

    def rows(non_tensor_batch, batch_size):
        spans_column = non_tensor_batch.get("flash_turn_spans")
        rewards_column = non_tensor_batch.get("flash_turn_rewards")
        if spans_column is None or rewards_column is None:
            return None
        if len(spans_column) != batch_size or len(rewards_column) != batch_size:
            return None
        result = []
        for spans, turns in zip(spans_column, rewards_column, strict=True):
            if spans is None or turns is None or len(spans) != len(turns):
                result.append(None)
                continue
            result.append(
                (
                    tuple((int(start), int(end)) for start, end in spans),
                    tuple(float(value) for value in turns),
                )
            )
        return result

    def per_turn_advantages(metadata_rows, index, episode_advantages):
        advantages = episode_advantages.clone()
        groups = {}
        for row_index, uid in enumerate(index):
            groups.setdefault(uid, []).append(row_index)
        for member_indexes in groups.values():
            if any(metadata_rows[row_index] is None for row_index in member_indexes):
                continue
            for row_index in member_indexes:
                advantages[row_index] = 0.0
            turn_total = max(len(metadata_rows[row_index][1]) for row_index in member_indexes)
            for turn_index in range(turn_total):
                scoring = [
                    row_index
                    for row_index in member_indexes
                    if turn_index < len(metadata_rows[row_index][1])
                    and metadata_rows[row_index][0][turn_index][1]
                    > metadata_rows[row_index][0][turn_index][0]
                ]
                if not scoring:
                    continue
                baseline = sum(
                    metadata_rows[row_index][1][turn_index] for row_index in scoring
                ) / len(scoring)
                for row_index in scoring:
                    start, end = metadata_rows[row_index][0][turn_index]
                    advantages[row_index, start:end] = (
                        metadata_rows[row_index][1][turn_index] - baseline
                    )
        return advantages

    def compute_advantage(data, *args, **kwargs):
        nonlocal logged
        data = original_compute_advantage(data, *args, **kwargs)
        episode = data.batch.get("advantages")
        if episode is None or episode.dim() != 2:
            return data
        batch_size, width = episode.shape
        metadata_rows = rows(data.non_tensor_batch, batch_size)
        if metadata_rows is None or all(row is None for row in metadata_rows):
            return data
        index = data.non_tensor_batch.get("uid")
        if index is None or len(index) != batch_size:
            return data
        for row in metadata_rows:
            if row is None:
                continue
            for start, end in row[0]:
                if not 0 <= start <= end <= width:
                    raise ValueError(
                        f"turn span [{start}, {end}) exceeds the response width {width}"
                    )
        advantages = per_turn_advantages(metadata_rows, index, episode)
        if not bool(torch.isfinite(advantages).all()):
            raise ValueError("per-turn advantages must be finite")
        response_mask = data.batch.get("response_mask")
        if response_mask is not None:
            advantages = advantages * response_mask.to(dtype=advantages.dtype)
        data.batch["advantages"] = advantages
        data.batch["returns"] = advantages
        if not logged:
            print("[rl-verl] multi-turn per-turn group-relative credit is active", flush=True)
            logged = True
        return data

    ray_trainer.compute_advantage = compute_advantage


def install_reentrant_checkpointing(*, multimodal: bool) -> None:
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    original = FSDPEngine._build_module
    if getattr(original, "_flash_reentrant_grpo", False):
        return

    def build_module(self):
        module = original(self)
        if getattr(self.model_config, "enable_gradient_checkpointing", False):
            module.enable_input_require_grads()
            module.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": True}
            )
            if multimodal:
                runtime.install_vision_input_grads(module)
            print("[rl-verl] reentrant gradient checkpointing is active", flush=True)
        return module

    build_module._flash_reentrant_grpo = True
    FSDPEngine._build_module = build_module


def install_entropy_quantile(entropy_quantile: float) -> None:
    import threading

    import torch
    import torch.distributed as dist
    from verl.workers.utils import losses
    from verl.workers.utils.padding import no_padding_2_padding

    threshold = 1.0 - float(entropy_quantile)
    original_ppo_loss = losses.ppo_loss
    original_get_policy_loss_fn = losses.get_policy_loss_fn
    if getattr(original_ppo_loss, "_flash_entropy_masked", False):
        return
    state = threading.local()

    def high_entropy_mask(entropy, response_mask):
        local = entropy[response_mask.bool()].float().reshape(-1)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            sizes = [
                torch.zeros(1, dtype=torch.long, device=local.device)
                for _ in range(dist.get_world_size())
            ]
            dist.all_gather(
                sizes,
                torch.tensor([local.numel()], dtype=torch.long, device=local.device),
            )
            largest = int(max(int(size.item()) for size in sizes))
            if largest == 0:
                return torch.zeros_like(entropy, dtype=torch.bool)
            padded = torch.full((largest,), -1e9, dtype=torch.float32, device=local.device)
            padded[: local.numel()] = local
            buckets = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
            dist.all_gather(buckets, padded)
            gathered = torch.cat(buckets)
            gathered = gathered[gathered != -1e9]
        else:
            gathered = local
        if gathered.numel() == 0:
            return torch.zeros_like(entropy, dtype=torch.bool)
        cutoff = torch.quantile(gathered, threshold)
        return ((entropy * response_mask.float()) >= cutoff) & response_mask.bool()

    def masked_policy_loss_fn(loss_mode):
        inner = original_get_policy_loss_fn(loss_mode)

        def masked(*args, **kwargs):
            entropy = getattr(state, "entropy", None)
            response_mask = kwargs.get("response_mask")
            if entropy is not None and response_mask is not None:
                kwargs["response_mask"] = high_entropy_mask(entropy, response_mask)
            return inner(*args, **kwargs)

        return masked

    def entropy_masked_ppo_loss(config, model_output, data, dp_group=None):
        entropy = model_output.get("entropy")
        if entropy is None:
            raise RuntimeError(
                "train.entropy_quantile needs per-token entropy, but verl produced none; "
                "actor.calculate_entropy must be true."
            )
        state.entropy = no_padding_2_padding(entropy, data)
        try:
            return original_ppo_loss(config, model_output, data, dp_group)
        finally:
            state.entropy = None

    entropy_masked_ppo_loss._flash_entropy_masked = True
    losses.get_policy_loss_fn = masked_policy_loss_fn
    losses.ppo_loss = entropy_masked_ppo_loss
    print(_ENTROPY_QUANTILE_MARKER + f" quantile={entropy_quantile:g}", flush=True)
