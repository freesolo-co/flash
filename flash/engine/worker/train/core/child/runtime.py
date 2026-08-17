"""dependency-minimal runtime patches copied into isolated verl children."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import sys
import traceback

FLASH_WANDB_LINK_MARKER = "FLASH_WANDB_LINK"
FLASH_GDN_VARLEN_MARKER = "[flash-verl] gdn packed-boundary resets active"
FLASH_FLASH_QLA_MARKER = "[flash-verl] flashqla gdn backend active"
FLASH_LORA_ROLLOUT_MARKER = "[flash-verl] lora rollout guard active"
LORA_ROLLOUT_GUARD_SHIM = "lora-rollout-guard"
SHIM_FRAGMENT_FAILED_EXIT_CODE = 97

_LORA_ROLLOUT_TARGET = "verl.workers.rollout.vllm_rollout.vllm_async_server"


def load_plugin_config(env_name: str) -> dict:
    """load one algorithm plugin's non-secret json configuration."""
    raw = os.environ.get(env_name)
    if not raw:
        raise RuntimeError(f"{env_name} is not configured")
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise TypeError(f"{env_name} must contain a json object")
    return config


def load_plugin_config_file(env_name: str) -> dict:
    """load one algorithm plugin's non-secret json configuration from a file."""
    path = os.environ.get(env_name)
    if not path:
        raise RuntimeError(f"{env_name} is not configured")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"{env_name} must point to a json object")
    return config


def record_applied_patch(marker_file: str, name: str) -> None:
    """record a patch only after its target was changed successfully."""
    with open(marker_file, "a", encoding="utf-8") as handle:
        handle.write(name + "\n")


def required_patch_failed(name: str) -> None:
    """exit permanently when a correctness patch cannot be installed."""
    traceback.print_exc()
    print(
        f"[flash-verl] required shim fragment {name} failed to apply; "
        f"exiting {SHIM_FRAGMENT_FAILED_EXIT_CODE} rather than training unpatched",
        file=sys.stderr,
        flush=True,
    )
    sys.stderr.flush()
    os._exit(SHIM_FRAGMENT_FAILED_EXIT_CODE)


def install_required(name: str, marker_file: str, installer, *args, **kwargs) -> None:
    """run an immediate correctness installer and record its success."""
    try:
        installer(*args, **kwargs)
        record_applied_patch(marker_file, name)
    except BaseException:
        required_patch_failed(name)


class _DeferredLoader:
    def __init__(self, finder, target: str, inner):
        self._finder = finder
        self._target = target
        self._inner = inner

    def create_module(self, spec):
        create = getattr(self._inner, "create_module", None)
        return create(spec) if callable(create) else None

    def exec_module(self, module):
        # a real loader failure must leave the queue armed for a later retry
        # and must not record any marker.
        self._inner.exec_module(module)
        self._finder.drain(self._target, module)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _DeferredFinder:
    """one target-keyed registry finder that never imports a target while armed."""

    def __init__(self):
        self.pending: dict[str, list] = {}
        self.active_targets: set[str] = set()

    def register(self, target: str, apply) -> None:
        """queue a callback and arm this finder, without importing the target."""
        self.pending.setdefault(target, []).append(apply)
        if self not in sys.meta_path:
            sys.meta_path.insert(0, self)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.pending:
            return None
        positions = [index for index, finder in enumerate(sys.meta_path) if finder is self]
        remaining = sys.meta_path[positions[0] + 1 :] if positions else sys.meta_path
        for finder in remaining:
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _DeferredLoader(self, fullname, spec.loader)
                return spec
        return None

    def drain(self, target: str, module) -> None:
        """apply every queued callback for one target in registration order."""
        self.active_targets.add(target)
        try:
            queue = self.pending.get(target)
            while queue:
                # pop before applying so a callback that registers another
                # same-target callback queues behind the remaining work.
                apply = queue.pop(0)
                apply(module)
                queue = self.pending.get(target)
        finally:
            self.active_targets.discard(target)
            if not self.pending.get(target):
                self.pending.pop(target, None)
            if not self.pending:
                self.uninstall()

    def uninstall(self) -> None:
        sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not self]


_DEFERRED_FINDER = _DeferredFinder()


def _arm_deferred(
    *,
    name: str,
    marker_file: str,
    target: str,
    patch,
    required: bool,
) -> None:
    def apply(module) -> None:
        try:
            applied = patch(module)
            if applied is not False:
                record_applied_patch(marker_file, name)
        except BaseException:
            if required:
                required_patch_failed(name)
            else:
                traceback.print_exc()

    loaded = sys.modules.get(target)
    # a target that is mid-drain is already in sys.modules, so applying here would
    # jump the queue. register instead and let the active drain pick it up in order.
    if loaded is not None and target not in _DEFERRED_FINDER.active_targets:
        apply(loaded)
        return

    _DEFERRED_FINDER.register(target, apply)


def install_deferred_required(
    name: str,
    marker_file: str,
    target: str,
    installer,
    *args,
    **kwargs,
) -> None:
    """arm a required patch without importing its cuda-sensitive target."""

    def patch(_module):
        return installer(*args, **kwargs)

    _arm_deferred(
        name=name,
        marker_file=marker_file,
        target=target,
        patch=patch,
        required=True,
    )


def install_wandb_link_reporting() -> None:
    """report the child-created wandb run without making logging fatal."""
    try:
        import wandb

        if getattr(wandb.init, "_flash_wandb_reporting", False):
            return
        original_init = wandb.init
        finished_run_ids: set[int] = set()

        def finish_run(run) -> None:
            run_id = id(run)
            if run_id in finished_run_ids:
                return
            try:
                run.finish(exit_code=0)
            except Exception:
                return
            finished_run_ids.add(run_id)

        def init_reporting(*args, **kwargs):
            run = original_init(*args, **kwargs)
            with contextlib.suppress(Exception):
                atexit.register(finish_run, run)
            try:
                url = getattr(run, "url", None)
                run_id = getattr(run, "id", None)
                if url:
                    print(
                        FLASH_WANDB_LINK_MARKER
                        + " "
                        + json.dumps({"wandb_url": url, "wandb_id": run_id}),
                        flush=True,
                    )
            except Exception:
                pass
            return run

        init_reporting._flash_wandb_reporting = True
        wandb.init = init_reporting
    except Exception:
        pass


def install_checkpoint_handler_filter(save_at_steps, total_steps: int) -> None:
    """keep only requested checkpoint handler saves and the final save."""
    required_steps = frozenset(int(step) for step in save_at_steps)
    if not required_steps:
        # an empty schedule keeps every save, so the wrapper is a pass-through.
        return

    from verl.utils.checkpoint.checkpoint_handler import CheckpointHandler

    total_steps = int(total_steps)
    current = CheckpointHandler.save_checkpoint
    if getattr(current, "_flash_exact_save_patched", False):
        return

    def save_checkpoint(self, step):
        if step not in required_steps and step != total_steps:
            return None
        return current(self, step)

    save_checkpoint._flash_exact_save_patched = True
    CheckpointHandler.save_checkpoint = save_checkpoint


def _gdn_seq_idx(position_ids, cu_seq_lens):
    import torch

    lengths = cu_seq_lens.diff()
    return (
        torch.repeat_interleave(
            torch.arange(lengths.numel(), device=position_ids.device, dtype=torch.int32),
            lengths.to(torch.int64),
        )
        .unsqueeze(0)
        .contiguous()
    )


def _patch_gdn_varlen(modeling, model_type: str) -> None:
    text_model = next(
        value
        for name, value in vars(modeling).items()
        if isinstance(value, type) and name.endswith("TextModel")
    )
    if getattr(text_model.forward, "_flash_gdn_varlen_patched", False):
        return
    from transformers.modeling_flash_attention_utils import (
        _is_packed_sequence,
        prepare_fa_kwargs_from_position_ids,
    )

    original = text_model.forward

    def forward(self, *args, **kwargs):
        if kwargs.get("cu_seq_lens_q") is None and kwargs.get("seq_idx") is None:
            position_ids = kwargs.get("position_ids")
            text_position_ids = position_ids
            if text_position_ids is not None and text_position_ids.ndim == 3:
                text_position_ids = text_position_ids[0]
            if (
                text_position_ids is not None
                and text_position_ids.ndim == 2
                and _is_packed_sequence(text_position_ids, text_position_ids.shape[0])
            ):
                (cu_seq_lens_q, cu_seq_lens_k), (max_q, max_k) = (
                    prepare_fa_kwargs_from_position_ids(text_position_ids)
                )
                kwargs["cu_seq_lens_q"] = cu_seq_lens_q
                kwargs["cu_seq_lens_k"] = cu_seq_lens_k
                kwargs["max_length_q"] = max_q
                kwargs["max_length_k"] = max_k
                kwargs["seq_idx"] = _gdn_seq_idx(text_position_ids, cu_seq_lens_q)
        return original(self, *args, **kwargs)

    forward._flash_gdn_varlen_patched = True
    text_model.forward = forward
    print(FLASH_GDN_VARLEN_MARKER, model_type, flush=True)


def install_deferred_gdn(model_type: str, marker_file: str) -> None:
    """arm the required packed-boundary patch without importing the model target."""
    target = f"transformers.models.{model_type}.modeling_{model_type}"
    _arm_deferred(
        name="gdn-varlen",
        marker_file=marker_file,
        target=target,
        patch=lambda module: _patch_gdn_varlen(module, model_type),
        required=True,
    )


def _flash_qla_supported() -> bool:
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _flash_qla_impl():
    try:
        from fla.ops.gated_delta_rule.backends import gdr_registry

        for backend in gdr_registry._get_sorted_backends():
            if getattr(backend, "backend_type", "") != "flash_qla":
                continue
            if not backend.is_available():
                return None
            return getattr(backend, "chunk_gated_delta_rule", None)
    except Exception:
        return None
    return None


def _patch_flash_qla(module) -> bool:
    if not _flash_qla_supported():
        return False
    implementation = _flash_qla_impl()
    if implementation is None:
        print(
            "[flash-verl] flashqla gdn backend unavailable (flash-qla wheel missing, or "
            "flash-linear-attention older than the 0.5.2 that carries the dispatch); "
            "continuing on fla's own kernel",
            file=sys.stderr,
            flush=True,
        )
        return False

    def flash_qla_call(*args, **kwargs):
        kwargs.pop("cu_seqlens_cpu", None)
        return implementation(*args, **kwargs)

    flash_qla_call._flash_qla_patched = True
    module.chunk_gated_delta_rule = flash_qla_call
    print(FLASH_FLASH_QLA_MARKER, module.__name__, flush=True)
    return True


def install_deferred_flash_qla(model_type: str, marker_file: str) -> None:
    """arm the optional sm90 flashqla binding without importing cuda-sensitive targets."""
    target = f"transformers.models.{model_type}.modeling_{model_type}"
    _arm_deferred(
        name="flashqla-gdn",
        marker_file=marker_file,
        target=target,
        patch=_patch_flash_qla,
        required=False,
    )


def _guard_lora_engine(engine, adapter_id) -> None:
    inner = engine.generate

    def guarded(*args, **kwargs):
        if kwargs.get("lora_request") is None:
            raise RuntimeError(
                "[flash-verl] refusing to roll out from the base model: this run trains a lora "
                f"adapter (id {adapter_id}), but it was not loaded in the vllm engine when this "
                "request was built, so verl passed no LoRARequest. generating here would train "
                "the adapter on tokens a different policy produced, which no loss curve can "
                "distinguish from a healthy run."
            )
        return inner(*args, **kwargs)

    engine.generate = guarded
    engine._flash_lora_guarded = True


def _patch_lora_rollout(module) -> None:
    server = module.vLLMHttpServer
    if getattr(server.generate, "_flash_lora_guarded", False):
        return
    original = server.generate

    async def generate(self, *args, **kwargs):
        engine = self.engine
        if self.lora_as_adapter and not getattr(engine, "_flash_lora_guarded", False):
            _guard_lora_engine(engine, module.VLLM_LORA_INT_ID)
        return await original(self, *args, **kwargs)

    generate._flash_lora_guarded = True
    server.generate = generate
    print(FLASH_LORA_ROLLOUT_MARKER, flush=True)


def install_deferred_lora_rollout_guard(marker_file: str) -> None:
    """arm the required lora rollout guard without importing vllm or torch."""
    _arm_deferred(
        name=LORA_ROLLOUT_GUARD_SHIM,
        marker_file=marker_file,
        target=_LORA_ROLLOUT_TARGET,
        patch=_patch_lora_rollout,
        required=True,
    )
