"""shared frozen training base with isolated per-run LoRA state."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flash.engine.worker.adapter import make_lora
from flash.engine.worker.openrlhf_shared_engine import AdapterHandle, SharedMultiLoRARolloutEngine


class TrainingIsolationError(RuntimeError):
    """raised when per-run adapter or optimizer isolation is violated."""


@dataclass(frozen=True, slots=True)
class RunOptimizerConfig:
    """optimizer settings matching the existing OpenRLHF paged AdamW recipe."""

    learning_rate: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass(slots=True)
class TrainingRunState:
    """mutable training state owned by one logical run."""

    run_id: str
    adapter_name: str
    adapter_version: int
    handle: AdapterHandle
    adapter_parameters: tuple[Any, ...]
    adapter_parameter_names: tuple[str, ...]
    optimizer: Any
    lr_scheduler: Any
    dataloader: Any
    prompt_cursor: int = 0
    global_step: int = 0


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    """result of one isolated optimizer step and adapter publication."""

    run_id: str
    global_step: int
    loss: float
    handle: AdapterHandle


OptimizerFactory = Callable[[tuple[Any, ...], RunOptimizerConfig], Any]
SchedulerFactory = Callable[[Any], Any]
AdapterAttacher = Callable[[Any, str, Any], Any]
AdapterExporter = Callable[[Any, str, Path], None]
LossFunction = Callable[[Any, TrainingRunState], Any]


def paged_adamw_8bit_optimizer(parameters: tuple[Any, ...], config: RunOptimizerConfig) -> Any:
    """build the paged 8-bit AdamW optimizer used by Flash OpenRLHF LoRA training."""

    import bitsandbytes as bnb

    return bnb.optim.PagedAdamW8bit(
        parameters,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def _constant_lr_scheduler(optimizer: Any) -> Any:
    import torch

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)


def _attach_peft_adapter(model: Any, adapter_name: str, config: Any) -> Any:
    from peft import get_peft_model

    if not getattr(model, "peft_config", None):
        return get_peft_model(model, config, adapter_name=adapter_name)
    model.add_adapter(adapter_name, config)
    return model


def _export_peft_adapter(model: Any, adapter_name: str, output_dir: Path) -> None:
    import torch
    from peft import get_peft_model_state_dict

    config = model.peft_config[adapter_name]
    config.save_pretrained(output_dir)
    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    if not state:
        raise TrainingIsolationError(f"adapter {adapter_name} has no exportable parameters")
    torch.save(state, output_dir / "adapter_model.bin")


def _optimizer_parameter_ids(optimizer: Any) -> tuple[int, ...]:
    return tuple(id(parameter) for group in optimizer.param_groups for parameter in group["params"])


def _adapter_name(run_id: str) -> str:
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    return f"flash_run_{digest}"


def _resolved_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser().resolve()


def _save_checkpoint_file(path: Path, checkpoint: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint_file(path: Path) -> Mapping[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


class SharedMultiLoRATrainingActor:
    """train named LoRA adapters over one frozen Hugging Face base.

    the actor owns one serialized training lease. each run has an exact adapter
    parameter registry, optimizer, learning-rate scheduler, and prompt cursor. the
    loss callable remains algorithm-owned so existing GRPO and OPD objective hooks
    can be reused without moving their math into this state layer.
    """

    def __init__(
        self,
        base_model_loader: Callable[[], Any],
        rollout_engine: SharedMultiLoRARolloutEngine,
        adapter_root: str | os.PathLike[str],
        *,
        model_id: str | None = None,
        optimizer_factory: OptimizerFactory = paged_adamw_8bit_optimizer,
        scheduler_factory: SchedulerFactory = _constant_lr_scheduler,
        adapter_attacher: AdapterAttacher = _attach_peft_adapter,
        adapter_exporter: AdapterExporter = _export_peft_adapter,
    ) -> None:
        self._model = base_model_loader()
        self._rollout_engine = rollout_engine
        self._adapter_root = Path(adapter_root).expanduser().resolve()
        self._adapter_root.mkdir(parents=True, exist_ok=True)
        self._model_id = model_id
        self._optimizer_factory = optimizer_factory
        self._scheduler_factory = scheduler_factory
        self._adapter_attacher = adapter_attacher
        self._adapter_exporter = adapter_exporter
        self._training_lease = asyncio.Lock()
        self._runs: dict[str, TrainingRunState] = {}
        self._adapter_parameter_ids: dict[str, frozenset[int]] = {}
        self._frozen_base_parameters = tuple(self._model.parameters())
        for parameter in self._frozen_base_parameters:
            parameter.requires_grad_(False)
            parameter.grad = None

    @property
    def model(self) -> Any:
        """return the single shared training model."""

        return self._model

    @property
    def run_ids(self) -> tuple[str, ...]:
        """return registered logical run ids in registration order."""

        return tuple(self._runs)

    def run_state(self, run_id: str) -> TrainingRunState:
        """return one run's isolated mutable state."""

        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown training run: {run_id}") from exc

    async def register_run(
        self,
        run_id: str,
        *,
        optimizer_config: RunOptimizerConfig,
        dataloader: Any,
        lora_config: Any | None = None,
        adapter_version: int = 0,
        prompt_cursor: int = 0,
        scheduler_factory: SchedulerFactory | None = None,
    ) -> TrainingRunState:
        """attach one named adapter and register its initial immutable rollout version."""

        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty")
        if adapter_version < 0:
            raise ValueError("adapter_version must be non-negative")
        if prompt_cursor < 0:
            raise ValueError("prompt_cursor must be non-negative")

        async with self._training_lease:
            if normalized_run_id in self._runs:
                raise ValueError(f"training run already registered: {normalized_run_id}")
            adapter_name = _adapter_name(normalized_run_id)
            before = {id(parameter) for parameter in self._model.parameters()}
            config = lora_config if lora_config is not None else make_lora(self._model_id)
            self._model = self._adapter_attacher(self._model, adapter_name, config)
            adapter_dir: Path | None = None
            try:
                named_parameters = tuple(self._model.named_parameters())
                adapter_entries = tuple(
                    (name, parameter)
                    for name, parameter in named_parameters
                    if id(parameter) not in before
                )
                if not adapter_entries:
                    raise TrainingIsolationError(
                        f"attaching adapter {adapter_name} introduced no trainable parameters"
                    )

                adapter_names = tuple(name for name, _parameter in adapter_entries)
                adapter_parameters = tuple(parameter for _name, parameter in adapter_entries)
                adapter_ids = frozenset(id(parameter) for parameter in adapter_parameters)
                self._assert_disjoint_parameter_set(normalized_run_id, adapter_ids)
                self._freeze_all_parameters()

                optimizer = self._optimizer_factory(adapter_parameters, optimizer_config)
                self._assert_optimizer_owns_exactly(adapter_ids, optimizer)
                selected_scheduler_factory = scheduler_factory or self._scheduler_factory
                lr_scheduler = selected_scheduler_factory(optimizer)

                adapter_dir = self._export_version(adapter_name, normalized_run_id, adapter_version)
                handle = await self._rollout_engine.register_run(
                    normalized_run_id, adapter_version, adapter_dir
                )
            except BaseException:
                self._delete_adapter(adapter_name)
                if adapter_dir is not None:
                    shutil.rmtree(adapter_dir, ignore_errors=True)
                raise

            state = TrainingRunState(
                run_id=normalized_run_id,
                adapter_name=adapter_name,
                adapter_version=adapter_version,
                handle=handle,
                adapter_parameters=adapter_parameters,
                adapter_parameter_names=adapter_names,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                dataloader=dataloader,
                prompt_cursor=prompt_cursor,
            )
            self._runs[normalized_run_id] = state
            self._adapter_parameter_ids[normalized_run_id] = adapter_ids
            self._assert_registry_isolation()
            return state

    async def step(self, run_id: str, loss_function: LossFunction) -> TrainingStepResult:
        """run one isolated optimizer step and publish only that run's adapter."""

        async with self._training_lease:
            state = self.run_state(run_id)
            self._assert_registry_isolation()
            self._activate_adapter(state)
            optimizer = state.optimizer
            optimizer.zero_grad(set_to_none=True)
            self._clear_non_active_gradients(state.run_id)

            try:
                loss = loss_function(self._model, state)
                if getattr(loss, "ndim", 0) != 0:
                    raise ValueError("loss function must return a scalar tensor")
                if not bool(getattr(loss, "requires_grad", False)):
                    raise ValueError("loss function returned a tensor without gradients")
                loss.backward()
                self._assert_inactive_gradients_absent(state.run_id)
                self._assert_frozen_base()
                optimizer.step()
                state.lr_scheduler.step()
            finally:
                optimizer.zero_grad(set_to_none=True)
                self._clear_all_gradients()
                self._freeze_all_parameters()

            old_version = state.adapter_version
            new_version = old_version + 1
            adapter_dir = self._export_version(state.adapter_name, state.run_id, new_version)
            new_handle = await self._rollout_engine.publish_adapter(
                state.run_id,
                expected_old_version=old_version,
                new_version=new_version,
                adapter_dir=adapter_dir,
            )
            state.adapter_version = new_version
            state.handle = new_handle
            state.global_step += 1
            return TrainingStepResult(
                run_id=state.run_id,
                global_step=state.global_step,
                loss=float(loss.detach().cpu().item()),
                handle=new_handle,
            )

    def advance_prompt_cursor(self, run_id: str, count: int = 1) -> int:
        """advance and return one run's independent prompt cursor."""

        if count < 0:
            raise ValueError("prompt cursor advance must be non-negative")
        state = self.run_state(run_id)
        state.prompt_cursor += count
        return state.prompt_cursor

    async def save_run_checkpoint(
        self, run_id: str, checkpoint_path: str | os.PathLike[str]
    ) -> Path:
        """atomically save only one run's adapter, optimizer, scheduler, and cursor."""

        path = _resolved_path(checkpoint_path)
        async with self._training_lease:
            state = self.run_state(run_id)
            checkpoint = {
                "format_version": 1,
                "run_id": state.run_id,
                "adapter_name": state.adapter_name,
                "adapter_version": state.adapter_version,
                "adapter_parameter_names": state.adapter_parameter_names,
                "adapter_parameters": tuple(
                    parameter.detach().cpu().clone() for parameter in state.adapter_parameters
                ),
                "optimizer": state.optimizer.state_dict(),
                "lr_scheduler": state.lr_scheduler.state_dict(),
                "prompt_cursor": state.prompt_cursor,
                "global_step": state.global_step,
            }
            await asyncio.to_thread(_save_checkpoint_file, path, checkpoint)
            return path

    async def restore_run_checkpoint(
        self, run_id: str, checkpoint_path: str | os.PathLike[str]
    ) -> AdapterHandle:
        """restore one live run and publish the restored weights as a new version."""

        import torch

        path = _resolved_path(checkpoint_path)
        async with self._training_lease:
            state = self.run_state(run_id)
            checkpoint = await asyncio.to_thread(_load_checkpoint_file, path)
            self._validate_checkpoint_identity(state, checkpoint)
            saved_names = tuple(checkpoint["adapter_parameter_names"])
            if saved_names != state.adapter_parameter_names:
                raise TrainingIsolationError(
                    "checkpoint adapter parameter names do not match the run"
                )
            saved_parameters = tuple(checkpoint["adapter_parameters"])
            if len(saved_parameters) != len(state.adapter_parameters):
                raise TrainingIsolationError(
                    "checkpoint adapter parameter count does not match the run"
                )

            with torch.no_grad():
                for parameter, saved in zip(
                    state.adapter_parameters, saved_parameters, strict=True
                ):
                    if parameter.shape != saved.shape:
                        raise TrainingIsolationError(
                            "checkpoint adapter parameter shape does not match the run"
                        )
                    parameter.copy_(saved.to(device=parameter.device, dtype=parameter.dtype))
            state.optimizer.load_state_dict(checkpoint["optimizer"])
            state.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            state.prompt_cursor = int(checkpoint["prompt_cursor"])
            state.global_step = int(checkpoint["global_step"])
            self._clear_all_gradients()
            self._freeze_all_parameters()
            self._assert_registry_isolation()

            old_version = state.adapter_version
            new_version = old_version + 1
            adapter_dir = self._export_version(state.adapter_name, state.run_id, new_version)
            new_handle = await self._rollout_engine.publish_adapter(
                state.run_id,
                expected_old_version=old_version,
                new_version=new_version,
                adapter_dir=adapter_dir,
            )
            state.adapter_version = new_version
            state.handle = new_handle
            return new_handle

    def adapter_parameter_snapshot(self, run_id: str) -> tuple[bytes, ...]:
        """return byte snapshots used to prove inactive-adapter immutability in tests."""

        import torch

        state = self.run_state(run_id)
        return tuple(
            parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            for parameter in state.adapter_parameters
        )

    def _activate_adapter(self, state: TrainingRunState) -> None:
        set_adapter = getattr(self._model, "set_adapter", None)
        if set_adapter is None:
            raise TrainingIsolationError("shared training model has no set_adapter method")
        set_adapter(state.adapter_name)
        active_ids = self._adapter_parameter_ids[state.run_id]
        for parameter in self._model.parameters():
            parameter.requires_grad_(id(parameter) in active_ids)
        self._assert_optimizer_owns_exactly(active_ids, state.optimizer)

    def _assert_disjoint_parameter_set(self, run_id: str, candidate_ids: frozenset[int]) -> None:
        for other_run_id, other_ids in self._adapter_parameter_ids.items():
            overlap = candidate_ids.intersection(other_ids)
            if overlap:
                raise TrainingIsolationError(
                    f"adapter parameters overlap between runs {run_id} and {other_run_id}"
                )

    def _assert_registry_isolation(self) -> None:
        seen: set[int] = set()
        for run_id, state in self._runs.items():
            expected = self._adapter_parameter_ids[run_id]
            actual = frozenset(id(parameter) for parameter in state.adapter_parameters)
            if actual != expected:
                raise TrainingIsolationError(f"adapter parameter registry changed for run {run_id}")
            if seen.intersection(actual):
                raise TrainingIsolationError("adapter parameter registries are not disjoint")
            seen.update(actual)
            self._assert_optimizer_owns_exactly(expected, state.optimizer)

    @staticmethod
    def _assert_optimizer_owns_exactly(expected_ids: frozenset[int], optimizer: Any) -> None:
        optimizer_ids = _optimizer_parameter_ids(optimizer)
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise TrainingIsolationError("optimizer contains duplicate adapter parameters")
        if frozenset(optimizer_ids) != expected_ids:
            raise TrainingIsolationError("optimizer parameter set does not match its run adapter")

    def _assert_inactive_gradients_absent(self, active_run_id: str) -> None:
        for run_id, state in self._runs.items():
            if run_id == active_run_id:
                continue
            if any(parameter.grad is not None for parameter in state.adapter_parameters):
                raise TrainingIsolationError(
                    f"inactive adapter for run {run_id} received gradients"
                )

    def _assert_frozen_base(self) -> None:
        for parameter in self._frozen_base_parameters:
            if parameter.requires_grad or parameter.grad is not None:
                raise TrainingIsolationError("shared base parameter became trainable")

    def _clear_non_active_gradients(self, active_run_id: str) -> None:
        for run_id, state in self._runs.items():
            if run_id == active_run_id:
                continue
            for parameter in state.adapter_parameters:
                parameter.grad = None

    def _clear_all_gradients(self) -> None:
        for parameter in self._model.parameters():
            parameter.grad = None

    def _freeze_all_parameters(self) -> None:
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _export_version(self, adapter_name: str, run_id: str, version: int) -> Path:
        run_dir = self._adapter_root / hashlib.sha256(run_id.encode()).hexdigest()[:24]
        final_dir = run_dir / f"v{version}"
        if final_dir.exists():
            raise FileExistsError(f"adapter version directory already exists: {final_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        staging = run_dir / f".v{version}.{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        try:
            self._adapter_exporter(self._model, adapter_name, staging)
            if not (staging / "adapter_config.json").is_file():
                raise TrainingIsolationError("adapter export has no adapter_config.json")
            tensor_files = (staging / "adapter_model.safetensors", staging / "adapter_model.bin")
            if not any(path.is_file() and path.stat().st_size > 0 for path in tensor_files):
                raise TrainingIsolationError("adapter export has no non-empty tensor file")
            os.replace(staging, final_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return final_dir

    def _delete_adapter(self, adapter_name: str) -> None:
        delete_adapter = getattr(self._model, "delete_adapter", None)
        if delete_adapter is not None:
            delete_adapter(adapter_name)
        self._freeze_all_parameters()

    @staticmethod
    def _validate_checkpoint_identity(
        state: TrainingRunState, checkpoint: Mapping[str, Any]
    ) -> None:
        if int(checkpoint.get("format_version", -1)) != 1:
            raise ValueError("unsupported shared training checkpoint format")
        if checkpoint.get("run_id") != state.run_id:
            raise ValueError("checkpoint run id does not match the requested run")
        if checkpoint.get("adapter_name") != state.adapter_name:
            raise ValueError("checkpoint adapter name does not match the requested run")
