"""Flash's SFT loop owns checkpoint scheduling and tracking around upstream training batches."""

from omegaconf import OmegaConf
from tensordict import NonTensorData
from verl.utils import tensordict_utils as tu

from flash.engine.worker.train.core.child.tracking import OwnedTracking


def _train_batch(trainer, batch, step):
    data = tu.get_tensordict(
        tensor_dict=batch,
        non_tensor_dict={
            "use_remove_padding": trainer.config.model.use_remove_padding,
            "use_dynamic_bsz": trainer.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": trainer.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": trainer.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": trainer.global_batch_size,
            "pad_mode": trainer.config.data.pad_mode,
            "pad_token_id": trainer.model_config.tokenizer.pad_token_id,
        },
    )
    lengths = trainer._get_batch_seqlens(data)
    tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=NonTensorData(lengths))
    if step == trainer.start_profile_step:
        trainer.training_client.start_profile()
    output = trainer.training_client.train_batch(data=data)
    trainer.training_client.step_profile()
    if step == trainer.end_profile_step:
        trainer.training_client.stop_profile()
    if not trainer.engine.is_mp_src_rank_with_outputs():
        return None
    metrics = dict(tu.get(output, "metrics"))
    for name in ("loss", "grad_norm", "lr", "mfu"):
        if name in metrics:
            metrics[f"train/{name}"] = metrics.pop(name)
    metrics["train/global_tokens"] = sum(lengths)
    return metrics


def fit_sft(trainer):
    logging = (
        trainer.engine.is_mp_src_rank_with_outputs()
        and trainer.engine.get_data_parallel_rank() == 0
    )
    tracking = None
    succeeded = False
    try:
        if logging:
            tracking = OwnedTracking(
                project_name=trainer.config.trainer.project_name,
                experiment_name=trainer.config.trainer.experiment_name,
                default_backend=trainer.config.trainer.logger,
                config=OmegaConf.to_container(trainer.config, resolve=True),
            )
        step = trainer.resume_global_step
        total_tokens = 0
        if step >= trainer.total_training_steps:
            succeeded = True
            return
        start_epoch = step // trainer.steps_per_epoch
        for epoch in range(start_epoch, trainer.config.trainer.total_epochs):
            trainer.train_sampler.set_epoch(epoch)
            for batch in trainer.train_dataloader:
                step += 1
                metrics = _train_batch(trainer, batch, step)
                if metrics is not None:
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9
                    if tracking is not None:
                        tracking.log(data=metrics, step=step)
                last = step >= trainer.total_training_steps
                if last or (trainer.save_freq > 0 and step % trainer.save_freq == 0):
                    trainer.ckpt_handler.save_checkpoint(step=step)
                if last:
                    succeeded = True
                    return
        if step < trainer.total_training_steps:
            raise RuntimeError("SFT dataset exhausted before the requested training horizon")
        succeeded = True
    finally:
        if tracking is not None:
            tracking.finish(exit_code=0 if succeeded else 1)
