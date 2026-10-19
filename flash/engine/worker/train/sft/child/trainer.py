"""SFT policy expressed through trainer, data loader, and checkpoint instances."""

import os
import time
from functools import partial

from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.sft_trainer import SFTTrainer
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator

from flash.engine.worker.train.core.child.options import RuntimeOptions
from flash.engine.worker.train.core.child.tracking import TrackingOwner
from flash.engine.worker.train.sft.child.dataset import FlashTokenizedSFTDataset


class FlashCheckpointHandler(CheckpointHandler):
    def __init__(self, *, runtime, **kwargs):
        self.runtime = runtime
        super().__init__(**kwargs)

    def save_checkpoint(self, step):
        if self.runtime.should_save(step):
            self._wait_for_publication(step)
            return super().save_checkpoint(step)
        return None

    def _wait_for_publication(self, step):
        for previous in sorted(self.runtime.save_at_steps):
            if previous >= step:
                break
            marker = os.path.join(self.runtime.checkpoint_published_dir, f"step-{previous}")
            next_notice = 0.0
            while not os.path.isfile(marker):
                if time.monotonic() >= next_notice:
                    print(
                        f"[ckpt] waiting for required step {previous} before saving step {step}",
                        flush=True,
                    )
                    next_notice = time.monotonic() + 5.0
                time.sleep(0.1)


class FlashSFTTrainer(TrackingOwner, SFTTrainer):
    def __init__(self, config):
        self.runtime = RuntimeOptions.decode(dict(config.model.runtime_options))
        super().__init__(config)
        # an exhausted stateful iterator must start the next epoch on boundary resume.
        if (
            0 < self.resume_global_step < self.total_training_steps
            and self.resume_global_step % self.steps_per_epoch == 0
        ):
            self._build_dataloader()
            self.ckpt_handler.train_dataloader = self.train_dataloader

    @property
    def checkpoint_handler_class(self):
        return partial(FlashCheckpointHandler, runtime=self.runtime)

    def _build_dataset(self):
        if self.config.data.val_files:
            raise ValueError("Flash SFT does not configure validation datasets")
        self.train_dataset = FlashTokenizedSFTDataset(
            self.config.data.train_files,
            self.model_config.tokenizer,
            self.config.data,
            self.model_config.processor,
            self.config.data.get("train_max_samples", -1),
        )
        self.val_dataset = None

    def _build_dataloader(self):
        dp_size = self.engine.get_data_parallel_size()
        self.global_batch_size = self.config.data.train_batch_size
        if self.global_batch_size % dp_size:
            raise ValueError("SFT global batch must be divisible by data parallel size")
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=False,
            num_replicas=dp_size,
            rank=self.engine.get_data_parallel_rank(),
            drop_last=True,
            seed=self.runtime.seed,
        )
        self.collate_fn = SFTTensorCollator(self.config.data.pad_mode)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=False,
        )
        self.val_dataloader = None
