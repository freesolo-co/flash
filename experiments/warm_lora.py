"""Resident LoRA reset proof, intentionally outside the production training backend.

Run under torchrun on CUDA. This validates model/optimizer/RNG isolation, not the
Flash lifecycle, serving, billing, or a production warm-worker API.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    owner: str
    model_revision: str
    runtime: str
    world_size: int
    seed: int


class ResidentModel:
    """One exclusive owner/profile; every lease starts from an immutable CPU snapshot."""

    def __init__(self, identity, model):
        import threading

        self.identity = identity
        self.model = model
        self._lock = threading.Lock()
        self._healthy = True
        # Include frozen parameters and persistent buffers: requires_grad=False is not a
        # promise that arbitrary training code will never mutate a tensor through .data.
        self._initial = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }

    def run(self, identity, train):
        import random

        import numpy as np
        import torch

        if identity != self.identity:
            raise ValueError("resident model belongs to another owner or runtime profile")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("resident model is already leased")
        try:
            if not self._healthy:
                raise RuntimeError("resident model must be rebuilt after a failed lease")
            self.model.load_state_dict(self._initial, strict=True)
            self.model.zero_grad(set_to_none=True)
            self.model.train()
            random.seed(identity.seed)
            np.random.seed(identity.seed % (2**32))
            torch.manual_seed(identity.seed)
            torch.cuda.manual_seed_all(identity.seed)
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=0.001,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            try:
                return train(self.model, optimizer, scheduler)
            finally:
                optimizer.zero_grad(set_to_none=True)
                del optimizer, scheduler
        except BaseException:
            # Cancellation, OOM, and partially updated state all evict this resident instance.
            self._healthy = False
            raise
        finally:
            self._lock.release()


def _assert_equal(first, second):
    import torch

    if isinstance(first, torch.Tensor):
        left = first.to_local() if hasattr(first, "to_local") else first
        right = second.to_local() if hasattr(second, "to_local") else second
        assert torch.equal(left, right)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_equal(left, right)
    else:
        assert first == second


def main():
    import copy
    from dataclasses import replace

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from torch.distributed.fsdp import fully_shard
    from transformers import LlamaConfig, LlamaForCausalLM

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    identity = Identity(
        "disposable-test-owner", "a" * 40, "tiny-llama-fsdp2-r4", dist.get_world_size(), 123
    )
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )

    def build():
        torch.manual_seed(identity.seed)
        model = get_peft_model(
            LlamaForCausalLM(config),
            LoraConfig(
                r=4,
                lora_alpha=8,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        ).cuda()
        model.register_buffer("mutable_counter", torch.zeros(1, device="cuda"))
        fully_shard(model)
        return ResidentModel(identity, model)

    def train(model, optimizer, scheduler):
        losses = []
        for _ in range(3):
            inputs = torch.randint(0, 128, (2, 16), device="cuda")
            loss = model(input_ids=inputs, labels=inputs).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            model.mutable_counter.add_(1)
            losses.append(loss.detach().clone())
        return copy.deepcopy(
            (losses, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
        )

    started = time.monotonic()
    resident = build()
    first = resident.run(identity, train)
    cold_seconds = time.monotonic() - started
    started = time.monotonic()
    second = resident.run(identity, train)
    warm_seconds = time.monotonic() - started
    _assert_equal(first, second)
    _assert_equal(first, build().run(identity, train))
    try:
        resident.run(replace(identity, owner="other-owner"), train)
    except ValueError:
        pass
    else:
        raise AssertionError("foreign owner accepted")

    def fail(*_):
        raise RuntimeError("injected cancellation")

    for callback in (fail, train):
        try:
            resident.run(identity, callback)
        except RuntimeError:
            pass
        else:
            raise AssertionError("failed lease reused")
    if rank == 0:
        print(
            "FLASH_WARM_RESULT "
            + json.dumps(
                {
                    "world_size": dist.get_world_size(),
                    "cold_seconds": cold_seconds,
                    "warm_seconds": warm_seconds,
                    "loss_model_optimizer_scheduler_equal": True,
                    "foreign_owner_rejected": True,
                    "failed_lease_evicted": True,
                    "fixture_only": True,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
