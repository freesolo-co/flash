"""Warm-start reference initialization on loaded and meta-only FSDP ranks."""

from __future__ import annotations

import warnings
from datetime import timedelta

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
accelerate = pytest.importorskip("accelerate")

from flash.engine.worker.train.core.child.reference import (
    adapter_name,
    snapshot,
    use_ref_adapter,
    validate_snapshot,
)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.q_proj(x)


@pytest.fixture
def adapter_path(tmp_path):
    torch.manual_seed(17)
    model = peft.get_peft_model(
        Tiny(), peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.25)
    model.save_pretrained(tmp_path)
    return str(tmp_path)


def load_adapter(path, *, meta):
    if meta:
        with accelerate.init_empty_weights():
            base = Tiny()
    else:
        base = Tiny()
    # PEFT's checkpoint copy into meta parameters is a no-op on secondary ranks;
    # the later FSDP broadcast supplies their values from rank zero.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")
        return peft.PeftModel.from_pretrained(base, path, is_trainable=True)


def test_meta_warm_start_preserves_placeholders_until_fsdp(adapter_path):
    model = load_adapter(adapter_path, meta=True)
    parameters = set(dict(model.named_parameters()))
    state = set(model.state_dict())
    assert all(value.is_meta for value in model.parameters())

    # This exact saved-adapter path used to raise "Cannot copy out of meta tensor".
    assert snapshot(model) is model
    reference = dict(model.named_buffers())
    assert len(reference) == 2
    assert all(value.is_meta for value in reference.values())
    assert set(dict(model.named_parameters())) == parameters
    assert set(model.state_dict()) == state
    with pytest.raises(RuntimeError, match="not materialized after FSDP"):
        validate_snapshot(model)


def test_reference_keeps_source_dtype_and_independent_storage(adapter_path):
    model = load_adapter(adapter_path, meta=False).to(dtype=torch.bfloat16)
    snapshot(model)
    validate_snapshot(model)
    for name, frozen in model.named_buffers():
        source = model.get_parameter(name.replace(f".{adapter_name}.", ".default."))
        assert frozen.dtype == source.dtype == torch.bfloat16
        assert frozen.device == source.device
        assert torch.equal(frozen, source)
        assert frozen.data_ptr() != source.data_ptr()


def test_missing_reference_fails_validation():
    with pytest.raises(RuntimeError, match="no frozen buffers"):
        validate_snapshot(Tiny())


def _distributed_reference(rank, rendezvous, path):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=45)
    )
    try:
        torch.manual_seed(29)
        model = load_adapter(path, meta=rank != 0)
        snapshot(model)
        state = model.state_dict()
        if rank == 0:
            expected = model(torch.ones(2, 8)).detach().clone()
            with model.disable_adapter():
                assert not torch.allclose(model(torch.ones(2, 8)), expected)
        else:
            expected = torch.empty(2, 8)
        torch.distributed.broadcast(expected, src=0)

        # Exercise real FSDP2 and the same public Torch load + buffer broadcast
        # sequence as pinned Verl's fsdp2_load_full_state_dict, on CPU/Gloo.
        # Nonpersistent references are deliberately absent from state_dict.
        assert not any(adapter_name in name for name in state)
        fully_shard(model, mesh=init_device_mesh("cpu", (2,)))
        if rank != 0:
            model.to_empty(device="cpu")
        set_model_state_dict(
            model,
            state,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        for _name, buf in sorted(model.named_buffers()):
            torch.distributed.broadcast(buf, src=0)
        # As in the engine, create the optimizer after sharding and before any
        # forward swaps sharded parameters for temporary all-gathered tensors.
        optimizer = torch.optim.SGD(
            [param for param in model.parameters() if param.requires_grad], lr=0.1
        )
        validate_snapshot(model)
        with use_ref_adapter(model):
            assert torch.allclose(model(torch.ones(2, 8)), expected)

        model(torch.ones(2, 8)).square().mean().backward()
        optimizer.step()
        trained = model(torch.ones(2, 8)).detach().clone()
        assert not torch.allclose(trained, expected)
        with use_ref_adapter(model):
            assert torch.allclose(model(torch.ones(2, 8)), expected)
        assert torch.equal(model(torch.ones(2, 8)), trained)
        assert not any(adapter_name in name for name in dict(model.named_parameters()))
        assert not any(adapter_name in name for name in model.state_dict())
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="requires the CPU Gloo backend",
)
def test_two_ranks_receive_the_same_frozen_reference(adapter_path, tmp_path):
    rendezvous = (tmp_path / "gloo-init").as_uri()
    torch.multiprocessing.spawn(
        _distributed_reference, args=(rendezvous, adapter_path), nprocs=2, join=True
    )
