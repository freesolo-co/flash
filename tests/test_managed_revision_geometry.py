"""A catalog model's own managed revision keeps its layer geometry for VRAM sizing.

Flash pins catalog models to ``managed_revision`` automatically. Sizing used to drop catalog
geometry for any pinned revision, so a GDN-hybrid such as Qwen3.5-9B-Base sized its GRPO rollout
KV as if every layer were full attention: a 24k-context run needed >=209 GB (an 8-GPU node)
instead of ~115 GB (one H200). Other pinned revisions stay unverified.
"""

from types import SimpleNamespace

import pytest

from flash.core.catalog import get_model, vocab_size_for
from flash.engine.plan import vram

MODEL = "Qwen/Qwen3.5-9B-Base"


def _train(context):
    return SimpleNamespace(
        max_context_tokens=context,
        max_completion_tokens=2048,
        group_size=8,
        prompts_per_step=8,
        lora_rank=32,
        batch_size=None,
    )


@pytest.fixture
def pinned_geometry(monkeypatch):
    info = get_model(MODEL)
    monkeypatch.setattr(
        vram,
        "_validated_revision_geometry",
        lambda model_id, revision, info: (info.params_b, vocab_size_for(model_id)),
    )
    return info


def test_only_the_catalog_managed_revision_is_trusted():
    info = get_model(MODEL)
    assert info.managed_revision
    assert vram.catalog_geometry_trusted(info, "")
    assert vram.catalog_geometry_trusted(info, info.managed_revision)
    assert not vram.catalog_geometry_trusted(info, "0" * 40)
    assert not vram.catalog_geometry_trusted(None, info.managed_revision)
    assert not vram.catalog_geometry_trusted(SimpleNamespace(managed_revision=""), "a" * 40)


@pytest.mark.parametrize("context", [12288, 16384, 24576])
def test_managed_pin_sizes_like_the_catalog_model(pinned_geometry, context):
    managed = vram.model_required_vram_gb(
        MODEL, "grpo", train=_train(context), model_revision=pinned_geometry.managed_revision
    )
    unpinned = vram.model_required_vram_gb(MODEL, "grpo", train=_train(context))
    unrelated = vram.model_required_vram_gb(
        MODEL, "grpo", train=_train(context), model_revision="0" * 40
    )
    assert managed == unpinned
    assert managed < unrelated


def test_24k_qwen35_9b_grpo_fits_one_h200(pinned_geometry):
    need = vram.model_required_vram_gb(
        MODEL, "grpo", train=_train(24576), model_revision=pinned_geometry.managed_revision
    )
    assert need <= 141


def test_worker_reserves_rollout_memory_from_managed_geometry(monkeypatch):
    from flash.engine.worker.train.rl.launch.verl_config import resolve_gpu_mem_util

    info = get_model(MODEL)
    monkeypatch.setattr("flash.engine.plan.vram.resolve_params_b", lambda *a, **kw: info.params_b)

    def util(revision):
        return resolve_gpu_mem_util(
            {
                "model_id": MODEL,
                "model_revision": revision,
                "engine_len": 24576,
                "group_size": 8,
                "lora_rank": 32,
            },
            gpu_type="H200",
            n_gpus=1,
            fp8_kv=False,
            sleep_unsupported=False,
        )

    assert util(info.managed_revision) <= util("0" * 40)
