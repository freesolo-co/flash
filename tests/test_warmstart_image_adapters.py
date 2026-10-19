"""Warm start from an adapter published by an image-bearing run.

Warm start and image support were built independently, so nothing pinned their intersection: no
test in the warm-start files referenced a processor, an image, or a multimodal model. That gap
matters because the adapter an image run publishes has a different file set from a text run's --
the checkpoint exporter copies processor sidecars next to the LoRA weights -- and because a
warm-started image run has to re-derive its processor from the base model rather than from the
source adapter.

The file lists here are the exact set published by the live 4B image gate runs
(`image-gate-qwen35-4b-sft-20260818-06` and `image-gate-qwen35-4b-opd-20260818-06`), so a change to
the exporter that drops a sidecar, or a change to the loadable check that starts demanding one,
turns these red.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save

from flash.core.spec import JobSpec
from flash.engine.worker.model.adapter import (
    _warmstart_adapter_is_loadable,
    validate_warmstart_adapter,
)

# exactly what the live image sft gate published: seven files, and note that
# `preprocessor_config.json` is NOT among them. an image run's processor is re-derived from the
# pinned base model, so warm start must not require that sidecar to be present.
_PUBLISHED_IMAGE_ADAPTER_SIDECARS = (
    "chat_template.jinja",
    "tokenizer_config.json",
    "base_model_provenance.json",
    "tokenizer.json",
    "processor_config.json",
)

_IMAGE_MODEL = "Qwen/Qwen3.5-9B"

# the live image sft gate run whose published adapter is the warm-start source for rl.
_IMAGE_SFT_SOURCE = "image-gate-qwen35-4b-sft-20260818-06"


def _adapter_config(model: str = _IMAGE_MODEL) -> dict:
    return {
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model,
        "exclude_modules": None,
    }


def _write_image_adapter(adir, *, sidecars=_PUBLISHED_IMAGE_ADAPTER_SIDECARS) -> None:
    """Materialize the file set an image run's checkpoint export actually publishes."""
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "adapter_config.json").write_text(json.dumps(_adapter_config()), encoding="utf-8")
    (adir / "adapter_model.safetensors").write_bytes(
        save(
            {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": np.zeros(
                    (16, 8), dtype=np.float32
                ),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": np.zeros(
                    (8, 16), dtype=np.float32
                ),
            }
        )
    )
    for name in sidecars:
        (adir / name).write_text("{}", encoding="utf-8")


def test_image_run_adapter_is_loadable_without_a_preprocessor_sidecar(tmp_path):
    """The published image adapter satisfies the worker's warm-start completeness check.

    The exporter copies `processor_config.json` but not `preprocessor_config.json`. If the loadable
    check ever grew a requirement for the latter, every warm start from an image run would fail on
    a healthy, fully published adapter.
    """
    adir = tmp_path / "adapter"
    _write_image_adapter(adir)
    assert not (adir / "preprocessor_config.json").exists()
    assert _warmstart_adapter_is_loadable(str(adir))


def test_image_run_adapter_passes_warmstart_validation(tmp_path):
    """Validation accepts the image adapter and does not demand multimodal-only tensors."""
    adir = tmp_path / "adapter"
    _write_image_adapter(adir)
    validate_warmstart_adapter(_adapter_config(), _IMAGE_MODEL, str(adir))


def test_an_adapter_missing_its_weights_is_still_rejected(tmp_path):
    """Paired control: the check above passes because the adapter is complete, not because it is lax."""
    adir = tmp_path / "adapter"
    _write_image_adapter(adir)
    (adir / "adapter_model.safetensors").unlink()
    assert not _warmstart_adapter_is_loadable(str(adir))


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_image_run_warm_start_is_accepted_for_supported_algorithms(algorithm):
    """GRPO and OPD accept `init_from_adapter` on an image environment.

    Validated live against the published image adapter
    `image-gate-qwen35-4b-opd-20260818-06`: preparation inherited rank 16 / alpha 32, pinned the
    source dataset revision, and propagated the base model pin.
    """
    spec = JobSpec.from_dict(_image_rl_spec(algorithm))
    assert spec.train.init_from_adapter == "image-gate-qwen35-4b-opd-20260818-06"
    assert spec.algorithm == algorithm


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_rl_warm_start_accepts_an_image_sft_adapter_as_its_source(algorithm):
    """SFT -> RL is the canonical warm start, and the SFT source may be an image run.

    The direction matters: SFT cannot itself be warm-started (see the sibling test below), but its
    published adapter is the normal starting point for GRPO and OPD. Nothing about that path was
    pinned for an image-bearing SFT source, whose adapter carries processor sidecars a text run's
    does not.

    Validated live against `image-gate-qwen35-4b-sft-20260818-06`: both target algorithms inherited
    rank 16 / alpha 32, pinned source revision a69435c9, and adopted the runner-chosen base pin
    851bf6e8 with `model_revision_auto` preserved as True so the child stays deployable.
    """
    spec = JobSpec.from_dict(_image_rl_spec(algorithm, source=_IMAGE_SFT_SOURCE))
    assert spec.train.init_from_adapter == _IMAGE_SFT_SOURCE
    assert spec.algorithm == algorithm
    # the source adapter's topology is authoritative, so the public round trip drops both knobs
    # rather than re-validating a combination the parser refuses.
    public_train = spec.to_dict()["train"]
    assert "lora_rank" not in public_train
    assert "lora_alpha" not in public_train


def test_image_sft_warm_start_inherits_the_source_pin_like_any_other_algorithm():
    """An image SFT run continues from an adapter on the same terms as a text one.

    SFT adapter continuation used to be rejected outright, and this test pinned that rejection so
    an image change could not quietly open a path the product did not implement. `dev` then
    implemented it for every algorithm combination, which makes the rejection the stale side: what
    still needs pinning is that image support does not carve out its own variant. So assert the
    inheritance the shared path performs -- an image SFT target takes the source's pin rather than
    resolving its own -- because a self-resolved pin would break warm start the moment the base
    model's hub tip moved, and would leave the child undeployable.
    """
    from dataclasses import replace

    from flash.runner.lifecycle.preparation import _adopted_warmstart_revision

    spec = JobSpec.from_dict(_image_sft_spec())
    assert spec.algorithm == "sft"
    assert spec.train.init_from_adapter == _IMAGE_SFT_SOURCE
    assert not spec.model_revision

    source_revision = "851bf6e8" * 5
    source = replace(spec, model_revision=source_revision, model_revision_auto=True)
    inherited = _adopted_warmstart_revision(spec, source)

    assert inherited.model_revision == source_revision
    assert inherited.model_revision_auto is True


def _image_environment() -> dict:
    return {
        "id": (
            "github:freesolo-co/environments@0bcf1e0fbbfe9b083da376e91d0689d9efc24a91:"
            "image-gate/environment.py"
        ),
        "params": {"mode": "rl"},
    }


def _image_rl_spec(algorithm: str, source: str = "image-gate-qwen35-4b-opd-20260818-06") -> dict:
    train = {
        "max_steps": 3,
        "max_examples": 1,
        "prompts_per_step": 1,
        "group_size": 4,
        "max_completion_tokens": 16,
        "max_context_tokens": 1024,
        "init_from_adapter": source,
    }
    if algorithm == "opd":
        train["teacher_model"] = "qwen3.5-397b-a17b"
        train["group_size"] = 1
    return {
        "model": _IMAGE_MODEL,
        "algorithm": algorithm,
        "seed": 42,
        "run_id": f"image-warmstart-{algorithm}",
        "environment": _image_environment(),
        "train": train,
        "gpu": {"provider": "runpod", "type": "H100", "count": 1},
    }


def _image_sft_spec() -> dict:
    environment = _image_environment()
    environment["params"] = {"mode": "sft", "split": "sft"}
    return {
        "model": _IMAGE_MODEL,
        "algorithm": "sft",
        "seed": 42,
        "run_id": "image-warmstart-sft",
        "environment": environment,
        "train": {
            "epochs": 1,
            "max_examples": 1,
            "batch_size": 1,
            "max_steps": 1,
            "max_context_tokens": 512,
            "init_from_adapter": "image-gate-qwen35-4b-sft-20260818-06",
        },
        "gpu": {"provider": "runpod", "type": "H100", "count": 1},
    }
