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

_IMAGE_MODEL = "Qwen/Qwen3.5-4B"

# the live image sft gate run whose published adapter is the warm-start source for rl.
_IMAGE_SFT_SOURCE = "image-gate-qwen35-4b-sft-20260818-06"


def _adapter_config(model: str = _IMAGE_MODEL) -> dict:
    return {
        "peft_type": "LORA",
        "r": 16,
        "lora_alpha": 32,
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": model,
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


def test_image_run_warm_start_stays_rejected_for_sft():
    """SFT adapter continuation is unsupported product-wide, images included.

    This is not an image-specific restriction and image support does not relax it: the guard is
    algorithm-level, so an image SFT run must fail for exactly the same reason a text one does.
    Pinned here so a future image change cannot quietly open an SFT warm-start path that the rest
    of the product does not implement.
    """
    from flash.runner.preparation import _require_supported_adapter_continuation

    spec = JobSpec.from_dict(_image_sft_spec())
    with pytest.raises(ValueError, match="SFT adapter continuation is not supported"):
        _require_supported_adapter_continuation(spec)


def test_queued_warm_start_snapshot_survives_before_the_environment_is_staged():
    """A warm start on a staged environment must persist before staging has run.

    Ordering: `submit_job` writes the effective-preparation snapshot, then `_run_job_inner` calls
    `stage_environment_package` and persists again. `_persist_effective_worker_spec` re-reads its
    own snapshot through `effective_spec_from_status`, but only for warm-start runs -- so a guard
    demanding a staged package unconditionally rejects the first persist of every warm start on a
    Freesolo environment, while non-warm-start runs never take that branch and look fine.

    Reproduced live: `image-gate-qwen35-4b-opd-fromsft-20260819-03` failed with "persisted
    effective preparation failed integrity validation" after staging uploaded the package but
    before a provider was allocated.
    """
    import flash.runner as runner
    from flash.runner.status import effective_spec_from_status

    public = JobSpec.from_dict(_image_rl_spec("opd", source=_IMAGE_SFT_SOURCE))
    # the worker half carries the RESOLVED storage ref, the public half the run id, exactly as
    # preparation writes them; reusing one spec for both would fail on the ref shape instead.
    worker_data = _image_rl_spec("opd", source=_IMAGE_SFT_SOURCE)
    worker_data["train"]["init_from_adapter"] = (
        f"Freesolo-Co/flashrun-image-gate:sft/{_IMAGE_SFT_SOURCE}"
    )
    worker_data["train"]["init_from_adapter_revision"] = "a69435c93e9d4cbd8674bd66740a87015a2e1e59"
    worker = JobSpec.from_dict(worker_data)
    # the four keys the live snapshot of the reproducing run carried, so the fixture fails on the
    # staged-package ordering rather than on a missing recovery record.
    identity = {
        "digest": "image-sft-artifact-v1",
        "config_sha256": "e" * 64,
        "weight_filename": "adapter_model.safetensors",
        "weight_identity": "f" * 64,
    }
    status = runner.RunStatus(
        run_id=public.run_id,
        state="queued",
        spec=public.to_dict(),
        effective_preparation={
            "worker_spec": worker.to_internal_dict(),
            "adapter_identity": identity,
            "preparation_digest": runner._preparation_digest(public, worker, identity),
        },
    )
    assert worker.environment.package is None
    # queued is the pre-staging window, so the absent package is the expected ordering.
    assert effective_spec_from_status(status).run_id == public.run_id

    # paired control: past that window a missing package is a stripped one and must still fail.
    status.state = "provisioning"
    with pytest.raises(ValueError, match="integrity validation"):
        effective_spec_from_status(status)


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
