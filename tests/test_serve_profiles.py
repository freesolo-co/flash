"""serving profile registry: catalog agreement, fail-closed lookup, exact placements."""

from __future__ import annotations

from dataclasses import replace

import pytest

from flash.core.catalog import get_model, supports_image_training
from flash.serve.control import ModalPlacement, RunPodPlacement
from flash.serve.profiles import (
    SERVE_RUNTIME_FAMILY,
    ProfileError,
    ServingProfile,
    get_profile,
    placement_for,
    supported_models,
)
from flash.serve.provisioning import ServingImage
from flash.serve.runtime.multimodal import _MAX_IMAGES

MODEL = "Qwen/Qwen3.5-9B"
DIGEST = "sha256:" + "a" * 64
REVISION = "b" * 40


def _image(digest: str = DIGEST) -> ServingImage:
    return ServingImage(
        reference=f"ghcr.io/freesolo-co/freesolo-flash-serve@{digest}",
        digest=digest,
    )


def _engine(profile: ServingProfile, **overrides: object):
    kwargs: dict[str, object] = {
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "image": _image(),
    }
    kwargs.update(overrides)
    return profile.engine(**kwargs)  # type: ignore[arg-type]


def test_supported_models_are_all_resolvable() -> None:
    models = supported_models()

    assert models, "the registry must expose at least one profile"
    for model_id in models:
        assert get_profile(model_id).model_id == model_id


def test_image_trainable_models_are_served_image_capable() -> None:
    # flash trains image loras on these models, so serving them text-only ships a dead end: the
    # engine loads no processor and passes no limit_mm_per_prompt, and every image request fails
    # with MultimodalRequestError even though the adapter was trained on images. both served
    # checkpoints are Qwen3_5ForConditionalGeneration with a vision_config, so the capability is
    # in the weights; only the profile decided not to expose it.
    for model_id in supported_models():
        profile = get_profile(model_id)
        if not supports_image_training(model_id):
            continue
        assert profile.modality == "multimodal", (
            f"{model_id} supports image training but is served as {profile.modality!r}"
        )
        assert profile.image_limit is not None, (
            f"{model_id} supports image training but declares no image_limit"
        )
        assert profile.image_limit > 0, (
            f"{model_id} supports image training but declares image_limit={profile.image_limit!r}"
        )
        # the runtime clamps to _MAX_IMAGES, so advertising more than that would promise a
        # capacity the request path silently trims away.
        assert profile.image_limit <= _MAX_IMAGES, (
            f"{model_id} advertises {profile.image_limit} images above the runtime cap {_MAX_IMAGES}"
        )


def test_no_profile_forces_a_quantization_the_checkpoint_would_reject() -> None:
    # every served checkpoint declares quant_method "compressed-tensors" in its own config, and
    # vllm treats a disagreeing `quantization` argument as a hard ValidationError rather than a
    # hint: "Quantization method specified in the model config (compressed-tensors) does not
    # match the quantization method specified in the `quantization` argument (fp8)". that killed
    # a live canary at engine construction, after the weights had already downloaded.
    #
    # leaving it unset lets vllm read the method from the checkpoint, which is the authority. a
    # future profile may set this only for a checkpoint that declares the same method.
    for model_id in supported_models():
        profile = get_profile(model_id)
        assert profile.quantization is None, (
            f"{model_id} forces quantization={profile.quantization!r}; the checkpoint declares its "
            "own method and vllm rejects an argument that disagrees"
        )


def test_unknown_model_fails_closed_rather_than_defaulting() -> None:
    # a guessed placement is a real gpu rental in the customer's account, so an unlisted model
    # must raise instead of falling back to any other profile.
    #
    # the id is derived from the registry rather than hardcoded: this test previously named a real
    # catalog model that simply had no profile yet, so adding that profile turned the test red for
    # the wrong reason. an id built from the registry cannot become registered behind the test's
    # back, which keeps this asserting "unlisted fails closed" rather than "this model is absent".
    unknown = "unlisted/" + "-".join(sorted(supported_models()))[:64]
    assert unknown not in supported_models()

    with pytest.raises(ProfileError) as excinfo:
        get_profile(unknown)

    assert "no customer-owned serving profile" in str(excinfo.value)


def test_profile_matches_the_catalog_serving_capacity() -> None:
    profile = get_profile(MODEL)
    serving = get_model(MODEL).serving
    assert serving is not None

    assert profile.max_model_len == serving.max_model_len
    assert profile.max_num_seqs == serving.max_num_seqs
    assert profile.max_loras == serving.max_loras
    assert profile.max_lora_rank == serving.max_lora_rank
    assert profile.gpu_memory_utilization == serving.gpu_memory_utilization
    assert profile.served_model == serving.serve_model_id
    assert profile.modal_gpu == serving.gpu


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_model_len", 65536),
        ("max_lora_rank", 64),
        ("max_num_seqs", 4),
        ("gpu_memory_utilization", 0.5),
        ("served_model", "Freesolo-Co/other"),
        ("modal_gpu", "H100"),
    ],
)
def test_profile_drifting_from_the_catalog_is_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    # the catalog is the advertised serving capacity contract. a profile that quietly serves a
    # longer context or a higher rank than the catalog would deploy a shape no gate checked.
    from flash.serve import profiles

    drifted = replace(get_profile(MODEL), **{field: value})
    monkeypatch.setitem(profiles._PROFILES, MODEL, drifted)

    with pytest.raises(ProfileError) as excinfo:
        get_profile(MODEL)

    assert "disagrees with the catalog" in str(excinfo.value)


def test_engine_derives_fingerprints_from_the_carried_kwargs() -> None:
    # build_serving_manifest recomputes these from the same kwargs and rejects a mismatch, so a
    # stored fingerprint would be a second source of truth that can drift.
    from flash.serve.control import canonical_mapping_fingerprint

    profile = get_profile(MODEL)

    engine = _engine(profile)

    assert engine.engine_args_fingerprint == canonical_mapping_fingerprint(profile.engine_args)
    assert engine.tokenizer_kwargs_fingerprint == canonical_mapping_fingerprint(
        profile.tokenizer_kwargs
    )
    assert engine.processor_kwargs_fingerprint == canonical_mapping_fingerprint(
        profile.processor_kwargs
    )


def test_engine_binds_the_supplied_image_and_revisions() -> None:
    profile = get_profile(MODEL)
    other = "sha256:" + "c" * 64

    engine = _engine(profile, image=_image(other))

    assert engine.image_digest == other
    assert engine.model_revision == REVISION
    assert engine.tokenizer_revision == REVISION
    assert engine.runtime_family == SERVE_RUNTIME_FAMILY
    assert engine.trust_remote_code is False


def test_engine_identity_changes_when_any_immutable_input_changes() -> None:
    profile = get_profile(MODEL)

    baseline = _engine(profile).engine_id
    other_image = _engine(profile, image=_image("sha256:" + "c" * 64)).engine_id
    other_revision = _engine(profile, model_revision="c" * 40).engine_id

    assert len({baseline, other_image, other_revision}) == 3


def test_modal_placement_uses_the_validated_gpu_and_matches_tensor_parallelism() -> None:
    profile = get_profile(MODEL)

    placement = placement_for(
        profile, "modal", workspace_name="workspace", environment="dev", region="us-east"
    )

    assert type(placement) is ModalPlacement
    assert placement.gpu == profile.modal_gpu
    # _validate_placement rejects a None region, so a placement built without one could never be
    # provisioned. asserting it here keeps the profile from silently reverting to that.
    assert placement.region == "us-east"
    # DeploymentSpec requires placement gpu_count == engine tensor_parallel_size.
    assert placement.gpu_count == profile.tensor_parallel_size


def test_runpod_placement_uses_the_runpod_gpu_id_not_the_modal_name() -> None:
    profile = get_profile(MODEL)

    placement = placement_for(profile, "runpod", account_id="account", data_center_id="US-KS-2")

    assert type(placement) is RunPodPlacement
    assert placement.gpu_type_id == profile.runpod_gpu.gpu_type_id
    assert placement.gpu_type_id != profile.modal_gpu
    assert placement.gpu_count == profile.tensor_parallel_size
    assert placement.container_disk_gb == profile.runpod_gpu.container_disk_gb
    assert placement.volume_size_gb == profile.runpod_gpu.volume_size_gb


@pytest.mark.parametrize(
    ("provider", "supplied"),
    [
        ("modal", {"workspace_name": "workspace", "region": "us-east"}),
        ("modal", {"environment": "dev", "region": "us-east"}),
        ("modal", {"workspace_name": "workspace", "environment": "dev"}),
        ("runpod", {"account_id": "account"}),
        ("runpod", {"data_center_id": "US-KS-2"}),
    ],
)
def test_incomplete_placement_inputs_are_rejected(provider: str, supplied: dict) -> None:
    profile = get_profile(MODEL)

    with pytest.raises(ProfileError) as excinfo:
        placement_for(profile, provider, **supplied)  # type: ignore[arg-type]

    assert "requires" in str(excinfo.value)


@pytest.mark.parametrize(
    ("provider", "supplied"),
    [
        (
            "modal",
            {
                "workspace_name": "w",
                "environment": "dev",
                "region": "us-east",
                "data_center_id": "US-KS-2",
            },
        ),
        ("runpod", {"account_id": "a", "data_center_id": "US-KS-2", "environment": "dev"}),
        ("runpod", {"account_id": "a", "data_center_id": "US-KS-2", "region": "us-east"}),
    ],
)
def test_foreign_provider_inputs_are_rejected_rather_than_ignored(
    provider: str, supplied: dict
) -> None:
    # silently dropping a runpod data center on a modal deployment would let the caller keep a
    # false belief about where this runs.
    profile = get_profile(MODEL)

    with pytest.raises(ProfileError) as excinfo:
        placement_for(profile, provider, **supplied)  # type: ignore[arg-type]

    assert "does not accept" in str(excinfo.value)


def test_unsupported_provider_is_rejected() -> None:
    profile = get_profile(MODEL)

    with pytest.raises(ProfileError):
        placement_for(profile, "vast")  # type: ignore[arg-type]


def test_profile_mappings_cannot_be_mutated_through_the_registry() -> None:
    # engine_args feed a fingerprint that is part of the engine id. a caller mutating them would
    # change every subsequent engine id derived from this profile.
    profile = get_profile(MODEL)

    with pytest.raises(TypeError):
        profile.engine_args["enforce_eager"] = True  # type: ignore[index]
