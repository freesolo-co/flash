"""serving profile registry: catalog agreement, fail-closed lookup, exact placements."""

from __future__ import annotations

from dataclasses import replace

import pytest

from flash.core.catalog import MODELS, get_model, supports_image_training
from flash.serve.control import ModalPlacement
from flash.serve.deployment.profiles import (
    SERVE_RUNTIME_FAMILY,
    ProfileError,
    ServingProfile,
    get_profile,
    placement_for,
    supported_models,
)
from flash.serve.provisioning import ServingImage
from flash.serve.runtime.multimodal import _MAX_IMAGES
from flash.serving.src.engine.model_config import reasoning_parser_for
from flash.serving.src.store.settings import KV_CACHE_DTYPE

MODEL = "Qwen/Qwen3.5-9B"
MODELS_WITH_PROFILES = tuple(sorted(MODELS))
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


def test_supported_models_exactly_cover_the_public_catalog() -> None:
    models = supported_models()

    assert models == tuple(sorted(MODELS))
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


def test_profiles_carry_the_qwen3_reasoning_parser() -> None:
    for model_id in supported_models():
        assert get_profile(model_id).reasoning_parser == "qwen3"
        if model_id != "Qwen/Qwen3.8-27B":
            assert reasoning_parser_for(model_id) == "qwen3"


def test_profiles_keep_the_validated_fp8_kv_cache() -> None:
    # unlike `quantization` above, the KV cache dtype is the engine's own and no checkpoint
    # rejects it. hosted serving runs fp8 for every base because it halves KV bytes, and the
    # 32k context / rank-128 / 16-hot-lora shape below was validated on that footprint.
    #
    # `None` is not a harmless "use the default": `engine_config_from_manifest` omits the key
    # entirely, so vllm falls back to auto and the same card holds half the cache blocks -- fewer
    # concurrent requests and preemption under load at the context the profile advertises.
    for model_id in supported_models():
        assert get_profile(model_id).kv_cache_dtype == KV_CACHE_DTYPE, (
            f"{model_id} would serve a different KV footprint than the validated shape"
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


@pytest.mark.parametrize("model_id", MODELS_WITH_PROFILES)
def test_profile_matches_the_catalog_serving_capacity(model_id: str) -> None:
    profile = get_profile(model_id)
    serving = get_model(model_id).serving
    assert serving is not None

    assert profile.max_model_len == serving.max_model_len
    assert profile.max_num_seqs == serving.max_num_seqs
    assert profile.max_num_batched_tokens == (serving.max_num_batched_tokens or None)
    assert profile.max_loras == serving.max_loras
    assert profile.max_cpu_loras == serving.max_cpu_loras
    assert profile.max_lora_rank == serving.max_lora_rank
    assert profile.tensor_parallel_size == serving.tensor_parallel_size
    assert profile.gpu_memory_utilization == serving.gpu_memory_utilization
    assert profile.image_limit == serving.image_limit
    assert profile.served_model == serving.serve_model_id
    # SHAPE agrees with the catalog (above); the CARD deliberately does not. serving.gpu is the
    # freesolo-owned hosted plane's card, modal_gpu is the card THIS customer-owned profile was
    # live-qualified on. Pinned per model rather than asserted "different", so a profile silently
    # drifting to some other unqualified card still fails here.
    assert serving.gpu == "B200"  # every hosted tier
    assert (
        profile.modal_gpu
        == {
            "Qwen/Qwen3.5-9B": "L40S",
            "Qwen/Qwen3.8-27B": "H100",
            "Qwen/Qwen3.6-35B-A3B": "H200",
        }[model_id]
    )
    assert profile.modal_live_qualified is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_model_len", 65536),
        ("max_lora_rank", 64),
        ("max_num_seqs", 4),
        ("max_num_batched_tokens", 2048),
        ("max_cpu_loras", 32),
        ("tensor_parallel_size", 2),
        ("gpu_memory_utilization", 0.5),
        ("image_limit", 2),
        ("served_model", "Freesolo-Co/other"),
    ],
)
def test_profile_drifting_from_the_catalog_is_rejected(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    # the catalog is the advertised serving capacity contract. a profile that quietly serves a
    # longer context or a higher rank than the catalog would deploy a shape no gate checked.
    from flash.serve.deployment import profiles

    drifted = replace(get_profile(MODEL), **{field: value})
    monkeypatch.setitem(profiles._PROFILES, MODEL, drifted)

    with pytest.raises(ProfileError) as excinfo:
        get_profile(MODEL)

    assert "disagrees with the catalog" in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("modal_live_qualified", 1, "must be an exact bool"),
        ("served_model", " ", "must be a nonempty unpadded string"),
        ("modal_gpu", "", "must be a nonempty unpadded string"),
        ("modality", "text", "text engines cannot declare an image_limit"),
        ("max_loras", 0, "max_loras must be a positive integer"),
        ("served_model_revision", "A" * 40, "not an immutable commit"),
        ("engine_args", {"bad": object()}, "invalid engine inputs"),
    ],
)
def test_structurally_invalid_profile_fails_the_whole_registry(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    from flash.serve.deployment import profiles

    monkeypatch.setitem(profiles._PROFILES, MODEL, replace(get_profile(MODEL), **{field: value}))

    with pytest.raises(ProfileError, match=message):
        supported_models()


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "sha512:" + "a" * 64,
        "sha256:" + "g" * 64,
    ],
)
def test_invalid_certified_image_digest_fails_the_whole_registry(
    monkeypatch: pytest.MonkeyPatch,
    digest: str,
) -> None:
    from flash.serve.deployment import profiles

    monkeypatch.setitem(
        profiles._PROFILES,
        MODEL,
        replace(get_profile(MODEL), modal_certified_image_digest=digest),
    )

    with pytest.raises(
        ProfileError, match="must be sha256: followed by 64 lowercase hex characters"
    ):
        supported_models()


def test_missing_profile_fails_the_whole_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from flash.serve.deployment import profiles

    monkeypatch.delitem(profiles._PROFILES, "Qwen/Qwen3.8-27B")
    with pytest.raises(ProfileError, match=r"missing=Qwen/Qwen3\.8-27B"):
        get_profile(MODEL)


def test_extra_profile_fails_the_whole_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from flash.serve.deployment import profiles

    monkeypatch.setitem(
        profiles._PROFILES, "extra/model", replace(get_profile(MODEL), model_id="extra/model")
    )
    with pytest.raises(ProfileError, match="extra=extra/model"):
        supported_models()


def test_profile_key_must_match_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from flash.serve.deployment import profiles

    monkeypatch.setitem(
        profiles._PROFILES,
        "Qwen/Qwen3.8-27B",
        replace(get_profile("Qwen/Qwen3.8-27B"), model_id="Qwen/Qwen3.5-9B"),
    )
    with pytest.raises(ProfileError, match=r"profile\.model_id"):
        get_profile(MODEL)


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
    assert placement.gpu == profile.modal_gpu_request
    # _validate_placement rejects a None region, so a placement built without one could never be
    # provisioned. asserting it here keeps the profile from silently reverting to that.
    assert placement.region == "us-east"
    # DeploymentSpec requires placement gpu_count == engine tensor_parallel_size.
    assert placement.gpu_count == profile.tensor_parallel_size


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


def test_image_capable_profiles_enable_tower_connector_lora() -> None:
    # flash trains image loras with target_modules="all-linear", which peft resolves to include the
    # vision tower: real published image adapters carry 196 visual.* tensors out of 692
    # (model.visual.blocks.N.attn.qkv/proj, .mlp.linear_fc1/fc2, model.visual.merger.*).
    #
    # vllm only wraps those modules when enable_tower_connector_lora is true. with it false they are
    # absent from expected_lora_modules, and LoRAModel.from_local_checkpoint RAISES ValueError
    # ("expected target modules in ... but received ...") rather than dropping them. none of the
    # four vision suffixes (qkv, proj, linear_fc1, linear_fc2) collide with a language-model
    # suffix, so the rejection is certain, not incidental: serving an image adapter would fail.
    for model_id in supported_models():
        profile = get_profile(model_id)
        if profile.modality != "multimodal":
            continue
        assert profile.enable_tower_connector_lora, (
            f"{model_id} serves images but disables tower connector lora, so vllm would reject "
            "every adapter trained on images"
        )


def test_model_specific_checkpoint_and_scheduler_choices() -> None:
    nine = get_profile("Qwen/Qwen3.5-9B")
    twenty_seven = get_profile("Qwen/Qwen3.8-27B")
    thirty_five = get_profile("Qwen/Qwen3.6-35B-A3B")

    assert nine.served_model == "Freesolo-Co/Qwen3.5-9B-FP8"
    assert twenty_seven.served_model == "Qwen/Qwen3.8-27B-FP8"
    assert twenty_seven.served_model_revision == "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
    assert twenty_seven.tokenizer_model == "Qwen/Qwen3.8-27B"
    assert twenty_seven.modal_gpu == "H100"
    assert twenty_seven.modal_gpu_request == "H100!"
    assert twenty_seven.modal_live_qualified is True
    assert twenty_seven.modal_certified_image_digest == (
        "sha256:2bf27b51f6e4b7f0b2d805d96202579d94868e2c594b7c496777d350ad6936f6"
    )
    assert thirty_five.served_model == "Qwen/Qwen3.6-35B-A3B"
    assert thirty_five.modal_gpu_request == "H200"
    assert thirty_five.quantization is None
    assert thirty_five.tensor_parallel_size == 1
    assert thirty_five.max_num_batched_tokens == 4096
    assert thirty_five.max_num_seqs == 8
    assert thirty_five.max_loras == 6
    assert thirty_five.max_lora_rank == 64
    assert thirty_five.modal_live_qualified is True
    assert thirty_five.modal_certified_image_digest == twenty_seven.modal_certified_image_digest
    assert nine.modal_certified_image_digest is None


@pytest.mark.parametrize(
    "supplied",
    [
        {"workspace_name": "workspace", "region": "us-east"},
        {"environment": "dev", "region": "us-east"},
        {"workspace_name": "workspace", "environment": "dev"},
    ],
)
def test_incomplete_placement_inputs_are_rejected(supplied: dict) -> None:
    profile = get_profile(MODEL)

    with pytest.raises(ProfileError) as excinfo:
        placement_for(profile, "modal", **supplied)  # type: ignore[arg-type]

    assert "requires" in str(excinfo.value)
