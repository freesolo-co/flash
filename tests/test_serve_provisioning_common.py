"""provider-neutral provisioning identities, secret boundaries, and manifest codec."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import pickle
import zlib
from dataclasses import replace

import pytest

from flash.serve.app.manifest import build_serving_manifest, load_serving_manifest
from flash.serve.provisioning import (
    LAUNCHER_ABI_ID,
    MAX_CANONICAL_MANIFEST_BYTES,
    MAX_ENCODED_MANIFEST_BYTES,
    DeploymentBundle,
    SanitizedProviderFailure,
    ServingImage,
    ServingRuntimeSecrets,
    _common,
    base64url_identity,
    decode_manifest_environment,
    encode_manifest_environment,
    failed_deployment_result,
    serving_resource_names,
)
from tests.test_serve_app_manifest import _spec_and_inputs

SECRET = "runtime-secret-sentinel"


def _manifest():
    return build_serving_manifest(*_spec_and_inputs())


def _image(digest: str | None = None, *, tag: bool = True) -> ServingImage:
    selected = digest or _manifest().expected_oci_digest
    tagged = ":release" if tag else ""
    return ServingImage(
        reference=f"registry.example/flash/serve{tagged}@{selected}",
        digest=selected,
    )


def _canonical_payload(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _mutated_manifest(manifest, mutate):
    payload = json.loads(manifest.canonical_json())
    mutate(payload)
    identity_payload = dict(payload)
    identity_payload.pop("manifest_id")
    payload["manifest_id"] = hashlib.sha256(_canonical_payload(identity_payload)).hexdigest()
    return load_serving_manifest(payload)


def test_serving_image_requires_one_canonical_digest_binding() -> None:
    digest = _manifest().expected_oci_digest
    assert _image().reference.endswith("@" + digest)
    assert _image(tag=False).digest == digest

    invalid = (
        ("registry.example/flash/serve:release", digest),
        (f"https://registry.example/flash/serve@{digest}", digest),
        (f"user@registry.example/flash/serve@{digest}", digest),
        (f"registry.example/flash/serve@{digest}?x=1", digest),
        (f"registry.example/flash/serve@{digest}#x", digest),
        (f"registry.example/flash/serve @ {digest}", digest),
        (f"REGISTRY.example/flash/serve@{digest}", digest),
        (f"registry.example/Flash/serve@{digest}", digest),
        (f"flash@{digest}", digest),
        (f"registry.example/flash/serve@{digest}", "sha256:" + "6" * 64),
    )
    for reference, declared_digest in invalid:
        with pytest.raises(ValueError, match=r"."):
            ServingImage(reference, declared_digest)


def test_serving_image_registry_conformance_matrix() -> None:
    digest = _manifest().expected_oci_digest
    valid = (
        "localhost",
        "localhost:5000",
        "registry.example",
        "registry-1.example.internal:443",
        "127.0.0.1",
        "10.20.30.40:65535",
    )
    for registry in valid:
        assert ServingImage(f"{registry}/flash/serve@{digest}", digest)

    invalid = (
        "",
        ".registry.example",
        "registry.example.",
        "registry..example",
        "-registry.example",
        "registry-.example",
        "REGISTRY.example",
        "registry_example",
        "registry.example:",
        "registry.example:1:2",
        f"{'a' * 64}.example",
        "localhost:0",
        "localhost:00",
        "localhost:080",
        "localhost:65536",
        "localhost:notaport",
        "127.00.0.1",
        "256.0.0.1",
        "1.2.3",
        "[::1]",
        "[2001:db8::1]:5000",
        "user:pass@registry.example",
    )
    for registry in invalid:
        reference = f"{registry}/flash/serve@{digest}"
        with pytest.raises(ValueError, match=r"registry|digest-qualified"):
            ServingImage(reference, digest)


def test_deployment_bundle_proves_all_control_manifest_and_image_bindings() -> None:
    spec, inputs = _spec_and_inputs()
    manifest = build_serving_manifest(spec, inputs)
    image = _image(manifest.expected_oci_digest)

    bundle = DeploymentBundle(spec, manifest, image)
    assert bundle.spec.spec_id == manifest.spec_id
    assert bundle.manifest.engine.engine_id == spec.engine.engine_id

    with pytest.raises(ValueError, match="deployment spec"):
        DeploymentBundle(replace(spec, generation=spec.generation + 1), manifest, image)
    other_digest = "sha256:" + "6" * 64
    with pytest.raises(ValueError, match="image binding"):
        DeploymentBundle(spec, manifest, _image(other_digest))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repo_id", "other/source"),
        ("repo_type", "dataset"),
        ("subfolder", "other/subfolder"),
    ],
)
def test_deployment_bundle_rejects_adapter_source_mutations(field: str, value: str) -> None:
    spec, inputs = _spec_and_inputs()
    manifest = build_serving_manifest(spec, inputs)

    def mutate(payload):
        source = payload["adapters"][0]["source"]
        source[field] = value

    changed = _mutated_manifest(manifest, mutate)
    with pytest.raises(ValueError, match="adapter does not match"):
        DeploymentBundle(spec, changed, _image())


def test_deployment_bundle_rejects_source_revision_and_aggregate_mutations() -> None:
    spec, inputs = _spec_and_inputs()
    manifest = build_serving_manifest(spec, inputs)
    new_revision = "9" * 40

    def mutate_revision(payload):
        adapter = payload["adapters"][0]
        adapter["adapter_revision"] = f"run-1@final.{new_revision}"
        adapter["source"]["revision"] = new_revision
        payload["aliases"]["run-1"] = adapter["adapter_revision"]

    changed_revision = _mutated_manifest(manifest, mutate_revision)
    with pytest.raises(ValueError, match="adapters do not exactly match"):
        DeploymentBundle(spec, changed_revision, _image())

    def mutate_aggregate(payload):
        adapter = payload["adapters"][0]
        adapter["files"][0]["sha256"] = "8" * 64
        adapter["aggregate_sha256"] = hashlib.sha256(
            _canonical_payload(adapter["files"])
        ).hexdigest()

    changed_aggregate = _mutated_manifest(manifest, mutate_aggregate)
    with pytest.raises(ValueError, match="adapter does not match"):
        DeploymentBundle(spec, changed_aggregate, _image())


@pytest.mark.parametrize("field", ["thinking_default", "structured_outputs_default"])
def test_deployment_bundle_rejects_adapter_default_mutations(field: str) -> None:
    spec, inputs = _spec_and_inputs()
    manifest = build_serving_manifest(spec, inputs)

    def mutate(payload):
        adapter = payload["adapters"][0]
        adapter[field] = (
            not adapter[field] if field == "thinking_default" else {"json_object": False}
        )

    changed = _mutated_manifest(manifest, mutate)
    with pytest.raises(ValueError, match="adapter does not match"):
        DeploymentBundle(spec, changed, _image())


def test_deployment_bundle_rejects_activation_alias_mutation() -> None:
    spec, inputs = _spec_and_inputs()
    manifest = build_serving_manifest(spec, inputs)
    changed = _mutated_manifest(manifest, lambda payload: payload["aliases"].clear())

    with pytest.raises(ValueError, match="aliases do not match"):
        DeploymentBundle(spec, changed, _image())


def test_resource_names_are_stable_sensitive_collision_resistant_and_bounded() -> None:
    spec, _inputs = _spec_and_inputs()
    first = serving_resource_names(
        spec.deployment_id,
        spec.generation,
        spec.engine.engine_id,
        workload_role="app",
    )
    assert first == serving_resource_names(
        spec.deployment_id,
        spec.generation,
        spec.engine.engine_id,
        workload_role="app",
    )
    changed = (
        serving_resource_names(
            "deployment-other",
            spec.generation,
            spec.engine.engine_id,
            workload_role="app",
        ),
        serving_resource_names(
            spec.deployment_id,
            spec.generation + 1,
            spec.engine.engine_id,
            workload_role="app",
        ),
        serving_resource_names(
            spec.deployment_id,
            spec.generation,
            "f" * 64,
            workload_role="app",
        ),
        serving_resource_names(
            spec.deployment_id,
            spec.generation,
            spec.engine.engine_id,
            workload_role="pod",
        ),
    )
    assert all(candidate != first for candidate in changed)
    names = tuple(first.__getattribute__(name) for name in first.__slots__)
    assert len(names) == len(set(names))
    assert all(len(name) <= 63 for name in names)

    generated = {
        serving_resource_names(
            f"deployment-{index}",
            index + 1,
            f"{index:064x}",
            workload_role="pod",
        ).app_or_pod
        for index in range(512)
    }
    assert len(generated) == 512


def test_runtime_secrets_are_exact_redacted_and_nonserializable() -> None:
    secrets = ServingRuntimeSecrets(SECRET, SECRET + "-artifact")
    assert SECRET not in repr(secrets)
    assert "redacted" in repr(secrets).lower()
    for operation in (
        lambda: copy.copy(secrets),
        lambda: copy.deepcopy(secrets),
        lambda: pickle.dumps(secrets),
        lambda: json.dumps(secrets),
        lambda: secrets.__getstate__(),
        lambda: vars(secrets),
        lambda: type("SecretSubclass", (ServingRuntimeSecrets,), {}),
    ):
        with pytest.raises(TypeError):
            operation()

    with pytest.raises(ValueError, match="artifact token") as exc_info:
        ServingRuntimeSecrets(SECRET, "")
    assert SECRET not in str(exc_info.value)


def test_provider_failures_use_only_fixed_allowlisted_data() -> None:
    spec, _inputs = _spec_and_inputs()
    failure = SanitizedProviderFailure("transport_failed")
    assert failure.code == "transport_failed"
    assert str(failure) == "provider transport failed"
    assert failure.__cause__ is None

    result = failed_deployment_result(spec, "resource_ambiguous", outcome_unknown=True)
    assert result.status == "outcome_unknown"
    assert result.error_code == "resource_ambiguous"
    with pytest.raises(ValueError, match="allowlisted") as exc_info:
        SanitizedProviderFailure(SECRET)
    assert SECRET not in str(exc_info.value)


def test_base64url_identity_and_launcher_abi_fit_provider_tags() -> None:
    encoded = base64url_identity(bytes(range(32)))
    assert len(encoded) == 43
    assert "=" not in encoded
    assert base64.urlsafe_b64decode(encoded + "=") == bytes(range(32))
    assert len(LAUNCHER_ABI_ID) <= 63
    assert LAUNCHER_ABI_ID.startswith("fsla1-")
    with pytest.raises(ValueError, match="32 bytes"):
        base64url_identity(b"short")


def test_manifest_codec_round_trip_is_canonical_and_bounded() -> None:
    manifest = _manifest()
    encoded = encode_manifest_environment(manifest)
    assert len(encoded.encode("ascii")) <= MAX_ENCODED_MANIFEST_BYTES
    assert len(manifest.canonical_json().encode("utf-8")) <= MAX_CANONICAL_MANIFEST_BYTES
    assert decode_manifest_environment(encoded) == manifest


def test_manifest_codec_rejects_canonical_and_encoded_limits_independently(
    monkeypatch,
) -> None:
    manifest = _manifest()
    canonical_size = len(manifest.canonical_json().encode("utf-8"))
    monkeypatch.setattr(_common, "MAX_CANONICAL_MANIFEST_BYTES", canonical_size - 1)
    with pytest.raises(ValueError, match="canonical"):
        encode_manifest_environment(manifest)

    monkeypatch.setattr(_common, "MAX_CANONICAL_MANIFEST_BYTES", canonical_size)
    encoded = encode_manifest_environment(manifest)
    monkeypatch.setattr(_common, "MAX_ENCODED_MANIFEST_BYTES", len(encoded) - 1)
    with pytest.raises(ValueError, match="encoded"):
        encode_manifest_environment(manifest)
    with pytest.raises(ValueError, match="encoded"):
        decode_manifest_environment(encoded)


def test_manifest_codec_bounds_decompression_without_compression_ratio_trust() -> None:
    oversized = b"x" * (MAX_CANONICAL_MANIFEST_BYTES + 1)
    encoded = base64.b64encode(zlib.compress(oversized, level=9)).decode("ascii")
    assert len(encoded) < MAX_ENCODED_MANIFEST_BYTES
    with pytest.raises(ValueError, match="canonical"):
        decode_manifest_environment(encoded)


def test_manifest_codec_rejects_trailing_malformed_noncanonical_and_wrong_identity() -> None:
    manifest = _manifest()
    canonical = manifest.canonical_json().encode("utf-8")
    compressed = zlib.compress(canonical, level=9)
    cases = (
        "!!!!",
        base64.b64encode(compressed).decode("ascii") + "\n",
        base64.b64encode(compressed + zlib.compress(b"extra")).decode("ascii"),
        base64.b64encode(compressed[:-1]).decode("ascii"),
        base64.b64encode(zlib.compress(b"\xff")).decode("ascii"),
    )
    for encoded in cases:
        with pytest.raises(ValueError, match=r"."):
            decode_manifest_environment(encoded)

    payload = json.loads(canonical)
    payload["manifest_id"] = "0" * 64
    wrong_identity = base64.b64encode(
        zlib.compress(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    ).decode("ascii")
    with pytest.raises(ValueError, match="invalid"):
        decode_manifest_environment(wrong_identity)

    noncanonical = base64.b64encode(
        zlib.compress(json.dumps(json.loads(canonical)).encode())
    ).decode("ascii")
    with pytest.raises(ValueError, match="not canonical"):
        decode_manifest_environment(noncanonical)


def test_the_launcher_abi_tracks_the_wrapper_source_that_actually_runs(tmp_path) -> None:
    """A changed wrapper must change the deployment identity.

    `deploy_app` copies the CLI's local `_modal_wrapper.py` over the copy inside the pinned image,
    so the executed code is NOT covered by `bundle.image.digest`. While the ABI was a hash of a
    fixed literal, a CLI and an image from different releases produced byte-identical provenance:
    the manifest, engine id, provider handle, and readiness proof all reported only the registry
    digest, so substituted wrapper code ran under an immutability claim it did not satisfy.

    Recomputed from bytes rather than asserting a pinned constant, so this keeps testing the
    binding itself instead of freezing today's digest.
    """
    import base64
    import hashlib
    from pathlib import Path

    from flash.serve.provisioning import _common

    wrapper = Path(_common.__file__).with_name("_modal_wrapper.py")
    expected = "fsla1-" + base64.urlsafe_b64encode(
        hashlib.sha256(b"flash.serve.app.launch:v1\0" + wrapper.read_bytes()).digest()
    ).decode("ascii").rstrip("=")
    assert expected == _common.LAUNCHER_ABI_ID

    # and a different wrapper yields a different id -- the property that makes it provenance.
    altered = hashlib.sha256(b"flash.serve.app.launch:v1\0" + wrapper.read_bytes() + b"# x\n")
    altered_id = "fsla1-" + base64.urlsafe_b64encode(altered.digest()).decode("ascii").rstrip("=")
    assert altered_id != _common.LAUNCHER_ABI_ID
