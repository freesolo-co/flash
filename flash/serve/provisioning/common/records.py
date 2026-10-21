"""provider-neutral serving provisioning records and codecs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import json
import math
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from flash.serve.app.manifest import ManifestError, ServingManifest, load_serving_manifest
from flash.serve.control import (
    DeploymentErrorCode,
    DeploymentErrorReason,
    DeploymentResult,
    DeploymentSpec,
    ProviderHandle,
)
from flash.serve.control.types import validate_deployment_spec

MAX_CANONICAL_MANIFEST_BYTES = 48 * 1024
MAX_ENCODED_MANIFEST_BYTES = 64 * 1024

Clock = Callable[[], float]
Sleeper = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class LifecycleFailure:
    """one provider-neutral lifecycle failure.

    Defined in the shared records module so lifecycle results and callers use one exact type.
    """

    code: DeploymentErrorCode
    outcome_unknown: bool = False
    reason: DeploymentErrorReason | None = None


def validate_deadline(deadline_at: float, clock: Clock) -> None:
    """reject a deadline that cannot describe a future instant.

    Neither the rule nor the clock is provider-specific.
    """

    if type(deadline_at) not in {int, float} or not math.isfinite(float(deadline_at)):
        raise ValueError("deadline_at must be finite")
    if float(deadline_at) <= clock():
        raise ValueError("deadline_at must be in the future")


_IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_DNS_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_REPOSITORY_COMPONENT_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_TAG_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
# rfc 7235 token68, the grammar a `Bearer <credential>` value must match.
_BEARER_TOKEN_RE = re.compile(r"[A-Za-z0-9\-._~+/]+=*")
_ENGINE_ID_RE = re.compile(r"[0-9a-f]{64}")
_PROVIDER_NAME_RE = re.compile(r"[a-z][a-z0-9-]*[a-z0-9]")
_ERROR_CODES = frozenset(
    {
        "authentication_failed",
        "conflict",
        "invalid_request",
        "provider_rejected",
        "readiness_failed",
        "resource_ambiguous",
        "transport_failed",
    }
)
_ERROR_MESSAGES = {
    "authentication_failed": "provider authentication failed",
    "conflict": "provider resource conflict",
    "invalid_request": "provider request is invalid",
    "provider_rejected": "provider rejected the operation",
    "readiness_failed": "provider resource did not become ready",
    "resource_ambiguous": "provider resource outcome is ambiguous",
    "transport_failed": "provider transport failed",
}


def _nonempty(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty unpadded string")
    return value


def _bearer_token(value: object, name: str) -> str:
    """a credential that can actually be sent as, and matched from, an http bearer header.

    `_nonempty` only strips the ends, so an interior space passed this boundary and provisioning
    created billable resources -- but the serving app parses `Authorization` by splitting on
    spaces and rejects any candidate containing whitespace, so no client could ever authenticate
    against the endpoint that was just paid for. non-ascii fails earlier still, while the header
    is being encoded. rfc 7235 defines the token68 grammar this checks; reject here, where nothing
    has been created yet.
    """

    text = _nonempty(value, name)
    if _BEARER_TOKEN_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a usable http bearer credential")
    return text


def _exact_digest(value: object, name: str) -> str:
    text = _nonempty(value, name)
    if _IMAGE_DIGEST_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be an exact lowercase sha256 digest")
    return text


def _validate_registry(value: str) -> None:
    if "[" in value or "]" in value or value.count(":") > 1:
        raise ValueError("image reference registry is not canonical")
    if ":" in value:
        host, port_text = value.rsplit(":", 1)
        if (
            not port_text.isascii()
            or not port_text.isdecimal()
            or len(port_text) > 5
            or port_text.startswith("0")
            or not 1 <= int(port_text) <= 65535
        ):
            raise ValueError("image reference registry port is not canonical")
    else:
        host = value
    if not host or host != host.lower() or len(host) > 253:
        raise ValueError("image reference registry host is not canonical")
    if host == "localhost":
        return
    if all(character.isdigit() or character == "." for character in host):
        try:
            address = ipaddress.IPv4Address(host)
        except ipaddress.AddressValueError as exc:
            raise ValueError("image reference registry ipv4 address is invalid") from exc
        if str(address) != host:
            raise ValueError("image reference registry ipv4 address is not canonical")
        return
    labels = host.split(".")
    if any(_DNS_LABEL_RE.fullmatch(label) is None for label in labels):
        raise ValueError("image reference registry dns name is not canonical")


_LOOPBACK_REGISTRY_HOSTS = frozenset({"localhost", "localhost.localdomain"})


def reject_unreachable_registry(image: ServingImage) -> None:
    """reject a registry only the operator's machine can reach.

    `_validate_registry` accepts loopback and private hosts because a local registry is a
    legitimate target for a locally executed pull. modal resolves the reference inside its build
    infrastructure instead, and the path neither uploads the image nor opens a tunnel back, so such
    a reference fails only after provider resources exist and bill. reject it
    while building the plan, before any provider call.
    """

    registry = image.reference.rsplit("@", 1)[0].split("/", 1)[0]
    host = registry.rsplit(":", 1)[0] if registry.count(":") == 1 else registry
    if (
        host in _LOOPBACK_REGISTRY_HOSTS
        or host == "local"
        or host.endswith((".localhost", ".local"))
    ):
        # rfc 6761 reserves `.localhost` for loopback, while rfc 6762 reserves `.local` for
        # link-local mdns. neither namespace is reachable from remote provider infrastructure.
        raise ValueError("remote provider image registry cannot be a loopback or link-local host")
    try:
        address = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError:
        return
    if address.is_loopback or address.is_private or address.is_link_local:
        raise ValueError("remote provider image registry cannot be a loopback or private address")


def base64url_identity(identity: bytes) -> str:
    """encode one exact 32-byte identity without base64 padding."""

    if type(identity) is not bytes or len(identity) != 32:
        raise ValueError("identity must be exactly 32 bytes")
    return base64.urlsafe_b64encode(identity).decode("ascii").rstrip("=")


@dataclass(frozen=True, slots=True)
class ServingImage:
    """one canonical digest-qualified oci image reference."""

    reference: str
    digest: str

    def __post_init__(self) -> None:
        if type(self) is not ServingImage:
            raise ValueError("image must be an exact ServingImage")
        reference = _nonempty(self.reference, "image reference")
        digest = _exact_digest(self.digest, "image digest")
        if any(character.isspace() for character in reference):
            raise ValueError("image reference cannot contain whitespace")
        if "://" in reference or "?" in reference or "#" in reference:
            raise ValueError("image reference must be a canonical oci reference")
        if reference.count("@") != 1:
            raise ValueError("image reference must be digest-qualified")
        named, bound_digest = reference.rsplit("@", 1)
        if bound_digest != digest:
            raise ValueError("image reference digest does not match image digest")
        if "/" not in named:
            raise ValueError("image reference must include a registry and repository")
        registry, repository = named.split("/", 1)
        _validate_registry(registry)
        last_slash = repository.rfind("/")
        last_colon = repository.rfind(":")
        if last_colon > last_slash:
            repository_name, tag = repository[:last_colon], repository[last_colon + 1 :]
            if _TAG_RE.fullmatch(tag) is None:
                raise ValueError("image reference tag is not canonical")
        else:
            repository_name = repository
        components = repository_name.split("/")
        if not components or any(
            _REPOSITORY_COMPONENT_RE.fullmatch(component) is None for component in components
        ):
            raise ValueError("image reference repository is not canonical")


def _validate_bundle_adapters(spec: DeploymentSpec, manifest: ServingManifest) -> None:
    manifest_by_checkpoint = {adapter.checkpoint_id: adapter for adapter in manifest.adapters}
    spec_by_checkpoint = {adapter.checkpoint_id: adapter for adapter in spec.adapters}
    if set(manifest_by_checkpoint) != set(spec_by_checkpoint):
        raise ValueError("bundle manifest adapters do not exactly match the deployment spec")
    for revision, planned in spec_by_checkpoint.items():
        actual = manifest_by_checkpoint[revision]
        structured_default = (
            None
            if planned.structured_outputs_default_json is None
            else json.loads(planned.structured_outputs_default_json)
        )
        if (
            actual.run_id != planned.run_id
            or actual.checkpoint_id != planned.checkpoint_id
            or actual.repo_id != planned.artifact_repo_id
            or actual.repo_type != planned.artifact_repo_type
            or actual.source_revision != planned.artifact_revision
            or actual.source_subfolder != planned.artifact_subfolder
            or actual.aggregate_sha256 != planned.artifact_digest
            or actual.base_model != planned.base_model
            or actual.base_model_revision != planned.base_model_revision
            or actual.lora_rank != planned.lora_rank
            or actual.thinking_default != planned.thinking_default
            or (
                None
                if actual.structured_outputs_default is None
                else dict(actual.structured_outputs_default)
            )
            != structured_default
        ):
            raise ValueError("bundle manifest adapter does not match the deployment spec")


@dataclass(frozen=True, slots=True)
class DeploymentBundle:
    """one exact control spec, execution manifest, and bound image."""

    spec: DeploymentSpec
    manifest: ServingManifest
    image: ServingImage

    def __post_init__(self) -> None:
        if type(self) is not DeploymentBundle:
            raise ValueError("bundle must be an exact DeploymentBundle")
        validate_deployment_spec(self.spec)
        if type(self.manifest) is not ServingManifest:
            raise ValueError("bundle manifest must be an exact ServingManifest")
        self.manifest.__post_init__()
        if type(self.image) is not ServingImage:
            raise ValueError("bundle image must be an exact ServingImage")
        self.image.__post_init__()
        if (
            self.manifest.deployment_id != self.spec.deployment_id
            or self.manifest.generation != self.spec.generation
            or self.manifest.spec_id != self.spec.spec_id
            or self.manifest.engine.engine_id != self.spec.engine.engine_id
        ):
            raise ValueError("bundle manifest does not match the exact deployment spec")
        _validate_bundle_adapters(self.spec, self.manifest)
        if (
            self.manifest.expected_oci_digest != self.image.digest
            or self.spec.engine.image_digest != self.image.digest
            or not self.image.reference.endswith("@" + self.image.digest)
        ):
            raise ValueError("bundle image binding does not match the manifest and spec")


@dataclass(frozen=True, slots=True)
class ServingResourceNames:
    """deterministic provider-safe names for one deployment generation."""

    app_or_pod: str
    volume: str
    template: str
    inference_secret: str
    artifact_secret: str

    def __post_init__(self) -> None:
        if type(self) is not ServingResourceNames:
            raise ValueError("resource names must use the exact record type")
        for name in (
            self.app_or_pod,
            self.volume,
            self.template,
            self.inference_secret,
            self.artifact_secret,
        ):
            if len(name) > 63 or _PROVIDER_NAME_RE.fullmatch(name) is None:
                raise ValueError("resource name is not provider-safe")


def _resource_name(payload: bytes, role: str, max_length: int) -> str:
    prefix = f"flash-{role}-"
    available = max_length - len(prefix)
    if available < 16:
        raise ValueError("resource name limit is too small for a safe identity")
    digest = base64.b32encode(hashlib.sha256(payload + b"\0" + role.encode()).digest())
    suffix = digest.decode("ascii").lower().rstrip("=")[: min(32, available)]
    return prefix + suffix


def serving_resource_names(
    deployment_id: str,
    generation: int,
    engine_id: str,
    *,
    workload_role: Literal["app", "pod"],
    max_length: int = 63,
) -> ServingResourceNames:
    """derive role-separated names from the complete deployment identity."""

    deployment = _nonempty(deployment_id, "deployment_id")
    if type(generation) is not int or generation <= 0:
        raise ValueError("generation must be a positive integer")
    if type(engine_id) is not str or _ENGINE_ID_RE.fullmatch(engine_id) is None:
        raise ValueError("engine_id must be an exact lowercase sha256 identity")
    if workload_role not in {"app", "pod"}:
        raise ValueError("workload_role must be app or pod")
    if type(max_length) is not int or max_length > 63:
        raise ValueError("resource name limit must be an integer no greater than 63")
    payload = b"\0".join(
        (deployment.encode("utf-8"), str(generation).encode("ascii"), engine_id.encode("ascii"))
    )
    return ServingResourceNames(
        app_or_pod=_resource_name(payload, workload_role, max_length),
        volume=_resource_name(payload, "volume", max_length),
        template=_resource_name(payload, "template", max_length),
        inference_secret=_resource_name(payload, "inference-secret", max_length),
        artifact_secret=_resource_name(payload, "artifact-secret", max_length),
    )


class ServingRuntimeSecrets:
    """redacted request boundary for runtime secret values.

    `artifact_token` is optional because teardown, resize, and read-only reconcile never hydrate and
    so have no artifact to fetch. A *fresh create* is the opposite: the volume starts empty, the
    container hydrates before the engine starts, and that path is token-only end to end, so
    provisioning without one would create billable resources for a container that cannot reach
    readiness. provisioning rejects that combination after observing the provider and before the
    first create mutation, so an existing generation can still be adopted without the token.
    """

    __slots__ = ("__artifact_token", "__inference_token")

    def __init_subclass__(cls, **_kwargs: object) -> None:
        raise TypeError("serving runtime secrets cannot be subclassed")

    def __init__(self, inference_token: str, artifact_token: str | None = None) -> None:
        self.__inference_token = _bearer_token(inference_token, "inference token")
        if artifact_token is not None:
            artifact_token = _nonempty(artifact_token, "artifact token")
        self.__artifact_token = artifact_token

    def __repr__(self) -> str:
        return "ServingRuntimeSecrets(<redacted>)"

    def __copy__(self) -> object:
        raise TypeError("serving runtime secrets cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> object:
        raise TypeError("serving runtime secrets cannot be copied")

    def __getstate__(self) -> object:
        raise TypeError("serving runtime secrets cannot expose serialization state")

    def __reduce__(self) -> object:
        raise TypeError("serving runtime secrets cannot be serialized")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("serving runtime secrets cannot be serialized")

    def _reveal_for_launch(self) -> tuple[str, str | None]:
        if type(self) is not ServingRuntimeSecrets:
            raise TypeError("runtime secrets must use the exact secret boundary")
        return self.__inference_token, self.__artifact_token


class InterruptedProvisioning(KeyboardInterrupt):
    """one interrupt whose provider cleanup could not be confirmed.

    subclasses `KeyboardInterrupt` because it stands in for the interrupt the user actually
    caused. that keeps both existing handlers correct without touching either: the `except
    Exception` in `flash/cli/__init__.py` still lets it through, and the `except KeyboardInterrupt`
    beside it still exits 130 with "aborted". a bare `BaseException` subclass would escape both and
    dump a traceback instead.

    `_abort_created_resources` is deliberately best-effort -- it suppresses each provider error so
    one failed delete cannot stop the rest, and so a provider exception never replaces the
    interrupt the user actually caused. that leaves nobody holding the knowledge that a stop or
    delete failed, and the generic cli handler prints only "aborted", which reads as "nothing was
    created". the gpu may still be live and billing. carrying the ambiguity out lets the cli say so.
    """

    __slots__ = ("provider",)

    def __init__(self, provider: str) -> None:
        if provider != "modal":
            raise ValueError("interrupted provisioning requires modal")
        self.provider = provider
        super().__init__(
            f"interrupted before the {provider} deployment was ready, and its cleanup could not "
            f"be confirmed"
        )


class FreshDeploymentArtifactTokenRequired(ValueError):
    """definite pre-mutation rejection for a fresh deployment without hydration access."""

    __slots__ = ("code", "outcome_unknown")

    def __init__(self) -> None:
        self.code = "invalid_request"
        self.outcome_unknown = False
        super().__init__(
            "a new deployment hydrates its serving cache from the hub before the engine starts, "
            "and that hydration requires a token even when the repositories are public"
        )


class SanitizedProviderFailure(RuntimeError):
    """one fixed provider failure without retained response or cause data."""

    __slots__ = ("code",)

    def __init__(self, code: DeploymentErrorCode) -> None:
        if type(code) is not str or code not in _ERROR_CODES:
            raise ValueError("provider failure code must be allowlisted")
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


def failed_deployment_result(
    spec: DeploymentSpec,
    code: DeploymentErrorCode,
    *,
    outcome_unknown: bool = False,
    handle: ProviderHandle | None = None,
    error_reason: DeploymentErrorReason | None = None,
) -> DeploymentResult:
    """build one sanitized failed or outcome-unknown result."""

    failure = SanitizedProviderFailure(code)
    status = "outcome_unknown" if outcome_unknown else "failed"
    return DeploymentResult.from_spec(
        spec,
        status=status,
        handle=handle,
        error_code=failure.code,
        error_reason=error_reason,
    )


def encode_manifest_environment(manifest: ServingManifest) -> str:
    """encode one canonical manifest for a bounded environment value."""

    if type(manifest) is not ServingManifest:
        raise ValueError("manifest must be an exact ServingManifest")
    canonical = manifest.canonical_json().encode("utf-8")
    if len(canonical) > MAX_CANONICAL_MANIFEST_BYTES:
        raise ValueError("canonical serving manifest exceeds its byte limit")
    compressed = zlib.compress(canonical, level=9)
    encoded = base64.b64encode(compressed)
    if len(encoded) > MAX_ENCODED_MANIFEST_BYTES:
        raise ValueError("encoded serving manifest exceeds its environment limit")
    return encoded.decode("ascii")


def _strict_base64(value: object) -> bytes:
    text = _nonempty(value, "encoded serving manifest")
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("encoded serving manifest must be canonical base64") from exc
    if len(encoded) > MAX_ENCODED_MANIFEST_BYTES:
        raise ValueError("encoded serving manifest exceeds its environment limit")
    try:
        compressed = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("encoded serving manifest must be canonical base64") from exc
    if base64.b64encode(compressed) != encoded:
        raise ValueError("encoded serving manifest must be canonical base64")
    return compressed


def _bounded_decompress(compressed: bytes) -> bytes:
    decompressor = zlib.decompressobj()
    output = bytearray()
    pending = compressed
    try:
        while pending:
            remaining = MAX_CANONICAL_MANIFEST_BYTES - len(output)
            chunk = decompressor.decompress(pending, remaining + 1)
            output.extend(chunk)
            if len(output) > MAX_CANONICAL_MANIFEST_BYTES:
                raise ValueError("canonical serving manifest exceeds its byte limit")
            pending = decompressor.unconsumed_tail
            if not pending:
                break
        remaining = MAX_CANONICAL_MANIFEST_BYTES - len(output)
        output.extend(decompressor.flush(remaining + 1))
    except zlib.error as exc:
        raise ValueError("encoded serving manifest is not valid zlib data") from exc
    if len(output) > MAX_CANONICAL_MANIFEST_BYTES:
        raise ValueError("canonical serving manifest exceeds its byte limit")
    if not decompressor.eof:
        raise ValueError("encoded serving manifest is truncated")
    if decompressor.unused_data or decompressor.unconsumed_tail:
        raise ValueError("encoded serving manifest contains trailing compressed data")
    return bytes(output)


def decode_manifest_environment(value: str) -> ServingManifest:
    """decode, bound, and fully revalidate one environment manifest."""

    canonical = _bounded_decompress(_strict_base64(value))
    try:
        text = canonical.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("serving manifest is not valid utf-8") from exc
    try:
        manifest = load_serving_manifest(text)
    except ManifestError as exc:
        raise ValueError("serving manifest is invalid") from exc
    if manifest.canonical_json().encode("utf-8") != canonical:
        raise ValueError("serving manifest bytes are not canonical")
    return manifest
