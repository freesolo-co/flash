"""prevalidated immutable modal deployment plan without secret values."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from flash.serve.control import ModalPlacement
from flash.serve.provisioning.common.records import (
    DeploymentBundle,
    ServingResourceNames,
    base64url_identity,
    encode_manifest_environment,
    reject_unreachable_registry,
    serving_resource_names,
)

MODAL_VOLUME_MOUNT = "/modal-volume"
MODAL_CACHE_ROOT = "/modal-volume/flash-serving"
MODAL_WEB_PORT = 8000
# a live cold boot projected about 21 minutes before readiness. 1200s is still below that, while
# 1800s adds roughly eight minutes of margin. this is modal's own container-boot limit, so it must
# stay strictly under the cli deadline that also has to cover finalize and cleanup after readiness.
MODAL_STARTUP_TIMEOUT_SECONDS = 1800
MODAL_SCALEDOWN_WINDOW_SECONDS = 60
MODAL_MAX_CONTAINERS = 1
MODAL_MIN_CONTAINERS = 0
MODAL_BUFFER_CONTAINERS = 0

ModalDeploymentPhase = Literal["bootstrap", "finalized"]
_MODAL_PHASES = frozenset({"bootstrap", "finalized"})

_SUBDOMAIN_RE = re.compile(r"(?![0-9]+$)(?!-)[a-z0-9-]{1,63}(?<!-)")
_ENVIRONMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{1,63}")
_REGION_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_GPU_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,31}!?")
_TAG_RE = re.compile(r"[A-Za-z0-9._-]{1,63}")


def _identity(value: str) -> str:
    return base64url_identity(bytes.fromhex(value))


def _hashed_identity(value: str) -> str:
    return base64url_identity(hashlib.sha256(value.encode("utf-8")).digest())


def _canonical_hash(value: dict[str, object]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64url_identity(hashlib.sha256(encoded).digest())


def _validate_placement(placement: ModalPlacement) -> None:
    if _SUBDOMAIN_RE.fullmatch(placement.workspace_name) is None:
        raise ValueError("modal workspace_name must be an exact lowercase subdomain label")
    # the suffix is concatenated into the same hostname label as the workspace
    # (`<workspace>-<suffix>--<label>.modal.run`), so it needs the workspace's charset, not merely
    # "nonempty". `dev_test` builds a syntactically valid-looking expected url that no modal
    # hostname can ever equal, and nothing downstream rejects it: the mismatch is only discovered
    # against modal's real url, after the secrets, volume, and app already exist and bill.
    if placement.web_suffix is not None and _SUBDOMAIN_RE.fullmatch(placement.web_suffix) is None:
        raise ValueError("modal web_suffix must be an exact lowercase subdomain label")
    if _ENVIRONMENT_RE.fullmatch(
        placement.environment
    ) is None or placement.environment.lower().startswith("en-"):
        raise ValueError("modal environment is not a canonical environment name")
    if placement.region is None or _REGION_RE.fullmatch(placement.region) is None:
        raise ValueError("modal region must be an exact lowercase region")
    if _GPU_RE.fullmatch(placement.gpu) is None:
        raise ValueError("modal gpu must be a canonical gpu type")


# modal rejects a deployment tag over 50 characters, and `fsm1-<phase>-` already spends 15 of
# them against a 43-character topology hash, so the untruncated tag was 58 and every deploy failed
# at app.deploy() -- after the secret and volume had been created, which is why the failure
# surfaced as outcome_unknown with orphaned resources rather than a clean rejection.
MODAL_DEPLOYMENT_TAG_LIMIT = 50

# separate limit, separate api: the one above bounds the length of the single `tag=` string on
# app.deploy(), while this bounds how many key/value pairs the app itself may carry. modal enforces
# it server-side ("Apps must have no more than 8 tags") and the client does not check it, so an
# over-budget tag set also fails inside app.deploy() with resources already created.
MODAL_APP_TAG_LIMIT = 8


def _deployment_tag(phase: str, topology: str) -> str:
    """build a modal deployment tag that fits modal's 50-character limit.

    the topology hash is truncated rather than dropped: it is what distinguishes one deployment's
    tag from another, and 35 base64url characters still carry ~208 bits. truncating keeps the tag
    deterministic for the same bundle, which the finalize/reconcile path relies on.
    """

    prefix = f"fsm1-{phase}-"
    tag = prefix + topology[: MODAL_DEPLOYMENT_TAG_LIMIT - len(prefix)]
    if len(tag) > MODAL_DEPLOYMENT_TAG_LIMIT or not tag.startswith(prefix):
        raise ValueError("modal deployment tag does not fit modal's tag limit")
    return tag


def _url_source(placement: ModalPlacement) -> str:
    """the `<source>` in `https://<source>--<label>.modal.run`.

    Modal disambiguates environments with a per-environment web suffix, so the source is
    `<workspace>-<suffix>` unless the environment is the one allowed to have none. Deriving it from
    the workspace alone made every suffixed environment's expected url wrong, which surfaces as an
    identity mismatch in `_modal_resources` rather than as a bad url: the app deploys, and then the
    deployment is rejected as not ours.
    """

    if placement.web_suffix is None:
        return placement.workspace_name
    return f"{placement.workspace_name}-{placement.web_suffix}"


def _endpoint_label(bundle: DeploymentBundle, placement: ModalPlacement) -> str:
    max_length = 63 - len(_url_source(placement)) - 2
    prefix = "fsw-"
    if max_length < len(prefix) + 16:
        raise ValueError("modal workspace_name is too long for a deterministic endpoint label")
    payload = b"\0".join(
        (
            bundle.spec.deployment_id.encode("utf-8"),
            str(bundle.spec.generation).encode("ascii"),
            bundle.spec.engine.engine_id.encode("ascii"),
        )
    )
    suffix = base64.b32encode(hashlib.sha256(payload).digest()).decode("ascii").lower().rstrip("=")
    return prefix + suffix[: max_length - len(prefix)]


def _topology_identity(
    bundle: DeploymentBundle,
    placement: ModalPlacement,
    names: ServingResourceNames,
    endpoint_label: str,
) -> str:
    return _canonical_hash(
        {
            "app": names.app_or_pod,
            "buffer_containers": MODAL_BUFFER_CONTAINERS,
            "endpoint_label": endpoint_label,
            "environment": placement.environment,
            "gpu": placement.gpu,
            "gpu_count": placement.gpu_count,
            "image": bundle.image.reference,
            "include_source": False,
            "max_containers": MODAL_MAX_CONTAINERS,
            "min_containers": MODAL_MIN_CONTAINERS,
            "region": placement.region,
            "scaledown_window": MODAL_SCALEDOWN_WINDOW_SECONDS,
            "startup_timeout": MODAL_STARTUP_TIMEOUT_SECONDS,
            "volume": names.volume,
            "web_port": MODAL_WEB_PORT,
        }
    )


def _tags(
    bundle: DeploymentBundle,
    placement: ModalPlacement,
    names: ServingResourceNames,
    endpoint_label: str,
    phase: ModalDeploymentPhase,
) -> tuple[tuple[str, str], ...]:
    # modal caps an app at MODAL_APP_TAG_LIMIT tags server-side, and this dict is the whole app
    # tag set. adding a ninth key here would exceed the cap, and because the failure lands at
    # app.deploy() -- after the secret and volume exist -- it surfaces as outcome_unknown with
    # orphaned billable resources rather than a clean rejection.
    values = {
        "flash-deployment": _hashed_identity(bundle.spec.deployment_id),
        "flash-engine": _identity(bundle.spec.engine.engine_id),
        "flash-generation": str(bundle.spec.generation),
        "flash-image": _identity(bundle.image.digest.removeprefix("sha256:")),
        "flash-manifest": _identity(bundle.manifest.manifest_id),
        "flash-phase": phase,
        "flash-spec": _identity(bundle.spec.spec_id),
        "flash-topology": _topology_identity(bundle, placement, names, endpoint_label),
    }
    if len(values) > MODAL_APP_TAG_LIMIT:
        raise ValueError("modal app tag set exceeds modal's tag limit")
    tags = tuple(sorted(values.items()))
    if any(
        _TAG_RE.fullmatch(key) is None or _TAG_RE.fullmatch(value) is None for key, value in tags
    ):
        raise ValueError("modal provenance tag is not canonical")
    return tags


def _environment(bundle: DeploymentBundle, encoded_manifest: str) -> tuple[tuple[str, str], ...]:
    values = {
        "FLASH_SERVING_CACHE_ROOT": MODAL_CACHE_ROOT,
        "FLASH_SERVING_HOST": "0.0.0.0",
        "FLASH_SERVING_IMAGE_DIGEST": bundle.image.digest,
        "FLASH_SERVING_MANIFEST": encoded_manifest,
        "FLASH_SERVING_MANIFEST_ID": bundle.manifest.manifest_id,
        "FLASH_SERVING_PORT": str(MODAL_WEB_PORT),
    }
    return tuple(sorted(values.items()))


@dataclass(frozen=True, slots=True)
class ModalCreatePlan:
    """one immutable modal resource plan with no plaintext secret values."""

    bundle: DeploymentBundle
    placement: ModalPlacement
    phase: ModalDeploymentPhase
    names: ServingResourceNames
    encoded_manifest: str
    environment: tuple[tuple[str, str], ...]
    tags: tuple[tuple[str, str], ...]
    deployment_tag: str
    gpu_request: str
    function_name: str
    endpoint_label: str
    expected_public_url: str
    include_source: bool = False
    web_port: int = MODAL_WEB_PORT
    startup_timeout_seconds: int = MODAL_STARTUP_TIMEOUT_SECONDS
    scaledown_window_seconds: int = MODAL_SCALEDOWN_WINDOW_SECONDS
    min_containers: int = MODAL_MIN_CONTAINERS
    max_containers: int = MODAL_MAX_CONTAINERS
    buffer_containers: int = MODAL_BUFFER_CONTAINERS


def validate_modal_plan(plan: ModalCreatePlan) -> None:
    if type(plan) is not ModalCreatePlan:
        raise ValueError("modal plan must use the exact plan type")
    if plan.phase not in _MODAL_PHASES or dict(plan.tags).get("flash-phase") != plan.phase:
        raise ValueError("modal plan phase does not match its exact provenance tags")
    if plan.include_source is not False:
        raise ValueError("modal source inclusion must remain disabled")
    if (
        plan.web_port != MODAL_WEB_PORT
        or plan.startup_timeout_seconds != MODAL_STARTUP_TIMEOUT_SECONDS
        or plan.scaledown_window_seconds != MODAL_SCALEDOWN_WINDOW_SECONDS
        or plan.min_containers != 0
        or plan.max_containers != 1
        or plan.buffer_containers != 0
    ):
        raise ValueError("modal topology does not match the fixed serving contract")
    combined_label = f"{_url_source(plan.placement)}--{plan.endpoint_label}"
    if (
        _SUBDOMAIN_RE.fullmatch(plan.endpoint_label) is None
        or len(combined_label) > 63
        or plan.expected_public_url != f"https://{combined_label}.modal.run"
    ):
        raise ValueError("modal endpoint does not match the deterministic label")


def build_modal_create_plan(
    bundle: DeploymentBundle,
    *,
    phase: ModalDeploymentPhase = "finalized",
) -> ModalCreatePlan:
    """validate every modal input before provider client construction or secret reveal."""

    if phase not in _MODAL_PHASES:
        raise ValueError("modal deployment phase must be bootstrap or finalized")
    if type(bundle) is not DeploymentBundle:
        raise ValueError("bundle must be an exact DeploymentBundle")
    bundle.__post_init__()
    if bundle.spec.provider != "modal" or type(bundle.spec.placement) is not ModalPlacement:
        raise ValueError("modal provisioning requires a modal deployment bundle")
    reject_unreachable_registry(bundle.image)
    placement = bundle.spec.placement
    _validate_placement(placement)
    names = serving_resource_names(
        bundle.spec.deployment_id,
        bundle.spec.generation,
        bundle.spec.engine.engine_id,
        workload_role="app",
    )
    encoded_manifest = encode_manifest_environment(bundle.manifest)
    endpoint_label = _endpoint_label(bundle, placement)
    tags = _tags(bundle, placement, names, endpoint_label, phase)
    topology = dict(tags)["flash-topology"]
    plan = ModalCreatePlan(
        bundle=bundle,
        placement=placement,
        phase=phase,
        names=names,
        encoded_manifest=encoded_manifest,
        environment=_environment(bundle, encoded_manifest),
        tags=tags,
        deployment_tag=_deployment_tag(phase, topology),
        gpu_request=f"{placement.gpu}:{placement.gpu_count}",
        function_name=names.template,
        endpoint_label=endpoint_label,
        expected_public_url=f"https://{_url_source(placement)}--{endpoint_label}.modal.run",
    )
    validate_modal_plan(plan)
    return plan
