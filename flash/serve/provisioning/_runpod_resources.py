"""pure runpod resource identity matching and operation-specific state policy."""

from __future__ import annotations

from typing import Literal

from flash.serve.control import RunPodProviderHandle

from ._runpod_plan import RunPodCreatePlan
from ._runpod_protocol import (
    LAUNCH_COMMAND,
    NETWORK_VOLUME_MOUNT,
    PROXY_PORT_SPEC,
    RunPodObservation,
    RunPodPodObservation,
    RunPodSecretObservation,
    RunPodTemplateObservation,
    RunPodVolumeObservation,
)

_RUNNING_STATUS = "RUNNING"
_PENDING_STATUSES = frozenset({"CREATED", "PENDING", "STARTING", "RESTARTING"})
_FAILED_STATUSES = frozenset({"DEAD", "EXITED", "FAILED", "STOPPED", "TERMINATED"})

ReadinessState = Literal["running", "pending", "failed", "invalid"]


class RunPodResourceConflict(RuntimeError):
    pass


def readiness_state(status: str) -> ReadinessState:
    """classify a parseable pod status only for readiness operations."""

    if status == _RUNNING_STATUS:
        return "running"
    if status in _PENDING_STATUSES:
        return "pending"
    if status in _FAILED_STATUSES:
        return "failed"
    return "invalid"


def _one(values: tuple[object, ...], name: str):
    if len(values) != 1:
        raise RunPodResourceConflict(f"{name} is not unique")
    return values[0]


def ensure_unique_resources(observation: RunPodObservation) -> None:
    groups = (
        observation.inference_secrets,
        observation.artifact_secrets,
        observation.templates,
        observation.volumes,
        observation.pods,
    )
    if any(len(group) > 1 for group in groups):
        raise RunPodResourceConflict("deterministic resource name is duplicated")


def template_identity_matches(
    plan: RunPodCreatePlan,
    template: RunPodTemplateObservation,
) -> bool:
    return (
        template.name == plan.names.template
        and template.image_name == plan.bundle.image.reference
        and template.docker_start_cmd == LAUNCH_COMMAND
        and template.container_disk_gb == plan.placement.container_disk_gb
        and template.volume_gb == 0
        and template.volume_mount_path == NETWORK_VOLUME_MOUNT
        and template.ports == (PROXY_PORT_SPEC,)
        and template.environment
        in {plan.environment_without_artifact, plan.environment_with_artifact}
        and not template.is_serverless
    )


def volume_identity_matches(
    plan: RunPodCreatePlan,
    volume: RunPodVolumeObservation,
) -> bool:
    return (
        volume.name == plan.names.volume
        and volume.data_center_id == plan.placement.data_center_id
        and volume.size_gb >= plan.placement.volume_size_gb
    )


def pod_identity_matches(
    plan: RunPodCreatePlan,
    pod: RunPodPodObservation,
    *,
    template_id: str,
    volume_id: str,
) -> bool:
    # placement is assigned by runpod, not by us, and is absent until it happens: a `CREATED` or
    # `PENDING` pod (including the one the create call just returned) reports no machine even with
    # `includeMachine=true`. comparing `None` for equality read that as "a different gpu than we
    # asked for" -- a permanent conflict for a pod that is merely still being placed, which failed
    # every fresh creation that had not been scheduled yet and every rerun that tried to adopt one.
    #
    # tolerating absence is scoped to exactly that window. once the pod is `RUNNING` it is on real
    # hardware, so a missing gpu type or data center is no longer "not yet decided" -- it is the one
    # moment we could have confirmed the customer got what they asked for, and `exact_core_resources`
    # runs immediately before the readiness probe reports `ready`. treating absence as
    # non-conflicting there would let a pod be declared ready on unverified hardware.
    #
    # the attachments below follow the same rule for the same reason. runpod reports
    # `networkVolumeId` and `templateId` only while the pod holds its machine: the create response
    # carries neither, and a released pod stops carrying them even with `includeNetworkVolume=true`
    # -- verified against the live api, where an `EXITED` pod still reports `machine.gpuTypeId` and
    # `machine.dataCenterId` but neither attachment. Comparing an absent id against the real one
    # read as "attached to something else", which rejected the pod the create call had just made
    # and, in `exact_teardown_resources`, raised a conflict *before* any delete was issued -- so the
    # pod, volume, template and secrets teardown exists to remove were left behind, still billing.
    only_running_reports_attachments = readiness_state(pod.desired_status) != "running"
    attachments_contradict = any(
        observed != expected if observed is not None else not only_running_reports_attachments
        for observed, expected in (
            (pod.network_volume_id, volume_id),
            (pod.template_id, template_id),
        )
    )
    placement_may_be_pending = readiness_state(pod.desired_status) == "pending"
    placement_contradicts = any(
        observed != planned if observed is not None else not placement_may_be_pending
        for observed, planned in (
            (pod.gpu_type_id, plan.placement.gpu_type_id),
            (pod.data_center_id, plan.placement.data_center_id),
        )
    )
    return (
        pod.name == plan.names.app_or_pod
        and pod.image_name == plan.bundle.image.reference
        and not placement_contradicts
        and pod.gpu_count == plan.placement.gpu_count
        and pod.container_disk_gb == plan.placement.container_disk_gb
        and not attachments_contradict
        and pod.ports == (PROXY_PORT_SPEC,)
    )


def exact_core_resources(
    plan: RunPodCreatePlan,
    observation: RunPodObservation,
) -> tuple[
    RunPodSecretObservation,
    RunPodTemplateObservation,
    RunPodVolumeObservation,
    RunPodPodObservation,
]:
    """return exact immutable resources without applying pod status policy."""

    ensure_unique_resources(observation)
    secret = _one(observation.inference_secrets, "inference secret")
    template = _one(observation.templates, "template")
    volume = _one(observation.volumes, "network volume")
    pod = _one(observation.pods, "pod")
    assert type(secret) is RunPodSecretObservation
    assert type(template) is RunPodTemplateObservation
    assert type(volume) is RunPodVolumeObservation
    assert type(pod) is RunPodPodObservation
    if secret.name != plan.names.inference_secret:
        raise RunPodResourceConflict("inference secret does not match")
    if not template_identity_matches(plan, template):
        raise RunPodResourceConflict("template does not match")
    if not volume_identity_matches(plan, volume):
        raise RunPodResourceConflict("network volume does not match")
    if not pod_identity_matches(plan, pod, template_id=template.id, volume_id=volume.id):
        raise RunPodResourceConflict("pod does not match")
    return secret, template, volume, pod


def build_handle(
    plan: RunPodCreatePlan,
    secret: RunPodSecretObservation,
    template: RunPodTemplateObservation,
    volume: RunPodVolumeObservation,
    pod: RunPodPodObservation,
) -> RunPodProviderHandle:
    return RunPodProviderHandle(
        deployment_id=plan.bundle.spec.deployment_id,
        generation=plan.bundle.spec.generation,
        engine_id=plan.bundle.spec.engine.engine_id,
        account_id=plan.placement.account_id,
        pod_id=pod.id,
        pod_name=pod.name,
        network_volume_id=volume.id,
        network_volume_name=volume.name,
        template_id=template.id,
        template_name=template.name,
        inference_secret_id=secret.id,
        inference_secret_name=secret.name,
        data_center_id=plan.placement.data_center_id,
        image_digest=plan.bundle.image.digest,
        public_url=f"https://{pod.id}-8000.proxy.runpod.net",
    )


def exact_teardown_resources(
    plan: RunPodCreatePlan,
    handle: RunPodProviderHandle,
    observation: RunPodObservation,
) -> tuple[
    RunPodSecretObservation | None,
    RunPodSecretObservation | None,
    RunPodTemplateObservation | None,
    RunPodVolumeObservation | None,
    RunPodPodObservation | None,
]:
    """validate partial teardown resources without restricting pod status."""

    ensure_unique_resources(observation)
    inference = observation.inference_secrets[0] if observation.inference_secrets else None
    artifact = observation.artifact_secrets[0] if observation.artifact_secrets else None
    template = observation.templates[0] if observation.templates else None
    volume = observation.volumes[0] if observation.volumes else None
    pod = observation.pods[0] if observation.pods else None
    identities = (
        (inference, handle.inference_secret_id),
        (template, handle.template_id),
        (volume, handle.network_volume_id),
        (pod, handle.pod_id),
    )
    if any(item is not None and item.id != expected for item, expected in identities):
        raise RunPodResourceConflict("provider resource id does not match the exact handle")
    if template is not None and not template_identity_matches(plan, template):
        raise RunPodResourceConflict("template does not match the exact handle")
    if volume is not None and not volume_identity_matches(plan, volume):
        raise RunPodResourceConflict("network volume does not match the exact handle")
    if pod is not None and not pod_identity_matches(
        plan,
        pod,
        template_id=handle.template_id,
        volume_id=handle.network_volume_id,
    ):
        raise RunPodResourceConflict("pod does not match the exact handle")
    return inference, artifact, template, volume, pod
