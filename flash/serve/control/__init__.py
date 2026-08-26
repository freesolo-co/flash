"""import-light deterministic serving deployment control foundation."""

from ._canonical import canonical_mapping_fingerprint
from .credentials import ModalCredentials
from .planning import PlanningError, plan_deployment
from .types import (
    DeploymentErrorCode,
    DeploymentErrorReason,
    DeploymentRequest,
    DeploymentResult,
    DeploymentSpec,
    DeploymentStatus,
    EngineIdentity,
    Modality,
    ModalPlacement,
    ModalProviderHandle,
    Placement,
    Provider,
    ProviderHandle,
    RepoType,
    ResolvedAdapter,
)

__all__ = [
    "DeploymentErrorCode",
    "DeploymentErrorReason",
    "DeploymentRequest",
    "DeploymentResult",
    "DeploymentSpec",
    "DeploymentStatus",
    "EngineIdentity",
    "ModalCredentials",
    "ModalPlacement",
    "ModalProviderHandle",
    "Modality",
    "Placement",
    "PlanningError",
    "Provider",
    "ProviderHandle",
    "RepoType",
    "ResolvedAdapter",
    "canonical_mapping_fingerprint",
    "plan_deployment",
]
