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
    EngineIdentity,
    Modality,
    ModalPlacement,
    ModalProviderHandle,
    Provider,
    ProviderHandle,
    ResolvedAdapter,
)

__all__ = [
    "DeploymentErrorCode",
    "DeploymentErrorReason",
    "DeploymentRequest",
    "DeploymentResult",
    "DeploymentSpec",
    "EngineIdentity",
    "ModalCredentials",
    "ModalPlacement",
    "ModalProviderHandle",
    "Modality",
    "PlanningError",
    "Provider",
    "ProviderHandle",
    "ResolvedAdapter",
    "canonical_mapping_fingerprint",
    "plan_deployment",
]
