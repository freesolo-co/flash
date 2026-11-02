"""Classify Azure artifact failures without inspecting credential-bearing messages."""

from .manifest import ArtifactNotFoundError


def artifact_error_transience(error: BaseException) -> bool | None:
    from azure.core.exceptions import ServiceRequestError, ServiceResponseError

    if isinstance(error, ArtifactNotFoundError):
        return False
    if isinstance(error, (ServiceRequestError, ServiceResponseError, TimeoutError)):
        return True
    status = getattr(error, "status_code", None) or getattr(
        getattr(error, "response", None), "status_code", None
    )
    if status is not None:
        return status in {408, 409, 412, 429} or 500 <= status <= 599
    return None
