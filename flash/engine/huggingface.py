"""hugging face helpers shared across control-plane and worker boundaries."""

from __future__ import annotations

from typing import TypedDict


class ModelRevisionKwargs(TypedDict, total=False):
    revision: str


def model_revision_kwargs(revision: str = "") -> ModelRevisionKwargs:
    """return the hub revision keyword for a nonempty pinned revision."""
    return {"revision": revision} if revision else {}


def hub_error_transience(exc: BaseException) -> bool | None:
    """classify known hub and transport failures, leaving unrelated errors unknown."""
    import httpx
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import RequestException
    from requests.exceptions import Timeout as RequestsTimeout

    if isinstance(exc, (RepositoryNotFoundError, RevisionNotFoundError, GatedRepoError)):
        return False
    if isinstance(exc, LocalEntryNotFoundError):
        return True
    if isinstance(exc, EntryNotFoundError):
        return False
    if isinstance(
        exc,
        (
            RequestsConnectionError,
            RequestsTimeout,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, HfHubHTTPError):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status in {401, 403, 404}:
            return False
        return (
            status is None
            or status in {408, 425, 429}
            or (isinstance(status, int) and 500 <= status <= 599)
        )
    if isinstance(exc, RequestException):
        return False
    return None
