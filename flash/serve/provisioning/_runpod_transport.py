"""single-attempt stdlib transport for request-scoped runpod lifecycle calls."""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Protocol

from flash.serve.control import DeploymentErrorCode

GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_BASE_URL = "https://rest.runpod.io/v1"
_DEFAULT_TIMEOUT_SECONDS = 30.0


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def build_no_redirect_opener(*handlers: object):
    """build an opener that never follows or forwards authorization across redirects."""

    return urllib.request.build_opener(_NoRedirectHandler(), *handlers)


class RunPodTransportFailure(RuntimeError):
    """sanitized transport failure with no retained response or credential data."""

    __slots__ = ("code", "outcome_unknown")

    def __init__(self, code: DeploymentErrorCode, *, outcome_unknown: bool = False) -> None:
        self.code = code
        self.outcome_unknown = outcome_unknown
        super().__init__("runpod transport operation failed")


class RunPodTransport(Protocol):
    """narrow raw transport used by the runpod lifecycle implementation."""

    def graphql(
        self,
        document: str,
        variables: Mapping[str, object],
        *,
        mutation: bool,
        deadline_at: float,
    ) -> object: ...

    def rest(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None,
        *,
        mutation: bool,
        deadline_at: float,
    ) -> object: ...


class StdlibRunPodTransport:
    """one request-scoped runpod client with no retries or global key state."""

    __slots__ = ("__api_key", "_clock", "_opener")

    def __init__(
        self,
        api_key: str,
        *,
        opener: object | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(api_key) is not str or not api_key:
            raise ValueError("runpod api key must be a nonempty string")
        self.__api_key = api_key
        self._opener = build_no_redirect_opener() if opener is None else opener
        self._clock = clock

    def __repr__(self) -> str:
        return "StdlibRunPodTransport(<redacted>)"

    def graphql(
        self,
        document: str,
        variables: Mapping[str, object],
        *,
        mutation: bool,
        deadline_at: float,
    ) -> object:
        if type(document) is not str or not document:
            raise ValueError("graphql document must be nonempty")
        result = self._request(
            GRAPHQL_URL,
            method="POST",
            payload={"query": document, "variables": dict(variables)},
            mutation=mutation,
            deadline_at=deadline_at,
        )
        if type(result) is dict and "errors" in result:
            if mutation:
                raise RunPodTransportFailure("resource_ambiguous", outcome_unknown=True)
            raise RunPodTransportFailure("provider_rejected")
        return result

    def rest(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None,
        *,
        mutation: bool,
        deadline_at: float,
    ) -> object:
        if method not in {"GET", "POST", "PATCH", "DELETE"}:
            raise ValueError("runpod rest method is unsupported")
        if type(path) is not str or not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("runpod rest path must be an absolute path")
        body = None if payload is None else dict(payload)
        return self._request(
            REST_BASE_URL + path,
            method=method,
            payload=body,
            mutation=mutation,
            deadline_at=deadline_at,
        )

    def _request(
        self,
        url: str,
        *,
        method: str,
        payload: Mapping[str, object] | None,
        mutation: bool,
        deadline_at: float,
    ) -> object:
        timeout = self._timeout(deadline_at)
        encoded = None
        if payload is not None:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.__api_key}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with self._open(request, timeout) as response:
                status = getattr(response, "status", None)
                if type(status) is not int or not 200 <= status < 300:
                    raise self._status_failure(status, mutation=mutation)
                raw = response.read()
        except urllib.error.HTTPError as exc:
            try:
                code = self._http_error_code(exc.code, mutation=mutation)
            finally:
                exc.close()
            raise code from None
        except (TimeoutError, urllib.error.URLError, OSError):
            raise RunPodTransportFailure(
                "resource_ambiguous" if mutation else "transport_failed",
                outcome_unknown=mutation,
            ) from None
        except RunPodTransportFailure:
            raise
        except Exception:
            raise RunPodTransportFailure(
                "resource_ambiguous" if mutation else "transport_failed",
                outcome_unknown=mutation,
            ) from None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            raise RunPodTransportFailure(
                "resource_ambiguous" if mutation else "transport_failed",
                outcome_unknown=mutation,
            ) from None

    def _open(self, request: urllib.request.Request, timeout: float):
        method = getattr(self._opener, "open", None)
        if callable(method):
            return method(request, timeout=timeout)
        if callable(self._opener):
            return self._opener(request, timeout=timeout)
        raise RunPodTransportFailure("transport_failed")

    @staticmethod
    def _status_failure(status: object, *, mutation: bool) -> RunPodTransportFailure:
        if type(status) is int:
            return StdlibRunPodTransport._http_error_code(status, mutation=mutation)
        return RunPodTransportFailure(
            "resource_ambiguous" if mutation else "transport_failed",
            outcome_unknown=mutation,
        )

    def _timeout(self, deadline_at: float) -> float:
        if type(deadline_at) not in {int, float} or not math.isfinite(float(deadline_at)):
            raise ValueError("deadline_at must be finite")
        remaining = float(deadline_at) - self._clock()
        if remaining <= 0:
            raise RunPodTransportFailure("transport_failed")
        return min(_DEFAULT_TIMEOUT_SECONDS, remaining)

    @staticmethod
    def _http_error_code(status: int, *, mutation: bool) -> RunPodTransportFailure:
        if 300 <= status < 400 or status == 408 or 500 <= status < 600:
            return RunPodTransportFailure(
                "resource_ambiguous" if mutation else "transport_failed",
                outcome_unknown=mutation,
            )
        if status in {401, 403}:
            return RunPodTransportFailure("authentication_failed")
        if status == 404:
            return RunPodTransportFailure("not_found")
        if status == 409:
            return RunPodTransportFailure("conflict")
        if status in {402, 429}:
            return RunPodTransportFailure("capacity_unavailable")
        if 400 <= status < 500:
            return RunPodTransportFailure("provider_rejected")
        return RunPodTransportFailure("transport_failed")
