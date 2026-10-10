"""request-scoped provider credentials with fail-closed serialization."""

from __future__ import annotations

from typing import Literal


def _credential_part(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


class _CredentialBoundary:
    __slots__ = ()

    def __init_subclass__(cls, **_kwargs: object) -> None:
        if cls.__module__ == __name__ and cls.__name__ == "ModalCredentials":
            return
        raise TypeError("provider credential wrappers cannot be subclassed")

    def __copy__(self) -> object:
        raise TypeError("provider credentials cannot be copied")

    def __deepcopy__(self, _memo: dict[int, object]) -> object:
        raise TypeError("provider credentials cannot be copied")

    def __getstate__(self) -> object:
        raise TypeError("provider credentials cannot expose serialization state")

    def __reduce__(self) -> object:
        raise TypeError("provider credentials cannot be serialized")

    def __reduce_ex__(self, _protocol: int) -> object:
        raise TypeError("provider credentials cannot be serialized")


class ModalCredentials(_CredentialBoundary):
    """one request's modal token pair, never a serializable control record."""

    __slots__ = ("__token_id", "__token_secret")

    provider: Literal["modal"] = "modal"

    def __init__(self, token_id: str, token_secret: str) -> None:
        self.__token_id = _credential_part(token_id, "token_id")
        self.__token_secret = _credential_part(token_secret, "token_secret")

    def __repr__(self) -> str:
        return "ModalCredentials(<redacted>)"

    def reveal(self) -> tuple[str, str]:
        """return the request-local values to the modal control implementation."""

        if type(self) is not ModalCredentials:
            raise TypeError("modal credentials must use the exact credential type")
        return self.__token_id, self.__token_secret
