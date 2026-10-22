import pytest
from fastapi import HTTPException, Request

from flash.serving.src.http.headers import assert_internal, is_trusted_internal


def test_non_ascii_internal_header_is_unauthorized():
    request = Request({"type": "http", "headers": [(b"x-freesolo-internal-key", b"invalid-\xff")]})
    with pytest.raises(HTTPException) as exc:
        assert_internal(request, "secret")
    assert exc.value.status_code == 401


def test_non_ascii_internal_header_does_not_break_external_auth_fallback():
    request = Request({"type": "http", "headers": [(b"x-freesolo-internal-key", b"invalid-\xff")]})
    assert not is_trusted_internal(request, ("secret", "other-secret"))
