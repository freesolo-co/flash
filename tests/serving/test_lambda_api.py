"""Provider failures must remain bounded and must not expose launch credentials."""

import pickle

import httpx
import pytest

from flash.serving.lambda_cloud.api import LambdaAPI, LambdaAPIError


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_ambiguous_post_is_not_replayed():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("provider did not answer")

    api = LambdaAPI("private", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(httpx.ReadTimeout, match="did not answer"):
            await api.call("POST", "instance-operations/launch", {"user_data": "secret bootstrap"})
        assert len(requests) == 1
        assert requests[0].headers["authorization"] == "Bearer private"
    finally:
        await api.aclose()


@pytest.mark.anyio
async def test_provider_errors_do_not_echo_secret_bodies_and_cross_rpc():
    api = LambdaAPI(
        "private",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                403, json={"error": {"code": "global/forbidden", "message": "secret bootstrap"}}
            )
        ),
    )
    try:
        with pytest.raises(LambdaAPIError, match="global/forbidden") as caught:
            await api.call("GET", "instances")
        assert "secret bootstrap" not in str(caught.value)
        restored = pickle.loads(pickle.dumps(caught.value))
        assert restored.status == 403
        assert restored.code == "global/forbidden"
    finally:
        await api.aclose()
