import pytest

from flash.serving.src.engine.generation import _payload
from flash.serving.src.io.cache_namespace import internal_generation_payload
from flash.serving.src.io.openai_request import OpenAIGenerateRequest


def test_authenticated_namespace_survives_private_rpc_without_becoming_a_public_field():
    request = OpenAIGenerateRequest(model_id="base", prompt="hello")
    request._cache_namespace = "a" * 64
    assert "_cache_namespace" not in request.model_dump()
    wire = internal_generation_payload(request)
    decoded, _ = _payload(wire)
    assert decoded._cache_namespace == "a" * 64
    assert wire["_cache_namespace"] == "a" * 64
    other = request.model_copy()
    other._cache_namespace = "b" * 64
    assert _payload(internal_generation_payload(other))[0]._cache_namespace == "b" * 64
    assert request._cache_namespace == "a" * 64


@pytest.mark.parametrize("namespace", ["", "a" * 63, "G" * 64, 1, []])
def test_invalid_private_cache_scope_fails_before_inference(namespace):
    with pytest.raises(ValueError, match="cache namespace"):
        _payload({"model_id": "base", "prompt": "hello", "_cache_namespace": namespace})
