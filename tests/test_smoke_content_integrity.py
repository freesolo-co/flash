"""Deployment readiness requires actual generated text."""

import pytest

from flash.serve.contract.errors import ServingError
from flash.server.routes.serving_smoke import _smoke_provenance


@pytest.mark.parametrize("content", [123, True, ["answer"], {"answer": "4"}])
def test_smoke_rejects_non_text_generation(content):
    checkpoint = "flash-run/final"
    result = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "freesolo": {"checkpoint_id": checkpoint},
        "_freesolo_headers": {"checkpoint_id": checkpoint},
    }
    with pytest.raises(ServingError, match="content"):
        _smoke_provenance(result, checkpoint, checkpoint)
