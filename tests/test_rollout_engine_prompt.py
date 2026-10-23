"""Reject receipts when the engine cannot confirm the requested conditioning tokens."""

import asyncio

import pytest

from flash.engine.worker.train.core.child.rollout_contract import (
    RESULT_FIELD,
    RolloutContractServer,
)
from flash.engine.worker.train.core.rollout.contract import digest, validate_result
from tests.test_versioned_rollout_contract import Backend, Server, generate, load, request


@pytest.mark.parametrize("engine_prompt", [None, [9], [1, 9]])
def test_missing_or_changed_engine_prompt_cannot_issue_a_receipt(engine_prompt):
    class ChangedBackend(Backend):
        async def generate(self, **kwargs):
            output = await super().generate(**kwargs)
            output.extra_fields.pop("_flash_engine_prompt_digest")
            if engine_prompt is not None:
                output.extra_fields["_flash_engine_prompt_digest"] = digest(engine_prompt)
            self.output = output
            return output

    class ChangedServer(RolloutContractServer, ChangedBackend):
        __init__ = Server.__init__

    async def scenario():
        server = ChangedServer()
        await load(server)
        with pytest.raises(ValueError, match="engine conditioning tokens"):
            await generate(server)
        assert RESULT_FIELD not in server.output.extra_fields
        # Failure must not release the request ID for a second execution.
        with pytest.raises(ValueError, match="duplicate"):
            await generate(server)

    asyncio.run(scenario())


def test_matching_engine_prompt_is_checked_and_internal_evidence_is_removed():
    async def scenario():
        server = Server()
        await load(server)
        output = await generate(server)
        validate_result(request(), output.extra_fields[RESULT_FIELD], [2, 3], [-0.2, -0.3])
        assert "_flash_engine_prompt_digest" not in output.extra_fields

    asyncio.run(scenario())
